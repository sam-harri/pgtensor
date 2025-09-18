use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use pgrx::bgworkers::*;
use pgrx::pg_sys;
use pgrx::prelude::*;
use std::cell::UnsafeCell;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::tensor_core::Tensor;

const MAX_WORKERS: usize = 4;
const QUEUE_CAP: usize = 16;
const MAX_BYTES: usize = 2048; // no big ahh tensors for the time being

// Anything we put into Postgres' shared memory must be safe to bitwise copy
// (no destructors/moves) and contain no non-'static borrowss
// We literally write its bytes into the segment, then other processes will read them
pub trait PGRXSharedMemory: Copy + 'static {}
impl<T: Copy + 'static> PGRXSharedMemory for T {}

// shemem follows Postgres' request -> init 2-phase process
// This trait enforces both steps
pub trait PgSharedMemoryInitialization {
    // The exact byte image we plant into shared memory on first boot
    type Value: PGRXSharedMemory;

    // please budget me N bytes and a tranche of LWLocks
    // from here on out, when you see tranche just think an array of LWLocks
    // C API: RequestAddinShmemSpace(size) + RequestNamedLWLockTranche(name, n)
    unsafe fn on_shmem_request(&'static self);

    // give me my pointer, then if I’m first, I’ll initialize it
    // bitwise copy PGRXSharedMemory into the shmem if first
    // C API: ShmemInitStruct(name, size, &found) + GetNamedLWLockTranche(name)
    unsafe fn on_shmem_startup(&'static self, value: Self::Value);
}

// Wrapper around postgres' LWLock (its cross process lock)
// name is basically a key to C's ShmemInitStruct to allocate/attach the shmem block
// and doubles up as tranche name for RequestNamedLWLockTranche/GetNamedLWLockTranche
// so we can obtain a pointer to a named LWLock we asked Postgres to createed for us
// Need the pointer to be in UnsafeCell since we will me mutating once when postgres gives us the pointer,
// but from that point on it will be read only
pub struct PgLwLock<T> {
    name: &'static CStr,
    inner: UnsafeCell<*mut Shared<T>>,
}

// since the T that impl's PGRXSharedMemory is guarded by the LWLock, we can safely share this between threads
// unsafe because we need to remember to actually use the lock
unsafe impl<T: PGRXSharedMemory> Sync for PgLwLock<T> {}

impl<T> PgLwLock<T> {
    pub const unsafe fn new(name: &'static CStr) -> Self {
        Self {
            name,
            // zero initialize the pointer, then we'll set it when we get from postgres
            inner: UnsafeCell::new(std::ptr::null_mut()),
        }
    }
    pub const fn name(&self) -> &'static CStr {
        self.name
    }
}

// then the logic for actually aquiriing the lock
impl<T: PGRXSharedMemory> PgLwLock<T> {
    // get a shared reference to T, &T
    pub fn share(&self) -> PgLwLockShareGuard<'_, T> {
        unsafe {
            let shared = self
                .inner
                .get()
                .read()
                .as_ref()
                .expect("PgLwLock not initialized");
            pg_sys::LWLockAcquire(shared.lock, pg_sys::LWLockMode::LW_SHARED);
            PgLwLockShareGuard {
                data: &*shared.data.get(),
                lock: shared.lock,
            }
        }
    }
    // get exlusive access to the lock, &mut T
    pub fn exclusive(&self) -> PgLwLockExclusiveGuard<'_, T> {
        unsafe {
            let shared = self
                .inner
                .get()
                .read()
                .as_ref()
                .expect("PgLwLock not initialized");
            pg_sys::LWLockAcquire(shared.lock, pg_sys::LWLockMode::LW_EXCLUSIVE);
            PgLwLockExclusiveGuard {
                data: &mut *shared.data.get(),
                lock: shared.lock,
            }
        }
    }
}

impl<T: PGRXSharedMemory> PgSharedMemoryInitialization for PgLwLock<T> {
    type Value = T;

    // hook that runs when we request shmem
    // tell Postgres reserve me size_of::<Shared<T>>() bytes and 1 named LWLock
    unsafe fn on_shmem_request(&'static self) {
        // returns a pointer into the shared memory segment that we can fill the first time and reuse after
        pg_sys::RequestAddinShmemSpace(size_of::<Shared<T>>());
        // give us 1 LWLock, we'll need to ask for more later perhaps? idk tbh cause it might be a lot of work
        // and I dont know if tht would be the bottleneck
        pg_sys::RequestNamedLWLockTranche(self.name.as_ptr(), 1);
    }

    // actually init the stuff were storing inside the shmem segment
    // postgres runs on a reserve first, init later basis
    unsafe fn on_shmem_startup(&'static self, value: T) {
        let addin_shmem_init_lock = &raw mut (*pg_sys::MainLWLockArray.add(21)).lock;
        pg_sys::LWLockAcquire(addin_shmem_init_lock, pg_sys::LWLockMode::LW_EXCLUSIVE);

        let mut found = false;
        let shm = pg_sys::ShmemInitStruct(self.name.as_ptr(), size_of::<Shared<T>>(), &mut found)
            .cast::<Shared<T>>();

        if !found {
            shm.write(Shared {
                data: UnsafeCell::new(value),
                lock: &raw mut (*pg_sys::GetNamedLWLockTranche(self.name.as_ptr())).lock,
            });
        }
        // this is where we use the UnsafeCell to actually set the pointer to the lock
        *self.inner.get() = shm;

        pg_sys::LWLockRelease(addin_shmem_init_lock);
    }
}

// wrapper around our shared state. Since the lock is managed by postgres, we have to use UnsafeCell
// to allow for interior mutabililty, and we use the LWLock to gate access to it.
// lock itself is a pointer to the only LWLock that postgres allocated for us.
// if you look at pg_sys::RequestNamedLWLockTranche, we only ask for 1 LWLock across at workers, makes it easier for now
#[repr(C)]
struct Shared<T> {
    data: UnsafeCell<T>,
    lock: *mut pg_sys::LWLock,
}

// impl of shared/read guard
// when this guard is active you get immutable view of the payload (SharedState)
// references only valid as long as the guard is active
pub struct PgLwLockShareGuard<'a, T> {
    data: &'a T,
    lock: *mut pg_sys::LWLock,
}

impl<T> Deref for PgLwLockShareGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.data
    }
}

impl<T> Drop for PgLwLockShareGuard<'_, T> {
    fn drop(&mut self) {
        unsafe {
            if pg_sys::InterruptHoldoffCount > 0 {
                pg_sys::LWLockRelease(self.lock);
            }
        }
    }
}

// impl for exclusive/write guard
// unique, mutable access while the guard lives
pub struct PgLwLockExclusiveGuard<'a, T> {
    data: &'a mut T,
    lock: *mut pg_sys::LWLock,
}

impl<T> Deref for PgLwLockExclusiveGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.data
    }
}
impl<T> DerefMut for PgLwLockExclusiveGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.data
    }
}
impl<T> Drop for PgLwLockExclusiveGuard<'_, T> {
    fn drop(&mut self) {
        unsafe {
            if pg_sys::InterruptHoldoffCount > 0 {
                pg_sys::LWLockRelease(self.lock);
            }
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
enum MsgState {
    Empty = 0,   // slot is unused (safe to overwrite)
    Pending = 1, // caller wrote a request into the slot; worker hasn’t produced a response yet
    Done = 2,    // worker wrote the response; caller can read it
                 // then when the caller reads the response, it sets it back to 0
}

// The thing queue is a buffer of Msgs. These are the messages were using for IPC
// Req and Resp contain the memory for passing the serialized tensors
#[repr(C)]
#[derive(Copy, Clone)]
struct Msg {
    id: u64,
    caller_pid: pg_sys::pid_t,
    state: MsgState,
    _pad: u8,
    req_len: u16,
    resp_len: u16,
    req: [u8; MAX_BYTES],
    resp: [u8; MAX_BYTES],
}
impl Msg {
    const fn empty() -> Self {
        Self {
            id: 0,
            caller_pid: 0,
            state: MsgState::Empty,
            _pad: 0,
            req_len: 0,
            resp_len: 0,
            req: [0; MAX_BYTES],
            resp: [0; MAX_BYTES],
        }
    }
}

// implementation of a ring queue for passing Msgs between the worker and calling processes
#[repr(C)]
#[derive(Copy, Clone)]
struct Ring {
    head: u32,
    tail: u32,
    slots: [Msg; QUEUE_CAP],
}
impl Ring {
    const fn new() -> Self {
        const M: Msg = Msg::empty();
        Self {
            head: 0,
            tail: 0,
            slots: [M; QUEUE_CAP],
        }
    }
    fn is_full(&self) -> bool {
        (self.head - self.tail) as usize >= QUEUE_CAP
    }
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }
    fn idx(i: u32) -> usize {
        (i as usize) % QUEUE_CAP
    }
}

// each worker process has its own state
// name will be for the name of the ONNX model is has loaded
// then queue_idx is it's the location of it's msg in the queue
#[repr(C)]
#[derive(Copy, Clone)]
struct WorkerEntry {
    in_use: bool,
    pid: pg_sys::pid_t,
    queue_idx: i32,
    name: [u8; 64],
}
impl WorkerEntry {
    const fn empty() -> Self {
        Self {
            in_use: false,
            pid: 0,
            queue_idx: -1,
            name: [0; 64],
        }
    }
    fn set_name(&mut self, s: &str) {
        let b = s.as_bytes();
        let n = b.len().min(63);
        self.name[..n].copy_from_slice(&b[..n]);
        self.name[n] = 0;
    }
    fn name_str(&self) -> &str {
        let len = self
            .name
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..len]).unwrap_or("") // gotta add a error jawn here
    }
}

// shared state that the LWLock protects
// contains all the workes and ring used for message passing
#[repr(C)]
#[derive(Copy, Clone)]
struct SharedState {
    workers: [WorkerEntry; MAX_WORKERS],
    queues: [Ring; MAX_WORKERS],
}
impl SharedState {
    const fn new() -> Self {
        const W: WorkerEntry = WorkerEntry::empty();
        const R: Ring = Ring::new();
        Self {
            workers: [W; MAX_WORKERS],
            queues: [R; MAX_WORKERS],
        }
    }
}

// global LWLock and the shared memory containing the SharedState that it protects)
const LW_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"tensor_bg\0") };
#[allow(non_upper_case_globals)]
static SHMEM: PgLwLock<SharedState> = unsafe { PgLwLock::new(LW_NAME) };

// Pre pg16, extentions could request shmem in _PG_init(), then initialize in shmem_startup_hook
// Starting in pg16 though, a new shmem_request_hook was added and postgres required that all shared-memory
// requests happen there (not in _PG_init() or shmem_startup_hook). Initialization
// still happens in shmem_startup_hook though
// really what we're doing here is building the _PG_init fn, and we do so differently based on the pg version
#[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
mod shmem_hooks {
    use super::*;
    // hook to request the shmem
    static mut PREV_SHMEM_REQUEST_HOOK: Option<unsafe extern "C-unwind" fn()> = None;
    // then another hook to actually init the shmem segment
    static mut PREV_SHMEM_STARTUP_HOOK: Option<unsafe extern "C-unwind" fn()> = None;

    #[pg_guard]
    pub extern "C-unwind" fn _PG_init() {
        // since we are asking for shmem, we need to declare in postgres.conf
        // shared_preload_libraries = 'pgtensor'"
        // look in the lib.rs `postgres_conf_options()` for example

        // this flag is only true while the postmaster is loading shared_preload_libraries
        if unsafe { !pg_sys::process_shared_preload_libraries_in_progress } {
            pgrx::error!("tensor_bg must be loaded via shared_preload_libraries");
        }
        unsafe {
            // then we register/chain these hooks, and let the postmaster call them when its time
            PREV_SHMEM_REQUEST_HOOK = pg_sys::shmem_request_hook;
            pg_sys::shmem_request_hook = Some(shmem_request);
            PREV_SHMEM_STARTUP_HOOK = pg_sys::shmem_startup_hook;
            pg_sys::shmem_startup_hook = Some(shmem_startup);
        }
    }

    // when the postmaster finally calls the hook, check that the hook is registered and run it
    #[pg_guard]
    unsafe extern "C-unwind" fn shmem_request() {
        if let Some(prev) = PREV_SHMEM_REQUEST_HOOK {
            prev();
        }
        // this is the trait fn from PgSharedMemoryInitialization
        SHMEM.on_shmem_request();
    }

    #[pg_guard]
    unsafe extern "C-unwind" fn shmem_startup() {
        if let Some(prev) = PREV_SHMEM_STARTUP_HOOK {
            prev();
        }
        // Other trait fn. Here we pass in the initalization logic for the SharedState
        SHMEM.on_shmem_startup(SharedState::new());
    }
}

#[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
mod shmem_hooks {
    use super::*;
    // no need for a request hook since we request right in the _PG_init
    static mut PREV_SHMEM_STARTUP_HOOK: Option<unsafe extern "C-unwind" fn()> = None;

    #[pg_guard]
    pub extern "C-unwind" fn _PG_init() {
        if unsafe { !pg_sys::process_shared_preload_libraries_in_progress } {
            pgrx::error!("tensor_bg must be loaded via shared_preload_libraries");
        }
        // so we run the request instead of registering the hook
        unsafe {
            SHMEM.on_shmem_request();
        }

        // but register the hook for the shmem init
        unsafe {
            PREV_SHMEM_STARTUP_HOOK = pg_sys::shmem_startup_hook;
            pg_sys::shmem_startup_hook = Some(shmem_startup);
        }
    }

    // and same as before we check the hook and run it when its time
    #[pg_guard]
    unsafe extern "C-unwind" fn shmem_startup() {
        if let Some(prev) = PREV_SHMEM_STARTUP_HOOK {
            prev();
        }
        SHMEM.on_shmem_startup(SharedState::new());
    }
}

// finally, we reexport the _PG_init for whatever cfg we're using
pub use shmem_hooks::_PG_init;

// monotic ID, use an Atomic so that we can incrment atomically across threads
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
// request ID is the worker PID XORed with the monotic ID
// should be good enough to avoid collisions but I have zero proof of that lmao
fn next_request_id() -> u64 {
    let base = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    (base << 16) ^ (unsafe { pg_sys::MyProcPid } as u64)
}

unsafe fn set_latch_for_pid(pid: pg_sys::pid_t) {
    let proc = pg_sys::BackendPidGetProc(pid);
    if !proc.is_null() {
        pg_sys::SetLatch(&mut (*proc).procLatch);
    }
}

// for now I have the most simple thing going where you just dynamically load a background
// worker. for hte inference we'll need to load up the ONNX runtime and the ONNX model
// in the initialization
#[pg_extern]
pub fn load_bgworker(name: &str) -> bool {
    // we only allocated 64 bytes for the name, and it has to be null terminate cause CStr
    if name.is_empty() || name.len() > 63 {
        pgrx::error!("name must be 1..63 chars");
    }

    // find the first unused index in the worker arr
    let mut queue_idx: i32 = -1;
    {
        let mut g = SHMEM.exclusive();
        for w in g.workers.iter() {
            if w.in_use && w.name_str() == name {
                pgrx::error!("bgworker \"{}\" already loaded", name);
            }
        }
        for (i, w) in g.workers.iter_mut().enumerate() {
            if !w.in_use {
                w.in_use = true;
                w.pid = 0;
                w.queue_idx = i as i32;
                w.set_name(name);
                queue_idx = i as i32;
                break;
            }
        }
        // no space left :(
        if queue_idx < 0 {
            pgrx::error!("no free worker slots (MAX_WORKERS={})", MAX_WORKERS);
        }
    }

    // actually start the worker
    let mut builder = BackgroundWorkerBuilder::new("tensor_bgworker");
    let builder = builder
        .set_function("tensor_worker_main")
        .set_library("pgtensor") // not sure what env var we have access to so hardcoding for now
        .set_extra(&format!("{}:{}", queue_idx, name))
        .set_notify_pid(unsafe { pg_sys::MyProcPid })
        .enable_spi_access();

    match builder.load_dynamic() {
        Ok(handle) => {
            let pid = handle
                .wait_for_startup()
                .map_err(|s| {
                    pgrx::error!("failed to start bgworker \"{}\": {:?}", name, s);
                })
                .unwrap();

            let mut g = SHMEM.exclusive();
            g.workers[queue_idx as usize].pid = pid;
            true
        }
        Err(_) => {
            // no more worker processes left
            // shouldnt happen as long as you keep the MAX_WORKERS under
            // the postgres.conf max workers - whatever other workers you have
            let mut g = SHMEM.exclusive();
            g.workers[queue_idx as usize] = WorkerEntry::empty();
            pgrx::error!("RegisterDynamicBackgroundWorker failed (max_worker_processes?)");
        }
    }
}

// once you have a worker created (aka when the model is loaded later)
// you can send tensors over the shmem block
#[pg_extern]
pub fn to_bgworker(name: &str, t: Tensor) -> Tensor {
    // find worker by it's name. This will be the model name
    let (qidx, pid) = {
        let g = SHMEM.share();
        let w = g
            .workers
            .iter()
            .find(|w| w.in_use && w.name_str() == name)
            .unwrap_or_else(|| pgrx::error!("bgworker \"{}\" not found", name));
        (w.queue_idx as usize, w.pid)
    };
    if pid <= 0 {
        // tried to access worker/model that isnt loaded
        pgrx::error!("bgworker \"{}\" not running", name);
    }

    // serialize the Tensor into CBOR
    let req =
        serde_cbor::to_vec(&t).unwrap_or_else(|e| pgrx::error!("CBOR serialize failed: {}", e));
    // limiting Tensors to this size for now but we can just change MAX_BYTES
    if req.len() > MAX_BYTES {
        pgrx::error!(
            "tensor too large for queue ({} > {} bytes)",
            req.len(),
            MAX_BYTES
        );
    }
    let req_id = next_request_id();
    // process we'll be sending the tensor back to
    let my_pid = unsafe { pg_sys::MyProcPid as pg_sys::pid_t };

    // enqueue the message
    let slot_idx = {
        let mut g = SHMEM.exclusive();
        let ring = &mut g.queues[qidx];
        if ring.is_full() {
            pgrx::error!("worker \"{}\" queue full", name);
        }
        let idx = Ring::idx(ring.head);
        let m = &mut ring.slots[idx];
        // we create/alter the message in place which is safe because we're holding hte lock to the shmem
        m.id = req_id;
        m.caller_pid = my_pid;
        m.state = MsgState::Pending;
        m.req_len = req.len() as u16;
        m.resp_len = 0;
        m.req[..req.len()].copy_from_slice(&req);
        ring.head = ring.head.wrapping_add(1);
        idx
    };

    // wake the worker, basically tell it that it has a Msg waiting for it
    unsafe {
        set_latch_for_pid(pid);
    }

    // wait for response. This will be us waiting for the inference to happen
    loop {
        {
            let g = SHMEM.share();
            let m = &g.queues[qidx].slots[slot_idx];
            if m.id == req_id && m.state == MsgState::Done {
                let resp = &m.resp[..m.resp_len as usize];
                let out: Tensor = serde_cbor::from_slice(resp)
                    .unwrap_or_else(|e| pgrx::error!("CBOR decode failed: {}", e));
                return out;
            }
        }

        unsafe {
            pg_sys::ResetLatch(pg_sys::MyLatch);
            let rc = pg_sys::WaitLatch(
                pg_sys::MyLatch,
                (pg_sys::WL_LATCH_SET | pg_sys::WL_TIMEOUT | pg_sys::WL_POSTMASTER_DEATH) as i32,
                50,
                pg_sys::PG_WAIT_EXTENSION,
            );
            if (rc & pg_sys::WL_POSTMASTER_DEATH as i32) != 0 {
                pgrx::error!("postmaster died");
            }
        }
    }
}

// program/method the worker processes are running
#[pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn tensor_worker_main(_arg: pg_sys::Datum) {
    let extra = BackgroundWorker::get_extra();
    let (qidx, wname) = {
        let mut it = extra.splitn(2, ':');
        let idx = it.next().unwrap_or("0").parse::<usize>().unwrap_or(0);
        let nm = it.next().unwrap_or("unnamed").to_owned();
        (idx, nm)
    };

    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    BackgroundWorker::connect_worker_to_spi(Some("postgres"), None);

    log!("tensor_worker \"{}\" started on queue {}", wname, qidx);

    // waiting for the latch from to_bgworker
    while BackgroundWorker::wait_latch(Some(Duration::from_millis(200))) {
        loop {
            // then pop the message when it's ready
            let (have, msg) = {
                let mut g = SHMEM.exclusive();
                let ring = &mut g.queues[qidx];
                if ring.is_empty() {
                    (false, Msg::empty())
                } else {
                    let idx = Ring::idx(ring.tail);
                    let m = &mut ring.slots[idx];
                    if m.state != MsgState::Pending {
                        (false, Msg::empty())
                    } else {
                        let local = *m; // copy out
                        ring.tail = ring.tail.wrapping_add(1);
                        (true, local)
                    }
                }
            };
            if !have {
                break;
            }

            // decode the CBOR repr of the tensor we sent through
            let t: Tensor = match serde_cbor::from_slice(&msg.req[..msg.req_len as usize]) {
                Ok(v) => v,
                Err(e) => {
                    log!("tensor_worker decode error: {}", e);
                    continue;
                }
            };

            // now we're fully in Rust land. This is where we'll run our inference, most likely
            // through a better abstracted
            let mut t_plus = t.clone();
            for x in &mut t_plus.elem_buffer {
                *x += 1.0;
            }

            // logging for now for debug purposes
            let _ = BackgroundWorker::transaction(|| {
                Spi::connect(|_| {
                    let s_in: String = t.clone().into();
                    let s_out: String = t_plus.clone().into();
                    log!("tensor_worker[{}] IN : {}", wname, s_in);
                    log!("tensor_worker[{}] OUT: {}", wname, s_out);
                    Ok::<(), pgrx::spi::Error>(())
                })
            });

            // once we're run out inference and get a Tensor backout
            // encode is back into cbor
            let out = serde_cbor::to_vec(&t_plus).unwrap();

            // and write response back to the same slot (tail-1)
            {
                let mut g = SHMEM.exclusive();
                let ring = &mut g.queues[qidx];
                let idx = Ring::idx(ring.tail.wrapping_sub(1));
                let m = &mut ring.slots[idx];
                let n = out.len().min(MAX_BYTES);
                m.resp[..n].copy_from_slice(&out[..n]);
                m.resp_len = n as u16;
                m.state = MsgState::Done;
                unsafe {
                    set_latch_for_pid(msg.caller_pid);
                }
            }
        }
    }

    log!("tensor_worker \"{}\" shutting down", wname);
}
