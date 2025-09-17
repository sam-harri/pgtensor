// src/bgworker.rs
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use pgrx::bgworkers::*;
use pgrx::pg_sys;
use pgrx::prelude::*;
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use crate::tensor_core::Tensor;

const MAX_WORKERS: usize = 4;
const QUEUE_CAP: usize = 16;
const MAX_BYTES: usize = 2048; // per message req/resp CBOR bytes

pub trait PGRXSharedMemory: Copy + 'static {}
impl<T: Copy + 'static> PGRXSharedMemory for T {}

pub trait PgSharedMemoryInitialization {
    type Value: PGRXSharedMemory;
    unsafe fn on_shmem_request(&'static self);
    unsafe fn on_shmem_startup(&'static self, value: Self::Value);
}

use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use std::cell::UnsafeCell;

pub struct PgLwLock<T> {
    name: &'static CStr,
    inner: UnsafeCell<*mut Shared<T>>,
}
unsafe impl<T: PGRXSharedMemory> Sync for PgLwLock<T> {}

impl<T> PgLwLock<T> {
    pub const unsafe fn new(name: &'static CStr) -> Self {
        Self {
            name,
            inner: UnsafeCell::new(std::ptr::null_mut()),
        }
    }
    pub const fn name(&self) -> &'static CStr {
        self.name
    }
}
impl<T: PGRXSharedMemory> PgLwLock<T> {
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

    unsafe fn on_shmem_request(&'static self) {
        pg_sys::RequestAddinShmemSpace(size_of::<Shared<T>>());
        pg_sys::RequestNamedLWLockTranche(self.name.as_ptr(), 1);
    }

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
        *self.inner.get() = shm;

        pg_sys::LWLockRelease(addin_shmem_init_lock);
    }
}

#[repr(C)]
struct Shared<T> {
    data: UnsafeCell<T>,
    lock: *mut pg_sys::LWLock,
}

pub struct PgLwLockShareGuard<'a, T> {
    data: &'a T,
    lock: *mut pg_sys::LWLock,
}
unsafe impl<T: PGRXSharedMemory> Sync for PgLwLockShareGuard<'_, T> {}
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

pub struct PgLwLockExclusiveGuard<'a, T> {
    data: &'a mut T,
    lock: *mut pg_sys::LWLock,
}
unsafe impl<T: PGRXSharedMemory> Sync for PgLwLockExclusiveGuard<'_, T> {}
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
    Empty = 0,
    Pending = 1,
    Done = 2,
}

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
        std::str::from_utf8(&self.name[..len]).unwrap_or("")
    }
}

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

// Global lock + state
const LW_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"tensor_bg\0") };
#[allow(non_upper_case_globals)]
static SHMEM: PgLwLock<SharedState> = unsafe { PgLwLock::new(LW_NAME) };

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    if unsafe { !pg_sys::process_shared_preload_libraries_in_progress } {
        pgrx::error!("tensor_bg must be loaded via shared_preload_libraries");
    }

    unsafe {
        SHMEM.on_shmem_request();
    }

    unsafe extern "C-unwind" fn shmem_startup() {
        unsafe {
            SHMEM.on_shmem_startup(SharedState::new());
        }
    }
    unsafe {
        pg_sys::shmem_startup_hook = Some(shmem_startup);
    }
}

// monotonic id, avoid casting TransactionId
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_request_id() -> u64 {
    // combine atomic counter with pid to reduce collision probability
    let base = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    (base << 16) ^ (unsafe { pg_sys::MyProcPid } as u64)
}

unsafe fn set_latch_for_pid(pid: pg_sys::pid_t) {
    let proc = pg_sys::BackendPidGetProc(pid);
    if !proc.is_null() {
        pg_sys::SetLatch(&mut (*proc).procLatch);
    }
}

#[pg_extern]
pub fn load_bgworker(name: &str) -> bool {
    if name.is_empty() || name.len() > 63 {
        pgrx::error!("name must be 1..63 chars");
    }

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
        if queue_idx < 0 {
            pgrx::error!("no free worker slots (MAX_WORKERS={})", MAX_WORKERS);
        }
    }

    let mut builder = BackgroundWorkerBuilder::new("tensor_bgworker");
    let builder = builder
        .set_function("tensor_worker_main")
        .set_library("pgtensor") // <-- macro, not function
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
            let mut g = SHMEM.exclusive();
            g.workers[queue_idx as usize] = WorkerEntry::empty();
            pgrx::error!("RegisterDynamicBackgroundWorker failed (max_worker_processes?)");
        }
    }
}

#[pg_extern]
pub fn to_bgworker(name: &str, t: Tensor) -> Tensor {
    // find worker
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
        pgrx::error!("bgworker \"{}\" not running", name);
    }

    // serialize
    let req =
        serde_cbor::to_vec(&t).unwrap_or_else(|e| pgrx::error!("CBOR serialize failed: {}", e));
    if req.len() > MAX_BYTES {
        pgrx::error!(
            "tensor too large for queue ({} > {} bytes)",
            req.len(),
            MAX_BYTES
        );
    }
    let req_id = next_request_id();
    let my_pid = unsafe { pg_sys::MyProcPid as pg_sys::pid_t };

    let slot_idx = {
        let mut g = SHMEM.exclusive();
        let ring = &mut g.queues[qidx];
        if ring.is_full() {
            pgrx::error!("worker \"{}\" queue full", name);
        }
        let idx = Ring::idx(ring.head);
        let m = &mut ring.slots[idx];
        m.id = req_id;
        m.caller_pid = my_pid;
        m.state = MsgState::Pending;
        m.req_len = req.len() as u16;
        m.resp_len = 0;
        m.req[..req.len()].copy_from_slice(&req);
        ring.head = ring.head.wrapping_add(1);
        idx
    };

    unsafe {
        set_latch_for_pid(pid);
    }

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

#[pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn tensor_worker_main(_arg: pg_sys::Datum) {
    let extra = BackgroundWorker::get_extra(); // "idx:name"
    let (qidx, wname) = {
        let mut it = extra.splitn(2, ':');
        let idx = it.next().unwrap_or("0").parse::<usize>().unwrap_or(0);
        let nm = it.next().unwrap_or("unnamed").to_owned();
        (idx, nm)
    };

    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    BackgroundWorker::connect_worker_to_spi(Some("postgres"), None);

    log!("tensor_worker \"{}\" started on queue {}", wname, qidx);

    while BackgroundWorker::wait_latch(Some(Duration::from_millis(200))) {
        loop {
            // pop one message if available
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
            let t: Tensor = match serde_cbor::from_slice(&msg.req[..msg.req_len as usize]) {
                Ok(v) => v,
                Err(e) => {
                    log!("tensor_worker decode error: {}", e);
                    continue;
                }
            };

            // do shit to the tensor
            // we are now fully in rust land here, can call a better abtracted mod here
            let mut t_plus = t.clone();
            for x in &mut t_plus.elem_buffer {
                *x += 1.0;
            }

            // loggin the baille for now to see
            let _ = BackgroundWorker::transaction(|| {
                Spi::connect(|_| {
                    let s: String = t.clone().into();
                    log!("tensor_worker[{}]: {}", wname, s);
                    Ok::<(), pgrx::spi::Error>(())
                })
            });

            let out = serde_cbor::to_vec(&t_plus).unwrap();

            // write response back to the same slot (tail-1)
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
