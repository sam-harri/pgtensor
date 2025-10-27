//! tensor_bg.rs — shm_mq-backed background inference workers for Postgres (pgrx)

use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use pgrx::bgworkers::*;
use pgrx::pg_sys;
use pgrx::pg_sys::shm_mq_result::{SHM_MQ_DETACHED, SHM_MQ_SUCCESS, SHM_MQ_WOULD_BLOCK};
use pgrx::prelude::*;
use std::cell::UnsafeCell;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::tensor_core::Tensor; // your existing Tensor type

// ------------------------------
// config
// ------------------------------

const MAX_WORKERS: usize = 4;

// shm_toc keys/magic
const TOC_MAGIC: u64 = 0x5445_4E53; // "TENS"
const KEY_INBOUND_MQ: u64 = 1;
const KEY_RESPONSE_MQ: u64 = 1;

// MVP: overallocate DSM by this much for TOC/header
const DSM_HEADROOM: usize = 8 * 1024;

// ------------------------------
// shared memory traits + LWLock wrapper
// ------------------------------

pub trait PGRXSharedMemory: Copy + 'static {}
impl<T: Copy + 'static> PGRXSharedMemory for T {}

pub trait PgSharedMemoryInitialization {
    type Value: PGRXSharedMemory;
    unsafe fn on_shmem_request(&'static self);
    unsafe fn on_shmem_startup(&'static self, value: Self::Value);
}

#[repr(C)]
struct Shared<T> {
    data: UnsafeCell<T>,
    lock: *mut pg_sys::LWLock,
}

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
            let shared = self.inner.get().read().as_ref().expect("PgLwLock not initialized");
            pg_sys::LWLockAcquire(shared.lock, pg_sys::LWLockMode::LW_SHARED);
            PgLwLockShareGuard {
                data: &*shared.data.get(),
                lock: shared.lock,
            }
        }
    }
    pub fn exclusive(&self) -> PgLwLockExclusiveGuard<'_, T> {
        unsafe {
            let shared = self.inner.get().read().as_ref().expect("PgLwLock not initialized");
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
        unsafe { pg_sys::LWLockRelease(self.lock) }
    }
}

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
        unsafe { pg_sys::LWLockRelease(self.lock) }
    }
}

// ------------------------------
// control-plane structures (no ring buffer)
// ------------------------------

#[repr(C)]
#[derive(Copy, Clone)]
struct WorkerEntry {
    in_use: bool,
    pid: pg_sys::pid_t,
    inbound_handle: pg_sys::dsm_handle, // DSM handle containing inbound shm_mq
    _pad0: u32,
    busy: bool, // one-sender-at-a-time gate
    queue_idx: i32,
    name: [u8; 64],
}
impl WorkerEntry {
    const fn empty() -> Self {
        Self {
            in_use: false,
            pid: 0,
            inbound_handle: 0,
            _pad0: 0,
            busy: false,
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
        let len = self.name.iter().position(|&c| c == 0).unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..len]).unwrap_or("")
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SharedState {
    workers: [WorkerEntry; MAX_WORKERS],
}
impl SharedState {
    const fn new() -> Self {
        const W: WorkerEntry = WorkerEntry::empty();
        Self { workers: [W; MAX_WORKERS] }
    }
}

const LW_NAME: &CStr = c"tensor_bg";
#[allow(non_upper_case_globals)]
static SHMEM: PgLwLock<SharedState> = unsafe { PgLwLock::new(LW_NAME) };

// ------------------------------
// hooks (pg13-18)
// ------------------------------

#[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
mod shmem_hooks {
    use super::*;
    static mut PREV_SHMEM_REQUEST_HOOK: Option<unsafe extern "C-unwind" fn()> = None;
    static mut PREV_SHMEM_STARTUP_HOOK: Option<unsafe extern "C-unwind" fn()> = None;

    #[pg_guard]
    pub extern "C-unwind" fn _PG_init() {
        if unsafe { !pg_sys::process_shared_preload_libraries_in_progress } {
            pgrx::error!("tensor_bg must be loaded via shared_preload_libraries");
        }
        unsafe {
            PREV_SHMEM_REQUEST_HOOK = pg_sys::shmem_request_hook;
            pg_sys::shmem_request_hook = Some(shmem_request);
            PREV_SHMEM_STARTUP_HOOK = pg_sys::shmem_startup_hook;
            pg_sys::shmem_startup_hook = Some(shmem_startup);
        }
    }

    #[pg_guard]
    unsafe extern "C-unwind" fn shmem_request() {
        if let Some(prev) = PREV_SHMEM_REQUEST_HOOK {
            prev();
        }
        SHMEM.on_shmem_request();
    }

    #[pg_guard]
    unsafe extern "C-unwind" fn shmem_startup() {
        if let Some(prev) = PREV_SHMEM_STARTUP_HOOK {
            prev();
        }
        SHMEM.on_shmem_startup(SharedState::new());
    }
}

#[cfg(any(feature = "pg13", feature = "pg14"))]
mod shmem_hooks {
    use super::*;
    static mut PREV_SHMEM_STARTUP_HOOK: Option<unsafe extern "C-unwind" fn()> = None;

    #[pg_guard]
    pub extern "C-unwind" fn _PG_init() {
        if unsafe { !pg_sys::process_shared_preload_libraries_in_progress } {
            pgrx::error!("tensor_bg must be loaded via shared_preload_libraries");
        }
        unsafe { SHMEM.on_shmem_request(); }
        unsafe {
            PREV_SHMEM_STARTUP_HOOK = pg_sys::shmem_startup_hook;
            pg_sys::shmem_startup_hook = Some(shmem_startup);
        }
    }

    #[pg_guard]
    unsafe extern "C-unwind" fn shmem_startup() {
        if let Some(prev) = PREV_SHMEM_STARTUP_HOOK {
            prev();
        }
        SHMEM.on_shmem_startup(SharedState::new());
    }
}

pub use shmem_hooks::_PG_init;

// ------------------------------
// request id + latch helper
// ------------------------------

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
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

// ------------------------------
// model path helpers (unchanged)
// ------------------------------

fn models_base_dir() -> std::path::PathBuf {
    use std::path::PathBuf;
    if let Ok(s) = std::env::var("PGTENSOR_MODELS_DIR") {
        return PathBuf::from(s);
    }
    PathBuf::from("/var/lib/postgresql/pgtensor_model")
}
fn resolve_model_path(rel: &str) -> std::path::PathBuf {
    use std::path::{Path, PathBuf};
    let base = models_base_dir();
    let p = Path::new(rel);
    if p.is_absolute() || rel.contains("..") {
        pgrx::error!("invalid model path: must be relative and not contain '..'");
    }
    base.join(p)
}

// ------------------------------
// shm_mq helpers (header)
// ------------------------------

#[repr(C)]
#[derive(Copy, Clone)]
struct RequestHeader {
    resp_handle: pg_sys::dsm_handle,
    req_id: u64,
}

// ------------------------------
// SQL: load_model (create per-worker inbound DSM+MQ; start worker)
// ------------------------------

#[pgrx::pg_extern(immutable, strict, parallel_safe, requires = ["cast_tensor_to_tensor"])]
pub fn load_model(name: &str, input_var: &str, output_var: &str) -> bool {
    if name.is_empty() || name.len() > 63 || name.contains(':') {
        pgrx::error!("name must be 1..63 chars and cannot contain ':'");
    }
    if input_var.contains(':') || output_var.contains(':') {
        pgrx::error!("input/output names cannot contain ':'");
    }

    let model_path = resolve_model_path(&format!("{}.onnx", name));
    if !model_path.exists() {
        pgrx::error!("model file not found: {}", model_path.display());
    }

    // claim free slot
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
                w.inbound_handle = 0;
                w.busy = false;
                w.set_name(name);
                queue_idx = i as i32;
                break;
            }
        }
        if queue_idx < 0 {
            pgrx::error!("no free worker slots (MAX_WORKERS={})", MAX_WORKERS);
        }
    }

    // Create inbound DSM + shm_mq; pin to keep it alive until worker attaches
    let inbound_seg = unsafe {
        let qsize: usize = 256 * 1024;
        let seg = pg_sys::dsm_create((qsize + DSM_HEADROOM).next_power_of_two(), 0);
        if seg.is_null() {
            pgrx::error!("dsm_create failed for worker inbound mq");
        }
        let addr = pg_sys::dsm_segment_address(seg);
        let segsz = pg_sys::dsm_segment_map_length(seg) as usize;
        let toc = pg_sys::shm_toc_create(TOC_MAGIC, addr, segsz);
        let space = pg_sys::shm_toc_allocate(toc, qsize);
        let _mq = pg_sys::shm_mq_create(space, qsize);
        pg_sys::shm_toc_insert(toc, KEY_INBOUND_MQ, space);
        pg_sys::dsm_pin_segment(seg);
        seg
    };
    let inbound_handle = unsafe { pg_sys::dsm_segment_handle(inbound_seg) };

    // publish handle
    {
        let mut g = SHMEM.exclusive();
        g.workers[queue_idx as usize].inbound_handle = inbound_handle;
    }

    // Start worker
    let mut builder = BackgroundWorkerBuilder::new("tensor_bgworker");
    let builder = builder
        .set_function("inference_worker_main")
        .set_library("pgtensor") // adjust if your lib name differs
        .set_extra(&format!("{}:{}:{}:{}", queue_idx, name, input_var, output_var))
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
            {
                let mut g = SHMEM.exclusive();
                g.workers[queue_idx as usize].pid = pid;
            }
            unsafe { pg_sys::dsm_detach(inbound_seg) }; // still pinned
            true
        }
        Err(_) => {
            let mut g = SHMEM.exclusive();
            g.workers[queue_idx as usize] = WorkerEntry::empty();
            unsafe { pg_sys::dsm_detach(inbound_seg) };
            pgrx::error!("RegisterDynamicBackgroundWorker failed (max_worker_processes?)");
        }
    }
}

// ------------------------------
// SQL: run_inference (per-call response DSM + send header||body)
// ------------------------------

#[pgrx::pg_extern(immutable, strict, parallel_safe, requires = ["cast_tensor_to_tensor"])]
pub fn run_inference(name: &str, t: Tensor) -> Tensor {
    let (idx, pid, inbound_handle) = {
        let g = SHMEM.share();
        let w = g
            .workers
            .iter()
            .find(|w| w.in_use && w.name_str() == name)
            .unwrap_or_else(|| pgrx::error!("bgworker \"{}\" not found", name));
        if w.pid <= 0 {
            pgrx::error!("bgworker \"{}\" not running", name);
        }
        (w.queue_idx as usize, w.pid, w.inbound_handle)
    };

    {
        let mut g = SHMEM.exclusive();
        let w = &mut g.workers[idx];
        if w.busy {
            pgrx::error!("worker \"{}\" is busy; try again", name);
        }
        w.busy = true;
    }

    // per-call response DSM + mq (we are the receiver)
    let (resp_seg, resp_handle, resp_mqh) = unsafe {
        let qsize: usize = 256 * 1024;
        let seg = pg_sys::dsm_create((qsize + DSM_HEADROOM).next_power_of_two(), 0);
        if seg.is_null() {
            pgrx::error!("dsm_create failed for response mq");
        }
        let addr = pg_sys::dsm_segment_address(seg);
        let segsz = pg_sys::dsm_segment_map_length(seg) as usize;
        let toc = pg_sys::shm_toc_create(TOC_MAGIC, addr, segsz);
        let space = pg_sys::shm_toc_allocate(toc, qsize);
        let mq = pg_sys::shm_mq_create(space, qsize);
        pg_sys::shm_mq_set_receiver(mq, pg_sys::MyProc);
        pg_sys::shm_toc_insert(toc, KEY_RESPONSE_MQ, space);
        let mqh = pg_sys::shm_mq_attach(mq, seg, std::ptr::null_mut());
        pg_sys::shm_mq_wait_for_attach(mqh);
        let handle = pg_sys::dsm_segment_handle(seg);
        (seg, handle, mqh)
    };

    // serialize tensor
    let body =
        serde_cbor::to_vec(&t).unwrap_or_else(|e| pgrx::error!("CBOR serialize failed: {}", e));
    let req_id = next_request_id();

    // send [header||body], then wait for response
    let result: Tensor;
    unsafe {
        let in_seg = pg_sys::dsm_attach(inbound_handle);
        if in_seg.is_null() {
            cleanup_busy(idx);
            pg_sys::dsm_detach(resp_seg);
            pgrx::error!("failed to dsm_attach inbound handle for worker");
        }
        let in_addr = pg_sys::dsm_segment_address(in_seg);
        let in_toc = pg_sys::shm_toc_attach(TOC_MAGIC, in_addr);
        let in_space = pg_sys::shm_toc_lookup(in_toc, KEY_INBOUND_MQ, false);
        if in_space.is_null() {
            pg_sys::dsm_detach(in_seg);
            cleanup_busy(idx);
            pg_sys::dsm_detach(resp_seg);
            pgrx::error!("inbound shm_mq not found in worker DSM");
        }

        let in_mq = in_space as *mut pg_sys::shm_mq;
        pg_sys::shm_mq_set_sender(in_mq, pg_sys::MyProc);
        let in_mqh = pg_sys::shm_mq_attach(in_mq, in_seg, std::ptr::null_mut());

        // header
        let hdr = RequestHeader { resp_handle, req_id };
        let hdr_ptr = &hdr as *const RequestHeader as *const u8;
        let hdr_bytes = std::slice::from_raw_parts(hdr_ptr, size_of::<RequestHeader>());

        // combine header + body (MVP; could use sendv)
        let mut buf = Vec::with_capacity(hdr_bytes.len() + body.len());
        buf.extend_from_slice(hdr_bytes);
        buf.extend_from_slice(&body);

        let res = pg_sys::shm_mq_send(in_mqh, buf.len(), buf.as_ptr() as *const _, false, true);
        if res != SHM_MQ_SUCCESS {
            pg_sys::dsm_detach(in_seg);
            cleanup_busy(idx);
            pg_sys::dsm_detach(resp_seg);
            pgrx::error!("shm_mq_send failed (result={})", res);
        }

        set_latch_for_pid(pid);

        let mut n: pg_sys::Size = 0;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        loop {
            let r = pg_sys::shm_mq_receive(resp_mqh, &mut n, &mut p, false);
            if r == SHM_MQ_SUCCESS {
                break;
            }
            if r == SHM_MQ_DETACHED {
                pg_sys::dsm_detach(in_seg);
                pg_sys::dsm_detach(resp_seg);
                cleanup_busy(idx);
                pgrx::error!("response queue detached");
            }
            pg_sys::ResetLatch(pg_sys::MyLatch);
            let rc = pg_sys::WaitLatch(
                pg_sys::MyLatch,
                (pg_sys::WL_LATCH_SET | pg_sys::WL_TIMEOUT | pg_sys::WL_POSTMASTER_DEATH) as i32,
                50,
                pg_sys::PG_WAIT_EXTENSION,
            );
            if (rc & pg_sys::WL_POSTMASTER_DEATH as i32) != 0 {
                pg_sys::dsm_detach(in_seg);
                pg_sys::dsm_detach(resp_seg);
                cleanup_busy(idx);
                pgrx::error!("postmaster died");
            }
        }

        let resp = std::slice::from_raw_parts(p as *const u8, n as usize);
        result = serde_cbor::from_slice(resp)
            .unwrap_or_else(|e| {
                pg_sys::dsm_detach(in_seg);
                pg_sys::dsm_detach(resp_seg);
                cleanup_busy(idx);
                pgrx::error!("CBOR decode failed: {}", e)
            });

        pg_sys::dsm_detach(in_seg);
        pg_sys::dsm_detach(resp_seg);
        cleanup_busy(idx);
    }

    result
}

fn cleanup_busy(idx: usize) {
    let mut g = SHMEM.exclusive();
    g.workers[idx].busy = false;
}

// ------------------------------
// BGWORKER MAIN: receive via inbound mq, run, reply via caller's response mq
// ------------------------------

#[pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn inference_worker_main(_arg: pg_sys::Datum) {
    use crate::onnx_runtime::InferenceSession;

    let extra = BackgroundWorker::get_extra();
    // expected: "qidx:name:input:output"
    let (qidx, model_name, input_var, output_var) = {
        let mut it = extra.splitn(4, ':');
        let idx = it.next().unwrap_or("0").parse::<usize>().unwrap_or(0);
        let name = it.next().unwrap_or("unnamed").to_owned();
        let ivar = it.next().unwrap_or("x").to_owned();
        let ovar = it.next().unwrap_or("y").to_owned();
        (idx, name, ivar, ovar)
    };

    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    BackgroundWorker::connect_worker_to_spi(Some("postgres"), None);

    let model_path = resolve_model_path(&format!("{}.onnx", model_name));
    log!(
        "inference_worker \"{}\" starting on slot {} with model {}",
        model_name,
        qidx,
        model_path.display()
    );

    let mut session = match InferenceSession::new(&model_path, &input_var, &output_var) {
        Ok(s) => s,
        Err(e) => {
            log!("inference_worker[{}]: failed to create session: {}", model_name, e);
            return;
        }
    };

    // Attach to inbound mq (from SHMEM)
    let (in_mqh, in_seg) = unsafe {
        let g = SHMEM.share();
        let w = g.workers[qidx];
        drop(g);

        let seg = pg_sys::dsm_attach(w.inbound_handle);
        if seg.is_null() {
            log!("inference_worker[{}]: dsm_attach inbound failed", model_name);
            return;
        }
        let addr = pg_sys::dsm_segment_address(seg);
        let toc = pg_sys::shm_toc_attach(TOC_MAGIC, addr);
        let space = pg_sys::shm_toc_lookup(toc, KEY_INBOUND_MQ, false);
        if space.is_null() {
            log!("inference_worker[{}]: inbound mq not found", model_name);
            pg_sys::dsm_detach(seg);
            return;
        }
        let mq = space as *mut pg_sys::shm_mq;
        pg_sys::shm_mq_set_receiver(mq, pg_sys::MyProc);
        let mqh = pg_sys::shm_mq_attach(mq, seg, std::ptr::null_mut());
        pg_sys::shm_mq_wait_for_attach(mqh);
        (mqh, seg)
    };

    while BackgroundWorker::wait_latch(Some(Duration::from_millis(200))) {
        unsafe {
            let mut n: pg_sys::Size = 0;
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();

            let r = pg_sys::shm_mq_receive(in_mqh, &mut n, &mut p, true);
            if r == SHM_MQ_WOULD_BLOCK {
                continue;
            }
            if r != SHM_MQ_SUCCESS {
                if r == SHM_MQ_DETACHED {
                    break;
                }
                continue;
            }

            let msg = std::slice::from_raw_parts(p as *const u8, n as usize);
            if msg.len() < size_of::<RequestHeader>() {
                log!("inference_worker[{}]: short message", model_name);
                continue;
            }

            // parse header
            let mut hdr = RequestHeader {
                resp_handle: 0,
                req_id: 0,
            };
            std::ptr::copy_nonoverlapping(
                msg.as_ptr(),
                &mut hdr as *mut RequestHeader as *mut u8,
                size_of::<RequestHeader>(),
            );
            let tensor_bytes = &msg[size_of::<RequestHeader>()..];

            // attach caller's response DSM/mq
            let resp_seg = pg_sys::dsm_attach(hdr.resp_handle);
            if resp_seg.is_null() {
                log!("inference_worker[{}]: dsm_attach(resp) failed", model_name);
                continue;
            }
            let raddr = pg_sys::dsm_segment_address(resp_seg);
            let rtoc = pg_sys::shm_toc_attach(TOC_MAGIC, raddr);
            let rspace = pg_sys::shm_toc_lookup(rtoc, KEY_RESPONSE_MQ, false);
            if rspace.is_null() {
                log!("inference_worker[{}]: response mq missing", model_name);
                pg_sys::dsm_detach(resp_seg);
                continue;
            }
            let rmq = rspace as *mut pg_sys::shm_mq;
            pg_sys::shm_mq_set_sender(rmq, pg_sys::MyProc);
            let rmqh = pg_sys::shm_mq_attach(rmq, resp_seg, std::ptr::null_mut());
            pg_sys::shm_mq_wait_for_attach(rmqh);

            // run inference
            let input: Tensor = match serde_cbor::from_slice(tensor_bytes) {
                Ok(v) => v,
                Err(e) => {
                    log!("inference_worker[{}] decode error: {}", model_name, e);
                    pg_sys::dsm_detach(resp_seg);
                    continue;
                }
            };
            let output = match session.infer(&input) {
                Ok(o) => o,
                Err(e) => {
                    log!("inference_worker[{}] infer error: {}", model_name, e);
                    pg_sys::dsm_detach(resp_seg);
                    continue;
                }
            };
            let out_bytes = match serde_cbor::to_vec(&output) {
                Ok(v) => v,
                Err(e) => {
                    log!("inference_worker[{}] encode error: {}", model_name, e);
                    pg_sys::dsm_detach(resp_seg);
                    continue;
                }
            };

            // send back
            let sr =
                pg_sys::shm_mq_send(rmqh, out_bytes.len(), out_bytes.as_ptr() as *const _, false, true);
            if sr != SHM_MQ_SUCCESS {
                log!("inference_worker[{}] shm_mq_send resp failed ({})", model_name, sr);
            }
            pg_sys::dsm_detach(resp_seg);
        }
    }

    unsafe { pg_sys::dsm_detach(in_seg) }
    log!("inference_worker \"{}\" shutting down", model_name);
}
