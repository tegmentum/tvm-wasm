//! Companion consumer that mixes `tvm_mm` substrate imports with
//! arbitrary host-supplied imports. Exercises
//! `tvm-guest-mm-link`'s import-forwarding path: the non-`tvm_mm`
//! imports survive the link step and are satisfied at instantiation
//! time by the embedder (wasmtime `Linker::define_func`, browser
//! `WebAssembly.instantiate(... importObject)`, etc.).
//!
//! Shape:
//!
//!   imports:
//!     "tvm_mm"."tvm_store_u32"
//!     "tvm_mm"."tvm_load_u32"
//!     "host"."log"          ← forwarded
//!     "host"."now_nanos"    ← forwarded
//!
//!   exports:
//!     "put_with_log" — store a u32 to pool 1 and call host.log
//!     "get"          — load a u32 from pool 1
//!     "stamped_put"  — store host.now_nanos() into pool 1 then call host.log
//!
//! Mirrors the shape sqlite-pcache-tvm hits in production: substrate
//! data-plane through `tvm_mm.*`, observability + clock through a
//! separate host-supplied module.

#![no_std]

use tvm_guest_mm_rt::{load_u32, store_u32, Pool};

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

const POOL: Pool = Pool::new(1);

// ---------- Host-supplied imports (forwarded by the linker) ----------
//
// These come from a separate import module (`host`) — not `tvm_mm`.
// The linker's old behavior was to bail when encountering these; the
// forwarding behavior preserves them in the merged module's import
// section so the embedder satisfies them at instantiation.

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "host")]
extern "C" {
    /// Host-side log callback: `(ptr, len)` of a UTF-8 byte string in
    /// the workload's default memory.
    fn log(ptr: u32, len: u32);
    /// Host-side monotonic clock in nanoseconds.
    fn now_nanos() -> i64;
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn log(_ptr: u32, _len: u32) {}
#[cfg(not(target_arch = "wasm32"))]
unsafe fn now_nanos() -> i64 {
    0
}

// ---------- Workload exports ----------

/// Stores `value` at `off` in pool 1 then asks the host to log a fixed
/// message. The string lives in the workload's default memory (rustc's
/// data section).
#[no_mangle]
pub extern "C" fn put_with_log(off: u32, value: u32) {
    store_u32(POOL, off, value);
    // Slice points into the rustc data section — i.e. memory 0 in the
    // user cdylib, which is pool 0 of the shell in the merged module.
    // The host log receives `(ptr, len)` against memory 0.
    let msg: &[u8] = b"put";
    unsafe { log(msg.as_ptr() as u32, msg.len() as u32) };
}

#[no_mangle]
pub extern "C" fn get(off: u32) -> u32 {
    load_u32(POOL, off)
}

/// Stamps `host.now_nanos()` into pool 1 at the given offset, then
/// emits a log line. Demonstrates that both a value-returning and a
/// void host import survive the link step.
#[no_mangle]
pub extern "C" fn stamped_put(off: u32) -> i64 {
    let ts = unsafe { now_nanos() };
    // High 32 bits.
    store_u32(POOL, off, (ts >> 32) as u32);
    // Low 32 bits.
    store_u32(POOL, off.wrapping_add(4), ts as u32);
    let msg: &[u8] = b"stamped";
    unsafe { log(msg.as_ptr() as u32, msg.len() as u32) };
    ts
}
