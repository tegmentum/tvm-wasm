//! Guest-side runtime for `tvm-guest-mm`'s multi-memory shell.
//!
//! ## What this is
//!
//! `tvm-guest-mm` (in the sibling crate) emits a self-contained WAT
//! shell that declares N memory pools plus a fixed set of dispatch
//! helpers (`tvm_load_u32`, `tvm_store_u32`, `tvm_copy_to_default`, …)
//! that are the only way to address a non-default memory from a runtime
//! `pool_id`. Today rustc cannot emit native multi-memory instructions
//! from Rust source (see `examples/multimem-probe/INVESTIGATION.md`).
//!
//! `tvm-guest-mm-rt` is the **Rust-source binding** to those helpers.
//! A Rust `cdylib` that uses this crate compiles to a core wasm with
//! `(import "tvm_mm" "tvm_load_u32" …)` and friends. The companion
//! linker in `tvm-guest-mm-link` then composes that core wasm with the
//! shell into a single self-contained component — see
//! `crates/tvm-guest-mm/docs/rust-cdylib.md` for the full pipeline.
//!
//! ## Quick start (guest crate)
//!
//! ```ignore
//! use tvm_guest_mm_rt::{Pool, load_u32, store_u32, copy_to_default};
//!
//! const PCACHE_HOT: Pool = Pool::new(1);
//! const PCACHE_WARM: Pool = Pool::new(2);
//!
//! #[no_mangle]
//! pub extern "C" fn write_page(pool: u32, off: u32, v: u32) {
//!     let p = Pool::new(pool);
//!     store_u32(p, off, v);
//! }
//!
//! #[no_mangle]
//! pub extern "C" fn read_page(pool: u32, off: u32) -> u32 {
//!     load_u32(Pool::new(pool), off)
//! }
//! ```
//!
//! ## Naming convention
//!
//! Every helper exported by `tvm_guest_mm_module_template(…)` is bound
//! here under the same identifier, with the wasm-level `tvm_` prefix
//! dropped (so `tvm_load_u32` → `load_u32`). For the per-pool
//! specialized helpers (`tvm_load_u32_pK`), the Rust binding picks the
//! right import at runtime via the generic dispatcher because there is
//! no clean way to dispatch to one of N statically-named imports from
//! Rust without knowing N upfront. Workloads that need the
//! dispatch-free hot path should call the generic dispatcher with a
//! known-constant pool — wasmtime constant-folds the BST when the pool
//! is a constant at call site.
//!
//! ## Safety
//!
//! Each dispatch helper is a wasm function exposed by the shell. The
//! shell traps on out-of-range pool ids (single comparison up front).
//! Out-of-range offsets within a valid pool produce a regular wasm OOB
//! trap. Both are surface-level traps — no UB on the Rust side.
//!
//! ## Host stubs
//!
//! When compiled for a non-wasm32 target (e.g. for unit-testing the
//! containing crate on the host triple), every binding returns a zero
//! / no-op value. This lets `cargo test` for a host-target build pass
//! without forcing a wasm runtime in the harness.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

/// Pool identifier — a wasm memory index, 0..N where N is the shell's
/// declared pool count. Pool 0 is the workload's default memory; pools
/// 1..N are pure data storage.
///
/// `Pool` is a transparent newtype over `u32` and is `const`-able so
/// you can declare workload-specific pool aliases:
///
/// ```ignore
/// const HOT: Pool = Pool::new(0);
/// const WARM: Pool = Pool::new(1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Pool(pub u32);

impl Pool {
    /// The workload's default memory. Always pool 0.
    pub const DEFAULT: Pool = Pool(0);

    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn id(self) -> u32 {
        self.0
    }
}

// ---------- Raw imports ----------

#[cfg(target_arch = "wasm32")]
mod raw {
    #[link(wasm_import_module = "tvm_mm")]
    extern "C" {
        pub fn tvm_load_u8(pool: u32, off: u32) -> u32;
        pub fn tvm_load_u32(pool: u32, off: u32) -> u32;
        pub fn tvm_load_i64(pool: u32, off: u32) -> i64;
        pub fn tvm_store_u8(pool: u32, off: u32, val: u32);
        pub fn tvm_store_u32(pool: u32, off: u32, val: u32);
        pub fn tvm_store_i64(pool: u32, off: u32, val: i64);
        pub fn tvm_copy_to_default(src_pool: u32, src_off: u32, dst_off: u32, len: u32);
        pub fn tvm_copy_from_default(dst_pool: u32, dst_off: u32, src_off: u32, len: u32);
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod raw {
    // Host stubs. Lets the crate compile on aarch64-apple-darwin /
    // x86_64-linux for unit tests of code that doesn't actually need
    // multi-memory at test time. Behavior is intentionally trivial:
    // load_* returns 0, store_*/copy_* are no-ops.
    pub unsafe fn tvm_load_u8(_: u32, _: u32) -> u32 {
        0
    }
    pub unsafe fn tvm_load_u32(_: u32, _: u32) -> u32 {
        0
    }
    pub unsafe fn tvm_load_i64(_: u32, _: u32) -> i64 {
        0
    }
    pub unsafe fn tvm_store_u8(_: u32, _: u32, _: u32) {}
    pub unsafe fn tvm_store_u32(_: u32, _: u32, _: u32) {}
    pub unsafe fn tvm_store_i64(_: u32, _: u32, _: i64) {}
    pub unsafe fn tvm_copy_to_default(_: u32, _: u32, _: u32, _: u32) {}
    pub unsafe fn tvm_copy_from_default(_: u32, _: u32, _: u32, _: u32) {}
}

// ---------- Safe wrappers ----------
//
// Each wrapper is a one-line forward to the import. The shell traps on
// out-of-range pool ids and offsets, so these are safe to call from
// safe Rust — failure modes are well-defined wasm traps, not Rust UB.

/// Load a single byte from `(pool, off)`.
#[inline]
pub fn load_u8(pool: Pool, off: u32) -> u8 {
    (unsafe { raw::tvm_load_u8(pool.0, off) } & 0xff) as u8
}

/// Load a u32 from `(pool, off)`. Little-endian on the wire (wasm-native).
#[inline]
pub fn load_u32(pool: Pool, off: u32) -> u32 {
    unsafe { raw::tvm_load_u32(pool.0, off) }
}

/// Load an i64 from `(pool, off)`. Little-endian on the wire.
#[inline]
pub fn load_i64(pool: Pool, off: u32) -> i64 {
    unsafe { raw::tvm_load_i64(pool.0, off) }
}

/// Store a single byte to `(pool, off)`.
#[inline]
pub fn store_u8(pool: Pool, off: u32, val: u8) {
    unsafe { raw::tvm_store_u8(pool.0, off, val as u32) }
}

/// Store a u32 to `(pool, off)`.
#[inline]
pub fn store_u32(pool: Pool, off: u32, val: u32) {
    unsafe { raw::tvm_store_u32(pool.0, off, val) }
}

/// Store an i64 to `(pool, off)`.
#[inline]
pub fn store_i64(pool: Pool, off: u32, val: i64) {
    unsafe { raw::tvm_store_i64(pool.0, off, val) }
}

/// Copy `len` bytes from `(src_pool, src_off)` into the default memory
/// at `dst_off`. Idiomatic for materializing a record from a cold pool
/// into the workload's default heap before processing.
#[inline]
pub fn copy_to_default(src_pool: Pool, src_off: u32, dst_off: u32, len: u32) {
    unsafe { raw::tvm_copy_to_default(src_pool.0, src_off, dst_off, len) }
}

/// Copy `len` bytes from `(default memory, src_off)` into
/// `(dst_pool, dst_off)`. Symmetric to `copy_to_default`. Idiomatic
/// for writing a freshly-prepared record from default into long-lived
/// storage in a data pool.
#[inline]
pub fn copy_from_default(dst_pool: Pool, dst_off: u32, src_off: u32, len: u32) {
    unsafe { raw::tvm_copy_from_default(dst_pool.0, dst_off, src_off, len) }
}

// ---------- Higher-level convenience ----------
//
// `read_bytes` and `write_bytes` reduce byte-level R/W to a single
// call by staging through the default memory. Inputs/outputs live in
// regular `&[u8]` / `&mut [u8]` slices, which already live in the
// default memory by definition.
//
// These avoid the per-byte trip through `load_u8` / `store_u8` (which
// would be 4-6 BST compares per byte on a 64-pool shell). For payloads
// > a few bytes, the bulk-copy path is strictly faster.

/// Read `dst.len()` bytes from `(pool, off)` into `dst` (which lives
/// in the default memory). One call into the shell regardless of size.
///
/// Pool 0 is the default memory — copying from pool 0 to the default
/// is degenerate; callers should slice directly in that case. We still
/// emit the dispatch for uniformity.
#[inline]
pub fn read_bytes(pool: Pool, off: u32, dst: &mut [u8]) {
    let dst_ptr = dst.as_mut_ptr() as u32;
    copy_to_default(pool, off, dst_ptr, dst.len() as u32);
}

/// Write `src` (which lives in the default memory) to `(pool, off)`.
/// One call into the shell.
#[inline]
pub fn write_bytes(pool: Pool, off: u32, src: &[u8]) {
    let src_ptr = src.as_ptr() as u32;
    copy_from_default(pool, off, src_ptr, src.len() as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Host-side smoke tests: the stubs are no-ops, so we just verify
    // the API compiles with the right shape and the const Pool helpers
    // behave.

    #[test]
    fn pool_const_helpers() {
        const P0: Pool = Pool::DEFAULT;
        const P5: Pool = Pool::new(5);
        assert_eq!(P0.id(), 0);
        assert_eq!(P5.id(), 5);
        assert_ne!(P0, P5);
    }

    #[test]
    fn host_stubs_compile() {
        // On the host these are no-ops; we just check the call sites
        // type-check.
        let p = Pool::new(1);
        store_u32(p, 0, 42);
        let _ = load_u32(p, 0);
        store_u8(p, 0, 7);
        let _ = load_u8(p, 0);
        store_i64(p, 0, -1);
        let _ = load_i64(p, 0);
        let mut buf = [0u8; 16];
        read_bytes(p, 0, &mut buf);
        write_bytes(p, 0, &[1, 2, 3]);
        copy_to_default(p, 0, 0, 16);
        copy_from_default(p, 0, 0, 16);
    }
}
