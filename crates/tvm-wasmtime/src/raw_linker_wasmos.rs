//! ADR-0029 Phase 6.9.a — the wasmos-runtime-api-backed equivalent of
//! [`crate::raw_linker`]. Registers the exact same `tvm.*` core-wasm import
//! surface, but through [`wasmos_runtime_api::CoreImports`] +
//! [`wasmos_runtime_api::CoreImportFn`] instead of wasmtime's `Linker` +
//! `Caller`. That makes the raw path portable across every wasmos-backed
//! runtime (wasmtime v48, wasmtime edge, WAMR), not just wasmtime.
//!
//! # Concurrency model
//!
//! The wasmtime raw_linker path expects `T: AsMut<TvmHost>` — the host lives
//! in the `Store<T>`'s data slot, so each handler grabs exclusive `&mut
//! TvmHost` access via `Caller::data_mut()` with zero locking. That model
//! doesn't fit `CoreImports`: an import fn is a stateless `Arc<dyn
//! CoreImportFn>` that any executor can call from any tokio task, without a
//! Store<T> to reach through.
//!
//! Under wasmos the raw path therefore uses [`SharedTvmHost`] — an
//! `Arc<Mutex<TvmHost>>` clone that each handler captures at registration
//! time. Every call takes the mutex, does its work, drops it. This matches
//! the semantics of the wasmtime `add_raw_shared` peer (see raw_linker.rs
//! lines 715+); we're using the "shared" concurrency model uniformly rather
//! than trying to bolt Caller-style exclusive access onto a stateless
//! trait-object dispatch path.
//!
//! Perf note: the mutex is per-call; the wasmtime path has no lock. For hot
//! loops this is a real cost. Phase 6.9.b is scheduled to benchmark this
//! and add a zero-copy fast path via `with_guest_memory_mut` if needed.
//! Correctness first here.
//!
//! # Handler inventory
//!
//! 26 handlers total, matching `raw_linker::add_raw_imports_with_memory_name`
//! 1:1. 21 are region-only (never touch guest memory); 5 memory-touching
//! (`read`, `write`, `read_gather`, `index_of`, `byte_histogram`). See the
//! module-level docstring in `raw_linker.rs` for the wire-level API. All
//! error codes are re-exports from `raw_linker`, so both paths report
//! identical results for identical error inputs.

use std::sync::Arc;

use async_trait::async_trait;
use tvm_core::{Handle, TvmError};
use wasmos_runtime_api::{
    CoreImportContext, CoreImportFn, CoreImports, CoreValue, RuntimeError, RuntimeResult,
};

use crate::raw_linker::{
    err_code, ERR_ALLOC_FAILED, ERR_GUEST_MEMORY, ERR_NOT_RESIDENT, ERR_OK, ERR_OTHER, ERR_PINNED,
    ERR_REGION_NOT_FOUND, ERR_STALE_HANDLE,
};
use crate::shared_host::SharedTvmHost;

// Re-export error codes so consumers can `use tvm_wasmtime::raw_linker_wasmos::*`
// and get the full public surface — mirrors the raw_linker module.
pub use crate::raw_linker::{
    ERR_ALLOC_FAILED as WASMOS_ERR_ALLOC_FAILED, ERR_GUEST_MEMORY as WASMOS_ERR_GUEST_MEMORY,
    ERR_NOT_RESIDENT as WASMOS_ERR_NOT_RESIDENT, ERR_OK as WASMOS_ERR_OK,
    ERR_OTHER as WASMOS_ERR_OTHER, ERR_OUT_OF_BOUNDS as WASMOS_ERR_OUT_OF_BOUNDS,
    ERR_PINNED as WASMOS_ERR_PINNED, ERR_REGION_NOT_FOUND as WASMOS_ERR_REGION_NOT_FOUND,
    ERR_STALE_HANDLE as WASMOS_ERR_STALE_HANDLE,
};

// Silence the unused-import warning triggered by the eight `use ERR_*` on
// symbols only referenced through the `pub use` re-exports above. Keeping
// the direct import block matches the shape used elsewhere in the crate and
// keeps grep-locality for the error taxonomy.
#[allow(dead_code)]
const _UNUSED: [i32; 8] = [
    ERR_ALLOC_FAILED,
    ERR_GUEST_MEMORY,
    ERR_NOT_RESIDENT,
    ERR_OK,
    ERR_OTHER,
    ERR_PINNED,
    ERR_REGION_NOT_FOUND,
    ERR_STALE_HANDLE,
];

// ── arg-shape guards ────────────────────────────────────────────────
//
// Each handler validates its argument list before doing work; a shape
// mismatch means the guest module was linked against a stale contract or
// the executor mis-routed the call. We surface a descriptive
// `RuntimeError::msg` in that case rather than returning a numeric error
// code — those codes are reserved for domain errors (region not found,
// out of bounds, etc.) that the guest expects to inspect.

fn arg_i32(name: &'static str, args: &[CoreValue], idx: usize) -> RuntimeResult<i32> {
    match args.get(idx) {
        Some(CoreValue::I32(v)) => Ok(*v),
        other => Err(RuntimeError::msg(format!(
            "tvm.{name}: arg {idx}: expected I32, got {other:?}",
        ))),
    }
}

fn arg_i64(name: &'static str, args: &[CoreValue], idx: usize) -> RuntimeResult<i64> {
    match args.get(idx) {
        Some(CoreValue::I64(v)) => Ok(*v),
        other => Err(RuntimeError::msg(format!(
            "tvm.{name}: arg {idx}: expected I64, got {other:?}",
        ))),
    }
}

fn expect_arity(name: &'static str, args: &[CoreValue], expected: usize) -> RuntimeResult<()> {
    if args.len() != expected {
        return Err(RuntimeError::msg(format!(
            "tvm.{name}: expected {expected} args, got {}",
            args.len()
        )));
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────
// Region-only handlers (21). Body pattern: lock host, delegate to
// TvmHost method, translate error to `err_code` + `last_raw_error`.
// ────────────────────────────────────────────────────────────────────

struct TvmAlloc {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmAlloc {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("alloc", &args, 2)?;
        let region_id = arg_i32("alloc", &args, 0)?;
        let size = arg_i32("alloc", &args, 1)?;
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret = match host.directory.alloc(region_id as u16, size as u32) {
            Ok(h) => {
                host.cache.invalidate(region_id as u16);
                h.pack() as i64
            }
            Err(e) => {
                host.last_raw_error = err_code(&e);
                0
            }
        };
        Ok(vec![CoreValue::I64(ret)])
    }
}

struct TvmDealloc {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmDealloc {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("dealloc", &args, 1)?;
        let packed = arg_i64("dealloc", &args, 0)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let ret = match g.directory.dealloc(h) {
            Ok(()) => ERR_OK,
            Err(e) => err_code(&e),
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmCopyRegion {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmCopyRegion {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("copy_region", &args, 5)?;
        let src_region = arg_i32("copy_region", &args, 0)?;
        let src_off = arg_i32("copy_region", &args, 1)?;
        let dst_region = arg_i32("copy_region", &args, 2)?;
        let dst_off = arg_i32("copy_region", &args, 3)?;
        let len = arg_i32("copy_region", &args, 4)?;
        let mut g = self.host.lock();
        let ret = match g.directory.cross_region_copy(
            src_region as u16,
            src_off as u32,
            dst_region as u16,
            dst_off as u32,
            len as u32,
        ) {
            Ok(()) => ERR_OK,
            Err(e) => err_code(&e),
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmLastError {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmLastError {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("last_error", &args, 0)?;
        let mut g = self.host.lock();
        let ret = std::mem::replace(&mut g.last_raw_error, ERR_OK);
        Ok(vec![CoreValue::I32(ret)])
    }
}

// ── reducer handlers ─────────────────────────────────────────────────
// All share: lock host, call region_*, translate result. Body varies
// only in the region_* method + the return-value packing.

/// Boilerplate for "reducer that returns i64 with -1 sentinel on err".
macro_rules! reducer_i64_neg1 {
    ($ty:ident, $wire_name:literal, $call:ident) => {
        struct $ty {
            host: SharedTvmHost,
        }
        #[async_trait]
        impl CoreImportFn for $ty {
            async fn call(
                &self,
                _ctx: &mut CoreImportContext<'_>,
                args: Vec<CoreValue>,
            ) -> RuntimeResult<Vec<CoreValue>> {
                expect_arity($wire_name, &args, 2)?;
                let packed = arg_i64($wire_name, &args, 0)?;
                let len = arg_i32($wire_name, &args, 1)?;
                let h = Handle::unpack(packed as u64);
                let mut g = self.host.lock();
                let host = &mut *g;
                let ret = match host.$call(h, len as u32) {
                    Ok(v) => v as i64,
                    Err(e) => {
                        host.last_raw_error = err_code(&e);
                        -1
                    }
                };
                Ok(vec![CoreValue::I64(ret)])
            }
        }
    };
}

/// Boilerplate for "reducer that returns i32 with -1 sentinel on err".
macro_rules! reducer_i32_neg1 {
    ($ty:ident, $wire_name:literal, $call:ident, $ok:ident => $ok_expr:expr) => {
        struct $ty {
            host: SharedTvmHost,
        }
        #[async_trait]
        impl CoreImportFn for $ty {
            async fn call(
                &self,
                _ctx: &mut CoreImportContext<'_>,
                args: Vec<CoreValue>,
            ) -> RuntimeResult<Vec<CoreValue>> {
                expect_arity($wire_name, &args, 2)?;
                let packed = arg_i64($wire_name, &args, 0)?;
                let len = arg_i32($wire_name, &args, 1)?;
                let h = Handle::unpack(packed as u64);
                let mut g = self.host.lock();
                let host = &mut *g;
                let ret: i32 = match host.$call(h, len as u32) {
                    Ok($ok) => $ok_expr,
                    Err(e) => {
                        host.last_raw_error = err_code(&e);
                        -1
                    }
                };
                Ok(vec![CoreValue::I32(ret)])
            }
        }
    };
}

reducer_i64_neg1!(TvmSumU8, "sum_u8", region_sum_u8);
reducer_i64_neg1!(TvmPopcount, "popcount", region_popcount);

reducer_i32_neg1!(TvmAndFoldU8, "and_fold_u8", region_and_fold_u8, v => v as i32);
reducer_i32_neg1!(TvmOrFoldU8, "or_fold_u8", region_or_fold_u8, v => v as i32);
reducer_i32_neg1!(TvmXorFoldU8, "xor_fold_u8", region_xor_fold_u8, v => v as i32);

/// hash_fnv1a returns i64; on err sets last_raw_error and returns 0.
struct TvmHashFnv1a {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmHashFnv1a {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("hash_fnv1a", &args, 2)?;
        let packed = arg_i64("hash_fnv1a", &args, 0)?;
        let len = arg_i32("hash_fnv1a", &args, 1)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret = match host.region_hash_fnv1a(h, len as u32) {
            Ok(v) => v as i64,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                0
            }
        };
        Ok(vec![CoreValue::I64(ret)])
    }
}

struct TvmFindByte {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmFindByte {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("find_byte", &args, 3)?;
        let packed = arg_i64("find_byte", &args, 0)?;
        let len = arg_i32("find_byte", &args, 1)?;
        let byte = arg_i32("find_byte", &args, 2)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_find_byte(h, len as u32, byte as u8) {
            Ok(Some(off)) => off as i32,
            Ok(None) => -1,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -2
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmCountByte {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmCountByte {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("count_byte", &args, 3)?;
        let packed = arg_i64("count_byte", &args, 0)?;
        let len = arg_i32("count_byte", &args, 1)?;
        let byte = arg_i32("count_byte", &args, 2)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_count_byte(h, len as u32, byte as u8) {
            Ok(c) => c as i32,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -1
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmEq {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmEq {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("eq", &args, 3)?;
        let packed_a = arg_i64("eq", &args, 0)?;
        let packed_b = arg_i64("eq", &args, 1)?;
        let len = arg_i32("eq", &args, 2)?;
        let ha = Handle::unpack(packed_a as u64);
        let hb = Handle::unpack(packed_b as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_eq(ha, hb, len as u32) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -1
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmMinMaxU8 {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmMinMaxU8 {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("min_max_u8", &args, 2)?;
        let packed = arg_i64("min_max_u8", &args, 0)?;
        let len = arg_i32("min_max_u8", &args, 1)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_min_max_u8(h, len as u32) {
            Ok((lo, hi)) => ((lo as i32) << 8) | (hi as i32),
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -1
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmXorIntoRegion {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmXorIntoRegion {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("xor_into_region", &args, 3)?;
        let packed_src = arg_i64("xor_into_region", &args, 0)?;
        let packed_dst = arg_i64("xor_into_region", &args, 1)?;
        let len = arg_i32("xor_into_region", &args, 2)?;
        let src = Handle::unpack(packed_src as u64);
        let dst = Handle::unpack(packed_dst as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_xor_into_region(src, dst, len as u32) {
            Ok(()) => ERR_OK,
            Err(e) => err_code(&e),
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmSumU32Le {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmSumU32Le {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("sum_u32_le", &args, 2)?;
        let packed = arg_i64("sum_u32_le", &args, 0)?;
        let len = arg_i32("sum_u32_le", &args, 1)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i64 = match host.region_sum_u32_le(h, len as u32) {
            Ok(s) if s <= i64::MAX as u128 => s as i64,
            Ok(_) => {
                host.last_raw_error = err_code(&TvmError::OutOfBounds);
                -1
            }
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -1
            }
        };
        Ok(vec![CoreValue::I64(ret)])
    }
}

struct TvmMaxU32Le {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmMaxU32Le {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("max_u32_le", &args, 2)?;
        let packed = arg_i64("max_u32_le", &args, 0)?;
        let len = arg_i32("max_u32_le", &args, 1)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i64 = match host.region_max_u32_le(h, len as u32) {
            Ok(Some(v)) => v as i64,
            Ok(None) => -2,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -1
            }
        };
        Ok(vec![CoreValue::I64(ret)])
    }
}

struct TvmCountInRange {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmCountInRange {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("count_in_range", &args, 4)?;
        let packed = arg_i64("count_in_range", &args, 0)?;
        let len = arg_i32("count_in_range", &args, 1)?;
        let lo = arg_i32("count_in_range", &args, 2)?;
        let hi = arg_i32("count_in_range", &args, 3)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_count_in_range(h, len as u32, lo as u8, hi as u8) {
            Ok(c) => c as i32,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -1
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmLexCmp {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmLexCmp {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("lex_cmp", &args, 3)?;
        let packed_a = arg_i64("lex_cmp", &args, 0)?;
        let packed_b = arg_i64("lex_cmp", &args, 1)?;
        let len = arg_i32("lex_cmp", &args, 2)?;
        let ha = Handle::unpack(packed_a as u64);
        let hb = Handle::unpack(packed_b as u64);
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_lex_cmp(ha, hb, len as u32) {
            Ok(core::cmp::Ordering::Less) => -1,
            Ok(core::cmp::Ordering::Equal) => 0,
            Ok(core::cmp::Ordering::Greater) => 1,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -2
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmFill {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmFill {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("fill", &args, 3)?;
        let packed = arg_i64("fill", &args, 0)?;
        let len = arg_i32("fill", &args, 1)?;
        let byte = arg_i32("fill", &args, 2)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let ret: i32 = match g.region_fill(h, len as u32, byte as u8) {
            Ok(()) => ERR_OK,
            Err(e) => err_code(&e),
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

struct TvmXorWithByte {
    host: SharedTvmHost,
}

#[async_trait]
impl CoreImportFn for TvmXorWithByte {
    async fn call(
        &self,
        _ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("xor_with_byte", &args, 3)?;
        let packed = arg_i64("xor_with_byte", &args, 0)?;
        let len = arg_i32("xor_with_byte", &args, 1)?;
        let byte = arg_i32("xor_with_byte", &args, 2)?;
        let h = Handle::unpack(packed as u64);
        let mut g = self.host.lock();
        let ret: i32 = match g.region_xor_with_byte(h, len as u32, byte as u8) {
            Ok(()) => ERR_OK,
            Err(e) => err_code(&e),
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

// ────────────────────────────────────────────────────────────────────
// Memory-touching handlers (5). Each does at least one
// `ctx.guest_memory_{read,write}` call. Uses safe scratch-buffer path;
// zero-copy via `with_guest_memory_mut` is a Phase 6.9.b tuning item.
// ────────────────────────────────────────────────────────────────────

/// tvm.read: read `len` bytes from region into guest memory at `dst_ptr`.
struct TvmRead {
    host: SharedTvmHost,
    memory_name: &'static str,
}

#[async_trait]
impl CoreImportFn for TvmRead {
    async fn call(
        &self,
        ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("read", &args, 3)?;
        let packed = arg_i64("read", &args, 0)?;
        let dst_ptr = arg_i32("read", &args, 1)?;
        let len = arg_i32("read", &args, 2)?;
        let h = Handle::unpack(packed as u64);
        let len_us = len as usize;
        // Read region into scratch, then copy scratch to guest memory.
        let mut scratch = vec![0u8; len_us];
        {
            let mut g = self.host.lock();
            let host = &mut *g;
            if let Err(e) = host.directory.read(h, &mut scratch) {
                host.last_raw_error = err_code(&e);
                return Ok(vec![CoreValue::I32(err_code(&e))]);
            }
        }
        if ctx
            .guest_memory_write(self.memory_name, dst_ptr as u64, &scratch)
            .is_err()
        {
            return Ok(vec![CoreValue::I32(ERR_GUEST_MEMORY)]);
        }
        Ok(vec![CoreValue::I32(ERR_OK)])
    }
}

/// tvm.write: read `len` bytes from guest memory at `src_ptr` and write to region.
struct TvmWrite {
    host: SharedTvmHost,
    memory_name: &'static str,
}

#[async_trait]
impl CoreImportFn for TvmWrite {
    async fn call(
        &self,
        ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("write", &args, 3)?;
        let packed = arg_i64("write", &args, 0)?;
        let src_ptr = arg_i32("write", &args, 1)?;
        let len = arg_i32("write", &args, 2)?;
        let h = Handle::unpack(packed as u64);
        let len_us = len as usize;
        let mut scratch = vec![0u8; len_us];
        if ctx
            .guest_memory_read(self.memory_name, src_ptr as u64, &mut scratch)
            .is_err()
        {
            return Ok(vec![CoreValue::I32(ERR_GUEST_MEMORY)]);
        }
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.directory.write(h, &scratch) {
            Ok(()) => ERR_OK,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                err_code(&e)
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

/// tvm.read_gather: strided/scatter read of `count` items of size `item_size`.
/// Indices live in guest memory at `indices_ptr` (LE u32s). Results are
/// concatenated into guest memory at `dst_ptr`.
struct TvmReadGather {
    host: SharedTvmHost,
    memory_name: &'static str,
}

#[async_trait]
impl CoreImportFn for TvmReadGather {
    async fn call(
        &self,
        ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("read_gather", &args, 5)?;
        let packed = arg_i64("read_gather", &args, 0)?;
        let indices_ptr = arg_i32("read_gather", &args, 1)?;
        let count = arg_i32("read_gather", &args, 2)?;
        let item_size = arg_i32("read_gather", &args, 3)?;
        let dst_ptr = arg_i32("read_gather", &args, 4)?;
        let h = Handle::unpack(packed as u64);
        let count_us = count as usize;
        let item = item_size as usize;

        // Fetch the LE-encoded indices from guest memory.
        let mut indices_bytes = vec![0u8; count_us * 4];
        if ctx
            .guest_memory_read(self.memory_name, indices_ptr as u64, &mut indices_bytes)
            .is_err()
        {
            return Ok(vec![CoreValue::I32(ERR_GUEST_MEMORY)]);
        }
        let mut indices = Vec::with_capacity(count_us);
        for i in 0..count_us {
            indices.push(u32::from_le_bytes([
                indices_bytes[i * 4],
                indices_bytes[i * 4 + 1],
                indices_bytes[i * 4 + 2],
                indices_bytes[i * 4 + 3],
            ]));
        }

        // Detect arithmetic-progression indices for a fused bulk read
        // (matches raw_linker.rs's optimization).
        let dense_contiguous = if count_us >= 2 {
            let stride = indices[1].wrapping_sub(indices[0]);
            stride as usize == item
                && (1..count_us).all(|k| indices[k].wrapping_sub(indices[k - 1]) == stride)
        } else {
            count_us == 1
        };

        let mut scratch = vec![0u8; count_us * item];
        {
            let mut g = self.host.lock();
            let host = &mut *g;
            if dense_contiguous && count_us > 0 {
                let cell = Handle {
                    offset: h.offset.wrapping_add(indices[0]),
                    ..h
                };
                if let Err(e) = host.directory.read(cell, &mut scratch) {
                    return Ok(vec![CoreValue::I32(err_code(&e))]);
                }
            } else {
                for i in 0..count_us {
                    let off = indices[i];
                    let cell = Handle {
                        offset: h.offset.wrapping_add(off),
                        ..h
                    };
                    if let Err(e) = host
                        .directory
                        .read(cell, &mut scratch[i * item..(i + 1) * item])
                    {
                        return Ok(vec![CoreValue::I32(err_code(&e))]);
                    }
                }
            }
        }

        if ctx
            .guest_memory_write(self.memory_name, dst_ptr as u64, &scratch)
            .is_err()
        {
            return Ok(vec![CoreValue::I32(ERR_GUEST_MEMORY)]);
        }
        Ok(vec![CoreValue::I32(ERR_OK)])
    }
}

/// tvm.index_of: search `region[..len]` for `needle` (living in guest memory).
struct TvmIndexOf {
    host: SharedTvmHost,
    memory_name: &'static str,
}

#[async_trait]
impl CoreImportFn for TvmIndexOf {
    async fn call(
        &self,
        ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("index_of", &args, 4)?;
        let packed = arg_i64("index_of", &args, 0)?;
        let len = arg_i32("index_of", &args, 1)?;
        let needle_ptr = arg_i32("index_of", &args, 2)?;
        let needle_len = arg_i32("index_of", &args, 3)?;
        let h = Handle::unpack(packed as u64);
        if !(0..=4096).contains(&needle_len) {
            return Ok(vec![CoreValue::I32(-2)]);
        }
        let needle_n = needle_len as usize;
        let mut needle = vec![0u8; needle_n];
        if ctx
            .guest_memory_read(self.memory_name, needle_ptr as u64, &mut needle)
            .is_err()
        {
            return Ok(vec![CoreValue::I32(ERR_GUEST_MEMORY)]);
        }
        let mut g = self.host.lock();
        let host = &mut *g;
        let ret: i32 = match host.region_index_of(h, len as u32, &needle) {
            Ok(Some(off)) => off as i32,
            Ok(None) => -1,
            Err(e) => {
                host.last_raw_error = err_code(&e);
                -2
            }
        };
        Ok(vec![CoreValue::I32(ret)])
    }
}

/// tvm.byte_histogram: compute 256-bucket u8 histogram (LE u32 per bucket)
/// over region and write 1024-byte result to guest memory at `out_ptr`.
struct TvmByteHistogram {
    host: SharedTvmHost,
    memory_name: &'static str,
}

#[async_trait]
impl CoreImportFn for TvmByteHistogram {
    async fn call(
        &self,
        ctx: &mut CoreImportContext<'_>,
        args: Vec<CoreValue>,
    ) -> RuntimeResult<Vec<CoreValue>> {
        expect_arity("byte_histogram", &args, 3)?;
        let packed = arg_i64("byte_histogram", &args, 0)?;
        let len = arg_i32("byte_histogram", &args, 1)?;
        let out_ptr = arg_i32("byte_histogram", &args, 2)?;
        let h = Handle::unpack(packed as u64);
        let mut buf = [0u8; 1024];
        {
            let mut g = self.host.lock();
            let host = &mut *g;
            if let Err(e) = host.region_byte_histogram(h, len as u32, &mut buf) {
                return Ok(vec![CoreValue::I32(err_code(&e))]);
            }
        }
        if ctx
            .guest_memory_write(self.memory_name, out_ptr as u64, &buf)
            .is_err()
        {
            return Ok(vec![CoreValue::I32(ERR_GUEST_MEMORY)]);
        }
        Ok(vec![CoreValue::I32(ERR_OK)])
    }
}

// ────────────────────────────────────────────────────────────────────
// Public entry points — match `raw_linker::add_raw_imports*` shape but
// consume + return `CoreImports` (fluent builder style, matching the
// wasmos idiom used elsewhere in the codebase and in the wasmos test
// suites). Consumers thread the returned `CoreImports` into their
// `ExecutionContext`.
// ────────────────────────────────────────────────────────────────────

/// Register the raw `tvm.*` import surface, expecting the guest's memory
/// to be exported as `"memory"`. See [`add_raw_imports_with_memory_name`]
/// for the custom-name variant.
pub fn add_raw_imports(imports: CoreImports, host: SharedTvmHost) -> CoreImports {
    add_raw_imports_with_memory_name(imports, host, "memory")
}

/// Register the raw `tvm.*` import surface, expecting the guest's memory
/// to be exported under a custom name (e.g. Emscripten toolchains).
///
/// `memory_name` must be a `&'static str` because handlers capture it by
/// value. To register against a non-static name, use `Box::leak` or clone
/// the string into a `String` inside a bespoke handler set.
pub fn add_raw_imports_with_memory_name(
    imports: CoreImports,
    host: SharedTvmHost,
    memory_name: &'static str,
) -> CoreImports {
    imports
        // Region-only handlers.
        .register(
            "tvm",
            "alloc",
            Arc::new(TvmAlloc { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "dealloc",
            Arc::new(TvmDealloc { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "copy_region",
            Arc::new(TvmCopyRegion { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "last_error",
            Arc::new(TvmLastError { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "sum_u8",
            Arc::new(TvmSumU8 { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "find_byte",
            Arc::new(TvmFindByte { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "hash_fnv1a",
            Arc::new(TvmHashFnv1a { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "count_byte",
            Arc::new(TvmCountByte { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "eq",
            Arc::new(TvmEq { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "min_max_u8",
            Arc::new(TvmMinMaxU8 { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "xor_into_region",
            Arc::new(TvmXorIntoRegion { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "sum_u32_le",
            Arc::new(TvmSumU32Le { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "max_u32_le",
            Arc::new(TvmMaxU32Le { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "and_fold_u8",
            Arc::new(TvmAndFoldU8 { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "or_fold_u8",
            Arc::new(TvmOrFoldU8 { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "xor_fold_u8",
            Arc::new(TvmXorFoldU8 { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "count_in_range",
            Arc::new(TvmCountInRange { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "lex_cmp",
            Arc::new(TvmLexCmp { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "popcount",
            Arc::new(TvmPopcount { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "fill",
            Arc::new(TvmFill { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "xor_with_byte",
            Arc::new(TvmXorWithByte { host: host.clone() }) as Arc<dyn CoreImportFn>,
        )
        // Memory-touching handlers.
        .register(
            "tvm",
            "read",
            Arc::new(TvmRead {
                host: host.clone(),
                memory_name,
            }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "write",
            Arc::new(TvmWrite {
                host: host.clone(),
                memory_name,
            }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "read_gather",
            Arc::new(TvmReadGather {
                host: host.clone(),
                memory_name,
            }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "index_of",
            Arc::new(TvmIndexOf {
                host: host.clone(),
                memory_name,
            }) as Arc<dyn CoreImportFn>,
        )
        .register(
            "tvm",
            "byte_histogram",
            Arc::new(TvmByteHistogram { host, memory_name }) as Arc<dyn CoreImportFn>,
        )
}

// ────────────────────────────────────────────────────────────────────
// `add_raw_shared*` — API-symmetric aliases (Phase 6.9.a Session 2).
//
// Under wasmtime, `add_raw_shared*` and `add_raw_imports*` are
// separate function surfaces because the store data type differs
// (`Store<T: AsMut<SharedTvmHost>>` vs `Store<T: AsMut<TvmHost>>`).
// The shared batch additionally fetches guest memory *uncached* on
// every call — the wasmtime non-shared batch caches a pointer inside
// `TvmHost.cached_memory` per store, and sharing that host across
// stores lets store B use store A's cached pointer (memory
// corruption). See raw_linker.rs docstring on `add_raw_shared` for
// the full rationale.
//
// Under wasmos, both hazards vanish by construction:
//
// * `CoreImports` handlers are stateless `Arc<dyn CoreImportFn>`
//   objects that carry their own captured `SharedTvmHost` — there's
//   no Store<T> to reach through, so the "shared" concurrency model
//   is the ONLY model available. That's what
//   `add_raw_imports_with_memory_name` above already does.
// * `ctx.guest_memory_{read,write,size}` fetches the memory from the
//   currently-executing instance via the adapter (v48's
//   `wasmtime::Caller` or WAMR's TLS-guarded `wamrx::Instance`) on
//   every call — no pointer is cached across calls, let alone across
//   instances.
//
// So the wasmos abstraction unified the two paths. These aliases
// exist to keep the migration ergonomic: consumers doing `add_raw_
// shared(&mut linker)?` under wasmtime can search-replace to
// `add_raw_shared(imports, host)` under wasmos without changing the
// function name, and the semantics they wanted (cross-store safety +
// shared host) hold.

/// Alias for [`add_raw_imports`] — API symmetric with the wasmtime
/// `add_raw_shared` entry point (`raw_linker.rs` line 715+). See the
/// module-level docstring for why the two paths unify under wasmos.
pub fn add_raw_shared(imports: CoreImports, host: SharedTvmHost) -> CoreImports {
    add_raw_imports(imports, host)
}

/// Alias for [`add_raw_imports_with_memory_name`] — API symmetric
/// with the wasmtime `add_raw_shared_with_memory_name` entry point.
pub fn add_raw_shared_with_memory_name(
    imports: CoreImports,
    host: SharedTvmHost,
    memory_name: &'static str,
) -> CoreImports {
    add_raw_imports_with_memory_name(imports, host, memory_name)
}
