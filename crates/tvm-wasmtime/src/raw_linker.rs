//! Raw core-wasm imports — the fast path that skips the component-model
//! canonical ABI entirely. See `docs/fast-paths.md` for the full tradeoff
//! analysis and recommended use.
//!
//! Module name: `tvm`. Functions:
//!   tvm.alloc   (region_id: i32, size: i32) -> i64    // packed handle, 0 on err
//!   tvm.dealloc (handle: i64)               -> i32    // 0 ok, !=0 error code
//!   tvm.read    (handle: i64, dst_ptr: i32, len: i32) -> i32
//!   tvm.write   (handle: i64, src_ptr: i32, len: i32) -> i32
//!   tvm.copy_region (src_region: i32, src_off: i32,
//!                    dst_region: i32, dst_off: i32, len: i32) -> i32
//!   tvm.last_error () -> i32                       // last error code on this thread
//!
//! Error codes (single-digit so they fit in i32 trivially):
//!   0 = ok
//!   1 = region not found
//!   2 = stale handle
//!   3 = out of bounds
//!   4 = not resident
//!   5 = allocation failed
//!   6 = pinned
//!   7 = guest memory missing or write failed
//!   9 = other
//!
//! The guest is expected to export a memory named `memory`. If your toolchain
//! exports a different name, use [`add_raw_imports_with_memory_name`].

use tvm_core::{Handle, TvmError};
use wasmtime::{Caller, Extern, Linker, Memory};

use crate::host::TvmHost;
use crate::shared_host::SharedTvmHost;

pub const ERR_OK: i32 = 0;
pub const ERR_REGION_NOT_FOUND: i32 = 1;
pub const ERR_STALE_HANDLE: i32 = 2;
pub const ERR_OUT_OF_BOUNDS: i32 = 3;
pub const ERR_NOT_RESIDENT: i32 = 4;
pub const ERR_ALLOC_FAILED: i32 = 5;
pub const ERR_PINNED: i32 = 6;
pub const ERR_GUEST_MEMORY: i32 = 7;
pub const ERR_OTHER: i32 = 9;

fn err_code(e: &TvmError) -> i32 {
    match e {
        TvmError::RegionNotFound(_) => ERR_REGION_NOT_FOUND,
        TvmError::StaleHandle => ERR_STALE_HANDLE,
        TvmError::OutOfBounds => ERR_OUT_OF_BOUNDS,
        TvmError::NotResident => ERR_NOT_RESIDENT,
        TvmError::AllocationFailed => ERR_ALLOC_FAILED,
        TvmError::Pinned => ERR_PINNED,
        TvmError::BackingStore(_) | TvmError::UnsupportedAllocator | TvmError::PolicyViolation => {
            ERR_OTHER
        }
    }
}

fn guest_memory<T>(caller: &mut Caller<'_, T>) -> Option<Memory> {
    match caller.get_export("memory") {
        Some(Extern::Memory(m)) => Some(m),
        _ => None,
    }
}

fn guest_memory_named<T>(caller: &mut Caller<'_, T>, name: &str) -> Option<Memory> {
    match caller.get_export(name) {
        Some(Extern::Memory(m)) => Some(m),
        _ => None,
    }
}

/// Cached fetch: populates `host.cached_memory` on first call, returns the
/// stored handle on subsequent calls. Saves the per-call HashMap lookup
/// inside `Caller::get_export`.
fn cached_guest_memory<T>(caller: &mut Caller<'_, T>, name: &'static str) -> Option<Memory>
where
    T: AsMut<TvmHost>,
{
    if let Some(m) = caller.data_mut().as_mut().cached_memory.memory {
        return Some(m);
    }
    let m = guest_memory_named(caller, name)?;
    caller.data_mut().as_mut().cached_memory.memory = Some(m);
    Some(m)
}

/// Returns (memory_handle, base_ptr, size_bytes), refreshing the cached
/// size/ptr from the live wasmtime memory if the cache is stale (zero) or
/// if the access we're about to do exceeds the cached size (memory may
/// have been grown).
#[inline]
fn cached_guest_memory_view<T>(
    caller: &mut Caller<'_, T>,
    name: &'static str,
    required_end: usize,
) -> Option<(Memory, usize, u64)>
where
    T: AsMut<TvmHost>,
{
    let mem = cached_guest_memory(caller, name)?;
    let host = caller.data_mut().as_mut();
    let mut size = host.cached_memory.size;
    let mut ptr = host.cached_memory.ptr;
    if size == 0 || (required_end as u64) > size {
        size = mem.data_size(&caller) as u64;
        ptr = mem.data_ptr(&caller) as usize;
        let host = caller.data_mut().as_mut();
        host.cached_memory.size = size;
        host.cached_memory.ptr = ptr;
    }
    Some((mem, ptr, size))
}

/// Register raw imports under module name `tvm`, expecting the guest's
/// linear memory to be exported as `memory`.
pub fn add_raw_imports<T>(linker: &mut Linker<T>) -> wasmtime::Result<()>
where
    T: AsMut<TvmHost> + 'static,
{
    add_raw_imports_with_memory_name(linker, "memory")
}

/// Same as `add_raw_imports`, but uses a custom guest memory export name.
pub fn add_raw_imports_with_memory_name<T>(
    linker: &mut Linker<T>,
    memory_name: &'static str,
) -> wasmtime::Result<()>
where
    T: AsMut<TvmHost> + 'static,
{
    linker.func_wrap(
        "tvm",
        "alloc",
        |mut caller: Caller<'_, T>, region_id: i32, size: i32| -> i64 {
            let host = caller.data_mut().as_mut();
            match host.directory.alloc(region_id as u16, size as u32) {
                Ok(h) => {
                    host.cache.invalidate(region_id as u16); // used updated
                    h.pack() as i64
                }
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "dealloc",
        |mut caller: Caller<'_, T>, packed: i64| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.directory.dealloc(h) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "read",
        move |mut caller: Caller<'_, T>, packed: i64, dst_ptr: i32, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let len_us = len as usize;
            let dst_off = dst_ptr as usize;
            let required_end = match dst_off.checked_add(len_us) {
                Some(e) => e,
                None => return ERR_GUEST_MEMORY,
            };
            let (_mem, ptr, _size) =
                match cached_guest_memory_view(&mut caller, memory_name, required_end) {
                    Some(v) => v,
                    None => return ERR_GUEST_MEMORY,
                };
            let host = caller.data_mut().as_mut();
            // SAFETY: ptr+required_end is bounds-checked against memory
            // size in the view fetch; fast_read checks the region side.
            match unsafe { host.fast_read(h, (ptr as *mut u8).add(dst_off), len as u32) } {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "write",
        move |mut caller: Caller<'_, T>, packed: i64, src_ptr: i32, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let len_us = len as usize;
            let src_off = src_ptr as usize;
            let required_end = match src_off.checked_add(len_us) {
                Some(e) => e,
                None => return ERR_GUEST_MEMORY,
            };
            let (_mem, ptr, _size) =
                match cached_guest_memory_view(&mut caller, memory_name, required_end) {
                    Some(v) => v,
                    None => return ERR_GUEST_MEMORY,
                };
            let host = caller.data_mut().as_mut();
            // SAFETY: ptr+required_end is bounds-checked.
            match unsafe { host.fast_write(h, (ptr as *const u8).add(src_off), len as u32) } {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "read_gather",
        move |mut caller: Caller<'_, T>,
              packed: i64,
              indices_ptr: i32,
              count: i32,
              item_size: i32,
              dst_ptr: i32|
              -> i32 {
            let h = Handle::unpack(packed as u64);
            let count_us = count as usize;
            let item = item_size as usize;
            let mem = match cached_guest_memory(&mut caller, memory_name) {
                Some(m) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let mut indices_bytes = vec![0u8; count_us * 4];
            if mem
                .read(&caller, indices_ptr as usize, &mut indices_bytes)
                .is_err()
            {
                return ERR_GUEST_MEMORY;
            }
            // Decode indices.
            let mut indices = Vec::with_capacity(count_us);
            for i in 0..count_us {
                indices.push(u32::from_le_bytes([
                    indices_bytes[i * 4],
                    indices_bytes[i * 4 + 1],
                    indices_bytes[i * 4 + 2],
                    indices_bytes[i * 4 + 3],
                ]));
            }
            // Detect arithmetic-progression indices and specialize when the
            // stride equals item_size — that's a contiguous range, served
            // by a single bulk read instead of count_us per-cell reads.
            let dense_contiguous = if count_us >= 2 {
                let stride = indices[1].wrapping_sub(indices[0]);
                stride as usize == item
                    && (1..count_us).all(|k| indices[k].wrapping_sub(indices[k - 1]) == stride)
            } else {
                count_us == 1
            };
            let mut scratch = vec![0u8; count_us * item];
            {
                let host = caller.data_mut().as_mut();
                if dense_contiguous && count_us > 0 {
                    // One bulk read.
                    let cell = Handle {
                        offset: h.offset.wrapping_add(indices[0]),
                        ..h
                    };
                    if let Err(e) = host.directory.read(cell, &mut scratch) {
                        return err_code(&e);
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
                            return err_code(&e);
                        }
                    }
                }
            }
            // Step 3: write results into guest memory in one write.
            match mem.write(&mut caller, dst_ptr as usize, &scratch) {
                Ok(()) => ERR_OK,
                Err(_) => ERR_GUEST_MEMORY,
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "copy_region",
        |mut caller: Caller<'_, T>,
         src_region: i32,
         src_off: i32,
         dst_region: i32,
         dst_off: i32,
         len: i32|
         -> i32 {
            let host = caller.data_mut().as_mut();
            match host.directory.cross_region_copy(
                src_region as u16,
                src_off as u32,
                dst_region as u16,
                dst_off as u32,
                len as u32,
            ) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap("tvm", "last_error", |mut caller: Caller<'_, T>| -> i32 {
        let host = caller.data_mut().as_mut();
        std::mem::replace(&mut host.last_raw_error, ERR_OK)
    })?;

    // ---- reducer imports: fold a region's bytes to a scalar ----
    //
    // These collapse the "host.read into guest mem, then guest sums it"
    // pattern into a single trampoline. Implementation is plain Rust
    // (autovec'd) over the region's slice; no SIMD wasm sidecar.
    //
    // Result conventions:
    //   sum_u8       -> i64 with the sum (≥0; -1 on error)
    //   find_byte    -> i32 offset, -1 if not found, < -1 reserved errno
    //   hash_fnv1a   -> i64 hash; on error sets last_raw_error and
    //                   returns 0 (rare collision, caller can check
    //                   last_error if 0 is returned)

    linker.func_wrap(
        "tvm",
        "sum_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_sum_u8(h, len as u32) {
                Ok(s) => s as i64,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "find_byte",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_find_byte(h, len as u32, byte as u8) {
                Ok(Some(off)) => off as i32,
                Ok(None) => -1,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -2
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "hash_fnv1a",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_hash_fnv1a(h, len as u32) {
                Ok(v) => v as i64,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "count_byte",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_count_byte(h, len as u32, byte as u8) {
                Ok(c) => c as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "eq",
        |mut caller: Caller<'_, T>, packed_a: i64, packed_b: i64, len: i32| -> i32 {
            let ha = Handle::unpack(packed_a as u64);
            let hb = Handle::unpack(packed_b as u64);
            let host = caller.data_mut().as_mut();
            match host.region_eq(ha, hb, len as u32) {
                Ok(true) => 1,
                Ok(false) => 0,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "min_max_u8",
        // Returns packed (min << 8) | max into low 16 bits; -1 on err.
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_min_max_u8(h, len as u32) {
                Ok((lo, hi)) => ((lo as i32) << 8) | (hi as i32),
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "xor_into_region",
        |mut caller: Caller<'_, T>, packed_src: i64, packed_dst: i64, len: i32| -> i32 {
            let src = Handle::unpack(packed_src as u64);
            let dst = Handle::unpack(packed_dst as u64);
            let host = caller.data_mut().as_mut();
            match host.region_xor_into_region(src, dst, len as u32) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "sum_u32_le",
        // Returns sum as i64 (low 60 bits used). -1 on err. Caller
        // promises len % 4 == 0.
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_sum_u32_le(h, len as u32) {
                Ok(s) if s <= i64::MAX as u128 => s as i64,
                Ok(_) => {
                    host.last_raw_error = err_code(&TvmError::OutOfBounds);
                    -1
                }
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "max_u32_le",
        // Returns the max u32 cast to i64; -1 on err; -2 on empty.
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_max_u32_le(h, len as u32) {
                Ok(Some(v)) => v as i64,
                Ok(None) => -2,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "and_fold_u8",
        // Returns folded byte in low 8 bits; -1 on err.
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_and_fold_u8(h, len as u32) {
                Ok(v) => v as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "or_fold_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_or_fold_u8(h, len as u32) {
                Ok(v) => v as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "xor_fold_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_xor_fold_u8(h, len as u32) {
                Ok(v) => v as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "count_in_range",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, lo: i32, hi: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_count_in_range(h, len as u32, lo as u8, hi as u8) {
                Ok(c) => c as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "lex_cmp",
        // Returns -1/0/1 for Less/Equal/Greater; -2 on err.
        |mut caller: Caller<'_, T>, packed_a: i64, packed_b: i64, len: i32| -> i32 {
            let ha = Handle::unpack(packed_a as u64);
            let hb = Handle::unpack(packed_b as u64);
            let host = caller.data_mut().as_mut();
            match host.region_lex_cmp(ha, hb, len as u32) {
                Ok(core::cmp::Ordering::Less) => -1,
                Ok(core::cmp::Ordering::Equal) => 0,
                Ok(core::cmp::Ordering::Greater) => 1,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -2
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "popcount",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_popcount(h, len as u32) {
                Ok(v) => v as i64,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "fill",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_fill(h, len as u32, byte as u8) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "xor_with_byte",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let host = caller.data_mut().as_mut();
            match host.region_xor_with_byte(h, len as u32, byte as u8) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    // index_of: needle lives in the guest's linear memory at
    // `needle_ptr`, length `needle_len`. Caller must ensure the
    // pointer is valid for that many bytes. Returns offset, -1 if not
    // found, -2 on error.
    linker.func_wrap(
        "tvm",
        "index_of",
        move |mut caller: Caller<'_, T>,
              packed: i64,
              len: i32,
              needle_ptr: i32,
              needle_len: i32|
              -> i32 {
            let h = Handle::unpack(packed as u64);
            if !(0..=4096).contains(&needle_len) {
                return -2;
            }
            let needle_off = needle_ptr as usize;
            let needle_n = needle_len as usize;
            let required_end = match needle_off.checked_add(needle_n) {
                Some(e) => e,
                None => return ERR_GUEST_MEMORY,
            };
            let mem = match cached_guest_memory_view(&mut caller, memory_name, required_end) {
                Some((m, _, _)) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let mut needle = vec![0u8; needle_n];
            if mem.read(&caller, needle_off, &mut needle).is_err() {
                return ERR_GUEST_MEMORY;
            }
            let host = caller.data_mut().as_mut();
            match host.region_index_of(h, len as u32, &needle) {
                Ok(Some(off)) => off as i32,
                Ok(None) => -1,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -2
                }
            }
        },
    )?;

    // Histogram needs to write 1024 bytes (256 u32s LE) into the guest's
    // memory at `out_ptr`. Caller must ensure the destination is long
    // enough.
    linker.func_wrap(
        "tvm",
        "byte_histogram",
        move |mut caller: Caller<'_, T>, packed: i64, len: i32, out_ptr: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let out_off = out_ptr as usize;
            let required_end = match out_off.checked_add(1024) {
                Some(e) => e,
                None => return ERR_GUEST_MEMORY,
            };
            let mem = match cached_guest_memory_view(&mut caller, memory_name, required_end) {
                Some((m, _, _)) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let host = caller.data_mut().as_mut();
            let mut buf = [0u8; 1024];
            if let Err(e) = host.region_byte_histogram(h, len as u32, &mut buf) {
                return err_code(&e);
            }
            if mem.write(&mut caller, out_off, &buf).is_err() {
                return ERR_GUEST_MEMORY;
            }
            ERR_OK
        },
    )?;

    // Suppress unused warning when memory name is the default.
    let _ = guest_memory::<T>;

    Ok(())
}

/// Shared-substrate variant of [`add_raw_imports`] — a **full drop-in
/// peer**: every function `add_raw_imports` defines, made cross-store
/// (cross-actor) correct.
///
/// `add_raw_imports` caches the guest's linear-memory pointer **inside
/// `TvmHost`** (`cached_memory`, per store). Sharing one `TvmHost` across
/// stores would let store B's raw read use store A's cached pointer —
/// memory corruption. This variant therefore, *uniformly* for every call:
///
/// * locks the shared `TvmHost` per call (the directory/region state is the
///   thing we *want* shared) — `let mut g = …lock(); let host = &mut *g;`,
///   after which every body is byte-identical to `add_raw_imports`; and
/// * for the memory-touching calls (`read`/`write`/`read_gather`/`index_of`/
///   `byte_histogram`) fetches the guest memory **uncached** (never touches
///   `cached_memory`), so no per-store pointer crosses stores.
///
/// Honest trade-offs: loses the raw-path memory-pointer micro-cache for
/// shared actors; and `last_raw_error` is now shared state, so concurrent
/// shared actors race on `tvm.last_error()` — the per-call `i32`/`i64`
/// return code remains the primary, correct error channel (this only
/// affects the secondary `last_error()` convenience).
pub fn add_raw_shared<T>(linker: &mut Linker<T>) -> wasmtime::Result<()>
where
    T: AsMut<SharedTvmHost> + Send + 'static,
{
    add_raw_shared_with_memory_name(linker, "memory")
}

/// Same as [`add_raw_shared`], custom guest-memory export name.
pub fn add_raw_shared_with_memory_name<T>(
    linker: &mut Linker<T>,
    memory_name: &'static str,
) -> wasmtime::Result<()>
where
    T: AsMut<SharedTvmHost> + Send + 'static,
{
    linker.func_wrap(
        "tvm",
        "alloc",
        |mut caller: Caller<'_, T>, region_id: i32, size: i32| -> i64 {
            let mut g = caller.data_mut().as_mut().lock();
            match g.directory.alloc(region_id as u16, size as u32) {
                Ok(h) => {
                    g.cache.invalidate(region_id as u16);
                    h.pack() as i64
                }
                Err(e) => {
                    g.last_raw_error = err_code(&e);
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "dealloc",
        |mut caller: Caller<'_, T>, packed: i64| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.directory.dealloc(h) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "read",
        move |mut caller: Caller<'_, T>, packed: i64, dst_ptr: i32, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let dst_off = dst_ptr as usize;
            let required_end = match dst_off.checked_add(len as usize) {
                Some(e) => e,
                None => return ERR_GUEST_MEMORY,
            };
            // Uncached fetch — never write `host.cached_memory` (the
            // cross-store hazard). Borrow of `caller` ends before the lock.
            let mem = match guest_memory_named(&mut caller, memory_name) {
                Some(m) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let size = mem.data_size(&caller) as u64;
            let base = mem.data_ptr(&caller) as usize;
            if (required_end as u64) > size {
                return ERR_GUEST_MEMORY;
            }
            let mut g = caller.data_mut().as_mut().lock();
            // SAFETY: base+required_end bounds-checked vs the live memory
            // size; with `memory_may_move(false)` the base is stable and the
            // guest cannot grow its memory mid host-call; `fast_read` checks
            // the region side.
            match unsafe { g.fast_read(h, (base as *mut u8).add(dst_off), len as u32) } {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "write",
        move |mut caller: Caller<'_, T>, packed: i64, src_ptr: i32, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let src_off = src_ptr as usize;
            let required_end = match src_off.checked_add(len as usize) {
                Some(e) => e,
                None => return ERR_GUEST_MEMORY,
            };
            let mem = match guest_memory_named(&mut caller, memory_name) {
                Some(m) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let size = mem.data_size(&caller) as u64;
            let base = mem.data_ptr(&caller) as usize;
            if (required_end as u64) > size {
                return ERR_GUEST_MEMORY;
            }
            let mut g = caller.data_mut().as_mut().lock();
            // SAFETY: as in `read` above.
            match unsafe { g.fast_write(h, (base as *const u8).add(src_off), len as u32) } {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "read_gather",
        move |mut caller: Caller<'_, T>,
              packed: i64,
              indices_ptr: i32,
              count: i32,
              item_size: i32,
              dst_ptr: i32|
              -> i32 {
            let h = Handle::unpack(packed as u64);
            let count_us = count as usize;
            let item = item_size as usize;
            // Uncached (never `cached_memory`).
            let mem = match guest_memory_named(&mut caller, memory_name) {
                Some(m) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let mut indices_bytes = vec![0u8; count_us * 4];
            if mem
                .read(&caller, indices_ptr as usize, &mut indices_bytes)
                .is_err()
            {
                return ERR_GUEST_MEMORY;
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
            let dense_contiguous = if count_us >= 2 {
                let stride = indices[1].wrapping_sub(indices[0]);
                stride as usize == item
                    && (1..count_us).all(|k| indices[k].wrapping_sub(indices[k - 1]) == stride)
            } else {
                count_us == 1
            };
            let mut scratch = vec![0u8; count_us * item];
            {
                let mut g = caller.data_mut().as_mut().lock();
                let host = &mut *g;
                if dense_contiguous && count_us > 0 {
                    let cell = Handle {
                        offset: h.offset.wrapping_add(indices[0]),
                        ..h
                    };
                    if let Err(e) = host.directory.read(cell, &mut scratch) {
                        return err_code(&e);
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
                            return err_code(&e);
                        }
                    }
                }
            }
            match mem.write(&mut caller, dst_ptr as usize, &scratch) {
                Ok(()) => ERR_OK,
                Err(_) => ERR_GUEST_MEMORY,
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "copy_region",
        |mut caller: Caller<'_, T>,
         src_region: i32,
         src_off: i32,
         dst_region: i32,
         dst_off: i32,
         len: i32|
         -> i32 {
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.directory.cross_region_copy(
                src_region as u16,
                src_off as u32,
                dst_region as u16,
                dst_off as u32,
                len as u32,
            ) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap("tvm", "last_error", |mut caller: Caller<'_, T>| -> i32 {
        let mut g = caller.data_mut().as_mut().lock();
        let host = &mut *g;
        std::mem::replace(&mut host.last_raw_error, ERR_OK)
    })?;

    linker.func_wrap(
        "tvm",
        "sum_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_sum_u8(h, len as u32) {
                Ok(s) => s as i64,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "find_byte",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_find_byte(h, len as u32, byte as u8) {
                Ok(Some(off)) => off as i32,
                Ok(None) => -1,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -2
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "hash_fnv1a",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_hash_fnv1a(h, len as u32) {
                Ok(v) => v as i64,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "count_byte",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_count_byte(h, len as u32, byte as u8) {
                Ok(c) => c as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "eq",
        |mut caller: Caller<'_, T>, packed_a: i64, packed_b: i64, len: i32| -> i32 {
            let ha = Handle::unpack(packed_a as u64);
            let hb = Handle::unpack(packed_b as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_eq(ha, hb, len as u32) {
                Ok(true) => 1,
                Ok(false) => 0,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "min_max_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_min_max_u8(h, len as u32) {
                Ok((lo, hi)) => ((lo as i32) << 8) | (hi as i32),
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "xor_into_region",
        |mut caller: Caller<'_, T>, packed_src: i64, packed_dst: i64, len: i32| -> i32 {
            let src = Handle::unpack(packed_src as u64);
            let dst = Handle::unpack(packed_dst as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_xor_into_region(src, dst, len as u32) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "sum_u32_le",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_sum_u32_le(h, len as u32) {
                Ok(s) if s <= i64::MAX as u128 => s as i64,
                Ok(_) => {
                    host.last_raw_error = err_code(&TvmError::OutOfBounds);
                    -1
                }
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "max_u32_le",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_max_u32_le(h, len as u32) {
                Ok(Some(v)) => v as i64,
                Ok(None) => -2,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "and_fold_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_and_fold_u8(h, len as u32) {
                Ok(v) => v as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "or_fold_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_or_fold_u8(h, len as u32) {
                Ok(v) => v as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "xor_fold_u8",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_xor_fold_u8(h, len as u32) {
                Ok(v) => v as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "count_in_range",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, lo: i32, hi: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_count_in_range(h, len as u32, lo as u8, hi as u8) {
                Ok(c) => c as i32,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "lex_cmp",
        |mut caller: Caller<'_, T>, packed_a: i64, packed_b: i64, len: i32| -> i32 {
            let ha = Handle::unpack(packed_a as u64);
            let hb = Handle::unpack(packed_b as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_lex_cmp(ha, hb, len as u32) {
                Ok(core::cmp::Ordering::Less) => -1,
                Ok(core::cmp::Ordering::Equal) => 0,
                Ok(core::cmp::Ordering::Greater) => 1,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -2
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "popcount",
        |mut caller: Caller<'_, T>, packed: i64, len: i32| -> i64 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_popcount(h, len as u32) {
                Ok(v) => v as i64,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "fill",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_fill(h, len as u32, byte as u8) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "xor_with_byte",
        |mut caller: Caller<'_, T>, packed: i64, len: i32, byte: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_xor_with_byte(h, len as u32, byte as u8) {
                Ok(()) => ERR_OK,
                Err(e) => err_code(&e),
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "index_of",
        move |mut caller: Caller<'_, T>,
              packed: i64,
              len: i32,
              needle_ptr: i32,
              needle_len: i32|
              -> i32 {
            let h = Handle::unpack(packed as u64);
            if !(0..=4096).contains(&needle_len) {
                return -2;
            }
            let needle_off = needle_ptr as usize;
            let needle_n = needle_len as usize;
            // Overflow guard retained; `mem.read` bounds-checks the rest.
            if needle_off.checked_add(needle_n).is_none() {
                return ERR_GUEST_MEMORY;
            }
            let mem = match guest_memory_named(&mut caller, memory_name) {
                Some(m) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let mut needle = vec![0u8; needle_n];
            if mem.read(&caller, needle_off, &mut needle).is_err() {
                return ERR_GUEST_MEMORY;
            }
            let mut g = caller.data_mut().as_mut().lock();
            let host = &mut *g;
            match host.region_index_of(h, len as u32, &needle) {
                Ok(Some(off)) => off as i32,
                Ok(None) => -1,
                Err(e) => {
                    host.last_raw_error = err_code(&e);
                    -2
                }
            }
        },
    )?;

    linker.func_wrap(
        "tvm",
        "byte_histogram",
        move |mut caller: Caller<'_, T>, packed: i64, len: i32, out_ptr: i32| -> i32 {
            let h = Handle::unpack(packed as u64);
            let out_off = out_ptr as usize;
            if out_off.checked_add(1024).is_none() {
                return ERR_GUEST_MEMORY;
            }
            let mem = match guest_memory_named(&mut caller, memory_name) {
                Some(m) => m,
                None => return ERR_GUEST_MEMORY,
            };
            let mut buf = [0u8; 1024];
            {
                let mut g = caller.data_mut().as_mut().lock();
                let host = &mut *g;
                if let Err(e) = host.region_byte_histogram(h, len as u32, &mut buf) {
                    return err_code(&e);
                }
            }
            if mem.write(&mut caller, out_off, &buf).is_err() {
                return ERR_GUEST_MEMORY;
            }
            ERR_OK
        },
    )?;

    Ok(())
}
