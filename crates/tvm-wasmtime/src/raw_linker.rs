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
        TvmError::BackingStore(_)
        | TvmError::UnsupportedAllocator
        | TvmError::PolicyViolation => ERR_OTHER,
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
fn cached_guest_memory<T>(
    caller: &mut Caller<'_, T>,
    name: &'static str,
) -> Option<Memory>
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
            let (_mem, ptr, _size) = match cached_guest_memory_view(
                &mut caller,
                memory_name,
                required_end,
            ) {
                Some(v) => v,
                None => return ERR_GUEST_MEMORY,
            };
            let host = caller.data_mut().as_mut();
            // SAFETY: ptr+required_end is bounds-checked against memory
            // size in the view fetch; fast_read checks the region side.
            match unsafe {
                host.fast_read(h, (ptr as *mut u8).add(dst_off), len as u32)
            } {
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
            let (_mem, ptr, _size) = match cached_guest_memory_view(
                &mut caller,
                memory_name,
                required_end,
            ) {
                Some(v) => v,
                None => return ERR_GUEST_MEMORY,
            };
            let host = caller.data_mut().as_mut();
            // SAFETY: ptr+required_end is bounds-checked.
            match unsafe {
                host.fast_write(h, (ptr as *const u8).add(src_off), len as u32)
            } {
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
            if mem.read(&caller, indices_ptr as usize, &mut indices_bytes).is_err() {
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
                    && (1..count_us)
                        .all(|k| indices[k].wrapping_sub(indices[k - 1]) == stride)
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
                        let cell = Handle { offset: h.offset.wrapping_add(off), ..h };
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

    linker.func_wrap(
        "tvm",
        "last_error",
        |mut caller: Caller<'_, T>| -> i32 {
            let host = caller.data_mut().as_mut();
            std::mem::replace(&mut host.last_raw_error, ERR_OK)
        },
    )?;

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

    // Suppress unused warning when memory name is the default.
    let _ = guest_memory::<T>;

    Ok(())
}
