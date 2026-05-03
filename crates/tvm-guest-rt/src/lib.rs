//! Guest-side runtime for the TVM raw fast path.
//!
//! This crate gives wasm32 guests a small, safe Rust API over the `tvm.*` raw
//! imports. Use it when you want the throughput of the raw linker without
//! writing `extern "C"` blocks and unsafe pointer arithmetic by hand.
//!
//! Quick start (guest crate):
//! ```ignore
//! use tvm_guest_rt::{Region, RegionKind};
//!
//! #[no_mangle]
//! pub extern "C" fn run() -> u32 {
//!     let region = Region::create(RegionKind::HotHeap, 4096).unwrap();
//!     let h = region.alloc(64).unwrap();
//!     h.write(b"hello").unwrap();
//!     let mut buf = [0u8; 5];
//!     h.read(&mut buf).unwrap();
//!     buf[0] as u32
//! }
//! ```
//!
//! See `docs/fast-paths.md` for when to use this crate vs the WIT bindings,
//! the performance characteristics, and the safety contract.

#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

#[cfg(target_arch = "wasm32")]
mod raw {
    #[link(wasm_import_module = "tvm")]
    extern "C" {
        pub fn alloc(region_id: i32, size: i32) -> i64;
        pub fn dealloc(handle: i64) -> i32;
        pub fn read(handle: i64, dst_ptr: i32, len: i32) -> i32;
        pub fn write(handle: i64, src_ptr: i32, len: i32) -> i32;
        pub fn read_gather(
            handle: i64,
            indices_ptr: i32,
            count: i32,
            item_size: i32,
            dst_ptr: i32,
        ) -> i32;
        pub fn copy_region(
            src_region: i32,
            src_offset: i32,
            dst_region: i32,
            dst_offset: i32,
            len: i32,
        ) -> i32;
        pub fn last_error() -> i32;

        // ---- reducer imports ----
        // Algebraic primitives that fold a region's bytes to a scalar
        // result on the host side. The host implementations are
        // autovec'd Rust hitting native SIMD bandwidth; calling these
        // collapses the classic "read into guest memory then loop"
        // pattern into a single trampoline returning the answer.

        pub fn sum_u8(handle: i64, len: i32) -> i64;
        pub fn sum_u32_le(handle: i64, len: i32) -> i64;
        pub fn max_u32_le(handle: i64, len: i32) -> i64;
        pub fn count_byte(handle: i64, len: i32, byte: i32) -> i32;
        pub fn count_in_range(handle: i64, len: i32, lo: i32, hi: i32) -> i32;
        pub fn popcount(handle: i64, len: i32) -> i64;
        pub fn min_max_u8(handle: i64, len: i32) -> i32;
        pub fn find_byte(handle: i64, len: i32, byte: i32) -> i32;
        pub fn index_of(handle: i64, len: i32, needle_ptr: i32, needle_len: i32) -> i32;
        pub fn eq(packed_a: i64, packed_b: i64, len: i32) -> i32;
        pub fn lex_cmp(packed_a: i64, packed_b: i64, len: i32) -> i32;
        pub fn hash_fnv1a(handle: i64, len: i32) -> i64;
        pub fn and_fold_u8(handle: i64, len: i32) -> i32;
        pub fn or_fold_u8(handle: i64, len: i32) -> i32;
        pub fn xor_fold_u8(handle: i64, len: i32) -> i32;
        pub fn fill(handle: i64, len: i32, byte: i32) -> i32;
        pub fn xor_with_byte(handle: i64, len: i32, byte: i32) -> i32;
        pub fn xor_into_region(packed_src: i64, packed_dst: i64, len: i32) -> i32;
        pub fn byte_histogram(handle: i64, len: i32, out_ptr: i32) -> i32;
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod raw {
    pub unsafe fn alloc(_: i32, _: i32) -> i64 { 0 }
    pub unsafe fn dealloc(_: i64) -> i32 { 0 }
    pub unsafe fn read(_: i64, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn write(_: i64, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn read_gather(_: i64, _: i32, _: i32, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn copy_region(_: i32, _: i32, _: i32, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn last_error() -> i32 { 0 }

    pub unsafe fn sum_u8(_: i64, _: i32) -> i64 { 0 }
    pub unsafe fn sum_u32_le(_: i64, _: i32) -> i64 { 0 }
    pub unsafe fn max_u32_le(_: i64, _: i32) -> i64 { 0 }
    pub unsafe fn count_byte(_: i64, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn count_in_range(_: i64, _: i32, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn popcount(_: i64, _: i32) -> i64 { 0 }
    pub unsafe fn min_max_u8(_: i64, _: i32) -> i32 { 0 }
    pub unsafe fn find_byte(_: i64, _: i32, _: i32) -> i32 { -1 }
    pub unsafe fn index_of(_: i64, _: i32, _: i32, _: i32) -> i32 { -1 }
    pub unsafe fn eq(_: i64, _: i64, _: i32) -> i32 { 0 }
    pub unsafe fn lex_cmp(_: i64, _: i64, _: i32) -> i32 { 0 }
    pub unsafe fn hash_fnv1a(_: i64, _: i32) -> i64 { 0 }
    pub unsafe fn and_fold_u8(_: i64, _: i32) -> i32 { 0 }
    pub unsafe fn or_fold_u8(_: i64, _: i32) -> i32 { 0 }
    pub unsafe fn xor_fold_u8(_: i64, _: i32) -> i32 { 0 }
    pub unsafe fn fill(_: i64, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn xor_with_byte(_: i64, _: i32, _: i32) -> i32 { 0 }
    pub unsafe fn xor_into_region(_: i64, _: i64, _: i32) -> i32 { 0 }
    pub unsafe fn byte_histogram(_: i64, _: i32, _: i32) -> i32 { 0 }
}

/// Mirrors `tvm:memory/types.region-kind`. Discriminants must match the
/// host's bindgen-generated enum (`#[repr(u8)]`, declaration order in WIT).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RegionKind {
    HotHeap = 0,
    ObjectArena = 1,
    BlobArena = 2,
    PageStore = 3,
    Scratch = 4,
    DeviceState = 5,
    CodeCache = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    RegionNotFound,
    StaleHandle,
    OutOfBounds,
    NotResident,
    AllocationFailed,
    Pinned,
    GuestMemory,
    Other(i32),
}

impl Error {
    fn from_code(code: i32) -> Self {
        match code {
            1 => Self::RegionNotFound,
            2 => Self::StaleHandle,
            3 => Self::OutOfBounds,
            4 => Self::NotResident,
            5 => Self::AllocationFailed,
            6 => Self::Pinned,
            7 => Self::GuestMemory,
            other => Self::Other(other),
        }
    }
}

pub type Result<T> = core::result::Result<T, Error>;

fn check(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(Error::from_code(code)) }
}

/// Opaque region identifier. Returned by `Region::create`; passed to host
/// management functions (which still go through the WIT path because they
/// aren't on the hot path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub id: u16,
}

impl Region {
    /// Calls into the WIT `manager.create-region` import. NB: this requires
    /// the WIT-bindgen-generated guest bindings to also be linked — the raw
    /// fast path doesn't provide region creation since it's a one-shot setup
    /// cost. See `examples/guest-fast-path` for the recommended layout.
    pub fn from_id(id: u16) -> Self { Self { id } }

    /// Allocate `size` bytes inside this region.
    pub fn alloc(self, size: u32) -> Result<RegionPtr> {
        let packed = unsafe { raw::alloc(self.id as i32, size as i32) };
        if packed == 0 {
            let code = unsafe { raw::last_error() };
            return Err(Error::from_code(code));
        }
        Ok(RegionPtr { packed })
    }
}

/// Resolved handle: a region pointer that the guest can read from and write
/// to. The packed `i64` representation is what crosses the FFI boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionPtr {
    pub packed: i64,
}

impl RegionPtr {
    pub fn read(self, dst: &mut [u8]) -> Result<()> {
        let code = unsafe { raw::read(self.packed, dst.as_mut_ptr() as i32, dst.len() as i32) };
        check(code)
    }

    pub fn write(self, src: &[u8]) -> Result<()> {
        let code = unsafe { raw::write(self.packed, src.as_ptr() as i32, src.len() as i32) };
        check(code)
    }

    pub fn dealloc(self) -> Result<()> {
        let code = unsafe { raw::dealloc(self.packed) };
        check(code)
    }
}

// ---------- Reducer methods on RegionPtr ----------
//
// These wrap the raw `tvm.<op>` imports added on the host side.
// Each is a single trampoline that returns the scalar (or writes to
// guest memory in the histogram case). Performance characteristics:
// the host implementations autovec to native SIMD; ~12–16 GiB/s on
// modern x86/aarch64 for byte-level reductions. See
// `tvm-wasmtime/benches/reducer_imports.rs` for measured speedups
// against the classic "read into guest, scalar loop" pattern.

impl RegionPtr {
    /// Sum every byte in the next `len` bytes from this pointer as a
    /// u64 (no overflow possible — max is `len × 255`).
    pub fn sum_u8(self, len: u32) -> Result<u64> {
        let v = unsafe { raw::sum_u8(self.packed, len as i32) };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok(v as u64)
        }
    }

    /// Sum of `len/4` little-endian u32 lanes. `len` must be a
    /// multiple of 4. Returns the low 60 bits of the sum (signed
    /// semantics; the host returns -1 on error).
    pub fn sum_u32_le(self, len: u32) -> Result<u64> {
        let v = unsafe { raw::sum_u32_le(self.packed, len as i32) };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok(v as u64)
        }
    }

    /// Max of `len/4` little-endian u32 lanes. Returns `None` when
    /// `len == 0`. `len` must be a multiple of 4.
    pub fn max_u32_le(self, len: u32) -> Result<Option<u32>> {
        let v = unsafe { raw::max_u32_le(self.packed, len as i32) };
        match v {
            -2 => Ok(None),
            -1 => Err(Error::from_code(unsafe { raw::last_error() })),
            v => Ok(Some(v as u32)),
        }
    }

    /// Count occurrences of `byte` in the next `len` bytes.
    pub fn count_byte(self, len: u32, byte: u8) -> Result<u32> {
        let v = unsafe { raw::count_byte(self.packed, len as i32, byte as i32) };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok(v as u32)
        }
    }

    /// Count bytes in `[lo, hi]` (inclusive on both ends).
    pub fn count_in_range(self, len: u32, lo: u8, hi: u8) -> Result<u32> {
        let v = unsafe {
            raw::count_in_range(self.packed, len as i32, lo as i32, hi as i32)
        };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok(v as u32)
        }
    }

    /// Total set bits across `len` bytes.
    pub fn popcount(self, len: u32) -> Result<u64> {
        let v = unsafe { raw::popcount(self.packed, len as i32) };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok(v as u64)
        }
    }

    /// Min and max of `len` bytes. Returns `(min, max)`.
    /// `len == 0` yields the conventional sentinel `(255, 0)`.
    pub fn min_max_u8(self, len: u32) -> Result<(u8, u8)> {
        let packed = unsafe { raw::min_max_u8(self.packed, len as i32) };
        if packed < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            let lo = ((packed >> 8) & 0xff) as u8;
            let hi = (packed & 0xff) as u8;
            Ok((lo, hi))
        }
    }

    /// First offset (relative to this pointer) where `byte` occurs in
    /// the next `len` bytes, or `None`.
    pub fn find_byte(self, len: u32, byte: u8) -> Result<Option<u32>> {
        let v = unsafe { raw::find_byte(self.packed, len as i32, byte as i32) };
        match v {
            -1 => Ok(None),
            -2 => Err(Error::from_code(unsafe { raw::last_error() })),
            v => Ok(Some(v as u32)),
        }
    }

    /// First offset where `needle` occurs in the next `len` bytes, or
    /// `None`. Needle length capped at 4096 by the host.
    pub fn index_of(self, len: u32, needle: &[u8]) -> Result<Option<u32>> {
        let v = unsafe {
            raw::index_of(
                self.packed,
                len as i32,
                needle.as_ptr() as i32,
                needle.len() as i32,
            )
        };
        match v {
            -1 => Ok(None),
            -2 => Err(Error::from_code(unsafe { raw::last_error() })),
            v => Ok(Some(v as u32)),
        }
    }

    /// Compare `len` bytes between two pointers; `true` iff equal.
    pub fn eq(self, other: RegionPtr, len: u32) -> Result<bool> {
        let v = unsafe { raw::eq(self.packed, other.packed, len as i32) };
        match v {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::from_code(unsafe { raw::last_error() })),
        }
    }

    /// Lexicographic compare; returns `Ordering`.
    pub fn lex_cmp(self, other: RegionPtr, len: u32) -> Result<core::cmp::Ordering> {
        let v = unsafe { raw::lex_cmp(self.packed, other.packed, len as i32) };
        match v {
            -1 => Ok(core::cmp::Ordering::Less),
            0 => Ok(core::cmp::Ordering::Equal),
            1 => Ok(core::cmp::Ordering::Greater),
            _ => Err(Error::from_code(unsafe { raw::last_error() })),
        }
    }

    /// FNV-1a 64-bit hash. Note: scalar loop, ~3 ns/byte. Use a real
    /// digest crate over a host-side bulk read for high throughput.
    pub fn hash_fnv1a(self, len: u32) -> Result<u64> {
        // Returns 0 on error; check last_error to disambiguate from a
        // legitimate hash of 0 (rare). Caller tolerant of the rare
        // collision on the error path can ignore last_error.
        let v = unsafe { raw::hash_fnv1a(self.packed, len as i32) };
        Ok(v as u64)
    }

    /// Bitwise AND of every byte. Useful for "all-bits-cleared" masks.
    pub fn and_fold_u8(self, len: u32) -> Result<u8> {
        let v = unsafe { raw::and_fold_u8(self.packed, len as i32) };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok((v & 0xff) as u8)
        }
    }

    /// Bitwise OR of every byte. Useful for "any-bits-set" masks.
    pub fn or_fold_u8(self, len: u32) -> Result<u8> {
        let v = unsafe { raw::or_fold_u8(self.packed, len as i32) };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok((v & 0xff) as u8)
        }
    }

    /// XOR of every byte (parity / lightweight checksum).
    pub fn xor_fold_u8(self, len: u32) -> Result<u8> {
        let v = unsafe { raw::xor_fold_u8(self.packed, len as i32) };
        if v < 0 {
            Err(Error::from_code(unsafe { raw::last_error() }))
        } else {
            Ok((v & 0xff) as u8)
        }
    }

    /// Set every byte in `[ptr, ptr + len)` to `value`. memset-style.
    pub fn fill(self, len: u32, value: u8) -> Result<()> {
        let code = unsafe { raw::fill(self.packed, len as i32, value as i32) };
        check(code)
    }

    /// XOR every byte in `[ptr, ptr + len)` with `value`. In-place.
    pub fn xor_with_byte(self, len: u32, value: u8) -> Result<()> {
        let code = unsafe { raw::xor_with_byte(self.packed, len as i32, value as i32) };
        check(code)
    }

    /// XOR `len` bytes from `src` into the same range starting at this
    /// pointer (`*self ^= *src`). Errors if both pointers refer to the
    /// same region.
    pub fn xor_from(self, src: RegionPtr, len: u32) -> Result<()> {
        let code = unsafe { raw::xor_into_region(src.packed, self.packed, len as i32) };
        check(code)
    }

    /// Byte-frequency histogram. Writes 256 little-endian u32s
    /// (1024 bytes total) into `out`, where `out[i]` is the count of
    /// bytes equal to `i`.
    pub fn byte_histogram(self, len: u32, out: &mut [u8; 1024]) -> Result<()> {
        let code = unsafe {
            raw::byte_histogram(self.packed, len as i32, out.as_mut_ptr() as i32)
        };
        check(code)
    }
}

impl RegionPtr {
    /// Scatter-read `count` items of `item_size` bytes each into a contiguous
    /// guest buffer. `indices` holds the per-item offsets within the region
    /// (relative to this RegionPtr). One host call regardless of count.
    pub fn read_gather(
        self,
        indices: &[u32],
        item_size: u32,
        dst: &mut [u8],
    ) -> Result<()> {
        let code = unsafe {
            raw::read_gather(
                self.packed,
                indices.as_ptr() as i32,
                indices.len() as i32,
                item_size as i32,
                dst.as_mut_ptr() as i32,
            )
        };
        check(code)
    }
}

/// Bulk-read iterator: pull a region's bytes into a stack buffer in one
/// host call, then iterate. This is the **idiomatic hot-path pattern**:
/// most workloads should read the working set once and process locally.
/// Returns an error from the closure or `Ok(acc)` after all chunks.
///
/// Per-call overhead: one host trampoline. Per-byte overhead: native
/// guest-side memory access.
///
/// True native-instruction access from a non-default wasm memory isn't
/// supported on stable Rust today (the wasm-multi-memory proposal
/// support in LLVM/rustc is incomplete). For that path use the WAT-based
/// TVM-MM pattern documented in `bench-framework/runner/src/main.rs`'s
/// `MM_WAT`.
pub fn for_chunks<const N: usize, F, T>(
    ptr: RegionPtr,
    total_len: u32,
    init: T,
    mut step: F,
) -> Result<T>
where
    F: FnMut(T, &[u8]) -> T,
{
    let mut buf = [0u8; N];
    let mut acc = init;
    let mut consumed: u32 = 0;
    while consumed < total_len {
        let to_read = core::cmp::min(N as u32, total_len - consumed);
        let cur = RegionPtr {
            packed: (ptr.packed
                & 0xFFFF_FFFF_0000_0000_u64 as i64)
                | ((((ptr.packed as u64) & 0xFFFF_FFFF) as u32 + consumed) as i64),
        };
        cur.read(&mut buf[..to_read as usize])?;
        acc = step(acc, &buf[..to_read as usize]);
        consumed += to_read;
    }
    Ok(acc)
}

/// Region-to-region copy that never touches guest linear memory. Use this
/// instead of `read` + `write` when both source and destination live in TVM
/// regions; it skips the host-to-guest copy on each side.
pub fn copy_region(src: Region, src_off: u32, dst: Region, dst_off: u32, len: u32) -> Result<()> {
    let code = unsafe {
        raw::copy_region(
            src.id as i32,
            src_off as i32,
            dst.id as i32,
            dst_off as i32,
            len as i32,
        )
    };
    check(code)
}
