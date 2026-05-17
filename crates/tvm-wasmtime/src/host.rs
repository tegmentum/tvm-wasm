use std::path::Path;

/// Bundles the wasmtime memory handle with its cached size + base
/// pointer. All three values are refreshed together on bounds-check
/// miss (the guest may have grown its memory).
#[derive(Default)]
pub struct CachedGuestMemory {
    pub memory: Option<wasmtime::Memory>,
    pub size: u64,
    pub ptr: usize,
}

impl CachedGuestMemory {
    pub fn invalidate(&mut self) {
        self.size = 0;
        self.ptr = 0;
    }
}

use tvm_core::{
    AllocatorKind, BackingStore, DynBackingStore, FileBackingStore, Handle as CoreHandle,
    Region as CoreRegion, RegionDirectory, RegionKind as CoreRegionKind,
    Residency as CoreResidency, ResolveCache, ResolveHit, TvmError as CoreError, TvmFacade,
    TvmSpill, VecBackedRegion,
};

/// Size-specialized memcpy. Hot path of `fast_read` / `fast_write`. Inline
/// typed loads for cell-grain sizes; let LLVM auto-vectorize the larger
/// fixed sizes via `u128` / paired-u128 idioms; fall back to
/// `copy_nonoverlapping` for variable lengths.
///
/// SAFETY: `src` valid for `len` bytes; `dst` valid for `len` bytes;
/// regions disjoint.
#[inline(always)]
unsafe fn copy_specialized(src: *const u8, dst: *mut u8, len: usize) {
    use std::ptr;
    unsafe {
        match len {
            1 => *dst = *src,
            2 => ptr::write_unaligned(dst as *mut u16, ptr::read_unaligned(src as *const u16)),
            4 => ptr::write_unaligned(dst as *mut u32, ptr::read_unaligned(src as *const u32)),
            8 => ptr::write_unaligned(dst as *mut u64, ptr::read_unaligned(src as *const u64)),
            16 => ptr::write_unaligned(dst as *mut u128, ptr::read_unaligned(src as *const u128)),
            32 => {
                let v0 = ptr::read_unaligned(src as *const u128);
                let v1 = ptr::read_unaligned((src as *const u128).add(1));
                ptr::write_unaligned(dst as *mut u128, v0);
                ptr::write_unaligned((dst as *mut u128).add(1), v1);
            }
            64 => {
                let v0 = ptr::read_unaligned(src as *const u128);
                let v1 = ptr::read_unaligned((src as *const u128).add(1));
                let v2 = ptr::read_unaligned((src as *const u128).add(2));
                let v3 = ptr::read_unaligned((src as *const u128).add(3));
                ptr::write_unaligned(dst as *mut u128, v0);
                ptr::write_unaligned((dst as *mut u128).add(1), v1);
                ptr::write_unaligned((dst as *mut u128).add(2), v2);
                ptr::write_unaligned((dst as *mut u128).add(3), v3);
            }
            _ => ptr::copy_nonoverlapping(src, dst, len),
        }
    }
}

use crate::bindings::tvm::memory::bytes::Host as BytesHost;
use crate::bindings::tvm::memory::diagnostics::Host as DiagnosticsHost;
use crate::bindings::tvm::memory::manager::Host as ManagerHost;
use crate::bindings::tvm::memory::types::{
    CompactResult, Handle, Host as TypesHost, RegionInfo, RegionKind, RegionMetrics, Residency,
    TvmError,
};

/// Host-side TVM. The directory + cache + (optional) backing store
/// live together in one type for ergonomic embedder use, but the
/// **spill capability is logically separate from the core TVM
/// interface**:
///
///   - `impl TvmFacade for TvmHost` covers the deployment-agnostic
///     ops (create / alloc / read / write / pin / unpin / region_info).
///   - `impl TvmSpill for TvmHost` covers spill/load — only meaningful
///     when `backing` is configured.
///
/// Embedders who don't want spill can simply not configure a backing
/// (`TvmHost::new()` instead of `TvmHost::with_backing(...)`); the
/// `TvmSpill` calls will return `BackingStore("no backing store
/// configured")`. Code that never calls spill never touches the
/// backing field.
///
/// Code generic over `T: TvmFacade` works against `TvmHost` without
/// caring whether spill is configured — that's the architectural
/// separation. Spill is layered above the facade.
pub struct TvmHost {
    pub directory: RegionDirectory<VecBackedRegion>,
    pub backing: Option<DynBackingStore>,
    pub default_allocator: AllocatorKind,
    pub cache: ResolveCache,
    /// Last error from a raw-linker call that returned a sentinel value (e.g.
    /// `tvm.alloc` returning 0). Cleared when the guest reads it via
    /// `tvm.last_error`.
    pub last_raw_error: i32,
    /// Cached view of the guest's linear memory (handle + size + base
    /// pointer). Populated lazily on the first raw-path call that needs
    /// it; refreshed on bounds-check miss. Skips the per-call
    /// `Caller::get_export` HashMap lookup and the two `Memory` getters.
    pub cached_memory: CachedGuestMemory,
    /// Imported regions — TVM regions whose backing is a wasmtime
    /// `Memory` exposed to the guest as an import. The guest accesses
    /// these natively via `i32.load`; TVM still owns lifecycle
    /// (alloc/dealloc/pin/spill/compact). See `imported.rs`.
    ///
    /// Keyed by region_id. ID assignment uses a separate counter from
    /// host regions; callers mixing both should partition the ID space
    /// (e.g. set `next_imported_id` to 0x8000 before creating any).
    pub imported: Vec<crate::imported::ImportedRegion>,
    /// Next region_id to assign for imported regions. Defaults to 0
    /// so the simple "imported only" case yields predictable names
    /// (`tvm.r0`, `tvm.r1`, ...). Mixed-mode callers should bump this
    /// before creating imported regions.
    pub next_imported_id: u16,
}

impl Default for TvmHost {
    fn default() -> Self {
        Self::new()
    }
}

impl TvmHost {
    pub fn new() -> Self {
        Self {
            directory: RegionDirectory::new(),
            backing: None,
            default_allocator: AllocatorKind::Bump,
            cache: ResolveCache::new(),
            last_raw_error: 0,
            cached_memory: CachedGuestMemory::default(),
            imported: Vec::new(),
            next_imported_id: 0,
        }
    }

    pub fn with_backing(path: impl AsRef<Path>) -> Result<Self, CoreError> {
        Ok(Self::with_custom_backing(Box::new(FileBackingStore::new(
            path.as_ref().to_path_buf(),
        )?)))
    }

    /// Construct a host with an arbitrary user-supplied backing store.
    /// Use this for custom impls (S3, in-memory test stub, network-attached
    /// shared store, etc.). The trait surface is intentionally tiny:
    /// `spill(region_id, generation, bytes)` and `load(region_id, generation)`.
    pub fn with_custom_backing(backing: DynBackingStore) -> Self {
        Self {
            directory: RegionDirectory::new(),
            backing: Some(backing),
            default_allocator: AllocatorKind::Bump,
            cache: ResolveCache::new(),
            last_raw_error: 0,
            cached_memory: CachedGuestMemory::default(),
            imported: Vec::new(),
            next_imported_id: 0,
        }
    }

    pub fn with_allocator(mut self, allocator: AllocatorKind) -> Self {
        self.default_allocator = allocator;
        self
    }

    // ---------- Ergonomic shortcuts (no bindgen-trait imports needed) ----------

    /// Create a new region with default allocator + policy. Returns the
    /// region id. Equivalent to calling `manager::create-region` over the
    /// WIT path but without needing the `ManagerHost` trait import.
    pub fn create_region(&mut self, kind: CoreRegionKind, capacity: u32) -> Result<u16, CoreError> {
        let allocator = self.default_allocator;
        self.directory
            .create_region_with(kind, capacity, allocator, VecBackedRegion::new(capacity))
    }

    /// Allocate inside an existing region. Same as the WIT `manager.alloc`
    /// but ergonomic.
    pub fn alloc(&mut self, region: u16, size: u32) -> Result<CoreHandle, CoreError> {
        self.directory.alloc(region, size)
    }

    /// One-call: create a fresh region of `capacity` bytes and immediately
    /// allocate `size` bytes inside it. Returns `(region_id, handle)`.
    /// Common pattern for "I just want a scratch buffer."
    pub fn alloc_in_new_region(
        &mut self,
        kind: CoreRegionKind,
        capacity: u32,
        size: u32,
    ) -> Result<(u16, CoreHandle), CoreError> {
        let region = self.create_region(kind, capacity)?;
        let handle = self.alloc(region, size)?;
        Ok((region, handle))
    }

    /// Read bytes into the supplied buffer. Ergonomic wrapper around the
    /// directory's read; uses the auto-fault path if a backing store is
    /// configured, strict mode otherwise.
    pub fn read_bytes(&mut self, handle: CoreHandle, buf: &mut [u8]) -> Result<(), CoreError> {
        let TvmHost {
            directory, backing, ..
        } = self;
        match backing.as_mut() {
            Some(b) => directory.read_or_fault(handle, buf, b),
            None => directory.read(handle, buf),
        }
    }

    /// Write bytes. Symmetric to `read_bytes`.
    pub fn write_bytes(&mut self, handle: CoreHandle, data: &[u8]) -> Result<(), CoreError> {
        let TvmHost {
            directory, backing, ..
        } = self;
        match backing.as_mut() {
            Some(b) => directory.write_or_fault(handle, data, b),
            None => directory.write(handle, data),
        }
    }

    // ------------------------------------------------------------------
    // Reducer API: operations that fold a region's bytes to a scalar
    // result, then return the scalar across the host boundary.
    //
    // For a wasm-guest caller, these replace the "host.read_bytes(...)
    // → scalar guest loop" two-step with a single trampoline that
    // returns the answer directly. The host-side implementation is
    // plain Rust over the region's `&[u8]`; rustc's autovectorizer
    // produces native SIMD for these shapes, so the host pays
    // hardware-bandwidth cost regardless of payload size.
    //
    // None of these require a guest-side SIMD sidecar — that pattern
    // turned out to be slower than autovec'd Rust for host execution.
    // The benefit is purely in collapsing the call sequence.
    //
    // ## API contract — closed algebraic core
    //
    // This module exposes a **closed set of algebraic primitives**: the
    // monoid reducers (sum, product-equivalents, AND, OR, XOR, max,
    // min, popcount), counting reducers (count_byte, count_in_range,
    // popcount), comparisons (eq, lex_cmp), searches (find_byte,
    // index_of), and the in-place mutator family (fill, xor_*).
    //
    // Domain-specific reducers (CRC32, base64, UTF-8 validation,
    // SHA / xxhash / blake3, JSON parsing) are **out of scope**.
    // Implement those in caller code by acquiring the slice via
    // `RegionDirectory::region_slice_at` (or use this trait's
    // operations for the autovec-friendly parts) and writing the
    // domain-specific fold inline. The trampoline-collapse benefit
    // generalizes — any user-defined fold gets the same speedup if
    // exposed as a single host import.
    //
    // The intentional contract: TVM owns the closed algebraic set;
    // the embedder owns everything else.
    // ------------------------------------------------------------------

    /// Sum every byte in the first `len` bytes of `handle`'s region as
    /// a u64 (no overflow possible — max value is len × 255). Returns
    /// `OutOfBounds` if `len` exceeds the region's capacity from
    /// `handle.offset`.
    pub fn region_sum_u8(&mut self, handle: CoreHandle, len: u32) -> Result<u64, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().map(|&b| b as u64).sum())
    }

    /// First offset (relative to `handle.offset`) at which `byte`
    /// occurs within `len` bytes of the region, or `None`. Returns
    /// `OutOfBounds` for an over-long `len`.
    pub fn region_find_byte(
        &mut self,
        handle: CoreHandle,
        len: u32,
        byte: u8,
    ) -> Result<Option<u32>, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().position(|&b| b == byte).map(|p| p as u32))
    }

    /// FNV-1a hash of `len` bytes. 64-bit variant. Cheap, decent
    /// distribution. For cryptographic hashes use a real digest crate
    /// over `region_slice_at`.
    ///
    /// Note: FNV-1a doesn't autovec — each byte's `wrapping_mul`
    /// carries state across iterations. Throughput is bounded by the
    /// 64-bit multiplier latency (~3 ns/byte on modern x86). Use a
    /// SIMD-friendly hash (xxhash, blake3) over `region_slice_at` if
    /// you need bandwidth.
    pub fn region_hash_fnv1a(&mut self, handle: CoreHandle, len: u32) -> Result<u64, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Ok(h)
    }

    /// Count how many bytes in the first `len` bytes equal `byte`.
    /// Maxes at `len`, so a u32 suffices.
    pub fn region_count_byte(
        &mut self,
        handle: CoreHandle,
        len: u32,
        byte: u8,
    ) -> Result<u32, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().filter(|&&b| b == byte).count() as u32)
    }

    /// Compare `len` bytes from two regions. Returns `true` iff every
    /// byte matches; uses `<[u8]>::eq` which short-circuits on first
    /// mismatch and autovec's the equal-prefix portion.
    pub fn region_eq(&mut self, a: CoreHandle, b: CoreHandle, len: u32) -> Result<bool, CoreError> {
        // Two slice borrows from the same directory require a small
        // dance: validate first into a fresh borrow, then the second.
        // Both immutable, no aliasing problem if the regions differ.
        let lhs = self.directory.region_slice_at(a, len)?;
        let lhs_ptr = lhs.as_ptr();
        let lhs_len = lhs.len();
        let rhs = self.directory.region_slice_at(b, len)?;
        // SAFETY: lhs_ptr/len came from a successful region_slice_at on
        // an immutable directory borrow that we still hold (no mutation
        // happened between the two calls). The slice is valid for the
        // duration of this function.
        let lhs = unsafe { core::slice::from_raw_parts(lhs_ptr, lhs_len) };
        Ok(lhs == rhs)
    }

    /// Min and max of bytes in the first `len` bytes. Returns
    /// `(min, max)`. `len == 0` returns `(255, 0)` as the conventional
    /// empty-range sentinel.
    pub fn region_min_max_u8(
        &mut self,
        handle: CoreHandle,
        len: u32,
    ) -> Result<(u8, u8), CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        let mut lo: u8 = 255;
        let mut hi: u8 = 0;
        for &b in bytes {
            if b < lo {
                lo = b;
            }
            if b > hi {
                hi = b;
            }
        }
        Ok((lo, hi))
    }

    /// XOR the first `len` bytes of `src` into the same range of
    /// `dst`, byte-wise (`dst[i] ^= src[i]`). Useful for delta
    /// encoding, parity, stream-cipher keystream application.
    /// `OutOfBounds` if either side is too short. Errors if `src` and
    /// `dst` resolve to the same region (would be a self-XOR-zero,
    /// silent footgun).
    pub fn region_xor_into_region(
        &mut self,
        src: CoreHandle,
        dst: CoreHandle,
        len: u32,
    ) -> Result<(), CoreError> {
        if src.region_id == dst.region_id {
            return Err(CoreError::PolicyViolation);
        }
        let src_ptr;
        let src_len_actual;
        {
            let s = self.directory.region_slice_at(src, len)?;
            src_ptr = s.as_ptr();
            src_len_actual = s.len();
        }
        let dst_slice = self.directory.region_slice_mut_at(dst, len)?;
        // SAFETY: src and dst belong to different regions (id check
        // above), so the immutable src borrow doesn't alias dst.
        // VecBackedRegion stores each region in its own Vec<u8>, so
        // distinct ids guarantee distinct allocations.
        let src_slice = unsafe { core::slice::from_raw_parts(src_ptr, src_len_actual) };
        for (d, s) in dst_slice.iter_mut().zip(src_slice.iter()) {
            *d ^= *s;
        }
        Ok(())
    }

    /// Byte-frequency histogram. Writes 256 little-endian u32s into
    /// `out` (must be `1024` bytes); position `i` holds the count of
    /// bytes equal to `i`. Counts saturate at `u32::MAX` (in practice
    /// unreachable since region capacity is also u32).
    pub fn region_byte_histogram(
        &mut self,
        handle: CoreHandle,
        len: u32,
        out: &mut [u8; 1024],
    ) -> Result<(), CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        let mut counts = [0u32; 256];
        for &b in bytes {
            counts[b as usize] = counts[b as usize].saturating_add(1);
        }
        // Pack as little-endian u32s.
        for (i, c) in counts.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&c.to_le_bytes());
        }
        Ok(())
    }

    /// Sum of `len/4` little-endian u32 lanes, returned as u128 to
    /// avoid overflow for full-region sums (max value =
    /// 2^32 × 2^30 / 4 = 2^60). `len` must be a multiple of 4.
    pub fn region_sum_u32_le(&mut self, handle: CoreHandle, len: u32) -> Result<u128, CoreError> {
        if len % 4 != 0 {
            return Err(CoreError::OutOfBounds);
        }
        let bytes = self.directory.region_slice_at(handle, len)?;
        // Manual chunk decode keeps us autovec-friendly without
        // requiring the slice to be 4-byte aligned.
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u128)
            .sum())
    }

    /// Max of `len/4` little-endian u32 lanes. `len` must be a
    /// multiple of 4. Returns `None` if `len == 0`.
    pub fn region_max_u32_le(
        &mut self,
        handle: CoreHandle,
        len: u32,
    ) -> Result<Option<u32>, CoreError> {
        if len % 4 != 0 {
            return Err(CoreError::OutOfBounds);
        }
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .max())
    }

    /// Total set bits across `len` bytes. Plain `u8::count_ones` per
    /// byte; rustc autovec's via vectorized popcount on x86_64 +SSE4
    /// and aarch64 +neon. Throughput ~16 GiB/s on modern hardware.
    pub fn region_popcount(&mut self, handle: CoreHandle, len: u32) -> Result<u64, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().map(|&b| b.count_ones() as u64).sum())
    }

    /// Set every byte in `[handle.offset, handle.offset + len)` to
    /// `value`. Equivalent to writing `len` copies of `value` but
    /// avoids the bytes-in copy from the guest.
    pub fn region_fill(
        &mut self,
        handle: CoreHandle,
        len: u32,
        value: u8,
    ) -> Result<(), CoreError> {
        let dst = self.directory.region_slice_mut_at(handle, len)?;
        // slice::fill compiles to `memset` for u8.
        dst.fill(value);
        Ok(())
    }

    /// XOR every byte in `[handle.offset, handle.offset + len)` with
    /// `value`. Useful for keystream-style obfuscation, polarity
    /// flips, etc. Autovec's to a vectorized xor loop.
    pub fn region_xor_with_byte(
        &mut self,
        handle: CoreHandle,
        len: u32,
        value: u8,
    ) -> Result<(), CoreError> {
        let dst = self.directory.region_slice_mut_at(handle, len)?;
        for b in dst.iter_mut() {
            *b ^= value;
        }
        Ok(())
    }

    /// Bitwise AND fold across `len` bytes. Identity is `0xff`.
    /// Returns the AND of every byte; useful for "all bits ever
    /// cleared" / common-prefix mask analysis.
    pub fn region_and_fold_u8(&mut self, handle: CoreHandle, len: u32) -> Result<u8, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().fold(0xffu8, |a, &b| a & b))
    }

    /// Bitwise OR fold across `len` bytes. Identity is `0x00`.
    /// Returns the OR of every byte; useful for "all bits ever set" /
    /// reachable-bit-mask analysis.
    pub fn region_or_fold_u8(&mut self, handle: CoreHandle, len: u32) -> Result<u8, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().fold(0u8, |a, &b| a | b))
    }

    /// XOR fold across `len` bytes. Identity is `0x00`. Returns the
    /// running XOR of every byte; useful for parity checks and
    /// simple checksums (note: not a hash — collisions are trivial).
    pub fn region_xor_fold_u8(&mut self, handle: CoreHandle, len: u32) -> Result<u8, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().fold(0u8, |a, &b| a ^ b))
    }

    /// Count bytes whose value falls within `[lo, hi]` (inclusive on
    /// both ends). Generalizes `count_byte` (which is the `lo == hi`
    /// case). Predicate-counting completes the monoid family of
    /// reducers (see also `count_byte`, `popcount`).
    pub fn region_count_in_range(
        &mut self,
        handle: CoreHandle,
        len: u32,
        lo: u8,
        hi: u8,
    ) -> Result<u32, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        Ok(bytes.iter().filter(|&&b| b >= lo && b <= hi).count() as u32)
    }

    /// Lexicographic comparison of `len` bytes between two regions.
    /// Returns `-1` if `a < b`, `0` if equal, `+1` if `a > b`.
    /// Generalizes `region_eq` from bool to ordering. Both regions
    /// must contain at least `len` bytes from their respective
    /// handle offsets.
    pub fn region_lex_cmp(
        &mut self,
        a: CoreHandle,
        b: CoreHandle,
        len: u32,
    ) -> Result<core::cmp::Ordering, CoreError> {
        let lhs = self.directory.region_slice_at(a, len)?;
        let lhs_ptr = lhs.as_ptr();
        let lhs_len = lhs.len();
        let rhs = self.directory.region_slice_at(b, len)?;
        // SAFETY: same justification as region_eq — both borrows are
        // immutable, taken in sequence on a directory we hold.
        let lhs = unsafe { core::slice::from_raw_parts(lhs_ptr, lhs_len) };
        Ok(lhs.cmp(rhs))
    }

    /// Find the first occurrence of `needle` within the first `len`
    /// bytes of `handle`'s region. Returns the offset (relative to
    /// `handle.offset`) of the match, or `None`. Uses the standard
    /// library's two-way string searcher under the hood, which is
    /// linear in `len` with a small constant.
    pub fn region_index_of(
        &mut self,
        handle: CoreHandle,
        len: u32,
        needle: &[u8],
    ) -> Result<Option<u32>, CoreError> {
        let bytes = self.directory.region_slice_at(handle, len)?;
        if needle.is_empty() {
            return Ok(Some(0));
        }
        // memchr-style first-byte gate, then memcmp the tail. For
        // production use a real `memmem` crate, but stdlib's windows()
        // is already vectorized for the common cases that matter to us.
        Ok(bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .map(|p| p as u32))
    }

    /// Construct a wasm-Memory-shaped view over a region. See
    /// [`crate::region_view::RegionView`] for the shape — `data`,
    /// `data_mut`, `read`, `write`, `copy_within`, `fill`, etc., as
    /// close to `wasmtime::Memory` as TVM semantics allow.
    pub fn region_view(
        &mut self,
        region_id: u16,
    ) -> Result<crate::region_view::RegionView<'_>, CoreError> {
        // Validate the region exists before handing out a view.
        let _ = self.directory.region_info(region_id)?;
        Ok(crate::region_view::RegionView::new(self, region_id))
    }

    /// Look up a region's metadata, hitting the cache first. On miss, falls
    /// back to the directory and populates the cache. Hot path for the raw
    /// linker.
    pub fn resolve(&mut self, region_id: u16) -> Result<ResolveHit, CoreError> {
        if let Some(hit) = self.cache.lookup(region_id) {
            return Ok(hit);
        }
        let info = *self.directory.region_info(region_id)?;
        Ok(self.cache.install(&info))
    }

    /// Run a closure that mutates a region's underlying memory, then
    /// invalidate that region's cache slot. Use this whenever you're
    /// adding a new method that replaces a region's data pointer or
    /// generation; it ensures the cache can never serve stale entries.
    #[inline]
    pub fn with_invalidated_cache<F, R>(&mut self, region_id: u16, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.cache.invalidate(region_id);
        f(self)
    }

    /// Hot-path read: validates handle via the resolve cache (zero
    /// directory traversal on cache hit), then performs a single memcpy
    /// directly from the cached region pointer to the supplied raw
    /// destination. Returns Ok on success, an error on bounds/staleness.
    ///
    /// SAFETY: `dst` must be valid for `len` bytes.
    #[inline]
    pub unsafe fn fast_read(
        &mut self,
        handle: CoreHandle,
        dst: *mut u8,
        len: u32,
    ) -> Result<(), CoreError> {
        let hit = match self.cache.lookup_fast(handle.region_id) {
            Some(h) => h,
            None => self.refresh_cache(handle.region_id)?,
        };
        if hit.generation != handle.generation {
            return Err(CoreError::StaleHandle);
        }
        if !hit.resident {
            return Err(CoreError::NotResident);
        }
        let end = handle
            .offset
            .checked_add(len)
            .ok_or(CoreError::OutOfBounds)?;
        if end > hit.capacity {
            return Err(CoreError::OutOfBounds);
        }
        // SAFETY: hit.data_ptr is the validated region's base; we just
        // bounds-checked offset+len against capacity. dst is the caller's
        // contract. Specialize the common cell-grain sizes — these are
        // single typed loads instead of memcpy calls. Wider sizes use
        // 128-bit (`u128`) or 256-bit (paired) loads where supported;
        // the compiler lowers these to SIMD on AVX/NEON targets.
        let src = unsafe { (hit.data_ptr as *const u8).add(handle.offset as usize) };
        unsafe { copy_specialized(src, dst, len as usize) }
        Ok(())
    }

    /// Symmetric to `fast_read` for writes.
    ///
    /// SAFETY: `src` must be valid for `len` bytes.
    #[inline]
    pub unsafe fn fast_write(
        &mut self,
        handle: CoreHandle,
        src: *const u8,
        len: u32,
    ) -> Result<(), CoreError> {
        let hit = match self.cache.lookup_fast(handle.region_id) {
            Some(h) => h,
            None => self.refresh_cache(handle.region_id)?,
        };
        if hit.generation != handle.generation {
            return Err(CoreError::StaleHandle);
        }
        if !hit.resident {
            return Err(CoreError::NotResident);
        }
        let end = handle
            .offset
            .checked_add(len)
            .ok_or(CoreError::OutOfBounds)?;
        if end > hit.capacity {
            return Err(CoreError::OutOfBounds);
        }
        let dst = unsafe { (hit.data_ptr as *mut u8).add(handle.offset as usize) };
        unsafe { copy_specialized(src, dst, len as usize) }
        Ok(())
    }

    fn refresh_cache(&mut self, region_id: u16) -> Result<tvm_core::FastHit, CoreError> {
        let info = *self.directory.region_info(region_id)?;
        let (ptr, len) = self.directory.region_data_raw(region_id)?;
        Ok(self.cache.install_with_data(&info, ptr, len))
    }

    // ---------- Imported regions (multi-memory unified path) ----------
    //
    // The canonical create path is the free function
    // `imported::create_imported_in_store(store, kind, capacity)` —
    // creating an imported region needs both `&mut store.data()` (the
    // host) and `&mut store.as_context_mut()` (the wasmtime store
    // context). Those two borrows are disjoint at the `Store` level but
    // can't both be acquired through `&mut self` here, so we provide
    // the helper at the free-function level instead. See
    // `imported::create_imported_in_store` for the canonical entry point.

    /// Find an imported region by id.
    pub fn imported_region(&self, region_id: u16) -> Option<&crate::imported::ImportedRegion> {
        self.imported.iter().find(|r| r.meta.id == region_id)
    }

    pub fn imported_region_mut(
        &mut self,
        region_id: u16,
    ) -> Option<&mut crate::imported::ImportedRegion> {
        self.imported.iter_mut().find(|r| r.meta.id == region_id)
    }

    /// Allocate inside an imported region. Returns the packed handle
    /// the guest can decode for native access.
    pub fn imported_alloc(
        &mut self,
        region_id: u16,
        size: u32,
    ) -> Result<tvm_core::Handle, CoreError> {
        let r = self
            .imported_region_mut(region_id)
            .ok_or(CoreError::RegionNotFound(region_id))?;
        r.alloc(size)
    }

    pub fn imported_dealloc(&mut self, handle: tvm_core::Handle) -> Result<(), CoreError> {
        let r = self
            .imported_region_mut(handle.region_id)
            .ok_or(CoreError::RegionNotFound(handle.region_id))?;
        r.dealloc(handle)
    }

    /// Register all imported regions in a wasmtime Linker so the guest
    /// can import them by name (`tvm.r<id>`). Call once before
    /// instantiating the guest module.
    pub fn register_imported<T>(
        &self,
        store: &mut wasmtime::StoreContextMut<'_, T>,
        linker: &mut wasmtime::Linker<T>,
    ) -> wasmtime::Result<()> {
        for region in &self.imported {
            let name = region.import_name();
            linker.define(&mut *store, "tvm", &name, region.memory)?;
        }
        Ok(())
    }
}

impl AsMut<TvmHost> for TvmHost {
    fn as_mut(&mut self) -> &mut TvmHost {
        self
    }
}

// ---------- TvmFacade: deployment-agnostic interface ----------
//
// Code generic over `T: TvmFacade` works against TvmHost the same way
// it works against guest-side `GuestTvm`. The facade only covers
// deployment-agnostic operations; spill / runtime / multi-store /
// imports stay on the concrete type.

impl TvmFacade for TvmHost {
    fn create_region(&mut self, kind: CoreRegionKind, capacity: u32) -> tvm_core::Result<u16> {
        TvmHost::create_region(self, kind, capacity)
    }
    fn alloc(&mut self, region: u16, size: u32) -> tvm_core::Result<CoreHandle> {
        TvmHost::alloc(self, region, size)
    }
    fn dealloc(&mut self, handle: CoreHandle) -> tvm_core::Result<()> {
        self.directory.dealloc(handle)
    }
    fn read(&mut self, handle: CoreHandle, buf: &mut [u8]) -> tvm_core::Result<()> {
        self.read_bytes(handle, buf)
    }
    fn write(&mut self, handle: CoreHandle, data: &[u8]) -> tvm_core::Result<()> {
        self.write_bytes(handle, data)
    }
    fn pin(&mut self, region: u16) -> tvm_core::Result<()> {
        self.directory.pin(region)
    }
    fn unpin(&mut self, region: u16) -> tvm_core::Result<()> {
        self.directory.unpin(region)
    }
    fn region_info(&self, region: u16) -> tvm_core::Result<CoreRegion> {
        self.directory.region_info(region).copied()
    }
}

impl TvmSpill for TvmHost {
    fn spill(&mut self, region: u16) -> tvm_core::Result<()> {
        let TvmHost {
            directory, backing, ..
        } = self;
        let backing = backing
            .as_mut()
            .ok_or_else(|| CoreError::BackingStore("no backing store configured".into()))?;
        directory.spill_region(region, backing)
    }
    fn load(&mut self, region: u16) -> tvm_core::Result<()> {
        let TvmHost {
            directory, backing, ..
        } = self;
        let backing = backing
            .as_mut()
            .ok_or_else(|| CoreError::BackingStore("no backing store configured".into()))?;
        directory.load_region(region, backing)
    }
}

impl TypesHost for TvmHost {}

impl ManagerHost for TvmHost {
    fn create_region(&mut self, kind: RegionKind, capacity: u32) -> Result<u16, TvmError> {
        let core_kind = to_core_kind(kind);
        self.directory
            .create_region_with(
                core_kind,
                capacity,
                self.default_allocator,
                VecBackedRegion::new(capacity),
            )
            .map_err(to_wit_err)
    }

    fn destroy_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.cache.invalidate(region_id);
        self.directory.destroy_region(region_id).map_err(to_wit_err)
    }

    fn alloc(&mut self, region_id: u16, size: u32) -> Result<Handle, TvmError> {
        self.directory
            .alloc(region_id, size)
            .map(to_wit_handle)
            .map_err(to_wit_err)
    }

    fn dealloc(&mut self, ptr: Handle) -> Result<(), TvmError> {
        self.directory
            .dealloc(to_core_handle(ptr))
            .map_err(to_wit_err)
    }

    fn describe_region(&mut self, region_id: u16) -> Result<RegionInfo, TvmError> {
        self.directory
            .region_info(region_id)
            .map(|info| RegionInfo {
                id: info.id,
                generation: info.generation,
                kind: from_core_kind(info.kind),
                capacity: info.capacity,
                used: info.used,
                residency: from_core_residency(info.residency),
            })
            .map_err(to_wit_err)
    }

    fn promote_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.cache.invalidate(region_id);
        let TvmHost {
            directory, backing, ..
        } = self;
        let backing = backing
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        directory
            .promote_region(region_id, backing)
            .map_err(to_wit_err)
    }

    fn demote_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.cache.invalidate(region_id);
        let TvmHost {
            directory, backing, ..
        } = self;
        let backing = backing
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        directory
            .demote_region(region_id, backing)
            .map_err(to_wit_err)
    }

    fn pin(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.directory.pin(region_id).map_err(to_wit_err)
    }

    fn unpin(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.directory.unpin(region_id).map_err(to_wit_err)
    }

    fn spill_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.cache.invalidate(region_id);
        let backing = self
            .backing
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        self.directory
            .spill_region(region_id, backing)
            .map_err(to_wit_err)
    }

    fn load_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        self.cache.invalidate(region_id);
        let backing = self
            .backing
            .as_mut()
            .ok_or_else(|| TvmError::BackingStore("no backing store configured".into()))?;
        self.directory
            .load_region(region_id, backing)
            .map_err(to_wit_err)
    }

    fn compact_region(&mut self, region_id: u16) -> Result<CompactResult, TvmError> {
        self.cache.invalidate(region_id);
        let remap = self
            .directory
            .compact_region(region_id)
            .map_err(to_wit_err)?;
        let mut mapping: Vec<(u32, u32)> = remap.mapping.into_iter().collect();
        mapping.sort_by_key(|p| p.0);
        Ok(CompactResult {
            old_generation: remap.old_generation,
            new_generation: remap.new_generation,
            mapping,
        })
    }
}

impl BytesHost for TvmHost {
    fn read(&mut self, ptr: Handle, len: u32) -> Result<Vec<u8>, TvmError> {
        let mut buf = vec![0u8; len as usize];
        let h = to_core_handle(ptr);
        let TvmHost {
            directory, backing, ..
        } = self;
        let result = match backing.as_mut() {
            Some(b) => directory.read_or_fault(h, &mut buf, b),
            None => directory.read(h, &mut buf),
        };
        result.map_err(to_wit_err)?;
        Ok(buf)
    }

    fn write(&mut self, ptr: Handle, data: Vec<u8>) -> Result<(), TvmError> {
        let h = to_core_handle(ptr);
        let TvmHost {
            directory, backing, ..
        } = self;
        let result = match backing.as_mut() {
            Some(b) => directory.write_or_fault(h, &data, b),
            None => directory.write(h, &data),
        };
        result.map_err(to_wit_err)
    }

    fn copy(&mut self, src: Handle, dst: Handle, len: u32) -> Result<(), TvmError> {
        let mut buf = vec![0u8; len as usize];
        self.directory
            .read(to_core_handle(src), &mut buf)
            .map_err(to_wit_err)?;
        self.directory
            .write(to_core_handle(dst), &buf)
            .map_err(to_wit_err)
    }

    fn read_into(
        &mut self,
        src: Handle,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<(), TvmError> {
        self.directory
            .read_into(to_core_handle(src), dst_region, dst_offset, len)
            .map_err(to_wit_err)
    }

    fn write_from(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst: Handle,
        len: u32,
    ) -> Result<(), TvmError> {
        self.directory
            .write_from(src_region, src_offset, to_core_handle(dst), len)
            .map_err(to_wit_err)
    }

    fn copy_region(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<(), TvmError> {
        self.directory
            .cross_region_copy(src_region, src_offset, dst_region, dst_offset, len)
            .map_err(to_wit_err)
    }
}

impl DiagnosticsHost for TvmHost {
    fn list_regions(&mut self) -> Vec<RegionInfo> {
        self.directory
            .iter()
            .map(|info| RegionInfo {
                id: info.id,
                generation: info.generation,
                kind: from_core_kind(info.kind),
                capacity: info.capacity,
                used: info.used,
                residency: from_core_residency(info.residency),
            })
            .collect()
    }

    fn fault_count(&mut self, region_id: u16) -> u64 {
        self.directory
            .metrics(region_id)
            .map(|m| m.snapshot().faults)
            .unwrap_or(0)
    }

    fn allocation_count(&mut self, region_id: u16) -> u64 {
        self.directory
            .metrics(region_id)
            .map(|m| m.snapshot().allocations)
            .unwrap_or(0)
    }

    fn bytes_read_count(&mut self, region_id: u16) -> u64 {
        self.directory
            .metrics(region_id)
            .map(|m| m.snapshot().bytes_read)
            .unwrap_or(0)
    }

    fn bytes_written_count(&mut self, region_id: u16) -> u64 {
        self.directory
            .metrics(region_id)
            .map(|m| m.snapshot().bytes_written)
            .unwrap_or(0)
    }

    fn metrics_snapshot(&mut self, region_id: u16) -> Result<RegionMetrics, TvmError> {
        let snap = self
            .directory
            .metrics(region_id)
            .map(|m| m.snapshot())
            .map_err(to_wit_err)?;
        Ok(RegionMetrics {
            allocations: snap.allocations,
            bytes_allocated: snap.bytes_allocated,
            bytes_read: snap.bytes_read,
            bytes_written: snap.bytes_written,
            faults: snap.faults,
            promotions: snap.promotions,
            demotions: snap.demotions,
        })
    }
}

fn to_core_kind(k: RegionKind) -> CoreRegionKind {
    match k {
        RegionKind::HotHeap => CoreRegionKind::HotHeap,
        RegionKind::ObjectArena => CoreRegionKind::ObjectArena,
        RegionKind::BlobArena => CoreRegionKind::BlobArena,
        RegionKind::PageStore => CoreRegionKind::PageStore,
        RegionKind::Scratch => CoreRegionKind::Scratch,
        RegionKind::DeviceState => CoreRegionKind::DeviceState,
        RegionKind::CodeCache => CoreRegionKind::CodeCache,
    }
}

fn from_core_kind(k: CoreRegionKind) -> RegionKind {
    match k {
        CoreRegionKind::HotHeap => RegionKind::HotHeap,
        CoreRegionKind::ObjectArena => RegionKind::ObjectArena,
        CoreRegionKind::BlobArena => RegionKind::BlobArena,
        CoreRegionKind::PageStore => RegionKind::PageStore,
        CoreRegionKind::Scratch => RegionKind::Scratch,
        CoreRegionKind::DeviceState => RegionKind::DeviceState,
        CoreRegionKind::CodeCache => RegionKind::CodeCache,
    }
}

fn from_core_residency(r: CoreResidency) -> Residency {
    match r {
        CoreResidency::Hot => Residency::Hot,
        CoreResidency::Warm => Residency::Warm,
        CoreResidency::Cold => Residency::Cold,
        CoreResidency::External => Residency::External,
    }
}

fn to_core_handle(h: Handle) -> CoreHandle {
    CoreHandle {
        region_id: h.region_id,
        generation: h.generation,
        offset: h.offset,
    }
}

fn to_wit_handle(h: CoreHandle) -> Handle {
    Handle {
        region_id: h.region_id,
        generation: h.generation,
        offset: h.offset,
    }
}

fn to_wit_err(e: CoreError) -> TvmError {
    match e {
        CoreError::RegionNotFound(id) => TvmError::RegionNotFound(id),
        CoreError::StaleHandle => TvmError::StaleHandle,
        CoreError::OutOfBounds => TvmError::OutOfBounds,
        CoreError::NotResident => TvmError::NotResident,
        CoreError::AllocationFailed => TvmError::AllocationFailed,
        CoreError::BackingStore(s) => TvmError::BackingStore(s),
        CoreError::Pinned => TvmError::Pinned,
        CoreError::UnsupportedAllocator => {
            TvmError::BackingStore("unsupported by allocator".into())
        }
        CoreError::PolicyViolation => TvmError::BackingStore("forbidden by region policy".into()),
    }
}
