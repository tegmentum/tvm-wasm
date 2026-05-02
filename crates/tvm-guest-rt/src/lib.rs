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
