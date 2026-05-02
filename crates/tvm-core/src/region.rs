use crate::residency::Residency;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    HotHeap,
    ObjectArena,
    BlobArena,
    PageStore,
    Scratch,
    DeviceState,
    CodeCache,
}

/// Region metadata. Field order is **load-bearing**: the first 12 bytes
/// (`id`, `generation`, `capacity`, `used`) are accessed by every cache
/// refresh and validate. Keeping them at the front ensures they share a
/// cache line and load in a single 16-byte access on most ISAs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Region {
    // --- hot fields (cache line 1) ---
    pub id: u16,
    pub generation: u16,
    pub capacity: u32,
    pub used: u32,
    // --- warm fields ---
    pub kind: RegionKind,
    pub residency: Residency,
    pub pinned: bool,
    pub pinnable: bool,
    pub spillable: bool,
}
