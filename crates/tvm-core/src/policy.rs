use crate::region::RegionKind;
use crate::residency::Residency;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementPolicy {
    pub initial_residency: Residency,
    pub pinnable: bool,
    pub spillable: bool,
}

impl PlacementPolicy {
    pub const fn for_kind(kind: RegionKind) -> Self {
        match kind {
            RegionKind::HotHeap | RegionKind::CodeCache | RegionKind::DeviceState => Self {
                initial_residency: Residency::Hot,
                pinnable: true,
                spillable: false,
            },
            RegionKind::ObjectArena | RegionKind::BlobArena => Self {
                initial_residency: Residency::Hot,
                pinnable: false,
                spillable: true,
            },
            RegionKind::PageStore => Self {
                initial_residency: Residency::Warm,
                pinnable: false,
                spillable: true,
            },
            RegionKind::Scratch => Self {
                initial_residency: Residency::Hot,
                pinnable: false,
                spillable: false,
            },
        }
    }
}
