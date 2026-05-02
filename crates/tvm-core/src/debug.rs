use std::fmt::Write;

use crate::directory::{MemoryRegion, RegionDirectory};
use crate::handle::Handle;
use crate::region::Region;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleStatus {
    Valid,
    StaleGeneration,
    UnknownRegion,
    OutOfBounds,
    NotResident,
}

pub fn dump_region_layout<M: MemoryRegion>(dir: &RegionDirectory<M>) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "regions: {}  (id  gen  kind          residency  pin  used/cap)",
        dir.len()
    );
    for r in dir.iter() {
        let _ = writeln!(
            out,
            "  {:>3}  {:>3}  {:<13?} {:<10?} {:<3}  {}/{}",
            r.id,
            r.generation,
            r.kind,
            r.residency,
            if r.pinned { "yes" } else { "no" },
            r.used,
            r.capacity,
        );
    }
    out
}

pub fn validate_handle<M: MemoryRegion>(
    dir: &RegionDirectory<M>,
    handle: Handle,
) -> HandleStatus {
    let info: &Region = match dir.region_info(handle.region_id) {
        Ok(i) => i,
        Err(_) => return HandleStatus::UnknownRegion,
    };
    if info.generation != handle.generation {
        return HandleStatus::StaleGeneration;
    }
    if handle.offset >= info.capacity {
        return HandleStatus::OutOfBounds;
    }
    if !matches!(info.residency, crate::residency::Residency::Hot | crate::residency::Residency::Warm) {
        return HandleStatus::NotResident;
    }
    HandleStatus::Valid
}

pub fn validate_handles<M: MemoryRegion>(
    dir: &RegionDirectory<M>,
    handles: &[Handle],
) -> Vec<(Handle, HandleStatus)> {
    handles
        .iter()
        .map(|h| (*h, validate_handle(dir, *h)))
        .collect()
}

pub fn fault_counts<M: MemoryRegion>(dir: &RegionDirectory<M>) -> Vec<(u16, u64)> {
    dir.iter()
        .map(|r| (r.id, dir.metrics(r.id).unwrap().snapshot().faults))
        .collect()
}
