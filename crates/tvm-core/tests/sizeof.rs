use std::mem::size_of;
use tvm_core::{Handle, Region, RegionAllocator, RegionMetrics, ResolveCache, VecBackedRegion};

#[test]
fn report_sizes() {
    println!("size_of::<Region>() = {}", size_of::<Region>());
    println!(
        "size_of::<RegionMetrics>() = {}",
        size_of::<RegionMetrics>()
    );
    println!(
        "size_of::<RegionAllocator>() = {}",
        size_of::<RegionAllocator>()
    );
    println!("size_of::<Handle>() = {}", size_of::<Handle>());
    println!("size_of::<ResolveCache>() = {}", size_of::<ResolveCache>());
    println!(
        "size_of::<Option<VecBackedRegion>>() = {}",
        size_of::<Option<VecBackedRegion>>()
    );
}
