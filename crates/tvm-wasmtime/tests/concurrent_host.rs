//! ConcurrentTvmHost integration: trait-level round-trip and multi-thread
//! sharing across separate stores.

use std::sync::Arc;
use std::thread;

use tvm_wasmtime::bindings::tvm::memory::bytes::Host as BytesHost;
use tvm_wasmtime::bindings::tvm::memory::diagnostics::Host as DiagnosticsHost;
use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::ConcurrentTvmHost;

#[test]
fn round_trip_via_traits() {
    let mut host = ConcurrentTvmHost::new();
    let r = ManagerHost::create_region(&mut host, RegionKind::HotHeap, 64).unwrap();
    let h = ManagerHost::alloc(&mut host, r, 8).unwrap();
    BytesHost::write(&mut host, h, b"concrntz".to_vec()).unwrap();
    let bytes = BytesHost::read(&mut host, h, 8).unwrap();
    assert_eq!(&bytes, b"concrntz");
    assert_eq!(DiagnosticsHost::allocation_count(&mut host, r), 1);
}

#[test]
fn shared_state_across_clones() {
    let host = ConcurrentTvmHost::new();
    let r = ManagerHost::create_region(&mut host.clone(), RegionKind::HotHeap, 64).unwrap();
    let h = ManagerHost::alloc(&mut host.clone(), r, 4).unwrap();
    BytesHost::write(&mut host.clone(), h, b"abcd".to_vec()).unwrap();
    let bytes = BytesHost::read(&mut host.clone(), h, 4).unwrap();
    assert_eq!(&bytes, b"abcd");
}

#[test]
fn concurrent_allocs_distinct_regions_via_host() {
    let host = Arc::new(ConcurrentTvmHost::new());
    let mut regions = Vec::new();
    for _ in 0..8 {
        let r = ManagerHost::create_region(&mut host.as_ref().clone(), RegionKind::HotHeap, 1024)
            .unwrap();
        regions.push(r);
    }

    let mut threads = Vec::new();
    for &region in &regions {
        let host = Arc::clone(&host);
        threads.push(thread::spawn(move || {
            for _ in 0..32 {
                let mut h = host.as_ref().clone();
                ManagerHost::alloc(&mut h, region, 8).unwrap();
            }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }

    for &region in &regions {
        let info = ManagerHost::describe_region(&mut host.as_ref().clone(), region).unwrap();
        assert_eq!(info.used, 32 * 8);
    }
}
