//! Multi-thread / multi-store sharing via SharedTvmHost.

use std::thread;

use tvm_wasmtime::bindings::tvm::memory::bytes::Host as BytesHost;
use tvm_wasmtime::bindings::tvm::memory::diagnostics::Host as DiagnosticsHost;
use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::{add_shared_to_linker, SharedTvmHost};
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};

#[test]
fn multiple_threads_share_directory() {
    let shared = SharedTvmHost::new();
    let region = ManagerHost::create_region(
        &mut *shared.lock(),
        RegionKind::HotHeap,
        4096,
    )
    .unwrap();

    let mut threads = Vec::new();
    for thread_id in 0..8u8 {
        let shared = shared.clone();
        threads.push(thread::spawn(move || {
            for _ in 0..16 {
                let h = ManagerHost::alloc(&mut *shared.lock(), region, 4).unwrap();
                BytesHost::write(&mut *shared.lock(), h, vec![thread_id; 4]).unwrap();
                let read = BytesHost::read(&mut *shared.lock(), h, 4).unwrap();
                assert_eq!(read, vec![thread_id; 4]);
            }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }

    // 8 threads × 16 allocations each.
    let allocs = DiagnosticsHost::allocation_count(&mut *shared.lock(), region);
    assert_eq!(allocs, 8 * 16);
}

#[test]
fn shared_host_visible_across_clones() {
    let shared = SharedTvmHost::new();
    let r = ManagerHost::create_region(&mut *shared.lock(), RegionKind::Scratch, 64).unwrap();
    let h = ManagerHost::alloc(&mut *shared.lock(), r, 4).unwrap();
    BytesHost::write(&mut *shared.lock(), h, b"test".to_vec()).unwrap();

    // Different clone, same backing state.
    let other = shared.clone();
    let bytes = BytesHost::read(&mut *other.lock(), h, 4).unwrap();
    assert_eq!(&bytes, b"test");
}

#[test]
fn shared_host_via_linker_and_two_stores() -> anyhow::Result<()> {
    let mut config = Config::new();
    config.wasm_component_model(true);
    let engine = Engine::new(&config)?;

    let shared = SharedTvmHost::new();
    // Pre-create a region from outside any store.
    let region =
        ManagerHost::create_region(&mut *shared.lock(), RegionKind::HotHeap, 256)?;

    // Empty component: just verify linker registration succeeds and two
    // stores can hold clones of the same SharedTvmHost.
    let component = Component::new(&engine, "(component)")?;

    let mut linker_a: Linker<SharedTvmHost> = Linker::new(&engine);
    add_shared_to_linker(&mut linker_a)?;
    let mut store_a = Store::new(&engine, shared.clone());
    let _instance_a = linker_a.instantiate(&mut store_a, &component)?;

    let mut linker_b: Linker<SharedTvmHost> = Linker::new(&engine);
    add_shared_to_linker(&mut linker_b)?;
    let mut store_b = Store::new(&engine, shared.clone());
    let _instance_b = linker_b.instantiate(&mut store_b, &component)?;

    // Mutations through one store visible to the other.
    let h = ManagerHost::alloc(&mut *store_a.data().lock(), region, 4)?;
    BytesHost::write(&mut *store_a.data().lock(), h, b"abcd".to_vec())?;
    let bytes = BytesHost::read(&mut *store_b.data().lock(), h, 4)?;
    assert_eq!(&bytes, b"abcd");

    Ok(())
}
