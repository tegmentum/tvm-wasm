#![allow(deprecated)] // ADR-0029 Phase 6.9.d Session 7 — this test/bench intentionally exercises the deprecated wit-bindgen raw entry points to guard the reference implementation while it coexists with `raw_linker_wasmos`.

//! Verify that the raw and WIT paths coexist correctly against the same
//! TvmHost: state mutated through one path is visible from the other.

use tvm_wasmtime::bindings::tvm::memory::bytes::Host as BytesHost;
use tvm_wasmtime::bindings::tvm::memory::diagnostics::Host as DiagnosticsHost;
use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::{add_raw_imports, TvmHost};
use wasmtime::{Engine, Linker, Module, Store};

const RAW_GUEST: &str = r#"
(module
  (import "tvm" "alloc" (func $alloc (param i32 i32) (result i64)))
  (import "tvm" "write" (func $write (param i64 i32 i32) (result i32)))
  (memory (export "memory") 1)

  (func (export "alloc_via_raw") (param $r i32) (param $size i32) (result i64)
    (call $alloc (local.get $r) (local.get $size)))

  (func (export "write_via_raw") (param $h i64) (param $ptr i32) (param $len i32) (result i32)
    (call $write (local.get $h) (local.get $ptr) (local.get $len)))
)
"#;

#[test]
fn raw_alloc_visible_to_wit_read() -> anyhow::Result<()> {
    let mut host = TvmHost::new();
    let region = ManagerHost::create_region(&mut host, RegionKind::HotHeap, 256)?;

    let engine = Engine::default();
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;
    let module = Module::new(&engine, RAW_GUEST)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;

    let alloc = instance.get_typed_func::<(i32, i32), i64>(&mut store, "alloc_via_raw")?;
    let write = instance.get_typed_func::<(i64, i32, i32), i32>(&mut store, "write_via_raw")?;
    let mem = instance.get_memory(&mut store, "memory").unwrap();

    let packed = alloc.call(&mut store, (region as i32, 8))?;
    assert!(packed != 0);

    mem.write(&mut store, 0, b"raw->wit")?;
    let err = write.call(&mut store, (packed, 0, 8))?;
    assert_eq!(err, 0);

    // Read back through the WIT bindings — the same host owns both paths,
    // so the bytes must be visible.
    use tvm_wasmtime::bindings::tvm::memory::types::Handle;
    let wit_handle = Handle {
        region_id: ((packed >> 48) & 0xFFFF) as u16,
        generation: ((packed >> 32) & 0xFFFF) as u16,
        offset: (packed & 0xFFFF_FFFF) as u32,
    };
    let bytes = BytesHost::read(store.data_mut(), wit_handle, 8)?;
    assert_eq!(&bytes, b"raw->wit");

    Ok(())
}

#[test]
fn wit_write_visible_via_metrics_after_raw_alloc() -> anyhow::Result<()> {
    let mut host = TvmHost::new();
    let r = ManagerHost::create_region(&mut host, RegionKind::Scratch, 64)?;
    // Allocate via WIT, write via WIT — count traffic.
    let h = ManagerHost::alloc(&mut host, r, 16)?;
    BytesHost::write(&mut host, h, vec![0u8; 16])?;
    BytesHost::write(&mut host, h, vec![0u8; 16])?;

    // Counters reflect WIT-path writes.
    assert_eq!(DiagnosticsHost::bytes_written_count(&mut host, r), 32);
    assert_eq!(DiagnosticsHost::allocation_count(&mut host, r), 1);

    Ok(())
}
