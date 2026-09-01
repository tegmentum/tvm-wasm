#![allow(deprecated)] // ADR-0029 Phase 6.9.d Session 7 — this test/bench intentionally exercises the deprecated wit-bindgen raw entry points to guard the reference implementation while it coexists with `raw_linker_wasmos`.

//! End-to-end coverage of the raw fast-path linker.

use tvm_core::RegionKind as CoreRegionKind;
use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::{add_raw_imports, TvmHost};
use wasmtime::{Engine, Instance, Linker, Module, Store};

const RAW_GUEST_WAT: &str = r#"
(module
  (import "tvm" "alloc"   (func $alloc   (param i32 i32) (result i64)))
  (import "tvm" "write"   (func $write   (param i64 i32 i32) (result i32)))
  (import "tvm" "read"    (func $read    (param i64 i32 i32) (result i32)))
  (memory (export "memory") 1)

  ;; Allocate `size` bytes in region `region_id`. Returns the packed handle
  ;; (or 0 on failure).
  (func (export "alloc") (param $region i32) (param $size i32) (result i64)
    (call $alloc (local.get $region) (local.get $size)))

  ;; Write `len` bytes from guest memory at ptr to the region pointed at by
  ;; the packed handle. Returns the host error code (0 on success).
  (func (export "write") (param $h i64) (param $ptr i32) (param $len i32) (result i32)
    (call $write (local.get $h) (local.get $ptr) (local.get $len)))

  (func (export "read") (param $h i64) (param $ptr i32) (param $len i32) (result i32)
    (call $read (local.get $h) (local.get $ptr) (local.get $len)))

  ;; Sum the first `len` bytes of guest memory starting at offset 0.
  (func (export "sum") (param $len i32) (result i32)
    (local $i i32) (local $acc i32)
    (block $break
      (loop $continue
        (br_if $break (i32.eq (local.get $i) (local.get $len)))
        (local.set $acc
          (i32.add (local.get $acc)
                   (i32.load8_u (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)))
    (local.get $acc))
)
"#;

fn instantiate(host: TvmHost) -> anyhow::Result<(Store<TvmHost>, Instance)> {
    let engine = Engine::default();
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;

    let module = Module::new(&engine, RAW_GUEST_WAT)?;
    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;
    Ok((store, instance))
}

#[test]
fn raw_alloc_write_read_round_trip() -> anyhow::Result<()> {
    let mut host = TvmHost::new();
    // Set up a region from the host side; the fast path doesn't expose
    // create-region.
    let region = ManagerHost::create_region(&mut host, RegionKind::HotHeap, 256)?;
    assert_eq!(region, 0);

    let (mut store, instance) = instantiate(host)?;

    let alloc = instance.get_typed_func::<(i32, i32), i64>(&mut store, "alloc")?;
    let write = instance.get_typed_func::<(i64, i32, i32), i32>(&mut store, "write")?;
    let read = instance.get_typed_func::<(i64, i32, i32), i32>(&mut store, "read")?;
    let sum = instance.get_typed_func::<i32, i32>(&mut store, "sum")?;

    let handle = alloc.call(&mut store, (region as i32, 4))?;
    assert!(handle != 0, "alloc returned null handle");

    // Stage 4 bytes in guest memory at offset 0, write them into the region.
    let mem = instance.get_memory(&mut store, "memory").unwrap();
    mem.write(&mut store, 0, &[1, 2, 3, 4])?;
    let err = write.call(&mut store, (handle, 0, 4))?;
    assert_eq!(err, 0);

    // Zero the buffer so the read is observable.
    mem.write(&mut store, 0, &[0, 0, 0, 0])?;
    let err = read.call(&mut store, (handle, 0, 4))?;
    assert_eq!(err, 0);

    let total = sum.call(&mut store, 4)?;
    assert_eq!(total, 10);

    // Host directory must reflect the alloc.
    let host = store.data();
    assert_eq!(host.directory.region_info(region)?.used, 4);

    Ok(())
}

#[test]
fn raw_alloc_failure_records_last_error() -> anyhow::Result<()> {
    let mut host = TvmHost::new();
    let region = ManagerHost::create_region(&mut host, RegionKind::Scratch, 16)?;
    let (mut store, instance) = instantiate(host)?;
    let alloc = instance.get_typed_func::<(i32, i32), i64>(&mut store, "alloc")?;

    // Capacity 16, ask for 32: should fail.
    let result = alloc.call(&mut store, (region as i32, 32))?;
    assert_eq!(result, 0);
    assert_eq!(
        store.data().last_raw_error,
        tvm_wasmtime::raw_linker::ERR_ALLOC_FAILED
    );

    Ok(())
}

#[test]
fn resolve_cache_hits_after_warm_up() {
    let mut host = TvmHost::new();
    let region = ManagerHost::create_region(&mut host, RegionKind::HotHeap, 64).unwrap();
    let _ = host.resolve(region).unwrap(); // miss
    let _ = host.resolve(region).unwrap(); // hit
    let _ = host.resolve(region).unwrap(); // hit
    assert_eq!(host.cache.hits, 2);
    assert_eq!(host.cache.misses, 1);
}

#[test]
fn resolve_cache_invalidated_on_destroy() {
    let mut host = TvmHost::new();
    let region = ManagerHost::create_region(&mut host, RegionKind::Scratch, 32).unwrap();
    let _ = host.resolve(region).unwrap();
    ManagerHost::destroy_region(&mut host, region).unwrap();
    let before = host.cache.invalidations;
    assert!(before >= 1);
    // Lookup after destroy is a miss + falls through to directory which errors.
    assert!(host.resolve(region).is_err());
}

#[test]
fn cross_region_copy_via_directory_helper() {
    let mut dir: tvm_core::RegionDirectory<tvm_core::VecBackedRegion> =
        tvm_core::RegionDirectory::new();
    let src = dir
        .create_region(
            CoreRegionKind::HotHeap,
            32,
            tvm_core::VecBackedRegion::new(32),
        )
        .unwrap();
    let dst = dir
        .create_region(
            CoreRegionKind::Scratch,
            32,
            tvm_core::VecBackedRegion::new(32),
        )
        .unwrap();
    let h = dir.alloc(src, 4).unwrap();
    dir.write(h, b"PING").unwrap();
    dir.cross_region_copy(src, h.offset, dst, 0, 4).unwrap();
    let dst_h = tvm_core::Handle {
        region_id: dst,
        generation: 1,
        offset: 0,
    };
    let mut buf = [0u8; 4];
    dir.read(dst_h, &mut buf).unwrap();
    assert_eq!(&buf, b"PING");
}
