//! `add_raw_shared`: cross-store region visibility via one `SharedTvmHost`,
//! and (decisively) that the per-store guest-memory cache hazard is avoided
//! — store B must read exactly what store A wrote, using B's OWN memory.

use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::{add_raw_shared, SharedTvmHost};
use wasmtime::{Engine, Linker, Module, Store};

const G: &str = r#"
(module
  (import "tvm" "alloc" (func $alloc (param i32 i32) (result i64)))
  (import "tvm" "write" (func $write (param i64 i32 i32) (result i32)))
  (import "tvm" "read"  (func $read  (param i64 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "fill") (param $n i32)
    (local $i i32)
    (block $b (loop $l (br_if $b (i32.eq (local.get $i) (local.get $n)))
      (i32.store8 (local.get $i) (i32.and (local.get $i) (i32.const 0xff)))
      (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $l))))
  (func (export "alloc") (param i32 i32) (result i64)
    (call $alloc (local.get 0) (local.get 1)))
  (func (export "write") (param i64 i32 i32) (result i32)
    (call $write (local.get 0) (local.get 1) (local.get 2)))
  (func (export "read") (param i64 i32 i32) (result i32)
    (call $read (local.get 0) (local.get 1) (local.get 2)))
  (func (export "sum") (param $n i32) (result i32)
    (local $i i32) (local $a i32)
    (block $b (loop $l (br_if $b (i32.eq (local.get $i) (local.get $n)))
      (local.set $a (i32.add (local.get $a) (i32.load8_u (local.get $i))))
      (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $l)))
    (local.get $a)))
"#;

#[test]
fn raw_shared_cross_store_region_visibility() -> anyhow::Result<()> {
    let engine = Engine::default();
    let shared = SharedTvmHost::new();
    // Host pre-creates the region (raw imports expose no `create`); the
    // lock guard is dropped at the block end, before any store runs (the
    // raw closures take the same lock — holding it here would deadlock).
    let region = {
        let mut g = shared.lock();
        ManagerHost::create_region(&mut *g, RegionKind::HotHeap, 4096)?
    };
    let module = Module::new(&engine, G)?;
    let mut linker: Linker<SharedTvmHost> = Linker::new(&engine);
    add_raw_shared(&mut linker)?;

    let n: i32 = 64;

    // Store A: fill its memory [0,n) with i&0xff, alloc in the shared
    // region, write those bytes into it.
    let mut a = Store::new(&engine, shared.clone());
    let ia = linker.instantiate(&mut a, &module)?;
    ia.get_typed_func::<i32, ()>(&mut a, "fill")?
        .call(&mut a, n)?;
    let h = ia
        .get_typed_func::<(i32, i32), i64>(&mut a, "alloc")?
        .call(&mut a, (region as i32, n))?;
    assert_ne!(h, 0, "alloc failed");
    let rc = ia
        .get_typed_func::<(i64, i32, i32), i32>(&mut a, "write")?
        .call(&mut a, (h, 0, n))?;
    assert_eq!(rc, 0, "write rc");

    // Store B (SAME SharedTvmHost, its OWN guest memory): read the same
    // handle and sum it.
    let mut b = Store::new(&engine, shared.clone());
    let ib = linker.instantiate(&mut b, &module)?;
    let rc = ib
        .get_typed_func::<(i64, i32, i32), i32>(&mut b, "read")?
        .call(&mut b, (h, 0, n))?;
    assert_eq!(rc, 0, "read rc");
    let sum = ib
        .get_typed_func::<i32, i32>(&mut b, "sum")?
        .call(&mut b, n)?;

    let expected: i32 = (0..64).map(|i| i & 0xff).sum();
    assert_eq!(
        sum, expected,
        "store B must read exactly what store A wrote — cross-store sharing \
         works AND no per-store cached-memory corruption"
    );
    Ok(())
}
