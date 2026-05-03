//! End-to-end: a self-contained TVM-MM module instantiated with **no
//! host imports whatsoever**. Proves the pure-guest design works.
//!
//! Workload: write a value into pool 1, write a different value into
//! pool 2, sum them via the dispatch helpers, return. If pool isolation
//! works, this returns the sum; if pool dispatch is wrong, we'd read
//! the wrong memory.

use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams};
use wasmtime::{Config, Engine, Linker, Module, Store};

const USER_BODY: &str = r#"
    ;; Write `lhs` to pool 1 offset 0, write `rhs` to pool 2 offset 0,
    ;; then read both back and sum them. Returns the sum.
    (func (export "exercise") (param $lhs i32) (param $rhs i32) (result i32)
      (call $tvm_store_u32 (i32.const 1) (i32.const 0) (local.get $lhs))
      (call $tvm_store_u32 (i32.const 2) (i32.const 0) (local.get $rhs))
      (i32.add
        (call $tvm_load_u32 (i32.const 1) (i32.const 0))
        (call $tvm_load_u32 (i32.const 2) (i32.const 0))))

    ;; Cross-pool isolation check: write to pool 1, read from pool 0
    ;; (should be zero — different memory).
    (func (export "isolation_check") (param $val i32) (result i32)
      (call $tvm_store_u32 (i32.const 1) (i32.const 0) (local.get $val))
      (call $tvm_load_u32 (i32.const 0) (i32.const 0)))
"#;

fn build_module(n_pools: u32) -> wasmtime::Result<(Engine, Module)> {
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let p = ModuleParams {
        n_pools,
        initial_pages_per_pool: 1,
        max_pages_per_pool: 16,
        user_body: USER_BODY.to_string(),
    };
    let wat = tvm_guest_mm_module_template(&p);
    let module = Module::new(&engine, &wat)?;
    Ok((engine, module))
}

#[test]
fn instantiates_with_zero_host_imports() -> anyhow::Result<()> {
    let (engine, module) = build_module(4)?;
    // Empty linker — no imports defined. Module must instantiate
    // because it imports nothing.
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let _instance = linker.instantiate(&mut store, &module)?;
    Ok(())
}

#[test]
fn cross_pool_writes_and_reads_round_trip() -> anyhow::Result<()> {
    let (engine, module) = build_module(4)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let exercise = instance.get_typed_func::<(i32, i32), i32>(&mut store, "exercise")?;
    let result = exercise.call(&mut store, (40, 2))?;
    assert_eq!(result, 42, "writes and dispatch reads must round-trip");
    Ok(())
}

#[test]
fn pools_are_isolated() -> anyhow::Result<()> {
    let (engine, module) = build_module(4)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let check = instance.get_typed_func::<i32, i32>(&mut store, "isolation_check")?;
    // Write 12345 to pool 1, read pool 0. Expect 0 (pools are independent).
    let result = check.call(&mut store, 12345)?;
    assert_eq!(result, 0, "pool 0 must be unaffected by writes to pool 1");
    Ok(())
}

#[test]
fn module_declares_n_memories() -> anyhow::Result<()> {
    use wasmtime::Extern;
    let (engine, module) = build_module(8)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    // Each pool was exported as memN; count them.
    let mut count = 0;
    for i in 0..16 {
        let name = format!("mem{}", i);
        if let Some(Extern::Memory(_)) = instance.get_export(&mut store, &name) {
            count += 1;
        }
    }
    assert_eq!(count, 8, "expected 8 memories exported as mem0..mem7");
    Ok(())
}
