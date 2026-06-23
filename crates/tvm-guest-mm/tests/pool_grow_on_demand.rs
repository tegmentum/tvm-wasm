//! Pool auto-grow on demand: a `tvm_copy_from_default` (or its
//! per-pool specialized sibling) that targets a destination offset
//! past the pool's `initial_pages_per_pool` must grow the pool to
//! cover the write rather than trapping with an OOB error.
//!
//! Motivation: rustc-emitted cdylib code targets only `memory 0`
//! (the user's default memory). It can't issue `memory.grow K` for
//! pool K directly, so the substrate is the only place that can
//! grow a pool. Without this, any callsite that lays out
//! per-region storage at offsets above `initial_pages * 64 KiB`
//! traps on its first cross-boundary write. sqlite-lib's
//! `MultiMemoryStorage` does exactly this: each open file reserves
//! a 256 MiB slice in pool 2 via a bump cursor, so the second-file
//! journal lands at offset 256 MiB and immediately hits the
//! shell's pool size unless the shell grows on demand.

use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams};
use wasmtime::{Config, Engine, Linker, Module, Store};

// Workload: write `len` bytes from default memory pool 0 into pool
// 2 at offset `dst_off`. Uses `tvm_copy_from_default`. The default
// memory has 1 page (64 KiB); we keep `len` small enough to fit.
//
// Also exposes `write_p2_specialized` that targets pool 2
// specifically via `tvm_copy_from_default_p2` so we can confirm
// the per-pool specialized helper grows the same way.
const USER_BODY: &str = r#"
    ;; Stage a 16-byte payload into the default memory at offset 0.
    (data (i32.const 0) "\01\02\03\04\05\06\07\08\09\0a\0b\0c\0d\0e\0f\10")

    (func (export "write_p2_generic")
        (param $dst_off i32) (param $len i32)
      (call $tvm_copy_from_default
        (i32.const 2)              ;; dst_pool
        (local.get $dst_off)
        (i32.const 0)              ;; src_off in default memory
        (local.get $len)))

    (func (export "write_p2_specialized")
        (param $dst_off i32) (param $len i32)
      (call $tvm_copy_from_default_p2
        (local.get $dst_off)
        (i32.const 0)              ;; src_off in default memory
        (local.get $len)))

    ;; Read a single u32 back from pool 2 at runtime offset.
    (func (export "read_p2_u32") (param $off i32) (result i32)
      (call $tvm_load_u32 (i32.const 2) (local.get $off)))
"#;

fn build_module(initial_pages: u32, max_pages: u32) -> wasmtime::Result<(Engine, Module)> {
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: initial_pages,
        max_pages_per_pool: max_pages,
        user_body: USER_BODY.to_string(),
    };
    let wat = tvm_guest_mm_module_template(&p);
    let module = Module::new(&engine, &wat)?;
    Ok((engine, module))
}

#[test]
fn generic_copy_from_default_grows_pool_past_initial_pages() -> anyhow::Result<()> {
    // Pool 2 starts at 1 page (64 KiB) and can grow to 16 pages
    // (1 MiB). We write 16 bytes at offset 128 KiB — well past the
    // initial page boundary — and expect the dispatcher to grow
    // the pool, perform the copy, and let us read the value back.
    //
    // Before the auto-grow fix this would trap with "out of bounds
    // memory access" inside `tvm_copy_from_default`.
    let (engine, module) = build_module(1, 16)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    let write =
        instance.get_typed_func::<(i32, i32), ()>(&mut store, "write_p2_generic")?;
    let read = instance.get_typed_func::<i32, i32>(&mut store, "read_p2_u32")?;

    let dst_off: i32 = 128 * 1024; // 2 × initial pool size
    write.call(&mut store, (dst_off, 16))?;

    // The first 4 bytes of the staged payload are 0x01, 0x02, 0x03,
    // 0x04 — little-endian read as 0x04030201.
    let got = read.call(&mut store, dst_off)?;
    assert_eq!(
        got as u32, 0x04030201,
        "value written past initial_pages must read back via tvm_load_u32"
    );
    Ok(())
}

#[test]
fn specialized_copy_from_default_grows_pool_past_initial_pages() -> anyhow::Result<()> {
    // Same as above but goes through the per-pool specialized
    // export `tvm_copy_from_default_p2`. Auto-grow has to happen
    // there too — sqlite-lib's `write_bytes(VFS_POOL, …)` ends up
    // monomorphizing on this path through `tvm-guest-mm-rt`.
    let (engine, module) = build_module(1, 16)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    let write = instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "write_p2_specialized")?;
    let read = instance.get_typed_func::<i32, i32>(&mut store, "read_p2_u32")?;

    let dst_off: i32 = 256 * 1024; // 4 × initial pool size
    write.call(&mut store, (dst_off, 16))?;
    let got = read.call(&mut store, dst_off)?;
    assert_eq!(
        got as u32, 0x04030201,
        "specialized helper must grow the destination pool too"
    );
    Ok(())
}

#[test]
fn grow_at_exact_page_boundary() -> anyhow::Result<()> {
    // Edge case: write 16 bytes starting at exactly the boundary
    // (offset = initial_pages * 65536). Crosses no boundary
    // partially; the whole copy lands in the page we have to grow
    // into. Before the fix this traps; after, it grows by exactly
    // 1 page and succeeds.
    let (engine, module) = build_module(1, 16)?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    let write =
        instance.get_typed_func::<(i32, i32), ()>(&mut store, "write_p2_generic")?;
    let read = instance.get_typed_func::<i32, i32>(&mut store, "read_p2_u32")?;

    let dst_off: i32 = 65536; // exact pool size in bytes
    write.call(&mut store, (dst_off, 16))?;
    let got = read.call(&mut store, dst_off)?;
    assert_eq!(got as u32, 0x04030201);
    Ok(())
}

#[test]
fn grow_to_max_pages_then_overflow_traps() -> anyhow::Result<()> {
    // Past `max_pages_per_pool` the grow returns -1 and the copy
    // traps — the grow request can't be satisfied. We accept the
    // trap; what we don't accept is corrupting memory or silently
    // succeeding past the configured cap.
    let (engine, module) = build_module(1, 2)?; // 1 page initial, 2 max
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let write =
        instance.get_typed_func::<(i32, i32), ()>(&mut store, "write_p2_generic")?;

    // Offset that requires more pages than max allows.
    let result = write.call(&mut store, (3 * 65536, 16));
    assert!(
        result.is_err(),
        "writing past max_pages must still trap; got Ok"
    );
    Ok(())
}
