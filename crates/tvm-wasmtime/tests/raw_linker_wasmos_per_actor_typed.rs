//! ADR-0029 D2 typed-args slice 4 (2026-09-02) — typed variant of
//! `raw_linker_wasmos_per_actor.rs`.
//!
//! Mirror of the per-actor test but the composite is built through
//! `add_raw_imports_per_actor_projected_typed`, which registers each
//! handler via `CoreImports::register_typed` — landing on wasmos's
//! `linker.func_new` (sync) path. Verifies the typed dispatch route
//! is at parity with the dynamic path on the same guest surface.

use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::raw_linker_wasmos::add_raw_imports_per_actor_projected_typed;
use tvm_wasmtime::TvmHost;
use wasmos_runtime_api::CoreImports;
use wasmos_runtime_wasmtime_v48::core_import_bridge;
use wasmtime::{Config, Engine, Linker, Module, Store};

const RAW_GUEST_WAT: &str = r#"
(module
  (import "tvm" "alloc"      (func $alloc      (param i32 i32) (result i64)))
  (import "tvm" "dealloc"    (func $dealloc    (param i64)     (result i32)))
  (import "tvm" "write"      (func $write      (param i64 i32 i32) (result i32)))
  (import "tvm" "read"       (func $read       (param i64 i32 i32) (result i32)))
  (import "tvm" "last_error" (func $last_error (result i32)))
  (memory (export "memory") 1)

  (func (export "stage_at") (param $ptr i32) (param $byte i32)
    local.get $ptr local.get $byte i32.store8)

  (func (export "do_alloc") (param $rid i32) (param $sz i32) (result i64)
    local.get $rid local.get $sz call $alloc)

  (func (export "do_write") (param $h i64) (param $p i32) (param $l i32) (result i32)
    local.get $h local.get $p local.get $l call $write)

  (func (export "do_read") (param $h i64) (param $p i32) (param $l i32) (result i32)
    local.get $h local.get $p local.get $l call $read)

  (func (export "do_last_error") (result i32) call $last_error)

  (func (export "load_at") (param $ptr i32) (result i32)
    local.get $ptr i32.load8_u)

  (func (export "do_dealloc") (param $h i64) (result i32)
    local.get $h call $dealloc)
)
"#;

/// Same round-trip flow as the dynamic per-actor test but every
/// handler comes from the typed registration path. Verifies typed
/// dispatch is byte-for-byte equivalent on the alloc / write / read /
/// dealloc quartet.
#[tokio::test(flavor = "multi_thread")]
async fn per_actor_typed_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = Config::new();
    #[allow(deprecated)]
    config.async_support(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, RAW_GUEST_WAT)?;

    // Typed composite — same shape as the dynamic per-actor entry
    // but every registration uses `register_typed`, so the bridge
    // installs each handler via linker.func_new (sync) rather than
    // func_new_async.
    let imports = add_raw_imports_per_actor_projected_typed::<TvmHost>(CoreImports::new());

    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    core_import_bridge::install_core_imports(&mut linker, &module, &imports)?;

    let mut host = TvmHost::default();
    let rid = ManagerHost::create_region(&mut host, RegionKind::HotHeap, 256)?;
    assert_eq!(rid, 0);

    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate_async(&mut store, &module).await?;

    let do_alloc =
        instance.get_typed_func::<(i32, i32), i64>(&mut store, "do_alloc")?;
    let do_write = instance
        .get_typed_func::<(i64, i32, i32), i32>(&mut store, "do_write")?;
    let do_read =
        instance.get_typed_func::<(i64, i32, i32), i32>(&mut store, "do_read")?;
    let stage_at =
        instance.get_typed_func::<(i32, i32), ()>(&mut store, "stage_at")?;
    let load_at = instance.get_typed_func::<i32, i32>(&mut store, "load_at")?;
    let do_dealloc =
        instance.get_typed_func::<i64, i32>(&mut store, "do_dealloc")?;

    let packed = do_alloc.call_async(&mut store, (rid as i32, 4)).await?;
    assert_ne!(packed, 0);

    let dst_ptr = 64i32;
    for (i, &b) in [0x11u8, 0x22, 0x33, 0x44].iter().enumerate() {
        stage_at
            .call_async(&mut store, (dst_ptr + i as i32, b as i32))
            .await?;
    }

    let wr = do_write.call_async(&mut store, (packed, dst_ptr, 4)).await?;
    assert_eq!(wr, 0, "typed write should return ERR_OK, got {wr}");

    for i in 0..4 {
        stage_at.call_async(&mut store, (dst_ptr + i, 0)).await?;
    }
    for i in 0..4 {
        let b = load_at.call_async(&mut store, dst_ptr + i).await?;
        assert_eq!(b, 0);
    }

    let rd = do_read.call_async(&mut store, (packed, dst_ptr, 4)).await?;
    assert_eq!(rd, 0, "typed read should return ERR_OK, got {rd}");
    for (i, &want) in [0x11u8, 0x22, 0x33, 0x44].iter().enumerate() {
        let got = load_at.call_async(&mut store, dst_ptr + i as i32).await?;
        assert_eq!(got, want as i32);
    }

    let dr = do_dealloc.call_async(&mut store, packed).await?;
    assert_eq!(dr, 0, "typed dealloc should return ERR_OK, got {dr}");

    Ok(())
}

/// The typed variant carries the same PerActor guard as the dynamic
/// variant; wrong store data must surface the diagnostic through the
/// wasmtime trap chain.
#[tokio::test(flavor = "multi_thread")]
async fn per_actor_typed_wrong_store_data_errors_gracefully()
-> Result<(), Box<dyn std::error::Error>> {
    let mut config = Config::new();
    #[allow(deprecated)]
    config.async_support(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, RAW_GUEST_WAT)?;

    let imports = add_raw_imports_per_actor_projected_typed::<TvmHost>(CoreImports::new());

    let mut linker: Linker<()> = Linker::new(&engine);
    core_import_bridge::install_core_imports(&mut linker, &module, &imports)?;
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate_async(&mut store, &module).await?;
    let do_alloc =
        instance.get_typed_func::<(i32, i32), i64>(&mut store, "do_alloc")?;

    let err = do_alloc
        .call_async(&mut store, (0, 4))
        .await
        .expect_err("PerActor typed with wrong store data must error");
    let combined: String = std::iter::successors(
        Some(&*err as &(dyn std::error::Error + 'static)),
        |e| e.source(),
    )
    .map(|e| e.to_string())
    .collect::<Vec<_>>()
    .join(" | ");
    assert!(
        combined.contains("TvmHost")
            && combined.contains("ctx.consumer_state")
            && combined.contains("install_core_imports"),
        "expected TvmHostSource::PerActor guard diagnostic in error chain, got: {combined}"
    );
    Ok(())
}
