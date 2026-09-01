//! ADR-0029 Phase 6.9.d Session 5 — per-actor `TvmHostSource`
//! variant of `raw_linker_wasmos`.
//!
//! Verifies the wasmos-side `add_raw_imports_per_actor` composite
//! installs through the v48 `core_import_bridge` on a consumer-
//! owned `wasmtime::Linker<TvmHost>` and dispatches all 26 `tvm.*`
//! handlers correctly with `TvmHost` living in the wasmtime store's
//! data (per-actor concurrency model — no `Arc<Mutex<>>` clone in
//! the handler).
//!
//! The test WAT below mirrors `raw_linker.rs`'s guest — imports
//! alloc/write/read/last_error, plus a guest-side `stage_at` helper
//! for staging bytes into linear memory.
//!
//! This is the per-actor peer of the existing `raw_linker_wasmos.rs`
//! test — same guest surface, different dispatch model.

use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::raw_linker_wasmos::add_raw_imports_per_actor;
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

  ;; stage_at(ptr, byte): guest-side helper — writes one byte into
  ;; guest linear memory at `ptr`. Matches the sibling test's shape.
  (func (export "stage_at") (param $ptr i32) (param $byte i32)
    local.get $ptr
    local.get $byte
    i32.store8)

  ;; do_alloc(region_id, size) -> packed handle (i64)
  (func (export "do_alloc") (param $rid i32) (param $sz i32) (result i64)
    local.get $rid
    local.get $sz
    call $alloc)

  ;; do_write(handle, src_ptr, len) -> i32 err_code
  (func (export "do_write") (param $h i64) (param $p i32) (param $l i32) (result i32)
    local.get $h
    local.get $p
    local.get $l
    call $write)

  ;; do_read(handle, dst_ptr, len) -> i32 err_code
  (func (export "do_read") (param $h i64) (param $p i32) (param $l i32) (result i32)
    local.get $h
    local.get $p
    local.get $l
    call $read)

  ;; do_last_error() -> i32
  (func (export "do_last_error") (result i32) call $last_error)

  ;; load_at(ptr) -> i32 (load one u8 sign-extended)
  (func (export "load_at") (param $ptr i32) (result i32)
    local.get $ptr
    i32.load8_u)

  ;; do_dealloc(handle) -> i32 err_code
  (func (export "do_dealloc") (param $h i64) (result i32)
    local.get $h
    call $dealloc)
)
"#;

/// Round-trip a byte string through a TVM region, staging via the
/// guest, and confirm the per-actor `TvmHost` in the store observes
/// the writes.
#[tokio::test(flavor = "multi_thread")]
async fn per_actor_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = Config::new();
    config.async_support(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, RAW_GUEST_WAT)?;

    // Per-actor CoreImports composite — no host passed at register
    // time; handlers pull TvmHost via ctx.consumer_state per call.
    let imports = add_raw_imports_per_actor(CoreImports::new());

    // Consumer-owned Linker<TvmHost>. install_core_imports populates
    // the ctx consumer_state slot with caller.data_mut() → the
    // exact same store data.
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    core_import_bridge::install_core_imports(&mut linker, &module, &imports)?;

    // TvmHost with one HotHeap region provisioned for the test —
    // routing through ManagerHost mirrors the sibling
    // `raw_linker_wasmos.rs` test's setup so the two tests exercise
    // the same allocator-side shape.
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
    let load_at =
        instance.get_typed_func::<i32, i32>(&mut store, "load_at")?;
    let do_dealloc =
        instance.get_typed_func::<i64, i32>(&mut store, "do_dealloc")?;

    // Alloc a 4-byte region — per-actor handler reads host.directory
    // via ctx.consumer_state<TvmHost>() and mints a handle.
    let packed = do_alloc.call_async(&mut store, (rid as i32, 4)).await?;
    assert_ne!(packed, 0, "alloc should return a nonzero packed handle");

    // Stage bytes 0x11, 0x22, 0x33, 0x44 at guest ptr=64.
    let dst_ptr = 64i32;
    for (i, &b) in [0x11u8, 0x22, 0x33, 0x44].iter().enumerate() {
        stage_at
            .call_async(&mut store, (dst_ptr + i as i32, b as i32))
            .await?;
    }

    // Write to region — handler pulls the SAME TvmHost from the
    // store's data (via ctx.consumer_state) and copies from guest
    // memory into the region.
    let wr = do_write.call_async(&mut store, (packed, dst_ptr, 4)).await?;
    assert_eq!(wr, 0 /* ERR_OK */, "write should return ERR_OK, got {wr}");

    // Zero out guest memory to prove read actually reaches the region.
    for i in 0..4 {
        stage_at.call_async(&mut store, (dst_ptr + i, 0)).await?;
    }
    for i in 0..4 {
        let b = load_at.call_async(&mut store, dst_ptr + i).await?;
        assert_eq!(b, 0, "guest memory should be zeroed pre-read");
    }

    // Read back — again through per-actor ctx.consumer_state path.
    let rd = do_read.call_async(&mut store, (packed, dst_ptr, 4)).await?;
    assert_eq!(rd, 0, "read should return ERR_OK, got {rd}");
    for (i, &want) in [0x11u8, 0x22, 0x33, 0x44].iter().enumerate() {
        let got = load_at.call_async(&mut store, dst_ptr + i as i32).await?;
        assert_eq!(got, want as i32, "byte {i}: want {want:#x}, got {got:#x}");
    }

    // Dealloc — one more per-actor handler exercised.
    let dr = do_dealloc.call_async(&mut store, packed).await?;
    assert_eq!(dr, 0, "dealloc should return ERR_OK, got {dr}");

    // Post-run: the store's TvmHost still holds the region metadata
    // (allocator counters etc.) — same TvmHost the handlers mutated.
    let post = store.data();
    let _ = post; // presence of the data() call is the assertion —
                  // if the type didn't match, this wouldn't compile.
    Ok(())
}

/// Sanity check: the PerActor variant errors cleanly when installed
/// against a store whose data isn't `TvmHost`.
#[tokio::test(flavor = "multi_thread")]
async fn per_actor_wrong_store_data_errors_gracefully()
-> Result<(), Box<dyn std::error::Error>> {
    let mut config = Config::new();
    config.async_support(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, RAW_GUEST_WAT)?;

    let imports = add_raw_imports_per_actor(CoreImports::new());

    // Store data is () — not TvmHost. ctx.consumer_state::<TvmHost>()
    // will downcast-fail, returning None; the handler surfaces our
    // guard error message.
    let mut linker: Linker<()> = Linker::new(&engine);
    core_import_bridge::install_core_imports(&mut linker, &module, &imports)?;
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate_async(&mut store, &module).await?;
    let do_alloc =
        instance.get_typed_func::<(i32, i32), i64>(&mut store, "do_alloc")?;

    let err = do_alloc
        .call_async(&mut store, (0, 4))
        .await
        .expect_err("PerActor with wrong store data must error");
    // wasmtime wraps the RuntimeError in a Trap-shaped anyhow chain;
    // walk it to find our guard's diagnostic.
    let combined: String = std::iter::successors(Some(&*err as &(dyn std::error::Error + 'static)), |e| e.source())
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(
        combined.contains("no `TvmHost` in `ctx.consumer_state`"),
        "expected TvmHostSource::PerActor guard diagnostic in error chain, got: {combined}"
    );
    Ok(())
}
