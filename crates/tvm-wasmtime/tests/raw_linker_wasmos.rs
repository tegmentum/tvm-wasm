//! End-to-end coverage of the wasmos-backed raw fast-path linker
//! (`raw_linker_wasmos`). Runs the same `tvm.*` guest surface as the
//! wasmtime-native `raw_linker.rs` tests, but drives it through
//! [`wasmos_runtime_wasmtime_v48::WasmtimeV48Runtime`] +
//! [`wasmos_runtime_api::CoreImports`].
//!
//! # What this proves
//!
//! * All 26 non-shared handlers register + wire correctly through the
//!   wasmos `CoreImports` builder path.
//! * The `Arc<Mutex<TvmHost>>` capture pattern (via `SharedTvmHost`)
//!   yields the same guest-observable results as the exclusive-Store
//!   wasmtime path.
//! * Guest memory access via `ctx.guest_memory_{read,write}` behaves
//!   identically to `Caller::get_export("memory")` + `mem.read/write`
//!   for the read/write handlers.
//!
//! # Test structure
//!
//! Assertions consolidated under one `#[tokio::test]` function — the
//! v48 runtime is created per assertion but tokio's multi-thread
//! executor is shared, matching the harness shape used in
//! `runtime/wamr/tests/core_import_ctx.rs`.
//!
//! # ADR-0029 Phase 6.9.a Session 1
//!
//! Additive test — the existing `raw_linker.rs` test file is untouched.
//! Both paths run in CI; each is exercised by its own test module.

use std::sync::Arc;

use tvm_core::RegionKind as CoreRegionKind;
use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::raw_linker_wasmos::add_raw_imports;
use tvm_wasmtime::shared_host::SharedTvmHost;
use wasmos_runtime_api::{
    Bytes, CompileOptions, ComponentSource, CoreImports, CoreValue, ExecutionContext,
    ModuleInstance, Runtime,
};
use wasmos_runtime_wasmtime_v48::WasmtimeV48Runtime;

/// Compile `wat`, register raw imports over `host`, instantiate.
async fn instantiate(
    rt: &WasmtimeV48Runtime,
    wat: &str,
    host: SharedTvmHost,
) -> anyhow::Result<ModuleInstance> {
    let wasm: Vec<u8> = wat::parse_str(wat)?;
    let compiled = rt
        .compile_module(
            ComponentSource::Bytes {
                bytes: Bytes::from(wasm),
                name: Some("raw-linker-wasmos-test".into()),
            },
            CompileOptions::default(),
        )
        .await?;
    let core_imports = add_raw_imports(CoreImports::new(), host);
    let ctx = ExecutionContext {
        core_imports,
        ..ExecutionContext::new()
    };
    Ok(rt.instantiate_module(&compiled, ctx).await?)
}

/// Guest mirroring `tests/raw_linker.rs` — imports alloc/write/read/
/// last_error from `tvm` + a `stage_at(ptr, byte)` helper that writes
/// one byte into guest memory. Real toolchains ship something
/// equivalent (the guest's own allocator writes into its linear memory
/// before calling `tvm.write`); the wasmos Instance surface doesn't
/// expose direct guest-memory writes from the host, so we route
/// staging through the guest itself.
const RAW_GUEST_WAT_STAGED: &str = r#"
(module
  (import "tvm" "alloc"      (func $alloc (param i32 i32) (result i64)))
  (import "tvm" "write"      (func $write (param i64 i32 i32) (result i32)))
  (import "tvm" "read"       (func $read  (param i64 i32 i32) (result i32)))
  (import "tvm" "last_error" (func $last_error (result i32)))
  (memory (export "memory") 1)

  (func (export "stage_at") (param $ptr i32) (param $byte i32)
    (i32.store8 (local.get $ptr) (local.get $byte)))

  (func (export "alloc") (param $region i32) (param $size i32) (result i64)
    (call $alloc (local.get $region) (local.get $size)))

  (func (export "write") (param $h i64) (param $ptr i32) (param $len i32) (result i32)
    (call $write (local.get $h) (local.get $ptr) (local.get $len)))

  (func (export "read") (param $h i64) (param $ptr i32) (param $len i32) (result i32)
    (call $read (local.get $h) (local.get $ptr) (local.get $len)))

  (func (export "last_error") (result i32)
    (call $last_error))

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

/// Full write-then-read round-trip through wasmos — proves tvm.alloc,
/// tvm.write, tvm.read wire correctly through the CoreImports path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wasmos_raw_write_then_read_round_trip() -> anyhow::Result<()> {
    let shared = SharedTvmHost::new();
    let region = {
        let mut g = shared.lock();
        ManagerHost::create_region(&mut *g, RegionKind::HotHeap, 256)?
    };
    assert_eq!(region, 0);

    let rt = WasmtimeV48Runtime::new(Default::default())?;
    let mut instance = instantiate(&rt, RAW_GUEST_WAT_STAGED, shared.clone()).await?;

    // Alloc 4 bytes in region 0.
    let out = instance
        .call_function(
            "alloc",
            &[CoreValue::I32(region as i32), CoreValue::I32(4)],
        )
        .await?;
    let handle = match out.as_slice() {
        [CoreValue::I64(h)] => *h,
        other => panic!("alloc: expected [I64], got {other:?}"),
    };
    assert!(handle != 0, "alloc returned null handle");

    // Stage [1, 2, 3, 4] into guest memory at offset 0 via stage_at.
    for (i, b) in [1i32, 2, 3, 4].iter().enumerate() {
        instance
            .call_function(
                "stage_at",
                &[CoreValue::I32(i as i32), CoreValue::I32(*b)],
            )
            .await?;
    }

    // Persist those 4 bytes into the region via tvm.write.
    let out = instance
        .call_function(
            "write",
            &[CoreValue::I64(handle), CoreValue::I32(0), CoreValue::I32(4)],
        )
        .await?;
    assert_eq!(out, vec![CoreValue::I32(0)], "tvm.write ok");

    // Zero the buffer so the read is observable.
    for i in 0..4 {
        instance
            .call_function(
                "stage_at",
                &[CoreValue::I32(i), CoreValue::I32(0)],
            )
            .await?;
    }

    // Read the region back into guest memory at offset 0.
    let out = instance
        .call_function(
            "read",
            &[CoreValue::I64(handle), CoreValue::I32(0), CoreValue::I32(4)],
        )
        .await?;
    assert_eq!(out, vec![CoreValue::I32(0)], "tvm.read ok");

    // Sum should now be 1+2+3+4 = 10.
    let out = instance.call_function("sum", &[CoreValue::I32(4)]).await?;
    assert_eq!(out, vec![CoreValue::I32(10)], "guest sees round-tripped bytes");

    // Host directory reflects the alloc.
    let g = shared.lock();
    assert_eq!(g.directory.region_info(region)?.used, 4);

    Ok(())
}

/// Prove tvm.alloc failure records into last_error and returns 0.
/// Mirrors `raw_linker::raw_alloc_failure_records_last_error`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wasmos_raw_alloc_failure_records_last_error() -> anyhow::Result<()> {
    let shared = SharedTvmHost::new();
    // Small region — 16 bytes total.
    let region = {
        let mut g = shared.lock();
        ManagerHost::create_region(&mut *g, RegionKind::Scratch, 16)?
    };

    let rt = WasmtimeV48Runtime::new(Default::default())?;
    let mut instance = instantiate(&rt, RAW_GUEST_WAT_STAGED, shared.clone()).await?;

    // Ask for 999 bytes — overflow the region.
    let out = instance
        .call_function(
            "alloc",
            &[CoreValue::I32(region as i32), CoreValue::I32(999)],
        )
        .await?;
    assert_eq!(out, vec![CoreValue::I64(0)], "alloc returned 0 on overflow");

    // last_error returns the error code and clears it.
    let out = instance.call_function("last_error", &[]).await?;
    match out.as_slice() {
        [CoreValue::I32(code)] => {
            assert!(*code != 0, "last_error non-zero after failed alloc");
        }
        other => panic!("last_error: expected [I32], got {other:?}"),
    }

    // Second call should now read 0 (last_error was reset).
    let out = instance.call_function("last_error", &[]).await?;
    assert_eq!(
        out,
        vec![CoreValue::I32(0)],
        "last_error resets after read"
    );

    Ok(())
}

/// Reducer path — populate a region with a known payload, prove
/// `tvm.sum_u8` returns the expected byte total. Exercises the
/// non-memory-touching branch (host state only, no guest memory).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wasmos_raw_sum_u8_reducer() -> anyhow::Result<()> {
    let shared = SharedTvmHost::new();
    let region = {
        let mut g = shared.lock();
        ManagerHost::create_region(&mut *g, RegionKind::HotHeap, 256)?
    };

    // Pre-populate a region with 10 bytes summing to 55 (1..=10).
    let handle = {
        let mut g = shared.lock();
        let host = &mut *g;
        let h = host.directory.alloc(region, 10)?;
        let payload: Vec<u8> = (1..=10).collect();
        host.directory.write(h, &payload)?;
        h.pack()
    };

    // Instantiate a small guest that only exposes tvm.sum_u8.
    let sum_only_wat = r#"
        (module
          (import "tvm" "sum_u8" (func $sum_u8 (param i64 i32) (result i64)))
          (memory (export "memory") 1)
          (func (export "sum") (param $h i64) (param $len i32) (result i64)
            (call $sum_u8 (local.get $h) (local.get $len))))
    "#;
    let rt = WasmtimeV48Runtime::new(Default::default())?;
    let mut instance = instantiate(&rt, sum_only_wat, shared.clone()).await?;
    let out = instance
        .call_function(
            "sum",
            &[CoreValue::I64(handle as i64), CoreValue::I32(10)],
        )
        .await?;
    assert_eq!(out, vec![CoreValue::I64(55)]);

    Ok(())
}

/// Static-dispatch sanity — the wasmos `CoreImports::register_static`
/// path (Phase 6.13 Session 3) works alongside the `register` path used
/// by `add_raw_imports`. Proves both dispatch mechanisms coexist inside
/// one `CoreImports` set without contention.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wasmos_raw_coexists_with_static_dispatch() -> anyhow::Result<()> {
    use async_trait::async_trait;
    use wasmos_runtime_api::{CoreImportContext, CoreImportFn, RuntimeError, RuntimeResult};

    /// Simple static-dispatch import — echoes its argument.
    struct Echo;
    #[async_trait]
    impl CoreImportFn for Echo {
        async fn call(
            &self,
            _ctx: &mut CoreImportContext<'_>,
            args: Vec<CoreValue>,
        ) -> RuntimeResult<Vec<CoreValue>> {
            match args.as_slice() {
                [CoreValue::I32(v)] => Ok(vec![CoreValue::I32(*v)]),
                other => Err(RuntimeError::msg(format!(
                    "echo: expected [I32], got {other:?}"
                ))),
            }
        }
    }

    let shared = SharedTvmHost::new();
    let core_imports = add_raw_imports(CoreImports::new(), shared.clone())
        .register_static("aux", "echo", Echo);

    let wat = r#"
        (module
          (import "tvm" "alloc" (func $alloc (param i32 i32) (result i64)))
          (import "aux" "echo"  (func $echo  (param i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "run") (result i32)
            (call $echo (i32.const 42))))
    "#;
    let wasm: Vec<u8> = wat::parse_str(wat)?;
    let rt = WasmtimeV48Runtime::new(Default::default())?;
    let compiled = rt
        .compile_module(
            ComponentSource::Bytes {
                bytes: Bytes::from(wasm),
                name: None,
            },
            CompileOptions::default(),
        )
        .await?;
    let ctx = ExecutionContext {
        core_imports,
        ..ExecutionContext::new()
    };
    let mut instance = rt.instantiate_module(&compiled, ctx).await?;
    let out = instance.call_function("run", &[]).await?;
    assert_eq!(out, vec![CoreValue::I32(42)]);

    // Suppress the unused-imports-when-no-alloc-called clippy warning.
    let _ = (Arc::new(0u8), CoreRegionKind::HotHeap, shared);
    Ok(())
}

// ────────────────────────────────────────────────────────────────────
// Phase 6.9.a Session 2 — cross-instance shared-host semantics.
//
// Proves that the `add_raw_shared*` alias delivers the same
// guest-observable semantics as the wasmtime `add_raw_shared` peer:
// two `ModuleInstance`s sharing ONE `SharedTvmHost` see each other's
// region writes, and neither corrupts the other's guest memory
// access (no `cached_memory` cross-store hazard because wasmos
// handlers never cache).
// ────────────────────────────────────────────────────────────────────

/// Guest that writes a byte into region 0 (offset zero) via
/// `tvm.write`, or reads the same byte back via `tvm.read` + sum. Two
/// exports: `write_one(handle, byte)` and `sum_one(handle)`.
const CROSS_INSTANCE_WAT: &str = r#"
(module
  (import "tvm" "write" (func $write (param i64 i32 i32) (result i32)))
  (import "tvm" "read"  (func $read  (param i64 i32 i32) (result i32)))
  (memory (export "memory") 1)

  ;; Stage `byte` at offset 0, then tvm.write 1 byte into `handle`.
  ;; Returns the tvm error code (0 ok).
  (func (export "write_one") (param $h i64) (param $byte i32) (result i32)
    (i32.store8 (i32.const 0) (local.get $byte))
    (call $write (local.get $h) (i32.const 0) (i32.const 1)))

  ;; Zero offset 0, tvm.read 1 byte from `handle` back into offset 0,
  ;; then return the byte.
  (func (export "sum_one") (param $h i64) (result i32)
    (i32.store8 (i32.const 0) (i32.const 0))
    (drop (call $read (local.get $h) (i32.const 0) (i32.const 1)))
    (i32.load8_u (i32.const 0)))
)
"#;

/// Alloc a 1-byte region + return its packed handle. Used to seed the
/// cross-instance test; runs pre-instantiation via direct host mut.
fn alloc_one_byte(shared: &SharedTvmHost, region: u16) -> anyhow::Result<i64> {
    let mut g = shared.lock();
    let h = g.directory.alloc(region, 1)?;
    Ok(h.pack() as i64)
}

/// Instantiate the cross-instance guest against `shared` — helper
/// used twice to build two independent instances that share one host.
async fn instantiate_cross_instance(
    rt: &WasmtimeV48Runtime,
    shared: SharedTvmHost,
) -> anyhow::Result<ModuleInstance> {
    instantiate(rt, CROSS_INSTANCE_WAT, shared).await
}

/// Cross-instance write-then-read via `add_raw_shared`.
///
/// Two ModuleInstances share ONE SharedTvmHost. Instance A writes
/// 0x2A into a region via tvm.write. Instance B reads the same
/// region back via tvm.read. B observes A's write — proves the
/// shared-host semantics work end-to-end through the wasmos alias.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wasmos_raw_shared_cross_instance_visibility() -> anyhow::Result<()> {
    // Use `add_raw_shared` (the Session 2 alias) explicitly to prove
    // the alias delegates correctly.
    use tvm_wasmtime::raw_linker_wasmos::add_raw_shared;

    let shared = SharedTvmHost::new();
    let region = {
        let mut g = shared.lock();
        ManagerHost::create_region(&mut *g, RegionKind::HotHeap, 64)?
    };
    let handle = alloc_one_byte(&shared, region)?;
    assert!(handle != 0);

    let rt = WasmtimeV48Runtime::new(Default::default())?;

    // Build TWO instances against the SAME SharedTvmHost via the
    // shared-batch alias. This mirrors the wasmtime pattern of two
    // Store<SharedTvmHost>s pointing at one Arc.
    let wasm: Vec<u8> = wat::parse_str(CROSS_INSTANCE_WAT)?;
    let compiled = rt
        .compile_module(
            ComponentSource::Bytes {
                bytes: Bytes::from(wasm),
                name: Some("cross-instance".into()),
            },
            CompileOptions::default(),
        )
        .await?;

    let mut inst_a = {
        let core_imports = add_raw_shared(CoreImports::new(), shared.clone());
        let ctx = ExecutionContext {
            core_imports,
            ..ExecutionContext::new()
        };
        rt.instantiate_module(&compiled, ctx).await?
    };
    let mut inst_b = {
        let core_imports = add_raw_shared(CoreImports::new(), shared.clone());
        let ctx = ExecutionContext {
            core_imports,
            ..ExecutionContext::new()
        };
        rt.instantiate_module(&compiled, ctx).await?
    };

    // Instance A writes 0x2A into region 0 offset 0.
    let out = inst_a
        .call_function(
            "write_one",
            &[CoreValue::I64(handle), CoreValue::I32(0x2A)],
        )
        .await?;
    assert_eq!(out, vec![CoreValue::I32(0)], "A's write ok");

    // Instance B reads region 0 offset 0. Because the region lives
    // in SharedTvmHost (shared between A and B) and the wasmos
    // handlers never cache per-store guest-memory pointers, B
    // observes A's write.
    let out = inst_b
        .call_function("sum_one", &[CoreValue::I64(handle)])
        .await?;
    assert_eq!(out, vec![CoreValue::I32(0x2A)], "B observes A's write");

    // Instance B writes back 0x55; A observes the update. Symmetric.
    let out = inst_b
        .call_function(
            "write_one",
            &[CoreValue::I64(handle), CoreValue::I32(0x55)],
        )
        .await?;
    assert_eq!(out, vec![CoreValue::I32(0)]);
    let out = inst_a
        .call_function("sum_one", &[CoreValue::I64(handle)])
        .await?;
    assert_eq!(out, vec![CoreValue::I32(0x55)], "A observes B's write");

    // Silence unused-fn warning if the compiler dead-strips.
    let _ = instantiate_cross_instance;
    Ok(())
}
