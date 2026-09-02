//! End-to-end test: build the `tvm-guest-demo` crate to wasm32-unknown-unknown,
//! wrap the core module as a component, instantiate against the TvmHost
//! linker, and call its exported `run-test` function.
//!
//! Skipped automatically if the wasm32-unknown-unknown target isn't installed
//! or if the cargo build fails for any reason.
//!
//! ADR-0029 Phase 6.9 D2 Session 15b — migrated off the retired
//! `add_to_linker` to the wasmos install path
//! (`install_tvm_imports_per_actor::<TvmHost>` + v48 async_bridge).

use std::path::PathBuf;
use std::process::Command;

use tvm_wasmtime::wasmos_bindings::install_tvm_imports_per_actor;
use tvm_wasmtime::TvmHost;
use wasmos_runtime_api::HostImports;
use wasmos_runtime_wasmtime_v48::async_bridge;
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};

// Demo-world bindings (with the run-test export); kept test-local because the
// production crate only ships the import-only `tvm-guest` world.
//
// ADR-0029 Phase 6.9 D2 Session 15b — async: true, matches the
// async engine + wasmos async_bridge the test now uses.
mod demo_bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "tvm-guest-demo",
        // ADR-0029 Phase 6.9 D2 Session 15b — async engine + wasmos
        // async_bridge require async guest exports too.
        exports: { default: async },
        with: {
            "tvm:memory/types": tvm_wasmtime::bindings::tvm::memory::types,
            "tvm:memory/manager": tvm_wasmtime::bindings::tvm::memory::manager,
            "tvm:memory/bytes": tvm_wasmtime::bindings::tvm::memory::bytes,
        },
    });
}
use demo_bindings::TvmGuestDemo;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn wasm32_target_installed() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).contains("wasm32-unknown-unknown"))
        .unwrap_or(false)
}

fn build_guest_core_wasm() -> anyhow::Result<Vec<u8>> {
    let root = workspace_root();
    let guest_dir = root.join("examples/guest-demo");

    let status = Command::new("cargo")
        .current_dir(&guest_dir)
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .status()?;
    if !status.success() {
        anyhow::bail!("guest cargo build failed");
    }

    let wasm = guest_dir.join("target/wasm32-unknown-unknown/release/tvm_guest_demo.wasm");
    Ok(std::fs::read(wasm)?)
}

fn encode_component(core_wasm: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut encoder = wit_component::ComponentEncoder::default()
        .module(core_wasm)?
        .validate(true);
    let bytes = encoder.encode()?;
    Ok(bytes)
}

#[test]
fn guest_calls_host_round_trip() -> anyhow::Result<()> {
    if !wasm32_target_installed() {
        eprintln!("skipping: wasm32-unknown-unknown target not installed");
        return Ok(());
    }

    let core = build_guest_core_wasm()?;
    let component_bytes = encode_component(&core)?;

    let mut config = Config::new();
    config.wasm_component_model(true);
    // Session 15b: the wasmos v48 async_bridge requires async_support.
    #[allow(deprecated)]
    config.async_support(true);
    let engine = Engine::new(&config)?;
    let component = Component::from_binary(&engine, &component_bytes)?;

    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    let imports = install_tvm_imports_per_actor::<TvmHost>(HostImports::new());
    async_bridge::install_host_imports(&engine, &mut linker, &component, &imports)
        .map_err(|e| anyhow::anyhow!("wasmos install: {e}"))?;

    let mut store = Store::new(&engine, TvmHost::new());
    // Session 15b: async engine requires instantiate_async + call_async.
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let (instance, sum) = rt.block_on(async {
        let instance =
            TvmGuestDemo::instantiate_async(&mut store, &component, &linker).await?;
        let sum = instance.call_run_test(&mut store).await?;
        anyhow::Ok((instance, sum))
    })?;
    let _ = instance;

    // 1+2+3+4 = 10
    assert_eq!(sum, 10);

    // Host state must reflect the guest's actions: one region, one allocation.
    let host = store.data();
    assert_eq!(host.directory.len(), 1);
    let metrics = host
        .directory
        .iter()
        .next()
        .map(|r| host.directory.metrics(r.id).unwrap().snapshot())
        .unwrap();
    assert_eq!(metrics.allocations, 1);
    assert_eq!(metrics.bytes_allocated, 4);

    Ok(())
}
