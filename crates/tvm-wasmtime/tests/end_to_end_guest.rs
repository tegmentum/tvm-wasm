//! End-to-end test: build the `tvm-guest-demo` crate to wasm32-unknown-unknown,
//! wrap the core module as a component, instantiate against the TvmHost
//! linker, and call its exported `run-test` function.
//!
//! Skipped automatically if the wasm32-unknown-unknown target isn't installed
//! or if the cargo build fails for any reason.

use std::path::PathBuf;
use std::process::Command;

use tvm_wasmtime::{add_to_linker, TvmHost};
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};

// Demo-world bindings (with the run-test export); kept test-local because the
// production crate only ships the import-only `tvm-guest` world.
mod demo_bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "tvm-guest-demo",
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

    let wasm = guest_dir
        .join("target/wasm32-unknown-unknown/release/tvm_guest_demo.wasm");
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
    let engine = Engine::new(&config)?;
    let component = Component::from_binary(&engine, &component_bytes)?;

    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_to_linker(&mut linker)?;

    let mut store = Store::new(&engine, TvmHost::new());
    let instance = TvmGuestDemo::instantiate(&mut store, &component, &linker)?;

    // 1+2+3+4 = 10
    let sum = instance.call_run_test(&mut store)?;
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
