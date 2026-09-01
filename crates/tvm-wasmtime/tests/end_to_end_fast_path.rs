#![allow(deprecated)] // ADR-0029 Phase 6.9.d Session 7 — this test/bench intentionally exercises the deprecated wit-bindgen raw entry points to guard the reference implementation while it coexists with `raw_linker_wasmos`.

//! End-to-end test for the raw fast path. Builds the
//! `examples/guest-fast-path` crate (which uses `tvm-guest-rt` for the
//! guest-side wrapper), instantiates it, and verifies the round-trip.

use std::path::PathBuf;
use std::process::Command;

use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::RegionKind;
use tvm_wasmtime::{add_raw_imports, TvmHost};
use wasmtime::{Engine, Linker, Module, Store};

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

fn build_fast_path_guest() -> anyhow::Result<Vec<u8>> {
    let root = workspace_root();
    let guest_dir = root.join("examples/guest-fast-path");
    let status = Command::new("cargo")
        .current_dir(&guest_dir)
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .status()?;
    if !status.success() {
        anyhow::bail!("guest cargo build failed");
    }
    let wasm = guest_dir.join("target/wasm32-unknown-unknown/release/tvm_guest_fast_path.wasm");
    Ok(std::fs::read(wasm)?)
}

#[test]
fn fast_path_guest_round_trip() -> anyhow::Result<()> {
    if !wasm32_target_installed() {
        eprintln!("skipping: wasm32-unknown-unknown target not installed");
        return Ok(());
    }

    let wasm = build_fast_path_guest()?;

    let engine = Engine::default();
    let mut linker: Linker<TvmHost> = Linker::new(&engine);
    add_raw_imports(&mut linker)?;

    let module = Module::new(&engine, &wasm)?;

    let mut host = TvmHost::new();
    // Pre-create region 0; the guest expects it to exist.
    let region = ManagerHost::create_region(&mut host, RegionKind::HotHeap, 256)?;
    assert_eq!(region, 0);

    let mut store = Store::new(&engine, host);
    let instance = linker.instantiate(&mut store, &module)?;
    let run = instance.get_typed_func::<(), u32>(&mut store, "run_test")?;

    // The guest writes [1, 2, 3, 4] and sums them: 1+2+3+4 = 10.
    let sum = run.call(&mut store, ())?;
    assert_eq!(sum, 10);

    // Host state reflects the guest's allocation.
    let host = store.data();
    let info = host.directory.region_info(region)?;
    assert_eq!(info.used, 4);

    Ok(())
}
