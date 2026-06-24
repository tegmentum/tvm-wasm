//! End-to-end test for the rust-cdylib substrate pattern.
//!
//! Builds the `examples/rust-cdylib-consumer` cdylib, links it
//! against a 4-pool multi-memory shell using `tvm-guest-mm-link`, and
//! exercises a pcache-shape workload (byte-level R/W against multiple
//! pools, page materialization through `tvm_copy_to_default`, sum over
//! a column of page-header u32s) against wasmtime.
//!
//! Requires the `wasm32-unknown-unknown` target to be installed:
//!   rustup target add wasm32-unknown-unknown
//!
//! Skipped (with a clear stderr message) when the target is missing
//! so a default `cargo test` from a freshly-cloned host doesn't fail
//! the suite — environments that intentionally test this path have
//! the target installed.

use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;

/// Locate the example crate manifest from this test's CARGO_MANIFEST_DIR.
fn example_manifest() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crate_dir = .../crates/tvm-guest-mm
    crate_dir
        .join("..")
        .join("..")
        .join("examples")
        .join("rust-cdylib-consumer")
        .join("Cargo.toml")
}

/// True if the wasm32-unknown-unknown target is installed.
fn has_wasm_target() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout).contains("wasm32-unknown-unknown")
        })
        .unwrap_or(false)
}

fn build_consumer_cdylib() -> anyhow::Result<Vec<u8>> {
    let manifest = example_manifest();
    let manifest_str = manifest.to_string_lossy().to_string();
    let target_dir = manifest
        .parent()
        .unwrap()
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("tvm_rust_cdylib_consumer.wasm");

    let status = Command::new(env!("CARGO"))
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "--manifest-path",
            &manifest_str,
        ])
        .status()
        .context("invoking cargo to build the rust-cdylib-consumer example")?;
    if !status.success() {
        anyhow::bail!("cargo build of the consumer example failed");
    }
    std::fs::read(&target_dir).with_context(|| format!("reading {}", target_dir.display()))
}

#[test]
fn rust_cdylib_consumer_round_trips_pages() -> anyhow::Result<()> {
    if !has_wasm_target() {
        eprintln!(
            "SKIP: rust_cdylib_consumer_round_trips_pages — wasm32-unknown-unknown target not installed.\n\
             Install with: rustup target add wasm32-unknown-unknown"
        );
        return Ok(());
    }

    let user_bytes = build_consumer_cdylib()?;

    let params = tvm_guest_mm_link::ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: 1,
        max_pages_per_pool: 256, // 16 MiB per pool — plenty for the 4-page test
        user_body: String::new(),
    };
    let linked_bytes = tvm_guest_mm_link::link_with_params(&params, &user_bytes)
        .context("linking consumer + shell")?;

    // Confirm validity + no leftover imports.
    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&linked_bytes)
    .context("validating linked module")?;

    let import_count = wasmparser::Parser::new(0)
        .parse_all(&linked_bytes)
        .filter_map(|p| p.ok())
        .filter(|p| matches!(p, wasmparser::Payload::ImportSection(_)))
        .count();
    assert_eq!(import_count, 0, "linked module must have no import section");

    // Instantiate on wasmtime.
    let mut config = wasmtime::Config::new();
    config.wasm_multi_memory(true);
    let engine = wasmtime::Engine::new(&config)?;
    let module = wasmtime::Module::new(&engine, &linked_bytes)?;
    let linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    // ----- 1. Byte-level R/W round-trip on the hot tier -----
    let write_hot = instance.get_typed_func::<(u32, u32, u32), ()>(&mut store, "write_hot_byte")?;
    let read_hot = instance.get_typed_func::<(u32, u32), u32>(&mut store, "read_hot_byte")?;
    // write_hot_byte takes (page_id, byte_off, value). value is u8 but
    // the ABI lifts it to u32.
    write_hot.call(&mut store, (3, 100, 0x5a))?;
    let v = read_hot.call(&mut store, (3, 100))?;
    assert_eq!(v, 0x5a, "byte round-trip on hot tier");

    // ----- 2. Pool isolation: writing the hot tier must not affect warm -----
    let read_warm = instance.get_typed_func::<(u32, u32), u32>(&mut store, "read_warm_byte")?;
    let v_warm = read_warm.call(&mut store, (3, 100))?;
    assert_eq!(v_warm, 0, "warm tier must be unaffected by hot-tier writes");

    // ----- 3. u32-shaped header field round-trip -----
    let write_hot_u32 =
        instance.get_typed_func::<(u32, u32, u32), ()>(&mut store, "write_hot_u32")?;
    let read_hot_u32 =
        instance.get_typed_func::<(u32, u32), u32>(&mut store, "read_hot_u32")?;
    write_hot_u32.call(&mut store, (1, 0, 0xdead_beef))?;
    write_hot_u32.call(&mut store, (2, 0, 0xc0fe_babe))?;
    assert_eq!(read_hot_u32.call(&mut store, (1, 0))?, 0xdead_beef);
    assert_eq!(read_hot_u32.call(&mut store, (2, 0))?, 0xc0fe_babe);

    // ----- 4. Page materialization through tvm_copy_to_default -----
    //
    // Seed page 0 in the hot tier with a recognizable pattern via the
    // per-pool typed-store helper exposed by the shell (a clearer test
    // than calling write_hot_byte 4 KiB times). Then call the
    // workload's materialize_hot_page to copy the page into default
    // memory at offset 0x10000. Read default memory back to confirm.
    //
    // The shell exports tvm_store_u8_p1 — direct typed store with mem
    // immediate baked in. (1 ⇒ pool index for HOT.) We use it to write
    // 4 KiB.
    let store_u8_p1 = instance.get_typed_func::<(u32, u32), ()>(&mut store, "tvm_store_u8_p1")?;
    for i in 0..tvm_rust_cdylib_consumer_consts::PAGE_SIZE {
        store_u8_p1.call(&mut store, (i, (i & 0xff) as u32))?;
    }
    let materialize =
        instance.get_typed_func::<(u32, u32), u32>(&mut store, "materialize_hot_page")?;
    let dst_off: u32 = 0x10000; // safe slot above rustc's data section
    let copied = materialize.call(&mut store, (0, dst_off))?;
    assert_eq!(copied, tvm_rust_cdylib_consumer_consts::PAGE_SIZE);

    // Read the materialized buffer from default memory (mem0).
    let mem0 = instance
        .get_memory(&mut store, "mem0")
        .expect("shell must export mem0 (default memory)");
    let mut buf = vec![0u8; tvm_rust_cdylib_consumer_consts::PAGE_SIZE as usize];
    mem0.read(&mut store, dst_off as usize, &mut buf)?;
    for (i, b) in buf.iter().enumerate() {
        assert_eq!(*b as u32, (i as u32) & 0xff, "page mat byte {} wrong", i);
    }

    // ----- 5. install_hot_page (symmetric: default → hot) -----
    // Build a page in default memory, install it into hot at page id 2,
    // then read back via read_hot_byte to confirm.
    let mut new_page = vec![0u8; tvm_rust_cdylib_consumer_consts::PAGE_SIZE as usize];
    for (i, b) in new_page.iter_mut().enumerate() {
        *b = ((i ^ 0xff) & 0xff) as u8;
    }
    mem0.write(&mut store, dst_off as usize, &new_page)?;
    let install = instance.get_typed_func::<(u32, u32), u32>(&mut store, "install_hot_page")?;
    install.call(&mut store, (2, dst_off))?;
    // Spot-check a few bytes via read_hot_byte.
    for &i in &[0u32, 1, 17, 4095] {
        let v = read_hot.call(&mut store, (2, i))?;
        assert_eq!(v, ((i ^ 0xff) & 0xff) as u32, "install byte {} wrong", i);
    }

    // ----- 6. sum_hot_page_headers: per-page u32 sum -----
    // write_hot_u32 set page 1 → 0xdead_beef, page 2 → 0xc0fe_babe.
    // Pages 0 and 3 are 0 unless install_hot_page just wrote page 2
    // — which we did. Page 2 was overwritten with the (i ^ 0xff)
    // pattern; its first u32 (i=0..4) is therefore [0xff, 0xfe, 0xfd,
    // 0xfc] = 0xfcfdfeff (LE). Page 0 was overwritten with the (i &
    // 0xff) pattern when we wrote it for the materialize test above:
    // first u32 is [0,1,2,3] LE = 0x03020100.
    let sum =
        instance.get_typed_func::<(u32, u32), u32>(&mut store, "sum_hot_page_headers")?;
    let result = sum.call(&mut store, (0, 4))?;
    let expected = 0x0302_0100u32
        .wrapping_add(0xdead_beefu32)
        .wrapping_add(0xfcfd_feffu32)
        .wrapping_add(0u32);
    assert_eq!(result, expected, "sum_hot_page_headers mismatch");

    Ok(())
}

// A small inline module mirroring the consumer's public constants so
// the test code doesn't have to depend on the cdylib crate as a Rust
// dep (which would force it into the host-target build graph).
mod tvm_rust_cdylib_consumer_consts {
    pub const PAGE_SIZE: u32 = 4096;
}
