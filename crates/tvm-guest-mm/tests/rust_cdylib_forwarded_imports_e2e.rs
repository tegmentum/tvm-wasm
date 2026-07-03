//! End-to-end test for the rust-cdylib substrate with **forwarded
//! imports**.
//!
//! Builds the `examples/rust-cdylib-consumer-with-imports` cdylib,
//! links it against a 2-pool shell, and verifies:
//!
//! 1. The merged module retains a non-empty import section with the
//!    forwarded `host.log` / `host.now_nanos` entries (no `tvm_mm.*`
//!    survives — those are rewired internally).
//! 2. Instantiating the module against a `wasmtime::Linker` that
//!    defines `host.log` and `host.now_nanos` succeeds.
//! 3. Calling the workload's exports invokes the forwarded host
//!    closures with the expected arguments — proving that the
//!    renumbering preserved every call site.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::Context;

fn example_manifest() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .join("..")
        .join("..")
        .join("examples")
        .join("rust-cdylib-consumer-with-imports")
        .join("Cargo.toml")
}

fn has_wasm_target() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("wasm32-unknown-unknown"))
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
        .join("tvm_rust_cdylib_consumer_with_imports.wasm");

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
        .context("invoking cargo to build the consumer-with-imports example")?;
    if !status.success() {
        anyhow::bail!("cargo build of the consumer-with-imports example failed");
    }
    std::fs::read(&target_dir).with_context(|| format!("reading {}", target_dir.display()))
}

#[derive(Default, Debug, Clone)]
struct HostState {
    log_calls: Vec<(u32, u32, Vec<u8>)>,
    now_calls: u32,
    clock_returns: i64,
}

#[test]
fn forwarded_imports_round_trip() -> anyhow::Result<()> {
    if !has_wasm_target() {
        eprintln!(
            "SKIP: forwarded_imports_round_trip — wasm32-unknown-unknown target not installed.\n\
             Install with: rustup target add wasm32-unknown-unknown"
        );
        return Ok(());
    }

    let user_bytes = build_consumer_cdylib()?;

    // Sanity: the pre-link cdylib must contain both flavors of import.
    let pre_link_imports = collect_imports(&user_bytes)?;
    assert!(
        pre_link_imports.iter().any(|(m, _)| m == "tvm_mm"),
        "pre-link cdylib should import from tvm_mm; got {:?}",
        pre_link_imports
    );
    assert!(
        pre_link_imports.iter().any(|(m, _)| m == "host"),
        "pre-link cdylib should import host.log / host.now_nanos; got {:?}",
        pre_link_imports
    );

    // Link against a 2-pool shell.
    let params = tvm_guest_mm_link::ModuleParams {
        n_pools: 2,
        initial_pages_per_pool: 1,
        max_pages_per_pool: 256,
        user_body: String::new(),
    };
    let linked_bytes = tvm_guest_mm_link::link_with_params(&params, &user_bytes)
        .context("linking consumer-with-imports + shell")?;

    // The merged module must validate and must have **no** tvm_mm
    // imports, while still carrying the forwarded host imports.
    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&linked_bytes)
    .context("validating linked module")?;

    let post_link_imports = collect_imports(&linked_bytes)?;
    assert!(
        post_link_imports.iter().all(|(m, _)| m != "tvm_mm"),
        "linked module must not import tvm_mm; got {:?}",
        post_link_imports
    );
    let host_imports: Vec<_> = post_link_imports
        .iter()
        .filter(|(m, _)| m == "host")
        .map(|(_, n)| n.as_str())
        .collect();
    assert!(
        host_imports.contains(&"log") && host_imports.contains(&"now_nanos"),
        "linked module must carry host.log + host.now_nanos through the link step; got {:?}",
        post_link_imports
    );

    // Stand up wasmtime with host-supplied imports.
    let mut config = wasmtime::Config::new();
    config.wasm_multi_memory(true);
    let engine = wasmtime::Engine::new(&config)?;
    let module = wasmtime::Module::new(&engine, &linked_bytes)?;

    let host_state = Arc::new(Mutex::new(HostState {
        clock_returns: 0x0123_4567_89ab_cdef_i64,
        ..HostState::default()
    }));

    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    let host_state_log = Arc::clone(&host_state);
    linker.func_wrap(
        "host",
        "log",
        move |mut caller: wasmtime::Caller<'_, ()>, ptr: u32, len: u32| {
            let mem = caller
                .get_export("mem0")
                .and_then(|e| e.into_memory())
                .expect("mem0 export missing");
            let mut buf = vec![0u8; len as usize];
            mem.read(&mut caller, ptr as usize, &mut buf)
                .expect("reading mem0 in host.log");
            host_state_log
                .lock()
                .unwrap()
                .log_calls
                .push((ptr, len, buf));
        },
    )?;
    let host_state_now = Arc::clone(&host_state);
    linker.func_wrap("host", "now_nanos", move || -> i64 {
        let mut s = host_state_now.lock().unwrap();
        s.now_calls += 1;
        s.clock_returns
    })?;

    let mut store = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    // ----- put_with_log: writes pool 1, then logs "put" -----
    let put = instance.get_typed_func::<(u32, u32), ()>(&mut store, "put_with_log")?;
    let get = instance.get_typed_func::<u32, u32>(&mut store, "get")?;
    put.call(&mut store, (16, 0xdead_beef))?;
    assert_eq!(get.call(&mut store, 16)?, 0xdead_beef);

    {
        let s = host_state.lock().unwrap();
        assert_eq!(s.log_calls.len(), 1, "host.log must have been called once");
        let (_, _, msg) = &s.log_calls[0];
        assert_eq!(msg.as_slice(), b"put", "log message must match `put`");
        assert_eq!(s.now_calls, 0, "now_nanos must not have been called yet");
    }

    // ----- stamped_put: reads now_nanos, splits into hi/lo u32 into
    // pool 1, then logs "stamped" -----
    let stamped = instance.get_typed_func::<u32, i64>(&mut store, "stamped_put")?;
    let ts = stamped.call(&mut store, 64)?;
    assert_eq!(ts, 0x0123_4567_89ab_cdef_i64);

    // Read back the two u32s the workload wrote into pool 1 via
    // tvm_store_u32, using the shell's typed-store. mem1 is the second
    // memory in the merged module.
    let mem1 = instance
        .get_memory(&mut store, "mem1")
        .expect("mem1 must be exported");
    let mut buf = [0u8; 8];
    mem1.read(&mut store, 64, &mut buf)?;
    let hi = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let lo = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    assert_eq!(hi, (0x0123_4567_89ab_cdef_i64 >> 32) as u32);
    assert_eq!(lo, 0x0123_4567_89ab_cdef_i64 as u32);

    {
        let s = host_state.lock().unwrap();
        assert_eq!(
            s.log_calls.len(),
            2,
            "host.log must have been called for stamped_put too"
        );
        assert_eq!(
            s.log_calls[1].2.as_slice(),
            b"stamped",
            "second log call must be `stamped`"
        );
        assert_eq!(
            s.now_calls, 1,
            "now_nanos must have been called exactly once"
        );
    }

    Ok(())
}

fn collect_imports(bytes: &[u8]) -> anyhow::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::ImportSection(reader) = payload? {
            for group in reader {
                let group = group?;
                match group {
                    wasmparser::Imports::Single(_, imp) => {
                        out.push((imp.module.to_string(), imp.name.to_string()));
                    }
                    wasmparser::Imports::Compact1 { module, items } => {
                        for item in items {
                            let item = item?;
                            out.push((module.to_string(), item.name.to_string()));
                        }
                    }
                    wasmparser::Imports::Compact2 { module, names, .. } => {
                        for name in names {
                            let name = name?;
                            out.push((module.to_string(), name.to_string()));
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}
