//! End-to-end test for `call_indirect` through a renumbered element
//! segment.
//!
//! This is the proof that the linker's element-segment renumbering
//! fix has put the wit-bindgen-shaped failure mode to rest. Pre-fix
//! symptom (downstream sqlite-lib): the linked + componentised module
//! validated structurally but trapped at instantiation with `wasm
//! trap: undefined element: out of bounds table access`, because the
//! `call_indirect` operand pointed at an out-of-range function index
//! in the merged module's func-index space.
//!
//! Shape:
//! - User cdylib has a static `[extern "C" fn(u32) -> u32; 4]` table
//!   plus a `dispatch(idx, arg) -> u32` exported function that
//!   `call_indirect`s into it.
//! - Each target function ends in a `host.bump(N)` call so the test
//!   can verify-from-outside which entry fired.
//! - One target function also exercises the `tvm_mm` data plane
//!   (round-trips a u32 through pool 1) — same workload, two
//!   substrate features.
//! - The host import forces forwarded-import shifting, so the user's
//!   element-segment func indices DO have to be renumbered.

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
        .join("rust-cdylib-consumer-indirect")
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
        .join("tvm_rust_cdylib_consumer_indirect.wasm");

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
        .context("invoking cargo to build the consumer-indirect example")?;
    if !status.success() {
        anyhow::bail!("cargo build of the consumer-indirect example failed");
    }
    std::fs::read(&target_dir).with_context(|| format!("reading {}", target_dir.display()))
}

#[test]
fn dispatch_routes_through_renumbered_element_segment() -> anyhow::Result<()> {
    if !has_wasm_target() {
        eprintln!(
            "SKIP: dispatch_routes_through_renumbered_element_segment — wasm32-unknown-unknown target not installed."
        );
        return Ok(());
    }

    let user_bytes = build_consumer_cdylib()?;

    // Confirm pre-link sanity: the cdylib declares an element segment.
    // (If rustc ever changes its mind and emits something else, the
    // test loses its bite — fail fast in that case.)
    let mut has_elem = false;
    for payload in wasmparser::Parser::new(0).parse_all(&user_bytes) {
        if matches!(payload?, wasmparser::Payload::ElementSection(_)) {
            has_elem = true;
            break;
        }
    }
    assert!(
        has_elem,
        "pre-link consumer-indirect cdylib must contain an element section; \
         rustc may have changed how it lowers static function tables"
    );

    // Link against a 2-pool shell with the conventional `--alias-memory0=memory`
    // alias so the resulting artifact also satisfies `wasm-tools component new`.
    let params = tvm_guest_mm_link::ModuleParams {
        n_pools: 2,
        initial_pages_per_pool: 1,
        max_pages_per_pool: 256,
        user_body: String::new(),
    };
    let options = tvm_guest_mm_link::LinkOptions {
        aliases: vec![("memory".to_string(), 0)],
    };
    let linked_bytes =
        tvm_guest_mm_link::link_with_params_and_options(&params, &user_bytes, &options)
            .context("linking consumer-indirect + shell")?;

    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&linked_bytes)
    .context("validating linked module")?;

    // Instantiate on wasmtime. host.bump records which dispatch entry
    // fired so the test can read it out.
    let mut config = wasmtime::Config::new();
    config.wasm_multi_memory(true);
    let engine = wasmtime::Engine::new(&config)?;
    let module = wasmtime::Module::new(&engine, &linked_bytes)?;

    let bumps: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    let bumps_clone = Arc::clone(&bumps);
    linker.func_wrap("host", "bump", move |by: u32| {
        bumps_clone.lock().unwrap().push(by);
    })?;

    let mut store = wasmtime::Store::new(&engine, ());
    // If this trips with `undefined element: out of bounds table
    // access`, the element-segment renumbering regressed.
    let instance = linker.instantiate(&mut store, &module)?;

    let dispatch = instance.get_typed_func::<(u32, u32), u32>(&mut store, "dispatch")?;

    // FUNCS table layout (per the example):
    //   FUNCS[0] = add_one          → bump(1), returns x+1
    //   FUNCS[1] = double           → bump(2), returns x*2
    //   FUNCS[2] = store_then_negate → bump(3), returns !(read_back of x)
    //   FUNCS[3] = always_42        → bump(4), returns 42

    // FUNCS[0] — add_one
    assert_eq!(
        dispatch.call(&mut store, (0, 10))?,
        11,
        "FUNCS[0] (add_one) must return x+1"
    );
    // FUNCS[1] — double
    assert_eq!(
        dispatch.call(&mut store, (1, 21))?,
        42,
        "FUNCS[1] (double) must return x*2"
    );
    // FUNCS[2] — store_then_negate
    let r = dispatch.call(&mut store, (2, 0xdead_beef))?;
    assert_eq!(
        r, !0xdead_beefu32,
        "FUNCS[2] (store_then_negate) must round-trip through pool 1 and return the bitwise negation"
    );
    // FUNCS[3] — always_42
    assert_eq!(
        dispatch.call(&mut store, (3, 0))?,
        42,
        "FUNCS[3] (always_42) must ignore arg and return 42"
    );

    // Each function bumped its tag once, in dispatch order.
    let final_bumps = bumps.lock().unwrap().clone();
    assert_eq!(
        final_bumps,
        vec![1, 2, 3, 4],
        "host.bump tags must match the dispatch order — proves each indirect call landed on the right target"
    );

    Ok(())
}
