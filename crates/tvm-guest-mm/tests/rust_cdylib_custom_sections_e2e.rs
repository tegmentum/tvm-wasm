//! End-to-end integration test for custom-section pass-through.
//!
//! Builds the `examples/rust-cdylib-consumer-with-imports` cdylib,
//! injects a synthetic `component-type:test` custom section into it
//! (mimicking what wit-bindgen emits when generating bindings for a
//! wasi-p2 world), links the resulting bytes against a 2-pool shell
//! using `--alias-memory0=memory`, and verifies:
//!
//! 1. The injected `component-type:test` custom section survives the
//!    link step byte-for-byte.
//! 2. The merged module gains the requested `memory` export alias in
//!    addition to the shell's existing `mem0..memN` exports.
//!
//! These two properties are exactly what downstream consumers need
//! before invoking `wasm-tools component new` on the linked artifact.

use std::path::PathBuf;
use std::process::Command;

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

/// Append a custom section to existing wasm bytes. A custom section is
/// section id 0 + LEB128-length + (LEB128-name-len + name + data). The
/// position is "at the end of the module" which is always valid for
/// custom sections.
fn append_custom_section(mut wasm: Vec<u8>, name: &str, data: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    leb128_u32(name.len() as u32, &mut body);
    body.extend_from_slice(name.as_bytes());
    body.extend_from_slice(data);

    wasm.push(0); // section id = 0 (custom)
    leb128_u32(body.len() as u32, &mut wasm);
    wasm.extend_from_slice(&body);
    wasm
}

fn leb128_u32(mut value: u32, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

#[test]
fn custom_sections_survive_link_with_alias() -> anyhow::Result<()> {
    if !has_wasm_target() {
        eprintln!(
            "SKIP: custom_sections_survive_link_with_alias — wasm32-unknown-unknown target not installed."
        );
        return Ok(());
    }

    // Build the user cdylib, then inject a synthetic component-type
    // section as if wit-bindgen had generated bindings for a wasi-p2
    // world. The exact bytes don't matter — we just need a recognizable
    // payload to assert byte-for-byte preservation.
    let user_bytes = build_consumer_cdylib()?;
    let test_payload: &[u8] =
        b"\x07\x00\x00\x00 fake component-type binary; first byte is the version tag";
    let user_with_custom = append_custom_section(user_bytes, "component-type:test", test_payload);

    // Link with --alias-memory0=memory equivalent.
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
        tvm_guest_mm_link::link_with_params_and_options(&params, &user_with_custom, &options)
            .context("linking consumer-with-imports + shell with aliases + customs")?;

    // Validate the merged module still parses and validates.
    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&linked_bytes)
    .context("validating linked module")?;

    // ---------- 1. Custom section survived byte-for-byte ----------
    let mut found = None;
    for payload in wasmparser::Parser::new(0).parse_all(&linked_bytes) {
        if let wasmparser::Payload::CustomSection(reader) = payload? {
            if reader.name() == "component-type:test" {
                found = Some(reader.data().to_vec());
                break;
            }
        }
    }
    let preserved = found.context("component-type:test custom section was dropped by linker")?;
    assert_eq!(
        preserved, test_payload,
        "custom section payload must survive byte-for-byte"
    );

    // ---------- 2. Memory alias export is present ----------
    let mut memory_alias_found = false;
    let mut mem0_present = false;
    let mut mem1_present = false;
    for payload in wasmparser::Parser::new(0).parse_all(&linked_bytes) {
        if let wasmparser::Payload::ExportSection(reader) = payload? {
            for e in reader {
                let e = e?;
                if matches!(e.kind, wasmparser::ExternalKind::Memory) {
                    match e.name {
                        "memory" if e.index == 0 => memory_alias_found = true,
                        "mem0" => mem0_present = true,
                        "mem1" => mem1_present = true,
                        _ => {}
                    }
                }
            }
        }
    }
    assert!(
        memory_alias_found,
        "expected an export named `memory` referencing memory 0 (the alias)"
    );
    assert!(
        mem0_present,
        "expected the shell's original `mem0` export to remain"
    );
    assert!(
        mem1_present,
        "expected the shell's original `mem1` export to remain"
    );
    Ok(())
}
