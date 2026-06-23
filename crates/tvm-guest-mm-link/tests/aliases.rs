//! Tests for `LinkOptions::aliases` — additional memory-export
//! aliases injected by the linker.
//!
//! The `--alias-memory0=memory` knob exists primarily so the merged
//! module satisfies `wasm-tools component new`, which expects an
//! export literally named `memory` when picking the default linear
//! memory of a component-model adapter.

use anyhow::Result;
use tvm_guest_mm_link::{link, link_with_options, LinkOptions};

/// Minimal shell with two memories so we can test aliasing an
/// arbitrary pool, not just pool 0.
const TWO_MEM_SHELL: &str = r#"
    (module
      (memory (export "mem0") 1 256)
      (memory (export "mem1") 1 256)
      (func (export "tvm_load_u8") (param $pool i32) (param $off i32) (result i32)
        (i32.const 0))
      (func (export "tvm_store_u8") (param $pool i32) (param $off i32) (param $val i32)))
"#;

const USER_TVM_ONLY: &str = r#"
    (module
      (import "tvm_mm" "tvm_load_u8" (func $load (param i32 i32) (result i32)))
      (func (export "go") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        call $load))
"#;

fn link_wat(shell: &str, user: &str, opts: &LinkOptions) -> Result<Vec<u8>> {
    let shell_bytes = wat::parse_str(shell)?;
    let user_bytes = wat::parse_str(user)?;
    link_with_options(&shell_bytes, &user_bytes, opts)
}

/// Walk the export section and return (name, kind, index) triples.
fn exports_of(bytes: &[u8]) -> Result<Vec<(String, wasmparser::ExternalKind, u32)>> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::ExportSection(reader) = payload? {
            for e in reader {
                let e = e?;
                out.push((e.name.to_string(), e.kind, e.index));
            }
        }
    }
    Ok(out)
}

#[test]
fn alias_memory0_adds_export_named_memory() -> Result<()> {
    let opts = LinkOptions {
        aliases: vec![("memory".to_string(), 0)],
    };
    let merged = link_wat(TWO_MEM_SHELL, USER_TVM_ONLY, &opts)?;
    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&merged)?;

    let exports = exports_of(&merged)?;
    let mem_exports: Vec<&(String, wasmparser::ExternalKind, u32)> = exports
        .iter()
        .filter(|(_, k, _)| matches!(k, wasmparser::ExternalKind::Memory))
        .collect();

    // mem0, mem1, plus the new "memory" alias for memory 0.
    assert!(
        mem_exports
            .iter()
            .any(|(n, _, idx)| n == "memory" && *idx == 0),
        "expected an export named `memory` referencing memory 0; got {:?}",
        mem_exports
    );
    // Originals must still be present.
    assert!(mem_exports.iter().any(|(n, _, idx)| n == "mem0" && *idx == 0));
    assert!(mem_exports.iter().any(|(n, _, idx)| n == "mem1" && *idx == 1));
    Ok(())
}

#[test]
fn alias_arbitrary_pool_index() -> Result<()> {
    let opts = LinkOptions {
        aliases: vec![("hot_memory".to_string(), 1)],
    };
    let merged = link_wat(TWO_MEM_SHELL, USER_TVM_ONLY, &opts)?;
    let exports = exports_of(&merged)?;
    assert!(
        exports
            .iter()
            .any(|(n, k, idx)| n == "hot_memory"
                && matches!(k, wasmparser::ExternalKind::Memory)
                && *idx == 1),
        "expected `hot_memory` to alias memory index 1; got {:?}",
        exports
    );
    Ok(())
}

#[test]
fn multiple_aliases_coexist() -> Result<()> {
    let opts = LinkOptions {
        aliases: vec![
            ("memory".to_string(), 0),
            ("scratch".to_string(), 1),
        ],
    };
    let merged = link_wat(TWO_MEM_SHELL, USER_TVM_ONLY, &opts)?;
    let exports = exports_of(&merged)?;
    assert!(exports.iter().any(|(n, _, idx)| n == "memory" && *idx == 0));
    assert!(exports.iter().any(|(n, _, idx)| n == "scratch" && *idx == 1));
    Ok(())
}

#[test]
fn default_options_emit_no_aliases() -> Result<()> {
    // Round-trip via `link()` (the no-options entry point) and confirm
    // we don't gain spurious "memory" exports — preserves the existing
    // strict-tvm_mm-only e2e expectation.
    let shell_bytes = wat::parse_str(TWO_MEM_SHELL)?;
    let user_bytes = wat::parse_str(USER_TVM_ONLY)?;
    let merged = link(&shell_bytes, &user_bytes)?;
    let exports = exports_of(&merged)?;
    assert!(
        !exports
            .iter()
            .any(|(n, _, _)| n == "memory"),
        "default link() must not synthesize a `memory` export; got {:?}",
        exports
    );
    Ok(())
}

#[test]
fn alias_out_of_range_index_is_rejected() {
    let opts = LinkOptions {
        aliases: vec![("memory".to_string(), 99)],
    };
    let err = link_wat(TWO_MEM_SHELL, USER_TVM_ONLY, &opts).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("99") && msg.to_lowercase().contains("memor"),
        "expected out-of-range alias rejection, got: {msg}"
    );
}

#[test]
fn alias_colliding_name_is_rejected() {
    let opts = LinkOptions {
        // `mem0` is already exported by the shell — duplicate name.
        aliases: vec![("mem0".to_string(), 0)],
    };
    let err = link_wat(TWO_MEM_SHELL, USER_TVM_ONLY, &opts).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.to_lowercase().contains("collide") || msg.contains("mem0"),
        "expected collision rejection, got: {msg}"
    );
}
