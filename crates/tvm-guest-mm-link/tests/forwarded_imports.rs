//! Focused tests for the linker's forwarded-imports behavior, driven
//! by hand-written WAT inputs (no rustc / wasm32 target required).
//!
//! These exercise the renumbering machinery in isolation: synthetic
//! shells + synthetic user modules let us assert exactly which import,
//! call, and ref.func indices end up where in the merged module.

use anyhow::Result;
use tvm_guest_mm_link::link;

/// A minimal shell that exports one `tvm_mm`-named function so the
/// linker has a wire target for matching user imports. No SIMD, no
/// multi-memory copies — just a single dispatch helper.
const MINI_SHELL: &str = r#"
    (module
      (memory (export "mem0") 1 256)
      (func (export "tvm_load_u8") (param $pool i32) (param $off i32) (result i32)
        (i32.const 0))
      (func (export "tvm_store_u8") (param $pool i32) (param $off i32) (param $val i32)))
"#;

/// A user module with **one** non-`tvm_mm` import (`host.log`), one
/// `tvm_mm` import (`tvm_load_u8`), and one defined function that
/// calls each.
const USER_MIXED: &str = r#"
    (module
      (import "tvm_mm" "tvm_load_u8" (func $load (param i32 i32) (result i32)))
      (import "host"   "log"         (func $log  (param i32 i32)))
      (func (export "go") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        call $log
        local.get 0
        local.get 1
        call $load))
"#;

/// User module that only imports from `tvm_mm` — exercises the strict
/// path that has to keep working after the refactor.
const USER_STRICT: &str = r#"
    (module
      (import "tvm_mm" "tvm_load_u8" (func $load (param i32 i32) (result i32)))
      (func (export "go") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        call $load))
"#;

fn link_wat(shell: &str, user: &str) -> Result<Vec<u8>> {
    let shell_bytes = wat::parse_str(shell)?;
    let user_bytes = wat::parse_str(user)?;
    link(&shell_bytes, &user_bytes)
}

#[derive(Debug, PartialEq, Eq)]
struct ImportRow {
    module: String,
    name: String,
}

fn imports_of(bytes: &[u8]) -> Result<Vec<ImportRow>> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::ImportSection(reader) = payload? {
            for group in reader {
                let group = group?;
                match group {
                    wasmparser::Imports::Single(_, imp) => out.push(ImportRow {
                        module: imp.module.to_string(),
                        name: imp.name.to_string(),
                    }),
                    wasmparser::Imports::Compact1 { module, items } => {
                        for item in items {
                            let item = item?;
                            out.push(ImportRow {
                                module: module.to_string(),
                                name: item.name.to_string(),
                            });
                        }
                    }
                    wasmparser::Imports::Compact2 { module, names, .. } => {
                        for name in names {
                            let name = name?;
                            out.push(ImportRow {
                                module: module.to_string(),
                                name: name.to_string(),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[test]
fn forwarded_import_appears_in_merged_module() -> Result<()> {
    let merged = link_wat(MINI_SHELL, USER_MIXED)?;
    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&merged)?;

    let imps = imports_of(&merged)?;
    assert_eq!(
        imps,
        vec![ImportRow {
            module: "host".into(),
            name: "log".into()
        }],
        "merged module must keep `host.log` and drop `tvm_mm.tvm_load_u8`"
    );
    Ok(())
}

#[test]
fn strict_tvm_mm_only_consumer_keeps_zero_imports() -> Result<()> {
    let merged = link_wat(MINI_SHELL, USER_STRICT)?;
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::default())
        .validate_all(&merged)?;
    let imps = imports_of(&merged)?;
    assert!(
        imps.is_empty(),
        "tvm_mm-only consumer must produce a fully self-contained module; got {:?}",
        imps
    );
    Ok(())
}

#[test]
fn forwarded_call_renumbers_user_call_target() -> Result<()> {
    // `go` calls (a) the forwarded host.log import and (b) the tvm_mm
    // import (which gets rewired to a shell helper). After linking the
    // merged module must instantiate and the `go` body's calls must
    // refer to valid functions of the right types.
    let merged = link_wat(MINI_SHELL, USER_MIXED)?;

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &merged)?;

    let log_calls = std::sync::Arc::new(std::sync::Mutex::new(0u32));
    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    let log_calls_clone = std::sync::Arc::clone(&log_calls);
    linker.func_wrap("host", "log", move |_a: i32, _b: i32| {
        *log_calls_clone.lock().unwrap() += 1;
    })?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    // tvm_load_u8 is wired to a shell stub that returns 0 — the
    // linked function returns whatever tvm_load_u8 returns.
    let go = instance.get_typed_func::<(i32, i32), i32>(&mut store, "go")?;
    let r = go.call(&mut store, (1, 2))?;
    assert_eq!(r, 0, "tvm_mm.tvm_load_u8 stub returns 0");
    assert_eq!(*log_calls.lock().unwrap(), 1, "host.log must be invoked once");
    Ok(())
}

#[test]
fn forwarded_import_position_zero_in_func_index_space() -> Result<()> {
    // After forwarding, the merged module's function-index space starts
    // with the forwarded imports. We expect the `host.log` import to be
    // function 0, then shell-defined funcs, then user-defined funcs.
    // Verify by parsing the import section and the function section.
    let merged = link_wat(MINI_SHELL, USER_MIXED)?;

    let mut import_func_count = 0u32;
    let mut shell_plus_user_def_count = 0u32;
    for payload in wasmparser::Parser::new(0).parse_all(&merged) {
        match payload? {
            wasmparser::Payload::ImportSection(reader) => {
                for group in reader {
                    let group = group?;
                    let inc = match group {
                        wasmparser::Imports::Single(_, imp) => {
                            matches!(imp.ty, wasmparser::TypeRef::Func(_)) as u32
                        }
                        wasmparser::Imports::Compact1 { items, .. } => {
                            let mut n = 0;
                            for item in items {
                                let item = item?;
                                if matches!(item.ty, wasmparser::TypeRef::Func(_)) {
                                    n += 1;
                                }
                            }
                            n
                        }
                        wasmparser::Imports::Compact2 { ty, names, .. } => {
                            let names_count = names.count() as u32;
                            if matches!(ty, wasmparser::TypeRef::Func(_)) {
                                names_count
                            } else {
                                0
                            }
                        }
                    };
                    import_func_count += inc;
                }
            }
            wasmparser::Payload::FunctionSection(reader) => {
                shell_plus_user_def_count = reader.count();
            }
            _ => {}
        }
    }

    assert_eq!(
        import_func_count, 1,
        "exactly one forwarded function import (host.log) should remain"
    );
    // Shell has 2 defined funcs (tvm_load_u8, tvm_store_u8) + user has 1
    // (the `go` workload) = 3.
    assert_eq!(
        shell_plus_user_def_count, 3,
        "function section should hold 2 shell + 1 user defined functions"
    );
    Ok(())
}

#[test]
fn memory_import_is_rejected() {
    let user_with_memory_import = r#"
        (module
          (import "host" "mem" (memory 1))
          (func (export "go") (result i32) (i32.const 0)))
    "#;
    let err = link_wat(MINI_SHELL, user_with_memory_import).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("memory") && msg.contains("host"),
        "expected memory-import rejection mentioning the import, got: {msg}"
    );
}

#[test]
fn forwarded_global_import_passes_through() -> Result<()> {
    // Globals are commonly imported from environment modules (e.g.
    // `env.__memory_base` from emscripten); forwarding them through
    // unchanged should yield a valid merged module.
    let user = r#"
        (module
          (import "env" "base" (global i32))
          (global $derived i32 (global.get 0))
          (func (export "get") (result i32) (global.get 0)))
    "#;
    let merged = link_wat(MINI_SHELL, user)?;
    let imps = imports_of(&merged)?;
    assert_eq!(
        imps,
        vec![ImportRow {
            module: "env".into(),
            name: "base".into()
        }]
    );
    // Validate against a wasmtime instance.
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &merged)?;
    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    let mut store = wasmtime::Store::new(&engine, ());
    let global = wasmtime::Global::new(
        &mut store,
        wasmtime::GlobalType::new(wasmtime::ValType::I32, wasmtime::Mutability::Const),
        wasmtime::Val::I32(0x1234),
    )?;
    linker.define(&mut store, "env", "base", global)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let get = instance.get_typed_func::<(), i32>(&mut store, "get")?;
    assert_eq!(get.call(&mut store, ())?, 0x1234);
    Ok(())
}
