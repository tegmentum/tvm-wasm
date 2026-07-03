//! Tests for element-segment renumbering through the linker.
//!
//! The bug under test: when a user module has forwarded (non-`tvm_mm`)
//! imports, the merged module's function-index space gets prepended
//! with those imports. Every `call`/`ref.func` operand in code bodies
//! is renumbered to compensate — but historically, function indices
//! inside element segments were NOT renumbered when carried in the
//! expression form (`(ref.func N)` const-exprs), which is the shape
//! modern LLVM and wit-bindgen emit. Result: `call_indirect` through
//! the user's table dispatched to the wrong function and the module
//! trapped at instantiation with `undefined element: out of bounds
//! table access`.
//!
//! These tests synthesise both element-segment shapes and assert the
//! shift is applied to func indices in either form.

use anyhow::Result;
use std::borrow::Cow;
use tvm_guest_mm_link::link;
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, ImportSection, Instruction, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection,
};

/// Mini shell with one exported memory and TWO shell-defined functions.
/// Only `tvm_load_u8` is exported (the wire target for user imports);
/// the second is a private helper. Two shell-defined funcs is enough to
/// force the user's defined-function indices to differ between the
/// user's pre-link space and the merged-module space, so a missed
/// renumbering would manifest as a wrong index in the output.
fn mini_shell() -> Vec<u8> {
    let mut m = Module::new();
    let mut types = TypeSection::new();
    // Type 0: (i32, i32) -> i32 — load helper signature.
    types.ty().function(
        [wasm_encoder::ValType::I32, wasm_encoder::ValType::I32],
        [wasm_encoder::ValType::I32],
    );
    // Type 1: () -> () — helper signature for the private helper.
    types.ty().function([], []);
    m.section(&types);

    let mut funcs = FunctionSection::new();
    funcs.function(0); // shell func 0: tvm_load_u8
    funcs.function(1); // shell func 1: private helper (not exported)
    m.section(&funcs);

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 1,
        maximum: Some(256),
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    m.section(&mems);

    let mut exports = ExportSection::new();
    exports.export("mem0", ExportKind::Memory, 0);
    exports.export("tvm_load_u8", ExportKind::Func, 0);
    m.section(&exports);

    let mut codes = CodeSection::new();
    // shell func 0 body: i32.const 0; end
    let mut body = Function::new(Vec::new());
    body.instruction(&Instruction::I32Const(0));
    body.instruction(&Instruction::End);
    codes.function(&body);
    // shell func 1 body: end (empty void function)
    let helper = Function::new(Vec::new());
    let mut helper = helper;
    helper.instruction(&Instruction::End);
    codes.function(&helper);
    m.section(&codes);

    m.finish()
}

/// Walk the element section of a merged module and return one entry
/// per element segment as a list of (target_function_indices). Both
/// `Functions(...)` and `Expressions(...) of ref.func` are recognized.
fn element_segments_target_funcs(bytes: &[u8]) -> Result<Vec<Vec<u32>>> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::ElementSection(reader) = payload? {
            for el in reader {
                let el = el?;
                let mut entries = Vec::new();
                match el.items {
                    wasmparser::ElementItems::Functions(r) => {
                        for f in r {
                            entries.push(f?);
                        }
                    }
                    wasmparser::ElementItems::Expressions(_, r) => {
                        for expr in r {
                            let expr = expr?;
                            let mut ops = expr.get_operators_reader();
                            let op = ops.read()?;
                            if let wasmparser::Operator::RefFunc { function_index } = op {
                                entries.push(function_index);
                            } else {
                                anyhow::bail!(
                                    "unexpected op in element expr: {:?}; test only handles ref.func",
                                    op
                                );
                            }
                            // Expect End.
                            let _ = ops.read()?;
                        }
                    }
                }
                out.push(entries);
            }
        }
    }
    Ok(out)
}

/// Build a user module with:
/// - One `tvm_mm.tvm_load_u8` import
/// - One `host.dummy` forwarded import (so the merged module shifts
///   the func-index space by 1)
/// - One funcref table
/// - One defined function (the indirect-call target). This function's
///   index in the user module is `2` (after both imports); after
///   linking, the forwarded `host.dummy` import is index 0 in the
///   merged module, the shell's `tvm_load_u8` is index 1, and the
///   user's defined function lands at index 2 + (whatever shell
///   defined funcs are emitted by the shell — exactly 1 for the
///   mini_shell). So the merged-module index is `1 (fwd) + 1 (shell
///   defined) + (user_idx - 1 user import) = 2 (host) + 1 (shell) - 0 = 3
///   for the user's `target` function`.
/// - One element segment, in the shape `element_shape` requests, with
///   a single entry pointing at the user's defined function (index 2
///   in the user's pre-link space).
/// - One exported function `go` that does `call_indirect` against
///   element 0 of the table.
fn user_with_element_segment(element_shape: ElementShape) -> Vec<u8> {
    let mut m = Module::new();

    // Types:
    //   0: (i32, i32) -> i32  — for the load helper (matches shell)
    //   1: ()         -> i32  — for the indirect-call target / go
    //   2: ()         -> ()   — for the host dummy import
    let mut types = TypeSection::new();
    types.ty().function(
        [wasm_encoder::ValType::I32, wasm_encoder::ValType::I32],
        [wasm_encoder::ValType::I32],
    );
    types.ty().function([], [wasm_encoder::ValType::I32]);
    types.ty().function([], []);
    m.section(&types);

    // Imports: tvm_mm.tvm_load_u8 (idx 0), host.dummy (idx 1).
    let mut imports = ImportSection::new();
    imports.import("tvm_mm", "tvm_load_u8", EntityType::Function(0));
    imports.import("host", "dummy", EntityType::Function(2));
    m.section(&imports);

    // Functions: one defined function of type 1, one defined function
    // of type 1 (the `go` body that issues the indirect call).
    //   user idx 2 = `target` (returns i32.const 42)
    //   user idx 3 = `go` (calls target via the table)
    let mut funcs = FunctionSection::new();
    funcs.function(1);
    funcs.function(1);
    m.section(&funcs);

    // Table: 1 funcref entry, holds the indirect-call target.
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: 1,
        maximum: Some(1),
        table64: false,
        shared: false,
    });
    m.section(&tables);

    // Exports: `go` is the public entrypoint.
    let mut exports = ExportSection::new();
    exports.export("go", ExportKind::Func, 3);
    m.section(&exports);

    // Element segment: active, table 0, offset i32.const 0, single
    // entry pointing at user function index 2 (the `target` function).
    let mut elems = ElementSection::new();
    let offset = ConstExpr::i32_const(0);
    match element_shape {
        ElementShape::Functions => {
            let entries: [u32; 1] = [2];
            elems.active(
                Some(0),
                &offset,
                Elements::Functions(Cow::Borrowed(&entries)),
            );
        }
        ElementShape::Expressions => {
            let exprs = [ConstExpr::ref_func(2)];
            elems.active(
                Some(0),
                &offset,
                Elements::Expressions(RefType::FUNCREF, Cow::Borrowed(&exprs)),
            );
        }
    }
    m.section(&elems);

    // Code:
    //   user idx 2 (target): i32.const 42; end
    //   user idx 3 (go):     i32.const 0; call_indirect 0 (type 1, table 0); end
    let mut codes = CodeSection::new();
    let mut target = Function::new(Vec::new());
    target.instruction(&Instruction::I32Const(42));
    target.instruction(&Instruction::End);
    codes.function(&target);

    let mut go = Function::new(Vec::new());
    go.instruction(&Instruction::I32Const(0));
    go.instruction(&Instruction::CallIndirect {
        type_index: 1,
        table_index: 0,
    });
    go.instruction(&Instruction::End);
    codes.function(&go);
    m.section(&codes);

    m.finish()
}

#[derive(Clone, Copy)]
enum ElementShape {
    /// Compact form: `(elem (func $a $b $c))` — direct function indices.
    Functions,
    /// Expression form: `(elem funcref (ref.func $a) (ref.func $b))` —
    /// the shape modern LLVM + wit-bindgen emit.
    Expressions,
}

/// Compute the expected merged-module index for the user's `target`
/// function. Layout after linking with a 1-`host.dummy` forwarded
/// import and the `mini_shell` (2 shell-defined funcs):
///   merged idx 0 = forwarded import `host.dummy`
///   merged idx 1 = shell-defined `tvm_load_u8`
///   merged idx 2 = shell-defined private helper
///   merged idx 3 = user's `target` (pre-link user idx 2 = first
///                  defined user fn)
///   merged idx 4 = user's `go`
///
/// Importantly, this expected merged idx (3) DIFFERS from the user's
/// pre-link idx (2) — so a linker that fails to renumber the element
/// segment's func operand would land on shell idx 2 (a private void
/// helper) instead of the user's `target` (a returning-i32 function),
/// either mistyped-trapping or returning a stale value.
fn expected_merged_target_idx() -> u32 {
    3
}

#[test]
fn element_segment_functions_form_renumbers_func_idx() -> Result<()> {
    let shell = mini_shell();
    let user = user_with_element_segment(ElementShape::Functions);
    let merged = link(&shell, &user)?;
    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&merged)?;

    let segs = element_segments_target_funcs(&merged)?;
    assert_eq!(segs.len(), 1, "expected exactly one element segment");
    assert_eq!(
        segs[0],
        vec![expected_merged_target_idx()],
        "Functions-form element entry must be shifted into the merged func-index space"
    );
    Ok(())
}

#[test]
fn element_segment_expressions_form_renumbers_ref_func() -> Result<()> {
    let shell = mini_shell();
    let user = user_with_element_segment(ElementShape::Expressions);
    let merged = link(&shell, &user)?;
    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&merged)?;

    let segs = element_segments_target_funcs(&merged)?;
    assert_eq!(segs.len(), 1, "expected exactly one element segment");
    assert_eq!(
        segs[0],
        vec![expected_merged_target_idx()],
        "Expressions-form `ref.func N` operand must be shifted into the merged func-index space"
    );
    Ok(())
}

#[test]
fn element_segment_expressions_call_indirect_runs_correctly() -> Result<()> {
    // Beyond static index inspection: actually instantiate the merged
    // module and confirm the indirect call lands on the right function.
    // This is the runtime check that catches the original
    // wit-bindgen-shaped trap "undefined element: out of bounds table
    // access".
    let shell = mini_shell();
    let user = user_with_element_segment(ElementShape::Expressions);
    let merged = link(&shell, &user)?;

    let mut config = wasmtime::Config::new();
    config.wasm_multi_memory(true);
    let engine = wasmtime::Engine::new(&config)?;
    let module = wasmtime::Module::new(&engine, &merged)?;

    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    linker.func_wrap("host", "dummy", || {})?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    let go = instance.get_typed_func::<(), i32>(&mut store, "go")?;
    let v = go.call(&mut store, ())?;
    assert_eq!(
        v, 42,
        "indirect call through the renumbered element segment must land on `target` which returns 42"
    );
    Ok(())
}

#[test]
fn user_default_table_element_targets_renumbered_table() -> Result<()> {
    // Subtlety the wit-bindgen-shaped failure mode also exercises:
    // a user element segment that omits the table index (the rustc
    // default for the single `__indirect_function_table`) MUST be
    // re-emitted with an explicit `Some(map_table(0))` so it ends up
    // initializing the user's renumbered table, not the shell's table
    // 0. Pre-fix, the default-table form encoded as table-0 in the
    // merged module too — which is the shell's first table, not the
    // user's — and the merged module trapped at instantiation with
    // `undefined element: out of bounds table access` because the
    // shell's table is sized for shell purposes (typically 2 funcref
    // slots, not 5).
    //
    // Drive a synthetic shell that also has tables, then a user with
    // a default-table element segment, and assert the linked module
    // instantiates successfully on wasmtime — the only way that can
    // happen is if the user's element segment lands on the renumbered
    // user table.
    let shell = shell_with_tables();
    let user = user_with_element_segment_default_table();
    let merged = link(&shell, &user)?;

    let mut config = wasmtime::Config::new();
    config.wasm_multi_memory(true);
    let engine = wasmtime::Engine::new(&config)?;
    let module = wasmtime::Module::new(&engine, &merged)?;
    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    linker.func_wrap("host", "dummy", || {})?;
    let mut store = wasmtime::Store::new(&engine, ());
    let _instance = linker.instantiate(&mut store, &module).map_err(|e| {
        anyhow::anyhow!(
            "instantiation failed — likely the user's default-table element segment is initializing the shell's table 0 instead of the user's renumbered table: {e}"
        )
    })?;
    Ok(())
}

/// Variant of `mini_shell` that ALSO carries a shell-side table — so
/// the user's "default table" (index 0 in user-space) gets renumbered
/// to a non-zero index in the merged module. This is what makes the
/// missed-renumbering bug observable at instantiation.
fn shell_with_tables() -> Vec<u8> {
    let mut m = Module::new();
    let mut types = TypeSection::new();
    types.ty().function(
        [wasm_encoder::ValType::I32, wasm_encoder::ValType::I32],
        [wasm_encoder::ValType::I32],
    );
    types.ty().function([], []);
    types.ty().function([], [wasm_encoder::ValType::I32]);
    m.section(&types);

    let mut funcs = FunctionSection::new();
    funcs.function(0); // tvm_load_u8
    funcs.function(1); // private helper
    m.section(&funcs);

    let mut tables = TableSection::new();
    // Shell table 0: size 2 — so a user element with 1 entry at offset
    // 0 would still fit, but with offset 1 (rustc's null-sentinel
    // pattern) we'd OOB. Sufficient to expose the bug at instantiation.
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: 2,
        maximum: Some(2),
        table64: false,
        shared: false,
    });
    m.section(&tables);

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 1,
        maximum: Some(256),
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    m.section(&mems);

    let mut exports = ExportSection::new();
    exports.export("mem0", ExportKind::Memory, 0);
    exports.export("tvm_load_u8", ExportKind::Func, 0);
    m.section(&exports);

    let mut codes = CodeSection::new();
    let mut body = Function::new(Vec::new());
    body.instruction(&Instruction::I32Const(0));
    body.instruction(&Instruction::End);
    codes.function(&body);
    let mut helper = Function::new(Vec::new());
    helper.instruction(&Instruction::End);
    codes.function(&helper);
    m.section(&codes);

    m.finish()
}

/// User module that:
/// - imports `host.dummy` (forwarded)
/// - declares one funcref table of size 5
/// - has one defined function `target` (returns i32)
/// - emits an element segment using the default-table shape (no
///   explicit table index) at offset 1, with one entry pointing at
///   `target`.
fn user_with_element_segment_default_table() -> Vec<u8> {
    let mut m = Module::new();

    let mut types = TypeSection::new();
    types.ty().function([], []); // type 0: host.dummy
    types.ty().function([], [wasm_encoder::ValType::I32]); // type 1: target / go
    m.section(&types);

    let mut imports = ImportSection::new();
    imports.import("host", "dummy", EntityType::Function(0));
    m.section(&imports);

    let mut funcs = FunctionSection::new();
    funcs.function(1); // target
    funcs.function(1); // go
    m.section(&funcs);

    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: 5,
        maximum: Some(5),
        table64: false,
        shared: false,
    });
    m.section(&tables);

    let mut exports = ExportSection::new();
    // user func idx layout:
    //   0 = host.dummy (import)
    //   1 = target
    //   2 = go
    exports.export("go", ExportKind::Func, 2);
    m.section(&exports);

    let mut elems = ElementSection::new();
    // Offset 4 + 1 entry → slot 4 in the user's size-5 table. The
    // shell table is size 2, so writing slot 4 there would OOB at
    // instantiation. This is what catches the "default-table maps to
    // shell-table-0" bug: without the fix the merged module fails to
    // instantiate; with the fix the user's element targets the
    // renumbered user table where slot 4 is in bounds.
    let offset = ConstExpr::i32_const(4);
    let entries: [u32; 1] = [1]; // target = user idx 1
                                 // Active element segment with TABLE INDEX = None (i.e. default —
                                 // table 0 in user-space). This is the shape rustc emits for the
                                 // implicit `__indirect_function_table` and is the path that needs
                                 // explicit-materialization of the renumbered table index.
    elems.active(None, &offset, Elements::Functions(Cow::Borrowed(&entries)));
    m.section(&elems);

    let mut codes = CodeSection::new();
    let mut target = Function::new(Vec::new());
    target.instruction(&Instruction::I32Const(7));
    target.instruction(&Instruction::End);
    codes.function(&target);

    let mut go = Function::new(Vec::new());
    go.instruction(&Instruction::I32Const(1)); // table index
    go.instruction(&Instruction::CallIndirect {
        type_index: 1,
        table_index: 0,
    });
    go.instruction(&Instruction::End);
    codes.function(&go);
    m.section(&codes);

    m.finish()
}

#[test]
fn element_segment_functions_call_indirect_runs_correctly() -> Result<()> {
    // Same as above but with the Functions-form element segment. This
    // is the path that already worked before the fix; the test locks
    // it in as a regression guard so the new code path doesn't break
    // the old shape.
    let shell = mini_shell();
    let user = user_with_element_segment(ElementShape::Functions);
    let merged = link(&shell, &user)?;

    let mut config = wasmtime::Config::new();
    config.wasm_multi_memory(true);
    let engine = wasmtime::Engine::new(&config)?;
    let module = wasmtime::Module::new(&engine, &merged)?;

    let mut linker: wasmtime::Linker<()> = wasmtime::Linker::new(&engine);
    linker.func_wrap("host", "dummy", || {})?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    let go = instance.get_typed_func::<(), i32>(&mut store, "go")?;
    let v = go.call(&mut store, ())?;
    assert_eq!(v, 42);
    Ok(())
}
