//! Tests for custom-section preservation through the linker.
//!
//! The downstream blocker that motivates this: `wasm-tools component
//! new` reads `component-type:*` custom sections (emitted by
//! wit-bindgen) to map the cdylib's raw imports to component-model
//! interfaces. The linker historically dropped all custom sections,
//! which produced malformed components.
//!
//! These tests synthesize wasm inputs with embedded custom sections
//! and assert they survive the link step byte-for-byte and in the
//! right position relative to the structural sections.

use anyhow::Result;
use std::borrow::Cow;
use tvm_guest_mm_link::link;
use wasm_encoder::{
    CodeSection, CustomSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
    MemorySection, MemoryType, Module, TypeSection,
};

/// Build a "shell-shaped" module that the linker accepts: at least one
/// memory exported as `mem0` and at least one function exported under
/// a `tvm_mm`-namespace name so the user can wire to it. Custom
/// sections can be injected at any position via `customs`.
fn build_shell_with_customs(customs: &[(SectionPosition, &str, &[u8])]) -> Vec<u8> {
    let mut module = Module::new();

    let emit_at = |module: &mut Module, pos: SectionPosition| {
        for (p, name, data) in customs {
            if *p == pos {
                module.section(&CustomSection {
                    name: Cow::Borrowed(*name),
                    data: Cow::Borrowed(*data),
                });
            }
        }
    };

    // Types: one (i32, i32) -> i32 for `tvm_load_u8`.
    let mut types = TypeSection::new();
    types.ty().function(
        [wasm_encoder::ValType::I32, wasm_encoder::ValType::I32],
        [wasm_encoder::ValType::I32],
    );
    emit_at(&mut module, SectionPosition::BeforeTypes);
    module.section(&types);
    emit_at(&mut module, SectionPosition::AfterTypes);

    // Functions: one defined function using type 0.
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);
    emit_at(&mut module, SectionPosition::AfterFunctions);

    // Memory.
    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 1,
        maximum: Some(256),
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&mems);
    emit_at(&mut module, SectionPosition::AfterMemories);

    // Exports: mem0 (memory 0) + tvm_load_u8 (function 0).
    let mut exports = ExportSection::new();
    exports.export("mem0", ExportKind::Memory, 0);
    exports.export("tvm_load_u8", ExportKind::Func, 0);
    module.section(&exports);
    emit_at(&mut module, SectionPosition::AfterExports);

    // Code: function body that just returns i32.const 0.
    let mut codes = CodeSection::new();
    let mut body = Function::new(Vec::new());
    body.instruction(&Instruction::I32Const(0));
    body.instruction(&Instruction::End);
    codes.function(&body);
    module.section(&codes);
    emit_at(&mut module, SectionPosition::AfterCode);

    module.finish()
}

/// Build a user-side cdylib-shaped module that imports the shell's
/// `tvm_mm.tvm_load_u8` and exports a single function. Custom sections
/// can be injected at the requested positions. This mirrors the shape
/// of a rustc-emitted cdylib with embedded `component-type:*` sections
/// from wit-bindgen.
fn build_user_with_customs(customs: &[(SectionPosition, &str, &[u8])]) -> Vec<u8> {
    let mut module = Module::new();

    let emit_at = |module: &mut Module, pos: SectionPosition| {
        for (p, name, data) in customs {
            if *p == pos {
                module.section(&CustomSection {
                    name: Cow::Borrowed(*name),
                    data: Cow::Borrowed(*data),
                });
            }
        }
    };

    // Types: one (i32, i32) -> i32 for matching the load helper.
    let mut types = TypeSection::new();
    types.ty().function(
        [wasm_encoder::ValType::I32, wasm_encoder::ValType::I32],
        [wasm_encoder::ValType::I32],
    );
    emit_at(&mut module, SectionPosition::BeforeTypes);
    module.section(&types);
    emit_at(&mut module, SectionPosition::AfterTypes);

    // Imports: tvm_mm.tvm_load_u8.
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import(
        "tvm_mm",
        "tvm_load_u8",
        wasm_encoder::EntityType::Function(0),
    );
    module.section(&imports);
    emit_at(&mut module, SectionPosition::AfterImports);

    // One defined function.
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);

    // Exports: `go` -> the user-defined function (function index 1
    // since the import occupies index 0).
    let mut exports = ExportSection::new();
    exports.export("go", ExportKind::Func, 1);
    module.section(&exports);
    emit_at(&mut module, SectionPosition::AfterExports);

    // Code: tail-call the import.
    let mut codes = CodeSection::new();
    let mut body = Function::new(Vec::new());
    body.instruction(&Instruction::LocalGet(0));
    body.instruction(&Instruction::LocalGet(1));
    body.instruction(&Instruction::Call(0));
    body.instruction(&Instruction::End);
    codes.function(&body);
    module.section(&codes);
    emit_at(&mut module, SectionPosition::AfterCode);

    module.finish()
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum SectionPosition {
    BeforeTypes,
    AfterTypes,
    AfterImports,
    AfterFunctions,
    AfterMemories,
    AfterExports,
    AfterCode,
}

/// Collect every custom section from a wasm in source order as
/// (name, data) pairs.
fn customs_of(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CustomSection(reader) = payload? {
            out.push((reader.name().to_string(), reader.data().to_vec()));
        }
    }
    Ok(out)
}

#[test]
fn user_custom_section_survives_link() -> Result<()> {
    // A wit-bindgen-shaped `component-type:foo` custom section right
    // after the user's exports — the position wit-bindgen actually
    // emits these.
    let payload: &[u8] = b"\x00\x01\x02\x03 component-type binary blob";
    let shell = build_shell_with_customs(&[]);
    let user = build_user_with_customs(&[(
        SectionPosition::AfterCode,
        "component-type:foo",
        payload,
    )]);
    let merged = link(&shell, &user)?;

    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&merged)?;

    let customs = customs_of(&merged)?;
    assert!(
        customs
            .iter()
            .any(|(n, d)| n == "component-type:foo" && d == payload),
        "user component-type custom section must survive byte-for-byte; got {:?}",
        customs.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn shell_custom_section_survives_link() -> Result<()> {
    // Shells today emit no customs but the linker should pass any
    // through anyway. Synthesize a shell with a `producers` section.
    let payload: &[u8] = b"\x01\x02 producers";
    let shell = build_shell_with_customs(&[(
        SectionPosition::AfterCode,
        "producers",
        payload,
    )]);
    let user = build_user_with_customs(&[]);
    let merged = link(&shell, &user)?;
    let customs = customs_of(&merged)?;
    assert!(
        customs
            .iter()
            .any(|(n, d)| n == "producers" && d == payload),
        "shell `producers` custom must survive; got {:?}",
        customs.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn name_section_lands_after_code() -> Result<()> {
    // The `name` section is positional: validators expect it after the
    // Code section. If it lands before Code, some tooling rejects the
    // module. Confirm we preserve that ordering.
    let user = build_user_with_customs(&[(
        SectionPosition::AfterCode,
        "name",
        b"\x00\x05hello", // name-section payload — exact bytes don't matter
    )]);
    let shell = build_shell_with_customs(&[]);
    let merged = link(&shell, &user)?;

    // Walk the merged module's payload stream and assert the `name`
    // custom appears after the Code section.
    let mut saw_code = false;
    let mut name_after_code = false;
    for payload in wasmparser::Parser::new(0).parse_all(&merged) {
        match payload? {
            wasmparser::Payload::CodeSectionStart { .. } => saw_code = true,
            wasmparser::Payload::CustomSection(cs) if cs.name() == "name" => {
                if saw_code {
                    name_after_code = true;
                }
            }
            _ => {}
        }
    }
    assert!(
        saw_code,
        "merged module must have a Code section for the ordering check"
    );
    assert!(
        name_after_code,
        "`name` custom must end up after the Code section in the merged module"
    );
    Ok(())
}

#[test]
fn multiple_customs_all_pass_through() -> Result<()> {
    let user = build_user_with_customs(&[
        (
            SectionPosition::AfterTypes,
            "component-type:foo",
            b"foo-payload",
        ),
        (
            SectionPosition::AfterCode,
            "component-type:bar",
            b"bar-payload",
        ),
    ]);
    let shell = build_shell_with_customs(&[(
        SectionPosition::AfterCode,
        "producers",
        b"producers-payload",
    )]);
    let merged = link(&shell, &user)?;
    let customs = customs_of(&merged)?;
    let names: Vec<&str> = customs.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.contains(&"component-type:foo"),
        "missing component-type:foo; got {:?}",
        names
    );
    assert!(
        names.contains(&"component-type:bar"),
        "missing component-type:bar; got {:?}",
        names
    );
    assert!(
        names.contains(&"producers"),
        "missing producers; got {:?}",
        names
    );
    Ok(())
}

#[test]
fn no_customs_means_no_customs() -> Result<()> {
    // Sanity: a module with no inputs carrying customs gains none. This
    // also locks in the absence of stray customs synthesized by the
    // linker for previously-emitted artifacts.
    let shell = build_shell_with_customs(&[]);
    let user = build_user_with_customs(&[]);
    let merged = link(&shell, &user)?;
    let customs = customs_of(&merged)?;
    assert!(
        customs.is_empty(),
        "expected no custom sections; got {:?}",
        customs.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    Ok(())
}
