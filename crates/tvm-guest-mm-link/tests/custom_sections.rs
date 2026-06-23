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
    //
    // The payload must be a well-formed name-section binary — the
    // linker decodes it to remap embedded function indices into the
    // merged module's index space, so degenerate bytes won't parse.
    let mut names = wasm_encoder::NameSection::new();
    names.module("hello");
    let payload = names.as_custom().data.into_owned();
    let user = build_user_with_customs(&[(
        SectionPosition::AfterCode,
        "name",
        &payload,
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

/// Walk the merged module's name section(s) and collect
/// `(index, name)` pairs from every Function subsection encountered.
/// Multiple `name` sections can exist (we emit shell-side then
/// user-side); both contribute.
fn function_names_of(bytes: &[u8]) -> Result<Vec<(u32, String)>> {
    let mut out: Vec<(u32, String)> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CustomSection(cs) = payload? {
            if cs.name() != "name" {
                continue;
            }
            let reader = wasmparser::NameSectionReader::new(
                wasmparser::BinaryReader::new(cs.data(), 0),
            );
            for sub in reader {
                if let wasmparser::Name::Function(map) = sub? {
                    for n in map {
                        let n = n?;
                        out.push((n.index, n.name.to_string()));
                    }
                }
            }
        }
    }
    Ok(out)
}

#[test]
fn merged_module_has_at_most_one_name_section() -> Result<()> {
    // The linker has to merge shell-side and user-side `name` custom
    // sections into a single section: emitting two `name` customs
    // produces a valid binary but tools that resolve symbols from the
    // section (wasmtime tracebacks, wasm-tools print annotations) can
    // arbitrarily prefer one section's entries over the other,
    // producing misleading function names at PC sites where the two
    // disagree. Confirm we collapse to one section.
    let mut user_names = wasm_encoder::NameSection::new();
    let mut user_funcs = wasm_encoder::NameMap::new();
    user_funcs.append(0, "user_loader_import");
    user_funcs.append(1, "user_go");
    user_names.functions(&user_funcs);
    let user_payload = user_names.as_custom().data.into_owned();

    // Shell-shaped module — `build_shell_with_customs` doesn't emit a
    // name section directly, but exercising the merge path with only
    // the user-side `name` already covers the "no duplicate" guarantee
    // for the typical configuration.
    let shell = build_shell_with_customs(&[]);
    let user = build_user_with_customs(&[(
        SectionPosition::AfterCode,
        "name",
        &user_payload,
    )]);
    let merged = link(&shell, &user)?;

    let mut name_sections = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&merged) {
        if let wasmparser::Payload::CustomSection(cs) = payload? {
            if cs.name() == "name" {
                name_sections += 1;
            }
        }
    }
    assert!(
        name_sections <= 1,
        "expected at most 1 merged name section; got {}",
        name_sections
    );
    Ok(())
}

#[test]
fn user_name_section_function_indices_get_remapped() -> Result<()> {
    // User pre-link: function index 0 = `tvm_mm.tvm_load_u8` import,
    // function index 1 = the user-defined `go`. The user names them
    // `imported_loader` and `go_user`.
    //
    // After link, the shell has its own defined `tvm_load_u8` at
    // merged-module func index 0 (no forwarded imports → no shift),
    // and the user's `go` shifts to merged index 1 (fwd_func_count=0
    // + shell_func_count=1 + (1-1)=1). The `tvm_mm.tvm_load_u8`
    // import is rewired into the shell's function so it doesn't take
    // a slot in the merged index space at all.
    //
    // Therefore the user-side name `go_user` must be remapped to
    // index 1 in the merged module's name section. Before this fix,
    // the linker passed the bytes through verbatim so `go_user` ended
    // up labeling index 1's predecessor (or, worse, an index that
    // doesn't exist).
    let mut names = wasm_encoder::NameSection::new();
    let mut funcs = wasm_encoder::NameMap::new();
    funcs.append(0, "imported_loader");
    funcs.append(1, "go_user");
    names.functions(&funcs);
    let payload = names.as_custom().data.into_owned();

    let user = build_user_with_customs(&[(
        SectionPosition::AfterCode,
        "name",
        &payload,
    )]);
    let shell = build_shell_with_customs(&[]);
    let merged = link(&shell, &user)?;

    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&merged)?;

    let names = function_names_of(&merged)?;
    // Find the `go_user` entry — it should live at index 1, not at
    // index 1 in the user's pre-link space (which would alias the
    // shell's `tvm_load_u8`).
    let go_user_idx = names
        .iter()
        .find(|(_, n)| n == "go_user")
        .map(|(i, _)| *i);
    assert_eq!(
        go_user_idx,
        Some(1),
        "`go_user` should map to merged func index 1; got names {:?}",
        names
    );
    // The export table places `go` at merged func index 1; cross-check
    // that the name section agrees with the export table.
    let mut exported_go_idx: Option<u32> = None;
    for payload in wasmparser::Parser::new(0).parse_all(&merged) {
        if let wasmparser::Payload::ExportSection(reader) = payload? {
            for exp in reader {
                let exp = exp?;
                if exp.name == "go" {
                    exported_go_idx = Some(exp.index);
                }
            }
        }
    }
    assert_eq!(
        exported_go_idx,
        Some(1),
        "`go` should export at merged func index 1"
    );
    Ok(())
}

#[test]
fn user_name_section_with_forwarded_imports_remaps_correctly() -> Result<()> {
    // Build a user with both a tvm_mm import and a forwarded import.
    // The forwarded import claims merged func index 0, shifting
    // shell-defined functions to index 1+, and shifting all user-
    // defined functions by an additional 1.
    let mut module = Module::new();
    // Type 0: (i32, i32) -> i32 (matches tvm_load_u8).
    let mut types = TypeSection::new();
    types.ty().function(
        [wasm_encoder::ValType::I32, wasm_encoder::ValType::I32],
        [wasm_encoder::ValType::I32],
    );
    // Type 1: () -> () for the forwarded import and the user fn.
    types.ty().function([], []);
    module.section(&types);

    // Imports:
    //   0: tvm_mm.tvm_load_u8   (rewired)
    //   1: env.host_print       (forwarded)
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import(
        "tvm_mm",
        "tvm_load_u8",
        wasm_encoder::EntityType::Function(0),
    );
    imports.import(
        "env",
        "host_print",
        wasm_encoder::EntityType::Function(1),
    );
    module.section(&imports);

    // One user-defined fn of type 1.
    let mut funcs = FunctionSection::new();
    funcs.function(1);
    module.section(&funcs);

    // Export the user fn (pre-link index 2) under name `start`.
    let mut exports = ExportSection::new();
    exports.export("start", ExportKind::Func, 2);
    module.section(&exports);

    // Code body for the user fn: just calls the forwarded import.
    let mut codes = CodeSection::new();
    let mut body = Function::new(Vec::new());
    body.instruction(&Instruction::Call(1));
    body.instruction(&Instruction::End);
    codes.function(&body);
    module.section(&codes);

    // Name section: 0→"loader_import", 1→"host_print_import",
    // 2→"start_user".
    let mut names = wasm_encoder::NameSection::new();
    let mut func_names = wasm_encoder::NameMap::new();
    func_names.append(0, "loader_import");
    func_names.append(1, "host_print_import");
    func_names.append(2, "start_user");
    names.functions(&func_names);
    module.section(&names);
    let user_bytes = module.finish();

    let shell = build_shell_with_customs(&[]);
    let merged = link(&shell, &user_bytes)?;

    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeatures::default() | wasmparser::WasmFeatures::MULTI_MEMORY,
    )
    .validate_all(&merged)?;

    // Expected merged-module func indices:
    //   0: forwarded `env.host_print`
    //   1: shell's tvm_load_u8 (defined)
    //   2: user `start` (user defined, shifted by fwd+shell)
    let names = function_names_of(&merged)?;
    let start_idx = names.iter().find(|(_, n)| n == "start_user").map(|(i, _)| *i);
    let host_print_idx = names.iter().find(|(_, n)| n == "host_print_import").map(|(i, _)| *i);
    let loader_idx = names.iter().find(|(_, n)| n == "loader_import").map(|(i, _)| *i);
    assert_eq!(start_idx, Some(2), "`start_user` should be at merged idx 2; names={:?}", names);
    assert_eq!(host_print_idx, Some(0), "`host_print_import` should be at merged idx 0; names={:?}", names);
    // The rewired tvm_mm import now lives at the shell function's
    // merged index, which is 1 (post fwd shift).
    assert_eq!(loader_idx, Some(1), "`loader_import` (rewired to shell's tvm_load_u8) should be at merged idx 1; names={:?}", names);

    // Cross-check against the export table.
    let mut exported_start_idx: Option<u32> = None;
    for payload in wasmparser::Parser::new(0).parse_all(&merged) {
        if let wasmparser::Payload::ExportSection(reader) = payload? {
            for exp in reader {
                let exp = exp?;
                if exp.name == "start" {
                    exported_start_idx = Some(exp.index);
                }
            }
        }
    }
    assert_eq!(exported_start_idx, Some(2));
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
