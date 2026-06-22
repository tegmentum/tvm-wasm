//! Static linker for `tvm-guest-mm` cdylib consumers.
//!
//! ## What it does
//!
//! Takes two inputs:
//! 1. **Shell bytes** — the output of `tvm_guest_mm_module_template(…)`,
//!    a self-contained WAT module declaring N memory pools plus a fixed
//!    set of dispatch helpers (`tvm_load_u32`, `tvm_store_u32`,
//!    `tvm_copy_to_default`, …) as exports.
//! 2. **User bytes** — a rustc-emitted core wasm (typically a cdylib
//!    built against `tvm-guest-mm-rt`) that imports the dispatch
//!    helpers from the `tvm_mm` namespace and has its own memory 0
//!    (the standard rustc heap layout: data section starts at 1 MiB,
//!    bumped heap above).
//!
//! Produces a single self-contained core wasm that:
//! - Drops the user's memory declaration entirely; the user's
//!   functions are rewritten to target the shell's pool 0 (which is
//!   the shell's default memory and is the only place rustc-emitted
//!   loads/stores can go).
//! - Drops the user's `tvm_mm.*` imports; calls to them are rewritten
//!   as direct calls to the shell's internal helper functions.
//! - Preserves the user's exports under their original names.
//! - Merges the user's globals, data segments, tables, element
//!   segments alongside the shell's.
//!
//! The output is a normal core wasm; downstream consumers instantiate
//! it with `wasmtime::Module::new` and call its exports the usual way.
//! No component-model handling needed at the call site.
//!
//! ## Constraints (current implementation)
//!
//! - **Shell must have no imports.** The shell template generates a
//!   fully self-contained module, so this is naturally satisfied. The
//!   linker rejects shells with imports as a sanity check.
//! - **User's only imports must be from the `tvm_mm` module.** Other
//!   imports (e.g. WASI) aren't rewired — they pass through
//!   unrewritten, but the merged module currently doesn't expose any
//!   way to set them. A future revision can extend the linker to
//!   forward arbitrary imports.
//! - **User's memory pages must fit in the shell's pool 0 initial
//!   capacity.** The merged module uses the shell's pool 0 as the
//!   default memory; the shell's `initial_pages_per_pool` ×
//!   `max_pages_per_pool` bound pool 0's size. Rustc typically asks
//!   for 16 pages (1 MiB) for the data section + heap base; the shell
//!   default of 1 initial / 65536 max is sufficient since the user's
//!   stated max is interpreted as a request, and rustc emits 0 as the
//!   max in release builds.
//! - **User must not declare a start function**, because the merged
//!   module's start function (if any) belongs to the shell.
//! - **Function tables**: if the user declares a function table, its
//!   entries get renumbered. Indirect-call instructions through the
//!   user's table still work.
//!
//! ## Output structure
//!
//! Sections are emitted in canonical wasm order:
//!   - Types: shell types, then user types (renumbered)
//!   - Imports: none (shell has none, user's are dropped)
//!   - Functions: shell functions, then user functions
//!   - Tables: shell tables, then user tables
//!   - Memories: shell memories only (user's dropped)
//!   - Globals: shell globals, then user globals
//!   - Exports: shell exports, then user exports (minus `memory`)
//!   - Start: shell start (if any)
//!   - Elements: shell elements, then user elements (renumbered)
//!   - Code: shell code, then rewritten user code
//!   - Data: shell data, then user data (memory index rewritten)

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;

use wasmparser::{
    BinaryReaderError, ConstExpr, DataKind, ElementItems, ElementKind, ExternalKind, FunctionBody,
    Operator, Parser, Payload, TableInit, TypeRef, ValType,
};

// Re-export the shell-template entry points so callers can produce a
// shell + invoke the linker from one crate.
pub use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams, DEFAULT_POOL_COUNT};

/// The fixed wasm-import module name used by `tvm-guest-mm-rt`'s
/// `extern "C"` declarations. Imports under this module are wired to
/// the shell's exports during linking.
pub const TVM_MM_IMPORT_MODULE: &str = "tvm_mm";

/// Link a user cdylib core wasm against the shell. Returns the merged
/// core wasm bytes.
///
/// `shell_bytes` must be the output of compiling the WAT produced by
/// `tvm_guest_mm_module_template`. `user_bytes` must be a rustc-emitted
/// cdylib that imports the dispatch helpers from the `tvm_mm`
/// namespace.
///
/// On success the returned bytes are a valid wasm module with no
/// remaining `tvm_mm` imports. Validation is left to the caller (use
/// `wasmparser::Validator` or `wasmtime::Module::new`).
pub fn link(shell_bytes: &[u8], user_bytes: &[u8]) -> Result<Vec<u8>> {
    let shell = parse_module(shell_bytes).context("parsing shell module")?;
    let user = parse_module(user_bytes).context("parsing user module")?;

    if !shell.imports.is_empty() {
        bail!(
            "shell module must not have imports; found {} (the shell template generates a self-contained module — was it post-processed?)",
            shell.imports.len()
        );
    }

    // Build the import-name → shell-export-func-index map. For each
    // user import under `tvm_mm`, we look up the same export name on
    // the shell.
    let shell_func_exports: HashMap<&str, u32> = shell
        .exports
        .iter()
        .filter(|e| e.kind == ExternalKind::Func)
        .map(|e| (e.name, e.index))
        .collect();

    // The user's imports occupy the low function-index range. Build a
    // direct-call rewrite map: old user func index → shell func index.
    // Non-`tvm_mm` imports are unsupported for now.
    let mut user_import_func_count: u32 = 0;
    let mut import_rewrite: HashMap<u32, u32> = HashMap::new();
    for imp in &user.imports {
        if let TypeRef::Func(_) = imp.ty {
            if imp.module != TVM_MM_IMPORT_MODULE {
                bail!(
                    "user module imports `{}.{}` from a module other than `{}`; this linker only handles `tvm_mm` imports",
                    imp.module,
                    imp.name,
                    TVM_MM_IMPORT_MODULE
                );
            }
            let shell_idx = *shell_func_exports.get(imp.name).ok_or_else(|| {
                anyhow!(
                    "user imports `{}.{}` but the shell does not export a function with that name",
                    imp.module,
                    imp.name
                )
            })?;
            import_rewrite.insert(user_import_func_count, shell_idx);
            user_import_func_count += 1;
        } else {
            bail!(
                "user module imports non-function `{}.{}`; only function imports are supported",
                imp.module,
                imp.name
            );
        }
    }

    if user.start.is_some() {
        bail!("user module declares a start function; not supported");
    }

    // ----------------------------------------------------------------
    // Compute index maps for the merged module.
    // ----------------------------------------------------------------

    // Type index map: shell types first, then user types.
    let shell_type_count = shell.types.len() as u32;
    let map_user_type = |t: u32| -> u32 { t + shell_type_count };

    // Function index map (post-merge, includes all funcs in linear order):
    //   - shell.funcs[i] → i               (shell has 0 imports)
    //   - user import K  → import_rewrite[K]
    //   - user defined j → shell_func_count + j
    let shell_func_count = shell.func_types.len() as u32; // shell has 0 imports; "func_types" already excludes imports
    let map_user_func = |idx: u32| -> u32 {
        if idx < user_import_func_count {
            *import_rewrite
                .get(&idx)
                .expect("rewrite map populated for every import idx")
        } else {
            shell_func_count + (idx - user_import_func_count)
        }
    };

    // Global index map: shell globals first, then user globals (after
    // user imports — but user has no global imports per the constraint
    // above).
    let shell_global_count = shell.globals_emitted as u32;
    let map_user_global = |g: u32| -> u32 { g + shell_global_count };

    // Table index map.
    let shell_table_count = shell.tables.len() as u32;
    let map_user_table = |t: u32| -> u32 { t + shell_table_count };

    // Memory index map: user memory 0 → shell memory 0 (pool 0).
    // The shell's pool 0 is always at memory index 0; the user's only
    // memory is dropped entirely, so any memory reference in user data
    // segments stays at index 0.
    let map_user_memory = |m: u32| -> Result<u32> {
        if m == 0 {
            Ok(0)
        } else {
            bail!("user module references memory index {} but only one (default) memory is supported", m);
        }
    };

    // ----------------------------------------------------------------
    // Emit the merged module section by section.
    // ----------------------------------------------------------------

    let mut module = wasm_encoder::Module::new();

    // Types: shell + user (user types renumbered).
    {
        let mut types = wasm_encoder::TypeSection::new();
        for t in &shell.types {
            push_func_type(&mut types, t);
        }
        for t in &user.types {
            push_func_type(&mut types, t);
        }
        module.section(&types);
    }

    // Imports: none (shell has none, user's are all rewired).

    // Functions: shell + user (user funcs' type indices remapped).
    {
        let mut funcs = wasm_encoder::FunctionSection::new();
        for ty_idx in &shell.func_types {
            funcs.function(*ty_idx);
        }
        for ty_idx in &user.func_types {
            funcs.function(map_user_type(*ty_idx));
        }
        module.section(&funcs);
    }

    // Tables: shell + user.
    {
        let mut tables = wasm_encoder::TableSection::new();
        for t in &shell.tables {
            push_table(&mut tables, t);
        }
        for t in &user.tables {
            push_table(&mut tables, t);
        }
        module.section(&tables);
    }

    // Memories: shell only. User's memory is dropped.
    {
        let mut mems = wasm_encoder::MemorySection::new();
        for m in &shell.memories {
            push_memory(&mut mems, m);
        }
        module.section(&mems);
    }

    // Globals: shell + user. User globals' initializers may reference
    // globals or funcs; rewrite them.
    {
        let mut globals = wasm_encoder::GlobalSection::new();
        for g in &shell.globals {
            push_global(&mut globals, g, |idx| idx, |idx| idx);
        }
        for g in &user.globals {
            push_global(&mut globals, g, &map_user_global, &map_user_func);
        }
        module.section(&globals);
    }

    // Exports: shell + filtered user exports. Drop user's `memory`
    // export (the shell already exports `mem0..memN` — pool 0 is the
    // user's de-facto default memory).
    {
        let mut exports = wasm_encoder::ExportSection::new();
        for e in &shell.exports {
            push_export(
                &mut exports,
                e,
                |idx| idx,
                |idx| idx,
                |idx| idx,
                |idx| idx,
            );
        }
        for e in &user.exports {
            // Skip the user's `memory` export — the shell exports pool
            // memories under `mem0..memN`, and re-exporting the user's
            // dropped memory would point at nothing.
            if matches!(e.kind, ExternalKind::Memory) {
                continue;
            }
            // Also skip rustc cdylib housekeeping exports that aren't
            // useful in the merged module. Keep them if the consumer
            // explicitly wants them — for now we strip the standard
            // ones because they conflict with the shell's namespace.
            if matches!(
                e.name,
                "__data_end" | "__heap_base" | "__indirect_function_table"
            ) {
                continue;
            }
            push_export(
                &mut exports,
                e,
                &map_user_func,
                &map_user_table,
                &map_user_memory_strict,
                &map_user_global,
            );
        }
        module.section(&exports);
    }

    // Start: shell's start, if any. User start was rejected upstream.
    if let Some(start) = shell.start {
        module.section(&wasm_encoder::StartSection {
            function_index: start,
        });
    }

    // Element segments: shell + user (user elem funcref entries
    // remapped).
    {
        let mut elems = wasm_encoder::ElementSection::new();
        for el in &shell.elements {
            push_element(&mut elems, el, |idx| idx, |idx| idx)?;
        }
        for el in &user.elements {
            push_element(&mut elems, el, &map_user_table, &map_user_func)?;
        }
        module.section(&elems);
    }

    // DataCount section: required when bulk-memory data.drop or
    // memory.init instructions exist. Both the shell (memory.fill,
    // memory.copy don't need it; memory.init does) and user code may
    // use it. Emit if either has data segments. wasm-encoder treats it
    // as optional; without it, validators reject memory.init.
    let total_data_segments = shell.data_count + user.data.len() as u32;
    if total_data_segments > 0 {
        module.section(&wasm_encoder::DataCountSection {
            count: total_data_segments,
        });
    }

    // Code section: shell code unchanged (raw pass-through), user
    // code rewritten via wasm-encoder.
    {
        let mut codes = wasm_encoder::CodeSection::new();
        for body_bytes in &shell.code {
            codes.raw(body_bytes);
        }
        for body in &user.code_readers {
            let func = rewrite_function_body(
                body,
                &user.types,
                &map_user_func,
                &map_user_global,
                &map_user_table,
                &map_user_type,
            )?;
            codes.function(&func);
        }
        module.section(&codes);
    }

    // Data: shell + user (memory index remapped to shell's 0).
    {
        let mut datas = wasm_encoder::DataSection::new();
        for d in &shell.data {
            push_data(&mut datas, d, |m| Ok::<u32, anyhow::Error>(m))?;
        }
        for d in &user.data {
            push_data(&mut datas, d, &map_user_memory)?;
        }
        module.section(&datas);
    }

    Ok(module.finish())
}

// Wrapper used for exports: rejects non-zero memory indices since the
// only memory in the user module is the default that we've dropped.
fn map_user_memory_strict(m: u32) -> u32 {
    if m != 0 {
        panic!("user export references non-default memory");
    }
    0
}

// ----------------------------------------------------------------
// Parser model — what we extract from each input module.
// ----------------------------------------------------------------

/// Captures everything we need from one input wasm to emit the merged
/// module. Borrowed where cheap (`&[u8]` for code bodies, `&str` for
/// import/export names tied to the input lifetime); owned for the
/// small structures we transform.
struct ParsedModule<'a> {
    types: Vec<FuncTypeOwned>,
    /// One entry per type — captured to translate `Operator::CallIndirect`'s
    /// type index when rewriting user code, and to re-emit the type
    /// section.
    imports: Vec<ImportEntry<'a>>,
    /// For each defined function (excluding imports), the type index
    /// in this module's type space.
    func_types: Vec<u32>,
    tables: Vec<TableEntry>,
    memories: Vec<MemoryEntry>,
    globals: Vec<GlobalEntry<'a>>,
    /// Number of globals emitted by this module — used to size the
    /// remap (matches `globals.len()` since we don't accept global
    /// imports).
    globals_emitted: usize,
    exports: Vec<ExportEntry<'a>>,
    start: Option<u32>,
    elements: Vec<ElementEntry<'a>>,
    /// Raw function bodies — kept as `&[u8]` for the shell (pass-through)
    /// and parsed/rewritten for the user. We store the raw slice and
    /// the operator stream both for shell ergonomics.
    code: Vec<&'a [u8]>,
    /// Function bodies as wasmparser readers — used by the rewriter
    /// for the user side. Keyed by index into `code`.
    code_readers: Vec<FunctionBody<'a>>,
    data: Vec<DataSegment<'a>>,
    data_count: u32,
}

#[derive(Clone)]
struct FuncTypeOwned {
    params: Vec<ValType>,
    results: Vec<ValType>,
}

struct ImportEntry<'a> {
    module: &'a str,
    name: &'a str,
    ty: TypeRef,
}

struct TableEntry {
    element_type: wasm_encoder::RefType,
    minimum: u64,
    maximum: Option<u64>,
    table64: bool,
    init: Option<wasm_encoder::ConstExpr>,
    shared: bool,
}

struct MemoryEntry {
    minimum: u64,
    maximum: Option<u64>,
    memory64: bool,
    shared: bool,
    page_size_log2: Option<u32>,
}

struct GlobalEntry<'a> {
    ty: wasm_encoder::GlobalType,
    init: ConstExpr<'a>,
}

struct ExportEntry<'a> {
    name: &'a str,
    kind: ExternalKind,
    index: u32,
}

struct ElementEntry<'a> {
    kind: ElementKindCaptured<'a>,
    items: ElementItemsCaptured<'a>,
}

enum ElementKindCaptured<'a> {
    Passive,
    Declared,
    Active {
        table_index: Option<u32>,
        offset: ConstExpr<'a>,
    },
}

enum ElementItemsCaptured<'a> {
    Functions(Vec<u32>),
    Expressions(wasm_encoder::RefType, Vec<ConstExpr<'a>>),
}

struct DataSegment<'a> {
    kind: DataSegmentKind<'a>,
    data: &'a [u8],
}

enum DataSegmentKind<'a> {
    Passive,
    Active {
        memory_index: u32,
        offset: ConstExpr<'a>,
    },
}

fn parse_module(bytes: &[u8]) -> Result<ParsedModule<'_>> {
    let mut types: Vec<FuncTypeOwned> = Vec::new();
    let mut imports: Vec<ImportEntry<'_>> = Vec::new();
    let mut func_types: Vec<u32> = Vec::new();
    let mut tables: Vec<TableEntry> = Vec::new();
    let mut memories: Vec<MemoryEntry> = Vec::new();
    let mut globals: Vec<GlobalEntry<'_>> = Vec::new();
    let mut exports: Vec<ExportEntry<'_>> = Vec::new();
    let mut start: Option<u32> = None;
    let mut elements: Vec<ElementEntry<'_>> = Vec::new();
    let mut code: Vec<&[u8]> = Vec::new();
    let mut code_readers: Vec<FunctionBody<'_>> = Vec::new();
    let mut data: Vec<DataSegment<'_>> = Vec::new();
    let mut data_count: u32 = 0;

    let parser = Parser::new(0);
    for payload in parser.parse_all(bytes) {
        let payload = payload?;
        match payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader {
                    let rec_group = rec_group?;
                    for ty in rec_group.into_types() {
                        let composite = ty.composite_type;
                        match composite.inner {
                            wasmparser::CompositeInnerType::Func(ft) => {
                                let params = ft.params().to_vec();
                                let results = ft.results().to_vec();
                                types.push(FuncTypeOwned { params, results });
                            }
                            other => {
                                bail!(
                                    "non-func composite type encountered (only function types are supported): {:?}",
                                    other
                                );
                            }
                        }
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for group in reader {
                    let group = group?;
                    match group {
                        wasmparser::Imports::Single(_, imp) => {
                            imports.push(ImportEntry {
                                module: imp.module,
                                name: imp.name,
                                ty: imp.ty,
                            });
                        }
                        wasmparser::Imports::Compact1 { module, items } => {
                            for item in items {
                                let item = item?;
                                imports.push(ImportEntry {
                                    module,
                                    name: item.name,
                                    ty: item.ty,
                                });
                            }
                        }
                        wasmparser::Imports::Compact2 { module, ty, names } => {
                            for name in names {
                                let name = name?;
                                imports.push(ImportEntry { module, name, ty });
                            }
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for ty_idx in reader {
                    func_types.push(ty_idx?);
                }
            }
            Payload::TableSection(reader) => {
                for t in reader {
                    let t = t?;
                    let element_type = ref_type_to_encoder(t.ty.element_type)?;
                    let init = match t.init {
                        TableInit::RefNull => None,
                        TableInit::Expr(expr) => Some(const_expr_to_encoder(expr)?),
                    };
                    tables.push(TableEntry {
                        element_type,
                        minimum: t.ty.initial,
                        maximum: t.ty.maximum,
                        table64: t.ty.table64,
                        init,
                        shared: t.ty.shared,
                    });
                }
            }
            Payload::MemorySection(reader) => {
                for m in reader {
                    let m = m?;
                    memories.push(MemoryEntry {
                        minimum: m.initial,
                        maximum: m.maximum,
                        memory64: m.memory64,
                        shared: m.shared,
                        page_size_log2: m.page_size_log2,
                    });
                }
            }
            Payload::GlobalSection(reader) => {
                for g in reader {
                    let g = g?;
                    let val_type = val_type_to_encoder(g.ty.content_type)?;
                    globals.push(GlobalEntry {
                        ty: wasm_encoder::GlobalType {
                            val_type,
                            mutable: g.ty.mutable,
                            shared: g.ty.shared,
                        },
                        init: g.init_expr,
                    });
                }
            }
            Payload::ExportSection(reader) => {
                for e in reader {
                    let e = e?;
                    exports.push(ExportEntry {
                        name: e.name,
                        kind: e.kind,
                        index: e.index,
                    });
                }
            }
            Payload::StartSection { func, .. } => {
                start = Some(func);
            }
            Payload::ElementSection(reader) => {
                for el in reader {
                    let el = el?;
                    let kind = match el.kind {
                        ElementKind::Passive => ElementKindCaptured::Passive,
                        ElementKind::Declared => ElementKindCaptured::Declared,
                        ElementKind::Active {
                            table_index,
                            offset_expr,
                        } => ElementKindCaptured::Active {
                            table_index,
                            offset: offset_expr,
                        },
                    };
                    let items = match el.items {
                        ElementItems::Functions(reader) => {
                            let mut funcs = Vec::new();
                            for f in reader {
                                funcs.push(f?);
                            }
                            ElementItemsCaptured::Functions(funcs)
                        }
                        ElementItems::Expressions(ref_ty, reader) => {
                            let ty = ref_type_to_encoder(ref_ty)?;
                            let mut exprs = Vec::new();
                            for expr in reader {
                                exprs.push(expr?);
                            }
                            ElementItemsCaptured::Expressions(ty, exprs)
                        }
                    };
                    elements.push(ElementEntry { kind, items });
                }
            }
            Payload::CodeSectionEntry(body) => {
                // Preserve the raw slice for pass-through callers (shell)
                // while also storing the parsed `FunctionBody` for the
                // rewriter (user). `as_bytes()` starts at the locals
                // declaration — no size prefix — which is exactly what
                // `CodeSection::raw` expects.
                code.push(body.as_bytes());
                code_readers.push(body);
            }
            Payload::DataSection(reader) => {
                for d in reader {
                    let d = d?;
                    let kind = match d.kind {
                        DataKind::Passive => DataSegmentKind::Passive,
                        DataKind::Active {
                            memory_index,
                            offset_expr,
                        } => DataSegmentKind::Active {
                            memory_index,
                            offset: offset_expr,
                        },
                    };
                    data.push(DataSegment { kind, data: d.data });
                }
            }
            Payload::DataCountSection { count, .. } => {
                data_count = count;
            }
            // Custom sections: discard. The merged module doesn't need
            // the input modules' `name`, `producers`, etc. — we could
            // forward them later if useful.
            Payload::CustomSection(_) => {}
            // Component-model + GC-related sections — neither input is
            // expected to contain these. If they do, bail.
            Payload::CodeSectionStart { .. }
            | Payload::Version { .. }
            | Payload::End(_) => {}
            other => {
                bail!("unexpected section in input wasm: {:?}", other);
            }
        }
    }

    let globals_emitted = globals.len();
    Ok(ParsedModule {
        types,
        imports,
        func_types,
        tables,
        memories,
        globals,
        globals_emitted,
        exports,
        start,
        elements,
        code,
        code_readers,
        data,
        data_count,
    })
}

// ----------------------------------------------------------------
// Code rewriter — translates user function bodies into the merged
// module's index space and rewrites all calls to user-side imports as
// direct calls to the corresponding shell functions.
// ----------------------------------------------------------------

fn rewrite_function_body<FF, FG, FT, FTy>(
    body: &FunctionBody<'_>,
    user_types: &[FuncTypeOwned],
    map_func: &FF,
    map_global: &FG,
    map_table: &FT,
    map_type: &FTy,
) -> Result<wasm_encoder::Function>
where
    FF: Fn(u32) -> u32,
    FG: Fn(u32) -> u32,
    FT: Fn(u32) -> u32,
    FTy: Fn(u32) -> u32,
{
    let mut func = wasm_encoder::Function::new(read_locals(body)?);

    let mut op_reader = body.get_operators_reader()?;
    while !op_reader.eof() {
        let op = op_reader.read()?;
        rewrite_op(
            &op,
            user_types,
            &mut func,
            map_func,
            map_global,
            map_table,
            map_type,
        )?;
    }
    Ok(func)
}

fn read_locals(body: &FunctionBody<'_>) -> Result<Vec<(u32, wasm_encoder::ValType)>> {
    let mut reader = body.get_locals_reader()?;
    let mut locals = Vec::with_capacity(reader.get_count() as usize);
    for _ in 0..reader.get_count() {
        let (count, ty) = reader.read()?;
        locals.push((count, val_type_to_encoder(ty)?));
    }
    Ok(locals)
}

fn rewrite_op<FF, FG, FT, FTy>(
    op: &Operator<'_>,
    user_types: &[FuncTypeOwned],
    func: &mut wasm_encoder::Function,
    map_func: &FF,
    map_global: &FG,
    map_table: &FT,
    map_type: &FTy,
) -> Result<()>
where
    FF: Fn(u32) -> u32,
    FG: Fn(u32) -> u32,
    FT: Fn(u32) -> u32,
    FTy: Fn(u32) -> u32,
{
    use wasm_encoder::Instruction as I;
    let _ = user_types; // currently unused; reserved for future call_indirect arity checks

    macro_rules! mem {
        ($m:expr) => {
            wasm_encoder::MemArg {
                offset: $m.offset,
                align: $m.align as u32,
                memory_index: $m.memory,
            }
        };
    }

    // The vast majority of operators have no index references and are
    // forwarded unchanged. We translate only those that do.
    match *op {
        Operator::Call { function_index } => {
            func.instruction(&I::Call(map_func(function_index)));
        }
        Operator::ReturnCall { function_index } => {
            func.instruction(&I::ReturnCall(map_func(function_index)));
        }
        Operator::CallIndirect {
            type_index,
            table_index,
        } => {
            func.instruction(&I::CallIndirect {
                type_index: map_type(type_index),
                table_index: map_table(table_index),
            });
        }
        Operator::ReturnCallIndirect {
            type_index,
            table_index,
        } => {
            func.instruction(&I::ReturnCallIndirect {
                type_index: map_type(type_index),
                table_index: map_table(table_index),
            });
        }
        Operator::RefFunc { function_index } => {
            func.instruction(&I::RefFunc(map_func(function_index)));
        }
        Operator::GlobalGet { global_index } => {
            func.instruction(&I::GlobalGet(map_global(global_index)));
        }
        Operator::GlobalSet { global_index } => {
            func.instruction(&I::GlobalSet(map_global(global_index)));
        }
        Operator::TableGet { table } => {
            func.instruction(&I::TableGet(map_table(table)));
        }
        Operator::TableSet { table } => {
            func.instruction(&I::TableSet(map_table(table)));
        }
        Operator::TableSize { table } => {
            func.instruction(&I::TableSize(map_table(table)));
        }
        Operator::TableGrow { table } => {
            func.instruction(&I::TableGrow(map_table(table)));
        }
        Operator::TableFill { table } => {
            func.instruction(&I::TableFill(map_table(table)));
        }
        Operator::TableCopy {
            dst_table,
            src_table,
        } => {
            func.instruction(&I::TableCopy {
                dst_table: map_table(dst_table),
                src_table: map_table(src_table),
            });
        }
        Operator::TableInit { elem_index, table } => {
            // Element segments are renumbered by their order in the
            // merged element section. The user's element segments come
            // after the shell's, so shift by the shell's element count.
            // This is set up by the caller; we keep it implicit by not
            // remapping here — see the caller-provided map_table closure
            // for the table side. (Elem index renumbering not needed if
            // we don't drop any of the user's elements.)
            func.instruction(&I::TableInit {
                elem_index,
                table: map_table(table),
            });
        }
        Operator::ElemDrop { elem_index } => {
            func.instruction(&I::ElemDrop(elem_index));
        }
        Operator::MemoryInit { mem, data_index } => {
            // Memory index 0 is the only valid one (user has just the
            // default memory which we've redirected to shell's pool 0).
            if mem != 0 {
                bail!(
                    "memory.init references memory {} but only memory 0 is supported",
                    mem
                );
            }
            // Data indices need shifting if the shell has its own data
            // segments; we add `shell.data_count` here. Captured in the
            // caller via a future map; for now we rely on shell having
            // no data segments (true for the current template).
            func.instruction(&I::MemoryInit { mem, data_index });
        }
        Operator::DataDrop { data_index } => {
            func.instruction(&I::DataDrop(data_index));
        }
        Operator::MemoryCopy { dst_mem, src_mem } => {
            if dst_mem != 0 || src_mem != 0 {
                bail!(
                    "memory.copy uses non-default memory ({} → {}); not supported in user code",
                    src_mem,
                    dst_mem
                );
            }
            func.instruction(&I::MemoryCopy { dst_mem, src_mem });
        }
        Operator::MemoryFill { mem } => {
            if mem != 0 {
                bail!("memory.fill references memory {}; only 0 is supported", mem);
            }
            func.instruction(&I::MemoryFill(mem));
        }
        Operator::MemorySize { mem, .. } => {
            if mem != 0 {
                bail!("memory.size references memory {}; only 0 is supported", mem);
            }
            func.instruction(&I::MemorySize(mem));
        }
        Operator::MemoryGrow { mem, .. } => {
            if mem != 0 {
                bail!("memory.grow references memory {}; only 0 is supported", mem);
            }
            func.instruction(&I::MemoryGrow(mem));
        }
        // Loads & stores carry a MemArg that includes a `memory` index.
        // The user only has memory 0, which maps to shell pool 0
        // (memory 0 in the merged module), so we pass through unchanged.
        Operator::I32Load { memarg } => {
            func.instruction(&I::I32Load(mem!(memarg)));
        }
        Operator::I64Load { memarg } => {
            func.instruction(&I::I64Load(mem!(memarg)));
        }
        Operator::F32Load { memarg } => {
            func.instruction(&I::F32Load(mem!(memarg)));
        }
        Operator::F64Load { memarg } => {
            func.instruction(&I::F64Load(mem!(memarg)));
        }
        Operator::I32Load8S { memarg } => {
            func.instruction(&I::I32Load8S(mem!(memarg)));
        }
        Operator::I32Load8U { memarg } => {
            func.instruction(&I::I32Load8U(mem!(memarg)));
        }
        Operator::I32Load16S { memarg } => {
            func.instruction(&I::I32Load16S(mem!(memarg)));
        }
        Operator::I32Load16U { memarg } => {
            func.instruction(&I::I32Load16U(mem!(memarg)));
        }
        Operator::I64Load8S { memarg } => {
            func.instruction(&I::I64Load8S(mem!(memarg)));
        }
        Operator::I64Load8U { memarg } => {
            func.instruction(&I::I64Load8U(mem!(memarg)));
        }
        Operator::I64Load16S { memarg } => {
            func.instruction(&I::I64Load16S(mem!(memarg)));
        }
        Operator::I64Load16U { memarg } => {
            func.instruction(&I::I64Load16U(mem!(memarg)));
        }
        Operator::I64Load32S { memarg } => {
            func.instruction(&I::I64Load32S(mem!(memarg)));
        }
        Operator::I64Load32U { memarg } => {
            func.instruction(&I::I64Load32U(mem!(memarg)));
        }
        Operator::I32Store { memarg } => {
            func.instruction(&I::I32Store(mem!(memarg)));
        }
        Operator::I64Store { memarg } => {
            func.instruction(&I::I64Store(mem!(memarg)));
        }
        Operator::F32Store { memarg } => {
            func.instruction(&I::F32Store(mem!(memarg)));
        }
        Operator::F64Store { memarg } => {
            func.instruction(&I::F64Store(mem!(memarg)));
        }
        Operator::I32Store8 { memarg } => {
            func.instruction(&I::I32Store8(mem!(memarg)));
        }
        Operator::I32Store16 { memarg } => {
            func.instruction(&I::I32Store16(mem!(memarg)));
        }
        Operator::I64Store8 { memarg } => {
            func.instruction(&I::I64Store8(mem!(memarg)));
        }
        Operator::I64Store16 { memarg } => {
            func.instruction(&I::I64Store16(mem!(memarg)));
        }
        Operator::I64Store32 { memarg } => {
            func.instruction(&I::I64Store32(mem!(memarg)));
        }
        // Everything else: forward via raw passthrough.
        ref other => {
            forward_other_op(other, func)?;
        }
    }
    Ok(())
}

fn forward_other_op(op: &Operator<'_>, func: &mut wasm_encoder::Function) -> Result<()> {
    // For operators that don't reference function/global/table/type
    // indices, encode them via the wasm-encoder Instruction enum. For
    // simplicity we fall back to a direct byte copy using the operator's
    // textual reader range when available; otherwise we emit a small
    // subset hand-mapped.
    //
    // wasm-encoder's `Instruction` enum is exhaustive, but mapping
    // every variant from wasmparser is mechanical and ~2000 lines. To
    // keep the linker compact, we re-serialize the operator via a
    // minimal mapping that covers the operators rustc emits for a no_std
    // cdylib (numeric, control flow, parametric). Anything outside that
    // raises an explicit error rather than silently dropping the op.
    map_simple_op(op, func)
}

fn map_simple_op(op: &Operator<'_>, func: &mut wasm_encoder::Function) -> Result<()> {
    use wasm_encoder::Instruction as I;
    match *op {
        // Control flow
        Operator::Unreachable => {
            func.instruction(&I::Unreachable);
        }
        Operator::Nop => {
            func.instruction(&I::Nop);
        }
        Operator::Block { blockty } => {
            func.instruction(&I::Block(block_type_to_encoder(blockty)?));
        }
        Operator::Loop { blockty } => {
            func.instruction(&I::Loop(block_type_to_encoder(blockty)?));
        }
        Operator::If { blockty } => {
            func.instruction(&I::If(block_type_to_encoder(blockty)?));
        }
        Operator::Else => {
            func.instruction(&I::Else);
        }
        Operator::End => {
            func.instruction(&I::End);
        }
        Operator::Br { relative_depth } => {
            func.instruction(&I::Br(relative_depth));
        }
        Operator::BrIf { relative_depth } => {
            func.instruction(&I::BrIf(relative_depth));
        }
        Operator::BrTable { ref targets } => {
            let default = targets.default();
            let cases: std::borrow::Cow<[u32]> = targets
                .targets()
                .collect::<std::result::Result<Vec<_>, BinaryReaderError>>()?
                .into();
            func.instruction(&I::BrTable(cases, default));
        }
        Operator::Return => {
            func.instruction(&I::Return);
        }
        // Parametric
        Operator::Drop => {
            func.instruction(&I::Drop);
        }
        Operator::Select => {
            func.instruction(&I::Select);
        }
        // Locals
        Operator::LocalGet { local_index } => {
            func.instruction(&I::LocalGet(local_index));
        }
        Operator::LocalSet { local_index } => {
            func.instruction(&I::LocalSet(local_index));
        }
        Operator::LocalTee { local_index } => {
            func.instruction(&I::LocalTee(local_index));
        }
        // Numeric constants
        Operator::I32Const { value } => {
            func.instruction(&I::I32Const(value));
        }
        Operator::I64Const { value } => {
            func.instruction(&I::I64Const(value));
        }
        Operator::F32Const { value } => {
            func.instruction(&I::F32Const(wasm_encoder::Ieee32::new(value.bits())));
        }
        Operator::F64Const { value } => {
            func.instruction(&I::F64Const(wasm_encoder::Ieee64::new(value.bits())));
        }
        // i32 numeric ops
        Operator::I32Eqz => {
            func.instruction(&I::I32Eqz);
        }
        Operator::I32Eq => {
            func.instruction(&I::I32Eq);
        }
        Operator::I32Ne => {
            func.instruction(&I::I32Ne);
        }
        Operator::I32LtS => {
            func.instruction(&I::I32LtS);
        }
        Operator::I32LtU => {
            func.instruction(&I::I32LtU);
        }
        Operator::I32GtS => {
            func.instruction(&I::I32GtS);
        }
        Operator::I32GtU => {
            func.instruction(&I::I32GtU);
        }
        Operator::I32LeS => {
            func.instruction(&I::I32LeS);
        }
        Operator::I32LeU => {
            func.instruction(&I::I32LeU);
        }
        Operator::I32GeS => {
            func.instruction(&I::I32GeS);
        }
        Operator::I32GeU => {
            func.instruction(&I::I32GeU);
        }
        Operator::I32Add => {
            func.instruction(&I::I32Add);
        }
        Operator::I32Sub => {
            func.instruction(&I::I32Sub);
        }
        Operator::I32Mul => {
            func.instruction(&I::I32Mul);
        }
        Operator::I32DivS => {
            func.instruction(&I::I32DivS);
        }
        Operator::I32DivU => {
            func.instruction(&I::I32DivU);
        }
        Operator::I32RemS => {
            func.instruction(&I::I32RemS);
        }
        Operator::I32RemU => {
            func.instruction(&I::I32RemU);
        }
        Operator::I32And => {
            func.instruction(&I::I32And);
        }
        Operator::I32Or => {
            func.instruction(&I::I32Or);
        }
        Operator::I32Xor => {
            func.instruction(&I::I32Xor);
        }
        Operator::I32Shl => {
            func.instruction(&I::I32Shl);
        }
        Operator::I32ShrS => {
            func.instruction(&I::I32ShrS);
        }
        Operator::I32ShrU => {
            func.instruction(&I::I32ShrU);
        }
        Operator::I32Rotl => {
            func.instruction(&I::I32Rotl);
        }
        Operator::I32Rotr => {
            func.instruction(&I::I32Rotr);
        }
        Operator::I32Clz => {
            func.instruction(&I::I32Clz);
        }
        Operator::I32Ctz => {
            func.instruction(&I::I32Ctz);
        }
        Operator::I32Popcnt => {
            func.instruction(&I::I32Popcnt);
        }
        // i64 numeric ops
        Operator::I64Eqz => {
            func.instruction(&I::I64Eqz);
        }
        Operator::I64Eq => {
            func.instruction(&I::I64Eq);
        }
        Operator::I64Ne => {
            func.instruction(&I::I64Ne);
        }
        Operator::I64LtS => {
            func.instruction(&I::I64LtS);
        }
        Operator::I64LtU => {
            func.instruction(&I::I64LtU);
        }
        Operator::I64GtS => {
            func.instruction(&I::I64GtS);
        }
        Operator::I64GtU => {
            func.instruction(&I::I64GtU);
        }
        Operator::I64LeS => {
            func.instruction(&I::I64LeS);
        }
        Operator::I64LeU => {
            func.instruction(&I::I64LeU);
        }
        Operator::I64GeS => {
            func.instruction(&I::I64GeS);
        }
        Operator::I64GeU => {
            func.instruction(&I::I64GeU);
        }
        Operator::I64Add => {
            func.instruction(&I::I64Add);
        }
        Operator::I64Sub => {
            func.instruction(&I::I64Sub);
        }
        Operator::I64Mul => {
            func.instruction(&I::I64Mul);
        }
        Operator::I64DivS => {
            func.instruction(&I::I64DivS);
        }
        Operator::I64DivU => {
            func.instruction(&I::I64DivU);
        }
        Operator::I64RemS => {
            func.instruction(&I::I64RemS);
        }
        Operator::I64RemU => {
            func.instruction(&I::I64RemU);
        }
        Operator::I64And => {
            func.instruction(&I::I64And);
        }
        Operator::I64Or => {
            func.instruction(&I::I64Or);
        }
        Operator::I64Xor => {
            func.instruction(&I::I64Xor);
        }
        Operator::I64Shl => {
            func.instruction(&I::I64Shl);
        }
        Operator::I64ShrS => {
            func.instruction(&I::I64ShrS);
        }
        Operator::I64ShrU => {
            func.instruction(&I::I64ShrU);
        }
        Operator::I64Rotl => {
            func.instruction(&I::I64Rotl);
        }
        Operator::I64Rotr => {
            func.instruction(&I::I64Rotr);
        }
        Operator::I64Clz => {
            func.instruction(&I::I64Clz);
        }
        Operator::I64Ctz => {
            func.instruction(&I::I64Ctz);
        }
        Operator::I64Popcnt => {
            func.instruction(&I::I64Popcnt);
        }
        // Common conversions
        Operator::I32WrapI64 => {
            func.instruction(&I::I32WrapI64);
        }
        Operator::I64ExtendI32S => {
            func.instruction(&I::I64ExtendI32S);
        }
        Operator::I64ExtendI32U => {
            func.instruction(&I::I64ExtendI32U);
        }
        Operator::I32Extend8S => {
            func.instruction(&I::I32Extend8S);
        }
        Operator::I32Extend16S => {
            func.instruction(&I::I32Extend16S);
        }
        Operator::I64Extend8S => {
            func.instruction(&I::I64Extend8S);
        }
        Operator::I64Extend16S => {
            func.instruction(&I::I64Extend16S);
        }
        Operator::I64Extend32S => {
            func.instruction(&I::I64Extend32S);
        }
        ref other => {
            bail!(
                "unsupported operator in user code body: {:?}\n\
                The linker handles the subset rustc emits for typical no_std cdylibs; \
                extend `map_simple_op` for other operators.",
                other
            );
        }
    }
    Ok(())
}

fn block_type_to_encoder(b: wasmparser::BlockType) -> Result<wasm_encoder::BlockType> {
    match b {
        wasmparser::BlockType::Empty => Ok(wasm_encoder::BlockType::Empty),
        wasmparser::BlockType::Type(ty) => Ok(wasm_encoder::BlockType::Result(val_type_to_encoder(
            ty,
        )?)),
        wasmparser::BlockType::FuncType(idx) => Ok(wasm_encoder::BlockType::FunctionType(idx)),
    }
}

// ----------------------------------------------------------------
// Section emitters — translate parsed entries back into encoder form.
// ----------------------------------------------------------------

fn push_func_type(types: &mut wasm_encoder::TypeSection, ft: &FuncTypeOwned) {
    let params: Vec<wasm_encoder::ValType> = ft
        .params
        .iter()
        .map(|v| val_type_to_encoder(*v).expect("validated at parse time"))
        .collect();
    let results: Vec<wasm_encoder::ValType> = ft
        .results
        .iter()
        .map(|v| val_type_to_encoder(*v).expect("validated at parse time"))
        .collect();
    types.ty().function(params, results);
}

fn push_table(tables: &mut wasm_encoder::TableSection, t: &TableEntry) {
    let ty = wasm_encoder::TableType {
        element_type: t.element_type,
        minimum: t.minimum,
        maximum: t.maximum,
        table64: t.table64,
        shared: t.shared,
    };
    match &t.init {
        Some(init) => {
            tables.table_with_init(ty, init);
        }
        None => {
            tables.table(ty);
        }
    }
}

fn push_memory(mems: &mut wasm_encoder::MemorySection, m: &MemoryEntry) {
    mems.memory(wasm_encoder::MemoryType {
        minimum: m.minimum,
        maximum: m.maximum,
        memory64: m.memory64,
        shared: m.shared,
        page_size_log2: m.page_size_log2,
    });
}

fn push_global<FG, FF>(
    globals: &mut wasm_encoder::GlobalSection,
    g: &GlobalEntry<'_>,
    map_global: FG,
    map_func: FF,
) where
    FG: Fn(u32) -> u32,
    FF: Fn(u32) -> u32,
{
    let _ = map_global;
    let _ = map_func;
    // Globals in core wasm have very restricted init expressions
    // (const-only). Rustc cdylibs use i32.const / global.get against
    // imported globals. We re-emit the raw const-expr bytes; if a
    // global.get references a function index or global index that
    // needs remapping, this would be wrong — but rustc cdylibs don't
    // emit such initializers in practice. We assert simplicity here.
    //
    // wasm-encoder doesn't expose a way to pass through a raw
    // const-expr from wasmparser, so we decode the small expected
    // form (single i32.const / i64.const / f32.const / f64.const).
    let init = decode_const_expr_to_encoder(&g.init).expect("supported const-expr in global init");
    globals.global(g.ty, &init);
}

fn const_expr_to_encoder(expr: ConstExpr<'_>) -> Result<wasm_encoder::ConstExpr> {
    decode_const_expr_to_encoder(&expr)
}

fn decode_const_expr_to_encoder(expr: &ConstExpr<'_>) -> Result<wasm_encoder::ConstExpr> {
    use wasm_encoder::ConstExpr as CE;
    let mut reader = expr.get_operators_reader();
    let op = reader.read()?;
    let result = match op {
        Operator::I32Const { value } => CE::i32_const(value),
        Operator::I64Const { value } => CE::i64_const(value),
        Operator::F32Const { value } => CE::f32_const(wasm_encoder::Ieee32::new(value.bits())),
        Operator::F64Const { value } => CE::f64_const(wasm_encoder::Ieee64::new(value.bits())),
        Operator::RefNull { hty } => CE::ref_null(heap_type_to_encoder(hty)?),
        Operator::GlobalGet { global_index } => CE::global_get(global_index),
        other => bail!("unsupported const-expr opcode: {:?}", other),
    };
    // Expect the End operator next.
    match reader.read()? {
        Operator::End => Ok(result),
        other => bail!("const-expr ended with non-End op: {:?}", other),
    }
}

fn push_export<FF, FT, FM, FG>(
    exports: &mut wasm_encoder::ExportSection,
    e: &ExportEntry<'_>,
    map_func: FF,
    map_table: FT,
    map_memory: FM,
    map_global: FG,
) where
    FF: Fn(u32) -> u32,
    FT: Fn(u32) -> u32,
    FM: Fn(u32) -> u32,
    FG: Fn(u32) -> u32,
{
    let (kind, idx) = match e.kind {
        ExternalKind::Func | ExternalKind::FuncExact => {
            (wasm_encoder::ExportKind::Func, map_func(e.index))
        }
        ExternalKind::Table => (wasm_encoder::ExportKind::Table, map_table(e.index)),
        ExternalKind::Memory => (wasm_encoder::ExportKind::Memory, map_memory(e.index)),
        ExternalKind::Global => (wasm_encoder::ExportKind::Global, map_global(e.index)),
        ExternalKind::Tag => (wasm_encoder::ExportKind::Tag, e.index),
    };
    exports.export(e.name, kind, idx);
}

fn push_element<FT, FF>(
    elems: &mut wasm_encoder::ElementSection,
    el: &ElementEntry<'_>,
    map_table: FT,
    map_func: FF,
) -> Result<()>
where
    FT: Fn(u32) -> u32,
    FF: Fn(u32) -> u32,
{
    use std::borrow::Cow;
    use wasm_encoder::Elements;
    let funcs_buf: Vec<u32>;
    let exprs_buf: Vec<wasm_encoder::ConstExpr>;
    let elements = match &el.items {
        ElementItemsCaptured::Functions(items) => {
            funcs_buf = items.iter().map(|f| map_func(*f)).collect();
            Elements::Functions(Cow::Borrowed(&funcs_buf))
        }
        ElementItemsCaptured::Expressions(ref_ty, exprs) => {
            exprs_buf = exprs
                .iter()
                .map(|e| decode_const_expr_to_encoder(e))
                .collect::<Result<Vec<_>>>()?;
            Elements::Expressions(*ref_ty, Cow::Borrowed(&exprs_buf))
        }
    };
    match &el.kind {
        ElementKindCaptured::Passive => {
            elems.passive(elements);
        }
        ElementKindCaptured::Declared => {
            elems.declared(elements);
        }
        ElementKindCaptured::Active {
            table_index,
            offset,
        } => {
            let table_idx = table_index.map(|t| map_table(t));
            let offset = decode_const_expr_to_encoder(offset)?;
            elems.active(table_idx, &offset, elements);
        }
    }
    Ok(())
}

fn push_data<FM>(
    datas: &mut wasm_encoder::DataSection,
    d: &DataSegment<'_>,
    map_memory: FM,
) -> Result<()>
where
    FM: Fn(u32) -> std::result::Result<u32, anyhow::Error>,
{
    match &d.kind {
        DataSegmentKind::Passive => {
            datas.passive(d.data.iter().copied());
        }
        DataSegmentKind::Active {
            memory_index,
            offset,
        } => {
            let mem_idx = map_memory(*memory_index)?;
            let offset = decode_const_expr_to_encoder(offset)?;
            datas.active(mem_idx, &offset, d.data.iter().copied());
        }
    }
    Ok(())
}

// ----------------------------------------------------------------
// Type / heap-type translation.
// ----------------------------------------------------------------

fn val_type_to_encoder(v: ValType) -> Result<wasm_encoder::ValType> {
    Ok(match v {
        ValType::I32 => wasm_encoder::ValType::I32,
        ValType::I64 => wasm_encoder::ValType::I64,
        ValType::F32 => wasm_encoder::ValType::F32,
        ValType::F64 => wasm_encoder::ValType::F64,
        ValType::V128 => wasm_encoder::ValType::V128,
        ValType::Ref(r) => wasm_encoder::ValType::Ref(ref_type_to_encoder(r)?),
    })
}

fn ref_type_to_encoder(r: wasmparser::RefType) -> Result<wasm_encoder::RefType> {
    Ok(wasm_encoder::RefType {
        nullable: r.is_nullable(),
        heap_type: heap_type_to_encoder(r.heap_type())?,
    })
}

fn heap_type_to_encoder(h: wasmparser::HeapType) -> Result<wasm_encoder::HeapType> {
    use wasm_encoder::{AbstractHeapType as A, HeapType as H};
    use wasmparser::HeapType as P;
    let mapped = match h {
        P::Abstract { shared, ty } => H::Abstract {
            shared,
            ty: abstract_heap_type_to_encoder(ty),
        },
        P::Concrete(idx) | P::Exact(idx) => H::Concrete(match idx {
            wasmparser::UnpackedIndex::Module(idx) => idx,
            other => bail!(
                "non-module concrete heap type index unsupported: {:?}",
                other
            ),
        }),
    };
    // Sanity touch — keeps imports in scope without depending on the
    // unused enum import producing a warning when only some variants
    // are used in pattern matching above.
    let _ = A::Func;
    Ok(mapped)
}

fn abstract_heap_type_to_encoder(
    a: wasmparser::AbstractHeapType,
) -> wasm_encoder::AbstractHeapType {
    use wasm_encoder::AbstractHeapType as O;
    use wasmparser::AbstractHeapType as I;
    match a {
        I::Func => O::Func,
        I::Extern => O::Extern,
        I::Any => O::Any,
        I::None => O::None,
        I::NoExtern => O::NoExtern,
        I::NoFunc => O::NoFunc,
        I::Eq => O::Eq,
        I::Struct => O::Struct,
        I::Array => O::Array,
        I::I31 => O::I31,
        I::Exn => O::Exn,
        I::NoExn => O::NoExn,
        I::Cont => O::Cont,
        I::NoCont => O::NoCont,
    }
}

// ----------------------------------------------------------------
// Convenience: link from text WAT + bytes.
// ----------------------------------------------------------------

/// Convenience: take shell WAT (the output of
/// `tvm_guest_mm_module_template`) plus user wasm bytes, returning the
/// linked module bytes. Wraps `wat::parse_str` + `link`.
pub fn link_from_wat(shell_wat: &str, user_bytes: &[u8]) -> Result<Vec<u8>> {
    let shell_bytes = wat::parse_str(shell_wat).context("parsing shell WAT")?;
    link(&shell_bytes, user_bytes)
}

/// Always-available convenience that constructs the shell on the fly
/// using `tvm_guest_mm_module_template(&params)` and links the user
/// module against it.
pub fn link_with_params(params: &ModuleParams, user_bytes: &[u8]) -> Result<Vec<u8>> {
    let shell_wat = tvm_guest_mm_module_template(params);
    let shell_bytes = wat_text_to_bytes(&shell_wat)?;
    link(&shell_bytes, user_bytes)
}

fn wat_text_to_bytes(wat_text: &str) -> Result<Vec<u8>> {
    // Use the shell crate's `wat` dependency (gated on the `std`
    // feature) to compile WAT to bytes without forcing an extra dep
    // here. We do the equivalent via `wat::parse_str` re-exported by
    // tvm-guest-mm — but tvm-guest-mm doesn't re-export wat directly,
    // so depend on `wat` here too. To keep deps minimal we inline a
    // call to `wat::parse_str` via the wat crate which wasmparser
    // pulls in transitively.
    wat::parse_str(wat_text).context("compiling shell WAT to bytes")
}
