//! Macro helpers for building TVM-MM (multi-memory imports) workloads
//! from WAT bodies. Until rustc gains source-level multi-memory support,
//! these macros are the most ergonomic way to write multi-region native
//! workloads.

/// Build a TVM-MM module string from a body and a list of imported memory
/// names. Wraps the body in a `(module)` declaration and emits one
/// `(import "tvm" "<name>" (memory $<name> 1))` per region name.
///
/// Use this with [`wasmtime::Module::new`] + the
/// [`build_imported_setup`](crate::build_imported_setup) helper to get an
/// instance whose memories are TVM-managed regions.
///
/// ```ignore
/// use tvm_wasmtime::tvm_mm_module;
/// use tvm_wasmtime::prelude::*;
///
/// let wat = tvm_mm_module! {
///     memories: [r0, r1],
///     body: r#"
///         (func (export "sum_in_r0") (param $ptr i32) (param $len i32) (result i64)
///           (local $i i32) (local $acc i64)
///           (block $break
///             (loop $continue
///               (br_if $break (i32.eq (local.get $i) (local.get $len)))
///               (local.set $acc
///                 (i64.add (local.get $acc)
///                          (i64.load8_u $r0
///                            (i32.add (local.get $ptr) (local.get $i)))))
///               (local.set $i (i32.add (local.get $i) (i32.const 1)))
///               (br $continue)))
///           (local.get $acc))
///     "#,
/// };
///
/// // wat is now a `&'static str` containing a complete WAT module ready
/// // for `wasmtime::Module::new(&engine, wat)`.
/// ```
#[macro_export]
macro_rules! tvm_mm_module {
    (memories: [$($name:ident),+ $(,)?], body: $body:expr $(,)?) => {{
        // Concatenate imports + body into a single &'static str at compile
        // time. Each memory gets a `(import "tvm" "<name>" (memory $<name> 1))`.
        concat!(
            "(module\n",
            $(
                "  (import \"tvm\" \"", stringify!($name), "\" ",
                "(memory $", stringify!($name), " 1))\n",
            )+
            $body,
            "\n)"
        )
    }};
}

/// Convenience: build a TVM-MM module + set up the matching host. Returns
/// `(engine, store, linker, module, region_handles)` ready for
/// `linker.instantiate`.
///
/// The `payloads` slice has one entry per declared memory; the
/// host-side bytes are pre-loaded into the corresponding region.
///
/// ```ignore
/// use tvm_wasmtime::{tvm_mm_setup, tvm_mm_module};
/// use tvm_wasmtime::prelude::*;
///
/// let wat = tvm_mm_module! {
///     memories: [r0],
///     body: r#"(func (export "sum") (result i64) (i64.const 42))"#,
/// };
/// let (engine, mut store, linker, _handles) = tvm_mm_setup(
///     wat,
///     &[b"hello"], // payload for r0
///     RegionKind::HotHeap,
/// )?;
/// let module = wasmtime::Module::new(&engine, wat)?;
/// let _instance = linker.instantiate(&mut store, &module)?;
/// ```
pub fn tvm_mm_setup(
    _wat: &str,
    payloads: &[&[u8]],
    kind: tvm_core::RegionKind,
) -> crate::imported::ImportedSetup<tvm_core::Handle> {
    crate::imported::build_imported_setup_with_data(payloads, kind, 4096)
}

#[cfg(test)]
mod tests {
    #[test]
    fn macro_emits_valid_wat_shell() {
        let wat = tvm_mm_module! {
            memories: [r0],
            body: r#"(func (export "noop"))"#,
        };
        assert!(wat.contains("(import \"tvm\" \"r0\""));
        assert!(wat.contains("(memory $r0 1)"));
        assert!(wat.contains("(func (export \"noop\"))"));
        // Roundtrip: must parse as valid WAT.
        let bytes = wat::parse_str(wat).expect("must parse");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn macro_supports_multiple_memories() {
        let wat = tvm_mm_module! {
            memories: [r0, r1, r2],
            body: r#"(func (export "noop"))"#,
        };
        assert!(wat.contains("(memory $r0 1)"));
        assert!(wat.contains("(memory $r1 1)"));
        assert!(wat.contains("(memory $r2 1)"));
    }
}
