//! Module-template generation. Builds a complete self-contained WAT
//! module string with N memory pools + dispatch helpers + a slot for
//! the user's workload body.

use crate::dispatch::{
    emit_bulk_copy_dispatcher, emit_bulk_copy_from_default_dispatcher,
    emit_indirect_load_dispatchers, emit_load_dispatcher,
    emit_specialized_copy_helpers, emit_specialized_intra_pool_copy,
    emit_specialized_simd_kernels, emit_specialized_simd_reducers,
    emit_specialized_typed_helpers, emit_store_dispatcher,
};

#[derive(Clone, Debug)]
pub struct ModuleParams {
    /// Number of memory pools. Each pool is a separate wasm memory.
    pub n_pools: u32,
    /// Initial pages per pool (each page = 64 KiB).
    pub initial_pages_per_pool: u32,
    /// Maximum pages per pool. Wasm32 caps at 65536 pages = 4 GiB.
    pub max_pages_per_pool: u32,
    /// Optional user WAT body to splice in. Should be one or more
    /// `(func ...)`, `(global ...)`, or `(export ...)` declarations.
    pub user_body: String,
}

impl Default for ModuleParams {
    fn default() -> Self {
        Self {
            n_pools: crate::DEFAULT_POOL_COUNT,
            initial_pages_per_pool: 1,
            max_pages_per_pool: 65536,
            user_body: String::new(),
        }
    }
}

/// Build a complete self-contained TVM-MM wasm module text. The returned
/// `String` is valid WAT that can be passed to `wat::parse_str` or
/// `wasmtime::Module::new`.
///
/// Generated structure:
///   - N internal memories (no imports)
///   - Dispatch helpers: `tvm_load_u8`, `tvm_load_u32`, `tvm_load_i64`,
///     `tvm_store_u8`, `tvm_store_u32`, `tvm_store_i64`
///   - User body spliced in
pub fn tvm_guest_mm_module_template(p: &ModuleParams) -> String {
    let mut s = String::new();
    s.push_str("(module\n");

    for i in 0..p.n_pools {
        s.push_str(&format!(
            "  (memory (export \"mem{}\") {} {})\n",
            i, p.initial_pages_per_pool, p.max_pages_per_pool
        ));
    }
    s.push('\n');

    // Load dispatchers.
    s.push_str(&emit_load_dispatcher("tvm_load_u8", "i32.load8_u", "i32", p.n_pools));
    s.push_str(&emit_load_dispatcher("tvm_load_u32", "i32.load", "i32", p.n_pools));
    s.push_str(&emit_load_dispatcher("tvm_load_i64", "i64.load", "i64", p.n_pools));

    // Store dispatchers.
    s.push_str(&emit_store_dispatcher("tvm_store_u8", "i32.store8", "i32", p.n_pools));
    s.push_str(&emit_store_dispatcher("tvm_store_u32", "i32.store", "i32", p.n_pools));
    s.push_str(&emit_store_dispatcher("tvm_store_i64", "i64.store", "i64", p.n_pools));

    // Bulk-copy dispatchers — one dispatch per len bytes, the right
    // idiom for sequential / range workloads.
    s.push_str(&emit_bulk_copy_dispatcher(p.n_pools));
    s.push_str(&emit_bulk_copy_from_default_dispatcher(p.n_pools));

    // Specialized direct-copy helpers — one per pool, no dispatch. The
    // workload calls these by indexing a function table at the resolved
    // pool index. Skips the dispatch chain entirely on the hot path.
    s.push_str(&emit_specialized_copy_helpers(p.n_pools));

    // Specialized typed load/store helpers per pool. Skips the BST
    // dispatch when the pool is already known.
    s.push_str(&emit_specialized_typed_helpers(p.n_pools));

    // Per-pool intra-pool copy helpers — used by the compactor.
    s.push_str(&emit_specialized_intra_pool_copy(p.n_pools));

    // Indirect-table dispatchers — `tvm_load_u8_indirect` etc. — same
    // (pool, off) signature as the BST versions but using
    // call_indirect against a function table populated with the
    // per-pool specialized helpers above. Drop-in alternative; whether
    // it beats the BST is engine-dependent (see bench).
    s.push_str(&emit_indirect_load_dispatchers(p.n_pools));

    // SIMD reduce kernels — `tvm_simd_sum_u8_p{K}` for each pool. 16
    // bytes per iter via v128.load + widening fused adds. Static
    // memory immediate so no dispatch on the hot path.
    s.push_str(&emit_specialized_simd_kernels(p.n_pools));

    // Symmetric SIMD reducer family — xor/and/or fold, count_byte,
    // popcount, find_byte, min_max. Each per-pool specialized.
    s.push_str(&emit_specialized_simd_reducers(p.n_pools));

    s.push_str("\n  ;; --- user body begins ---\n");
    s.push_str(&p.user_body);
    s.push_str("\n  ;; --- user body ends ---\n");

    s.push_str(")\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_module_with_4_pools_compiles() {
        let p = ModuleParams {
            n_pools: 4,
            initial_pages_per_pool: 1,
            max_pages_per_pool: 16,
            user_body: String::new(),
        };
        let wat = tvm_guest_mm_module_template(&p);
        let bytes = wat::parse_str(&wat).expect("must parse");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn module_with_user_body_compiles() {
        let user = r#"
            (func (export "noop"))
            (func (export "answer") (result i32) (i32.const 42))
        "#;
        let p = ModuleParams {
            n_pools: 2,
            initial_pages_per_pool: 1,
            max_pages_per_pool: 4,
            user_body: user.to_string(),
        };
        let wat = tvm_guest_mm_module_template(&p);
        let bytes = wat::parse_str(&wat).expect("must parse");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn module_with_16_pools_compiles() {
        let p = ModuleParams::default();
        let wat = tvm_guest_mm_module_template(&p);
        wat::parse_str(&wat).expect("must parse with default 16 pools");
    }
}
