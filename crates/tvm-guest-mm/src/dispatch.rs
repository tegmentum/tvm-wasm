//! Dispatch-function generators.
//!
//! Wasm's load/store memory immediate is a static parameter — you can't
//! say "load from memory N" where N is a runtime value. We work around
//! this by generating WAT functions that dispatch on a runtime
//! `pool_id` to one of N statically-known load/store instructions.
//!
//! The current emitter uses a chain of `if/else` comparisons over the
//! pool id. Wasmtime compiles this to a series of compare-and-branch
//! instructions; for small N (≤16) the cost is negligible compared to
//! the load/store itself, especially when amortized over a bulk-copy.

/// Emit the dispatcher for a typed load (e.g. `i32.load`, `i32.load8_u`).
pub(crate) fn emit_load_dispatcher(
    func_name: &str,
    op_name: &str,
    result_ty: &str,
    n_pools: u32,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "  (func ${name} (export \"{name}\") (param $pool i32) (param $off i32) (result {ty})\n",
        name = func_name,
        ty = result_ty
    ));
    for i in 0..n_pools {
        s.push_str(&format!(
            "    (if (result {ty}) (i32.eq (local.get $pool) (i32.const {i}))\n",
            ty = result_ty,
            i = i
        ));
        s.push_str(&format!(
            "      (then ({op} {memarg} (local.get $off)))\n",
            op = op_name,
            memarg = memory_arg(i)
        ));
        s.push_str("      (else ");
    }
    s.push_str("(unreachable)");
    for _ in 0..n_pools {
        s.push_str("))\n    ");
    }
    s.push_str(")\n");
    s
}

/// Emit the dispatcher for a typed store.
pub(crate) fn emit_store_dispatcher(
    func_name: &str,
    op_name: &str,
    value_ty: &str,
    n_pools: u32,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "  (func ${name} (export \"{name}\") (param $pool i32) (param $off i32) (param $val {ty})\n",
        name = func_name,
        ty = value_ty
    ));
    for i in 0..n_pools {
        s.push_str(&format!(
            "    (if (i32.eq (local.get $pool) (i32.const {i}))\n",
            i = i
        ));
        s.push_str(&format!(
            "      (then ({op} {memarg} (local.get $off) (local.get $val)) (return))\n",
            op = op_name,
            memarg = memory_arg(i)
        ));
        s.push_str("      (else ");
    }
    s.push_str("(unreachable)");
    for _ in 0..n_pools {
        s.push_str("))\n    ");
    }
    s.push_str(")\n");
    s
}

/// Bulk-copy dispatcher: copy from a runtime-selected pool into the
/// default memory. One dispatch decision amortized over `len` bytes —
/// the right idiom for sequential / range workloads.
pub(crate) fn emit_bulk_copy_dispatcher(n_pools: u32) -> String {
    let mut s = String::new();
    s.push_str(
        "  (func $tvm_copy_to_default (export \"tvm_copy_to_default\")\n        \
         (param $src_pool i32) (param $src_off i32) (param $dst_off i32) (param $len i32)\n",
    );
    for i in 0..n_pools {
        s.push_str(&format!(
            "    (if (i32.eq (local.get $src_pool) (i32.const {i}))\n",
            i = i
        ));
        s.push_str(&format!(
            "      (then (memory.copy 0 {i} (local.get $dst_off) (local.get $src_off) (local.get $len)) (return))\n",
            i = i
        ));
        s.push_str("      (else ");
    }
    s.push_str("(unreachable)");
    for _ in 0..n_pools {
        s.push_str("))\n    ");
    }
    s.push_str(")\n");
    s
}

/// Symmetric: copy from default memory into a runtime-selected pool.
pub(crate) fn emit_bulk_copy_from_default_dispatcher(n_pools: u32) -> String {
    let mut s = String::new();
    s.push_str(
        "  (func $tvm_copy_from_default (export \"tvm_copy_from_default\")\n        \
         (param $dst_pool i32) (param $dst_off i32) (param $src_off i32) (param $len i32)\n",
    );
    for i in 0..n_pools {
        s.push_str(&format!(
            "    (if (i32.eq (local.get $dst_pool) (i32.const {i}))\n",
            i = i
        ));
        s.push_str(&format!(
            "      (then (memory.copy {i} 0 (local.get $dst_off) (local.get $src_off) (local.get $len)) (return))\n",
            i = i
        ));
        s.push_str("      (else ");
    }
    s.push_str("(unreachable)");
    for _ in 0..n_pools {
        s.push_str("))\n    ");
    }
    s.push_str(")\n");
    s
}

fn memory_arg(memory_index: u32) -> String {
    // Multi-memory WAT syntax: the memory immediate is a bare integer
    // right after the instruction name.
    format!("{}", memory_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap(n_pools: u32, body: &str) -> String {
        let mut s = String::from("(module\n");
        for i in 0..n_pools {
            s.push_str(&format!("  (memory {} 1)\n", i));
        }
        s.push_str(body);
        s.push_str(")\n");
        s
    }

    #[test]
    fn load_dispatcher_parses() {
        let body = emit_load_dispatcher("tvm_load_u8", "i32.load8_u", "i32", 4);
        wat::parse_str(&wrap(4, &body)).expect("parse");
    }

    #[test]
    fn store_dispatcher_parses() {
        let body = emit_store_dispatcher("tvm_store_u32", "i32.store", "i32", 2);
        wat::parse_str(&wrap(2, &body)).expect("parse");
    }

    #[test]
    fn copy_dispatchers_parse() {
        let mut body = String::new();
        body.push_str(&emit_bulk_copy_dispatcher(4));
        body.push_str(&emit_bulk_copy_from_default_dispatcher(4));
        wat::parse_str(&wrap(4, &body)).expect("parse");
    }

    #[test]
    fn dispatcher_handles_single_pool() {
        let body = emit_load_dispatcher("tvm_load_u32", "i32.load", "i32", 1);
        wat::parse_str(&wrap(1, &body)).expect("parse");
    }
}
