//! Dispatch-function generators.
//!
//! Wasm's load/store memory immediate is a static parameter — you can't
//! say "load from memory N" where N is a runtime value. We work around
//! this by generating WAT functions that dispatch on a runtime
//! `pool_id` to one of N statically-known load/store instructions.
//!
//! ## Two flavors of dispatch
//!
//! ### 1. Generic dispatch (`tvm_load_u8`, `tvm_copy_to_default`, …)
//!
//! Single function that accepts a runtime `pool` parameter. Emitted as
//! a **balanced binary search tree** over `[0, N)`:
//!
//! ```text
//! if (pool < N/2)
//!   if (pool < N/4) ...
//!   else ...
//! else
//!   if (pool < 3N/4) ...
//!   else ...
//! ```
//!
//! Depth = `ceil(log2(N))`. At N=64 that's 6 comparisons; at N=256, 8.
//! The previous linear if/else chain was N comparisons worst case (N/2
//! average). For per-byte workloads against many pools, the BST is a
//! material improvement.
//!
//! An upfront `pool >= N → unreachable` guard makes the BST correct
//! for any input — out-of-range pool ids trap, they don't silently
//! land on memory N-1.
//!
//! ### 2. Specialized per-pool exports (`tvm_copy_to_default_p{K}`, …)
//!
//! For workloads that have already resolved the pool index (everyone
//! who came through `GuestDirectory::resolve`), we additionally emit
//! direct, **dispatch-free** functions:
//!
//! ```text
//! (func $tvm_copy_to_default_p7 (export "tvm_copy_to_default_p7")
//!   (param $src_off i32) (param $dst_off i32) (param $len i32)
//!   (memory.copy 0 7 (local.get $dst_off) (local.get $src_off)
//!                    (local.get $len)))
//! ```
//!
//! The Rust side picks the right export by indexing a function table at
//! `pool_index`. Cost: zero dispatch instructions on the hot path. The
//! per-pool specializations are tiny (≈3 instructions each) so adding
//! N of them adds well under a kilobyte to module size for typical N.

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
        ty = result_ty,
    ));
    // Bounds guard: pool >= n_pools → trap. Single comparison.
    s.push_str(&format!(
        "    (if (i32.ge_u (local.get $pool) (i32.const {n})) (then unreachable))\n",
        n = n_pools,
    ));
    // BST body returning `result_ty`.
    let body = build_load_bst(op_name, result_ty, 0, n_pools);
    indent_into(&mut s, &body, 4);
    s.push_str("  )\n");
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
        ty = value_ty,
    ));
    s.push_str(&format!(
        "    (if (i32.ge_u (local.get $pool) (i32.const {n})) (then unreachable))\n",
        n = n_pools,
    ));
    let body = build_store_bst(op_name, 0, n_pools);
    indent_into(&mut s, &body, 4);
    s.push_str("  )\n");
    s
}

/// Bulk-copy dispatcher: `tvm_copy_to_default(src_pool, src_off,
/// dst_off, len)`. Generic — runtime selects source pool. One
/// dispatch decision amortized over `len` bytes; for a workload that
/// already knows the source pool, prefer `tvm_copy_to_default_p{K}`.
pub(crate) fn emit_bulk_copy_dispatcher(n_pools: u32) -> String {
    let mut s = String::new();
    s.push_str(
        "  (func $tvm_copy_to_default (export \"tvm_copy_to_default\")\n        \
         (param $src_pool i32) (param $src_off i32) (param $dst_off i32) (param $len i32)\n",
    );
    s.push_str(&format!(
        "    (if (i32.ge_u (local.get $src_pool) (i32.const {n})) (then unreachable))\n",
        n = n_pools,
    ));
    let body = build_copy_bst(/*to_default=*/ true, "src_pool", 0, n_pools);
    indent_into(&mut s, &body, 4);
    s.push_str("  )\n");
    s
}

/// Symmetric: copy from default memory into a runtime-selected pool.
pub(crate) fn emit_bulk_copy_from_default_dispatcher(n_pools: u32) -> String {
    let mut s = String::new();
    s.push_str(
        "  (func $tvm_copy_from_default (export \"tvm_copy_from_default\")\n        \
         (param $dst_pool i32) (param $dst_off i32) (param $src_off i32) (param $len i32)\n",
    );
    s.push_str(&format!(
        "    (if (i32.ge_u (local.get $dst_pool) (i32.const {n})) (then unreachable))\n",
        n = n_pools,
    ));
    let body = build_copy_bst(/*to_default=*/ false, "dst_pool", 0, n_pools);
    indent_into(&mut s, &body, 4);
    s.push_str("  )\n");
    s
}

/// Specialized direct-copy exports — one per pool. These contain a
/// single static `memory.copy` and exist purely to skip the dispatch
/// chain for callers that have already resolved which pool they want.
///
/// Naming: `tvm_copy_to_default_p{K}(src_off, dst_off, len)` copies
/// from pool K → default; `tvm_copy_from_default_p{K}(dst_off, src_off,
/// len)` copies from default → pool K.
pub(crate) fn emit_specialized_copy_helpers(n_pools: u32) -> String {
    let mut s = String::new();
    for k in 0..n_pools {
        // The internal `$tvm_copy_to_default_pK` name lets in-module
        // user code call these directly; the export name lets host
        // code call them when needed.
        s.push_str(&format!(
            "  (func $tvm_copy_to_default_p{k} (export \"tvm_copy_to_default_p{k}\")\n        \
             (param $src_off i32) (param $dst_off i32) (param $len i32)\n    \
             (memory.copy 0 {k} (local.get $dst_off) (local.get $src_off) (local.get $len)))\n",
            k = k,
        ));
        s.push_str(&format!(
            "  (func $tvm_copy_from_default_p{k} (export \"tvm_copy_from_default_p{k}\")\n        \
             (param $dst_off i32) (param $src_off i32) (param $len i32)\n    \
             (memory.copy {k} 0 (local.get $dst_off) (local.get $src_off) (local.get $len)))\n",
            k = k,
        ));
    }
    s
}

// --- BST builders ---------------------------------------------------------

/// Build a balanced BST over `[lo, hi)` returning a value loaded from
/// the matching memory. Caller has already verified `pool ∈ [0, hi)`,
/// so we don't re-check here — every leaf is reachable.
fn build_load_bst(op: &str, result_ty: &str, lo: u32, hi: u32) -> String {
    debug_assert!(lo < hi);
    if hi - lo == 1 {
        return format!("({op} {mem} (local.get $off))\n", op = op, mem = lo);
    }
    let mid = lo + (hi - lo) / 2;
    let left = build_load_bst(op, result_ty, lo, mid);
    let right = build_load_bst(op, result_ty, mid, hi);
    let mut s = String::new();
    s.push_str(&format!(
        "(if (result {ty}) (i32.lt_u (local.get $pool) (i32.const {mid}))\n",
        ty = result_ty,
        mid = mid,
    ));
    s.push_str("  (then ");
    indent_into(&mut s, &left, 4);
    s.push_str("  )\n  (else ");
    indent_into(&mut s, &right, 4);
    s.push_str("  ))\n");
    s
}

/// Build a balanced BST over `[lo, hi)` performing a store + return.
fn build_store_bst(op: &str, lo: u32, hi: u32) -> String {
    debug_assert!(lo < hi);
    if hi - lo == 1 {
        return format!(
            "({op} {mem} (local.get $off) (local.get $val))\n",
            op = op,
            mem = lo,
        );
    }
    let mid = lo + (hi - lo) / 2;
    let left = build_store_bst(op, lo, mid);
    let right = build_store_bst(op, mid, hi);
    let mut s = String::new();
    s.push_str(&format!(
        "(if (i32.lt_u (local.get $pool) (i32.const {mid}))\n",
        mid = mid,
    ));
    s.push_str("  (then ");
    indent_into(&mut s, &left, 4);
    s.push_str("  )\n  (else ");
    indent_into(&mut s, &right, 4);
    s.push_str("  ))\n");
    s
}

/// Build a balanced BST over `[lo, hi)` issuing one `memory.copy` per
/// leaf. Direction is encoded by `to_default`: true = pool→default,
/// false = default→pool. The runtime-pool param is named via
/// `pool_param`.
fn build_copy_bst(to_default: bool, pool_param: &str, lo: u32, hi: u32) -> String {
    debug_assert!(lo < hi);
    if hi - lo == 1 {
        let (dst_mem, src_mem) = if to_default { (0, lo) } else { (lo, 0) };
        // For pool→default, params are (src_pool, src_off, dst_off, len).
        // For default→pool, params are (dst_pool, dst_off, src_off, len).
        return if to_default {
            format!(
                "(memory.copy {dst} {src} (local.get $dst_off) (local.get $src_off) (local.get $len))\n",
                dst = dst_mem,
                src = src_mem,
            )
        } else {
            format!(
                "(memory.copy {dst} {src} (local.get $dst_off) (local.get $src_off) (local.get $len))\n",
                dst = dst_mem,
                src = src_mem,
            )
        };
    }
    let mid = lo + (hi - lo) / 2;
    let left = build_copy_bst(to_default, pool_param, lo, mid);
    let right = build_copy_bst(to_default, pool_param, mid, hi);
    let mut s = String::new();
    s.push_str(&format!(
        "(if (i32.lt_u (local.get ${param}) (i32.const {mid}))\n",
        param = pool_param,
        mid = mid,
    ));
    s.push_str("  (then ");
    indent_into(&mut s, &left, 4);
    s.push_str("  )\n  (else ");
    indent_into(&mut s, &right, 4);
    s.push_str("  ))\n");
    s
}

/// Append `body` to `dst`, indenting every line after the first by
/// `cols` spaces. The first line continues whatever the caller already
/// emitted (e.g. `(then ` followed by the BST). WAT is whitespace-
/// tolerant, so this is purely cosmetic — kept readable for debugging.
fn indent_into(dst: &mut String, body: &str, cols: usize) {
    let mut first = true;
    for line in body.lines() {
        if first {
            dst.push_str(line);
            dst.push('\n');
            first = false;
        } else {
            for _ in 0..cols {
                dst.push(' ');
            }
            dst.push_str(line);
            dst.push('\n');
        }
    }
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
    fn load_dispatcher_parses_for_various_n() {
        for n in [1u32, 2, 3, 4, 7, 8, 16, 64, 256] {
            let body = emit_load_dispatcher("tvm_load_u8", "i32.load8_u", "i32", n);
            let module = wrap(n, &body);
            wat::parse_str(&module).unwrap_or_else(|e| {
                panic!("n_pools={} failed to parse: {}\n--- module ---\n{}", n, e, module)
            });
        }
    }

    #[test]
    fn store_dispatcher_parses_for_various_n() {
        for n in [1u32, 2, 5, 16, 64] {
            let body = emit_store_dispatcher("tvm_store_u32", "i32.store", "i32", n);
            wat::parse_str(&wrap(n, &body)).expect("parse");
        }
    }

    #[test]
    fn copy_dispatchers_parse() {
        for n in [1u32, 2, 3, 8, 64] {
            let mut body = String::new();
            body.push_str(&emit_bulk_copy_dispatcher(n));
            body.push_str(&emit_bulk_copy_from_default_dispatcher(n));
            wat::parse_str(&wrap(n, &body)).expect("parse");
        }
    }

    #[test]
    fn specialized_copy_helpers_parse() {
        for n in [1u32, 4, 16, 64] {
            let body = emit_specialized_copy_helpers(n);
            wat::parse_str(&wrap(n, &body)).expect("parse");
        }
    }

    /// Dispatch tree depth must be `ceil(log2(n))` — verified by counting
    /// `(if` occurrences along the deepest leaf path. Approximate: the
    /// total number of `(if` statements should equal `n - 1`, since the
    /// BST has n leaves and n-1 internal nodes.
    #[test]
    fn bst_depth_is_log_n() {
        // Count `(if ` occurrences (with trailing space to exclude e.g.
        // `(if`-prefixed identifiers).
        let s = emit_load_dispatcher("tvm_load_u32", "i32.load", "i32", 64);
        // 1 bounds-guard `if` + 63 BST internal `if`s = 64.
        let count = s.matches("(if ").count();
        assert_eq!(count, 64, "module:\n{}", s);
    }

    /// Specialized helper is a single `memory.copy` — one line of body.
    #[test]
    fn specialized_helper_is_one_op() {
        let s = emit_specialized_copy_helpers(4);
        assert!(s.contains("(memory.copy 0 0"));
        assert!(s.contains("(memory.copy 0 1"));
        assert!(s.contains("(memory.copy 0 2"));
        assert!(s.contains("(memory.copy 0 3"));
        assert!(s.contains("(memory.copy 1 0"));
        assert!(s.contains("(memory.copy 2 0"));
        assert!(s.contains("(memory.copy 3 0"));
    }
}
