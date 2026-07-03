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
    let body = build_copy_bst_with_grow(
        /*to_default=*/ true, "src_pool", /*grow_dst=*/ false, 0, n_pools,
    );
    indent_into(&mut s, &body, 4);
    s.push_str("  )\n");
    s
}

/// Symmetric: copy from default memory into a runtime-selected pool.
///
/// Before issuing the copy, this dispatcher grows the target pool on
/// demand to cover `dst_off + len`. The user / forwarded import side
/// of the merged module can't `memory.grow` a pool directly (rustc
/// cdylib code only references memory 0); without auto-grow the
/// dispatcher would trap any time a write crosses the pool's
/// `initial_pages` boundary. With auto-grow, callers can address up
/// to `max_pages` worth of bytes per pool without coordinating
/// memory.grow at every call site.
pub(crate) fn emit_bulk_copy_from_default_dispatcher(n_pools: u32) -> String {
    let mut s = String::new();
    s.push_str(
        "  (func $tvm_copy_from_default (export \"tvm_copy_from_default\")\n        \
         (param $dst_pool i32) (param $dst_off i32) (param $src_off i32) (param $len i32)\n        \
         (local $end i32)\n",
    );
    s.push_str(&format!(
        "    (if (i32.ge_u (local.get $dst_pool) (i32.const {n})) (then unreachable))\n",
        n = n_pools,
    ));
    // Compute end = dst_off + len; trap on u32 overflow.
    s.push_str(
        "    (local.set $end (i32.add (local.get $dst_off) (local.get $len)))\n\
         \
             (if (i32.lt_u (local.get $end) (local.get $dst_off)) (then unreachable))\n",
    );
    let body = build_copy_bst_with_grow(
        /*to_default=*/ false, "dst_pool", /*grow_dst=*/ true, 0, n_pools,
    );
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
/// SIMD-accelerated kernels that read from one of the data pools.
///
/// `tvm_simd_sum_u8_p{K}(off, len) -> i64` returns the unsigned sum
/// of `len` bytes starting at `(pool K, off)`. The loop processes 16
/// bytes per iteration:
///
/// ```text
/// v = v128.load <pool K> off
/// // 16 u8 → 8 u16 (pairwise add)
/// pairs = i16x8.extadd_pairwise_i8x16_u(v)
/// // 8 u16  → 4 u32 (pairwise add)
/// quads = i32x4.extadd_pairwise_i16x8_u(pairs)
/// // 4 u32  → 2 u64 (split + extend) and accumulate
/// acc += i64x2.extend_low_i32x4_u(quads)
///      + i64x2.extend_high_i32x4_u(quads)
/// ```
///
/// Wasm SIMD doesn't have a 32→64 pairwise extadd, so we widen with
/// `extend_low/high` and add both halves into the i64x2 accumulator.
/// At the tail, any remaining `len % 16` bytes are summed with a
/// scalar `i64.load8_u` loop. Final reduce is `lane0 + lane1`.
///
/// **Statically specialized per pool**, so the `v128.load` and
/// `i64.load8_u` carry constant memory immediates — zero dispatch on
/// the hot path. Real wins for sum / hash / xor / count workloads.
/// For pure copy-pool-to-default, `tvm_copy_to_default_p{K}` already
/// runs at hardware bandwidth and SIMD adds nothing on top.
pub(crate) fn emit_specialized_simd_kernels(n_pools: u32) -> String {
    let mut s = String::new();
    for k in 0..n_pools {
        s.push_str(&format!(
            r#"  (func $tvm_simd_sum_u8_p{k} (export "tvm_simd_sum_u8_p{k}")
        (param $off i32) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $acc v128) (local $quads v128) (local $scalar i64)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $acc (v128.const i64x2 0 0))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $quads
          (i32x4.extadd_pairwise_i16x8_u
            (i16x8.extadd_pairwise_i8x16_u
              (v128.load {k} (local.get $cur)))))
        (local.set $acc
          (i64x2.add
            (i64x2.add
              (local.get $acc)
              (i64x2.extend_low_i32x4_u (local.get $quads)))
            (i64x2.extend_high_i32x4_u (local.get $quads))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (local.set $scalar
      (i64.add
        (i64x2.extract_lane 0 (local.get $acc))
        (i64x2.extract_lane 1 (local.get $acc))))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (local.set $scalar
          (i64.add (local.get $scalar)
                   (i64.load8_u {k} (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (local.get $scalar))
"#,
            k = k,
        ));
    }
    s
}

/// Symmetric SIMD reducer family — mirrors the host-side reducer API
/// for pure-guest deployments that don't have a host to delegate to.
///
/// All kernels are per-pool specialized (zero dispatch), process 16
/// bytes per SIMD iteration, and handle a scalar tail. Each follows
/// the same shape:
///
///   - load a v128 from `pool K`
///   - apply lane-wise op + accumulate
///   - tail loop for `len % 16` remaining bytes
///   - horizontal reduce the v128 acc into the scalar return
///
/// Naming: `tvm_simd_<op>_p{K}`. All are reducers (single scalar
/// result). Mutators (fill, xor_with_byte) and multi-output
/// (histogram) aren't included here — the dedicated `memory.fill` op
/// handles the former and the latter doesn't SIMD well.
pub(crate) fn emit_specialized_simd_reducers(n_pools: u32) -> String {
    let mut s = String::new();
    for k in 0..n_pools {
        s.push_str(&simd_xor_fold(k));
        s.push_str(&simd_and_fold(k));
        s.push_str(&simd_or_fold(k));
        s.push_str(&simd_count_byte(k));
        s.push_str(&simd_popcount(k));
        s.push_str(&simd_find_byte(k));
        s.push_str(&simd_min_max(k));
    }
    s
}

/// XOR fold: v128.xor accumulator, horizontal-XOR at end.
fn simd_xor_fold(k: u32) -> String {
    format!(
        r#"  (func $tvm_simd_xor_fold_u8_p{k} (export "tvm_simd_xor_fold_u8_p{k}")
        (param $off i32) (param $len i32) (result i32)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $acc v128) (local $tmp i64) (local $byte i32)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $acc (v128.const i64x2 0 0))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $acc
          (v128.xor (local.get $acc) (v128.load {k} (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (local.set $tmp
      (i64.xor
        (i64x2.extract_lane 0 (local.get $acc))
        (i64x2.extract_lane 1 (local.get $acc))))
    (local.set $tmp
      (i64.xor (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 32))))
    (local.set $tmp
      (i64.xor (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 16))))
    (local.set $tmp
      (i64.xor (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 8))))
    (local.set $byte (i32.and (i32.wrap_i64 (local.get $tmp)) (i32.const 0xff)))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (local.set $byte
          (i32.xor (local.get $byte)
                   (i32.load8_u {k} (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (local.get $byte))
"#,
        k = k,
    )
}

/// AND fold: identity is all-ones; v128.and per iter; horizontal AND at end.
fn simd_and_fold(k: u32) -> String {
    format!(
        r#"  (func $tvm_simd_and_fold_u8_p{k} (export "tvm_simd_and_fold_u8_p{k}")
        (param $off i32) (param $len i32) (result i32)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $acc v128) (local $tmp i64) (local $byte i32)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $acc (v128.const i64x2 -1 -1))
    (local.set $byte (i32.const 0xff))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $acc
          (v128.and (local.get $acc) (v128.load {k} (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (if (i32.ne (local.get $vec_end) (local.get $off))
      (then
        (local.set $tmp
          (i64.and
            (i64x2.extract_lane 0 (local.get $acc))
            (i64x2.extract_lane 1 (local.get $acc))))
        (local.set $tmp
          (i64.and (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 32))))
        (local.set $tmp
          (i64.and (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 16))))
        (local.set $tmp
          (i64.and (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 8))))
        (local.set $byte
          (i32.and (i32.wrap_i64 (local.get $tmp)) (i32.const 0xff)))))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (local.set $byte
          (i32.and (local.get $byte)
                   (i32.load8_u {k} (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (local.get $byte))
"#,
        k = k,
    )
}

/// OR fold: identity is zero; v128.or per iter.
fn simd_or_fold(k: u32) -> String {
    format!(
        r#"  (func $tvm_simd_or_fold_u8_p{k} (export "tvm_simd_or_fold_u8_p{k}")
        (param $off i32) (param $len i32) (result i32)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $acc v128) (local $tmp i64) (local $byte i32)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $acc (v128.const i64x2 0 0))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $acc
          (v128.or (local.get $acc) (v128.load {k} (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (local.set $tmp
      (i64.or
        (i64x2.extract_lane 0 (local.get $acc))
        (i64x2.extract_lane 1 (local.get $acc))))
    (local.set $tmp
      (i64.or (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 32))))
    (local.set $tmp
      (i64.or (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 16))))
    (local.set $tmp
      (i64.or (local.get $tmp) (i64.shr_u (local.get $tmp) (i64.const 8))))
    (local.set $byte (i32.and (i32.wrap_i64 (local.get $tmp)) (i32.const 0xff)))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (local.set $byte
          (i32.or (local.get $byte)
                  (i32.load8_u {k} (local.get $cur))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (local.get $byte))
"#,
        k = k,
    )
}

/// count_byte: i8x16.eq + i8x16.bitmask + i32.popcnt of mask.
fn simd_count_byte(k: u32) -> String {
    format!(
        r#"  (func $tvm_simd_count_byte_p{k} (export "tvm_simd_count_byte_p{k}")
        (param $off i32) (param $len i32) (param $byte i32) (result i32)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $broadcast v128) (local $count i32)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $broadcast (i8x16.splat (local.get $byte)))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $count
          (i32.add (local.get $count)
            (i32.popcnt
              (i8x16.bitmask
                (i8x16.eq
                  (v128.load {k} (local.get $cur))
                  (local.get $broadcast))))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (if (i32.eq (i32.load8_u {k} (local.get $cur)) (local.get $byte))
          (then (local.set $count (i32.add (local.get $count) (i32.const 1)))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (local.get $count))
"#,
        k = k,
    )
}

/// popcount: i8x16.popcnt then accumulate into i32x4 via pairwise widens.
fn simd_popcount(k: u32) -> String {
    format!(
        r#"  (func $tvm_simd_popcount_p{k} (export "tvm_simd_popcount_p{k}")
        (param $off i32) (param $len i32) (result i64)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $acc v128) (local $tmp v128) (local $scalar i64)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $acc (v128.const i64x2 0 0))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $tmp
          (i32x4.extadd_pairwise_i16x8_u
            (i16x8.extadd_pairwise_i8x16_u
              (i8x16.popcnt
                (v128.load {k} (local.get $cur))))))
        (local.set $acc
          (i64x2.add
            (i64x2.add
              (local.get $acc)
              (i64x2.extend_low_i32x4_u (local.get $tmp)))
            (i64x2.extend_high_i32x4_u (local.get $tmp))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (local.set $scalar
      (i64.add
        (i64x2.extract_lane 0 (local.get $acc))
        (i64x2.extract_lane 1 (local.get $acc))))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (local.set $scalar
          (i64.add (local.get $scalar)
            (i64.extend_i32_u
              (i32.popcnt (i32.load8_u {k} (local.get $cur))))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (local.get $scalar))
"#,
        k = k,
    )
}

/// find_byte: i8x16.eq + i8x16.bitmask + i32.ctz to locate first match.
/// Returns offset within `[off, off+len)` or -1.
fn simd_find_byte(k: u32) -> String {
    format!(
        r#"  (func $tvm_simd_find_byte_p{k} (export "tvm_simd_find_byte_p{k}")
        (param $off i32) (param $len i32) (param $byte i32) (result i32)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $broadcast v128) (local $mask i32)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $broadcast (i8x16.splat (local.get $byte)))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $mask
          (i8x16.bitmask
            (i8x16.eq
              (v128.load {k} (local.get $cur))
              (local.get $broadcast))))
        (if (i32.ne (local.get $mask) (i32.const 0))
          (then
            (return (i32.sub
              (i32.add (local.get $cur) (i32.ctz (local.get $mask)))
              (local.get $off)))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (if (i32.eq (i32.load8_u {k} (local.get $cur)) (local.get $byte))
          (then
            (return (i32.sub (local.get $cur) (local.get $off)))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (i32.const -1))
"#,
        k = k,
    )
}

/// min_max_u8: i8x16.min_u / max_u accumulators, horizontal reduce at end.
/// Returns ((min as i32) << 8) | (max as i32). For len==0 returns
/// (255 << 8) | 0 — the conventional empty-range sentinel.
fn simd_min_max(k: u32) -> String {
    format!(
        r#"  (func $tvm_simd_min_max_u8_p{k} (export "tvm_simd_min_max_u8_p{k}")
        (param $off i32) (param $len i32) (result i32)
    (local $cur i32) (local $end i32) (local $vec_end i32)
    (local $vmin v128) (local $vmax v128) (local $vec v128)
    (local $any_vec i32) (local $lo i32) (local $hi i32) (local $b i32)
    (local.set $cur (local.get $off))
    (local.set $end (i32.add (local.get $off) (local.get $len)))
    (local.set $vec_end
      (i32.add (local.get $off)
               (i32.and (local.get $len) (i32.const -16))))
    (local.set $vmin (v128.const i8x16 -1 -1 -1 -1 -1 -1 -1 -1
                                       -1 -1 -1 -1 -1 -1 -1 -1))
    (local.set $vmax (v128.const i64x2 0 0))
    (local.set $lo (i32.const 255))
    (local.set $hi (i32.const 0))
    (block $vec_done
      (loop $vec_continue
        (br_if $vec_done (i32.eq (local.get $cur) (local.get $vec_end)))
        (local.set $vec (v128.load {k} (local.get $cur)))
        (local.set $vmin (i8x16.min_u (local.get $vmin) (local.get $vec)))
        (local.set $vmax (i8x16.max_u (local.get $vmax) (local.get $vec)))
        (local.set $any_vec (i32.const 1))
        (local.set $cur (i32.add (local.get $cur) (i32.const 16)))
        (br $vec_continue)))
    (if (i32.eq (local.get $any_vec) (i32.const 1))
      (then
        ;; Reduce vmin / vmax across all 16 lanes by extract+scalar-min.
        (local.set $b (i8x16.extract_lane_u 0 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 0 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 1 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 1 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 2 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 2 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 3 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 3 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 4 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 4 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 5 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 5 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 6 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 6 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 7 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 7 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 8 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 8 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 9 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 9 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 10 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 10 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 11 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 11 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 12 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 12 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 13 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 13 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 14 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 14 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 15 (local.get $vmin)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (local.set $b (i8x16.extract_lane_u 15 (local.get $vmax)))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))))
    (block $tail_done
      (loop $tail_continue
        (br_if $tail_done (i32.eq (local.get $cur) (local.get $end)))
        (local.set $b (i32.load8_u {k} (local.get $cur)))
        (if (i32.lt_u (local.get $b) (local.get $lo))
          (then (local.set $lo (local.get $b))))
        (if (i32.gt_u (local.get $b) (local.get $hi))
          (then (local.set $hi (local.get $b))))
        (local.set $cur (i32.add (local.get $cur) (i32.const 1)))
        (br $tail_continue)))
    (i32.or
      (i32.shl (local.get $lo) (i32.const 8))
      (local.get $hi)))
"#,
        k = k,
    )
}

pub(crate) fn emit_specialized_copy_helpers(n_pools: u32) -> String {
    let mut s = String::new();
    for k in 0..n_pools {
        // The internal `$tvm_copy_to_default_pK` name lets in-module
        // user code call these directly; the export name lets host
        // code call them when needed.
        //
        // `tvm_copy_to_default_p{K}` reads pool K → default memory.
        // No auto-grow on the source pool (reading past the pool's
        // size is a genuine error and should trap loud), and the
        // destination (default memory) is the user's responsibility
        // to size — pool 0 is already linked to the user's default
        // memory and grows via the user's own `memory.grow`.
        s.push_str(&format!(
            "  (func $tvm_copy_to_default_p{k} (export \"tvm_copy_to_default_p{k}\")\n        \
             (param $src_off i32) (param $dst_off i32) (param $len i32)\n    \
             (memory.copy 0 {k} (local.get $dst_off) (local.get $src_off) (local.get $len)))\n",
            k = k,
        ));
        // `tvm_copy_from_default_p{K}` writes default memory → pool
        // K. The dispatcher must grow pool K on demand because user
        // code can't issue `memory.grow K` directly. Mirrors the
        // grow-then-copy behavior in the runtime-dispatched
        // `tvm_copy_from_default` for the same reason.
        s.push_str(&format!(
            "  (func $tvm_copy_from_default_p{k} (export \"tvm_copy_from_default_p{k}\")\n        \
             (param $dst_off i32) (param $src_off i32) (param $len i32)\n        \
             (local $end i32)\n    \
             (local.set $end (i32.add (local.get $dst_off) (local.get $len)))\n    \
             (if (i32.lt_u (local.get $end) (local.get $dst_off)) (then unreachable))\n    \
             (if (i32.gt_u\n          (i32.shr_u (i32.add (local.get $end) (i32.const 65535)) (i32.const 16))\n          (memory.size {k}))\n      (then\n        (drop\n          (memory.grow {k}\n            (i32.sub\n              (i32.shr_u (i32.add (local.get $end) (i32.const 65535)) (i32.const 16))\n              (memory.size {k}))))))\n    \
             (memory.copy {k} 0 (local.get $dst_off) (local.get $src_off) (local.get $len)))\n",
            k = k,
        ));
    }
    s
}

/// Indirect-table dispatchers for typed loads. Builds a function
/// table populated with the per-pool specialized helpers, dispatches
/// via `call_indirect` rather than the BST comparison cascade.
///
/// **Empirical result on wasmtime: ~5× slower than BST** (per-byte
/// sum at n_pools=64: BST = 45 µs, call_indirect = 225 µs). Wasmtime's
/// indirect-call overhead — signature check, indirect branch, frame
/// setup — dominates the savings from skipping ~6 compares. The BST
/// stays the default; these dispatchers are kept emitted so the
/// `dispatch_shape_comparison` bench remains reproducible and so a
/// user on a different runtime can A/B test without re-emitting the
/// module template.
///
/// Naming: `tvm_load_u8_indirect`, `tvm_load_u32_indirect`,
/// `tvm_load_i64_indirect`. Same signature as the BST dispatcher
/// (`(pool, off) → typed`); drop-in if you somehow find an engine
/// where indirect wins.
pub(crate) fn emit_indirect_load_dispatchers(n_pools: u32) -> String {
    if n_pools == 0 {
        return String::new();
    }
    let mut s = String::new();
    // Type signatures (use distinct names so `(type ...)` is referable).
    s.push_str("  (type $tvm_indirect_load_u8 (func (param i32) (result i32)))\n");
    s.push_str("  (type $tvm_indirect_load_u32 (func (param i32) (result i32)))\n");
    s.push_str("  (type $tvm_indirect_load_i64 (func (param i32) (result i64)))\n");

    // One table per typed-load family. The inline `(elem ...)` form
    // declares table size and initial contents in one shot — avoids
    // the WAT-active-segment ambiguity around `(table $t)` vs bare
    // `$t` for elem-segment table refs.
    let mut elem_u8 = String::new();
    let mut elem_u32 = String::new();
    let mut elem_i64 = String::new();
    for k in 0..n_pools {
        elem_u8.push_str(&format!(" $tvm_load_u8_p{}", k));
        elem_u32.push_str(&format!(" $tvm_load_u32_p{}", k));
        elem_i64.push_str(&format!(" $tvm_load_i64_p{}", k));
    }
    s.push_str(&format!(
        "  (table $tvm_tbl_load_u8 funcref (elem{}))\n",
        elem_u8,
    ));
    s.push_str(&format!(
        "  (table $tvm_tbl_load_u32 funcref (elem{}))\n",
        elem_u32,
    ));
    s.push_str(&format!(
        "  (table $tvm_tbl_load_i64 funcref (elem{}))\n",
        elem_i64,
    ));

    // Dispatcher functions.
    s.push_str(&format!(
        r#"  (func $tvm_load_u8_indirect (export "tvm_load_u8_indirect")
        (param $pool i32) (param $off i32) (result i32)
    (if (i32.ge_u (local.get $pool) (i32.const {n})) (then unreachable))
    (call_indirect $tvm_tbl_load_u8 (type $tvm_indirect_load_u8)
      (local.get $off) (local.get $pool)))
  (func $tvm_load_u32_indirect (export "tvm_load_u32_indirect")
        (param $pool i32) (param $off i32) (result i32)
    (if (i32.ge_u (local.get $pool) (i32.const {n})) (then unreachable))
    (call_indirect $tvm_tbl_load_u32 (type $tvm_indirect_load_u32)
      (local.get $off) (local.get $pool)))
  (func $tvm_load_i64_indirect (export "tvm_load_i64_indirect")
        (param $pool i32) (param $off i32) (result i64)
    (if (i32.ge_u (local.get $pool) (i32.const {n})) (then unreachable))
    (call_indirect $tvm_tbl_load_i64 (type $tvm_indirect_load_i64)
      (local.get $off) (local.get $pool)))
"#,
        n = n_pools,
    ));
    s
}

/// Per-pool intra-pool copy helpers — `memory.copy K K`. Used by the
/// compactor to slide live blocks within their region's pool. One per
/// pool, statically specialized, zero dispatch on the hot path.
///
/// Source and destination may overlap; wasm `memory.copy` semantics
/// are equivalent to copying via a scratch buffer, so direction
/// doesn't need to be sorted out by the caller.
pub(crate) fn emit_specialized_intra_pool_copy(n_pools: u32) -> String {
    let mut s = String::new();
    for k in 0..n_pools {
        s.push_str(&format!(
            "  (func $tvm_intra_pool_copy_p{k} (export \"tvm_intra_pool_copy_p{k}\")\n        \
             (param $dst_off i32) (param $src_off i32) (param $len i32)\n    \
             (memory.copy {k} {k} (local.get $dst_off) (local.get $src_off) (local.get $len)))\n",
            k = k,
        ));
    }
    s
}

/// Specialized typed load/store helpers — one per pool, no dispatch.
/// Mirrors the specialized copy helpers but for single typed
/// load/store ops. Useful in tight per-element loops where the pool
/// is already resolved.
///
/// Per pool K, emits:
///   tvm_load_u8_p{K}, tvm_load_u32_p{K}, tvm_load_i64_p{K}
///   tvm_store_u8_p{K}, tvm_store_u32_p{K}, tvm_store_i64_p{K}
///
/// Each is a single static load/store with the memory immediate baked
/// in. Skips the BST dispatch entirely.
pub(crate) fn emit_specialized_typed_helpers(n_pools: u32) -> String {
    let mut s = String::new();
    for k in 0..n_pools {
        // Loads.
        s.push_str(&format!(
            "  (func $tvm_load_u8_p{k} (export \"tvm_load_u8_p{k}\") (param $off i32) (result i32) (i32.load8_u {k} (local.get $off)))\n",
            k = k,
        ));
        s.push_str(&format!(
            "  (func $tvm_load_u32_p{k} (export \"tvm_load_u32_p{k}\") (param $off i32) (result i32) (i32.load {k} (local.get $off)))\n",
            k = k,
        ));
        s.push_str(&format!(
            "  (func $tvm_load_i64_p{k} (export \"tvm_load_i64_p{k}\") (param $off i32) (result i64) (i64.load {k} (local.get $off)))\n",
            k = k,
        ));
        // Stores.
        s.push_str(&format!(
            "  (func $tvm_store_u8_p{k} (export \"tvm_store_u8_p{k}\") (param $off i32) (param $v i32) (i32.store8 {k} (local.get $off) (local.get $v)))\n",
            k = k,
        ));
        s.push_str(&format!(
            "  (func $tvm_store_u32_p{k} (export \"tvm_store_u32_p{k}\") (param $off i32) (param $v i32) (i32.store {k} (local.get $off) (local.get $v)))\n",
            k = k,
        ));
        s.push_str(&format!(
            "  (func $tvm_store_i64_p{k} (export \"tvm_store_i64_p{k}\") (param $off i32) (param $v i64) (i64.store {k} (local.get $off) (local.get $v)))\n",
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
/// `pool_param`. When `grow_dst` is true, each leaf preambles the
/// copy with a `memory.grow` call against the destination pool to
/// cover `(dst_off + len)`; the dispatcher is expected to have set up
/// a `$end` local holding `dst_off + len` and to have already trapped
/// on `$end < dst_off` (u32 overflow). For pool→default copies the
/// destination is memory 0 (the default), which grows independently
/// and is the user's responsibility, so `grow_dst` is unused in that
/// direction.
fn build_copy_bst_with_grow(
    to_default: bool,
    pool_param: &str,
    grow_dst: bool,
    lo: u32,
    hi: u32,
) -> String {
    debug_assert!(lo < hi);
    debug_assert!(
        !grow_dst || !to_default,
        "grow_dst only makes sense for default→pool"
    );
    if hi - lo == 1 {
        let (dst_mem, src_mem) = if to_default { (0, lo) } else { (lo, 0) };
        // For pool→default, params are (src_pool, src_off, dst_off, len).
        // For default→pool, params are (dst_pool, dst_off, src_off, len).
        // The emitted instruction is identical either way; direction is
        // already encoded in (dst_mem, src_mem) above.
        let copy = format!(
            "(memory.copy {dst} {src} (local.get $dst_off) (local.get $src_off) (local.get $len))\n",
            dst = dst_mem,
            src = src_mem,
        );
        if grow_dst {
            // Grow pool `lo` to cover `$end` bytes if it isn't
            // already that big. Page count is `ceil($end / 65536)`,
            // computed as `($end + 65535) >> 16`. If the grow fails
            // (returns -1), let the subsequent memory.copy trap with
            // the natural OOB error — same behavior as before this
            // change.
            return format!(
                "(if (i32.gt_u\n      (i32.shr_u (i32.add (local.get $end) (i32.const 65535)) (i32.const 16))\n      (memory.size {k}))\n  (then\n    (drop\n      (memory.grow {k}\n        (i32.sub\n          (i32.shr_u (i32.add (local.get $end) (i32.const 65535)) (i32.const 16))\n          (memory.size {k}))))))\n{copy}",
                k = lo,
                copy = copy,
            );
        }
        return copy;
    }
    let mid = lo + (hi - lo) / 2;
    let left = build_copy_bst_with_grow(to_default, pool_param, grow_dst, lo, mid);
    let right = build_copy_bst_with_grow(to_default, pool_param, grow_dst, mid, hi);
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
                panic!(
                    "n_pools={} failed to parse: {}\n--- module ---\n{}",
                    n, e, module
                )
            });
        }
    }

    #[test]
    fn store_dispatcher_parses_for_various_n() {
        for n in [1u32, 2, 5, 16, 64] {
            let body = emit_store_dispatcher("tvm_store_u32", "i32.store", "i32", n);
            wat::parse_str(wrap(n, &body)).expect("parse");
        }
    }

    #[test]
    fn copy_dispatchers_parse() {
        for n in [1u32, 2, 3, 8, 64] {
            let mut body = String::new();
            body.push_str(&emit_bulk_copy_dispatcher(n));
            body.push_str(&emit_bulk_copy_from_default_dispatcher(n));
            wat::parse_str(wrap(n, &body)).expect("parse");
        }
    }

    #[test]
    fn specialized_copy_helpers_parse() {
        for n in [1u32, 4, 16, 64] {
            let body = emit_specialized_copy_helpers(n);
            wat::parse_str(wrap(n, &body)).expect("parse");
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

    #[test]
    fn simd_kernels_parse() {
        for n in [1u32, 4, 16] {
            let body = emit_specialized_simd_kernels(n);
            wat::parse_str(wrap(n, &body)).expect("parse");
        }
    }

    #[test]
    fn indirect_load_dispatchers_parse() {
        for n in [1u32, 2, 4, 16, 64] {
            let mut body = emit_specialized_typed_helpers(n);
            body.push_str(&emit_indirect_load_dispatchers(n));
            wat::parse_str(wrap(n, &body))
                .unwrap_or_else(|e| panic!("indirect dispatchers n={} failed to parse: {}", n, e));
        }
    }

    #[test]
    fn intra_pool_copy_helpers_parse() {
        for n in [1u32, 2, 4, 16] {
            let body = emit_specialized_intra_pool_copy(n);
            wat::parse_str(wrap(n, &body)).expect("parse");
        }
    }

    #[test]
    fn specialized_typed_helpers_parse() {
        for n in [1u32, 2, 4, 16] {
            let body = emit_specialized_typed_helpers(n);
            wat::parse_str(wrap(n, &body)).expect("parse");
        }
    }

    #[test]
    fn simd_reducers_parse() {
        for n in [1u32, 2, 4, 8] {
            let body = emit_specialized_simd_reducers(n);
            wat::parse_str(wrap(n, &body)).unwrap_or_else(|e| {
                panic!(
                    "simd reducers n={} failed to parse: {}\n--- module ---\n{}",
                    n,
                    e,
                    wrap(n, &body)
                )
            });
        }
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
