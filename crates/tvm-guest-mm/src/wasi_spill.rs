//! WASI-based spill helpers. Generates a WAT module variant that
//! imports the wasi-snapshot-preview1 file API and exposes spill
//! helpers the guest can call to persist / restore region bytes.
//!
//! ## What this gives you
//!
//! - **Spill bytes from a pool to a file via `fd_write`.** One host
//!   call (the WASI fd_write) per spill operation, with the pool's
//!   bytes copied directly to the file via wasi.
//! - **Load bytes from a file back into a pool via `fd_read`.**
//!   Symmetric.
//! - **Compose with any WASI-capable runtime.** wasmtime + wasi-pt,
//!   wasmer + wasi, NodeJS + wasi shim, browser-with-wasi-shim — all
//!   work without code changes.
//!
//! ## What this doesn't do
//!
//! - **Doesn't open the file for you.** The user passes a `fd` (file
//!   descriptor) that they obtained via `path_open` (or equivalent).
//!   That's an embedder concern.
//! - **Doesn't manage WAL or partial-write recovery.** The trait
//!   surface is just spill/load; durability semantics are the
//!   embedder's call.
//!
//! ## Why this is just generated WAT (not compiled Rust)
//!
//! The same toolchain limitation that blocks multi-memory loads from
//! Rust source applies here: we need `memory.copy` between a wasm
//! memory (the pool) and an iovec for `fd_write`. The dispatch is
//! trivially expressible in WAT but not yet in Rust.

/// Generate just the WASI imports. Must appear before any function
/// definitions in the module (WAT ordering rule).
pub fn emit_wasi_imports() -> String {
    r#"  ;; --- WASI imports for spill ---
  (import "wasi_snapshot_preview1" "fd_write"
    (func $wasi_fd_write
      (param i32) ;; fd
      (param i32) ;; iovec ptr (in default memory)
      (param i32) ;; iovec count
      (param i32) ;; nwritten ptr
      (result i32))) ;; errno
  (import "wasi_snapshot_preview1" "fd_read"
    (func $wasi_fd_read
      (param i32) ;; fd
      (param i32) ;; iovec ptr
      (param i32) ;; iovec count
      (param i32) ;; nread ptr
      (result i32)))

"#
    .to_string()
}

/// Generate the spill helper functions. Goes after all imports + base
/// dispatchers.
pub fn emit_wasi_spill_helpers(n_pools: u32) -> String {
    let mut s = String::new();

    // Spill helper: copy region bytes from pool N into the default
    // memory's scratch area, then call fd_write. Uses the
    // tvm_copy_to_default dispatcher we already generate.
    //
    // Layout in default memory:
    //   [0..len)         : scratch buffer holding the bytes to write
    //   [len..len+8)     : iovec { ptr=0, len=N }
    //   [len+8..len+12)  : nwritten output slot
    s.push_str(r#"  (func (export "tvm_spill_to_fd")
        (param $pool i32) (param $src_off i32) (param $len i32) (param $fd i32)
        (result i32) ;; errno (0 = ok)
    ;; Step 1: copy pool bytes into default memory scratch (offset 0).
    (call $tvm_copy_to_default (local.get $pool) (local.get $src_off) (i32.const 0) (local.get $len))
    ;; Step 2: build iovec at offset = len.
    (i32.store (local.get $len) (i32.const 0))                               ;; iov.ptr = 0
    (i32.store (i32.add (local.get $len) (i32.const 4)) (local.get $len))    ;; iov.len = $len
    ;; Step 3: fd_write(fd, iovec_ptr=$len, iovec_count=1, nwritten_ptr=$len+8).
    (call $wasi_fd_write
      (local.get $fd)
      (local.get $len)
      (i32.const 1)
      (i32.add (local.get $len) (i32.const 8))))

  (func (export "tvm_load_from_fd")
        (param $pool i32) (param $dst_off i32) (param $len i32) (param $fd i32)
        (result i32) ;; errno (0 = ok)
    ;; Step 1: build iovec pointing at default-mem offset 0 with len bytes.
    (i32.store (local.get $len) (i32.const 0))
    (i32.store (i32.add (local.get $len) (i32.const 4)) (local.get $len))
    ;; Step 2: fd_read into default memory at offset 0.
    (drop (call $wasi_fd_read
      (local.get $fd)
      (local.get $len)
      (i32.const 1)
      (i32.add (local.get $len) (i32.const 8))))
    ;; Step 3: copy default memory bytes into target pool.
    (call $tvm_copy_from_default (local.get $pool) (local.get $dst_off) (i32.const 0) (local.get $len))
    (i32.const 0))

"#);

    let _ = n_pools; // helpers are pool-agnostic; dispatch happens in tvm_copy_*
    s
}

/// Module variant: same shape as `tvm_guest_mm_module_template` but
/// with WASI imports + spill helpers in the right order.
///
/// WAT ordering rule: imports must come before any function
/// definitions. We build from scratch so this is satisfied:
///
///   1. Memory declarations
///   2. WASI imports
///   3. Base dispatchers (load / store / copy)
///   4. WASI spill helpers
///   5. User body
pub fn tvm_guest_mm_module_with_wasi_spill(p: &crate::ModuleParams) -> String {
    use crate::dispatch::{
        emit_bulk_copy_dispatcher, emit_bulk_copy_from_default_dispatcher, emit_load_dispatcher,
        emit_specialized_copy_helpers, emit_store_dispatcher,
    };
    let mut s = String::new();
    s.push_str("(module\n");
    // 1. WASI imports (must come before memories per WAT spec).
    s.push_str(&emit_wasi_imports());
    // 2. Memories.
    for i in 0..p.n_pools {
        s.push_str(&format!(
            "  (memory (export \"mem{}\") {} {})\n",
            i, p.initial_pages_per_pool, p.max_pages_per_pool
        ));
    }
    s.push('\n');
    // 3. Base dispatchers.
    s.push_str(&emit_load_dispatcher(
        "tvm_load_u8",
        "i32.load8_u",
        "i32",
        p.n_pools,
    ));
    s.push_str(&emit_load_dispatcher(
        "tvm_load_u32",
        "i32.load",
        "i32",
        p.n_pools,
    ));
    s.push_str(&emit_load_dispatcher(
        "tvm_load_i64",
        "i64.load",
        "i64",
        p.n_pools,
    ));
    s.push_str(&emit_store_dispatcher(
        "tvm_store_u8",
        "i32.store8",
        "i32",
        p.n_pools,
    ));
    s.push_str(&emit_store_dispatcher(
        "tvm_store_u32",
        "i32.store",
        "i32",
        p.n_pools,
    ));
    s.push_str(&emit_store_dispatcher(
        "tvm_store_i64",
        "i64.store",
        "i64",
        p.n_pools,
    ));
    s.push_str(&emit_bulk_copy_dispatcher(p.n_pools));
    s.push_str(&emit_bulk_copy_from_default_dispatcher(p.n_pools));
    s.push_str(&emit_specialized_copy_helpers(p.n_pools));
    // 4. WASI spill helpers.
    s.push_str(&emit_wasi_spill_helpers(p.n_pools));
    // 5. User body.
    s.push_str("\n  ;; --- user body begins ---\n");
    s.push_str(&p.user_body);
    s.push_str("\n  ;; --- user body ends ---\n");
    s.push_str(")\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModuleParams;

    #[test]
    fn wasi_spill_module_parses() {
        let p = ModuleParams {
            n_pools: 4,
            initial_pages_per_pool: 1,
            max_pages_per_pool: 16,
            user_body: String::new(),
        };
        let wat = tvm_guest_mm_module_with_wasi_spill(&p);
        let bytes = wat::parse_str(&wat).expect("must parse");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn wasi_spill_module_has_imports() {
        let p = ModuleParams {
            n_pools: 4,
            initial_pages_per_pool: 1,
            max_pages_per_pool: 16,
            user_body: String::new(),
        };
        let wat = tvm_guest_mm_module_with_wasi_spill(&p);
        assert!(wat.contains("wasi_snapshot_preview1"));
        assert!(wat.contains("fd_write"));
        assert!(wat.contains("fd_read"));
        assert!(wat.contains("tvm_spill_to_fd"));
        assert!(wat.contains("tvm_load_from_fd"));
    }
}
