# tvm-guest-mm

Guest-side TVM with multi-memory pools — emits a self-contained WAT
shell declaring N internal wasm memories plus dispatch helpers
(`tvm_load_u32`, `tvm_store_u32`, `tvm_copy_to_default`, …). No host
imports needed; the same `.wasm` runs anywhere the wasm multi-memory
proposal is implemented (wasmtime, wasmer, modern browsers, …).

## When to use this

- You want a TVM-shaped working set above 4 GiB and your runtime has
  the multi-memory proposal.
- You don't have, or don't want, a host that exposes a `tvm:memory`
  interface — pure-guest sandboxing.

Compare with `tvm-wasmtime` for the host-side path that adds
spill-to-disk + host-side observability.

## How to consume

Three documented patterns, in increasing order of ergonomics:

1. **Splice raw WAT** into `ModuleParams::user_body`. Lowest-level —
   useful for the bench framework, internal probes, and one-off
   experiments. See `tests/end_to_end.rs`.

2. **Post-build composition** with `wasm-tools` against the bytes
   emitted by the `gen_guest_wasm` binary. Useful when the workload
   is a separate Rust crate compiled in isolation. Predates the
   linker; mostly superseded by (3).

3. **Static linking from a Rust cdylib via `tvm-guest-mm-link`.** The
   recommended path for non-toy consumers (sqlite-pcache-tvm,
   sqlite-vfs-tvm, …). Write ordinary Rust source against the safe
   API in **[`tvm-guest-mm-rt`](../tvm-guest-mm-rt/)**, build a
   wasm32 cdylib, then run `tvm-mm-link` (from
   **[`tvm-guest-mm-link`](../tvm-guest-mm-link/)**) to produce a
   single self-contained `.wasm`.

   Full walkthrough: [`docs/rust-cdylib.md`](docs/rust-cdylib.md).
   Working example: [`examples/rust-cdylib-consumer/`](../../examples/rust-cdylib-consumer/).

## Layout

| Module | Purpose |
|---|---|
| `module` | `tvm_guest_mm_module_template(&ModuleParams) → String` — emits the WAT shell |
| `dispatch` | Per-helper WAT codegen (BST dispatch, per-pool specialized helpers, SIMD reducers) |
| `directory` | `GuestDirectory` + `Pool` — pool allocation, region creation, handle resolution |
| `facade` | `GuestTvm` — `TvmFacade` impl bridging directory + dispatch helpers |
| `wasi_spill` | Optional WASI-driven spill to disk for the shell |
| `multi` | Multi-shard guest TVM (one TVM per shard) |

## Quick start (template-only path)

```rust
use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams};

let params = ModuleParams {
    n_pools: 4,
    initial_pages_per_pool: 1,
    max_pages_per_pool: 256,
    user_body: r#"
        (func (export "answer") (result i32)
          (call $tvm_store_u32 (i32.const 1) (i32.const 0) (i32.const 42))
          (call $tvm_load_u32  (i32.const 1) (i32.const 0)))
    "#.to_string(),
};
let wat = tvm_guest_mm_module_template(&params);
let bytes = wat::parse_str(&wat)?;
```

For the cdylib path, see [`docs/rust-cdylib.md`](docs/rust-cdylib.md).
