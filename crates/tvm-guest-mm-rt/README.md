# tvm-guest-mm-rt

Guest-side Rust runtime for the multi-memory TVM shell emitted by
**[`tvm-guest-mm`](../tvm-guest-mm/)**.

Provides safe Rust wrappers (`Pool`, `load_u32`, `store_u32`,
`copy_to_default`, …) over `extern "C"` declarations against the
`tvm_mm` import namespace. A Rust cdylib that uses this crate compiles
to a core wasm with `(import "tvm_mm" "tvm_load_u32" …)` imports; the
companion static linker
**[`tvm-guest-mm-link`](../tvm-guest-mm-link/)** rewires those imports
to the shell's internal helper functions and emits a single
self-contained `.wasm`.

## When to use this

- You're writing a non-toy multi-memory TVM consumer in Rust source
  (sqlite-pcache-tvm, sqlite-vfs-tvm, anything that wants more than
  the `ModuleParams::user_body` WAT splice).
- You want a `.wasm` that runs anywhere multi-memory is supported,
  with no host-side `tvm:memory` import implementations needed.

For the WIT path (host-mediated `tvm:memory/*` calls), see
**[`tvm-guest-rt`](../tvm-guest-rt/)** instead.

## Quick start

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
tvm-guest-mm-rt = { path = "path/to/tvm-wasm/crates/tvm-guest-mm-rt" }
```

```rust
#![no_std]
use tvm_guest_mm_rt::{Pool, load_u32, store_u32, copy_to_default};

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }

const HOT: Pool = Pool::new(1);

#[no_mangle]
pub extern "C" fn put(off: u32, v: u32) {
    store_u32(HOT, off, v);
}

#[no_mangle]
pub extern "C" fn materialize(off: u32, dst: u32, len: u32) {
    copy_to_default(HOT, off, dst, len);
}
```

```sh
cargo build --release --target wasm32-unknown-unknown
tvm-mm-link --pools 4 --user target/wasm32-unknown-unknown/release/<crate>.wasm \
            -o <crate>.linked.wasm
```

Full pipeline walkthrough:
**[`tvm-guest-mm/docs/rust-cdylib.md`](../tvm-guest-mm/docs/rust-cdylib.md)**.
Working example:
**[`examples/rust-cdylib-consumer/`](../../examples/rust-cdylib-consumer/)**.

## API surface

| Function | WAT counterpart in `tvm-guest-mm` shell |
|---|---|
| `load_u8`, `load_u32`, `load_i64` | `tvm_load_u8`, `tvm_load_u32`, `tvm_load_i64` |
| `store_u8`, `store_u32`, `store_i64` | `tvm_store_u8`, `tvm_store_u32`, `tvm_store_i64` |
| `copy_to_default` | `tvm_copy_to_default` |
| `copy_from_default` | `tvm_copy_from_default` |
| `read_bytes` (convenience) | `tvm_copy_to_default` with the slice's pointer |
| `write_bytes` (convenience) | `tvm_copy_from_default` with the slice's pointer |

The per-pool specialized helpers (`tvm_load_u32_p7`,
`tvm_simd_sum_u8_p3`, …) are exported by the shell but not bound here.
Rust cannot dispatch to one of N statically-named imports without
knowing N at the binding site; call the generic dispatcher with a
`const Pool` instead — wasmtime constant-folds the BST when the pool
is a constant at the call site.

## Host stubs

On non-wasm32 targets the imports compile down to no-ops (zero / no
effect). This lets host-target unit tests for consumer crates pass
without needing a wasm runtime — useful when the consumer crate
exposes both wasm-only and host-portable code paths.
