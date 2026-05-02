# examples

Each example is a separate crate that compiles to `wasm32-unknown-unknown`
(or, in one case, the host triple). They demonstrate the three guest
access paths to TVM and one investigation of toolchain limits.

## Pick by access path

| Directory | Access path | Toolchain | When to use |
|---|---|---|---|
| `guest-demo/` | **WIT bindings (component model)** | stable Rust + `wit-bindgen` | Multi-language portability. Type-safe variant returns. Slowest path. |
| `guest-fast-path/` | **Raw imports** (`tvm.alloc`, `tvm.read`, `tvm.write`) | stable Rust + `tvm-guest-rt` | Hot loops where component-model overhead dominates. |
| `multimem-probe/` | (investigation) | nightly + `+multimemory` + `asm_experimental_arch` | Reference for what doesn't yet work; see `INVESTIGATION.md`. |

For the **imported-memory** access path (TVM-Unified — regions exposed as
imported wasm memories that the guest accesses natively), see the
hand-written WAT modules in `bench-framework/runner/src/main.rs`
(`MM_WAT`, `UNIFIED_WAT`). Rust source can't drive multi-memory imports
yet; see `multimem-probe/INVESTIGATION.md` for why.

## Building

The `bench-framework/build.sh` script builds the workload guests used by
the benchmark runner. The example guests are built separately by their
own `cargo build --target wasm32-unknown-unknown` invocations.

For the multimem probe specifically:
```sh
cd multimem-probe
RUSTFLAGS="-C target-feature=+multimemory" \
  cargo +nightly build --release \
  -Zbuild-std=panic_abort,std \
  --target wasm64-unknown-unknown
```
(See `multimem-probe/INVESTIGATION.md` for what this proves.)

## Why three guest crates and not one with feature flags?

Each crate has different deps (`wit-bindgen` vs. `tvm-guest-rt` vs. nightly
asm), and each compiles to a separate `.wasm` artifact that the bench
framework or test harness loads. Feature flags would conflate the build
products and force rebuilds when switching paths.
