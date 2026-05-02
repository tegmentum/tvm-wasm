# Multi-memory probe — investigation notes

## Goal

Find out whether Rust source code can emit native wasm multi-memory loads
(`i32.load <mem_idx>`) without going through hand-written WAT.

## What we tried

### Approach 1: stable Rust + inline `asm!`
Build:  `cargo build --release --target wasm32-unknown-unknown`

Result: **fails to compile**. Inline asm on wasm32 is gated on the
`asm_experimental_arch` feature, which is nightly-only as of 2026-05.
Tracking issue: <https://github.com/rust-lang/rust/issues/93335>.

### Approach 2: nightly + `asm!` + `+multimemory` target feature
Build:
```
RUSTFLAGS="-C target-feature=+multimemory" \
  cargo +nightly build --release --target wasm32-unknown-unknown
```
With `#![feature(asm_experimental_arch)]`.

Result: **compiles, but emits the wrong instruction.** The asm
```
local.get {ptr}
i32.load 1
local.set {result}
```
gets parsed by LLVM's wasm asm frontend as `i32.load offset=1` (memarg
offset = 1), not as `i32.load (memory 1)`. The compiled module has only
one memory, declared as memory 0 — no multi-memory ops emitted.

Verified by running:
```
wasm2wat target/wasm32-unknown-unknown/release/tvm_multimem_probe.wasm \
  --enable-multi-memory
```
which shows:
```
i32.load offset=1
i32.load8_u offset=1
(memory (;0;) 16)
```

### Approach 3: alternate asm syntaxes
Other syntaxes attempted: `i32.load 1:p2align=2`, `i32.load (memory 1)`,
named memory `i32.load $r0`. None parse correctly in the LLVM wasm asm
frontend.

## Conclusion

**The Rust toolchain cannot currently emit multi-memory loads/stores.**
The required pieces:
- Inline asm on wasm32 → nightly only.
- LLVM wasm asm parser → no syntax for multi-memory immediate.
- Rust source-level multi-memory → no language support.

This is the upstream gap. Until rustc lands proper multi-memory support
(tracked under the wasm-multi-memory proposal and downstream LLVM work),
Rust guests must use hand-written WAT for multi-region native access.

## What does work today

**Single imported memory** (the default-memory case): if the Rust source
declares one imported memory and uses it as the default, standard `i32.load`
instructions target the imported memory natively. This works on stable Rust
because the source emits `i32.load offset=0` against memory 0, and memory 0
*is* the imported memory.

We exploit this in the existing TVM-MM benchmarks (which use one imported
memory per workload).

**Multiple imported memories**: must be hand-written in WAT. The
`bench-framework` runner inlines WAT strings for these cases.

## Migration path for when the toolchain catches up

The `tvm-guest-rt` crate's API is structured so a future rustc that gains
multi-memory support could provide native-load helpers without changing
the public surface. The current host-mediated `RegionPtr::read/write`
methods stay; new methods like `RegionPtr::native_load_u32_unchecked()`
would be added behind a `multimemory` cargo feature, gated on
`#[cfg(target_feature = "multimemory")]`.

When this lands, expected wins on small-cell-grain access patterns
(random chase, list walk per-cell) of 5–50× over the host-mediated path.
