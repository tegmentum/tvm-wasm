# tvm-wasm

**Tiered Virtual Memory for WebAssembly.**

A composable memory substrate that replaces a single monolithic 64-bit linear
memory with a directory of coordinated 32-bit memories. Each *region* is
typed (heap, arena, blob, page-store, etc.), policy-driven (pinnable,
spillable, initial tier), and accessed through a generation-checked *handle*
— never a raw cross-region pointer.

| Why | What you get |
|---|---|
| Lower runtime overhead | 32-bit linear memories, no memory64 cost |
| Better locality | Hot data stays hot; cold regions spill to disk |
| Explicit lifecycle | `pin`, `demote`, `compact`, `spill` are first-class |
| Beyond 4 GB | Compose multiple 4 GB regions instead of one giant one |
| Capability-style isolation | Handles cross thread and FFI boundaries safely |

## Quick start

```sh
# Run the host-side test suite (no toolchain dependencies).
cargo test --workspace

# Run the wasm32 end-to-end tests too (requires the target).
rustup target add wasm32-unknown-unknown
cargo test --workspace
```

## Crates

| Crate | What it does |
|---|---|
| `tvm-core` | Region directory, allocators, residency tiering, handles, metrics |
| `tvm-wasmtime` | Wasmtime bindings: WIT host impl, raw fast-path linker, multi-store sharing |
| `tvm-guest-rt` | Guest-side safe Rust API over the raw fast path |
| `tvm-tests` | Integration tests for `tvm-core` |

Plus example guests under `examples/guest-demo/` (WIT path) and
`examples/guest-fast-path/` (raw path).

## Two ways to call into TVM from a guest

```rust
// 1. WIT path — type-safe, multi-language portable, slowest. Use for setup
//    and rare calls.
manager::create_region(RegionKind::HotHeap, 4096)?;
bytes::write(handle, &payload)?;

// 2. Raw path — i32/i64 imports, single host scratch copy, no canonical ABI.
//    Use for hot loops.
use tvm_guest_rt::Region;
let region = Region::from_id(0);
let h = region.alloc(64)?;
h.write(&payload)?;
```

See [`docs/fast-paths.md`](docs/fast-paths.md) for the detailed cost model
and when to use which.

## Design

See [`docs/architecture.md`](docs/architecture.md) for the implementation
overview, residency tiers, the handle/generation model, and how the WIT and
raw paths share state.

The original specification is preserved in this README's git history; the
architecture doc describes what was actually built.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the build/test workflow,
including how to regenerate WIT bindings and run the wasm32 examples.

## Status

Pre-1.0. The shape of the WIT package and the host trait surface is stable
enough to build against. Test count: 114 across the workspace.

### Headline benchmark results (Apple Silicon, wasmtime, 256 KiB working set)

| Class | M32 | M64 | TVM-MM | TVM | TVM/M64 |
|---|---:|---:|---:|---:|---:|
| Sequential sum | 47 µs | 2599 µs | 163 µs | **45 µs** | **58.3×** |
| List walk | 138 µs | 647 µs | **150 µs** | 251 µs | 2.6× |
| Multi-region 90/9/1 | 32 µs | 154 µs | — | 37 µs | 4.2× |
| Columnar filter+sum | 16 µs | 487 µs | 21 µs | 29 µs | **17.0×** |
| Large-WS probe | 27 µs | 106 µs | — | 33 µs | 3.2× |
| Growth (alloc + touch) | 3 µs | 65 µs | — | 41 µs | 1.6× |
| JVM gen-alloc-scan | 6 µs | 152 µs | — | 41 µs | **3.7×** |
| Spill-driven (>resident budget) | infeasible | infeasible | — | 0.5 µs/cycle | TVM only |

**TVM beats Memory64 in 20 of 24 measured (class × size) pairs**, with up to
**58.3× speedup** on sequential.

For sequential workloads, TVM is **at parity with native M32** — within
measurement noise across 50-sample runs (TVM 2788–3151 ns; M32 2963–3019 ns
at 16 KiB). The host-mediated bulk-read pattern compensates for its trampoline
cost via vectorized memcpy at the host level — fast enough to keep up with
wasmtime's per-byte engine-emitted load loop. We are not claiming TVM is
*faster than* M32; we are claiming it is *not measurably slower than M32*
on bulk-read workloads, which is itself an architecturally surprising result.

**TVM-MM** (multi-memory imports — each region exposed as a native imported
wasm memory) ties M32 on list-walk (140 µs vs 138 µs) and is within 30% on
columnar. The 3× gap on sequential is in wasmtime's imported-memory codegen,
not in our WAT — and closes automatically as the engine improves.

The honest framing: **TVM gives you region-and-handle abstraction, lifecycle
management, and >4 GiB working sets at performance parity with native M32 for
bulk workloads, while beating Memory64 by 2.5–58× across the matrix.** That's
the load-bearing claim.

See `bench-framework/README.md` for the full design and `THREATS.md` for
mitigations against engine-specific bias / cherry-picking. Run reproducibly
with `./bench-framework/build.sh && cargo run -p tvm-bench-runner --release`.

## License

Apache-2.0.
