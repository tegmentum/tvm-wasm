# tvm-bench — TVM benchmark framework

Adversarial, structured benchmark suite for comparing memory architectures
on WebAssembly runtimes. The goal is not to demonstrate that TVM is faster
than Memory64 in one workload — it's to characterize *where and why* each
model wins or loses across an orthogonal matrix of stressors, so the
results survive scrutiny.

## Hypotheses (must be falsifiable)

| | Hypothesis | What it predicts |
|---|---|---|
| H1 | Address-width cost | M64 measurably slower than M32/TVM on pointer-heavy or index-heavy code |
| H2 | Locality advantage | TVM produces fewer cache/TLB misses on partitioned working sets |
| H3 | Growth behavior | TVM's incremental expansion has more predictable latency than monolithic linear-memory growth |
| H4 | Bounds-check behavior | Engines emit tighter bounds-check code for smaller memories — TVM benefits |

## Test matrix

| Variant | Memory model | Status |
|---|---|---|
| `m32` | single 32-bit linear memory | implemented (sequential) |
| `m64` | single 64-bit linear memory | **deferred** — wasm64 toolchain |
| `tvm` | multi-32-bit through `tvm-guest-rt` raw fast path | implemented (sequential) |

Runtimes:

| Runtime | Status |
|---|---|
| Wasmtime | implemented — primary runner (all variants) |
| Wasmer | implemented — cross-engine validator (M32 only; no TVM raw imports) |
| V8 | **deferred** |

See `BACKLOG.md` for the explicit list of deferred classes/backends/runtimes,
each slotted into the framework.

## Benchmark classes

Each class is an orthogonal stressor — testing one aspect of memory
behavior. A claim that "TVM is better" requires showing wins in the classes
where the underlying mechanism predicts a win, *and* no significant
regression in the others.

| # | Class | What it stresses | Status |
|---|---|---|---|
| 4.1 | Sequential access | raw bandwidth | **implemented** |
| 4.2 | Random access | cache + TLB | deferred |
| 4.3 | Pointer-heavy structures | pointer footprint, dereference cost | deferred |
| 4.4 | Allocation / growth stress | memory expansion latency | deferred |
| 4.5 | Multi-region workloads | locality across hot/warm/cold | **partial — design only** |
| 4.6 | Database-like | columnar scans, indexed lookups | deferred |
| 4.7 | JVM heap simulation | object allocation, GC scan | deferred |

## How to run

```sh
# Build the wasm guest workloads.
./build.sh

# Run the harness (wasmtime — primary).
cargo run -p tvm-bench-runner --release

# Cross-engine validator (wasmer, M32 only).
cargo run -p tvm-bench-runner-wasmer --release
```

Results land in `results/` as JSON. Reproducibility is enforced by a fixed
RNG seed, identical dataset sizes per variant, and explicit warm-up rounds.

## Determinism

- Fixed seed (`0xDEADBEEF`) for any randomized access pattern.
- Identical dataset sizes across variants — only the memory model differs.
- Warm-up rounds are explicit and excluded from the recorded samples.
- Sample size and confidence interval reported with every measurement.

## Reporting

For every (variant × class × size) tuple we record:

- Mean latency (ns).
- p50 / p95 / p99 latency.
- Throughput (bytes/sec).
- Sample size.
- Wall-clock duration of the harness run.

Derived metrics (computed offline):

- Bytes per cycle (when CPU cycles available).
- Locality efficiency: `(observed throughput) / (peak DRAM bandwidth)`.

## Threats to validity

See `THREATS.md`. Read it before claiming any win.
