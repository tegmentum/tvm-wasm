# Threats to validity

If you intend to claim a win from tvm-bench results, you must address each
of these. Ignoring them is how the results get dismissed as workload bias.

## Engine-specific optimization artifacts

Wasmtime, Wasmer, and V8 each apply different bounds-check elimination,
register allocation, and memory layout strategies. A win on one engine may
not generalize.

**Mitigation:** run the suite on at least two independent engines before
claiming a hypothesis is supported. Wasmer and V8 backends are deferred
but planned (see BACKLOG.md). Until they land, every claim is tagged
"wasmtime-only."

## Bounds-check elimination variance

Both M32 and M64 may have their bounds checks elided when the engine can
prove safety. TVM goes through host-trusted code paths and never sees the
engine's bounds-check optimizer. Comparing TVM read/write performance to
M32/M64 is therefore comparing **two different things**: a host-side scratch
copy vs. an engine-emitted memory access.

**Mitigation:** report TVM and M32/M64 results separately and label the
comparison as "host-mediated TVM vs. native linear memory." Any "TVM wins"
claim must explicitly acknowledge the architectural difference, not pitch
it as apples-to-apples.

## wasm64 maturity

The wasm64 spec is still evolving. Toolchain support (wasm-bindgen, LLVM
backend) is younger than wasm32 and likely emits suboptimal code in places.
A measured M64 slowdown could be a transient toolchain artifact rather than
a fundamental cost of 64-bit indexing.

**Mitigation:** record the LLVM/Rust toolchain version with every M64
measurement. Re-run on at least two toolchain releases before claiming H1.

## Synthetic benchmarks vs. real workloads

The 7 classes are synthetic. They exercise specific mechanisms but no one
of them mirrors a production workload.

**Mitigation:** the framework is designed for orthogonal stress, not
end-to-end production simulation. Claims should be of the form "TVM wins
in workloads that are pointer-heavy *and* multi-region" — not "TVM is
faster than M64."

When a real workload becomes available (FijiVM, DataFission), wire it into
the suite as an additional class and treat it as the validation point.

## Cache-effect variance

Cache and TLB behavior depends on the host CPU. Apple Silicon, Intel
server-class, and AMD Zen all cache differently.

**Mitigation:** every result must record CPU model, L1/L2/L3 sizes, and
TLB count. Cross-architecture replication is required before any locality
claim.

## RNG / seed bias

A randomized access pattern with a poorly chosen seed can advantage one
memory layout over another by accident.

**Mitigation:** every randomized class runs with at least 3 distinct seeds.
Report results per seed; if seeds disagree, the comparison is invalid.

## Warm-up bias

JIT engines (V8) and AOT engines (wasmtime, wasmer) have different
warm-up profiles. Recording a "first call" measurement penalizes JITs;
recording only "steady state" hides startup costs.

**Mitigation:** report **both** cold-start and steady-state numbers
separately. Don't aggregate.

## Allocator-strategy bias on TVM

TVM's `bump` vs `freelist` vs `slab` allocator selection materially affects
per-call latency. A TVM benchmark must declare which allocator is in use.

**Mitigation:** every TVM result tags the allocator. When comparing to
M32, the M32 workload must use the same allocation strategy on its own
linear memory (e.g. a hand-rolled bump pointer for fairness).

## Sample-size and confidence

A single benchmark run is meaningless. Microbenchmarks need enough samples
to bound the confidence interval; coarse-grained workloads need enough
runs to average out OS scheduler noise.

**Mitigation:** the runner enforces a minimum sample count (criterion
default of 100) and reports the 95% confidence interval with every mean.
Don't report a result whose CI overlaps with the variant being compared.

## Reporting bias

Cherry-picking the workloads where TVM wins is the most common attack
vector. Even an honest experimenter slides into this if they choose what
to highlight.

**Mitigation:** publish all (variant × class × size) combinations, not
just the favorable ones. The README and BACKLOG must list every benchmark
in the matrix; missing classes are deferrals, never silent omissions.
