# tvm-bench backlog

Every deferred piece of the matrix has an explicit slot here. A claim
"the framework supports X" is only true if X has a working implementation
*or* an open backlog item.

## Backends

- [ ] **M64 backend** — needs `wasm64-unknown-unknown` Rust target +
  toolchain pinning. Workload source must be `cfg(target_pointer_width =
  "64")` aware. Estimated effort: medium; main risk is LLVM emitting
  suboptimal i64 code that masks the actual cost of 64-bit indexing.

## Runtimes

- [x] **Wasmer backend** — `tvm-bench-runner-wasmer` runs the M32 baseline
  workloads on the Wasmer engine as a cross-engine validator so the
  wasmtime M32/M64 numbers aren't taken as engine-specific artifacts.
  TVM variants stay wasmtime-only because they depend on the
  raw_linker host functions.
- [ ] **V8 backend** — node.js harness or a C++ shim. JIT warm-up needs
  separate cold/steady reporting (see THREATS.md).

## Benchmark classes

Each one mirrors a numbered section in the design doc. All workloads
share the abstraction in `workloads/m32/src/lib.rs`'s `Workload` trait;
adding a class means implementing that trait for each backend.

- [x] 4.1 Sequential access
- [ ] **4.2 Random access** — pointer-chasing pattern over a working set
  larger than L3. Stresses TLB and cache. The TVM variant should use
  multiple regions to test whether smaller bounded address spaces
  improve TLB locality.
- [ ] **4.3 Pointer-heavy structures** — linked list / binary tree / hash
  map workloads. Predicts H1 (M64 penalty for larger pointers). TVM
  variant should use packed handles (`u64` pack of `region_id + offset`)
  vs M64's bare `i64` pointer.
- [ ] **4.4 Allocation / growth stress** — repeated alloc + dealloc with
  growing footprint. TVM variant exercises freelist coalescing; M32/M64
  variants must use a comparable arena allocator (hand-rolled bump or
  reuse one from the standard ecosystem).
- [ ] **4.5 Multi-region workload** — *the critical proof for TVM.*
  Partitions the working set into `hot` / `warm` / `cold` regions. TVM
  maps each to a separate memory; M32/M64 share one address space with
  some discipline (offsets reserved per tier). The hypothesis is that
  TVM's smaller bounded memories improve cache locality on the hot path.
  See section 4.5 of the design doc.
- [ ] **4.6 Database-like access** — columnar scans + indexed lookups.
  Aligns with DataFission. TVM keeps each column in its own region; M64
  layouts them sequentially. Predicted advantage: TVM avoids the
  cross-column false-sharing pattern.
- [ ] **4.7 JVM heap simulation** — generational allocation pattern.
  Aligns with FijiVM. Young-gen on a hot region, old-gen demoted to a
  warm region.

## Instrumentation

- [x] Wall-clock timing (criterion).
- [ ] CPU cycles (`cargo-criterion --instruments` on Apple, `perf stat`
  on Linux).
- [ ] Cache miss / TLB miss rates via `perf` integration.
- [ ] Branch misprediction rates.

## Reporting

- [x] Per-(variant × class × size) JSON output.
- [ ] Cross-engine comparison plots.
- [ ] Significance tests (Mann-Whitney U) for variant comparisons.
- [ ] Locality-efficiency derived metric (observed throughput / peak DRAM).

## Determinism

- [x] Fixed seed.
- [x] Identical dataset sizes per variant.
- [ ] CPU pinning + governor settings (Linux: `performance` governor;
  macOS: not exposed, document as a threat).

## Workflow

- [x] Single source for the workload abstraction.
- [ ] Backend-specific guest crates auto-built by the runner before each
  invocation (currently requires manual `./build.sh`).
- [ ] CI integration: nightly benchmark runs, regression detection
  against committed baselines.
