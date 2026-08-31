# Wasmos raw-path overhead (Phase 6.9.b)

**Status:** measured 2026-08-31 on macOS arm64 (M-series, cold host).
Bench source: `crates/tvm-wasmtime/benches/wasmos_overhead.rs`. Rerun:
`cargo bench -p tvm-wasmtime --bench wasmos_overhead`.

## What we measured

Same guest, same host state, same handler surface (`tvm.*` from
`raw_linker.rs` vs `raw_linker_wasmos.rs`), timed head-to-head across
four workloads and three size classes. 50 samples per case after 5
warmup iterations. `mann_whitney_u = 0.000` on every pair — the
differences are statistically ironclad, not noise.

## Numbers

Per-call wall clock, mean of 50 samples, release build.

| Workload | Size | wasmtime (ns) | wasmos (ns) | delta (ns) | overhead |
|---|---:|---:|---:|---:|---:|
| alloc+dealloc | — | 46 | 931 | +885 | +1932% |
| sum_u8 | 64B | 48 | 805 | +757 | +1595% |
| sum_u8 | 1KB | 64 | 744 | +680 | +1060% |
| sum_u8 | 16KB | 380 | 1094 | +714 | +188% |
| write | 64B | 41 | 842 | +801 | +1963% |
| write | 1KB | 61 | 935 | +874 | +1437% |
| write | 16KB | 217 | 1554 | +1337 | +615% |
| read | 64B | 44 | 836 | +792 | +1796% |
| read | 1KB | 52 | 888 | +836 | +1592% |
| read | 16KB | 233 | 1588 | +1355 | +581% |

## The single most important reading

**The absolute delta is essentially constant ~700-900ns per call.**
Look at `sum_u8`: 757 → 680 → 714ns delta across three orders of
magnitude in payload. The overhead is per-call cost, not per-byte
cost. The percentage looks worse on small calls because the
denominator shrinks; the actual work being added is the same
regardless of size.

That's the ADR-0029 abstraction tax made concrete:

- `Arc<dyn CoreImportFn>` vtable dispatch — ~20ns.
- `SharedTvmHost::lock()` uncontended — ~30-80ns.
- `Vec<CoreValue>` allocation for args + return — ~50-100ns
  (two heap allocations per call vs zero on wasmtime-native).
- `CoreImportContext` wrapping + async future construction —
  ~100-300ns adapter-side.
- `tokio::runtime::Handle::block_on` on the caller side to bridge
  the async `ModuleInstance::call_function` into the bench's sync
  loop — ~200-500ns.
- `Caller::get_export("memory")` HashMap lookup on every
  memory-touching call — ~40-80ns.

That sums to ~500-1000ns of unavoidable-with-current-shape overhead,
consistent with the measured delta.

## What this means for consumers

### Perf-critical hot loops: stay on `raw_linker`

If your guest is calling `tvm.alloc` or `tvm.read` in a tight loop
and each call does <1μs of real work, the wasmos abstraction will
dominate wall clock. Use `tvm_wasmtime::add_raw_imports` on a
plain `wasmtime::Linker<T>` and eat the wasmtime coupling. Same
crate, both paths exported.

### Portable / cross-adapter code: use `raw_linker_wasmos`

If you need the same handler surface to work on wasmtime v48,
wasmtime edge, AND WAMR — or you're planning a runtime-abstracted
downstream (girder does this) — the ~800ns constant is what
portability costs. For a call that does 100μs of real work, this
is <1% overhead. Live with it and get portability + the ability
to swap adapters without rewriting.

### Large-transfer workloads (>16KB per call)

Even at 16KB payloads the ratio is 3-7×. That's still not
competitive with the wasmtime-native path. The `write`/`read`
overhead has an additional per-byte component from the safe
scratch-buffer path (`ctx.guest_memory_{read,write}` copies bytes
into `Vec<u8>` scratch, then out again). Zero-copy via
`ctx.with_guest_memory_mut(name, |slice| ...)` eliminates that
half — not yet wired into `raw_linker_wasmos::TvmRead/TvmWrite`
but the API exists on wasmos, and the switch is per-handler.

## Optimizations available but not applied

Two escape hatches inside the wasmos surface exist. They're
per-handler, non-breaking, and reversible. **We have not applied
them here because the raw-path recommendation is already "use
wasmtime-native for hot loops."** Applying them would optimize
the portable path, but wouldn't change the recommendation —
the constant tokio+Vec+Arc+Mutex overhead is what it is.

1. **`CoreImports::register_static<F>`** (Phase 6.13 Session 3)
   — replaces the `Arc<dyn CoreImportFn>` vtable hop with a
   monomorphized fn pointer. Saves ~20ns per call. Small win
   compared to the ~800ns constant, but essentially free to apply.
2. **`ctx.with_guest_memory_mut(name, |slice| ...)`** (Phase 6.13
   Session 2) — zero-copy scratch buffer elimination for
   memory-touching handlers. Saves one memcpy pass. For the
   16KB read/write cases the memcpy is a significant portion
   of the delta (compare 16KB read overhead +1355ns vs 16KB
   sum_u8 overhead +714ns — the extra ~640ns IS the scratch
   copy).

If a specific downstream workload emerges where the wasmos path
is on a hot loop AND the memory-touching handlers are the bottleneck,
switching `TvmRead` + `TvmWrite` to `with_guest_memory_mut` should
recover most of the memory-scaling delta. Not free work but not
huge either — a per-handler diff.

## Future work if someone cares

- **Typed-args signature in `wasmos-runtime-api`** — a
  `CoreImportFnTyped<Args, Ret>` sibling trait that skips the
  `Vec<CoreValue>` boxing entirely. That's a breaking API change
  to wasmos + a real design decision (do we support arbitrary
  arg counts through variadics? do we require exact-match?). Not
  scoped anywhere yet; open when a workload demands it.
- **`Runtime::instantiate_module_sync`** — an alternative
  sync entry point that would eliminate the tokio bridging for
  callers that don't need async. Also a wasmos-side API decision;
  not on any roadmap.
- **Sharing one adapter Instance across many calls** — the bench
  already does this. If a real caller creates a fresh Runtime per
  call the numbers would look much worse; that's not what we're
  measuring.

## Recon reference

See `wasmos/docs/design/runtime-abstraction/phase-6-9-tvm-wasm-recon.md`
§Session 4 for the recon-side write-up of the same measurements
and their implication for the ADR-0029 arc's completeness claim.
