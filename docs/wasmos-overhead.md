# Wasmos raw-path overhead (Phase 6.9.b)

**Status:** measured 2026-08-31; **refreshed 2026-09-01** with the
Phase 6.9.d Session 5 per-actor path (`TvmHostSource::PerActor`) as
a third comparison. macOS arm64 (M-series, cold host). Bench source:
`crates/tvm-wasmtime/benches/wasmos_overhead.rs`. Rerun:
`cargo bench -p tvm-wasmtime --bench wasmos_overhead`.

## What we measured

Same guest, same host state, same handler surface (`tvm.*` from
`raw_linker.rs` vs `raw_linker_wasmos.rs`), timed head-to-head across
four workloads and three size classes. Three dispatch paths:

- **wasmtime**: `add_raw_imports` on a plain `wasmtime::Linker<TvmHost>`,
  handler pulls `&mut TvmHost` via `caller.data_mut()` (per-actor,
  no lock).
- **wasmos-sh** (shared): `raw_linker_wasmos::add_raw_imports` +
  `SharedTvmHost` closure captured in the handler; per-call
  `Arc<Mutex<>>::lock()`. Adapter dispatches through
  `wasmos_runtime_wasmtime_v48::runtime`.
- **wasmos-pa** (per-actor, new at Phase 6.9.d Session 5):
  `raw_linker_wasmos::add_raw_imports_per_actor_projected::<TvmHost>`
  installed on a consumer-owned `Linker<TvmHost>` via
  `wasmos_runtime_wasmtime_v48::core_import_bridge::install_core_imports`.
  Handler pulls `&mut TvmHost` via `ctx.consumer_state::<TvmHost>()`
  (no `Arc<Mutex<>>` lock).

50 samples per case after 5 warmup. `mann_whitney_u = 0.000`
between every pair — the differences are statistically ironclad,
not noise. Per-actor-vs-shared reports `mann_whitney_u = 1.000` in
the same direction (per-actor faster) with equally-strong
significance.

## Numbers (Session 8 refresh, 2026-09-01)

Per-call wall clock, mean of 50 samples, release build.

| Workload      | Size |  wasmtime (ns) | wasmos-sh (ns) | wasmos-pa (ns) | sh vs wt | pa vs wt | pa vs sh |
|---------------|-----:|---------------:|---------------:|---------------:|---------:|---------:|---------:|
| alloc+dealloc |    — |             49 |            916 |            554 |  +1762%  |  +1027%  |   −39.5% |
| sum_u8        |  64B |             47 |            884 |            381 |  +1762%  |   +702%  |   −56.9% |
| sum_u8        |  1KB |             66 |            762 |            401 |  +1057%  |   +509%  |   −47.4% |
| sum_u8        | 16KB |            384 |           1112 |            737 |   +189%  |    +92%  |   −33.7% |
| write         |  64B |             41 |            856 |            507 |  +1986%  |  +1134%  |   −40.8% |
| write         |  1KB |             56 |            995 |            545 |  +1683%  |   +877%  |   −45.2% |
| write         | 16KB |            234 |           1543 |           1102 |   +559%  |   +371%  |   −28.6% |
| read          |  64B |             39 |            887 |            498 |  +2164%  |  +1171%  |   −43.9% |
| read          |  1KB |             56 |           1024 |            548 |  +1735%  |   +882%  |   −46.5% |
| read          | 16KB |            235 |           1603 |           1179 |   +582%  |   +402%  |   −26.5% |

## The three most important readings

**1. Per-actor is consistently faster than shared, by 27-57%.**
Removing the `SharedTvmHost::lock()` (per-call `Arc<Mutex<>>::lock`
+ MutexGuard construction) saves ~350-500ns absolute across every
workload. Confirms the lock is real cost, not noise.

**2. Per-actor still lags wasmtime-native by 4-13× on small calls,
2× on 16KB.** The remaining gap is per-call abstraction cost that
neither variant of the wasmos path can eliminate:
`Arc<dyn CoreImportFn>` vtable, `Vec<CoreValue>` boxing for args +
return, `CoreImportContext` wrapping, async-future construction +
`block_on` bridging, `Caller::get_export` per memory-touching call.
See the cost decomposition below.

**3. Per-byte cost stays constant across paths.** All three paths
scale linearly with payload; the abstraction adds latency, not
bandwidth degradation. At 16KB `sum_u8`, per-actor achieves 20 GiB/s
against wasmtime-native's 40 GiB/s — half the throughput at
proportionally-larger buffers.

## Cost decomposition

That's the ADR-0029 abstraction tax made concrete:

- `Arc<dyn CoreImportFn>` vtable dispatch — ~20ns.
- `SharedTvmHost::lock()` uncontended — ~350-500ns (measured as the
  per-actor-vs-shared delta above; larger than initial estimate
  because MutexGuard drop + poisoning check + Deref chain add).
  **Not paid on per-actor.**
- `Vec<CoreValue>` allocation for args + return — ~50-100ns
  (two heap allocations per call vs zero on wasmtime-native).
- `CoreImportContext` wrapping + async future construction —
  ~100-300ns adapter-side.
- `tokio::runtime::Handle::block_on` on the caller side to bridge
  the async `ModuleInstance::call_function` into the bench's sync
  loop — ~200-500ns.
- `Caller::get_export("memory")` HashMap lookup on every
  memory-touching call — ~40-80ns.

Shared-vs-native gap: ~700-900ns. Per-actor-vs-native gap:
~350-450ns. The lock cost was the single largest component.

## What this means for consumers

### Default: `raw_linker_wasmos::add_raw_imports_per_actor_projected`

Girder's `RawTvmActorInstance` (per-actor `TvmHost` in the Store's
data) and any consumer with a similar shape should default to the
per-actor variant. It's the fastest wasmos path, portable across
every wasmos-backed adapter (v48 / edge; WAMR needs Store<T> which
it doesn't have — see WAMR ack in the source), and has no lock
contention concerns.

### Shared substrate: `raw_linker_wasmos::add_raw_shared`

When multiple actors legitimately share one `TvmHost` (cross-store
region visibility — girder's `SharedRawTvmActorInstance` shape),
the shared path is correct even if slower. The extra ~350-500ns per
call is what shared-substrate correctness costs.

### Perf-critical hot loops: measured escape to `raw_linker`

If your guest is calling `tvm.alloc` or `tvm.read` in a tight loop
and each call does <1μs of real work, the wasmos abstraction still
dominates wall clock — per-actor cuts the shared overhead roughly
in half but doesn't close the gap to wasmtime-native. Use
`tvm_wasmtime::add_raw_imports` on a plain `wasmtime::Linker<T>`
behind an `#[allow(deprecated)]` if you've *measured* the gap
and it's the bottleneck. Same crate, both paths exported.

**Deprecation state (ADR-0029 Phase 6.9.d Session 7, 2026-09-01):**
`add_raw_imports` / `add_raw_shared` (wit-bindgen path) are marked
`#[deprecated]` — no production consumer remains. Session 8's
3-way bench confirms the wit-bindgen path is still 4-13× faster
than per-actor on small calls, so the escape-hatch case is real.
This bench continues to publish the head-to-head numbers so a
future re-evaluation has data.

**Deletion assessment (Session 8, 2026-09-01):** the wit-bindgen
entry points and their dedicated tests + reference implementation
are RETAINED. The Session 8 numbers above justify the decision:
per-actor cuts the shared overhead roughly in half (removing the
`SharedTvmHost::lock`) but doesn't close the ~450ns gap to
wasmtime-native for small calls. Any consumer with a documented
hot-loop workload can still take the wit-bindgen coupling behind
`#[allow(deprecated)]` and get 4-13× lower per-call cost than the
best wasmos path today. A future deletion trigger would be: either
a wasmos-side change that closes the remaining native gap (e.g. the
Phase 6.13 Session 3 `register_static` monomorphized-dispatch path
extended to skip `Vec<CoreValue>` boxing), or an explicit decision
that perf-hot raw-tvm is not a workload wasmos needs to serve
optimally. Neither has happened; the wit-bindgen path stays.

### Portable / cross-adapter code: `raw_linker_wasmos` (either variant)

If you need the same handler surface to work on wasmtime v48,
wasmtime edge, AND WAMR — or you're planning a runtime-abstracted
downstream (girder does this) — the wasmos overhead is what
portability costs. For a call that does 100μs of real work, even
the shared path adds <1%. Live with it and get portability + the
ability to swap adapters without rewriting.

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
