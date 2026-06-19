# Architecture

This describes the system as built. The original aspirational specification
lives in version control; this document is the authoritative description of
what's actually shipping.

## What TVM is (and isn't)

**TVM's load-bearing primitive is region-and-handle memory management
with multi-pool composition.** That delivers:

- Working sets > 4 GiB by composing multiple 32-bit-indexed memories.
- Hardware-isolated regions (engine bounds checks per pool).
- Generation-checked stale-handle safety.
- Region lifecycle (pin, compact, residency tier transitions).

**Spill to disk is an optional layer built on top of TVM**, not a core
capability. The `BackingStore` trait + `FileBackingStore` impl ship in
`tvm-core` as a convenience for embedders who want them, but a TVM
deployment doesn't need spill to be a TVM deployment. The pure guest-
side `tvm-guest-mm` crate is a complete TVM implementation with no
spill at all.

## Two deployment models

TVM has two flavors that share types and architecture:

| Crate | Where it runs | Use when |
|---|---|---|
| `tvm-wasmtime` (host-side) | Host process running wasmtime; guest accesses via imports | You control the wasm host. Useful for: spill-to-disk, multi-tenant accounting, cross-component sharing, observability into guest memory state from outside the guest. |
| `tvm-guest-mm` (guest-side) | Self-contained inside the wasm module; no host imports needed | Browser deployments, sandboxed platforms, anywhere you can't extend the host. Same region/handle abstraction; no I/O capabilities (no spill). |

Both flavors give you the core TVM properties (multi-pool >4 GiB
scaling, region lifecycle, stale-handle safety). Choose by deployment
constraint, not by feature.

## Layering

```
   ┌──────────────────────────────────────────────────────────────┐
   │             host-side TVM (server runtimes)                  │
   │  ┌────────────────────────────────────────┐                  │
   │  │ TvmHost / SharedTvmHost /              │                  │
   │  │ ConcurrentTvmHost                      │                  │
   │  │   • WIT manager/bytes/diagnostics      │                  │
   │  │   • raw_linker tvm.{alloc,read,write}  │                  │
   │  │   • imported regions (TVM-MM Unified)  │                  │
   │  │   • OPTIONAL: BackingStore for spill   │                  │
   │  └────────────────────────────────────────┘                  │
   └──────────────────────────────────────────────────────────────┘
                              ↓ both build on
   ┌──────────────────────────────────────────────────────────────┐
   │  tvm-core: regions + handles + allocators                    │
   │    • RegionDirectory / ConcurrentDirectory                   │
   │    • create / alloc / dealloc / read / write                 │
   │    • Bump / Freelist / Slab allocators                       │
   │    • residency tiers + LRU + compaction                      │
   │    • generation-checked handle validation                    │
   └──────────────────────────────────────────────────────────────┘
                              ↑ also built on
   ┌──────────────────────────────────────────────────────────────┐
   │             guest-side TVM (browser / sandboxed)             │
   │  ┌────────────────────────────────────────┐                  │
   │  │ tvm-guest-mm                           │                  │
   │  │   • Self-contained WAT modules          │                  │
   │  │   • N internal wasm memories (pools)   │                  │
   │  │   • Generated dispatch helpers          │                  │
   │  │   • Reuses tvm-core types               │                  │
   │  │   • NO spill (toolchain has no I/O)    │                  │
   │  └────────────────────────────────────────┘                  │
   └──────────────────────────────────────────────────────────────┘

  ┌─────────────────── OPTIONAL LAYERS ABOVE TVM ──────────────────┐
  │ BackingStore impls (FileBackingStore, S3, etc.)                │
  │   "spill cold regions to slower-but-larger storage"            │
  │   ← embedder's policy decision, not TVM's responsibility       │
  │                                                                │
  │ Memory pressure controller / multi-tenant scheduler            │
  │   ← also embedder's responsibility                             │
  └────────────────────────────────────────────────────────────────┘
```

## Core abstractions

### Region

A typed memory segment. Identified by a `region_id: u16` assigned at
creation. Carries a `generation: u16` that bumps on compaction; old
handles fail validation immediately after their generation goes stale.

A region's behavior is constrained by a `PlacementPolicy`:

- `initial_residency`: tier on creation (Hot or Warm)
- `pinnable`: whether `pin()` is allowed
- `spillable`: whether `spill_region` / `demote_region` are allowed

`PlacementPolicy::for_kind` provides defaults per `RegionKind`:

| RegionKind | initial_residency | pinnable | spillable |
|---|---|---|---|
| HotHeap, CodeCache, DeviceState | Hot | yes | no |
| ObjectArena, BlobArena | Hot | no | yes |
| PageStore | Warm | no | yes |
| Scratch | Hot | no | no |

### Handle

```rust
#[repr(C)]
struct Handle {
    region_id: u16,
    generation: u16,
    offset: u32,
}
```

64 bits total when packed, designed to cross FFI/WIT boundaries cleanly. A
handle is opaque to the caller — it cannot point into another region's
memory and it cannot survive a generation bump.

`pack() / unpack()` for raw fast-path use; the WIT path uses the structured
form.

### Residency tiers

```
   Hot ──demote──▶ Warm ──demote──▶ Cold ──external storage──▶ External
    ▲                ▲                │ │
    │                │                │ │
    └────promote─────┴────promote─────┘ │
    └────────────────promote─────────────┘
```

- **Hot**: resident, normal access.
- **Warm**: resident but eligible for eviction by an external policy.
  Tracked in an LRU. The "eligibility" is a hint — TVM provides the LRU
  + state machine, the embedder decides what to do with it.
- **Cold**: a region whose bytes have been moved out of TVM's resident
  storage. *How* and *where* they went is the embedder's concern via
  the optional `BackingStore` layer. From TVM's perspective, "Cold"
  just means "the region's bytes aren't currently in our pool — if you
  want to read it, you (or your `BackingStore` layer) need to bring
  them back."
- **External**: reserved for callbacks (`ExternalLoader`) that fetch
  bytes from non-local sources on demand. Same shape as Cold but with
  a different recovery mechanism.

`evict_warm_region(&mut backing)` is provided **as a convenience** for
embedders who use `BackingStore`; it pops the oldest non-pinned Warm
region and spills it. The embedder's pressure-handling policy lives
above TVM, not inside it.

**`tvm-guest-mm` doesn't use the Cold tier** *in a bare wasm engine*.
Pure-guest deployments have no I/O capability, so cold regions can't go
anywhere. Regions in guest-mm are always Hot or Warm (LRU eligibility for
application-level policy decisions, not for actual eviction).

### Browser Cold tier (`web/tvm-web`)

A browser embedding *does* have a storage target — the Origin Private File
System (OPFS) — so the `web/tvm-web` host adds a real Cold tier on top of an
unmodified guest module. The guest `.wasm` is exactly what
`tvm-guest-mm` emits: N exported pool memories (`mem0..memN`) plus the
exported dispatch helpers. Everything else lives in TypeScript:

- **The directory/policy is JS-owned.** `web/tvm-web` ports `GuestDirectory`
  placement, the `RegionDirectory` residency/LRU machine, and
  `PlacementPolicy` into TS. It reads/writes pool bytes directly through
  `WebAssembly.Memory.buffer`, so the guest needs no introspection ABI.
- **Spill target is OPFS.** A Web Worker holds `FileSystemSyncAccessHandle`s
  (the only synchronous, high-throughput OPFS path) and stores one file per
  region snapshot, `region-{id}-gen-{gen}.bin` — the same key shape as the
  host-side `FileBackingStore`.

Two constraints shape the design and are worth stating plainly:

1. **No transparent fault.** Native wasm loads cannot trap, so a spilled pool
   cannot be faulted back in mid-access (unlike the host path's
   `read_or_fault`). The browser Cold tier is therefore **cooperative**: the
   host spills only unpinned, spillable regions while the guest is quiesced,
   and the application must `await tvm.promote(region)` before touching a
   region again. `read`/`write` against a Cold region throw `NotResident`.
2. **Eviction reclaims address space, not pages.** wasm linear memory cannot
   shrink, so spilling does not hand pages back to the OS. Its value is that a
   Cold region's pool span is returned to a free-list and **reused** by a
   later allocation while the bytes sit safely in OPFS. The resident working
   set is thus bounded by an LRU byte budget (reuse) rather than by RAM, and
   total capacity is bounded by the OPFS storage quota.

An optional synchronous load path (`SharedArrayBuffer` + `Atomics.wait`, for a
guest that must reload inside a tight loop without yielding) is provided as a
channel primitive; it requires the guest to run in a Web Worker and the page
to be cross-origin isolated.

This closes the browser scaling gap: addressing already scaled to the module's
N × 4 GiB, but the *backable* working set was capped by the tab's RAM budget.
With the OPFS Cold tier it is capped by disk quota instead.

## Allocators

Each region owns its allocator. Three implementations:

| Kind | alloc cost | dealloc | Fragmentation | Use when |
|---|---|---|---|---|
| `Bump` | O(1), no free | no-op | n/a | one-shot scratch, generation-scoped state |
| `Freelist` | first-fit | O(n) coalesce | possible | mixed-size allocations with reuse |
| `Slab { class_size }` | O(1) | O(1) (linear no-double-free check) | none by construction | uniform-size pools |

Compaction is supported only for `Freelist` (the only allocator that tracks
allocations by offset). Returns a `HandleRemap` mapping old offsets to new
offsets; bumps the region's generation; old handles fail validation.

## Two paths from guest to host

### WIT (component model)

```
guest → wit-bindgen-generated lower
      → component-model trampoline
      → wasmtime canonical ABI lift
      → host trait impl
      → RegionDirectory
```

Type-safe, multi-language portable. Cost dominated by the canonical ABI for
list-shaped returns (e.g. `bytes.read` lifts a `Vec<u8>` through guest
linear memory). Use for control plane and any call where the cost is
amortized across many bytes per call.

### Raw imports

```
guest → extern "C" call
      → wasmtime::Linker::func_wrap
      → host closure
      → RegionDirectory
```

Plain `(i32 | i64) → i32 | i64` core-wasm calls. One scratch copy on the
host side, no canonical ABI. ~4× faster on small payloads. Errors surface as
i32 codes; `tvm.last_error` recovers the code from sentinel-value returns.

Both paths share the same `TvmHost` and the same underlying directory — a
guest can use both in the same store.

## Multi-store sharing

Two flavors, picked according to contention shape:

- **`SharedTvmHost = Arc<Mutex<TvmHost>>`** — single global mutex. Cheaper
  under no contention; serializes all calls. Use for simple multi-store
  setups where most calls hit the same regions or where contention is rare.
- **`ConcurrentTvmHost`** — backed by `ConcurrentDirectory`, which gives
  per-region locking. The regions vector sits under an outer `RwLock` for
  membership changes; each region's `RegionEntry` is behind its own
  `Mutex`. Operations on **different regions** run in parallel.
  Multi-region operations (`cross_region_copy`) acquire per-region locks
  in ascending `region_id` order to avoid deadlock. Use when stores
  typically hit distinct regions per call.

Both flavors are `Clone`-able and implement the same `Host` traits, so
swapping them is a one-line change in the linker setup
(`add_shared_to_linker` ↔ `add_concurrent_to_linker`).

`ConcurrentTvmHost` does not yet implement compaction (which would need a
write-lock on the outer regions vector and an in-place memory swap); call
`compact_region` only on `TvmHost` / `SharedTvmHost`.

## Backing stores

`BackingStore` trait:
```rust
fn spill(region_id, generation, bytes) -> Result<()>;
fn load(region_id, generation) -> Result<Vec<u8>>;
```

Two implementations: `FileBackingStore` (one file per region, named by id +
generation) and `SingleFileBackingStore` (one named file, useful for
snapshotting one region at a time).

Wasmtime memories use the `RuntimeMemoryRegion<Cx>` trait and the
`spill_runtime_region` / `load_runtime_region` helpers — they need a
`Store` context to access bytes, so they don't go through the
`RegionDirectory` API directly.

## ResolveCache

Tiny 8-slot direct-mapped cache keyed by `region_id & 7`. Stores
`(generation, capacity, residency_hot)`. Invalidated automatically on
mutating ops (`destroy_region`, `spill_region`, `load_region`,
`compact_region`).

The cache makes the per-call validation a branch+compare instead of a
Vec lookup. Counters (`hits`, `misses`, `invalidations`) are exposed for
tuning region IDs to avoid collisions.

## What's not yet built

- **Compaction on `ConcurrentTvmHost`.** Compaction needs to swap a region's
  underlying memory and bump its generation atomically; under per-region
  locks this also wants the outer regions write-lock, which existing
  callers don't expect to wait on. Use `TvmHost` / `SharedTvmHost` if you
  need compaction.
- **Wasmtime spill via the directory.** Today, spilling a wasmtime-backed
  region uses the runtime-aware helpers (`spill_runtime_region` /
  `load_runtime_region`) and is not unified with `RegionDirectory`'s
  spill/load. The unified path would need `MemoryRegion` to thread a
  context type, which is a heavy refactor we've deferred.
- **External tier.** The fourth `Residency` value is reserved but nothing
  implements it.
