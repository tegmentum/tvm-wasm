# Architecture

This describes the system as built. The original aspirational specification
lives in version control; this document is the authoritative description of
what's actually shipping.

## Layering

```
                ┌─────────────────────────────────────────────┐
                │   guest wasm component / core wasm module   │
                └─────────────────────────────────────────────┘
                          ↓ WIT bindings              ↓ raw imports
   ┌──────────────────────────────────────┐  ┌──────────────────────┐
   │ TvmHost / SharedTvmHost /            │  │ tvm-wasmtime::       │
   │ ConcurrentTvmHost                    │  │ raw_linker           │
   │   • impl manager/bytes/diagnostics   │  │   • tvm.{alloc,read, │
   │   • auto-fault on Cold + backing     │  │     write,copy_…}    │
   │   • cache invalidation               │  │                      │
   └──────────────────────────────────────┘  └──────────────────────┘
                              ↓
   ┌─────────────────────────────────────────────────────────────────┐
   │ tvm-core::RegionDirectory       (single-threaded, &mut self)    │
   │ tvm-core::ConcurrentDirectory   (per-region locks, &self)       │
   │   • regions + warm_lru                                          │
   │   • create / alloc / dealloc / read / write                     │
   │   • promote_region / demote_region / spill_region / load_region │
   │   • compact_region (RegionDirectory only — returns HandleRemap) │
   │   • read_or_fault / write_or_fault (auto-load Cold)             │
   │   • cross_region_copy / read_into / write_from                  │
   └─────────────────────────────────────────────────────────────────┘
        ↓                   ↓                    ↓
 ┌────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
 │ MemoryRegion   │  │ RegionAllocator  │  │ BackingStore            │
 │ (host-managed) │  │ Bump | Freelist  │  │ FileBackingStore        │
 │ VecBackedRegion│  │ | Slab           │  │ SingleFileBackingStore  │
 └────────────────┘  └──────────────────┘  └─────────────────────────┘
        ↑                                       ↑
 ┌────────────────────────────────┐
 │ RuntimeMemoryRegion<Cx>        │  ← runtime-bound (wasmtime memory)
 │ + spill/load helpers           │
 └────────────────────────────────┘
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
- **Warm**: resident but eligible for spill. Tracked in an LRU.
- **Cold**: spilled to a `BackingStore`. Reads via `read_or_fault` auto-load;
  reads via plain `read` return `NotResident`.
- **External**: future hook for non-local stores. Not implemented.

`evict_warm_region(&mut backing)` pops the oldest non-pinned Warm region and
spills it. Designed as the foundation for memory-pressure-driven eviction.

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
