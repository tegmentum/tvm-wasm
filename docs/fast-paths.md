# TVM fast paths

There are three ways a guest can talk to the TVM host. Pick the one that
matches what the call is doing — the wrong choice can cost you a 5-10x
slowdown on hot loops or, conversely, force you into hand-written unsafe FFI
where a slower call would have been fine.

| Path | When to use | Cost per call |
|---|---|---|
| **WIT bindings** (`tvm:memory/manager`, `bytes`, `diagnostics`) | Setup, control plane, cross-language portability, anything called once or rarely | Component-model canonical ABI: ~2 copies + return-area lift/lower for `list<u8>` returns |
| **WIT region-to-region** (`bytes.read-into`, `write-from`, `copy-region`) | Bulk byte movement that stays inside TVM regions | Same ABI overhead as above, but **no copy through guest linear memory** |
| **Raw imports** (`tvm.alloc/dealloc/read/write/copy_region`) | Hot per-element loops, large transfers between region and guest, throughput-sensitive paths | Plain core-wasm call: 1 copy through a host scratch buffer |

The `tvm-guest-rt` crate gives a safe Rust wrapper over the raw imports so
you don't have to write `extern "C"` yourself.

---

## What the WIT path costs

Every `bytes.read(handle, len) -> result<list<u8>, tvm-error>` call:

1. Guest allocates a return area in its linear memory.
2. Host runs the function and produces a `Vec<u8>`.
3. Component-model ABI **lifts** the Vec into the return area (one copy).
4. Guest **lowers** it from the return area into a local `Vec<u8>` (second copy
   when the guest binding clones — most do).
5. Result-variant tag dispatch on the guest side.

For 4-byte payloads on a hot loop, steps 1-5 dominate. For 64KB transfers, the
double copy is the dominant cost.

There is also a per-call trampoline in component-model that wraps every host
call — small (low hundreds of ns), but it does add up.

## What the raw imports do differently

Raw imports skip the component model entirely. `tvm.read(handle, dst_ptr, len)`
is a plain `(i64, i32, i32) -> i32` core-wasm call:

1. Host reads `len` bytes from the region into a scratch `Vec`.
2. Host writes the scratch buffer into guest linear memory at `dst_ptr`.

One scratch copy on the host side. The guest never sees a `Vec`, never lifts
or lowers anything, never inspects a result variant — errors come back as a
single `i32` code (0 = ok).

**Tradeoffs:**

| | WIT | Raw |
|---|---|---|
| Type safety on the wire | Variant + result types, compile-time checked | `i32` error codes |
| Multi-language portability | Yes (Python, JS, Go, etc. via wit-bindgen) | Rust + C only realistically |
| Toolchain | `wit-bindgen` or `cargo component` | `extern "C"` blocks (or `tvm-guest-rt`) |
| Cost per byte transferred | ~2 copies | ~1 copy |
| Cost per call (constant) | Trampoline + ABI handling | Plain function call |
| Failure mode on ABI mismatch | Fails to instantiate (visible) | Silent data corruption (unsafe FFI) |

## What region-to-region copies do differently

`copy-region(src_region, src_off, dst_region, dst_off, len)` never touches the
guest's linear memory at all. The bytes go region → host scratch → region.
This is **always** faster than `read` + `write` from the guest, because it
avoids the round-trip through guest memory in both directions.

If the bytes are eventually going to be processed by guest code, you still
need `read` to bring them in. But for "move data from page-store region to
column region," `copy-region` is the right call.

---

## Recommendation by call site

**Setup, lifecycle, diagnostics** — use WIT.

```rust
// Once at startup. Latency doesn't matter here.
let region = manager::create_region(RegionKind::HotHeap, 4096)?;
manager::pin(region)?;
```

**Hot inner loop** — use raw imports via `tvm-guest-rt`.

```rust
use tvm_guest_rt::{Region, RegionPtr};

let region = Region::from_id(known_region_id);
for chunk in chunks {
    let h = region.alloc(chunk.len() as u32)?;
    h.write(chunk)?;
    process(h);
}
```

**Cross-region pipeline** — use the region-to-region variant from WIT (or
`copy_region` from the raw path). Don't `read` then `write`.

```rust
// Shovel bytes from the page-store region into the staging region without
// ever materializing them in guest linear memory.
bytes::copy_region(page_store_region, page_offset, staging_region, 0, len)?;
```

---

## ResolveCache: how host-side validation gets cheap

Every host-side call validates the region exists and the handle's generation
is current. The `ResolveCache` in `TvmHost` is an 8-slot direct-mapped cache
keyed by `region_id & 7`. On hit, validation is one branch + compare. On
miss, it falls back to the `RegionDirectory` (still O(1)) and populates the
slot.

You don't interact with the cache directly. It is invalidated automatically
on `destroy_region`, `spill_region`, `load_region`, and `compact_region`. If
your workload uses many regions and they collide on the same slot, the cache
degrades to "always-miss" but never returns wrong data.

`host.cache.hits` / `host.cache.misses` / `host.cache.invalidations` are
public counters useful for tuning region IDs to avoid collisions.

---

## Pitfalls

**Mixing fast and slow paths against the same handle.** Safe, but the cache
has no insight into raw-path mutations. If a raw `tvm.alloc` updates `used`,
the cache slot's `capacity` is still correct (capacity doesn't change), and
generation is unchanged. You only need to manually invalidate after operations
that bump generation (compaction does this automatically through `TvmHost`).

**Raw error codes are not stable across crashes.** `tvm.alloc` returns 0 for
NULL; `tvm.last_error` returns the last code from this `Store`. If two raw
calls run concurrently against the same Store (you can't — wasmtime stores
aren't `Sync`), the codes would race. With the standard one-Store-per-thread
pattern this isn't a concern.

**Aligned reads.** Both paths copy through scratch buffers, so neither
guarantees alignment of guest memory writes. If your guest reads as `f64`
from `dst_ptr`, ensure `dst_ptr % 8 == 0` yourself.

**Guest memory export name.** The raw linker assumes the guest exports its
linear memory as `memory`. If your toolchain uses a different name, call
`add_raw_imports_with_memory_name(linker, "your-name")`.

**Generation invalidation after compaction.** `compact_region` returns a
`HandleRemap`. Old packed handles you're carrying around in raw-FFI calls
will be reported as `STALE_HANDLE` (error code 2) until you migrate them.
The fast path doesn't auto-migrate — that would defeat the point of skipping
the component-model wrapper.

---

## Numbers (measured)

Apple Silicon, wasmtime, single-threaded, 256 KiB working set, 50 samples
each. **TVM here uses the optimized "bulk-read once, process locally" idiom**
described above; per-cell host calls would be 50–150× slower.

The realistic baselines for a >4 GiB workload are M64 (the only built-in
WebAssembly answer) and TVM (this project). Both are reported; M32 is
included as the lower bound that 32-bit-bounded workloads enjoy.

| Class | M32 baseline | M64 | TVM | TVM/M64 |
|---|---:|---:|---:|---:|
| Sequential sum | 47 µs | **2547 µs** | 45 µs | **56.6× faster than M64** |
| Random chase | 2.0 ms | 2.7 ms | 2.6 ms | 1.05× |
| Linked-list walk | 147 µs | 651 µs | 252 µs | **2.6× faster** |
| Bump-alloc + touch | 3 µs | 76 µs | 39 µs | **2.0× faster** |
| Multi-region 90/9/1 | 32 µs | 155 µs | 37 µs | **4.1× faster** |
| Columnar filter+sum | 16 µs | 486 µs | 28 µs | **17.1× faster** |
| JVM gen-alloc-scan | 6 µs | 157 µs | 41 µs | **3.8× faster** |
| Large WS probe | 26 µs | 107 µs | 34 µs | **3.1× faster** |
| Spill-driven | infeasible | infeasible | 0.5 µs/cycle | **TVM only** |

### What the data says

1. **M64 has a real per-instruction tax on wasmtime.** Its sequential cost
   is 55× the M32 cost — the address-width hypothesis (H1) is confirmed
   for this engine. M32 + bounded 32-bit working sets beat M64 even on
   tiny payloads.

2. **TVM is a faster alternative to M64 on most workloads.** On the
   workloads where the working set genuinely needs >4 GiB, M64 is the
   only built-in WebAssembly answer — and TVM beats it by 2–35×, because
   TVM keeps each region 32-bit-indexed internally.

3. **TVM is not a faster alternative to M32 for inner loops.** Per-byte
   cost still exceeds direct linear-memory access because every host
   call has overhead. Use M32 directly when the working set fits.

4. **The growth and JVM benchmarks expose TVM's allocator overhead.**
   These workloads call `alloc()` per element. Each alloc is a host
   round-trip; that's a deliberate design choice (handles validate
   centrally) but it kills throughput on alloc-heavy workloads. A future
   `bulk_alloc` primitive that returns a contiguous range in one call
   would close this gap; tracked in BACKLOG.md.

5. **Spill-driven is TVM-exclusive.** Neither M32 nor M64 has a way to
   represent a working set that exceeds the resident memory budget;
   they would require user-implemented swap. TVM serves this through
   its tier transitions natively.

### Path-level cost (all paths, 4-byte read)

| Path | ns/call (relative) |
|---|---|
| WIT `bytes.read` | 1.0× baseline |
| WIT `bytes.copy-region` (no guest round-trip) | 0.45× |
| Raw `tvm.read` | 0.25× |
| Raw `tvm.copy_region` | 0.15× |

For 64KB transfers all paths converge — the actual byte copy dominates.
The WIT path is ~30% slower than raw at large sizes from the canonical-ABI
double-copy.

### When this changes

- **Cross-engine.** Wasmer's M32 is ~70% slower than wasmtime's M32 on
  sequential. Cross-engine replication is required before claiming any
  result generalizes; tracked in BACKLOG.md.
- **Future `bulk_alloc` primitive.** Predicted to bring TVM's growth /
  JVM costs within 2–3× of M32 instead of 100×.
- **Real CPU counters.** Cache-miss / TLB-miss data would let us validate
  H2 (locality advantage) — currently we only have wall-clock numbers.

---

## The wasmos raw path (Phase 6.9.a)

The raw path historically lived on wasmtime — `add_raw_imports` takes a
`wasmtime::Linker<T>` where `T: AsMut<TvmHost>`. That works well when your
whole pipeline is wasmtime-based, but locks the raw path to one engine.

Since ADR-0029, a second raw path lives alongside the wasmtime one, sitting
on the wasmos runtime-abstraction layer. It's in
`tvm_wasmtime::raw_linker_wasmos` and offers the same `tvm.*` import surface,
but registers handlers through `wasmos_runtime_api::CoreImports` instead of
`wasmtime::Linker`. Any wasmos-backed adapter (wasmtime v48, wasmtime edge,
WAMR) can now host the raw path.

### API shape

```rust,ignore
use tvm_wasmtime::raw_linker_wasmos::add_raw_imports;
use tvm_wasmtime::SharedTvmHost;
use wasmos_runtime_api::CoreImports;

let shared = SharedTvmHost::new();
let imports = add_raw_imports(CoreImports::new(), shared.clone());
// Thread `imports` into your ExecutionContext, then instantiate:
//   let mut ctx = ExecutionContext::new();
//   ctx.core_imports = imports;
//   let instance = runtime.instantiate_module(&compiled, ctx).await?;
```

Two entry points, both fluent-builder-style:

- `add_raw_imports(imports, host)` — memory export named `"memory"`.
- `add_raw_imports_with_memory_name(imports, host, name)` — custom name.

`add_raw_shared*` aliases exist for API symmetry with the wasmtime path but
delegate to the same implementation — the wasmos abstraction unified the
two batches (see the module docstring for why).

### Which path to pick

| Situation | Path |
|---|---|
| Wasmtime-only pipeline, single-threaded, perf-critical | wasmtime `raw_linker` (unchanged) |
| Multi-runtime dispatch (wasmtime + WAMR) or planning to portability-test | `raw_linker_wasmos` |
| Cross-store sharing (multiple actors, one region directory) | either, but `raw_linker_wasmos` handles it uniformly (no per-Store cache to invalidate) |
| Benchmarking wasmos-overhead vs wasmtime-native | run both paths side-by-side; the crate exports both |

### Perf cost of the wasmos path

The wasmtime `raw_linker` handlers get exclusive `&mut TvmHost` via
`Caller::data_mut()` — zero locking, direct memory pointer via a
cached-per-store `Memory` handle. The wasmos handlers capture a
`SharedTvmHost` (`Arc<Mutex<TvmHost>>`) at registration and grab the lock
per call; guest memory is fetched fresh through the adapter's Caller/Instance
surface every invocation.

Phase 6.9.b measured this. **The absolute overhead is ~700-900ns per call,
essentially constant across payload sizes** (measured on macOS arm64,
release build, 50 samples per case). That's the abstraction tax: `Arc<dyn>`
vtable dispatch, `Vec<CoreValue>` allocation for args + return, tokio's
`block_on` bridging into an async runtime, `Caller::get_export` HashMap
lookup for memory-touching handlers, uncontended mutex acquisition. All
real, none apparent from the API surface.

For a raw `tvm.alloc` call (~46ns wasmtime-native), that's a 20× regression.
For a 16KB `tvm.write` call (~217ns wasmtime-native), 7×. For a call that
does 100μs of real work, <1% — portable and fine.

**See `docs/wasmos-overhead.md` for the full numbers table + optimization
escape hatches.** Short version: if you're doing hot-loop
call-per-element with tiny work, stay on `raw_linker`. If you need
portability across adapters and can spend an extra microsecond per call,
`raw_linker_wasmos` is the shape.

### What's covered

`raw_linker_wasmos` registers all 26 handlers of the wasmtime non-shared
batch:

- `alloc`, `dealloc`, `copy_region`, `last_error` — region-lifecycle
- `read`, `write`, `read_gather`, `index_of`, `byte_histogram` — memory
- `sum_u8`, `find_byte`, `hash_fnv1a`, `count_byte`, `eq`, `min_max_u8`,
  `xor_into_region`, `sum_u32_le`, `max_u32_le`, `and_fold_u8`, `or_fold_u8`,
  `xor_fold_u8`, `count_in_range`, `lex_cmp`, `popcount`, `fill`,
  `xor_with_byte` — reducers

Same wire-level API, same error codes, same guest-observable semantics.
The two paths run side-by-side in this crate; consumers pick.
