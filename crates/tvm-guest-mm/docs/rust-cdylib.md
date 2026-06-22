# Building a Rust cdylib against `tvm-guest-mm`

This document describes the **rust-cdylib substrate** for
`tvm-guest-mm`: how a downstream crate compiles ordinary Rust source
into a self-contained multi-memory `.wasm` that runs on wasmtime (and
any other engine with the wasm-multi-memory proposal) with no host
imports.

The path replaces both of the previously-documented patterns:

| Old pattern | New pattern (this doc) |
|---|---|
| Splice user-written WAT into `ModuleParams::user_body` | Write ordinary Rust against `tvm-guest-mm-rt`; the linker does the splice |
| Post-build wasm-tools composition steps in the consumer | One `cargo build` + one `tvm-mm-link` invocation |
| Host-side `tvm:memory/*` WIT imports for cold-tier access | Zero unresolved imports in the output |

It is the **first non-toy v2 substrate consumer** unblocker for the
sqlite-pcache-tvm and sqlite-vfs-tvm crates that today still talk to
the host-side `tvm:memory` interface via wit-bindgen.

## Architecture

```text
┌───────────────────────────────┐
│ user crate (cdylib)           │     uses tvm-guest-mm-rt
│   src/lib.rs                  │     extern "C" against "tvm_mm"
│   #[no_mangle] pub extern …   │
└─────────────┬─────────────────┘
              │ cargo build --target wasm32-unknown-unknown
              ▼
┌───────────────────────────────┐
│ user_core.wasm                │     imports tvm_mm.tvm_load_u32 …
│   memory 0 (rustc heap)       │     own data section + heap
│   func[0..K] = imports        │
│   func[K..]  = workload       │
│   exports: write_page, …      │
└─────────────┬─────────────────┘
              │ tvm-mm-link (uses wasmparser + wasm-encoder)
              │   + tvm_guest_mm_module_template(ModuleParams)
              ▼
┌───────────────────────────────┐
│ linked.wasm   ← single .wasm  │
│   memory 0 (= shell pool 0,   │     ← user's mem 0 dropped;
│             also user heap)   │       its data segments retargeted
│   memory 1..N (data pools)    │
│   shell helpers (mem-immed.)  │
│   user funcs (calls rewired)  │     ← every `call tvm_mm.X` is now
│   exports: shell + workload   │       a direct internal call
└───────────────────────────────┘
```

Three crates collaborate:

1. **`tvm-guest-mm`** (existing) — the shell template generator. Emits
   the WAT shell on demand from a `ModuleParams { n_pools, … }`.
2. **`tvm-guest-mm-rt`** — guest-side rlib that exposes a safe Rust
   API (`Pool`, `load_u32`, `store_u32`, `copy_to_default`, …) over
   `extern "C"` declarations against `wasm_import_module = "tvm_mm"`.
3. **`tvm-guest-mm-link`** — static linker that consumes (shell bytes,
   user wasm) and produces a merged self-contained core wasm.

The downstream consumer crate depends only on `tvm-guest-mm-rt`. The
shell + linker are invoked at build time, typically from a `Makefile`
or `xtask`.

## What the linker does, in detail

`tvm-guest-mm-link::link(shell_bytes, user_bytes)` walks both modules
with `wasmparser` and re-emits a single core wasm via `wasm-encoder`.
The transformations:

| Section | Behavior |
|---|---|
| **Types** | Concatenate (shell first, then user). User type indices renumbered. |
| **Imports** | Shell must have none. User's `tvm_mm.*` imports are stripped; their func indices map to the corresponding shell-internal function indices via the shell's export table. Non-`tvm_mm` imports are rejected. |
| **Functions** | Shell function entries unchanged; user's appended. User type indices renumbered. |
| **Tables** | Concatenate. User table indices renumbered. |
| **Memories** | Shell memories only. User's default memory (memory 0) is dropped; the linker rewrites all data segments + memargs referencing it to target pool 0 of the shell (memory 0 in the merged module). Pool 0's initial-page count is auto-bumped to the user's declared initial when it's higher (rustc requests 16 initial pages for the data section + heap base). |
| **Globals** | Concatenate. User globals renumbered. |
| **Exports** | Shell exports unchanged. User exports kept except for `memory`, `__data_end`, `__heap_base`, `__indirect_function_table` (rustc cdylib housekeeping that conflicts with the shell's namespace). |
| **Start** | Shell's start function (if any) is preserved. User start function is rejected. |
| **Elements** | Concatenate. User elem funcref entries renumbered. |
| **Code** | Shell code is passed through unmodified (`CodeSection::raw`). User code is re-emitted via `wasm-encoder`; every `call`, `return_call`, `ref.func`, `call_indirect`, `global.*`, `table.*` is renumbered through the maps above. |
| **Data** | Concatenate. User data segments' `memory_index` is rewritten to point at pool 0. |

Operators outside the subset rustc emits for a typical `no_std` cdylib
return an explicit error from the linker (rather than silently
dropping the op). The current subset covers everything rustc emits
for the example consumer and the probe; broaden `map_simple_op` to
handle SIMD or threads as needed.

## Quick start

### 1. Set up the consumer crate

```toml
# Cargo.toml
[package]
name = "my-tvm-workload"
version = "0.0.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
tvm-guest-mm-rt = { path = "path/to/tvm-wasm/crates/tvm-guest-mm-rt" }

[profile.release]
opt-level = "s"
lto = true
strip = true
```

```rust
// src/lib.rs
#![no_std]
use tvm_guest_mm_rt::{Pool, load_u32, store_u32};

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }

const HOT: Pool = Pool::new(1);

#[no_mangle]
pub extern "C" fn put(off: u32, v: u32) {
    store_u32(HOT, off, v);
}

#[no_mangle]
pub extern "C" fn get(off: u32) -> u32 {
    load_u32(HOT, off)
}
```

### 2. Build the cdylib

```sh
cargo build --release --target wasm32-unknown-unknown
```

Output: `target/wasm32-unknown-unknown/release/my_tvm_workload.wasm`.
This wasm has `tvm_mm.*` imports and is not yet runnable on a host
that doesn't provide them — that's the linker's job.

### 3. Link against the shell

Build the linker once:

```sh
cargo build --release -p tvm-guest-mm-link --manifest-path path/to/tvm-wasm/Cargo.toml
```

Then link:

```sh
path/to/tvm-wasm/target/release/tvm-mm-link \
  --pools 4 \
  --initial-pages 1 \
  --max-pages 256 \
  --user target/wasm32-unknown-unknown/release/my_tvm_workload.wasm \
  -o my_tvm_workload.linked.wasm
```

Now `my_tvm_workload.linked.wasm` is a self-contained core wasm. It
exports:

- Every `mem0..memN` (the pool memories — direct JS buffer access for
  browsers; `instance.get_memory(...)` for wasmtime hosts)
- Every shell dispatch helper (`tvm_load_u32`, `tvm_store_u32`,
  `tvm_copy_to_default`, `tvm_simd_sum_u8_p0`, …) — exposed for
  host-side reducers and direct manipulation
- Every `#[no_mangle] pub extern "C"` from the consumer crate

And it imports nothing. Confirm with `wasm-tools print` or the
component-model wit inspector:

```sh
wasm-tools print my_tvm_workload.linked.wasm | grep '(import' || echo "no imports"
# → no imports
```

### 4. Run on wasmtime

```rust
use wasmtime::{Config, Engine, Linker, Module, Store};

let mut config = Config::new();
config.wasm_multi_memory(true);
let engine = Engine::new(&config)?;
let module = Module::new(&engine, &std::fs::read("my_tvm_workload.linked.wasm")?)?;
let linker: Linker<()> = Linker::new(&engine);
let mut store = Store::new(&engine, ());
let instance = linker.instantiate(&mut store, &module)?;

let put = instance.get_typed_func::<(u32, u32), ()>(&mut store, "put")?;
let get = instance.get_typed_func::<u32, u32>(&mut store, "get")?;
put.call(&mut store, (0, 0xdead_beef))?;
assert_eq!(get.call(&mut store, 0)?, 0xdead_beef);
```

## Naming convention reference

The shell's WAT-level helpers map to `tvm-guest-mm-rt` Rust functions
by dropping the `tvm_` prefix:

| WAT export (shell) | Rust binding (`tvm-guest-mm-rt`) |
|---|---|
| `tvm_load_u8(pool, off) → u32` | `load_u8(Pool, u32) → u8` |
| `tvm_load_u32(pool, off) → u32` | `load_u32(Pool, u32) → u32` |
| `tvm_load_i64(pool, off) → i64` | `load_i64(Pool, u32) → i64` |
| `tvm_store_u8(pool, off, val)` | `store_u8(Pool, u32, u8)` |
| `tvm_store_u32(pool, off, val)` | `store_u32(Pool, u32, u32)` |
| `tvm_store_i64(pool, off, val)` | `store_i64(Pool, u32, i64)` |
| `tvm_copy_to_default(src_pool, src_off, dst_off, len)` | `copy_to_default(Pool, u32, u32, u32)` |
| `tvm_copy_from_default(dst_pool, dst_off, src_off, len)` | `copy_from_default(Pool, u32, u32, u32)` |

Plus the two byte-slice convenience helpers:

| Function | Behavior |
|---|---|
| `read_bytes(pool, off, dst)` | One-call bulk read from pool into a `&mut [u8]` in the default memory. Wraps `tvm_copy_to_default` with the slice's pointer. |
| `write_bytes(pool, off, src)` | One-call bulk write from a `&[u8]` in the default memory into the pool. Wraps `tvm_copy_from_default`. |

The per-pool specialized helpers (`tvm_load_u32_p7`, `tvm_simd_sum_u8_p3`,
…) are **not** exposed as Rust bindings because Rust can't dispatch to
one of N statically-named imports without knowing N at the binding
site. They remain available via `wasm-tools print` for hosts that
want to call them directly; workloads that need the dispatch-free hot
path should call the generic dispatcher with a `const Pool`, which
wasmtime constant-folds the BST against.

## Constraints + sharp edges

The current implementation handles the common case (rustc-emitted
no_std cdylibs with only `tvm_mm` imports). The following are
intentional v1 limits:

1. **Shell must have no imports.** The shell template generates a
   self-contained module, so this is naturally satisfied. The linker
   rejects shells with imports as a sanity check; remove the check
   and forward shell imports as a follow-up.
2. **User's only imports are from `tvm_mm`.** Other imports (WASI,
   etc.) are rejected. A future revision can forward arbitrary
   imports to the merged module's import section.
3. **User must not declare a start function.** Conflicts with the
   shell's start. Drop or forward as a follow-up.
4. **Pool 0 doubles as the user's default memory.** This is by design
   (rustc's loads/stores target memory 0; the shell template makes
   pool 0 the metadata + default memory). The linker auto-bumps pool
   0's initial pages to match the user's; if the user's data section
   exceeds the shell's pool 0 max, instantiation will fail with a
   memory-size error — bump `max_pages_per_pool` at link time.
5. **No host-mediated memory growth.** Pool 0 still has the rustc
   `memory.grow` instruction; growth is fine if the shell's pool 0
   max allows it.
6. **Operator coverage.** The rewriter handles the operators rustc
   emits for typical no_std cdylibs (control flow, numeric, memory
   load/store, bulk-memory, call/ref). SIMD, threads, GC, exceptions,
   and tail calls beyond `return_call`/`return_call_indirect` are not
   covered. The linker bails with an explicit error if it hits an
   unhandled operator; extend `map_simple_op` in
   `crates/tvm-guest-mm-link/src/lib.rs` with the missing variant.
7. **Custom sections are dropped.** Producer/name sections are
   discarded by the linker. For symbol-bearing debug info, the
   consumer should produce a separate `.dwp` and pair it with the
   linked module — the linked module's function indices will not
   match the cdylib's function indices.

## Worked example: the rust-cdylib-consumer

`examples/rust-cdylib-consumer/` is a working consumer that mirrors
the shape sqlite-pcache-tvm needs:

- Two pools (`HOT=1`, `WARM=2`) modeling a tiered page cache.
- 4 KiB pages with byte-level + u32-level accessors.
- Page materialization (`tvm_copy_to_default`) for moving a page from
  a data pool into the workload's default heap before processing.
- Page install (`tvm_copy_from_default`) for the symmetric path.
- Hot → warm eviction (double-copy via the default-memory scratch).
- A small reducer (`sum_hot_page_headers`) that scans the first u32
  of N consecutive pages — representative of "scan over a column of
  header fields" patterns pcache uses for cache-warmup heuristics.

Build:

```sh
cd examples/rust-cdylib-consumer
make
```

Inspect:

```sh
make show
```

The `tvm-guest-mm` crate's `tests/rust_cdylib_e2e.rs` integration test
drives the full pipeline (cargo + linker + wasmtime) and verifies
round-trip, pool isolation, page materialization, and the header sum.
Run it directly:

```sh
cargo test -p tvm-guest-mm --test rust_cdylib_e2e
```
