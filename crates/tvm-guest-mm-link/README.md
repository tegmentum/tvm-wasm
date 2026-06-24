# tvm-guest-mm-link

Static linker that composes a rustc-emitted cdylib core wasm with the
multi-memory shell from **[`tvm-guest-mm`](../tvm-guest-mm/)** into a
single self-contained `.wasm`.

See [`tvm-guest-mm/docs/rust-cdylib.md`](../tvm-guest-mm/docs/rust-cdylib.md)
for the full pipeline walkthrough.

## CLI

The `tvm-mm-link` binary takes a rustc-emitted cdylib core wasm plus a
shell configuration and emits a merged module:

```sh
tvm-mm-link \
  --pools 4 \
  --initial-pages 1 \
  --max-pages 256 \
  --user my_workload.wasm \
  -o my_workload.linked.wasm
```

Defaults: 64 pools, 1 initial page per pool, 65536 max pages per pool.

### Memory-export aliases

`--alias-memory0=<name>` adds an additional `(export "<name>" (memory 0))`
to the merged module — the same memory the shell already exports as
`mem0`, just under a second name. Typical use:

```sh
tvm-mm-link --alias-memory0=memory --user my_workload.wasm -o linked.wasm
```

The merged module then gains an export literally named `memory`, which
is what `wasm-tools component new` looks for when picking the default
linear memory of a component-model adapter. Without it, the component
shape can't be inferred and `wasm-tools component new` rejects the
merged module.

For aliasing arbitrary pool memories, use the general form:

```sh
tvm-mm-link --alias-memory 1=hot_memory --alias-memory 2=warm_memory \
  --user my_workload.wasm -o linked.wasm
```

The flag may be repeated. Out-of-range indices and aliases that collide
with an existing export name are rejected at link time.

## Library

```rust
use tvm_guest_mm_link::{link_with_params, ModuleParams};

let params = ModuleParams { n_pools: 4, ..Default::default() };
let user_bytes = std::fs::read("my_workload.wasm")?;
let linked_bytes = link_with_params(&params, &user_bytes)?;
std::fs::write("my_workload.linked.wasm", &linked_bytes)?;
```

Or, for callers that already have the shell bytes:

```rust
use tvm_guest_mm_link::link;
let linked_bytes = link(&shell_bytes, &user_bytes)?;
```

To pass options (e.g. memory-export aliases) use the `_with_options`
entry points:

```rust
use tvm_guest_mm_link::{link_with_options, LinkOptions};

let options = LinkOptions {
    aliases: vec![("memory".into(), 0)],
};
let linked_bytes = link_with_options(&shell_bytes, &user_bytes, &options)?;
```

The shell bytes can be produced from
`tvm_guest_mm::tvm_guest_mm_module_template(&params)` + `wat::parse_str`,
or pre-built via the `gen_guest_wasm` binary in `tvm-guest-mm`.

## What it does

Walks both modules with `wasmparser` and re-emits a single core wasm
via `wasm-encoder`. Drops the user's `(memory ...)` declaration (the
shell's pool 0 takes its place), strips the user's `tvm_mm` imports,
and rewires every `call` to those imports as a direct call to the
corresponding shell-internal helper function. Every other user import
(function, global, table) is **forwarded** through to the merged
module's import section so the embedder satisfies it at instantiation
time exactly as it would have for the pre-link cdylib — WASI
components, host logging hooks, custom SPI contracts all survive the
link step unchanged.

See the linker source's module-level rustdoc for the section-by-section
transformation table, and `tvm-guest-mm/docs/rust-cdylib.md` for the
constraints and sharp edges.
