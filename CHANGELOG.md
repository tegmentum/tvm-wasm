# Changelog

All notable changes to the tvm-wasm workspace crates.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [SemVer](https://semver.org/) and is pre-1.0, so
0.y.z minor bumps may break API.

## [Unreleased]

## [0.2.0] — 2026-09-02

Patch-level dep bump on top of v0.1.0. No API changes, no
consumer-visible surface changes.

### Changed

- Workspace wasmtime pin bumped from `48.0.0` to `48.0.1`. Same-
  major patch. Consumers on the abstraction path (via
  `wasmos-runtime-select` etc.) transparently pick up the patch
  when they upgrade the tag pin.

## [0.1.0] — 2026-09-02

First tagged release. Snapshot of tvm-wasm main at commit
`6f3bae38` — the rev every downstream consumer (ducklink, sqlink,
sqlite-wasm) was already pinning at the time of tagging.

Consumers can pin to a tag instead of a git rev:

```toml
tvm-core        = { git = "https://github.com/tegmentum/tvm-wasm.git", tag = "v0.1.0" }
tvm-wasmtime    = { git = "https://github.com/tegmentum/tvm-wasm.git", tag = "v0.1.0" }
tvm-guest-mm-rt = { git = "https://github.com/tegmentum/tvm-wasm.git", tag = "v0.1.0" }
```

### Workspace crates in the v0.1.0 shape

- `tvm-core` — the Tiered Virtual Memory substrate (host-owned
  regions backing >4 GiB spill tiers). Wasmtime-free; the
  manager/bytes host traits are implemented against a caller's
  bindgen.
- `tvm-wasmtime` — wasmtime-native host implementations
  (`SharedTvmHost`, per-actor variants, ADR-0029 wasmos peers via
  `#[host_iface(sync)]`). Feeds sqlink and ducklink host stacks.
- `tvm-guest-mm-rt` — the guest-side runtime substrate for the
  `tvm-guest-mm` shell. Consumed by sqlite-wasm's TVM providers.
- `tvm-guest-mm-link` — build-time helper linking guest
  components against `tvm-guest-mm-rt`.
- `tvm-guest-mm` — the guest-side shell interface.
- `tvm-tests`, `tvm-test-harness`, `tvm-bench-runner` — test +
  bench infrastructure.

### Reference material

- The wasmos runtime-abstraction docs (see wasmos
  `docs/design/runtime-abstraction/`) document how consumers
  reach `tvm-wasmtime` through the abstraction path.

[Unreleased]: https://github.com/tegmentum/tvm-wasm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tegmentum/tvm-wasm/releases/tag/v0.2.0
[0.1.0]: https://github.com/tegmentum/tvm-wasm/releases/tag/v0.1.0
