# @tvm/web

Browser host for TVM multi-memory guests, with an **OPFS-backed Cold tier** —
spill cold regions to disk and reload them, so a browser deployment's resident
working set is bounded by storage quota instead of the tab's RAM ceiling.

## Why

`tvm-guest-mm` already scales addressing to *N × 4 GiB* in the browser
(multi-memory shipped in Wasm 3.0 on Chrome/Firefox). But without a storage
target, the *backable* working set is capped by the tab's memory budget
(low-GB to low-tens-of-GB). The host build (`tvm-wasmtime`) relieves this with
`FileBackingStore` spill-to-disk; the browser had no equivalent. This package
is that equivalent, built on the Origin Private File System.

See [`docs/architecture.md`](../../docs/architecture.md) ("Browser Cold tier")
for the full design and its two hard constraints (cooperative eviction; span
reuse rather than page reclaim).

## Architecture

```
 main thread                                   OPFS worker
 ┌──────────────────────────────────────┐      ┌─────────────────────┐
 │ guest .wasm (mem0..memN + dispatch)   │ post │ FileSystemSync-     │
 │ Tvm + Directory (JS-owned policy/LRU) │◄────►│ AccessHandle per    │
 │ reads/writes exports.memK.buffer      │ msg  │ region-{id}-gen-N   │
 └──────────────────────────────────────┘      └─────────────────────┘
```

- The guest `.wasm` is emitted unmodified by `tvm-guest-mm` (pools + dispatch
  helpers, both exported). All region metadata and the residency/LRU policy
  live here in TypeScript.
- `evict(region)` reads the region's pool bytes, writes them to OPFS, and
  reclaims the pool span. `promote(region)` reloads from OPFS into a fresh
  span. Both are cooperative: the app `await`s them; `read`/`write` on a Cold
  region throw `NotResident`.

## Usage

```ts
import { Tvm, OpfsBackend, RegionKind } from "@tvm/web";

const worker = new Worker(new URL("./opfs-worker.js", import.meta.url), { type: "module" });
const backend = await OpfsBackend.create(worker);

const tvm = await Tvm.instantiate(fetch("/tvm-guest.wasm"), {
  backend,
  budgetBytes: 256 * 1024 * 1024, // soft resident cap; warm regions auto-evict
});

const r = await tvm.createRegion(RegionKind.ObjectArena, 8 * 1024 * 1024);
const h = tvm.alloc(r, 1024);
tvm.write(h, new Uint8Array([1, 2, 3]));

tvm.demote(r);          // mark eviction-eligible
await tvm.evict(r);     // spill to OPFS, reclaim the span
// ... allocate other regions into the reclaimed space ...
await tvm.promote(r);   // reload from OPFS before touching r again
tvm.read(h, 3);         // => 1,2,3
```

## Build & test

```sh
# 1. generate the guest module (writes public/tvm-guest.wasm)
# --max-pages-per-pool 256 caps the per-tab wasm virtual reservation at
# 1 GiB (pools × max_pages × 64 KiB); without it Chromium's V8 refuses to
# instantiate 64 memories declaring the 4-GiB wasm32 max.
cargo run -p tvm-guest-mm --bin gen_guest_wasm -- --pools 64 --max-pages-per-pool 256 web/tvm-web/public/tvm-guest.wasm

# 2. install + verify
cd web/tvm-web
npm install
npm run typecheck
npm test            # vitest: directory logic, sync channel, and a full
                    # evict/promote round-trip against the real wasm in Node
npm run test:e2e    # playwright: the real OPFS Cold tier in headless Chromium
```

The Vitest suite runs the real multi-memory guest under Node's V8 (which
supports multi-memory) with an in-memory backend stub; only OPFS itself is
browser-only, which is what the Playwright e2e covers.

## Notes

- **Cross-origin isolation:** the optional `SharedArrayBuffer` sync-load path
  (`src/sync-channel.ts`) needs `COOP: same-origin` + `COEP: require-corp`. The
  Vite config sets these. The default async `OpfsBackend` does not require it.
- **Browser support:** OPFS `FileSystemSyncAccessHandle` is available in
  Chrome/Firefox and Safari 17+. Verify multi-memory support on your target
  Safari version.
