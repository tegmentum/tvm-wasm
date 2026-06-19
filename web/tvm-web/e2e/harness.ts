/**
 * Browser harness for the Playwright Cold-tier e2e. Exposes
 * `window.runColdTier()` which drives the real OPFS-backed spill/promote
 * cycle and returns a structured result the test asserts on.
 */
import { OpfsBackend } from "../src/opfs-backend.js";
import { RegionKind, Residency } from "../src/types.js";
import { Tvm } from "../src/tvm.js";

export interface ColdTierResult {
  poolCount: number;
  coldAfterEvict: boolean;
  readThrewWhileCold: boolean;
  residentUnderBudget: boolean;
  dataMatchesAfterPromote: boolean;
  budgetEvictions: number;
  allBudgetRegionsRestored: boolean;
}

function pattern(len: number, seed: number): Uint8Array {
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = (i * 31 + seed) & 0xff;
  return out;
}

function eq(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

async function runColdTier(): Promise<ColdTierResult> {
  const worker = new Worker(new URL("../src/opfs-worker.ts", import.meta.url), {
    type: "module",
  });
  const backend = await OpfsBackend.create(worker, { dir: "tvm-e2e" });

  // Budget so the third 8 KiB arena cannot stay resident with the others.
  const tvm = await Tvm.instantiate(fetch("/tvm-guest.wasm"), {
    backend,
    budgetBytes: 20 * 1024,
  });

  // --- single region evict -> clobber -> promote round trip ---
  const r = await tvm.createRegion(RegionKind.ObjectArena, 8 * 1024);
  const h = tvm.alloc(r, 8 * 1024);
  const data = pattern(8 * 1024, 99);
  tvm.write(h, data);

  tvm.demote(r);
  await tvm.evict(r);
  const coldAfterEvict = tvm.regionInfo(r).residency === Residency.Cold;

  let readThrewWhileCold = false;
  try {
    tvm.read(h, 8 * 1024);
  } catch {
    readThrewWhileCold = true;
  }

  // Clobber the reclaimed span to prove the reload comes from OPFS, not RAM.
  const span = tvm._directory().spanOf(r);
  tvm._pools().writeRange(span.memoryIndex, span.baseOffset, pattern(8 * 1024, 7));

  await tvm.promote(r);
  const dataMatchesAfterPromote = eq(tvm.read(h, 8 * 1024), data);

  // --- budget-driven auto-eviction across several regions ---
  const ids: number[] = [r];
  for (let i = 0; i < 3; i++) {
    const id = await tvm.createRegion(RegionKind.ObjectArena, 8 * 1024);
    const hi = tvm.alloc(id, 8 * 1024);
    tvm.write(hi, pattern(8 * 1024, i));
    tvm.demote(id);
    ids.push(id);
  }
  const residentUnderBudget = tvm.residentBytes() <= 20 * 1024;
  const budgetEvictions = ids.filter(
    (id) => tvm.regionInfo(id).residency === Residency.Cold,
  ).length;

  let allBudgetRegionsRestored = true;
  for (let i = 0; i < 3; i++) {
    const id = ids[i + 1]!;
    await tvm.promote(id);
    const info = tvm.regionInfo(id);
    const handle = { regionId: id, generation: info.generation, offset: 0 };
    if (!eq(tvm.read(handle, 8 * 1024), pattern(8 * 1024, i))) {
      allBudgetRegionsRestored = false;
    }
  }

  const result: ColdTierResult = {
    poolCount: tvm.poolCount(),
    coldAfterEvict,
    readThrewWhileCold,
    residentUnderBudget,
    dataMatchesAfterPromote,
    budgetEvictions,
    allBudgetRegionsRestored,
  };
  await tvm.close();
  return result;
}

declare global {
  interface Window {
    runColdTier(): Promise<ColdTierResult>;
  }
}

window.runColdTier = runColdTier;

// Auto-run for manual inspection when the page is opened directly.
const log = document.getElementById("log");
runColdTier().then(
  (r) => {
    if (log) log.textContent = JSON.stringify(r, null, 2);
  },
  (e: unknown) => {
    if (log) log.textContent = `error: ${e instanceof Error ? e.message : String(e)}`;
  },
);
