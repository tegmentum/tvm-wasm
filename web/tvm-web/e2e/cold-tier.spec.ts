import { expect, test } from "@playwright/test";
import type { ColdTierResult } from "./harness.js";

test("OPFS Cold tier: spill to disk and reload survives span reuse + budget", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/");

  // Drive the real OPFS-backed spill/promote cycle in the page context.
  const result = (await page.evaluate(() => window.runColdTier())) as ColdTierResult;

  expect(errors, `page errors: ${errors.join("; ")}`).toEqual([]);
  expect(result.poolCount).toBe(64);
  // Region genuinely spilled to OPFS and was unreadable while Cold.
  expect(result.coldAfterEvict).toBe(true);
  expect(result.readThrewWhileCold).toBe(true);
  // Reload came from OPFS even though the span's RAM was clobbered.
  expect(result.dataMatchesAfterPromote).toBe(true);
  // Budget kept the resident set bounded by evicting to OPFS...
  expect(result.residentUnderBudget).toBe(true);
  expect(result.budgetEvictions).toBeGreaterThan(0);
  // ...and every evicted region's data round-tripped intact.
  expect(result.allBudgetRegionsRestored).toBe(true);
});
