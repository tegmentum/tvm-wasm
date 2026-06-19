import { defineConfig } from "@playwright/test";

// The e2e suite exercises the real OPFS Cold tier in headless Chromium. Vite
// serves the harness with COOP/COEP set (see vite.config.ts) so both the
// async OPFS path and the optional SharedArrayBuffer sync path are available.
export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  use: {
    baseURL: "http://localhost:5173",
  },
  webServer: {
    command: "npx vite --port 5173 --strictPort",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
