import { defineConfig } from "vite";

// OPFS `FileSystemSyncAccessHandle` runs inside the worker without special
// headers, but the optional synchronous-load fast path uses
// `SharedArrayBuffer` + `Atomics.wait`, which require the page to be
// cross-origin isolated. Set COOP/COEP on the dev server and preview so the
// SAB path is exercisable locally and in the Playwright harness.
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: { middlewares: { use: (fn: unknown) => void } }) {
    server.middlewares.use((_req: unknown, res: { setHeader: (k: string, v: string) => void }, next: () => void) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
};

export default defineConfig({
  plugins: [crossOriginIsolation],
  build: {
    target: "es2022",
    lib: {
      entry: {
        index: "src/index.ts",
        "opfs-worker": "src/opfs-worker.ts",
      },
      formats: ["es"],
    },
  },
  worker: {
    format: "es",
  },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    exclude: ["e2e/**", "node_modules/**", "dist/**"],
  },
});
