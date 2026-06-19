/**
 * OPFS spill worker. Runs inside a Web Worker because
 * `FileSystemSyncAccessHandle` — the high-throughput, synchronous OPFS I/O
 * path — is only available off the main thread. One file per region
 * snapshot, keyed `region-{id}-gen-{gen}.bin`, matching the Rust
 * `FileBackingStore` naming (crates/tvm-core/src/backing.rs:126-129).
 *
 * The protocol is request/response over `postMessage`, mirroring the Rust
 * `BackingStore` trait: spill / load / delete. Buffers are transferred
 * (zero-copy) in both directions.
 */
import { backingKey } from "./backend.js";

type Req =
  | { type: "init"; dir: string; reqId: number }
  | { type: "spill"; id: number; gen: number; buffer: ArrayBuffer; reqId: number }
  | { type: "load"; id: number; gen: number; reqId: number }
  | { type: "delete"; id: number; gen: number; reqId: number };

type Res =
  | { reqId: number; ok: true; buffer?: ArrayBuffer }
  | { reqId: number; ok: false; error: string };

let dirHandle: FileSystemDirectoryHandle | null = null;

async function getDir(name: string): Promise<FileSystemDirectoryHandle> {
  if (dirHandle) return dirHandle;
  const root = await navigator.storage.getDirectory();
  dirHandle = await root.getDirectoryHandle(name, { create: true });
  return dirHandle;
}

async function handleFor(name: string, create: boolean): Promise<FileSystemSyncAccessHandle> {
  const dir = await getDir(currentDir);
  const file = await dir.getFileHandle(name, { create });
  return file.createSyncAccessHandle();
}

let currentDir = "tvm";

async function dispatch(req: Req): Promise<Res> {
  switch (req.type) {
    case "init": {
      currentDir = req.dir;
      await getDir(currentDir);
      return { reqId: req.reqId, ok: true };
    }
    case "spill": {
      const name = backingKey(req.id, req.gen);
      const handle = await handleFor(name, true);
      try {
        const view = new Uint8Array(req.buffer);
        handle.truncate(view.byteLength);
        handle.write(view, { at: 0 });
        handle.flush();
      } finally {
        handle.close();
      }
      return { reqId: req.reqId, ok: true };
    }
    case "load": {
      const name = backingKey(req.id, req.gen);
      const handle = await handleFor(name, false);
      try {
        const size = handle.getSize();
        const out = new Uint8Array(size);
        handle.read(out, { at: 0 });
        return { reqId: req.reqId, ok: true, buffer: out.buffer };
      } finally {
        handle.close();
      }
    }
    case "delete": {
      const dir = await getDir(currentDir);
      try {
        await dir.removeEntry(backingKey(req.id, req.gen));
      } catch {
        // missing file is a no-op delete
      }
      return { reqId: req.reqId, ok: true };
    }
  }
}

self.onmessage = (ev: MessageEvent<Req>) => {
  const req = ev.data;
  dispatch(req).then(
    (res) => {
      const transfer = res.ok && res.buffer ? [res.buffer] : [];
      (self as unknown as Worker).postMessage(res, transfer);
    },
    (err: unknown) => {
      const res: Res = {
        reqId: req.reqId,
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      };
      (self as unknown as Worker).postMessage(res);
    },
  );
};

export type { Req as OpfsWorkerRequest, Res as OpfsWorkerResponse };
