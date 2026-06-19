/**
 * Main-thread proxy to the OPFS spill worker. Implements {@link SpillBackend}
 * by round-tripping requests over `postMessage`. Because the JS host owns the
 * eviction policy and spills only while the guest is quiesced (between calls
 * into the guest), an async channel is sufficient — the guest is never
 * mid-execution during a spill/load. A synchronous fast path is available
 * separately in bridge.ts for callers that must load inside a tight loop.
 */
import { type SpillBackend } from "./backend.js";
import type { OpfsWorkerRequest, OpfsWorkerResponse } from "./opfs-worker.js";

interface Pending {
  resolve: (buffer?: ArrayBuffer) => void;
  reject: (err: Error) => void;
}

export interface OpfsBackendOptions {
  /** Subdirectory under the origin's private file system. Default "tvm". */
  dir?: string;
}

export class OpfsBackend implements SpillBackend {
  private nextReqId = 1;
  private readonly pending = new Map<number, Pending>();

  private constructor(private readonly worker: Worker) {
    this.worker.onmessage = (ev: MessageEvent<OpfsWorkerResponse>) => {
      const res = ev.data;
      const p = this.pending.get(res.reqId);
      if (!p) return;
      this.pending.delete(res.reqId);
      if (res.ok) p.resolve(res.buffer);
      else p.reject(new Error(res.error));
    };
  }

  /**
   * Spin up the worker and initialize the OPFS directory. The caller supplies
   * a constructed Worker so bundlers can resolve the worker entry point
   * (`new Worker(new URL("./opfs-worker.js", import.meta.url), { type: "module" })`).
   */
  static async create(worker: Worker, opts: OpfsBackendOptions = {}): Promise<OpfsBackend> {
    const backend = new OpfsBackend(worker);
    await backend.send({ type: "init", dir: opts.dir ?? "tvm", reqId: 0 });
    return backend;
  }

  async spill(regionId: number, generation: number, bytes: Uint8Array): Promise<void> {
    // Copy into a standalone, transferable ArrayBuffer (guest memory must not
    // be neutered, and may be a view over a larger buffer).
    const buffer = bytes.slice().buffer;
    await this.send(
      { type: "spill", id: regionId, gen: generation, buffer, reqId: 0 },
      [buffer],
    );
  }

  async load(regionId: number, generation: number): Promise<Uint8Array> {
    const buffer = await this.send({ type: "load", id: regionId, gen: generation, reqId: 0 });
    if (!buffer) throw new Error("load returned no buffer");
    return new Uint8Array(buffer);
  }

  async delete(regionId: number, generation: number): Promise<void> {
    await this.send({ type: "delete", id: regionId, gen: generation, reqId: 0 });
  }

  async close(): Promise<void> {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error("backend closed"));
    this.pending.clear();
  }

  private send(req: OpfsWorkerRequest, transfer: Transferable[] = []): Promise<ArrayBuffer | undefined> {
    const reqId = this.nextReqId++;
    req.reqId = reqId;
    return new Promise<ArrayBuffer | undefined>((resolve, reject) => {
      this.pending.set(reqId, { resolve, reject });
      this.worker.postMessage(req, transfer);
    });
  }
}
