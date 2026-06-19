/**
 * Backing-store abstraction for the Cold tier. Mirrors the Rust
 * `BackingStore` trait (crates/tvm-core/src/backing.rs:70-73): keyed by
 * `(regionId, generation)`, moves whole region snapshots. Async here because
 * every browser storage target (OPFS, IndexedDB) is async.
 */
export interface SpillBackend {
  spill(regionId: number, generation: number, bytes: Uint8Array): Promise<void>;
  load(regionId: number, generation: number): Promise<Uint8Array>;
  delete(regionId: number, generation: number): Promise<void>;
  close?(): Promise<void>;
}

export function backingKey(regionId: number, generation: number): string {
  return `region-${regionId}-gen-${generation}.bin`;
}

/**
 * In-memory backend — the test/dev stub. Holds spilled bytes in a Map
 * instead of OPFS, so directory/eviction logic can be unit-tested in Node
 * without a real file system. Bytes are copied on spill and load so callers
 * can't alias stored data.
 */
export class InMemoryBackend implements SpillBackend {
  private readonly store = new Map<string, Uint8Array>();

  async spill(regionId: number, generation: number, bytes: Uint8Array): Promise<void> {
    this.store.set(backingKey(regionId, generation), bytes.slice());
  }

  async load(regionId: number, generation: number): Promise<Uint8Array> {
    const found = this.store.get(backingKey(regionId, generation));
    if (!found) throw new Error(`no spilled bytes for region ${regionId} gen ${generation}`);
    return found.slice();
  }

  async delete(regionId: number, generation: number): Promise<void> {
    this.store.delete(backingKey(regionId, generation));
  }

  /** Test helper: how many region snapshots are currently spilled. */
  size(): number {
    return this.store.size;
  }
}
