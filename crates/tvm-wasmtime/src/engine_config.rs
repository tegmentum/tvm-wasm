//! Engine config helpers for imported-region workloads.
//!
//! These helpers consolidate the wasmtime tunings that matter for our
//! access pattern: many fixed-capacity imported memories, no growth, hot
//! inner loops that benefit from base-pointer LICM.
//!
//! Two flavors:
//!
//! * [`imported_region_engine_config`] — the default. Each imported memory
//!   reserves exactly its declared capacity (when paired with the bounded
//!   `MemoryType` we set in [`crate::ImportedRegion::new`]) and the JIT is
//!   allowed to hoist the memory base pointer out of loops. This is the
//!   right config for everyday workloads where total aggregate capacity
//!   stays within reason.
//!
//! * [`pooling_imported_region_engine_config`] — for workloads that
//!   instantiate dozens of imported memories in a single store. The
//!   pooling allocator pre-reserves slots for memories and tables so
//!   instantiation stays cheap and the memories don't move. Requires the
//!   `pooling-allocator` cargo feature on `wasmtime` (enabled in this
//!   workspace's `Cargo.toml`).

use wasmtime::Config;

/// Build a `Config` tuned for TVM imported-region workloads.
///
/// Sets:
/// * `wasm_multi_memory(true)` — required to import more than one memory.
/// * `memory_may_move(false)` — tells the JIT every memory's base pointer
///   is stable for its lifetime, enabling loop-invariant code motion of
///   the base out of hot loops. Safe because [`crate::ImportedRegion::new`]
///   declares its `MemoryType` with `max == min`, so memories never grow.
/// * `memory_init_cow(true)` — explicit (default is already on); enables
///   copy-on-write initialization of memory images, cheap for our zero-init
///   regions.
/// * `wasm_custom_page_sizes(true)` — enables the custom-page-sizes
///   proposal so callers that want sub-64 KiB granularity can opt into
///   `(pagesize 1)` memories. No effect on memories that don't use it.
pub fn imported_region_engine_config() -> Config {
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    config.memory_may_move(false);
    config.memory_init_cow(true);
    config.wasm_custom_page_sizes(true);
    config
}

/// Build a `Config` that uses wasmtime's pooling allocator, sized for `N`
/// imported memories of `region_bytes` each within a single store.
///
/// Use this instead of [`imported_region_engine_config`] when:
/// * the store will host many imported memories (≥ ~16), AND
/// * stores get instantiated repeatedly (per-request, per-test, etc.) and
///   per-store mmap churn matters.
///
/// `region_bytes` is rounded up to wasm pages (64 KiB). The pool reserves
/// `n_regions` slots, each sized to hold a region of `region_bytes`.
/// `total_core_instances`, `total_tables`, and `total_stacks` get a small
/// multiplier so a store can hold one instance plus the imported
/// memories.
pub fn pooling_imported_region_engine_config(
    n_regions: u32,
    region_bytes: u32,
) -> Config {
    use wasmtime::{InstanceAllocationStrategy, PoolingAllocationConfig};

    const PAGE: u32 = 65_536;
    let pages = ((region_bytes as u64 + PAGE as u64 - 1) / PAGE as u64).max(1);
    let memory_max_bytes = pages.saturating_mul(PAGE as u64);

    let mut pool = PoolingAllocationConfig::default();
    pool.total_memories(n_regions.saturating_add(2));
    pool.total_tables(n_regions.saturating_add(2));
    pool.total_core_instances(n_regions.saturating_add(2));
    pool.total_stacks(n_regions.saturating_add(2));
    pool.max_memories_per_module(n_regions.saturating_add(1));
    pool.max_memory_size(memory_max_bytes as usize);

    let mut config = imported_region_engine_config();
    config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool));
    config
}
