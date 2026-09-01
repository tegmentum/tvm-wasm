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
pub fn pooling_imported_region_engine_config(n_regions: u32, region_bytes: u32) -> Config {
    use wasmtime::{InstanceAllocationStrategy, PoolingAllocationConfig};

    const PAGE: u32 = 65_536;
    let pages = (region_bytes as u64).div_ceil(PAGE as u64).max(1);
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

// ── D2 Session 9 — wasmos RuntimeConfig peers ───────────────────────
//
// Wasmos-native alternatives to the wasmtime::Config helpers above.
// Return [`wasmos_runtime_api::RuntimeConfig`] which the caller
// passes to a wasmos adapter's constructor
// (`WasmtimeV48Runtime::new(config)`, etc.). The knobs that map
// cleanly to the portable surface are set here; adapter-specific
// tunings that don't (memory_may_move, wasm_multi_memory,
// wasm_custom_page_sizes) remain the adapter's concern — a
// wasmos-backed wasmtime adapter enables multi-memory by default,
// so consumers who need custom shapes still reach the wasmtime
// Config via ADR-0029's native escape hatch on the adapter.

/// Wasmos peer of [`imported_region_engine_config`]. Returns a
/// [`wasmos_runtime_api::RuntimeConfig`] tuned for tvm imported-
/// region workloads. Adapter picks the concrete allocator when
/// [`wasmos_runtime_api::config::AllocationStrategy::OnDemand`] is set —
/// this is the default and every runtime supports it.
///
/// See [`imported_region_engine_config`] for the semantic intent;
/// this version speaks it through the portable RuntimeConfig
/// vocabulary.
pub fn imported_region_runtime_config() -> wasmos_runtime_api::RuntimeConfig {
    wasmos_runtime_api::RuntimeConfig {
        // Balanced maps to wasmtime's default OptLevel — matches the
        // omitted opt-level in imported_region_engine_config above.
        optimization: wasmos_runtime_api::config::OptimizationLevel::Balanced,
        // tvm-wasmtime's raw + WIT paths both need the async
        // execution surface (raw_linker_wasmos::add_raw_imports_*
        // dispatches through async CoreImportFn::call).
        async_support: true,
        default_limits: wasmos_runtime_api::ExecutionLimits::default(),
        required_capabilities: None,
        profiling: wasmos_runtime_api::config::ProfilingStrategy::None,
        allocation_strategy: wasmos_runtime_api::config::AllocationStrategy::OnDemand,
    }
}

/// Wasmos peer of [`pooling_imported_region_engine_config`].
/// Returns a [`wasmos_runtime_api::RuntimeConfig`] with the pooling
/// allocator sized for `n_regions` imported memories of
/// `region_bytes` each. Requires the runtime to declare the
/// [`wasmos_runtime_api::capability::well_known::WASMOS_POOLING_ALLOCATOR`]
/// capability — adapters that don't may honour it as OnDemand or
/// reject at construction (ADR §12).
pub fn pooling_imported_region_runtime_config(
    n_regions: u32,
    region_bytes: u32,
) -> wasmos_runtime_api::RuntimeConfig {
    const PAGE: u32 = 65_536;
    let pages = (region_bytes as u64).div_ceil(PAGE as u64).max(1);
    let memory_max_bytes = pages.saturating_mul(PAGE as u64);

    wasmos_runtime_api::RuntimeConfig {
        optimization: wasmos_runtime_api::config::OptimizationLevel::Balanced,
        async_support: true,
        default_limits: wasmos_runtime_api::ExecutionLimits::default(),
        required_capabilities: None,
        profiling: wasmos_runtime_api::config::ProfilingStrategy::None,
        allocation_strategy: wasmos_runtime_api::config::AllocationStrategy::Pooling(
            wasmos_runtime_api::config::PoolingConfig {
                // `max_component_instances` isn't the right axis for
                // core-module-only workloads — leave None so the
                // adapter's default kicks in.
                max_component_instances: None,
                max_memories: Some(n_regions.saturating_add(2)),
                max_memory_bytes: Some(memory_max_bytes),
            },
        ),
    }
}

#[cfg(test)]
mod wasmos_config_tests {
    use super::*;

    #[test]
    fn imported_region_runtime_config_sets_async_and_default_optimization() {
        let c = imported_region_runtime_config();
        assert!(c.async_support);
        assert!(matches!(
            c.optimization,
            wasmos_runtime_api::config::OptimizationLevel::Balanced
        ));
        assert!(matches!(
            c.allocation_strategy,
            wasmos_runtime_api::config::AllocationStrategy::OnDemand
        ));
        assert!(matches!(c.profiling, wasmos_runtime_api::config::ProfilingStrategy::None));
    }

    #[test]
    fn pooling_runtime_config_sizes_pool_from_arguments() {
        let c = pooling_imported_region_runtime_config(/* n_regions */ 8, /* bytes */ 128 * 1024);
        match c.allocation_strategy {
            wasmos_runtime_api::config::AllocationStrategy::Pooling(pool) => {
                assert_eq!(pool.max_memories, Some(10)); // n + 2
                assert_eq!(pool.max_memory_bytes, Some(128 * 1024)); // 2 pages
                assert!(pool.max_component_instances.is_none());
            }
            other => panic!("expected Pooling, got {other:?}"),
        }
    }

    #[test]
    fn pooling_runtime_config_rounds_region_bytes_up_to_full_pages() {
        // 100 KiB rounds to 2 pages = 128 KiB.
        let c = pooling_imported_region_runtime_config(1, 100 * 1024);
        match c.allocation_strategy {
            wasmos_runtime_api::config::AllocationStrategy::Pooling(pool) => {
                assert_eq!(pool.max_memory_bytes, Some(128 * 1024));
            }
            _ => panic!(),
        }
    }
}
