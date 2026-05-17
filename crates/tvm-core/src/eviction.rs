//! Bulk eviction primitives for [`crate::ConcurrentDirectory`].
//!
//! Adds [`ConcurrentDirectory::demote_until`] and
//! [`ConcurrentDirectory::alloc_or_demote`] so callers can express
//! "evict regions until total resident bytes ≤ target" without
//! re-implementing the loop. The policy chooses which regions to
//! spill; [`crate::ConcurrentDirectory::spill_region`] does the
//! actual demotion.
//!
//! ## Semantics
//!
//! - **`target` is absolute.** The number of bytes of total resident
//!   capacity (sum of `used` across regions in [`Residency::Hot`] or
//!   [`Residency::Warm`]) that the caller wants *after* the call.
//!   Calling `demote_until(target)` twice in a row is idempotent —
//!   if the directory is already at or below `target`, the second
//!   call is a no-op.
//! - **Pinned regions are skipped silently.** A pinned region's
//!   bytes are not counted toward what `demote_until` can free, but
//!   they *are* counted in "total resident bytes" — so a pinned
//!   region above `target` can prevent the target from being met.
//!   That case is reported via [`EvictionReport::target_met`] =
//!   `false`, not an error.
//! - **Non-spillable regions** (regions where
//!   [`crate::Region::spillable`] is `false`) are also skipped.
//! - **`Cold` and `External` residency tiers** contribute 0 to the
//!   resident-bytes total and are never visited.
//! - **Errors** are returned for I/O failures (`BackingStore::spill`
//!   errors). `target_met=false` is *not* an error — it's the
//!   caller's decision whether shortfall is fatal.

use crate::residency::Residency;

/// Within a residency tier, the order in which regions are
/// considered for eviction. The first variants do not require any
/// new bookkeeping in [`crate::Region`]; LRU is deferred to a
/// future revision once a `last_access` field is added.
#[derive(Clone, Copy, Debug)]
pub enum WithinTier {
    /// Sort by `used` descending: spill the largest region first.
    /// Maximizes bytes freed per spill IO; minimizes the number of
    /// regions touched to meet a target. Good default for workloads
    /// (e.g. SQL stages) where each region is roughly stage-sized.
    LargestFirst,
    /// Sort by `used` ascending: spill the smallest region first.
    /// Minimizes per-spill disruption (small regions are less
    /// expensive to lose), at the cost of more IO ops and more
    /// regions touched.
    SmallestFirst,
    /// Walk regions in `region_id` ascending order. No size bias;
    /// reproducible for tests.
    InsertionOrder,
    // LeastRecentlyUsed — deferred; requires `Region::last_access`.
}

/// The eviction policy passed to
/// [`crate::ConcurrentDirectory::demote_until`] and
/// [`crate::ConcurrentDirectory::alloc_or_demote`]. Only one
/// top-level variant exists today (`ColdestFirst`); future
/// extensions land additively here.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum EvictionPolicy {
    /// Walk residency tiers `Warm` → `Hot` (`Cold` and `External`
    /// contribute zero to resident bytes and are skipped). Within
    /// each tier, order regions per `within_tier`. Pinned and
    /// non-spillable regions are silently skipped.
    ColdestFirst { within_tier: WithinTier },
}

/// Result of a [`crate::ConcurrentDirectory::demote_until`] call.
///
/// `bytes_freed` counts the `used` bytes of every region
/// successfully spilled. `regions_spilled` is the count of
/// successful `spill_region` calls. `target_met` is `true` iff the
/// directory's total resident bytes are at or below the requested
/// target *after* this call; it can be `false` even on success
/// (e.g. when too many regions are pinned).
///
/// Marked `#[non_exhaustive]` so adding fields is not a breaking
/// change. Callers should pattern-match by field name.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct EvictionReport {
    pub bytes_freed: u64,
    pub regions_spilled: u16,
    pub target_met: bool,
}

/// Sort key shape used internally by `demote_until`: `(region_id, used)`.
pub(crate) type EvictCandidate = (u16, u32);

/// Comparator factory: turn a [`WithinTier`] into a `sort_by`
/// closure operating on [`EvictCandidate`]s.
pub(crate) fn within_tier_cmp(
    w: WithinTier,
) -> fn(&EvictCandidate, &EvictCandidate) -> std::cmp::Ordering {
    match w {
        WithinTier::LargestFirst => |a, b| b.1.cmp(&a.1),
        WithinTier::SmallestFirst => |a, b| a.1.cmp(&b.1),
        WithinTier::InsertionOrder => |a, b| a.0.cmp(&b.0),
    }
}

/// True when the given residency contributes to the
/// resident-bytes total used by `demote_until`'s target check.
#[inline]
pub(crate) fn counts_toward_resident(r: Residency) -> bool {
    matches!(r, Residency::Hot | Residency::Warm)
}
