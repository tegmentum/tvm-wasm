//! `PerActorTvmHost` — a per-store wrapper around [`SharedTvmHost`] that
//! enforces a per-actor allocation budget.
//!
//! The default `SharedTvmHost` is shared substrate: one directory, many
//! actors/stores. A misbehaving actor on an untrusted host can exhaust
//! the substrate for the well-behaved ones. `PerActorTvmHost` lifts the
//! budget into the *per-store* wrapper: the inner substrate is still
//! shared, but each actor has its own outstanding-bytes accounting and
//! its own overrun flag. The embedder gives one actor 64 MiB, another
//! 16 GiB; they cooperate on the same substrate without trusting each
//! other's behaviour.
//!
//! ## Semantics
//!
//! - **Outstanding bytes**, not cumulative. `alloc` adds to the
//!   per-actor counter; `dealloc` releases. A real workload's
//!   *steady-state working set* is what the budget bounds.
//! - **Unlimited mode** (`TvmBudget::unlimited()`) preserves the prior
//!   behaviour exactly — every host call delegates straight to the
//!   inner `SharedTvmHost` with no overhead beyond the wrapping mutex.
//! - **Soft failure shape**: a budget overrun returns
//!   `TvmError::AllocationFailed` — the *existing* error variant
//!   substrate-wide exhaustion uses. The wrapper additionally sets an
//!   `overrun` flag the embedder can poll
//!   ([`PerActorTvmHost::budget_overrun`]); that lets a runtime such
//!   as girder reclassify the *otherwise graceful* `AllocationFailed`
//!   as fatal-and-restartable for this actor's incarnation (a restart
//!   releases the actor's regions, so it's actually remediable —
//!   unlike substrate-wide exhaustion). No WIT change needed; the
//!   host-side flag is invisible to the guest.
//!
//! ## Per-store wrapper, shared substrate
//!
//! Every method that doesn't touch the budget delegates to the inner
//! `SharedTvmHost` — including the bytes/diagnostics surfaces. Two
//! `PerActorTvmHost` instances with the same inner `SharedTvmHost`
//! see the same regions; their budget accounting is independent.
//!
//! ```ignore
//! let shared = SharedTvmHost::new();
//! // Actor A: 64 MiB outstanding budget.
//! let a = PerActorTvmHost::new(shared.clone(), TvmBudget { max_outstanding_bytes: 64 * 1024 * 1024 });
//! // Actor B: 16 GiB.
//! let b = PerActorTvmHost::new(shared.clone(), TvmBudget { max_outstanding_bytes: 16 * 1024 * 1024 * 1024 });
//! // A and B share the directory; their budgets are private.
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::bindings::tvm::memory::bytes::Host as BytesHost;
use crate::bindings::tvm::memory::diagnostics::Host as DiagnosticsHost;
use crate::bindings::tvm::memory::manager::Host as ManagerHost;
use crate::bindings::tvm::memory::types::{
    CompactResult, Handle, Host as TypesHost, RegionInfo, RegionKind, RegionMetrics, TvmError,
};
use crate::shared_host::SharedTvmHost;

/// Per-actor allocation budget for [`PerActorTvmHost`].
#[derive(Debug, Clone, Copy)]
pub struct TvmBudget {
    /// Maximum *outstanding* bytes allowed for this actor across all
    /// regions. `0` is treated as unlimited — preserves the prior
    /// behaviour for embedders that don't want enforcement.
    pub max_outstanding_bytes: u64,
}

impl TvmBudget {
    /// No enforcement. Equivalent to a bare `SharedTvmHost`.
    pub fn unlimited() -> Self {
        Self {
            max_outstanding_bytes: 0,
        }
    }

    /// Limit outstanding bytes to `bytes`.
    pub fn outstanding_bytes(bytes: u64) -> Self {
        Self {
            max_outstanding_bytes: bytes,
        }
    }
}

impl Default for TvmBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

#[derive(Default)]
struct BudgetState {
    outstanding_bytes: u64,
    overrun: bool,
    /// `(region_id, generation, offset) → size`. Lets `dealloc` release
    /// the right number of bytes for the budget.
    allocations: HashMap<(u16, u16, u32), u32>,
}

/// A per-store budget wrapper around `SharedTvmHost`. See module docs.
#[derive(Clone)]
pub struct PerActorTvmHost {
    inner: SharedTvmHost,
    budget: TvmBudget,
    state: Arc<Mutex<BudgetState>>,
}

impl PerActorTvmHost {
    pub fn new(inner: SharedTvmHost, budget: TvmBudget) -> Self {
        Self {
            inner,
            budget,
            state: Arc::new(Mutex::new(BudgetState::default())),
        }
    }

    /// The inner shared substrate (unwrapping the per-actor view).
    pub fn shared(&self) -> &SharedTvmHost {
        &self.inner
    }

    /// `true` if any `alloc` on this actor has been denied by the
    /// budget since the last [`reset_overrun`](Self::reset_overrun)
    /// (typically called by the embedder once per turn / per incarnation).
    pub fn budget_overrun(&self) -> bool {
        match self.state.lock() {
            Ok(s) => s.overrun,
            Err(p) => p.into_inner().overrun,
        }
    }

    /// Clear the overrun flag (e.g. after the embedder has acted on it).
    pub fn reset_overrun(&self) {
        match self.state.lock() {
            Ok(mut s) => s.overrun = false,
            Err(p) => p.into_inner().overrun = false,
        }
    }

    /// Currently outstanding (alloced-but-not-dealloced) bytes for this
    /// actor. Useful for debugging / introspection.
    pub fn outstanding_bytes(&self) -> u64 {
        match self.state.lock() {
            Ok(s) => s.outstanding_bytes,
            Err(p) => p.into_inner().outstanding_bytes,
        }
    }

    fn handle_key(h: &Handle) -> (u16, u16, u32) {
        (h.region_id, h.generation, h.offset)
    }
}

impl AsMut<PerActorTvmHost> for PerActorTvmHost {
    fn as_mut(&mut self) -> &mut PerActorTvmHost {
        self
    }
}

impl TypesHost for PerActorTvmHost {}

// --- ManagerHost: alloc/dealloc enforce the budget; everything else
//     just delegates to the inner SharedTvmHost. ----------------------

impl ManagerHost for PerActorTvmHost {
    fn create_region(&mut self, kind: RegionKind, capacity: u32) -> Result<u16, TvmError> {
        ManagerHost::create_region(&mut self.inner, kind, capacity)
    }

    fn destroy_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        ManagerHost::destroy_region(&mut self.inner, region_id)
    }

    fn alloc(&mut self, region_id: u16, size: u32) -> Result<Handle, TvmError> {
        // Fast path: unlimited budget → no accounting overhead.
        if self.budget.max_outstanding_bytes == 0 {
            return ManagerHost::alloc(&mut self.inner, region_id, size);
        }
        // Budget check before the substrate touches the directory.
        {
            let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if s.outstanding_bytes.saturating_add(size as u64) > self.budget.max_outstanding_bytes {
                s.overrun = true;
                return Err(TvmError::AllocationFailed);
            }
        }
        let handle = ManagerHost::alloc(&mut self.inner, region_id, size)?;
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        s.outstanding_bytes = s.outstanding_bytes.saturating_add(size as u64);
        s.allocations.insert(Self::handle_key(&handle), size);
        Ok(handle)
    }

    fn dealloc(&mut self, ptr: Handle) -> Result<(), TvmError> {
        ManagerHost::dealloc(&mut self.inner, ptr)?;
        if self.budget.max_outstanding_bytes == 0 {
            return Ok(());
        }
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(size) = s.allocations.remove(&Self::handle_key(&ptr)) {
            s.outstanding_bytes = s.outstanding_bytes.saturating_sub(size as u64);
        }
        Ok(())
    }

    fn describe_region(&mut self, region_id: u16) -> Result<RegionInfo, TvmError> {
        ManagerHost::describe_region(&mut self.inner, region_id)
    }

    fn promote_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        ManagerHost::promote_region(&mut self.inner, region_id)
    }

    fn demote_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        ManagerHost::demote_region(&mut self.inner, region_id)
    }

    fn spill_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        ManagerHost::spill_region(&mut self.inner, region_id)
    }

    fn load_region(&mut self, region_id: u16) -> Result<(), TvmError> {
        ManagerHost::load_region(&mut self.inner, region_id)
    }

    fn pin(&mut self, region_id: u16) -> Result<(), TvmError> {
        ManagerHost::pin(&mut self.inner, region_id)
    }

    fn unpin(&mut self, region_id: u16) -> Result<(), TvmError> {
        ManagerHost::unpin(&mut self.inner, region_id)
    }

    fn compact_region(&mut self, region_id: u16) -> Result<CompactResult, TvmError> {
        ManagerHost::compact_region(&mut self.inner, region_id)
    }
}

// --- BytesHost: pure delegation. Bytes don't change outstanding totals;
//     they cross the host/guest boundary, they don't allocate. -----------

impl BytesHost for PerActorTvmHost {
    fn read(&mut self, ptr: Handle, len: u32) -> Result<Vec<u8>, TvmError> {
        BytesHost::read(&mut self.inner, ptr, len)
    }

    fn write(&mut self, ptr: Handle, data: Vec<u8>) -> Result<(), TvmError> {
        BytesHost::write(&mut self.inner, ptr, data)
    }

    fn copy(&mut self, src: Handle, dst: Handle, len: u32) -> Result<(), TvmError> {
        BytesHost::copy(&mut self.inner, src, dst, len)
    }

    fn read_into(
        &mut self,
        src: Handle,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<(), TvmError> {
        BytesHost::read_into(&mut self.inner, src, dst_region, dst_offset, len)
    }

    fn write_from(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst: Handle,
        len: u32,
    ) -> Result<(), TvmError> {
        BytesHost::write_from(&mut self.inner, src_region, src_offset, dst, len)
    }

    fn copy_region(
        &mut self,
        src_region: u16,
        src_offset: u32,
        dst_region: u16,
        dst_offset: u32,
        len: u32,
    ) -> Result<(), TvmError> {
        BytesHost::copy_region(
            &mut self.inner,
            src_region,
            src_offset,
            dst_region,
            dst_offset,
            len,
        )
    }
}

// --- DiagnosticsHost: pure delegation. ----------------------------------

impl DiagnosticsHost for PerActorTvmHost {
    fn list_regions(&mut self) -> Vec<RegionInfo> {
        DiagnosticsHost::list_regions(&mut self.inner)
    }

    fn fault_count(&mut self, region_id: u16) -> u64 {
        DiagnosticsHost::fault_count(&mut self.inner, region_id)
    }

    fn allocation_count(&mut self, region_id: u16) -> u64 {
        DiagnosticsHost::allocation_count(&mut self.inner, region_id)
    }

    fn bytes_read_count(&mut self, region_id: u16) -> u64 {
        DiagnosticsHost::bytes_read_count(&mut self.inner, region_id)
    }

    fn bytes_written_count(&mut self, region_id: u16) -> u64 {
        DiagnosticsHost::bytes_written_count(&mut self.inner, region_id)
    }

    fn metrics_snapshot(&mut self, region_id: u16) -> Result<RegionMetrics, TvmError> {
        DiagnosticsHost::metrics_snapshot(&mut self.inner, region_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::tvm::memory::types::RegionKind;

    #[test]
    fn unlimited_budget_passes_through() {
        let shared = SharedTvmHost::new();
        let mut h = PerActorTvmHost::new(shared, TvmBudget::unlimited());
        let region = ManagerHost::create_region(&mut h, RegionKind::HotHeap, 4096).unwrap();
        // Plenty of allocations succeed; no overrun tracking on unlimited.
        for _ in 0..16 {
            ManagerHost::alloc(&mut h, region, 64).unwrap();
        }
        assert!(!h.budget_overrun());
        assert_eq!(h.outstanding_bytes(), 0); // unlimited fast-path skips counter
    }

    #[test]
    fn budget_denies_alloc_past_quota_and_flags_overrun() {
        let shared = SharedTvmHost::new();
        let mut h = PerActorTvmHost::new(shared, TvmBudget::outstanding_bytes(128));
        let region = ManagerHost::create_region(&mut h, RegionKind::HotHeap, 4096).unwrap();

        // Within budget: 64 + 32 = 96 bytes outstanding.
        let _h1 = ManagerHost::alloc(&mut h, region, 64).unwrap();
        let _h2 = ManagerHost::alloc(&mut h, region, 32).unwrap();
        assert_eq!(h.outstanding_bytes(), 96);
        assert!(!h.budget_overrun());

        // Past budget: 96 + 64 = 160 > 128 → denied + overrun flag.
        let denied = ManagerHost::alloc(&mut h, region, 64);
        assert!(matches!(denied, Err(TvmError::AllocationFailed)));
        assert!(h.budget_overrun());
        // Failed alloc must NOT have touched outstanding.
        assert_eq!(h.outstanding_bytes(), 96);
    }

    #[test]
    fn dealloc_releases_budget_under_limit() {
        let shared = SharedTvmHost::new();
        let mut h = PerActorTvmHost::new(shared, TvmBudget::outstanding_bytes(128));
        let region = ManagerHost::create_region(&mut h, RegionKind::HotHeap, 4096).unwrap();

        let h1 = ManagerHost::alloc(&mut h, region, 100).unwrap();
        assert_eq!(h.outstanding_bytes(), 100);
        ManagerHost::dealloc(&mut h, h1).unwrap();
        assert_eq!(h.outstanding_bytes(), 0);

        // After dealloc, the full budget is available again.
        let _h2 = ManagerHost::alloc(&mut h, region, 100).unwrap();
        assert_eq!(h.outstanding_bytes(), 100);
        assert!(!h.budget_overrun());
    }

    #[test]
    fn reset_overrun_clears_flag() {
        let shared = SharedTvmHost::new();
        let mut h = PerActorTvmHost::new(shared, TvmBudget::outstanding_bytes(8));
        let region = ManagerHost::create_region(&mut h, RegionKind::HotHeap, 4096).unwrap();
        let _ = ManagerHost::alloc(&mut h, region, 16); // denied → overrun=true
        assert!(h.budget_overrun());
        h.reset_overrun();
        assert!(!h.budget_overrun());
    }

    #[test]
    fn two_actors_share_substrate_independent_budgets() {
        // The point of the wrapper: shared substrate, per-store accounting.
        let shared = SharedTvmHost::new();
        let mut a = PerActorTvmHost::new(shared.clone(), TvmBudget::outstanding_bytes(64));
        let mut b = PerActorTvmHost::new(shared, TvmBudget::outstanding_bytes(64));

        let r = ManagerHost::create_region(&mut a, RegionKind::HotHeap, 4096).unwrap();
        // a fills its budget exactly.
        let _ = ManagerHost::alloc(&mut a, r, 64).unwrap();
        assert_eq!(a.outstanding_bytes(), 64);
        // b sees the SAME directory (it can reach the region a created)
        // but its budget accounting is independent — b's 0 outstanding.
        assert_eq!(b.outstanding_bytes(), 0);
        let _ = ManagerHost::alloc(&mut b, r, 64).unwrap();
        assert_eq!(b.outstanding_bytes(), 64);
        // Past-budget alloc on a does NOT poison b's counter.
        let denied = ManagerHost::alloc(&mut a, r, 1);
        assert!(matches!(denied, Err(TvmError::AllocationFailed)));
        assert!(a.budget_overrun());
        assert!(!b.budget_overrun());
    }
}
