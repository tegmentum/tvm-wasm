use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default, Debug)]
pub struct RegionMetrics {
    pub allocations: AtomicU64,
    pub bytes_allocated: AtomicU64,
    pub bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,
    pub faults: AtomicU64,
    pub promotions: AtomicU64,
    pub demotions: AtomicU64,
}

impl RegionMetrics {
    #[cfg(feature = "metrics")]
    #[inline(always)]
    pub fn record_alloc(&self, bytes: u64) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_add(bytes, Ordering::Relaxed);
    }
    #[cfg(not(feature = "metrics"))]
    #[inline(always)]
    pub fn record_alloc(&self, _bytes: u64) {}

    #[cfg(feature = "metrics")]
    #[inline(always)]
    pub fn record_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }
    #[cfg(not(feature = "metrics"))]
    #[inline(always)]
    pub fn record_read(&self, _bytes: u64) {}

    #[cfg(feature = "metrics")]
    #[inline(always)]
    pub fn record_write(&self, bytes: u64) {
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }
    #[cfg(not(feature = "metrics"))]
    #[inline(always)]
    pub fn record_write(&self, _bytes: u64) {}

    #[cfg(feature = "metrics")]
    #[inline(always)]
    pub fn record_fault(&self) {
        self.faults.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(not(feature = "metrics"))]
    #[inline(always)]
    pub fn record_fault(&self) {}

    #[cfg(feature = "metrics")]
    #[inline(always)]
    pub fn record_promotion(&self) {
        self.promotions.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(not(feature = "metrics"))]
    #[inline(always)]
    pub fn record_promotion(&self) {}

    #[cfg(feature = "metrics")]
    #[inline(always)]
    pub fn record_demotion(&self) {
        self.demotions.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(not(feature = "metrics"))]
    #[inline(always)]
    pub fn record_demotion(&self) {}

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            faults: self.faults.load(Ordering::Relaxed),
            promotions: self.promotions.load(Ordering::Relaxed),
            demotions: self.demotions.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetricsSnapshot {
    pub allocations: u64,
    pub bytes_allocated: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub faults: u64,
    pub promotions: u64,
    pub demotions: u64,
}
