//! Async variants of [`BackingStore`] for I/O-bound implementations.
//!
//! ## When to use this
//!
//! The synchronous [`BackingStore`](crate::BackingStore) is fine when
//! `spill`/`load` are CPU-bound or local-disk-bound — the kernel's page
//! cache absorbs most of the cost. When the cold tier is genuinely I/O
//! bound (S3, network-attached storage, IPC to another process), blocking
//! the calling thread on every spill is wasteful in async runtimes.
//! Implement [`AsyncBackingStore`] instead and use the `async_*` methods
//! on [`RegionDirectory`](crate::RegionDirectory).
//!
//! ## What stays sync
//!
//! Hot-path operations (`alloc`, `dealloc`, `read`, `write`) remain
//! synchronous. They're CPU-bound and adding async machinery would only
//! impose overhead. Async support is targeted exactly at the spill/load
//! boundary and the auto-fault path that funnels through it.
//!
//! ## Compatibility
//!
//! Any `BackingStore` automatically becomes an `AsyncBackingStore` via
//! [`SyncAdapter`] — useful when callers are async but the configured
//! store happens to be sync. The reverse isn't free; calling an async
//! store from sync code requires the caller to provide an executor.

use core::future::Future;
use core::pin::Pin;

use crate::backing::BackingStore;
use crate::error::Result;

/// Future returned by `AsyncBackingStore::spill`. Pinned + boxed so the
/// trait stays object-safe; users who care about avoiding the box can
/// implement the trait by hand and return a concrete future type.
pub type SpillFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
pub type LoadFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>>;

/// Async cousin of [`BackingStore`]. Same `(region_id, generation)` keying
/// — only the methods are now async-friendly.
pub trait AsyncBackingStore: Send {
    fn spill_async<'a>(
        &'a mut self,
        region_id: u16,
        generation: u16,
        bytes: &'a [u8],
    ) -> SpillFuture<'a>;

    fn load_async<'a>(&'a mut self, region_id: u16, generation: u16) -> LoadFuture<'a>;
}

/// Adapt a sync `BackingStore` to the async trait. The returned futures
/// resolve immediately — useful when an async caller is configured with a
/// local-filesystem store and doesn't want to special-case sync vs. async
/// paths.
pub struct SyncAdapter<B>(pub B);

impl<B: BackingStore> AsyncBackingStore for SyncAdapter<B> {
    fn spill_async<'a>(
        &'a mut self,
        region_id: u16,
        generation: u16,
        bytes: &'a [u8],
    ) -> SpillFuture<'a> {
        Box::pin(async move { self.0.spill(region_id, generation, bytes) })
    }

    fn load_async<'a>(&'a mut self, region_id: u16, generation: u16) -> LoadFuture<'a> {
        Box::pin(async move { self.0.load(region_id, generation) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::{BackingStore, FileBackingStore};
    use tempfile::tempdir;

    // Trivial executor that drives a future to completion on the current
    // thread. Avoids pulling in tokio for unit tests.
    fn block_on<F: Future>(mut fut: F) -> F::Output {
        use core::task::{Context, Poll, Waker};
        let waker = Waker::noop().clone();
        let mut cx = Context::from_waker(&waker);
        let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
        loop {
            if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
                return out;
            }
        }
    }

    #[test]
    fn sync_adapter_spills_and_loads() {
        let tmp = tempdir().unwrap();
        let inner = FileBackingStore::new(tmp.path()).unwrap();
        let mut adapter = SyncAdapter(inner);

        block_on(adapter.spill_async(7, 1, b"async-spill-test")).unwrap();
        let bytes = block_on(adapter.load_async(7, 1)).unwrap();
        assert_eq!(&bytes, b"async-spill-test");
    }
}
