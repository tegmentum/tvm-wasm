//! End-to-end tests for the async spill/load path. Uses a hand-rolled
//! single-thread executor (no tokio dep) and a custom async backing that
//! records call counts and yields once per operation, proving that the
//! async path actually awaits.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use tvm_core::async_backing::{AsyncBackingStore, LoadFuture, SpillFuture};
use tvm_core::{RegionDirectory, RegionKind, Residency, Result, TvmError, VecBackedRegion};

// ---------- Tiny block-on executor ----------

fn block_on<F: Future>(mut fut: F) -> F::Output {
    let waker = Waker::noop().clone();
    let mut cx = Context::from_waker(&waker);
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
            return out;
        }
    }
}

// ---------- Yield-once future ----------
//
// Returns Pending the first time it's polled, then Ready forever. Lets us
// prove the async path actually awaits — a fully-sync impl would never
// see Pending.

struct YieldOnce {
    yielded: bool,
    value: Option<std::result::Result<Vec<u8>, TvmError>>,
}

impl Future for YieldOnce {
    type Output = std::result::Result<Vec<u8>, TvmError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        Poll::Ready(self.value.take().unwrap())
    }
}

// ---------- Test backing: in-memory + yields once per op ----------

struct AsyncTestBacking {
    storage: HashMap<(u16, u16), Vec<u8>>,
    spill_calls: Arc<AtomicU32>,
    load_calls: Arc<AtomicU32>,
}

impl AsyncTestBacking {
    fn new() -> Self {
        Self {
            storage: HashMap::new(),
            spill_calls: Arc::new(AtomicU32::new(0)),
            load_calls: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl AsyncBackingStore for AsyncTestBacking {
    fn spill_async<'a>(
        &'a mut self,
        region_id: u16,
        generation: u16,
        bytes: &'a [u8],
    ) -> SpillFuture<'a> {
        self.spill_calls.fetch_add(1, Ordering::Relaxed);
        let bytes = bytes.to_vec();
        Box::pin(async move {
            // Simulate async work by yielding once.
            YieldOnce {
                yielded: false,
                value: Some(Ok(Vec::new())),
            }
            .await
            .ok();
            self.storage.insert((region_id, generation), bytes);
            Ok(())
        })
    }

    fn load_async<'a>(&'a mut self, region_id: u16, generation: u16) -> LoadFuture<'a> {
        self.load_calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            // Yield once before the synchronous lookup.
            YieldOnce {
                yielded: false,
                value: Some(Ok(Vec::new())),
            }
            .await
            .ok();
            self.storage
                .get(&(region_id, generation))
                .cloned()
                .ok_or_else(|| TvmError::BackingStore("no such key".into()))
        })
    }
}

// ---------- Tests ----------

#[test]
fn async_spill_then_load_round_trips() -> Result<()> {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(r, 16).unwrap();
    dir.write(h, b"async-payload-16").unwrap();

    let mut backing = AsyncTestBacking::new();
    block_on(dir.spill_region_async(r, &mut backing))?;
    assert_eq!(backing.spill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Cold);

    let mut buf = [0u8; 16];
    assert!(matches!(dir.read(h, &mut buf), Err(TvmError::NotResident)));

    block_on(dir.load_region_async(r, &mut backing))?;
    assert_eq!(backing.load_calls.load(Ordering::Relaxed), 1);
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);

    dir.read(h, &mut buf).unwrap();
    assert_eq!(&buf, b"async-payload-16");
    Ok(())
}

#[test]
fn async_read_or_fault_auto_loads_cold_region() -> Result<()> {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(r, 8).unwrap();
    dir.write(h, b"AUTOFAUL").unwrap();

    let mut backing = AsyncTestBacking::new();
    block_on(dir.spill_region_async(r, &mut backing))?;

    let mut buf = [0u8; 8];
    block_on(dir.read_or_fault_async(h, &mut buf, &mut backing))?;
    assert_eq!(&buf, b"AUTOFAUL");
    assert_eq!(backing.load_calls.load(Ordering::Relaxed), 1);
    assert_eq!(dir.region_info(r).unwrap().residency, Residency::Hot);
    Ok(())
}

#[test]
fn async_write_or_fault_auto_loads() -> Result<()> {
    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 16, VecBackedRegion::new(16))
        .unwrap();
    let h = dir.alloc(r, 4).unwrap();
    dir.write(h, b"OLD!").unwrap();

    let mut backing = AsyncTestBacking::new();
    block_on(dir.spill_region_async(r, &mut backing))?;

    block_on(dir.write_or_fault_async(h, b"NEW!", &mut backing))?;
    let mut buf = [0u8; 4];
    dir.read(h, &mut buf).unwrap();
    assert_eq!(&buf, b"NEW!");
    Ok(())
}

#[test]
fn async_path_actually_yields() -> Result<()> {
    // Verify the async path goes through Pending → poll → Ready, not just
    // sync-collapse. Counts how many times the executor polled the future.

    struct CountingPoller {
        polls: u32,
    }

    fn drive_count<F: Future>(mut fut: F) -> (F::Output, u32) {
        let waker = Waker::noop().clone();
        let mut cx = Context::from_waker(&waker);
        let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
        let mut polls = 0u32;
        loop {
            polls += 1;
            if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
                return (out, polls);
            }
        }
    }
    let _ = CountingPoller { polls: 0 };

    let mut dir: RegionDirectory<VecBackedRegion> = RegionDirectory::new();
    let r = dir
        .create_region(RegionKind::ObjectArena, 32, VecBackedRegion::new(32))
        .unwrap();
    let h = dir.alloc(r, 4).unwrap();
    dir.write(h, b"DATA").unwrap();
    let mut backing = AsyncTestBacking::new();

    let (_, polls) = drive_count(dir.spill_region_async(r, &mut backing));
    assert!(
        polls >= 2,
        "expected at least 2 polls (Pending + Ready), got {polls}"
    );
    Ok(())
}
