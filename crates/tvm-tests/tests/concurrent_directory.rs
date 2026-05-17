use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tvm_core::directory::MemoryRegion;
use tvm_core::{ConcurrentDirectory, RegionKind, Result, VecBackedRegion};

/// Custom memory region whose `write` sleeps for a fixed duration before
/// performing the actual write. Used to simulate slow per-region operations
/// so we can measure whether two threads serialize.
struct SlowRegion {
    inner: VecBackedRegion,
    sleep: Duration,
}

impl SlowRegion {
    fn new(capacity: u32, sleep: Duration) -> Self {
        Self {
            inner: VecBackedRegion::new(capacity),
            sleep,
        }
    }
}

impl MemoryRegion for SlowRegion {
    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn read(&self, offset: u32, buf: &mut [u8]) -> Result<()> {
        self.inner.read(offset, buf)
    }

    fn write(&mut self, offset: u32, buf: &[u8]) -> Result<()> {
        thread::sleep(self.sleep);
        self.inner.write(offset, buf)
    }

    fn snapshot(&self) -> Vec<u8> {
        self.inner.snapshot()
    }

    fn restore(bytes: Vec<u8>) -> Self {
        Self {
            inner: VecBackedRegion::from_bytes(bytes),
            sleep: Duration::ZERO,
        }
    }
}

#[test]
fn parallel_writes_to_distinct_regions_do_not_serialize() {
    let dir: Arc<ConcurrentDirectory<SlowRegion>> = Arc::new(ConcurrentDirectory::new());

    // r0: sleeps 200ms on every write.
    let r0 = dir
        .create_region(
            RegionKind::Scratch,
            16,
            SlowRegion::new(16, Duration::from_millis(200)),
        )
        .unwrap();
    // r1: writes immediately.
    let r1 = dir
        .create_region(RegionKind::Scratch, 16, SlowRegion::new(16, Duration::ZERO))
        .unwrap();

    let h0 = dir.alloc(r0, 4).unwrap();
    let h1 = dir.alloc(r1, 4).unwrap();

    let start = Instant::now();

    // Thread A: slow write to r0 — will hold r0's lock for ~200ms.
    let dir_a = Arc::clone(&dir);
    let thread_a = thread::spawn(move || {
        dir_a.write(h0, &[1, 2, 3, 4]).unwrap();
        Instant::now()
    });

    // Give A a moment to actually start its write.
    thread::sleep(Duration::from_millis(30));

    // Thread B: fast write to r1. With per-region locks, this should complete
    // immediately even though thread A is mid-write to r0.
    let dir_b = Arc::clone(&dir);
    let thread_b = thread::spawn(move || {
        dir_b.write(h1, &[5, 6, 7, 8]).unwrap();
        Instant::now()
    });

    let b_done = thread_b.join().unwrap();
    let a_done = thread_a.join().unwrap();

    let b_elapsed = b_done.duration_since(start);
    let a_elapsed = a_done.duration_since(start);

    // B must finish well before A — if locks serialized across regions, B
    // would wait for A's 200ms sleep to complete.
    assert!(
        b_elapsed < Duration::from_millis(150),
        "B took too long: {b_elapsed:?} (likely serialized behind A)"
    );
    assert!(
        a_elapsed > Duration::from_millis(150),
        "A finished suspiciously fast: {a_elapsed:?}"
    );
}

#[test]
fn parallel_allocs_distinct_regions() {
    let dir: Arc<ConcurrentDirectory<VecBackedRegion>> = Arc::new(ConcurrentDirectory::new());
    let mut regions = Vec::new();
    for _ in 0..8 {
        regions.push(
            dir.create_region(RegionKind::HotHeap, 1024, VecBackedRegion::new(1024))
                .unwrap(),
        );
    }

    let mut threads = Vec::new();
    for &region in &regions {
        let dir = Arc::clone(&dir);
        threads.push(thread::spawn(move || {
            for _ in 0..32 {
                dir.alloc(region, 8).unwrap();
            }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }

    for &region in &regions {
        let info = dir.region_info(region).unwrap();
        assert_eq!(info.used, 32 * 8);
    }
}

#[test]
fn destroy_serializes_against_outer_lock() {
    let dir: Arc<ConcurrentDirectory<VecBackedRegion>> = Arc::new(ConcurrentDirectory::new());
    let r = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    dir.destroy_region(r).unwrap();
    assert!(dir.region_info(r).is_err());
}

#[test]
fn cross_region_copy_locks_in_order() {
    let dir: Arc<ConcurrentDirectory<VecBackedRegion>> = Arc::new(ConcurrentDirectory::new());
    let a = dir
        .create_region(RegionKind::HotHeap, 32, VecBackedRegion::new(32))
        .unwrap();
    let b = dir
        .create_region(RegionKind::Scratch, 32, VecBackedRegion::new(32))
        .unwrap();
    let h_a = dir.alloc(a, 4).unwrap();
    dir.write(h_a, b"PING").unwrap();
    // Copy b → a (descending region IDs); locks should still acquire safely.
    dir.cross_region_copy(b, 0, a, h_a.offset, 4).unwrap();
    let mut buf = [0u8; 4];
    dir.read(h_a, &mut buf).unwrap();
    // b is zeroed by default, so the read after copy returns zeros.
    assert_eq!(&buf, &[0, 0, 0, 0]);
}
