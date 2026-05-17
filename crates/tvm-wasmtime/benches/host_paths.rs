//! Microbenchmarks comparing the WIT path, raw path, region-to-region copy,
//! and resolve-cache cost. Run with `cargo bench -p tvm-wasmtime`. These
//! benchmarks call the host trait impls / directory methods directly — they
//! don't go through a wasmtime instance — so they isolate the host-side
//! work from canonical-ABI lift/lower (which is the actual cost difference
//! between WIT and raw in production).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use tvm_wasmtime::bindings::tvm::memory::bytes::Host as BytesHost;
use tvm_wasmtime::bindings::tvm::memory::manager::Host as ManagerHost;
use tvm_wasmtime::bindings::tvm::memory::types::{Handle, RegionKind};
use tvm_wasmtime::TvmHost;

fn bench_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytes_read");
    for size in [4usize, 256, 64 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("wit", size), &size, |b, &size| {
            let mut host = TvmHost::new();
            let r = ManagerHost::create_region(&mut host, RegionKind::HotHeap, (size * 2) as u32)
                .unwrap();
            let h = ManagerHost::alloc(&mut host, r, size as u32).unwrap();
            BytesHost::write(&mut host, h, vec![0u8; size]).unwrap();
            b.iter(|| {
                let bytes = BytesHost::read(&mut host, h, size as u32).unwrap();
                criterion::black_box(bytes);
            });
        });
        group.bench_with_input(BenchmarkId::new("directory", size), &size, |b, &size| {
            let mut host = TvmHost::new();
            let r = ManagerHost::create_region(&mut host, RegionKind::HotHeap, (size * 2) as u32)
                .unwrap();
            let h = ManagerHost::alloc(&mut host, r, size as u32).unwrap();
            BytesHost::write(&mut host, h, vec![0u8; size]).unwrap();
            let core_h = tvm_core::Handle {
                region_id: h.region_id,
                generation: h.generation,
                offset: h.offset,
            };
            let mut buf = vec![0u8; size];
            b.iter(|| {
                host.directory.read(core_h, &mut buf).unwrap();
                criterion::black_box(&buf);
            });
        });
    }
    group.finish();
}

fn bench_copy_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_region");
    for size in [256usize, 64 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("read_then_write", size),
            &size,
            |b, &size| {
                let mut host = TvmHost::new();
                let src =
                    ManagerHost::create_region(&mut host, RegionKind::HotHeap, size as u32 * 2)
                        .unwrap();
                let dst =
                    ManagerHost::create_region(&mut host, RegionKind::Scratch, size as u32 * 2)
                        .unwrap();
                let h_src = ManagerHost::alloc(&mut host, src, size as u32).unwrap();
                let h_dst = ManagerHost::alloc(&mut host, dst, size as u32).unwrap();
                BytesHost::write(&mut host, h_src, vec![1u8; size]).unwrap();
                b.iter(|| {
                    let payload = BytesHost::read(&mut host, h_src, size as u32).unwrap();
                    BytesHost::write(&mut host, h_dst, payload).unwrap();
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("copy_region", size), &size, |b, &size| {
            let mut host = TvmHost::new();
            let src = ManagerHost::create_region(&mut host, RegionKind::HotHeap, size as u32 * 2)
                .unwrap();
            let dst = ManagerHost::create_region(&mut host, RegionKind::Scratch, size as u32 * 2)
                .unwrap();
            let h_src = ManagerHost::alloc(&mut host, src, size as u32).unwrap();
            let h_dst = ManagerHost::alloc(&mut host, dst, size as u32).unwrap();
            BytesHost::write(&mut host, h_src, vec![1u8; size]).unwrap();
            b.iter(|| {
                BytesHost::copy_region(
                    &mut host,
                    src,
                    h_src.offset,
                    dst,
                    h_dst.offset,
                    size as u32,
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_resolve_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolve");
    let mut host = TvmHost::new();
    for _ in 0..4 {
        ManagerHost::create_region(&mut host, RegionKind::HotHeap, 64).unwrap();
    }

    group.bench_function("hit", |b| {
        // Warm cache for region 0.
        host.resolve(0).unwrap();
        b.iter(|| {
            criterion::black_box(host.resolve(0).unwrap());
        });
    });

    group.bench_function("miss_then_install", |b| {
        b.iter(|| {
            host.cache.invalidate(1);
            criterion::black_box(host.resolve(1).unwrap());
        });
    });
    group.finish();
}

// Avoid unused import warning if Handle is only used in some paths.
#[allow(dead_code)]
fn _use_handle() -> Handle {
    Handle {
        region_id: 0,
        generation: 0,
        offset: 0,
    }
}

criterion_group!(benches, bench_read, bench_copy_region, bench_resolve_cache);
criterion_main!(benches);
