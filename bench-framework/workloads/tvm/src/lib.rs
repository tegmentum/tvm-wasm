//! TVM workload (optimized). All workloads use **bulk reads** — a single
//! `tvm.read` call to pull the working set into guest linear memory, then
//! the access pattern runs purely on guest memory. This is the proper way
//! to use the TVM raw fast path: amortize the host-call cost across many
//! bytes per call, not pay it per cell.
//!
//! Compare to the v1 implementation in git history, which paid one host
//! call per byte and was 50–150× slower on hot inner loops.

use tvm_guest_rt::{Region, RegionPtr};

const CHUNK: usize = 65536; // 64 KiB; one read per chunk

#[no_mangle]
pub extern "C" fn tvm_sum_sequential(handle_packed: i64, len: u32) -> u64 {
    // Bulk-read into a guest static buffer 64KB at a time, sum locally.
    static mut SCRATCH: [u8; CHUNK] = [0; CHUNK];
    let ptr = RegionPtr { packed: handle_packed };
    let mut acc: u64 = 0;
    let mut consumed: u32 = 0;
    while consumed < len {
        let to_read = core::cmp::min(CHUNK as u32, len - consumed);
        let buf = unsafe { &mut SCRATCH[..to_read as usize] };
        let _ = ptr.read(buf);
        for &b in buf.iter() {
            acc = acc.wrapping_add(b as u64);
        }
        consumed += to_read;
    }
    acc
}

/// Helper used by the runner: allocate a region+handle of the requested
/// size in the well-known region 0, return the packed handle.
#[no_mangle]
pub extern "C" fn tvm_alloc_in_region0(size: u32) -> i64 {
    let region = Region::from_id(0);
    region
        .alloc(size)
        .map(|p| p.packed)
        .unwrap_or(0)
}

/// Bulk write helper: write `len` bytes from local guest memory into the
/// TVM region. The runner uses this to seed the test data.
#[no_mangle]
pub extern "C" fn tvm_write_bytes(handle_packed: i64, src_ptr: u32, len: u32) -> i32 {
    let ptr = RegionPtr { packed: handle_packed };
    let slice = unsafe { core::slice::from_raw_parts(src_ptr as *const u8, len as usize) };
    match ptr.write(slice) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

// ---------- 4.2 random access ----------
//
// Bulk-read the entire ring once, then chase in guest memory. One host
// call regardless of step count.
const RANDOM_BUF_SIZE: usize = 1 << 20; // 1 MiB; covers our test sizes
static mut RANDOM_BUF: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];

#[no_mangle]
pub extern "C" fn tvm_random_chase(handle_packed: i64, cells: u32, steps: u32) -> u64 {
    let ptr = RegionPtr { packed: handle_packed };
    let len = (cells * 4) as usize;
    let buf = unsafe { &mut RANDOM_BUF[..len] };
    let _ = ptr.read(buf);
    // Now chase locally — exactly the M32 inner loop.
    let mut idx: u32 = 0;
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < steps {
        let off = (idx * 4) as usize;
        let next = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        acc = acc.wrapping_add(next as u64);
        idx = next % cells;
        i += 1;
    }
    acc
}

// ---------- 4.3 pointer-heavy ----------
//
// Bulk-read all nodes once; walk locally.
static mut LIST_BUF: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];

#[no_mangle]
pub extern "C" fn tvm_list_walk(handle_packed: i64, head: u32, total_bytes: u32) -> u64 {
    let ptr = RegionPtr { packed: handle_packed };
    let buf = unsafe { &mut LIST_BUF[..total_bytes as usize] };
    let _ = ptr.read(buf);
    let mut acc: u64 = 0;
    let mut cur = head;
    let sentinel: u32 = 0xFFFF_FFFF;
    while cur != sentinel {
        let off = cur as usize;
        let next =
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        let payload = u32::from_le_bytes([
            buf[off + 4],
            buf[off + 5],
            buf[off + 6],
            buf[off + 7],
        ]);
        acc = acc.wrapping_add(payload as u64);
        cur = next;
    }
    acc
}

// ---------- 4.4 growth ----------
//
// Bulk-alloc-once pattern: a single alloc of count*size bytes, then build
// the data in a guest static buffer and bulk-write it. The fair comparison
// to M32's "fill an array of count slots" — both do the same operation,
// not "alloc N times" vs "no allocs at all."
static mut GROWTH_BUF: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];

#[no_mangle]
pub extern "C" fn tvm_bump_alloc_touch(region_id: u32, count: u32, size: u32) -> u64 {
    let region = Region::from_id(region_id as u16);
    let total = (count * size) as usize;
    let buf = unsafe { &mut GROWTH_BUF[..total] };
    // Fill in guest memory.
    let mut i: u32 = 0;
    while i < count {
        let off = (i * size) as usize;
        buf[off] = (i & 0xFF) as u8;
        i += 1;
    }
    // One alloc, one write.
    let ptr = match region.alloc(count * size) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let _ = ptr.write(buf);
    // Read back to mimic the M32 pattern of touching each slot.
    let _ = ptr.read(buf);
    let mut acc: u64 = 0;
    let mut j: u32 = 0;
    while j < count {
        let off = (j * size) as usize;
        acc = acc.wrapping_add(buf[off] as u64);
        j += 1;
    }
    acc
}

// ---------- 4.5 multi-region ----------
//
// THE critical proof. Three separate regions for hot/warm/cold. Bulk-read
// each region once, then run the access pattern locally. Three host calls
// total regardless of iter count.
static mut MR_HOT: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];
static mut MR_WARM: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];
static mut MR_COLD: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];

#[no_mangle]
pub extern "C" fn tvm_multi_region_mix(
    hot_packed: i64,
    hot_size: u32,
    warm_packed: i64,
    warm_size: u32,
    cold_packed: i64,
    cold_size: u32,
    iters: u32,
    seed: u32,
) -> u64 {
    let hot_ptr = RegionPtr { packed: hot_packed };
    let warm_ptr = RegionPtr { packed: warm_packed };
    let cold_ptr = RegionPtr { packed: cold_packed };
    let hot = unsafe { &mut MR_HOT[..hot_size as usize] };
    let warm = unsafe { &mut MR_WARM[..warm_size as usize] };
    let cold = unsafe { &mut MR_COLD[..cold_size as usize] };
    let _ = hot_ptr.read(hot);
    let _ = warm_ptr.read(warm);
    let _ = cold_ptr.read(cold);
    let mut state: u32 = seed.wrapping_add(1);
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < iters {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let bucket = state % 100;
        let v = if bucket < 90 {
            hot[(state as usize) % hot.len()]
        } else if bucket < 99 {
            warm[(state as usize) % warm.len()]
        } else {
            cold[(state as usize) % cold.len()]
        };
        acc = acc.wrapping_add(v as u64);
        i += 1;
    }
    acc
}

// ---------- 4.6 database (columnar) ----------
//
// Bulk-read each column once, then filter+sum locally. Two host calls
// total regardless of n.
static mut COL_A_BUF: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];
static mut COL_B_BUF: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];

#[no_mangle]
pub extern "C" fn tvm_columnar_filter_sum(
    col_a_packed: i64,
    col_b_packed: i64,
    n: u32,
    threshold: u32,
) -> u64 {
    let col_a_ptr = RegionPtr { packed: col_a_packed };
    let col_b_ptr = RegionPtr { packed: col_b_packed };
    let bytes = (n * 4) as usize;
    let col_a = unsafe { &mut COL_A_BUF[..bytes] };
    let col_b = unsafe { &mut COL_B_BUF[..bytes] };
    let _ = col_a_ptr.read(col_a);
    let _ = col_b_ptr.read(col_b);
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < n {
        let off = (i * 4) as usize;
        let k = u32::from_le_bytes([
            col_a[off],
            col_a[off + 1],
            col_a[off + 2],
            col_a[off + 3],
        ]);
        if k < threshold {
            let v = u32::from_le_bytes([
                col_b[off],
                col_b[off + 1],
                col_b[off + 2],
                col_b[off + 3],
            ]);
            acc = acc.wrapping_add(v as u64);
        }
        i += 1;
    }
    acc
}

// ---------- 4.8 large-working-set probe ----------
//
// TVM's natural shape: one region per block. Bulk-read each block once
// (warming up the cache) then run the probe locally. The runner sets
// up `n_blocks` regions and packs handles into a guest array.
static mut LWS_BUF: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];

#[no_mangle]
pub extern "C" fn tvm_large_ws_probe(
    handles_ptr: u32,
    n_blocks: u32,
    block_size: u32,
    iters: u32,
    seed: u32,
) -> u64 {
    // Read all blocks into one packed guest buffer (sequential layout).
    let total = (n_blocks * block_size) as usize;
    let buf = unsafe { &mut LWS_BUF[..total] };
    for i in 0..n_blocks {
        let h_off = (handles_ptr + i * 8) as usize;
        let packed = unsafe { *(h_off as *const i64) };
        let ptr = RegionPtr { packed };
        let block_off = (i * block_size) as usize;
        let _ = ptr.read(&mut buf[block_off..block_off + block_size as usize]);
    }
    let mut state: u32 = seed.wrapping_add(1);
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < iters {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let block = state % n_blocks;
        let off = block * block_size + (state % block_size);
        acc = acc.wrapping_add(buf[off as usize] as u64);
        i += 1;
    }
    acc
}

// ---------- 4.7 JVM heap ----------
//
// Bump-allocate `n` 32-byte objects in a region, then walk them.
// Bulk-alloc-once pattern for the JVM workload too. Single alloc of n*32
// bytes; build the headers in a guest static buffer; one write, one read.
static mut JVM_BUF: [u8; RANDOM_BUF_SIZE] = [0; RANDOM_BUF_SIZE];

#[no_mangle]
pub extern "C" fn tvm_gen_alloc_scan(region_id: u32, n: u32) -> u64 {
    let region = Region::from_id(region_id as u16);
    let total = (n * 32) as usize;
    let buf = unsafe { &mut JVM_BUF[..total] };
    // Build headers in guest memory: each 32-byte object's first 4 bytes
    // hold its index.
    let mut i = 0u32;
    while i < n {
        let off = (i * 32) as usize;
        let bytes = i.to_le_bytes();
        buf[off..off + 4].copy_from_slice(&bytes);
        i += 1;
    }
    let ptr = match region.alloc(n * 32) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let _ = ptr.write(buf);
    let _ = ptr.read(buf);
    let mut acc: u64 = 0;
    let mut j = 0u32;
    while j < n {
        let off = (j * 32) as usize;
        let v = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        acc = acc.wrapping_add(v as u64);
        j += 1;
    }
    acc
}
