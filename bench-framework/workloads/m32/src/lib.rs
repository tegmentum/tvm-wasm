//! M32 workload — single 32-bit linear memory. The runner instantiates this
//! module, writes the test data into the exported `memory`, then calls
//! `sum_sequential(ptr, len)` and times the call.
//!
//! Keeping the workload as raw as possible (no allocator, no panic
//! handlers, no std) so the engine sees clean code and bounds-check
//! elimination is uncontaminated.

#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// One slab of guest memory we own. The runner writes test bytes here
/// before calling the workload.
#[no_mangle]
pub static mut BUFFER: [u8; 1 << 20] = [0; 1 << 20]; // 1 MiB

#[no_mangle]
pub extern "C" fn buffer_ptr() -> *mut u8 {
    unsafe { BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn buffer_len() -> u32 {
    unsafe { BUFFER.len() as u32 }
}

/// Sequential workload — sum every byte in `[ptr, ptr+len)`. Returns the
/// sum so the optimizer can't elide the loads.
#[no_mangle]
pub extern "C" fn sum_sequential(ptr: u32, len: u32) -> u64 {
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < len {
        let byte = unsafe { *((ptr + i) as *const u8) };
        acc = acc.wrapping_add(byte as u64);
        i += 1;
    }
    acc
}

// ---------- 4.2 random access ----------

/// Random-access workload — N pointer-chase steps starting at the head of
/// a u32 index ring at `ptr`. Each cell holds the next index (mod cells).
/// Returns the final accumulator. Caller must seed the ring first.
#[no_mangle]
pub extern "C" fn random_chase(ptr: u32, cells: u32, steps: u32) -> u64 {
    let mut idx: u32 = 0;
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < steps {
        let cell_addr = ptr + (idx * 4);
        let next: u32 = unsafe { *(cell_addr as *const u32) };
        acc = acc.wrapping_add(next as u64);
        idx = next % cells;
        i += 1;
    }
    acc
}

// ---------- 4.3 pointer-heavy (linked list) ----------

/// Linked-list traversal. Each node is `[next: u32, payload: u32]` (8B).
/// `head` is the offset of the first node. Walk until next == 0xFFFF_FFFF.
#[no_mangle]
pub extern "C" fn list_walk(ptr: u32, head: u32) -> u64 {
    let mut acc: u64 = 0;
    let mut cur = head;
    let sentinel: u32 = 0xFFFF_FFFF;
    while cur != sentinel {
        let node_ptr = ptr + cur;
        let next: u32 = unsafe { *(node_ptr as *const u32) };
        let payload: u32 = unsafe { *((node_ptr + 4) as *const u32) };
        acc = acc.wrapping_add(payload as u64);
        cur = next;
    }
    acc
}

// ---------- 4.4 growth ----------

/// Bump-allocate `count` blocks of `size` bytes each starting at `ptr`,
/// touching the first byte of each block. Returns the touched-byte sum.
#[no_mangle]
pub extern "C" fn bump_alloc_touch(ptr: u32, count: u32, size: u32) -> u64 {
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < count {
        let off = ptr + i * size;
        unsafe { *(off as *mut u8) = (i & 0xFF) as u8 };
        acc = acc.wrapping_add(unsafe { *(off as *const u8) } as u64);
        i += 1;
    }
    acc
}

// ---------- 4.5 multi-region ----------
//
// Layout in the single linear memory:
//   [0 .. hot_size)                    — hot
//   [hot_size .. hot_size + warm_size) — warm
//   [hot_size + warm_size .. )         — cold
// Workload: 90% reads from hot, 9% from warm, 1% from cold, indexed by an
// xorshift PRNG seeded from `seed`. Returns the accumulator.
#[no_mangle]
pub extern "C" fn multi_region_mix(
    base: u32,
    hot_size: u32,
    warm_size: u32,
    cold_size: u32,
    iters: u32,
    seed: u32,
) -> u64 {
    let hot_start = base;
    let warm_start = base + hot_size;
    let cold_start = base + hot_size + warm_size;
    let mut state: u32 = seed.wrapping_add(1);
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < iters {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let bucket = state % 100;
        let (start, size) = if bucket < 90 {
            (hot_start, hot_size)
        } else if bucket < 99 {
            (warm_start, warm_size)
        } else {
            (cold_start, cold_size)
        };
        let off = start + (state % size);
        acc = acc.wrapping_add(unsafe { *(off as *const u8) } as u64);
        i += 1;
    }
    acc
}

// ---------- 4.6 database (columnar) ----------

/// Two columns laid out one after the other. `col_a` is u32 keys at
/// [base .. base + n*4); `col_b` is u32 values at [base + n*4 .. base +
/// n*8). Filter `key < threshold` and sum the matching values.
#[no_mangle]
pub extern "C" fn columnar_filter_sum(base: u32, n: u32, threshold: u32) -> u64 {
    let col_a = base;
    let col_b = base + n * 4;
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < n {
        let k: u32 = unsafe { *((col_a + i * 4) as *const u32) };
        if k < threshold {
            let v: u32 = unsafe { *((col_b + i * 4) as *const u32) };
            acc = acc.wrapping_add(v as u64);
        }
        i += 1;
    }
    acc
}

// ---------- 4.8 large-working-set probe ----------
//
// The motivating scenario: working set spans more bytes than one wasm32
// memory can hold. The proxy here is "many random reads scattered across
// `n_regions` blocks of `block_size` bytes, all packed into one M32
// memory." M64 / TVM run the equivalent. The interesting comparison is
// the M64 row (which is what real >4GB workloads must use today) vs TVM.
#[no_mangle]
pub extern "C" fn large_ws_probe(
    base: u32,
    block_size: u32,
    n_blocks: u32,
    iters: u32,
    seed: u32,
) -> u64 {
    let mut state: u32 = seed.wrapping_add(1);
    let mut acc: u64 = 0;
    let mut i = 0u32;
    while i < iters {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let block = state % n_blocks;
        let off = block * block_size + (state % block_size);
        acc = acc.wrapping_add(unsafe { *((base + off) as *const u8) } as u64);
        i += 1;
    }
    acc
}

// ---------- 4.7 JVM heap simulation ----------

/// Bump-allocate `n` 32-byte objects then "scan" them by reading the
/// first 4 bytes of each. Approximates a young-generation walk.
#[no_mangle]
pub extern "C" fn gen_alloc_scan(ptr: u32, n: u32) -> u64 {
    let mut i = 0u32;
    while i < n {
        let off = ptr + i * 32;
        unsafe { *(off as *mut u32) = i };
        i += 1;
    }
    let mut acc: u64 = 0;
    let mut j = 0u32;
    while j < n {
        let off = ptr + j * 32;
        acc = acc.wrapping_add(unsafe { *(off as *const u32) } as u64);
        j += 1;
    }
    acc
}
