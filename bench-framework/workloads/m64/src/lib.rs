//! M64 workload — single 64-bit linear memory, all 7 benchmark classes.
//! Source-identical structure to M32 but uses u64 pointers / indices.
//! The pointer-width difference is the critical comparison for H1.
//!
//! Build (Rust nightly + rust-src):
//!   rustup install nightly
//!   rustup +nightly component add rust-src
//!   cargo +nightly build -Zbuild-std=panic_abort,std \
//!     --target wasm64-unknown-unknown --release

#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub static mut BUFFER: [u8; 1 << 20] = [0; 1 << 20];

#[no_mangle]
pub extern "C" fn buffer_ptr() -> u64 {
    unsafe { BUFFER.as_mut_ptr() as u64 }
}

#[no_mangle]
pub extern "C" fn buffer_len() -> u64 {
    unsafe { BUFFER.len() as u64 }
}

// ---------- 4.1 sequential ----------

#[no_mangle]
pub extern "C" fn sum_sequential(ptr: u64, len: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut i: u64 = 0;
    while i < len {
        let byte = unsafe { *((ptr + i) as *const u8) };
        acc = acc.wrapping_add(byte as u64);
        i += 1;
    }
    acc
}

// ---------- 4.2 random access ----------

#[no_mangle]
pub extern "C" fn random_chase(ptr: u64, cells: u64, steps: u64) -> u64 {
    let mut idx: u64 = 0;
    let mut acc: u64 = 0;
    let mut i: u64 = 0;
    while i < steps {
        let cell_addr = ptr + (idx * 4);
        let next: u32 = unsafe { *(cell_addr as *const u32) };
        acc = acc.wrapping_add(next as u64);
        idx = (next as u64) % cells;
        i += 1;
    }
    acc
}

// ---------- 4.3 pointer-heavy linked list ----------

#[no_mangle]
pub extern "C" fn list_walk(ptr: u64, head: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut cur: u64 = head;
    let sentinel: u64 = 0xFFFF_FFFF;
    while cur != sentinel {
        let node_ptr = ptr + cur;
        let next: u32 = unsafe { *(node_ptr as *const u32) };
        let payload: u32 = unsafe { *((node_ptr + 4) as *const u32) };
        acc = acc.wrapping_add(payload as u64);
        cur = next as u64;
    }
    acc
}

// ---------- 4.4 growth ----------

#[no_mangle]
pub extern "C" fn bump_alloc_touch(ptr: u64, count: u64, size: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut i: u64 = 0;
    while i < count {
        let off = ptr + i * size;
        unsafe { *(off as *mut u8) = (i & 0xFF) as u8 };
        acc = acc.wrapping_add(unsafe { *(off as *const u8) } as u64);
        i += 1;
    }
    acc
}

// ---------- 4.5 multi-region (single linear memory layout) ----------

#[no_mangle]
pub extern "C" fn multi_region_mix(
    base: u64,
    hot_size: u64,
    warm_size: u64,
    cold_size: u64,
    iters: u64,
    seed: u32,
) -> u64 {
    let hot_start = base;
    let warm_start = base + hot_size;
    let cold_start = base + hot_size + warm_size;
    let mut state: u32 = seed.wrapping_add(1);
    let mut acc: u64 = 0;
    let mut i: u64 = 0;
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
        let off = start + ((state as u64) % size);
        acc = acc.wrapping_add(unsafe { *(off as *const u8) } as u64);
        i += 1;
    }
    acc
}

// ---------- 4.6 columnar ----------

#[no_mangle]
pub extern "C" fn columnar_filter_sum(base: u64, n: u64, threshold: u32) -> u64 {
    let col_a = base;
    let col_b = base + n * 4;
    let mut acc: u64 = 0;
    let mut i: u64 = 0;
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

#[no_mangle]
pub extern "C" fn large_ws_probe(
    base: u64,
    block_size: u64,
    n_blocks: u64,
    iters: u64,
    seed: u32,
) -> u64 {
    let mut state: u32 = seed.wrapping_add(1);
    let mut acc: u64 = 0;
    let mut i: u64 = 0;
    while i < iters {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let block = (state as u64) % n_blocks;
        let off = block * block_size + ((state as u64) % block_size);
        acc = acc.wrapping_add(unsafe { *((base + off) as *const u8) } as u64);
        i += 1;
    }
    acc
}

// ---------- 4.7 JVM ----------

#[no_mangle]
pub extern "C" fn gen_alloc_scan(ptr: u64, n: u64) -> u64 {
    let mut i: u64 = 0;
    while i < n {
        let off = ptr + i * 32;
        unsafe { *(off as *mut u32) = i as u32 };
        i += 1;
    }
    let mut acc: u64 = 0;
    let mut j: u64 = 0;
    while j < n {
        let off = ptr + j * 32;
        acc = acc.wrapping_add(unsafe { *(off as *const u32) } as u64);
        j += 1;
    }
    acc
}
