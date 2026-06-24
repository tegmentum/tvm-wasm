//! Representative consumer of the rust-cdylib + multi-memory shell
//! substrate pattern.
//!
//! Mimics the **pcache** shape that the downstream sqlite-pcache-tvm
//! crate needs: multiple memory pools used as page-stores, byte-level
//! page R/W, and copy-to-default for materializing records into the
//! workload's default heap before further processing.
//!
//! The crate compiles to a wasm cdylib with `tvm_mm.*` imports; the
//! companion `tvm-guest-mm-link` static linker rewires those imports
//! and emits a single self-contained `.wasm` declaring N memory pools
//! plus the workload's exports.

#![no_std]

use tvm_guest_mm_rt::{
    copy_from_default, copy_to_default, load_u32, load_u8, store_u32, store_u8, Pool,
};

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// ---------- Page-cache-shape API ----------
//
// Two pools, used as separate page tiers. A real pcache might use 8 or
// 16 — that's just a constant in the shell template. Pool 0 is the
// default memory (workload heap); pools 1..N are pure data storage.
//
// Page layout: each page is `PAGE_SIZE` bytes. `page_off(pool_off,
// page_id) → byte_offset` is the trivial multiplication. The shell
// traps on OOB, so unsafely-wide page ids manifest as a wasm trap, not
// memory corruption.

/// Page size in bytes. 4 KiB matches sqlite's default page granularity.
pub const PAGE_SIZE: u32 = 4096;

const HOT: Pool = Pool::new(1);
const WARM: Pool = Pool::new(2);

// ---------- Byte-level page R/W ----------

/// Write a single byte to a hot-tier page.
#[no_mangle]
pub extern "C" fn write_hot_byte(page_id: u32, byte_off: u32, value: u8) {
    let off = page_id.wrapping_mul(PAGE_SIZE).wrapping_add(byte_off);
    store_u8(HOT, off, value);
}

/// Read a single byte from a hot-tier page.
#[no_mangle]
pub extern "C" fn read_hot_byte(page_id: u32, byte_off: u32) -> u8 {
    let off = page_id.wrapping_mul(PAGE_SIZE).wrapping_add(byte_off);
    load_u8(HOT, off)
}

/// Same shape for the warm tier.
#[no_mangle]
pub extern "C" fn write_warm_byte(page_id: u32, byte_off: u32, value: u8) {
    let off = page_id.wrapping_mul(PAGE_SIZE).wrapping_add(byte_off);
    store_u8(WARM, off, value);
}

#[no_mangle]
pub extern "C" fn read_warm_byte(page_id: u32, byte_off: u32) -> u8 {
    let off = page_id.wrapping_mul(PAGE_SIZE).wrapping_add(byte_off);
    load_u8(WARM, off)
}

// ---------- u32-shaped header field accessors ----------
//
// pcache uses u32 lanes for the page header (page id, checksum, etc.).
// These exercise the u32 dispatch path.

#[no_mangle]
pub extern "C" fn write_hot_u32(page_id: u32, byte_off: u32, value: u32) {
    let off = page_id.wrapping_mul(PAGE_SIZE).wrapping_add(byte_off);
    store_u32(HOT, off, value);
}

#[no_mangle]
pub extern "C" fn read_hot_u32(page_id: u32, byte_off: u32) -> u32 {
    let off = page_id.wrapping_mul(PAGE_SIZE).wrapping_add(byte_off);
    load_u32(HOT, off)
}

// ---------- Bulk page operations ----------
//
// The intended hot path: rather than dispatching per-byte, copy a full
// page into the default heap and process in place. One dispatch per
// PAGE_SIZE bytes. The shell's `tvm_copy_to_default` does the
// memory.copy with a static memory immediate per pool.
//
// `dst_off` is a pointer into the workload's default memory; the
// caller passes a buffer they own (e.g., from a `Vec<u8>` they
// pre-allocated host-side and whose pointer they passed in).

/// Materialize a page from the hot tier into the workload's default
/// memory at `dst_off`. Returns the number of bytes copied (PAGE_SIZE).
#[no_mangle]
pub extern "C" fn materialize_hot_page(page_id: u32, dst_off: u32) -> u32 {
    let src_off = page_id.wrapping_mul(PAGE_SIZE);
    copy_to_default(HOT, src_off, dst_off, PAGE_SIZE);
    PAGE_SIZE
}

/// Symmetric: write a page from default-memory buffer into the hot
/// tier. Idiomatic for "client built a page in their heap; now persist
/// it into the long-lived store".
#[no_mangle]
pub extern "C" fn install_hot_page(page_id: u32, src_off: u32) -> u32 {
    let dst_off = page_id.wrapping_mul(PAGE_SIZE);
    copy_from_default(HOT, dst_off, src_off, PAGE_SIZE);
    PAGE_SIZE
}

/// Move a page from the hot tier into the warm tier (eviction). Uses
/// `default` as a scratch staging area — this is one byte-level
/// pattern; a true intra-shell pool-to-pool copy would skip the
/// default-staging trip, but the shell's per-pool helpers handle pool↔
/// default only; cross-data-pool copy is a follow-up. For pcache,
/// hot→warm is rare enough that the double-copy is fine.
#[no_mangle]
pub extern "C" fn evict_hot_to_warm(page_id: u32, scratch_off: u32) -> u32 {
    let off = page_id.wrapping_mul(PAGE_SIZE);
    // hot → default (scratch)
    copy_to_default(HOT, off, scratch_off, PAGE_SIZE);
    // default (scratch) → warm
    copy_from_default(WARM, off, scratch_off, PAGE_SIZE);
    PAGE_SIZE
}

// ---------- Tiny header inspection: sum the page-id-checksum field ----------
//
// A more realistic micro-workload than echo. Reads the first u32 of
// each requested page and sums them — the kind of "scan over a column
// of header fields" pattern pcache uses for cache-warmup heuristics.
//
// Reads PAGE_SIZE 4-byte slots at the start of each page in [first,
// first+count). Returns the wrapping sum.

#[no_mangle]
pub extern "C" fn sum_hot_page_headers(first_page: u32, count: u32) -> u32 {
    let mut acc: u32 = 0;
    let mut i: u32 = 0;
    while i < count {
        let off = (first_page.wrapping_add(i)).wrapping_mul(PAGE_SIZE);
        acc = acc.wrapping_add(load_u32(HOT, off));
        i = i.wrapping_add(1);
    }
    acc
}
