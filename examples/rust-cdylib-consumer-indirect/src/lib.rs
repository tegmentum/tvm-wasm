//! Indirect-call consumer.
//!
//! Exercises the path that the linker's element-segment renumbering
//! has to handle correctly: a static `[extern "C" fn() -> i32; N]`
//! table that rustc lowers into a wasm function table populated by an
//! element segment. The exported `dispatch` function takes an index
//! and `call_indirect`s through the table — every entry of which
//! points at a function whose pre-link wasm index gets shifted into
//! the merged module's index space.
//!
//! To make the shift non-trivial (so a missed renumbering would
//! manifest as a wrong-function trap), the cdylib also imports a
//! `host.bump` function from a non-`tvm_mm` module. That import is
//! forwarded by the linker into the merged module's import section
//! and prepends to the func-index space, displacing every other
//! function index by 1 compared to the pre-link wasm.

#![no_std]

use tvm_guest_mm_rt::{load_u32, store_u32, Pool};

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

const POOL: Pool = Pool::new(1);

// ---------- Host import (forwarded by the linker) ----------

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "host")]
extern "C" {
    /// Host-side counter bump — used to disambiguate which user
    /// function fired in the indirect-call tests.
    fn bump(by: u32);
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn bump(_by: u32) {}

// ---------- Indirect-call target functions ----------
//
// Each one performs a different observable side effect so the test
// can tell which one fired from outside the wasm.

extern "C" fn add_one(x: u32) -> u32 {
    unsafe { bump(1) };
    x.wrapping_add(1)
}

extern "C" fn double(x: u32) -> u32 {
    unsafe { bump(2) };
    x.wrapping_mul(2)
}

extern "C" fn store_then_negate(x: u32) -> u32 {
    // Round-trip through pool 1: store the input, read it back, return
    // its bitwise negation. Exercises both the tvm_mm path and the
    // indirect dispatch in the same call.
    store_u32(POOL, 0, x);
    let read_back = load_u32(POOL, 0);
    unsafe { bump(3) };
    !read_back
}

extern "C" fn always_42(_x: u32) -> u32 {
    unsafe { bump(4) };
    42
}

// ---------- Static function-pointer table ----------
//
// `#[no_mangle] static` on a function-pointer array lands rustc in
// the "emit a function table + element segment" path. The element
// segment is populated by an `(elem (i32.const 0) ...)` entry whose
// items are each a `ref.func N` const-expr — the exact shape that
// the linker has to renumber.

type Fp = extern "C" fn(u32) -> u32;

#[no_mangle]
pub static FUNCS: [Fp; 4] = [add_one, double, store_then_negate, always_42];

// ---------- Public dispatcher ----------
//
// Calls into the table by index. wasm-level: `call_indirect 0`
// against a funcref table. The element segment populating the table
// is what the renumbering pass has to handle.

#[no_mangle]
pub extern "C" fn dispatch(idx: u32, arg: u32) -> u32 {
    // SAFETY: bounds-checked at the wasm-trap level by call_indirect;
    // out-of-range idx traps. We deliberately don't add a Rust-level
    // bounds check so rustc keeps the indirect-call shape.
    let f = FUNCS[idx as usize];
    f(arg)
}
