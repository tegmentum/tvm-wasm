//! Probe: can we emit native multi-memory loads from Rust source?
//!
//! Build attempts (in order of preference, stopping at the first that works):
//!
//!   1. Stable Rust + `-Ctarget-feature=+multimemory` + inline `asm!`.
//!   2. Same but using LLVM's wasm intrinsics directly.
//!   3. Custom `extern` declarations matching wasm-bindgen's pattern.
//!
//! See `BUILD.md` in this directory for what we tried and what the
//! toolchain reports.

#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// ---------- Approach 1: inline asm targeting memory index 1 ----------
//
// The wasm `i32.load` instruction with the multi-memory proposal takes
// a memory index immediate. In LLVM's wasm asm syntax that's:
//
//     i32.load  1                  ;; loads from memory 1, default offset/align
//     i32.load  offset=0 align=2 1 ;; explicit memargs
//
// Rust's `asm!` accepts this on stable since 1.59.

#[no_mangle]
pub unsafe extern "C" fn load_u32_from_mem1(ptr: u32) -> u32 {
    let result: u32;
    core::arch::asm!(
        "local.get {ptr}",
        "i32.load 1",
        "local.set {result}",
        ptr = in(local) ptr,
        result = out(local) result,
    );
    result
}

// ---------- Approach 2: byte load, what most TVM workloads need ----------

#[no_mangle]
pub unsafe extern "C" fn load_u8_from_mem1(ptr: u32) -> u32 {
    let result: u32;
    core::arch::asm!(
        "local.get {ptr}",
        "i32.load8_u 1",
        "local.set {result}",
        ptr = in(local) ptr,
        result = out(local) result,
    );
    result
}
