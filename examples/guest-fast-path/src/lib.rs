//! Guest demo for the raw fast path.
//!
//! The host pre-creates a region (id 0) before instantiation; this guest
//! exports a single function that allocates inside it, writes a 4-byte
//! pattern, reads it back, and returns the sum (10).
//!
//! Compare with `examples/guest-demo/`, which does the same thing through
//! the WIT-generated bindings — same logic, ~2-3x more bytes copied.

use tvm_guest_rt::Region;

#[no_mangle]
pub extern "C" fn run_test() -> u32 {
    let region = Region::from_id(0);
    let h = region.alloc(4).expect("alloc");
    h.write(&[1, 2, 3, 4]).expect("write");
    let mut buf = [0u8; 4];
    h.read(&mut buf).expect("read");
    buf.iter().map(|b| *b as u32).sum()
}
