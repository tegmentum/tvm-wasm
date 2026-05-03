//! End-to-end integration test for the `tvm-guest-rt` API surface.
//!
//! `tvm-guest-rt` only declares its `extern "C"` imports when compiled
//! to `wasm32-*` targets, so we can't unit-test the typed load/store
//! and reducer methods host-side directly. Instead, this test loads a
//! WAT module that **faithfully reproduces** what `tvm-guest-rt`
//! emits — same raw imports, same byte arithmetic, same call sequence
//! — and exercises every reducer + typed-accessor path.
//!
//! If the host-side `tvm.<op>` ABI ever drifts, this test catches it
//! before users running real guest code do.

use anyhow::Result;
use tvm_core::RegionKind;
use tvm_wasmtime::{add_raw_imports, TvmHost};
use wasmtime::{Config, Engine, Linker, Module, Store};

// ----------------------------------------------------------------------
// Typed load/store round-trip — mirrors the byte arithmetic in
// `tvm-guest-rt`'s `load_u32_le` / `store_u32_le` etc.
//
// For a u32 LE load: tvm-guest-rt does `raw::read(packed, dst_ptr, 4)`
// then `u32::from_le_bytes(dst)`. The WAT below does the same: calls
// `tvm.read` to copy 4 bytes into a stack scratch slot, then loads
// them with i32.load (which is little-endian on every wasm engine).
// ----------------------------------------------------------------------

const TYPED_WAT: &str = r#"
(module
  (import "tvm" "read"  (func $read  (param i64 i32 i32) (result i32)))
  (import "tvm" "write" (func $write (param i64 i32 i32) (result i32)))
  (memory (export "memory") 1)

  ;; load_u8 — emulates RegionPtr::load_u8(offset).
  (func (export "load_u8")
        (param $packed i64) (param $delta i32) (result i32)
    ;; Adjust packed by delta in the low 32 bits (matches with_offset).
    (drop (call $read
      (i64.or
        (i64.and (local.get $packed) (i64.const 0xFFFFFFFF00000000))
        (i64.and
          (i64.add
            (i64.and (local.get $packed) (i64.const 0xFFFFFFFF))
            (i64.extend_i32_u (local.get $delta)))
          (i64.const 0xFFFFFFFF)))
      (i32.const 0) (i32.const 1)))
    (i32.load8_u (i32.const 0)))

  (func (export "store_u8")
        (param $packed i64) (param $delta i32) (param $val i32) (result i32)
    (i32.store8 (i32.const 0) (local.get $val))
    (call $write
      (i64.or
        (i64.and (local.get $packed) (i64.const 0xFFFFFFFF00000000))
        (i64.and
          (i64.add
            (i64.and (local.get $packed) (i64.const 0xFFFFFFFF))
            (i64.extend_i32_u (local.get $delta)))
          (i64.const 0xFFFFFFFF)))
      (i32.const 0) (i32.const 1)))

  (func (export "load_u32_le")
        (param $packed i64) (param $delta i32) (result i32)
    (drop (call $read
      (i64.or
        (i64.and (local.get $packed) (i64.const 0xFFFFFFFF00000000))
        (i64.and
          (i64.add
            (i64.and (local.get $packed) (i64.const 0xFFFFFFFF))
            (i64.extend_i32_u (local.get $delta)))
          (i64.const 0xFFFFFFFF)))
      (i32.const 0) (i32.const 4)))
    (i32.load (i32.const 0)))

  (func (export "store_u32_le")
        (param $packed i64) (param $delta i32) (param $val i32) (result i32)
    (i32.store (i32.const 0) (local.get $val))
    (call $write
      (i64.or
        (i64.and (local.get $packed) (i64.const 0xFFFFFFFF00000000))
        (i64.and
          (i64.add
            (i64.and (local.get $packed) (i64.const 0xFFFFFFFF))
            (i64.extend_i32_u (local.get $delta)))
          (i64.const 0xFFFFFFFF)))
      (i32.const 0) (i32.const 4)))

  (func (export "load_i64_le")
        (param $packed i64) (param $delta i32) (result i64)
    (drop (call $read
      (i64.or
        (i64.and (local.get $packed) (i64.const 0xFFFFFFFF00000000))
        (i64.and
          (i64.add
            (i64.and (local.get $packed) (i64.const 0xFFFFFFFF))
            (i64.extend_i32_u (local.get $delta)))
          (i64.const 0xFFFFFFFF)))
      (i32.const 0) (i32.const 8)))
    (i64.load (i32.const 0)))

  (func (export "store_i64_le")
        (param $packed i64) (param $delta i32) (param $val i64) (result i32)
    (i64.store (i32.const 0) (local.get $val))
    (call $write
      (i64.or
        (i64.and (local.get $packed) (i64.const 0xFFFFFFFF00000000))
        (i64.and
          (i64.add
            (i64.and (local.get $packed) (i64.const 0xFFFFFFFF))
            (i64.extend_i32_u (local.get $delta)))
          (i64.const 0xFFFFFFFF)))
      (i32.const 0) (i32.const 8)))
)
"#;

fn setup_with_region(size: u32) -> Result<(Store<TvmHost>, i64)> {
    let host = TvmHost::new();
    let engine = Engine::new(&Config::new())?;
    let mut store = Store::new(&engine, host);
    let r = store.data_mut().create_region(RegionKind::HotHeap, size + 256)?;
    let h = store.data_mut().alloc(r, size)?;
    Ok((store, h.pack() as i64))
}

#[test]
fn typed_u8_round_trip() -> Result<()> {
    let (mut store, packed) = setup_with_region(64)?;
    let module = Module::new(store.engine(), TYPED_WAT)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let inst = linker.instantiate(&mut store, &module)?;

    let store_u8 = inst.get_typed_func::<(i64, i32, i32), i32>(&mut store, "store_u8")?;
    let load_u8  = inst.get_typed_func::<(i64, i32), i32>(&mut store, "load_u8")?;

    for off in 0..64i32 {
        store_u8.call(&mut store, (packed, off, ((off ^ 0xa5) & 0xff)))?;
    }
    for off in 0..64i32 {
        let got = load_u8.call(&mut store, (packed, off))?;
        assert_eq!(got, (off ^ 0xa5) & 0xff, "u8 round-trip at offset {}", off);
    }
    Ok(())
}

#[test]
fn typed_u32_le_round_trip() -> Result<()> {
    let (mut store, packed) = setup_with_region(64)?;
    let module = Module::new(store.engine(), TYPED_WAT)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let inst = linker.instantiate(&mut store, &module)?;

    let store_u32 = inst.get_typed_func::<(i64, i32, i32), i32>(&mut store, "store_u32_le")?;
    let load_u32  = inst.get_typed_func::<(i64, i32), i32>(&mut store, "load_u32_le")?;

    let values = [0u32, 1, 0xdeadbeef, 0xffffffff, 0x80000000];
    for (i, &v) in values.iter().enumerate() {
        let off = (i * 4) as i32;
        store_u32.call(&mut store, (packed, off, v as i32))?;
    }
    for (i, &v) in values.iter().enumerate() {
        let off = (i * 4) as i32;
        let got = load_u32.call(&mut store, (packed, off))? as u32;
        assert_eq!(got, v, "u32 round-trip at offset {}", off);
    }
    Ok(())
}

#[test]
fn typed_i64_le_round_trip() -> Result<()> {
    let (mut store, packed) = setup_with_region(128)?;
    let module = Module::new(store.engine(), TYPED_WAT)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let inst = linker.instantiate(&mut store, &module)?;

    let store_i64 = inst.get_typed_func::<(i64, i32, i64), i32>(&mut store, "store_i64_le")?;
    let load_i64  = inst.get_typed_func::<(i64, i32), i64>(&mut store, "load_i64_le")?;

    let values = [
        0i64, -1, i64::MAX, i64::MIN, 0x0102030405060708, -0x0102030405060708,
    ];
    for (i, &v) in values.iter().enumerate() {
        let off = (i * 8) as i32;
        store_i64.call(&mut store, (packed, off, v))?;
    }
    for (i, &v) in values.iter().enumerate() {
        let off = (i * 8) as i32;
        let got = load_i64.call(&mut store, (packed, off))?;
        assert_eq!(got, v, "i64 round-trip at offset {}", off);
    }
    Ok(())
}

#[test]
fn typed_load_against_host_seeded_bytes() -> Result<()> {
    // Seed the region's bytes via TvmHost::write_bytes, then read them
    // through the typed load path. Checks the with_offset arithmetic
    // by reading at non-zero deltas.
    let (mut store, packed) = setup_with_region(64)?;
    let h = tvm_core::Handle::unpack(packed as u64);
    let mut bytes = [0u8; 64];
    for i in 0..64 {
        bytes[i] = (i as u8).wrapping_mul(7).wrapping_add(0x42);
    }
    store.data_mut().write_bytes(h, &bytes)?;

    let module = Module::new(store.engine(), TYPED_WAT)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    let inst = linker.instantiate(&mut store, &module)?;
    let load_u32 = inst.get_typed_func::<(i64, i32), i32>(&mut store, "load_u32_le")?;
    let load_u8 = inst.get_typed_func::<(i64, i32), i32>(&mut store, "load_u8")?;

    // u8 at every offset
    for off in 0..64i32 {
        let got = load_u8.call(&mut store, (packed, off))? as u8;
        assert_eq!(got, bytes[off as usize], "u8 at +{}", off);
    }
    // u32 LE at aligned offsets
    for off in (0..60i32).step_by(4) {
        let got = load_u32.call(&mut store, (packed, off))? as u32;
        let want = u32::from_le_bytes(bytes[off as usize..off as usize + 4].try_into()?);
        assert_eq!(got, want, "u32 at +{}", off);
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Reducer raw imports — every safe method on `RegionPtr` (in
// tvm-guest-rt) calls one of these. Validating each one via WAT is
// equivalent to validating the safe wrapper that delegates to it.
// ----------------------------------------------------------------------

const REDUCER_WAT: &str = r#"
(module
  (import "tvm" "sum_u8"          (func $sum_u8 (param i64 i32) (result i64)))
  (import "tvm" "sum_u32_le"      (func $sum_u32 (param i64 i32) (result i64)))
  (import "tvm" "max_u32_le"      (func $max_u32 (param i64 i32) (result i64)))
  (import "tvm" "count_byte"      (func $count_byte (param i64 i32 i32) (result i32)))
  (import "tvm" "count_in_range"  (func $count_in_range (param i64 i32 i32 i32) (result i32)))
  (import "tvm" "popcount"        (func $popcount (param i64 i32) (result i64)))
  (import "tvm" "min_max_u8"      (func $min_max (param i64 i32) (result i32)))
  (import "tvm" "find_byte"       (func $find_byte (param i64 i32 i32) (result i32)))
  (import "tvm" "eq"              (func $eq (param i64 i64 i32) (result i32)))
  (import "tvm" "lex_cmp"         (func $lex_cmp (param i64 i64 i32) (result i32)))
  (import "tvm" "hash_fnv1a"      (func $hash (param i64 i32) (result i64)))
  (import "tvm" "and_fold_u8"     (func $and_fold (param i64 i32) (result i32)))
  (import "tvm" "or_fold_u8"      (func $or_fold (param i64 i32) (result i32)))
  (import "tvm" "xor_fold_u8"     (func $xor_fold (param i64 i32) (result i32)))
  (import "tvm" "fill"            (func $fill (param i64 i32 i32) (result i32)))
  (import "tvm" "xor_with_byte"   (func $xor_with_byte (param i64 i32 i32) (result i32)))
  (import "tvm" "xor_into_region" (func $xor_into_region (param i64 i64 i32) (result i32)))
  (memory (export "memory") 1)

  (func (export "call_sum_u8") (param $h i64) (param $n i32) (result i64)
    (call $sum_u8 (local.get $h) (local.get $n)))
  (func (export "call_sum_u32") (param $h i64) (param $n i32) (result i64)
    (call $sum_u32 (local.get $h) (local.get $n)))
  (func (export "call_max_u32") (param $h i64) (param $n i32) (result i64)
    (call $max_u32 (local.get $h) (local.get $n)))
  (func (export "call_count_byte") (param $h i64) (param $n i32) (param $b i32) (result i32)
    (call $count_byte (local.get $h) (local.get $n) (local.get $b)))
  (func (export "call_count_in_range") (param $h i64) (param $n i32) (param $lo i32) (param $hi i32) (result i32)
    (call $count_in_range (local.get $h) (local.get $n) (local.get $lo) (local.get $hi)))
  (func (export "call_popcount") (param $h i64) (param $n i32) (result i64)
    (call $popcount (local.get $h) (local.get $n)))
  (func (export "call_min_max") (param $h i64) (param $n i32) (result i32)
    (call $min_max (local.get $h) (local.get $n)))
  (func (export "call_find_byte") (param $h i64) (param $n i32) (param $b i32) (result i32)
    (call $find_byte (local.get $h) (local.get $n) (local.get $b)))
  (func (export "call_eq") (param $a i64) (param $b i64) (param $n i32) (result i32)
    (call $eq (local.get $a) (local.get $b) (local.get $n)))
  (func (export "call_lex_cmp") (param $a i64) (param $b i64) (param $n i32) (result i32)
    (call $lex_cmp (local.get $a) (local.get $b) (local.get $n)))
  (func (export "call_hash") (param $h i64) (param $n i32) (result i64)
    (call $hash (local.get $h) (local.get $n)))
  (func (export "call_and_fold") (param $h i64) (param $n i32) (result i32)
    (call $and_fold (local.get $h) (local.get $n)))
  (func (export "call_or_fold") (param $h i64) (param $n i32) (result i32)
    (call $or_fold (local.get $h) (local.get $n)))
  (func (export "call_xor_fold") (param $h i64) (param $n i32) (result i32)
    (call $xor_fold (local.get $h) (local.get $n)))
  (func (export "call_fill") (param $h i64) (param $n i32) (param $b i32) (result i32)
    (call $fill (local.get $h) (local.get $n) (local.get $b)))
  (func (export "call_xor_with_byte") (param $h i64) (param $n i32) (param $b i32) (result i32)
    (call $xor_with_byte (local.get $h) (local.get $n) (local.get $b)))
  (func (export "call_xor_into_region") (param $a i64) (param $b i64) (param $n i32) (result i32)
    (call $xor_into_region (local.get $a) (local.get $b) (local.get $n)))
)
"#;

fn build_reducer_inst(
    store: &mut Store<TvmHost>,
) -> Result<wasmtime::Instance> {
    let module = Module::new(store.engine(), REDUCER_WAT)?;
    let mut linker: Linker<TvmHost> = Linker::new(store.engine());
    add_raw_imports(&mut linker)?;
    Ok(linker.instantiate(store, &module)?)
}

#[test]
fn reducer_roundtrip_all_ops() -> Result<()> {
    let host = TvmHost::new();
    let engine = Engine::new(&Config::new())?;
    let mut store = Store::new(&engine, host);

    // Seed two regions: r1 with a known pattern, r2 with the same.
    let r1 = store.data_mut().create_region(RegionKind::HotHeap, 1024)?;
    let r2 = store.data_mut().create_region(RegionKind::HotHeap, 1024)?;
    let h1 = store.data_mut().alloc(r1, 256)?;
    let h2 = store.data_mut().alloc(r2, 256)?;
    let bytes: Vec<u8> = (0..256u32).map(|i| (i & 0xff) as u8).collect();
    store.data_mut().write_bytes(h1, &bytes)?;
    store.data_mut().write_bytes(h2, &bytes)?;
    let p1 = h1.pack() as i64;
    let p2 = h2.pack() as i64;

    let inst = build_reducer_inst(&mut store)?;

    // sum_u8: 0+1+...+255 = 32640
    let f = inst.get_typed_func::<(i64, i32), i64>(&mut store, "call_sum_u8")?;
    assert_eq!(f.call(&mut store, (p1, 256))?, 32640);

    // popcount: each byte 0..255 contributes its bit count; total = 1024
    let f = inst.get_typed_func::<(i64, i32), i64>(&mut store, "call_popcount")?;
    let want: u64 = bytes.iter().map(|&b| b.count_ones() as u64).sum();
    assert_eq!(f.call(&mut store, (p1, 256))? as u64, want);

    // count_byte
    let f = inst.get_typed_func::<(i64, i32, i32), i32>(&mut store, "call_count_byte")?;
    assert_eq!(f.call(&mut store, (p1, 256, 0x42))?, 1);

    // count_in_range
    let f = inst.get_typed_func::<(i64, i32, i32, i32), i32>(&mut store, "call_count_in_range")?;
    assert_eq!(f.call(&mut store, (p1, 256, 0x10, 0x1f))?, 16);

    // min_max — packed (lo<<8) | hi
    let f = inst.get_typed_func::<(i64, i32), i32>(&mut store, "call_min_max")?;
    let packed = f.call(&mut store, (p1, 256))?;
    let lo = (packed >> 8) & 0xff;
    let hi = packed & 0xff;
    assert_eq!((lo, hi), (0, 255));

    // find_byte: byte 0x80 lives at offset 0x80
    let f = inst.get_typed_func::<(i64, i32, i32), i32>(&mut store, "call_find_byte")?;
    assert_eq!(f.call(&mut store, (p1, 256, 0x80))?, 0x80);
    assert_eq!(f.call(&mut store, (p1, 256, 0x00))?, 0); // first
    // Not present: only 0..=255 in 256 bytes; pick a value in range but
    // limit search to first 16 bytes where it doesn't appear.
    assert_eq!(f.call(&mut store, (p1, 16, 0xff))?, -1);

    // and_fold / or_fold / xor_fold — at length 256 every byte 0..255
    // appears exactly once.
    //   AND of 0..255 = 0 (0x00 zeros every bit)
    //   OR  of 0..255 = 0xff (every bit eventually set)
    //   XOR of 0..255 = 0 (each bit flipped 128 times)
    let f = inst.get_typed_func::<(i64, i32), i32>(&mut store, "call_and_fold")?;
    assert_eq!(f.call(&mut store, (p1, 256))?, 0x00);
    let f = inst.get_typed_func::<(i64, i32), i32>(&mut store, "call_or_fold")?;
    assert_eq!(f.call(&mut store, (p1, 256))?, 0xff);
    let f = inst.get_typed_func::<(i64, i32), i32>(&mut store, "call_xor_fold")?;
    assert_eq!(f.call(&mut store, (p1, 256))?, 0x00);

    // eq / lex_cmp: identical regions
    let f = inst.get_typed_func::<(i64, i64, i32), i32>(&mut store, "call_eq")?;
    assert_eq!(f.call(&mut store, (p1, p2, 256))?, 1);
    let f = inst.get_typed_func::<(i64, i64, i32), i32>(&mut store, "call_lex_cmp")?;
    assert_eq!(f.call(&mut store, (p1, p2, 256))?, 0);

    // hash_fnv1a: nonzero, deterministic
    let f = inst.get_typed_func::<(i64, i32), i64>(&mut store, "call_hash")?;
    let h1_hash = f.call(&mut store, (p1, 256))?;
    let h2_hash = f.call(&mut store, (p2, 256))?;
    assert_eq!(h1_hash, h2_hash);
    assert_ne!(h1_hash, 0);

    // sum_u32_le: sum of u32s decoded from the bytes
    let f = inst.get_typed_func::<(i64, i32), i64>(&mut store, "call_sum_u32")?;
    let want_sum_u32: u128 = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u128)
        .sum();
    assert_eq!(f.call(&mut store, (p1, 256))? as u128, want_sum_u32);

    // max_u32_le
    let f = inst.get_typed_func::<(i64, i32), i64>(&mut store, "call_max_u32")?;
    let want_max_u32 = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .max()
        .unwrap();
    assert_eq!(f.call(&mut store, (p1, 256))? as u32, want_max_u32);

    Ok(())
}

#[test]
fn reducer_mutators_modify_in_place() -> Result<()> {
    let host = TvmHost::new();
    let engine = Engine::new(&Config::new())?;
    let mut store = Store::new(&engine, host);

    let r = store.data_mut().create_region(RegionKind::HotHeap, 1024)?;
    let h = store.data_mut().alloc(r, 128)?;
    store.data_mut().write_bytes(h, &[0u8; 128])?;
    let packed = h.pack() as i64;

    let inst = build_reducer_inst(&mut store)?;

    // fill: set all 128 bytes to 0x5a
    let f = inst.get_typed_func::<(i64, i32, i32), i32>(&mut store, "call_fill")?;
    assert_eq!(f.call(&mut store, (packed, 128, 0x5a))?, 0);
    let mut buf = [0u8; 128];
    store.data_mut().read_bytes(h, &mut buf)?;
    assert!(buf.iter().all(|&b| b == 0x5a));

    // xor_with_byte: flip each byte with 0xa5; 0x5a ^ 0xa5 = 0xff
    let f = inst.get_typed_func::<(i64, i32, i32), i32>(&mut store, "call_xor_with_byte")?;
    assert_eq!(f.call(&mut store, (packed, 128, 0xa5))?, 0);
    store.data_mut().read_bytes(h, &mut buf)?;
    assert!(buf.iter().all(|&b| b == 0xff));

    // xor_into_region: src ^= dst-style; verify the contract via a
    // separate region that holds 0x0f.
    let r2 = store.data_mut().create_region(RegionKind::HotHeap, 256)?;
    let h2 = store.data_mut().alloc(r2, 128)?;
    store.data_mut().write_bytes(h2, &[0x0fu8; 128])?;
    let packed2 = h2.pack() as i64;
    let f = inst.get_typed_func::<(i64, i64, i32), i32>(&mut store, "call_xor_into_region")?;
    // XOR src (0x0f) into dst (0xff) → 0xf0
    assert_eq!(f.call(&mut store, (packed2, packed, 128))?, 0);
    store.data_mut().read_bytes(h, &mut buf)?;
    assert!(buf.iter().all(|&b| b == 0xf0), "xor_into_region must apply src bytes");

    Ok(())
}
