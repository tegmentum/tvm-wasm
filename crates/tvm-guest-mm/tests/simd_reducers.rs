//! Correctness checks for the symmetric SIMD reducer kernels emitted
//! per pool. Each test seeds pool 1 with a known byte pattern, calls
//! the SIMD kernel, and compares against a scalar reference computed
//! host-side.

use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams};
use wasmtime::{Config, Engine, Linker, Module, Store};

const USER_BODY: &str = r#"
    ;; Each thunk just calls the per-pool-1 SIMD kernel with the
    ;; supplied (off, len, …) and returns its scalar result.
    (func (export "xor_fold") (param $off i32) (param $len i32) (result i32)
      (call $tvm_simd_xor_fold_u8_p1 (local.get $off) (local.get $len)))
    (func (export "and_fold") (param $off i32) (param $len i32) (result i32)
      (call $tvm_simd_and_fold_u8_p1 (local.get $off) (local.get $len)))
    (func (export "or_fold") (param $off i32) (param $len i32) (result i32)
      (call $tvm_simd_or_fold_u8_p1 (local.get $off) (local.get $len)))
    (func (export "count_byte") (param $off i32) (param $len i32) (param $b i32) (result i32)
      (call $tvm_simd_count_byte_p1 (local.get $off) (local.get $len) (local.get $b)))
    (func (export "popcount") (param $off i32) (param $len i32) (result i64)
      (call $tvm_simd_popcount_p1 (local.get $off) (local.get $len)))
    (func (export "find_byte") (param $off i32) (param $len i32) (param $b i32) (result i32)
      (call $tvm_simd_find_byte_p1 (local.get $off) (local.get $len) (local.get $b)))
    (func (export "min_max") (param $off i32) (param $len i32) (result i32)
      (call $tvm_simd_min_max_u8_p1 (local.get $off) (local.get $len)))
"#;

fn build() -> anyhow::Result<(Engine, Module)> {
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: 1,
        max_pages_per_pool: 16,
        user_body: USER_BODY.to_string(),
    };
    let module = Module::new(&engine, tvm_guest_mm_module_template(&p))?;
    Ok((engine, module))
}

fn instantiate_with_data(data: &[u8]) -> anyhow::Result<(Store<()>, wasmtime::Instance)> {
    let (engine, module) = build()?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;
    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    mem1.write(&mut store, 0, data)?;
    Ok((store, instance))
}

fn make_data(len: u32, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| i.wrapping_mul(2654435761) as u8 ^ seed)
        .collect()
}

#[test]
fn simd_xor_fold_matches_scalar() -> anyhow::Result<()> {
    for &len in &[0u32, 1, 15, 16, 17, 33, 1024, 4099] {
        let data = make_data(len, 0xa5);
        let (mut store, inst) = instantiate_with_data(&data)?;
        let f = inst.get_typed_func::<(i32, i32), i32>(&mut store, "xor_fold")?;
        let got = f.call(&mut store, (0, len as i32))? as u8;
        let want: u8 = data.iter().fold(0u8, |a, &b| a ^ b);
        assert_eq!(got, want, "xor_fold len={}", len);
    }
    Ok(())
}

#[test]
fn simd_and_fold_matches_scalar() -> anyhow::Result<()> {
    for &len in &[0u32, 1, 15, 16, 17, 33, 1024, 4099] {
        let data = make_data(len, 0x33);
        let (mut store, inst) = instantiate_with_data(&data)?;
        let f = inst.get_typed_func::<(i32, i32), i32>(&mut store, "and_fold")?;
        let got = f.call(&mut store, (0, len as i32))? as u8;
        let want: u8 = data.iter().fold(0xffu8, |a, &b| a & b);
        assert_eq!(got, want, "and_fold len={}", len);
    }
    Ok(())
}

#[test]
fn simd_or_fold_matches_scalar() -> anyhow::Result<()> {
    for &len in &[0u32, 1, 15, 16, 17, 33, 1024, 4099] {
        let data = make_data(len, 0x12);
        let (mut store, inst) = instantiate_with_data(&data)?;
        let f = inst.get_typed_func::<(i32, i32), i32>(&mut store, "or_fold")?;
        let got = f.call(&mut store, (0, len as i32))? as u8;
        let want: u8 = data.iter().fold(0u8, |a, &b| a | b);
        assert_eq!(got, want, "or_fold len={}", len);
    }
    Ok(())
}

#[test]
fn simd_count_byte_matches_scalar() -> anyhow::Result<()> {
    for &len in &[0u32, 1, 15, 16, 17, 33, 1024, 4099] {
        let data = make_data(len, 0x00);
        let (mut store, inst) = instantiate_with_data(&data)?;
        let f = inst.get_typed_func::<(i32, i32, i32), i32>(&mut store, "count_byte")?;
        for &needle in &[0u8, 0x42, 0xff] {
            let got = f.call(&mut store, (0, len as i32, needle as i32))?;
            let want = data.iter().filter(|&&b| b == needle).count() as i32;
            assert_eq!(got, want, "count_byte len={} needle={:#x}", len, needle);
        }
    }
    Ok(())
}

#[test]
fn simd_popcount_matches_scalar() -> anyhow::Result<()> {
    for &len in &[0u32, 1, 15, 16, 17, 33, 1024, 4099] {
        let data = make_data(len, 0xc3);
        let (mut store, inst) = instantiate_with_data(&data)?;
        let f = inst.get_typed_func::<(i32, i32), i64>(&mut store, "popcount")?;
        let got = f.call(&mut store, (0, len as i32))? as u64;
        let want: u64 = data.iter().map(|&b| b.count_ones() as u64).sum();
        assert_eq!(got, want, "popcount len={}", len);
    }
    Ok(())
}

#[test]
fn simd_find_byte_matches_scalar() -> anyhow::Result<()> {
    let mut data = vec![0u8; 200];
    data[100] = 0xab;
    data[150] = 0xab;
    let (mut store, inst) = instantiate_with_data(&data)?;
    let f = inst.get_typed_func::<(i32, i32, i32), i32>(&mut store, "find_byte")?;
    assert_eq!(f.call(&mut store, (0, 200, 0xab))?, 100);
    assert_eq!(f.call(&mut store, (0, 200, 0xff))?, -1);
    assert_eq!(f.call(&mut store, (0, 50, 0xab))?, -1);
    // From offset 110, the first 0xab is at byte index 150 → relative offset 40.
    assert_eq!(f.call(&mut store, (110, 90, 0xab))?, 40);
    Ok(())
}

#[test]
fn simd_min_max_matches_scalar() -> anyhow::Result<()> {
    for &len in &[1u32, 15, 16, 17, 33, 1024, 4099] {
        let data = make_data(len, 0x77);
        let (mut store, inst) = instantiate_with_data(&data)?;
        let f = inst.get_typed_func::<(i32, i32), i32>(&mut store, "min_max")?;
        let packed = f.call(&mut store, (0, len as i32))?;
        let lo = ((packed >> 8) & 0xff) as u8;
        let hi = (packed & 0xff) as u8;
        let want_lo = *data.iter().min().unwrap();
        let want_hi = *data.iter().max().unwrap();
        assert_eq!((lo, hi), (want_lo, want_hi), "min_max len={}", len);
    }
    Ok(())
}
