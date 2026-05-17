//! End-to-end: drives the compactor against a *real* wasm module.
//! Pool 1 holds three live data ranges with a hole between them;
//! we call `tvm_intra_pool_copy_p1` from outside to slide the live
//! ranges into a contiguous prefix and verify the bytes survive.

use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams};
use wasmtime::{Config, Engine, Linker, Module, Store};

const USER_BODY: &str = r#"
    ;; Test thunk: read a u8 from pool 1 at the given offset.
    (func (export "read_p1") (param $off i32) (result i32)
      (i32.load8_u 1 (local.get $off)))
"#;

#[test]
fn intra_pool_copy_slides_live_data() -> anyhow::Result<()> {
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let p = ModuleParams {
        n_pools: 4,
        initial_pages_per_pool: 1,
        max_pages_per_pool: 16,
        user_body: USER_BODY.to_string(),
    };
    let module = Module::new(&engine, &tvm_guest_mm_module_template(&p))?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    // Seed pool 1 with three live blocks separated by a hole:
    //   [0..64)    = 0xaa  (block A)
    //   [64..192)  = 0x00  (hole, was block B)
    //   [192..224) = 0xcc  (block C)
    //   [224..320) = 0xdd  (block D)
    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    let mut layout = vec![0u8; 320];
    layout[0..64].fill(0xaa);
    layout[192..224].fill(0xcc);
    layout[224..320].fill(0xdd);
    mem1.write(&mut store, 0, &layout)?;

    // Slide C from 192 → 64 and D from 224 → 96 by calling the
    // per-pool intra-copy helper from the host. After compaction the
    // first 192 bytes should be: [0..64)=0xaa, [64..96)=0xcc, [96..192)=0xdd.
    let copy =
        instance.get_typed_func::<(i32, i32, i32), ()>(&mut store, "tvm_intra_pool_copy_p1")?;
    copy.call(&mut store, (64, 192, 32))?; // C → 64
    copy.call(&mut store, (96, 224, 96))?; // D → 96

    // Verify the packed layout via a guest-side read.
    let read = instance.get_typed_func::<i32, i32>(&mut store, "read_p1")?;
    for off in 0..64 {
        assert_eq!(read.call(&mut store, off)?, 0xaa, "A byte at {off}");
    }
    for off in 64..96 {
        assert_eq!(read.call(&mut store, off)?, 0xcc, "C byte at {off}");
    }
    for off in 96..192 {
        assert_eq!(read.call(&mut store, off)?, 0xdd, "D byte at {off}");
    }
    Ok(())
}

#[test]
fn intra_pool_copy_handles_overlap() -> anyhow::Result<()> {
    // Wasm memory.copy semantics: source and destination may overlap;
    // the operation behaves as if bytes were copied via a scratch
    // buffer. Verify by sliding a 16-byte run by 8 bytes.
    let mut config = Config::new();
    config.wasm_multi_memory(true);
    let engine = Engine::new(&config)?;
    let p = ModuleParams {
        n_pools: 2,
        initial_pages_per_pool: 1,
        max_pages_per_pool: 16,
        user_body: USER_BODY.to_string(),
    };
    let module = Module::new(&engine, &tvm_guest_mm_module_template(&p))?;
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module)?;

    let mem1 = instance.get_memory(&mut store, "mem1").unwrap();
    let pattern: Vec<u8> = (0..16u8).collect();
    mem1.write(&mut store, 100, &pattern)?;

    // Slide [100..116) → [108..124). Overlaps in [108..116).
    let copy =
        instance.get_typed_func::<(i32, i32, i32), ()>(&mut store, "tvm_intra_pool_copy_p1")?;
    copy.call(&mut store, (108, 100, 16))?;

    // Verify destination has the original pattern; source is undefined
    // in the overlapping region but unchanged in the non-overlap.
    let read = instance.get_typed_func::<i32, i32>(&mut store, "read_p1")?;
    for i in 0..16i32 {
        assert_eq!(
            read.call(&mut store, 108 + i)?,
            i as i32,
            "destination byte at +{i}"
        );
    }
    Ok(())
}
