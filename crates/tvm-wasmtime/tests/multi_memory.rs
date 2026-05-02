use tvm_wasmtime::{RuntimeMemoryRegion, WasmtimeMemoryRegion, WASM_PAGE_SIZE};
use wasmtime::{AsContextMut, Engine, Store};

fn store() -> Store<()> {
    let engine = Engine::default();
    Store::new(&engine, ())
}

#[test]
fn create_multiple_memories_in_one_store() -> anyhow::Result<()> {
    let mut store = store();

    let r0 = WasmtimeMemoryRegion::new(store.as_context_mut(), 1, Some(2))?;
    let r1 = WasmtimeMemoryRegion::new(store.as_context_mut(), 1, Some(4))?;
    let r2 = WasmtimeMemoryRegion::new(store.as_context_mut(), 2, Some(2))?;

    assert_eq!(r0.len(&store), WASM_PAGE_SIZE);
    assert_eq!(r1.len(&store), WASM_PAGE_SIZE);
    assert_eq!(r2.len(&store), WASM_PAGE_SIZE * 2);

    Ok(())
}

#[test]
fn read_write_isolated_per_memory() -> anyhow::Result<()> {
    let mut store = store();

    let a = WasmtimeMemoryRegion::new(store.as_context_mut(), 1, Some(1))?;
    let b = WasmtimeMemoryRegion::new(store.as_context_mut(), 1, Some(1))?;

    a.write(&mut store, 0, b"hello")?;
    b.write(&mut store, 0, b"world")?;

    let mut out_a = [0u8; 5];
    let mut out_b = [0u8; 5];
    a.read(&store, 0, &mut out_a)?;
    b.read(&store, 0, &mut out_b)?;

    assert_eq!(&out_a, b"hello");
    assert_eq!(&out_b, b"world");
    Ok(())
}

#[test]
fn out_of_bounds_read_returns_error() -> anyhow::Result<()> {
    let mut store = store();
    let m = WasmtimeMemoryRegion::new(store.as_context_mut(), 1, Some(1))?;
    let mut buf = [0u8; 8];
    let beyond = m.len(&store);
    let result = m.read(&store, beyond, &mut buf);
    assert!(result.is_err());
    Ok(())
}
