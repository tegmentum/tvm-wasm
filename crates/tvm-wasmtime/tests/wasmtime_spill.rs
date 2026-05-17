//! End-to-end coverage: spill a wasmtime memory to disk and load it back.

use tempfile::tempdir;
use tvm_core::FileBackingStore;
use tvm_wasmtime::{
    load_runtime_region, memory_factory::RuntimeMemoryRegion, spill_runtime_region,
    WasmtimeMemoryRegion, WASM_PAGE_SIZE,
};
use wasmtime::{AsContextMut, Engine, Store};

#[test]
fn wasmtime_memory_round_trips_through_backing_store() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let mut backing = FileBackingStore::new(tmp.path())?;

    let engine = Engine::default();
    let mut store: Store<()> = Store::new(&engine, ());

    // Stage 1: live region with known bytes.
    let region = WasmtimeMemoryRegion::new(store.as_context_mut(), 1, Some(4))?;
    region.write(&mut store, 0, b"hello-wasmtime!!")?;

    spill_runtime_region(&region, &store, &mut backing, 7, 1)?;

    // Drop the live region to free the memory slot.
    drop(region);

    // Stage 2: load into a freshly minted region.
    let restored: WasmtimeMemoryRegion = load_runtime_region(&mut store, &mut backing, 7, 1)?;

    let mut buf = [0u8; 16];
    restored.read(&store, 0, &mut buf)?;
    assert_eq!(&buf, b"hello-wasmtime!!");

    // Restored region should be at least one page.
    assert!(restored.len(&store) >= WASM_PAGE_SIZE);
    Ok(())
}

#[test]
fn snapshot_returns_full_memory_size() -> anyhow::Result<()> {
    let engine = Engine::default();
    let mut store: Store<()> = Store::new(&engine, ());
    let region = WasmtimeMemoryRegion::new(store.as_context_mut(), 2, Some(2))?;
    region.write(&mut store, 0, b"prefix")?;

    let snap = region.snapshot(&store)?;
    assert_eq!(snap.len(), (WASM_PAGE_SIZE * 2) as usize);
    assert_eq!(&snap[..6], b"prefix");
    // The remainder must be zeroed (wasm page initialization guarantee).
    assert!(snap[6..].iter().all(|b| *b == 0));
    Ok(())
}

#[test]
fn restore_rounds_up_to_page_size() -> anyhow::Result<()> {
    let engine = Engine::default();
    let mut store: Store<()> = Store::new(&engine, ());
    // Bytes shorter than a page should still produce at least a 1-page memory.
    let region: WasmtimeMemoryRegion = WasmtimeMemoryRegion::restore(&mut store, vec![1, 2, 3])?;
    assert_eq!(region.len(&store), WASM_PAGE_SIZE);

    let mut head = [0u8; 3];
    region.read(&store, 0, &mut head)?;
    assert_eq!(&head, &[1, 2, 3]);
    Ok(())
}
