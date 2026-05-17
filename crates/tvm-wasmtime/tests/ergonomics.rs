//! Smoke tests for the ergonomic API surface added in the user-DX round.
//! Each test demonstrates a common usage pattern; if these break, the
//! "quick start" examples in the docs are wrong.

use tvm_wasmtime::prelude::*;

#[test]
fn one_liner_alloc_in_new_region() {
    let mut host = TvmHost::new();
    let (region, handle) = host
        .alloc_in_new_region(RegionKind::HotHeap, 4096, 64)
        .expect("alloc_in_new_region");
    assert_eq!(region, 0);
    assert_eq!(handle.region_id, region);
    assert_eq!(handle.offset, 0);

    host.write_bytes(handle, b"hello-world").unwrap();
    let mut buf = [0u8; 11];
    host.read_bytes(handle, &mut buf).unwrap();
    assert_eq!(&buf, b"hello-world");
}

#[test]
fn handle_display_and_conversions() {
    let h = Handle {
        region_id: 7,
        generation: 5,
        offset: 1024,
    };
    assert_eq!(h.to_string(), "r7@5+1024");
    assert_eq!(format!("{:?}", h), "Handle(r7@gen5+0x400)");

    // Round-trip through i64.
    let packed: i64 = h.into();
    let back: Handle = packed.into();
    assert_eq!(h, back);

    // u64 too.
    let packed_u: u64 = h.into();
    assert_eq!(packed_u, h.pack());

    // NULL display.
    assert_eq!(Handle::NULL.to_string(), "<null>");
    assert_eq!(format!("{:?}", Handle::NULL), "Handle::NULL");
}

#[test]
fn builder_quickstart() -> wasmtime::Result<()> {
    let (_engine, _store, _linker) = TvmBuilder::new().with_raw_imports().build()?;
    Ok(())
}

#[test]
fn builder_with_backing_and_allocator() -> wasmtime::Result<()> {
    let tmp = tempfile::tempdir().unwrap();
    let (_engine, store, _linker) = TvmBuilder::new()
        .with_backing(tmp.path())
        .with_allocator(AllocatorKind::Freelist)
        .build()?;
    assert!(store.data().backing.is_some());
    assert_eq!(store.data().default_allocator, AllocatorKind::Freelist);
    Ok(())
}

#[test]
fn build_imported_setup_with_data_round_trip() -> wasmtime::Result<()> {
    let (_engine, store, _linker, handles) =
        build_imported_setup_with_data(&[b"abcd", b"efgh"], RegionKind::HotHeap, 64)?;
    assert_eq!(handles.len(), 2);

    // Verify pre-loaded payloads via direct memory inspection.
    let r0_mem = store
        .data()
        .imported_region(handles[0].region_id)
        .unwrap()
        .memory();
    let mut buf = [0u8; 4];
    r0_mem
        .read(&store, handles[0].offset as usize, &mut buf)
        .unwrap();
    assert_eq!(&buf, b"abcd");

    let r1_mem = store
        .data()
        .imported_region(handles[1].region_id)
        .unwrap()
        .memory();
    r1_mem
        .read(&store, handles[1].offset as usize, &mut buf)
        .unwrap();
    assert_eq!(&buf, b"efgh");
    Ok(())
}

#[test]
fn error_context_populated_on_oob() {
    let mut host = TvmHost::new();
    let (_r, h) = host
        .alloc_in_new_region(RegionKind::HotHeap, 64, 8)
        .unwrap();
    // Read 100 bytes into 8-byte allocation → out of bounds.
    let mut buf = [0u8; 100];
    let result = host.read_bytes(h, &mut buf);
    assert!(matches!(result, Err(TvmError::OutOfBounds)));

    let ctx = take_last_error_context().expect("error context populated");
    assert_eq!(ctx.region_id, Some(h.region_id));
    assert_eq!(ctx.generation, Some(h.generation));
    assert_eq!(ctx.len, Some(100));
    assert!(ctx.capacity.is_some());
    assert!(ctx.note.unwrap().contains("end > capacity"));
}

#[test]
fn error_context_cleared_on_take() {
    let mut host = TvmHost::new();
    let (_r, h) = host
        .alloc_in_new_region(RegionKind::HotHeap, 64, 8)
        .unwrap();
    let mut buf = [0u8; 100];
    let _ = host.read_bytes(h, &mut buf);
    let _first = take_last_error_context();
    let second = take_last_error_context();
    assert!(second.is_none(), "context should be cleared after take");
}
