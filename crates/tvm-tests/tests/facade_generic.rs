//! Proves the facade is genuinely deployment-agnostic: the same generic
//! function compiles and runs against both `TvmHost` (host-side) and
//! `GuestTvm` (guest-side, via stubbed dispatch in unit tests).

use tvm_core::prelude::*;

/// A library function written against the generic facade. Allocates a
/// region, writes a payload, reads it back, returns the SUM of bytes.
fn library_workload<T: TvmFacade>(
    tvm: &mut T,
    payload: &[u8],
) -> Result<u32> {
    let region = tvm.create_region(RegionKind::HotHeap, payload.len() as u32 * 2)?;
    let handle = tvm.alloc(region, payload.len() as u32)?;
    tvm.write(handle, payload)?;
    let mut buf = vec![0u8; payload.len()];
    tvm.read(handle, &mut buf)?;
    Ok(buf.iter().map(|b| *b as u32).sum())
}

#[test]
fn library_workload_against_tvm_host() {
    use tvm_wasmtime::TvmHost;
    let mut host = TvmHost::new();
    let payload: Vec<u8> = (0..=63u8).collect();
    let sum = library_workload(&mut host, &payload).unwrap();
    let expected: u32 = (0u32..=63).sum();
    assert_eq!(sum, expected);
}

#[test]
fn library_workload_against_guest_tvm() {
    use std::sync::Mutex;
    use tvm_guest_mm::{Dispatch, GuestTvm, Pool};

    // Stub backing for the guest-side dispatch (host-side test).
    static STUB: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
    fn stub_read(pool: u32, off: u32, dst: &mut [u8]) -> Result<()> {
        let pools = STUB.lock().unwrap();
        let p = &pools[pool as usize];
        dst.copy_from_slice(&p[off as usize..off as usize + dst.len()]);
        Ok(())
    }
    fn stub_write(pool: u32, off: u32, src: &[u8]) -> Result<()> {
        let mut pools = STUB.lock().unwrap();
        let p = &mut pools[pool as usize];
        p[off as usize..off as usize + src.len()].copy_from_slice(src);
        Ok(())
    }

    *STUB.lock().unwrap() = (0..4).map(|_| vec![0u8; 4096]).collect();

    let pools: Vec<Pool> = (0..4)
        .map(|i| Pool {
            memory_index: i,
            used: 0,
            capacity: 4096,
        })
        .collect();
    let mut guest = GuestTvm::new(
        pools,
        Dispatch {
            read_bytes: stub_read,
            write_bytes: stub_write,
        },
    );

    let payload: Vec<u8> = (0..=63u8).collect();
    let sum = library_workload(&mut guest, &payload).unwrap();
    let expected: u32 = (0u32..=63).sum();
    assert_eq!(sum, expected);
}

/// Same function, this time exercising pin/unpin and region_info.
fn lifecycle_workload<T: TvmFacade>(tvm: &mut T) -> Result<()> {
    // Pinnable kind (HotHeap policy: pinnable=true).
    let region = tvm.create_region(RegionKind::HotHeap, 256)?;
    let _info_before = tvm.region_info(region)?;
    tvm.pin(region)?;
    let info_after = tvm.region_info(region)?;
    assert!(info_after.pinned);
    tvm.unpin(region)?;
    Ok(())
}

#[test]
fn lifecycle_workload_against_both() {
    use tvm_wasmtime::TvmHost;
    let mut host = TvmHost::new();
    lifecycle_workload(&mut host).unwrap();

    // Guest-side variant — same code path.
    use tvm_guest_mm::{Dispatch, GuestTvm, Pool};
    fn noop_read(_p: u32, _o: u32, _dst: &mut [u8]) -> Result<()> { Ok(()) }
    fn noop_write(_p: u32, _o: u32, _src: &[u8]) -> Result<()> { Ok(()) }
    let pools: Vec<Pool> = (0..2)
        .map(|i| Pool { memory_index: i, used: 0, capacity: 4096 })
        .collect();
    let mut guest = GuestTvm::new(
        pools,
        Dispatch { read_bytes: noop_read, write_bytes: noop_write },
    );
    lifecycle_workload(&mut guest).unwrap();
}
