use tvm_core::{Result, TvmError};
use wasmtime::{AsContext, AsContextMut, Memory, MemoryType, StoreContextMut};

pub const WASM_PAGE_SIZE: u32 = 65_536;

/// Memory region backed by a runtime that requires a context for access
/// (e.g. a `wasmtime::Store`). Mirrors `tvm_core::MemoryRegion` but threads a
/// `Cx` through every operation.
///
/// Implementations should treat snapshot/restore as authoritative full-state
/// serialization: the bytes returned by `snapshot` must round-trip through
/// `restore` to a region with identical observable behavior.
pub trait RuntimeMemoryRegion<Cx> {
    fn len(&self, cx: &Cx) -> u32;
    fn read(&self, cx: &Cx, offset: u32, buf: &mut [u8]) -> Result<()>;
    fn write(&self, cx: &mut Cx, offset: u32, data: &[u8]) -> Result<()>;

    /// Read the full backing memory into a host-side `Vec<u8>`.
    fn snapshot(&self, cx: &Cx) -> Result<Vec<u8>>;

    /// Construct a fresh region from a previously-captured snapshot. The
    /// implementation is responsible for sizing the underlying memory to fit
    /// `bytes.len()` (rounding up to the runtime's page size if needed).
    fn restore(cx: &mut Cx, bytes: Vec<u8>) -> Result<Self>
    where
        Self: Sized;
}

pub struct WasmtimeMemoryRegion {
    memory: Memory,
}

impl WasmtimeMemoryRegion {
    pub fn new<T>(
        mut store: StoreContextMut<'_, T>,
        min_pages: u32,
        max_pages: Option<u32>,
    ) -> anyhow::Result<Self> {
        let ty = MemoryType::new(min_pages, max_pages);
        let memory = Memory::new(&mut store, ty)?;
        Ok(Self { memory })
    }

    pub fn from_memory(memory: Memory) -> Self {
        Self { memory }
    }

    pub fn raw(&self) -> Memory {
        self.memory
    }
}

impl<T> RuntimeMemoryRegion<wasmtime::Store<T>> for WasmtimeMemoryRegion {
    fn len(&self, cx: &wasmtime::Store<T>) -> u32 {
        let bytes = self.memory.data_size(cx.as_context());
        u32::try_from(bytes).unwrap_or(u32::MAX)
    }

    fn read(&self, cx: &wasmtime::Store<T>, offset: u32, buf: &mut [u8]) -> Result<()> {
        self.memory
            .read(cx.as_context(), offset as usize, buf)
            .map_err(|_| TvmError::OutOfBounds)
    }

    fn write(&self, cx: &mut wasmtime::Store<T>, offset: u32, data: &[u8]) -> Result<()> {
        self.memory
            .write(cx.as_context_mut(), offset as usize, data)
            .map_err(|_| TvmError::OutOfBounds)
    }

    fn snapshot(&self, cx: &wasmtime::Store<T>) -> Result<Vec<u8>> {
        let len = self.memory.data_size(cx.as_context());
        let mut buf = vec![0u8; len];
        self.memory
            .read(cx.as_context(), 0, &mut buf)
            .map_err(|_| TvmError::OutOfBounds)?;
        Ok(buf)
    }

    fn restore(cx: &mut wasmtime::Store<T>, bytes: Vec<u8>) -> Result<Self> {
        let pages = bytes.len().div_ceil(WASM_PAGE_SIZE as usize).max(1) as u32;
        let ty = MemoryType::new(pages, None);
        let memory = Memory::new(cx.as_context_mut(), ty)
            .map_err(|e| TvmError::BackingStore(e.to_string()))?;
        memory
            .write(cx.as_context_mut(), 0, &bytes)
            .map_err(|_| TvmError::OutOfBounds)?;
        Ok(Self { memory })
    }
}

// ── D2 Session 6 — WasmosMemoryRegion (wasmos SharedMemory-backed) ──
//
// Wasmos-native peer of [`WasmtimeMemoryRegion`]. Backed by
// [`wasmos_runtime_api::SharedMemory`] instead of `wasmtime::Memory`.
// Both share the [`RuntimeMemoryRegion`] trait but with different
// `Cx` types:
//
// - `impl RuntimeMemoryRegion<wasmtime::Store<T>>` for the
//   wasmtime-native path (needs the Store to reach the memory).
// - `impl RuntimeMemoryRegion<()>` for the wasmos-native path
//   (SharedMemory carries its own state; no ctx needed).
//
// The unit-typed Cx keeps the trait shape symmetric while
// acknowledging that wasmos SharedMemory doesn't need any per-call
// context. Consumers pick the impl matching their runtime.
//
// # When to use
//
// - Girder's `SharedRegion` / imported-tvm patterns — the host
//   creates one memory that multiple guest instances import.
// - Any tvm-wasm consumer moving off `WasmtimeMemoryRegion` to
//   run on non-wasmtime adapters (WAMR, edge).

/// Wasmos-native memory region backed by a [`SharedMemory`] handle.
#[derive(Debug)]
pub struct WasmosMemoryRegion {
    memory: wasmos_runtime_api::SharedMemory,
}

impl WasmosMemoryRegion {
    /// **Adapter-facing.** Wrap a wasmos [`SharedMemory`] handle.
    /// Consumers typically obtain the handle from
    /// [`wasmos_runtime_api::Runtime::create_shared_memory`].
    pub fn from_shared_memory(memory: wasmos_runtime_api::SharedMemory) -> Self {
        Self { memory }
    }

    /// Return a fresh clone of the underlying shared-memory handle.
    /// Cheap; shares the same allocation.
    pub fn shared_memory(&self) -> wasmos_runtime_api::SharedMemory {
        self.memory.clone_handle()
    }
}

impl RuntimeMemoryRegion<()> for WasmosMemoryRegion {
    fn len(&self, _cx: &()) -> u32 {
        u32::try_from(self.memory.data_size_bytes()).unwrap_or(u32::MAX)
    }

    fn read(&self, _cx: &(), offset: u32, buf: &mut [u8]) -> Result<()> {
        // Wasmos SharedMemory::read returns Vec<u8>; copy into the
        // caller's buffer. The single Vec allocation is unavoidable
        // with the current SharedMemoryImpl surface — a
        // `read_into(&mut [u8])` primitive would remove it if a hot
        // workload demands (Phase 6.5.c candidate).
        let bytes = self
            .memory
            .read(offset as usize, buf.len())
            .map_err(|_| TvmError::OutOfBounds)?;
        if bytes.len() != buf.len() {
            return Err(TvmError::OutOfBounds);
        }
        buf.copy_from_slice(&bytes);
        Ok(())
    }

    fn write(&self, _cx: &mut (), offset: u32, data: &[u8]) -> Result<()> {
        self.memory
            .write(offset as usize, data)
            .map_err(|_| TvmError::OutOfBounds)
    }

    fn snapshot(&self, _cx: &()) -> Result<Vec<u8>> {
        let len = self.memory.data_size_bytes();
        self.memory
            .read(0, len)
            .map_err(|_| TvmError::OutOfBounds)
    }

    /// Wasmos SharedMemory has no direct "create-a-new-memory"
    /// primitive tied to this region's descriptor — the memory
    /// allocation is a runtime-level concern
    /// ([`wasmos_runtime_api::Runtime::create_shared_memory`]).
    /// Restore is therefore unsupported on this impl; consumers
    /// that need snapshot/restore should either stay on
    /// [`WasmtimeMemoryRegion`] or create a fresh shared memory
    /// via the runtime and use [`Self::from_shared_memory`] +
    /// [`Self::write`] to load the bytes.
    fn restore(_cx: &mut (), _bytes: Vec<u8>) -> Result<Self> {
        Err(TvmError::BackingStore(
            "WasmosMemoryRegion::restore is unsupported — allocate a fresh SharedMemory \
             via Runtime::create_shared_memory and use from_shared_memory + write instead"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use wasmos_runtime_api::{SharedMemory, SharedMemoryImpl};

    /// Vec-backed shared memory for the tests — mirrors the
    /// api-crate test double.
    struct FakeShared {
        buf: Mutex<Vec<u8>>,
    }

    impl SharedMemoryImpl for FakeShared {
        fn size_pages(&self) -> u64 {
            (self.buf.lock().unwrap().len() / WASM_PAGE_SIZE as usize) as u64
        }
        fn data_size_bytes(&self) -> usize {
            self.buf.lock().unwrap().len()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn read(&self, offset: usize, len: usize) -> wasmos_runtime_api::RuntimeResult<Vec<u8>> {
            let b = self.buf.lock().unwrap();
            let end = offset.saturating_add(len);
            if end > b.len() {
                return Err(wasmos_runtime_api::RuntimeError::msg("oob"));
            }
            Ok(b[offset..end].to_vec())
        }
        fn write(&self, offset: usize, bytes: &[u8]) -> wasmos_runtime_api::RuntimeResult<()> {
            let mut b = self.buf.lock().unwrap();
            let end = offset.saturating_add(bytes.len());
            if end > b.len() {
                return Err(wasmos_runtime_api::RuntimeError::msg("oob"));
            }
            b[offset..end].copy_from_slice(bytes);
            Ok(())
        }
    }

    fn fake_shared(size: usize) -> SharedMemory {
        SharedMemory::from_impl(Arc::new(FakeShared {
            buf: Mutex::new(vec![0u8; size]),
        }))
    }

    #[test]
    fn wasmos_memory_region_len_reflects_shared_memory_size() {
        let region = WasmosMemoryRegion::from_shared_memory(fake_shared(4096));
        assert_eq!(region.len(&()), 4096);
    }

    #[test]
    fn wasmos_memory_region_read_write_round_trip() {
        let region = WasmosMemoryRegion::from_shared_memory(fake_shared(WASM_PAGE_SIZE as usize));
        region
            .write(&mut (), 128, &[10, 20, 30, 40])
            .expect("write");
        let mut buf = [0u8; 4];
        region.read(&(), 128, &mut buf).expect("read");
        assert_eq!(buf, [10, 20, 30, 40]);
    }

    #[test]
    fn wasmos_memory_region_snapshot_captures_full_backing() {
        let region = WasmosMemoryRegion::from_shared_memory(fake_shared(8));
        region.write(&mut (), 0, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let snap = region.snapshot(&()).expect("snapshot");
        assert_eq!(snap, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn wasmos_memory_region_read_out_of_bounds_errors() {
        let region = WasmosMemoryRegion::from_shared_memory(fake_shared(16));
        let mut buf = [0u8; 32];
        let err = region.read(&(), 0, &mut buf).unwrap_err();
        assert!(matches!(err, TvmError::OutOfBounds));
    }

    #[test]
    fn wasmos_memory_region_restore_is_unsupported() {
        let err =
            <WasmosMemoryRegion as RuntimeMemoryRegion<()>>::restore(&mut (), vec![0u8; 16])
                .unwrap_err();
        assert!(matches!(err, TvmError::BackingStore(_)));
    }

    #[test]
    fn shared_memory_handle_survives_wrapping_and_unwrapping() {
        let orig = fake_shared(2048);
        let region = WasmosMemoryRegion::from_shared_memory(orig);
        let unwrapped = region.shared_memory();
        assert_eq!(unwrapped.data_size_bytes(), 2048);
    }
}
