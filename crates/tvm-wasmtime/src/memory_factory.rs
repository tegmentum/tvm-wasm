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
        let pages = bytes
            .len()
            .div_ceil(WASM_PAGE_SIZE as usize)
            .max(1) as u32;
        let ty = MemoryType::new(pages, None);
        let memory = Memory::new(cx.as_context_mut(), ty)
            .map_err(|e| TvmError::BackingStore(e.to_string()))?;
        memory
            .write(cx.as_context_mut(), 0, &bytes)
            .map_err(|_| TvmError::OutOfBounds)?;
        Ok(Self { memory })
    }
}
