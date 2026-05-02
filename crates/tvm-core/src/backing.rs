use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::directory::MemoryRegion;
use crate::error::{Result, TvmError};

pub struct VecBackedRegion {
    data: Vec<u8>,
}

impl VecBackedRegion {
    pub fn new(capacity: u32) -> Self {
        Self { data: vec![0u8; capacity as usize] }
    }

    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

impl MemoryRegion for VecBackedRegion {
    fn len(&self) -> u32 {
        self.data.len() as u32
    }

    fn read(&self, offset: u32, buf: &mut [u8]) -> Result<()> {
        let start = offset as usize;
        let end = start.checked_add(buf.len()).ok_or(TvmError::OutOfBounds)?;
        if end > self.data.len() {
            return Err(TvmError::OutOfBounds);
        }
        buf.copy_from_slice(&self.data[start..end]);
        Ok(())
    }

    fn write(&mut self, offset: u32, buf: &[u8]) -> Result<()> {
        let start = offset as usize;
        let end = start.checked_add(buf.len()).ok_or(TvmError::OutOfBounds)?;
        if end > self.data.len() {
            return Err(TvmError::OutOfBounds);
        }
        self.data[start..end].copy_from_slice(buf);
        Ok(())
    }

    fn snapshot(&self) -> Vec<u8> {
        self.data.clone()
    }

    fn restore(bytes: Vec<u8>) -> Self {
        Self::from_bytes(bytes)
    }
}

pub trait BackingStore: Send {
    fn spill(&mut self, region_id: u16, generation: u16, bytes: &[u8]) -> Result<()>;
    fn load(&mut self, region_id: u16, generation: u16) -> Result<Vec<u8>>;
}

/// Box wrapper for trait-object use. Lets `TvmHost` accept any
/// `BackingStore` impl rather than being hard-coded to `FileBackingStore`.
pub type DynBackingStore = Box<dyn BackingStore + Send>;

// Forward `BackingStore` through Box so generic `spill_region<B:
// BackingStore>` callers can pass a `&mut DynBackingStore` directly.
impl BackingStore for DynBackingStore {
    fn spill(&mut self, region_id: u16, generation: u16, bytes: &[u8]) -> Result<()> {
        (**self).spill(region_id, generation, bytes)
    }
    fn load(&mut self, region_id: u16, generation: u16) -> Result<Vec<u8>> {
        (**self).load(region_id, generation)
    }
}

/// Single-file backing store: a complete drop-in for snapshot-by-path.
///
/// Unlike `FileBackingStore` (one file per region), this writes everything to
/// the same path on every call. Useful for serializing one region to a named
/// file and reading it back.
pub struct SingleFileBackingStore {
    path: PathBuf,
}

impl SingleFileBackingStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl BackingStore for SingleFileBackingStore {
    fn spill(&mut self, _region_id: u16, _generation: u16, bytes: &[u8]) -> Result<()> {
        write_all(&self.path, bytes)
    }

    fn load(&mut self, _region_id: u16, _generation: u16) -> Result<Vec<u8>> {
        read_all(&self.path)
    }
}

pub struct FileBackingStore {
    root: PathBuf,
}

impl FileBackingStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)
            .map_err(|e| TvmError::BackingStore(e.to_string()))?;
        Ok(Self { root })
    }

    fn path(&self, region_id: u16, generation: u16) -> PathBuf {
        self.root.join(format!("region-{region_id}-gen-{generation}.bin"))
    }
}

impl BackingStore for FileBackingStore {
    fn spill(&mut self, region_id: u16, generation: u16, bytes: &[u8]) -> Result<()> {
        let path = self.path(region_id, generation);
        write_all(&path, bytes)
    }

    fn load(&mut self, region_id: u16, generation: u16) -> Result<Vec<u8>> {
        let path = self.path(region_id, generation);
        read_all(&path)
    }
}

fn write_all(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| TvmError::BackingStore(e.to_string()))?;
    f.write_all(bytes).map_err(|e| TvmError::BackingStore(e.to_string()))?;
    Ok(())
}

fn read_all(path: &Path) -> Result<Vec<u8>> {
    let mut f = File::open(path).map_err(|e| TvmError::BackingStore(e.to_string()))?;
    let mut buf = Vec::new();
    f.seek(SeekFrom::Start(0)).map_err(|e| TvmError::BackingStore(e.to_string()))?;
    f.read_to_end(&mut buf).map_err(|e| TvmError::BackingStore(e.to_string()))?;
    Ok(buf)
}
