use std::collections::BTreeMap;

use crate::error::{Result, TvmError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocatorKind {
    Bump,
    Freelist,
    /// Fixed-class-size slab allocator. All allocations must be exactly
    /// `class_size` bytes; `dealloc` recycles the slot for the next `alloc`.
    /// Region capacity should be a multiple of `class_size`; the trailing
    /// partial slot (if any) is wasted.
    Slab { class_size: u32 },
}

pub enum RegionAllocator {
    Bump(BumpAllocator),
    Freelist(FreelistAllocator),
    Slab(SlabAllocator),
}

impl RegionAllocator {
    pub fn new(kind: AllocatorKind, capacity: u32) -> Self {
        match kind {
            AllocatorKind::Bump => Self::Bump(BumpAllocator::new(capacity)),
            AllocatorKind::Freelist => Self::Freelist(FreelistAllocator::new(capacity)),
            AllocatorKind::Slab { class_size } => {
                Self::Slab(SlabAllocator::new(capacity, class_size))
            }
        }
    }

    pub fn alloc(&mut self, size: u32, align: u32) -> Result<u32> {
        match self {
            Self::Bump(a) => a.alloc(size, align),
            Self::Freelist(a) => a.alloc(size, align),
            Self::Slab(a) => a.alloc(size),
        }
    }

    pub fn dealloc(&mut self, offset: u32) -> Result<u32> {
        match self {
            Self::Bump(_) => Ok(0),
            Self::Freelist(a) => a.dealloc(offset),
            Self::Slab(a) => a.dealloc(offset),
        }
    }

    pub fn used(&self) -> u32 {
        match self {
            Self::Bump(a) => a.used(),
            Self::Freelist(a) => a.used(),
            Self::Slab(a) => a.used(),
        }
    }

    /// Sorted (offset, size) of every live allocation. Returns `None` for
    /// allocators that don't track allocations (e.g. bump).
    pub fn allocated_blocks(&self) -> Option<Vec<(u32, u32)>> {
        match self {
            Self::Bump(_) => None,
            Self::Freelist(a) => Some(
                a.allocated.iter().map(|(off, size)| (*off, *size)).collect(),
            ),
            // Slab allocations are uniform; compaction would still pack live
            // slots toward 0, but doing so is rarely useful (no fragmentation
            // by construction). Skip for now.
            Self::Slab(_) => None,
        }
    }

    /// Reset internal state to the supplied packed layout. `new_blocks` must be
    /// sorted by offset and contiguous starting at 0. Only meaningful after a
    /// successful compaction; calling on an unsupported allocator is a no-op.
    pub fn rebuild_after_compact(&mut self, new_blocks: &[(u32, u32)], capacity: u32) {
        if let Self::Freelist(a) = self {
            a.rebuild_after_compact(new_blocks, capacity);
        }
    }

    pub fn supports_compaction(&self) -> bool {
        matches!(self, Self::Freelist(_))
    }
}

pub struct BumpAllocator {
    capacity: u32,
    used: u32,
}

impl BumpAllocator {
    pub const fn new(capacity: u32) -> Self {
        Self { capacity, used: 0 }
    }

    pub fn alloc(&mut self, size: u32, align: u32) -> Result<u32> {
        let aligned = align_up(self.used, align).ok_or(TvmError::AllocationFailed)?;
        let end = aligned.checked_add(size).ok_or(TvmError::AllocationFailed)?;
        if end > self.capacity {
            return Err(TvmError::AllocationFailed);
        }
        self.used = end;
        Ok(aligned)
    }

    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}

pub struct FreelistAllocator {
    capacity: u32,
    /// (offset, size) pairs of free blocks, sorted by offset.
    free: Vec<(u32, u32)>,
    /// Live allocations: offset -> size, for dealloc lookup.
    allocated: BTreeMap<u32, u32>,
    used: u32,
}

impl FreelistAllocator {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            free: vec![(0, capacity)],
            allocated: BTreeMap::new(),
            used: 0,
        }
    }

    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn alloc(&mut self, size: u32, align: u32) -> Result<u32> {
        if size == 0 {
            return Err(TvmError::AllocationFailed);
        }
        for i in 0..self.free.len() {
            let (block_off, block_size) = self.free[i];
            let aligned = align_up(block_off, align).ok_or(TvmError::AllocationFailed)?;
            let pad = aligned - block_off;
            let need = pad.checked_add(size).ok_or(TvmError::AllocationFailed)?;
            if need <= block_size {
                let remainder = block_size - need;
                if pad == 0 && remainder == 0 {
                    self.free.remove(i);
                } else if pad == 0 {
                    self.free[i] = (aligned + size, remainder);
                } else if remainder == 0 {
                    self.free[i] = (block_off, pad);
                } else {
                    self.free[i] = (block_off, pad);
                    self.free.insert(i + 1, (aligned + size, remainder));
                }
                self.allocated.insert(aligned, size);
                self.used += size;
                return Ok(aligned);
            }
        }
        Err(TvmError::AllocationFailed)
    }

    pub fn dealloc(&mut self, offset: u32) -> Result<u32> {
        let size = self.allocated.remove(&offset).ok_or(TvmError::OutOfBounds)?;
        self.used -= size;
        // Insert sorted, then coalesce with neighbors.
        let pos = self.free.partition_point(|(o, _)| *o < offset);
        self.free.insert(pos, (offset, size));
        self.coalesce(pos);
        Ok(size)
    }

    fn coalesce(&mut self, idx: usize) {
        // Coalesce with right neighbor first so left-neighbor index stays stable.
        if idx + 1 < self.free.len() {
            let (a_off, a_size) = self.free[idx];
            let (b_off, b_size) = self.free[idx + 1];
            if a_off + a_size == b_off {
                self.free[idx] = (a_off, a_size + b_size);
                self.free.remove(idx + 1);
            }
        }
        if idx > 0 {
            let (a_off, a_size) = self.free[idx - 1];
            let (b_off, b_size) = self.free[idx];
            if a_off + a_size == b_off {
                self.free[idx - 1] = (a_off, a_size + b_size);
                self.free.remove(idx);
            }
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn rebuild_after_compact(&mut self, new_blocks: &[(u32, u32)], capacity: u32) {
        self.capacity = capacity;
        self.allocated.clear();
        let mut used = 0u32;
        for &(off, size) in new_blocks {
            self.allocated.insert(off, size);
            used += size;
        }
        self.used = used;
        self.free.clear();
        if used < capacity {
            self.free.push((used, capacity - used));
        }
    }
}

/// Fixed-class-size pool. Every allocation is exactly `class_size` bytes;
/// `dealloc` returns the slot to the free list for reuse. Zero-fragmentation
/// by construction — the only failure mode is "no free slots."
pub struct SlabAllocator {
    capacity: u32,
    class_size: u32,
    free_slots: Vec<u32>,
    n_slots: u32,
    used: u32,
}

impl SlabAllocator {
    pub fn new(capacity: u32, class_size: u32) -> Self {
        let n_slots = if class_size == 0 { 0 } else { capacity / class_size };
        // Push slots in reverse so `pop()` hands out offset 0 first.
        let free_slots: Vec<u32> = (0..n_slots).rev().map(|i| i * class_size).collect();
        Self { capacity, class_size, free_slots, n_slots, used: 0 }
    }

    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn class_size(&self) -> u32 {
        self.class_size
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn alloc(&mut self, size: u32) -> Result<u32> {
        if self.class_size == 0 || size != self.class_size {
            return Err(TvmError::AllocationFailed);
        }
        match self.free_slots.pop() {
            Some(off) => {
                self.used += self.class_size;
                Ok(off)
            }
            None => Err(TvmError::AllocationFailed),
        }
    }

    pub fn dealloc(&mut self, offset: u32) -> Result<u32> {
        if self.class_size == 0
            || offset >= self.n_slots * self.class_size
            || offset % self.class_size != 0
        {
            return Err(TvmError::OutOfBounds);
        }
        if self.free_slots.contains(&offset) {
            return Err(TvmError::OutOfBounds);
        }
        self.free_slots.push(offset);
        self.used -= self.class_size;
        Ok(self.class_size)
    }
}

fn align_up(value: u32, align: u32) -> Option<u32> {
    if align == 0 || (align & (align - 1)) != 0 {
        return None;
    }
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_alloc_basic() {
        let mut a = BumpAllocator::new(64);
        assert_eq!(a.alloc(8, 1).unwrap(), 0);
        assert_eq!(a.alloc(8, 1).unwrap(), 8);
        assert!(a.alloc(64, 1).is_err());
    }

    #[test]
    fn bump_alloc_alignment() {
        let mut a = BumpAllocator::new(64);
        assert_eq!(a.alloc(1, 1).unwrap(), 0);
        assert_eq!(a.alloc(4, 8).unwrap(), 8);
    }

    #[test]
    fn freelist_alloc_dealloc_reuse() {
        let mut a = FreelistAllocator::new(64);
        let x = a.alloc(16, 1).unwrap();
        let y = a.alloc(16, 1).unwrap();
        assert_eq!(x, 0);
        assert_eq!(y, 16);
        a.dealloc(x).unwrap();
        // Reuse the freed block (first-fit).
        let z = a.alloc(8, 1).unwrap();
        assert_eq!(z, 0);
    }

    #[test]
    fn freelist_coalesces_on_dealloc() {
        let mut a = FreelistAllocator::new(64);
        let x = a.alloc(16, 1).unwrap();
        let y = a.alloc(16, 1).unwrap();
        let z = a.alloc(16, 1).unwrap();
        a.dealloc(x).unwrap();
        a.dealloc(z).unwrap();
        a.dealloc(y).unwrap(); // middle frees → should coalesce all three
        // After full coalesce, single 64-byte alloc must succeed.
        let big = a.alloc(64, 1).unwrap();
        assert_eq!(big, 0);
    }

    #[test]
    fn freelist_dealloc_unknown_offset_errors() {
        let mut a = FreelistAllocator::new(32);
        assert!(a.dealloc(0).is_err());
    }
}
