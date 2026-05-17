#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle {
    pub region_id: u16,
    pub generation: u16,
    pub offset: u32,
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_null() {
            return write!(f, "Handle::NULL");
        }
        write!(
            f,
            "Handle(r{}@gen{}+{:#x})",
            self.region_id, self.generation, self.offset
        )
    }
}

impl core::fmt::Display for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_null() {
            return write!(f, "<null>");
        }
        write!(f, "r{}@{}+{}", self.region_id, self.generation, self.offset)
    }
}

impl From<u64> for Handle {
    fn from(packed: u64) -> Self {
        Self::unpack(packed)
    }
}

impl From<i64> for Handle {
    fn from(packed: i64) -> Self {
        Self::unpack(packed as u64)
    }
}

impl From<Handle> for u64 {
    fn from(h: Handle) -> Self {
        h.pack()
    }
}

impl From<Handle> for i64 {
    fn from(h: Handle) -> Self {
        h.pack() as i64
    }
}

impl Handle {
    pub const NULL: Handle = Handle {
        region_id: 0,
        generation: 0,
        offset: 0,
    };

    pub const fn is_null(self) -> bool {
        self.region_id == 0 && self.generation == 0 && self.offset == 0
    }

    pub const fn pack(self) -> u64 {
        ((self.region_id as u64) << 48) | ((self.generation as u64) << 32) | (self.offset as u64)
    }

    pub const fn unpack(packed: u64) -> Self {
        Self {
            region_id: (packed >> 48) as u16,
            generation: ((packed >> 32) & 0xFFFF) as u16,
            offset: (packed & 0xFFFF_FFFF) as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let h = Handle {
            region_id: 7,
            generation: 42,
            offset: 0xDEAD_BEEF,
        };
        assert_eq!(Handle::unpack(h.pack()), h);
    }

    #[test]
    fn null_handle() {
        assert!(Handle::NULL.is_null());
    }
}
