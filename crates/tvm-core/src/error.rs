use alloc::string::String;

#[derive(Debug, thiserror::Error)]
pub enum TvmError {
    #[error("region not found: id={0}")]
    RegionNotFound(u16),
    #[error("stale handle: generation mismatch")]
    StaleHandle,
    #[error("out of bounds: access exceeds region capacity")]
    OutOfBounds,
    #[error("region is not resident (Cold tier; load via load_region or auto-fault)")]
    NotResident,
    #[error("allocation failed: region full or unsupported size")]
    AllocationFailed,
    #[error("backing store error: {0}")]
    BackingStore(String),
    #[error("region is pinned: cannot spill or demote")]
    Pinned,
    #[error("operation not supported by this allocator (e.g. compaction on Bump)")]
    UnsupportedAllocator,
    #[error("policy violation: operation forbidden by region's PlacementPolicy")]
    PolicyViolation,
}

pub type Result<T> = core::result::Result<T, TvmError>;

// ErrorContext and the thread_local!-backed context-passing surface
// require libstd (`thread_local!` is std-only). Gated behind the
// `std` feature so the no_std subset of the crate doesn't pull them
// in. Guest consumers can still construct `TvmError` and inspect
// it; they just can't use the per-thread last-error context.
#[cfg(feature = "std")]
mod context {
    use super::ErrorContext;

    std::thread_local! {
        static LAST_ERROR_CONTEXT: std::cell::RefCell<Option<ErrorContext>> =
            std::cell::RefCell::new(None);
    }

    /// Set the per-thread last-error context. Called internally by
    /// error sites; users typically don't call this directly.
    pub fn set_last_error_context(ctx: ErrorContext) {
        LAST_ERROR_CONTEXT.with(|cell| *cell.borrow_mut() = Some(ctx));
    }

    /// Take and clear the per-thread last-error context. Returns
    /// whatever was last recorded for *this thread*, or `None` if
    /// no error site has fired since the last call.
    pub fn take_last_error_context() -> Option<ErrorContext> {
        LAST_ERROR_CONTEXT.with(|cell| cell.borrow_mut().take())
    }
}

#[cfg(feature = "std")]
pub use context::{set_last_error_context, take_last_error_context};

/// Last error context. Populated by error sites with enough
/// information to debug what went wrong (which region, what offset,
/// how big a request). Read via [`take_last_error_context`] after an
/// error is observed (std-only).
///
/// The struct itself is always defined so it can appear in type
/// signatures regardless of feature gating, but its consumers
/// (`set_last_error_context` / `take_last_error_context`) require
/// `std`.
#[derive(Clone, Debug, Default)]
pub struct ErrorContext {
    pub region_id: Option<u16>,
    pub generation: Option<u16>,
    pub offset: Option<u32>,
    pub len: Option<u32>,
    pub capacity: Option<u32>,
    pub note: Option<&'static str>,
}

impl core::fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        let mut sep = |f: &mut core::fmt::Formatter<'_>| -> core::fmt::Result {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            Ok(())
        };
        write!(f, "[")?;
        if let Some(r) = self.region_id {
            sep(f)?;
            write!(f, "region={}", r)?;
        }
        if let Some(g) = self.generation {
            sep(f)?;
            write!(f, "gen={}", g)?;
        }
        if let Some(o) = self.offset {
            sep(f)?;
            write!(f, "offset={:#x}", o)?;
        }
        if let Some(l) = self.len {
            sep(f)?;
            write!(f, "len={}", l)?;
        }
        if let Some(c) = self.capacity {
            sep(f)?;
            write!(f, "capacity={}", c)?;
        }
        if let Some(n) = self.note {
            sep(f)?;
            write!(f, "note=\"{}\"", n)?;
        }
        write!(f, "]")
    }
}
