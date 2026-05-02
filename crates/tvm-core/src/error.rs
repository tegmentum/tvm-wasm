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

/// Last error context. Populated by error sites with enough information to
/// debug what went wrong (which region, what offset, how big a request).
/// Read via [`take_last_error_context`] after an error is observed.
///
/// Thread-local — each thread gets its own context. Cleared on take.
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

std::thread_local! {
    static LAST_ERROR_CONTEXT: std::cell::RefCell<Option<ErrorContext>> =
        std::cell::RefCell::new(None);
}

/// Set the per-thread last-error context. Called internally by error
/// sites; users typically don't call this directly.
pub fn set_last_error_context(ctx: ErrorContext) {
    LAST_ERROR_CONTEXT.with(|cell| *cell.borrow_mut() = Some(ctx));
}

/// Take and clear the per-thread last-error context. Returns whatever
/// was last recorded for *this thread*, or `None` if no error site has
/// fired since the last call.
///
/// ```ignore
/// match host.write_bytes(handle, &data) {
///     Ok(()) => { /* fine */ }
///     Err(e) => {
///         let ctx = tvm_core::take_last_error_context();
///         eprintln!("write failed: {} {:?}", e, ctx);
///     }
/// }
/// ```
pub fn take_last_error_context() -> Option<ErrorContext> {
    LAST_ERROR_CONTEXT.with(|cell| cell.borrow_mut().take())
}

pub type Result<T> = std::result::Result<T, TvmError>;
