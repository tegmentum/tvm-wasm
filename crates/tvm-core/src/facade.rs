//! `TvmFacade` — common interface implemented by both host-side and
//! guest-side TVM. Code generic over `T: TvmFacade` is **deployment-
//! agnostic** — the same logic compiles against `TvmHost` (host-side
//! wasmtime embedding) or a guest-side `GuestTvm` (running inside a
//! `tvm-guest-mm` wasm module).
//!
//! ## What's in the facade
//!
//! Only the operations that are meaningful in both deployments. Things
//! that are inherently host-only (registering imports in a wasmtime
//! `Linker`, multi-store sharing) or inherently guest-only (calling the
//! WAT-generated dispatch helpers from inside a wasm module) live on
//! the concrete types.
//!
//! ## What's deliberately NOT here
//!
//! - **Spill / backing-store ops.** Those go through a separate
//!   `Spill` trait so callers can opt in. A facade impl that doesn't
//!   support spill simply doesn't impl `Spill`.
//! - **Async variants.** Async support is layered above the facade for
//!   the same reason — opt-in via separate trait.
//!
//! ## Why this matters
//!
//! It lets a library author write region-management code once, against
//! `T: TvmFacade`, and have it work with whatever deployment the user
//! picked. The library doesn't care whether bytes live in the host or
//! the guest's linear memory. That separation is the architectural
//! payoff of the two-flavor design.

use crate::error::Result;
use crate::handle::Handle;
use crate::region::{Region, RegionKind};

/// Operations that are meaningful in any TVM deployment.
pub trait TvmFacade {
    /// Create a region with default policy + allocator. Returns the
    /// new region's id.
    fn create_region(&mut self, kind: RegionKind, capacity: u32) -> Result<u16>;

    /// Allocate `size` bytes within an existing region. Returns the
    /// validated handle.
    fn alloc(&mut self, region: u16, size: u32) -> Result<Handle>;

    /// Free a previously-allocated handle. May be a no-op for
    /// allocators that don't track per-handle state (e.g. bump).
    fn dealloc(&mut self, handle: Handle) -> Result<()>;

    /// Read `buf.len()` bytes from the region pointed at by `handle`
    /// into `buf`. Validates generation, bounds.
    fn read(&mut self, handle: Handle, buf: &mut [u8]) -> Result<()>;

    /// Symmetric write.
    fn write(&mut self, handle: Handle, data: &[u8]) -> Result<()>;

    /// Forbid spill/demote on a region. Region must have
    /// `pinnable=true` per its `PlacementPolicy`.
    fn pin(&mut self, region: u16) -> Result<()>;

    fn unpin(&mut self, region: u16) -> Result<()>;

    /// Snapshot of the region's metadata. Returned by value so the
    /// trait stays object-safe.
    fn region_info(&self, region: u16) -> Result<Region>;
}

/// Optional spill/load surface. Layered above `TvmFacade`. Implement
/// when your deployment has somewhere to spill bytes — host-side via
/// `BackingStore`, guest-side via WASI fs / keyvalue / sockets.
pub trait TvmSpill {
    /// Move a region's bytes out of resident storage. Region transitions
    /// to `Cold`. Subsequent reads via plain `read` return `NotResident`
    /// until a `load` happens.
    fn spill(&mut self, region: u16) -> Result<()>;

    /// Bring a `Cold` region's bytes back. Region transitions to `Hot`.
    fn load(&mut self, region: u16) -> Result<()>;
}
