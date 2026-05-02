use wasmtime::component::Linker;

use crate::bindings::tvm::memory::{bytes, diagnostics, manager, types};
use crate::bindings::TvmGuest;
use crate::concurrent_host::ConcurrentTvmHost;
use crate::host::TvmHost;
use crate::shared_host::SharedTvmHost;

/// Convenience wrapper for stores whose data implements `AsMut<TvmHost>`.
/// Single-threaded use.
pub fn add_to_linker<T: AsMut<TvmHost> + Send>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    TvmGuest::add_to_linker(linker, |state: &mut T| state.as_mut())
}

/// Convenience wrapper for stores whose data implements `AsMut<SharedTvmHost>`.
/// The shared host serializes calls through an internal mutex, allowing
/// multiple stores (typically on different threads) to share region state.
pub fn add_shared_to_linker<T: AsMut<SharedTvmHost> + Send>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    TvmGuest::add_to_linker(linker, |state: &mut T| state.as_mut())
}

/// Convenience wrapper for stores whose data implements
/// `AsMut<ConcurrentTvmHost>`. Per-region locking lets operations on
/// different regions run in parallel — preferred over `add_shared_to_linker`
/// when stores hit distinct regions.
pub fn add_concurrent_to_linker<T: AsMut<ConcurrentTvmHost> + Send>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    TvmGuest::add_to_linker(linker, |state: &mut T| state.as_mut())
}

/// Generic helper. Use when your store data isn't `AsMut<H>` directly — for
/// instance when `T` is an application struct that owns the host alongside
/// other state. The closure must extract a `&mut H` from `&mut T`.
pub fn add_to_linker_with<T, H>(
    linker: &mut Linker<T>,
    get: impl Fn(&mut T) -> &mut H + Send + Sync + Copy + 'static,
) -> wasmtime::Result<()>
where
    H: types::Host + manager::Host + bytes::Host + diagnostics::Host + Send,
    T: Send,
{
    TvmGuest::add_to_linker(linker, get)
}
