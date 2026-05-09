use wasmtime::component::{HasSelf, Linker};

use crate::bindings::tvm::memory::{bytes, diagnostics, manager, types};
use crate::bindings::TvmGuest;
use crate::concurrent_host::ConcurrentTvmHost;
use crate::host::TvmHost;
use crate::shared_host::SharedTvmHost;

/// Convenience wrapper for stores whose data implements `AsMut<TvmHost>`.
/// Single-threaded use.
pub fn add_to_linker<T: AsMut<TvmHost> + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    fn project<T: AsMut<TvmHost>>(state: &mut T) -> &mut TvmHost {
        state.as_mut()
    }
    TvmGuest::add_to_linker::<T, HasSelf<TvmHost>>(linker, project::<T>)
}

/// Convenience wrapper for stores whose data implements `AsMut<SharedTvmHost>`.
/// The shared host serializes calls through an internal mutex, allowing
/// multiple stores (typically on different threads) to share region state.
pub fn add_shared_to_linker<T: AsMut<SharedTvmHost> + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    fn project<T: AsMut<SharedTvmHost>>(state: &mut T) -> &mut SharedTvmHost {
        state.as_mut()
    }
    TvmGuest::add_to_linker::<T, HasSelf<SharedTvmHost>>(linker, project::<T>)
}

/// Convenience wrapper for stores whose data implements
/// `AsMut<ConcurrentTvmHost>`. Per-region locking lets operations on
/// different regions run in parallel — preferred over `add_shared_to_linker`
/// when stores hit distinct regions.
pub fn add_concurrent_to_linker<T: AsMut<ConcurrentTvmHost> + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    fn project<T: AsMut<ConcurrentTvmHost>>(state: &mut T) -> &mut ConcurrentTvmHost {
        state.as_mut()
    }
    TvmGuest::add_to_linker::<T, HasSelf<ConcurrentTvmHost>>(linker, project::<T>)
}

/// Generic helper. Use when your store data isn't `AsMut<H>` directly — for
/// instance when `T` is an application struct that owns the host alongside
/// other state. Pass a `fn` pointer that projects from `&mut T` to `&mut H`.
pub fn add_to_linker_with<T, H>(
    linker: &mut Linker<T>,
    get: fn(&mut T) -> &mut H,
) -> wasmtime::Result<()>
where
    H: types::Host + manager::Host + bytes::Host + diagnostics::Host + Send + 'static,
    T: Send + 'static,
{
    TvmGuest::add_to_linker::<T, HasSelf<H>>(linker, get)
}
