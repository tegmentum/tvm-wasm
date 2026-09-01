//! Wit-bindgen linker installers for the component-model host
//! interfaces. See the deprecation notes below for the migration
//! path to the wasmos-native install fns landed at D2 Sessions 3-4.

use wasmtime::component::{HasSelf, Linker};

use crate::bindings::tvm::memory::{bytes, diagnostics, manager, types};
use crate::bindings::TvmGuest;
use crate::concurrent_host::ConcurrentTvmHost;
use crate::host::TvmHost;
use crate::per_actor::PerActorTvmHost;
use crate::shared_host::SharedTvmHost;

/// Convenience wrapper for stores whose data implements `AsMut<TvmHost>`.
/// Single-threaded use.
///
/// **Deprecated (ADR-0029 Phase 6.9 D2 Session 5).** The wit-bindgen
/// path is being wound down as the wasmos-native install fns
/// mature. Prefer
/// [`crate::wasmos_bindings::install_tvm_imports_per_actor`]`::<T>`
/// with the wasmos [`ExecutionContext::with_host_imports`] flow.
/// Same set of interfaces installed, portable across every
/// wasmos-backed adapter (v48, edge, WAMR). The wit-bindgen entries
/// stay callable for the crate's own tests + benches; new consumers
/// should route through the wasmos path.
#[deprecated(
    since = "0.1.1",
    note = "Use `wasmos_bindings::install_tvm_imports_per_actor::<T>` \
            with a wasmos-backed ExecutionContext. See linker.rs \
            module docstring for the migration."
)]
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
///
/// **Deprecated (ADR-0029 Phase 6.9 D2 Session 5).** Prefer
/// [`crate::wasmos_bindings::install_tvm_imports_shared`] — same
/// shared-mutex semantics, portable across adapters.
#[deprecated(
    since = "0.1.1",
    note = "Use `wasmos_bindings::install_tvm_imports_shared(imports, SharedTvmHost)` \
            with a wasmos-backed ExecutionContext."
)]
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
///
/// **Deprecated (ADR-0029 Phase 6.9 D2 Session 5).** No wasmos-side
/// equivalent yet; the `ConcurrentTvmHost` per-region-locking shape
/// is not directly mirrored by `TvmHostSource`. If your workload
/// hits this warning, either stay on this entry point behind
/// `#[allow(deprecated)]` or open a wasmos-side design session for
/// a concurrent variant (`TvmHostSource::Concurrent(ConcurrentTvmHost)`
/// would be the natural extension).
#[deprecated(
    since = "0.1.1",
    note = "No wasmos-side ConcurrentTvmHost variant yet. Stay on this \
            entry behind `#[allow(deprecated)]` if you need per-region \
            parallelism, or use install_tvm_imports_shared for the \
            single-mutex shape."
)]
pub fn add_concurrent_to_linker<T: AsMut<ConcurrentTvmHost> + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    fn project<T: AsMut<ConcurrentTvmHost>>(state: &mut T) -> &mut ConcurrentTvmHost {
        state.as_mut()
    }
    TvmGuest::add_to_linker::<T, HasSelf<ConcurrentTvmHost>>(linker, project::<T>)
}

/// Convenience wrapper for stores whose data implements
/// `AsMut<PerActorTvmHost>`. Each store gets its own outstanding-bytes
/// accounting + overrun flag, while the *inner* `SharedTvmHost` they
/// each wrap stays a shared substrate — the right shape for embedders
/// that host *untrusted* actors on a shared directory and want one to
/// be unable to exhaust the substrate for the others.
///
/// **Deprecated (ADR-0029 Phase 6.9 D2 Session 5).** The per-actor
/// budget-accounting shape is not directly mirrored by the wasmos-
/// side per-actor variant (which pulls raw `TvmHost` via
/// `ctx.consumer_state::<T>()`). If your workload needs
/// `PerActorTvmHost`'s outstanding-bytes accounting, stay on this
/// entry behind `#[allow(deprecated)]`; a wasmos-side variant
/// would need a design session.
#[deprecated(
    since = "0.1.1",
    note = "PerActorTvmHost's budget-accounting shape has no direct \
            wasmos mirror yet. Stay on this entry behind \
            `#[allow(deprecated)]` if you need it, or use \
            install_tvm_imports_per_actor::<T> for the raw per-actor \
            TvmHost shape."
)]
pub fn add_per_actor_to_linker<T: AsMut<PerActorTvmHost> + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    fn project<T: AsMut<PerActorTvmHost>>(state: &mut T) -> &mut PerActorTvmHost {
        state.as_mut()
    }
    TvmGuest::add_to_linker::<T, HasSelf<PerActorTvmHost>>(linker, project::<T>)
}

/// Generic helper. Use when your store data isn't `AsMut<H>` directly — for
/// instance when `T` is an application struct that owns the host alongside
/// other state. Pass a `fn` pointer that projects from `&mut T` to `&mut H`.
///
/// **Deprecated (ADR-0029 Phase 6.9 D2 Session 5).** This escape
/// hatch predates `TvmHostExtractor`; the wasmos-side pattern is
/// [`crate::wasmos_bindings::TvmHostExtractor`] which takes the
/// same idea to the `HostCallContext::consumer_state<T>()` path.
/// New code should implement `TvmHostExtractor` and pass an
/// `Arc<dyn TvmHostExtractor>` to `TvmHostSource::PerActor`.
#[deprecated(
    since = "0.1.1",
    note = "Use `wasmos_bindings::TvmHostExtractor` + \
            TvmHostSource::PerActor for the same generic-projection shape."
)]
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
