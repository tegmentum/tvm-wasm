// `pub mod bindings` is intentionally public. The bindgen-generated
// `Host` traits (`tvm::memory::manager::Host` etc.) are part of the public
// API: downstream crates (including our own tests) implement them when
// building custom hosts. The `prelude` module re-exports the helper
// functions; the trait paths here remain available for users who need
// them by name.
//
// # ADR-0029 Phase 6.9 D2 Session 11 — retirement stance
//
// **Status: RETAINED, retirement gated on downstream design work.**
//
// The D2 Sessions 2-10 arc landed a full wasmos-native mirror of every
// interface this bindgen! generates (see `wasmos_bindings` module).
// The wit-bindgen wrapper stays in place because:
//
//  1. `linker.rs` still exposes 5 `add_*_to_linker` entry points marked
//     `#[deprecated]` (Phase 6.9 D2 Session 5). They consume the `Host`
//     traits + record types this bindgen! emits. Their deprecation
//     notes name the wasmos-native replacements for callers who can
//     migrate today.
//  2. Two of those 5 have NO wasmos-native peer yet:
//        * `add_concurrent_to_linker` (backed by `ConcurrentTvmHost`
//          with per-region locking) — needs a `TvmHostSource::Concurrent`
//          variant in `wasmos_bindings` before its callers can migrate.
//        * `add_per_actor_to_linker` (backed by `PerActorTvmHost` with
//          outstanding-bytes accounting) — needs a wasmos-side model
//          for per-actor budget accounting that the current TvmHostSource
//          doesn't cover.
//     Both are design work (a new enum variant / a companion trait);
//     both are gated on a concrete workload asking for the wasmos
//     variant.
//  3. The crate's own tests (`tests/{raw_linker,shared_host,end_to_end_
//     guest,coexistence,guest_rt_integration,end_to_end_fast_path,
//     component_host}.rs`) carry `#![allow(deprecated)]` and continue
//     to exercise the wit-bindgen path as the reference implementation
//     for both raw + component-model dispatch. Retaining the bindings!
//     macro is what keeps that reference alive.
//
// # Retirement path
//
// When both wasmos-side design tasks (Concurrent variant + per-actor
// budget accounting) land AND downstream consumers finish migrating
// off the deprecated `add_*_to_linker` entries, the sequence is:
//
//  1. Delete the 5 `add_*_to_linker` fns from `linker.rs`.
//  2. Migrate the internal reference tests to the wasmos install fns.
//  3. Delete the `pub mod bindings` re-export from `lib.rs`.
//  4. Remove this `bindings.rs` file + the WIT-file dep in Cargo.toml.
//
// The mirror types in `wasmos_bindings` become the sole public shape.

wasmtime::component::bindgen!({
    path: "../../wit",
    world: "tvm-guest",
});
