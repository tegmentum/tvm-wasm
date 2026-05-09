// `pub mod bindings` is intentionally public. The bindgen-generated
// `Host` traits (`tvm::memory::manager::Host` etc.) are part of the public
// API: downstream crates (including our own tests) implement them when
// building custom hosts. The `prelude` module re-exports the helper
// functions; the trait paths here remain available for users who need
// them by name.

wasmtime::component::bindgen!({
    path: "../../wit",
    world: "tvm-guest",
});
