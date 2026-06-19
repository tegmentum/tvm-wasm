//! Generate a deployable multi-memory guest `.wasm` (or `.wat`).
//!
//! The browser Cold-tier embedding (`web/tvm-web`) ships a concrete
//! `.wasm`, not runtime-generated WAT. This binary emits the default
//! 64-pool TVM-MM module — N exported memories (`mem0..memN`) plus the
//! exported dispatch helpers (`tvm_load_u32`, `tvm_store_u32`,
//! `tvm_copy_to_default`, …) — and writes it out.
//!
//! Usage:
//!   gen_guest_wasm [--pools N] [--wat] <out_path>
//!
//! With `--wat` the raw WAT text is written instead of compiled bytes
//! (handy for inspection / `wasm-tools print` diffing).

use std::process::ExitCode;

use tvm_guest_mm::{tvm_guest_mm_module_template, ModuleParams, DEFAULT_POOL_COUNT};

fn main() -> ExitCode {
    let mut pools = DEFAULT_POOL_COUNT;
    let mut emit_wat = false;
    let mut out: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pools" => {
                let Some(v) = args.next() else {
                    return fail("--pools needs a value");
                };
                match v.parse() {
                    Ok(n) => pools = n,
                    Err(_) => return fail("--pools value must be an integer"),
                }
            }
            "--wat" => emit_wat = true,
            "-h" | "--help" => {
                println!("usage: gen_guest_wasm [--pools N] [--wat] <out_path>");
                return ExitCode::SUCCESS;
            }
            other if !other.starts_with('-') => out = Some(other.to_string()),
            other => return fail(&format!("unknown flag: {other}")),
        }
    }

    let Some(out) = out else {
        return fail("missing <out_path>");
    };

    let params = ModuleParams {
        n_pools: pools,
        ..ModuleParams::default()
    };
    let wat = tvm_guest_mm_module_template(&params);

    let result = if emit_wat {
        std::fs::write(&out, wat.as_bytes())
    } else {
        match wat::parse_str(&wat) {
            Ok(bytes) => std::fs::write(&out, bytes),
            Err(e) => return fail(&format!("WAT did not compile: {e}")),
        }
    };

    match result {
        Ok(()) => {
            eprintln!("wrote {out} ({pools} pools, {} bytes WAT)", wat.len());
            ExitCode::SUCCESS
        }
        Err(e) => fail(&format!("write {out}: {e}")),
    }
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("gen_guest_wasm: {msg}");
    ExitCode::FAILURE
}
