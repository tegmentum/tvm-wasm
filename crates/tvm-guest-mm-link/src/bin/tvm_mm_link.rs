//! CLI: link a rustc-emitted cdylib core wasm against the
//! `tvm-guest-mm` multi-memory shell into a single self-contained
//! `.wasm`.
//!
//! Usage:
//!   tvm-mm-link [--pools N] [--initial-pages P] [--max-pages M] \
//!               --user <user.wasm> -o <out.wasm>

use std::process::ExitCode;

use anyhow::Context;
use tvm_guest_mm_link::{link_with_params, ModuleParams, DEFAULT_POOL_COUNT};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tvm-mm-link: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let mut pools = DEFAULT_POOL_COUNT;
    let mut initial_pages = 1u32;
    let mut max_pages = 65536u32;
    let mut user: Option<String> = None;
    let mut out: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pools" => {
                pools = args
                    .next()
                    .context("--pools needs a value")?
                    .parse()
                    .context("--pools value must be an integer")?;
            }
            "--initial-pages" => {
                initial_pages = args
                    .next()
                    .context("--initial-pages needs a value")?
                    .parse()
                    .context("--initial-pages value must be an integer")?;
            }
            "--max-pages" => {
                max_pages = args
                    .next()
                    .context("--max-pages needs a value")?
                    .parse()
                    .context("--max-pages value must be an integer")?;
            }
            "--user" => {
                user = args.next();
                if user.is_none() {
                    anyhow::bail!("--user needs a value");
                }
            }
            "-o" | "--output" => {
                out = args.next();
                if out.is_none() {
                    anyhow::bail!("-o needs a value");
                }
            }
            "-h" | "--help" => {
                println!(
                    "Usage:\n  tvm-mm-link [--pools N] [--initial-pages P] [--max-pages M] \\\n              --user <user.wasm> -o <out.wasm>\n\nDefaults: pools={DEFAULT_POOL_COUNT}, initial-pages=1, max-pages=65536"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let user_path = user.context("--user is required")?;
    let out_path = out.context("-o/--output is required")?;

    let user_bytes = std::fs::read(&user_path)
        .with_context(|| format!("reading user wasm at {user_path}"))?;

    let params = ModuleParams {
        n_pools: pools,
        initial_pages_per_pool: initial_pages,
        max_pages_per_pool: max_pages,
        user_body: String::new(),
    };

    let bytes = link_with_params(&params, &user_bytes).context("linking modules")?;

    std::fs::write(&out_path, &bytes)
        .with_context(|| format!("writing merged wasm to {out_path}"))?;

    eprintln!(
        "tvm-mm-link: wrote {out_path} ({} bytes, {pools} pools)",
        bytes.len()
    );
    Ok(())
}
