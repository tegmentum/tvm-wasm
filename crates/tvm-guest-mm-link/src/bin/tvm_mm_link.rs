//! CLI: link a rustc-emitted cdylib core wasm against the
//! `tvm-guest-mm` multi-memory shell into a single self-contained
//! `.wasm`.
//!
//! Usage:
//!   tvm-mm-link [--pools N] [--initial-pages P] [--max-pages M] \
//!               --user <user.wasm> -o <out.wasm>

use std::process::ExitCode;

use anyhow::Context;
use tvm_guest_mm_link::{link_with_params_and_options, LinkOptions, ModuleParams, DEFAULT_POOL_COUNT};

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
    let mut aliases: Vec<(String, u32)> = Vec::new();

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
            // `--alias-memory0=<name>` — shorthand for memory 0.
            // `--alias-memory0 <name>` — same, space-separated.
            arg if arg == "--alias-memory0" || arg.starts_with("--alias-memory0=") => {
                let name = if let Some(eq) = arg.strip_prefix("--alias-memory0=") {
                    eq.to_string()
                } else {
                    args.next().context("--alias-memory0 needs a value")?
                };
                if name.is_empty() {
                    anyhow::bail!("--alias-memory0 needs a non-empty name");
                }
                aliases.push((name, 0));
            }
            // `--alias-memory <idx>=<name>` — explicit index.
            // `--alias-memory=<idx>=<name>` — equals form.
            arg if arg == "--alias-memory" || arg.starts_with("--alias-memory=") => {
                let spec = if let Some(eq) = arg.strip_prefix("--alias-memory=") {
                    eq.to_string()
                } else {
                    args.next().context("--alias-memory needs <idx>=<name>")?
                };
                let (idx, name) = spec.split_once('=').with_context(|| {
                    format!("--alias-memory expects <idx>=<name>, got `{spec}`")
                })?;
                let idx: u32 = idx
                    .parse()
                    .with_context(|| format!("--alias-memory index must be u32, got `{idx}`"))?;
                if name.is_empty() {
                    anyhow::bail!("--alias-memory expects a non-empty name");
                }
                aliases.push((name.to_string(), idx));
            }
            "-h" | "--help" => {
                println!(
                    "Usage:\n  tvm-mm-link [--pools N] [--initial-pages P] [--max-pages M] \\\n              [--alias-memory0=<name>] [--alias-memory <idx>=<name>] \\\n              --user <user.wasm> -o <out.wasm>\n\nDefaults: pools={DEFAULT_POOL_COUNT}, initial-pages=1, max-pages=65536\n\n--alias-memory0=<name> adds `(export \"<name>\" (memory 0))` to the merged module.\n  Typical use: `--alias-memory0=memory` for `wasm-tools component new` compat.\n--alias-memory <idx>=<name> aliases any pool memory; may repeat."
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

    let options = LinkOptions { aliases };
    let bytes = link_with_params_and_options(&params, &user_bytes, &options)
        .context("linking modules")?;

    std::fs::write(&out_path, &bytes)
        .with_context(|| format!("writing merged wasm to {out_path}"))?;

    eprintln!(
        "tvm-mm-link: wrote {out_path} ({} bytes, {pools} pools)",
        bytes.len()
    );
    Ok(())
}
