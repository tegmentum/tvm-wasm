#!/usr/bin/env bash
# Build the wasm guest workloads. Requires the wasm32-unknown-unknown
# Rust target. Idempotent; safe to re-run.
set -euo pipefail

if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
  echo "ERROR: wasm32-unknown-unknown target is not installed. Run:"
  echo "  rustup target add wasm32-unknown-unknown"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> building m32 workload..."
( cd "$ROOT/workloads/m32" && cargo build --release --target wasm32-unknown-unknown )

echo "==> building tvm workload..."
( cd "$ROOT/workloads/tvm" && cargo build --release --target wasm32-unknown-unknown )

# Optional: m64 workload (requires nightly + rust-src for build-std).
if rustup +nightly --version > /dev/null 2>&1; then
  if rustup +nightly component list --installed 2>/dev/null | grep -q rust-src; then
    echo "==> building m64 workload (nightly)..."
    ( cd "$ROOT/workloads/m64" && cargo +nightly build --release \
        -Zbuild-std=panic_abort,std \
        --target wasm64-unknown-unknown ) || \
      echo "    (m64 build failed; runner will skip the M64 column)"
  else
    echo "==> skipping m64: nightly toolchain present but rust-src missing."
    echo "    install with: rustup +nightly component add rust-src"
  fi
else
  echo "==> skipping m64: nightly toolchain not installed."
  echo "    install with: rustup install nightly"
fi

echo "==> done."
echo "    run benchmarks with:"
echo "      cargo run -p tvm-bench-runner --release"
