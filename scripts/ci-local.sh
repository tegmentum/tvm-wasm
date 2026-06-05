#!/usr/bin/env bash
# Run the GitHub Actions CI locally with nektos/act (no paid private minutes
# until public). tvm-wasm is self-contained, so its `test` job runs standalone.
#
#   scripts/ci-local.sh           # run the workflow (all jobs)
#   scripts/ci-local.sh test      # run a single job
#
# Machine notes: the active Docker daemon is colima (act otherwise hits the
# inactive /var/run/docker.sock); colima can't bind-mount the daemon socket, so
# we disable that; Apple silicon needs linux/amd64. The Rust build/test job
# wants a roomy colima VM, e.g. `colima start --cpu 6 --memory 8`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${DOCKER_HOST:=unix://$HOME/.colima/default/docker.sock}"
export DOCKER_HOST

ARGS=(-W .github/workflows/ci.yml
  --container-architecture linux/amd64
  --container-daemon-socket -
  -P ubuntu-latest=catthehacker/ubuntu:act-latest)
[[ -n "${1:-}" ]] && ARGS+=(-j "$1")
exec act "${ARGS[@]}"
