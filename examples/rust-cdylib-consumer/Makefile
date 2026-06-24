# Build pipeline for the rust-cdylib-consumer example.
#
# Pipeline:
#   1. cargo build --release --target wasm32-unknown-unknown
#      → core wasm with `tvm_mm.*` imports
#   2. tvm-mm-link --user <core>.wasm -o <linked>.wasm
#      → single self-contained .wasm with N memories and zero
#        unresolved imports
#
# Usage:
#   make           — full pipeline
#   make clean     — drop build artifacts (cargo's target/ stays)

POOLS ?= 4
INITIAL_PAGES ?= 1
# 256 pages = 16 MiB per pool, enough for the example's 4-page test
# load. Bump for real workloads; the shell template defaults max to
# 65536 (4 GiB).
MAX_PAGES ?= 256

# The workspace root (relative to this Makefile). Used to find the
# `tvm-mm-link` binary, which lives in the workspace `target/release/`.
ROOT ?= ../..

# This example is `exclude`d from the host workspace (so a workspace-
# wide `cargo build` doesn't try to compile a cdylib for the host
# triple) — cargo therefore puts its output in this directory's own
# `target/` rather than the workspace target.
CORE_WASM = target/wasm32-unknown-unknown/release/tvm_rust_cdylib_consumer.wasm
LINKED_WASM = target/wasm32-unknown-unknown/release/consumer.linked.wasm
TVM_MM_LINK = $(ROOT)/target/release/tvm-mm-link

.PHONY: all clean show

all: $(LINKED_WASM)

$(CORE_WASM): src/lib.rs Cargo.toml
	cargo build --release --target wasm32-unknown-unknown \
	  --manifest-path Cargo.toml

$(TVM_MM_LINK):
	cargo build --release -p tvm-guest-mm-link --manifest-path $(ROOT)/Cargo.toml

$(LINKED_WASM): $(CORE_WASM) $(TVM_MM_LINK)
	$(TVM_MM_LINK) \
	  --pools $(POOLS) \
	  --initial-pages $(INITIAL_PAGES) \
	  --max-pages $(MAX_PAGES) \
	  --user $(CORE_WASM) \
	  -o $(LINKED_WASM)

# Print the linked module's exports + imports so reviewers can confirm
# `tvm:memory` imports are gone and the workload exports are intact.
show: $(LINKED_WASM)
	@echo "--- imports (should be none) ---"
	@wasm-tools print $(LINKED_WASM) | grep -E "^  \(import" || echo "(none)"
	@echo "--- memories ---"
	@wasm-tools print $(LINKED_WASM) | grep -E "^  \(memory" | head
	@echo "--- user exports ---"
	@wasm-tools print $(LINKED_WASM) | grep -E "_page|_byte|_u32|sum_hot|materialize|install|evict" | head -20

clean:
	rm -f $(LINKED_WASM)
