#!/usr/bin/env python3
"""
Regression checker for nightly CI.

Compares a fresh bench JSON against `results/baseline.json` and exits
non-zero if any TVM variant's mean_ns regresses by more than 25% on any
(class, size) pair. Tolerant of cross-machine noise — the threshold is
intentionally loose; tighten in CI by pinning the runner to a known
machine type.

Usage:
  ./check_regression.py results/all-1234567890.json
"""

import json
import sys
from pathlib import Path

THRESHOLD = 1.25  # 25% regression


def load(path):
    with open(path) as f:
        return json.load(f)


def index(samples):
    return {(s["variant"], s["class"], s["size_bytes"]): s for s in samples}


def main():
    if len(sys.argv) < 2:
        print("usage: check_regression.py <fresh.json>")
        sys.exit(1)
    fresh_path = Path(sys.argv[1])
    base_path = fresh_path.parent / "baseline.json"
    if not base_path.exists():
        print(f"baseline not found: {base_path}; nothing to compare against.")
        sys.exit(0)

    fresh = index(load(fresh_path))
    base = index(load(base_path))

    regressions = []
    for key, b in base.items():
        if key[0] not in ("tvm", "tvm-mm"):
            continue
        f = fresh.get(key)
        if not f:
            continue
        if b["mean_ns"] <= 0 or f["mean_ns"] <= 0:
            continue
        ratio = f["mean_ns"] / b["mean_ns"]
        if ratio > THRESHOLD:
            regressions.append((key, b["mean_ns"], f["mean_ns"], ratio))

    if not regressions:
        print(f"no regressions vs baseline (threshold {THRESHOLD:.0%}).")
        sys.exit(0)

    print(f"REGRESSIONS DETECTED (>{THRESHOLD:.0%} slowdown):")
    for (variant, cls, size), b, f, r in regressions:
        print(f"  {variant:<8} {cls:<24} size={size:<8} {b:>10.0f}ns -> {f:>10.0f}ns  ({r:.2f}x)")
    sys.exit(1)


if __name__ == "__main__":
    main()
