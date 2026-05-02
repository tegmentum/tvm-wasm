#!/usr/bin/env python3
"""
Render bench-framework JSON output as a comparison chart.

Usage:
  ./plot.py results/all-1234567890.json [output.png]

Reads the JSON, produces a grouped bar chart per (class, size) showing
mean_ns for each variant. M64 and TVM are highlighted; M32 is the
baseline. If matplotlib isn't installed, the script falls back to ASCII.
"""

import json
import math
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def ascii_chart(samples):
    """Fallback when matplotlib isn't available."""
    by_key = {}
    for s in samples:
        key = (s["class"], s["size_bytes"])
        by_key.setdefault(key, []).append(s)
    # Sort by class then size.
    keys = sorted(by_key.keys(), key=lambda k: (k[0], k[1]))
    print(f"{'class':<14} {'size':>8} {'m32':>10} {'m64':>10} {'tvm-mm':>10} {'tvm':>10} {'TVM/M64':>10}")
    print("-" * 80)
    for key in keys:
        cls, size = key
        rows = by_key[key]
        d = {r["variant"]: r["mean_ns"] for r in rows}
        ratio = (d.get("m64", float("nan")) / d.get("tvm", float("nan"))) if d.get("tvm") else float("nan")
        print(
            f"{cls:<14} {size:>8} "
            f"{d.get('m32', float('nan')):>10.0f} "
            f"{d.get('m64', float('nan')):>10.0f} "
            f"{d.get('tvm-mm', float('nan')):>10.0f} "
            f"{d.get('tvm', float('nan')):>10.0f} "
            f"{ratio:>9.2f}x"
        )


def matplotlib_chart(samples, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    by_class_size = {}
    for s in samples:
        key = (s["class"], s["size_bytes"])
        by_class_size.setdefault(key, []).append(s)

    keys = sorted(by_class_size.keys(), key=lambda k: (k[0], k[1]))
    variants = ["m32", "m64", "tvm-mm", "tvm"]
    colors = {"m32": "#4caf50", "m64": "#f44336", "tvm-mm": "#2196f3", "tvm": "#ff9800"}

    n_keys = len(keys)
    width = 0.2
    x = np.arange(n_keys)
    fig, ax = plt.subplots(figsize=(max(10, n_keys * 0.6), 6))
    for i, var in enumerate(variants):
        means = []
        for key in keys:
            rows = {r["variant"]: r for r in by_class_size[key]}
            v = rows.get(var, {}).get("mean_ns", 0)
            means.append(max(v, 1))  # log axis: avoid zero
        ax.bar(x + (i - 1.5) * width, means, width, label=var, color=colors[var])

    ax.set_yscale("log")
    ax.set_ylabel("mean ns/call (log)")
    ax.set_title("tvm-bench: variants vs class × size  (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{s}B" for c, s in keys], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else str(path).replace(".json", ".png")
    samples = load(path)
    try:
        matplotlib_chart(samples, out)
    except ImportError:
        print("matplotlib not installed; falling back to ASCII chart.")
        ascii_chart(samples)


if __name__ == "__main__":
    main()
