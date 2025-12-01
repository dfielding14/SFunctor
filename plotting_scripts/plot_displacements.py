#!/usr/bin/env python3
"""
Create quick scatter plots of displacement positions from displacements.npz files.

For every displacements.npz under the given results root, save displacements.png
in the same directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "figure.dpi": 120,
        }
    )


def plot_displacements(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    if "displacements" not in data:
        print(f"  Skipping (no displacements): {npz_path}")
        return
    disp = np.asarray(data["displacements"])
    if disp.ndim != 2 or disp.shape[1] < 2:
        print(f"  Skipping (unexpected shape {disp.shape}): {npz_path}")
        return

    out_path = npz_path.with_name("displacements.png")
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(disp[:, 0], disp[:, 1], s=3, alpha=0.35, c="tab:blue", linewidths=0)
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.set_aspect("equal", adjustable="box")
    margin = 0.05
    xmin, xmax = disp[:, 0].min(), disp[:, 0].max()
    ymin, ymax = disp[:, 1].min(), disp[:, 1].max()
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
    ax.set_ylim(ymin - margin * dy, ymax + margin * dy)
    ax.grid(True, alpha=0.25)
    rel = npz_path.parent
    ax.set_title(str(rel))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot displacement positions from displacements.npz files.")
    parser.add_argument("--results-root", type=Path, default=Path("Results"), help="Root directory to search (default: Results)")
    args = parser.parse_args()

    apply_style()

    paths = sorted(args.results_root.rglob("displacements.npz"))
    if not paths:
        print("No displacements.npz files found.")
        return 1

    print(f"Found {len(paths)} displacements.npz files.")
    for p in paths:
        plot_displacements(p)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
