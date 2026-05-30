#!/usr/bin/env python3
"""Combine unified histogram files with streaming schema validation."""
from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sfunctor.io.histogram_results import combine_histogram_files


def _default_output(mode: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sf_results{'_all_slices' if mode == 'slice' else ''}_{timestamp}.npz"


def combine_files(input_files, *, mode: str, output: str | None = None, verbose: bool = False) -> int:
    """Stream, validate, and save node or slice histogram products."""

    if verbose:
        print(f"Combining {len(input_files)} {mode} histogram files")
    combined = combine_histogram_files(input_files, mode=mode)
    output = output or _default_output(mode)
    payload = {
        "hist": combined["hist"],
        "channels": combined["channels"],
        "ell_bin_edges": combined["ell_bin_edges"],
        "theta_bin_edges": combined["theta_bin_edges"],
        "phi_bin_edges": combined["phi_bin_edges"],
        "delta_bin_edges": combined["delta_bin_edges"],
        "metadata": combined["metadata"],
    }
    for key in ("hist_censoring", "censor_names"):
        if key in combined:
            payload[key] = combined[key]
    if mode == "node":
        payload["node_infos"] = combined["node_infos"]
    else:
        payload["slice_metadata"] = combined["slice_metadata"]
    np.savez_compressed(output, **payload)
    print(f"Combined results saved to {output}")
    print(f"Total counts in histograms: {combined['hist'].sum()}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True, help="Glob pattern for histogram files")
    parser.add_argument("--output", default=None, help="Output NPZ filename")
    parser.add_argument("--mode", choices=("node", "slice"), default="node")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    input_files = sorted(glob(args.pattern))
    if not input_files:
        parser.error(f"No files found matching pattern {args.pattern!r}")
    return combine_files(input_files, mode=args.mode, output=args.output, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
