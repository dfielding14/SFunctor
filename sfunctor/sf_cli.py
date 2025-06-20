"""Command-line interface helper for structure-function analysis.

All options common across scripts are centralised here so that other modules
can import :func:`parse_cli` and avoid duplicating *argparse* boilerplate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

__all__ = [
    "RunConfig",
    "parse_cli",
]


@dataclass(slots=True)
class RunConfig:
    """Aggregated, type-checked runtime options."""

    stride: int
    file_name: Optional[Path]
    slice_list: Optional[Path]
    n_disp_total: int
    N_random_subsamples: int
    n_ell_bins: int
    n_processes: int  # per-rank shared-memory pool size
    stencil_width: int  # 2, 3, or 5


# ----------------------------------------------------------------------------
# Parser ---------------------------------------------------------------------
# ----------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"expected positive int, got {value}")
    return ivalue


def parse_cli(argv: Optional[List[str]] = None) -> RunConfig:
    """Parse *argv* (default: *sys.argv[1:])* and return :class:`RunConfig`."""
    parser = argparse.ArgumentParser(
        description="2-D slice structure-function histogram builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--stride",
        type=_positive_int,
        default=1,
        help="Down-sampling factor applied equally in both in-plane directions.",
    )
    parser.add_argument(
        "--file_name",
        type=Path,
        default=None,
        help="Path to single slice *.npz* file (mutually exclusive with --slice_list).",
    )
    parser.add_argument(
        "--slice_list",
        type=Path,
        default=None,
        help="Text file with one slice-filename per line; each MPI rank processes one line.",
    )

    parser.add_argument(
        "--n_disp_total",
        type=_positive_int,
        default=1000,
        help="Total number of (unique) displacement vectors over all ℓ-bins.",
    )
    parser.add_argument(
        "--N_random_subsamples",
        type=_positive_int,
        default=1000,
        help="Number of random spatial points per displacement.",
    )
    parser.add_argument(
        "--n_ell_bins",
        type=_positive_int,
        default=128,
        help="Number of logarithmic ℓ-bins.",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="Processes per node (0 → auto: cpu_count() − 2).",
    )
    parser.add_argument(
        "--stencil_width",
        type=int,
        choices=[2, 3, 5],
        default=2,
        help="Structure-function stencil width (2, 3, or 5).",
    )

    args = parser.parse_args(argv)

    # Mutual-exclusion logic ---------------------------------------------------
    if args.file_name is None and args.slice_list is None:
        parser.error("One of --file_name or --slice_list is required.")
    if args.file_name is not None and args.slice_list is not None:
        parser.error("Options --file_name and --slice_list are mutually exclusive.")

    return RunConfig(
        stride=args.stride,
        file_name=args.file_name.resolve() if args.file_name else None,
        slice_list=args.slice_list.resolve() if args.slice_list else None,
        n_disp_total=args.n_disp_total,
        N_random_subsamples=args.N_random_subsamples,
        n_ell_bins=args.n_ell_bins,
        n_processes=args.n_processes,
        stencil_width=args.stencil_width,
    )
