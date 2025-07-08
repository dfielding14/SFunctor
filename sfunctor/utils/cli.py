"""Command-line interface helper for structure-function analysis.

All options common across scripts are centralised here so that other modules
can import :func:`parse_cli` and avoid duplicating *argparse* boilerplate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "RunConfig",
    "parse_cli",
]


@dataclass(slots=True)
class RunConfig:
    """Aggregated, type-checked runtime options.
    
    Attributes
    ----------
    stride : int
        Spatial downsampling factor applied uniformly to both dimensions.
        A value of 2 keeps every 2nd point, reducing data by 4x.
    file_name : Path or None
        Path to a single slice .npz file for analysis.
        Mutually exclusive with slice_list.
    slice_list : Path or None
        Path to text file containing one slice filename per line.
        Used for batch processing with MPI.
    n_disp_total : int
        Total number of unique displacement vectors to generate across
        all magnitude bins. More vectors improve statistics.
    N_random_subsamples : int
        Number of random spatial positions sampled per displacement.
        Increases statistical convergence.
    n_ell_bins : int
        Number of logarithmically-spaced displacement magnitude bins.
    n_processes : int
        Number of worker processes for shared-memory parallelism per node.
        0 means auto-detect (cpu_count - 2).
    stencil_width : int
        Finite difference stencil width for derivatives: 2, 3, or 5.
        Larger stencils are more accurate but require more boundary padding.
    """

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
    """Argument parser type for positive integers.
    
    Parameters
    ----------
    value : str
        String representation of the integer.
    
    Returns
    -------
    int
        The parsed positive integer.
    
    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a positive integer.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"expected positive int, got {value}")
    return ivalue


def parse_cli(argv: Optional[List[str]] = None) -> RunConfig:
    """Parse command-line arguments for structure function analysis.
    
    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments to parse. If None, uses sys.argv[1:].
    
    Returns
    -------
    RunConfig
        Validated configuration object containing all runtime parameters.
    
    Raises
    ------
    SystemExit
        If arguments are invalid or help is requested.
    
    Notes
    -----
    The function enforces mutual exclusion between --file_name and
    --slice_list options. At least one must be provided.
    """
    parser = argparse.ArgumentParser(
        description="2-D slice structure-function histogram builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Analysis arguments
    parser.add_argument("--stride", type=_positive_int, default=None, help="Down-sampling factor applied equally in both in-plane directions.")
    parser.add_argument("--file_name", type=Path, default=None, help="Path to single slice *.npz* file (mutually exclusive with --slice_list).")
    parser.add_argument("--slice_list", type=Path, default=None, help="Text file with one slice-filename per line; each MPI rank processes one line.")

    parser.add_argument("--n_disp_total", type=_positive_int, default=None, help="Total number of (unique) displacement vectors over all ℓ-bins.")
    parser.add_argument("--N_random_subsamples", type=_positive_int, default=None, help="Number of random spatial points per displacement.")
    parser.add_argument("--n_ell_bins", type=_positive_int, default=None, help="Number of logarithmic ℓ-bins.")
    parser.add_argument("--n_processes", type=int, default=None, help="Processes per node (0 → auto: cpu_count() − 2).")
    parser.add_argument("--stencil_width", type=int, choices=[2,3,5], default=None, help="Structure-function stencil width (2, 3, or 5).")

    args = parser.parse_args(argv)
    
    # Build RunConfig with defaults
    config = RunConfig(
        stride=args.stride if args.stride is not None else 1,
        file_name=args.file_name.resolve() if args.file_name else None,
        slice_list=args.slice_list.resolve() if args.slice_list else None,
        n_disp_total=args.n_disp_total if args.n_disp_total is not None else 1000,
        N_random_subsamples=args.N_random_subsamples if args.N_random_subsamples is not None else 1000,
        n_ell_bins=args.n_ell_bins if args.n_ell_bins is not None else 128,
        n_processes=args.n_processes if args.n_processes is not None else 0,
        stencil_width=args.stencil_width if args.stencil_width is not None else 2,
    )
    
    # Validate configuration
    if config.file_name is None and config.slice_list is None:
        parser.error("One of --file_name or --slice_list is required.")
    if config.file_name is not None and config.slice_list is not None:
        parser.error("Options --file_name and --slice_list are mutually exclusive.")
    
    
    return config 