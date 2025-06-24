#!/usr/bin/env python3
"""MPI-capable front-end for extracting 2-D slices from a 3-D Athena-K run.

Overview
--------
This script is the *extraction* counterpart to ``run_sf.py``.  It calls
:pyfunc:`extract_2d_slice.extract_2d_slice` once per MPI rank to pull out
pre-defined 2-D planes (density, velocity, magnetic field, vorticity and
current) from a large 3-D dataset and stores them in ``*.npz`` archives that
``run_sf.py`` can ingest directly.

Default slice set
~~~~~~~~~~~~~~~~~
• Simulation name: ``Turb_10240_beta25_dedt025_plm``
• Axes extracted: 1, 2, 3 (i.e. *x*, *y*, *z* normals)
• Slice offsets (fraction of box size): −3/8, −1/8, +1/8, +3/8
  → total of 12 slices (3 axes × 4 offsets)

Command-line usage
------------------
Single node (serial):
    python run_extract_slice.py --plot

Multi-node / cluster:
    mpirun ‑n 12 python run_extract_slice.py \
        --sim_name Turb_10240_beta25_dedt025_plm \
        --offsets -0.3,0.0,0.3 --axes 1,3 --plot

Available options (also shown with ``-h``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--sim_name     Base directory name of the Athena-K run (default shown above)
--offsets      Comma-separated list of slice positions in the range [-0.5, 0.5]
--axes         Comma-separated list of axes to slice (1=x, 2=y, 3=z)
--file_number  Specific snapshot index to read (None → latest file per rank)
--plot         If given, create a *.png* density image for each slice
--cache_dir    Destination directory for *.npz* and *.png* outputs

Output layout
-------------
Each rank writes exactly one archive
    slice_x{axis}_{offset}_{sim_name}[_{fileIdx}].npz
where *offset* is rendered with up to 6 decimal places, ``m`` replaces minus
(e.g. ``-0.125`` → ``-0p125``) and *fileIdx* appears only when ``--file_number``
is specified.  The file contains the 2-D arrays:
    dens, velx, vely, velz, bcc1, bcc2, bcc3,
    vortx, vorty, vortz, currx, curry, currz
which match exactly the field names expected by ``sf_io.load_slice_npz``.

The script is safe to run repeatedly; existing files will be overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# MPI import with graceful fallback -----------------------------------------
# ---------------------------------------------------------------------------
# The extraction step can run perfectly fine on a single workstation without
# ``mpi4py``.  We therefore attempt to import it, but fall back to a minimal
# stub that mimics the subset of the API we rely on when the package is not
# available.

try:
    from mpi4py import MPI  # type: ignore
    _mpi_enabled = True
except ModuleNotFoundError:
    _mpi_enabled = False

    class _SerialComm:
        def Get_rank(self):  # noqa: N802
            return 0

        def Get_size(self):  # noqa: N802
            return 1

        def bcast(self, obj, root=0):  # noqa: ANN001, N802
            return obj

        def Barrier(self):  # noqa: N802
            pass

    class _FakeMPI:  # pylint: disable=too-few-public-methods
        COMM_WORLD = _SerialComm()

    MPI = _FakeMPI()  # type: ignore

# ---------------------------------------------------------------------------
# Basic MPI variables (valid in both real and stubbed mode) ------------------
# ---------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Local imports --------------------------------------------------------------
from extract_2d_slice import extract_2d_slice

# ----------------------------------------------------------------------------
# CLI helper ------------------------------------------------------------------
# ----------------------------------------------------------------------------

@dataclass(slots=True)
class RunConfig:
    """Aggregated, type-checked runtime options."""

    sim_name: str
    slice_offsets: Tuple[float, ...]
    axes: Tuple[int, ...]
    file_number: Optional[int]
    plot: bool
    cache_dir: Path


def _float_list(arg: str) -> Tuple[float, ...]:
    """Parse a comma-separated list of floats."""
    try:
        values = tuple(float(x) for x in arg.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None
    if not values:
        raise argparse.ArgumentTypeError("list of floats cannot be empty")
    return values


def _int_list(arg: str) -> Tuple[int, ...]:
    """Parse a comma-separated list of ints."""
    try:
        values = tuple(int(x) for x in arg.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None
    if not values:
        raise argparse.ArgumentTypeError("list of ints cannot be empty")
    if any(v not in (1, 2, 3) for v in values):
        raise argparse.ArgumentTypeError("axes must be 1, 2, or 3")
    return values


def parse_cli(argv: Optional[List[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Extract 2-D slices from a 3-D AthenaK simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sim_name",
        default="Turb_10240_beta25_dedt025_plm",
        help="Base name of the AthenaK simulation (matches directory names).",
    )
    parser.add_argument(
        "--offsets",
        type=_float_list,
        default=(-3/8, -1/8, 1/8, 3/8),
        help="Comma-separated list of slice positions along each axis (−0.5 … 0.5).",
    )
    parser.add_argument(
        "--axes",
        type=_int_list,
        default=(1, 2, 3),
        help="Comma-separated list of axis indices to extract (1=x, 2=y, 3=z).",
    )
    parser.add_argument(
        "--file_number",
        type=int,
        default=None,
        help="Index of time-snapshot binary to read (None → latest).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a quick density plot for each slice as a sanity check.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("slice_data"),
        help="Directory where extracted slices and images are saved.",
    )

    args = parser.parse_args(argv)

    return RunConfig(
        sim_name=args.sim_name,
        slice_offsets=args.offsets,
        axes=args.axes,
        file_number=args.file_number,
        plot=args.plot,
        cache_dir=args.cache_dir.resolve(),
    )

# ----------------------------------------------------------------------------
# Utility functions -----------------------------------------------------------
# ----------------------------------------------------------------------------

def _format_float(val: float) -> str:
    """Return *val* as filename-friendly string with up to 6 decimals."""
    s = f"{val:+.6f}"  # include sign, fixed decimals
    s = s.rstrip("0").rstrip(".")  # trim trailing zeros/dot
    return s.replace("+", "")  # drop plus sign for positives


def _save_slice_npz(
    data: dict[str, np.ndarray],
    axis: int,
    offset: float,
    sim_name: str,
    file_number: Optional[int],
    cache_dir: Path,
) -> Path:
    """Write *data* to disk in the canonical naming scheme and return the path."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    offset_str = _format_float(offset)
    file_part = (
        f"_{file_number:04d}" if file_number is not None else ""
    )
    out_name = cache_dir / f"slice_x{axis}_{offset_str}_{sim_name}{file_part}.npz"

    np.savez(out_name, **data)
    return out_name


def _plot_density(slice_path: Path, density: np.ndarray) -> None:
    """Save a density image next to *slice_path* (PNG, same stem)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(density, origin="lower", cmap="cmasher.iceburn", aspect="equal")
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        ax.set_title(slice_path.stem)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(slice_path.with_suffix(".png"), dpi=200)
        plt.close(fig)
    except Exception as err:
        print(f"[run_extract_slice] Warning: could not plot slice '{slice_path}': {err}")

# ----------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main() -> None:
    cfg = parse_cli()

    # ------------------------------------------------------------------
    # Build the full task list on rank-0 and share with everyone ---------
    # ------------------------------------------------------------------
    if rank == 0:
        tasks: List[Tuple[int, float]] = [
            (ax, off) for ax in cfg.axes for off in cfg.slice_offsets
        ]
    else:
        tasks = None  # type: ignore[assignment]

    tasks = comm.bcast(tasks, root=0)

    # Distribute tasks --------------------------------------------------
    if size > 1:
        tasks_my_rank = tasks[rank::size]
        if not tasks_my_rank:
            # More ranks than tasks
            return
    else:
        tasks_my_rank = tasks

    if rank == 0:
        mode = "MPI" if _mpi_enabled and size > 1 else "serial"
        print(
            f"[run_extract_slice] Starting extraction in {mode} mode with {size} rank(s) -> {len(tasks)} slices"
        )

    # ------------------------------------------------------------------
    # Main processing loop ---------------------------------------------
    # ------------------------------------------------------------------
    for axis, offset in tasks_my_rank:
        _process_single_slice(axis, offset, cfg)


def _process_single_slice(axis: int, offset: float, cfg) -> None:  # noqa: D401, ANN001
    """Extract one slice and handle optional plotting/saving."""

    slice_data = extract_2d_slice(
        cfg.sim_name,
        axis,
        offset,
        file_number=cfg.file_number,
        save=False,
    )

    out_path = _save_slice_npz(
        slice_data,
        axis,
        offset,
        cfg.sim_name,
        cfg.file_number,
        cfg.cache_dir,
    )

    if rank == 0:
        print(f"[run_extract_slice] Saved slice to {out_path}")

    if cfg.plot:
        dens = slice_data.get("dens")
        if dens is not None:
            _plot_density(out_path, dens)


if __name__ == "__main__":
    main() 