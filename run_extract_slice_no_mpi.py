#!/usr/bin/env python3
"""Non-MPI version for extracting 2-D slices from a 3-D Athena-K run.

This is a simplified version of run_extract_slice.py that works without MPI
for testing and development purposes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

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
        description="Extract 2-D slices from a 3-D AthenaK simulation (no MPI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sim_name",
        default="Turb_320_beta100_dedt025_plm",
        help="Base name of the AthenaK simulation (matches directory names).",
    )
    parser.add_argument(
        "--offsets",
        type=_float_list,
        default=(-0.25, 0.0, 0.25),
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
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(density, origin="lower", cmap="viridis")
        ax.set_title(f"Density slice: {slice_path.stem}")
        plt.colorbar(im)
        
        png_path = slice_path.with_suffix(".png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[extract] Saved plot: {png_path}")
    except ImportError:
        print("[extract] matplotlib not available; skipping plot")
    except Exception as e:
        print(f"[extract] Plot failed: {e}")


def main() -> None:
    cfg = parse_cli()
    
    print(f"[extract] Starting extraction (no MPI)")
    print(f"[extract] Simulation: {cfg.sim_name}")
    print(f"[extract] Axes: {cfg.axes}")
    print(f"[extract] Offsets: {cfg.slice_offsets}")

    # Generate all (axis, offset) combinations
    tasks = [(axis, offset) for axis in cfg.axes for offset in cfg.slice_offsets]
    
    print(f"[extract] Total slices to extract: {len(tasks)}")

    for i, (axis, offset) in enumerate(tasks):
        print(f"[extract] Processing slice {i+1}/{len(tasks)}: axis={axis}, offset={offset}")
        
        try:
            # Extract the slice
            slice_data = extract_2d_slice(
                cfg.sim_name,
                axis,
                offset,
                file_number=cfg.file_number,
                save=False,  # We'll save manually with our naming convention
                cache_dir=str(cfg.cache_dir)
            )
            
            # Save with standard naming convention
            slice_path = _save_slice_npz(
                slice_data,
                axis,
                offset,
                cfg.sim_name,
                cfg.file_number,
                cfg.cache_dir,
            )
            
            print(f"[extract] Saved: {slice_path}")
            
            # Optional plotting
            if cfg.plot:
                _plot_density(slice_path, slice_data["dens"])
                
        except Exception as e:
            print(f"[extract] Failed to extract axis={axis}, offset={offset}: {e}")
            continue

    print(f"[extract] Extraction complete!")


if __name__ == "__main__":
    main() 