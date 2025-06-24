#!/usr/bin/env python3
"""Demo pipeline: slice extraction → structure-function histograms.

This script shows a minimal, *single-node* workflow using the unified drivers
``run_extract_slice.py`` and ``run_sf.py``.  It performs the following steps:

1. Extract a small set of 2-D slices from the example simulation located in
   ``data/``.
2. Build a temporary *slice-list* text file enumerating the generated slices.
3. Compute structure-function histograms for each slice with moderate (i.e.
   quick-to-run) settings.

Both stages call the drivers via *subprocess* so the behaviour is identical to
running the commands in a shell.  Feel free to adjust the CLI options below
(e.g., offsets, axes, stride) to suit your needs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Utility helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> None:
    """Run *cmd* (list of strings) with ``subprocess.run`` – exits on error."""
    print("\n≫", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main demo -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    # ------------------------------------------------------------------
    # Stage 1 – 2-D slice extraction -----------------------------------
    # ------------------------------------------------------------------
    sim_name = "Turb_320_beta100_dedt025_plm"  # matches the example data tree
    offsets = "-0.25,0.0,0.25"                 # three positions along each axis
    axes = "1,3"                               # extract slices normal to *x* and *z*

    _run(
        [
            sys.executable,
            "run_extract_slice.py",
            "--sim_name",
            sim_name,
            "--offsets",
            offsets,
            "--axes",
            axes,
            "--file_number",
            "0",           # use snapshot index 0 to keep runtime short
            "--plot",      # quick density PNG for visual verification (optional)
        ]
    )

    # ------------------------------------------------------------------
    # Stage 2 – Build slice-list file ----------------------------------
    # ------------------------------------------------------------------
    slice_dir = Path("slice_data")
    list_path = slice_dir / "demo_slices.txt"
    with list_path.open("w", encoding="utf-8") as fh:
        for path in sorted(slice_dir.glob(f"slice_*_{sim_name}_*.npz")):
            fh.write(str(path) + "\n")

    print(f"Recorded {list_path.stat().st_size} bytes → {list_path}")

    # ------------------------------------------------------------------
    # Stage 3 – Structure-function histograms ---------------------------
    # ------------------------------------------------------------------
    _run(
        [
            sys.executable,
            "run_sf.py",
            "--slice_list",
            str(list_path),
            "--stride",
            "2",          # down-sample grid by 2 to speed things up
            "--n_disp_total",
            "5000",       # modest number of displacement vectors
            "--N_random_subsamples",
            "500",        # per-displacement sampling for quick demo
        ]
    )

    print("\n✔ Demo finished – histograms written to slice_data/ ...")


if __name__ == "__main__":
    main() 