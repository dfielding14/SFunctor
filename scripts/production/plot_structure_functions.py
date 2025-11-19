#!/usr/bin/env python3
"""Compatibility wrapper for the plotting CLI.

Production job scripts expect ``plot_structure_functions.py`` to live in the
repository root.  The real implementation has moved to
``plotting_scripts/plot_structure_functions.py`` but we keep this thin wrapper
so existing SLURM jobs continue to work without modifications.
"""
from __future__ import annotations

from plotting_scripts.plot_structure_functions import main


if __name__ == "__main__":
    main()
