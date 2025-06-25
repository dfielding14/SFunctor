#!/usr/bin/env python3
"""Main entry point for SFunctor analysis.

This script provides backward compatibility with the old run_sf.py interface
while using the new package structure.

Usage:
    python run_analysis.py --file_name slice.npz --stride 2
    mpirun -n 64 python run_analysis.py --slice_list slices.txt
"""

import sys
from sfunctor.analysis.batch import main

if __name__ == "__main__":
    main()