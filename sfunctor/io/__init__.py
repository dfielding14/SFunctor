"""Input/output utilities for SFunctor.

This subpackage handles data loading, saving, and format conversion
for slice data and analysis results.

Modules
-------
slice_io
    Loading and parsing 2D slice data files
extract
    Extracting 2D slices from 3D simulation data
results
    Saving and loading structure function results
"""

from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata

__all__ = [
    "load_slice_npz",
    "parse_slice_metadata",
]