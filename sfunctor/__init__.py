"""SFunctor: Structure Function Analysis for MHD Turbulence.

A high-performance Python package for computing anisotropic, angle-resolved
structure functions from 2D slices of 3D magnetohydrodynamic (MHD) simulations.

Subpackages
-----------
core
    Core computational modules (histograms, physics, parallel processing)
analysis
    Main analysis pipelines and structure function computation
io
    Input/output utilities for slice data and results
visualization
    Plotting and visualization tools
utils
    Utility functions (displacement generation, CLI parsing)

Examples
--------
Basic usage for single slice analysis:

    from sfunctor.analysis import analyze_slice
    from sfunctor.io import load_slice_npz
    
    # Load slice data
    data = load_slice_npz("slice_x1_0.0_beta25.npz")
    
    # Compute structure functions
    results = analyze_slice(data, n_displacements=1000)
    
    # Visualize results
    from sfunctor.visualization import plot_structure_functions
    plot_structure_functions(results)

For batch processing with MPI:

    mpirun -n 64 python -m sfunctor.analysis.batch --slice_list slices.txt
"""

__version__ = "0.1.0"
__author__ = "SFunctor Development Team"

# Import key functions for convenience
from sfunctor.io import load_slice_npz, parse_slice_metadata
from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.visualization import plot_structure_functions

__all__ = [
    "load_slice_npz",
    "parse_slice_metadata", 
    "analyze_slice",
    "plot_structure_functions",
]