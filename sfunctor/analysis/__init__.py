"""Analysis pipelines for structure function computation.

This subpackage provides the main entry points for running structure
function analysis on MHD simulation data.

Modules
-------
single_slice
    Analysis pipeline for single slice processing
batch
    MPI-enabled batch processing for multiple slices
simple
    Simplified analysis without Numba (for testing/education)
"""

from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.analysis.batch import main as batch_analyze

__all__ = [
    "analyze_slice",
    "batch_analyze",
]