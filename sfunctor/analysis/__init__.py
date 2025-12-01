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


def batch_analyze(*args, **kwargs):
    """Lazy import wrapper for MPI batch analysis to avoid mpi4py at import time."""
    from sfunctor.analysis import batch as _batch  # local import to defer mpi4py load
    return _batch.main(*args, **kwargs)


__all__ = [
    "analyze_slice",
    "batch_analyze",
]
