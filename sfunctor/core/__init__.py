"""Core computational modules for structure function analysis.

This subpackage contains the performance-critical components of SFunctor,
including Numba-accelerated histogram computation, physics calculations,
and parallel processing utilities.

Modules
-------
histograms
    Numba-accelerated histogram building for structure functions
physics
    MHD physics calculations (Alfvén velocity, Elsasser variables)
parallel
    Shared-memory multiprocessing utilities
"""

from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.core.histograms import (
    Channel,
    N_CHANNELS,
    compute_histogram_for_disp_2D,
)
from sfunctor.core.parallel import compute_histograms_shared

__all__ = [
    # Physics
    "compute_vA",
    "compute_z_plus_minus",
    # Histograms
    "Channel",
    "N_CHANNELS",
    "compute_histogram_for_disp_2D",
    # Parallel
    "compute_histograms_shared",
]