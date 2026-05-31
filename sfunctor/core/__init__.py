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
    CENSOR_NAMES,
    Channel,
    N_CHANNELS,
    compute_histogram_for_disp_2D,
)
from sfunctor.core.parallel import compute_histograms_shared
from sfunctor.core.directional import (
    DirectionalConfig,
    DirectionalResult,
    build_q_variants,
    compute_directional_structure_functions,
    slice_offset_to_vector,
)
from sfunctor.core.finite_domain import (
    FiniteDomainConfig,
    FiniteDomainResult,
    build_cube_q_variants,
    compute_finite_domain_structure_functions,
    cube_offset_to_vector,
    generate_fibonacci_displacements,
    nested_core_bounds_kji,
    stencil_definition,
    valid_origin_bounds_kji,
)

__all__ = [
    # Physics
    "compute_vA",
    "compute_z_plus_minus",
    # Histograms
    "Channel",
    "CENSOR_NAMES",
    "N_CHANNELS",
    "compute_histogram_for_disp_2D",
    # Parallel
    "compute_histograms_shared",
    # Strict pairwise three-direction diagnostics
    "DirectionalConfig",
    "DirectionalResult",
    "build_q_variants",
    "compute_directional_structure_functions",
    "slice_offset_to_vector",
    # Finite-domain 3-D conditional statistics
    "FiniteDomainConfig",
    "FiniteDomainResult",
    "build_cube_q_variants",
    "compute_finite_domain_structure_functions",
    "cube_offset_to_vector",
    "generate_fibonacci_displacements",
    "nested_core_bounds_kji",
    "stencil_definition",
    "valid_origin_bounds_kji",
]
