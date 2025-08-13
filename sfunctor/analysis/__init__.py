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
time_series
    Time-series analysis across multiple snapshots
anisotropy
    Scale-dependent anisotropy analysis
cross_correlation
    Cross-correlation between different field quantities
wavelet
    Wavelet-based decomposition and multifractal analysis
"""

from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.analysis.batch import process_slices_batch

__all__ = [
    "analyze_slice",
    "process_slices_batch",
]

# Import new scientific modules if available
try:
    from sfunctor.analysis.time_series import (
        analyze_time_series,
        compute_anisotropy_measures,
        compute_energy_content,
        compute_scaling_exponents,
        analyze_evolution,
        detect_transition_times
    )
    __all__.extend([
        "analyze_time_series",
        "compute_anisotropy_measures",
        "compute_energy_content",
        "compute_scaling_exponents",
        "analyze_evolution",
        "detect_transition_times"
    ])
except ImportError:
    pass

try:
    from sfunctor.analysis.anisotropy import (
        compute_scale_dependent_anisotropy,
        create_anisotropy_spectrogram,
        decompose_anisotropy_modes
    )
    __all__.extend([
        "compute_scale_dependent_anisotropy",
        "create_anisotropy_spectrogram",
        "decompose_anisotropy_modes"
    ])
except ImportError:
    pass

try:
    from sfunctor.analysis.cross_correlation import (
        compute_field_correlations,
        compute_scale_dependent_transfer,
        compute_nonlinear_coupling,
        compute_conditional_statistics,
        compute_mutual_information
    )
    __all__.extend([
        "compute_field_correlations",
        "compute_scale_dependent_transfer",
        "compute_nonlinear_coupling",
        "compute_conditional_statistics",
        "compute_mutual_information"
    ])
except ImportError:
    pass

try:
    from sfunctor.analysis.wavelet import (
        wavelet_decompose_sf,
        continuous_wavelet_analysis,
        wavelet_coherence_analysis,
        multifractal_wavelet_analysis,
        identify_coherent_structures
    )
    __all__.extend([
        "wavelet_decompose_sf",
        "continuous_wavelet_analysis",
        "wavelet_coherence_analysis",
        "multifractal_wavelet_analysis",
        "identify_coherent_structures"
    ])
except ImportError:
    pass