"""Single slice analysis functionality.

This module provides a high-level interface for analyzing individual
2D slices without the complexity of MPI or batch processing.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np

from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.utils.displacements import find_ell_bin_edges, build_displacement_list
from sfunctor.core.parallel import compute_histograms_shared
from sfunctor.core.histograms import MAG_CHANNELS, OTHER_CHANNELS


def analyze_slice(
    slice_data: Dict[str, np.ndarray],
    *,
    n_displacements: int = 1000,
    n_ell_bins: int = 128,
    n_random_subsamples: int = 1000,
    stencil_width: int = 2,
    n_processes: Optional[int] = None,
    axis: int = 1,
) -> Dict[str, np.ndarray]:
    """Analyze a single 2D slice to compute structure functions.
    
    This is a high-level convenience function that performs the complete
    analysis pipeline on a single slice.
    
    Parameters
    ----------
    slice_data : dict[str, np.ndarray]
        Dictionary containing field arrays. Required keys:
        'rho', 'v_x', 'v_y', 'v_z', 'B_x', 'B_y', 'B_z'.
        Optional: 'omega_x/y/z', 'j_x/y/z' (will be filled with NaN if missing).
    n_displacements : int, optional
        Total number of displacement vectors to generate. Default 1000.
    n_ell_bins : int, optional
        Number of logarithmic displacement magnitude bins. Default 128.
    n_random_subsamples : int, optional
        Number of random positions per displacement. Default 1000.
    stencil_width : int, optional
        Finite difference stencil width (2, 3, or 5). Default 2.
    n_processes : int, optional
        Number of worker processes. None means auto-detect.
    axis : int, optional
        Slice orientation (1, 2, or 3). Default 1.
    
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - 'hist_mag': Magnitude histogram array
        - 'hist_other': Cross-product histogram array
        - 'ell_bin_edges': Displacement magnitude bin edges
        - 'theta_bin_edges': Theta angle bin edges
        - 'phi_bin_edges': Phi angle bin edges
        - 'sf_bin_edges': Structure function value bin edges
        - 'product_bin_edges': Cross-product value bin edges
        - 'displacements': Array of displacement vectors used
        - 'mag_channels': List of magnitude channel names
        - 'other_channels': List of cross-product channel names
    
    Examples
    --------
    >>> from sfunctor.io import load_slice_npz
    >>> data = load_slice_npz("slice.npz")
    >>> results = analyze_slice(data, n_displacements=5000)
    >>> print(results['hist_mag'].shape)
    """
    # Extract fields
    rho = slice_data["rho"]
    v_x = slice_data["v_x"]
    v_y = slice_data["v_y"]
    v_z = slice_data["v_z"]
    B_x = slice_data["B_x"]
    B_y = slice_data["B_y"]
    B_z = slice_data["B_z"]
    
    # Compute derived fields
    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (zp_x, zp_y, zp_z), (zm_x, zm_y, zm_z) = compute_z_plus_minus(
        v_x, v_y, v_z, vA_x, vA_y, vA_z
    )
    
    # Generate displacements
    N_res = rho.shape[0]
    if stencil_width == 2:
        ell_max = N_res // 2
    elif stencil_width == 3:
        ell_max = N_res // 4
    else:
        ell_max = N_res // 8  # 5-point stencil
    
    ell_bin_edges = find_ell_bin_edges(1.0, ell_max, n_ell_bins)
    displacements = build_displacement_list(ell_bin_edges, n_displacements)
    
    # Set up bins
    n_theta_bins = 18
    theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)
    n_phi_bins = 18
    phi_bin_edges = np.linspace(0, np.pi, n_phi_bins + 1)
    sf_bin_edges = np.logspace(-4, 1, 128)
    product_bin_edges = np.logspace(-5, 5, 128)
    
    # Prepare fields dictionary
    fields = {
        "v_x": v_x, "v_y": v_y, "v_z": v_z,
        "B_x": B_x, "B_y": B_y, "B_z": B_z,
        "rho": rho,
        "vA_x": vA_x, "vA_y": vA_y, "vA_z": vA_z,
        "zp_x": zp_x, "zp_y": zp_y, "zp_z": zp_z,
        "zm_x": zm_x, "zm_y": zm_y, "zm_z": zm_z,
        "omega_x": slice_data.get("omega_x", np.full_like(rho, np.nan)),
        "omega_y": slice_data.get("omega_y", np.full_like(rho, np.nan)),
        "omega_z": slice_data.get("omega_z", np.full_like(rho, np.nan)),
        "j_x": slice_data.get("j_x", np.full_like(rho, np.nan)),
        "j_y": slice_data.get("j_y", np.full_like(rho, np.nan)),
        "j_z": slice_data.get("j_z", np.full_like(rho, np.nan)),
        "curv_x": slice_data.get("curv_x", np.full_like(rho, np.nan)),
        "curv_y": slice_data.get("curv_y", np.full_like(rho, np.nan)),
        "curv_z": slice_data.get("curv_z", np.full_like(rho, np.nan)),
        "grad_rho_x": slice_data.get("grad_rho_x", np.full_like(rho, np.nan)),
        "grad_rho_y": slice_data.get("grad_rho_y", np.full_like(rho, np.nan)),
        "grad_rho_z": slice_data.get("grad_rho_z", np.full_like(rho, np.nan)),
    }
    
    # Compute histograms
    hist_mag, hist_other = compute_histograms_shared(
        fields,
        displacements,
        axis=axis,
        N_random_subsamples=n_random_subsamples,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges,
        product_bin_edges=product_bin_edges,
        stencil_width=stencil_width,
        n_processes=n_processes,
    )
    
    # Return results dictionary
    return {
        "hist_mag": hist_mag,
        "hist_other": hist_other,
        "ell_bin_edges": ell_bin_edges,
        "theta_bin_edges": theta_bin_edges,
        "phi_bin_edges": phi_bin_edges,
        "sf_bin_edges": sf_bin_edges,
        "product_bin_edges": product_bin_edges,
        "displacements": displacements,
        "mag_channels": [ch.name for ch in MAG_CHANNELS],
        "other_channels": [ch.name for ch in OTHER_CHANNELS],
    }