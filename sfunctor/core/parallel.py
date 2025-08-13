"""Simplified parallel processing for structure function computation.

This module removes the shared memory complexity and uses simple chunked
processing instead. For GPU optimization, we need clean, simple code.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from sfunctor.core.histograms import (
    compute_histogram_for_disp_2D,
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
)


def compute_histograms_chunked(
    fields: Dict[str, np.ndarray],
    displacements: np.ndarray,
    axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_channel_bin_edges: Optional[List[np.ndarray]] = None,
    product_bin_edges: Optional[np.ndarray] = None,
    stencil_width: int = 2,
    chunk_size: Optional[int] = None,
    # Backwards compatibility parameters
    sf_bin_edges: Optional[np.ndarray] = None,
    n_processes: int = 1,  # Ignored - kept for compatibility
    **kwargs  # Catch any other legacy parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histograms for multiple displacements using chunked processing.
    
    This is a simplified version that removes shared memory complexity.
    For large datasets, process displacements in chunks to manage memory.
    
    Args:
        fields: Dictionary of field arrays
        displacements: Array of (dx, dy) displacement pairs
        axis: Slice axis (1, 2, or 3)
        N_random_subsamples: Number of random samples per displacement
        ell_bin_edges: Bin edges for displacement magnitude
        theta_bin_edges: Bin edges for theta angle
        phi_bin_edges: Bin edges for phi angle
        sf_channel_bin_edges: List of bin edges for each channel
        product_bin_edges: Bin edges for cross products
        stencil_width: Stencil width (2, 3, or 5)
        chunk_size: Number of displacements to process at once (None = all)
        sf_bin_edges: Legacy parameter - converted to sf_channel_bin_edges
        n_processes: Legacy parameter - ignored
    
    Returns:
        Tuple of (hist_mag, hist_other) accumulated histograms
    """
    # Handle backwards compatibility - convert sf_bin_edges to sf_channel_bin_edges
    if sf_channel_bin_edges is None and sf_bin_edges is not None:
        # Use same bins for all channels (backwards compatibility)
        sf_channel_bin_edges = [sf_bin_edges] * N_MAG_CHANNELS
    
    # Default product bin edges if not provided
    if product_bin_edges is None:
        product_bin_edges = np.logspace(-4, 2, 128)
    
    # Initialize output histograms
    n_ell_bins = len(ell_bin_edges) - 1
    n_theta_bins = len(theta_bin_edges) - 1
    n_phi_bins = len(phi_bin_edges) - 1
    n_sf_bins = len(sf_channel_bin_edges[0]) - 1
    n_product_bins = len(product_bin_edges) - 1
    
    hist_mag_total = np.zeros(
        (N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins),
        dtype=np.int64
    )
    hist_other_total = np.zeros(
        (N_OTHER_CHANNELS, n_ell_bins, n_product_bins),
        dtype=np.int64
    )
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = len(displacements)
    
    # Process displacements in chunks
    for i in range(0, len(displacements), chunk_size):
        chunk_end = min(i + chunk_size, len(displacements))
        
        for j in range(i, chunk_end):
            dx, dy = displacements[j]
            
            # Compute histogram for this displacement
            hist_mag, hist_other = compute_histogram_for_disp_2D(
                fields["v_x"], fields["v_y"], fields["v_z"],
                fields["B_x"], fields["B_y"], fields["B_z"],
                fields["rho"],
                fields["vA_x"], fields["vA_y"], fields["vA_z"],
                fields["zp_x"], fields["zp_y"], fields["zp_z"],
                fields["zm_x"], fields["zm_y"], fields["zm_z"],
                fields["omega_x"], fields["omega_y"], fields["omega_z"],
                fields["j_x"], fields["j_y"], fields["j_z"],
                fields["curv_x"], fields["curv_y"], fields["curv_z"],
                fields["grad_rho_x"], fields["grad_rho_y"], fields["grad_rho_z"],
                int(dx), int(dy), axis,
                N_random_subsamples,
                ell_bin_edges, theta_bin_edges, phi_bin_edges,
                sf_channel_bin_edges, product_bin_edges,
                stencil_width
            )
            
            # Accumulate results
            hist_mag_total += hist_mag
            hist_other_total += hist_other
    
    return hist_mag_total, hist_other_total


# Backwards compatibility alias
compute_histograms_shared = compute_histograms_chunked