"""Simplified batch processing for structure function analysis.

Removes fake MPI classes - just use simple conditionals.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    HAS_MPI = False
    rank = 0
    size = 1

from sfunctor.utils.cli import parse_cli
from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.utils.displacements import find_ell_bin_edges, build_displacement_list
from sfunctor.core.histograms import (
    MAG_CHANNELS,
    OTHER_CHANNELS,
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
    compute_histogram_for_disp_2D,
)


def process_slices_batch(
    slice_files: List[Union[str, Path]],
    config: Optional[Dict] = None,
    output_file: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Process multiple slice files in parallel using MPI if available.
    
    Simplified version without fake MPI classes - just use conditionals.
    
    Args:
        slice_files: List of slice file paths
        config: Configuration dictionary (or use defaults)
        output_file: Output file path for results
        
    Returns:
        Dictionary with aggregated histograms and metadata
    """
    if config is None:
        config = parse_cli([])  # Get defaults
    
    # Distribute files across MPI ranks (if available)
    if HAS_MPI and size > 1:
        # Simple round-robin distribution
        my_files = [f for i, f in enumerate(slice_files) if i % size == rank]
    else:
        my_files = slice_files
    
    if rank == 0:
        print(f"Processing {len(slice_files)} files across {size} ranks")
    
    # Initialize bins
    axis, beta = parse_slice_metadata(slice_files[0])
    ell_bin_edges = find_ell_bin_edges(
        config["n_disp_per_decade"],
        config["ell_min"],
        config["ell_max"]
    )
    theta_bin_edges = np.linspace(0, np.pi, config["n_theta_bins"] + 1)
    phi_bin_edges = np.linspace(-np.pi, np.pi, config["n_phi_bins"] + 1)
    sf_bin_edges = np.logspace(-4, 1, 128)
    product_bin_edges = np.logspace(-4, 2, 128)
    
    # Initialize histograms
    n_ell = len(ell_bin_edges) - 1
    n_theta = len(theta_bin_edges) - 1
    n_phi = len(phi_bin_edges) - 1
    n_sf = len(sf_bin_edges) - 1
    n_prod = len(product_bin_edges) - 1
    
    hist_mag = np.zeros((N_MAG_CHANNELS, n_ell, n_theta, n_phi, n_sf), dtype=np.int64)
    hist_other = np.zeros((N_OTHER_CHANNELS, n_ell, n_prod), dtype=np.int64)
    
    # Process files
    for file_idx, file_path in enumerate(my_files):
        if rank == 0:
            print(f"Rank {rank}: Processing file {file_idx + 1}/{len(my_files)}")
        
        # Load slice
        fields = load_slice_npz(file_path, stride=config["stride"])
        
        # Compute derived fields
        fields["vA_x"], fields["vA_y"], fields["vA_z"] = compute_vA(
            fields["B_x"], fields["B_y"], fields["B_z"], fields["rho"]
        )
        fields["zp_x"], fields["zp_y"], fields["zp_z"], \
        fields["zm_x"], fields["zm_y"], fields["zm_z"] = compute_z_plus_minus(
            fields["v_x"], fields["v_y"], fields["v_z"],
            fields["vA_x"], fields["vA_y"], fields["vA_z"]
        )
        
        # Generate displacements
        displacements = build_displacement_list(
            ell_bin_edges,
            config["n_disp_per_bin"],
            config["n_disp_total"]
        )
        
        # Compute histograms for each displacement
        for dx, dy in displacements:
            h_mag, h_other = compute_histogram_for_disp_2D(
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
                config["N_random_subsamples"],
                ell_bin_edges, theta_bin_edges, phi_bin_edges,
                [sf_bin_edges] * N_MAG_CHANNELS, product_bin_edges,
                config["stencil_width"]
            )
            hist_mag += h_mag
            hist_other += h_other
    
    # Reduce across MPI ranks if available
    if HAS_MPI and size > 1:
        # Sum histograms across all ranks
        hist_mag_total = np.zeros_like(hist_mag)
        hist_other_total = np.zeros_like(hist_other)
        comm.Reduce(hist_mag, hist_mag_total, op=MPI.SUM, root=0)
        comm.Reduce(hist_other, hist_other_total, op=MPI.SUM, root=0)
        hist_mag = hist_mag_total
        hist_other = hist_other_total
    
    # Save results (rank 0 only)
    if rank == 0 and output_file:
        np.savez_compressed(
            output_file,
            hist_mag=hist_mag,
            hist_other=hist_other,
            ell_bin_edges=ell_bin_edges,
            theta_bin_edges=theta_bin_edges,
            phi_bin_edges=phi_bin_edges,
            sf_bin_edges=sf_bin_edges,
            product_bin_edges=product_bin_edges,
            mag_channels=[ch.name for ch in MAG_CHANNELS],
            other_channels=[ch.name for ch in OTHER_CHANNELS],
            n_files=len(slice_files),
            beta=beta,
            axis=axis,
        )
        print(f"Results saved to {output_file}")
    
    return {
        "hist_mag": hist_mag,
        "hist_other": hist_other,
        "ell_bin_edges": ell_bin_edges,
        "theta_bin_edges": theta_bin_edges,
        "phi_bin_edges": phi_bin_edges,
        "sf_bin_edges": sf_bin_edges,
        "product_bin_edges": product_bin_edges,
    }