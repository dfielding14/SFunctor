#!/usr/bin/env python3
"""
Minimal integration of shared memory implementation for node analysis.

This script demonstrates how to replace the Pool-based approach with
compute_histograms_shared to avoid memory duplication.
"""

import argparse
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
import sys
import time

# Add sfunctor to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sfunctor.core.parallel import compute_histograms_shared
from sf_physics import (
    compute_vA, compute_z_plus_minus, compute_vorticity,
    compute_current, compute_magnetic_curvature, compute_density_gradient
)
from sf_histograms import N_MAG_CHANNELS, N_OTHER_CHANNELS


def load_and_prepare_fields(slice_path, stride=1):
    """Load slice data and compute derived fields."""
    print(f"Loading slice: {slice_path}")
    slice_data = np.load(slice_path)
    
    # Extract basic fields with stride
    rho = slice_data['rho'][::stride, ::stride].astype(np.float32)
    B_x = slice_data['B_x'][::stride, ::stride].astype(np.float32)
    B_y = slice_data['B_y'][::stride, ::stride].astype(np.float32)
    B_z = slice_data['B_z'][::stride, ::stride].astype(np.float32)
    v_x = slice_data['v_x'][::stride, ::stride].astype(np.float32)
    v_y = slice_data['v_y'][::stride, ::stride].astype(np.float32)
    v_z = slice_data['v_z'][::stride, ::stride].astype(np.float32)
    
    print(f"  Slice shape after stride: {rho.shape}")
    
    # Get axis from metadata
    axis = slice_data.get('axis', 3)
    
    # Compute derived fields
    print("Computing derived fields...")
    
    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (zp_x, zp_y, zp_z), (zm_x, zm_y, zm_z) = compute_z_plus_minus(
        v_x, v_y, v_z, vA_x, vA_y, vA_z
    )
    
    omega_x, omega_y, omega_z = compute_vorticity(v_x, v_y, v_z, axis)
    j_x, j_y, j_z = compute_current(B_x, B_y, B_z, axis)
    curv_x, curv_y, curv_z = compute_magnetic_curvature(B_x, B_y, B_z, axis)
    grad_rho_x, grad_rho_y, grad_rho_z = compute_density_gradient(rho, axis)
    
    # Package all fields
    fields = {
        'v_x': v_x, 'v_y': v_y, 'v_z': v_z,
        'B_x': B_x, 'B_y': B_y, 'B_z': B_z,
        'rho': rho,
        'vA_x': vA_x, 'vA_y': vA_y, 'vA_z': vA_z,
        'zp_x': zp_x, 'zp_y': zp_y, 'zp_z': zp_z,
        'zm_x': zm_x, 'zm_y': zm_y, 'zm_z': zm_z,
        'omega_x': omega_x, 'omega_y': omega_y, 'omega_z': omega_z,
        'j_x': j_x, 'j_y': j_y, 'j_z': j_z,
        'curv_x': curv_x, 'curv_y': curv_y, 'curv_z': curv_z,
        'grad_rho_x': grad_rho_x, 'grad_rho_y': grad_rho_y, 'grad_rho_z': grad_rho_z,
    }
    
    # Calculate total memory usage
    total_size = sum(arr.nbytes for arr in fields.values())
    print(f"  Total field memory: {total_size / 1024 / 1024:.1f} MB")
    
    return fields, axis


def main():
    parser = argparse.ArgumentParser(description="Run node analysis with shared memory")
    parser.add_argument("--slice", type=str, required=True,
                        help="Path to slice NPZ file")
    parser.add_argument("--displacements", type=str, required=True,
                        help="Path to displacements NPZ file")
    parser.add_argument("--node_id", type=int, required=True,
                        help="Node ID (0-based)")
    parser.add_argument("--total_nodes", type=int, required=True,
                        help="Total number of nodes")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for loading slice data")
    parser.add_argument("--N_random_subsamples", type=int, default=2000,
                        help="Random samples per displacement")
    parser.add_argument("--stencil_width", type=int, default=2,
                        help="Finite difference stencil width")
    parser.add_argument("--n_processes", type=int, default=0,
                        help="Number of processes (0=auto)")
    
    # Bin edge parameters - channel-specific for sf_bin_edges
    parser.add_argument("--log_sf_bin_edges_min", type=float, nargs='+', 
                        default=[-5, -5, -5, -5, -5, -5, -2, -2, -2, -2, -8],
                        help="Log10 of minimum sf bin edge for each channel")
    parser.add_argument("--log_sf_bin_edges_max", type=float, nargs='+',
                        default=[1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 8],
                        help="Log10 of maximum sf bin edge for each channel")
    parser.add_argument("--N_sf_bin_edges", type=int, default=128,
                        help="Number of sf bins")
    
    # Bin edge parameters for product_bin_edges
    parser.add_argument("--log_product_bin_edges_min", type=float, default=-5,
                        help="Log10 of minimum product bin edge")
    parser.add_argument("--log_product_bin_edges_max", type=float, default=5,
                        help="Log10 of maximum product bin edge")
    parser.add_argument("--N_product_bin_edges", type=int, default=128,
                        help="Number of product bins")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Running Node Analysis with Shared Memory Implementation")
    print("=" * 70)
    
    # Determine number of processes
    n_processes = args.n_processes if args.n_processes > 0 else max(1, cpu_count() - 2)
    print(f"Using {n_processes} processes")
    
    # Load displacements and determine this node's subset
    print(f"\nLoading displacements from {args.displacements}")
    disp_data = np.load(args.displacements)
    all_displacements = disp_data['displacements']
    
    # Split displacements across nodes
    n_disp = len(all_displacements)
    disp_per_node = n_disp // args.total_nodes
    remainder = n_disp % args.total_nodes
    
    start_idx = args.node_id * disp_per_node + min(args.node_id, remainder)
    end_idx = start_idx + disp_per_node + (1 if args.node_id < remainder else 0)
    
    node_displacements = all_displacements[start_idx:end_idx]
    print(f"Node {args.node_id}: Processing {len(node_displacements)} displacements")
    print(f"  (indices {start_idx} to {end_idx} of {n_disp} total)")
    
    # Load and prepare fields
    fields, axis = load_and_prepare_fields(args.slice, args.stride)
    
    # Set up bin edges
    print("\nSetting up bin edges...")
    ell_bin_edges = disp_data['ell_bin_edges']
    n_theta_bins = 18
    theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)
    n_phi_bins = 18
    phi_bin_edges = np.linspace(0, np.pi, n_phi_bins + 1)
    
    # Create channel-specific bin edges (for compute_histograms_shared we need a single array)
    # Use the first channel's bin edges as they should be representative
    sf_bin_edges = np.logspace(
        args.log_sf_bin_edges_min[0], 
        args.log_sf_bin_edges_max[0], 
        args.N_sf_bin_edges
    )
    
    product_bin_edges = np.logspace(
        args.log_product_bin_edges_min, 
        args.log_product_bin_edges_max, 
        args.N_product_bin_edges
    )
    
    # === KEY CHANGE: Use compute_histograms_shared instead of Pool ===
    print(f"\nProcessing with shared memory implementation...")
    print(f"  Expected memory usage: ~{sum(f.nbytes for f in fields.values()) / 1024 / 1024:.0f} MB total")
    print(f"  (vs ~{sum(f.nbytes for f in fields.values()) * n_processes / 1024 / 1024:.0f} MB with Pool)")
    
    start_time = time.time()
    
    # Call compute_histograms_shared directly with ALL node displacements
    hist_mag, hist_other = compute_histograms_shared(
        fields=fields,
        displacements=node_displacements,
        axis=axis,
        N_random_subsamples=args.N_random_subsamples,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges,
        product_bin_edges=product_bin_edges,
        stencil_width=args.stencil_width,
        n_processes=n_processes
    )
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slice_name = Path(args.slice).stem
    output_file = output_dir / f"histogram_{slice_name}_node{args.node_id:04d}.npz"
    
    print(f"\nSaving results to {output_file}")
    
    # Note: For full compatibility, we'd need to save channel-specific bin edges
    # but this minimal example uses single bin edges for simplicity
    np.savez_compressed(
        output_file,
        hist_mag=hist_mag,
        hist_other=hist_other,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges,
        product_bin_edges=product_bin_edges,
        node_id=args.node_id,
        total_nodes=args.total_nodes,
        n_displacements=len(node_displacements),
        axis=axis
    )
    
    print(f"\n✓ Successfully processed node {args.node_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()