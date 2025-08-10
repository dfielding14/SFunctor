#!/usr/bin/env python3
"""Run structure function analysis on a single node with a subset of displacements.

This script is designed to be run on each compute node independently.
It processes a fraction of the total displacements and saves partial results.
"""

import argparse
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
import sys

# Add sfunctor to path
sys.path.insert(0, str(Path(__file__).parent))

from sfunctor.io.slice_io import load_slice_npz
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.core.histograms import (
    compute_histogram_for_disp_2D,
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
    MAG_CHANNELS,
    OTHER_CHANNELS,
)
from sfunctor.core.parallel import compute_histograms_shared


def main():
    parser = argparse.ArgumentParser(description="Run node analysis")
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
    
    # Bin edge parameters for sf_bin_edges - now channel-specific
    parser.add_argument("--log_sf_bin_edges_min", type=float, nargs='+', 
                        default=[-5, -5, -5, -5, -5, -5, -2, -2, -2, -2, -8],
                        help="Log10 of minimum sf bin edge for each channel (11 values)")
    parser.add_argument("--log_sf_bin_edges_max", type=float, nargs='+',
                        default=[1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 8],
                        help="Log10 of maximum sf bin edge for each channel (11 values)")
    parser.add_argument("--N_sf_bin_edges", type=int, default=128,
                        help="Number of sf bins (default: 128)")
    
    # Bin edge parameters for product_bin_edges
    parser.add_argument("--log_product_bin_edges_min", type=float, default=-5,
                        help="Log10 of minimum product bin edge (default: -5)")
    parser.add_argument("--log_product_bin_edges_max", type=float, default=5,
                        help="Log10 of maximum product bin edge (default: 5)")
    parser.add_argument("--N_product_bin_edges", type=int, default=128,
                        help="Number of product bins (default: 128)")
    
    args = parser.parse_args()
    
    # Determine number of processes
    n_processes = args.n_processes if args.n_processes > 0 else max(1, cpu_count() - 2)
    
    # Load displacements and determine this node's subset
    disp_data = np.load(args.displacements)
    all_displacements = disp_data['displacements']
    ell_bin_edges = disp_data['ell_bin_edges']
    
    # Calculate displacement range for this node
    n_disp = len(all_displacements)
    start_idx = (args.node_id * n_disp) // args.total_nodes
    end_idx = ((args.node_id + 1) * n_disp) // args.total_nodes
    node_displacements = all_displacements[start_idx:end_idx]
    
    print(f"Node {args.node_id}/{args.total_nodes}: Processing {len(node_displacements)} displacements")
    print(f"Displacement indices: {start_idx} to {end_idx}")
    
    # Load slice data
    slice_path = Path(args.slice)
    slice_data = load_slice_npz(slice_path, stride=args.stride)
    
    # Extract slice metadata from filename
    # Expected format: *_axis{N}_*.npz
    filename = slice_path.name
    if '_axis' in filename:
        axis = int(filename.split('_axis')[1].split('_')[0])
    else:
        axis = 3  # default to z-axis
    
    # Compute derived fields
    rho = slice_data["rho"]
    B_x = slice_data["B_x"]; B_y = slice_data["B_y"]; B_z = slice_data["B_z"]
    v_x = slice_data["v_x"]; v_y = slice_data["v_y"]; v_z = slice_data["v_z"]
    
    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (z_plus_x, z_plus_y, z_plus_z), (z_minus_x, z_minus_y, z_minus_z) = compute_z_plus_minus(
        v_x, v_y, v_z, vA_x, vA_y, vA_z
    )
    
    # Prepare fields dictionary
    fields = {
        "v_x": v_x, "v_y": v_y, "v_z": v_z,
        "B_x": B_x, "B_y": B_y, "B_z": B_z,
        "rho": rho,
        "vA_x": vA_x, "vA_y": vA_y, "vA_z": vA_z,
        "zp_x": z_plus_x, "zp_y": z_plus_y, "zp_z": z_plus_z,
        "zm_x": z_minus_x, "zm_y": z_minus_y, "zm_z": z_minus_z,
        "omega_x": slice_data.get("omega_x", np.zeros_like(rho)),
        "omega_y": slice_data.get("omega_y", np.zeros_like(rho)),
        "omega_z": slice_data.get("omega_z", np.zeros_like(rho)),
        "j_x": slice_data.get("j_x", np.zeros_like(rho)),
        "j_y": slice_data.get("j_y", np.zeros_like(rho)),
        "j_z": slice_data.get("j_z", np.zeros_like(rho)),
        "curv_x": slice_data.get("curv_x", np.zeros_like(rho)),
        "curv_y": slice_data.get("curv_y", np.zeros_like(rho)),
        "curv_z": slice_data.get("curv_z", np.zeros_like(rho)),
        "grad_rho_x": slice_data.get("grad_rho_x", np.zeros_like(rho)),
        "grad_rho_y": slice_data.get("grad_rho_y", np.zeros_like(rho)),
        "grad_rho_z": slice_data.get("grad_rho_z", np.zeros_like(rho)),
    }
    
    # Validate bin edge arguments
    if len(args.log_sf_bin_edges_min) != N_MAG_CHANNELS:
        raise ValueError(f"Expected {N_MAG_CHANNELS} values for log_sf_bin_edges_min, got {len(args.log_sf_bin_edges_min)}")
    if len(args.log_sf_bin_edges_max) != N_MAG_CHANNELS:
        raise ValueError(f"Expected {N_MAG_CHANNELS} values for log_sf_bin_edges_max, got {len(args.log_sf_bin_edges_max)}")
    
    # Set up histogram bins
    n_theta_bins = 18
    theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)
    n_phi_bins = 18
    phi_bin_edges = np.linspace(0, np.pi, n_phi_bins + 1)
    
    # Create channel-specific bin edges
    sf_channel_bin_edges = []
    for i in range(N_MAG_CHANNELS):
        bin_edges = np.logspace(args.log_sf_bin_edges_min[i], args.log_sf_bin_edges_max[i], args.N_sf_bin_edges)
        sf_channel_bin_edges.append(bin_edges)
    
    product_bin_edges = np.logspace(args.log_product_bin_edges_min, args.log_product_bin_edges_max, args.N_product_bin_edges)
    
    # Process using shared memory implementation
    print(f"Processing with shared memory ({n_processes} processes)...")
    
    # Report expected memory usage
    total_field_memory = sum(f.nbytes for f in fields.values()) / 1024 / 1024
    print(f"  Field memory: {total_field_memory:.1f} MB (shared across processes)")
    print(f"  Previous approach would use: {total_field_memory * n_processes:.1f} MB")
    print(f"  Processing {len(node_displacements)} displacements")
    
    # Note: compute_histograms_shared expects a single sf_bin_edges array
    # For now, use the first channel's bin edges as representative
    # TODO: Modify compute_histograms_shared to handle channel-specific bins
    sf_bin_edges_single = sf_channel_bin_edges[0]
    
    # Use compute_histograms_shared for efficient memory usage
    hist_mag, hist_other = compute_histograms_shared(
        fields=fields,
        displacements=node_displacements,
        axis=axis,
        N_random_subsamples=args.N_random_subsamples,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges_single,
        product_bin_edges=product_bin_edges,
        stencil_width=args.stencil_width,
        n_processes=n_processes
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slice_name = slice_path.stem
    output_file = output_dir / f"histogram_{slice_name}_node{args.node_id:04d}.npz"
    
    np.savez_compressed(
        output_file,
        hist_mag=hist_mag,
        hist_other=hist_other,
        mag_channels=[ch.name for ch in MAG_CHANNELS],
        other_channels=[ch.name for ch in OTHER_CHANNELS],
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_channel_bin_edges=sf_channel_bin_edges,
        product_bin_edges=product_bin_edges,
        node_info={
            'node_id': args.node_id,
            'total_nodes': args.total_nodes,
            'displacement_start': start_idx,
            'displacement_end': end_idx,
            'n_displacements': len(node_displacements)
        },
        metadata={
            'slice': str(slice_path),
            'axis': axis,
            'stride': args.stride,
            'stencil_width': args.stencil_width,
            'N_random_subsamples': args.N_random_subsamples,
            'n_processes': n_processes,
            'log_sf_bin_edges_min': args.log_sf_bin_edges_min,
            'log_sf_bin_edges_max': args.log_sf_bin_edges_max,
            'N_sf_bin_edges': args.N_sf_bin_edges
        }
    )
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()