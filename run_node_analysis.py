#!/usr/bin/env python3
"""Run structure function analysis on a single node with a subset of displacements.

This script is designed to be run on each compute node independently.
It processes a fraction of the total displacements and saves partial results.
"""

import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
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


def process_displacement_batch(args):
    """Process a batch of displacements (for multiprocessing pool)."""
    (fields, displacements, axis, N_random_subsamples, 
     ell_bin_edges, theta_bin_edges, phi_bin_edges, 
     sf_bin_edges, product_bin_edges, stencil_width) = args
    
    # Initialize histograms
    hist_mag = np.zeros(
        (N_MAG_CHANNELS, len(ell_bin_edges)-1, len(theta_bin_edges)-1, 
         len(phi_bin_edges)-1, len(sf_bin_edges)-1),
        dtype=np.int64
    )
    hist_other = np.zeros(
        (N_OTHER_CHANNELS, len(ell_bin_edges)-1, len(product_bin_edges)-1),
        dtype=np.int64
    )
    
    # Process each displacement
    for dx, dy in displacements:
        hm, ho = compute_histogram_for_disp_2D(
            fields['v_x'], fields['v_y'], fields['v_z'],
            fields['B_x'], fields['B_y'], fields['B_z'],
            fields['rho'],
            fields['vA_x'], fields['vA_y'], fields['vA_z'],
            fields['zp_x'], fields['zp_y'], fields['zp_z'],
            fields['zm_x'], fields['zm_y'], fields['zm_z'],
            fields['omega_x'], fields['omega_y'], fields['omega_z'],
            fields['j_x'], fields['j_y'], fields['j_z'],
            fields['curv_x'], fields['curv_y'], fields['curv_z'],
            fields['grad_rho_x'], fields['grad_rho_y'], fields['grad_rho_z'],
            int(dx), int(dy), axis,
            N_random_subsamples,
            ell_bin_edges, theta_bin_edges, phi_bin_edges,
            sf_bin_edges, product_bin_edges,
            stencil_width
        )
        hist_mag += hm
        hist_other += ho
    
    return hist_mag, hist_other


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
    
    # Set up histogram bins
    n_theta_bins = 18
    theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)
    n_phi_bins = 18
    phi_bin_edges = np.linspace(0, np.pi, n_phi_bins + 1)
    sf_bin_edges = np.logspace(-4, 1, 128)
    product_bin_edges = np.logspace(-5, 5, 128)
    
    # Split displacements for multiprocessing
    if n_processes > 1:
        # Split displacements into batches
        batch_size = max(1, len(node_displacements) // n_processes)
        batches = []
        for i in range(0, len(node_displacements), batch_size):
            batch = node_displacements[i:i+batch_size]
            batches.append((
                fields, batch, axis, args.N_random_subsamples,
                ell_bin_edges, theta_bin_edges, phi_bin_edges,
                sf_bin_edges, product_bin_edges, args.stencil_width
            ))
        
        # Process in parallel
        print(f"Processing with {n_processes} processes...")
        with Pool(n_processes) as pool:
            results = pool.map(process_displacement_batch, batches)
        
        # Combine results
        hist_mag = sum(r[0] for r in results)
        hist_other = sum(r[1] for r in results)
    else:
        # Process serially
        print("Processing serially...")
        hist_mag, hist_other = process_displacement_batch((
            fields, node_displacements, axis, args.N_random_subsamples,
            ell_bin_edges, theta_bin_edges, phi_bin_edges,
            sf_bin_edges, product_bin_edges, args.stencil_width
        ))
    
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
        sf_bin_edges=sf_bin_edges,
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
            'n_processes': n_processes
        }
    )
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()