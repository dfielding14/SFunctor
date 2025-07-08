#!/usr/bin/env python3
"""Generate displacement vectors for structure function analysis.

This script generates all displacement vectors needed for the analysis
and saves them to a file. This allows all nodes to use the same set
of displacements for consistent results.
"""

import argparse
import numpy as np
from pathlib import Path


def find_ell_bin_edges(ell_min, ell_max, n_bins):
    """Create logarithmically spaced bin edges for displacement magnitudes."""
    return np.logspace(np.log10(ell_min), np.log10(ell_max), n_bins + 1)


def build_displacement_list(ell_bin_edges, n_disp_total):
    """Generate random displacement vectors distributed across magnitude bins."""
    n_bins = len(ell_bin_edges) - 1
    n_disp_per_bin = max(1, n_disp_total // n_bins)
    
    displacements = []
    
    for i in range(n_bins):
        ell_min = ell_bin_edges[i]
        ell_max = ell_bin_edges[i + 1]
        
        # Generate random displacements in this bin
        for _ in range(n_disp_per_bin):
            # Random magnitude in bin
            ell = np.random.uniform(ell_min, ell_max)
            # Random angle
            theta = np.random.uniform(0, 2 * np.pi)
            
            # Convert to Cartesian
            dx = int(np.round(ell * np.cos(theta)))
            dy = int(np.round(ell * np.sin(theta)))
            
            # Skip zero displacements
            if dx != 0 or dy != 0:
                displacements.append([dx, dy])
    
    return np.array(displacements, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Generate displacement vectors")
    parser.add_argument("--n_disp_total", type=int, default=5000,
                        help="Total number of displacement vectors")
    parser.add_argument("--n_ell_bins", type=int, default=64,
                        help="Number of displacement magnitude bins")
    parser.add_argument("--ell_min", type=float, default=1.0,
                        help="Minimum displacement magnitude")
    parser.add_argument("--ell_max", type=float, default=512.0,
                        help="Maximum displacement magnitude")
    parser.add_argument("--output", type=str, default="displacements.npz",
                        help="Output filename")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Generate bin edges
    ell_bin_edges = find_ell_bin_edges(args.ell_min, args.ell_max, args.n_ell_bins)
    
    # Generate displacements
    displacements = build_displacement_list(ell_bin_edges, args.n_disp_total)
    
    # Save to file
    np.savez_compressed(
        args.output,
        displacements=displacements,
        ell_bin_edges=ell_bin_edges,
        config={
            'n_disp_total': args.n_disp_total,
            'n_ell_bins': args.n_ell_bins,
            'ell_min': args.ell_min,
            'ell_max': args.ell_max,
            'seed': args.seed
        }
    )
    
    print(f"Generated {len(displacements)} displacement vectors")
    print(f"Saved to {args.output}")
    print(f"Displacement range: ({displacements.min()}, {displacements.max()})")


if __name__ == "__main__":
    main()