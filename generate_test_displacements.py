#!/usr/bin/env python3
"""
Simple displacement generation for testing without sfunctor dependencies.
"""

import numpy as np
import argparse


def find_ell_bin_edges(ell_min, ell_max, n_ell_bins):
    """Create logarithmically spaced bin edges."""
    return np.logspace(np.log10(ell_min), np.log10(ell_max), n_ell_bins + 1)


def build_displacement_list(ell_bin_edges, n_disp_total, nres):
    """Build list of displacement vectors."""
    n_ell_bins = len(ell_bin_edges) - 1
    n_displ_per_bin = n_disp_total // n_ell_bins
    
    displacements = []
    
    for i in range(n_ell_bins):
        ell_min = ell_bin_edges[i]
        ell_max = ell_bin_edges[i + 1]
        
        # Generate random displacements in this magnitude range
        for _ in range(n_displ_per_bin):
            # Random magnitude in [ell_min, ell_max]
            ell = np.random.uniform(ell_min, ell_max)
            # Random angle
            theta = np.random.uniform(0, 2 * np.pi)
            
            # Convert to integer pixel displacements
            dx = int(ell * np.cos(theta))
            dy = int(ell * np.sin(theta))
            
            # Ensure within bounds
            if abs(dx) < nres//2 and abs(dy) < nres//2:
                displacements.append([dx, dy])
    
    return np.array(displacements, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Generate test displacements")
    parser.add_argument("--n_disp_total", type=int, default=10000,
                        help="Total number of displacements")
    parser.add_argument("--n_ell_bins", type=int, default=32,
                        help="Number of magnitude bins")
    parser.add_argument("--nres", type=int, default=10240,
                        help="Resolution of the grid")
    parser.add_argument("--output", type=str, default="test_displacements.npz",
                        help="Output file path")
    args = parser.parse_args()
    
    # Set up bin edges
    ell_min = 1
    ell_max = args.nres // 4  # Maximum displacement is 1/4 of grid size
    ell_bin_edges = find_ell_bin_edges(ell_min, ell_max, args.n_ell_bins)
    
    # Generate displacements
    np.random.seed(42)  # For reproducibility
    displacements = build_displacement_list(ell_bin_edges, args.n_disp_total, args.nres)
    
    print(f"Generated {len(displacements)} displacements")
    print(f"Displacement range: [{np.min(displacements)}, {np.max(displacements)}]")
    print(f"Saving to {args.output}")
    
    # Save
    np.savez(args.output,
             displacements=displacements,
             ell_bin_edges=ell_bin_edges,
             n_disp_total=args.n_disp_total,
             nres=args.nres)
    
    print("Done!")


if __name__ == "__main__":
    main()