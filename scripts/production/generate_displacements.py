#!/usr/bin/env python3
"""Generate displacement vectors for structure function analysis.

This script generates all displacement vectors needed for the analysis
and saves them to a file. This allows all nodes to use the same set
of displacements for consistent results.
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add repository root to path so `import sfunctor` works when running script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sfunctor.utils.displacements import find_ell_bin_edges, build_displacement_list


def main():
    parser = argparse.ArgumentParser(description="Generate displacement vectors")
    parser.add_argument("--n_disp_total", type=int, default=5000,
                        help="Total number of displacement vectors")
    parser.add_argument("--n_ell_bins", type=int, default=64,
                        help="Number of displacement magnitude bins")
    parser.add_argument("--ell_min", type=float, default=1.0,
                        help="Minimum displacement magnitude")
    parser.add_argument("--Nres", type=int, default=1024,
                        help="Grid resolution")
    parser.add_argument("--stencil_width", type=int, default=2, choices=[2, 3, 5],
                        help="Stencil width for derivatives (2, 3, or 5)")
    parser.add_argument("--output", type=str, default="displacements.npz",
                        help="Output filename")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Calculate ell_max based on stencil width
    if args.stencil_width == 2:
        ell_max = args.Nres / 2
    elif args.stencil_width == 3:
        ell_max = args.Nres / 4
    elif args.stencil_width == 5:
        ell_max = args.Nres / 8
    else:
        raise ValueError(f"Invalid stencil_width: {args.stencil_width}. Must be 2, 3, or 5.")
    
    # Generate bin edges
    ell_bin_edges = find_ell_bin_edges(args.ell_min, ell_max, args.n_ell_bins)
    
    # Generate displacements
    displacements = build_displacement_list(ell_bin_edges, args.n_disp_total, seed=args.seed)
    
    # Save to file
    np.savez_compressed(
        args.output,
        displacements=displacements,
        ell_bin_edges=ell_bin_edges,
        config={
            'n_disp_total': args.n_disp_total,
            'n_ell_bins': args.n_ell_bins,
            'ell_min': args.ell_min,
            'ell_max': ell_max,
            'Nres': args.Nres,
            'stencil_width': args.stencil_width,
            'seed': args.seed
        }
    )
    
    print(f"Generated {len(displacements)} displacement vectors")
    print(f"Saved to {args.output}")
    print(f"Displacement range: ({displacements.min()}, {displacements.max()})")


if __name__ == "__main__":
    main()
