#!/usr/bin/env python
"""
Simple script to extract a single 2D slice from 3D simulation data.
Called by SLURM script with command-line arguments.
"""

import sys
import os
import argparse

# Add SFunctor to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sfunctor.io.extract import extract_2d_slice

def main():
    parser = argparse.ArgumentParser(description="Extract a 2D slice from 3D data")
    parser.add_argument("--sim_name", required=True, help="Simulation name")
    parser.add_argument("--axis", type=int, required=True, help="Axis (1, 2, or 3)")
    parser.add_argument("--position", type=float, required=True, help="Slice position")
    parser.add_argument("--file_number", type=int, required=True, help="File number")
    
    args = parser.parse_args()
    
    print(f"Extracting: file {args.file_number}, axis {args.axis}, position {args.position}")
    
    try:
        # Set cache directory to new structure
        cache_dir = f"sfunctor_results/slice_{args.sim_name}"
        
        # Extract the slice
        slice_data = extract_2d_slice(
            sim_name=args.sim_name,
            axis=args.axis,
            slice_value=args.position,
            file_number=args.file_number,
            save=True,
            cache_dir=cache_dir
        )
        
        print(f"Successfully extracted slice with shape {slice_data['dens'].shape}")
        print(f"Saved to: {cache_dir}")
        
    except Exception as e:
        print(f"Error extracting slice: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()