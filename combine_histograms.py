#!/usr/bin/env python3
"""Combine partial histograms from multiple nodes into final result.

This script loads all partial histogram files from different nodes
and sums them together to produce the final structure function result.
"""

import argparse
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Combine partial histograms")
    parser.add_argument("--pattern", type=str, required=True,
                        help="Glob pattern for histogram files (e.g., 'histogram_*_node*.npz')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: sf_results_TIMESTAMP.npz)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Find all matching histogram files
    histogram_files = sorted(glob(args.pattern))
    
    if not histogram_files:
        print(f"Error: No files found matching pattern '{args.pattern}'")
        return 1
    
    print(f"Found {len(histogram_files)} histogram files to combine")
    
    # Load first file to get structure
    first_data = np.load(histogram_files[0])
    hist_mag_total = first_data['hist_mag'].copy()
    hist_other_total = first_data['hist_other'].copy()
    
    # Get metadata from first file
    mag_channels = first_data['mag_channels']
    other_channels = first_data['other_channels']
    ell_bin_edges = first_data['ell_bin_edges']
    theta_bin_edges = first_data['theta_bin_edges']
    phi_bin_edges = first_data['phi_bin_edges']
    sf_bin_edges = first_data['sf_bin_edges']
    product_bin_edges = first_data['product_bin_edges']
    
    # Keep track of node info
    node_infos = [dict(first_data['node_info'].item())]
    total_displacements = node_infos[0]['n_displacements']
    
    if args.verbose:
        print(f"File 0: {Path(histogram_files[0]).name} - {node_infos[0]['n_displacements']} displacements")
    
    # Add histograms from remaining files
    for i, hist_file in enumerate(histogram_files[1:], 1):
        data = np.load(hist_file)
        hist_mag_total += data['hist_mag']
        hist_other_total += data['hist_other']
        
        node_info = dict(data['node_info'].item())
        node_infos.append(node_info)
        total_displacements += node_info['n_displacements']
        
        if args.verbose:
            print(f"File {i}: {Path(hist_file).name} - {node_info['n_displacements']} displacements")
    
    # Verify we have all nodes
    expected_nodes = max(info['total_nodes'] for info in node_infos)
    found_nodes = sorted(set(info['node_id'] for info in node_infos))
    
    print(f"\nTotal displacements processed: {total_displacements}")
    print(f"Expected nodes: {expected_nodes}")
    print(f"Found nodes: {len(found_nodes)} - IDs: {found_nodes}")
    
    if len(found_nodes) != expected_nodes:
        missing = set(range(expected_nodes)) - set(found_nodes)
        print(f"WARNING: Missing nodes: {sorted(missing)}")
    
    # Get metadata from first file
    metadata = dict(first_data['metadata'].item())
    metadata['n_nodes'] = len(found_nodes)
    metadata['total_displacements'] = total_displacements
    metadata['combined_from'] = len(histogram_files)
    
    # Determine output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sf_results_{timestamp}.npz"
    else:
        output_file = args.output
    
    # Save combined results
    np.savez_compressed(
        output_file,
        hist_mag=hist_mag_total,
        hist_other=hist_other_total,
        mag_channels=mag_channels,
        other_channels=other_channels,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges,
        product_bin_edges=product_bin_edges,
        metadata=metadata,
        node_infos=node_infos
    )
    
    print(f"\nCombined results saved to {output_file}")
    print(f"Total counts in histograms: {hist_mag_total.sum() + hist_other_total.sum()}")
    
    return 0


if __name__ == "__main__":
    exit(main())