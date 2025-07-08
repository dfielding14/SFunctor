#!/usr/bin/env python3
"""Combine partial histograms from multiple nodes into final result.

This script has two modes:
1. Node mode: Combines partial histogram files from different nodes for a single slice
2. Slice mode: Merges sf_results files from different slices into a combined result
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
    parser.add_argument("--mode", type=str, default="node", choices=["node", "slice"],
                        help="Mode: 'node' combines node histograms, 'slice' merges slice results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Find all matching files
    input_files = sorted(glob(args.pattern))
    
    if not input_files:
        print(f"Error: No files found matching pattern '{args.pattern}'")
        return 1
    
    if args.mode == "node":
        print(f"Found {len(input_files)} histogram files to combine")
        return combine_node_histograms(input_files, args)
    else:  # mode == "slice"
        print(f"Found {len(input_files)} slice result files to merge")
        return merge_slice_results(input_files, args)


def combine_node_histograms(histogram_files, args):
    """Combine histograms from different nodes for a single slice."""
    # Load first file to get structure
    first_data = np.load(histogram_files[0], allow_pickle=True)
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
        data = np.load(hist_file, allow_pickle=True)
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


def merge_slice_results(slice_files, args):
    """Merge sf_results from different slices into a combined result."""
    # Load first file to get structure
    first_data = np.load(slice_files[0], allow_pickle=True)
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
    
    # Keep track of slice info
    slice_names = [Path(slice_files[0]).stem]
    total_displacements = 0
    all_metadata = []
    
    # Get displacement count from first file
    if 'metadata' in first_data:
        first_metadata = dict(first_data['metadata'].item())
        total_displacements += first_metadata.get('total_displacements', 0)
        all_metadata.append(first_metadata)
    
    if args.verbose:
        print(f"Slice 0: {slice_names[0]}")
    
    # Add histograms from remaining files
    for i, slice_file in enumerate(slice_files[1:], 1):
        data = np.load(slice_file, allow_pickle=True)
        hist_mag_total += data['hist_mag']
        hist_other_total += data['hist_other']
        
        slice_name = Path(slice_file).stem
        slice_names.append(slice_name)
        
        if 'metadata' in data:
            metadata = dict(data['metadata'].item())
            total_displacements += metadata.get('total_displacements', 0)
            all_metadata.append(metadata)
        
        if args.verbose:
            print(f"Slice {i}: {slice_name}")
    
    print(f"\nTotal slices merged: {len(slice_files)}")
    print(f"Total displacements across all slices: {total_displacements}")
    
    # Create combined metadata
    combined_metadata = {
        'n_slices': len(slice_files),
        'slice_names': slice_names,
        'total_displacements': total_displacements,
        'merged_from': len(slice_files),
        'mode': 'slice_merge'
    }
    
    # Include common metadata from first file
    if all_metadata:
        for key in ['stride', 'N_random_subsamples', 'stencil_width', 'n_ell_bins']:
            if key in all_metadata[0]:
                combined_metadata[key] = all_metadata[0][key]
    
    # Determine output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sf_results_all_slices_{timestamp}.npz"
    else:
        output_file = args.output
    
    # Save merged results
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
        metadata=combined_metadata,
        slice_metadata=all_metadata  # Keep individual slice metadata
    )
    
    print(f"\nMerged results saved to {output_file}")
    print(f"Total counts in histograms: {hist_mag_total.sum() + hist_other_total.sum()}")
    
    # List slice names for confirmation
    if args.verbose:
        print("\nSlices included:")
        for name in slice_names:
            print(f"  - {name}")
    
    return 0


if __name__ == "__main__":
    exit(main())