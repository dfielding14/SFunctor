#!/usr/bin/env python3
"""Fast parallel version of combine_histograms.py

This optimized version uses parallel file loading and efficient array operations
to speed up combining large numbers of histogram files.
"""

import argparse
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os
import sys

# Try to import tqdm for progress bars, but make it optional
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


def load_histogram_data(hist_file):
    """Load histogram data from a single file (for parallel loading)."""
    try:
        data = np.load(hist_file, allow_pickle=True)
        # Return only the essential data needed for combining
        return {
            'hist_mag': data['hist_mag'],
            'hist_other': data['hist_other'],
            'node_info': dict(data['node_info'].item()),
            'file': hist_file
        }
    except Exception as e:
        print(f"Error loading {hist_file}: {e}")
        return None


def load_first_file_metadata(hist_file):
    """Load all metadata from the first file."""
    data = np.load(hist_file, allow_pickle=True)
    
    # Get all the metadata fields
    result = {
        'hist_mag': data['hist_mag'],
        'hist_other': data['hist_other'],
        'mag_channels': data['mag_channels'],
        'other_channels': data['other_channels'],
        'ell_bin_edges': data['ell_bin_edges'],
        'theta_bin_edges': data['theta_bin_edges'],
        'phi_bin_edges': data['phi_bin_edges']
    }
    
    # Handle optional fields
    if 'metadata' in data:
        result['metadata'] = dict(data['metadata'].item())
    else:
        result['metadata'] = {}
    
    if 'node_info' in data:
        result['node_info'] = dict(data['node_info'].item())
    
    # Handle optional fields
    if 'sf_channel_bin_edges' in data:
        result['sf_channel_bin_edges'] = data['sf_channel_bin_edges']
    if 'product_bin_edges' in data:
        result['product_bin_edges'] = data['product_bin_edges']
    
    return result


def reconstruct_bin_edges(metadata):
    """Reconstruct bin edges from metadata if needed."""
    sf_channel_bin_edges = None
    product_bin_edges = None
    
    # Reconstruct sf_channel_bin_edges
    if 'log_sf_bin_edges_min' in metadata and 'log_sf_bin_edges_max' in metadata:
        sf_channel_bin_edges = []
        for i in range(11):  # 11 channels
            edges = np.logspace(
                metadata['log_sf_bin_edges_min'][i],
                metadata['log_sf_bin_edges_max'][i],
                metadata['N_sf_bin_edges']
            )
            sf_channel_bin_edges.append(edges)
    
    # Reconstruct product_bin_edges
    if 'log_product_bin_edges_min' in metadata and 'log_product_bin_edges_max' in metadata:
        product_bin_edges = np.logspace(
            metadata['log_product_bin_edges_min'],
            metadata['log_product_bin_edges_max'],
            metadata['N_product_bin_edges']
        )
    else:
        # Default values if not specified
        product_bin_edges = np.logspace(-5, 5, 201)
    
    return sf_channel_bin_edges, product_bin_edges


def combine_node_histograms_parallel(histogram_files, args):
    """Combine histograms from different nodes using parallel loading."""
    print(f"Found {len(histogram_files)} histogram files to combine")
    
    # Load first file to get structure and metadata
    print("Loading metadata from first file...")
    first_data = load_first_file_metadata(histogram_files[0])
    
    # Initialize totals with correct dtype for efficiency
    hist_mag_total = first_data['hist_mag'].astype(np.float64, copy=True)
    hist_other_total = first_data['hist_other'].astype(np.float64, copy=True)
    
    # Get metadata
    mag_channels = first_data['mag_channels']
    other_channels = first_data['other_channels']
    ell_bin_edges = first_data['ell_bin_edges']
    theta_bin_edges = first_data['theta_bin_edges']
    phi_bin_edges = first_data['phi_bin_edges']
    metadata = first_data['metadata']
    
    # Get or reconstruct bin edges
    sf_channel_bin_edges = first_data.get('sf_channel_bin_edges', None)
    product_bin_edges = first_data.get('product_bin_edges', None)
    
    if sf_channel_bin_edges is None or product_bin_edges is None:
        sf_channel_bin_edges_rec, product_bin_edges_rec = reconstruct_bin_edges(metadata)
        if sf_channel_bin_edges is None:
            sf_channel_bin_edges = sf_channel_bin_edges_rec
        if product_bin_edges is None:
            product_bin_edges = product_bin_edges_rec
    
    # Track node info
    node_infos = [first_data['node_info']]
    total_displacements = first_data['node_info']['n_displacements']
    
    # Determine number of workers
    n_workers = min(len(histogram_files) - 1, cpu_count(), 32)  # Cap at 32 workers
    
    if len(histogram_files) > 1:
        print(f"Loading remaining {len(histogram_files) - 1} files in parallel with {n_workers} workers...")
        
        # Load remaining files in parallel
        with Pool(n_workers) as pool:
            if HAS_TQDM:
                results = list(tqdm(
                    pool.imap(load_histogram_data, histogram_files[1:]),
                    total=len(histogram_files) - 1,
                    desc="Loading files"
                ))
            else:
                results = pool.map(load_histogram_data, histogram_files[1:])
        
        # Combine results
        print("Combining histograms...")
        for result in tqdm(results, desc="Summing", disable=not HAS_TQDM):
            if result is None:
                continue
            
            # Use in-place addition for efficiency
            np.add(hist_mag_total, result['hist_mag'], out=hist_mag_total)
            np.add(hist_other_total, result['hist_other'], out=hist_other_total)
            
            node_infos.append(result['node_info'])
            total_displacements += result['node_info']['n_displacements']
    
    # Verify we have all nodes
    expected_nodes = max(info['total_nodes'] for info in node_infos)
    found_nodes = sorted(set(info['node_id'] for info in node_infos))
    
    print(f"\nTotal displacements processed: {total_displacements}")
    print(f"Expected nodes: {expected_nodes}")
    print(f"Found nodes: {len(found_nodes)} - IDs: {found_nodes}")
    
    if len(found_nodes) != expected_nodes:
        missing = set(range(expected_nodes)) - set(found_nodes)
        print(f"WARNING: Missing nodes: {sorted(missing)}")
    
    # Update metadata
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
    print(f"Saving combined results to {output_file}...")
    np.savez_compressed(
        output_file,
        hist_mag=hist_mag_total,
        hist_other=hist_other_total,
        mag_channels=mag_channels,
        other_channels=other_channels,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_channel_bin_edges=sf_channel_bin_edges,
        product_bin_edges=product_bin_edges,
        metadata=metadata,
        node_infos=node_infos
    )
    
    print(f"\nCombined results saved to {output_file}")
    print(f"Total counts in histograms: {hist_mag_total.sum() + hist_other_total.sum()}")
    
    return 0


def load_slice_data(slice_file):
    """Load slice data for parallel loading."""
    try:
        data = np.load(slice_file, allow_pickle=True)
        result = {
            'hist_mag': data['hist_mag'],
            'hist_other': data['hist_other'],
            'file': slice_file
        }
        
        if 'metadata' in data:
            result['metadata'] = dict(data['metadata'].item())
        
        return result
    except Exception as e:
        print(f"Error loading {slice_file}: {e}")
        return None


def merge_slice_results_parallel(slice_files, args):
    """Merge sf_results from different slices using parallel loading."""
    print(f"Found {len(slice_files)} slice result files to merge")
    
    # Load first file to get structure
    print("Loading metadata from first file...")
    first_data = load_first_file_metadata(slice_files[0])
    
    # Initialize totals
    hist_mag_total = first_data['hist_mag'].astype(np.float64, copy=True)
    hist_other_total = first_data['hist_other'].astype(np.float64, copy=True)
    
    # Get metadata
    mag_channels = first_data['mag_channels']
    other_channels = first_data['other_channels']
    ell_bin_edges = first_data['ell_bin_edges']
    theta_bin_edges = first_data['theta_bin_edges']
    phi_bin_edges = first_data['phi_bin_edges']
    metadata = first_data.get('metadata', {})
    
    # Get or reconstruct bin edges
    sf_channel_bin_edges = first_data.get('sf_channel_bin_edges', None)
    product_bin_edges = first_data.get('product_bin_edges', None)
    
    if sf_channel_bin_edges is None or product_bin_edges is None:
        sf_channel_bin_edges_rec, product_bin_edges_rec = reconstruct_bin_edges(metadata)
        if sf_channel_bin_edges is None:
            sf_channel_bin_edges = sf_channel_bin_edges_rec
        if product_bin_edges is None:
            product_bin_edges = product_bin_edges_rec
    
    # Track slice info
    slice_names = [Path(slice_files[0]).stem]
    total_displacements = metadata.get('total_displacements', 0)
    all_metadata = [metadata] if metadata else []
    
    # Determine number of workers
    n_workers = min(len(slice_files) - 1, cpu_count(), 32)
    
    if len(slice_files) > 1:
        print(f"Loading remaining {len(slice_files) - 1} files in parallel with {n_workers} workers...")
        
        # Load remaining files in parallel
        with Pool(n_workers) as pool:
            if HAS_TQDM:
                results = list(tqdm(
                    pool.imap(load_slice_data, slice_files[1:]),
                    total=len(slice_files) - 1,
                    desc="Loading files"
                ))
            else:
                results = pool.map(load_slice_data, slice_files[1:])
        
        # Combine results
        print("Combining slice histograms...")
        for result in tqdm(results, desc="Summing", disable=not HAS_TQDM):
            if result is None:
                continue
            
            # Use in-place addition
            np.add(hist_mag_total, result['hist_mag'], out=hist_mag_total)
            np.add(hist_other_total, result['hist_other'], out=hist_other_total)
            
            slice_names.append(Path(result['file']).stem)
            
            if 'metadata' in result:
                total_displacements += result['metadata'].get('total_displacements', 0)
                all_metadata.append(result['metadata'])
    
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
        for key in ['stride', 'N_random_subsamples', 'stencil_width', 'n_ell_bins',
                    'log_sf_bin_edges_min', 'log_sf_bin_edges_max', 'N_sf_bin_edges',
                    'log_product_bin_edges_min', 'log_product_bin_edges_max', 'N_product_bin_edges']:
            if key in all_metadata[0]:
                combined_metadata[key] = all_metadata[0][key]
    
    # Determine output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sf_results_all_slices_{timestamp}.npz"
    else:
        output_file = args.output
    
    # Save merged results
    print(f"Saving merged results to {output_file}...")
    np.savez_compressed(
        output_file,
        hist_mag=hist_mag_total,
        hist_other=hist_other_total,
        mag_channels=mag_channels,
        other_channels=other_channels,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_channel_bin_edges=sf_channel_bin_edges,
        product_bin_edges=product_bin_edges,
        metadata=combined_metadata,
        slice_metadata=all_metadata
    )
    
    print(f"\nMerged results saved to {output_file}")
    print(f"Total counts in histograms: {hist_mag_total.sum() + hist_other_total.sum()}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel combine of structure function histograms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine node histograms from a single slice
  %(prog)s --pattern "histogram_*_node*.npz" --output sf_results_slice0.npz --mode node
  
  # Merge multiple slice results into final result
  %(prog)s --pattern "sf_results_slice*.npz" --output sf_results_all.npz --mode slice
  
  # Use more workers for very large jobs
  %(prog)s --pattern "histogram_*.npz" --workers 64 --mode node
        """
    )
    
    parser.add_argument("--pattern", type=str, required=True,
                        help="Glob pattern for histogram files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: sf_results_TIMESTAMP.npz)")
    parser.add_argument("--mode", type=str, default="node", choices=["node", "slice"],
                        help="Mode: 'node' combines node histograms, 'slice' merges slice results")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto-detect)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Override worker count if specified
    if args.workers:
        global cpu_count
        old_cpu_count = cpu_count
        cpu_count = lambda: args.workers
    
    # Find all matching files
    input_files = sorted(glob(args.pattern))
    
    if not input_files:
        print(f"Error: No files found matching pattern '{args.pattern}'")
        return 1
    
    # Choose the appropriate function based on mode
    if args.mode == "node":
        return combine_node_histograms_parallel(input_files, args)
    else:  # mode == "slice"
        return merge_slice_results_parallel(input_files, args)


if __name__ == "__main__":
    exit(main())