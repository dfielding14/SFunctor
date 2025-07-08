#!/usr/bin/env python3
"""Verify consistency of SFunctor results across different runs.

This script compares structure function results from different parallelization
strategies to ensure numerical consistency.
"""

import sys
import argparse
from pathlib import Path
import numpy as np


def load_results(filepath):
    """Load SFunctor results from npz file."""
    try:
        data = np.load(filepath)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compare_histograms(data1, data2, rtol=1e-10, atol=1e-12):
    """Compare histogram data between two results.
    
    Parameters
    ----------
    data1, data2 : dict
        Loaded npz data dictionaries
    rtol : float
        Relative tolerance for comparison
    atol : float
        Absolute tolerance for comparison
        
    Returns
    -------
    bool
        True if histograms match within tolerance
    dict
        Dictionary of comparison metrics
    """
    metrics = {}
    all_match = True
    
    # Get histogram keys (mag_histograms, other_histograms)
    hist_keys = [k for k in data1.keys() if 'histogram' in k]
    
    for key in hist_keys:
        if key not in data2:
            print(f"WARNING: {key} not found in second file")
            all_match = False
            continue
            
        hist1 = data1[key]
        hist2 = data2[key]
        
        # Check shapes
        if hist1.shape != hist2.shape:
            print(f"ERROR: Shape mismatch for {key}: {hist1.shape} vs {hist2.shape}")
            all_match = False
            continue
        
        # Compute differences
        abs_diff = np.abs(hist1 - hist2)
        rel_diff = np.abs(hist1 - hist2) / (np.abs(hist1) + 1e-16)
        
        # Get statistics
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        # Check tolerance
        matches = np.allclose(hist1, hist2, rtol=rtol, atol=atol)
        if not matches:
            all_match = False
            
        metrics[key] = {
            'matches': matches,
            'max_abs_diff': max_abs_diff,
            'max_rel_diff': max_rel_diff,
            'mean_abs_diff': mean_abs_diff,
            'mean_rel_diff': mean_rel_diff,
            'n_nonzero_diffs': np.sum(abs_diff > 0)
        }
    
    return all_match, metrics


def compare_metadata(data1, data2):
    """Compare metadata between two results."""
    meta_keys = ['ell_bin_edges', 'theta_bin_edges', 'phi_bin_edges', 
                 'sf_bin_edges', 'product_bin_edges']
    
    all_match = True
    for key in meta_keys:
        if key in data1 and key in data2:
            if not np.array_equal(data1[key], data2[key]):
                print(f"WARNING: {key} differs between files")
                all_match = False
        elif key in data1 or key in data2:
            print(f"WARNING: {key} present in only one file")
            all_match = False
            
    return all_match


def print_comparison_summary(file1, file2, all_match, metrics):
    """Print a summary of the comparison."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print("-"*60)
    
    if all_match:
        print("✓ All histograms match within tolerance!")
    else:
        print("✗ Histograms do not match within tolerance")
    
    print("\nDetailed metrics:")
    for key, metric in metrics.items():
        print(f"\n{key}:")
        print(f"  Matches: {metric['matches']}")
        print(f"  Max absolute difference: {metric['max_abs_diff']:.2e}")
        print(f"  Max relative difference: {metric['max_rel_diff']:.2e}")
        print(f"  Mean absolute difference: {metric['mean_abs_diff']:.2e}")
        print(f"  Mean relative difference: {metric['mean_rel_diff']:.2e}")
        print(f"  Non-zero differences: {metric['n_nonzero_diffs']}")


def compare_multiple_files(filepaths, reference_idx=0):
    """Compare multiple result files against a reference."""
    if len(filepaths) < 2:
        print("Need at least 2 files to compare")
        return
        
    print(f"\nComparing {len(filepaths)} files...")
    print(f"Reference: {filepaths[reference_idx]}")
    
    # Load reference
    ref_data = load_results(filepaths[reference_idx])
    if ref_data is None:
        print("Failed to load reference file")
        return
    
    all_comparisons_match = True
    
    # Compare each file to reference
    for i, filepath in enumerate(filepaths):
        if i == reference_idx:
            continue
            
        print(f"\n{'='*60}")
        print(f"Comparing to: {filepath}")
        
        data = load_results(filepath)
        if data is None:
            print("Failed to load file")
            all_comparisons_match = False
            continue
        
        # Compare histograms
        match, metrics = compare_histograms(ref_data, data)
        if not match:
            all_comparisons_match = False
            
        # Print summary for this comparison
        for key, metric in metrics.items():
            if not metric['matches']:
                print(f"  ✗ {key}: max_diff={metric['max_abs_diff']:.2e}")
            else:
                print(f"  ✓ {key}: matches")
    
    print(f"\n{'='*60}")
    if all_comparisons_match:
        print("✓ ALL FILES MATCH THE REFERENCE!")
    else:
        print("✗ Some files differ from the reference")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Verify consistency of SFunctor results"
    )
    parser.add_argument(
        'files', 
        nargs='+', 
        help='Result files to compare (npz format)'
    )
    parser.add_argument(
        '--rtol', 
        type=float, 
        default=1e-10,
        help='Relative tolerance for comparison (default: 1e-10)'
    )
    parser.add_argument(
        '--atol', 
        type=float, 
        default=1e-12,
        help='Absolute tolerance for comparison (default: 1e-12)'
    )
    parser.add_argument(
        '--reference', 
        type=int, 
        default=0,
        help='Index of reference file for multiple comparisons (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    filepaths = [Path(f) for f in args.files]
    
    # Check files exist
    for filepath in filepaths:
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
    
    if len(filepaths) == 2:
        # Compare two files directly
        data1 = load_results(filepaths[0])
        data2 = load_results(filepaths[1])
        
        if data1 is None or data2 is None:
            print("Failed to load one or both files")
            sys.exit(1)
        
        # Compare metadata
        meta_match = compare_metadata(data1, data2)
        
        # Compare histograms
        hist_match, metrics = compare_histograms(data1, data2, 
                                                 rtol=args.rtol, 
                                                 atol=args.atol)
        
        # Print summary
        print_comparison_summary(filepaths[0], filepaths[1], 
                               hist_match and meta_match, metrics)
        
        # Exit with appropriate code
        sys.exit(0 if hist_match and meta_match else 1)
        
    else:
        # Compare multiple files
        compare_multiple_files(filepaths, reference_idx=args.reference)


if __name__ == "__main__":
    main()