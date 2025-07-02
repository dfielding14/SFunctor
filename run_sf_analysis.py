#!/usr/bin/env python3
"""Run just the structure function analysis on existing slices."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sfunctor.io.slice_io import load_slice_npz
from sfunctor.analysis.single_slice import analyze_slice

def run_structure_functions():
    """Run structure function analysis on the extracted slices."""
    print("Running structure function analysis...")
    
    # Find slice files
    slice_files = list(Path("slice_data").glob("Turb_320_beta100_dedt025_plm_axis*_slice0_file0000.npz"))
    
    if not slice_files:
        print("❌ No slice files found. Make sure extraction completed successfully.")
        return
    
    results = {}
    
    for slice_file in slice_files:
        print(f"\nAnalyzing {slice_file.name}...")
        
        try:
            # Load slice data with stride=2 for speed
            slice_data = load_slice_npz(slice_file, stride=2)
            
            # Print available fields
            print(f"   Available fields: {sorted(slice_data.keys())}")
            
            # Determine axis from filename
            if "axis1" in str(slice_file):
                axis = 1
            elif "axis2" in str(slice_file):
                axis = 2
            else:
                axis = 3
            
            # Run analysis
            result = analyze_slice(
                slice_data,
                n_displacements=500,
                n_ell_bins=32,
                n_random_subsamples=500,
                stencil_width=2,
                axis=axis
            )
            
            results[slice_file.name] = result
            print(f"✅ Analysis complete for {slice_file.name}")
            
            # Save results
            output_file = f"sf_results_{slice_file.stem}.npz"
            np.savez_compressed(
                output_file,
                **result  # Save all result fields
            )
            print(f"   Saved results to {output_file}")
            
        except Exception as e:
            print(f"❌ Error analyzing {slice_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def plot_structure_functions(results):
    """Create structure function plots."""
    print("\n\nCreating structure function plots...")
    
    for name, result in results.items():
        # Extract key data
        hist_mag = result['hist_mag']
        hist_other = result['hist_other']
        ell_bin_edges = result['ell_bin_edges']
        ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
        
        # Create figure for magnitude channels
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Structure Function Magnitudes - {name}', fontsize=16)
        
        mag_channel_names = result['mag_channels']
        
        for idx, channel_name in enumerate(mag_channel_names):
            if idx < 10:  # We have 10 magnitude channels now
                ax = axes.flat[idx]
                
                # Sum over angular bins to get ell dependence
                sf_vs_ell = np.sum(hist_mag[idx], axis=(1, 2, 3))
                
                if np.sum(sf_vs_ell) > 0:
                    ax.loglog(ell_centers, sf_vs_ell, 'o-')
                    ax.set_xlabel('ℓ')
                    ax.set_ylabel('Counts')
                    ax.set_title(channel_name)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(channel_name)
        
        plt.tight_layout()
        output_file = f'sf_magnitudes_{name.replace(".npz", "")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved structure function plot: {output_file}")
        plt.close()
        
        # Create detailed plots for new channels
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'New Channels - {name}', fontsize=16)
        
        # Plot curvature structure function
        ax = axes[0, 0]
        curv_idx = mag_channel_names.index('D_CURV')
        sf_curv = np.sum(hist_mag[curv_idx], axis=(1, 2, 3))
        if np.sum(sf_curv) > 0:
            ax.loglog(ell_centers, sf_curv, 'o-', label='|δ(b·∇b)|')
            ax.set_xlabel('ℓ')
            ax.set_ylabel('Counts')
            ax.set_title('Magnetic Curvature Structure Function')
            ax.grid(True, alpha=0.3)
        
        # Plot density gradient structure function
        ax = axes[0, 1]
        grad_idx = mag_channel_names.index('D_GRAD_RHO')
        sf_grad = np.sum(hist_mag[grad_idx], axis=(1, 2, 3))
        if np.sum(sf_grad) > 0:
            ax.loglog(ell_centers, sf_grad, 'o-', label='|δ(∇ρ)|')
            ax.set_xlabel('ℓ')
            ax.set_ylabel('Counts')
            ax.set_title('Density Gradient Structure Function')
            ax.grid(True, alpha=0.3)
        
        # Plot cross product (for angle)
        ax = axes[1, 0]
        other_channel_names = result['other_channels']
        cross_idx = other_channel_names.index('D_CURV_CROSS_GRAD_RHO')
        cross_vs_ell = np.sum(hist_other[cross_idx], axis=1)
        if np.sum(cross_vs_ell) > 0:
            ax.loglog(ell_centers, cross_vs_ell, 'o-')
            ax.set_xlabel('ℓ')
            ax.set_ylabel('Counts')
            ax.set_title('|δ(curv) × δ(∇ρ)|')
            ax.grid(True, alpha=0.3)
        
        # Plot magnitude product (for angle normalization)
        ax = axes[1, 1]
        mag_idx = other_channel_names.index('D_CURV_D_GRAD_RHO_MAG')
        mag_vs_ell = np.sum(hist_other[mag_idx], axis=1)
        if np.sum(mag_vs_ell) > 0:
            ax.loglog(ell_centers, mag_vs_ell, 'o-')
            ax.set_xlabel('ℓ')
            ax.set_ylabel('Counts')
            ax.set_title('|δ(curv)| × |δ(∇ρ)|')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f'sf_new_channels_{name.replace(".npz", "")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved new channels plot: {output_file}")
        plt.close()

def main():
    """Run structure function analysis and plotting."""
    results = run_structure_functions()
    
    if results:
        plot_structure_functions(results)
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ No results to plot.")

if __name__ == "__main__":
    main()