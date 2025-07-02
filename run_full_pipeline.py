#!/usr/bin/env python3
"""Run the full pipeline: extract slices and compute structure functions."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sfunctor.io.extract import extract_2d_slice
from sfunctor.io.slice_io import load_slice_npz
from sfunctor.analysis.single_slice import analyze_slice

def extract_slices():
    """Extract one slice in each direction."""
    print("Extracting slices...")
    
    sim_name = "Turb_320_beta100_dedt025_plm"
    
    # Extract slices at the middle of the domain (0.0)
    slices = {}
    
    for axis in [1, 2, 3]:
        print(f"\nExtracting slice for axis {axis} (x{axis}-normal)...")
        try:
            slice_data = extract_2d_slice(
                sim_name=sim_name,
                axis=axis,
                slice_value=0.0,
                file_number=0,  # Use first available file
                save=True,
                cache_dir="slice_data"
            )
            slices[f"axis{axis}"] = slice_data
            print(f"✅ Successfully extracted slice for axis {axis}")
            
            # Print available fields
            print(f"   Available fields: {list(slice_data.keys())}")
            
            # Check if new fields are present
            new_fields = ['curvx', 'curvy', 'curvz', 'grad_rho_x', 'grad_rho_y', 'grad_rho_z']
            for field in new_fields:
                if field in slice_data:
                    valid_data = slice_data[field][~np.isnan(slice_data[field])]
                    if len(valid_data) > 0:
                        print(f"   ✅ {field}: min={np.min(valid_data):.3e}, max={np.max(valid_data):.3e}")
                    else:
                        print(f"   ⚠️  {field}: all NaN")
                else:
                    print(f"   ❌ {field}: missing")
                    
        except Exception as e:
            print(f"❌ Error extracting slice for axis {axis}: {e}")
            
    return slices

def visualize_slices(slices):
    """Create visualizations of the extracted slices."""
    print("\n\nCreating slice visualizations...")
    
    for name, slice_data in slices.items():
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Slice {name}', fontsize=16)
        
        # Fields to plot
        fields = [
            ('dens', 'Density', 'viridis', 'linear'),
            ('velx', 'Velocity X', 'RdBu_r', 'linear'),
            ('vely', 'Velocity Y', 'RdBu_r', 'linear'),
            ('velz', 'Velocity Z', 'RdBu_r', 'linear'),
            ('bcc1', 'B_x', 'RdBu_r', 'linear'),
            ('bcc2', 'B_y', 'RdBu_r', 'linear'),
            ('bcc3', 'B_z', 'RdBu_r', 'linear'),
            ('vortx', 'Vorticity X', 'RdBu_r', 'linear'),
            ('curvx', 'Curvature X', 'RdBu_r', 'linear'),
            ('curvy', 'Curvature Y', 'RdBu_r', 'linear'),
            ('curvz', 'Curvature Z', 'RdBu_r', 'linear'),
            ('grad_rho_x', 'Grad ρ X', 'viridis', 'linear'),
        ]
        
        for idx, (field, title, cmap, scale) in enumerate(fields):
            ax = axes.flat[idx]
            
            if field in slice_data:
                data = slice_data[field]
                if scale == 'log':
                    im = ax.imshow(np.abs(data), cmap=cmap, origin='lower', 
                                  norm=plt.matplotlib.colors.LogNorm())
                else:
                    im = ax.imshow(data, cmap=cmap, origin='lower')
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'{field} not available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        output_file = f'slice_visualization_{name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_file}")
        plt.close()

def run_structure_functions():
    """Run structure function analysis on the extracted slices."""
    print("\n\nRunning structure function analysis...")
    
    # Find slice files
    slice_files = list(Path("slice_data").glob("Turb_320_beta100_dedt025_plm_axis*_slice0_file0000.npz"))
    
    if not slice_files:
        print("❌ No slice files found. Make sure extraction completed successfully.")
        return
    
    results = {}
    
    for slice_file in slice_files:
        print(f"\nAnalyzing {slice_file.name}...")
        
        try:
            # Load slice data
            slice_data = load_slice_npz(slice_file, stride=2)  # Use stride=2 for faster computation
            
            # Determine axis from filename
            if "axis1" in str(slice_file):
                axis = 1
            elif "axis2" in str(slice_file):
                axis = 2
            else:
                axis = 3
            
            # Ensure all required fields are present in slice_data
            # The analyze_slice function expects certain fields, so let's make sure they're all there
            required_fields = ['rho', 'v_x', 'v_y', 'v_z', 'B_x', 'B_y', 'B_z',
                             'omega_x', 'omega_y', 'omega_z', 'j_x', 'j_y', 'j_z',
                             'curv_x', 'curv_y', 'curv_z', 'grad_rho_x', 'grad_rho_y', 'grad_rho_z']
            
            # Check and report missing fields
            missing = [f for f in required_fields if f not in slice_data]
            if missing:
                print(f"   ⚠️  Missing fields: {missing}")
            
            # Run analysis
            result = analyze_slice(
                slice_data,
                n_displacements=500,  # Reduced for faster computation
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
                hist_mag=result['hist_mag'],
                hist_other=result['hist_other'],
                ell_bin_edges=result['ell_bin_edges'],
                theta_bin_edges=result['theta_bin_edges'],
                phi_bin_edges=result['phi_bin_edges'],
                sf_bin_edges=result['sf_bin_edges'],
                product_bin_edges=result['product_bin_edges'],
                mag_channels=result['mag_channels'],
                other_channels=result['other_channels']
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
        
        # Create figure for cross products (angles)
        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        fig.suptitle(f'Cross Products and Angles - {name}', fontsize=16)
        
        other_channel_names = result['other_channels']
        
        for idx, channel_name in enumerate(other_channel_names):
            if idx < 18:  # We have 18 OTHER channels now
                ax = axes.flat[idx]
                
                # Sum over product bins to get ell dependence
                counts_vs_ell = np.sum(hist_other[idx], axis=1)
                
                if np.sum(counts_vs_ell) > 0:
                    ax.loglog(ell_centers, counts_vs_ell, 'o-')
                    ax.set_xlabel('ℓ')
                    ax.set_ylabel('Counts')
                    ax.set_title(channel_name)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(channel_name)
        
        plt.tight_layout()
        output_file = f'sf_angles_{name.replace(".npz", "")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved angle plot: {output_file}")
        plt.close()

def main():
    """Run the full pipeline."""
    print("Starting full pipeline execution...\n")
    
    # Step 1: Extract slices
    slices = extract_slices()
    
    if not slices:
        print("❌ No slices were extracted. Exiting.")
        return
    
    # Step 2: Visualize slices
    visualize_slices(slices)
    
    # Step 3: Run structure function analysis
    results = run_structure_functions()
    
    if not results:
        print("❌ No structure function results. Exiting.")
        return
    
    # Step 4: Plot structure functions
    plot_structure_functions(results)
    
    print("\n✅ Pipeline completed successfully!")
    print("\nGenerated files:")
    print("  - Slice visualizations: slice_visualization_axis*.png")
    print("  - Structure function results: sf_results_*.npz")
    print("  - Structure function plots: sf_magnitudes_*.png, sf_angles_*.png")

if __name__ == "__main__":
    main()