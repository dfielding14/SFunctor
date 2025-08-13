#!/usr/bin/env python
"""
Basic Structure Function Analysis Example

This example demonstrates how to:
1. Load an MHD simulation slice
2. Compute structure functions
3. Visualize results
4. Analyze anisotropy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import SFunctor
from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.core.histograms_gpu import get_gpu_info

def main():
    # Check GPU availability
    gpu_info = get_gpu_info()
    print("=" * 60)
    print("SFunctor Basic Analysis Example")
    print("=" * 60)
    print(f"GPU Available: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"GPU Backend: {gpu_info['backend']}")
        print(f"GPU Memory: {gpu_info['memory_gb']:.1f} GB")
    print()
    
    # Configuration for analysis
    config = {
        'stride': 2,              # Downsample by 2x for speed
        'n_disp_per_decade': 8,   # Displacement sampling density
        'n_disp_per_bin': 5,      # Samples per bin
        'n_disp_total': 100,      # Total displacements
        'ell_min': 1,             # Minimum displacement
        'ell_max': 50,            # Maximum displacement
        'n_random_subsamples': 5000,  # Random samples per displacement
        'n_theta_bins': 16,       # Angular resolution (polar)
        'n_phi_bins': 32,         # Angular resolution (azimuthal)
        'stencil_width': 2,       # 2-point stencil
    }
    
    # Find example data
    slice_dir = Path("slice_data")
    if not slice_dir.exists():
        print("Error: slice_data directory not found!")
        print("Please ensure you have example data in ./slice_data/")
        return
    
    slice_files = list(slice_dir.glob("*.npz"))
    if not slice_files:
        print("Error: No .npz files found in slice_data/")
        return
    
    # Use first available slice
    slice_file = slice_files[0]
    print(f"Analyzing: {slice_file.name}")
    print(f"Configuration: {config}")
    print()
    
    # Run analysis
    print("Running structure function analysis...")
    results = analyze_slice(slice_file, **config)
    
    # Extract results
    hist_mag = results['hist_mag']
    hist_other = results['hist_other']
    ell_bin_edges = results['ell_bin_edges']
    theta_bin_edges = results['theta_bin_edges']
    phi_bin_edges = results['phi_bin_edges']
    sf_bin_edges = results['sf_bin_edges']
    
    print(f"Analysis complete!")
    print(f"  Histogram shape (mag): {hist_mag.shape}")
    print(f"  Histogram shape (other): {hist_other.shape}")
    print(f"  Total counts: {hist_mag.sum()}")
    print()
    
    # Visualization
    print("Creating visualizations...")
    
    # Compute bin centers
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Structure Function Analysis Results', fontsize=16)
    
    # Channel names
    channel_names = ['Velocity', 'Magnetic', 'Density', 'Alfvén', 'z⁺', 'z⁻']
    
    for idx, (ax, name) in enumerate(zip(axes.flat, channel_names)):
        if idx >= hist_mag.shape[0]:
            break
            
        # Get structure function for this channel
        sf_channel = hist_mag[idx]
        
        # Average over angles for isotropic view
        sf_iso = sf_channel.mean(axis=(1, 2))  # Average over θ and φ
        
        # Plot 2D histogram
        im = ax.pcolormesh(ell_centers, sf_centers, sf_iso.T, 
                          shading='auto', cmap='viridis')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Displacement ℓ')
        ax.set_ylabel(f'|δ{name[0].lower()}|')
        ax.set_title(f'{name} Structure Function')
        plt.colorbar(im, ax=ax, label='Counts')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'structure_functions_basic.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    
    # Anisotropy analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Anisotropy Analysis', fontsize=16)
    
    # Velocity structure function
    velocity_sf = hist_mag[0]
    
    # Define parallel and perpendicular regions
    parallel_mask = (theta_centers < np.pi/6) | (theta_centers > 5*np.pi/6)
    perp_mask = (theta_centers > np.pi/3) & (theta_centers < 2*np.pi/3)
    
    # Average over appropriate angles
    sf_parallel = velocity_sf[:, parallel_mask, :, :].mean(axis=(1, 2))
    sf_perp = velocity_sf[:, perp_mask, :, :].mean(axis=(1, 2))
    
    # Plot parallel vs perpendicular
    ax = axes2[0]
    for ell_idx in range(0, len(ell_centers), len(ell_centers)//5):
        ax.plot(sf_centers, sf_parallel[ell_idx], 
                label=f'ℓ = {ell_centers[ell_idx]:.1f} (∥)', linestyle='-')
        ax.plot(sf_centers, sf_perp[ell_idx], 
                label=f'ℓ = {ell_centers[ell_idx]:.1f} (⊥)', linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('|δv|')
    ax.set_ylabel('PDF')
    ax.set_title('Parallel vs Perpendicular SF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot anisotropy ratio
    ax = axes2[1]
    for ell_idx in range(0, len(ell_centers), len(ell_centers)//5):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = sf_parallel[ell_idx] / sf_perp[ell_idx]
            ratio[~np.isfinite(ratio)] = np.nan
        ax.plot(sf_centers, ratio, label=f'ℓ = {ell_centers[ell_idx]:.1f}')
    ax.set_xscale('log')
    ax.set_xlabel('|δv|')
    ax.set_ylabel('SF∥ / SF⊥')
    ax.set_title('Anisotropy Ratio')
    ax.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='Isotropic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save anisotropy figure
    output_file2 = 'anisotropy_analysis.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved anisotropy analysis to {output_file2}")
    
    plt.show()
    
    print("\nAnalysis complete!")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"  Mean anisotropy (velocity): {(sf_parallel.sum() / sf_perp.sum()):.2f}")
    
    # Scaling analysis
    velocity_moments = []
    for p in [1, 2, 3]:
        moment = np.sum(sf_centers**p * sf_iso.T, axis=0) / sf_iso.sum(axis=1)
        velocity_moments.append(moment)
    
    print("\nScaling exponents (velocity):")
    for ell_idx in range(0, len(ell_centers), len(ell_centers)//3):
        if ell_idx > 0:
            # Estimate scaling exponent from ratio of moments
            zeta_2 = np.log(velocity_moments[1][ell_idx] / velocity_moments[1][0]) / \
                     np.log(ell_centers[ell_idx] / ell_centers[0])
            print(f"  ℓ = {ell_centers[ell_idx]:.1f}: ζ₂ ≈ {zeta_2:.2f}")


if __name__ == "__main__":
    main()