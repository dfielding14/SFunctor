#!/usr/bin/env python3
"""Plot structure functions from combined results.

This script creates visualizations of the structure functions computed
from the histogram data.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib.colors as colors


def main():
    parser = argparse.ArgumentParser(description="Plot structure functions")
    parser.add_argument("input_file", type=str,
                        help="Path to sf_results file (e.g., sf_results_all_slices.npz)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as input file)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Output format for plots")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for raster formats")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = np.load(args.input_file, allow_pickle=True)
    
    # Extract arrays
    hist_mag = data['hist_mag']
    hist_other = data['hist_other']
    mag_channels = data['mag_channels']
    other_channels = data['other_channels']
    ell_bin_edges = data['ell_bin_edges']
    theta_bin_edges = data['theta_bin_edges']
    phi_bin_edges = data['phi_bin_edges']
    sf_bin_edges = data['sf_bin_edges']
    product_bin_edges = data['product_bin_edges']
    
    # Compute bin centers
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    phi_centers = 0.5 * (phi_bin_edges[:-1] + phi_bin_edges[1:])
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    product_centers = 0.5 * (product_bin_edges[:-1] + product_bin_edges[1:])
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.input_file).parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(args.input_file).stem
    
    print(f"Creating plots in {output_dir}...")
    
    # 1. Plot mean structure functions vs ell for key channels
    plot_mean_structure_functions(hist_mag, mag_channels, ell_centers, sf_centers,
                                  output_dir, base_name, args.format, args.dpi)
    
    # 2. Plot 2D histograms for selected channels
    plot_2d_histograms(hist_mag, mag_channels, ell_centers, sf_centers,
                       output_dir, base_name, args.format, args.dpi)
    
    # 3. Plot angular distributions
    plot_angular_distributions(hist_mag, mag_channels, ell_centers, theta_centers, phi_centers,
                               output_dir, base_name, args.format, args.dpi)
    
    # 4. Plot cross-product structure functions
    plot_cross_products(hist_other, other_channels, ell_centers, product_centers,
                        output_dir, base_name, args.format, args.dpi)
    
    print(f"\nPlots saved to {output_dir}")
    
    if args.show:
        plt.show()
    
    return 0


def plot_mean_structure_functions(hist_mag, mag_channels, ell_centers, sf_centers, 
                                   output_dir, base_name, fmt, dpi):
    """Plot mean structure functions vs ell for key channels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Select key channels to plot
    key_channels = ['dv_ell', 'dv_perp', 'dB_ell', 'dB_perp']
    
    for idx, channel_name in enumerate(key_channels):
        if channel_name not in mag_channels:
            continue
            
        ax = axes[idx]
        channel_idx = list(mag_channels).index(channel_name)
        
        # Sum over angles to get total histogram for this channel
        hist_ell_sf = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi
        
        # Compute mean and std
        mean_sf = np.zeros(len(ell_centers))
        std_sf = np.zeros(len(ell_centers))
        
        for i in range(len(ell_centers)):
            if hist_ell_sf[i].sum() > 0:
                # Compute weighted mean
                mean_sf[i] = np.average(sf_centers, weights=hist_ell_sf[i])
                # Compute weighted std
                variance = np.average((sf_centers - mean_sf[i])**2, weights=hist_ell_sf[i])
                std_sf[i] = np.sqrt(variance)
        
        # Plot
        ax.errorbar(ell_centers, mean_sf, yerr=std_sf, fmt='o-', capsize=3, label=channel_name)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(f'⟨{channel_name}⟩')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Mean {channel_name} vs $\ell$')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_mean_structure_functions.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_2d_histograms(hist_mag, mag_channels, ell_centers, sf_centers,
                       output_dir, base_name, fmt, dpi):
    """Plot 2D histograms of SF vs ell for selected channels."""
    # Select channels to plot
    channels_to_plot = ['dv_ell', 'dB_ell', 'drho', 'dz+_ell']
    
    n_channels = len(channels_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, channel_name in enumerate(channels_to_plot):
        if channel_name not in mag_channels:
            continue
        if idx >= 4:
            break
            
        ax = axes[idx]
        channel_idx = list(mag_channels).index(channel_name)
        
        # Sum over angles
        hist_2d = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi
        
        # Create 2D plot
        ell_edges = np.concatenate([[ell_centers[0] * 0.9], 
                                    0.5 * (ell_centers[:-1] + ell_centers[1:]),
                                    [ell_centers[-1] * 1.1]])
        sf_edges = np.concatenate([[sf_centers[0] * 0.9],
                                   0.5 * (sf_centers[:-1] + sf_centers[1:]),
                                   [sf_centers[-1] * 1.1]])
        
        # Use logarithmic normalization
        pcm = ax.pcolormesh(ell_edges, sf_edges, hist_2d.T,
                            norm=colors.LogNorm(vmin=1, vmax=hist_2d.max()),
                            cmap='viridis', shading='flat')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(channel_name)
        ax.set_title(f'2D Histogram: {channel_name} vs $\ell$')
        
        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, label='Counts')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_2d_histograms.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_angular_distributions(hist_mag, mag_channels, ell_centers, theta_centers, phi_centers,
                               output_dir, base_name, fmt, dpi):
    """Plot angular distributions for selected channels and ell bins."""
    # Select one channel and a few ell bins
    channel_name = 'dv_ell'
    if channel_name not in mag_channels:
        channel_name = mag_channels[0]
    
    channel_idx = list(mag_channels).index(channel_name)
    
    # Select 3 ell bins (small, medium, large)
    n_ell = len(ell_centers)
    ell_indices = [n_ell//4, n_ell//2, 3*n_ell//4]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
    
    for idx, ell_idx in enumerate(ell_indices):
        ax = axes[idx]
        
        # Sum over SF values and phi to get theta distribution
        theta_dist = hist_mag[channel_idx, ell_idx, :, :, :].sum(axis=(1, 2))
        
        # Plot on polar axis
        theta_rad = np.deg2rad(theta_centers)
        ax.plot(theta_rad, theta_dist, 'b-', linewidth=2)
        ax.fill_between(theta_rad, 0, theta_dist, alpha=0.3)
        
        ax.set_title(f'$\ell$ = {ell_centers[ell_idx]:.1f}')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
    
    fig.suptitle(f'Angular Distribution of {channel_name}', fontsize=14)
    plt.tight_layout()
    filename = output_dir / f"{base_name}_angular_distributions.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_cross_products(hist_other, other_channels, ell_centers, product_centers,
                        output_dir, base_name, fmt, dpi):
    """Plot cross-product structure functions."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot first few channels
    n_plot = min(6, len(other_channels))
    
    for idx in range(n_plot):
        channel_name = other_channels[idx]
        
        # Sum over ell bins to get product distribution
        hist_product = hist_other[idx]
        
        # Compute mean for each ell
        mean_product = np.zeros(len(ell_centers))
        
        for i in range(len(ell_centers)):
            if hist_product[i].sum() > 0:
                mean_product[i] = np.average(product_centers, weights=hist_product[i])
        
        # Plot
        ax.plot(ell_centers, mean_product, 'o-', label=channel_name, markersize=4)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Mean Cross Product')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Cross-Product Structure Functions')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_cross_products.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


if __name__ == "__main__":
    exit(main())