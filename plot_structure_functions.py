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
    product_bin_edges = data['product_bin_edges']
    
    # Handle channel-specific bin edges
    if 'sf_channel_bin_edges' in data:
        # New format with channel-specific bins
        sf_channel_bin_edges = data['sf_channel_bin_edges']
        # For backward compatibility in plotting, use the first channel's bins as default
        sf_bin_edges = sf_channel_bin_edges[0]
        
        # Also extract metadata if available
        if 'metadata' in data:
            metadata = dict(data['metadata'].item())
            if 'log_sf_bin_edges_min' in metadata:
                print(f"Bin edge parameters found:")
                print(f"  log_sf_bin_edges_min: {metadata['log_sf_bin_edges_min']}")
                print(f"  log_sf_bin_edges_max: {metadata['log_sf_bin_edges_max']}")
                print(f"  N_sf_bin_edges: {metadata['N_sf_bin_edges']}")
    else:
        # Old format - single set of bins
        sf_bin_edges = data['sf_bin_edges']
        sf_channel_bin_edges = [sf_bin_edges] * len(mag_channels)
    
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
    # Pass channel-specific bins if available
    if 'sf_channel_bin_edges' in data:
        plot_2d_histograms_with_channel_bins(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                           output_dir, base_name, args.format, args.dpi)
    else:
        plot_2d_histograms(hist_mag, mag_channels, ell_centers, sf_centers,
                           output_dir, base_name, args.format, args.dpi)
    
    # 3. Plot angular distributions (2D histogram of D_V vs ell and theta)
    plot_angular_distributions(hist_mag, mag_channels, ell_centers, theta_centers, phi_centers,
                               output_dir, base_name, args.format, args.dpi)
    
    # 4. Plot cross-product ratios
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
    
    # Select key channels to plot - use actual channel names
    key_channels = ['D_V', 'D_B', 'D_RHO', 'D_ZPLUS']
    
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
        
        # Plot only non-zero values
        mask = mean_sf > 0
        if np.any(mask):
            ax.errorbar(ell_centers[mask], mean_sf[mask], yerr=std_sf[mask], 
                       fmt='o-', capsize=3, label=channel_name)
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
    # Select channels to plot - use actual channel names
    channels_to_plot = ['D_V', 'D_B', 'D_RHO', 'D_ZPLUS']
    
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


def plot_2d_histograms_with_channel_bins(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                                          output_dir, base_name, fmt, dpi):
    """Plot normalized 2D histograms for selected channels using channel-specific bins."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Select channels to plot
    channels_to_plot = ['D_V', 'D_B', 'D_OMEGA', 'D_B_over_Bmean_loc']
    
    plot_idx = 0
    for channel_name in channels_to_plot:
        if channel_name not in mag_channels or plot_idx >= 4:
            continue
            
        ax = axes[plot_idx]
        channel_idx = list(mag_channels).index(channel_name)
        
        # Get channel-specific bin edges and centers
        bin_edges = sf_channel_bin_edges[channel_idx]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Sum over angles to get 2D histogram
        hist_2d = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi
        
        # Normalize each ell bin by total counts in that ell
        hist_2d_norm = hist_2d.copy().astype(float)
        for i in range(len(ell_centers)):
            total_counts = hist_2d[i].sum()
            if total_counts > 0:
                hist_2d_norm[i] = hist_2d[i] / total_counts
        
        # Create meshgrid for plotting
        ell_mesh, sf_mesh = np.meshgrid(ell_centers, bin_centers)
        
        # Plot normalized histogram
        pcm = ax.pcolormesh(ell_mesh, sf_mesh, hist_2d_norm.T, 
                            norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(channel_name)
        ax.set_title(f'Normalized 2D Histogram: {channel_name} vs $\ell$')
        
        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, label='Probability')
        plot_idx += 1
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_2d_histograms_normalized.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_angular_distributions(hist_mag, mag_channels, ell_centers, theta_centers, phi_centers,
                               output_dir, base_name, fmt, dpi):
    """Plot 2D histogram of D_V versus ell and theta (summed over phi)."""
    # Get D_V channel
    channel_name = 'D_V'
    if channel_name not in mag_channels:
        print(f"Warning: {channel_name} not found, using first channel")
        channel_name = mag_channels[0]
    
    channel_idx = list(mag_channels).index(channel_name)
    
    # Sum over phi and sf values to get 2D histogram
    # hist_mag shape: (channel, ell, theta, phi, sf)
    # Sum over phi (axis 2) and sf (axis 3)
    hist_2d_theta = hist_mag[channel_idx].sum(axis=2).sum(axis=2)  # Shape: (ell, theta)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert theta to degrees for display
    theta_degrees = np.rad2deg(theta_centers)
    
    # Create meshgrid for plotting
    ell_edges = np.concatenate([[ell_centers[0] * 0.9], 
                                0.5 * (ell_centers[:-1] + ell_centers[1:]),
                                [ell_centers[-1] * 1.1]])
    theta_edges_deg = np.concatenate([[theta_degrees[0] - 0.5 * (theta_degrees[1] - theta_degrees[0])],
                                      0.5 * (theta_degrees[:-1] + theta_degrees[1:]),
                                      [theta_degrees[-1] + 0.5 * (theta_degrees[-1] - theta_degrees[-2])]])
    
    # Plot 2D histogram
    pcm = ax.pcolormesh(ell_edges, theta_edges_deg, hist_2d_theta.T,
                        norm=colors.LogNorm(vmin=1, vmax=hist_2d_theta.max()),
                        cmap='viridis', shading='flat')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\theta$ (degrees)')
    ax.set_title(f'2D Histogram: {channel_name} vs $\ell$ and $\\theta$ (summed over $\phi$)')
    
    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, label='Counts')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_angular_distribution_2d.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_cross_products(hist_other, other_channels, ell_centers, product_centers,
                        output_dir, base_name, fmt, dpi):
    """Plot ratios of cross-products to their corresponding MAG products."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Define cross product / MAG pairs
    ratio_pairs = [
        ('D_Vperp_CROSS_Bperp', 'D_Vperp_D_Bperp_MAG'),
        ('D_Vperp_CROSS_VAperp', 'D_Vperp_D_VAperp_MAG'),
        ('D_Vperp_CROSS_Omegaperp', 'D_Vperp_D_Omegaperp_MAG'),
        ('D_Bperp_CROSS_Jperp', 'D_Bperp_D_Jperp_MAG')
    ]
    
    for idx, (cross_name, mag_name) in enumerate(ratio_pairs):
        ax = axes[idx]
        
        if cross_name not in other_channels or mag_name not in other_channels:
            continue
            
        cross_idx = list(other_channels).index(cross_name)
        mag_idx = list(other_channels).index(mag_name)
        
        # Compute mean values for each ell
        mean_cross = np.zeros(len(ell_centers))
        mean_mag = np.zeros(len(ell_centers))
        
        for i in range(len(ell_centers)):
            if hist_other[cross_idx][i].sum() > 0:
                mean_cross[i] = np.average(product_centers, weights=hist_other[cross_idx][i])
            if hist_other[mag_idx][i].sum() > 0:
                mean_mag[i] = np.average(product_centers, weights=hist_other[mag_idx][i])
        
        # Compute ratio
        mask = (mean_mag > 0) & (mean_cross > 0)
        if np.any(mask):
            ratio = mean_cross[mask] / mean_mag[mask]
            
            # Plot ratio
            ax.plot(ell_centers[mask], ratio, 'o-', markersize=6)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(f'{cross_name} / {mag_name}')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Ratio: {cross_name} / {mag_name}')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_cross_product_ratios.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


if __name__ == "__main__":
    exit(main())