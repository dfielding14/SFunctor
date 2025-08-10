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
from scipy.optimize import curve_fit

# LaTeX labels for channels
CHANNEL_LABELS = {
    'D_V': r'$\delta v$',
    'D_B': r'$\delta B$',
    'D_RHO': r'$\delta \rho$',
    'D_VA': r'$\delta v_A$',
    'D_ZPLUS': r'$\delta z^+$',
    'D_ZMINUS': r'$\delta z^-$',
    'D_OMEGA': r'$\delta \omega$',
    'D_J': r'$\delta j$',
    'D_CURV': r'$\delta K$',
    'D_GRAD_RHO': r'$\delta |\nabla \rho|$',
    'D_B_over_Bmean_loc': r'$\delta B / \langle B \rangle_{loc}$',
    # Cross products
    'D_Vperp_CROSS_Bperp': r'$\delta v_\perp \times \delta B_\perp$',
    'D_Vperp_CROSS_VAperp': r'$\delta v_\perp \times \delta v_{A\perp}$',
    'D_Vperp_CROSS_Omegaperp': r'$\delta v_\perp \times \delta \omega_\perp$',
    'D_Bperp_CROSS_Jperp': r'$\delta B_\perp \times \delta j_\perp$',
    'D_Vperp_D_Bperp_MAG': r'$|\delta v_\perp| |\delta B_\perp|$',
    'D_Vperp_D_VAperp_MAG': r'$|\delta v_\perp| |\delta v_{A\perp}|$',
    'D_Vperp_D_Omegaperp_MAG': r'$|\delta v_\perp| |\delta \omega_\perp|$',
    'D_Bperp_D_Jperp_MAG': r'$|\delta B_\perp| |\delta j_\perp|$',
}

def get_channel_label(channel_name):
    """Get LaTeX label for channel, with fallback to original name."""
    return CHANNEL_LABELS.get(channel_name, channel_name)


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
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])  # Arithmetic mean for ell
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])  # Arithmetic mean (linear bins)
    phi_centers = 0.5 * (phi_bin_edges[:-1] + phi_bin_edges[1:])  # Arithmetic mean (linear bins)
    sf_centers = np.sqrt(sf_bin_edges[:-1] * sf_bin_edges[1:])  # Geometric mean for SF (logarithmic bins)
    product_centers = np.sqrt(np.abs(product_bin_edges[:-1] * product_bin_edges[1:]))  # Geometric mean with abs for negative values
    
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
    plot_mean_structure_functions(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                                  output_dir, base_name, args.format, args.dpi)
    
    # 2. Plot 2D histograms for selected channels
    # Pass channel-specific bins if available
    if 'sf_channel_bin_edges' in data:
        plot_2d_histograms_with_channel_bins(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                           output_dir, base_name, args.format, args.dpi)
        # 2b. Plot individual 2D histograms with power law fits
        plot_individual_2d_histograms_with_fits(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                           output_dir, base_name, args.format, args.dpi)
    else:
        plot_2d_histograms(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
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


def plot_mean_structure_functions(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges, 
                                   output_dir, base_name, fmt, dpi):
    """Plot mean structure functions vs ell for all channels with power law fits."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for different channels
    colors = plt.cm.tab20(np.linspace(0, 1, len(mag_channels)))
    
    # Fit range
    ell_min_fit = 32
    ell_max_fit = ell_centers.max() / 4
    
    for idx, channel_name in enumerate(mag_channels):
        channel_idx = idx
        
        # Get channel-specific bin centers using geometric mean for logarithmic bins
        bin_edges = sf_channel_bin_edges[channel_idx]
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
        
        # Sum over angles to get total histogram for this channel
        hist_ell_sf = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi
        
        # Compute mean
        mean_sf = np.zeros(len(ell_centers))
        
        for i in range(len(ell_centers)):
            if hist_ell_sf[i].sum() > 0:
                # Compute weighted mean using channel-specific bin centers
                mean_sf[i] = np.average(bin_centers, weights=hist_ell_sf[i])
        
        # Plot only non-zero values
        mask = mean_sf > 0
        if np.any(mask):
            # Fit power law
            fit_mask = mask & (ell_centers >= ell_min_fit) & (ell_centers <= ell_max_fit)
            if np.sum(fit_mask) > 2:
                # Perform linear fit in log space
                log_ell_fit = np.log10(ell_centers[fit_mask])
                log_sf_fit = np.log10(mean_sf[fit_mask])
                
                # Linear regression
                coeffs = np.polyfit(log_ell_fit, log_sf_fit, 1)
                slope = coeffs[0]
                
                # Create label with power law
                label = f'{get_channel_label(channel_name)} $\propto \ell^{{{slope:.2f}}}$'
            else:
                label = get_channel_label(channel_name)
            
            # Plot data
            ax.plot(ell_centers[mask], mean_sf[mask], 'o-', 
                   color=colors[idx], markersize=4, linewidth=1.5, label=label)
    
    # Add shaded region for fit range
    ax.axvspan(ell_min_fit, ell_max_fit, alpha=0.1, color='gray')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Mean Structure Function')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_title('Mean Structure Functions for All Channels')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_mean_structure_functions.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_2d_histograms(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
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
        
        # Get channel-specific bin centers using geometric mean for logarithmic bins
        bin_edges = sf_channel_bin_edges[channel_idx]
        sf_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
        
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
        ax.set_ylabel(get_channel_label(channel_name))
        ax.set_title(f'2D Histogram: {get_channel_label(channel_name)} vs $\ell$')
        
        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, label='Counts')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_2d_histograms.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_2d_histograms_with_channel_bins(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                                          output_dir, base_name, fmt, dpi):
    """Plot normalized 2D histograms for all channels with statistical moments."""
    fig, axes = plt.subplots(6, 2, figsize=(12, 20))
    axes = axes.flatten()
    
    for channel_idx, channel_name in enumerate(mag_channels):
        if channel_idx >= len(axes) - 1:  # Skip if we run out of axes
            break
            
        ax = axes[channel_idx]
        
        # Get channel-specific bin edges and centers
        bin_edges = sf_channel_bin_edges[channel_idx]
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for logarithmic bins
        
        # Sum over angles to get 2D histogram
        hist_2d = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi
        
        # Normalize each ell bin by total counts in that ell
        hist_2d_norm = hist_2d.copy().astype(float)
        
        # Calculate statistical moments for each ell
        median_sf = np.zeros(len(ell_centers))
        mean_sf = np.zeros(len(ell_centers))
        second_moment = np.zeros(len(ell_centers))
        
        for i in range(len(ell_centers)):
            total_counts = hist_2d[i].sum()
            if total_counts > 0:
                hist_2d_norm[i] = hist_2d[i] / total_counts
                
                # Calculate median (cumulative sum approach)
                cumsum = np.cumsum(hist_2d[i])
                median_idx = np.searchsorted(cumsum, 0.5 * total_counts)
                if median_idx < len(bin_centers):
                    median_sf[i] = bin_centers[median_idx]
                
                # Calculate mean (first moment)
                mean_sf[i] = np.average(bin_centers, weights=hist_2d[i])
                
                # Calculate second moment
                second_moment[i] = np.average(bin_centers**2, weights=hist_2d[i])
        
        # Create meshgrid for plotting
        ell_mesh, sf_mesh = np.meshgrid(ell_centers, bin_centers)
        
        # Plot normalized histogram
        pcm = ax.pcolormesh(ell_mesh, sf_mesh, hist_2d_norm.T, 
                            norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
        
        # Overplot statistical moments
        mask = mean_sf > 0
        if np.any(mask):
            ax.plot(ell_centers[mask], median_sf[mask], 'w-', linewidth=2, label='Median')
            ax.plot(ell_centers[mask], mean_sf[mask], 'r-', linewidth=2, label='SF_1')
            ax.plot(ell_centers[mask], np.sqrt(second_moment[mask]), 'y-', linewidth=2, label='SF_2')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$' if channel_idx >= 10 else '')
        ax.set_ylabel(get_channel_label(channel_name))
        ax.set_title(get_channel_label(channel_name), fontsize=10)
        
        # Add legend only to first plot
        if channel_idx == 0:
            ax.legend(loc='upper left', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, label='P' if channel_idx % 2 == 0 else '')
        cbar.ax.tick_params(labelsize=8)
    
    # Hide the last empty axis
    if len(mag_channels) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_2d_histograms_normalized.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_individual_2d_histograms_with_fits(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                                            output_dir, base_name, fmt, dpi):
    """Create individual 2D histogram plots for each channel with power law fits."""
    
    # Fit range
    ell_min_fit = 32
    ell_max_fit = ell_centers.max() / 4
    
    for channel_idx, channel_name in enumerate(mag_channels):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get channel-specific bin edges and centers
        bin_edges = sf_channel_bin_edges[channel_idx]
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for logarithmic bins
        
        # Sum over angles to get 2D histogram
        hist_2d = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi
        
        # Normalize each ell bin by total counts in that ell
        hist_2d_norm = hist_2d.copy().astype(float)
        
        # Calculate statistical moments for each ell
        median_sf = np.zeros(len(ell_centers))
        mean_sf = np.zeros(len(ell_centers))
        second_moment = np.zeros(len(ell_centers))
        
        for i in range(len(ell_centers)):
            total_counts = hist_2d[i].sum()
            if total_counts > 0:
                hist_2d_norm[i] = hist_2d[i] / total_counts
                
                # Calculate median (cumulative sum approach)
                cumsum = np.cumsum(hist_2d[i])
                median_idx = np.searchsorted(cumsum, 0.5 * total_counts)
                if median_idx < len(bin_centers):
                    median_sf[i] = bin_centers[median_idx]
                
                # Calculate mean (first moment)
                mean_sf[i] = np.average(bin_centers, weights=hist_2d[i])
                
                # Calculate second moment
                second_moment[i] = np.average(bin_centers**2, weights=hist_2d[i])
        
        # Create meshgrid for plotting
        ell_mesh, sf_mesh = np.meshgrid(ell_centers, bin_centers)
        
        # Plot normalized histogram
        pcm = ax.pcolormesh(ell_mesh, sf_mesh, hist_2d_norm.T, 
                            norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
        
        # Fit power law to mean
        mask = mean_sf > 0
        fit_mask = mask & (ell_centers >= ell_min_fit) & (ell_centers <= ell_max_fit)
        
        if np.sum(fit_mask) > 2:
            # Perform linear fit in log space
            log_ell_fit = np.log10(ell_centers[fit_mask])
            log_sf_fit = np.log10(mean_sf[fit_mask])
            
            # Linear regression
            coeffs = np.polyfit(log_ell_fit, log_sf_fit, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Plot fitted line
            ell_fit = np.logspace(np.log10(ell_min_fit), np.log10(ell_max_fit), 50)
            sf_fit = 10**(intercept) * ell_fit**slope
            ax.plot(ell_fit, sf_fit, 'k--', linewidth=2, alpha=0.7, 
                   label=f'Fit: $\propto \ell^{{{slope:.2f}}}$')
        
        # Overplot statistical moments
        if np.any(mask):
            ax.plot(ell_centers[mask], median_sf[mask], 'w-', linewidth=2, label='Median')
            ax.plot(ell_centers[mask], mean_sf[mask], 'r-', linewidth=2, label='SF_1')
            ax.plot(ell_centers[mask], np.sqrt(second_moment[mask]), 'y-', linewidth=2, label='SF_2')
        
        # Add shaded region for fit range
        ax.axvspan(ell_min_fit, ell_max_fit, alpha=0.1, color='gray')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(get_channel_label(channel_name))
        ax.set_title(f'2D Histogram: {get_channel_label(channel_name)}')
        ax.legend(loc='upper left', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, label='Normalized Probability')
        
        plt.tight_layout()
        # Create filename with sanitized channel name
        safe_name = channel_name.replace('_', '').lower()
        filename = output_dir / f"{base_name}_2d_histogram_{safe_name}.{fmt}"
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
    
    # Use centers for contourf - simpler and avoids dimension mismatch
    # Plot 2D histogram with ell on x-axis and theta on y-axis
    levels = np.logspace(0, np.log10(hist_2d_theta.max()), num=20)
    pcm = ax.contourf(ell_centers, theta_degrees, hist_2d_theta.T,
                      levels=levels, norm=colors.LogNorm(vmin=1, vmax=hist_2d_theta.max()),
                      cmap='viridis')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\theta$ (degrees)')
    ax.set_title(f'2D Histogram: {get_channel_label(channel_name)} vs $\ell$ and $\\theta$ (summed over $\phi$)')
    
    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, label='Counts')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_angular_distribution_2d.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_cross_products(hist_other, other_channels, ell_centers, product_centers,
                        output_dir, base_name, fmt, dpi):
    """Plot ratios of cross-products to their corresponding MAG products with power law fits."""
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
        # Exclude first and last 4 bins to avoid out-of-bounds values
        mean_cross = np.zeros(len(ell_centers))
        mean_mag = np.zeros(len(ell_centers))
        
        for i in range(len(ell_centers)):
            # Use interior bins only (exclude first and last 4 bins)
            if len(hist_other[cross_idx][i]) > 8:
                interior_hist_cross = hist_other[cross_idx][i][4:-4]
                interior_centers = product_centers[4:-4]
                if interior_hist_cross.sum() > 0:
                    mean_cross[i] = np.average(interior_centers, weights=interior_hist_cross)
            
            if len(hist_other[mag_idx][i]) > 8:
                interior_hist_mag = hist_other[mag_idx][i][4:-4]
                interior_centers = product_centers[4:-4]
                if interior_hist_mag.sum() > 0:
                    mean_mag[i] = np.average(interior_centers, weights=interior_hist_mag)
        
        # Compute ratio
        mask = (mean_mag > 0) & (mean_cross > 0)
        if np.any(mask):
            ratio = mean_cross[mask] / mean_mag[mask]
            ell_valid = ell_centers[mask]
            
            # Plot ratio
            ax.plot(ell_valid, ratio, 'o-', markersize=6, label='Data')
            
            # Fit power law from ell ~ 32 to ell ~ max(ell)/4
            ell_min_fit = 32
            ell_max_fit = ell_centers.max() / 4
            fit_mask = (ell_valid >= ell_min_fit) & (ell_valid <= ell_max_fit)
            
            if np.sum(fit_mask) > 2:  # Need at least 3 points for a good fit
                # Perform linear fit in log space
                log_ell_fit = np.log10(ell_valid[fit_mask])
                log_ratio_fit = np.log10(ratio[fit_mask])
                
                # Linear regression
                coeffs = np.polyfit(log_ell_fit, log_ratio_fit, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                
                # Create fit line
                ell_fit_range = np.logspace(np.log10(ell_min_fit), np.log10(ell_max_fit), 100)
                ratio_fit = 10**(intercept) * ell_fit_range**slope
                
                # Plot fit
                ax.plot(ell_fit_range, ratio_fit, 'r--', linewidth=2, 
                       label=f'Power law: $\ell^{{{slope:.2f}}}$')
                
                # Add shaded region to show fit range
                ax.axvspan(ell_min_fit, ell_max_fit, alpha=0.1, color='gray')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(f'{get_channel_label(cross_name)} / {get_channel_label(mag_name)}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f'Ratio: {get_channel_label(cross_name)} / {get_channel_label(mag_name)}')
    
    plt.tight_layout()
    filename = output_dir / f"{base_name}_cross_product_ratios.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


if __name__ == "__main__":
    exit(main())