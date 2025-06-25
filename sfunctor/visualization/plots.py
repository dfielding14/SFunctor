#!/usr/bin/env python3
"""Visualization tools for structure function analysis results.

This module provides functions to load and visualize structure function
results from the simplified analysis pipeline. It creates publication-quality
plots showing structure functions versus displacement and probability distributions.

Features:
- Log-log plots of structure functions vs displacement
- Theoretical scaling comparisons (Kolmogorov 1/3, etc.)
- Probability distribution histograms
- Automatic plot saving

Usage:
    python visualize_sf_results.py [results_file.npz]
    
If no file is specified, it will look for simple_sf_*.npz files in slice_data/.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_structure_functions(sf_file):
    """Plot structure function results from simplified analysis.
    
    Creates a two-panel figure showing:
    1. Structure functions vs displacement (log-log plot)
    2. Probability distributions of structure function values
    
    Parameters
    ----------
    sf_file : str or Path
        Path to the .npz file containing structure function results.
        Expected to contain:
        - 'displacements': Array of displacement magnitudes
        - 'dv_values': Velocity structure function values
        - 'dB_values': Magnetic field structure function values
        - 'drho_values': Density structure function values
        - 'meta': Metadata dictionary
    
    Returns
    -------
    None
        Displays the plot and saves it as a PNG file.
    
    Notes
    -----
    The function bins the data logarithmically to compute mean structure
    functions at different scales. It also overlays theoretical scaling
    predictions for comparison.
    """
    
    print(f"Loading structure function results from: {sf_file}")
    
    # Load the data
    data = np.load(sf_file, allow_pickle=True)
    
    displacements = data['displacements']
    dv_values = data['dv_values']
    dB_values = data['dB_values']
    drho_values = data['drho_values']
    meta = data['meta'].item()
    
    print(f"Metadata: {meta}")
    print(f"Number of samples: {len(displacements)}")
    print(f"Displacement range: {displacements.min():.3f} - {displacements.max():.3f}")
    
    # Create bins for displacement
    r_bins = np.logspace(np.log10(displacements.min()), np.log10(displacements.max()), 20)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    # Bin the data and compute mean structure functions
    dv_binned = []
    dB_binned = []
    drho_binned = []
    
    for i in range(len(r_bins) - 1):
        mask = (displacements >= r_bins[i]) & (displacements < r_bins[i+1])
        if np.sum(mask) > 0:
            dv_binned.append(np.mean(dv_values[mask]))
            dB_binned.append(np.mean(dB_values[mask]))
            drho_binned.append(np.mean(drho_values[mask]))
        else:
            dv_binned.append(np.nan)
            dB_binned.append(np.nan)
            drho_binned.append(np.nan)
    
    dv_binned = np.array(dv_binned)
    dB_binned = np.array(dB_binned)
    drho_binned = np.array(drho_binned)
    
    # Remove NaN values
    valid = ~(np.isnan(dv_binned) | np.isnan(dB_binned) | np.isnan(drho_binned))
    r_centers = r_centers[valid]
    dv_binned = dv_binned[valid]
    dB_binned = dB_binned[valid]
    drho_binned = drho_binned[valid]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Structure functions vs displacement
    ax1.loglog(r_centers, dv_binned, 'o-', label='Velocity |δv|', color='blue')
    ax1.loglog(r_centers, dB_binned, 's-', label='Magnetic |δB|', color='red')
    ax1.loglog(r_centers, drho_binned, '^-', label='Density |δρ|', color='green')
    
    # Add theoretical scaling lines
    r_theory = np.linspace(r_centers.min(), r_centers.max(), 100)
    
    # Kolmogorov 1/3 scaling for velocity
    if len(dv_binned) > 0:
        C_v = dv_binned[len(dv_binned)//2] / (r_centers[len(r_centers)//2]**(1/3))
        ax1.loglog(r_theory, C_v * r_theory**(1/3), '--', alpha=0.7, color='blue', label='r^(1/3)')
    
    # 1/2 scaling for magnetic field
    if len(dB_binned) > 0:
        C_B = dB_binned[len(dB_binned)//2] / (r_centers[len(r_centers)//2]**(1/2))
        ax1.loglog(r_theory, C_B * r_theory**(1/2), '--', alpha=0.7, color='red', label='r^(1/2)')
    
    ax1.set_xlabel('Displacement r')
    ax1.set_ylabel('Structure Function')
    ax1.set_title(f'Structure Functions - {meta["slice"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of structure function values
    ax2.hist(dv_values, bins=50, alpha=0.7, label='Velocity |δv|', color='blue', density=True)
    ax2.hist(dB_values, bins=50, alpha=0.7, label='Magnetic |δB|', color='red', density=True)
    ax2.hist(drho_values, bins=50, alpha=0.7, label='Density |δρ|', color='green', density=True)
    
    ax2.set_xlabel('Structure Function Value')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Structure Function Distributions')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = sf_file.replace('.npz', '_plot.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    
    plt.show()

def main():
    """Command-line interface for structure function visualization.
    
    Accepts an optional command-line argument specifying the results file.
    If no argument is provided, searches for simple_sf_*.npz files in the
    slice_data directory and uses the first one found.
    
    Returns
    -------
    None
        Exits after creating and displaying the plots.
    """
    if len(sys.argv) > 1:
        sf_file = sys.argv[1]
    else:
        # Find the first simple SF file
        sf_files = list(Path("slice_data").glob("simple_sf_*.npz"))
        if not sf_files:
            print("No structure function files found!")
            return
        sf_file = str(sf_files[0])
    
    plot_structure_functions(sf_file)

if __name__ == "__main__":
    main() 