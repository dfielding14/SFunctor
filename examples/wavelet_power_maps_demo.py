#!/usr/bin/env python
"""
Demonstration of Wavelet Power Spectrum Mapping

This example shows how to:
1. Compute spatial maps of turbulence properties
2. Identify regions with different spectral characteristics
3. Analyze scale-dependent spatial variations
4. Study anisotropic turbulence patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Import SFunctor modules
from sfunctor.io.slice_io import load_slice_npz
from sfunctor.analysis.wavelet_maps import (
    WaveletMapConfig,
    compute_wavelet_power_maps,
    compute_multiscale_maps,
    compute_anisotropic_wavelet_maps,
    compute_cross_scale_correlation_map,
    identify_turbulence_regions,
    visualize_wavelet_maps
)

def main():
    """Run wavelet power spectrum mapping demonstration."""
    
    print("="*60)
    print("WAVELET POWER SPECTRUM MAPPING DEMONSTRATION")
    print("="*60)
    
    # Check for PyWavelets
    try:
        import pywt
        print("✓ PyWavelets is installed")
    except ImportError:
        print("✗ PyWavelets not installed")
        print("  Install with: pip install PyWavelets")
        return
    
    # Load example data
    slice_dir = Path("slice_data")
    if not slice_dir.exists():
        print("Error: slice_data directory not found")
        print("Creating synthetic turbulence field for demonstration...")
        field = create_synthetic_turbulence(512, 512)
    else:
        slice_files = list(slice_dir.glob("*.npz"))
        if slice_files:
            print(f"Loading: {slice_files[0].name}")
            data = load_slice_npz(slice_files[0], stride=1)
            # Use velocity magnitude as example field
            field = np.sqrt(data['v_x']**2 + data['v_y']**2 + data['v_z']**2)
            print(f"  Field shape: {field.shape}")
        else:
            print("No slice files found, creating synthetic field...")
            field = create_synthetic_turbulence(512, 512)
    
    print()
    
    # 1. Basic wavelet power spectrum mapping
    print("1. Computing wavelet power spectrum maps...")
    print("-" * 40)
    
    config = WaveletMapConfig(
        outer_scale=64.0,    # Analyze scales up to 64 pixels
        inner_scale=2.0,     # Minimum scale of 2 pixels
        n_scales=32,         # 32 logarithmic scales
        window_size=64,      # 64x64 local windows
        window_overlap=0.5,  # 50% overlap
        wavelet='morl',      # Morlet wavelet
        detrend='linear',    # Remove linear trends
        fit_method='robust'  # Robust power law fitting
    )
    
    results = compute_wavelet_power_maps(field, config, verbose=True)
    
    # Visualize results
    print("\n  Creating visualization...")
    visualize_wavelet_maps(results, field, save_path=Path("wavelet_maps_basic.png"))
    
    # 2. Identify turbulence regions
    print("\n2. Identifying turbulence regions...")
    print("-" * 40)
    
    regions = identify_turbulence_regions(
        results,
        slope_threshold=(-2.0, -1.3),  # Custom range
        r_squared_threshold=0.7
    )
    
    print("  Region statistics:")
    for name, stats in regions['statistics'].items():
        print(f"    {name:12s}: {stats['fraction']*100:5.1f}% of area, "
              f"slope={stats['mean_slope']:.2f}±{stats['std_slope']:.2f}")
    
    # Visualize regions
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for ax, (name, mask) in zip(axes.flat, regions['regions'].items()):
        ax.imshow(mask, cmap='RdYlBu_r', origin='lower')
        ax.set_title(f'{name.capitalize()} Regions')
        ax.axis('off')
    
    # Remove unused axes
    for i in range(len(regions['regions']), 6):
        fig.delaxes(axes.flat[i])
    
    plt.suptitle('Identified Turbulence Regions', fontsize=14)
    plt.tight_layout()
    plt.savefig('turbulence_regions.png', dpi=150, bbox_inches='tight')
    print("  Saved turbulence_regions.png")
    
    # 3. Multi-scale analysis
    print("\n3. Multi-scale analysis...")
    print("-" * 40)
    
    scale_ranges = [
        (2, 8),    # Small scales
        (8, 32),   # Medium scales
        (32, 128)  # Large scales
    ]
    
    print(f"  Analyzing {len(scale_ranges)} scale ranges:")
    for inner, outer in scale_ranges:
        print(f"    - Scales {inner}-{outer} pixels")
    
    multiscale_results = compute_multiscale_maps(field, scale_ranges, config)
    
    # Compare slopes across scales
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, ((inner, outer), key) in zip(axes, zip(scale_ranges, multiscale_results.keys())):
        slope_map = multiscale_results[key]['slope_map']
        im = ax.imshow(slope_map, cmap='RdBu_r', origin='lower',
                      vmin=-3, vmax=-1)
        ax.set_title(f'Slopes: {inner}-{outer} px\n(mean={np.nanmean(slope_map):.2f})')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Scale-Dependent Spectral Slopes', fontsize=14)
    plt.tight_layout()
    plt.savefig('multiscale_slopes.png', dpi=150, bbox_inches='tight')
    print("  Saved multiscale_slopes.png")
    
    # 4. Anisotropic analysis
    print("\n4. Anisotropic wavelet analysis...")
    print("-" * 40)
    
    # Use smaller field for speed
    field_small = field[::2, ::2]
    
    config_aniso = WaveletMapConfig(
        outer_scale=32.0,
        inner_scale=2.0,
        window_size=32,
        window_overlap=0.5
    )
    
    aniso_results = compute_anisotropic_wavelet_maps(
        field_small, 
        config_aniso,
        angles=np.array([0, 45, 90, 135]) * np.pi / 180
    )
    
    print(f"  Mean anisotropy ratio: {np.nanmean(aniso_results['anisotropy_ratio']):.2f}")
    print(f"  Max anisotropy ratio: {np.nanmax(aniso_results['anisotropy_ratio']):.2f}")
    
    # Visualize anisotropy
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Anisotropy ratio
    ax = axes[0]
    im = ax.imshow(aniso_results['anisotropy_ratio'], cmap='hot', origin='lower')
    ax.set_title('Anisotropy Ratio\n(max slope / min slope)')
    plt.colorbar(im, ax=ax)
    
    # Preferred direction
    ax = axes[1]
    im = ax.imshow(aniso_results['preferred_direction'] * 180/np.pi, 
                  cmap='hsv', origin='lower', vmin=0, vmax=180)
    ax.set_title('Preferred Direction (degrees)')
    plt.colorbar(im, ax=ax)
    
    # Alignment strength
    ax = axes[2]
    im = ax.imshow(aniso_results['alignment_strength'], cmap='viridis', origin='lower')
    ax.set_title('Alignment Strength')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Anisotropic Turbulence Properties', fontsize=14)
    plt.tight_layout()
    plt.savefig('anisotropic_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved anisotropic_analysis.png")
    
    # 5. Cross-scale correlation
    print("\n5. Cross-scale correlation analysis...")
    print("-" * 40)
    
    scale1 = 4.0
    scale2 = 16.0
    print(f"  Computing correlation between scales {scale1} and {scale2}")
    
    cross_scale = compute_cross_scale_correlation_map(
        field_small, scale1, scale2, config_aniso
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Correlation
    ax = axes[0]
    im = ax.imshow(cross_scale['correlation_map'], cmap='RdBu_r', 
                  origin='lower', vmin=-1, vmax=1)
    ax.set_title(f'Correlation\n({scale1:.0f} vs {scale2:.0f} px)')
    plt.colorbar(im, ax=ax)
    
    # Coherence
    ax = axes[1]
    im = ax.imshow(cross_scale['coherence_map'], cmap='magma', 
                  origin='lower', vmin=0, vmax=1)
    ax.set_title('Coherence')
    plt.colorbar(im, ax=ax)
    
    # Phase
    ax = axes[2]
    im = ax.imshow(cross_scale['phase_map'], cmap='hsv', 
                  origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Phase Difference')
    plt.colorbar(im, ax=ax, label='radians')
    
    plt.suptitle('Cross-Scale Correlations', fontsize=14)
    plt.tight_layout()
    plt.savefig('cross_scale_correlation.png', dpi=150, bbox_inches='tight')
    print("  Saved cross_scale_correlation.png")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Spectral slope statistics:")
    print(f"  Mean: {np.nanmean(results['slope_map']):.3f}")
    print(f"  Std:  {np.nanstd(results['slope_map']):.3f}")
    print(f"  Min:  {np.nanmin(results['slope_map']):.3f}")
    print(f"  Max:  {np.nanmax(results['slope_map']):.3f}")
    
    print(f"\nFit quality (R²):")
    print(f"  Mean: {np.nanmean(results['r_squared_map']):.3f}")
    print(f"  > 0.8: {np.sum(results['r_squared_map'] > 0.8) / results['r_squared_map'].size * 100:.1f}%")
    print(f"  > 0.9: {np.sum(results['r_squared_map'] > 0.9) / results['r_squared_map'].size * 100:.1f}%")
    
    print(f"\nSpatial variations:")
    slope_std = np.nanstd(results['slope_map'])
    slope_mean = np.nanmean(results['slope_map'])
    print(f"  Coefficient of variation: {abs(slope_std/slope_mean):.3f}")
    
    # Check for Kolmogorov scaling
    kolmogorov_fraction = np.sum(np.abs(results['slope_map'] + 5/3) < 0.2) / results['slope_map'].size
    print(f"\nKolmogorov-like regions (slope ≈ -5/3):")
    print(f"  {kolmogorov_fraction*100:.1f}% of area")
    
    print("\n" + "="*60)
    print("Analysis complete! Generated files:")
    print("  - wavelet_maps_basic.png")
    print("  - turbulence_regions.png")
    print("  - multiscale_slopes.png")
    print("  - anisotropic_analysis.png")
    print("  - cross_scale_correlation.png")
    print("="*60)


def create_synthetic_turbulence(ny: int, nx: int, 
                               spectral_slope: float = -5/3) -> np.ndarray:
    """Create synthetic turbulence field for testing.
    
    Args:
        ny, nx: Field dimensions
        spectral_slope: Power spectrum slope
        
    Returns:
        2D turbulent field
    """
    # Create field in Fourier space
    ky = np.fft.fftfreq(ny, d=1.0).reshape(-1, 1)
    kx = np.fft.fftfreq(nx, d=1.0).reshape(1, -1)
    k = np.sqrt(ky**2 + kx**2)
    
    # Power spectrum
    k[0, 0] = 1  # Avoid division by zero
    power = k**(spectral_slope)
    power[0, 0] = 0
    
    # Random phases
    phases = np.random.uniform(-np.pi, np.pi, (ny, nx))
    
    # Create complex field
    field_k = np.sqrt(power) * np.exp(1j * phases)
    
    # Ensure Hermitian symmetry for real field
    field_k[0, 0] = 0
    for i in range(1, ny//2):
        for j in range(1, nx//2):
            field_k[-i, -j] = np.conj(field_k[i, j])
    
    # Transform to real space
    field = np.real(np.fft.ifft2(field_k))
    
    # Add some spatial variation in intensity
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y)
    modulation = 1 + 0.3 * np.sin(2*X) * np.cos(2*Y)
    
    field *= modulation
    
    # Normalize
    field = (field - field.mean()) / field.std()
    
    print(f"  Created synthetic {ny}×{nx} turbulence field")
    print(f"  Spectral slope: {spectral_slope:.2f}")
    
    return field


if __name__ == "__main__":
    main()