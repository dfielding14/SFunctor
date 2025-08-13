#!/usr/bin/env python
"""
Advanced Analysis Example

Demonstrates the new scientific features:
1. Time-series analysis
2. Scale-dependent anisotropy
3. Cross-correlation between fields
4. Wavelet decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Import SFunctor modules
from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.analysis.time_series import (
    analyze_time_series, 
    compute_anisotropy_measures,
    compute_scaling_exponents
)
from sfunctor.analysis.anisotropy import (
    compute_scale_dependent_anisotropy,
    create_anisotropy_spectrogram
)
from sfunctor.analysis.cross_correlation import (
    compute_field_correlations,
    compute_scale_dependent_transfer,
    compute_nonlinear_coupling,
    compute_mutual_information
)

# Try to import wavelet module
try:
    from sfunctor.analysis.wavelet import (
        wavelet_decompose_sf,
        continuous_wavelet_analysis,
        wavelet_coherence_analysis,
        multifractal_wavelet_analysis
    )
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    warnings.warn("Wavelet analysis not available. Install PyWavelets: pip install PyWavelets")


def demo_time_series_analysis():
    """Demonstrate time-series analysis across multiple snapshots."""
    print("\n" + "="*60)
    print("TIME-SERIES ANALYSIS")
    print("="*60)
    
    # Find slice files
    slice_dir = Path("slice_data")
    if not slice_dir.exists():
        print("Error: slice_data directory not found")
        return None
    
    slice_files = sorted(slice_dir.glob("*.npz"))[:5]  # Use first 5 files
    
    if len(slice_files) < 2:
        print("Need at least 2 slice files for time-series analysis")
        return None
    
    print(f"Analyzing {len(slice_files)} snapshots...")
    
    # Configuration for fast testing
    config = {
        'stride': 4,
        'n_disp_total': 50,
        'n_random_subsamples': 1000,
        'n_theta_bins': 8,
        'n_phi_bins': 16,
    }
    
    # Run time-series analysis
    ts_results = analyze_time_series(
        slice_files,
        config=config,
        verbose=True
    )
    
    # Plot evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Turbulence Evolution', fontsize=16)
    
    times = ts_results['times']
    
    # Plot anisotropy evolution
    ax = axes[0, 0]
    aniso = ts_results['anisotropy_series']
    if len(aniso.shape) > 1:
        ax.plot(times, aniso[:, 0], 'o-', label='Parallel/Perp Ratio')
    ax.set_xlabel('Time')
    ax.set_ylabel('Anisotropy')
    ax.set_title('Anisotropy Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot energy evolution
    ax = axes[0, 1]
    energy = ts_results['energy_series']
    if isinstance(energy, np.ndarray) and len(energy) > 0:
        if len(energy.shape) == 2:
            for key in ['velocity', 'magnetic', 'density']:
                if key in energy.dtype.names:
                    ax.plot(times, energy[key], 'o-', label=key.capitalize())
        elif len(energy.shape) == 1:
            ax.plot(times, energy, 'o-', label='Total Energy')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot scaling exponents
    ax = axes[1, 0]
    scaling = ts_results['scaling_exponents']
    if len(scaling) > 0 and 'zeta_2' in scaling[0]:
        zeta_2 = [s['zeta_2'][0] if len(s['zeta_2']) > 0 else np.nan 
                  for s in scaling]
        ax.plot(times, zeta_2, 'o-', label='ζ₂ (velocity)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Scaling Exponent')
    ax.set_title('Scaling Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot structure function evolution
    ax = axes[1, 1]
    hist_series = ts_results['hist_mag_series']
    # Show velocity SF at middle scale
    mid_ell = hist_series.shape[2] // 2
    sf_evolution = hist_series[:, 0, mid_ell].sum(axis=(1, 2, 3))
    ax.plot(times, sf_evolution, 'o-')
    ax.set_xlabel('Time')
    ax.set_ylabel('SF Amplitude')
    ax.set_title('Structure Function Evolution (mid-scale)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved time_series_analysis.png")
    
    return ts_results


def demo_anisotropy_analysis():
    """Demonstrate scale-dependent anisotropy analysis."""
    print("\n" + "="*60)
    print("SCALE-DEPENDENT ANISOTROPY ANALYSIS")
    print("="*60)
    
    # Find a slice file
    slice_dir = Path("slice_data")
    slice_files = list(slice_dir.glob("*.npz"))
    
    if not slice_files:
        print("No slice files found")
        return None
    
    slice_file = slice_files[0]
    print(f"Analyzing: {slice_file.name}")
    
    # Quick analysis
    results = analyze_slice(
        slice_file,
        stride=4,
        n_disp_total=100,
        n_random_subsamples=2000,
        n_theta_bins=16,
        n_phi_bins=32
    )
    
    # Compute anisotropy measures
    methods = ['ratio', 'variance', 'entropy', 'alignment']
    aniso_results = {}
    
    for method in methods:
        aniso = compute_scale_dependent_anisotropy(results, method=method)
        aniso_results[method] = aniso
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Scale-Dependent Anisotropy', fontsize=16)
    
    # Plot parallel/perpendicular ratio
    ax = axes[0, 0]
    aniso = aniso_results['ratio']
    ell_centers = aniso['ell_centers']
    for ch in range(min(3, aniso['parallel_perp_ratio'].shape[0])):
        ax.semilogx(ell_centers, aniso['parallel_perp_ratio'][ch], 
                   label=f'Channel {ch}')
    ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('SF∥ / SF⊥')
    ax.set_title('Anisotropy Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot angular variance
    ax = axes[0, 1]
    aniso = aniso_results['variance']
    for ch in range(min(3, aniso['angular_variance'].shape[0])):
        ax.semilogx(ell_centers, aniso['angular_variance'][ch], 
                   label=f'Channel {ch}')
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Angular Variance')
    ax.set_title('Angular Distribution Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot entropy
    ax = axes[1, 0]
    aniso = aniso_results['entropy']
    for ch in range(min(3, aniso['angular_entropy'].shape[0])):
        ax.semilogx(ell_centers, aniso['angular_entropy'][ch], 
                   label=f'Channel {ch}')
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Angular Entropy (0=ordered, 1=random)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot alignment
    ax = axes[1, 1]
    aniso = aniso_results['alignment']
    for ch in range(min(3, aniso['alignment'].shape[0])):
        ax.semilogx(ell_centers, aniso['alignment'][ch], 
                   label=f'Channel {ch}')
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Alignment')
    ax.set_title('Field Alignment (with z-axis)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anisotropy_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved anisotropy_analysis.png")
    
    # Create anisotropy spectrogram
    spectrogram = create_anisotropy_spectrogram(results, channel=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Anisotropy Spectrogram (Velocity)', fontsize=16)
    
    # Plot raw spectrogram
    ax = axes[0]
    im = ax.pcolormesh(spectrogram['ell_centers'], 
                      spectrogram['theta_centers'] * 180/np.pi,
                      spectrogram['spectrogram'].T,
                      shading='auto', cmap='viridis')
    ax.set_xscale('log')
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Angle θ (degrees)')
    ax.set_title('Angular Distribution')
    plt.colorbar(im, ax=ax, label='Probability')
    
    # Plot anisotropy ratio
    ax = axes[1]
    im = ax.pcolormesh(spectrogram['ell_centers'],
                      spectrogram['theta_centers'] * 180/np.pi,
                      spectrogram['anisotropy_ratio'].T,
                      shading='auto', cmap='RdBu_r', 
                      vmin=-1, vmax=1)
    ax.set_xscale('log')
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Angle θ (degrees)')
    ax.set_title('Anisotropy Ratio')
    plt.colorbar(im, ax=ax, label='(SF - mean) / mean')
    
    plt.tight_layout()
    plt.savefig('anisotropy_spectrogram.png', dpi=150, bbox_inches='tight')
    print("Saved anisotropy_spectrogram.png")
    
    return aniso_results


def demo_cross_correlation():
    """Demonstrate cross-correlation analysis between fields."""
    print("\n" + "="*60)
    print("CROSS-CORRELATION ANALYSIS")
    print("="*60)
    
    # Get a slice
    slice_dir = Path("slice_data")
    slice_files = list(slice_dir.glob("*.npz"))
    
    if not slice_files:
        print("No slice files found")
        return None
    
    slice_file = slice_files[0]
    print(f"Analyzing: {slice_file.name}")
    
    # Quick analysis
    results = analyze_slice(
        slice_file,
        stride=4,
        n_disp_total=100,
        n_random_subsamples=2000
    )
    
    # Compute correlations
    field_pairs = [(0, 1), (0, 2), (1, 2)]  # vel-mag, vel-dens, mag-dens
    correlations = compute_field_correlations(results, field_pairs=field_pairs)
    
    # Compute transfer functions
    transfer = compute_scale_dependent_transfer(results, 
                                               source_channel=0, 
                                               target_channel=1)
    
    # Compute nonlinear coupling
    coupling = compute_nonlinear_coupling(results, triplet=(0, 1, 7))
    
    # Compute mutual information
    mi_results = compute_mutual_information(results, channel_pairs=field_pairs)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cross-Correlation Analysis', fontsize=16)
    
    # Plot scale-dependent correlations
    ax = axes[0, 0]
    ell_centers = correlations['ell_centers']
    labels = ['Vel-Mag', 'Vel-Dens', 'Mag-Dens']
    for (ch1, ch2), label in zip(field_pairs, labels):
        name1 = ['velocity', 'magnetic', 'density'][ch1]
        name2 = ['velocity', 'magnetic', 'density'][ch2]
        key = f'{name1}_{name2}_scale'
        if key in correlations:
            ax.semilogx(ell_centers, correlations[key], 'o-', label=label)
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Correlation')
    ax.set_title('Field Correlations vs Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot transfer function
    ax = axes[0, 1]
    if 'transfer_function' in transfer:
        ax.semilogx(transfer['ell_centers'], transfer['transfer_function'], 'o-')
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Transfer Function')
    ax.set_title('Velocity → Magnetic Transfer')
    ax.grid(True, alpha=0.3)
    
    # Plot triadic coupling
    ax = axes[1, 0]
    if 'triadic_strength' in coupling:
        ax.semilogx(coupling['ell_centers'], coupling['triadic_strength'], 'o-')
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Coupling Strength')
    ax.set_title('Nonlinear Triadic Coupling (V-B-J)')
    ax.grid(True, alpha=0.3)
    
    # Plot mutual information
    ax = axes[1, 1]
    for (ch1, ch2), label in zip(field_pairs, labels):
        key = f'nmi_ch{ch1}_ch{ch2}'
        if key in mi_results:
            ax.semilogx(mi_results['ell_centers'], mi_results[key], 'o-', label=label)
    ax.set_xlabel('Scale ℓ')
    ax.set_ylabel('Normalized MI')
    ax.set_title('Mutual Information')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_correlation_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved cross_correlation_analysis.png")
    
    return correlations


def demo_wavelet_analysis():
    """Demonstrate wavelet-based decomposition."""
    if not WAVELET_AVAILABLE:
        print("\n" + "="*60)
        print("WAVELET ANALYSIS")
        print("="*60)
        print("PyWavelets not installed. Skipping wavelet analysis.")
        print("Install with: pip install PyWavelets")
        return None
    
    print("\n" + "="*60)
    print("WAVELET DECOMPOSITION ANALYSIS")
    print("="*60)
    
    # Get a slice
    slice_dir = Path("slice_data")
    slice_files = list(slice_dir.glob("*.npz"))
    
    if not slice_files:
        print("No slice files found")
        return None
    
    slice_file = slice_files[0]
    print(f"Analyzing: {slice_file.name}")
    
    # Quick analysis
    results = analyze_slice(
        slice_file,
        stride=4,
        n_disp_total=100,
        n_random_subsamples=2000
    )
    
    # Perform wavelet decomposition
    wavelet_decomp = wavelet_decompose_sf(results, channel=0, wavelet='db4')
    
    # Continuous wavelet transform
    cwt_results = continuous_wavelet_analysis(results, channel=0)
    
    # Wavelet coherence
    coherence = wavelet_coherence_analysis(results, channel1=0, channel2=1)
    
    # Multifractal analysis
    multifractal = multifractal_wavelet_analysis(results, channel=0)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Wavelet Analysis', fontsize=16)
    
    # Plot original SF
    ax = axes[0, 0]
    im = ax.imshow(wavelet_decomp['original'], aspect='auto', 
                   origin='lower', cmap='viridis')
    ax.set_xlabel('θ index')
    ax.set_ylabel('ℓ index')
    ax.set_title('Original Structure Function')
    plt.colorbar(im, ax=ax)
    
    # Plot wavelet energy
    ax = axes[0, 1]
    ax.bar(range(len(wavelet_decomp['energy_per_scale'])), 
           wavelet_decomp['energy_per_scale'])
    ax.set_xlabel('Wavelet Level')
    ax.set_ylabel('Energy Fraction')
    ax.set_title('Energy per Scale')
    ax.grid(True, alpha=0.3)
    
    # Plot CWT scalogram
    ax = axes[0, 2]
    im = ax.pcolormesh(cwt_results['ell_centers'], 
                      cwt_results['scales'],
                      cwt_results['scalogram'],
                      shading='auto', cmap='hot')
    ax.set_xlabel('Position (ℓ)')
    ax.set_ylabel('Scale')
    ax.set_title('Wavelet Scalogram')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Power')
    
    # Plot wavelet coherence
    ax = axes[1, 0]
    im = ax.pcolormesh(range(coherence['coherence'].shape[1]),
                      coherence['scales'],
                      coherence['coherence'],
                      shading='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Scale')
    ax.set_title('Wavelet Coherence (Vel-Mag)')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Coherence')
    
    # Plot phase difference
    ax = axes[1, 1]
    im = ax.pcolormesh(range(coherence['phase_difference'].shape[1]),
                      coherence['scales'],
                      coherence['phase_difference'],
                      shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel('Position')
    ax.set_ylabel('Scale')
    ax.set_title('Phase Difference')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    
    # Plot multifractal spectrum
    ax = axes[1, 2]
    ax.plot(multifractal['alpha'], multifractal['f_alpha'], 'o-')
    ax.set_xlabel('α (Singularity)')
    ax.set_ylabel('f(α)')
    ax.set_title(f'Multifractal Spectrum (width={multifractal["width"]:.2f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wavelet_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved wavelet_analysis.png")
    
    return wavelet_decomp


def main():
    """Run all demonstrations."""
    print("="*60)
    print("SFunctor Advanced Analysis Demonstrations")
    print("="*60)
    
    # Check for data
    slice_dir = Path("slice_data")
    if not slice_dir.exists():
        print("\nError: slice_data directory not found!")
        print("Please ensure you have example data in ./slice_data/")
        return
    
    slice_files = list(slice_dir.glob("*.npz"))
    print(f"\nFound {len(slice_files)} slice files")
    
    if len(slice_files) == 0:
        print("No data files found. Exiting.")
        return
    
    # Run demonstrations
    try:
        # Time-series analysis (needs multiple files)
        if len(slice_files) >= 2:
            ts_results = demo_time_series_analysis()
        else:
            print("\nSkipping time-series analysis (need at least 2 files)")
        
        # Scale-dependent anisotropy
        aniso_results = demo_anisotropy_analysis()
        
        # Cross-correlation
        corr_results = demo_cross_correlation()
        
        # Wavelet analysis
        wavelet_results = demo_wavelet_analysis()
        
        print("\n" + "="*60)
        print("All demonstrations complete!")
        print("Generated visualizations:")
        print("  - time_series_analysis.png")
        print("  - anisotropy_analysis.png")
        print("  - anisotropy_spectrogram.png")
        print("  - cross_correlation_analysis.png")
        if WAVELET_AVAILABLE:
            print("  - wavelet_analysis.png")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()