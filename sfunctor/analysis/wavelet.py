"""Wavelet-based decomposition for multi-scale turbulence analysis.

This module provides:
1. Wavelet transforms for structure functions
2. Scale-space decomposition
3. Intermittency detection via wavelet coefficients
4. Coherent structure identification
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import warnings
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets not available. Install with: pip install PyWavelets")


def wavelet_decompose_sf(
    results: Dict,
    channel: int = 0,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    mode: str = 'symmetric'
) -> Dict[str, np.ndarray]:
    """Decompose structure functions using wavelet transform.
    
    Args:
        results: Output from analyze_slice
        channel: Which channel to decompose
        wavelet: Wavelet type (e.g., 'db4', 'sym5', 'coif3')
        level: Decomposition level (None for maximum)
        mode: Signal extension mode
        
    Returns:
        Dictionary with wavelet coefficients and reconstructions
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required for wavelet analysis")
    
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    theta_bin_edges = results['theta_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    
    # Get structure function for this channel
    sf_channel = hist_mag[channel]  # Shape: (n_ell, n_theta, n_phi, n_sf)
    
    # Average over phi for 2D analysis (ell vs theta)
    sf_2d = sf_channel.sum(axis=2).sum(axis=2)  # Sum over phi and sf values
    
    # Normalize
    for i in range(sf_2d.shape[0]):
        if sf_2d[i].sum() > 0:
            sf_2d[i] = sf_2d[i] / sf_2d[i].sum()
    
    # Determine decomposition level
    if level is None:
        level = pywt.dwt_max_level(min(sf_2d.shape), wavelet)
    
    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(sf_2d, wavelet, level=level, mode=mode)
    
    # Extract coefficients
    decomposition = {
        'wavelet': wavelet,
        'level': level,
        'ell_centers': ell_centers,
        'theta_centers': theta_centers,
        'original': sf_2d,
    }
    
    # Approximation coefficients (lowest frequency)
    decomposition['approx'] = coeffs[0]
    
    # Detail coefficients at each level
    details_h = []  # Horizontal
    details_v = []  # Vertical
    details_d = []  # Diagonal
    
    for lev in range(1, len(coeffs)):
        cH, cV, cD = coeffs[lev]
        details_h.append(cH)
        details_v.append(cV)
        details_d.append(cD)
    
    decomposition['details_horizontal'] = details_h
    decomposition['details_vertical'] = details_v
    decomposition['details_diagonal'] = details_d
    
    # Compute energy at each scale
    energy_per_scale = compute_wavelet_energy(coeffs)
    decomposition['energy_per_scale'] = energy_per_scale
    
    # Identify intermittent events
    intermittency = detect_intermittency(coeffs, threshold_sigma=3.0)
    decomposition['intermittency_map'] = intermittency
    
    # Reconstruct at each level
    reconstructions = []
    for lev in range(level):
        # Zero out higher levels
        coeffs_truncated = [coeffs[0]] + coeffs[1:lev+1]
        recon = pywt.waverec2(coeffs_truncated, wavelet, mode=mode)
        # Handle size mismatch
        if recon.shape != sf_2d.shape:
            recon = recon[:sf_2d.shape[0], :sf_2d.shape[1]]
        reconstructions.append(recon)
    
    decomposition['reconstructions'] = reconstructions
    
    return decomposition


def continuous_wavelet_analysis(
    results: Dict,
    channel: int = 0,
    scales: Optional[np.ndarray] = None,
    wavelet: str = 'morl',
    angle_idx: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Perform continuous wavelet transform for scale analysis.
    
    Args:
        results: Output from analyze_slice
        channel: Which channel to analyze
        scales: Scales for CWT (None for automatic)
        wavelet: Mother wavelet ('morl', 'mexh', 'gaus8', etc.)
        angle_idx: Specific angle to analyze (None for average)
        
    Returns:
        Dictionary with CWT coefficients and scalogram
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required for wavelet analysis")
    
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    sf_bin_edges = results['sf_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    # Get structure function
    sf_channel = hist_mag[channel]
    
    # Select signal for analysis
    if angle_idx is not None:
        # Specific angle
        signal = sf_channel[:, angle_idx].sum(axis=1).sum(axis=1)
    else:
        # Average over all angles
        signal = sf_channel.sum(axis=(1, 2, 3))
    
    # Normalize
    if signal.sum() > 0:
        signal = signal / signal.sum()
    
    # Default scales
    if scales is None:
        scales = np.logspace(0, np.log10(len(signal)/2), 32)
    
    # Perform CWT
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    
    cwt_results = {
        'coefficients': coeffs,
        'scales': scales,
        'frequencies': freqs,
        'signal': signal,
        'ell_centers': ell_centers,
        'wavelet': wavelet,
    }
    
    # Compute scalogram (power)
    scalogram = np.abs(coeffs)**2
    cwt_results['scalogram'] = scalogram
    
    # Identify dominant scales
    dominant_scales = identify_dominant_scales(scalogram, scales)
    cwt_results['dominant_scales'] = dominant_scales
    
    # Compute scale-dependent intermittency
    intermittency_factor = compute_scale_intermittency(coeffs)
    cwt_results['intermittency_factor'] = intermittency_factor
    
    # Ridge detection for coherent structures
    ridges = detect_wavelet_ridges(scalogram)
    cwt_results['ridges'] = ridges
    
    return cwt_results


def wavelet_coherence_analysis(
    results: Dict,
    channel1: int = 0,
    channel2: int = 1,
    scales: Optional[np.ndarray] = None,
    wavelet: str = 'morl'
) -> Dict[str, np.ndarray]:
    """Compute wavelet coherence between two channels.
    
    Reveals scale-dependent phase relationships and correlations.
    
    Args:
        results: Output from analyze_slice
        channel1: First channel
        channel2: Second channel
        scales: CWT scales
        wavelet: Mother wavelet
        
    Returns:
        Dictionary with coherence and phase information
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required for wavelet analysis")
    
    hist_mag = results['hist_mag']
    
    # Get signals (averaged over angles)
    signal1 = hist_mag[channel1].sum(axis=(1, 2, 3))
    signal2 = hist_mag[channel2].sum(axis=(1, 2, 3))
    
    # Normalize
    if signal1.sum() > 0:
        signal1 = signal1 / signal1.sum()
    if signal2.sum() > 0:
        signal2 = signal2 / signal2.sum()
    
    # Default scales
    if scales is None:
        scales = np.logspace(0, np.log10(len(signal1)/2), 32)
    
    # Compute CWT for both signals
    coeffs1, _ = pywt.cwt(signal1, scales, wavelet)
    coeffs2, _ = pywt.cwt(signal2, scales, wavelet)
    
    # Compute cross-wavelet spectrum
    cross_spectrum = coeffs1 * np.conj(coeffs2)
    
    # Compute wavelet coherence
    # Smooth in scale and time
    smoothing_scale = 3
    from scipy.ndimage import gaussian_filter
    
    smooth_cross = gaussian_filter(np.abs(cross_spectrum), smoothing_scale)
    smooth_auto1 = gaussian_filter(np.abs(coeffs1)**2, smoothing_scale)
    smooth_auto2 = gaussian_filter(np.abs(coeffs2)**2, smoothing_scale)
    
    coherence = smooth_cross / (np.sqrt(smooth_auto1 * smooth_auto2) + 1e-10)
    
    # Compute phase difference
    phase_diff = np.angle(cross_spectrum)
    
    coherence_results = {
        'coherence': coherence,
        'phase_difference': phase_diff,
        'scales': scales,
        'cross_spectrum': cross_spectrum,
        'coeffs1': coeffs1,
        'coeffs2': coeffs2,
    }
    
    # Identify significant coherence regions
    significant_coherence = coherence > 0.8  # Threshold
    coherence_results['significant_regions'] = significant_coherence
    
    # Compute mean phase lag at each scale
    mean_phase_lag = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        if significant_coherence[i].sum() > 0:
            mean_phase_lag[i] = np.mean(phase_diff[i][significant_coherence[i]])
    
    coherence_results['mean_phase_lag'] = mean_phase_lag
    
    return coherence_results


def multifractal_wavelet_analysis(
    results: Dict,
    channel: int = 0,
    q_values: Optional[np.ndarray] = None,
    n_scales: int = 16
) -> Dict[str, np.ndarray]:
    """Perform multifractal analysis using wavelet leaders.
    
    Args:
        results: Output from analyze_slice
        channel: Channel to analyze
        q_values: Moment orders for multifractal spectrum
        n_scales: Number of scales for analysis
        
    Returns:
        Dictionary with multifractal measures
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required for wavelet analysis")
    
    hist_mag = results['hist_mag']
    
    # Get signal
    signal = hist_mag[channel].sum(axis=(1, 2, 3))
    if signal.sum() > 0:
        signal = signal / signal.sum()
    
    # Default q values
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    
    # Perform discrete wavelet transform
    wavelet = 'db3'
    level = min(pywt.dwt_max_level(len(signal), wavelet), n_scales)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Compute wavelet leaders (simplified)
    leaders = []
    for j in range(1, len(coeffs)):
        # Take absolute value of coefficients
        abs_coeffs = np.abs(coeffs[j])
        
        # Compute local maxima (simplified leaders)
        if len(abs_coeffs) > 2:
            local_max = np.zeros_like(abs_coeffs)
            for i in range(1, len(abs_coeffs)-1):
                local_max[i] = max(abs_coeffs[i-1:i+2])
            leaders.append(local_max[local_max > 0])
        else:
            leaders.append(abs_coeffs[abs_coeffs > 0])
    
    # Compute partition function
    scales = 2**np.arange(1, len(leaders)+1)
    tau_q = np.zeros(len(q_values))
    
    for q_idx, q in enumerate(q_values):
        zq = []
        for j, leader in enumerate(leaders):
            if len(leader) > 0:
                if q == 0:
                    zq.append(np.exp(np.mean(np.log(leader + 1e-10))))
                else:
                    zq.append(np.mean(leader**q)**(1/q))
        
        if len(zq) > 1:
            # Fit scaling exponent
            log_scales = np.log(scales[:len(zq)])
            log_zq = np.log(np.array(zq) + 1e-10)
            
            # Linear fit in log-log
            valid = np.isfinite(log_zq)
            if valid.sum() >= 2:
                coeffs_fit = np.polyfit(log_scales[valid], log_zq[valid], 1)
                tau_q[q_idx] = coeffs_fit[0]
    
    # Compute singularity spectrum
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q
    
    multifractal = {
        'q_values': q_values,
        'tau_q': tau_q,
        'alpha': alpha,
        'f_alpha': f_alpha,
        'scales': scales,
    }
    
    # Compute multifractal width
    alpha_range = alpha[np.isfinite(alpha)]
    if len(alpha_range) > 0:
        multifractal['width'] = alpha_range.max() - alpha_range.min()
    else:
        multifractal['width'] = 0
    
    # Identify most singular and regular structures
    if len(alpha) > 0:
        multifractal['alpha_min'] = np.nanmin(alpha)  # Most singular
        multifractal['alpha_max'] = np.nanmax(alpha)  # Most regular
        multifractal['alpha_peak'] = alpha[np.nanargmax(f_alpha)]  # Most probable
    
    return multifractal


def identify_coherent_structures(
    wavelet_decomp: Dict,
    threshold_percentile: float = 95.0,
    min_size: int = 3
) -> Dict[str, np.ndarray]:
    """Identify coherent structures from wavelet decomposition.
    
    Args:
        wavelet_decomp: Output from wavelet_decompose_sf
        threshold_percentile: Percentile for thresholding coefficients
        min_size: Minimum structure size
        
    Returns:
        Dictionary with identified structures
    """
    from scipy import ndimage
    
    structures = {
        'structures_per_level': [],
        'structure_maps': [],
        'structure_properties': [],
    }
    
    # Analyze each detail level
    for level_idx, (dH, dV, dD) in enumerate(zip(
        wavelet_decomp['details_horizontal'],
        wavelet_decomp['details_vertical'],
        wavelet_decomp['details_diagonal']
    )):
        # Combine directional information
        detail_magnitude = np.sqrt(dH**2 + dV**2 + dD**2)
        
        # Threshold to identify structures
        threshold = np.percentile(detail_magnitude, threshold_percentile)
        binary_map = detail_magnitude > threshold
        
        # Label connected components
        labeled, n_structures = ndimage.label(binary_map)
        
        # Filter by size
        structure_list = []
        for i in range(1, n_structures + 1):
            structure_mask = labeled == i
            size = structure_mask.sum()
            
            if size >= min_size:
                # Compute structure properties
                props = {
                    'level': level_idx,
                    'size': size,
                    'mean_intensity': detail_magnitude[structure_mask].mean(),
                    'max_intensity': detail_magnitude[structure_mask].max(),
                    'centroid': ndimage.center_of_mass(structure_mask),
                }
                
                # Compute anisotropy of structure
                y_coords, x_coords = np.where(structure_mask)
                if len(x_coords) > 2:
                    cov = np.cov(x_coords, y_coords)
                    eigenvalues = np.linalg.eigvalsh(cov)
                    if eigenvalues[0] > 0:
                        props['anisotropy'] = eigenvalues[1] / eigenvalues[0]
                    else:
                        props['anisotropy'] = 1.0
                else:
                    props['anisotropy'] = 1.0
                
                structure_list.append(props)
        
        structures['structures_per_level'].append(structure_list)
        structures['structure_maps'].append(labeled)
    
    # Compute statistics
    total_structures = sum(len(s) for s in structures['structures_per_level'])
    structures['total_count'] = total_structures
    
    if total_structures > 0:
        all_sizes = [s['size'] for level_structs in structures['structures_per_level'] 
                     for s in level_structs]
        structures['mean_size'] = np.mean(all_sizes)
        structures['size_distribution'] = np.histogram(all_sizes, bins=20)
    
    return structures


# Helper functions

def compute_wavelet_energy(coeffs: List) -> np.ndarray:
    """Compute energy at each wavelet scale."""
    energies = []
    
    # Approximation energy
    energies.append(np.sum(coeffs[0]**2))
    
    # Detail energies
    for level in range(1, len(coeffs)):
        if len(coeffs[level]) == 3:  # 2D decomposition
            cH, cV, cD = coeffs[level]
            energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
        else:  # 1D decomposition
            energy = np.sum(coeffs[level]**2)
        energies.append(energy)
    
    energies = np.array(energies)
    return energies / energies.sum()  # Normalize


def detect_intermittency(coeffs: List, threshold_sigma: float = 3.0) -> np.ndarray:
    """Detect intermittent events from wavelet coefficients."""
    # Use finest scale details for intermittency detection
    if len(coeffs) > 1:
        if len(coeffs[-1]) == 3:  # 2D
            cH, cV, cD = coeffs[-1]
            detail_magnitude = np.sqrt(cH**2 + cV**2 + cD**2)
        else:  # 1D
            detail_magnitude = np.abs(coeffs[-1])
        
        # Identify outliers
        mean = np.mean(detail_magnitude)
        std = np.std(detail_magnitude)
        intermittency_map = detail_magnitude > (mean + threshold_sigma * std)
        
        return intermittency_map
    
    return np.array([])


def identify_dominant_scales(scalogram: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Identify dominant scales from scalogram."""
    # Compute mean power at each scale
    mean_power = scalogram.mean(axis=1)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(mean_power, prominence=mean_power.std())
    
    if len(peaks) > 0:
        return scales[peaks]
    else:
        # Return scale with maximum power
        return np.array([scales[np.argmax(mean_power)]])


def compute_scale_intermittency(coeffs: np.ndarray) -> np.ndarray:
    """Compute intermittency factor at each scale."""
    n_scales = coeffs.shape[0]
    intermittency = np.zeros(n_scales)
    
    for scale_idx in range(n_scales):
        scale_coeffs = coeffs[scale_idx]
        if scale_coeffs.std() > 0:
            # Flatness (kurtosis) as intermittency measure
            flatness = np.mean(scale_coeffs**4) / (np.mean(scale_coeffs**2)**2 + 1e-10)
            intermittency[scale_idx] = flatness - 3  # Excess kurtosis
    
    return intermittency


def detect_wavelet_ridges(scalogram: np.ndarray) -> List[np.ndarray]:
    """Detect ridges in wavelet scalogram."""
    from scipy.ndimage import maximum_filter
    
    # Find local maxima
    local_max = maximum_filter(scalogram, size=3)
    ridges_mask = (scalogram == local_max) & (scalogram > scalogram.mean())
    
    # Extract ridge lines
    ridges = []
    ridge_points = np.where(ridges_mask)
    
    if len(ridge_points[0]) > 0:
        # Simple ridge following (could be improved)
        for scale_idx in range(scalogram.shape[0]):
            ridge_line = np.where(ridges_mask[scale_idx])[0]
            if len(ridge_line) > 0:
                ridges.append(ridge_line)
    
    return ridges