"""Wavelet power spectrum mapping for spatially-resolved turbulence analysis.

This module provides tools to create spatial maps of turbulence properties
by computing local wavelet power spectra and fitting power laws.

Key features:
1. Local wavelet power spectrum at each spatial point
2. Power law fitting to extract amplitude and spectral index
3. Spatial maps of turbulence properties
4. Multi-scale analysis with user-defined scale ranges
5. Quality metrics for fit reliability
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path
import warnings
from scipy import ndimage, signal, optimize
from dataclasses import dataclass

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets required for wavelet maps. Install with: pip install PyWavelets")


@dataclass
class WaveletMapConfig:
    """Configuration for wavelet power spectrum mapping."""
    outer_scale: float = 32.0          # Maximum scale to analyze
    inner_scale: float = 2.0           # Minimum scale to analyze  
    n_scales: int = 32                 # Number of scales
    window_size: int = 64              # Size of local analysis window
    window_overlap: float = 0.5        # Overlap fraction between windows
    wavelet: str = 'morl'              # Mother wavelet
    detrend: str = 'linear'            # Detrending method ('none', 'mean', 'linear')
    fit_method: str = 'robust'         # Fitting method ('least_squares', 'robust', 'bootstrap')
    min_fit_points: int = 5            # Minimum points for power law fit
    edge_handling: str = 'mirror'      # Edge handling ('mirror', 'periodic', 'zero')
    compute_confidence: bool = True    # Compute confidence intervals
    n_bootstrap: int = 100             # Bootstrap samples for confidence


def compute_wavelet_power_maps(
    field: np.ndarray,
    config: Optional[WaveletMapConfig] = None,
    mask: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Compute spatial maps of wavelet power spectrum properties.
    
    For each location in the field, computes a local wavelet power spectrum,
    fits a power law, and extracts amplitude and spectral index.
    
    Args:
        field: 2D field array to analyze
        config: Configuration object (uses defaults if None)
        mask: Optional mask for valid regions
        verbose: Print progress information
        
    Returns:
        Dictionary containing:
        - 'amplitude_map': Power law amplitude at each location
        - 'slope_map': Spectral index (power law slope) at each location
        - 'r_squared_map': Fit quality (R²) at each location
        - 'energy_map': Total energy in scale range at each location
        - 'peak_scale_map': Scale of maximum power at each location
        - 'intermittency_map': Local intermittency measure
        - 'scales': Scale values used
        - 'mean_spectrum': Spatially-averaged power spectrum
        - 'config': Configuration used
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required for wavelet power maps")
    
    if config is None:
        config = WaveletMapConfig()
    
    # Validate input
    if field.ndim != 2:
        raise ValueError(f"Field must be 2D, got shape {field.shape}")
    
    ny, nx = field.shape
    
    if verbose:
        print(f"Computing wavelet power maps for {ny}×{nx} field")
        print(f"  Outer scale: {config.outer_scale}")
        print(f"  Inner scale: {config.inner_scale}")
        print(f"  Window size: {config.window_size}")
    
    # Generate scales (logarithmic spacing)
    scales = np.logspace(
        np.log10(config.inner_scale),
        np.log10(config.outer_scale),
        config.n_scales
    )
    
    # Determine window positions
    window_size = config.window_size
    stride = int(window_size * (1 - config.window_overlap))
    
    # Calculate number of windows
    n_windows_y = (ny - window_size) // stride + 1
    n_windows_x = (nx - window_size) // stride + 1
    
    # Initialize output maps (at window resolution)
    amplitude_map = np.zeros((n_windows_y, n_windows_x))
    slope_map = np.zeros((n_windows_y, n_windows_x))
    r_squared_map = np.zeros((n_windows_y, n_windows_x))
    energy_map = np.zeros((n_windows_y, n_windows_x))
    peak_scale_map = np.zeros((n_windows_y, n_windows_x))
    intermittency_map = np.zeros((n_windows_y, n_windows_x))
    
    # Store all local spectra for averaging
    all_spectra = []
    
    # Process each window
    for iy in range(n_windows_y):
        for ix in range(n_windows_x):
            # Extract window
            y_start = iy * stride
            x_start = ix * stride
            y_end = y_start + window_size
            x_end = x_start + window_size
            
            window = field[y_start:y_end, x_start:x_end]
            
            # Check mask if provided
            if mask is not None:
                window_mask = mask[y_start:y_end, x_start:x_end]
                if np.sum(window_mask) < window_size * window_size * 0.5:
                    # Skip if more than 50% masked
                    amplitude_map[iy, ix] = np.nan
                    slope_map[iy, ix] = np.nan
                    r_squared_map[iy, ix] = np.nan
                    continue
            
            # Detrend window
            window = detrend_window(window, method=config.detrend)
            
            # Apply edge tapering
            window = apply_window_taper(window)
            
            # Compute local wavelet power spectrum
            power_spectrum = compute_local_power_spectrum(
                window, scales, config.wavelet
            )
            
            all_spectra.append(power_spectrum)
            
            # Fit power law
            fit_result = fit_power_law(
                scales, power_spectrum,
                min_points=config.min_fit_points,
                method=config.fit_method
            )
            
            # Store results
            amplitude_map[iy, ix] = fit_result['amplitude']
            slope_map[iy, ix] = fit_result['slope']
            r_squared_map[iy, ix] = fit_result['r_squared']
            
            # Compute additional metrics
            energy_map[iy, ix] = np.sum(power_spectrum)
            peak_scale_map[iy, ix] = scales[np.argmax(power_spectrum)]
            
            # Intermittency measure (departure from mean spectrum)
            if len(all_spectra) > 1:
                mean_spec = np.mean(all_spectra, axis=0)
                intermittency = np.std(np.log(power_spectrum + 1e-10) - 
                                      np.log(mean_spec + 1e-10))
            else:
                intermittency = 0
            intermittency_map[iy, ix] = intermittency
    
    # Interpolate maps back to original resolution
    if n_windows_y != ny or n_windows_x != nx:
        amplitude_map = interpolate_map_to_original(
            amplitude_map, (ny, nx), stride, window_size
        )
        slope_map = interpolate_map_to_original(
            slope_map, (ny, nx), stride, window_size
        )
        r_squared_map = interpolate_map_to_original(
            r_squared_map, (ny, nx), stride, window_size
        )
        energy_map = interpolate_map_to_original(
            energy_map, (ny, nx), stride, window_size
        )
        peak_scale_map = interpolate_map_to_original(
            peak_scale_map, (ny, nx), stride, window_size
        )
        intermittency_map = interpolate_map_to_original(
            intermittency_map, (ny, nx), stride, window_size
        )
    
    # Compute mean spectrum
    if all_spectra:
        mean_spectrum = np.mean(all_spectra, axis=0)
    else:
        mean_spectrum = np.zeros(len(scales))
    
    results = {
        'amplitude_map': amplitude_map,
        'slope_map': slope_map,
        'r_squared_map': r_squared_map,
        'energy_map': energy_map,
        'peak_scale_map': peak_scale_map,
        'intermittency_map': intermittency_map,
        'scales': scales,
        'mean_spectrum': mean_spectrum,
        'config': config
    }
    
    if verbose:
        print(f"  Mean spectral slope: {np.nanmean(slope_map):.2f}")
        print(f"  Slope std dev: {np.nanstd(slope_map):.2f}")
        print(f"  Mean R²: {np.nanmean(r_squared_map):.3f}")
    
    return results


def compute_multiscale_maps(
    field: np.ndarray,
    scale_ranges: List[Tuple[float, float]],
    config: Optional[WaveletMapConfig] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute wavelet power maps for multiple scale ranges.
    
    Useful for studying scale-dependent spatial variations.
    
    Args:
        field: 2D field array
        scale_ranges: List of (inner_scale, outer_scale) tuples
        config: Base configuration (scales will be overridden)
        
    Returns:
        Dictionary with results for each scale range
    """
    if config is None:
        config = WaveletMapConfig()
    
    results = {}
    
    for i, (inner, outer) in enumerate(scale_ranges):
        # Create config for this scale range
        range_config = WaveletMapConfig(
            inner_scale=inner,
            outer_scale=outer,
            n_scales=config.n_scales,
            window_size=min(config.window_size, int(outer * 2)),
            window_overlap=config.window_overlap,
            wavelet=config.wavelet,
            detrend=config.detrend,
            fit_method=config.fit_method
        )
        
        # Compute maps for this range
        range_results = compute_wavelet_power_maps(
            field, range_config, verbose=False
        )
        
        # Store with descriptive key
        key = f"scale_{inner:.1f}_{outer:.1f}"
        results[key] = range_results
    
    return results


def compute_anisotropic_wavelet_maps(
    field: np.ndarray,
    config: Optional[WaveletMapConfig] = None,
    angles: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Compute directional wavelet power maps using anisotropic wavelets.
    
    Args:
        field: 2D field array
        config: Configuration
        angles: Angles for directional analysis (default: 0, 45, 90, 135 degrees)
        
    Returns:
        Dictionary with directional power maps and anisotropy measures
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required")
    
    if config is None:
        config = WaveletMapConfig()
    
    if angles is None:
        angles = np.array([0, 45, 90, 135]) * np.pi / 180
    
    ny, nx = field.shape
    n_angles = len(angles)
    
    # Store directional results
    directional_slopes = np.zeros((n_angles, ny, nx))
    directional_amplitudes = np.zeros((n_angles, ny, nx))
    
    for i, angle in enumerate(angles):
        # Rotate field
        rotated = ndimage.rotate(field, -angle * 180 / np.pi, 
                                 reshape=False, order=3)
        
        # Compute wavelet maps for rotated field
        results = compute_wavelet_power_maps(rotated, config, verbose=False)
        
        # Rotate back and store
        directional_slopes[i] = ndimage.rotate(
            results['slope_map'], angle * 180 / np.pi, 
            reshape=False, order=3
        )[:ny, :nx]
        
        directional_amplitudes[i] = ndimage.rotate(
            results['amplitude_map'], angle * 180 / np.pi,
            reshape=False, order=3
        )[:ny, :nx]
    
    # Compute anisotropy metrics
    anisotropy_ratio = (np.max(directional_slopes, axis=0) / 
                       (np.min(directional_slopes, axis=0) + 1e-10))
    
    preferred_direction = angles[np.argmax(directional_amplitudes, axis=0)]
    
    # Compute alignment strength (how much variance across directions)
    alignment_strength = np.std(directional_slopes, axis=0) / \
                        (np.mean(directional_slopes, axis=0) + 1e-10)
    
    return {
        'directional_slopes': directional_slopes,
        'directional_amplitudes': directional_amplitudes,
        'anisotropy_ratio': anisotropy_ratio,
        'preferred_direction': preferred_direction,
        'alignment_strength': alignment_strength,
        'angles': angles
    }


def compute_cross_scale_correlation_map(
    field: np.ndarray,
    scale1: float,
    scale2: float,
    config: Optional[WaveletMapConfig] = None
) -> Dict[str, np.ndarray]:
    """Compute spatial map of correlation between two scales.
    
    Reveals regions where energy transfer between scales is enhanced.
    
    Args:
        field: 2D field array
        scale1: First scale
        scale2: Second scale (should be different from scale1)
        config: Configuration
        
    Returns:
        Dictionary with correlation maps
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required")
    
    if config is None:
        config = WaveletMapConfig()
    
    # Compute CWT at both scales
    coeffs1, _ = pywt.cwt(field.flatten(), [scale1], config.wavelet)
    coeffs2, _ = pywt.cwt(field.flatten(), [scale2], config.wavelet)
    
    # Reshape to 2D
    coeffs1 = coeffs1.reshape(field.shape)
    coeffs2 = coeffs2.reshape(field.shape)
    
    # Compute local correlation in sliding windows
    window_size = config.window_size
    stride = int(window_size * (1 - config.window_overlap))
    
    ny, nx = field.shape
    n_windows_y = (ny - window_size) // stride + 1
    n_windows_x = (nx - window_size) // stride + 1
    
    correlation_map = np.zeros((n_windows_y, n_windows_x))
    coherence_map = np.zeros((n_windows_y, n_windows_x))
    phase_map = np.zeros((n_windows_y, n_windows_x))
    
    for iy in range(n_windows_y):
        for ix in range(n_windows_x):
            y_start = iy * stride
            x_start = ix * stride
            y_end = y_start + window_size
            x_end = x_start + window_size
            
            w1 = coeffs1[y_start:y_end, x_start:x_end].flatten()
            w2 = coeffs2[y_start:y_end, x_start:x_end].flatten()
            
            # Correlation
            if np.std(w1) > 0 and np.std(w2) > 0:
                correlation_map[iy, ix] = np.corrcoef(w1, w2)[0, 1]
            
            # Coherence (magnitude squared coherence)
            cross_spec = np.mean(w1 * np.conj(w2))
            auto_spec1 = np.mean(np.abs(w1)**2)
            auto_spec2 = np.mean(np.abs(w2)**2)
            
            if auto_spec1 > 0 and auto_spec2 > 0:
                coherence_map[iy, ix] = np.abs(cross_spec)**2 / (auto_spec1 * auto_spec2)
            
            # Phase difference
            phase_map[iy, ix] = np.angle(cross_spec)
    
    # Interpolate to original resolution
    correlation_map = interpolate_map_to_original(
        correlation_map, field.shape, stride, window_size
    )
    coherence_map = interpolate_map_to_original(
        coherence_map, field.shape, stride, window_size
    )
    phase_map = interpolate_map_to_original(
        phase_map, field.shape, stride, window_size
    )
    
    return {
        'correlation_map': correlation_map,
        'coherence_map': coherence_map,
        'phase_map': phase_map,
        'scale1': scale1,
        'scale2': scale2
    }


def identify_turbulence_regions(
    wavelet_maps: Dict[str, np.ndarray],
    slope_threshold: Optional[Tuple[float, float]] = None,
    amplitude_threshold: Optional[float] = None,
    r_squared_threshold: float = 0.8
) -> Dict[str, np.ndarray]:
    """Identify regions with specific turbulence characteristics.
    
    Args:
        wavelet_maps: Output from compute_wavelet_power_maps
        slope_threshold: (min, max) range for spectral slope
        amplitude_threshold: Minimum amplitude
        r_squared_threshold: Minimum fit quality
        
    Returns:
        Dictionary with region masks and statistics
    """
    slope_map = wavelet_maps['slope_map']
    amplitude_map = wavelet_maps['amplitude_map']
    r_squared_map = wavelet_maps['r_squared_map']
    
    # Quality mask
    quality_mask = r_squared_map > r_squared_threshold
    
    # Initialize region masks
    regions = {}
    
    # Kolmogorov-like turbulence (slope ≈ -5/3)
    kolmogorov_mask = quality_mask & (np.abs(slope_map + 5/3) < 0.2)
    regions['kolmogorov'] = kolmogorov_mask
    
    # Steep spectrum (dissipation range)
    steep_mask = quality_mask & (slope_map < -2.5)
    regions['dissipation'] = steep_mask
    
    # Shallow spectrum (injection or inverse cascade)
    shallow_mask = quality_mask & (slope_map > -1.0)
    regions['injection'] = shallow_mask
    
    # High amplitude regions (intense turbulence)
    if amplitude_threshold is not None:
        intense_mask = quality_mask & (amplitude_map > amplitude_threshold)
    else:
        # Use top 10% as intense
        threshold = np.nanpercentile(amplitude_map[quality_mask], 90)
        intense_mask = quality_mask & (amplitude_map > threshold)
    regions['intense'] = intense_mask
    
    # Custom slope range
    if slope_threshold is not None:
        custom_mask = quality_mask & \
                     (slope_map > slope_threshold[0]) & \
                     (slope_map < slope_threshold[1])
        regions['custom'] = custom_mask
    
    # Compute region statistics
    stats = {}
    for name, mask in regions.items():
        if np.any(mask):
            stats[name] = {
                'fraction': np.sum(mask) / mask.size,
                'mean_slope': np.mean(slope_map[mask]),
                'std_slope': np.std(slope_map[mask]),
                'mean_amplitude': np.mean(amplitude_map[mask]),
                'mean_r_squared': np.mean(r_squared_map[mask])
            }
    
    return {
        'regions': regions,
        'statistics': stats
    }


# Helper functions

def detrend_window(window: np.ndarray, method: str = 'linear') -> np.ndarray:
    """Remove trend from window."""
    if method == 'none':
        return window
    elif method == 'mean':
        return window - np.mean(window)
    elif method == 'linear':
        # Remove linear trend
        ny, nx = window.shape
        y, x = np.meshgrid(range(ny), range(nx), indexing='ij')
        
        # Flatten for fitting
        coords = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
        z = window.ravel()
        
        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(coords, z, rcond=None)
        
        # Remove trend
        trend = (coeffs[0] * x + coeffs[1] * y + coeffs[2])
        return window - trend
    else:
        return window - np.mean(window)


def apply_window_taper(window: np.ndarray, taper_fraction: float = 0.1) -> np.ndarray:
    """Apply cosine taper to window edges."""
    ny, nx = window.shape
    
    # Create taper
    taper_y = np.ones(ny)
    taper_x = np.ones(nx)
    
    n_taper_y = int(ny * taper_fraction)
    n_taper_x = int(nx * taper_fraction)
    
    if n_taper_y > 0:
        taper_y[:n_taper_y] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper_y) / n_taper_y))
        taper_y[-n_taper_y:] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper_y, 0, -1) / n_taper_y))
    
    if n_taper_x > 0:
        taper_x[:n_taper_x] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper_x) / n_taper_x))
        taper_x[-n_taper_x:] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper_x, 0, -1) / n_taper_x))
    
    # Apply 2D taper
    taper_2d = np.outer(taper_y, taper_x)
    
    return window * taper_2d


def compute_local_power_spectrum(
    window: np.ndarray,
    scales: np.ndarray,
    wavelet: str = 'morl'
) -> np.ndarray:
    """Compute wavelet power spectrum for a window."""
    # Compute 2D CWT by analyzing rows and columns
    ny, nx = window.shape
    power = np.zeros(len(scales))
    
    # Average over rows
    for row in window:
        coeffs, _ = pywt.cwt(row, scales, wavelet)
        power += np.mean(np.abs(coeffs)**2, axis=1)
    
    # Average over columns
    for col in window.T:
        coeffs, _ = pywt.cwt(col, scales, wavelet)
        power += np.mean(np.abs(coeffs)**2, axis=1)
    
    # Normalize
    power /= (ny + nx)
    
    return power


def fit_power_law(
    scales: np.ndarray,
    spectrum: np.ndarray,
    min_points: int = 5,
    method: str = 'robust'
) -> Dict[str, float]:
    """Fit power law to spectrum."""
    # Remove invalid points
    valid = (spectrum > 0) & np.isfinite(spectrum) & np.isfinite(scales)
    
    if np.sum(valid) < min_points:
        return {
            'amplitude': np.nan,
            'slope': np.nan,
            'r_squared': 0.0,
            'error': np.nan
        }
    
    log_scales = np.log10(scales[valid])
    log_spectrum = np.log10(spectrum[valid])
    
    if method == 'least_squares':
        # Simple linear regression in log-log space
        coeffs = np.polyfit(log_scales, log_spectrum, 1)
        slope = coeffs[0]
        log_amplitude = coeffs[1]
        
        # Compute R²
        fit_values = np.polyval(coeffs, log_scales)
        ss_res = np.sum((log_spectrum - fit_values)**2)
        ss_tot = np.sum((log_spectrum - np.mean(log_spectrum))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
    elif method == 'robust':
        # Use robust regression (Huber regressor)
        from scipy.optimize import minimize
        
        def huber_loss(params, x, y, delta=1.0):
            """Huber loss for robust regression."""
            residuals = y - (params[0] * x + params[1])
            loss = np.where(
                np.abs(residuals) <= delta,
                0.5 * residuals**2,
                delta * (np.abs(residuals) - 0.5 * delta)
            )
            return np.sum(loss)
        
        # Initial guess from least squares
        initial = np.polyfit(log_scales, log_spectrum, 1)
        
        # Minimize Huber loss
        result = minimize(huber_loss, initial, args=(log_scales, log_spectrum))
        
        slope = result.x[0]
        log_amplitude = result.x[1]
        
        # Compute R² for comparison
        fit_values = slope * log_scales + log_amplitude
        ss_res = np.sum((log_spectrum - fit_values)**2)
        ss_tot = np.sum((log_spectrum - np.mean(log_spectrum))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
    else:  # Bootstrap
        n_bootstrap = 100
        slopes = []
        amplitudes = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(log_scales), len(log_scales), replace=True)
            boot_scales = log_scales[indices]
            boot_spectrum = log_spectrum[indices]
            
            # Fit
            coeffs = np.polyfit(boot_scales, boot_spectrum, 1)
            slopes.append(coeffs[0])
            amplitudes.append(coeffs[1])
        
        slope = np.median(slopes)
        log_amplitude = np.median(amplitudes)
        
        # Compute R² with median parameters
        fit_values = slope * log_scales + log_amplitude
        ss_res = np.sum((log_spectrum - fit_values)**2)
        ss_tot = np.sum((log_spectrum - np.mean(log_spectrum))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'amplitude': 10**log_amplitude,
        'slope': slope,
        'r_squared': r_squared,
        'error': np.std(log_spectrum - (slope * log_scales + log_amplitude))
    }


def interpolate_map_to_original(
    map_window: np.ndarray,
    original_shape: Tuple[int, int],
    stride: int,
    window_size: int
) -> np.ndarray:
    """Interpolate windowed map back to original resolution."""
    from scipy.interpolate import RegularGridInterpolator
    
    ny_orig, nx_orig = original_shape
    ny_win, nx_win = map_window.shape
    
    # Create coordinates for window centers
    y_centers = np.arange(ny_win) * stride + window_size // 2
    x_centers = np.arange(nx_win) * stride + window_size // 2
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (y_centers, x_centers), 
        map_window,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Create output grid
    y_out, x_out = np.meshgrid(
        np.arange(ny_orig),
        np.arange(nx_orig),
        indexing='ij'
    )
    
    # Interpolate
    points = np.column_stack([y_out.ravel(), x_out.ravel()])
    interpolated = interpolator(points).reshape(original_shape)
    
    # Fill edges if needed
    if np.any(np.isnan(interpolated)):
        # Use nearest neighbor for edges
        mask = ~np.isnan(interpolated)
        if np.any(mask):
            from scipy.interpolate import NearestNDInterpolator
            y_valid, x_valid = np.where(mask)
            values_valid = interpolated[mask]
            
            nearest = NearestNDInterpolator(
                np.column_stack([y_valid, x_valid]),
                values_valid
            )
            
            nan_mask = np.isnan(interpolated)
            y_nan, x_nan = np.where(nan_mask)
            interpolated[nan_mask] = nearest(np.column_stack([y_nan, x_nan]))
    
    return interpolated


def visualize_wavelet_maps(
    results: Dict[str, np.ndarray],
    field: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> None:
    """Create visualization of wavelet power spectrum maps.
    
    Args:
        results: Output from compute_wavelet_power_maps
        field: Original field for comparison
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    n_plots = 5 if field is not None else 4
    fig, axes = plt.subplots(2, 3 if n_plots > 4 else 2, 
                             figsize=(15 if n_plots > 4 else 10, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Original field
    if field is not None:
        ax = axes[plot_idx]
        im = ax.imshow(field, cmap='RdBu_r', origin='lower')
        ax.set_title('Original Field')
        plt.colorbar(im, ax=ax)
        plot_idx += 1
    
    # Amplitude map
    ax = axes[plot_idx]
    amplitude_map = results['amplitude_map']
    im = ax.imshow(np.log10(amplitude_map + 1e-10), cmap='viridis', origin='lower')
    ax.set_title('Log₁₀(Amplitude)')
    plt.colorbar(im, ax=ax)
    plot_idx += 1
    
    # Slope map
    ax = axes[plot_idx]
    slope_map = results['slope_map']
    # Use diverging colormap centered on -5/3
    vmax = max(abs(np.nanmin(slope_map) + 5/3), abs(np.nanmax(slope_map) + 5/3))
    im = ax.imshow(slope_map, cmap='RdBu_r', origin='lower',
                   vmin=-5/3-vmax, vmax=-5/3+vmax)
    ax.set_title(f'Spectral Slope (mean={np.nanmean(slope_map):.2f})')
    cbar = plt.colorbar(im, ax=ax)
    # Add reference line at -5/3
    cbar.ax.axhline(y=-5/3, color='k', linestyle='--', linewidth=1)
    plot_idx += 1
    
    # R² map
    ax = axes[plot_idx]
    r_squared_map = results['r_squared_map']
    im = ax.imshow(r_squared_map, cmap='magma', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Fit Quality R² (mean={np.nanmean(r_squared_map):.2f})')
    plt.colorbar(im, ax=ax)
    plot_idx += 1
    
    # Intermittency map
    if 'intermittency_map' in results:
        ax = axes[plot_idx]
        intermittency_map = results['intermittency_map']
        im = ax.imshow(intermittency_map, cmap='hot', origin='lower')
        ax.set_title('Intermittency')
        plt.colorbar(im, ax=ax)
        plot_idx += 1
    
    # Mean spectrum
    if plot_idx < len(axes):
        ax = axes[plot_idx]
        scales = results['scales']
        mean_spectrum = results['mean_spectrum']
        
        ax.loglog(scales, mean_spectrum, 'b-', label='Mean spectrum')
        
        # Add reference slopes
        ax.loglog(scales, mean_spectrum[0] * (scales/scales[0])**(-5/3), 
                 'k--', alpha=0.5, label='k⁻⁵/³')
        ax.loglog(scales, mean_spectrum[0] * (scales/scales[0])**(-3), 
                 'k:', alpha=0.5, label='k⁻³')
        
        ax.set_xlabel('Scale')
        ax.set_ylabel('Power')
        ax.set_title('Mean Power Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove unused axes
    for i in range(plot_idx + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Wavelet Power Spectrum Maps', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()