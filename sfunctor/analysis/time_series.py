"""Time-series analysis across multiple snapshots for studying turbulence evolution.

This module provides tools to:
1. Track structure function evolution over time
2. Compute time-dependent anisotropy measures
3. Analyze spectral evolution
4. Identify intermittency changes
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
from tqdm import tqdm

from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.io.slice_io import parse_slice_metadata


def analyze_time_series(
    slice_files: List[Union[str, Path]],
    time_values: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    save_intermediate: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Analyze structure functions across multiple time snapshots.
    
    This function processes a series of snapshots to study the temporal
    evolution of turbulence properties.
    
    Args:
        slice_files: List of paths to slice files in temporal order
        time_values: Optional array of time values for each snapshot
        config: Analysis configuration (same as analyze_slice)
        output_dir: Directory to save intermediate results
        save_intermediate: Whether to save results for each snapshot
        verbose: Print progress information
        
    Returns:
        Dictionary containing:
        - 'times': Time values for each snapshot
        - 'hist_mag_series': Structure functions vs time (n_times, n_channels, ...)
        - 'hist_other_series': Cross products vs time
        - 'anisotropy_series': Anisotropy measures vs time
        - 'energy_series': Energy in different channels vs time
        - 'scaling_exponents': Scaling exponents vs time
    """
    n_snapshots = len(slice_files)
    
    if verbose:
        print(f"Analyzing {n_snapshots} snapshots for time series")
    
    # Default configuration
    if config is None:
        config = {
            'stride': 2,
            'n_disp_total': 100,
            'n_random_subsamples': 5000,
            'n_theta_bins': 16,
            'n_phi_bins': 32,
            'stencil_width': 2,
        }
    
    # Extract time values if not provided
    if time_values is None:
        time_values = np.arange(n_snapshots, dtype=float)
        # Try to extract from filenames if they contain time info
        try:
            times_from_files = []
            for f in slice_files:
                # Assuming format like "...file0123.npz" where 0123 is timestep
                fname = Path(f).stem
                if 'file' in fname:
                    timestep = int(fname.split('file')[-1].split('_')[0])
                    times_from_files.append(timestep)
            if len(times_from_files) == n_snapshots:
                time_values = np.array(times_from_files, dtype=float)
        except:
            pass
    
    # Initialize storage
    results_series = []
    anisotropy_series = []
    energy_series = []
    scaling_exponents_series = []
    
    # Create output directory if needed
    if output_dir and save_intermediate:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each snapshot
    iterator = tqdm(slice_files, desc="Processing snapshots") if verbose else slice_files
    
    for idx, slice_file in enumerate(iterator):
        # Load slice data
        from sfunctor.io.slice_io import load_slice_npz
        
        # Extract stride if present in config, otherwise use 1
        stride = config.pop('stride', 1)
        slice_data = load_slice_npz(slice_file, stride=stride)
        
        # Analyze this snapshot with corrected parameters
        # Map old parameter names to new ones
        analysis_config = {}
        if 'n_disp_total' in config:
            analysis_config['n_displacements'] = config['n_disp_total']
        if 'n_random_subsamples' in config:
            analysis_config['n_random_subsamples'] = config['n_random_subsamples']
        if 'n_theta_bins' in config or 'n_phi_bins' in config:
            # Use average for n_ell_bins
            n_theta = config.get('n_theta_bins', 16)
            n_phi = config.get('n_phi_bins', 32)
            analysis_config['n_ell_bins'] = (n_theta + n_phi) // 2
        if 'stencil_width' in config:
            analysis_config['stencil_width'] = config['stencil_width']
        
        results = analyze_slice(slice_data, **analysis_config)
        
        # Store results
        results_series.append(results)
        
        # Compute derived quantities
        anisotropy = compute_anisotropy_measures(results)
        anisotropy_series.append(anisotropy)
        
        energy = compute_energy_content(results)
        energy_series.append(energy)
        
        scaling = compute_scaling_exponents(results)
        scaling_exponents_series.append(scaling)
        
        # Save intermediate if requested
        if save_intermediate and output_dir:
            snapshot_file = output_dir / f"snapshot_{idx:04d}.npz"
            np.savez_compressed(
                snapshot_file,
                time=time_values[idx],
                **results,
                anisotropy=anisotropy,
                energy=energy,
                scaling=scaling
            )
    
    # Combine results into time series arrays
    n_channels_mag = results_series[0]['hist_mag'].shape[0]
    n_channels_other = results_series[0]['hist_other'].shape[0]
    hist_shape_mag = results_series[0]['hist_mag'].shape
    hist_shape_other = results_series[0]['hist_other'].shape
    
    hist_mag_series = np.zeros((n_snapshots,) + hist_shape_mag)
    hist_other_series = np.zeros((n_snapshots,) + hist_shape_other)
    
    for idx, results in enumerate(results_series):
        hist_mag_series[idx] = results['hist_mag']
        hist_other_series[idx] = results['hist_other']
    
    # Compile time series results
    time_series_results = {
        'times': time_values,
        'hist_mag_series': hist_mag_series,
        'hist_other_series': hist_other_series,
        'anisotropy_series': np.array(anisotropy_series),
        'energy_series': np.array(energy_series),
        'scaling_exponents': np.array(scaling_exponents_series),
        'ell_bin_edges': results_series[0]['ell_bin_edges'],
        'theta_bin_edges': results_series[0]['theta_bin_edges'],
        'phi_bin_edges': results_series[0]['phi_bin_edges'],
        'sf_bin_edges': results_series[0]['sf_bin_edges'],
    }
    
    # Save complete time series
    if output_dir:
        output_file = output_dir / 'time_series_complete.npz'
        np.savez_compressed(output_file, **time_series_results)
        if verbose:
            print(f"Saved time series to {output_file}")
    
    return time_series_results


def compute_anisotropy_measures(results: Dict) -> Dict[str, np.ndarray]:
    """Compute various anisotropy measures from structure function results.
    
    Args:
        results: Output from analyze_slice
        
    Returns:
        Dictionary with anisotropy measures:
        - 'parallel_perp_ratio': Ratio of parallel to perpendicular SF
        - 'alignment_angle': Mean alignment angle with B field
        - 'anisotropy_index': Scalar anisotropy measure
    """
    hist_mag = results['hist_mag']
    theta_bin_edges = results['theta_bin_edges']
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    
    # Define parallel and perpendicular masks
    parallel_mask = (theta_centers < np.pi/6) | (theta_centers > 5*np.pi/6)
    perp_mask = (theta_centers > np.pi/3) & (theta_centers < 2*np.pi/3)
    
    anisotropy = {}
    
    # Compute for each channel
    n_channels = hist_mag.shape[0]
    parallel_perp_ratios = np.zeros(n_channels)
    
    for ch in range(n_channels):
        sf_channel = hist_mag[ch]
        
        # Integrate over angles
        sf_parallel = sf_channel[:, parallel_mask, :, :].sum()
        sf_perp = sf_channel[:, perp_mask, :, :].sum()
        
        if sf_perp > 0:
            parallel_perp_ratios[ch] = sf_parallel / sf_perp
        else:
            parallel_perp_ratios[ch] = np.nan
    
    anisotropy['parallel_perp_ratio'] = parallel_perp_ratios
    
    # Compute mean alignment angle
    # Weight theta by counts to get mean angle
    velocity_sf = hist_mag[0]  # Use velocity channel
    theta_weights = velocity_sf.sum(axis=(0, 2, 3))  # Sum over ell, phi, sf
    if theta_weights.sum() > 0:
        mean_theta = np.average(theta_centers, weights=theta_weights)
    else:
        mean_theta = np.nan
    
    anisotropy['alignment_angle'] = mean_theta
    
    # Compute anisotropy index (variance of angular distribution)
    # Higher values mean more anisotropic
    theta_dist = theta_weights / theta_weights.sum() if theta_weights.sum() > 0 else theta_weights
    theta_variance = np.sum(theta_dist * (theta_centers - mean_theta)**2) if not np.isnan(mean_theta) else np.nan
    anisotropy['anisotropy_index'] = np.sqrt(theta_variance)
    
    return anisotropy


def compute_energy_content(results: Dict) -> Dict[str, float]:
    """Compute energy content in different channels.
    
    Args:
        results: Output from analyze_slice
        
    Returns:
        Dictionary with energy measures for each channel
    """
    hist_mag = results['hist_mag']
    sf_bin_edges = results['sf_bin_edges']
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    energy = {}
    channel_names = ['velocity', 'magnetic', 'density', 'alfven', 'zplus', 'zminus',
                     'vorticity', 'current', 'curvature', 'grad_rho', 'b_norm']
    
    for ch, name in enumerate(channel_names):
        if ch >= hist_mag.shape[0]:
            break
        
        # Compute second moment (energy-like quantity)
        sf_channel = hist_mag[ch]
        counts = sf_channel.sum(axis=(1, 2))  # Sum over angles
        
        # Weight by sf^2 for energy
        energy_ch = 0.0
        for i, sf_val in enumerate(sf_centers):
            energy_ch += sf_val**2 * counts[:, i].sum()
        
        total_counts = counts.sum()
        if total_counts > 0:
            energy[name] = energy_ch / total_counts
        else:
            energy[name] = 0.0
    
    return energy


def compute_scaling_exponents(results: Dict, p_values: List[float] = [1, 2, 3]) -> Dict[str, np.ndarray]:
    """Compute scaling exponents ζ_p for different orders p.
    
    The scaling exponent ζ_p is defined by: SF_p(ℓ) ∝ ℓ^ζ_p
    
    Args:
        results: Output from analyze_slice
        p_values: List of moment orders to compute
        
    Returns:
        Dictionary with scaling exponents for each channel and order
    """
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    sf_bin_edges = results['sf_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    n_channels = hist_mag.shape[0]
    n_ell = len(ell_centers)
    
    scaling = {}
    
    for p in p_values:
        exponents = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # Average over angles
            sf_channel = hist_mag[ch].mean(axis=(1, 2))  # Shape: (n_ell, n_sf)
            
            # Compute p-th moment for each ell
            moments = np.zeros(n_ell)
            for ell_idx in range(n_ell):
                pdf = sf_channel[ell_idx]
                if pdf.sum() > 0:
                    pdf_norm = pdf / pdf.sum()
                    moments[ell_idx] = np.sum(sf_centers**p * pdf_norm)
                else:
                    moments[ell_idx] = np.nan
            
            # Fit scaling exponent (log-log slope)
            valid = np.isfinite(moments) & (moments > 0)
            if valid.sum() >= 2:
                # Use least squares fit in log space
                log_ell = np.log(ell_centers[valid])
                log_moment = np.log(moments[valid])
                
                # Fit only in inertial range (middle third of scales)
                n_valid = len(log_ell)
                i_start = n_valid // 3
                i_end = 2 * n_valid // 3
                
                if i_end > i_start + 1:
                    coeffs = np.polyfit(log_ell[i_start:i_end], 
                                       log_moment[i_start:i_end], 1)
                    exponents[ch] = coeffs[0]
                else:
                    exponents[ch] = np.nan
            else:
                exponents[ch] = np.nan
        
        scaling[f'zeta_{p}'] = exponents
    
    return scaling


def analyze_evolution(
    time_series_results: Dict,
    channel: int = 0,
    ell_idx: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Analyze the temporal evolution of specific structure function properties.
    
    Args:
        time_series_results: Output from analyze_time_series
        channel: Which channel to analyze (0=velocity, 1=magnetic, etc.)
        ell_idx: Which length scale to focus on (None = average over all)
        
    Returns:
        Dictionary with evolution metrics
    """
    times = time_series_results['times']
    hist_series = time_series_results['hist_mag_series']
    
    # Extract channel evolution
    channel_evolution = hist_series[:, channel]  # Shape: (n_times, n_ell, n_theta, n_phi, n_sf)
    
    if ell_idx is not None:
        channel_evolution = channel_evolution[:, ell_idx]
    else:
        channel_evolution = channel_evolution.mean(axis=1)
    
    # Compute time derivatives
    dt = np.diff(times)
    if len(dt) > 0 and dt[0] > 0:
        # Compute growth/decay rates
        total_counts = channel_evolution.sum(axis=(1, 2, 3))
        growth_rate = np.gradient(np.log(total_counts + 1e-10), times)
    else:
        growth_rate = np.zeros(len(times))
    
    # Compute intermittency evolution (flatness)
    sf_bin_edges = time_series_results['sf_bin_edges']
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    flatness_series = []
    for t in range(len(times)):
        sf_t = channel_evolution[t].sum(axis=(0, 1))  # Average over angles
        if sf_t.sum() > 0:
            pdf = sf_t / sf_t.sum()
            moment2 = np.sum(sf_centers**2 * pdf)
            moment4 = np.sum(sf_centers**4 * pdf)
            if moment2 > 0:
                flatness = moment4 / moment2**2
            else:
                flatness = np.nan
        else:
            flatness = np.nan
        flatness_series.append(flatness)
    
    evolution = {
        'times': times,
        'growth_rate': growth_rate,
        'flatness': np.array(flatness_series),
        'total_counts': channel_evolution.sum(axis=(1, 2, 3)),
    }
    
    return evolution


def detect_transition_times(
    time_series_results: Dict,
    threshold: float = 0.1
) -> List[float]:
    """Detect times when turbulence properties undergo significant changes.
    
    Args:
        time_series_results: Output from analyze_time_series
        threshold: Relative change threshold for detection
        
    Returns:
        List of transition times
    """
    anisotropy = time_series_results['anisotropy_series']
    times = time_series_results['times']
    
    # Look for sudden changes in anisotropy
    aniso_index = anisotropy[:, 2] if len(anisotropy.shape) > 1 else anisotropy
    
    # Compute relative changes
    rel_changes = np.abs(np.diff(aniso_index)) / (np.abs(aniso_index[:-1]) + 1e-10)
    
    # Find peaks in change rate
    transition_indices = np.where(rel_changes > threshold)[0]
    transition_times = times[transition_indices + 1]  # +1 because diff reduces length
    
    return transition_times.tolist()