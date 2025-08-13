"""Cross-correlation analysis between different MHD field quantities.

This module provides tools for:
1. Computing cross-correlations between field pairs
2. Scale-dependent correlation analysis
3. Phase relationships and coherence
4. Transfer function estimation
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import warnings
from scipy import signal, stats

def compute_field_correlations(
    results: Dict,
    field_pairs: Optional[List[Tuple[int, int]]] = None,
    normalize: bool = True,
    compute_phase: bool = False
) -> Dict[str, np.ndarray]:
    """Compute cross-correlations between different field channels.
    
    Args:
        results: Output from analyze_slice
        field_pairs: List of (channel1, channel2) tuples to correlate
                    Default: [(0,1), (0,2), (1,2)] for vel-mag, vel-dens, mag-dens
        normalize: Whether to normalize correlations to [-1, 1]
        compute_phase: Whether to compute phase relationships
        
    Returns:
        Dictionary with correlation measures for each field pair
    """
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    theta_bin_edges = results['theta_bin_edges']
    sf_bin_edges = results['sf_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    # Default field pairs: velocity-magnetic, velocity-density, magnetic-density
    if field_pairs is None:
        field_pairs = [(0, 1), (0, 2), (1, 2)]
    
    correlations = {
        'field_pairs': field_pairs,
        'ell_centers': ell_centers,
        'theta_centers': theta_centers,
    }
    
    # Channel names for reference
    channel_names = ['velocity', 'magnetic', 'density', 'alfven', 'zplus', 'zminus',
                     'vorticity', 'current', 'curvature', 'grad_rho', 'b_norm']
    
    for ch1, ch2 in field_pairs:
        if ch1 >= hist_mag.shape[0] or ch2 >= hist_mag.shape[0]:
            warnings.warn(f"Channel pair ({ch1}, {ch2}) out of bounds")
            continue
        
        name1 = channel_names[ch1] if ch1 < len(channel_names) else f"ch{ch1}"
        name2 = channel_names[ch2] if ch2 < len(channel_names) else f"ch{ch2}"
        pair_name = f"{name1}_{name2}"
        
        # Get structure functions for both channels
        sf1 = hist_mag[ch1]  # Shape: (n_ell, n_theta, n_phi, n_sf)
        sf2 = hist_mag[ch2]
        
        # Compute correlation at each scale
        n_ell = len(ell_centers)
        n_theta = len(theta_centers)
        
        corr_vs_scale = np.zeros(n_ell)
        corr_vs_angle = np.zeros((n_ell, n_theta))
        
        for ell_idx in range(n_ell):
            for theta_idx in range(n_theta):
                # Get distributions at this scale and angle
                dist1 = sf1[ell_idx, theta_idx].sum(axis=0)  # Sum over phi
                dist2 = sf2[ell_idx, theta_idx].sum(axis=0)
                
                if dist1.sum() > 0 and dist2.sum() > 0:
                    # Normalize to probabilities
                    p1 = dist1 / dist1.sum()
                    p2 = dist2 / dist2.sum()
                    
                    # Compute correlation coefficient
                    if normalize:
                        # Pearson correlation
                        mean1 = np.sum(sf_centers * p1)
                        mean2 = np.sum(sf_centers * p2)
                        
                        var1 = np.sum((sf_centers - mean1)**2 * p1)
                        var2 = np.sum((sf_centers - mean2)**2 * p2)
                        
                        if var1 > 0 and var2 > 0:
                            cov = np.sum((sf_centers - mean1) * (sf_centers - mean2) * 
                                       np.sqrt(p1 * p2))
                            corr = cov / np.sqrt(var1 * var2)
                        else:
                            corr = 0
                    else:
                        # Simple product correlation
                        corr = np.sum(p1 * p2)
                    
                    corr_vs_angle[ell_idx, theta_idx] = corr
            
            # Average over angles for scale-dependent correlation
            corr_vs_scale[ell_idx] = np.nanmean(corr_vs_angle[ell_idx])
        
        correlations[f'{pair_name}_scale'] = corr_vs_scale
        correlations[f'{pair_name}_angle'] = corr_vs_angle
        
        # Compute phase relationships if requested
        if compute_phase:
            phase_diff = compute_phase_difference(sf1, sf2, sf_centers)
            correlations[f'{pair_name}_phase'] = phase_diff
    
    return correlations


def compute_scale_dependent_transfer(
    results: Dict,
    source_channel: int = 0,  # e.g., velocity
    target_channel: int = 1,  # e.g., magnetic
    method: str = 'spectral'
) -> Dict[str, np.ndarray]:
    """Compute scale-dependent energy transfer between fields.
    
    Args:
        results: Output from analyze_slice
        source_channel: Source field channel index
        target_channel: Target field channel index
        method: Transfer calculation method ('spectral', 'flux', 'cascade')
        
    Returns:
        Dictionary with transfer functions and rates
    """
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    sf_bin_edges = results['sf_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    n_ell = len(ell_centers)
    
    transfer = {
        'ell_centers': ell_centers,
        'method': method,
    }
    
    # Get structure functions
    sf_source = hist_mag[source_channel]
    sf_target = hist_mag[target_channel]
    
    if method == 'spectral':
        # Compute spectral transfer function
        transfer_function = np.zeros(n_ell)
        coherence = np.zeros(n_ell)
        
        for ell_idx in range(n_ell):
            # Get PDFs at this scale (averaged over angles)
            pdf_source = sf_source[ell_idx].sum(axis=(0, 1))
            pdf_target = sf_target[ell_idx].sum(axis=(0, 1))
            
            if pdf_source.sum() > 0 and pdf_target.sum() > 0:
                # Normalize
                pdf_source = pdf_source / pdf_source.sum()
                pdf_target = pdf_target / pdf_target.sum()
                
                # Compute energy at this scale
                energy_source = np.sum(sf_centers**2 * pdf_source)
                energy_target = np.sum(sf_centers**2 * pdf_target)
                
                if energy_source > 0:
                    transfer_function[ell_idx] = energy_target / energy_source
                
                # Compute coherence
                coh = compute_coherence(pdf_source, pdf_target, sf_centers)
                coherence[ell_idx] = coh
        
        transfer['transfer_function'] = transfer_function
        transfer['coherence'] = coherence
        
    elif method == 'flux':
        # Compute energy flux across scales
        flux_forward = np.zeros(n_ell - 1)
        flux_backward = np.zeros(n_ell - 1)
        
        for ell_idx in range(n_ell - 1):
            # Compare adjacent scales
            sf_curr = sf_source[ell_idx].sum(axis=(0, 1))
            sf_next = sf_source[ell_idx + 1].sum(axis=(0, 1))
            
            if sf_curr.sum() > 0 and sf_next.sum() > 0:
                # Normalize
                pdf_curr = sf_curr / sf_curr.sum()
                pdf_next = sf_next / sf_next.sum()
                
                # Compute flux (simplified)
                energy_curr = np.sum(sf_centers**2 * pdf_curr)
                energy_next = np.sum(sf_centers**2 * pdf_next)
                
                scale_ratio = ell_centers[ell_idx + 1] / ell_centers[ell_idx]
                expected_scaling = scale_ratio**(2/3)  # Kolmogorov
                
                actual_ratio = energy_next / energy_curr if energy_curr > 0 else 0
                
                if actual_ratio > expected_scaling:
                    flux_forward[ell_idx] = actual_ratio - expected_scaling
                else:
                    flux_backward[ell_idx] = expected_scaling - actual_ratio
        
        transfer['flux_forward'] = flux_forward
        transfer['flux_backward'] = flux_backward
        transfer['net_flux'] = flux_forward - flux_backward
        
    elif method == 'cascade':
        # Compute cascade rates
        cascade_rate = compute_cascade_rate(sf_source, sf_target, ell_centers, sf_centers)
        transfer['cascade_rate'] = cascade_rate
    
    return transfer


def compute_nonlinear_coupling(
    results: Dict,
    triplet: Tuple[int, int, int] = (0, 1, 7),  # velocity, magnetic, current
    compute_bispectrum: bool = True
) -> Dict[str, np.ndarray]:
    """Compute nonlinear coupling between three field quantities.
    
    This measures three-wave interactions and energy transfer mechanisms.
    
    Args:
        results: Output from analyze_slice
        triplet: Three channel indices for coupling analysis
        compute_bispectrum: Whether to compute bispectral measures
        
    Returns:
        Dictionary with coupling strengths and phase relationships
    """
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    sf_bin_edges = results['sf_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    
    ch1, ch2, ch3 = triplet
    
    coupling = {
        'triplet': triplet,
        'ell_centers': ell_centers,
    }
    
    # Get structure functions
    sf1 = hist_mag[ch1]
    sf2 = hist_mag[ch2]
    sf3 = hist_mag[ch3]
    
    n_ell = len(ell_centers)
    
    # Compute triadic interaction strength
    triadic_strength = np.zeros(n_ell)
    
    for ell_idx in range(n_ell):
        # Get PDFs at this scale
        pdf1 = sf1[ell_idx].sum(axis=(0, 1))
        pdf2 = sf2[ell_idx].sum(axis=(0, 1))
        pdf3 = sf3[ell_idx].sum(axis=(0, 1))
        
        if pdf1.sum() > 0 and pdf2.sum() > 0 and pdf3.sum() > 0:
            # Normalize
            pdf1 = pdf1 / pdf1.sum()
            pdf2 = pdf2 / pdf2.sum()
            pdf3 = pdf3 / pdf3.sum()
            
            # Compute third moment (skewness-like)
            m3_1 = np.sum(sf_centers**3 * pdf1)
            m3_2 = np.sum(sf_centers**3 * pdf2)
            m3_3 = np.sum(sf_centers**3 * pdf3)
            
            # Triadic coupling strength (simplified)
            m2_1 = np.sum(sf_centers**2 * pdf1)
            m2_2 = np.sum(sf_centers**2 * pdf2)
            m2_3 = np.sum(sf_centers**2 * pdf3)
            
            if m2_1 > 0 and m2_2 > 0 and m2_3 > 0:
                # Normalized triple correlation
                triadic_strength[ell_idx] = (m3_1 * m3_2 * m3_3)**(1/3) / (m2_1 * m2_2 * m2_3)**(1/2)
    
    coupling['triadic_strength'] = triadic_strength
    
    if compute_bispectrum:
        # Compute bispectral measures
        bicoherence = compute_bicoherence(sf1, sf2, sf3, ell_centers)
        coupling['bicoherence'] = bicoherence
    
    # Compute energy exchange rates
    exchange_12_3 = np.zeros(n_ell)  # From fields 1,2 to field 3
    exchange_13_2 = np.zeros(n_ell)  # From fields 1,3 to field 2
    exchange_23_1 = np.zeros(n_ell)  # From fields 2,3 to field 1
    
    for ell_idx in range(n_ell):
        # Simplified energy exchange calculation
        e1 = compute_field_energy(sf1[ell_idx], sf_centers)
        e2 = compute_field_energy(sf2[ell_idx], sf_centers)
        e3 = compute_field_energy(sf3[ell_idx], sf_centers)
        
        total_e = e1 + e2 + e3
        if total_e > 0:
            # Compute exchange based on relative energies and coupling
            coupling_factor = triadic_strength[ell_idx]
            
            exchange_12_3[ell_idx] = coupling_factor * (e1 + e2 - 2*e3) / total_e
            exchange_13_2[ell_idx] = coupling_factor * (e1 + e3 - 2*e2) / total_e
            exchange_23_1[ell_idx] = coupling_factor * (e2 + e3 - 2*e1) / total_e
    
    coupling['exchange_12_to_3'] = exchange_12_3
    coupling['exchange_13_to_2'] = exchange_13_2
    coupling['exchange_23_to_1'] = exchange_23_1
    
    return coupling


def compute_conditional_statistics(
    results: Dict,
    condition_channel: int = 2,  # e.g., density
    target_channel: int = 0,     # e.g., velocity
    n_bins: int = 5,
    percentile_bins: bool = True
) -> Dict[str, np.ndarray]:
    """Compute statistics of one field conditioned on another.
    
    This reveals how turbulence properties depend on local conditions.
    
    Args:
        results: Output from analyze_slice
        condition_channel: Channel to condition on
        target_channel: Channel to compute statistics for
        n_bins: Number of conditioning bins
        percentile_bins: Use percentile-based bins (True) or uniform (False)
        
    Returns:
        Dictionary with conditional statistics
    """
    hist_mag = results['hist_mag']
    sf_bin_edges = results['sf_bin_edges']
    ell_bin_edges = results['ell_bin_edges']
    
    sf_centers = 0.5 * (sf_bin_edges[:-1] + sf_bin_edges[1:])
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    
    # Get structure functions
    sf_condition = hist_mag[condition_channel]
    sf_target = hist_mag[target_channel]
    
    n_ell = len(ell_centers)
    
    conditional = {
        'ell_centers': ell_centers,
        'sf_centers': sf_centers,
        'n_bins': n_bins,
    }
    
    # For each scale, compute conditional statistics
    cond_mean = np.zeros((n_ell, n_bins))
    cond_std = np.zeros((n_ell, n_bins))
    cond_skew = np.zeros((n_ell, n_bins))
    cond_pdf = np.zeros((n_ell, n_bins, len(sf_centers)))
    
    for ell_idx in range(n_ell):
        # Get joint distribution at this scale
        cond_dist = sf_condition[ell_idx].sum(axis=(0, 1))  # Average over angles
        target_dist = sf_target[ell_idx].sum(axis=(0, 1))
        
        if cond_dist.sum() > 0 and target_dist.sum() > 0:
            # Normalize
            cond_pdf_norm = cond_dist / cond_dist.sum()
            
            # Determine bin edges for conditioning variable
            if percentile_bins:
                # Use quantiles
                cumsum = np.cumsum(cond_pdf_norm)
                bin_edges = []
                for i in range(n_bins + 1):
                    q = i / n_bins
                    idx = np.searchsorted(cumsum, q)
                    if idx < len(sf_centers):
                        bin_edges.append(sf_centers[idx])
                    else:
                        bin_edges.append(sf_centers[-1])
                bin_edges = np.array(bin_edges)
            else:
                # Uniform bins
                bin_edges = np.linspace(sf_centers[0], sf_centers[-1], n_bins + 1)
            
            # Compute statistics in each bin
            for bin_idx in range(n_bins):
                # Find points in this conditioning bin
                mask = (sf_centers >= bin_edges[bin_idx]) & \
                       (sf_centers < bin_edges[bin_idx + 1])
                
                if bin_idx == n_bins - 1:  # Include right edge in last bin
                    mask = (sf_centers >= bin_edges[bin_idx]) & \
                           (sf_centers <= bin_edges[bin_idx + 1])
                
                if mask.sum() > 0:
                    # Weight by conditioning distribution
                    weights = cond_pdf_norm[mask]
                    values = sf_centers[mask]
                    
                    # Get target distribution in this bin
                    target_in_bin = target_dist[mask]
                    
                    if target_in_bin.sum() > 0:
                        target_pdf = target_in_bin / target_in_bin.sum()
                        
                        # Compute moments
                        mean = np.average(values, weights=target_pdf)
                        var = np.average((values - mean)**2, weights=target_pdf)
                        
                        cond_mean[ell_idx, bin_idx] = mean
                        cond_std[ell_idx, bin_idx] = np.sqrt(var)
                        
                        if var > 0:
                            skew = np.average((values - mean)**3, weights=target_pdf)
                            cond_skew[ell_idx, bin_idx] = skew / var**(3/2)
                        
                        # Store full PDF
                        cond_pdf[ell_idx, bin_idx, mask] = target_pdf
    
    conditional['mean'] = cond_mean
    conditional['std'] = cond_std
    conditional['skewness'] = cond_skew
    conditional['pdf'] = cond_pdf
    conditional['bin_edges'] = bin_edges if 'bin_edges' in locals() else None
    
    return conditional


def compute_mutual_information(
    results: Dict,
    channel_pairs: Optional[List[Tuple[int, int]]] = None,
    n_bins: int = 20
) -> Dict[str, np.ndarray]:
    """Compute mutual information between field pairs.
    
    Mutual information measures nonlinear dependencies beyond correlation.
    
    Args:
        results: Output from analyze_slice
        channel_pairs: List of channel pairs to analyze
        n_bins: Number of bins for entropy estimation
        
    Returns:
        Dictionary with mutual information measures
    """
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    
    if channel_pairs is None:
        channel_pairs = [(0, 1), (0, 2), (1, 2)]
    
    n_ell = len(ell_centers)
    
    mi_results = {
        'ell_centers': ell_centers,
        'channel_pairs': channel_pairs,
    }
    
    for ch1, ch2 in channel_pairs:
        mi_scale = np.zeros(n_ell)
        
        for ell_idx in range(n_ell):
            # Get marginal distributions
            p1 = hist_mag[ch1][ell_idx].sum(axis=(0, 1))
            p2 = hist_mag[ch2][ell_idx].sum(axis=(0, 1))
            
            if p1.sum() > 0 and p2.sum() > 0:
                # Normalize
                p1 = p1 / p1.sum()
                p2 = p2 / p2.sum()
                
                # Compute entropies
                h1 = -np.sum(p1[p1 > 0] * np.log(p1[p1 > 0]))
                h2 = -np.sum(p2[p2 > 0] * np.log(p2[p2 > 0]))
                
                # Estimate joint entropy (simplified - assumes some independence)
                # In practice, would need actual joint distribution
                h_joint = h1 + h2 - 0.5 * min(h1, h2)  # Approximate
                
                # Mutual information
                mi = h1 + h2 - h_joint
                mi_scale[ell_idx] = mi
        
        pair_name = f"ch{ch1}_ch{ch2}"
        mi_results[f'mi_{pair_name}'] = mi_scale
        
        # Normalized mutual information (0 to 1)
        max_mi = np.log(n_bins)  # Maximum possible MI
        mi_results[f'nmi_{pair_name}'] = mi_scale / max_mi
    
    return mi_results


# Helper functions

def compute_phase_difference(sf1: np.ndarray, sf2: np.ndarray, 
                            sf_centers: np.ndarray) -> np.ndarray:
    """Compute phase difference between two fields."""
    n_ell = sf1.shape[0]
    phase_diff = np.zeros(n_ell)
    
    for ell_idx in range(n_ell):
        # Get PDFs
        pdf1 = sf1[ell_idx].sum(axis=(0, 1))
        pdf2 = sf2[ell_idx].sum(axis=(0, 1))
        
        if pdf1.sum() > 0 and pdf2.sum() > 0:
            # Use cross-correlation to find phase shift
            xcorr = np.correlate(pdf1, pdf2, mode='same')
            lag = np.argmax(xcorr) - len(pdf1) // 2
            
            # Convert to phase
            phase_diff[ell_idx] = 2 * np.pi * lag / len(pdf1)
    
    return phase_diff


def compute_coherence(pdf1: np.ndarray, pdf2: np.ndarray, 
                     sf_centers: np.ndarray) -> float:
    """Compute coherence between two PDFs."""
    if len(pdf1) != len(pdf2):
        return 0.0
    
    # Compute cross-spectral density (simplified)
    fft1 = np.fft.fft(pdf1)
    fft2 = np.fft.fft(pdf2)
    
    cross_spec = fft1 * np.conj(fft2)
    auto_spec1 = fft1 * np.conj(fft1)
    auto_spec2 = fft2 * np.conj(fft2)
    
    # Coherence
    coh_squared = np.abs(cross_spec)**2 / (np.abs(auto_spec1) * np.abs(auto_spec2) + 1e-10)
    
    return np.mean(np.real(coh_squared))


def compute_cascade_rate(sf_source: np.ndarray, sf_target: np.ndarray,
                         ell_centers: np.ndarray, sf_centers: np.ndarray) -> np.ndarray:
    """Compute energy cascade rate between scales."""
    n_ell = len(ell_centers)
    cascade_rate = np.zeros(n_ell - 1)
    
    for ell_idx in range(n_ell - 1):
        # Energy at current and next scale
        e_curr = compute_field_energy(sf_source[ell_idx], sf_centers)
        e_next = compute_field_energy(sf_source[ell_idx + 1], sf_centers)
        
        # Time scale (eddy turnover)
        if e_curr > 0:
            tau = ell_centers[ell_idx] / np.sqrt(e_curr)
            cascade_rate[ell_idx] = (e_curr - e_next) / tau
    
    return cascade_rate


def compute_field_energy(sf: np.ndarray, sf_centers: np.ndarray) -> float:
    """Compute energy from structure function."""
    pdf = sf.sum(axis=(0, 1))  # Average over angles
    if pdf.sum() > 0:
        pdf = pdf / pdf.sum()
        return np.sum(sf_centers**2 * pdf)
    return 0.0


def compute_bicoherence(sf1: np.ndarray, sf2: np.ndarray, sf3: np.ndarray,
                        ell_centers: np.ndarray) -> np.ndarray:
    """Compute bicoherence for three-wave interactions."""
    n_ell = len(ell_centers)
    bicoherence = np.zeros(n_ell)
    
    for ell_idx in range(n_ell):
        # Get PDFs
        pdf1 = sf1[ell_idx].sum(axis=(0, 1))
        pdf2 = sf2[ell_idx].sum(axis=(0, 1))
        pdf3 = sf3[ell_idx].sum(axis=(0, 1))
        
        if pdf1.sum() > 0 and pdf2.sum() > 0 and pdf3.sum() > 0:
            # FFT of each
            fft1 = np.fft.fft(pdf1 / pdf1.sum())
            fft2 = np.fft.fft(pdf2 / pdf2.sum())
            fft3 = np.fft.fft(pdf3 / pdf3.sum())
            
            # Bispectrum (simplified - diagonal only)
            n = len(fft1)
            bispec = 0
            norm = 0
            
            for i in range(n // 2):
                for j in range(n // 2):
                    k = (i + j) % n
                    if k < n // 2:
                        bispec += fft1[i] * fft2[j] * np.conj(fft3[k])
                        norm += np.abs(fft1[i] * fft2[j])**2
            
            if norm > 0:
                bicoherence[ell_idx] = np.abs(bispec) / np.sqrt(norm)
    
    return bicoherence