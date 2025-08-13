"""Scale-dependent anisotropy analysis for MHD turbulence.

This module provides advanced tools for analyzing anisotropic properties:
1. Scale-dependent anisotropy measures
2. Directional structure functions
3. Alignment statistics
4. Anisotropy spectrograms
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings


def compute_scale_dependent_anisotropy(
    results: Dict,
    method: str = 'ratio',
    reference_direction: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Compute anisotropy measures as a function of scale.
    
    Args:
        results: Output from analyze_slice
        method: Anisotropy measure ('ratio', 'variance', 'entropy', 'alignment')
        reference_direction: Reference direction for alignment (e.g., mean B field)
        
    Returns:
        Dictionary with scale-dependent anisotropy measures
    """
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    theta_bin_edges = results['theta_bin_edges']
    phi_bin_edges = results['phi_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    phi_centers = 0.5 * (phi_bin_edges[:-1] + phi_bin_edges[1:])
    
    n_channels = hist_mag.shape[0]
    n_ell = len(ell_centers)
    
    anisotropy = {
        'ell_centers': ell_centers,
        'theta_centers': theta_centers,
        'phi_centers': phi_centers,
    }
    
    if method == 'ratio':
        # Compute parallel/perpendicular ratio at each scale
        aniso_ratio = compute_parallel_perp_ratio(hist_mag, theta_centers)
        anisotropy['parallel_perp_ratio'] = aniso_ratio
        
    elif method == 'variance':
        # Compute angular variance at each scale
        aniso_var = compute_angular_variance(hist_mag, theta_centers, phi_centers)
        anisotropy['angular_variance'] = aniso_var
        
    elif method == 'entropy':
        # Compute angular entropy at each scale
        aniso_entropy = compute_angular_entropy(hist_mag)
        anisotropy['angular_entropy'] = aniso_entropy
        
    elif method == 'alignment':
        # Compute alignment with reference direction
        if reference_direction is None:
            # Default: assume B field along z (θ=0)
            reference_direction = np.array([0, 0, 1])
        aniso_align = compute_alignment_measure(hist_mag, theta_centers, phi_centers, 
                                               reference_direction)
        anisotropy['alignment'] = aniso_align
    
    # Compute anisotropy strength (scale-dependent)
    aniso_strength = compute_anisotropy_strength(hist_mag, theta_centers)
    anisotropy['strength'] = aniso_strength
    
    # Compute preferential directions
    pref_directions = compute_preferential_directions(hist_mag, theta_centers, phi_centers)
    anisotropy['preferential_theta'] = pref_directions[0]
    anisotropy['preferential_phi'] = pref_directions[1]
    
    return anisotropy


def compute_parallel_perp_ratio(
    hist_mag: np.ndarray,
    theta_centers: np.ndarray
) -> np.ndarray:
    """Compute ratio of parallel to perpendicular structure functions.
    
    Args:
        hist_mag: Histogram array from analyze_slice
        theta_centers: Centers of theta bins
        
    Returns:
        Array of shape (n_channels, n_ell) with ratios
    """
    n_channels, n_ell = hist_mag.shape[:2]
    
    # Define parallel and perpendicular regions
    parallel_mask = (theta_centers < np.pi/6) | (theta_centers > 5*np.pi/6)
    perp_mask = (theta_centers > np.pi/3) & (theta_centers < 2*np.pi/3)
    
    ratios = np.zeros((n_channels, n_ell))
    
    for ch in range(n_channels):
        for ell_idx in range(n_ell):
            # Get SF at this scale
            sf_ell = hist_mag[ch, ell_idx]  # Shape: (n_theta, n_phi, n_sf)
            
            # Sum over parallel directions
            sf_parallel = sf_ell[parallel_mask].sum()
            
            # Sum over perpendicular directions
            sf_perp = sf_ell[perp_mask].sum()
            
            if sf_perp > 0:
                ratios[ch, ell_idx] = sf_parallel / sf_perp
            else:
                ratios[ch, ell_idx] = np.nan
    
    return ratios


def compute_angular_variance(
    hist_mag: np.ndarray,
    theta_centers: np.ndarray,
    phi_centers: np.ndarray
) -> np.ndarray:
    """Compute variance of angular distribution at each scale.
    
    Higher variance indicates stronger anisotropy.
    
    Args:
        hist_mag: Histogram array
        theta_centers: Centers of theta bins
        phi_centers: Centers of phi bins
        
    Returns:
        Array of shape (n_channels, n_ell) with variances
    """
    n_channels, n_ell = hist_mag.shape[:2]
    variances = np.zeros((n_channels, n_ell))
    
    for ch in range(n_channels):
        for ell_idx in range(n_ell):
            # Get angular distribution at this scale
            angular_dist = hist_mag[ch, ell_idx].sum(axis=2)  # Sum over SF values
            
            if angular_dist.sum() > 0:
                # Normalize to probability
                angular_dist = angular_dist / angular_dist.sum()
                
                # Compute mean direction
                mean_theta = np.sum(angular_dist.sum(axis=1) * theta_centers)
                mean_phi = np.sum(angular_dist.sum(axis=0) * phi_centers)
                
                # Compute variance
                theta_var = 0
                phi_var = 0
                for i, theta in enumerate(theta_centers):
                    for j, phi in enumerate(phi_centers):
                        weight = angular_dist[i, j]
                        theta_var += weight * (theta - mean_theta)**2
                        # Handle circular variance for phi
                        dphi = np.abs(phi - mean_phi)
                        dphi = min(dphi, 2*np.pi - dphi)
                        phi_var += weight * dphi**2
                
                variances[ch, ell_idx] = np.sqrt(theta_var + phi_var)
            else:
                variances[ch, ell_idx] = 0
    
    return variances


def compute_angular_entropy(hist_mag: np.ndarray) -> np.ndarray:
    """Compute entropy of angular distribution at each scale.
    
    Lower entropy indicates stronger anisotropy (more ordered).
    
    Args:
        hist_mag: Histogram array
        
    Returns:
        Array of shape (n_channels, n_ell) with entropies
    """
    n_channels, n_ell = hist_mag.shape[:2]
    entropies = np.zeros((n_channels, n_ell))
    
    for ch in range(n_channels):
        for ell_idx in range(n_ell):
            # Get angular distribution
            angular_dist = hist_mag[ch, ell_idx].sum(axis=2)  # Sum over SF values
            
            if angular_dist.sum() > 0:
                # Normalize and flatten
                p = angular_dist.flatten()
                p = p / p.sum()
                
                # Compute entropy
                # Add small epsilon to avoid log(0)
                p_nonzero = p[p > 0]
                entropy = -np.sum(p_nonzero * np.log(p_nonzero))
                
                # Normalize by maximum entropy (uniform distribution)
                max_entropy = np.log(len(p))
                entropies[ch, ell_idx] = entropy / max_entropy
            else:
                entropies[ch, ell_idx] = 1.0  # Maximum entropy for empty
    
    return entropies


def compute_alignment_measure(
    hist_mag: np.ndarray,
    theta_centers: np.ndarray,
    phi_centers: np.ndarray,
    reference_direction: np.ndarray
) -> np.ndarray:
    """Compute alignment with a reference direction at each scale.
    
    Args:
        hist_mag: Histogram array
        theta_centers: Centers of theta bins
        phi_centers: Centers of phi bins
        reference_direction: 3D unit vector for reference direction
        
    Returns:
        Array of shape (n_channels, n_ell) with alignment measures [0, 1]
    """
    n_channels, n_ell = hist_mag.shape[:2]
    alignments = np.zeros((n_channels, n_ell))
    
    # Normalize reference direction
    ref_dir = reference_direction / np.linalg.norm(reference_direction)
    ref_theta = np.arccos(ref_dir[2])
    ref_phi = np.arctan2(ref_dir[1], ref_dir[0])
    
    for ch in range(n_channels):
        for ell_idx in range(n_ell):
            # Get angular distribution
            angular_dist = hist_mag[ch, ell_idx].sum(axis=2)
            
            if angular_dist.sum() > 0:
                angular_dist = angular_dist / angular_dist.sum()
                
                # Compute mean alignment
                alignment = 0
                for i, theta in enumerate(theta_centers):
                    for j, phi in enumerate(phi_centers):
                        # Convert to Cartesian
                        x = np.sin(theta) * np.cos(phi)
                        y = np.sin(theta) * np.sin(phi)
                        z = np.cos(theta)
                        
                        # Dot product with reference
                        dot = x * ref_dir[0] + y * ref_dir[1] + z * ref_dir[2]
                        
                        # Weight by probability
                        alignment += angular_dist[i, j] * np.abs(dot)
                
                alignments[ch, ell_idx] = alignment
            else:
                alignments[ch, ell_idx] = 0
    
    return alignments


def compute_anisotropy_strength(
    hist_mag: np.ndarray,
    theta_centers: np.ndarray
) -> np.ndarray:
    """Compute overall anisotropy strength at each scale.
    
    Uses the ratio of maximum to minimum directional SF.
    
    Args:
        hist_mag: Histogram array
        theta_centers: Centers of theta bins
        
    Returns:
        Array of shape (n_channels, n_ell) with strength values
    """
    n_channels, n_ell = hist_mag.shape[:2]
    strengths = np.zeros((n_channels, n_ell))
    
    for ch in range(n_channels):
        for ell_idx in range(n_ell):
            # Get directional SFs
            directional_sf = hist_mag[ch, ell_idx].sum(axis=2)  # Sum over SF values
            
            # Sum over phi for each theta
            theta_sf = directional_sf.sum(axis=1)
            
            if theta_sf.sum() > 0 and len(theta_sf) > 1:
                max_sf = theta_sf.max()
                min_sf = theta_sf[theta_sf > 0].min() if any(theta_sf > 0) else 0
                
                if min_sf > 0:
                    strengths[ch, ell_idx] = max_sf / min_sf
                else:
                    strengths[ch, ell_idx] = np.inf if max_sf > 0 else 1.0
            else:
                strengths[ch, ell_idx] = 1.0
    
    return strengths


def compute_preferential_directions(
    hist_mag: np.ndarray,
    theta_centers: np.ndarray,
    phi_centers: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Find preferential directions at each scale.
    
    Args:
        hist_mag: Histogram array
        theta_centers: Centers of theta bins
        phi_centers: Centers of phi bins
        
    Returns:
        Tuple of (preferred_theta, preferred_phi) arrays of shape (n_channels, n_ell)
    """
    n_channels, n_ell = hist_mag.shape[:2]
    pref_theta = np.zeros((n_channels, n_ell))
    pref_phi = np.zeros((n_channels, n_ell))
    
    for ch in range(n_channels):
        for ell_idx in range(n_ell):
            # Get angular distribution
            angular_dist = hist_mag[ch, ell_idx].sum(axis=2)
            
            if angular_dist.sum() > 0:
                # Find maximum
                max_idx = np.unravel_index(angular_dist.argmax(), angular_dist.shape)
                pref_theta[ch, ell_idx] = theta_centers[max_idx[0]]
                pref_phi[ch, ell_idx] = phi_centers[max_idx[1]]
            else:
                pref_theta[ch, ell_idx] = np.nan
                pref_phi[ch, ell_idx] = np.nan
    
    return pref_theta, pref_phi


def create_anisotropy_spectrogram(
    results: Dict,
    channel: int = 0,
    cmap: str = 'RdBu_r'
) -> Dict[str, np.ndarray]:
    """Create a spectrogram showing anisotropy as a function of scale and angle.
    
    Args:
        results: Output from analyze_slice
        channel: Which channel to analyze
        cmap: Colormap name
        
    Returns:
        Dictionary with spectrogram data ready for plotting
    """
    hist_mag = results['hist_mag']
    ell_bin_edges = results['ell_bin_edges']
    theta_bin_edges = results['theta_bin_edges']
    
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    
    # Extract channel and average over phi and SF values
    channel_data = hist_mag[channel]  # Shape: (n_ell, n_theta, n_phi, n_sf)
    spectrogram = channel_data.sum(axis=(2, 3)).astype(float)  # Sum over phi and SF, ensure float
    
    # Normalize each scale
    for ell_idx in range(len(ell_centers)):
        if spectrogram[ell_idx].sum() > 0:
            spectrogram[ell_idx] = spectrogram[ell_idx] / spectrogram[ell_idx].sum()
    
    # Compute anisotropy ratio at each (ell, theta)
    mean_per_scale = spectrogram.mean(axis=1, keepdims=True)
    aniso_ratio = np.zeros_like(spectrogram)
    
    for ell_idx in range(len(ell_centers)):
        if mean_per_scale[ell_idx] > 0:
            aniso_ratio[ell_idx] = (spectrogram[ell_idx] - mean_per_scale[ell_idx]) / mean_per_scale[ell_idx]
    
    return {
        'spectrogram': spectrogram,
        'anisotropy_ratio': aniso_ratio,
        'ell_centers': ell_centers,
        'theta_centers': theta_centers,
        'cmap': cmap
    }


def decompose_anisotropy_modes(
    results: Dict,
    n_modes: int = 3
) -> Dict[str, np.ndarray]:
    """Decompose anisotropy into principal modes using SVD.
    
    Args:
        results: Output from analyze_slice
        n_modes: Number of modes to extract
        
    Returns:
        Dictionary with mode decomposition
    """
    hist_mag = results['hist_mag']
    n_channels, n_ell, n_theta, n_phi, n_sf = hist_mag.shape
    
    # Reshape for SVD: (n_channels, n_ell) x (n_theta * n_phi)
    data_matrix = []
    
    for ch in range(n_channels):
        for ell_idx in range(n_ell):
            angular_pattern = hist_mag[ch, ell_idx].sum(axis=2)  # Sum over SF
            if angular_pattern.sum() > 0:
                angular_pattern = angular_pattern / angular_pattern.sum()
            data_matrix.append(angular_pattern.flatten())
    
    data_matrix = np.array(data_matrix)
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    
    # Extract principal modes
    modes = {
        'spatial_modes': U[:, :n_modes],  # Scale/channel patterns
        'angular_modes': Vt[:n_modes].reshape(n_modes, n_theta, n_phi),  # Angular patterns
        'singular_values': S[:n_modes],
        'variance_explained': S[:n_modes]**2 / (S**2).sum()
    }
    
    return modes