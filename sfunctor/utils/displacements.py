"""Utility functions for generating displacement vectors used in SF histograms.

This module provides functions for creating displacement vectors used in
structure function calculations. The displacements are sampled in log-space
to ensure good coverage across multiple scales.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple
import logging
import warnings

# Set up module logger
logger = logging.getLogger(__name__)

__all__ = [
    "find_ell_bin_edges",
    "build_displacement_list",
]


def find_ell_bin_edges(r_min: float, r_max: float, n_ell_bins: int) -> np.ndarray:
    """Return logarithmically spaced *integer* ℓ-bin edges (length = n_ell_bins+1).

    This function finds optimal integer bin edges for logarithmic binning
    of displacement magnitudes. It uses a binary search to find the right
    number of points that yield exactly n_ell_bins+1 unique integer edges.

    Parameters
    ----------
    r_min : float
        Minimum displacement magnitude. Must be positive.
    r_max : float
        Maximum displacement magnitude. Must be greater than r_min.
    n_ell_bins : int
        Number of bins desired. Must be positive.

    Returns
    -------
    np.ndarray
        Array of n_ell_bins+1 unique integer bin edges, logarithmically spaced.

    Raises
    ------
    ValueError
        If r_min <= 0, r_max <= r_min, or n_ell_bins < 1.
    RuntimeWarning
        If the exact number of requested bins cannot be achieved.

    Examples
    --------
    >>> edges = find_ell_bin_edges(1.0, 100.0, 10)
    >>> print(len(edges))  # Should be 11
    """
    # Input validation
    if r_min <= 0:
        raise ValueError(f"r_min must be positive, got {r_min}")
    if r_max <= r_min:
        raise ValueError(f"r_max ({r_max}) must be greater than r_min ({r_min})")
    if n_ell_bins < 1:
        raise ValueError(f"n_ell_bins must be at least 1, got {n_ell_bins}")
    
    # Binary search for optimal number of points
    n_points_low = n_ell_bins + 1
    n_points_high = min(3 * n_ell_bins, 1000)  # Cap to prevent excessive iterations
    
    best_edges = None
    best_diff = float('inf')
    
    while n_points_low <= n_points_high:
        mid = (n_points_low + n_points_high) // 2
        try:
            # Generate logarithmically spaced points and round to integers
            points = np.geomspace(r_min, r_max, mid)
            edges = np.unique(np.around(points).astype(int))
            
            # Track best result even if not exact
            diff = abs(len(edges) - (n_ell_bins + 1))
            if diff < best_diff:
                best_diff = diff
                best_edges = edges
            
            if len(edges) < n_ell_bins + 1:
                n_points_low = mid + 1
            elif len(edges) > n_ell_bins + 1:
                n_points_high = mid - 1
            else:
                # Found exact match
                break
        except Exception as e:
            logger.error(f"Error in binary search: {e}")
            raise RuntimeError(f"Failed to generate bin edges: {e}") from e
    
    # Use best result found
    edges = best_edges
    
    if len(edges) != n_ell_bins + 1:
        warnings.warn(
            f"Could not achieve exactly {n_ell_bins+1} edges. "
            f"Got {len(edges)} edges instead. Consider adjusting r_min/r_max.",
            RuntimeWarning
        )
        logger.warning(
            f"Bin edge mismatch: requested {n_ell_bins+1}, got {len(edges)} "
            f"for range [{r_min}, {r_max}]"
        )
    
    return edges.astype(float)


def build_displacement_list(
    ell_bin_edges: np.ndarray,
    n_disp_total: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate displacement vectors grouped per ℓ-bin then de-duplicated.

    This function creates a set of 2D displacement vectors distributed
    logarithmically in magnitude and uniformly in angle (0 to π). Mirror
    duplicates are removed to avoid redundant calculations.

    Parameters
    ----------
    ell_bin_edges : np.ndarray
        Array of bin edges from :func:`find_ell_bin_edges`. Must be
        monotonically increasing with at least 2 elements.
    n_disp_total : int
        Total number of displacement vectors desired (before mirror removal).
        Must be positive.
    seed : int, optional
        Random seed for reproducibility. If None, uses numpy's current state.

    Returns
    -------
    np.ndarray
        Array of shape *(N, 2)* with integer *(dx, dy)* displacements sorted
        by |Δ|. N will be less than n_disp_total due to de-duplication and
        mirror removal.

    Raises
    ------
    ValueError
        If inputs are invalid (empty edges, non-monotonic edges, n_disp_total <= 0).

    Notes
    -----
    Mirror duplicates (dx, dy) and (-dx, -dy) represent the same displacement
    for structure function calculations, so only one is kept.

    Examples
    --------
    >>> edges = np.array([1.0, 10.0, 100.0])
    >>> disps = build_displacement_list(edges, 1000, seed=42)
    >>> print(disps.shape)  # Will be (N, 2) where N < 1000
    """
    # Input validation
    if not isinstance(ell_bin_edges, np.ndarray):
        raise TypeError(
            f"ell_bin_edges must be a numpy array, got {type(ell_bin_edges).__name__}"
        )
    
    if len(ell_bin_edges) < 2:
        raise ValueError(
            f"ell_bin_edges must have at least 2 elements, got {len(ell_bin_edges)}"
        )
    
    if not np.all(np.diff(ell_bin_edges) > 0):
        raise ValueError("ell_bin_edges must be monotonically increasing")
    
    if n_disp_total <= 0:
        raise ValueError(f"n_disp_total must be positive, got {n_disp_total}")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    n_ell_bins = len(ell_bin_edges) - 1
    n_per_bin = max(1, n_disp_total // n_ell_bins)  # At least 1 per bin
    
    if n_per_bin < 10:
        warnings.warn(
            f"Only {n_per_bin} displacements per bin. Consider increasing n_disp_total "
            f"(currently {n_disp_total}) for better sampling.",
            RuntimeWarning
        )
    
    disp_list = []
    for i in range(n_ell_bins):
        r_low = ell_bin_edges[i]
        r_high = ell_bin_edges[i + 1]
        
        if r_low >= r_high:
            logger.warning(f"Skipping bin {i}: r_low={r_low} >= r_high={r_high}")
            continue
        
        try:
            # Generate radii logarithmically spaced within bin
            r_values = np.geomspace(r_low, r_high, n_per_bin)
            
            # Random angles from 0 to pi (half circle due to mirror symmetry)
            angles = np.random.uniform(0, np.pi, n_per_bin)
            
            # Convert to integer displacements
            dx = np.rint(r_values * np.cos(angles)).astype(np.int32)
            dy = np.rint(r_values * np.sin(angles)).astype(np.int32)
            
            # Remove duplicates within this bin
            disp_bin = np.unique(np.column_stack((dx, dy)), axis=0)
            
            # Filter out zero displacement
            nonzero_mask = (disp_bin[:, 0] != 0) | (disp_bin[:, 1] != 0)
            disp_bin = disp_bin[nonzero_mask]
            
            if len(disp_bin) > 0:
                disp_list.append(disp_bin)
            
        except Exception as e:
            logger.error(f"Error generating displacements for bin {i}: {e}")
            raise RuntimeError(
                f"Failed to generate displacements for bin {i} [{r_low}, {r_high}]: {e}"
            ) from e
    
    if not disp_list:
        raise RuntimeError("No valid displacements generated")
    
    displacements = np.vstack(disp_list)
    
    # Remove mirror duplicates: (dx,dy) and (-dx,-dy) represent same displacement
    # Keep the one with positive dx, or positive dy if dx=0
    mask = (displacements[:, 0] < 0) | (
        (displacements[:, 0] == 0) & (displacements[:, 1] < 0)
    )
    canonical = np.where(mask[:, None], -displacements, displacements)
    unique_canonical = np.unique(canonical, axis=0)
    
    # Sort by absolute length
    lengths = np.sqrt(unique_canonical[:, 0] ** 2 + unique_canonical[:, 1] ** 2)
    idx_sort = np.argsort(lengths)
    
    result = unique_canonical[idx_sort]
    
    logger.info(
        f"Generated {len(result)} unique displacements from {n_disp_total} requested "
        f"(reduction due to de-duplication and mirror removal)"
    )
    
    return result 