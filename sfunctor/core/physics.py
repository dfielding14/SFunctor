"""Physics calculations for MHD structure function analysis.

This module provides functions for computing derived physical quantities
from primitive MHD variables. All functions are optimized for performance
and include proper error handling for numerical edge cases.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
import logging
import warnings

# Set up module logger
logger = logging.getLogger(__name__)

__all__ = [
    "compute_vA",
    "compute_z_plus_minus",
]


def compute_vA(B_x: np.ndarray, B_y: np.ndarray, B_z: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vector Alfvén speed **v_A = B / sqrt(rho)** for a 2-D slice.

    Parameters
    ----------
    B_x, B_y, B_z : np.ndarray
        Magnetic-field components on the slice. Must be 2D arrays with
        identical shapes.
    rho : np.ndarray
        Mass density on the slice. Must have the same shape as B components
        and contain positive values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Alfvén velocity components (vA_x, vA_y, vA_z) with same shape as inputs.

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have inconsistent shapes, are not 2D, or if rho has
        non-positive values.

    Notes
    -----
    For numerical stability, very small or negative density values are
    handled by setting the corresponding Alfvén speed to zero with a warning.

    Examples
    --------
    >>> B_x = np.ones((100, 100))
    >>> B_y = np.zeros((100, 100))
    >>> B_z = np.zeros((100, 100))
    >>> rho = np.full((100, 100), 4.0)
    >>> vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    >>> print(vA_x[0, 0])  # Should be 0.5
    """
    # Input validation
    arrays = {'B_x': B_x, 'B_y': B_y, 'B_z': B_z, 'rho': rho}
    
    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got {arr.ndim}D")
    
    # Check shape consistency
    ref_shape = B_x.shape
    for name, arr in arrays.items():
        if arr.shape != ref_shape:
            raise ValueError(
                f"All arrays must have the same shape. {name} has shape {arr.shape}, "
                f"expected {ref_shape}"
            )
    
    # Handle density edge cases
    min_rho = 1e-10  # Minimum density threshold
    
    if np.any(rho <= 0):
        n_negative = np.sum(rho <= 0)
        warnings.warn(
            f"Found {n_negative} non-positive density values. "
            f"Setting corresponding Alfvén speeds to zero.",
            RuntimeWarning
        )
        logger.warning(f"Non-positive density values detected: min={np.min(rho)}")
    
    # Create a safe density array for computation
    safe_rho = np.maximum(rho, min_rho)
    
    # Compute Alfvén speed
    inv_sqrt_rho = 1.0 / np.sqrt(safe_rho)
    
    # Set Alfvén speed to zero where density was non-positive
    mask = rho > 0
    vA_x = np.where(mask, B_x * inv_sqrt_rho, 0.0)
    vA_y = np.where(mask, B_y * inv_sqrt_rho, 0.0)
    vA_z = np.where(mask, B_z * inv_sqrt_rho, 0.0)
    
    return vA_x, vA_y, vA_z


def compute_z_plus_minus(
    v_x: np.ndarray,
    v_y: np.ndarray,
    v_z: np.ndarray,
    vA_x: np.ndarray,
    vA_y: np.ndarray,
    vA_z: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return Elsasser variables **z⁺ = v + v_A**, **z⁻ = v − v_A**.

    The Elsasser variables are fundamental to MHD turbulence analysis,
    representing wave-like fluctuations propagating along magnetic field lines.

    Parameters
    ----------
    v_x, v_y, v_z : np.ndarray
        Velocity components. Must be 2D arrays with identical shapes.
    vA_x, vA_y, vA_z : np.ndarray
        Alfvén velocity components (typically from compute_vA).
        Must have the same shape as velocity components.

    Returns
    -------
    tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
        Two triplets:
        - First: (z_plus_x, z_plus_y, z_plus_z) representing z⁺ = v + v_A
        - Second: (z_minus_x, z_minus_y, z_minus_z) representing z⁻ = v − v_A

    Raises
    ------
    TypeError
        If inputs are not numpy arrays.
    ValueError
        If arrays have inconsistent shapes or are not 2D.

    Notes
    -----
    The Elsasser variables z⁺ and z⁻ represent Alfvén waves propagating
    in opposite directions along the magnetic field. They are crucial for
    understanding energy transfer in MHD turbulence.

    Examples
    --------
    >>> v_x = np.ones((100, 100))
    >>> v_y = np.zeros((100, 100))
    >>> v_z = np.zeros((100, 100))
    >>> vA_x = np.full((100, 100), 0.5)
    >>> vA_y = np.zeros((100, 100))
    >>> vA_z = np.zeros((100, 100))
    >>> z_plus, z_minus = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
    >>> print(z_plus[0][0, 0])  # Should be 1.5
    >>> print(z_minus[0][0, 0])  # Should be 0.5
    """
    # Input validation
    arrays = {
        'v_x': v_x, 'v_y': v_y, 'v_z': v_z,
        'vA_x': vA_x, 'vA_y': vA_y, 'vA_z': vA_z
    }
    
    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got {arr.ndim}D")
    
    # Check shape consistency
    ref_shape = v_x.shape
    for name, arr in arrays.items():
        if arr.shape != ref_shape:
            raise ValueError(
                f"All arrays must have the same shape. {name} has shape {arr.shape}, "
                f"expected {ref_shape}"
            )
    
    # Check for NaN or infinite values
    for name, arr in arrays.items():
        if np.any(~np.isfinite(arr)):
            n_invalid = np.sum(~np.isfinite(arr))
            warnings.warn(
                f"{name} contains {n_invalid} non-finite values (NaN or Inf). "
                "Results may be unreliable.",
                RuntimeWarning
            )
            logger.warning(f"Non-finite values in {name}: NaN={np.sum(np.isnan(arr))}, Inf={np.sum(np.isinf(arr))}")
    
    # Compute Elsasser variables
    z_plus_x  = v_x + vA_x
    z_plus_y  = v_y + vA_y
    z_plus_z  = v_z + vA_z

    z_minus_x = v_x - vA_x
    z_minus_y = v_y - vA_y
    z_minus_z = v_z - vA_z

    return (z_plus_x, z_plus_y, z_plus_z), (z_minus_x, z_minus_y, z_minus_z) 