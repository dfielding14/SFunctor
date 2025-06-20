from __future__ import annotations

import numpy as np
from typing import Tuple

__all__ = [
    "compute_vA",
    "compute_z_plus_minus",
]


def compute_vA(B_x: np.ndarray, B_y: np.ndarray, B_z: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vector Alfvén speed **v_A = B / sqrt(rho)** for a 2-D slice.

    Parameters
    ----------
    B_x, B_y, B_z
        Magnetic-field components on the slice.
    rho
        Mass density on the slice.

    Notes
    -----
    A small floor of ``rho = 1e-30`` is applied to avoid division by zero.
    """
    rho_safe = np.maximum(rho, 1e-30)
    inv_sqrt_rho = 1.0 / np.sqrt(rho_safe)
    vA_x = B_x * inv_sqrt_rho
    vA_y = B_y * inv_sqrt_rho
    vA_z = B_z * inv_sqrt_rho
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

    All inputs must have identical shapes.  The function returns two triplets
    ``(z_plus_x, z_plus_y, z_plus_z)`` and ``(z_minus_x, z_minus_y, z_minus_z)``.
    """
    z_plus_x  = v_x + vA_x
    z_plus_y  = v_y + vA_y
    z_plus_z  = v_z + vA_z

    z_minus_x = v_x - vA_x
    z_minus_y = v_y - vA_y
    z_minus_z = v_z - vA_z

    return (z_plus_x, z_plus_y, z_plus_z), (z_minus_x, z_minus_y, z_minus_z) 