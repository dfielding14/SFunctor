"""Histogram-building kernels for 2-D slice structure-function analysis.

This module is Numba-accelerated and therefore avoids importing anything
that would block compilation (e.g. matplotlib).  All heavy lifting of per-
displacement statistics happens here.

This is a fixed version that handles the stencil_width Numba compilation issue.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from enum import IntEnum
from typing import Tuple

__all__ = [
    "Channel",
    "N_CHANNELS",
    "find_bin_index_binary",
    "compute_histogram_for_disp_2D",
]


class Channel(IntEnum):
    """Histogram channels – all names start with D_ for consistency."""

    # --- Structure-function magnitudes -------------------------------------
    D_V = 0        # |δv|
    D_B = 1        # |δB|
    D_RHO = 2      # |δρ|
    D_VA = 3       # |δv_A|
    D_ZPLUS = 4    # |δz⁺|
    D_ZMINUS = 5   # |δz⁻|
    D_OMEGA = 6    # |δω|
    D_J = 7        # |δj|

    # --- Angle numerators: perpendicular components -----------------------
    D_Vperp_CROSS_Bperp = 8      # |δv_⊥ × δB_⊥|
    D_Vperp_CROSS_VAperp = 9     # |δv_⊥ × δv_A⊥|
    D_Vperp_CROSS_Omegaperp = 10 # |δv_⊥ × δω_⊥|
    D_Bperp_CROSS_Jperp = 11     # |δB_⊥ × δj_⊥|

    # --- MAG (product magnitudes) of perpendicular components -------------
    D_Vperp_D_Bperp_MAG = 12       # |δv_⊥||δB_⊥|
    D_Vperp_D_VAperp_MAG = 13      # |δv_⊥||δv_A⊥|
    D_Vperp_D_Omegaperp_MAG = 14   # |δv_⊥||δω_⊥|
    D_Bperp_D_Jperp_MAG = 15       # |δB_⊥||δj_⊥|

    # --- Non-perpendicular (full-vector) numerators -----------------------
    D_V_CROSS_B = 16
    D_V_CROSS_VA = 17
    D_V_CROSS_OMEGA = 18
    D_B_CROSS_J = 19

    # --- Non-perpendicular magnitudes ------------------------------------
    D_V_D_B_MAG = 20
    D_V_D_VA_MAG = 21
    D_V_D_OMEGA_MAG = 22
    D_B_D_J_MAG = 23


N_CHANNELS = len(Channel)

# Channel groupings -----------------------------------------------------------
# First 8 channels correspond to structure-function magnitudes which require
# θ and φ binning.  The remaining channels are angle numerators/denominators
# that only need ℓ and SF magnitude bins.

MAG_CHANNELS = (
    Channel.D_V,
    Channel.D_B,
    Channel.D_RHO,
    Channel.D_VA,
    Channel.D_ZPLUS,
    Channel.D_ZMINUS,
    Channel.D_OMEGA,
    Channel.D_J,
)

OTHER_CHANNELS = tuple(ch for ch in Channel if ch not in MAG_CHANNELS)

N_MAG_CHANNELS = len(MAG_CHANNELS)
N_OTHER_CHANNELS = len(OTHER_CHANNELS)

# Offset of first OTHER channel so we can map global Channel.value → local
# index inside ``hist_other`` arrays (θ/φ collapsed).
OFFSET_OTHER = OTHER_CHANNELS[0].value  # 8

# Mapping from global Channel → local index inside the split histograms ------
MAG_IDX = {ch: i for i, ch in enumerate(MAG_CHANNELS)}
OTHER_IDX = {ch: i for i, ch in enumerate(OTHER_CHANNELS)}


@njit(cache=True)
def find_bin_index_binary(value: float, bin_edges: np.ndarray) -> int:
    """Binary search helper for non-uniform *bin_edges* (monotonic ascending)."""
    left, right = 0, len(bin_edges) - 2
    while left <= right:
        mid = (left + right) // 2
        if bin_edges[mid] <= value < bin_edges[mid + 1]:
            return mid
        elif value < bin_edges[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1  # value outside range


# Helper functions for 2-point stencil
@njit(inline="always")
def _diff_2pt(arr, jp, ip, j, i):
    return arr[jp, ip] - arr[j, i]


@njit(inline="always")
def _mean_B_2pt(B1, B2):
    return (B1 + B2) / 2.0


@njit(inline="always")
def _perp(vec, B_unit):
    return vec - (vec @ B_unit) * B_unit


@njit(cache=True)
def compute_histogram_for_disp_2D_stencil_2(
    v_x: np.ndarray,
    v_y: np.ndarray,
    v_z: np.ndarray,
    B_x: np.ndarray,
    B_y: np.ndarray,
    B_z: np.ndarray,
    rho: np.ndarray,
    vA_x: np.ndarray,
    vA_y: np.ndarray,
    vA_z: np.ndarray,
    zp_x: np.ndarray,
    zp_y: np.ndarray,
    zp_z: np.ndarray,
    zm_x: np.ndarray,
    zm_y: np.ndarray,
    zm_z: np.ndarray,
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    omega_z: np.ndarray,
    J_x: np.ndarray,
    J_y: np.ndarray,
    J_z: np.ndarray,
    delta_i: int,
    delta_j: int,
    slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """2-point stencil version."""
    
    N, M = v_x.shape
    n_ell_bins = ell_bin_edges.shape[0] - 1
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    n_sf_bins = sf_bin_edges.shape[0] - 1

    if slice_axis == 1:
        dx = 0
        dy = delta_i
        dz = delta_j
    elif slice_axis == 2:
        dx = delta_i
        dy = 0
        dz = delta_j
    else:  # slice_axis == 3
        dx = delta_i
        dy = delta_j
        dz = 0

    # Allocate histograms
    hist_mag = np.zeros((N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins), dtype=np.int64)
    hist_other = np.zeros((N_OTHER_CHANNELS, n_ell_bins, n_sf_bins), dtype=np.int64)

    # Magnitude of displacement r and ℓ-bin index
    r = (delta_i * delta_i + delta_j * delta_j) ** 0.5
    ell_idx = -1
    for i in range(n_ell_bins):
        if ell_bin_edges[i] <= r < ell_bin_edges[i + 1]:
            ell_idx = i
            break
    if ell_idx == -1:
        return hist_mag, hist_other

    # Random spatial samples
    flat_indices = np.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M
    random_points_x = flat_indices % M

    for idx in range(N_random_subsamples):
        i = random_points_x[idx]
        j = random_points_y[idx]
        ip = (i + delta_i) % M
        jp = (j + delta_j) % N

        # Compute differences using 2-point stencil
        dvx = _diff_2pt(v_x, jp, ip, j, i)
        dvy = _diff_2pt(v_y, jp, ip, j, i)
        dvz = _diff_2pt(v_z, jp, ip, j, i)

        dBx = _diff_2pt(B_x, jp, ip, j, i)
        dBy = _diff_2pt(B_y, jp, ip, j, i)
        dBz = _diff_2pt(B_z, jp, ip, j, i)

        # Mean B field components
        Bmx = _mean_B_2pt(B_x[jp, ip], B_x[j, i])
        Bmy = _mean_B_2pt(B_y[jp, ip], B_y[j, i])
        Bmz = _mean_B_2pt(B_z[jp, ip], B_z[j, i])

        Bmean_mag = (Bmx * Bmx + Bmy * Bmy + Bmz * Bmz) ** 0.5
        if Bmean_mag == 0.0:
            continue

        cos_theta = abs(dx * Bmx + dy * Bmy + dz * Bmz) / (r * Bmean_mag)
        if cos_theta > 1.0:
            cos_theta = 1.0
        theta_val = np.arccos(cos_theta)

        # Compute perpendicular components
        B_unit = np.array([Bmz, Bmy, Bmx]) / Bmean_mag
        dB_vec = np.array([dBz, dBy, dBx])
        dv_vec = np.array([dvz, dvy, dvx])

        dB_perp = _perp(dB_vec, B_unit)
        dB_perp_mag = np.sqrt((dB_perp ** 2).sum())

        dv_perp = _perp(dv_vec, B_unit)
        dv_perp_mag = np.sqrt((dv_perp ** 2).sum())

        displacement_perp = np.array([dz, dy, dx]) - (
            (dz * Bmz + dy * Bmy + dx * Bmx) / (Bmean_mag ** 2)
        ) * np.array([Bmz, Bmy, Bmx])
        displacement_perp_mag = np.sqrt((displacement_perp ** 2).sum())

        if displacement_perp_mag == 0.0 or dB_perp_mag == 0.0:
            phi_val = 0.0
        else:
            cos_phi = (
                (displacement_perp * dB_perp).sum()
                / (displacement_perp_mag * dB_perp_mag)
            )
            if cos_phi > 1.0:
                cos_phi = 1.0
            elif cos_phi < -1.0:
                cos_phi = -1.0
            phi_val = np.arccos(abs(cos_phi))

        # Structure function magnitudes
        dv = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
        dB = np.sqrt(dBx * dBx + dBy * dBy + dBz * dBz)

        drho = _diff_2pt(rho, jp, ip, j, i)
        drho = abs(drho)

        # Additional quantities (simplified for demonstration)
        dvAx = _diff_2pt(vA_x, jp, ip, j, i)
        dvAy = _diff_2pt(vA_y, jp, ip, j, i)
        dvAz = _diff_2pt(vA_z, jp, ip, j, i)
        dVA = np.sqrt(dvAx*dvAx + dvAy*dvAy + dvAz*dvAz)

        # Bin the results
        theta_idx = find_bin_index_binary(theta_val, theta_bin_edges)
        phi_idx = find_bin_index_binary(phi_val, phi_bin_edges)

        v_idx = find_bin_index_binary(dv, sf_bin_edges)
        if v_idx >= 0 and theta_idx >= 0 and phi_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_V], ell_idx, theta_idx, phi_idx, v_idx] += 1

        b_idx = find_bin_index_binary(dB, sf_bin_edges)
        if b_idx >= 0 and theta_idx >= 0 and phi_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_B], ell_idx, theta_idx, phi_idx, b_idx] += 1

        rho_idx = find_bin_index_binary(drho, sf_bin_edges)
        if rho_idx >= 0 and theta_idx >= 0 and phi_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_RHO], ell_idx, theta_idx, phi_idx, rho_idx] += 1

        va_idx = find_bin_index_binary(dVA, sf_bin_edges)
        if va_idx >= 0 and theta_idx >= 0 and phi_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_VA], ell_idx, theta_idx, phi_idx, va_idx] += 1

        # Cross products and magnitudes for other channels
        if dv_perp_mag > 0.0 and dB_perp_mag > 0.0:
            vperp_cross_bperp = np.sqrt((np.cross(dv_perp, dB_perp) ** 2).sum())
            vperp_bperp = dv_perp_mag * dB_perp_mag

            x_idx = find_bin_index_binary(vperp_cross_bperp, product_bin_edges)
            if x_idx >= 0:
                hist_other[OTHER_IDX[Channel.D_Vperp_CROSS_Bperp], ell_idx, x_idx] += 1

            p_idx = find_bin_index_binary(vperp_bperp, product_bin_edges)
            if p_idx >= 0:
                hist_other[OTHER_IDX[Channel.D_Vperp_D_Bperp_MAG], ell_idx, p_idx] += 1

    return hist_mag, hist_other


def compute_histogram_for_disp_2D(
    v_x: np.ndarray,
    v_y: np.ndarray,
    v_z: np.ndarray,
    B_x: np.ndarray,
    B_y: np.ndarray,
    B_z: np.ndarray,
    rho: np.ndarray,
    vA_x: np.ndarray,
    vA_y: np.ndarray,
    vA_z: np.ndarray,
    zp_x: np.ndarray,
    zp_y: np.ndarray,
    zp_z: np.ndarray,
    zm_x: np.ndarray,
    zm_y: np.ndarray,
    zm_z: np.ndarray,
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    omega_z: np.ndarray,
    J_x: np.ndarray,
    J_y: np.ndarray,
    J_z: np.ndarray,
    delta_i: int,
    delta_j: int,
    slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
    stencil_width: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to appropriate stencil-width specific function."""
    
    if stencil_width == 2:
        return compute_histogram_for_disp_2D_stencil_2(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, delta_i, delta_j, slice_axis,
            N_random_subsamples, ell_bin_edges, theta_bin_edges,
            phi_bin_edges, sf_bin_edges, product_bin_edges
        )
    else:
        raise NotImplementedError(f"Stencil width {stencil_width} not implemented in fixed version") 