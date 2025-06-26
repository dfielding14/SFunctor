"""Histogram-building kernels for 2-D slice structure-function analysis.

This module is Numba-accelerated and therefore avoids importing anything
that would block compilation (e.g. matplotlib).  All heavy lifting of per-
displacement statistics happens here.

This improved version fixes Numba compilation issues by creating separate
functions for each stencil width instead of runtime checks.
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
    "compute_histogram_for_disp_2D_stencil2",
    "compute_histogram_for_disp_2D_stencil3",
    "compute_histogram_for_disp_2D_stencil5",
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

    # --- MAG (product magnitudes) of full vectors ------------------------
    D_V_D_B_MAG = 20      # |δv||δB|
    D_V_D_VA_MAG = 21     # |δv||δv_A|
    D_V_D_OMEGA_MAG = 22  # |δv||δω|
    D_B_D_J_MAG = 23      # |δB||δj|


N_CHANNELS = len(Channel)

# Channels that go into the (ℓ, θ, φ, sf) histogram
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
N_MAG_CHANNELS = len(MAG_CHANNELS)

# Channels that go into the (ℓ, product) histogram
OTHER_CHANNELS = (
    Channel.D_Vperp_CROSS_Bperp,
    Channel.D_Vperp_CROSS_VAperp,
    Channel.D_Vperp_CROSS_Omegaperp,
    Channel.D_Bperp_CROSS_Jperp,
    Channel.D_Vperp_D_Bperp_MAG,
    Channel.D_Vperp_D_VAperp_MAG,
    Channel.D_Vperp_D_Omegaperp_MAG,
    Channel.D_Bperp_D_Jperp_MAG,
    Channel.D_V_CROSS_B,
    Channel.D_V_CROSS_VA,
    Channel.D_V_CROSS_OMEGA,
    Channel.D_B_CROSS_J,
    Channel.D_V_D_B_MAG,
    Channel.D_V_D_VA_MAG,
    Channel.D_V_D_OMEGA_MAG,
    Channel.D_B_D_J_MAG,
)
N_OTHER_CHANNELS = len(OTHER_CHANNELS)

# Maps from Channel enum to the index within each histogram type
MAG_IDX = {ch: i for i, ch in enumerate(MAG_CHANNELS)}
OTHER_IDX = {ch: i for i, ch in enumerate(OTHER_CHANNELS)}


# -----------------------------------------------------------------------------
# Binary search for histogram binning ----------------------------------------
# -----------------------------------------------------------------------------

@njit(cache=True)
def find_bin_index_binary(value: float, bin_edges: np.ndarray) -> int:
    """Binary search for the bin containing *value*.  Return -1 if not found."""
    left = 0
    right = len(bin_edges) - 2
    while left <= right:
        mid = (left + right) // 2
        if bin_edges[mid] <= value < bin_edges[mid + 1]:
            return mid
        elif value < bin_edges[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1  # value outside range


# -----------------------------------------------------------------------------
# Helper functions for different stencil computations ------------------------
# -----------------------------------------------------------------------------

@njit(inline="always")
def _diff_2pt(arr, jp, ip, j, i):
    """2-point stencil difference."""
    return arr[jp, ip] - arr[j, i]


@njit(inline="always")
def _diff_3pt(arr, jp, ip, j, i, jm, im):
    """3-point stencil difference."""
    return arr[jp, ip] - 2.0 * arr[j, i] + arr[jm, im]


@njit(inline="always")
def _diff_5pt(arr, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2):
    """5-point stencil difference."""
    return (-arr[jp2, ip2] + 16.0 * arr[jp, ip] - 30.0 * arr[j, i] + 
            16.0 * arr[jm, im] - arr[jm2, im2])


@njit(inline="always")
def _mean_B_2pt(B_plus, B_center):
    """Mean B for 2-point stencil."""
    return (B_plus + B_center) / 2.0


@njit(inline="always")
def _mean_B_3pt(B_plus, B_center, B_minus):
    """Mean B for 3-point stencil."""
    return (B_plus + B_center + B_minus) / 3.0


@njit(inline="always")
def _mean_B_5pt(B_plus2, B_plus, B_center, B_minus, B_minus2):
    """Mean B for 5-point stencil."""
    return (B_plus2 + B_plus + B_center + B_minus + B_minus2) / 5.0


@njit(inline="always")
def _perp(vec, B_unit):
    """Component of *vec* perpendicular to *B_unit*."""
    return vec - (vec @ B_unit) * B_unit


# -----------------------------------------------------------------------------
# Common histogram computation logic -----------------------------------------
# -----------------------------------------------------------------------------

@njit
def _compute_histogram_core(
    dvx, dvy, dvz, dBx, dBy, dBz, drho,
    dvAx, dvAy, dvAz, dzpx, dzpy, dzpz,
    dzmx, dzmy, dzmz, domegax, domegay, domegaz,
    dJx, dJy, dJz,
    Bmx, Bmy, Bmz,
    dx, dy, dz, r,
    ell_idx, theta_bin_edges, phi_bin_edges, 
    sf_bin_edges, product_bin_edges,
    hist_mag, hist_other
):
    """Common histogram computation logic shared by all stencil widths."""
    
    Bmean_mag = (Bmx * Bmx + Bmy * Bmy + Bmz * Bmz) ** 0.5
    if Bmean_mag < 1e-10:  # Use small epsilon instead of exact zero
        return
    
    cos_theta = abs(dx * Bmx + dy * Bmy + dz * Bmz) / (r * Bmean_mag)
    if cos_theta > 1.0:
        cos_theta = 1.0
    theta_val = np.arccos(cos_theta)
    
    # Pre-compute B_unit and reuse for all perpendicular projections
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
    
    if displacement_perp_mag > 0 and dB_perp_mag > 0:
        cos_phi = (
            (displacement_perp * dB_perp).sum()
            / (displacement_perp_mag * dB_perp_mag)
        )
        if cos_phi > 1.0:
            cos_phi = 1.0
        elif cos_phi < -1.0:
            cos_phi = -1.0
        phi_val = np.arccos(cos_phi)
    else:
        phi_val = 0.0
    
    vperp_cross_bperp = np.sqrt((np.cross(dv_perp, dB_perp) ** 2).sum())
    vperp_bperp = dv_perp_mag * dB_perp_mag
    
    dv = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
    dB = np.sqrt(dBx * dBx + dBy * dBy + dBz * dBz)
    
    dVA = np.sqrt(dvAx*dvAx + dvAy*dvAy + dvAz*dvAz)
    dZp = np.sqrt(dzpx*dzpx + dzpy*dzpy + dzpz*dzpz)
    dZm = np.sqrt(dzmx*dzmx + dzmy*dzmy + dzmz*dzmz)
    dOmega = np.sqrt(domegax*domegax + domegay*domegay + domegaz*domegaz)
    dJ = np.sqrt(dJx*dJx + dJy*dJy + dJz*dJz)
    
    # Bin indices
    theta_idx = find_bin_index_binary(theta_val, theta_bin_edges)
    phi_idx = find_bin_index_binary(phi_val, phi_bin_edges)
    
    # Check if angular indices are valid
    if theta_idx < 0 or phi_idx < 0:
        return
    
    # Update histograms for magnitude channels
    v_idx = find_bin_index_binary(dv, sf_bin_edges)
    if v_idx >= 0:
        hist_mag[0, ell_idx, theta_idx, phi_idx, v_idx] += 1  # Channel.D_V = 0
    
    b_idx = find_bin_index_binary(dB, sf_bin_edges)
    if b_idx >= 0:
        hist_mag[1, ell_idx, theta_idx, phi_idx, b_idx] += 1  # Channel.D_B = 1
    
    rho_idx = find_bin_index_binary(drho, sf_bin_edges)
    if rho_idx >= 0:
        hist_mag[2, ell_idx, theta_idx, phi_idx, rho_idx] += 1  # Channel.D_RHO = 2
    
    va_idx = find_bin_index_binary(dVA, sf_bin_edges)
    if va_idx >= 0:
        hist_mag[3, ell_idx, theta_idx, phi_idx, va_idx] += 1  # Channel.D_VA = 3
    
    zp_idx = find_bin_index_binary(dZp, sf_bin_edges)
    if zp_idx >= 0:
        hist_mag[4, ell_idx, theta_idx, phi_idx, zp_idx] += 1  # Channel.D_ZPLUS = 4
    
    zm_idx = find_bin_index_binary(dZm, sf_bin_edges)
    if zm_idx >= 0:
        hist_mag[5, ell_idx, theta_idx, phi_idx, zm_idx] += 1  # Channel.D_ZMINUS = 5
    
    om_idx = find_bin_index_binary(dOmega, sf_bin_edges)
    if om_idx >= 0:
        hist_mag[6, ell_idx, theta_idx, phi_idx, om_idx] += 1  # Channel.D_OMEGA = 6
    
    j_idx = find_bin_index_binary(dJ, sf_bin_edges)
    if j_idx >= 0:
        hist_mag[7, ell_idx, theta_idx, phi_idx, j_idx] += 1  # Channel.D_J = 7
    
    # Update histograms for cross product channels
    x_idx = find_bin_index_binary(vperp_cross_bperp, product_bin_edges)
    if x_idx >= 0:
        hist_other[0, ell_idx, x_idx] += 1  # Channel.D_Vperp_CROSS_Bperp = 8
    
    p_idx = find_bin_index_binary(vperp_bperp, product_bin_edges)
    if p_idx >= 0:
        hist_other[4, ell_idx, p_idx] += 1  # Channel.D_Vperp_D_Bperp_MAG = 12
    
    # Compute perpendicular components for other fields
    dVA_vec = np.array([dvAz, dvAy, dvAx])
    dOmega_vec = np.array([domegaz, domegay, domegax])
    dJ_vec = np.array([dJz, dJy, dJx])
    
    dVA_perp = _perp(dVA_vec, B_unit)
    dOmega_perp = _perp(dOmega_vec, B_unit)
    dJ_perp = _perp(dJ_vec, B_unit)
    
    dVA_perp_mag = np.sqrt((dVA_perp**2).sum())
    dOmega_perp_mag = np.sqrt((dOmega_perp**2).sum())
    dJ_perp_mag = np.sqrt((dJ_perp**2).sum())
    
    # Cross products and magnitudes
    cross_v_va = np.sqrt((np.cross(dv_perp, dVA_perp)**2).sum())
    cross_v_omega = np.sqrt((np.cross(dv_perp, dOmega_perp)**2).sum())
    cross_B_j = np.sqrt((np.cross(dB_perp, dJ_perp)**2).sum())
    
    c_idx = find_bin_index_binary(cross_v_va, product_bin_edges)
    if c_idx >= 0:
        hist_other[1, ell_idx, c_idx] += 1  # Channel.D_Vperp_CROSS_VAperp = 9
    
    c_idx = find_bin_index_binary(cross_v_omega, product_bin_edges)
    if c_idx >= 0:
        hist_other[2, ell_idx, c_idx] += 1  # Channel.D_Vperp_CROSS_Omegaperp = 10
    
    c_idx = find_bin_index_binary(cross_B_j, product_bin_edges)
    if c_idx >= 0:
        hist_other[3, ell_idx, c_idx] += 1  # Channel.D_Bperp_CROSS_Jperp = 11
    
    # Product magnitudes
    MAG_v_va = dv_perp_mag * dVA_perp_mag
    MAG_v_omega = dv_perp_mag * dOmega_perp_mag
    MAG_B_j = dB_perp_mag * dJ_perp_mag
    
    d_idx = find_bin_index_binary(MAG_v_va, product_bin_edges)
    if d_idx >= 0:
        hist_other[5, ell_idx, d_idx] += 1  # Channel.D_Vperp_D_VAperp_MAG = 13
    
    d_idx = find_bin_index_binary(MAG_v_omega, product_bin_edges)
    if d_idx >= 0:
        hist_other[6, ell_idx, d_idx] += 1  # Channel.D_Vperp_D_Omegaperp_MAG = 14
    
    d_idx = find_bin_index_binary(MAG_B_j, product_bin_edges)
    if d_idx >= 0:
        hist_other[7, ell_idx, d_idx] += 1  # Channel.D_Bperp_D_Jperp_MAG = 15
    
    # Full vector cross products
    cross_v_B_full = np.sqrt((np.cross(dv_vec, dB_vec)**2).sum())
    cross_v_VA_full = np.sqrt((np.cross(dv_vec, dVA_vec)**2).sum())
    cross_v_Omega_full = np.sqrt((np.cross(dv_vec, dOmega_vec)**2).sum())
    cross_B_J_full = np.sqrt((np.cross(dB_vec, dJ_vec)**2).sum())
    
    f_idx = find_bin_index_binary(cross_v_B_full, product_bin_edges)
    if f_idx >= 0:
        hist_other[8, ell_idx, f_idx] += 1  # Channel.D_V_CROSS_B = 16
    
    f_idx = find_bin_index_binary(cross_v_VA_full, product_bin_edges)
    if f_idx >= 0:
        hist_other[9, ell_idx, f_idx] += 1  # Channel.D_V_CROSS_VA = 17
    
    f_idx = find_bin_index_binary(cross_v_Omega_full, product_bin_edges)
    if f_idx >= 0:
        hist_other[10, ell_idx, f_idx] += 1  # Channel.D_V_CROSS_OMEGA = 18
    
    f_idx = find_bin_index_binary(cross_B_J_full, product_bin_edges)
    if f_idx >= 0:
        hist_other[11, ell_idx, f_idx] += 1  # Channel.D_B_CROSS_J = 19
    
    # Full vector product magnitudes
    MAG_v_B_full = dv * dB
    MAG_v_VA_full = dv * dVA
    MAG_v_Omega_full = dv * dOmega
    MAG_B_J_full = dB * dJ
    
    g_idx = find_bin_index_binary(MAG_v_B_full, product_bin_edges)
    if g_idx >= 0:
        hist_other[12, ell_idx, g_idx] += 1  # Channel.D_V_D_B_MAG = 20
    
    g_idx = find_bin_index_binary(MAG_v_VA_full, product_bin_edges)
    if g_idx >= 0:
        hist_other[13, ell_idx, g_idx] += 1  # Channel.D_V_D_VA_MAG = 21
    
    g_idx = find_bin_index_binary(MAG_v_Omega_full, product_bin_edges)
    if g_idx >= 0:
        hist_other[14, ell_idx, g_idx] += 1  # Channel.D_V_D_OMEGA_MAG = 22
    
    g_idx = find_bin_index_binary(MAG_B_J_full, product_bin_edges)
    if g_idx >= 0:
        hist_other[15, ell_idx, g_idx] += 1  # Channel.D_B_D_J_MAG = 23


# -----------------------------------------------------------------------------
# Specialized functions for each stencil width -------------------------------
# -----------------------------------------------------------------------------

@njit(cache=True)
def compute_histogram_for_disp_2D_stencil2(
    v_x: np.ndarray, v_y: np.ndarray, v_z: np.ndarray,
    B_x: np.ndarray, B_y: np.ndarray, B_z: np.ndarray,
    rho: np.ndarray,
    vA_x: np.ndarray, vA_y: np.ndarray, vA_z: np.ndarray,
    zp_x: np.ndarray, zp_y: np.ndarray, zp_z: np.ndarray,
    zm_x: np.ndarray, zm_y: np.ndarray, zm_z: np.ndarray,
    omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
    J_x: np.ndarray, J_y: np.ndarray, J_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """2-point stencil version of histogram computation."""
    
    N, M = v_x.shape
    n_ell_bins = ell_bin_edges.shape[0] - 1
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    n_sf_bins = sf_bin_edges.shape[0] - 1
    
    if slice_axis == 1:
        dx, dy, dz = 0, delta_i, delta_j
    elif slice_axis == 2:
        dx, dy, dz = delta_i, 0, delta_j
    else:
        dx, dy, dz = delta_i, delta_j, 0
    
    hist_mag = np.zeros((N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins), dtype=np.int64)
    n_product_bins = product_bin_edges.shape[0] - 1
    hist_other = np.zeros((N_OTHER_CHANNELS, n_ell_bins, n_product_bins), dtype=np.int64)
    
    r = (delta_i * delta_i + delta_j * delta_j) ** 0.5
    ell_idx = find_bin_index_binary(r, ell_bin_edges)
    if ell_idx == -1:
        return hist_mag, hist_other
    
    flat_indices = np.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M
    random_points_x = flat_indices % M
    
    for idx in range(N_random_subsamples):
        i = random_points_x[idx]
        j = random_points_y[idx]
        ip = (i + delta_i) % M
        jp = (j + delta_j) % N
        
        # Compute all differences
        dvx = _diff_2pt(v_x, jp, ip, j, i)
        dvy = _diff_2pt(v_y, jp, ip, j, i)
        dvz = _diff_2pt(v_z, jp, ip, j, i)
        
        dBx = _diff_2pt(B_x, jp, ip, j, i)
        dBy = _diff_2pt(B_y, jp, ip, j, i)
        dBz = _diff_2pt(B_z, jp, ip, j, i)
        
        drho = abs(_diff_2pt(rho, jp, ip, j, i))
        
        dvAx = _diff_2pt(vA_x, jp, ip, j, i)
        dvAy = _diff_2pt(vA_y, jp, ip, j, i)
        dvAz = _diff_2pt(vA_z, jp, ip, j, i)
        
        dzpx = _diff_2pt(zp_x, jp, ip, j, i)
        dzpy = _diff_2pt(zp_y, jp, ip, j, i)
        dzpz = _diff_2pt(zp_z, jp, ip, j, i)
        
        dzmx = _diff_2pt(zm_x, jp, ip, j, i)
        dzmy = _diff_2pt(zm_y, jp, ip, j, i)
        dzmz = _diff_2pt(zm_z, jp, ip, j, i)
        
        domegax = _diff_2pt(omega_x, jp, ip, j, i)
        domegay = _diff_2pt(omega_y, jp, ip, j, i)
        domegaz = _diff_2pt(omega_z, jp, ip, j, i)
        
        dJx = _diff_2pt(J_x, jp, ip, j, i)
        dJy = _diff_2pt(J_y, jp, ip, j, i)
        dJz = _diff_2pt(J_z, jp, ip, j, i)
        
        # Mean B field
        Bmx = _mean_B_2pt(B_x[jp, ip], B_x[j, i])
        Bmy = _mean_B_2pt(B_y[jp, ip], B_y[j, i])
        Bmz = _mean_B_2pt(B_z[jp, ip], B_z[j, i])
        
        # Call common histogram computation
        _compute_histogram_core(
            dvx, dvy, dvz, dBx, dBy, dBz, drho,
            dvAx, dvAy, dvAz, dzpx, dzpy, dzpz,
            dzmx, dzmy, dzmz, domegax, domegay, domegaz,
            dJx, dJy, dJz,
            Bmx, Bmy, Bmz,
            dx, dy, dz, r,
            ell_idx, theta_bin_edges, phi_bin_edges,
            sf_bin_edges, product_bin_edges,
            hist_mag, hist_other
        )
    
    return hist_mag, hist_other


@njit(cache=True)
def compute_histogram_for_disp_2D_stencil3(
    v_x: np.ndarray, v_y: np.ndarray, v_z: np.ndarray,
    B_x: np.ndarray, B_y: np.ndarray, B_z: np.ndarray,
    rho: np.ndarray,
    vA_x: np.ndarray, vA_y: np.ndarray, vA_z: np.ndarray,
    zp_x: np.ndarray, zp_y: np.ndarray, zp_z: np.ndarray,
    zm_x: np.ndarray, zm_y: np.ndarray, zm_z: np.ndarray,
    omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
    J_x: np.ndarray, J_y: np.ndarray, J_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """3-point stencil version of histogram computation."""
    
    N, M = v_x.shape
    n_ell_bins = ell_bin_edges.shape[0] - 1
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    n_sf_bins = sf_bin_edges.shape[0] - 1
    
    if slice_axis == 1:
        dx, dy, dz = 0, delta_i, delta_j
    elif slice_axis == 2:
        dx, dy, dz = delta_i, 0, delta_j
    else:
        dx, dy, dz = delta_i, delta_j, 0
    
    hist_mag = np.zeros((N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins), dtype=np.int64)
    n_product_bins = product_bin_edges.shape[0] - 1
    hist_other = np.zeros((N_OTHER_CHANNELS, n_ell_bins, n_product_bins), dtype=np.int64)
    
    r = (delta_i * delta_i + delta_j * delta_j) ** 0.5
    ell_idx = find_bin_index_binary(r, ell_bin_edges)
    if ell_idx == -1:
        return hist_mag, hist_other
    
    flat_indices = np.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M
    random_points_x = flat_indices % M
    
    for idx in range(N_random_subsamples):
        i = random_points_x[idx]
        j = random_points_y[idx]
        ip = (i + delta_i) % M
        jp = (j + delta_j) % N
        im = (i - delta_i) % M
        jm = (j - delta_j) % N
        
        # Compute all differences
        dvx = _diff_3pt(v_x, jp, ip, j, i, jm, im)
        dvy = _diff_3pt(v_y, jp, ip, j, i, jm, im)
        dvz = _diff_3pt(v_z, jp, ip, j, i, jm, im)
        
        dBx = _diff_3pt(B_x, jp, ip, j, i, jm, im)
        dBy = _diff_3pt(B_y, jp, ip, j, i, jm, im)
        dBz = _diff_3pt(B_z, jp, ip, j, i, jm, im)
        
        drho = abs(_diff_3pt(rho, jp, ip, j, i, jm, im))
        
        dvAx = _diff_3pt(vA_x, jp, ip, j, i, jm, im)
        dvAy = _diff_3pt(vA_y, jp, ip, j, i, jm, im)
        dvAz = _diff_3pt(vA_z, jp, ip, j, i, jm, im)
        
        dzpx = _diff_3pt(zp_x, jp, ip, j, i, jm, im)
        dzpy = _diff_3pt(zp_y, jp, ip, j, i, jm, im)
        dzpz = _diff_3pt(zp_z, jp, ip, j, i, jm, im)
        
        dzmx = _diff_3pt(zm_x, jp, ip, j, i, jm, im)
        dzmy = _diff_3pt(zm_y, jp, ip, j, i, jm, im)
        dzmz = _diff_3pt(zm_z, jp, ip, j, i, jm, im)
        
        domegax = _diff_3pt(omega_x, jp, ip, j, i, jm, im)
        domegay = _diff_3pt(omega_y, jp, ip, j, i, jm, im)
        domegaz = _diff_3pt(omega_z, jp, ip, j, i, jm, im)
        
        dJx = _diff_3pt(J_x, jp, ip, j, i, jm, im)
        dJy = _diff_3pt(J_y, jp, ip, j, i, jm, im)
        dJz = _diff_3pt(J_z, jp, ip, j, i, jm, im)
        
        # Mean B field
        Bmx = _mean_B_3pt(B_x[jp, ip], B_x[j, i], B_x[jm, im])
        Bmy = _mean_B_3pt(B_y[jp, ip], B_y[j, i], B_y[jm, im])
        Bmz = _mean_B_3pt(B_z[jp, ip], B_z[j, i], B_z[jm, im])
        
        
        # Call common histogram computation
        _compute_histogram_core(
            dvx, dvy, dvz, dBx, dBy, dBz, drho,
            dvAx, dvAy, dvAz, dzpx, dzpy, dzpz,
            dzmx, dzmy, dzmz, domegax, domegay, domegaz,
            dJx, dJy, dJz,
            Bmx, Bmy, Bmz,
            dx, dy, dz, r,
            ell_idx, theta_bin_edges, phi_bin_edges,
            sf_bin_edges, product_bin_edges,
            hist_mag, hist_other
        )
    
    return hist_mag, hist_other


@njit(cache=True)
def compute_histogram_for_disp_2D_stencil5(
    v_x: np.ndarray, v_y: np.ndarray, v_z: np.ndarray,
    B_x: np.ndarray, B_y: np.ndarray, B_z: np.ndarray,
    rho: np.ndarray,
    vA_x: np.ndarray, vA_y: np.ndarray, vA_z: np.ndarray,
    zp_x: np.ndarray, zp_y: np.ndarray, zp_z: np.ndarray,
    zm_x: np.ndarray, zm_y: np.ndarray, zm_z: np.ndarray,
    omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
    J_x: np.ndarray, J_y: np.ndarray, J_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """5-point stencil version of histogram computation."""
    
    N, M = v_x.shape
    n_ell_bins = ell_bin_edges.shape[0] - 1
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    n_sf_bins = sf_bin_edges.shape[0] - 1
    
    if slice_axis == 1:
        dx, dy, dz = 0, delta_i, delta_j
    elif slice_axis == 2:
        dx, dy, dz = delta_i, 0, delta_j
    else:
        dx, dy, dz = delta_i, delta_j, 0
    
    hist_mag = np.zeros((N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins), dtype=np.int64)
    n_product_bins = product_bin_edges.shape[0] - 1
    hist_other = np.zeros((N_OTHER_CHANNELS, n_ell_bins, n_product_bins), dtype=np.int64)
    
    r = (delta_i * delta_i + delta_j * delta_j) ** 0.5
    ell_idx = find_bin_index_binary(r, ell_bin_edges)
    if ell_idx == -1:
        return hist_mag, hist_other
    
    flat_indices = np.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M
    random_points_x = flat_indices % M
    
    for idx in range(N_random_subsamples):
        i = random_points_x[idx]
        j = random_points_y[idx]
        ip = (i + delta_i) % M
        jp = (j + delta_j) % N
        im = (i - delta_i) % M
        jm = (j - delta_j) % N
        ip2 = (i + 2 * delta_i) % M
        jp2 = (j + 2 * delta_j) % N
        im2 = (i - 2 * delta_i) % M
        jm2 = (j - 2 * delta_j) % N
        
        # Compute all differences
        dvx = _diff_5pt(v_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dvy = _diff_5pt(v_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dvz = _diff_5pt(v_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        dBx = _diff_5pt(B_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dBy = _diff_5pt(B_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dBz = _diff_5pt(B_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        drho = abs(_diff_5pt(rho, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2))
        
        dvAx = _diff_5pt(vA_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dvAy = _diff_5pt(vA_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dvAz = _diff_5pt(vA_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        dzpx = _diff_5pt(zp_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dzpy = _diff_5pt(zp_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dzpz = _diff_5pt(zp_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        dzmx = _diff_5pt(zm_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dzmy = _diff_5pt(zm_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dzmz = _diff_5pt(zm_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        domegax = _diff_5pt(omega_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        domegay = _diff_5pt(omega_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        domegaz = _diff_5pt(omega_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        dJx = _diff_5pt(J_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dJy = _diff_5pt(J_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dJz = _diff_5pt(J_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        # Mean B field
        Bmx = _mean_B_5pt(B_x[jp2, ip2], B_x[jp, ip], B_x[j, i], B_x[jm, im], B_x[jm2, im2])
        Bmy = _mean_B_5pt(B_y[jp2, ip2], B_y[jp, ip], B_y[j, i], B_y[jm, im], B_y[jm2, im2])
        Bmz = _mean_B_5pt(B_z[jp2, ip2], B_z[jp, ip], B_z[j, i], B_z[jm, im], B_z[jm2, im2])
        
        
        # Call common histogram computation
        _compute_histogram_core(
            dvx, dvy, dvz, dBx, dBy, dBz, drho,
            dvAx, dvAy, dvAz, dzpx, dzpy, dzpz,
            dzmx, dzmy, dzmz, domegax, domegay, domegaz,
            dJx, dJy, dJz,
            Bmx, Bmy, Bmz,
            dx, dy, dz, r,
            ell_idx, theta_bin_edges, phi_bin_edges,
            sf_bin_edges, product_bin_edges,
            hist_mag, hist_other
        )
    
    return hist_mag, hist_other


def compute_histogram_for_disp_2D(
    v_x: np.ndarray, v_y: np.ndarray, v_z: np.ndarray,
    B_x: np.ndarray, B_y: np.ndarray, B_z: np.ndarray,
    rho: np.ndarray,
    vA_x: np.ndarray, vA_y: np.ndarray, vA_z: np.ndarray,
    zp_x: np.ndarray, zp_y: np.ndarray, zp_z: np.ndarray,
    zm_x: np.ndarray, zm_y: np.ndarray, zm_z: np.ndarray,
    omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
    J_x: np.ndarray, J_y: np.ndarray, J_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
    stencil_width: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to appropriate stencil-specific function."""
    
    if stencil_width == 2:
        return compute_histogram_for_disp_2D_stencil2(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, delta_i, delta_j, slice_axis,
            N_random_subsamples, ell_bin_edges, theta_bin_edges,
            phi_bin_edges, sf_bin_edges, product_bin_edges
        )
    elif stencil_width == 3:
        return compute_histogram_for_disp_2D_stencil3(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, delta_i, delta_j, slice_axis,
            N_random_subsamples, ell_bin_edges, theta_bin_edges,
            phi_bin_edges, sf_bin_edges, product_bin_edges
        )
    elif stencil_width == 5:
        return compute_histogram_for_disp_2D_stencil5(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, delta_i, delta_j, slice_axis,
            N_random_subsamples, ell_bin_edges, theta_bin_edges,
            phi_bin_edges, sf_bin_edges, product_bin_edges
        )
    else:
        raise ValueError(f"Unsupported stencil_width: {stencil_width}")