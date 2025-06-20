"""Histogram-building kernels for 2-D slice structure-function analysis.

This module is Numba-accelerated and therefore avoids importing anything
that would block compilation (e.g. matplotlib).  All heavy lifting of per-
displacement statistics happens here.
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


# -----------------------------------------------------------------------------
# Main kernel ------------------------------------------------------------------
# -----------------------------------------------------------------------------

@njit(cache=True)
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
    """Return 5-D histogram for a single displacement.

    Helper inline functions are used to minimise repeated code and keep the
    main body readable without sacrificing Numba performance.
    """

    # ------------------------------------------------------------------
    # Local helpers (eligible for inlining) -----------------------------
    # ------------------------------------------------------------------

    @njit(inline="always")
    def _diff(arr, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2):
        """Return δq for 2/3/5-point stencils (inline)."""
        if stencil_width == 2:
            return arr[jp, ip] - arr[j, i]
        elif stencil_width == 3:
            return arr[jp, ip] - 2.0 * arr[j, i] + arr[jm, im]
        else:  # 5-point
            return (
                -arr[jp2, ip2] + 16.0 * arr[jp, ip]
                - 30.0 * arr[j, i]
                + 16.0 * arr[jm, im] - arr[jm2, im2]
            )

    @njit(inline="always")
    def _mean_B(B_arr):
        """Local mean B component according to stencil width."""
        if stencil_width == 5:
            return (B_arr[0] + B_arr[1] + B_arr[2] + B_arr[3] + B_arr[4]) / 5.0
        elif stencil_width == 3:
            return (B_arr[0] + B_arr[1] + B_arr[2]) / 3.0
        else:
            return (B_arr[0] + B_arr[1]) / 2.0

    @njit(inline="always")
    def _perp(vec, B_unit):
        """Component of *vec* perpendicular to *B_unit*."""
        return vec - (vec @ B_unit) * B_unit

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

    # Allocate histograms -------------------------------------------------
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
        return hist_mag, hist_other  # displacement outside requested ℓ range

    # Random spatial samples (flat indexing)
    flat_indices = np.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M  # row (j)
    random_points_x = flat_indices % M   # col (i)

    for idx in range(N_random_subsamples):
        i = random_points_x[idx]
        j = random_points_y[idx]
        ip = (i + delta_i) % M
        jp = (j + delta_j) % N
        im = (i - delta_i) % M
        jm = (j - delta_j) % N

        # Secondary neighbours for 5-pt stencil -------------------------
        ip2 = (i + 2 * delta_i) % M
        jp2 = (j + 2 * delta_j) % N
        im2 = (i - 2 * delta_i) % M
        jm2 = (j - 2 * delta_j) % N

        # Vectorised diff calls -------------------------------------
        dvx = _diff(v_x, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dvy = _diff(v_y, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dvz = _diff(v_z, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)

        dBx = _diff(B_x, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dBy = _diff(B_y, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dBz = _diff(B_z, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)

        if stencil_width == 5:
            Bmx = _mean_B((B_x[jp2, ip2], B_x[jp, ip], B_x[j, i], B_x[jm, im], B_x[jm2, im2]))
            Bmy = _mean_B((B_y[jp2, ip2], B_y[jp, ip], B_y[j, i], B_y[jm, im], B_y[jm2, im2]))
            Bmz = _mean_B((B_z[jp2, ip2], B_z[jp, ip], B_z[j, i], B_z[jm, im], B_z[jm2, im2]))
        elif stencil_width == 3:
            Bmx = _mean_B((B_x[jp, ip], B_x[j, i], B_x[jm, im]))
            Bmy = _mean_B((B_y[jp, ip], B_y[j, i], B_y[jm, im]))
            Bmz = _mean_B((B_z[jp, ip], B_z[j, i], B_z[jm, im]))
        else:
            Bmx = _mean_B((B_x[jp, ip], B_x[j, i]))
            Bmy = _mean_B((B_y[jp, ip], B_y[j, i]))
            Bmz = _mean_B((B_z[jp, ip], B_z[j, i]))

        Bmean_mag = (Bmx * Bmx + Bmy * Bmy + Bmz * Bmz) ** 0.5
        if Bmean_mag == 0.0:
            continue

        cos_theta = abs(dx * Bmx + dy * Bmy + dz * Bmz) / (r * Bmean_mag)
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

        cos_phi = (
            (displacement_perp * dB_perp).sum()
            / (displacement_perp_mag * dB_perp_mag)
        )
        phi_val = np.arccos(cos_phi)

        vperp_cross_bperp = np.sqrt((np.cross(dv_perp, dB_perp) ** 2).sum())
        vperp_bperp = dv_perp_mag * dB_perp_mag

        dv = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
        dB = np.sqrt(dBx * dBx + dBy * dBy + dBz * dBz)

        drho = _diff(rho, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        drho = abs(drho)

        # Alfvén, Elsasser, vorticity, and current fluctuations ---------
        dvAx = _diff(vA_x, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dvAy = _diff(vA_y, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dvAz = _diff(vA_z, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)

        dzpx = _diff(zp_x, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dzpy = _diff(zp_y, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dzpz = _diff(zp_z, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)

        dzmx = _diff(zm_x, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dzmy = _diff(zm_y, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dzmz = _diff(zm_z, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)

        domegax = _diff(omega_x, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        domegay = _diff(omega_y, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        domegaz = _diff(omega_z, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)

        dJx = _diff(J_x, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dJy = _diff(J_y, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)
        dJz = _diff(J_z, jp, ip, j, i, jm, im, jp2, ip2, jm2, im2)

        dVA = np.sqrt(dvAx*dvAx + dvAy*dvAy + dvAz*dvAz)
        dZp = np.sqrt(dzpx*dzpx + dzpy*dzpy + dzpz*dzpz)
        dZm = np.sqrt(dzmx*dzmx + dzmy*dzmy + dzmz*dzmz)
        dOmega = np.sqrt(domegax*domegax + domegay*domegay + domegaz*domegaz)

        # Current fluctuation
        dJ = np.sqrt(dJx*dJx + dJy*dJy + dJz*dJz)

        # Bin indices ------------------------------------------------------
        theta_idx = find_bin_index_binary(theta_val, theta_bin_edges)
        phi_idx = find_bin_index_binary(phi_val, phi_bin_edges)

        v_idx = find_bin_index_binary(dv, sf_bin_edges)
        if v_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_V], ell_idx, theta_idx, phi_idx, v_idx] += 1

        b_idx = find_bin_index_binary(dB, sf_bin_edges)
        if b_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_B], ell_idx, theta_idx, phi_idx, b_idx] += 1

        x_idx = find_bin_index_binary(vperp_cross_bperp, product_bin_edges)
        if x_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Vperp_CROSS_Bperp], ell_idx, x_idx] += 1

        p_idx = find_bin_index_binary(vperp_bperp, product_bin_edges)
        if p_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Vperp_D_Bperp_MAG], ell_idx, p_idx] += 1

        rho_idx = find_bin_index_binary(drho, sf_bin_edges)
        if rho_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_RHO], ell_idx, theta_idx, phi_idx, rho_idx] += 1

        j_idx = find_bin_index_binary(dJ, sf_bin_edges)
        if j_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_J], ell_idx, theta_idx, phi_idx, j_idx] += 1

        # Bin Alfvén speed
        va_idx = find_bin_index_binary(dVA, sf_bin_edges)
        if va_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_VA], ell_idx, theta_idx, phi_idx, va_idx] += 1

        # Bin Elsasser magnitudes
        zp_idx = find_bin_index_binary(dZp, sf_bin_edges)
        if zp_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_ZPLUS], ell_idx, theta_idx, phi_idx, zp_idx] += 1

        zm_idx = find_bin_index_binary(dZm, sf_bin_edges)
        if zm_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_ZMINUS], ell_idx, theta_idx, phi_idx, zm_idx] += 1

        # Bin vorticity
        om_idx = find_bin_index_binary(dOmega, sf_bin_edges)
        if om_idx >= 0:
            hist_mag[MAG_IDX[Channel.D_OMEGA], ell_idx, theta_idx, phi_idx, om_idx] += 1

        # Compute perpendicular components relative to mean B ---------
        dVA_vec = np.array([dvAz, dvAy, dvAx])
        dOmega_vec = np.array([domegaz, domegay, domegax])
        dJ_vec = np.array([dJz, dJy, dJx])

        dVA_perp = dVA_vec - (dVA_vec @ np.array([Bmz, Bmy, Bmx])) / (Bmean_mag**2) * np.array([Bmz, Bmy, Bmx])
        dOmega_perp = dOmega_vec - (dOmega_vec @ np.array([Bmz, Bmy, Bmx])) / (Bmean_mag**2) * np.array([Bmz, Bmy, Bmx])
        dJ_perp = dJ_vec - (dJ_vec @ np.array([Bmz, Bmy, Bmx])) / (Bmean_mag**2) * np.array([Bmz, Bmy, Bmx])

        dVA_perp_mag = np.sqrt((dVA_perp**2).sum())
        dOmega_perp_mag = np.sqrt((dOmega_perp**2).sum())
        dJ_perp_mag = np.sqrt((dJ_perp**2).sum())

        cross_v_va = np.sqrt((np.cross(dv_perp, dVA_perp)**2).sum())
        cross_v_omega = np.sqrt((np.cross(dv_perp, dOmega_perp)**2).sum())
        cross_B_j = np.sqrt((np.cross(dB_perp, dJ_perp)**2).sum())

        c_idx = find_bin_index_binary(cross_v_va, product_bin_edges)
        if c_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Vperp_CROSS_VAperp], ell_idx, c_idx] += 1

        c_idx = find_bin_index_binary(cross_v_omega, product_bin_edges)
        if c_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Vperp_CROSS_Omegaperp], ell_idx, c_idx] += 1

        c_idx = find_bin_index_binary(cross_B_j, product_bin_edges)
        if c_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Bperp_CROSS_Jperp], ell_idx, c_idx] += 1

        # MAG (product magnitude) using perp magnitudes -----------------
        MAG_v_va = dv_perp_mag * dVA_perp_mag
        MAG_v_omega = dv_perp_mag * dOmega_perp_mag
        MAG_B_j = dB_perp_mag * dJ_perp_mag

        d_idx = find_bin_index_binary(MAG_v_va, product_bin_edges)
        if d_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Vperp_D_VAperp_MAG], ell_idx, d_idx] += 1

        d_idx = find_bin_index_binary(MAG_v_omega, product_bin_edges)
        if d_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Vperp_D_Omegaperp_MAG], ell_idx, d_idx] += 1

        d_idx = find_bin_index_binary(MAG_B_j, product_bin_edges)
        if d_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_Bperp_D_Jperp_MAG], ell_idx, d_idx] += 1

        # Full (non-perp) vectors --------------------------------------
        dv_vec = np.array([dvz, dvy, dvx])
        dB_vec = np.array([dBz, dBy, dBx])
        dVA_vec = np.array([dvAz, dvAy, dvAx])
        dOmega_vec = np.array([domegaz, domegay, domegax])
        dJ_vec_full = np.array([dJz, dJy, dJx])

        cross_v_B_full = np.sqrt((np.cross(dv_vec, dB_vec)**2).sum())
        cross_v_VA_full = np.sqrt((np.cross(dv_vec, dVA_vec)**2).sum())
        cross_v_Omega_full = np.sqrt((np.cross(dv_vec, dOmega_vec)**2).sum())
        cross_B_J_full = np.sqrt((np.cross(dB_vec, dJ_vec_full)**2).sum())

        f_idx = find_bin_index_binary(cross_v_B_full, product_bin_edges)
        if f_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_V_CROSS_B], ell_idx, f_idx] += 1

        f_idx = find_bin_index_binary(cross_v_VA_full, product_bin_edges)
        if f_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_V_CROSS_VA], ell_idx, f_idx] += 1

        f_idx = find_bin_index_binary(cross_v_Omega_full, product_bin_edges)
        if f_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_V_CROSS_OMEGA], ell_idx, f_idx] += 1

        f_idx = find_bin_index_binary(cross_B_J_full, product_bin_edges)
        if f_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_B_CROSS_J], ell_idx, f_idx] += 1

        # Full magnitudes ---------------------------------------------
        mag_v_B_full = dv * dB
        mag_v_VA_full = dv * dVA
        mag_v_Omega_full = dv * dOmega
        mag_B_J_full = dB * dJ

        g_idx = find_bin_index_binary(mag_v_B_full, product_bin_edges)
        if g_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_V_D_B_MAG], ell_idx, g_idx] += 1

        g_idx = find_bin_index_binary(mag_v_VA_full, product_bin_edges)
        if g_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_V_D_VA_MAG], ell_idx, g_idx] += 1

        g_idx = find_bin_index_binary(mag_v_Omega_full, product_bin_edges)
        if g_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_V_D_OMEGA_MAG], ell_idx, g_idx] += 1

        g_idx = find_bin_index_binary(mag_B_J_full, product_bin_edges)
        if g_idx >= 0:
            hist_other[OTHER_IDX[Channel.D_B_D_J_MAG], ell_idx, g_idx] += 1

    return hist_mag, hist_other 