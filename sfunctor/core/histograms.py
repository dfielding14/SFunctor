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
    "CENSOR_NAMES",
    "N_CENSOR_KINDS",
    "Channel",
    "N_CHANNELS",
    "find_bin_index_binary",
    "compute_histogram_for_disp_2D_stencil2",
    "compute_histogram_for_disp_2D_stencil3",
    "compute_histogram_for_disp_2D_stencil5",
    "compute_histogram_for_disp_2D",
]

# Normalisation factors so that ⟨|Δu|²⟩ → 2σ² for uncorrelated samples
NORM_2PT = 1.0
NORM_3PT = 1.0 / np.sqrt(3.0)
NORM_5PT = 1.0 / np.sqrt(35.0)


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
    D_CURV = 8     # |δ(b·∇b)|
    D_GRAD_RHO = 9 # |δ(∇ρ)|
    D_B_over_Bmean_loc = 10  # |δB|/B_mean_local

    # --- Angle numerators: perpendicular components -----------------------
    D_Vperp_CROSS_D_Bperp = 11      # |δv_⊥ × δB_⊥|
    D_Vperp_CROSS_D_Omegaperp = 12  # |δv_⊥ × δω_⊥|
    D_Bperp_CROSS_D_Jperp = 13      # |δB_⊥ × δj_⊥|
    D_Omegaperp_CROSS_D_Jperp = 14  # |δω_⊥ × δj_⊥|

    # --- MAG (product magnitudes) of perpendicular components -------------
    D_Vperp_D_Bperp_MAG = 15        # |δv_⊥||δB_⊥|
    D_Vperp_D_Omegaperp_MAG = 16    # |δv_⊥||δω_⊥|
    D_Bperp_D_Jperp_MAG = 17        # |δB_⊥||δj_⊥|
    D_Omegaperp_D_Jperp_MAG = 18    # |δω_⊥||δj_⊥|

    # --- Cross product to magnitude ratios ---------------------------------
    D_Vperp_D_Bperp_CROSS_MAG_RATIO = 19      # |δv_⊥ × δB_⊥| / |δv_⊥||δB_⊥|
    D_Vperp_D_Omegaperp_CROSS_MAG_RATIO = 20  # |δv_⊥ × δω_⊥| / |δv_⊥||δω_⊥|
    D_Bperp_D_Jperp_CROSS_MAG_RATIO = 21      # |δB_⊥ × δj_⊥| / |δB_⊥||δj_⊥|
    D_Omegaperp_D_Jperp_CROSS_MAG_RATIO = 22  # |δω_⊥ × δj_⊥| / |δω_⊥||δj_⊥|

    # --- Elsasser alignment angle: θ^(z+,z-) = ⟨δz⁺_⊥ × δz⁻_⊥⟩ / ⟨|δz⁺_⊥||δz⁻_⊥|⟩
    D_Zplusperp_CROSS_D_Zminusperp = 23       # |δz⁺_⊥ × δz⁻_⊥|
    D_Zplusperp_D_Zminusperp_MAG = 24         # |δz⁺_⊥||δz⁻_⊥|
    D_Zplusperp_D_Zminusperp_CROSS_MAG_RATIO = 25  # |δz⁺_⊥ × δz⁻_⊥| / |δz⁺_⊥||δz⁻_⊥|


N_CHANNELS = len(Channel)
CENSOR_NAMES = ("accepted", "underflow", "overflow", "invalid")
N_CENSOR_KINDS = len(CENSOR_NAMES)
_CENSOR_ACCEPTED = 0
_CENSOR_UNDERFLOW = 1
_CENSOR_OVERFLOW = 2
_CENSOR_INVALID = 3


# -----------------------------------------------------------------------------
# Binary search for histogram binning ----------------------------------------
# -----------------------------------------------------------------------------

@njit(cache=True)
def find_bin_index_binary(value: float, bin_edges: np.ndarray) -> int:
    """Binary search for the bin containing *value*. Return -1 if not found.

    Histogram bins are left-inclusive and right-exclusive except for the final
    bin, which includes its right edge.
    """
    if value == bin_edges[-1]:
        return len(bin_edges) - 2
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
    return NORM_2PT * (arr[jp, ip] - arr[j, i])


@njit(inline="always")
def _diff_3pt(arr, jp, ip, j, i, jm, im):
    """3-point stencil difference."""
    return NORM_3PT * (arr[jp, ip] - 2.0 * arr[j, i] + arr[jm, im])


@njit(inline="always")
def _diff_5pt(arr, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2):
    """5-point stencil difference."""
    return NORM_5PT * (
        arr[jm2, im2]
        - 4.0 * arr[jm, im]
        + 6.0 * arr[j, i]
        - 4.0 * arr[jp, ip]
        + arr[jp2, ip2]
    )


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
    return (
        B_plus2
        + 4.0 * B_plus
        + 6.0 * B_center
        + 4.0 * B_minus
        + B_minus2
    ) / 16.0


@njit(inline="always")
def _perp(vec, B_unit):
    """Component of *vec* perpendicular to *B_unit*."""
    return vec - (vec @ B_unit) * B_unit


@njit(inline="always")
def _accumulate_bin(value, edges, hist, censoring, channel_idx, ell_idx, theta_idx, phi_idx):
    """Accumulate one sample and record whether its delta bin censored it."""
    if not np.isfinite(value):
        censoring[channel_idx, ell_idx, theta_idx, phi_idx, _CENSOR_INVALID] += 1
        return
    if value < edges[0]:
        censoring[channel_idx, ell_idx, theta_idx, phi_idx, _CENSOR_UNDERFLOW] += 1
        return
    if value > edges[-1]:
        censoring[channel_idx, ell_idx, theta_idx, phi_idx, _CENSOR_OVERFLOW] += 1
        return
    bin_idx = find_bin_index_binary(value, edges)
    if bin_idx >= 0:
        hist[channel_idx, ell_idx, theta_idx, phi_idx, bin_idx] += 1
        censoring[channel_idx, ell_idx, theta_idx, phi_idx, _CENSOR_ACCEPTED] += 1
    else:
        censoring[channel_idx, ell_idx, theta_idx, phi_idx, _CENSOR_INVALID] += 1


@njit(inline="always")
def _accumulate_invalid(censoring, channel_idx, ell_idx, theta_idx, phi_idx):
    """Record a channel value that is undefined despite valid angular geometry."""
    censoring[channel_idx, ell_idx, theta_idx, phi_idx, _CENSOR_INVALID] += 1


# -----------------------------------------------------------------------------
# Common histogram computation logic -----------------------------------------
# -----------------------------------------------------------------------------

@njit
def _compute_histogram_core(
    dvx, dvy, dvz, dBx, dBy, dBz, drho,
    dvAx, dvAy, dvAz, dzpx, dzpy, dzpz,
    dzmx, dzmy, dzmz, domegax, domegay, domegaz,
    dJx, dJy, dJz,
    dcurvx, dcurvy, dcurvz,
    dgradrhox, dgradrhoy, dgradrhoz,
    Bmx, Bmy, Bmz,
    dx, dy, dz, r,
    ell_idx, theta_bin_edges, phi_bin_edges,
    delta_bin_edges,
    hist, censoring
):
    """Common histogram computation logic shared by all stencil widths.

    All channels are binned on (ℓ, θ, ϕ, Δ) using per-channel Δ bin edges.
    """

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
    # Cast displacement to float so _perp uses consistent dtype under numba
    displacement_vec = np.array([dz, dy, dx], dtype=np.float64)

    dB_perp = _perp(dB_vec, B_unit)
    dB_perp_mag = np.sqrt((dB_perp ** 2).sum())

    dv_perp = _perp(dv_vec, B_unit)
    dv_perp_mag = np.sqrt((dv_perp ** 2).sum())

    displacement_perp = _perp(displacement_vec, B_unit)
    displacement_perp_mag = np.sqrt((displacement_perp ** 2).sum())

    if displacement_perp_mag > 0 and dB_perp_mag > 0:
        cos_phi = (
            np.abs((displacement_perp * dB_perp).sum())
            / (displacement_perp_mag * dB_perp_mag)
        )
        if cos_phi > 1.0:
            cos_phi = 1.0
        elif cos_phi < -1.0:
            cos_phi = -1.0
        phi_val = np.arccos(cos_phi)
    else:
        phi_val = 0.0

    dv = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
    dB = np.sqrt(dBx * dBx + dBy * dBy + dBz * dBz)

    dVA = np.sqrt(dvAx*dvAx + dvAy*dvAy + dvAz*dvAz)
    dZp = np.sqrt(dzpx*dzpx + dzpy*dzpy + dzpz*dzpz)
    dZm = np.sqrt(dzmx*dzmx + dzmy*dzmy + dzmz*dzmz)
    dOmega = np.sqrt(domegax*domegax + domegay*domegay + domegaz*domegaz)
    dJ = np.sqrt(dJx*dJx + dJy*dJy + dJz*dJz)
    dCurv = np.sqrt(dcurvx*dcurvx + dcurvy*dcurvy + dcurvz*dcurvz)
    dGradRho = np.sqrt(dgradrhox*dgradrhox + dgradrhoy*dgradrhoy + dgradrhoz*dgradrhoz)

    # Bin indices
    theta_idx = find_bin_index_binary(theta_val, theta_bin_edges)
    phi_idx = find_bin_index_binary(phi_val, phi_bin_edges)

    # Check if angular indices are valid
    if theta_idx < 0 or phi_idx < 0:
        return

    # Precompute integer channel indices for numba-friendly indexing
    c_D_V = Channel.D_V.value
    c_D_B = Channel.D_B.value
    c_D_RHO = Channel.D_RHO.value
    c_D_VA = Channel.D_VA.value
    c_D_ZPLUS = Channel.D_ZPLUS.value
    c_D_ZMINUS = Channel.D_ZMINUS.value
    c_D_OMEGA = Channel.D_OMEGA.value
    c_D_J = Channel.D_J.value
    c_D_CURV = Channel.D_CURV.value
    c_D_GRAD_RHO = Channel.D_GRAD_RHO.value
    c_D_B_ratio = Channel.D_B_over_Bmean_loc.value
    c_cross_v_b = Channel.D_Vperp_CROSS_D_Bperp.value
    c_cross_v_omega = Channel.D_Vperp_CROSS_D_Omegaperp.value
    c_cross_b_j = Channel.D_Bperp_CROSS_D_Jperp.value
    c_cross_omega_j = Channel.D_Omegaperp_CROSS_D_Jperp.value
    c_mag_v_b = Channel.D_Vperp_D_Bperp_MAG.value
    c_mag_v_omega = Channel.D_Vperp_D_Omegaperp_MAG.value
    c_mag_b_j = Channel.D_Bperp_D_Jperp_MAG.value
    c_mag_omega_j = Channel.D_Omegaperp_D_Jperp_MAG.value
    c_ratio_v_b = Channel.D_Vperp_D_Bperp_CROSS_MAG_RATIO.value
    c_ratio_v_omega = Channel.D_Vperp_D_Omegaperp_CROSS_MAG_RATIO.value
    c_ratio_b_j = Channel.D_Bperp_D_Jperp_CROSS_MAG_RATIO.value
    c_ratio_omega_j = Channel.D_Omegaperp_D_Jperp_CROSS_MAG_RATIO.value
    c_cross_zp_zm = Channel.D_Zplusperp_CROSS_D_Zminusperp.value
    c_mag_zp_zm = Channel.D_Zplusperp_D_Zminusperp_MAG.value
    c_ratio_zp_zm = Channel.D_Zplusperp_D_Zminusperp_CROSS_MAG_RATIO.value

    # Magnitudes
    _accumulate_bin(dv, delta_bin_edges[c_D_V], hist, censoring, c_D_V, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dB, delta_bin_edges[c_D_B], hist, censoring, c_D_B, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(drho, delta_bin_edges[c_D_RHO], hist, censoring, c_D_RHO, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dVA, delta_bin_edges[c_D_VA], hist, censoring, c_D_VA, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dZp, delta_bin_edges[c_D_ZPLUS], hist, censoring, c_D_ZPLUS, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dZm, delta_bin_edges[c_D_ZMINUS], hist, censoring, c_D_ZMINUS, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dOmega, delta_bin_edges[c_D_OMEGA], hist, censoring, c_D_OMEGA, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dJ, delta_bin_edges[c_D_J], hist, censoring, c_D_J, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dCurv, delta_bin_edges[c_D_CURV], hist, censoring, c_D_CURV, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(dGradRho, delta_bin_edges[c_D_GRAD_RHO], hist, censoring, c_D_GRAD_RHO, ell_idx, theta_idx, phi_idx)

    # Normalized |δB| / |B_mean|
    dB_over_Bmean = dB / Bmean_mag
    _accumulate_bin(dB_over_Bmean, delta_bin_edges[c_D_B_ratio], hist, censoring, c_D_B_ratio, ell_idx, theta_idx, phi_idx)

    # Perpendicular field components for cross/product channels
    dOmega_vec = np.array([domegaz, domegay, domegax])
    dJ_vec = np.array([dJz, dJy, dJx])

    dOmega_perp = _perp(dOmega_vec, B_unit)
    dJ_perp = _perp(dJ_vec, B_unit)

    dOmega_perp_mag = np.sqrt((dOmega_perp**2).sum())
    dJ_perp_mag = np.sqrt((dJ_perp**2).sum())

    # Cross products (perpendicular)
    cross_v_b = np.sqrt((np.cross(dv_perp, dB_perp) ** 2).sum())
    cross_v_omega = np.sqrt((np.cross(dv_perp, dOmega_perp) ** 2).sum())
    cross_b_j = np.sqrt((np.cross(dB_perp, dJ_perp) ** 2).sum())
    cross_omega_j = np.sqrt((np.cross(dOmega_perp, dJ_perp) ** 2).sum())

    _accumulate_bin(cross_v_b, delta_bin_edges[c_cross_v_b], hist, censoring, c_cross_v_b, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(cross_v_omega, delta_bin_edges[c_cross_v_omega], hist, censoring, c_cross_v_omega, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(cross_b_j, delta_bin_edges[c_cross_b_j], hist, censoring, c_cross_b_j, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(cross_omega_j, delta_bin_edges[c_cross_omega_j], hist, censoring, c_cross_omega_j, ell_idx, theta_idx, phi_idx)

    # Magnitude products (perpendicular)
    mag_v_b = dv_perp_mag * dB_perp_mag
    mag_v_omega = dv_perp_mag * dOmega_perp_mag
    mag_b_j = dB_perp_mag * dJ_perp_mag
    mag_omega_j = dOmega_perp_mag * dJ_perp_mag

    _accumulate_bin(mag_v_b, delta_bin_edges[c_mag_v_b], hist, censoring, c_mag_v_b, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(mag_v_omega, delta_bin_edges[c_mag_v_omega], hist, censoring, c_mag_v_omega, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(mag_b_j, delta_bin_edges[c_mag_b_j], hist, censoring, c_mag_b_j, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(mag_omega_j, delta_bin_edges[c_mag_omega_j], hist, censoring, c_mag_omega_j, ell_idx, theta_idx, phi_idx)

    # Ratios (cross/product) – only bin when denominator is positive
    if mag_v_b > 0.0:
        _accumulate_bin(min(1.0, cross_v_b / mag_v_b), delta_bin_edges[c_ratio_v_b], hist, censoring, c_ratio_v_b, ell_idx, theta_idx, phi_idx)
    else:
        _accumulate_invalid(censoring, c_ratio_v_b, ell_idx, theta_idx, phi_idx)
    if mag_v_omega > 0.0:
        _accumulate_bin(min(1.0, cross_v_omega / mag_v_omega), delta_bin_edges[c_ratio_v_omega], hist, censoring, c_ratio_v_omega, ell_idx, theta_idx, phi_idx)
    else:
        _accumulate_invalid(censoring, c_ratio_v_omega, ell_idx, theta_idx, phi_idx)
    if mag_b_j > 0.0:
        _accumulate_bin(min(1.0, cross_b_j / mag_b_j), delta_bin_edges[c_ratio_b_j], hist, censoring, c_ratio_b_j, ell_idx, theta_idx, phi_idx)
    else:
        _accumulate_invalid(censoring, c_ratio_b_j, ell_idx, theta_idx, phi_idx)
    if mag_omega_j > 0.0:
        _accumulate_bin(min(1.0, cross_omega_j / mag_omega_j), delta_bin_edges[c_ratio_omega_j], hist, censoring, c_ratio_omega_j, ell_idx, theta_idx, phi_idx)
    else:
        _accumulate_invalid(censoring, c_ratio_omega_j, ell_idx, theta_idx, phi_idx)

    # Elsasser alignment angle: θ^(z+,z-) = ⟨δz⁺_⊥ × δz⁻_⊥⟩ / ⟨|δz⁺_⊥||δz⁻_⊥|⟩
    dZplus_vec = np.array([dzpz, dzpy, dzpx])
    dZminus_vec = np.array([dzmz, dzmy, dzmx])

    dZplus_perp = _perp(dZplus_vec, B_unit)
    dZminus_perp = _perp(dZminus_vec, B_unit)

    dZplus_perp_mag = np.sqrt((dZplus_perp**2).sum())
    dZminus_perp_mag = np.sqrt((dZminus_perp**2).sum())

    cross_zp_zm = np.sqrt((np.cross(dZplus_perp, dZminus_perp) ** 2).sum())
    mag_zp_zm = dZplus_perp_mag * dZminus_perp_mag

    _accumulate_bin(cross_zp_zm, delta_bin_edges[c_cross_zp_zm], hist, censoring, c_cross_zp_zm, ell_idx, theta_idx, phi_idx)
    _accumulate_bin(mag_zp_zm, delta_bin_edges[c_mag_zp_zm], hist, censoring, c_mag_zp_zm, ell_idx, theta_idx, phi_idx)

    if mag_zp_zm > 0.0:
        _accumulate_bin(min(1.0, cross_zp_zm / mag_zp_zm), delta_bin_edges[c_ratio_zp_zm], hist, censoring, c_ratio_zp_zm, ell_idx, theta_idx, phi_idx)
    else:
        _accumulate_invalid(censoring, c_ratio_zp_zm, ell_idx, theta_idx, phi_idx)


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
    curv_x: np.ndarray, curv_y: np.ndarray, curv_z: np.ndarray,
    grad_rho_x: np.ndarray, grad_rho_y: np.ndarray, grad_rho_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    delta_bin_edges: list,
    cell_sizes: np.ndarray,
    random_seed: int,
    compact: bool,
) -> np.ndarray:
    """2-point stencil version of histogram computation."""

    N, M = v_x.shape
    n_ell_bins = ell_bin_edges.shape[0] - 1
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    n_delta_bins = delta_bin_edges[0].shape[0] - 1

    if slice_axis == 1:
        dx, dy, dz = 0.0, delta_i * cell_sizes[1], delta_j * cell_sizes[2]
    elif slice_axis == 2:
        dx, dy, dz = delta_i * cell_sizes[0], 0.0, delta_j * cell_sizes[2]
    elif slice_axis == 3:
        dx, dy, dz = delta_i * cell_sizes[0], delta_j * cell_sizes[1], 0.0
    else:
        raise ValueError("slice_axis must be 1, 2, or 3")

    hist = np.zeros((N_CHANNELS, 1 if compact else n_ell_bins, n_theta_bins, n_phi_bins, n_delta_bins), dtype=np.int64)
    censoring = np.zeros((N_CHANNELS, 1 if compact else n_ell_bins, n_theta_bins, n_phi_bins, N_CENSOR_KINDS), dtype=np.int64)

    r = (dx * dx + dy * dy + dz * dz) ** 0.5
    ell_idx = find_bin_index_binary(r, ell_bin_edges)
    if ell_idx == -1:
        return hist, censoring
    output_ell_idx = 0 if compact else ell_idx

    if random_seed >= 0:
        np.random.seed(random_seed)
    flat_indices = np.random.randint(0, M * N, size=N_random_subsamples)
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

        dcurvx = _diff_2pt(curv_x, jp, ip, j, i)
        dcurvy = _diff_2pt(curv_y, jp, ip, j, i)
        dcurvz = _diff_2pt(curv_z, jp, ip, j, i)

        dgradrhox = _diff_2pt(grad_rho_x, jp, ip, j, i)
        dgradrhoy = _diff_2pt(grad_rho_y, jp, ip, j, i)
        dgradrhoz = _diff_2pt(grad_rho_z, jp, ip, j, i)

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
            dcurvx, dcurvy, dcurvz,
            dgradrhox, dgradrhoy, dgradrhoz,
            Bmx, Bmy, Bmz,
            dx, dy, dz, r,
            output_ell_idx, theta_bin_edges, phi_bin_edges,
            delta_bin_edges,
            hist, censoring,
        )

    return hist, censoring


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
    curv_x: np.ndarray, curv_y: np.ndarray, curv_z: np.ndarray,
    grad_rho_x: np.ndarray, grad_rho_y: np.ndarray, grad_rho_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    delta_bin_edges: list,
    cell_sizes: np.ndarray,
    random_seed: int,
    compact: bool,
) -> np.ndarray:
    """3-point stencil version of histogram computation."""

    N, M = v_x.shape
    n_ell_bins = ell_bin_edges.shape[0] - 1
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    n_delta_bins = delta_bin_edges[0].shape[0] - 1

    if slice_axis == 1:
        dx, dy, dz = 0.0, delta_i * cell_sizes[1], delta_j * cell_sizes[2]
    elif slice_axis == 2:
        dx, dy, dz = delta_i * cell_sizes[0], 0.0, delta_j * cell_sizes[2]
    elif slice_axis == 3:
        dx, dy, dz = delta_i * cell_sizes[0], delta_j * cell_sizes[1], 0.0
    else:
        raise ValueError("slice_axis must be 1, 2, or 3")

    hist = np.zeros((N_CHANNELS, 1 if compact else n_ell_bins, n_theta_bins, n_phi_bins, n_delta_bins), dtype=np.int64)
    censoring = np.zeros((N_CHANNELS, 1 if compact else n_ell_bins, n_theta_bins, n_phi_bins, N_CENSOR_KINDS), dtype=np.int64)

    r = (dx * dx + dy * dy + dz * dz) ** 0.5
    ell_idx = find_bin_index_binary(r, ell_bin_edges)
    if ell_idx == -1:
        return hist, censoring

    output_ell_idx = 0 if compact else ell_idx
    if random_seed >= 0:
        np.random.seed(random_seed)
    flat_indices = np.random.randint(0, M * N, size=N_random_subsamples)
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

        dcurvx = _diff_3pt(curv_x, jp, ip, j, i, jm, im)
        dcurvy = _diff_3pt(curv_y, jp, ip, j, i, jm, im)
        dcurvz = _diff_3pt(curv_z, jp, ip, j, i, jm, im)

        dgradrhox = _diff_3pt(grad_rho_x, jp, ip, j, i, jm, im)
        dgradrhoy = _diff_3pt(grad_rho_y, jp, ip, j, i, jm, im)
        dgradrhoz = _diff_3pt(grad_rho_z, jp, ip, j, i, jm, im)

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
            dcurvx, dcurvy, dcurvz,
            dgradrhox, dgradrhoy, dgradrhoz,
            Bmx, Bmy, Bmz,
            dx, dy, dz, r,
            output_ell_idx, theta_bin_edges, phi_bin_edges,
            delta_bin_edges,
            hist, censoring,
        )

    return hist, censoring


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
    curv_x: np.ndarray, curv_y: np.ndarray, curv_z: np.ndarray,
    grad_rho_x: np.ndarray, grad_rho_y: np.ndarray, grad_rho_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    delta_bin_edges: list,
    cell_sizes: np.ndarray,
    random_seed: int,
    compact: bool,
) -> np.ndarray:
    """5-point stencil version of histogram computation."""

    N, M = v_x.shape
    n_ell_bins = ell_bin_edges.shape[0] - 1
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    n_delta_bins = delta_bin_edges[0].shape[0] - 1

    if slice_axis == 1:
        dx, dy, dz = 0.0, delta_i * cell_sizes[1], delta_j * cell_sizes[2]
    elif slice_axis == 2:
        dx, dy, dz = delta_i * cell_sizes[0], 0.0, delta_j * cell_sizes[2]
    elif slice_axis == 3:
        dx, dy, dz = delta_i * cell_sizes[0], delta_j * cell_sizes[1], 0.0
    else:
        raise ValueError("slice_axis must be 1, 2, or 3")

    hist = np.zeros((N_CHANNELS, 1 if compact else n_ell_bins, n_theta_bins, n_phi_bins, n_delta_bins), dtype=np.int64)
    censoring = np.zeros((N_CHANNELS, 1 if compact else n_ell_bins, n_theta_bins, n_phi_bins, N_CENSOR_KINDS), dtype=np.int64)

    r = (dx * dx + dy * dy + dz * dz) ** 0.5
    ell_idx = find_bin_index_binary(r, ell_bin_edges)
    if ell_idx == -1:
        return hist, censoring

    output_ell_idx = 0 if compact else ell_idx
    if random_seed >= 0:
        np.random.seed(random_seed)
    flat_indices = np.random.randint(0, M * N, size=N_random_subsamples)
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

        dcurvx = _diff_5pt(curv_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dcurvy = _diff_5pt(curv_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dcurvz = _diff_5pt(curv_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)

        dgradrhox = _diff_5pt(grad_rho_x, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dgradrhoy = _diff_5pt(grad_rho_y, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        dgradrhoz = _diff_5pt(grad_rho_z, jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)

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
            dcurvx, dcurvy, dcurvz,
            dgradrhox, dgradrhoy, dgradrhoz,
            Bmx, Bmy, Bmz,
            dx, dy, dz, r,
            output_ell_idx, theta_bin_edges, phi_bin_edges,
            delta_bin_edges,
            hist, censoring,
        )

    return hist, censoring


def compute_histogram_for_disp_2D(
    v_x: np.ndarray, v_y: np.ndarray, v_z: np.ndarray,
    B_x: np.ndarray, B_y: np.ndarray, B_z: np.ndarray,
    rho: np.ndarray,
    vA_x: np.ndarray, vA_y: np.ndarray, vA_z: np.ndarray,
    zp_x: np.ndarray, zp_y: np.ndarray, zp_z: np.ndarray,
    zm_x: np.ndarray, zm_y: np.ndarray, zm_z: np.ndarray,
    omega_x: np.ndarray, omega_y: np.ndarray, omega_z: np.ndarray,
    J_x: np.ndarray, J_y: np.ndarray, J_z: np.ndarray,
    curv_x: np.ndarray, curv_y: np.ndarray, curv_z: np.ndarray,
    grad_rho_x: np.ndarray, grad_rho_y: np.ndarray, grad_rho_z: np.ndarray,
    delta_i: int, delta_j: int, slice_axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    delta_bin_edges: list,
    stencil_width: int = 2,
    cell_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    random_seed: int | None = None,
    compact: bool = False,
    return_censoring: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Dispatch to the selected normalized increment-filter kernel.

    Spatial origins are Monte Carlo samples drawn with replacement.  This is
    unbiased and makes sampling cost proportional to the requested sample
    count rather than to the full slice area.
    """
    if slice_axis not in (1, 2, 3):
        raise ValueError("slice_axis must be 1, 2, or 3")
    if delta_i == 0 and delta_j == 0:
        raise ValueError("zero displacement is not a valid structure-function offset")
    if not isinstance(N_random_subsamples, int) or N_random_subsamples <= 0:
        raise ValueError("N_random_subsamples must be a positive integer")

    # Convert to tuple so numba sees a fixed, indexable container
    delta_bin_edges = tuple(delta_bin_edges)
    cell_sizes_array = np.asarray(cell_sizes, dtype=float)
    if cell_sizes_array.shape != (3,) or not np.all(np.isfinite(cell_sizes_array)) or np.any(cell_sizes_array <= 0.0):
        raise ValueError("cell_sizes must contain three finite positive Cartesian spacings")
    seed = -1 if random_seed is None else int(random_seed)

    if stencil_width == 2:
        output = compute_histogram_for_disp_2D_stencil2(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, curv_x, curv_y, curv_z,
            grad_rho_x, grad_rho_y, grad_rho_z,
            delta_i, delta_j, slice_axis,
            N_random_subsamples, ell_bin_edges, theta_bin_edges,
            phi_bin_edges, delta_bin_edges, cell_sizes_array, seed, compact,
        )
    elif stencil_width == 3:
        output = compute_histogram_for_disp_2D_stencil3(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, curv_x, curv_y, curv_z,
            grad_rho_x, grad_rho_y, grad_rho_z,
            delta_i, delta_j, slice_axis,
            N_random_subsamples, ell_bin_edges, theta_bin_edges,
            phi_bin_edges, delta_bin_edges, cell_sizes_array, seed, compact,
        )
    elif stencil_width == 5:
        output = compute_histogram_for_disp_2D_stencil5(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, curv_x, curv_y, curv_z,
            grad_rho_x, grad_rho_y, grad_rho_z,
            delta_i, delta_j, slice_axis,
            N_random_subsamples, ell_bin_edges, theta_bin_edges,
            phi_bin_edges, delta_bin_edges, cell_sizes_array, seed, compact,
        )
    else:
        raise ValueError(f"Unsupported stencil_width: {stencil_width}")
    return output if return_censoring else output[0]
