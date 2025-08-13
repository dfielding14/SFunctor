"""Simplified histogram-building kernels for 2-D slice structure-function analysis.

This module is Numba-accelerated for performance. All per-displacement 
statistics computation happens here.

Key simplifications from original:
- Single unified stencil function instead of three duplicates
- Plain integer channel indices instead of complex enum system
- Minimal validation in hot paths
"""
import numpy as np
from numba import njit
from typing import Tuple
from enum import IntEnum

# Simple channel constants - no enum overhead
# Structure-function magnitudes (go into 5D histogram)
D_V = 0
D_B = 1
D_RHO = 2
D_VA = 3
D_ZPLUS = 4
D_ZMINUS = 5
D_OMEGA = 6
D_J = 7
D_CURV = 8
D_GRAD_RHO = 9
D_B_OVER_BMEAN = 10
N_MAG_CHANNELS = 11

# Cross products and magnitudes (go into 3D histogram)
D_VPERP_CROSS_BPERP = 0
D_VPERP_CROSS_VAPERP = 1
D_VPERP_CROSS_OMEGAPERP = 2
D_BPERP_CROSS_JPERP = 3
D_VPERP_D_BPERP_MAG = 4
D_VPERP_D_VAPERP_MAG = 5
D_VPERP_D_OMEGAPERP_MAG = 6
D_BPERP_D_JPERP_MAG = 7
D_V_CROSS_B = 8
D_V_CROSS_VA = 9
D_V_CROSS_OMEGA = 10
D_B_CROSS_J = 11
D_V_D_B_MAG = 12
D_V_D_VA_MAG = 13
D_V_D_OMEGA_MAG = 14
D_B_D_J_MAG = 15
D_CURV_CROSS_GRAD_RHO = 16
D_CURV_D_GRAD_RHO_MAG = 17
N_OTHER_CHANNELS = 18

# Backwards compatibility - create enum that maps to our constants
class Channel(IntEnum):
    """Channel enum for backwards compatibility."""
    D_V = D_V
    D_B = D_B
    D_RHO = D_RHO
    D_VA = D_VA
    D_ZPLUS = D_ZPLUS
    D_ZMINUS = D_ZMINUS
    D_OMEGA = D_OMEGA
    D_J = D_J
    D_CURV = D_CURV
    D_GRAD_RHO = D_GRAD_RHO
    D_B_over_Bmean_loc = D_B_OVER_BMEAN
    # Cross products
    D_Vperp_CROSS_Bperp = 11
    D_Vperp_CROSS_VAperp = 12
    D_Vperp_CROSS_Omegaperp = 13
    D_Bperp_CROSS_Jperp = 14
    D_Vperp_D_Bperp_MAG = 15
    D_Vperp_D_VAperp_MAG = 16
    D_Vperp_D_Omegaperp_MAG = 17
    D_Bperp_D_Jperp_MAG = 18
    D_V_CROSS_B = 19
    D_V_CROSS_VA = 20
    D_V_CROSS_OMEGA = 21
    D_B_CROSS_J = 22
    D_CURV_CROSS_GRAD_RHO = 23
    D_V_D_B_MAG = 24
    D_V_D_VA_MAG = 25
    D_V_D_OMEGA_MAG = 26
    D_B_D_J_MAG = 27
    D_CURV_D_GRAD_RHO_MAG = 28

N_CHANNELS = 29  # Total for backwards compatibility

# Backwards compatibility lists
MAG_CHANNELS = (
    Channel.D_V,
    Channel.D_B,
    Channel.D_RHO,
    Channel.D_VA,
    Channel.D_ZPLUS,
    Channel.D_ZMINUS,
    Channel.D_OMEGA,
    Channel.D_J,
    Channel.D_CURV,
    Channel.D_GRAD_RHO,
    Channel.D_B_over_Bmean_loc,
)

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
    Channel.D_CURV_CROSS_GRAD_RHO,
    Channel.D_CURV_D_GRAD_RHO_MAG,
)

# Maps for backwards compatibility
MAG_IDX = {ch: i for i, ch in enumerate(MAG_CHANNELS)}
OTHER_IDX = {ch: i for i, ch in enumerate(OTHER_CHANNELS)}


@njit(inline="always")
def find_bin_index_binary(value, bin_edges):
    """Binary search for histogram bin."""
    left, right = 0, len(bin_edges) - 1
    while left < right:
        mid = (left + right) // 2
        if value < bin_edges[mid]:
            right = mid
        else:
            left = mid + 1
    if left == 0 or left == len(bin_edges):
        return -1
    return left - 1


@njit(inline="always")
def compute_stencil_diff(arr, indices, stencil):
    """Unified difference computation for all stencil types.
    
    Args:
        arr: 2D array to compute difference on
        indices: tuple of (jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        stencil: stencil width (2, 3, or 5)
    """
    jp2, ip2, jp, ip, j, i, jm, im, jm2, im2 = indices
    
    if stencil == 2:
        return arr[jp, ip] - arr[j, i]
    elif stencil == 3:
        return arr[jp, ip] - 2.0 * arr[j, i] + arr[jm, im]
    else:  # stencil == 5
        return (-arr[jp2, ip2] + 16.0 * arr[jp, ip] - 30.0 * arr[j, i] + 
                16.0 * arr[jm, im] - arr[jm2, im2])


@njit(inline="always")
def compute_mean_B(B_vals, stencil):
    """Unified mean B computation for all stencil types.
    
    Args:
        B_vals: tuple of (B_plus2, B_plus, B_center, B_minus, B_minus2)
        stencil: stencil width (2, 3, or 5)
    """
    B_plus2, B_plus, B_center, B_minus, B_minus2 = B_vals
    
    if stencil == 2:
        return (B_plus + B_center) / 2.0
    elif stencil == 3:
        return (B_plus + B_center + B_minus) / 3.0
    else:  # stencil == 5
        return (B_plus2 + B_plus + B_center + B_minus + B_minus2) / 5.0


@njit(inline="always")
def perp_component(vec, B_unit):
    """Perpendicular component of vector relative to B."""
    dot = vec[0] * B_unit[0] + vec[1] * B_unit[1] + vec[2] * B_unit[2]
    return (
        vec[0] - dot * B_unit[0],
        vec[1] - dot * B_unit[1],
        vec[2] - dot * B_unit[2]
    )


@njit(inline="always")
def cross_product(a, b):
    """3D cross product."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


@njit(inline="always")
def magnitude(vec):
    """3D vector magnitude."""
    return (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5


@njit(cache=True)
def compute_histogram_unified(
    # Field arrays
    v_x, v_y, v_z, B_x, B_y, B_z, rho,
    vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
    zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
    J_x, J_y, J_z, curv_x, curv_y, curv_z,
    grad_rho_x, grad_rho_y, grad_rho_z,
    # Parameters
    delta_i, delta_j, slice_axis,
    N_random_subsamples,
    ell_bin_edges, theta_bin_edges, phi_bin_edges,
    sf_channel_bin_edges, product_bin_edges,
    stencil_width
):
    """Single unified histogram computation function for all stencil widths.
    
    This replaces three nearly-identical 350+ line functions with one.
    The only differences between stencils are:
    - How differences are computed
    - How mean B is computed
    - Index calculations for additional points
    """
    N, M = v_x.shape
    n_ell_bins = len(ell_bin_edges) - 1
    n_theta_bins = len(theta_bin_edges) - 1
    n_phi_bins = len(phi_bin_edges) - 1
    n_sf_bins = len(sf_channel_bin_edges[0]) - 1
    n_product_bins = len(product_bin_edges) - 1
    
    # Determine displacement vector in 3D
    if slice_axis == 1:
        dx, dy, dz = 0, delta_i, delta_j
    elif slice_axis == 2:
        dx, dy, dz = delta_i, 0, delta_j
    else:
        dx, dy, dz = delta_i, delta_j, 0
    
    # Initialize histograms
    hist_mag = np.zeros((N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins), dtype=np.int64)
    hist_other = np.zeros((N_OTHER_CHANNELS, n_ell_bins, n_product_bins), dtype=np.int64)
    
    # Compute displacement magnitude and bin
    r = (delta_i * delta_i + delta_j * delta_j) ** 0.5
    ell_idx = find_bin_index_binary(r, ell_bin_edges)
    if ell_idx == -1:
        return hist_mag, hist_other
    
    # Random sampling
    flat_indices = np.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M
    random_points_x = flat_indices % M
    
    for idx in range(N_random_subsamples):
        i = random_points_x[idx]
        j = random_points_y[idx]
        
        # Calculate all necessary indices based on stencil width
        # For stencil=2, only jp/ip are used
        # For stencil=3, jp/ip and jm/im are used  
        # For stencil=5, all indices are used
        jp = (j + delta_j) % N
        ip = (i + delta_i) % M
        
        if stencil_width >= 3:
            jm = (j - delta_j) % N
            im = (i - delta_i) % M
        else:
            jm = j  # Won't be used
            im = i
            
        if stencil_width == 5:
            jp2 = (j + 2*delta_j) % N
            ip2 = (i + 2*delta_i) % M
            jm2 = (j - 2*delta_j) % N
            im2 = (i - 2*delta_i) % M
        else:
            jp2 = jp  # Won't be used
            ip2 = ip
            jm2 = jm
            im2 = im
        
        indices = (jp2, ip2, jp, ip, j, i, jm, im, jm2, im2)
        
        # Compute all field differences using unified function
        dvx = compute_stencil_diff(v_x, indices, stencil_width)
        dvy = compute_stencil_diff(v_y, indices, stencil_width)
        dvz = compute_stencil_diff(v_z, indices, stencil_width)
        
        dBx = compute_stencil_diff(B_x, indices, stencil_width)
        dBy = compute_stencil_diff(B_y, indices, stencil_width)
        dBz = compute_stencil_diff(B_z, indices, stencil_width)
        
        drho = abs(compute_stencil_diff(rho, indices, stencil_width))
        
        dvAx = compute_stencil_diff(vA_x, indices, stencil_width)
        dvAy = compute_stencil_diff(vA_y, indices, stencil_width)
        dvAz = compute_stencil_diff(vA_z, indices, stencil_width)
        
        dzpx = compute_stencil_diff(zp_x, indices, stencil_width)
        dzpy = compute_stencil_diff(zp_y, indices, stencil_width)
        dzpz = compute_stencil_diff(zp_z, indices, stencil_width)
        
        dzmx = compute_stencil_diff(zm_x, indices, stencil_width)
        dzmy = compute_stencil_diff(zm_y, indices, stencil_width)
        dzmz = compute_stencil_diff(zm_z, indices, stencil_width)
        
        domegax = compute_stencil_diff(omega_x, indices, stencil_width)
        domegay = compute_stencil_diff(omega_y, indices, stencil_width)
        domegaz = compute_stencil_diff(omega_z, indices, stencil_width)
        
        dJx = compute_stencil_diff(J_x, indices, stencil_width)
        dJy = compute_stencil_diff(J_y, indices, stencil_width)
        dJz = compute_stencil_diff(J_z, indices, stencil_width)
        
        dcurvx = compute_stencil_diff(curv_x, indices, stencil_width)
        dcurvy = compute_stencil_diff(curv_y, indices, stencil_width)
        dcurvz = compute_stencil_diff(curv_z, indices, stencil_width)
        
        dgradrhox = compute_stencil_diff(grad_rho_x, indices, stencil_width)
        dgradrhoy = compute_stencil_diff(grad_rho_y, indices, stencil_width)
        dgradrhoz = compute_stencil_diff(grad_rho_z, indices, stencil_width)
        
        # Compute mean B field based on stencil
        B_vals = (
            B_x[jp2, ip2] if stencil_width == 5 else B_x[jp, ip],
            B_x[jp, ip],
            B_x[j, i],
            B_x[jm, im] if stencil_width >= 3 else B_x[j, i],
            B_x[jm2, im2] if stencil_width == 5 else B_x[j, i]
        )
        Bmx = compute_mean_B(B_vals, stencil_width)
        
        B_vals = (
            B_y[jp2, ip2] if stencil_width == 5 else B_y[jp, ip],
            B_y[jp, ip],
            B_y[j, i],
            B_y[jm, im] if stencil_width >= 3 else B_y[j, i],
            B_y[jm2, im2] if stencil_width == 5 else B_y[j, i]
        )
        Bmy = compute_mean_B(B_vals, stencil_width)
        
        B_vals = (
            B_z[jp2, ip2] if stencil_width == 5 else B_z[jp, ip],
            B_z[jp, ip],
            B_z[j, i],
            B_z[jm, im] if stencil_width >= 3 else B_z[j, i],
            B_z[jm2, im2] if stencil_width == 5 else B_z[j, i]
        )
        Bmz = compute_mean_B(B_vals, stencil_width)
        
        # Compute magnitudes
        dv_mag = magnitude((dvx, dvy, dvz))
        dB_mag = magnitude((dBx, dBy, dBz))
        dvA_mag = magnitude((dvAx, dvAy, dvAz))
        dzp_mag = magnitude((dzpx, dzpy, dzpz))
        dzm_mag = magnitude((dzmx, dzmy, dzmz))
        domega_mag = magnitude((domegax, domegay, domegaz))
        dJ_mag = magnitude((dJx, dJy, dJz))
        dcurv_mag = magnitude((dcurvx, dcurvy, dcurvz))
        dgradrho_mag = magnitude((dgradrhox, dgradrhoy, dgradrhoz))
        
        # Mean B magnitude
        Bm_mag = magnitude((Bmx, Bmy, Bmz))
        dB_over_Bmean = dB_mag / Bm_mag if Bm_mag > 0 else 0.0
        
        # Compute angles
        if r > 0:
            theta = np.arccos(dz / r) if abs(dz / r) <= 1.0 else 0.0
        else:
            theta = 0.0
        phi = np.arctan2(dy, dx)
        
        theta_idx = find_bin_index_binary(theta, theta_bin_edges)
        phi_idx = find_bin_index_binary(phi, phi_bin_edges)
        
        if theta_idx == -1 or phi_idx == -1:
            continue
        
        # Bin magnitude channels
        mag_values = [
            dv_mag, dB_mag, drho, dvA_mag, dzp_mag, dzm_mag,
            domega_mag, dJ_mag, dcurv_mag, dgradrho_mag, dB_over_Bmean
        ]
        
        for ch_idx, value in enumerate(mag_values):
            sf_idx = find_bin_index_binary(value, sf_channel_bin_edges[ch_idx])
            if sf_idx != -1:
                hist_mag[ch_idx, ell_idx, theta_idx, phi_idx, sf_idx] += 1
        
        # Compute perpendicular components and cross products
        if Bm_mag > 0:
            B_unit = (Bmx / Bm_mag, Bmy / Bm_mag, Bmz / Bm_mag)
            
            dv_perp = perp_component((dvx, dvy, dvz), B_unit)
            dB_perp = perp_component((dBx, dBy, dBz), B_unit)
            dvA_perp = perp_component((dvAx, dvAy, dvAz), B_unit)
            domega_perp = perp_component((domegax, domegay, domegaz), B_unit)
            dJ_perp = perp_component((dJx, dJy, dJz), B_unit)
            
            # Cross products
            cross_values = [
                magnitude(cross_product(dv_perp, dB_perp)),
                magnitude(cross_product(dv_perp, dvA_perp)),
                magnitude(cross_product(dv_perp, domega_perp)),
                magnitude(cross_product(dB_perp, dJ_perp)),
                magnitude(dv_perp) * magnitude(dB_perp),
                magnitude(dv_perp) * magnitude(dvA_perp),
                magnitude(dv_perp) * magnitude(domega_perp),
                magnitude(dB_perp) * magnitude(dJ_perp),
                magnitude(cross_product((dvx, dvy, dvz), (dBx, dBy, dBz))),
                magnitude(cross_product((dvx, dvy, dvz), (dvAx, dvAy, dvAz))),
                magnitude(cross_product((dvx, dvy, dvz), (domegax, domegay, domegaz))),
                magnitude(cross_product((dBx, dBy, dBz), (dJx, dJy, dJz))),
                dv_mag * dB_mag,
                dv_mag * dvA_mag,
                dv_mag * domega_mag,
                dB_mag * dJ_mag,
                magnitude(cross_product((dcurvx, dcurvy, dcurvz), (dgradrhox, dgradrhoy, dgradrhoz))),
                dcurv_mag * dgradrho_mag
            ]
            
            for ch_idx, value in enumerate(cross_values):
                prod_idx = find_bin_index_binary(value, product_bin_edges)
                if prod_idx != -1:
                    hist_other[ch_idx, ell_idx, prod_idx] += 1
    
    return hist_mag, hist_other


# Backwards compatibility - old function names that call unified version
@njit(cache=True)
def compute_histogram_for_disp_2D_stencil2(
    v_x, v_y, v_z, B_x, B_y, B_z, rho,
    vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
    zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
    J_x, J_y, J_z, curv_x, curv_y, curv_z,
    grad_rho_x, grad_rho_y, grad_rho_z,
    delta_i, delta_j, slice_axis,
    N_random_subsamples,
    ell_bin_edges, theta_bin_edges, phi_bin_edges,
    sf_channel_bin_edges, product_bin_edges
):
    """Backwards compatibility wrapper for 2-point stencil."""
    return compute_histogram_unified(
        v_x, v_y, v_z, B_x, B_y, B_z, rho,
        vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
        zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
        J_x, J_y, J_z, curv_x, curv_y, curv_z,
        grad_rho_x, grad_rho_y, grad_rho_z,
        delta_i, delta_j, slice_axis,
        N_random_subsamples,
        ell_bin_edges, theta_bin_edges, phi_bin_edges,
        sf_channel_bin_edges, product_bin_edges,
        stencil_width=2
    )

@njit(cache=True)
def compute_histogram_for_disp_2D_stencil3(
    v_x, v_y, v_z, B_x, B_y, B_z, rho,
    vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
    zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
    J_x, J_y, J_z, curv_x, curv_y, curv_z,
    grad_rho_x, grad_rho_y, grad_rho_z,
    delta_i, delta_j, slice_axis,
    N_random_subsamples,
    ell_bin_edges, theta_bin_edges, phi_bin_edges,
    sf_channel_bin_edges, product_bin_edges
):
    """Backwards compatibility wrapper for 3-point stencil."""
    return compute_histogram_unified(
        v_x, v_y, v_z, B_x, B_y, B_z, rho,
        vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
        zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
        J_x, J_y, J_z, curv_x, curv_y, curv_z,
        grad_rho_x, grad_rho_y, grad_rho_z,
        delta_i, delta_j, slice_axis,
        N_random_subsamples,
        ell_bin_edges, theta_bin_edges, phi_bin_edges,
        sf_channel_bin_edges, product_bin_edges,
        stencil_width=3
    )

@njit(cache=True)
def compute_histogram_for_disp_2D_stencil5(
    v_x, v_y, v_z, B_x, B_y, B_z, rho,
    vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
    zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
    J_x, J_y, J_z, curv_x, curv_y, curv_z,
    grad_rho_x, grad_rho_y, grad_rho_z,
    delta_i, delta_j, slice_axis,
    N_random_subsamples,
    ell_bin_edges, theta_bin_edges, phi_bin_edges,
    sf_channel_bin_edges, product_bin_edges
):
    """Backwards compatibility wrapper for 5-point stencil."""
    return compute_histogram_unified(
        v_x, v_y, v_z, B_x, B_y, B_z, rho,
        vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
        zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
        J_x, J_y, J_z, curv_x, curv_y, curv_z,
        grad_rho_x, grad_rho_y, grad_rho_z,
        delta_i, delta_j, slice_axis,
        N_random_subsamples,
        ell_bin_edges, theta_bin_edges, phi_bin_edges,
        sf_channel_bin_edges, product_bin_edges,
        stencil_width=5
    )

# Main dispatcher function
@njit(cache=True)
def compute_histogram_for_disp_2D(
    v_x, v_y, v_z, B_x, B_y, B_z, rho,
    vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
    zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
    J_x, J_y, J_z, curv_x, curv_y, curv_z,
    grad_rho_x, grad_rho_y, grad_rho_z,
    delta_i, delta_j, slice_axis,
    N_random_subsamples,
    ell_bin_edges, theta_bin_edges, phi_bin_edges,
    sf_channel_bin_edges, product_bin_edges,
    stencil_width=2
):
    """Backwards-compatible wrapper that calls unified function."""
    return compute_histogram_unified(
        v_x, v_y, v_z, B_x, B_y, B_z, rho,
        vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
        zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
        J_x, J_y, J_z, curv_x, curv_y, curv_z,
        grad_rho_x, grad_rho_y, grad_rho_z,
        delta_i, delta_j, slice_axis,
        N_random_subsamples,
        ell_bin_edges, theta_bin_edges, phi_bin_edges,
        sf_channel_bin_edges, product_bin_edges,
        stencil_width
    )