#!/usr/bin/env python3
"""Plot structure functions from combined results.

This script creates visualizations of the structure functions computed
from the histogram data.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib.ticker import LogLocator
import matplotlib.patheffects as pe
import cmasher as cmr  # type: ignore
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from sfunctor.core.histograms import Channel

# LaTeX labels for channels
CHANNEL_LABELS = {
    'D_V': r'$\delta v$',
    'D_B': r'$\delta B$',
    'D_RHO': r'$\delta \rho$',
    'D_VA': r'$\delta v_A$',
    'D_ZPLUS': r'$\delta z^+$',
    'D_ZMINUS': r'$\delta z^-$',
    'D_OMEGA': r'$\delta \omega$',
    'D_J': r'$\delta j$',
    'D_CURV': r'$\delta K$',
    'D_GRAD_RHO': r'$\delta |\nabla \rho|$',
    'D_B_over_Bmean_loc': r'$\delta B_\ell / \overline{B}_\ell$',
    # Cross products and ratios
    'D_Vperp_CROSS_D_Bperp': r'$|\delta v_\perp \times \delta B_\perp|$',
    'D_Vperp_CROSS_D_Omegaperp': r'$|\delta v_\perp \times \delta \omega_\perp|$',
    'D_Bperp_CROSS_D_Jperp': r'$|\delta B_\perp \times \delta j_\perp|$',
    'D_Omegaperp_CROSS_D_Jperp': r'$|\delta \omega_\perp \times \delta j_\perp|$',
    'D_Vperp_D_Bperp_MAG': r'$|\delta v_\perp| |\delta B_\perp|$',
    'D_Vperp_D_Omegaperp_MAG': r'$|\delta v_\perp| |\delta \omega_\perp|$',
    'D_Bperp_D_Jperp_MAG': r'$|\delta B_\perp| |\delta j_\perp|$',
    'D_Omegaperp_D_Jperp_MAG': r'$|\delta \omega_\perp| |\delta j_\perp|$',
    'D_Vperp_D_Bperp_CROSS_MAG_RATIO': r'$\sin \theta_{vB}$',
    'D_Vperp_D_Omegaperp_CROSS_MAG_RATIO': r'$\sin \theta_{v\omega}$',
    'D_Bperp_D_Jperp_CROSS_MAG_RATIO': r'$\sin \theta_{Bj}$',
    'D_Omegaperp_D_Jperp_CROSS_MAG_RATIO': r'$\sin \theta_{\omega j}$',
}

# Channels to skip for individual/aniso SF plotting (cross, mag, ratio variants)
SKIP_PLOT_CHANNELS = {
    'D_Vperp_CROSS_D_Bperp',
    'D_Vperp_CROSS_D_Omegaperp',
    'D_Bperp_CROSS_D_Jperp',
    'D_Omegaperp_CROSS_D_Jperp',
    'D_Vperp_D_Bperp_MAG',
    'D_Vperp_D_Omegaperp_MAG',
    'D_Bperp_D_Jperp_MAG',
    'D_Omegaperp_D_Jperp_MAG',
    'D_Vperp_D_Bperp_CROSS_MAG_RATIO',
    'D_Vperp_D_Omegaperp_CROSS_MAG_RATIO',
    'D_Bperp_D_Jperp_CROSS_MAG_RATIO',
    'D_Omegaperp_D_Jperp_CROSS_MAG_RATIO',
}

def get_channel_label(channel_name):
    """Get LaTeX label for channel, with fallback to original name."""
    return CHANNEL_LABELS.get(channel_name, channel_name)


HIST_CMAP = cmr.ocean_r
ORDER_CMAP = cmr.neon
ELL_CMAP = cmr.tropical
DEFAULT_MOMENT_ORDERS = (1, 2, 3, 4, 6, 8, 10)


def apply_house_style():
    """Styling closer to plot_mine_dbb_ratio_kde (no LaTeX dependency)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
    })


def geometric_centers(edges: np.ndarray) -> np.ndarray:
    """Geometric bin centers for log-spaced edges."""
    return np.sqrt(edges[:-1] * edges[1:])


def compute_median_from_hist(hist_2d: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """Return median for each ell bin given counts over bin_centers."""
    medians = np.full(hist_2d.shape[0], np.nan, dtype=float)
    totals = hist_2d.sum(axis=1)
    for i, (row, total) in enumerate(zip(hist_2d, totals)):
        if total <= 0:
            continue
        cdf = np.cumsum(row)
        idx = np.searchsorted(cdf, 0.5 * total)
        idx = min(idx, len(bin_centers) - 1)
        medians[i] = bin_centers[idx]
    return medians


def compute_sf_orders(hist_2d: np.ndarray, bin_centers: np.ndarray, orders=DEFAULT_MOMENT_ORDERS):
    """Compute S_p = <|delta|^p> for each ell, returning dict of arrays."""
    totals = hist_2d.sum(axis=1)
    sf = {p: np.full(hist_2d.shape[0], np.nan, dtype=float) for p in orders}
    for i, (row, total) in enumerate(zip(hist_2d, totals)):
        if total <= 0:
            continue
        for p in orders:
            sf[p][i] = np.sum((bin_centers ** p) * row) / total
    return sf


def compute_skew_kurtosis(hist_2d: np.ndarray, bin_centers: np.ndarray):
    """Compute skewness and kurtosis for each ell bin."""
    totals = hist_2d.sum(axis=1)
    skew = np.full(hist_2d.shape[0], np.nan, dtype=float)
    kurt = np.full(hist_2d.shape[0], np.nan, dtype=float)
    for i, (row, total) in enumerate(zip(hist_2d, totals)):
        if total <= 0:
            continue
        pdf = row / total
        mean = np.sum(bin_centers * pdf)
        var = np.sum(((bin_centers - mean) ** 2) * pdf)
        if var <= 0:
            continue
        std = np.sqrt(var)
        skew[i] = np.sum(((bin_centers - mean) ** 3) * pdf) / (std ** 3)
        kurt[i] = np.sum(((bin_centers - mean) ** 4) * pdf) / (std ** 4)
    return skew, kurt


def compute_qpdf(hist_2d: np.ndarray, bin_edges: np.ndarray, bin_centers: np.ndarray) -> np.ndarray:
    """Return q * P(q|ℓ) for each ℓ using linear PDFs."""
    totals = hist_2d.sum(axis=1, keepdims=True)
    widths = np.diff(bin_edges)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf_linear = np.where(totals > 0, hist_2d / (totals * widths), np.nan)
        qpdf = bin_centers[None, :] * pdf_linear
    return qpdf


def kde_smooth_qp(counts: np.ndarray, bin_centers: np.ndarray, bin_edges: np.ndarray, bandwidth: float = 0.05) -> np.ndarray:
    """
    Smooth q * P(q | ℓ) using a Gaussian kernel in log10-space for each ℓ.
    Adapted from plot_mine_dbb_ratio_kde for general channels.
    """
    log_ratio = np.log10(bin_centers)
    ln10 = np.log(10.0)

    diff = log_ratio[:, None] - log_ratio[None, :]
    kernel = np.exp(-0.5 * (diff / bandwidth) ** 2)
    kernel /= np.sqrt(2.0 * np.pi) * bandwidth

    widths = np.diff(bin_edges)
    q_times_pdf = np.full_like(counts, np.nan, dtype=float)

    for i, row in enumerate(counts):
        total = row.sum()
        if total <= 0:
            continue

        weights = row / total
        density_log = kernel @ weights  # with respect to d(log10 q)
        smoothed = density_log / ln10

        if np.any(~np.isfinite(smoothed)):
            pdf_linear = (row / total) / widths
            smoothed = bin_centers * pdf_linear

        q_times_pdf[i] = smoothed

    return q_times_pdf


def compute_qpdf_slopes(qpdf: np.ndarray, bin_centers: np.ndarray, q_min: float = 40.0, q_max: float = 200.0) -> np.ndarray:
    """Fit slopes to qP(q) profiles between q/qmax in [q_min, q_max]."""
    slopes = np.full(qpdf.shape[0], np.nan, dtype=float)
    for i, profile in enumerate(qpdf):
        if not np.any(profile > 0):
            continue
        peak_idx = np.nanargmax(profile)
        if peak_idx <= 0 or peak_idx >= len(bin_centers):
            continue
        q_norm = bin_centers / bin_centers[peak_idx]
        mask = (q_norm >= q_min) & (q_norm <= q_max) & (profile > 0)
        if np.sum(mask) < 2:
            continue
        logx = np.log(q_norm[mask])
        logy = np.log(profile[mask])
        slope, _ = np.polyfit(logx, logy, 1)
        slopes[i] = -slope  # report positive magnitude
    return slopes


def compute_local_powerlaw_slopes(ell: np.ndarray, values: np.ndarray, window: int = 6) -> np.ndarray:
    """Local log–log slopes using a rolling 6-point linear fit."""
    slopes = np.full_like(values, np.nan, dtype=float)
    log_ell = np.log10(ell)
    log_val = np.log10(values)
    half = window // 2
    for i in range(len(values)):
        start = max(i - half, 0)
        stop = min(i + half, len(values))
        idx = slice(start, stop)
        x = log_ell[idx]
        y = log_val[idx]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < max(3, window - 1):
            continue
        coeffs = np.polyfit(x[mask], y[mask], 1)
        slopes[i] = coeffs[0]
    return slopes


def find_matching_scales(target_x: np.ndarray, target_y: np.ndarray,
                         ref_x: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    """For each target_y, find ref_x where ref_y matches target_y using log-space interpolation."""
    matches = np.full_like(target_x, np.nan, dtype=float)
    valid_target = np.isfinite(target_x) & np.isfinite(target_y) & (target_x > 0) & (target_y > 0)
    valid_ref = np.isfinite(ref_x) & np.isfinite(ref_y) & (ref_x > 0) & (ref_y > 0)
    if np.count_nonzero(valid_ref) < 2:
        return matches
    rx = ref_x[valid_ref]
    ry = ref_y[valid_ref]
    for idx, (tx, ty) in enumerate(zip(target_x, target_y)):
        if not valid_target[idx]:
            continue
        log_ty = np.log(ty)
        for j in range(len(rx) - 1):
            y0, y1 = ry[j], ry[j + 1]
            if y0 <= 0 or y1 <= 0:
                continue
            log_y0 = np.log(y0)
            log_y1 = np.log(y1)
            if (log_ty - log_y0) * (log_ty - log_y1) <= 0:
                log_x0 = np.log(rx[j])
                log_x1 = np.log(rx[j + 1])
                t = (log_ty - log_y0) / (log_y1 - log_y0) if log_y1 != log_y0 else 0.0
                log_match = log_x0 + t * (log_x1 - log_x0)
                matches[idx] = np.exp(log_match)
                break
    return matches


def make_subdir(base: Path, name: str) -> Path:
    """Create and return a subdirectory under base."""
    path = base / name
    path.mkdir(exist_ok=True)
    return path


def build_anisotropic_masks(n_theta_bins: int, n_phi_bins: int, theta_wedge_bins: int, phi_wedge_bins: int):
    """Return masks for L, perp, xi, and lambda bins.

    With explicit angular bins, we take only the first (≈0°) and last (≈90°) bins.
    """
    theta_L_idx = np.array([0])
    theta_perp_idx = np.array([n_theta_bins - 1])
    phi_xi_idx = np.array([0])
    phi_lambda_idx = np.array([n_phi_bins - 1])

    L_mask = np.zeros((n_theta_bins, n_phi_bins), dtype=bool)
    perp_mask = np.zeros_like(L_mask)
    xi_mask = np.zeros_like(L_mask)
    lambda_mask = np.zeros_like(L_mask)

    L_mask[theta_L_idx, :] = True
    perp_mask[theta_perp_idx, :] = True
    xi_mask[np.ix_(theta_perp_idx, phi_xi_idx)] = True
    lambda_mask[np.ix_(theta_perp_idx, phi_lambda_idx)] = True
    return L_mask, perp_mask, xi_mask, lambda_mask


def angular_average(S2_slice: np.ndarray, N_slice: np.ndarray, mask: np.ndarray) -> float:
    """Weighted average of S2 over theta/phi using pair counts as weights."""
    weights = N_slice * mask
    w_sum = weights.sum()
    if w_sum <= 0:
        return np.nan
    return float((S2_slice * weights).sum() / w_sum)


def compute_anisotropic_s2(hist_mag: np.ndarray,
                           sf_channel_bin_edges,
                           theta_bin_edges: np.ndarray,
                           phi_bin_edges: np.ndarray,
                           theta_wedge_bins: int,
                           phi_wedge_bins: int):
    """Compute S2(ell, theta, phi) and directional averages for each channel."""
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    L_mask, perp_mask, xi_mask, lambda_mask = build_anisotropic_masks(
        n_theta_bins, n_phi_bins, theta_wedge_bins, phi_wedge_bins
    )

    results = {}

    for channel_idx in range(hist_mag.shape[0]):
        counts_Q = hist_mag[channel_idx]  # shape (ell, theta, phi, dQ)

        bin_edges = sf_channel_bin_edges[channel_idx]
        dQ_centers = geometric_centers(bin_edges)
        dQ2 = dQ_centers ** 2

        N_pairs = counts_Q.sum(axis=-1)  # (ell, theta, phi)
        numerator = (counts_Q * dQ2[None, None, None, :]).sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            S2 = numerator / N_pairs
        S2 = np.where(N_pairs > 0, S2, np.nan)

        results[channel_idx] = {
            "S2": S2,
            "N_pairs": N_pairs,
            "L": np.array([angular_average(S2[i], N_pairs[i], L_mask) for i in range(S2.shape[0])]),
            "perp": np.array([angular_average(S2[i], N_pairs[i], perp_mask) for i in range(S2.shape[0])]),
            "xi": np.array([angular_average(S2[i], N_pairs[i], xi_mask) for i in range(S2.shape[0])]),
            "lambda": np.array([angular_average(S2[i], N_pairs[i], lambda_mask) for i in range(S2.shape[0])]),
            "iso": np.array([angular_average(S2[i], N_pairs[i], np.ones_like(L_mask, dtype=bool))
                            for i in range(S2.shape[0])]),
            "dQ_centers": dQ_centers,
        }
    return results


def _mean_product_from_hist(counts_theta_phi_prod: np.ndarray, product_centers: np.ndarray, mask: np.ndarray) -> float:
    """Weighted mean over product centers with theta/phi mask."""
    masked = counts_theta_phi_prod * mask[..., None]
    weights = masked.sum(axis=(0, 1))
    total = weights.sum()
    if total <= 0:
        return np.nan
    return float(np.average(product_centers, weights=weights))


def compute_anisotropic_alignment(hist_other_angle: np.ndarray,
                                  alignment_ratio_sum: np.ndarray | None,
                                  alignment_ratio_count: np.ndarray | None,
                                  product_centers: np.ndarray,
                                  other_channels,
                                  theta_bin_edges: np.ndarray,
                                  phi_bin_edges: np.ndarray,
                                  theta_wedge_bins: int,
                                  phi_wedge_bins: int,
                                  alignment_ratio_pairs=None):
    """Compute ~θ and θ over anisotropic wedges (L, perp, xi, lambda, iso)."""
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    L_mask, perp_mask, xi_mask, lambda_mask = build_anisotropic_masks(
        n_theta_bins, n_phi_bins, theta_wedge_bins, phi_wedge_bins
    )
    masks = {
        "iso": np.ones((n_theta_bins, n_phi_bins), dtype=bool),
        "L": L_mask,
        "perp": perp_mask,
        "xi": xi_mask,
        "lambda": lambda_mask,
    }

    ratio_map = {}
    if alignment_ratio_pairs is not None:
        ratio_map = {cross: idx for idx, (cross, _) in enumerate(alignment_ratio_pairs)}

    results = {}
    for cross_name, mag_name in alignment_ratio_pairs or []:
        if cross_name not in other_channels or mag_name not in other_channels:
            continue
        cross_idx = list(other_channels).index(cross_name)
        mag_idx = list(other_channels).index(mag_name)

        tilde_stats = {k: np.full(hist_other_angle.shape[1], np.nan, dtype=float) for k in masks}
        theta_stats = {k: np.full(hist_other_angle.shape[1], np.nan, dtype=float) for k in masks}

        ratio_idx = ratio_map.get(cross_name, None)

        for ell_i in range(hist_other_angle.shape[1]):
            cross_counts = hist_other_angle[cross_idx, ell_i]
            mag_counts = hist_other_angle[mag_idx, ell_i]

            for name, mask in masks.items():
                mean_cross = _mean_product_from_hist(cross_counts, product_centers, mask)
                mean_mag = _mean_product_from_hist(mag_counts, product_centers, mask)
                if np.isfinite(mean_cross) and np.isfinite(mean_mag) and mean_mag > 0:
                    tilde_stats[name][ell_i] = mean_cross / mean_mag

                if ratio_idx is not None and alignment_ratio_sum is not None and alignment_ratio_count is not None:
                    rs = alignment_ratio_sum[ratio_idx, ell_i]
                    rc = alignment_ratio_count[ratio_idx, ell_i]
                    masked_rs = np.where(mask, rs, 0.0)
                    masked_rc = np.where(mask, rc, 0)
                    total_c = masked_rc.sum()
                    if total_c > 0:
                        theta_stats[name][ell_i] = masked_rs.sum() / total_c

        results[cross_name] = {"tilde": tilde_stats, "theta": theta_stats}
    return results


def plot_anisotropic_s2(ell_centers: np.ndarray,
                        ell_bin_edges: np.ndarray,
                        channel_name: str,
                        channel_label: str,
                        s2_stats: dict,
                        output_dir: Path,
                        fmt: str,
                        dpi: int):
    """Generate anisotropic plots with matching-scale panels."""
    s2_iso = s2_stats["iso"]
    s2_L = s2_stats["L"]
    s2_perp = s2_stats["perp"]
    s2_xi = s2_stats["xi"]
    s2_lambda = s2_stats["lambda"]
    ell_max = ell_bin_edges[-1]
    safe_name = channel_name.replace("_", "").lower()
    ch_plain = channel_label.replace("$", "")
    y_label = rf"$S_{{2}}({ch_plain})$"

    def _loglog_safe(ax, x, y, *args, **kwargs) -> bool:
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0) & (x > 0)
        if np.any(mask):
            ax.loglog(x[mask], y[mask], *args, **kwargs)
            return True
        return False

    def find_matching_scales(target_x, target_y, ref_x, ref_y):
        """For each target_y, find ref_x where ref_y == target_y (log-space interpolation)."""
        matches = np.full_like(target_x, np.nan, dtype=float)
        valid_target = np.isfinite(target_x) & np.isfinite(target_y) & (target_x > 0) & (target_y > 0)
        valid_ref = np.isfinite(ref_x) & np.isfinite(ref_y) & (ref_x > 0) & (ref_y > 0)
        if np.count_nonzero(valid_ref) < 2:
            return matches
        rx = ref_x[valid_ref]
        ry = ref_y[valid_ref]
        for idx, (tx, ty) in enumerate(zip(target_x, target_y)):
            if not valid_target[idx]:
                continue
            log_ty = np.log(ty)
            for j in range(len(rx) - 1):
                y0, y1 = ry[j], ry[j + 1]
                if y0 <= 0 or y1 <= 0:
                    continue
                log_y0 = np.log(y0)
                log_y1 = np.log(y1)
                if (log_ty - log_y0) * (log_ty - log_y1) <= 0:
                    log_x0 = np.log(rx[j])
                    log_x1 = np.log(rx[j + 1])
                    t = (log_ty - log_y0) / (log_y1 - log_y0) if log_y1 != log_y0 else 0.0
                    log_match = log_x0 + t * (log_x1 - log_x0)
                    matches[idx] = np.exp(log_match)
                    break
        return matches

    def choose_best_powerlaw_fit(x_vals: np.ndarray, y_vals: np.ndarray):
        """Return mask, slope, intercept for best 8× window in [32, ell_max/2]."""
        mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
        if not np.any(mask):
            return None, None, None, None, None
        max_end = ell_max / 2.0
        # Enforce full 8× dynamic range: start * 8 <= max_end
        start_candidates = np.unique(x_vals[mask & (x_vals >= 32.0) & (x_vals * 8.0 <= max_end)])
        start_candidates = np.sort(start_candidates)
        best = None
        for start in start_candidates:
            end_limit = start * 8.0
            fit_mask = mask & (x_vals >= start) & (x_vals <= end_limit)
            if np.count_nonzero(fit_mask) < 3:
                continue
            log_x = np.log10(x_vals[fit_mask])
            log_y = np.log10(y_vals[fit_mask])
            m, b = np.polyfit(log_x, log_y, 1)
            residual = log_y - (m * log_x + b)
            chi2 = np.sum(residual ** 2)
            end_actual = x_vals[fit_mask].max()
            if best is None or chi2 < best[0] - 1e-12 or (
                np.isclose(chi2, best[0]) and (start < best[3] - 1e-12 or end_actual > best[4])
            ):
                best = (chi2, m, b, start, end_actual, fit_mask)
        if best is None:
            return None, None, None, None, None
        return best[5], best[1], best[2], best[3], best[4]

    def plot_match(ax, x_vals, y_vals, base_label, color):
        """Plot matching curve and best-fit power law over the chosen 8× window."""
        mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
        if not np.any(mask):
            return None
        label = base_label
        base_math = base_label[1:-1] if base_label.startswith("$") and base_label.endswith("$") else base_label
        fit_mask, m, b, start_fit, end_fit = choose_best_powerlaw_fit(x_vals, y_vals)
        if fit_mask is not None:
            x_fit = np.logspace(np.log10(start_fit), np.log10(end_fit), 100)
            y_fit = 10 ** (b + m * np.log10(x_fit))
            ax.loglog(
                x_fit,
                y_fit,
                linestyle="--",
                color=color,
                linewidth=3.0,
                alpha=0.5,
                label=None,
            )
            label = rf"${base_math} \propto \ell_\parallel^{{{m:.2f}}}$"
        ax.loglog(x_vals[mask], y_vals[mask], color=color, lw=2.0, label=label)
        return mask

    def plot_zeta(ax, x_vals, y_vals, color, label):
        """Plot local log–log slope using a wider stencil for smoothness."""
        slopes = compute_local_powerlaw_slopes(x_vals, y_vals, window=16)
        mask = np.isfinite(x_vals) & np.isfinite(slopes) & (x_vals > 0)
        if not np.any(mask):
            return
        ax.plot(x_vals[mask], slopes[mask], color=color, lw=1.6, label=label)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)

    # Theta anisotropy with matching/zeta panels
    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.25)
    ax_top = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.0)
    ax_zeta = fig.add_subplot(right_gs[0, 0])
    ax_right = fig.add_subplot(right_gs[1, 0], sharex=ax_zeta)
    _loglog_safe(ax_top, ell_centers, s2_iso, label=r"isotropic ($\ell$)", color="k", lw=2)
    _loglog_safe(ax_top, ell_centers, s2_L, label=r"$\ell_\parallel$", color="tab:blue")
    _loglog_safe(ax_top, ell_centers, s2_perp, label=r"$\ell_\perp$", color="tab:orange")
    ax_top.set_xlabel(r"$\ell$")
    ax_top.set_ylabel(y_label)
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend()
    ell_match_perp = find_matching_scales(ell_centers, s2_L, ell_centers, s2_perp)
    plot_zeta(ax_zeta, ell_centers, ell_match_perp, "tab:orange", r"$\ell_\perp$")
    plot_match(ax_right, ell_centers, ell_match_perp, r"$\ell_\perp$", "tab:orange")
    ax_zeta.set_ylabel(r"$\zeta$")
    ax_zeta.tick_params(labelbottom=False)
    ax_right.set_xlabel(r"$\ell_\parallel$")
    ax_right.set_ylabel(r"$\ell_\perp$")
    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.grid(True, which="both", alpha=0.3)
    ax_right.legend()

    fname = output_dir / f"{safe_name}_S2_theta_anisotropy.{fmt}"
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {fname}")

    # Xi/Lambda anisotropy with matching/zeta panels
    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.25)
    ax_top2 = fig.add_subplot(gs[0, 0])
    right_gs2 = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.0)
    ax_zeta2 = fig.add_subplot(right_gs2[0, 0])
    ax_right2 = fig.add_subplot(right_gs2[1, 0], sharex=ax_zeta2)
    _loglog_safe(ax_top2, ell_centers, s2_L, label=r"$\ell_\parallel$", color="tab:blue")
    _loglog_safe(ax_top2, ell_centers, s2_xi, label=r"$\xi$", color="tab:green")
    _loglog_safe(ax_top2, ell_centers, s2_lambda, label=r"$\lambda$", color="tab:red")
    ax_top2.set_xlabel(r"$\ell$")
    ax_top2.set_ylabel(y_label)
    ax_top2.grid(True, which="both", alpha=0.3)
    ax_top2.legend()

    ell_match_xi = find_matching_scales(ell_centers, s2_L, ell_centers, s2_xi)
    ell_match_lambda = find_matching_scales(ell_centers, s2_L, ell_centers, s2_lambda)
    plot_zeta(ax_zeta2, ell_centers, ell_match_xi, "tab:green", r"$\xi$")
    plot_zeta(ax_zeta2, ell_centers, ell_match_lambda, "tab:red", r"$\lambda$")
    plot_match(ax_right2, ell_centers, ell_match_xi, r"$\xi$", "tab:green")
    plot_match(ax_right2, ell_centers, ell_match_lambda, r"$\lambda$", "tab:red")
    ax_zeta2.set_ylabel(r"$\zeta$")
    ax_zeta2.tick_params(labelbottom=False)
    ax_right2.set_xlabel(r"$\ell_\parallel$")
    ax_right2.set_ylabel(r"$\ell$")
    ax_right2.set_xscale("log")
    ax_right2.set_yscale("log")
    ax_right2.grid(True, which="both", alpha=0.3)
    ax_right2.legend()

    fname = output_dir / f"{safe_name}_S2_xi_lambda_anisotropy.{fmt}"
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {fname}")


def estimate_taylor_scale(hist_mag: np.ndarray, mag_channels: np.ndarray,
                          ell_centers: np.ndarray, sf_channel_bin_edges) -> float | None:
    """Estimate λ_T from the velocity second-order structure function."""
    names = list(mag_channels)
    if 'D_V' not in names:
        return None
    dv_idx = names.index('D_V')
    sf_edges = sf_channel_bin_edges[dv_idx]
    sf_centers = geometric_centers(sf_edges)
    hist_2d = hist_mag[dv_idx].sum(axis=(1, 2))
    totals = hist_2d.sum(axis=1)
    s2 = np.full_like(totals, np.nan, dtype=float)
    for i, (row, total) in enumerate(zip(hist_2d, totals)):
        if total <= 0:
            continue
        s2[i] = np.sum((sf_centers ** 2) * row) / total

    valid = np.isfinite(s2) & (s2 > 0)
    if valid.sum() < 4:
        return None

    tail = s2[valid][-5:] if valid.sum() >= 5 else s2[valid]
    u_rms2 = 0.5 * np.nanmedian(tail)
    if not np.isfinite(u_rms2) or u_rms2 <= 0:
        return None

    def model(r, lam):
        return 2.0 * u_rms2 * (1.0 - np.exp(-(r * r) / (lam * lam)))

    fit_mask = valid & (ell_centers <= ell_centers.max() / 3.0)
    try:
        popt, _ = curve_fit(model, ell_centers[fit_mask], s2[fit_mask],
                            p0=[ell_centers[fit_mask][np.argmax(s2[fit_mask])]],
                            bounds=(0.0, np.inf))
        lam_t = float(popt[0])
    except Exception:
        lam_t = np.nan

    if not np.isfinite(lam_t):
        first_idx = np.where(valid)[0][0]
        lam_t = ell_centers[first_idx] * np.sqrt(2.0 * u_rms2 / max(s2[first_idx], 1e-30))

    # Guardrails: Taylor scale should not exceed L/2 or be smaller than minimum ell
    lam_t = float(np.clip(lam_t, ell_centers.min(), ell_centers.max() / 2.0))

    return lam_t


def main():
    parser = argparse.ArgumentParser(description="Plot structure functions")
    parser.add_argument("input_file", type=str,
                        help="Path to sf_results file (e.g., sf_results_all_slices.npz)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as input file)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Output format for plots")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for raster formats")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    parser.add_argument("--plot_qp_slopes", action="store_true",
                        help="Also generate increment slope plots (default: off)")
    parser.add_argument("--theta_wedge_bins", type=int, default=3,
                        help="Number of theta bins in each wedge for anisotropic S2 (default: 3)")
    parser.add_argument("--phi_wedge_bins", type=int, default=3,
                        help="Number of phi bins in each wedge for anisotropic S2 (default: 3)")

    args = parser.parse_args()

    apply_house_style()

    # Load data
    print(f"Loading data from {args.input_file}...")
    data = np.load(args.input_file, allow_pickle=True)
    metadata = dict(data['metadata'].item()) if 'metadata' in data else {}

    hist = data['hist']
    channels = list(data['channels'])
    ell_bin_edges = data['ell_bin_edges']
    theta_bin_edges = data['theta_bin_edges']
    phi_bin_edges = data['phi_bin_edges']

    delta_bin_edges = data.get('delta_bin_edges', None)
    if delta_bin_edges is None:
        if 'log_delta_bin_edges_min' in metadata and 'log_delta_bin_edges_max' in metadata:
            delta_bin_edges = []
            for lo, hi in zip(metadata['log_delta_bin_edges_min'], metadata['log_delta_bin_edges_max']):
                delta_bin_edges.append(np.logspace(lo, hi, metadata['N_delta_bin_edges']))
        else:
            raise KeyError("delta_bin_edges missing and metadata reconstruction unavailable.")
    # Force numeric dtype to avoid object arrays leaking from saved files
    delta_bin_edges = [np.asarray(edges, dtype=float) for edges in delta_bin_edges]

    # Order channels according to the Enum definition and slice hist/edges accordingly
    channel_index = {name: idx for idx, name in enumerate(channels)}
    mag_channels = [name for name in [ch.name for ch in Channel] if name in channel_index]
    hist_mag = np.stack([hist[channel_index[name]] for name in mag_channels], axis=0)
    sf_channel_bin_edges = [delta_bin_edges[channel_index[name]] for name in mag_channels]

    # Compute bin centers
    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])  # Arithmetic mean for ell
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])  # Arithmetic mean (linear bins)
    phi_centers = 0.5 * (phi_bin_edges[:-1] + phi_bin_edges[1:])  # Arithmetic mean (linear bins)
    sf_bin_edges = sf_channel_bin_edges[0]
    sf_centers = np.sqrt(sf_bin_edges[:-1] * sf_bin_edges[1:])  # Geometric mean for SF (logarithmic bins)
    taylor_scale = estimate_taylor_scale(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges)

    # Setup output directory
    if args.output_dir is None:
        output_root = Path(args.input_file).parent
    else:
        output_root = Path(args.output_dir)
    output_root.mkdir(exist_ok=True)

    # Organized subdirectories
    subdirs = {
        "mean_sf": make_subdir(output_root, "mean_structure_functions"),
        "individual_raw": make_subdir(output_root, "individual_2d_raw"),
        "individual_kde": make_subdir(output_root, "individual_2d_kde"),
        "angular": make_subdir(output_root, "angular_distributions"),
        "anisotropic": make_subdir(output_root, "anisotropic_S2"),
        "alignment": make_subdir(output_root, "alignment_angles"),
    }

    base_name = Path(args.input_file).stem

    print(f"Creating plots in {output_root}...")

    # 1. Plot mean structure functions vs ell for key channels
    plot_mean_structure_functions(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                                  subdirs["mean_sf"], base_name, args.format, args.dpi)

    # 2. Plot individual 2D histograms with power law fits (raw + KDE variants)
    plot_individual_2d_histograms_with_fits(
        hist_mag,
        mag_channels,
        ell_centers,
        ell_bin_edges,
        sf_channel_bin_edges,
        taylor_scale,
        args.plot_qp_slopes,
        subdirs["individual_raw"],
        subdirs["individual_kde"],
        base_name,
        args.format,
        args.dpi,
    )

    # 3. Plot angular distributions (2D histogram of D_V vs ell and theta)
    plot_angular_distributions(
        hist_mag,
        mag_channels,
        ell_centers,
        ell_bin_edges,
        theta_centers,
        theta_bin_edges,
        phi_centers,
        phi_bin_edges,
        subdirs["angular"],
        base_name,
        args.format,
        args.dpi,
    )

    # 4. Anisotropic S2 directional cuts
    anisotropic_results = compute_anisotropic_s2(
        hist_mag,
        sf_channel_bin_edges,
        theta_bin_edges,
        phi_bin_edges,
        args.theta_wedge_bins,
        args.phi_wedge_bins,
    )
    anisotropic_npz = {
        "ell_centers": ell_centers,
        "ell_bin_edges": ell_bin_edges,
        "theta_bin_edges": theta_bin_edges,
        "phi_bin_edges": phi_bin_edges,
        "theta_wedge_bins": args.theta_wedge_bins,
        "phi_wedge_bins": args.phi_wedge_bins,
    }
    for channel_idx, channel_name in enumerate(mag_channels):
        s2_stats = anisotropic_results[channel_idx]
        anisotropic_npz[f"{channel_name}_iso"] = s2_stats["iso"]
        anisotropic_npz[f"{channel_name}_L"] = s2_stats["L"]
        anisotropic_npz[f"{channel_name}_perp"] = s2_stats["perp"]
        anisotropic_npz[f"{channel_name}_xi"] = s2_stats["xi"]
        anisotropic_npz[f"{channel_name}_lambda"] = s2_stats["lambda"]
        if channel_name not in SKIP_PLOT_CHANNELS:
            plot_anisotropic_s2(
                ell_centers,
                ell_bin_edges,
                channel_name,
                get_channel_label(channel_name),
                s2_stats,
                subdirs["anisotropic"],
                args.format,
                args.dpi,
            )
    np.savez(subdirs["anisotropic"] / f"{base_name}_anisotropic_S2.npz", **anisotropic_npz)
    print(f"  Saved anisotropic S2 master: {subdirs['anisotropic'] / (base_name + '_anisotropic_S2.npz')}")

    # 5. Alignment angles (perpendicular cross/product ratios) in multiple projections
    plot_alignment_angles_new(
        hist,
        channels,
        delta_bin_edges,
        ell_bin_edges,
        theta_bin_edges,
        phi_bin_edges,
        subdirs["alignment"],
        base_name,
        args.format,
        args.dpi,
    )

    print(f"\nPlots saved to {output_root}")

    if args.show:
        plt.show()

    return 0


def plot_mean_structure_functions(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                                   output_dir, base_name, fmt, dpi):
    """Plot mean structure functions vs ell for all channels with power law fits."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors for different channels
    colors = plt.cm.tab20(np.linspace(0, 1, len(mag_channels)))

    # Fit range
    ell_min_fit = 32
    ell_max_fit = ell_centers.max() / 4

    for idx, channel_name in enumerate(mag_channels):
        channel_idx = idx

        # Get channel-specific bin centers using geometric mean for logarithmic bins
        bin_edges = sf_channel_bin_edges[channel_idx]
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean

        # Sum over angles to get total histogram for this channel
        hist_ell_sf = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi

        # Compute mean
        mean_sf = np.zeros(len(ell_centers))

        for i in range(len(ell_centers)):
            if hist_ell_sf[i].sum() > 0:
                # Compute weighted mean using channel-specific bin centers
                mean_sf[i] = np.average(bin_centers, weights=hist_ell_sf[i])

        # Plot only non-zero values
        mask = mean_sf > 0
        if np.any(mask):
            # Fit power law
            fit_mask = mask & (ell_centers >= ell_min_fit) & (ell_centers <= ell_max_fit)
            if np.sum(fit_mask) > 2:
                # Perform linear fit in log space
                log_ell_fit = np.log10(ell_centers[fit_mask])
                log_sf_fit = np.log10(mean_sf[fit_mask])

                # Linear regression
                coeffs = np.polyfit(log_ell_fit, log_sf_fit, 1)
                slope = coeffs[0]

                # Create label with power law
                label = f'{get_channel_label(channel_name)} $\propto \ell^{{{slope:.2f}}}$'
            else:
                label = get_channel_label(channel_name)

            # Plot data
            ax.plot(ell_centers[mask], mean_sf[mask], 'o-',
                   color=colors[idx], markersize=4, linewidth=1.5, label=label)

    # Add shaded region for fit range
    ax.axvspan(ell_min_fit, ell_max_fit, alpha=0.1, color='gray')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Mean Structure Function')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_title('Mean Structure Functions for All Channels')

    plt.tight_layout()
    filename = output_dir / f"{base_name}_mean_structure_functions.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")

    # Save 2nd/3rd-order structure functions for reuse
    s2 = np.full((len(mag_channels), len(ell_centers)), np.nan, dtype=float)
    s3 = np.full_like(s2, np.nan, dtype=float)
    for idx, channel_name in enumerate(mag_channels):
        bin_edges = sf_channel_bin_edges[idx]
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        hist_2d = hist_mag[idx].sum(axis=(1, 2))  # sum over theta, phi
        totals = hist_2d.sum(axis=1)
        with np.errstate(invalid="ignore"):
            s2[idx] = np.where(
                totals > 0,
                (hist_2d * (bin_centers ** 2)).sum(axis=1) / totals,
                np.nan,
            )
            s3[idx] = np.where(
                totals > 0,
                (hist_2d * (bin_centers ** 3)).sum(axis=1) / totals,
                np.nan,
            )
    np.savez(
        output_dir / f"{base_name}_structure_functions_2nd.npz",
        ell_centers=ell_centers,
        channels=np.array(mag_channels, dtype=object),
        S2=s2,
    )
    np.savez(
        output_dir / f"{base_name}_structure_functions_3rd.npz",
        ell_centers=ell_centers,
        channels=np.array(mag_channels, dtype=object),
        S3=s3,
    )


def plot_2d_histograms(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                       output_dir, base_name, fmt, dpi):
    """Plot 2D histograms of SF vs ell for selected channels."""
    # Select channels to plot - use actual channel names
    channels_to_plot = ['D_V', 'D_B', 'D_RHO', 'D_ZPLUS']

    n_channels = len(channels_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, channel_name in enumerate(channels_to_plot):
        if channel_name not in mag_channels:
            continue
        if idx >= 4:
            break

        ax = axes[idx]
        channel_idx = list(mag_channels).index(channel_name)

        # Get channel-specific bin centers using geometric mean for logarithmic bins
        bin_edges = sf_channel_bin_edges[channel_idx]
        sf_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean

        # Sum over angles
        hist_2d = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi

        # Create 2D plot
        ell_edges = np.concatenate([[ell_centers[0] * 0.9],
                                    0.5 * (ell_centers[:-1] + ell_centers[1:]),
                                    [ell_centers[-1] * 1.1]])
        sf_edges = np.concatenate([[sf_centers[0] * 0.9],
                                   0.5 * (sf_centers[:-1] + sf_centers[1:]),
                                   [sf_centers[-1] * 1.1]])

        # Use logarithmic normalization
        pcm = ax.pcolormesh(ell_edges, sf_edges, hist_2d.T,
                            norm=colors.LogNorm(vmin=1, vmax=hist_2d.max()),
                            cmap='viridis', shading='flat')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(get_channel_label(channel_name))
        ax.set_title(f'2D Histogram: {get_channel_label(channel_name)} vs $\ell$')

        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, label='Counts')

    plt.tight_layout()
    filename = output_dir / f"{base_name}_2d_histograms.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_2d_histograms_with_channel_bins(hist_mag, mag_channels, ell_centers, sf_channel_bin_edges,
                                          output_dir, base_name, fmt, dpi):
    """Plot normalized 2D histograms for all channels with statistical moments."""
    fig, axes = plt.subplots(6, 2, figsize=(12, 20))
    axes = axes.flatten()

    for channel_idx, channel_name in enumerate(mag_channels):
        if channel_idx >= len(axes) - 1:  # Skip if we run out of axes
            break

        ax = axes[channel_idx]

        # Get channel-specific bin edges and centers
        bin_edges = sf_channel_bin_edges[channel_idx]
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for logarithmic bins

        # Sum over angles to get 2D histogram
        hist_2d = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi

        # Normalize each ell bin by total counts in that ell
        hist_2d_norm = hist_2d.copy().astype(float)

        # Calculate statistical moments for each ell
        median_sf = np.zeros(len(ell_centers))
        mean_sf = np.zeros(len(ell_centers))
        second_moment = np.zeros(len(ell_centers))

        for i in range(len(ell_centers)):
            total_counts = hist_2d[i].sum()
            if total_counts > 0:
                hist_2d_norm[i] = hist_2d[i] / total_counts

                # Calculate median (cumulative sum approach)
                cumsum = np.cumsum(hist_2d[i])
                median_idx = np.searchsorted(cumsum, 0.5 * total_counts)
                if median_idx < len(bin_centers):
                    median_sf[i] = bin_centers[median_idx]

                # Calculate mean (first moment)
                mean_sf[i] = np.average(bin_centers, weights=hist_2d[i])

                # Calculate second moment
                second_moment[i] = np.average(bin_centers**2, weights=hist_2d[i])

        # Create meshgrid for plotting
        ell_mesh, sf_mesh = np.meshgrid(ell_centers, bin_centers)

        # Plot normalized histogram
        pcm = ax.pcolormesh(ell_mesh, sf_mesh, hist_2d_norm.T,
                            norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')

        # Overplot statistical moments
        mask = mean_sf > 0
        if np.any(mask):
            ax.plot(ell_centers[mask], median_sf[mask], 'w-', linewidth=2, label='Median')
            ax.plot(ell_centers[mask], mean_sf[mask], 'r-', linewidth=2, label='SF_1')
            ax.plot(ell_centers[mask], np.sqrt(second_moment[mask]), 'y-', linewidth=2, label='SF_2')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$' if channel_idx >= 10 else '')
        ax.set_ylabel(get_channel_label(channel_name))
        ax.set_title(get_channel_label(channel_name), fontsize=10)

        # Add legend only to first plot
        if channel_idx == 0:
            ax.legend(loc='upper left', fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, label='P' if channel_idx % 2 == 0 else '')
        cbar.ax.tick_params(labelsize=8)

    # Hide the last empty axis
    if len(mag_channels) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    filename = output_dir / f"{base_name}_2d_histograms_normalized.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_individual_2d_histograms_with_fits(hist_mag, mag_channels, ell_centers, ell_bin_edges,
                                            sf_channel_bin_edges, taylor_scale, plot_qp_slopes,
                                            output_dir_raw: Path, output_dir_kde: Path,
                                            base_name, fmt, dpi):
    """Create individual 2D histogram plots with SF overlays, slopes, and qP profiles."""

    moment_orders = DEFAULT_MOMENT_ORDERS
    order_colors = ORDER_CMAP(np.linspace(0, 1, len(moment_orders)))
    ell_norm = ell_centers / ell_bin_edges[-1]

    for channel_idx, channel_name in enumerate(mag_channels):
        if channel_name in SKIP_PLOT_CHANNELS:
            continue
        bin_edges = sf_channel_bin_edges[channel_idx]
        bin_centers = geometric_centers(bin_edges)
        hist_2d = hist_mag[channel_idx].sum(axis=(1, 2))  # Sum over theta, phi
        totals = hist_2d.sum(axis=1)
        hist_2d_norm = np.where(totals[:, None] > 0, hist_2d / totals[:, None], 0.0)

        medians = compute_median_from_hist(hist_2d, bin_centers)
        sf_orders = compute_sf_orders(hist_2d, bin_centers, orders=moment_orders)
        sf_roots = {
            p: np.where(sf_orders[p] > 0, sf_orders[p] ** (1.0 / p), np.nan)
            for p in moment_orders
        }
        slopes = {
            p: compute_local_powerlaw_slopes(ell_centers, sf_roots[p])
            for p in moment_orders
        }
        skew, kurt = compute_skew_kurtosis(hist_2d, bin_centers)

        qpdf = compute_qpdf(hist_2d, bin_edges, bin_centers)
        qpdf_kde = kde_smooth_qp(hist_2d, bin_centers, bin_edges)
        qpdf_slopes = compute_qpdf_slopes(qpdf_kde, bin_centers)

        qpdf = compute_qpdf(hist_2d, bin_edges, bin_centers)
        qpdf_kde = kde_smooth_qp(hist_2d, bin_centers, bin_edges)

        ell_centers_mesh, sf_centers_mesh = np.meshgrid(ell_centers, bin_centers, indexing="ij")
        safe_name = channel_name.replace('_', '').lower()

        def stats_from_qpdf(qpdf_data: np.ndarray):
            """Compute median, S_p^{1/p}, slopes from q*P(q|ell) (raw or KDE)."""
            pdf_linear = np.where(qpdf_data > 0, qpdf_data / bin_centers[None, :], 0.0)
            widths = np.diff(bin_edges)[None, :]  # shape (1, N_bins)
            weights = pdf_linear * widths
            totals = weights.sum(axis=1)

            med = np.full(len(ell_centers), np.nan)
            sf_root_local = {p: np.full(len(ell_centers), np.nan) for p in moment_orders}

            for i, total in enumerate(totals):
                if total <= 0:
                    continue
                cdf = np.cumsum(weights[i])
                idx = np.searchsorted(cdf, 0.5 * total)
                idx = min(idx, len(bin_centers) - 1)
                med[i] = bin_centers[idx]
                for p in moment_orders:
                    sf_val = np.sum((bin_centers ** p) * weights[i]) / total
                    sf_root_local[p][i] = sf_val ** (1.0 / p) if sf_val > 0 else np.nan

            slopes_local = {
                p: compute_local_powerlaw_slopes(ell_centers, sf_root_local[p]) for p in moment_orders
            }
            return med, sf_root_local, slopes_local

        med_raw, sf_raw, slopes_raw = stats_from_qpdf(qpdf)
        med_kde, sf_kde, slopes_kde = stats_from_qpdf(qpdf_kde)

        def render_variant(data_array, suffix: str, out_dir: Path):
            data_masked = np.ma.masked_invalid(data_array)
            data_masked = np.ma.masked_less_equal(data_masked, 0)
            vmax_local = max(float(data_masked.max()) if data_masked.count() else 1e-6, 1e-6)
            levels = 10 ** np.arange(-6, np.log10(vmax_local) + 0.5, 0.5)

            fig, (ax_top, ax_bottom) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(6.5, 6.6),
                sharex=True,
                gridspec_kw={"height_ratios": [3.6, 1.3], "hspace": 0.04},
            )

            pcm_local = ax_top.contourf(
                ell_centers_mesh,
                sf_centers_mesh,
                data_masked,
                levels=levels,
                norm=colors.LogNorm(vmin=levels.min(), vmax=levels.max()),
                cmap=HIST_CMAP,
                extend="both",
            )

            ax_top.plot(
                ell_centers,
                medians_variant,
                color="white",
                lw=1.6,
                zorder=4,
                path_effects=[pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()],
            )
            for color, p in zip(order_colors, moment_orders):
                label = fr"$S_{{{p}}}^{{1/{p}}}$"
                ax_top.plot(ell_centers, sf_variant[p], color=color, lw=1.3, zorder=3)

            if taylor_scale is not None and np.isfinite(taylor_scale):
                ylim = ax_top.get_ylim()
                y0 = 10 ** (np.log10(ylim[1]) - 0.1 * (np.log10(ylim[1]) - np.log10(ylim[0])))
                ax_top.plot(
                    [taylor_scale, taylor_scale],
                    [ylim[1], y0],
                    color="0.6",
                    lw=1.0,
                    ls="--",
                    zorder=4,
                )
                ax_bottom.plot(
                    [taylor_scale, taylor_scale],
                    [1.0, 0.9],
                    color="0.6",
                    lw=1.0,
                    ls="--",
                    zorder=3,
                )

            # Set consistent x-limits with a slightly larger minimum (no data at ell=1)
            x_min = max(1.5, ell_bin_edges[0])
            x_max = ell_bin_edges[-1]
            ax_top.set_xlim(left=x_min, right=x_max)
            ax_bottom.set_xlim(left=x_min, right=x_max)

            ax_top.set_xscale("log")
            ax_top.set_yscale("log")
            ax_top.set_ylabel(get_channel_label(channel_name))
            inset = ax_top.inset_axes([0.58, 0.05, 0.4, 0.045])
            cbar = fig.colorbar(pcm_local, cax=inset, orientation="horizontal")
            ch_plain = get_channel_label(channel_name).replace("$", "")
            cbar.set_label(rf"${ch_plain}\,P({ch_plain}\mid \ell)$", labelpad=8)
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.tick_top()
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.tick_params(axis="x", labelsize=8, width=0.6, length=2, direction="in", pad=2)
            cbar.ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10, dtype=float)))
            for spine in cbar.ax.spines.values():
                spine.set_linewidth(0.6)

            for color, p in zip(order_colors, moment_orders):
                ax_bottom.plot(ell_centers, slopes_variant[p], color=color, lw=1.2)

            # Fit global slope for S3^{1/3} between 32 cells and the upper quarter of the domain
            fit_handle = None
            if 3 in moment_orders:
                fit_min = 32.0
                fit_max = ell_bin_edges[-1] / 4.0
                if fit_max > fit_min:
                    s3_vals = sf_variant[3]
                    mask_fit = (
                        np.isfinite(s3_vals)
                        & (s3_vals > 0)
                        & (ell_centers >= fit_min)
                        & (ell_centers <= fit_max)
                    )
                    if np.count_nonzero(mask_fit) >= 2:
                        log_ell = np.log10(ell_centers[mask_fit])
                        log_s3 = np.log10(s3_vals[mask_fit])
                        slope_fit, intercept_fit = np.polyfit(log_ell, log_s3, 1)
                        ell_fit_min = ell_centers[mask_fit].min()
                        ell_fit_max = ell_centers[mask_fit].max()
                        fit_handle = ax_bottom.hlines(
                            slope_fit,
                            ell_fit_min,
                            ell_fit_max,
                            colors=[order_colors[moment_orders.index(3)]],
                            linestyles="--",
                            linewidth=1.1,
                            zorder=4,
                            label=fr"$S_3$ fit ({slope_fit:.2f})",
                        )

            ax_bottom.set_xscale("log")
            ax_bottom.set_xlabel(r"$\ell$")
            ax_bottom.set_ylabel(r"$\zeta_p(\ell)$")
            ax_bottom.set_ylim(-0.1, 1.1)
            for y in (0.25, 1.0/3.0, 0.4, 0.5, 0.6, 2.0/3.0, 0.75):
                ax_bottom.hlines(
                    y,
                    x_min,
                    x_max,
                    colors="0.7",
                    linestyles=":",
                    linewidth=0.7,
                    zorder=1,
                )
            if fit_handle is not None:
                ax_bottom.legend(
                    [fit_handle],
                    [fit_handle.get_label()],
                    fontsize=7,
                    loc="lower left",
                    frameon=False,
                    handlelength=1.3,
                )

            # Inset colorbar mapping p -> color for slope curves
            order_cmap = colors.ListedColormap(order_colors)
            order_norm = colors.BoundaryNorm(
                boundaries=np.arange(len(moment_orders) + 1) - 0.5,
                ncolors=len(moment_orders),
            )
            inset_orders = ax_bottom.inset_axes([0.62, 0.82, 0.33, 0.10])
            sm_orders = plt.cm.ScalarMappable(norm=order_norm, cmap=order_cmap)
            sm_orders.set_array([])
            cbar_orders = fig.colorbar(sm_orders, cax=inset_orders, orientation="horizontal")
            cbar_orders.set_ticks([])
            cbar_orders.set_ticklabels([])
            cbar_orders.ax.xaxis.set_ticks_position("top")
            cbar_orders.ax.xaxis.set_label_position("top")
            cbar_orders.ax.minorticks_off()
            cbar_orders.ax.tick_params(
                axis="x",
                labelsize=7,
                width=0,
                length=0,  # remove tick marks; leave only labels
                direction="in",
                pad=-8,  # pull labels deeper into the bar to center vertically
            )
            # Manually place non-rotated label at left-middle of the bar
            cbar_orders.ax.text(
                -0.08,
                0.5,
                r"$p$",
                transform=cbar_orders.ax.transAxes,
                va="center",
                ha="left",
                fontsize=8,
            )
            # Manually place numeric labels at band centers
            for i, p_order in enumerate(moment_orders):
                cbar_orders.ax.text(
                    (i + 0.5) / len(moment_orders),
                    0.45,
                    str(p_order),
                    ha="center",
                    va="center",
                    fontsize=7,
                    transform=cbar_orders.ax.transAxes,
                )
            for spine in cbar_orders.ax.spines.values():
                spine.set_linewidth(0.6)

            fig.tight_layout()
            fname = out_dir / f"{safe_name}_SFp_2D_zeta{suffix}.{fmt}"
            plt.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Created: {fname}")

        # Raw and KDE variants
        # Render raw (uses raw stats) and KDE (uses smoothed stats)
        medians_variant, sf_variant, slopes_variant = med_raw, sf_raw, slopes_raw
        render_variant(qpdf, "", output_dir_raw)
        medians_variant, sf_variant, slopes_variant = med_kde, sf_kde, slopes_kde
        render_variant(qpdf_kde, "_KDE", output_dir_kde)

        profiles_path_raw = output_dir_raw / f"{safe_name}_increment_distributions.{fmt}"
        plot_qp_profiles_for_channel(
            channel_name,
            ell_norm,
            bin_centers,
            qpdf,
            profiles_path_raw,
            dpi,
            use_kde=False,
        )

        profiles_path_kde = output_dir_kde / f"{safe_name}_increment_distributions_KDE.{fmt}"
        plot_qp_profiles_for_channel(
            channel_name,
            ell_norm,
            bin_centers,
            qpdf_kde,
            profiles_path_kde,
            dpi,
            use_kde=True,
        )

        if plot_qp_slopes:
            slopes_path = output_dir_kde / f"{base_name}_{safe_name}_qp_profiles_kde_slopes.{fmt}"
            plot_qp_slopes_for_channel(
                channel_name,
                ell_norm,
                qpdf_slopes,
                slopes_path,
                dpi,
            )


def plot_qp_profiles_for_channel(channel_name: str,
                                 ell_norm: np.ndarray,
                                 bin_centers: np.ndarray,
                                 qpdf: np.ndarray,
                                 output_path: Path,
                                 dpi: int,
                                 use_kde: bool):
    """Plot qP(q) profiles (raw or KDE) without a skew/kurtosis panel."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    curves = []
    ell_colors = []
    for ell_val, profile in zip(ell_norm, qpdf):
        if not np.any(profile > 0):
            continue
        peak_idx = np.nanargmax(profile)
        if peak_idx <= 0 or peak_idx >= len(bin_centers):
            continue
        x = bin_centers / bin_centers[peak_idx]
        y = np.where(profile > 0, profile, np.nan)
        curves.append(np.column_stack([x, y]))
        ell_colors.append(ell_val)

    if curves:
        ell_colors = np.asarray(ell_colors)
        finite_colors = ell_colors[np.isfinite(ell_colors) & (ell_colors > 0)]
        if finite_colors.size:
            norm = colors.LogNorm(
                vmin=max(np.nanmin(finite_colors), 1e-6),
                vmax=np.nanmax(finite_colors),
            )
            lc = LineCollection(curves, array=ell_colors, cmap=ELL_CMAP, norm=norm)
            lc.set_linewidth(1.0)
            ax.add_collection(lc)
            sm = plt.cm.ScalarMappable(cmap=ELL_CMAP, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.01)
            cbar.set_label(r"$\ell / L$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e3)
    ch_plain = get_channel_label(channel_name).replace("$", "")
    ch_safe = ch_plain.replace("_", r"\_")
    ax.set_xlabel(rf"${ch_safe}/({ch_safe})_{{\mathrm{{peak}}}}$")
    ax.set_ylabel(rf"${ch_safe}\,P({ch_safe}\mid \ell)$")
    ax.grid(True, alpha=0.3, which="both")
    valid_y = np.concatenate([curve[:, 1][np.isfinite(curve[:, 1])] for curve in curves]) if curves else np.array([])
    if valid_y.size:
        ymin = np.nanmax([valid_y[valid_y > 0].min(), 1e-8])
        ymax = np.nanpercentile(valid_y, 99.5)
        ax.set_ylim(ymin, ymax * 1.1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {output_path}")


def plot_qp_slopes_for_channel(channel_name: str,
                               ell_norm: np.ndarray,
                               slopes: np.ndarray,
                               output_path: Path,
                               dpi: int):
    """Plot slopes of qP(q) profiles versus scale."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    mask = np.isfinite(slopes) & np.isfinite(ell_norm) & (ell_norm > 0) & (slopes > 0)
    ax.loglog(ell_norm[mask], slopes[mask], color="tab:purple", lw=1.4)
    ax.set_xlabel(r"$\ell / L$")
    ax.set_ylabel(r"$-\,\mathrm{slope}$")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(f"{get_channel_label(channel_name)} qP(q) slopes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {output_path}")


def plot_angular_distributions(hist_mag, mag_channels, ell_centers, ell_bin_edges,
                               theta_centers, theta_bin_edges,
                               phi_centers, phi_bin_edges,
                               output_dir, base_name, fmt, dpi):
    """Plot ℓ–θ and ℓ–φ distributions for δv and δB side-by-side with ℓ on the y-axis."""
    targets = []
    for name in ('D_V', 'D_B'):
        if name in mag_channels:
            targets.append((name, list(mag_channels).index(name)))
    if not targets and len(mag_channels) > 0:
        targets.append((mag_channels[0], 0))

    theta_deg_edges = np.rad2deg(theta_bin_edges)
    theta_deg_centers = np.rad2deg(theta_centers)
    phi_deg_edges = np.rad2deg(phi_bin_edges)
    phi_deg_centers = np.rad2deg(phi_centers)

    def render_ang_plot(hist_data_func, x_centers, x_edges, xlabel, file_suffix, smooth=False, sum_over_label=r"$\phi$"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, (channel_name, idx) in zip(axes, targets):
            hist_raw = hist_data_func(hist_mag[idx])
            hist_use = gaussian_filter(hist_raw, sigma=1.0) if smooth else hist_raw
            hist_plot = np.ma.masked_less_equal(hist_use, 0)
            vmax = float(hist_plot.max()) if hist_plot.count() else 1.0
            vmin = max(float(hist_plot.min()) if hist_plot.count() else 1.0, 1.0)
            log_min, log_max = np.log10(vmin), np.log10(vmax)
            log_lo = np.floor(log_min) - 0.25
            log_hi = np.ceil(log_max) + 0.25
            levels = 10 ** np.arange(log_lo, log_hi + 1e-6, 0.25) if vmax > vmin else [vmin]
            pcm = ax.contourf(
                x_centers,
                ell_centers,
                hist_plot,
                levels=levels,
                norm=colors.LogNorm(vmin=levels.min(), vmax=levels.max()),
                cmap=cmr.rainforest,
                extend="both",
            )
            ax.set_xlabel(xlabel)
            ax.set_xlim(x_edges.min(), x_edges.max())
            ax.set_yscale("log")
            cbar = fig.colorbar(pcm, ax=ax, pad=0.01, extend="both")
            cbar.set_label(f"{get_channel_label(channel_name)} counts (sum over {sum_over_label}, SF bins)")
            decade_ticks = 10 ** np.arange(np.floor(log_min), np.ceil(log_max) + 1)
            cbar.set_ticks(decade_ticks)
            cbar.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        axes[0].set_ylabel(r"$\ell$")
        fig.tight_layout()
        suffix = "_KDE" if smooth else ""
        filename = output_dir / f"dv_db_ell_{file_suffix}{suffix}.{fmt}"
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Created: {filename}")

    # ℓ–θ distributions (sum over φ and SF)
    render_ang_plot(
        hist_data_func=lambda h: np.sum(h, axis=(2, 3)),
        x_centers=theta_deg_centers,
        x_edges=theta_deg_edges,
        xlabel=r"$\theta$ (degrees)",
        file_suffix="theta_distribution",
        smooth=False,
    )
    render_ang_plot(
        hist_data_func=lambda h: np.sum(h, axis=(2, 3)),
        x_centers=theta_deg_centers,
        x_edges=theta_deg_edges,
        xlabel=r"$\theta$ (degrees)",
        file_suffix="theta_distribution",
        smooth=True,
    )

    # ℓ–φ distributions (restrict θ near 90° to emphasize ℓ_perp; sum over SF)
    theta_high_mask = (theta_deg_centers >= 75.0)
    render_ang_plot(
        hist_data_func=lambda h: np.sum(h[:, theta_high_mask, :, :], axis=(1, 3)),
        x_centers=phi_deg_centers,
        x_edges=phi_deg_edges,
        xlabel=r"$\phi$",
        file_suffix="phi_distribution_perp",
        smooth=False,
        sum_over_label=r"$\theta \approx 90^\circ$",
    )
    render_ang_plot(
        hist_data_func=lambda h: np.sum(h[:, theta_high_mask, :, :], axis=(1, 3)),
        x_centers=phi_deg_centers,
        x_edges=phi_deg_edges,
        xlabel=r"$\phi$",
        file_suffix="phi_distribution_perp",
        smooth=True,
        sum_over_label=r"$\theta \approx 90^\circ$",
    )


def plot_cross_products(hist_other, other_channels, ell_centers, product_centers,
                        output_dir, base_name, fmt, dpi, taylor_scale=None,
                        alignment_ratio_sum=None, alignment_ratio_count=None, alignment_ratio_pairs=None):
    """Plot ratios of cross-products to their corresponding MAG products with power law fits.

    By default plots the perpendicular versions; if full-vector counterparts exist,
    overlay them as dashed lines.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    edge_trim = 4
    min_fraction = 0.9
    min_counts = 200

    ratio_map = {}
    if alignment_ratio_pairs is not None and alignment_ratio_sum is not None and alignment_ratio_count is not None:
        ratio_map = {cross: idx for idx, (cross, _) in enumerate(alignment_ratio_pairs)}

    # Define cross product / MAG pairs
    ratio_pairs = [
        ('D_Vperp_CROSS_Bperp', 'D_Vperp_D_Bperp_MAG', 'D_V_CROSS_B', 'D_V_D_B_MAG'),
        ('D_Vperp_CROSS_VAperp', 'D_Vperp_D_VAperp_MAG', 'D_V_CROSS_VA', 'D_V_D_VA_MAG'),
        ('D_Vperp_CROSS_Omegaperp', 'D_Vperp_D_Omegaperp_MAG', 'D_V_CROSS_OMEGA', 'D_V_D_OMEGA_MAG'),
        ('D_Bperp_CROSS_Jperp', 'D_Bperp_D_Jperp_MAG', 'D_B_CROSS_J', 'D_B_D_J_MAG')
    ]

    for idx, (cross_name, mag_name, full_cross_name, full_mag_name) in enumerate(ratio_pairs):
        ax = axes[idx]

        if cross_name not in other_channels or mag_name not in other_channels:
            continue

        cross_idx = list(other_channels).index(cross_name)
        mag_idx = list(other_channels).index(mag_name)

        mean_cross = np.full(len(ell_centers), np.nan, dtype=float)
        mean_mag = np.full(len(ell_centers), np.nan, dtype=float)
        sin_ratio = None
        if ratio_map and cross_name in ratio_map:
            idx_ratio = ratio_map[cross_name]
            counts = alignment_ratio_count[idx_ratio]
            with np.errstate(divide="ignore", invalid="ignore"):
                sin_ratio = np.where(counts > 0, alignment_ratio_sum[idx_ratio] / counts, np.nan)

        for i in range(len(ell_centers)):
            cross_row = hist_other[cross_idx][i]
            mag_row = hist_other[mag_idx][i]
            if len(cross_row) > 2 * edge_trim:
                interior_cross = cross_row[edge_trim:-edge_trim]
                total_cross = cross_row.sum()
                if total_cross >= min_counts:
                    frac_cross = interior_cross.sum() / total_cross if total_cross > 0 else 0.0
                    if frac_cross >= min_fraction and interior_cross.sum() > 0:
                        mean_cross[i] = np.average(product_centers[edge_trim:-edge_trim], weights=interior_cross)
            if len(mag_row) > 2 * edge_trim:
                interior_mag = mag_row[edge_trim:-edge_trim]
                total_mag = mag_row.sum()
                if total_mag >= min_counts:
                    frac_mag = interior_mag.sum() / total_mag if total_mag > 0 else 0.0
                    if frac_mag >= min_fraction and interior_mag.sum() > 0:
                        mean_mag[i] = np.average(product_centers[edge_trim:-edge_trim], weights=interior_mag)

        mask = np.isfinite(mean_mag) & np.isfinite(mean_cross) & (mean_mag > 0) & (mean_cross > 0)
        if np.any(mask):
            ratio = mean_cross[mask] / mean_mag[mask]
            ell_valid = ell_centers[mask]

            # Plot ratio
            ax.plot(ell_valid, ratio, 'o-', markersize=6, label='~$\\theta$ (mean cross/product)')
            if taylor_scale is not None and np.isfinite(taylor_scale):
                ax.axvline(taylor_scale, color='0.6', lw=1.0, ls='--', label=r'$\lambda_T$')

            # Fit over the upper portion of available scales (60–90% in log space)
            ell_min_fit = max(32.0, ell_valid.min())
            ell_max_fit = min(ell_valid.max(), ell_centers.max() / 8.0)
            fit_mask = (ell_valid >= ell_min_fit) & (ell_valid <= ell_max_fit)

            if np.sum(fit_mask) > 2:  # Need at least 3 points for a good fit
                # Perform linear fit in log space
                log_ell_fit = np.log10(ell_valid[fit_mask])
                log_ratio_fit = np.log10(ratio[fit_mask])

                # Linear regression
                coeffs = np.polyfit(log_ell_fit, log_ratio_fit, 1)
                slope = coeffs[0]
                intercept = coeffs[1]

                # Create fit line
                ell_fit_range = np.logspace(np.log10(ell_min_fit), np.log10(ell_max_fit), 100)
                ratio_fit = 10**(intercept) * ell_fit_range**slope

                # Plot fit
                ax.plot(
                    ell_fit_range,
                    ratio_fit,
                    'r--',
                    linewidth=2.2,
                    alpha=0.75,
                    label=rf'Power law: $\ell^{{{slope:.2f}}}$',
                )

                # Add shaded region to show fit range
                ax.axvspan(ell_min_fit, ell_max_fit, alpha=0.1, color='gray')

        # Plot instantaneous θ if available
        if sin_ratio is not None:
            mask_theta = np.isfinite(sin_ratio) & (sin_ratio > 0)
            if np.any(mask_theta):
                ax.plot(
                    ell_centers[mask_theta],
                    sin_ratio[mask_theta],
                    linestyle="-.",
                    color="tab:purple",
                    linewidth=2.0,
                    label=r"$\theta$ (instantaneous ratio)",
                )

        # Optionally overlay full-vector version if available
        if full_cross_name in other_channels and full_mag_name in other_channels:
            cross_idx_full = list(other_channels).index(full_cross_name)
            mag_idx_full = list(other_channels).index(full_mag_name)
            mean_cross_full = np.full(len(ell_centers), np.nan, dtype=float)
            mean_mag_full = np.full(len(ell_centers), np.nan, dtype=float)
            for i in range(len(ell_centers)):
                cross_row = hist_other[cross_idx_full][i]
                mag_row = hist_other[mag_idx_full][i]
                if len(cross_row) > 2 * edge_trim:
                    interior_cross = cross_row[edge_trim:-edge_trim]
                    total_cross = cross_row.sum()
                    if total_cross >= min_counts:
                        frac_cross = interior_cross.sum() / total_cross if total_cross > 0 else 0.0
                        if frac_cross >= min_fraction and interior_cross.sum() > 0:
                            mean_cross_full[i] = np.average(product_centers[edge_trim:-edge_trim], weights=interior_cross)
                if len(mag_row) > 2 * edge_trim:
                    interior_mag = mag_row[edge_trim:-edge_trim]
                    total_mag = mag_row.sum()
                    if total_mag >= min_counts:
                        frac_mag = interior_mag.sum() / total_mag if total_mag > 0 else 0.0
                        if frac_mag >= min_fraction and interior_mag.sum() > 0:
                            mean_mag_full[i] = np.average(product_centers[edge_trim:-edge_trim], weights=interior_mag)
            mask_full = np.isfinite(mean_mag_full) & np.isfinite(mean_cross_full) & (mean_mag_full > 0) & (mean_cross_full > 0)
            if np.any(mask_full):
                ratio_full = mean_cross_full[mask_full] / mean_mag_full[mask_full]
                ell_full = ell_centers[mask_full]
                ax.plot(ell_full, ratio_full, linestyle='--', color='tab:gray', linewidth=1.5, label='Full (no ⊥)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        cross_lbl = get_channel_label(cross_name)
        mag_lbl = get_channel_label(mag_name)
        cross_plain = cross_lbl.replace("$", "").strip()
        mag_plain = mag_lbl.replace("$", "").strip()
        if r"\times" in cross_plain:
            parts = cross_plain.split(r"\times")
            a = parts[0].strip()
            b = parts[1].strip()
            theta_sub = rf"{b},{a}"
        else:
            theta_sub = cross_plain
        ax.set_ylabel(r"$\theta$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # No titles per preference

    plt.tight_layout()
    filename = output_dir / f"alignment_angles.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Created: {filename}")


def plot_alignment_comparison(
    hist_other,
    other_channels,
    ell_centers,
    product_centers,
    output_dir: Path,
    fmt: str,
    alignment_ratio_sum=None,
    alignment_ratio_count=None,
    alignment_ratio_pairs=None,
):
    """Compare theta_(Q,P)(ell) for v-B, v-omega, B-j using both ~θ (tilde) and θ (ratio)."""
    edge_trim = 4
    min_fraction = 0.9
    min_counts = 200
    ratio_map = {}
    if alignment_ratio_pairs is not None and alignment_ratio_sum is not None and alignment_ratio_count is not None:
        ratio_map = {cross: idx for idx, (cross, _) in enumerate(alignment_ratio_pairs)}

    pairs = [
        ("D_Vperp_CROSS_Bperp", "D_Vperp_D_Bperp_MAG", "v,B", "tab:blue"),
        ("D_Vperp_CROSS_Omegaperp", "D_Vperp_D_Omegaperp_MAG", "v,\\omega", "tab:green"),
        ("D_Bperp_CROSS_Jperp", "D_Bperp_D_Jperp_MAG", "B,j", "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for cross_name, mag_name, subscript, color in pairs:
        if cross_name not in other_channels or mag_name not in other_channels:
            continue
        cross_idx = list(other_channels).index(cross_name)
        mag_idx = list(other_channels).index(mag_name)

        mean_ratio = np.full(len(ell_centers), np.nan, dtype=float)
        sin_ratio = None
        if ratio_map and cross_name in ratio_map:
            idx_ratio = ratio_map[cross_name]
            counts = alignment_ratio_count[idx_ratio]
            with np.errstate(divide="ignore", invalid="ignore"):
                sin_ratio = np.where(counts > 0, alignment_ratio_sum[idx_ratio] / counts, np.nan)

        for i in range(len(ell_centers)):
            cross_row = hist_other[cross_idx][i]
            mag_row = hist_other[mag_idx][i]
            if len(cross_row) > 2 * edge_trim and len(mag_row) > 2 * edge_trim:
                interior_cross = cross_row[edge_trim:-edge_trim]
                interior_mag = mag_row[edge_trim:-edge_trim]
                total_cross = cross_row.sum()
                total_mag = mag_row.sum()
                if total_cross >= min_counts and total_mag >= min_counts:
                    frac_cross = interior_cross.sum() / total_cross if total_cross > 0 else 0.0
                    frac_mag = interior_mag.sum() / total_mag if total_mag > 0 else 0.0
                    if frac_cross >= min_fraction and frac_mag >= min_fraction:
                        mean_c = np.average(product_centers[edge_trim:-edge_trim], weights=interior_cross)
                        mean_m = np.average(product_centers[edge_trim:-edge_trim], weights=interior_mag)
                        if mean_m > 0:
                            mean_ratio[i] = mean_c / mean_m

        mask = np.isfinite(mean_ratio) & (mean_ratio > 0)
        if not np.any(mask):
            continue

        solid_label = None

        ell_min_fit = max(32.0, ell_centers[mask].min())
        ell_max_fit = min(ell_centers[mask].max(), ell_centers.max() / 8.0)
        fit_mask = (ell_centers[mask] >= ell_min_fit) & (ell_centers[mask] <= ell_max_fit)
        if np.count_nonzero(fit_mask) >= 3:
            log_ell_fit = np.log10(ell_centers[mask][fit_mask])
            log_ratio_fit = np.log10(mean_ratio[mask][fit_mask])
            slope, intercept = np.polyfit(log_ell_fit, log_ratio_fit, 1)
            ell_fit_range = np.logspace(np.log10(ell_min_fit), np.log10(ell_max_fit), 100)
            ratio_fit = 10 ** (intercept) * ell_fit_range ** slope
            ax.loglog(
                ell_fit_range,
                ratio_fit,
                linestyle="--",
                color=color,
                linewidth=2.2,
                alpha=0.75,
                label=None,
            )
            solid_label = rf"$\theta_{{{subscript}}}\propto \ell^{{{slope:.2f}}}$"

        ax.loglog(
            ell_centers[mask],
            mean_ratio[mask],
            color=color,
            lw=1.6,
            label=solid_label,
        )

        # Plot instantaneous θ (ratio) if available
        if sin_ratio is not None:
            mask_theta = np.isfinite(sin_ratio) & (sin_ratio > 0)
            if np.any(mask_theta):
                ax.loglog(
                    ell_centers[mask_theta],
                    sin_ratio[mask_theta],
                    color=color,
                    lw=1.6,
                    linestyle="-.",
                    alpha=0.9,
                    label=rf"$\theta_{{{subscript}}}$ (ratio)",
                )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\,\theta$')
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    filename = output_dir / f"alignment_angles_comparison.{fmt}"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {filename}")


def plot_alignment_anisotropy(
    ell_centers: np.ndarray,
    align_stats: dict,
    output_dir: Path,
    fmt: str,
):
    """Plot anisotropic alignment (~θ and θ) over L, perp, xi, lambda, iso for each pair."""
    wedge_order = ["iso", "L", "perp", "xi", "lambda"]
    wedge_styles = {
        "iso": ("k", "-"),
        "L": ("tab:blue", "-"),
        "perp": ("tab:red", "-"),
        "xi": ("tab:green", "-"),
        "lambda": ("tab:purple", "-"),
    }
    for pair_name, stats in align_stats.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for wedge in wedge_order:
            color, style = wedge_styles[wedge]
            tilde_vals = stats["tilde"].get(wedge, None)
            theta_vals = stats["theta"].get(wedge, None)
            if tilde_vals is not None:
                mask = np.isfinite(tilde_vals) & (tilde_vals > 0)
                if np.any(mask):
                    ax.loglog(
                        ell_centers[mask],
                        tilde_vals[mask],
                        linestyle=style,
                        color=color,
                        linewidth=1.6,
                        label=rf"{wedge} ~$\theta$",
                    )
            if theta_vals is not None:
                mask_t = np.isfinite(theta_vals) & (theta_vals > 0)
                if np.any(mask_t):
                    ax.loglog(
                        ell_centers[mask_t],
                        theta_vals[mask_t],
                        linestyle="--",
                        color=color,
                        linewidth=1.6,
                        alpha=0.9,
                        label=rf"{wedge} $\theta$",
                    )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$\theta$')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        filename = output_dir / f"alignment_anisotropy_{pair_name}.{fmt}"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Created: {filename}")


# -----------------------------------------------------------------------------
# New alignment plots from unified histograms ---------------------------------
# -----------------------------------------------------------------------------

def _mean_from_hist(hist_chan: np.ndarray, bin_edges: np.ndarray, axis_sum=(2, 3)):
    """Weighted mean of a histogram over specified axes using geometric centers."""
    centers = geometric_centers(bin_edges)
    # move last axis (delta bins) to end for broadcasting
    counts = hist_chan
    # Collapse over requested axes, leaving ell (and optionally theta)
    weights = np.sum(counts, axis=axis_sum)
    with np.errstate(invalid="ignore"):
        num = np.sum(counts * centers, axis=axis_sum)
        mean = np.where(weights > 0, num / weights, np.nan)
    return mean, weights


def plot_alignment_angles_new(
    hist: np.ndarray,
    channels: list,
    delta_bin_edges,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    output_dir: Path,
    base_name: str,
    fmt: str,
    dpi: int,
):
    """Plot θ_{x,y} vs ℓ, ℓ_parallel, ℓ_perp, ξ, λ using cross/mag channels."""
    output_dir.mkdir(exist_ok=True)
    ch_index = {name: idx for idx, name in enumerate(channels)}
    n_theta_bins = theta_bin_edges.shape[0] - 1
    n_phi_bins = phi_bin_edges.shape[0] - 1
    # Use explicit edge-based masks: first (~0°) and last (~90°) bins only.
    L_mask, perp_mask, xi_mask, lambda_mask = build_anisotropic_masks(
        n_theta_bins, n_phi_bins, theta_wedge_bins=1, phi_wedge_bins=1
    )
    pairs = [
        ("D_Vperp_CROSS_D_Bperp", "D_Vperp_D_Bperp_MAG", "D_Vperp_D_Bperp_CROSS_MAG_RATIO", "v,B", "tab:blue"),
        ("D_Bperp_CROSS_D_Jperp", "D_Bperp_D_Jperp_MAG", "D_Bperp_D_Jperp_CROSS_MAG_RATIO", "B,j", "tab:red"),
        ("D_Vperp_CROSS_D_Omegaperp", "D_Vperp_D_Omegaperp_MAG", "D_Vperp_D_Omegaperp_CROSS_MAG_RATIO", "v,\\omega", "tab:green"),
        ("D_Omegaperp_CROSS_D_Jperp", "D_Omegaperp_D_Jperp_MAG", "D_Omegaperp_D_Jperp_CROSS_MAG_RATIO", "\\omega,j", "tab:purple"),
    ]

    ell_centers = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
    theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
    phi_centers = 0.5 * (phi_bin_edges[:-1] + phi_bin_edges[1:])
    cos_theta = np.cos(theta_centers)
    sin_theta = np.sin(theta_centers)
    cos_phi = np.cos(phi_centers)
    sin_phi = np.sin(phi_centers)

    def choose_best_powerlaw_fit(x_vals: np.ndarray, y_vals: np.ndarray):
        """Return slope/intercept for best 8× window in [32, max/2] minimizing chi^2."""
        mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
        if not np.any(mask):
            return None
        xmax = np.nanmax(x_vals[mask])
        max_end = xmax / 2.0
        # Require full 8× span
        start_candidates = np.unique(x_vals[mask & (x_vals >= 32.0) & (x_vals * 8.0 <= max_end)])
        start_candidates = np.sort(start_candidates)
        best = None
        for start in start_candidates:
            end_limit = start * 8.0
            fit_mask = mask & (x_vals >= start) & (x_vals <= end_limit)
            if np.count_nonzero(fit_mask) < 3:
                continue
            logx = np.log10(x_vals[fit_mask])
            logy = np.log10(y_vals[fit_mask])
            slope, intercept = np.polyfit(logx, logy, 1)
            resid = logy - (slope * logx + intercept)
            chi2 = np.sum(resid ** 2)
            end_actual = x_vals[fit_mask].max()
            if best is None or chi2 < best[0] - 1e-12 or (
                np.isclose(chi2, best[0]) and (start < best[3] - 1e-12 or end_actual > best[4])
            ):
                best = (chi2, slope, intercept, start, end_actual)
        return best

    def plot_variant(x_vals, ratios, xlabel, suffix, y_label, prime=False):
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for (_, _, _, sub, color), ratio in zip(pairs, ratios):
            mask = np.isfinite(ratio) & (ratio > 0)
            if not np.any(mask):
                continue
            label = rf"$\theta_{{{sub}}}$" if not prime else rf"$\theta'_{{{sub}}}$"
            best = choose_best_powerlaw_fit(x_vals, ratio)
            if best is not None:
                _, slope, intercept, start_fit, end_fit = best
                x_fit = np.logspace(np.log10(start_fit), np.log10(end_fit), 200)
                y_fit = 10 ** (intercept) * x_fit ** slope
                ax.loglog(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    color=color,
                    alpha=0.5,
                    linewidth=3.0,
                    label=None,
                )
                if not prime:
                    label = rf"$\theta_{{{sub}}} \propto \ell^{{{slope:.2f}}}$"
                else:
                    label = rf"$\theta'_{{{sub}}} \propto \ell^{{{slope:.2f}}}$"
            ax.loglog(x_vals[mask], ratio[mask], color=color, lw=1.8, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fname = output_dir / f"alignment_angles_{suffix}.{fmt}"
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Created: {fname}")

    ratios_ell = []
    ratios_par = []
    ratios_perp = []
    ratios_xi = []
    ratios_lambda = []
    ratios_prime_ell = []
    ratios_prime_par = []
    ratios_prime_perp = []
    ratios_prime_xi = []
    ratios_prime_lambda = []
    x_xi = None
    x_lambda = None
    for cross_name, mag_name, ratio_name, _, _ in pairs:
        if cross_name not in ch_index or mag_name not in ch_index:
            ratios_ell.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_par.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_perp.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_xi.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_lambda.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_prime_ell.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_prime_par.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_prime_perp.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_prime_xi.append(np.full_like(ell_centers, np.nan, dtype=float))
            ratios_prime_lambda.append(np.full_like(ell_centers, np.nan, dtype=float))
            continue
        cross_idx = ch_index[cross_name]
        mag_idx = ch_index[mag_name]
        ratio_idx = ch_index.get(ratio_name, None)
        cross_hist = hist[cross_idx]
        mag_hist = hist[mag_idx]
        ratio_hist = hist[ratio_idx] if ratio_idx is not None else None
        cross_edges = delta_bin_edges[cross_idx]
        mag_edges = delta_bin_edges[mag_idx]
        ratio_edges = delta_bin_edges[ratio_idx] if ratio_idx is not None else None

        # Mean over theta, phi
        mean_cross_ell, w_cross_ell = _mean_from_hist(cross_hist, cross_edges, axis_sum=(1, 2, 3))
        mean_mag_ell, w_mag_ell = _mean_from_hist(mag_hist, mag_edges, axis_sum=(1, 2, 3))
        mean_ratio_ell = None
        if ratio_hist is not None and ratio_edges is not None:
            mean_ratio_ell, _ = _mean_from_hist(ratio_hist, ratio_edges, axis_sum=(1, 2, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_ell = np.where((mean_mag_ell > 0) & (mean_cross_ell > 0),
                                 mean_cross_ell / mean_mag_ell,
                                 np.nan)
        ratios_ell.append(ratio_ell)
        ratios_prime_ell.append(mean_ratio_ell if mean_ratio_ell is not None else np.full_like(ratio_ell, np.nan))

        # Mean per (ell, theta), collapsing phi
        mean_cross_theta, w_cross_theta = _mean_from_hist(cross_hist, cross_edges, axis_sum=(2, 3))
        mean_mag_theta, w_mag_theta = _mean_from_hist(mag_hist, mag_edges, axis_sum=(2, 3))
        mean_ratio_theta = None
        weights_ratio_theta = None
        if ratio_hist is not None and ratio_edges is not None:
            mean_ratio_theta, weights_ratio_theta = _mean_from_hist(ratio_hist, ratio_edges, axis_sum=(2, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_theta = np.where((mean_mag_theta > 0) & (mean_cross_theta > 0),
                                   mean_cross_theta / mean_mag_theta,
                                   np.nan)
        # Weighted average over theta to build parallel/perp curves (masked)
        weights_theta = np.where(np.isfinite(ratio_theta), w_mag_theta, 0.0)
        weights_ratio_theta = np.where(np.isfinite(mean_ratio_theta), weights_ratio_theta, 0.0) if mean_ratio_theta is not None else None
        ell_par = ell_centers[:, None] * cos_theta[None, :]
        ell_perp = ell_centers[:, None] * sin_theta[None, :]
        theta_L = L_mask.any(axis=1)
        theta_perp = perp_mask.any(axis=1)
        ratio_prime_par = np.full_like(ratio_theta, np.nan)
        ratio_prime_perp = np.full_like(ratio_theta, np.nan)
        with np.errstate(invalid="ignore"):
            denom_par = (weights_theta * theta_L[None, :]).sum(axis=1)
            ratio_par = np.where(
                denom_par > 0,
                np.nansum(ratio_theta * weights_theta * theta_L[None, :], axis=1) / denom_par,
                np.nan,
            )
            if mean_ratio_theta is not None and weights_ratio_theta is not None:
                denom_par_p = (weights_ratio_theta * theta_L[None, :]).sum(axis=1)
                ratio_prime_par = np.where(
                    denom_par_p > 0,
                    np.nansum(mean_ratio_theta * weights_ratio_theta * theta_L[None, :], axis=1) / denom_par_p,
                    np.nan,
                )
            denom_perp = (weights_theta * theta_perp[None, :]).sum(axis=1)
            ratio_perp = np.where(
                denom_perp > 0,
                np.nansum(ratio_theta * weights_theta * theta_perp[None, :], axis=1) / denom_perp,
                np.nan,
            )
            if mean_ratio_theta is not None and weights_ratio_theta is not None:
                denom_perp_p = (weights_ratio_theta * theta_perp[None, :]).sum(axis=1)
                ratio_prime_perp = np.where(
                    denom_perp_p > 0,
                    np.nansum(mean_ratio_theta * weights_ratio_theta * theta_perp[None, :], axis=1) / denom_perp_p,
                    np.nan,
                )
        ratios_par.append(ratio_par)
        ratios_perp.append(ratio_perp)
        ratios_prime_par.append(ratio_prime_par if mean_ratio_theta is not None else np.full_like(ratio_par, np.nan))
        ratios_prime_perp.append(ratio_prime_perp if mean_ratio_theta is not None else np.full_like(ratio_perp, np.nan))
        # ξ and λ components using phi decomposition of ℓ_perp (masked)
        mean_cross_tp, w_cross_tp = _mean_from_hist(cross_hist, cross_edges, axis_sum=(3,))
        mean_mag_tp, w_mag_tp = _mean_from_hist(mag_hist, mag_edges, axis_sum=(3,))
        mean_ratio_tp = _mean_from_hist(ratio_hist, ratio_edges, axis_sum=(3)) if ratio_hist is not None and ratio_edges is not None else (None, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_tp = np.where((mean_mag_tp > 0) & (mean_cross_tp > 0),
                                mean_cross_tp / mean_mag_tp,
                                np.nan)
        weights_tp = np.where(np.isfinite(ratio_tp), w_mag_tp, 0.0)
        mean_ratio_tp_vals, w_ratio_tp = mean_ratio_tp
        weights_ratio_tp = np.where(np.isfinite(mean_ratio_tp_vals), w_ratio_tp, 0.0) if mean_ratio_tp_vals is not None else None
        xi_grid = ell_perp[:, :, None] * cos_phi[None, None, :]
        lambda_grid = ell_perp[:, :, None] * sin_phi[None, None, :]
        ratio_prime_xi = np.full_like(ratio_tp, np.nan)
        ratio_prime_lambda = np.full_like(ratio_tp, np.nan)
        with np.errstate(invalid="ignore"):
            denom_xi = (weights_tp * xi_mask[None, :, :]).sum(axis=(1, 2))
            ratio_xi = np.where(
                denom_xi > 0,
                np.nansum(ratio_tp * weights_tp * xi_mask[None, :, :], axis=(1, 2)) / denom_xi,
                np.nan,
            )
            if mean_ratio_tp_vals is not None and weights_ratio_tp is not None:
                denom_xi_p = (weights_ratio_tp * xi_mask[None, :, :]).sum(axis=(1, 2))
                ratio_prime_xi = np.where(
                    denom_xi_p > 0,
                    np.nansum(mean_ratio_tp_vals * weights_ratio_tp * xi_mask[None, :, :], axis=(1, 2)) / denom_xi_p,
                    np.nan,
                )
            denom_lambda = (weights_tp * lambda_mask[None, :, :]).sum(axis=(1, 2))
            ratio_lambda = np.where(
                denom_lambda > 0,
                np.nansum(ratio_tp * weights_tp * lambda_mask[None, :, :], axis=(1, 2)) / denom_lambda,
                np.nan,
            )
            if mean_ratio_tp_vals is not None and weights_ratio_tp is not None:
                denom_lambda_p = (weights_ratio_tp * lambda_mask[None, :, :]).sum(axis=(1, 2))
                ratio_prime_lambda = np.where(
                    denom_lambda_p > 0,
                    np.nansum(mean_ratio_tp_vals * weights_ratio_tp * lambda_mask[None, :, :], axis=(1, 2)) / denom_lambda_p,
                    np.nan,
                )
            if x_xi is None:
                x_xi = np.where(
                    denom_xi > 0,
                    np.nansum(xi_grid * weights_tp * xi_mask[None, :, :], axis=(1, 2)) / denom_xi,
                    np.nan,
                )
                x_lambda = np.where(
                    denom_lambda > 0,
                    np.nansum(lambda_grid * weights_tp * lambda_mask[None, :, :], axis=(1, 2)) / denom_lambda,
                    np.nan,
                )
        ratios_xi.append(ratio_xi)
        ratios_lambda.append(ratio_lambda)
        ratios_prime_xi.append(ratio_prime_xi if mean_ratio_tp_vals is not None else np.full_like(ratio_xi, np.nan))
        ratios_prime_lambda.append(ratio_prime_lambda if mean_ratio_tp_vals is not None else np.full_like(ratio_lambda, np.nan))

    plot_variant(ell_centers, ratios_ell, r"$\ell$", "ell", y_label=r"$\theta$")
    plot_variant(ell_centers, ratios_par, r"$\ell_{\parallel}$", "ell_parallel", y_label=r"$\theta$")
    plot_variant(ell_centers, ratios_perp, r"$\ell_{\perp}$", "ell_perp", y_label=r"$\theta$")
    plot_variant(x_xi if x_xi is not None else ell_centers, ratios_xi, r"$\xi$", "xi", y_label=r"$\theta$")
    plot_variant(x_lambda if x_lambda is not None else ell_centers, ratios_lambda, r"$\lambda$", "lambda", y_label=r"$\theta$")
    plot_variant(ell_centers, ratios_prime_ell, r"$\ell$", "ell_prime", y_label=r"$\theta'$", prime=True)
    plot_variant(ell_centers, ratios_prime_par, r"$\ell_{\parallel}$", "ell_parallel_prime", y_label=r"$\theta'$", prime=True)
    plot_variant(ell_centers, ratios_prime_perp, r"$\ell_{\perp}$", "ell_perp_prime", y_label=r"$\theta'$", prime=True)
    plot_variant(x_xi if x_xi is not None else ell_centers, ratios_prime_xi, r"$\xi$", "xi_prime", y_label=r"$\theta'$", prime=True)
    plot_variant(x_lambda if x_lambda is not None else ell_centers, ratios_prime_lambda, r"$\lambda$", "lambda_prime", y_label=r"$\theta'$", prime=True)
if __name__ == "__main__":
    exit(main())
