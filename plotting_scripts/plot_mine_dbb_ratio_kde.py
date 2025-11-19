#!/usr/bin/env python

"""Variant of the δB/B_mean plot using KDE-smoothed contours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

from plot_mine_dbb_ratio import (  # Reuse styling helpers and metadata.
    CHANNEL_NAME,
    HIST_CMAP,
    Dataset,
    add_power_law_guide,
    _configure_matplotlib,
)


@dataclass(frozen=True)
class HistogramData:
    ell_edges_norm: np.ndarray
    ell_centers_norm: np.ndarray
    ratio_edges: np.ndarray
    ratio_centers: np.ndarray
    counts: np.ndarray
    medians: np.ndarray


def load_histogram(dataset: Dataset) -> HistogramData:
    """Load the raw histogram data for the requested dataset."""
    data = np.load(dataset.path, allow_pickle=True)

    mag_channels = data["mag_channels"]
    if CHANNEL_NAME not in mag_channels:
        raise ValueError(f"{CHANNEL_NAME} not available in {dataset.path}")
    channel_index = int(np.where(mag_channels == CHANNEL_NAME)[0][0])

    ell_edges = data["ell_bin_edges"]
    ratio_edges = data["sf_channel_bin_edges"][channel_index]
    ell_centers = 0.5 * (ell_edges[1:] + ell_edges[:-1])
    ratio_centers = np.sqrt(ratio_edges[1:] * ratio_edges[:-1])

    counts = data["hist_mag"][channel_index]
    counts = counts.sum(axis=(1, 2))  # (N_ell, N_bins)

    totals = counts.sum(axis=1, keepdims=True)
    totals_safe = np.where(totals > 0, totals, np.nan)

    cdf = np.cumsum(counts, axis=1) / totals_safe
    medians = np.empty(len(ell_centers))
    medians[:] = np.nan
    for i, (cdf_row, total) in enumerate(zip(cdf, totals.flatten())):
        if not np.isfinite(total) or total <= 0:
            continue
        medians[i] = np.interp(0.5, cdf_row, ratio_centers)

    return HistogramData(
        ell_edges_norm=ell_edges / dataset.nres,
        ell_centers_norm=ell_centers / dataset.nres,
        ratio_edges=ratio_edges,
        ratio_centers=ratio_centers,
        counts=counts,
        medians=medians,
    )


def kde_smooth_qp(
    counts: np.ndarray,
    ratio_centers: np.ndarray,
    ratio_edges: np.ndarray,
    bandwidth: float = 0.05,
) -> np.ndarray:
    """
    Smooth q * P(q | ℓ) using a Gaussian kernel in log10-space for each ℓ.
    """
    log_ratio = np.log10(ratio_centers)
    ln10 = np.log(10.0)

    diff = log_ratio[:, None] - log_ratio[None, :]
    kernel = np.exp(-0.5 * (diff / bandwidth) ** 2)
    kernel /= np.sqrt(2.0 * np.pi) * bandwidth

    widths = np.diff(ratio_edges)
    q_times_pdf = np.full_like(counts, np.nan, dtype=float)

    for i, w in enumerate(counts):
        total = w.sum()
        if total <= 0:
            continue

        weights = w / total
        density_log = kernel @ weights  # with respect to d(log10 q)
        smoothed = density_log / ln10

        if np.any(~np.isfinite(smoothed)):
            pdf_linear = (w / total) / widths
            smoothed = ratio_centers * pdf_linear

        q_times_pdf[i] = smoothed

    return q_times_pdf


def plot_kde_contours(
    datasets: Iterable[Dataset],
    output_path: str,
    levels: Iterable[float],
) -> None:
    datasets = list(datasets)
    histograms: Dict[str, HistogramData] = {
        ds.label: load_histogram(ds) for ds in datasets
    }

    high_data = histograms[datasets[0].label]
    q_times_pdf = kde_smooth_qp(
        high_data.counts,
        high_data.ratio_centers,
        high_data.ratio_edges,
    )

    X = high_data.ell_edges_norm[:-1]
    x_min = 1.0 / datasets[0].nres
    x_max = 0.5
    X[-1] = x_max
    Y = high_data.ratio_centers
    Z = q_times_pdf.T  # contourf expects (len(Y), len(X))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap=HIST_CMAP,
        norm=LogNorm(vmin=min(levels), vmax=max(levels)),
        extend="both",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", width=0.5, length=2)
    ax.tick_params(axis="both", which="minor", width=0.5, length=1)

    inset = ax.inset_axes([0.55, 0.025, 0.425, 0.04])
    cb = fig.colorbar(contour, cax=inset, orientation="horizontal")
    cb.set_label(
        r"$\left(\frac{\left|\delta B\right|}{\overline{B}(\ell)}\right)"
        r"P\!\left( \frac{\left|\delta B\right|}{\overline{B}(\ell)} \,\middle|\, \ell\right)$",
        labelpad=12,
    )
    decade_ticks = [10.0**exp for exp in (-5, -4, -3, -2, -1)]
    cb.set_ticks(decade_ticks)
    cb.set_ticklabels([r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.tick_top()
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(axis="x", labelsize=9, width=0.5, length=0, direction="in", which="major", pad=0)
    # cb.ax.tick_params(axis="x", width=0.5, length=1, direction="in", which="minor")
    cb.ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10, dtype=float)))
    for spine in cb.ax.spines.values():
        spine.set_linewidth(0.5)

    for ds in datasets:
        data = histograms[ds.label]
        ell = data.ell_centers_norm
        med = data.medians
        mask = np.isfinite(med) & (med > 0)
        if not np.any(mask):
            continue

        ell_valid = ell[mask]
        med_valid = med[mask]
        ell_break = 16.0 / ds.nres
        split_idx = int(np.searchsorted(ell_valid, ell_break, side="right"))
        solid_end = min(split_idx, len(ell_valid))

        def draw_segment(x_vals: np.ndarray, y_vals: np.ndarray, style: str) -> None:
            if len(x_vals) < 2:
                return
            line = ax.loglog(
                x_vals,
                y_vals,
                color=ds.color,
                lw=ds.linewidth,
                zorder=3,
            )[0]
            line.set_solid_capstyle("round")
            line.set_solid_joinstyle("round")
            line.set_dash_capstyle("round")
            if style == "dotted":
                line.set_linestyle((0, (1, 2)))

        draw_segment(ell_valid[:solid_end], med_valid[:solid_end], "dotted")
        if solid_end < len(ell_valid):
            start = max(solid_end - 1, 0)
            draw_segment(ell_valid[start:], med_valid[start:], "solid")

        n_points = len(ell_valid)
        if n_points == 0:
            continue

        first_idx = 0
        second_idx = min(1, n_points - 1)

        x0 = ell_valid[first_idx]
        y0 = med_valid[first_idx]
        x_label = x0
        y_label = y0

        if n_points > 1:
            x1 = ell_valid[second_idx]
            y1 = med_valid[second_idx]
            log_x0 = np.log10(x0)
            log_x1 = np.log10(x1)
            log_y0 = np.log10(y0)
            log_y1 = np.log10(y1)
            frac = 0.12
            x_label = 10 ** (log_x0 + frac * (log_x1 - log_x0))
            y_label = 10 ** (log_y0 + frac * (log_y1 - log_y0))
        else:
            x1 = x0
            y1 = y0

        delta_x = x1 - x0
        delta_y = y1 - y0
        angle_data = 0.0
        if not (np.isclose(delta_x, 0.0) and np.isclose(delta_y, 0.0)):
            angle_data = np.degrees(np.arctan2(delta_y, delta_x))

        angle = ax.transData.transform_angles(
            np.array([angle_data]),
            np.array([[x_label, y_label]]),
        )[0]

        y_offset = y_label * 1.1
        x_offset = x_label * 0.8

        ax.annotate(
            rf"${ds.label}$",
            xy=(x_offset, y_offset),
            xytext=(6, 0),
            textcoords="offset points",
            color=ds.color,
            fontsize=12,
            rotation=angle,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
            zorder=4,
        )

    add_power_law_guide(
        ax,
        high_data.ell_centers_norm,
        histograms[datasets[0].label].medians,
        slope=0.5,
        boost=1.15,
    )

    b0_line = ax.axhline(1.0, color="#ffcc66", linestyle="--")
    b0_line.set_dash_capstyle("round")
    b0_line.set_solid_capstyle("round")
    ax.text(
        X[0] * 1.15,
        1 / 1.15,
        r"$ |\delta B (\ell)| = \overline B (\ell)$",
        color="#ffcc66",
        va="top",
        ha="left",
        fontsize=12,
    )

    ax.set_xlabel(r"$\ell / L$")
    ax.set_ylabel(r"$\left|\delta B(\ell)\right| / \overline{B}(\ell)$")
    ax.set_ylim(8e-4, 125)
    ax.set_xlim(x_min, x_max)

    primary = datasets[0]
    legend_handle = Line2D(
        [0, 1],
        [0, 1],
        color=primary.color,
        lw=primary.linewidth,
        solid_capstyle="round",
        solid_joinstyle="round",
        dash_capstyle="round",
    )
    ax.legend(
        handles=[legend_handle],
        labels=[r"$\mathrm{median}$"],
        loc="upper left",
        fontsize=10,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path[:-4]+".png", dpi=300, bbox_inches="tight")


def main() -> None:
    _configure_matplotlib()

    datasets = [
        Dataset(
            path=(
                "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
                "sfunctor_results/results_Turb_10240_beta25_dedt025_plm/"
                "ndisp100_000_nrand100_000_nell128_sw3_job3663362/"
                "sf_results_all_slices.npz"
            ),
            nres=10240,
            label=r"10{,}240",
            color="#ff3399",
            linewidth=2.0,
        ),
        Dataset(
            path=(
                "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
                "sfunctor_results/results_Turb_2560_beta25_dedt025_plm/"
                "ndisp25000_nrand25000_nell128_sw3_job3665055/"
                "sf_results_all_slices.npz"
            ),
            nres=2560,
            label=r"2560",
            color="#ff6699",
            linewidth=1.0,
        ),
        Dataset(
            path=(
                "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
                "sfunctor_results/results_Turb_640_beta25_dedt025_plm/"
                "ndisp12500_nrand12500_nell128_sw3_job3665054/"
                "sf_results_all_slices.npz"
            ),
            nres=640,
            label=r"640",
            color="#ff9999",
            linewidth=0.75,
        ),
    ]

    levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    plot_kde_contours(
        datasets,
        output_path="db_over_bmean_median_comparison_kde.pdf",
        levels=levels,
    )


if __name__ == "__main__":
    main()
