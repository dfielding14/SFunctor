#!/usr/bin/env python

"""Plot qℓ P(qℓ) profiles for raw histograms and KDE-smoothed data."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import cmasher as cmr

from plot_mine_dbb_ratio import (
    CHANNEL_NAME,
    Dataset,
    _configure_matplotlib,
)
from plot_mine_dbb_ratio_kde import load_histogram, kde_smooth_qp


RAW_PATH = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "sfunctor_results/results_Turb_10240_beta25_dedt025_plm/"
    "ndisp100_000_nrand100_000_nell128_sw3_job3663362/sf_results_all_slices.npz"
)


def load_raw_qp(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ell centers, ratio centers, and qP(q) from raw histograms."""
    data = np.load(dataset.path, allow_pickle=True)
    channels = list(data["channels"])
    if CHANNEL_NAME not in channels:
        raise ValueError(f"{CHANNEL_NAME} not available in {dataset.path}")
    channel_index = channels.index(CHANNEL_NAME)

    ell_edges = data["ell_bin_edges"]
    ell_centers = 0.5 * (ell_edges[1:] + ell_edges[:-1])

    delta_bin_edges = data.get("delta_bin_edges", None)
    if delta_bin_edges is None:
        metadata = dict(data["metadata"].item()) if "metadata" in data else {}
        if "log_delta_bin_edges_min" not in metadata or "log_delta_bin_edges_max" not in metadata:
            raise KeyError("delta_bin_edges missing and metadata reconstruction unavailable.")
        delta_bin_edges = []
        for lo, hi in zip(metadata["log_delta_bin_edges_min"], metadata["log_delta_bin_edges_max"]):
            delta_bin_edges.append(np.logspace(lo, hi, metadata["N_delta_bin_edges"]))
    ratio_edges = np.asarray(delta_bin_edges[channel_index], dtype=float)
    ratio_centers = np.sqrt(ratio_edges[1:] * ratio_edges[:-1])

    counts = data["hist"][channel_index].sum(axis=(1, 2))
    totals = counts.sum(axis=1, keepdims=True)
    widths = np.diff(ratio_edges)

    pdf_linear = (counts / totals) / widths
    q_times_pdf = ratio_centers * pdf_linear

    return ell_centers / dataset.nres, ratio_centers, q_times_pdf


def plot_profiles(
    ell_norm: np.ndarray,
    ratios: np.ndarray,
    qpdf: np.ndarray,
    output_file: str,
) -> None:
    """Plot qP(q) curves colored by ℓ."""
    fig, ax = plt.subplots(figsize=(5, 3))

    curves = []
    for profile in qpdf:
        peak_idx = np.argmax(profile)
        x = ratios / ratios[peak_idx]
        y = np.where(profile > 0, profile, np.nan)
        curves.append(np.column_stack([x, y]))

    norm = LogNorm(vmin=ell_norm.min(), vmax=ell_norm.max())
    lc = LineCollection(curves, array=ell_norm, cmap=cmr.tropical, norm=norm)
    lc.set_linewidth(1.0)
    ax.add_collection(lc)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(1e-6, 1)

    # all_values = np.concatenate([curve[:, 1] for curve in curves])
    # finite = all_values[np.isfinite(all_values) & (all_values > 0)]
    # if finite.size:
    #     ymin = finite.min()
    #     ymax = np.percentile(finite, 99.5)
    #     ax.set_ylim(ymin, ymax * 1.1)

    ax.set_xlabel(
        r"$\left(\delta B(\ell)/\overline{B}(\ell)\right) /"
        r"\left(\delta B(\ell)/\overline{B}(\ell)\right)_{\rm max}$"
    )
    ax.set_ylabel(
        r"$\left(\delta B(\ell)/\overline{B}(\ell)\right)"
        r" P\!\left(\delta B(\ell)/\overline{B}(\ell) \middle| \ell \right)$"
    )
    # ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmr.tropical, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$\ell / L$")
    # tick_values = np.array([1 / 10240, 1 / 2560, 1 / 640, 1 / 160, 1 / 40, 1 / 10, 0.5])
    # tick_labels = np.array([1 / 10240, 1 / 2560, 1 / 640, 1 / 160, 1 / 40, 1 / 10, 0.5])
    # # tick_values = tick_values[(tick_values >= ell_norm.min()) & (tick_values <= ell_norm.max())]
    # if tick_values.size:
    #     cbar.set_ticks(tick_values)
    #     cbar.set_ticklabels([f"{val:.3g}" for val in tick_values])
    # cbar.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    fig.savefig(output_file[:-4]+".png", dpi=300)


def compute_slopes(
    ratios: np.ndarray,
    qpdf: np.ndarray,
    ell_centers: np.ndarray,
    filename: str,
) -> None:
    """Fit slopes to qP(q) profiles between q/qmax=40 and 200."""
    slopes = np.zeros(len(ell_centers))

    for i, profile in enumerate(qpdf):
        peak_idx = np.argmax(profile)
        q_norm = ratios / ratios[peak_idx]
        mask = (q_norm >= 40) & (q_norm <= 200) & (profile > 0)
        if np.sum(mask) < 2:
            slopes[i] = np.nan
            continue
        logx = np.log(q_norm[mask])
        logy = np.log(profile[mask])
        slope, intercept = np.polyfit(logx, logy, 1)
        slopes[i] = -slope  # report positive magnitude

    fig, ax = plt.subplots()
    mask = np.isfinite(slopes) & (slopes > 0)
    ax.loglog(ell_centers[mask], slopes[mask])
    ax.set_xlabel(r"$\ell / L$")
    ax.set_ylabel(r"$-\,\mathrm{slope}$")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    fig.savefig(filename[:-4]+".png", dpi=300)


def main() -> None:
    _configure_matplotlib()

    dataset = Dataset(
        path=RAW_PATH,
        nres=10240,
        label=r"L/\Delta x = 10{,}240",
        color="#ff3399",
        linewidth=2.0,
    )

    ell_norm, qbins, qpdf_raw = load_raw_qp(dataset)
    plot_profiles(
        ell_norm,
        qbins,
        qpdf_raw,
        output_file="qp_profiles_raw.pdf",
    )
    compute_slopes(qbins, qpdf_raw, ell_norm, "qp_profiles_raw_slopes.pdf")

    hist = load_histogram(dataset)
    qpdf_kde = kde_smooth_qp(hist.counts, hist.ratio_centers, hist.ratio_edges)
    plot_profiles(
        hist.ell_centers_norm,
        hist.ratio_centers,
        qpdf_kde,
        output_file="qp_profiles_kde.pdf",
    )
    compute_slopes(hist.ratio_centers, qpdf_kde, hist.ell_centers_norm, "qp_profiles_kde_slopes.pdf")


if __name__ == "__main__":
    main()
