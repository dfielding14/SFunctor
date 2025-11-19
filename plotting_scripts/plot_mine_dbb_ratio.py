#!/usr/bin/env python

"""Make the referee-requested plot of δB/B_mean,loc versus scale ℓ."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import cmasher as cmr  # type: ignore

HIST_CMAP = cmr.ocean_r


# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

def _configure_matplotlib() -> None:
    """Match the house style used for the existing figures."""
    texlive_path = (
        "/sw/andes/spack-envs/base/opt/linux-rhel8-x86_64/"
        "gcc-8.3.1/texlive-20210325-ari2ztcowrqldrwjblfqpdiphkzd3nhu/bin/x86_64-linux"
    )
    os.environ["PATH"] = f"{texlive_path}:{os.environ['PATH']}"

    import matplotlib  # Local import so PATH is updated first.

    matplotlib.rc("font", family="serif", size=12)
    matplotlib.rcParams["xtick.direction"] = "out"
    matplotlib.rcParams["ytick.direction"] = "out"
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True
    matplotlib.rcParams["xtick.minor.visible"] = True
    matplotlib.rcParams["ytick.minor.visible"] = True
    matplotlib.rcParams["lines.dash_capstyle"] = "round"
    matplotlib.rcParams["figure.dpi"] = 200
    matplotlib.rcParams["text.usetex"] = True


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

CHANNEL_NAME = "D_B_over_Bmean_loc"


@dataclass(frozen=True)
class Dataset:
    path: str
    nres: int
    label: str
    color: str
    linewidth: float


def load_ratio_statistics(dataset: Dataset) -> Dict[str, np.ndarray]:
    """Load the δB/B_mean histogram and derived statistics for one run."""
    data = np.load(dataset.path, allow_pickle=True)

    mag_channels = data["mag_channels"]
    if CHANNEL_NAME not in mag_channels:
        raise ValueError(f"{CHANNEL_NAME} not available in {dataset.path}")
    channel_index = int(np.where(mag_channels == CHANNEL_NAME)[0][0])

    ell_edges = data["ell_bin_edges"]  # (N_ell + 1,)
    ratio_edges = data["sf_channel_bin_edges"][channel_index]  # (N_bins + 1,)
    ell_centers = 0.5 * (ell_edges[1:] + ell_edges[:-1])
    ratio_centers = np.sqrt(ratio_edges[1:] * ratio_edges[:-1])

    # hist_mag[channel] has shape (N_ell, N_theta, N_phi, N_bins)
    counts = data["hist_mag"][channel_index]
    counts = counts.sum(axis=(1, 2))  # (N_ell, N_bins)

    totals = counts.sum(axis=1, keepdims=True)  # (N_ell, 1)
    totals_safe = np.where(totals > 0, totals, np.nan)

    widths = np.diff(ratio_edges)  # linear bin widths
    widths_safe = np.where(widths > 0, widths, np.nan)
    pdf_linear = counts / (totals_safe * widths_safe)  # integrates to 1 over dq
    pdf_weighted = pdf_linear * ratio_centers  # q * P(q)
    pdf_weighted = np.where(np.isfinite(pdf_weighted) & (pdf_weighted > 0), pdf_weighted, np.nan)

    # Median δB/B_mean for each ℓ bin.
    cdf = np.cumsum(counts, axis=1) / totals_safe
    medians = np.empty(len(ell_centers))
    medians[:] = np.nan
    for i, (cdf_row, total) in enumerate(zip(cdf, totals.flatten())):
        if not np.isfinite(total) or total <= 0:
            continue
        medians[i] = np.interp(0.5, cdf_row, ratio_centers)

    return {
        "ell_edges_norm": ell_edges / dataset.nres,
        "ell_centers_norm": ell_centers / dataset.nres,
        "ratio_edges": ratio_edges,
        "ratio_centers": ratio_centers,
        "pdf": pdf_weighted,  # shape (N_ell, N_bins)
        "median": medians,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def add_power_law_guide(
    ax: plt.Axes,
    ell_centers: np.ndarray,
    medians: np.ndarray,
    slope: float = 0.5,
    boost: float = 1.15,
) -> None:
    """Overlay an offset power-law guide aligned with the high-resolution median."""
    valid = np.isfinite(medians) & (medians > 0)
    if not np.any(valid):
        return

    x_valid = ell_centers[valid]
    y_valid = medians[valid]

    ref_x = 1e-2
    if not (x_valid.min() <= ref_x <= x_valid.max()):
        ref_x = x_valid[len(x_valid) // 2]

    log_ref_y = np.interp(np.log10(ref_x), np.log10(x_valid), np.log10(y_valid))
    ref_y = 10 ** log_ref_y
    amplitude = boost * ref_y / (ref_x ** slope)

    x_line = np.geomspace(1e-3, 3e-1, 200)
    y_line = amplitude * (x_line ** slope)

    line = ax.loglog(x_line, y_line, color="white", lw=0.5, ls=":")[0]
    line.set_dash_capstyle("round")
    line.set_solid_capstyle("round")
    line.set_solid_joinstyle("round")
    line.set_zorder(10)

    label_x = 5e-2
    label_y = amplitude * (label_x ** slope) * 1.1

    pt1 = ax.transData.transform([label_x, amplitude * (label_x ** slope)])
    pt2 = ax.transData.transform([label_x * 1.5, amplitude * ((label_x * 1.5) ** slope)])
    angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

    ax.text(
        label_x,
        label_y,
        r"$\propto \ell^{1/2}$",
        color="white",
        ha="left",
        va="bottom",
        fontsize=10,
        rotation=angle,
        rotation_mode="anchor",
        zorder=10,
    )


def plot_ratio_histogram(datasets: Iterable[Dataset], output_path: str) -> None:
    """Create the δB/B_mean,loc comparison plot."""
    datasets = list(datasets)
    high_res = datasets[0]
    stats = {ds.label: load_ratio_statistics(ds) for ds in datasets}

    high_stats = stats[high_res.label]
    ell_edges = high_stats["ell_edges_norm"]
    ratio_edges = high_stats["ratio_edges"]
    pdf = high_stats["pdf"].T  # (N_bins, N_ell) for pcolormesh

    fig, ax = plt.subplots(figsize=(5, 3.5))
    mesh = ax.pcolormesh(
        ell_edges,
        ratio_edges,
        pdf,
        norm=LogNorm(vmin=1e-4, vmax=1.0),
        cmap=HIST_CMAP,
        shading="auto",
        rasterized=True,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", width=0.5, length=2)
    ax.tick_params(axis="both", which="minor", width=0.5, length=1)

    inset = ax.inset_axes([0.5, 0.05, 0.45, 0.05])
    cb = fig.colorbar(mesh, cax=inset, orientation="horizontal")
    cb.set_label(
        r"$\left(\frac{|\delta B|}{B_{\rm mean, loc}}\right) P\!\left(\frac{|\delta B|}{B_{\rm mean, loc}} \mid \ell\right)$",
        labelpad=5,
    )
    decade_ticks = [10.0**exp for exp in (-4, -3, -2, -1, 0)]
    cb.set_ticks(decade_ticks)
    cb.set_ticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
    cb.ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10, dtype=float)))
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.tick_top()
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(
        axis="x",
        labelsize=10,
        width=0.5,
        length=2,
        direction="in",
        which="major",
    )
    cb.ax.tick_params(axis="x", width=0.5, length=1, direction="in", which="minor")
    for spine in cb.ax.spines.values():
        spine.set_linewidth(0.5)

    for ds in datasets:
        ds_stats = stats[ds.label]
        ell_centers = ds_stats["ell_centers_norm"]
        medians = ds_stats["median"]
        mask = np.isfinite(medians) & (medians > 0)
        if not np.any(mask):
            continue

        ell_valid = ell_centers[mask]
        median_valid = medians[mask]

        ell_break = 16.0 / ds.nres
        split_idx = int(np.searchsorted(ell_valid, ell_break, side="right"))
        solid_end = min(split_idx, len(ell_valid))

        def draw_segment(x_vals, y_vals, style: str = "solid") -> None:
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
            elif style == "dashed":
                line.set_linestyle((0, (6, 4)))

        draw_segment(ell_valid[:solid_end], median_valid[:solid_end], style="dotted")
        if solid_end < len(ell_valid):
            solid_start = max(solid_end - 1, 0)
            draw_segment(
                ell_valid[solid_start:],
                median_valid[solid_start:],
                style="solid",
            )

        ax.text(
            ell_valid[0] / 1.05,
            median_valid[0] / 1.05,
            rf"${ds.label}$",
            color=ds.color,
            ha="left",
            va="top",
            fontsize=10,
        )

    high_medians = high_stats["median"]
    add_power_law_guide(
        ax,
        high_stats["ell_centers_norm"],
        high_medians,
        slope=0.5,
        boost=1.08,
    )

    b0_line = ax.axhline(1.0, color="#ffcc66", linestyle="--")
    b0_line.set_dash_capstyle("round")
    b0_line.set_solid_capstyle("round")
    ax.text(
        ell_edges[-1] * 0.9,
        1.0 / 1.15,
        r"$\delta B = B_{\rm mean, loc}$",
        color="#ffcc66",
        va="top",
        ha="right",
        fontsize=12,
    )

    ax.set_xlabel(r"$\ell / L$")
    ax.set_ylabel(r"$|\delta B(\ell)| / B_{\rm mean, loc}(\ell)$")
    ax.set_ylim(8e-4, 125)

    legend_handle = Line2D(
        [0, 1],
        [0, 1],
        color=datasets[0].color,
        lw=datasets[0].linewidth,
        solid_capstyle="round",
        solid_joinstyle="round",
        dash_capstyle="round",
    )
    ax.legend(
        handles=[legend_handle],
        labels=[r"$\mathrm{median}\ |\delta B(\ell)|$"],
        loc="upper left",
        fontsize=10,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
            label=r"L/\Delta x = 10{,}240",
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
            label=r"L/\Delta x = 2{,}560",
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
            label=r"L/\Delta x = 640",
            color="#ff9999",
            linewidth=0.75,
        ),
    ]

    output_path = "db_over_bmean_median_comparison.pdf"
    plot_ratio_histogram(datasets, output_path)


if __name__ == "__main__":
    main()
