#!/usr/bin/env python3
"""Test 01: Core structure-function analysis on the unified histogram API."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.io.slice_io import load_slice_npz

TEST_FILE = "slice_data/Turb_2560_beta25_dedt025_plm_axis3_slicem0p375_file0024.npz"
if not (PROJECT_ROOT / TEST_FILE).exists():
    TEST_FILE = "slice_data/Turb_320_beta100_dedt025_plm_axis2_slice0_file0000.npz"

RESULTS_DIR = PROJECT_ROOT / "test_suite" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _infer_axis_from_name(path: str) -> int:
    name = Path(path).name
    if "_axis" not in name:
        return 3
    try:
        return int(name.split("_axis", maxsplit=1)[1].split("_", maxsplit=1)[0])
    except Exception:  # noqa: BLE001
        return 3


def _mean_delta_per_ell(hist_channel: np.ndarray, delta_edges: np.ndarray) -> np.ndarray:
    """Compute <delta> for each ell from count histograms."""
    counts = hist_channel.sum(axis=(1, 2))
    delta_centers = np.sqrt(delta_edges[:-1] * delta_edges[1:])
    totals = counts.sum(axis=1)
    out = np.full(counts.shape[0], np.nan, dtype=float)
    valid = totals > 0
    out[valid] = (counts[valid] * delta_centers[None, :]).sum(axis=1) / totals[valid]
    return out


def _normalized_by_ell(counts_ell_delta: np.ndarray) -> np.ndarray:
    totals = counts_ell_delta.sum(axis=1, keepdims=True)
    return np.divide(counts_ell_delta, totals, out=np.zeros_like(counts_ell_delta, dtype=float), where=totals > 0)


def test_core_structure_functions(test_file: str | None = None) -> bool:
    print("=" * 60)
    print("TEST 01: CORE STRUCTURE FUNCTIONS (UNIFIED API)")
    print("=" * 60)

    file_name = test_file or TEST_FILE
    data_path = PROJECT_ROOT / file_name
    if not data_path.exists():
        print(f"  Missing input file: {data_path}")
        return False

    # Keep runtime reasonable while preserving coverage.
    if "2560" in file_name:
        stride = 8
        n_disp = 800
        n_samples = 8000
        n_ell_bins = 40
    else:
        stride = 1
        n_disp = 400
        n_samples = 4000
        n_ell_bins = 24

    axis = _infer_axis_from_name(file_name)

    print(f"Data: {file_name}")
    print(f"Settings: stride={stride}, n_disp={n_disp}, n_samples={n_samples}, n_ell_bins={n_ell_bins}")

    load_start = time.time()
    slice_data = load_slice_npz(data_path, stride=stride)
    print(f"Loaded shape: {slice_data['rho'].shape} in {time.time() - load_start:.2f}s")

    run_start = time.time()
    results = analyze_slice(
        slice_data,
        n_displacements=n_disp,
        n_ell_bins=n_ell_bins,
        n_random_subsamples=n_samples,
        stencil_width=2,
        n_processes=1,
        axis=axis,
    )
    elapsed = time.time() - run_start

    hist = results["hist"]
    channels = list(results["channels"])
    ell_edges = np.asarray(results["ell_bin_edges"], dtype=float)
    theta_edges = np.asarray(results["theta_bin_edges"], dtype=float)
    phi_edges = np.asarray(results["phi_bin_edges"], dtype=float)
    delta_edges_all = [np.asarray(e, dtype=float) for e in results["delta_bin_edges"]]

    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])

    print(f"Analysis complete in {elapsed:.2f}s")
    print(f"Histogram shape: {hist.shape}")
    print(f"Total counts: {int(hist.sum()):,}")
    print(f"Channels: {len(channels)}")

    # ------------------------------------------------------------------
    # Plot 1: mean structure functions
    # ------------------------------------------------------------------
    channel_names = ["D_V", "D_B", "D_RHO", "D_ZPLUS", "D_ZMINUS", "D_OMEGA"]
    plt.figure(figsize=(9, 6))
    for name in channel_names:
        if name not in channels:
            continue
        idx = channels.index(name)
        mean_delta = _mean_delta_per_ell(hist[idx], delta_edges_all[idx])
        valid = np.isfinite(mean_delta) & (mean_delta > 0)
        if np.any(valid):
            plt.loglog(ell_centers[valid], mean_delta[valid], "o-", label=name, alpha=0.85)

    plt.xlabel("Scale $\\ell$")
    plt.ylabel(r"$\langle \delta \rangle$")
    plt.title("Mean Increment vs Scale")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_mean_structure_functions.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: 2D histograms for key channels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for ax, name in zip(axes, channel_names, strict=False):
        if name not in channels:
            ax.set_visible(False)
            continue
        idx = channels.index(name)
        counts = hist[idx].sum(axis=(1, 2))
        delta_edges = delta_edges_all[idx]
        vmax = max(2.0, float(np.nanmax(counts + 1.0)))
        pcm = ax.pcolormesh(
            ell_edges,
            delta_edges,
            (counts + 1.0).T,
            shading="auto",
            cmap="viridis",
            norm=LogNorm(vmin=1.0, vmax=vmax),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\delta$")
        ax.set_title(name)
        fig.colorbar(pcm, ax=ax)

    fig.suptitle("2D Histograms: counts(ell, delta)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_2d_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: angular distributions for D_V
    # ------------------------------------------------------------------
    if "D_V" in channels:
        idx = channels.index("D_V")
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        selected = np.linspace(0, len(ell_centers) - 1, num=min(4, len(ell_centers)), dtype=int)

        plt.figure(figsize=(8, 5))
        for ell_i in selected:
            counts_theta = hist[idx, ell_i].sum(axis=(1, 2))
            norm = counts_theta.sum()
            if norm > 0:
                plt.plot(np.rad2deg(theta_centers), counts_theta / norm, "o-", label=f"ell bin {ell_i}")

        plt.xlabel(r"$\theta$ (deg)")
        plt.ylabel("Probability")
        plt.title("Angular Distribution for D_V")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "01_angular_distributions.png", dpi=180, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 4: multiscale PDFs for D_B_over_Bmean_loc (if present)
    # ------------------------------------------------------------------
    if "D_B_over_Bmean_loc" in channels:
        idx = channels.index("D_B_over_Bmean_loc")
        counts = hist[idx].sum(axis=(1, 2))
        delta_edges = delta_edges_all[idx]
        delta_centers = np.sqrt(delta_edges[:-1] * delta_edges[1:])
        pdf = _normalized_by_ell(counts)

        selected = np.linspace(0, len(ell_centers) - 1, num=min(5, len(ell_centers)), dtype=int)
        plt.figure(figsize=(8, 5))
        for ell_i in selected:
            p = pdf[ell_i]
            valid = p > 0
            if np.any(valid):
                plt.loglog(delta_centers[valid], p[valid], label=f"ell bin {ell_i}")

        plt.xlabel(r"$q = |\delta B|/B_{\mathrm{loc}}$")
        plt.ylabel("P(q | ell)")
        plt.title("Multiscale PDFs for D_B_over_Bmean_loc")
        plt.grid(alpha=0.25, which="both")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "01_pdfs_multiscale.png", dpi=180, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 5: local scaling exponents (log slope)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for name in ["D_V", "D_B", "D_RHO"]:
        if name not in channels:
            continue
        idx = channels.index(name)
        mean_delta = _mean_delta_per_ell(hist[idx], delta_edges_all[idx])
        valid = np.isfinite(mean_delta) & (mean_delta > 0) & (ell_centers > 0)
        if np.sum(valid) < 3:
            continue

        x = np.log10(ell_centers[valid])
        y = np.log10(mean_delta[valid])
        slope = np.gradient(y, x)
        plt.plot(ell_centers[valid], slope, "o-", label=name)

    plt.xscale("log")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$d\log\langle\delta\rangle / d\log\ell$")
    plt.title("Local Scaling Exponents")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_scaling_analysis.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Plot 6: cross-product ratios vs ell
    # ------------------------------------------------------------------
    ratio_channels = [
        c for c in channels if c.endswith("CROSS_MAG_RATIO")
    ]
    if ratio_channels:
        plt.figure(figsize=(8, 5))
        for name in ratio_channels[:6]:
            idx = channels.index(name)
            mean_ratio = _mean_delta_per_ell(hist[idx], delta_edges_all[idx])
            valid = np.isfinite(mean_ratio) & (mean_ratio > 0)
            if np.any(valid):
                plt.plot(ell_centers[valid], mean_ratio[valid], "o-", label=name)

        plt.xscale("log")
        plt.xlabel(r"$\ell$")
        plt.ylabel("Mean ratio")
        plt.title("Cross-Product Ratio Channels")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "01_cross_products.png", dpi=180, bbox_inches="tight")
        plt.close()

    np.savez_compressed(
        DATA_DIR / "01_structure_functions.npz",
        hist=hist,
        channels=np.array(channels, dtype=object),
        ell_bin_edges=ell_edges,
        theta_bin_edges=theta_edges,
        phi_bin_edges=phi_edges,
        delta_bin_edges=np.array(delta_edges_all, dtype=object),
    )

    print("Generated plots:")
    for plot in sorted(PLOTS_DIR.glob("01_*.png")):
        print(f"  - {plot.name}")
    print(f"Saved data: {DATA_DIR / '01_structure_functions.npz'}")
    return True


def main(test_file: str | None = None) -> bool:
    return test_core_structure_functions(test_file=test_file)


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
