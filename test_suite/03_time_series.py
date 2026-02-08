#!/usr/bin/env python3
"""Test 03: pseudo time-series analysis using available slice snapshots."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.io.slice_io import load_slice_npz

RESULTS_DIR = PROJECT_ROOT / "test_suite" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _infer_axis(path: Path) -> int:
    name = path.name
    if "_axis" not in name:
        return 3
    try:
        return int(name.split("_axis", maxsplit=1)[1].split("_", maxsplit=1)[0])
    except Exception:  # noqa: BLE001
        return 3


def _resolve_snapshots(test_file: str | None) -> list[Path]:
    if test_file is not None:
        p = (PROJECT_ROOT / test_file).resolve()
        if p.exists():
            return [p, p, p]

    candidates = sorted((PROJECT_ROOT / "slice_data").glob("Turb_320_beta100_dedt025_plm_axis*_slice0_file0000.npz"))
    if len(candidates) >= 3:
        return candidates[:3]

    fallback = sorted((PROJECT_ROOT / "slice_data").glob("*.npz"))
    if not fallback:
        return []
    return [fallback[0], fallback[0], fallback[0]]


def _mean_delta(hist_channel: np.ndarray, delta_edges: np.ndarray) -> np.ndarray:
    counts = hist_channel.sum(axis=(1, 2))
    centers = np.sqrt(delta_edges[:-1] * delta_edges[1:])
    totals = counts.sum(axis=1)
    return np.divide((counts * centers[None, :]).sum(axis=1), totals, out=np.full_like(totals, np.nan, dtype=float), where=totals > 0)


def _parallel_perp_ratio(hist_channel: np.ndarray, theta_edges: np.ndarray) -> np.ndarray:
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    parallel_mask = theta_centers <= np.deg2rad(15.0)
    perp_mask = theta_centers >= np.deg2rad(75.0)

    counts_theta = hist_channel.sum(axis=(2, 3))
    parallel = counts_theta[:, parallel_mask].sum(axis=1)
    perp = counts_theta[:, perp_mask].sum(axis=1)
    return np.divide(parallel, perp, out=np.full_like(parallel, np.nan, dtype=float), where=perp > 0)


def _fit_scaling_exponent(ell_centers: np.ndarray, mean_curve: np.ndarray) -> float:
    valid = np.isfinite(mean_curve) & (mean_curve > 0) & (ell_centers > 0)
    if np.sum(valid) < 3:
        return float("nan")
    x = np.log10(ell_centers[valid])
    y = np.log10(mean_curve[valid])
    return float(np.polyfit(x, y, 1)[0])


def test_time_series(test_file: str | None = None) -> bool:
    print("=" * 60)
    print("TEST 03: PSEUDO TIME-SERIES ANALYSIS")
    print("=" * 60)
    print("Note: this uses available snapshots as a proxy for temporal evolution.")

    snapshots = _resolve_snapshots(test_file)
    if not snapshots:
        print("No slice snapshots found in slice_data/")
        return False

    print("Snapshots:")
    for i, snap in enumerate(snapshots):
        print(f"  t{i}: {snap.name}")

    metrics = {
        "times": [],
        "total_counts": [],
        "ratio_mean": [],
        "scaling_exponent": [],
    }

    ratio_pdfs = []
    ratio_centers_ref = None

    for t_idx, snap in enumerate(snapshots):
        run_start = time.time()

        stride = 4 if "2560" in snap.name else 2
        n_disp = 180
        n_samples = 1500

        data = load_slice_npz(snap, stride=stride)
        results = analyze_slice(
            data,
            n_displacements=n_disp,
            n_ell_bins=24,
            n_random_subsamples=n_samples,
            stencil_width=2,
            n_processes=1,
            axis=_infer_axis(snap),
        )

        hist = results["hist"]
        channels = list(results["channels"])
        ell_edges = np.asarray(results["ell_bin_edges"], dtype=float)
        theta_edges = np.asarray(results["theta_bin_edges"], dtype=float)
        delta_edges = [np.asarray(e, dtype=float) for e in results["delta_bin_edges"]]
        ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])

        dv_idx = channels.index("D_V")
        dv_mean = _mean_delta(hist[dv_idx], delta_edges[dv_idx])
        ratio_curve = _parallel_perp_ratio(hist[dv_idx], theta_edges)

        metrics["times"].append(float(t_idx))
        metrics["total_counts"].append(float(hist.sum()))
        metrics["ratio_mean"].append(float(np.nanmean(ratio_curve)))
        metrics["scaling_exponent"].append(_fit_scaling_exponent(ell_centers, dv_mean))

        if "D_B_over_Bmean_loc" in channels:
            q_idx = channels.index("D_B_over_Bmean_loc")
            counts = hist[q_idx].sum(axis=(1, 2))
            totals = counts.sum(axis=1, keepdims=True)
            pdf = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
            ratio_pdfs.append(pdf[np.nanargmax(np.isfinite(pdf).sum(axis=1))])
            q_edges = delta_edges[q_idx]
            ratio_centers_ref = np.sqrt(q_edges[:-1] * q_edges[1:])

        print(f"  t{t_idx}: done in {time.time() - run_start:.2f}s, counts={int(hist.sum()):,}")

    times = np.asarray(metrics["times"], dtype=float)
    total_counts = np.asarray(metrics["total_counts"], dtype=float)
    ratio_mean = np.asarray(metrics["ratio_mean"], dtype=float)
    scaling = np.asarray(metrics["scaling_exponent"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(times, total_counts, "o-")
    axes[0, 0].set_title("Total Histogram Counts")
    axes[0, 0].set_xlabel("Pseudo-time index")
    axes[0, 0].set_ylabel("Counts")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(times, ratio_mean, "o-")
    axes[0, 1].axhline(1.0, color="k", ls=":", alpha=0.5)
    axes[0, 1].set_title("Mean Parallel/Perp Ratio (D_V)")
    axes[0, 1].set_xlabel("Pseudo-time index")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(times, scaling, "o-")
    axes[1, 0].set_title("Global Scaling Exponent (D_V)")
    axes[1, 0].set_xlabel("Pseudo-time index")
    axes[1, 0].set_ylabel("Slope")
    axes[1, 0].grid(alpha=0.25)

    if ratio_pdfs and ratio_centers_ref is not None:
        for i, pdf in enumerate(ratio_pdfs):
            valid = pdf > 0
            axes[1, 1].loglog(ratio_centers_ref[valid], pdf[valid], label=f"t{i}")
        axes[1, 1].set_title("PDF of D_B_over_Bmean_loc")
        axes[1, 1].set_xlabel(r"$q$")
        axes[1, 1].set_ylabel("P(q)")
        axes[1, 1].grid(alpha=0.25, which="both")
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "Ratio channel unavailable", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig.suptitle("Pseudo Time-Series Diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_time_series.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        DATA_DIR / "03_time_series_metrics.npz",
        times=times,
        total_counts=total_counts,
        ratio_mean=ratio_mean,
        scaling_exponent=scaling,
    )

    print(f"Saved plot: {PLOTS_DIR / '03_time_series.png'}")
    print(f"Saved data: {DATA_DIR / '03_time_series_metrics.npz'}")
    return True


def main(test_file: str | None = None) -> bool:
    return test_time_series(test_file=test_file)


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
