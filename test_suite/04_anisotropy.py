#!/usr/bin/env python3
"""Test 04: scale-dependent anisotropy diagnostics on unified histograms."""

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

DEFAULT_TEST_FILE = "slice_data/Turb_320_beta100_dedt025_plm_axis2_slice0_file0000.npz"
RESULTS_DIR = PROJECT_ROOT / "test_suite" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _infer_axis(path: Path) -> int:
    if "_axis" not in path.name:
        return 3
    try:
        return int(path.name.split("_axis", maxsplit=1)[1].split("_", maxsplit=1)[0])
    except Exception:  # noqa: BLE001
        return 3


def _angular_cube(hist_channel: np.ndarray) -> np.ndarray:
    """Return angular counts C(ell, theta, phi) by summing over delta."""
    return hist_channel.sum(axis=3)


def _parallel_perp_ratio(ang: np.ndarray, theta_edges: np.ndarray) -> np.ndarray:
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    parallel = theta_centers <= np.deg2rad(15.0)
    perp = theta_centers >= np.deg2rad(75.0)

    c_theta = ang.sum(axis=2)
    num = c_theta[:, parallel].sum(axis=1)
    den = c_theta[:, perp].sum(axis=1)
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)


def _angular_entropy(ang: np.ndarray) -> np.ndarray:
    flat = ang.reshape(ang.shape[0], -1)
    totals = flat.sum(axis=1, keepdims=True)
    p = np.divide(flat, totals, out=np.zeros_like(flat, dtype=float), where=totals > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.log(p + 1e-30)).sum(axis=1)
    return ent


def _anisotropy_strength(ang: np.ndarray) -> np.ndarray:
    """Simple anisotropy index from theta-marginal variance."""
    c_theta = ang.sum(axis=2)
    totals = c_theta.sum(axis=1, keepdims=True)
    p_theta = np.divide(c_theta, totals, out=np.zeros_like(c_theta, dtype=float), where=totals > 0)
    uniform = 1.0 / p_theta.shape[1]
    return np.sqrt(((p_theta - uniform) ** 2).sum(axis=1))


def _preferential_theta_phi(ang: np.ndarray, theta_edges: np.ndarray, phi_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    pref_theta = np.full(ang.shape[0], np.nan, dtype=float)
    pref_phi = np.full(ang.shape[0], np.nan, dtype=float)

    for i in range(ang.shape[0]):
        if ang[i].sum() <= 0:
            continue
        idx = np.argmax(ang[i])
        th_i, ph_i = np.unravel_index(idx, ang[i].shape)
        pref_theta[i] = theta_centers[th_i]
        pref_phi[i] = phi_centers[ph_i]

    return pref_theta, pref_phi


def _svd_modes(ang: np.ndarray, n_modes: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD over ell of angular distributions."""
    mat = ang.reshape(ang.shape[0], -1)
    totals = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, totals, out=np.zeros_like(mat, dtype=float), where=totals > 0)
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    n = min(n_modes, len(s))
    variance = (s[:n] ** 2) / np.sum(s**2)
    modes = vt[:n].reshape(n, ang.shape[1], ang.shape[2])
    return variance, u[:, :n], modes


def test_anisotropy(test_file: str | None = None) -> bool:
    file_name = test_file or DEFAULT_TEST_FILE
    path = (PROJECT_ROOT / file_name).resolve()
    if not path.exists():
        print(f"Missing test file: {path}")
        return False

    print("=" * 60)
    print("TEST 04: SCALE-DEPENDENT ANISOTROPY")
    print(f"Data: {path.name}")
    print("=" * 60)

    stride = 4 if "2560" in path.name else 2
    n_disp = 300
    n_samples = 2500

    load_start = time.time()
    data = load_slice_npz(path, stride=stride)
    print(f"Loaded shape: {data['rho'].shape} in {time.time() - load_start:.2f}s")

    run_start = time.time()
    results = analyze_slice(
        data,
        n_displacements=n_disp,
        n_ell_bins=28,
        n_random_subsamples=n_samples,
        stencil_width=2,
        n_processes=1,
        axis=_infer_axis(path),
    )
    print(f"Analysis in {time.time() - run_start:.2f}s")

    hist = results["hist"]
    channels = list(results["channels"])
    ell_edges = np.asarray(results["ell_bin_edges"], dtype=float)
    theta_edges = np.asarray(results["theta_bin_edges"], dtype=float)
    phi_edges = np.asarray(results["phi_bin_edges"], dtype=float)
    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])

    channel_subset = [c for c in ["D_V", "D_B", "D_ZPLUS"] if c in channels]
    diagnostics: dict[str, dict[str, np.ndarray]] = {}

    for name in channel_subset:
        idx = channels.index(name)
        ang = _angular_cube(hist[idx])
        ratio = _parallel_perp_ratio(ang, theta_edges)
        entropy = _angular_entropy(ang)
        strength = _anisotropy_strength(ang)
        pref_theta, pref_phi = _preferential_theta_phi(ang, theta_edges, phi_edges)
        diagnostics[name] = {
            "ratio": ratio,
            "entropy": entropy,
            "strength": strength,
            "pref_theta": pref_theta,
            "pref_phi": pref_phi,
            "angular": ang,
        }

    if not diagnostics:
        print("No expected channels found for anisotropy diagnostics")
        return False

    # Use D_V as the reference channel for spectrogram/modes when available.
    ref_name = "D_V" if "D_V" in diagnostics else next(iter(diagnostics))
    ref_ang = diagnostics[ref_name]["angular"]
    spectrogram = np.divide(
        ref_ang.sum(axis=2),
        ref_ang.sum(axis=(1, 2), keepdims=False)[:, None],
        out=np.zeros((ref_ang.shape[0], ref_ang.shape[1]), dtype=float),
        where=ref_ang.sum(axis=(1, 2), keepdims=False)[:, None] > 0,
    )
    variance, mode_amplitude, modes = _svd_modes(ref_ang, n_modes=3)

    fig = plt.figure(figsize=(16, 11))

    ax1 = fig.add_subplot(3, 3, 1)
    for name in channel_subset:
        ax1.semilogx(ell_centers, diagnostics[name]["ratio"], "o-", label=name)
    ax1.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax1.set_title("Parallel/Perp Ratio")
    ax1.set_xlabel(r"$\ell$")
    ax1.set_ylabel("ratio")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(3, 3, 2)
    for name in channel_subset:
        ax2.semilogx(ell_centers, diagnostics[name]["entropy"], "o-", label=name)
    ax2.set_title("Angular Entropy")
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel("entropy")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(3, 3, 3)
    for name in channel_subset:
        ax3.semilogx(ell_centers, diagnostics[name]["strength"], "o-", label=name)
    ax3.set_title("Anisotropy Strength")
    ax3.set_xlabel(r"$\ell$")
    ax3.set_ylabel("index")
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(3, 3, 4)
    for name in channel_subset:
        ax4.semilogx(ell_centers, np.rad2deg(diagnostics[name]["pref_theta"]), "o-", label=name)
    ax4.set_title("Preferential Theta")
    ax4.set_xlabel(r"$\ell$")
    ax4.set_ylabel("deg")
    ax4.grid(alpha=0.25)

    ax5 = fig.add_subplot(3, 3, 5)
    for name in channel_subset:
        ax5.semilogx(ell_centers, np.rad2deg(diagnostics[name]["pref_phi"]), "o-", label=name)
    ax5.set_title("Preferential Phi")
    ax5.set_xlabel(r"$\ell$")
    ax5.set_ylabel("deg")
    ax5.grid(alpha=0.25)

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.bar(np.arange(1, len(variance) + 1), variance)
    ax6.set_title("SVD Variance Explained")
    ax6.set_xlabel("mode")
    ax6.set_ylabel("fraction")
    ax6.grid(alpha=0.25)

    ax7 = fig.add_subplot(3, 3, 7)
    theta_centers_deg = np.rad2deg(0.5 * (theta_edges[:-1] + theta_edges[1:]))
    pcm = ax7.pcolormesh(ell_edges, theta_edges, spectrogram.T, shading="auto", cmap="viridis")
    ax7.set_xscale("log")
    ax7.set_ylim(theta_edges.min(), theta_edges.max())
    ax7.set_title(f"Theta Spectrogram ({ref_name})")
    ax7.set_xlabel(r"$\ell$")
    ax7.set_ylabel(r"$\theta$ (rad)")
    fig.colorbar(pcm, ax=ax7)

    ax8 = fig.add_subplot(3, 3, 8)
    if modes.shape[0] > 0:
        im = ax8.imshow(modes[0], cmap="RdBu_r", origin="lower", aspect="auto")
        ax8.set_title("Angular Mode 1")
        ax8.set_xlabel("phi bin")
        ax8.set_ylabel("theta bin")
        fig.colorbar(im, ax=ax8)

    ax9 = fig.add_subplot(3, 3, 9)
    for i in range(mode_amplitude.shape[1]):
        ax9.semilogx(ell_centers, mode_amplitude[:, i], label=f"mode {i+1}")
    ax9.set_title("Mode Amplitudes vs Scale")
    ax9.set_xlabel(r"$\ell$")
    ax9.set_ylabel("amplitude")
    ax9.grid(alpha=0.25)
    ax9.legend(fontsize=8)

    fig.suptitle("Anisotropy Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_anisotropy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        DATA_DIR / "04_anisotropy_metrics.npz",
        ell_centers=ell_centers,
        channel_subset=np.array(channel_subset, dtype=object),
        variance_explained=variance,
        spectrogram=spectrogram,
    )

    print(f"Saved plot: {PLOTS_DIR / '04_anisotropy.png'}")
    print(f"Saved data: {DATA_DIR / '04_anisotropy_metrics.npz'}")
    return True


def main(test_file: str | None = None) -> bool:
    return test_anisotropy(test_file=test_file)


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
