#!/usr/bin/env python3
"""Test 02: Physics-field calculations compatible with the current API."""

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
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.io.slice_io import load_slice_npz

TEST_FILE = "slice_data/Turb_320_beta100_dedt025_plm_axis2_slice0_file0000.npz"
RESULTS_DIR = PROJECT_ROOT / "test_suite" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _grad_xy(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return gradients (d/dy, d/dx) on the 2D slice grid."""
    d_dy = np.gradient(arr, axis=0)
    d_dx = np.gradient(arr, axis=1)
    return d_dy, d_dx


def _compute_derived_fields(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute derived physics fields using 2D finite differences."""
    rho = data["rho"]
    v_x = data["v_x"]
    v_y = data["v_y"]
    v_z = data["v_z"]
    B_x = data["B_x"]
    B_y = data["B_y"]
    B_z = data["B_z"]

    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (zp_x, zp_y, zp_z), (zm_x, zm_y, zm_z) = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)

    dvz_dy, dvz_dx = _grad_xy(v_z)
    dvy_dy, dvy_dx = _grad_xy(v_y)
    dvx_dy, dvx_dx = _grad_xy(v_x)

    dBz_dy, dBz_dx = _grad_xy(B_z)
    dBy_dy, dBy_dx = _grad_xy(B_y)
    dBx_dy, dBx_dx = _grad_xy(B_x)

    omega_x = dvz_dy
    omega_y = -dvz_dx
    omega_z = dvy_dx - dvx_dy

    j_x = dBz_dy
    j_y = -dBz_dx
    j_z = dBy_dx - dBx_dy

    drho_dy, drho_dx = _grad_xy(rho)
    grad_rho_x = drho_dx
    grad_rho_y = drho_dy
    grad_rho_z = np.zeros_like(rho)

    # Approximate curvature K=(b·∇)b on the slice using x/y derivatives.
    B_mag = np.sqrt(B_x * B_x + B_y * B_y + B_z * B_z)
    safe = np.maximum(B_mag, 1e-12)
    b_x = B_x / safe
    b_y = B_y / safe
    b_z = B_z / safe

    dbx_dy, dbx_dx = _grad_xy(b_x)
    dby_dy, dby_dx = _grad_xy(b_y)
    dbz_dy, dbz_dx = _grad_xy(b_z)

    curv_x = b_x * dbx_dx + b_y * dbx_dy
    curv_y = b_x * dby_dx + b_y * dby_dy
    curv_z = b_x * dbz_dx + b_y * dbz_dy

    return {
        "vA_x": vA_x,
        "vA_y": vA_y,
        "vA_z": vA_z,
        "zp_x": zp_x,
        "zp_y": zp_y,
        "zp_z": zp_z,
        "zm_x": zm_x,
        "zm_y": zm_y,
        "zm_z": zm_z,
        "omega_x": omega_x,
        "omega_y": omega_y,
        "omega_z": omega_z,
        "j_x": j_x,
        "j_y": j_y,
        "j_z": j_z,
        "curv_x": curv_x,
        "curv_y": curv_y,
        "curv_z": curv_z,
        "grad_rho_x": grad_rho_x,
        "grad_rho_y": grad_rho_y,
        "grad_rho_z": grad_rho_z,
    }


def _norm3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt(x * x + y * y + z * z)


def test_physics_calculations(test_file: str | None = None) -> bool:
    print("=" * 60)
    print("TEST 02: PHYSICS CALCULATIONS")
    print("=" * 60)

    file_name = test_file or TEST_FILE
    data_path = PROJECT_ROOT / file_name
    if not data_path.exists():
        print(f"Missing input file: {data_path}")
        return False

    print(f"Data: {file_name}")
    load_start = time.time()
    data = load_slice_npz(data_path, stride=2)
    print(f"Loaded shape: {data['rho'].shape} in {time.time() - load_start:.2f}s")

    start = time.time()
    derived = _compute_derived_fields(data)
    elapsed = time.time() - start
    print(f"Computed derived fields in {elapsed:.2f}s")

    omega_mag = _norm3(derived["omega_x"], derived["omega_y"], derived["omega_z"])
    j_mag = _norm3(derived["j_x"], derived["j_y"], derived["j_z"])
    curv_mag = _norm3(derived["curv_x"], derived["curv_y"], derived["curv_z"])
    grad_rho_mag = _norm3(derived["grad_rho_x"], derived["grad_rho_y"], derived["grad_rho_z"])

    print(f"|omega| range: [{omega_mag.min():.3e}, {omega_mag.max():.3e}]")
    print(f"|j| range: [{j_mag.min():.3e}, {j_mag.max():.3e}]")
    print(f"|curv| range: [{curv_mag.min():.3e}, {curv_mag.max():.3e}]")
    print(f"|grad_rho| range: [{grad_rho_mag.min():.3e}, {grad_rho_mag.max():.3e}]")

    # Add derived fields and run unified histogram analysis.
    full_data = dict(data)
    full_data.update({k: v for k, v in derived.items() if k in {
        "omega_x", "omega_y", "omega_z",
        "j_x", "j_y", "j_z",
        "curv_x", "curv_y", "curv_z",
        "grad_rho_x", "grad_rho_y", "grad_rho_z",
    }})

    results = analyze_slice(
        full_data,
        n_displacements=250,
        n_ell_bins=24,
        n_random_subsamples=2000,
        stencil_width=2,
        n_processes=1,
        axis=2,
    )

    hist = results["hist"]
    channels = list(results["channels"])
    ell_edges = np.asarray(results["ell_bin_edges"], dtype=float)
    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    delta_edges = [np.asarray(e, dtype=float) for e in results["delta_bin_edges"]]

    # ------------------------------------------------------------------
    # Plot 1: field snapshots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fields = [
        (data["rho"], "rho", "viridis"),
        (data["v_x"], "v_x", "RdBu_r"),
        (data["B_x"], "B_x", "RdBu_r"),
        (_norm3(data["B_x"], data["B_y"], data["B_z"]), "|B|", "plasma"),
        (omega_mag, "|omega|", "magma"),
        (j_mag, "|j|", "magma"),
        (curv_mag, "|curv|", "cividis"),
        (grad_rho_mag, "|grad rho|", "cividis"),
        (_norm3(derived["zp_x"], derived["zp_y"], derived["zp_z"]), "|z+|", "viridis"),
        (_norm3(derived["zm_x"], derived["zm_y"], derived["zm_z"]), "|z-|", "viridis"),
        (_norm3(derived["vA_x"], derived["vA_y"], derived["vA_z"]), "|vA|", "viridis"),
        (data["v_z"], "v_z", "RdBu_r"),
    ]

    for ax, (arr, title, cmap) in zip(axes.ravel(), fields, strict=True):
        im = ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Physics Field Snapshots", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_field_snapshots.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: PDFs
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    pdf_fields = [
        (data["v_x"], "v_x"),
        (data["B_x"], "B_x"),
        (omega_mag, "|omega|"),
        (j_mag, "|j|"),
        (curv_mag, "|curv|"),
        (grad_rho_mag, "|grad rho|"),
    ]

    for ax, (arr, label) in zip(axes.ravel(), pdf_fields, strict=True):
        hist_pdf, bins = np.histogram(arr.ravel(), bins=80, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        valid = hist_pdf > 0
        ax.semilogy(centers[valid], hist_pdf[valid], lw=1.7)
        ax.set_title(label)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel(label)
        ax.set_ylabel("PDF")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_field_pdfs.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: Derived-channel structure functions
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for name in ["D_OMEGA", "D_J", "D_CURV", "D_GRAD_RHO"]:
        if name not in channels:
            continue
        idx = channels.index(name)
        counts = hist[idx].sum(axis=(1, 2))
        centers = np.sqrt(delta_edges[idx][:-1] * delta_edges[idx][1:])
        totals = counts.sum(axis=1)
        mean_delta = np.divide((counts * centers[None, :]).sum(axis=1), totals, out=np.full_like(totals, np.nan, dtype=float), where=totals > 0)
        valid = np.isfinite(mean_delta) & (mean_delta > 0)
        if np.any(valid):
            plt.loglog(ell_centers[valid], mean_delta[valid], "o-", label=name)

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\langle\delta\rangle$")
    plt.title("Derived-Field Mean Increments")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_derived_structure_functions.png", dpi=170, bbox_inches="tight")
    plt.close()

    np.savez_compressed(
        DATA_DIR / "02_physics_fields.npz",
        omega_mag=omega_mag,
        j_mag=j_mag,
        curv_mag=curv_mag,
        grad_rho_mag=grad_rho_mag,
        hist=hist,
        channels=np.array(channels, dtype=object),
        ell_bin_edges=ell_edges,
        delta_bin_edges=np.array(delta_edges, dtype=object),
    )

    print("Generated plots:")
    for plot in sorted(PLOTS_DIR.glob("02_*.png")):
        print(f"  - {plot.name}")
    print(f"Saved data: {DATA_DIR / '02_physics_fields.npz'}")
    return True


def main(test_file: str | None = None) -> bool:
    return test_physics_calculations(test_file=test_file)


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
