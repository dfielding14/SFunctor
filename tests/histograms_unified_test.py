"""Unit tests for unified histogram kernels and aggregation helpers."""

from __future__ import annotations

import numpy as np

from sfunctor.core.histograms import Channel, N_CHANNELS, compute_histogram_for_disp_2D
from sfunctor.core.parallel import compute_histograms_shared
from sfunctor.core.physics import compute_vA, compute_z_plus_minus


def _make_synthetic_fields(shape: tuple[int, int], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    rho = np.full(shape, 1.0, dtype=float)

    # Small-amplitude fields so all Δ magnitudes fit in simple linear bins.
    v_x = rng.normal(0.0, 0.1, size=shape)
    v_y = rng.normal(0.0, 0.1, size=shape)
    v_z = rng.normal(0.0, 0.1, size=shape)

    B_x = 1.0 + rng.normal(0.0, 0.01, size=shape)
    B_y = rng.normal(0.0, 0.01, size=shape)
    B_z = rng.normal(0.0, 0.01, size=shape)

    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (zp_x, zp_y, zp_z), (zm_x, zm_y, zm_z) = compute_z_plus_minus(
        v_x, v_y, v_z, vA_x, vA_y, vA_z
    )

    zeros = np.zeros(shape, dtype=float)

    return {
        "rho": rho,
        "v_x": v_x,
        "v_y": v_y,
        "v_z": v_z,
        "B_x": B_x,
        "B_y": B_y,
        "B_z": B_z,
        "vA_x": vA_x,
        "vA_y": vA_y,
        "vA_z": vA_z,
        "zp_x": zp_x,
        "zp_y": zp_y,
        "zp_z": zp_z,
        "zm_x": zm_x,
        "zm_y": zm_y,
        "zm_z": zm_z,
        "omega_x": zeros,
        "omega_y": zeros,
        "omega_z": zeros,
        "j_x": zeros,
        "j_y": zeros,
        "j_z": zeros,
        "curv_x": zeros,
        "curv_y": zeros,
        "curv_z": zeros,
        "grad_rho_x": zeros,
        "grad_rho_y": zeros,
        "grad_rho_z": zeros,
    }


def test_compute_histogram_for_disp_2d_shape_and_counts():
    fields = _make_synthetic_fields((32, 32), seed=1)

    # Displacement (dx=1, dy=0) on an xy slice (axis=3) keeps theta/phi away from edge cases.
    delta_i, delta_j, axis = 1, 0, 3
    n_samples = 64

    ell_bin_edges = np.array([0.0, 1.5, 4.0], dtype=float)  # r=1 lands in bin 0
    theta_bin_edges = np.linspace(0.0, np.pi / 2, 5, dtype=float)
    phi_bin_edges = np.linspace(0.0, np.pi / 2, 5, dtype=float)

    delta_edges = np.linspace(0.0, 2.0, 33, dtype=float)
    delta_bin_edges = [delta_edges] * N_CHANNELS

    hist = compute_histogram_for_disp_2D(
        fields["v_x"],
        fields["v_y"],
        fields["v_z"],
        fields["B_x"],
        fields["B_y"],
        fields["B_z"],
        fields["rho"],
        fields["vA_x"],
        fields["vA_y"],
        fields["vA_z"],
        fields["zp_x"],
        fields["zp_y"],
        fields["zp_z"],
        fields["zm_x"],
        fields["zm_y"],
        fields["zm_z"],
        fields["omega_x"],
        fields["omega_y"],
        fields["omega_z"],
        fields["j_x"],
        fields["j_y"],
        fields["j_z"],
        fields["curv_x"],
        fields["curv_y"],
        fields["curv_z"],
        fields["grad_rho_x"],
        fields["grad_rho_y"],
        fields["grad_rho_z"],
        delta_i,
        delta_j,
        axis,
        n_samples,
        ell_bin_edges,
        theta_bin_edges,
        phi_bin_edges,
        delta_bin_edges,
        stencil_width=2,
    )

    assert hist.dtype == np.int64
    assert hist.shape == (
        N_CHANNELS,
        ell_bin_edges.shape[0] - 1,
        theta_bin_edges.shape[0] - 1,
        phi_bin_edges.shape[0] - 1,
        delta_edges.shape[0] - 1,
    )
    assert hist.sum() > 0
    assert hist[Channel.D_V.value].sum() > 0


def test_compute_histograms_shared_single_process_runs():
    fields = _make_synthetic_fields((32, 32), seed=2)

    displacements = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.int32)
    ell_bin_edges = np.array([0.0, 1.5, 2.5, 4.5], dtype=float)
    theta_bin_edges = np.linspace(0.0, np.pi / 2, 5, dtype=float)
    phi_bin_edges = np.linspace(0.0, np.pi / 2, 5, dtype=float)

    delta_edges = np.linspace(0.0, 2.0, 33, dtype=float)
    delta_bin_edges = [delta_edges] * N_CHANNELS

    hist = compute_histograms_shared(
        fields,
        displacements,
        axis=3,
        N_random_subsamples=16,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        delta_bin_edges=delta_bin_edges,
        stencil_width=2,
        n_processes=1,
    )

    assert hist.dtype == np.int64
    assert hist.shape == (
        N_CHANNELS,
        ell_bin_edges.shape[0] - 1,
        theta_bin_edges.shape[0] - 1,
        phi_bin_edges.shape[0] - 1,
        delta_edges.shape[0] - 1,
    )
    assert hist.sum() > 0
