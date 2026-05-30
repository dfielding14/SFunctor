"""Unit tests for unified histogram kernels and aggregation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from sfunctor.core.histograms import (
    CENSOR_NAMES,
    Channel,
    N_CHANNELS,
    compute_histogram_for_disp_2D,
    find_bin_index_binary,
)
from sfunctor.core.parallel import compute_histograms_shared
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.reference import compute_unified_channel_values_reference


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


def _compute_histogram(
    fields: dict[str, np.ndarray],
    *,
    delta_i: int,
    delta_j: int,
    axis: int,
    stencil_width: int = 2,
    n_samples: int = 8,
) -> np.ndarray:
    ell_bin_edges = np.array([0.0, 2.0], dtype=float)
    theta_bin_edges = np.array([0.0, np.pi / 4, np.pi / 2], dtype=float)
    phi_bin_edges = np.array([0.0, np.pi / 2], dtype=float)
    delta_edges = np.linspace(0.0, 10.0, 33, dtype=float)
    delta_bin_edges = [delta_edges] * N_CHANNELS

    return compute_histogram_for_disp_2D(
        fields["v_x"], fields["v_y"], fields["v_z"],
        fields["B_x"], fields["B_y"], fields["B_z"],
        fields["rho"],
        fields["vA_x"], fields["vA_y"], fields["vA_z"],
        fields["zp_x"], fields["zp_y"], fields["zp_z"],
        fields["zm_x"], fields["zm_y"], fields["zm_z"],
        fields["omega_x"], fields["omega_y"], fields["omega_z"],
        fields["j_x"], fields["j_y"], fields["j_z"],
        fields["curv_x"], fields["curv_y"], fields["curv_z"],
        fields["grad_rho_x"], fields["grad_rho_y"], fields["grad_rho_z"],
        delta_i, delta_j, axis, n_samples,
        ell_bin_edges, theta_bin_edges, phi_bin_edges, delta_bin_edges,
        stencil_width=stencil_width,
    )


def _set_constant_guide_field(fields: dict[str, np.ndarray], guide: tuple[float, float, float]) -> None:
    for name, value in zip(("B_x", "B_y", "B_z"), guide, strict=True):
        fields[name] = np.full_like(fields[name], value)


def test_find_bin_index_binary_includes_final_edge():
    edges = np.array([0.0, np.pi / 4, np.pi / 2])
    assert find_bin_index_binary(np.pi / 2, edges) == 1


@pytest.mark.parametrize("stencil_width", [2, 3, 5])
@pytest.mark.parametrize(
    ("axis", "delta_i", "delta_j", "guide"),
    [
        (1, 1, 0, (0.0, 1.0, 0.0)),  # axis1 plane is (k=z, j=y)
        (1, 0, 1, (0.0, 0.0, 1.0)),
        (2, 1, 0, (1.0, 0.0, 0.0)),  # axis2 plane is (k=z, i=x)
        (2, 0, 1, (0.0, 0.0, 1.0)),
        (3, 1, 0, (1.0, 0.0, 0.0)),  # axis3 plane is (j=y, i=x)
        (3, 0, 1, (0.0, 1.0, 0.0)),
    ],
)
def test_kji_displacement_mapping_aligns_with_physical_guide_field(axis, delta_i, delta_j, guide, stencil_width):
    fields = _make_synthetic_fields((8, 8), seed=axis * 10 + stencil_width)
    _set_constant_guide_field(fields, guide)
    hist = _compute_histogram(
        fields,
        delta_i=delta_i,
        delta_j=delta_j,
        axis=axis,
        stencil_width=stencil_width,
    )

    dv_hist = hist[Channel.D_V.value]
    assert dv_hist.sum() == 8
    assert dv_hist[:, 0, :, :].sum() == 8


def test_perpendicular_theta_endpoint_is_retained():
    fields = _make_synthetic_fields((8, 8), seed=4)
    _set_constant_guide_field(fields, (1.0, 0.0, 0.0))
    hist = _compute_histogram(fields, delta_i=0, delta_j=1, axis=3)

    dv_hist = hist[Channel.D_V.value]
    assert dv_hist.sum() == 8
    assert dv_hist[:, -1, :, :].sum() == 8


def test_histogram_rejects_invalid_slice_axis():
    fields = _make_synthetic_fields((8, 8), seed=5)
    with pytest.raises(ValueError, match="slice_axis must be 1, 2, or 3"):
        _compute_histogram(fields, delta_i=1, delta_j=0, axis=0)


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


def test_histogram_censoring_records_accepted_underflow_overflow_and_invalid():
    fields = _make_synthetic_fields((16, 16), seed=21)
    delta_bin_edges = [np.array([0.0, 100.0])] * N_CHANNELS
    delta_bin_edges = list(delta_bin_edges)
    delta_bin_edges[Channel.D_V] = np.array([0.0, 1.0e-12])
    delta_bin_edges[Channel.D_OMEGA] = np.array([0.01, 100.0])
    hist, censoring = compute_histogram_for_disp_2D(
        fields["v_x"], fields["v_y"], fields["v_z"],
        fields["B_x"], fields["B_y"], fields["B_z"], fields["rho"],
        fields["vA_x"], fields["vA_y"], fields["vA_z"],
        fields["zp_x"], fields["zp_y"], fields["zp_z"],
        fields["zm_x"], fields["zm_y"], fields["zm_z"],
        fields["omega_x"], fields["omega_y"], fields["omega_z"],
        fields["j_x"], fields["j_y"], fields["j_z"],
        fields["curv_x"], fields["curv_y"], fields["curv_z"],
        fields["grad_rho_x"], fields["grad_rho_y"], fields["grad_rho_z"],
        0, 1, 3, 16,
        np.array([0.0, 2.0]),
        np.array([0.0, np.pi / 2]),
        np.array([0.0, np.pi / 2]),
        delta_bin_edges,
        random_seed=7,
        return_censoring=True,
    )
    censor_index = {name: index for index, name in enumerate(CENSOR_NAMES)}
    assert np.array_equal(hist.sum(axis=-1), censoring[..., censor_index["accepted"]])
    assert censoring[Channel.D_V, ..., censor_index["overflow"]].sum() == 16
    assert censoring[Channel.D_OMEGA, ..., censor_index["underflow"]].sum() == 16
    assert censoring[Channel.D_Vperp_D_Omegaperp_CROSS_MAG_RATIO, ..., censor_index["invalid"]].sum() == 16


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


def test_compute_histograms_shared_uses_physical_cell_sizes_for_ell_bins():
    fields = _make_synthetic_fields((8, 8), seed=3)
    displacements = np.array([[1, 1]], dtype=np.int32)
    ell_bin_edges = np.array([0.0, 2.0, 4.0], dtype=float)
    angle_edges = np.array([0.0, np.pi / 2], dtype=float)
    delta_edges = np.linspace(0.0, 10.0, 33, dtype=float)

    hist = compute_histograms_shared(
        fields,
        displacements,
        axis=3,
        N_random_subsamples=8,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=angle_edges,
        phi_bin_edges=angle_edges,
        delta_bin_edges=[delta_edges] * N_CHANNELS,
        n_processes=1,
        cell_sizes=(1.0, 3.0, 1.0),
        random_seed=5,
    )
    # The physical length is sqrt(1^2 + 3^2), not the pixel length sqrt(2).
    assert hist[Channel.D_V.value, 0].sum() == 0
    assert hist[Channel.D_V.value, 1].sum() == 8


def test_seeded_shared_histograms_are_process_count_invariant():
    fields = _make_synthetic_fields((16, 16), seed=7)
    displacements = np.array([[1, 0], [0, 1], [2, 1], [-1, 2]], dtype=np.int32)
    ell_bin_edges = np.array([0.0, 1.5, 2.5, 4.0], dtype=float)
    angle_edges = np.linspace(0.0, np.pi / 2, 4, dtype=float)
    delta_edges = np.linspace(0.0, 10.0, 33, dtype=float)
    kwargs = dict(
        fields=fields,
        displacements=displacements,
        axis=3,
        N_random_subsamples=32,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=angle_edges,
        phi_bin_edges=angle_edges,
        delta_bin_edges=[delta_edges] * N_CHANNELS,
        random_seed=1234,
    )

    single = compute_histograms_shared(**kwargs, n_processes=1)
    parallel = compute_histograms_shared(**kwargs, n_processes=2)
    assert np.array_equal(single, parallel)


def test_seeded_shared_censoring_is_process_count_invariant():
    fields = _make_synthetic_fields((16, 16), seed=17)
    kwargs = dict(
        fields=fields,
        displacements=np.array([[1, 0], [0, 1], [2, 1], [-1, 2]], dtype=np.int32),
        axis=3,
        N_random_subsamples=32,
        ell_bin_edges=np.array([0.0, 1.5, 2.5, 4.0]),
        theta_bin_edges=np.linspace(0.0, np.pi / 2, 4),
        phi_bin_edges=np.linspace(0.0, np.pi / 2, 4),
        delta_bin_edges=[np.linspace(0.01, 0.2, 8)] * N_CHANNELS,
        random_seed=71,
        return_censoring=True,
    )
    single = compute_histograms_shared(**kwargs, n_processes=1)
    parallel = compute_histograms_shared(**kwargs, n_processes=2)
    assert np.array_equal(single[0], parallel[0])
    assert np.array_equal(single[1], parallel[1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"displacements": np.array([[0, 0]])}, "zero displacement"),
        ({"N_random_subsamples": 0}, "positive integer"),
        ({"n_processes": -1}, "positive or zero"),
        ({"cell_sizes": (1.0, 0.0, 1.0)}, "finite positive"),
    ],
)
def test_compute_histograms_shared_rejects_invalid_runtime_inputs(kwargs, message):
    fields = _make_synthetic_fields((8, 8), seed=9)
    arguments = dict(
        fields=fields,
        displacements=np.array([[1, 0]], dtype=np.int32),
        axis=3,
        N_random_subsamples=8,
        ell_bin_edges=np.array([0.0, 2.0]),
        theta_bin_edges=np.array([0.0, np.pi / 2]),
        phi_bin_edges=np.array([0.0, np.pi / 2]),
        delta_bin_edges=[np.linspace(0.0, 10.0, 8)] * N_CHANNELS,
        n_processes=1,
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        compute_histograms_shared(**arguments)


def test_all_legacy_channels_match_readable_reference_values():
    shape = (8, 8)
    sign = np.tile((-1.0) ** np.arange(shape[1]), (shape[0], 1))
    zeros = np.zeros(shape)

    def scalar(value):
        return value * sign

    def components(values):
        return tuple(scalar(value) for value in values)

    v = components((1.0, 2.0, 0.0))
    B = (scalar(2.0), scalar(-1.0), np.full(shape, 10.0))
    vA = components((0.5, 1.0, 0.0))
    zp = components((1.0, 1.0, 0.0))
    zm = components((1.0, -2.0, 0.0))
    omega = components((1.0, -0.5, 0.0))
    j = components((0.25, 1.0, 0.0))
    curv = components((0.2, 0.3, 0.0))
    grad_rho = components((0.4, -0.1, 0.0))
    fields = {
        "v_x": v[0], "v_y": v[1], "v_z": v[2],
        "B_x": B[0], "B_y": B[1], "B_z": B[2],
        "rho": 2.0 + scalar(0.25),
        "vA_x": vA[0], "vA_y": vA[1], "vA_z": vA[2],
        "zp_x": zp[0], "zp_y": zp[1], "zp_z": zp[2],
        "zm_x": zm[0], "zm_y": zm[1], "zm_z": zm[2],
        "omega_x": omega[0], "omega_y": omega[1], "omega_z": omega[2],
        "j_x": j[0], "j_y": j[1], "j_z": j[2],
        "curv_x": curv[0], "curv_y": curv[1], "curv_z": curv[2],
        "grad_rho_x": grad_rho[0], "grad_rho_y": grad_rho[1], "grad_rho_z": grad_rho[2],
    }
    delta_edges = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    hist = compute_histogram_for_disp_2D(
        fields["v_x"], fields["v_y"], fields["v_z"],
        fields["B_x"], fields["B_y"], fields["B_z"], fields["rho"],
        fields["vA_x"], fields["vA_y"], fields["vA_z"],
        fields["zp_x"], fields["zp_y"], fields["zp_z"],
        fields["zm_x"], fields["zm_y"], fields["zm_z"],
        fields["omega_x"], fields["omega_y"], fields["omega_z"],
        fields["j_x"], fields["j_y"], fields["j_z"],
        fields["curv_x"], fields["curv_y"], fields["curv_z"],
        fields["grad_rho_x"], fields["grad_rho_y"], fields["grad_rho_z"],
        1, 0, 3, 64,
        np.array([0.0, 2.0]),
        np.array([0.0, np.pi / 2]),
        np.array([0.0, np.pi / 2]),
        [delta_edges] * N_CHANNELS,
        random_seed=5,
    )
    expected = compute_unified_channel_values_reference(
        {
            "v": 2.0 * np.array((1.0, 2.0, 0.0)),
            "B": 2.0 * np.array((2.0, -1.0, 0.0)),
            "rho": 0.5,
            "vA": 2.0 * np.array((0.5, 1.0, 0.0)),
            "zp": 2.0 * np.array((1.0, 1.0, 0.0)),
            "zm": 2.0 * np.array((1.0, -2.0, 0.0)),
            "omega": 2.0 * np.array((1.0, -0.5, 0.0)),
            "j": 2.0 * np.array((0.25, 1.0, 0.0)),
            "curv": 2.0 * np.array((0.2, 0.3, 0.0)),
            "grad_rho": 2.0 * np.array((0.4, -0.1, 0.0)),
        },
        np.array((0.0, 0.0, 10.0)),
    )
    assert set(expected) == {channel.name for channel in Channel}
    for channel in Channel:
        bin_index = find_bin_index_binary(expected[channel.name], delta_edges)
        assert bin_index >= 0, (channel.name, expected[channel.name])
        assert hist[channel.value, 0, 0, 0, bin_index] == 64, channel.name
        assert hist[channel.value].sum() == 64, channel.name
