"""Validation tests for strict pairwise three-direction structure functions."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from sfunctor.analysis.directional import constant_s2_shapes, coverage_rows, fit_directional_slopes
from sfunctor.core.directional import (
    DirectionalConfig,
    build_local_basis,
    build_q_variants,
    compute_directional_structure_functions,
    slice_offset_to_vector,
)
from sfunctor.reference import (
    compute_directional_structure_functions_reference,
    compute_standard_s2_reference,
)


def _slice_data(shape=(7, 9)) -> dict[str, np.ndarray]:
    y, x = np.indices(shape)
    return {
        "rho": 1.0 + 0.2 * np.sin(2.0 * np.pi * x / shape[1]),
        "B_x": 1.0 + 0.1 * np.cos(2.0 * np.pi * y / shape[0]),
        "B_y": 0.2 * np.sin(2.0 * np.pi * x / shape[1]),
        "B_z": 0.1 * np.cos(2.0 * np.pi * x / shape[1]),
        "v_x": 0.2 * np.sin(2.0 * np.pi * y / shape[0]),
        "v_y": np.sin(2.0 * np.pi * x / shape[1]),
        "v_z": np.cos(2.0 * np.pi * y / shape[0]),
    }


@pytest.mark.parametrize(
    ("axis", "displacement", "expected"),
    [
        (1, (2, 3), (0.0, 10.0, 21.0)),
        (2, (2, 3), (6.0, 0.0, 21.0)),
        (3, (2, 3), (6.0, 15.0, 0.0)),
    ],
)
def test_slice_offset_to_vector_uses_athenak_kji_contract(axis, displacement, expected):
    assert np.array_equal(slice_offset_to_vector(axis, displacement, (3.0, 5.0, 7.0)), expected)


def test_build_local_basis_is_orthonormal_and_sign_invariant():
    B_left = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -3.0]])
    B_right = B_left.copy()
    delta_q = np.array([[2.0, 0.0, 1.0], [-4.0, 0.0, 2.0]])
    e_parallel, e_xi, e_lambda, valid = build_local_basis(B_left, B_right, delta_q)

    assert valid.all()
    for basis in (e_parallel, e_xi, e_lambda):
        assert np.allclose(np.linalg.norm(basis, axis=1), 1.0)
    assert np.allclose(np.einsum("ij,ij->i", e_parallel, e_xi), 0.0)
    assert np.allclose(np.einsum("ij,ij->i", e_parallel, e_lambda), 0.0)
    assert np.allclose(np.einsum("ij,ij->i", e_xi, e_lambda), 0.0)
    assert np.allclose(e_lambda, np.cross(e_parallel, e_xi))

    _, reversed_xi, reversed_lambda, _ = build_local_basis(B_left, B_right, -delta_q)
    assert np.allclose(reversed_xi, -e_xi)
    assert np.allclose(reversed_lambda, -e_lambda)


def test_q_variants_keep_parallel_geometry_separate_from_density_dependent_increments():
    data = _slice_data((4, 5))
    B, q_fields, rho0 = build_q_variants(data)
    assert rho0 == pytest.approx(np.mean(data["rho"]))
    assert np.allclose(q_fields["vA_ref"].values, B / np.sqrt(rho0))
    assert not np.allclose(q_fields["vA"].values, q_fields["vA_ref"].values)
    assert np.array_equal(q_fields["B"].valid, q_fields["vA_ref"].valid)


def test_vectorized_directional_calculator_matches_loop_reference():
    data = _slice_data()
    displacements = np.array([[1, 0], [0, 1], [2, 1], [-1, 2]], dtype=int)
    config = DirectionalConfig(
        np.array([0.5, 1.5, 2.5, 5.0]),
        cell_sizes=(1.0, 1.5, 2.0),
        sample_count=31,
        seed=17,
    )
    fast = compute_directional_structure_functions(data, displacements, slice_axis=3, config=config)
    slow = compute_directional_structure_functions_reference(data, displacements, slice_axis=3, config=config)

    assert fast.q_names == slow.q_names
    assert fast.density_conventions == slow.density_conventions
    assert fast.geometry_names == slow.geometry_names
    assert np.array_equal(fast.attempts, slow.attempts)
    assert fast.out_of_range_attempts == slow.out_of_range_attempts
    assert np.array_equal(fast.counts, slow.counts)
    assert np.array_equal(fast.exclusions, slow.exclusions)
    assert np.allclose(fast.sums, slow.sums, rtol=2e-15, atol=2e-15)
    assert np.allclose(fast.sums_sq, slow.sums_sq, rtol=2e-15, atol=2e-15)


def test_phi_degeneracy_is_excluded_without_dropping_parallel_s2():
    shape = (4, 5)
    data = {
        "rho": np.ones(shape),
        "B_x": np.ones(shape),
        "B_y": np.zeros(shape),
        "B_z": np.zeros(shape),
        "v_x": np.zeros(shape),
        "v_y": np.tile(np.arange(shape[1], dtype=float), (shape[0], 1)),
        "v_z": np.zeros(shape),
    }
    config = DirectionalConfig(np.array([0.5, 1.5]), include_global=False)
    result = compute_directional_structure_functions(
        data,
        np.array([[1, 0]]),
        slice_axis=3,
        config=config,
        q_names=("u",),
    )
    directions = {name: index for index, name in enumerate(result.direction_names)}
    exclusions = {name: index for index, name in enumerate(result.exclusion_names)}

    assert result.counts[0, 0, directions["parallel"], 0] == np.prod(shape)
    assert result.counts[0, 0, directions["xi"], 0] == 0
    assert result.exclusions[0, 0, exclusions["weak_r_perp_for_phi"], 0] == np.prod(shape)


def test_invalid_density_and_canceling_local_field_are_counted():
    shape = (2, 3)
    data = {
        "rho": np.ones(shape),
        "B_x": np.tile(np.array([1.0, -1.0, 1.0]), (shape[0], 1)),
        "B_y": np.zeros(shape),
        "B_z": np.zeros(shape),
        "v_x": np.zeros(shape),
        "v_y": np.tile(np.arange(shape[1], dtype=float), (shape[0], 1)),
        "v_z": np.zeros(shape),
    }
    data["rho"][0, 0] = np.nan
    config = DirectionalConfig(np.array([0.5, 1.5]), include_global=False)
    result = compute_directional_structure_functions(
        data,
        np.array([[1, 0]]),
        slice_axis=3,
        config=config,
        q_names=("vA",),
    )
    exclusions = {name: index for index, name in enumerate(result.exclusion_names)}
    assert result.exclusions[0, 0, exclusions["weak_B_direction"], 0] == 4
    assert result.exclusions[0, 0, exclusions["invalid_q"], 0] == 1


def test_entirely_invalid_B_is_excluded_without_empty_mean_warning():
    shape = (2, 3)
    data = {
        "rho": np.ones(shape),
        "B_x": np.full(shape, np.nan),
        "B_y": np.full(shape, np.nan),
        "B_z": np.full(shape, np.nan),
        "v_x": np.zeros(shape),
        "v_y": np.ones(shape),
        "v_z": np.zeros(shape),
    }
    config = DirectionalConfig(np.array([0.5, 1.5]))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fast = compute_directional_structure_functions(
            data, np.array([[1, 0]]), slice_axis=3, config=config, q_names=("u",)
        )
        slow = compute_directional_structure_functions_reference(
            data, np.array([[1, 0]]), slice_axis=3, config=config, q_names=("u",)
        )

    assert not [warning for warning in recorded if issubclass(warning.category, RuntimeWarning)]
    assert np.array_equal(fast.counts, slow.counts)
    assert np.array_equal(fast.exclusions, slow.exclusions)
    assert not fast.counts.any()


def test_xi_and_lambda_classification_use_each_q_increment():
    shape = (8, 8)
    _, x = np.indices(shape)
    base = {
        "rho": np.ones(shape),
        "B_x": np.zeros(shape),
        "B_y": np.zeros(shape),
        "B_z": np.ones(shape),
        "v_x": np.sin(2.0 * np.pi * x / shape[1]),
        "v_y": np.zeros(shape),
        "v_z": np.zeros(shape),
    }
    config = DirectionalConfig(np.array([0.5, 1.5]), include_global=False)
    result = compute_directional_structure_functions(
        base,
        np.array([[1, 0]]),
        slice_axis=3,
        config=config,
        q_names=("u",),
    )
    directions = {name: index for index, name in enumerate(result.direction_names)}
    assert result.counts[0, 0, directions["xi"], 0] > 0
    assert result.counts[0, 0, directions["lambda"], 0] == 0

    rotated = dict(base)
    rotated["v_x"] = np.zeros(shape)
    rotated["v_y"] = base["v_x"]
    result = compute_directional_structure_functions(
        rotated,
        np.array([[1, 0]]),
        slice_axis=3,
        config=config,
        q_names=("u",),
    )
    assert result.counts[0, 0, directions["xi"], 0] == 0
    assert result.counts[0, 0, directions["lambda"], 0] > 0


def test_periodic_translation_and_fixed_seed_are_reproducible():
    data = _slice_data()
    config = DirectionalConfig(np.array([0.5, 1.5, 3.5]), sample_count=21, seed=9)
    displacements = np.array([[1, 0], [0, 1], [2, 1]])
    first = compute_directional_structure_functions(data, displacements, slice_axis=3, config=config)
    second = compute_directional_structure_functions(data, displacements, slice_axis=3, config=config)
    assert np.array_equal(first.counts, second.counts)
    assert np.array_equal(first.sums, second.sums)

    full_config = DirectionalConfig(np.array([0.5, 1.5, 3.5]), sample_count=None)
    shifted = {name: np.roll(value, shift=(2, -3), axis=(0, 1)) for name, value in data.items()}
    base = compute_directional_structure_functions(data, displacements, slice_axis=3, config=full_config)
    rolled = compute_directional_structure_functions(shifted, displacements, slice_axis=3, config=full_config)
    assert np.array_equal(base.counts, rolled.counts)
    assert np.allclose(base.sums, rolled.sums, rtol=1e-14, atol=1e-14)


def test_standard_reference_matches_periodic_sine_formula():
    n = 32
    x = np.arange(n)
    amplitude = 2.5
    mode = 3
    wave = amplitude * np.sin(2.0 * np.pi * mode * x / n)
    field = np.zeros((3, 4, n))
    field[0] = wave
    output = compute_standard_s2_reference(field, np.array([[1, 0], [3, 0]]))
    for (delta_i, _), measured in output.items():
        expected = 2.0 * amplitude**2 * np.sin(np.pi * mode * delta_i / n) ** 2
        assert measured == pytest.approx(expected, rel=2e-15, abs=2e-15)


def test_analysis_helpers_recover_power_law_slopes_and_shapes():
    data = _slice_data((3, 3))
    config = DirectionalConfig(np.array([1.0, 2.0, 4.0, 8.0, 16.0]), include_global=False)
    result = compute_directional_structure_functions(
        data,
        np.array([[1, 0]]),
        slice_axis=3,
        config=config,
        q_names=("u",),
    )
    ell = np.sqrt(config.ell_bin_edges[:-1] * config.ell_bin_edges[1:])
    result.counts[:] = 20
    result.sums[:] = 0.0
    for direction, coefficient in {"parallel": 1.0, "xi": 2.0, "lambda": 4.0}.items():
        index = result.direction_names.index(direction)
        result.sums[0, 0, index] = result.counts[0, 0, index] * coefficient * ell**2

    slopes = fit_directional_slopes(result, (1.0, 16.0))
    assert slopes["u"]["local"]["parallel"]["slope"] == pytest.approx(2.0)
    assert slopes["u"]["local"]["parallel"]["quality"] == "ok"
    shapes = constant_s2_shapes(result, "u", (8.0, 32.0))
    assert np.allclose(shapes["parallel"], np.sqrt(np.array([8.0, 32.0])))
    assert np.allclose(shapes["xi"], np.sqrt(np.array([8.0, 32.0]) / 2.0))
    assert np.allclose(shapes["lambda"], np.sqrt(np.array([8.0, 32.0]) / 4.0))
    assert np.allclose(shapes["xi_over_lambda"], np.sqrt(2.0))


def test_coverage_rows_and_fit_quality_gate_sparse_directional_bins():
    data = _slice_data((3, 3))
    config = DirectionalConfig(np.array([1.0, 2.0, 4.0, 8.0]), include_global=False)
    result = compute_directional_structure_functions(
        data,
        np.array([[1, 0]]),
        slice_axis=3,
        config=config,
        q_names=("u",),
    )
    direction = result.direction_names.index("parallel")
    result.attempts[:] = 100
    result.counts[0, 0, direction] = np.array([20, 30, 40])
    result.sums[0, 0, direction] = np.array([20.0, 120.0, 640.0])

    rows = coverage_rows(result, min_count=25, min_accepted_fraction=0.25)
    parallel = [row for row in rows if row["direction"] == "parallel"]
    assert [row["fit_eligible"] for row in parallel] == [False, True, True]
    assert [row["accepted_fraction"] for row in parallel] == pytest.approx([0.2, 0.3, 0.4])

    rejected = fit_directional_slopes(
        result,
        (1.0, 8.0),
        min_count=35,
        min_accepted_fraction=0.25,
    )
    assert rejected["u"]["local"]["parallel"]["quality"] == "insufficient_coverage"
    assert np.isnan(rejected["u"]["local"]["parallel"]["slope"])

    accepted = fit_directional_slopes(
        result,
        (1.0, 8.0),
        min_count=25,
        min_accepted_fraction=0.25,
    )
    assert accepted["u"]["local"]["parallel"]["quality"] == "ok"
