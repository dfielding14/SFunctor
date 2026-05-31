"""Focused validation tests for finite-domain Phase 3 structure functions."""
from __future__ import annotations

import numpy as np
import pytest

from sfunctor.analysis.finite_domain import (
    constant_sp_shapes,
    geometric_bin_centers,
    result_to_npz_payload,
    summary_rows,
)
from sfunctor.core.finite_domain import (
    FiniteDomainConfig,
    build_cube_q_variants,
    compute_finite_domain_structure_functions,
    cube_offset_to_vector,
    generate_fibonacci_displacements,
    nested_core_bounds_kji,
    valid_origin_bounds_kji,
)
from sfunctor.reference_3d import compute_finite_domain_structure_functions_reference


def _cube_data(shape: tuple[int, int, int] = (4, 5, 6)) -> dict[str, np.ndarray]:
    """Return a finite, nonuniform synthetic cube in native KJI storage order."""

    k, j, i = np.indices(shape, dtype=float)
    return {
        "rho": 1.5 + 0.02 * k + 0.03 * j + 0.05 * i,
        "B_x": 1.0 + 0.04 * i - 0.01 * j,
        "B_y": 0.3 + 0.02 * j + 0.01 * k,
        "B_z": 0.2 + 0.03 * k - 0.01 * i,
        "v_x": 0.11 * i + 0.03 * j**2 - 0.02 * k,
        "v_y": -0.07 * i**2 + 0.05 * j + 0.04 * k,
        "v_z": 0.02 * i * j + 0.06 * k**2 - 0.01 * j,
    }


def _linear_cube(values: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return a one-row cube with a guide field along x1."""

    i = np.arange(4, dtype=float) if values is None else np.asarray(values, dtype=float)
    shape = (1, 1, i.size)
    zeros = np.zeros(shape)
    return {
        "rho": np.ones(shape),
        "B_x": np.ones(shape),
        "B_y": zeros.copy(),
        "B_z": zeros.copy(),
        "v_x": (3.0 * i).reshape(shape),
        "v_y": (4.0 * i).reshape(shape),
        "v_z": zeros.copy(),
    }


def _assert_results_match(left, right) -> None:
    assert left.q_names == right.q_names
    assert left.density_conventions == right.density_conventions
    assert left.geometry_names == right.geometry_names
    assert left.measurement_names == right.measurement_names
    assert left.direction_names == right.direction_names
    assert left.exclusion_names == right.exclusion_names
    assert left.p_values == right.p_values
    assert left.pair_mode == right.pair_mode
    assert left.cube_shape_kji == right.cube_shape_kji
    assert left.nested_core_bounds_kji == right.nested_core_bounds_kji
    assert left.rho0_provenance == right.rho0_provenance
    if np.isnan(left.rho0):
        assert np.isnan(right.rho0)
    else:
        assert left.rho0 == pytest.approx(right.rho0)
    for name in (
        "counts",
        "exclusions",
        "sampled_pairs",
        "eligible_pairs",
        "cube_candidate_pairs",
        "excluded_boundary_pairs",
        "displacements_per_bin",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name)), name
    assert left.out_of_range_displacements == right.out_of_range_displacements
    for name in ("sums", "sums_sq", "moments"):
        assert np.allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=2.0e-13,
            atol=2.0e-13,
            equal_nan=True,
        ), name
    assert np.allclose(
        left.standard_error,
        right.standard_error,
        rtol=2.0e-13,
        atol=5.0e-10,
        equal_nan=True,
    ), "standard_error"


def test_cube_offsets_use_ijk_vectors_and_kji_bounds_with_unequal_spacings():
    assert np.array_equal(cube_offset_to_vector((2, -3, 4), (0.5, 2.0, 7.0)), (1.0, -6.0, 28.0))
    assert valid_origin_bounds_kji((5, 6, 7), (2, -3, 4)) == ((0, 1), (3, 6), (0, 5))
    assert valid_origin_bounds_kji((5, 6, 7), (-2, 3, -4)) == ((4, 5), (0, 3), (2, 7))
    assert valid_origin_bounds_kji((5, 6, 7), (8, 0, 0)) == ((0, 0), (0, 0), (0, 0))


def test_nested_core_bounds_are_the_exact_intersection_of_valid_origins():
    displacements = np.array([[2, -3, 4], [-4, 1, -2], [1, 5, -1]])
    assert nested_core_bounds_kji((8, 9, 10), displacements) == ((2, 4), (3, 4), (4, 8))

    with pytest.raises(ValueError, match="no non-empty nested core"):
        nested_core_bounds_kji((2, 2, 2), np.array([[2, 0, 0]]))


@pytest.mark.parametrize(
    ("helper", "args"),
    [
        (cube_offset_to_vector, ((1.5, 0, 0),)),
        (cube_offset_to_vector, ((np.inf, 0, 0),)),
        (valid_origin_bounds_kji, ((5, 6, 7), (1.5, 0, 0))),
        (valid_origin_bounds_kji, ((5, 6, 7), (np.nan, 0, 0))),
        (valid_origin_bounds_kji, ((5, 6.5, 7), (1, 0, 0))),
        (nested_core_bounds_kji, ((5, 6, 7), np.array([[1.5, 0, 0]]))),
        (nested_core_bounds_kji, ((5, 6, 7), np.array([[np.inf, 0, 0]]))),
        (nested_core_bounds_kji, ((5, np.nan, 7), np.array([[1, 0, 0]]))),
    ],
    ids=[
        "vector-fractional-offset",
        "vector-nonfinite-offset",
        "valid-bounds-fractional-offset",
        "valid-bounds-nonfinite-offset",
        "valid-bounds-fractional-shape",
        "nested-bounds-fractional-offset",
        "nested-bounds-nonfinite-offset",
        "nested-bounds-nonfinite-shape",
    ],
)
def test_exported_geometry_helpers_reject_fractional_and_nonfinite_integer_inputs(helper, args):
    with pytest.raises(ValueError):
        helper(*args)


def _offset_support_array(result, aggregate_name):
    candidates = (
        f"{aggregate_name}_by_offset",
        f"{aggregate_name}_per_offset",
        f"{aggregate_name}_by_displacement",
        f"{aggregate_name}_per_displacement",
        f"offset_{aggregate_name}",
    )
    for name in candidates:
        if hasattr(result, name):
            return np.asarray(getattr(result, name))
    pytest.fail(f"result is missing offset-resolved {aggregate_name}: tried {candidates}")


def test_offset_resolved_support_arrays_reproduce_aggregate_support_bins():
    data = _cube_data((4, 5, 7))
    displacements = np.array(
        [[1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 1, 0], [0, -1, 0]]
    )
    config = FiniteDomainConfig(
        np.array([0.5, 1.5, 2.5]),
        pair_mode="all_valid_pairs",
        sample_count=3,
        include_subvolume_mean=False,
    )
    result = compute_finite_domain_structure_functions(
        data, displacements, config=config, q_names=("u",)
    )
    ell = np.linalg.norm(
        np.asarray([cube_offset_to_vector(offset, config.cell_sizes) for offset in displacements]),
        axis=1,
    )
    ell_bin = np.searchsorted(config.ell_bin_edges, ell, side="right") - 1

    offset_support = {}
    for name in (
        "sampled_pairs",
        "eligible_pairs",
        "cube_candidate_pairs",
        "excluded_boundary_pairs",
    ):
        values = _offset_support_array(result, name)
        assert values.shape == (len(displacements),), name
        expected = np.zeros_like(getattr(result, name))
        np.add.at(expected, ell_bin, values)
        assert np.array_equal(getattr(result, name), expected), name
        offset_support[name] = values

    assert np.array_equal(
        offset_support["cube_candidate_pairs"] - offset_support["eligible_pairs"],
        offset_support["excluded_boundary_pairs"],
    )
    assert np.all(offset_support["sampled_pairs"] <= offset_support["eligible_pairs"])


def test_npz_payload_metadata_is_readable_without_pickle(tmp_path):
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        pair_mode="all_valid_pairs",
        sample_count=3,
        include_subvolume_mean=False,
    )
    result = compute_finite_domain_structure_functions(
        _cube_data((3, 3, 4)),
        np.array([[1, 0, 0], [-1, 0, 0]]),
        config=config,
        q_names=("u",),
    )
    path = tmp_path / "finite_domain.npz"
    np.savez_compressed(path, **result_to_npz_payload(result))
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            payload[key]
        assert payload["metadata_json"].dtype.kind == "U"


@pytest.mark.parametrize(
    "calculator",
    [compute_finite_domain_structure_functions, compute_finite_domain_structure_functions_reference],
    ids=["optimized", "oracle"],
)
@pytest.mark.parametrize(
    "displacements",
    [np.array([[0, 0, 0]]), np.array([[1, 0, 0], [1, 0, 0]])],
    ids=["zero", "duplicate"],
)
def test_calculators_reject_zero_and_duplicate_offsets(calculator, displacements):
    config = FiniteDomainConfig(np.array([0.5, 1.5]))
    with pytest.raises(ValueError):
        calculator(_cube_data((2, 2, 3)), displacements, config=config, q_names=("u",))


def test_signed_fibonacci_generator_is_reproducible_unique_and_closed():
    first = generate_fibonacci_displacements((1.5, 3.25), directions_per_radius=12, phase=0.37)
    second = generate_fibonacci_displacements((1.5, 3.25), directions_per_radius=12, phase=0.37)
    tuples = {tuple(row) for row in first.tolist()}

    assert np.array_equal(first, second)
    assert first.dtype == np.int64
    assert len(tuples) == len(first)
    assert (0, 0, 0) not in tuples
    assert all(tuple(-value for value in offset) in tuples for offset in tuples)

    with pytest.raises(ValueError, match="even for signed closure"):
        generate_fibonacci_displacements((2.0,), directions_per_radius=7)


def test_signed_fibonacci_generator_reports_rounding_losses_without_changing_default_api():
    default = generate_fibonacci_displacements((0.75,), directions_per_radius=12)
    accounted, rounding = generate_fibonacci_displacements(
        (0.75,), directions_per_radius=12, return_accounting=True
    )
    tuples = {tuple(row) for row in accounted.tolist()}

    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, accounted)
    assert rounding == {
        "post_rounding_zero_offset_removed": 2,
        "post_rounding_duplicate_offset_removed": 4,
    }
    assert len(accounted) + sum(rounding.values()) == 12
    assert all(tuple(-value for value in offset) in tuples for offset in tuples)


def test_nested_core_requires_signed_closure_and_ignores_out_of_range_offsets():
    data = _cube_data((3, 3, 6))
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        pair_mode="nested_core",
        include_subvolume_mean=False,
    )
    with pytest.raises(ValueError, match="signed closure"):
        compute_finite_domain_structure_functions(
            data, np.array([[1, 0, 0]]), config=config, q_names=("u",)
        )

    baseline = compute_finite_domain_structure_functions(
        data, np.array([[1, 0, 0], [-1, 0, 0]]), config=config, q_names=("u",)
    )
    with_out_of_range = compute_finite_domain_structure_functions(
        data,
        np.array([[1, 0, 0], [-1, 0, 0], [4, 0, 0], [-4, 0, 0]]),
        config=config,
        q_names=("u",),
    )
    assert baseline.nested_core_bounds_kji == with_out_of_range.nested_core_bounds_kji
    assert np.array_equal(baseline.sampled_pairs, with_out_of_range.sampled_pairs)
    assert with_out_of_range.out_of_range_displacements == 2


def test_all_valid_pairs_never_wraps_boundary_sentinel():
    data = _linear_cube(np.array([0.0, 1.0, 2.0, 100.0]))
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        p_values=(2.0,),
        pair_mode="all_valid_pairs",
        include_subvolume_mean=False,
    )
    result = compute_finite_domain_structure_functions(
        data, np.array([[1, 0, 0]]), config=config, q_names=("u",)
    )
    all_direction = result.direction_names.index("all")
    total = result.measurement_names.index("total")

    assert np.array_equal(result.cube_candidate_pairs, (4,))
    assert np.array_equal(result.eligible_pairs, (3,))
    assert np.array_equal(result.excluded_boundary_pairs, (1,))
    assert np.array_equal(result.sampled_pairs, (3,))
    assert result.counts[0, 0, total, all_direction, 0, 0] == 3
    assert result.moments[0, 0, total, all_direction, 0, 0] == pytest.approx(
        (5.0**2 + 5.0**2 + 490.0**2) / 3.0
    )


@pytest.mark.parametrize("pair_mode", ["nested_core", "all_valid_pairs"])
def test_optimized_matches_exhaustive_oracle_on_small_cubes(pair_mode):
    data = _cube_data()
    displacements = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 1],
            [1, 0, -1],
            [0, -1, 0],
            [0, 1, 0],
            [1, 1, 0],
            [-1, -1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ]
    )
    config = FiniteDomainConfig(
        np.array([0.5, 0.9, 1.1, 1.35, 1.6]),
        p_values=(1.0, 2.0, 2.5),
        cell_sizes=(0.75, 1.0, 1.25),
        pair_mode=pair_mode,
        pair_batch_size=5,
        seed=19,
    )
    fast = compute_finite_domain_structure_functions(data, displacements, config=config)
    slow = compute_finite_domain_structure_functions_reference(data, displacements, config=config)
    _assert_results_match(fast, slow)


def test_pair_batch_size_does_not_change_products():
    data = _cube_data()
    displacements = np.array([[1, 0, 0], [0, -1, 0], [-1, 0, 1], [1, 1, 0]])
    common = {
        "ell_bin_edges": np.array([0.5, 0.9, 1.1, 1.35, 1.6]),
        "p_values": (1.0, 2.0, 2.5),
        "cell_sizes": (0.75, 1.0, 1.25),
        "pair_mode": "all_valid_pairs",
    }
    one_pair = compute_finite_domain_structure_functions(
        data,
        displacements,
        config=FiniteDomainConfig(**common, pair_batch_size=1),
        q_names=("B", "u"),
    )
    one_batch = compute_finite_domain_structure_functions(
        data,
        displacements,
        config=FiniteDomainConfig(**common, pair_batch_size=100_000),
        q_names=("B", "u"),
    )
    _assert_results_match(one_pair, one_batch)


def test_arbitrary_positive_real_p_and_total_vs_perpendicular_products():
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        p_values=(0.5, 1.0, 2.5, 4.0),
        pair_mode="all_valid_pairs",
        include_subvolume_mean=False,
    )
    result = compute_finite_domain_structure_functions(
        _linear_cube(), np.array([[1, 0, 0]]), config=config, q_names=("u",)
    )
    all_direction = result.direction_names.index("all")
    total = result.measurement_names.index("total")
    perpendicular = result.measurement_names.index("perpendicular")

    for p_index, p_value in enumerate(config.p_values):
        assert result.counts[0, 0, total, all_direction, p_index, 0] == 3
        assert result.counts[0, 0, perpendicular, all_direction, p_index, 0] == 3
        assert result.moments[0, 0, total, all_direction, p_index, 0] == pytest.approx(5.0**p_value)
        assert result.moments[0, 0, perpendicular, all_direction, p_index, 0] == pytest.approx(
            4.0**p_value
        )


@pytest.mark.parametrize(
    "p_values",
    [(), (0.0,), (-0.5,), (np.nan,), (np.inf,), (2.0, 2.0)],
    ids=["empty", "zero", "negative", "nan", "inf", "duplicate"],
)
def test_config_rejects_invalid_p_values(p_values):
    with pytest.raises(ValueError):
        FiniteDomainConfig(np.array([0.5, 1.5]), p_values=p_values)


@pytest.mark.parametrize(
    "override",
    [
        {"B_epsilon": np.nan},
        {"q_perp_epsilon": np.inf},
        {"r_perp_epsilon": np.nan},
        {"sample_count": 1.5},
        {"pair_batch_size": 1.5},
        {"seed": 1.5},
        {"sample_count": True},
        {"pair_batch_size": True},
        {"seed": True},
    ],
)
def test_config_rejects_nonfinite_epsilons_and_noninteger_controls(override):
    with pytest.raises(ValueError):
        FiniteDomainConfig(np.array([0.5, 1.5]), **override)


def test_calculator_rejects_fractional_offsets_and_power_overflow():
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        p_values=(4.0,),
        pair_mode="all_valid_pairs",
        include_subvolume_mean=False,
    )
    with pytest.raises(ValueError, match="exact integer"):
        compute_finite_domain_structure_functions(
            _linear_cube(), np.array([[1.9, 0.0, 0.0]]), config=config, q_names=("u",)
        )

    data = _linear_cube(np.array([0.0, 1.0e100]))
    with pytest.raises(FloatingPointError, match="non-finite powered"):
        compute_finite_domain_structure_functions(
            data, np.array([[1, 0, 0]]), config=config, q_names=("u",)
        )


@pytest.mark.parametrize(
    "calculator",
    [
        compute_finite_domain_structure_functions,
        compute_finite_domain_structure_functions_reference,
    ],
    ids=["batched", "oracle"],
)
def test_calculator_rejects_sampling_error_accumulator_overflow(calculator):
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        p_values=(2.0,),
        pair_mode="all_valid_pairs",
        include_subvolume_mean=False,
    )
    data = _linear_cube(np.array([0.0, 1.0e100]))
    with pytest.raises(FloatingPointError, match="accumulator contribution"):
        calculator(data, np.array([[1, 0, 0]]), config=config, q_names=("u",))


@pytest.mark.parametrize(
    "calculator",
    [
        compute_finite_domain_structure_functions,
        compute_finite_domain_structure_functions_reference,
    ],
    ids=["batched", "oracle"],
)
def test_calculator_rejects_cumulative_sampling_error_accumulator_overflow(calculator):
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        p_values=(2.0,),
        pair_mode="all_valid_pairs",
        pair_batch_size=1,
        include_subvolume_mean=False,
    )
    data = _linear_cube(np.array([0.0, 2.0e76, 4.0e76]))
    with pytest.raises(FloatingPointError, match="cumulative structure-function accumulator"):
        calculator(data, np.array([[1, 0, 0]]), config=config, q_names=("u",))


def test_compressible_variants_use_pointwise_and_reference_density_formulas():
    shape = (1, 1, 3)
    data = {
        "rho": np.array([1.0, 4.0, np.nan]).reshape(shape),
        "B_x": np.array([2.0, 4.0, 6.0]).reshape(shape),
        "B_y": np.array([1.0, 2.0, 3.0]).reshape(shape),
        "B_z": np.array([-1.0, 1.0, 5.0]).reshape(shape),
        "v_x": np.array([0.5, 1.0, 1.5]).reshape(shape),
        "v_y": np.array([2.0, 3.0, 4.0]).reshape(shape),
        "v_z": np.array([-2.0, 0.0, 2.0]).reshape(shape),
    }
    names = ("B", "u", "vA", "vA_ref", "z_plus", "z_minus", "z_plus_ref", "z_minus_ref")
    B, fields, rho0 = build_cube_q_variants(data, q_names=names, rho0=9.0)
    u = np.stack([data[f"v_{component}"] for component in "xyz"])
    valid_rho = np.array([True, True, False]).reshape(shape)
    expected_vA = np.full(B.shape, np.nan)
    expected_vA[:, valid_rho] = B[:, valid_rho] / np.sqrt(data["rho"][valid_rho])
    expected_vA_ref = B / 3.0

    assert rho0 == 9.0
    assert np.allclose(fields["vA"].values, expected_vA, equal_nan=True)
    assert np.allclose(fields["vA_ref"].values, expected_vA_ref)
    assert np.allclose(fields["z_plus"].values, u + expected_vA, equal_nan=True)
    assert np.allclose(fields["z_minus"].values, u - expected_vA, equal_nan=True)
    assert np.allclose(fields["z_plus_ref"].values, u + expected_vA_ref)
    assert np.allclose(fields["z_minus_ref"].values, u - expected_vA_ref)
    assert np.array_equal(fields["vA"].valid, valid_rho)
    assert np.array_equal(fields["z_plus"].valid, valid_rho)
    assert fields["vA_ref"].valid.all()
    assert fields["z_plus_ref"].valid.all()


def test_invalid_density_excludes_pointwise_variants_but_not_reference_variants():
    data = _linear_cube(np.arange(3, dtype=float))
    data["rho"][0, 0, 1] = np.nan
    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        pair_mode="all_valid_pairs",
        include_subvolume_mean=False,
    )
    result = compute_finite_domain_structure_functions(
        data,
        np.array([[1, 0, 0]]),
        config=config,
        q_names=("vA", "vA_ref"),
        rho0=1.0,
    )
    invalid_q = result.exclusion_names.index("invalid_q")
    all_direction = result.direction_names.index("all")
    total = result.measurement_names.index("total")

    assert result.exclusions[0, 0, invalid_q, 0] == 2
    assert result.exclusions[1, 0, invalid_q, 0] == 0
    assert result.counts[0, 0, total, all_direction, 0, 0] == 0
    assert result.counts[1, 0, total, all_direction, 0, 0] == 2

    data["rho"][:] = np.nan
    with pytest.raises(ValueError, match="rho0 cannot be inferred"):
        build_cube_q_variants(data, q_names=("vA_ref",))


def test_explicit_reference_density_variants_do_not_require_rho_array():
    data = _linear_cube()
    del data["rho"]
    B, fields, rho0 = build_cube_q_variants(data, q_names=("vA_ref",), rho0=4.0)
    assert rho0 == 4.0
    assert np.allclose(fields["vA_ref"].values, B / 2.0)


def test_fixed_seed_sampling_is_deterministic():
    data = _cube_data((5, 6, 7))
    displacements = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def run(seed: int):
        return compute_finite_domain_structure_functions(
            data,
            displacements,
            config=FiniteDomainConfig(
                np.array([0.5, 1.5]),
                pair_mode="all_valid_pairs",
                sample_count=7,
                pair_batch_size=2,
                seed=seed,
                include_subvolume_mean=False,
            ),
            q_names=("u",),
        )

    first = run(91)
    second = run(91)
    changed_seed = run(92)
    _assert_results_match(first, second)
    assert np.array_equal(first.sampled_pairs, (21,))
    assert not np.array_equal(first.sums, changed_seed.sums)


def _swap_x1_x3(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    transpose = lambda values: np.transpose(values, (2, 1, 0))
    return {
        "rho": transpose(data["rho"]),
        "B_x": transpose(data["B_z"]),
        "B_y": transpose(data["B_y"]),
        "B_z": transpose(data["B_x"]),
        "v_x": transpose(data["v_z"]),
        "v_y": transpose(data["v_y"]),
        "v_z": transpose(data["v_x"]),
    }


def test_axis_permutation_preserves_summary_products():
    data = _cube_data()
    displacements = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, -1], [1, 0, 1]]
    )
    base = compute_finite_domain_structure_functions(
        data,
        displacements,
        config=FiniteDomainConfig(
            np.array([0.5, 0.9, 1.3, 1.7, 2.1]),
            cell_sizes=(0.8, 1.1, 1.4),
            pair_mode="all_valid_pairs",
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    permuted = compute_finite_domain_structure_functions(
        _swap_x1_x3(data),
        displacements[:, [2, 1, 0]],
        config=FiniteDomainConfig(
            np.array([0.5, 0.9, 1.3, 1.7, 2.1]),
            cell_sizes=(1.4, 1.1, 0.8),
            pair_mode="all_valid_pairs",
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )

    for name in ("counts", "exclusions", "sampled_pairs", "eligible_pairs", "excluded_boundary_pairs"):
        assert np.array_equal(getattr(base, name), getattr(permuted, name)), name
    assert np.allclose(base.sums, permuted.sums, rtol=2.0e-13, atol=2.0e-13)

    base_rows = summary_rows(base, (0.5, 2.1))
    permuted_rows = summary_rows(permuted, (0.5, 2.1))
    assert len(base_rows) == len(permuted_rows)
    for left, right in zip(base_rows, permuted_rows):
        for name in (
            "q",
            "density_convention",
            "geometry",
            "measurement",
            "p",
            "fit_interval",
            "fit_quality",
            "fit_bin_counts",
            "sample_counts",
            "excluded_pair_counts",
        ):
            assert left[name] == right[name], name
        assert np.allclose(
            list(left["slopes"].values()),
            list(right["slopes"].values()),
            rtol=2.0e-13,
            atol=2.0e-13,
            equal_nan=True,
        )


@pytest.mark.parametrize("pair_mode", ["nested_core", "all_valid_pairs"])
def test_proper_rotation_preserves_summary_products(pair_mode):
    data = _cube_data((5, 6, 7))
    displacements = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0]]
    )
    config = FiniteDomainConfig(
        np.array([0.5, 1.1, 1.7]),
        pair_mode=pair_mode,
        include_subvolume_mean=False,
    )
    base = compute_finite_domain_structure_functions(data, displacements, config=config)

    rotate = lambda values: values[:, ::-1, ::-1]
    rotated_data = {
        "rho": rotate(data["rho"]),
        "B_x": -rotate(data["B_x"]),
        "B_y": -rotate(data["B_y"]),
        "B_z": rotate(data["B_z"]),
        "v_x": -rotate(data["v_x"]),
        "v_y": -rotate(data["v_y"]),
        "v_z": rotate(data["v_z"]),
    }
    rotated_displacements = displacements.copy()
    rotated_displacements[:, :2] *= -1
    rotated = compute_finite_domain_structure_functions(
        rotated_data, rotated_displacements, config=config
    )

    for name in (
        "counts",
        "exclusions",
        "sampled_pairs",
        "eligible_pairs",
        "cube_candidate_pairs",
        "excluded_boundary_pairs",
    ):
        assert np.array_equal(getattr(base, name), getattr(rotated, name)), name
    assert np.allclose(base.sums, rotated.sums, rtol=2.0e-13, atol=2.0e-13)


def test_translation_pair_swap_and_sign_reversals_preserve_statistics():
    shape = (7, 8, 9)
    k, j, i = np.indices(shape, dtype=float)
    full = {
        "rho": np.ones(shape),
        "B_x": np.ones(shape),
        "B_y": np.full(shape, 0.1),
        "B_z": np.full(shape, 0.2),
        "v_x": 0.2 * i + 0.1 * j,
        "v_y": -0.1 * i + 0.3 * k,
        "v_z": 0.2 * j - 0.1 * k,
    }

    def crop(offset):
        k0, j0, i0 = offset
        return {name: values[k0 : k0 + 4, j0 : j0 + 5, i0 : i0 + 6] for name, values in full.items()}

    config = FiniteDomainConfig(
        np.array([0.5, 1.5]),
        pair_mode="all_valid_pairs",
        include_subvolume_mean=False,
    )

    def calculate(data, displacement):
        return compute_finite_domain_structure_functions(
            data, np.asarray([displacement]), config=config, q_names=("B", "u")
        )

    base = calculate(crop((0, 0, 0)), (1, 0, 0))
    translated = calculate(crop((2, 2, 2)), (1, 0, 0))
    swapped = calculate(crop((0, 0, 0)), (-1, 0, 0))
    reversed_B_data = crop((0, 0, 0))
    for name in ("B_x", "B_y", "B_z"):
        reversed_B_data[name] = -reversed_B_data[name]
    reversed_B = calculate(reversed_B_data, (1, 0, 0))
    reversed_u_data = crop((0, 0, 0))
    for name in ("v_x", "v_y", "v_z"):
        reversed_u_data[name] = -reversed_u_data[name]
    reversed_u = calculate(reversed_u_data, (1, 0, 0))

    for result in (translated, swapped, reversed_B, reversed_u):
        assert np.array_equal(base.counts, result.counts)
        assert np.allclose(base.sums, result.sums, rtol=2.0e-13, atol=2.0e-13)


def test_weak_subvolume_mean_and_invalid_fields_are_excluded_not_classified():
    shape = (1, 2, 2)
    data = {
        "rho": np.ones(shape),
        "B_x": np.array([[[1.0, -1.0], [1.0, -1.0]]]),
        "B_y": np.zeros(shape),
        "B_z": np.zeros(shape),
        "v_x": np.array([[[0.0, 0.0], [1.0, 1.0]]]),
        "v_y": np.zeros(shape),
        "v_z": np.zeros(shape),
    }
    config = FiniteDomainConfig(np.array([0.5, 1.5]), pair_mode="all_valid_pairs")
    result = compute_finite_domain_structure_functions(
        data, np.array([[0, 1, 0]]), config=config, q_names=("u",)
    )
    weak_B = result.exclusion_names.index("weak_B_direction")
    all_direction = result.direction_names.index("all")
    total = result.measurement_names.index("total")
    assert result.counts[0, 0, total, all_direction, 0, 0] == 2
    assert result.counts[0, 1, total, all_direction, 0, 0] == 0
    assert result.exclusions[0, 1, weak_B, 0] == 2

    data["v_x"][0, 0, 0] = np.nan
    data["B_x"][0, 1, 1] = np.inf
    invalid = compute_finite_domain_structure_functions(
        data, np.array([[0, 1, 0]]), config=config, q_names=("u",)
    )
    invalid_B = invalid.exclusion_names.index("invalid_B")
    invalid_q = invalid.exclusion_names.index("invalid_q")
    assert invalid.exclusions[0, 0, invalid_B, 0] > 0
    assert invalid.exclusions[0, 0, invalid_q, 0] > 0


def test_sample_count_and_wedge_width_sensitivity_are_explicit():
    data = _cube_data((5, 6, 7))
    displacements = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])

    def calculate(sample_count, theta_parallel_max=np.deg2rad(15.0)):
        return compute_finite_domain_structure_functions(
            data,
            displacements,
            config=FiniteDomainConfig(
                np.array([0.5, 1.5]),
                pair_mode="all_valid_pairs",
                sample_count=sample_count,
                theta_parallel_max=theta_parallel_max,
                include_subvolume_mean=False,
                seed=71,
            ),
            q_names=("u",),
        )

    small, larger, full = calculate(8), calculate(64), calculate(None)
    all_direction = full.direction_names.index("all")
    total = full.measurement_names.index("total")
    full_value = full.moments[0, 0, total, all_direction, 0, 0]
    assert abs(larger.moments[0, 0, total, all_direction, 0, 0] - full_value) < abs(
        small.moments[0, 0, total, all_direction, 0, 0] - full_value
    )

    narrow, wide = calculate(None, np.deg2rad(5.0)), calculate(None, np.deg2rad(35.0))
    parallel = full.direction_names.index("parallel")
    assert wide.counts[0, 0, total, parallel, 0, 0] >= narrow.counts[0, 0, total, parallel, 0, 0]


def test_separation_bin_width_changes_census_without_losing_pairs():
    data = _cube_data((5, 6, 7))
    displacements = np.array([[1, 0, 0], [-1, 0, 0], [1, 1, 0], [-1, -1, 0]])

    def calculate(edges):
        return compute_finite_domain_structure_functions(
            data,
            displacements,
            config=FiniteDomainConfig(
                np.asarray(edges),
                pair_mode="all_valid_pairs",
                include_subvolume_mean=False,
            ),
            q_names=("u",),
        )

    narrow = calculate([0.5, 1.2, 1.8])
    wide = calculate([0.5, 1.8])
    assert np.array_equal(narrow.displacements_per_bin, (2, 2))
    assert np.array_equal(wide.displacements_per_bin, (4,))
    assert narrow.sampled_pairs.sum() == wide.sampled_pairs.sum()


def _controlled_reducer_result():
    edges = np.array(
        [1.0 / np.sqrt(2.0), np.sqrt(2.0), 2.0 * np.sqrt(2.0), 4.0 * np.sqrt(2.0), 8.0 * np.sqrt(2.0)]
    )
    result = compute_finite_domain_structure_functions(
        _linear_cube(),
        np.array([[1, 0, 0]]),
        config=FiniteDomainConfig(
            edges,
            pair_mode="all_valid_pairs",
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    result.counts[:] = 0
    result.sums[:] = 0.0
    result.sums_sq[:] = 0.0
    result.sampled_pairs[:] = 100
    ell = geometric_bin_centers(edges)
    measurement = result.measurement_names.index("perpendicular")
    for direction, coefficient in {"parallel": 1.0, "xi": 2.0, "lambda": 4.0}.items():
        direction_index = result.direction_names.index(direction)
        index = (0, 0, measurement, direction_index, 0)
        moments = coefficient * ell**2
        result.counts[index] = 20
        result.sums[index] = 20.0 * moments
        result.sums_sq[index] = 20.0 * moments**2
    return result


def test_constant_sp_reducer_recovers_known_curves_and_aspect_ratios():
    result = _controlled_reducer_result()
    targets = np.array([10.0, 40.0])
    shapes = constant_sp_shapes(result, "u", 2.0, targets)

    assert np.allclose(shapes["parallel"], np.sqrt(targets))
    assert np.allclose(shapes["xi"], np.sqrt(targets / 2.0))
    assert np.allclose(shapes["lambda"], np.sqrt(targets / 4.0))
    assert np.allclose(shapes["xi_over_lambda"], np.sqrt(2.0))
    assert np.allclose(shapes["parallel_over_lambda"], 2.0)
    assert np.array_equal(shapes["parallel_quality"], ("ok", "ok"))
    assert np.array_equal(shapes["xi_quality"], ("ok", "ok"))
    assert np.array_equal(shapes["lambda_quality"], ("ok", "ok"))


def test_constant_sp_reducer_rejects_sparse_and_multiple_crossing_curves():
    result = _controlled_reducer_result()
    measurement = result.measurement_names.index("perpendicular")
    parallel = result.direction_names.index("parallel")
    xi = result.direction_names.index("xi")

    result.counts[0, 0, measurement, parallel, 0] = np.array([20, 0, 0, 0])
    xi_moments = np.array([1.0, 4.0, 1.0, 4.0])
    result.sums[0, 0, measurement, xi, 0] = 20.0 * xi_moments
    result.sums_sq[0, 0, measurement, xi, 0] = 20.0 * xi_moments**2

    shapes = constant_sp_shapes(result, "u", 2.0, (2.0,), min_count=10)
    assert np.isnan(shapes["parallel"][0])
    assert shapes["parallel_quality"][0] == "no_crossing"
    assert np.isnan(shapes["xi"][0])
    assert shapes["xi_quality"][0] == "multiple_crossings"


def test_constant_sp_reducer_does_not_bridge_sparse_middle_bin():
    result = _controlled_reducer_result()
    measurement = result.measurement_names.index("perpendicular")
    parallel = result.direction_names.index("parallel")
    result.counts[0, 0, measurement, parallel, 0] = np.array([20, 0, 20, 20])
    moments = np.array([1.0, 2.0, 4.0, 8.0])
    result.sums[0, 0, measurement, parallel, 0] = 20.0 * moments
    result.sums_sq[0, 0, measurement, parallel, 0] = 20.0 * moments**2

    shapes = constant_sp_shapes(result, "u", 2.0, (2.0,), min_count=10)
    assert np.isnan(shapes["parallel"][0])
    assert shapes["parallel_quality"][0] == "no_crossing"
