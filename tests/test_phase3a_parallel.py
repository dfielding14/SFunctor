"""Focused Phase 3a stencil, support-policy, and block-accumulator tests."""
from __future__ import annotations

import numpy as np
import pytest

from sfunctor.analysis.phase3a import reduce_finite_domain_shards
from sfunctor.core.finite_domain import (
    FiniteDomainConfig,
    _block_population_for_bounds,
    _origin_block_ids,
    compute_finite_domain_structure_functions,
    stencil_definition,
    valid_origin_bounds_kji,
)
from sfunctor.core.phase3a import (
    compute_finite_domain_structure_functions_parallel,
    dense_displacement_manifest,
    dense_separation_centers,
    displacement_shard,
)
from sfunctor.reference_3d import compute_finite_domain_structure_functions_reference


def _cube(shape: tuple[int, int, int] = (5, 6, 7)) -> dict[str, np.ndarray]:
    k, j, i = np.indices(shape, dtype=float)
    return {
        "rho": 1.0 + 0.01 * (i + j + k),
        "B_x": 1.0 + 0.02 * i,
        "B_y": 0.2 + 0.01 * j,
        "B_z": 0.1 + 0.01 * k,
        "v_x": 0.1 * i**4 + 0.2 * j**2 + 0.3 * k,
        "v_y": -0.1 * i + 0.05 * j**3 + 0.2 * k**2,
        "v_z": 0.03 * i * j + 0.04 * k**3,
    }


def _line(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=float)
    shape = (1, 1, values.size)
    zeros = np.zeros(shape)
    return {
        "rho": np.ones(shape),
        "B_x": np.ones(shape),
        "B_y": zeros.copy(),
        "B_z": zeros.copy(),
        "v_x": values.reshape(shape),
        "v_y": zeros.copy(),
        "v_z": zeros.copy(),
    }


def _assert_moments_match(left, right) -> None:
    for name in (
        "counts",
        "exclusions",
        "sampled_pairs",
        "eligible_pairs",
        "cube_candidate_pairs",
        "excluded_boundary_pairs",
        "displacements_per_bin",
        "ell_bin_index_per_displacement",
        "sampled_pairs_per_displacement",
        "eligible_pairs_per_displacement",
        "cube_candidate_pairs_per_displacement",
        "excluded_boundary_pairs_per_displacement",
        "intrinsic_eligible_origins",
        "boundary_excluded_origins",
        "support_policy_excluded_origins",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name)), name
    assert np.allclose(left.sums, right.sums, rtol=2.0e-13, atol=2.0e-13)
    assert np.allclose(left.sums_sq, right.sums_sq, rtol=2.0e-13, atol=2.0e-13)


def test_stencil_definitions_match_phase3a_normalizations():
    multipliers, increment, local_B = stencil_definition(2)
    assert multipliers == (0, 1)
    assert np.allclose(increment, (-1.0, 1.0))
    assert np.allclose(local_B, (0.5, 0.5))

    multipliers, increment, local_B = stencil_definition(3)
    assert multipliers == (-1, 0, 1)
    assert np.allclose(increment, np.asarray((1.0, -2.0, 1.0)) / np.sqrt(3.0))
    assert np.allclose(local_B, np.asarray((1.0, 1.0, 1.0)) / 3.0)

    multipliers, increment, local_B = stencil_definition(5)
    assert multipliers == (-2, -1, 0, 1, 2)
    assert np.allclose(increment, np.asarray((1.0, -4.0, 6.0, -4.0, 1.0)) / np.sqrt(35.0))
    assert np.allclose(local_B, np.asarray((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0)


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (2, ((0, 9), (1, 11), (0, 11))),
        (3, ((1, 9), (1, 10), (1, 11))),
        (5, ((2, 8), (2, 9), (2, 10))),
    ],
)
def test_stencil_aware_bounds_keep_every_point_inside(width, expected):
    assert valid_origin_bounds_kji((10, 11, 12), (1, -1, 1), width) == expected


@pytest.mark.parametrize("stencil_width", [2, 3, 5])
@pytest.mark.parametrize("pair_mode", ["nested_core", "shell_local", "all_valid_origins"])
def test_optimized_stencils_match_slow_oracle(stencil_width, pair_mode):
    displacements = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)),
        dtype=np.int64,
    )
    config = FiniteDomainConfig(
        np.asarray((0.5, 1.5)),
        p_values=(1.0, 2.0),
        pair_mode=pair_mode,
        sample_count=9,
        pair_batch_size=4,
        include_subvolume_mean=True,
        stencil_width=stencil_width,
    )
    optimized = compute_finite_domain_structure_functions(
        _cube(), displacements, config=config, q_names=("B", "u")
    )
    oracle = compute_finite_domain_structure_functions_reference(
        _cube(), displacements, config=config, q_names=("B", "u")
    )
    _assert_moments_match(optimized, oracle)
    assert optimized.stencil_width == stencil_width
    assert optimized.shell_core_bounds_kji == oracle.shell_core_bounds_kji


@pytest.mark.parametrize(
    ("stencil_width", "values", "expected"),
    [
        (2, np.asarray([0.0, 1.0, 4.0, 9.0, 16.0]), 21.0),
        (3, np.asarray([0.0, 1.0, 4.0, 9.0, 16.0]), 4.0 / 3.0),
        (5, np.asarray([0.0, 1.0, 16.0, 81.0, 256.0]), 576.0 / 35.0),
    ],
)
def test_direct_polynomial_stencil_recovery(stencil_width, values, expected):
    result = compute_finite_domain_structure_functions(
        _line(values),
        np.asarray(((1, 0, 0),)),
        config=FiniteDomainConfig(
            np.asarray((0.5, 1.5)),
            pair_mode="all_valid_origins",
            stencil_width=stencil_width,
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    all_direction = result.direction_names.index("all")
    total = result.measurement_names.index("total")
    assert result.moments[0, 0, total, all_direction, 0, 0] == pytest.approx(expected)


def test_shell_local_uses_one_shared_box_within_each_shell():
    displacements = np.asarray(((1, 0, 0), (-1, 0, 0), (0, 2, 0), (0, -2, 0)))
    result = compute_finite_domain_structure_functions(
        _cube(),
        displacements,
        config=FiniteDomainConfig(
            np.asarray((0.5, 1.5, 2.5)),
            pair_mode="shell_local",
            stencil_width=3,
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    assert result.shell_core_bounds_kji == (
        ((0, 5), (0, 6), (1, 6)),
        ((0, 5), (2, 4), (0, 7)),
    )
    assert np.array_equal(result.eligible_pairs_per_displacement, (150, 150, 70, 70))


def test_slow_oracle_matches_optimized_with_larger_frozen_support_census():
    measured = np.asarray(((1, 0, 0), (-1, 0, 0)))
    support = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 2, 0), (0, -2, 0)),
        dtype=np.int64,
    )
    config = FiniteDomainConfig(
        np.asarray((0.5, 1.5, 2.5)),
        pair_mode="shell_local",
        sample_count=9,
        stencil_width=3,
        block_shape_kji=(2, 3, 4),
        include_subvolume_mean=False,
    )
    optimized = compute_finite_domain_structure_functions(
        _cube(),
        measured,
        config=config,
        q_names=("B", "u"),
        support_displacements_ijk=support,
    )
    oracle = compute_finite_domain_structure_functions_reference(
        _cube(),
        measured,
        config=config,
        q_names=("B", "u"),
        support_displacements_ijk=support,
    )
    _assert_moments_match(optimized, oracle)
    assert optimized.shell_core_bounds_kji == oracle.shell_core_bounds_kji
    assert optimized.support_displacements_sha256 == oracle.support_displacements_sha256
    assert optimized.support_displacement_count == oracle.support_displacement_count == len(support)


def test_origin_block_accumulators_sum_to_global_statistics():
    result = compute_finite_domain_structure_functions(
        _cube((5, 6, 7)),
        np.asarray(((1, 0, 0), (-1, 0, 0))),
        config=FiniteDomainConfig(
            np.asarray((0.5, 1.5)),
            pair_mode="all_valid_origins",
            stencil_width=3,
            block_shape_kji=(2, 3, 4),
            include_subvolume_mean=False,
        ),
        q_names=("B", "u"),
    )
    assert result.block_counts is not None
    assert result.block_sums is not None
    assert result.block_sums_sq is not None
    assert result.block_counts.shape[0] == 12
    assert np.array_equal(result.block_counts.sum(axis=0), result.counts)
    assert np.allclose(result.block_sums.sum(axis=0), result.sums)
    assert np.allclose(result.block_sums_sq.sum(axis=0), result.sums_sq)


@pytest.mark.parametrize("stencil_width", [2, 3, 5])
def test_block_enabled_optimized_path_matches_slow_oracle(stencil_width):
    displacements = np.asarray(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)))
    config = FiniteDomainConfig(
        np.asarray((0.5, 1.5)),
        pair_mode="shell_local",
        sample_count=9,
        stencil_width=stencil_width,
        block_shape_kji=(2, 3, 4),
        include_subvolume_mean=False,
    )
    optimized = compute_finite_domain_structure_functions(
        _cube(), displacements, config=config, q_names=("B", "u")
    )
    oracle = compute_finite_domain_structure_functions_reference(
        _cube(), displacements, config=config, q_names=("B", "u")
    )
    _assert_moments_match(optimized, oracle)
    assert np.array_equal(optimized.block_counts, oracle.block_counts)
    assert np.allclose(optimized.block_sums, oracle.block_sums)
    assert np.allclose(optimized.block_sums_sq, oracle.block_sums_sq)
    assert np.array_equal(optimized.block_sampled_origins, oracle.block_sampled_origins)
    assert np.array_equal(optimized.block_eligible_origins, oracle.block_eligible_origins)
    assert np.array_equal(optimized.block_exclusions, oracle.block_exclusions)


def test_dense_phase3a_design_is_reproducible_closed_and_strictly_bounded():
    centers = dense_separation_centers(320, 64)
    assert len(centers) == 64
    assert np.array_equal(centers[:16], np.arange(1, 17))
    assert centers[-1] == 320
    assert np.count_nonzero(centers > 160) >= 10
    assert np.count_nonzero((centers >= 64) & (centers <= 256)) >= 20
    assert np.array_equal(dense_separation_centers(32, 2), (1, 32))

    first, edges, manifest = dense_displacement_manifest(
        stencil_width=2, ell_max=320, bin_count=64, directions_per_bin=24
    )
    second, second_edges, second_manifest = dense_displacement_manifest(
        stencil_width=2, ell_max=320, bin_count=64, directions_per_bin=24
    )
    offsets = {tuple(row) for row in first.tolist()}
    assert np.array_equal(first, second)
    assert np.array_equal(edges, second_edges)
    assert manifest == second_manifest
    assert np.linalg.norm(first.astype(float), axis=1).max() <= 320
    assert all(tuple(-value for value in offset) in offsets for offset in offsets)
    assert (
        manifest["realized_directional_occupancy_per_bin"]
        == manifest["realized_offsets_per_bin"]
    )
    assert manifest["post_rounding_zero_offset_removed"] >= 0
    assert manifest["post_rounding_duplicate_offset_removed"] >= 0
    assert manifest["post_rounding_zero_or_duplicate_removed"] == (
        manifest["post_rounding_zero_offset_removed"]
        + manifest["post_rounding_duplicate_offset_removed"]
    )
    assert (
        manifest["requested_candidate_count"]
        - manifest["post_rounding_zero_offset_removed"]
        - manifest["post_rounding_duplicate_offset_removed"]
        - manifest["post_rounding_out_of_range_removed"]
        == manifest["realized_offset_count"]
    )


def test_round_robin_displacement_shards_are_disjoint_and_complete():
    displacements, _, _ = dense_displacement_manifest(
        stencil_width=5, ell_max=80, bin_count=32, directions_per_bin=12
    )
    shards = [displacement_shard(displacements, index, 3) for index in range(3)]
    rows = [tuple(row) for shard in shards for row in shard.tolist()]
    assert len(rows) == len(set(rows)) == len(displacements)
    assert set(rows) == {tuple(row) for row in displacements.tolist()}


@pytest.mark.parametrize("stencil_width", [3, 5])
def test_historical_all_valid_pairs_label_is_restricted_to_two_point(stencil_width):
    with pytest.raises(ValueError, match="historical 2-point"):
        FiniteDomainConfig(
            np.asarray((0.5, 1.5)),
            pair_mode="all_valid_pairs",
            stencil_width=stencil_width,
        )


def test_shell_local_separates_intrinsic_boundary_and_support_policy_loss():
    displacements = np.asarray(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)))
    result = compute_finite_domain_structure_functions(
        _cube(),
        displacements,
        config=FiniteDomainConfig(
            np.asarray((0.5, 1.5)),
            pair_mode="shell_local",
            stencil_width=3,
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    assert result.intrinsic_eligible_origins_per_displacement is not None
    assert result.support_policy_excluded_origins_per_displacement is not None
    assert np.array_equal(result.intrinsic_eligible_origins_per_displacement, (150, 150, 140, 140))
    assert np.array_equal(result.support_policy_excluded_origins_per_displacement, (50, 50, 40, 40))


def test_two_point_midpoint_block_assignment_is_signed_offset_stable():
    positive = _origin_block_ids(
        (np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.asarray((0, 1, 2))),
        (8, 8, 8),
        (2, 2, 2),
        (1, 0, 0),
        2,
    )
    negative = _origin_block_ids(
        (np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.asarray((1, 2, 3))),
        (8, 8, 8),
        (2, 2, 2),
        (-1, 0, 0),
        2,
    )
    assert np.array_equal(positive, negative)


@pytest.mark.parametrize(
    ("stencil_width", "positive_origins", "negative_origins", "expected"),
    [
        (2, np.asarray((0, 3)), np.asarray((1, 4)), np.asarray((0, 10))),
        (3, np.asarray((1, 3)), np.asarray((1, 3)), np.asarray((0, 10))),
        (5, np.asarray((2, 4)), np.asarray((2, 4)), np.asarray((1, 14))),
    ],
)
def test_hand_calculated_block_assignments_cover_signed_two_three_and_five_point_stencils(
    stencil_width, positive_origins, negative_origins, expected
):
    shape = (3, 3, 7) if stencil_width == 5 else (3, 3, 5)
    edge = np.asarray((0, 2), dtype=np.int64)
    positive = _origin_block_ids(
        (edge, edge, positive_origins),
        shape,
        (2, 2, 2),
        (1, 0, 0),
        stencil_width,
    )
    negative = _origin_block_ids(
        (edge, edge, negative_origins),
        shape,
        (2, 2, 2),
        (-1, 0, 0),
        stencil_width,
    )
    assert np.array_equal(positive, expected)
    assert np.array_equal(negative, expected)


@pytest.mark.parametrize(
    ("stencil_width", "bounds", "displacement", "expected"),
    [
        (
            2,
            ((0, 3), (0, 3), (0, 4)),
            (1, 0, 0),
            np.asarray((8, 8, 0, 4, 4, 0, 4, 4, 0, 2, 2, 0)),
        ),
        (
            2,
            ((0, 3), (0, 3), (1, 5)),
            (-1, 0, 0),
            np.asarray((8, 8, 0, 4, 4, 0, 4, 4, 0, 2, 2, 0)),
        ),
        (
            3,
            ((0, 3), (0, 3), (1, 4)),
            (1, 0, 0),
            np.asarray((4, 8, 0, 2, 4, 0, 2, 4, 0, 1, 2, 0)),
        ),
        (
            3,
            ((0, 3), (0, 3), (1, 4)),
            (-1, 0, 0),
            np.asarray((4, 8, 0, 2, 4, 0, 2, 4, 0, 1, 2, 0)),
        ),
        (
            5,
            ((0, 3), (0, 3), (2, 3)),
            (1, 0, 0),
            np.asarray((0, 4, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0)),
        ),
        (
            5,
            ((0, 3), (0, 3), (2, 3)),
            (-1, 0, 0),
            np.asarray((0, 4, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0)),
        ),
    ],
)
def test_hand_calculated_block_populations_include_negative_offsets_and_truncated_edges(
    stencil_width, bounds, displacement, expected
):
    population = _block_population_for_bounds(
        bounds,
        (3, 3, 5),
        (2, 2, 2),
        displacement,
        stencil_width,
    )
    assert np.array_equal(population, expected)
    assert int(population.sum()) == np.prod([stop - start for start, stop in bounds])


@pytest.mark.parametrize(
    ("displacements", "support_displacements"),
    [
        (np.asarray(((1.5, 0.0, 0.0),)), None),
        (np.asarray(((1, 0, 0),)), np.asarray(((1.5, 0.0, 0.0),))),
    ],
)
def test_parallel_wrapper_rejects_fractional_displacements_before_coercion(
    displacements, support_displacements
):
    with pytest.raises(ValueError, match="exact integer"):
        compute_finite_domain_structure_functions_parallel(
            _cube(),
            displacements,
            config=FiniteDomainConfig(np.asarray((0.5, 1.5)), include_subvolume_mean=False),
            q_names=("u",),
            worker_count=1,
            support_displacements_ijk=support_displacements,
        )


@pytest.mark.parametrize(
    ("support_displacements", "message"),
    [
        (np.asarray(((0, 0, 0), (1, 0, 0))), "zero support displacement"),
        (np.asarray(((-1, 0, 0),)), "every measured displacement"),
    ],
)
def test_support_census_rejects_zero_and_missing_measured_displacements(
    support_displacements, message
):
    with pytest.raises(ValueError, match=message):
        compute_finite_domain_structure_functions(
            _cube(),
            np.asarray(((1, 0, 0),)),
            config=FiniteDomainConfig(np.asarray((0.5, 1.5)), include_subvolume_mean=False),
            q_names=("u",),
            support_displacements_ijk=support_displacements,
        )


@pytest.mark.parametrize(("prefix", "q_name"), [("B", "B"), ("v", "u")])
def test_prebuilt_vector_spatial_shape_must_match_scalar_components(prefix, q_name):
    cube = _cube()
    cube[f"_{prefix}_vector"] = np.stack(
        [cube[f"{prefix}_{component}"] for component in "xyz"],
        axis=0,
    )[:, :, :, :-1]
    with pytest.raises(ValueError, match="spatial shape must match"):
        compute_finite_domain_structure_functions(
            cube,
            np.asarray(((1, 0, 0),)),
            config=FiniteDomainConfig(
                np.asarray((0.5, 1.5)),
                pair_mode="all_valid_origins",
                include_subvolume_mean=False,
            ),
            q_names=(q_name,),
        )


@pytest.mark.parametrize(("stencil_width", "ell_max"), [(2, 321), (3, 161), (5, 81)])
def test_dense_manifest_rejects_scales_beyond_stencil_footprint_limit(stencil_width, ell_max):
    with pytest.raises(ValueError, match="approved"):
        dense_displacement_manifest(
            stencil_width=stencil_width,
            ell_max=ell_max,
            bin_count=32,
            directions_per_bin=12,
        )


@pytest.mark.parametrize("stencil_width", [2, 3, 5])
def test_fork_workers_reduce_to_serial_with_frozen_shell_support(stencil_width):
    displacements = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)),
        dtype=np.int64,
    )
    config = FiniteDomainConfig(
        np.asarray((0.5, 1.5)),
        p_values=(2.0,),
        pair_mode="shell_local",
        sample_count=9,
        pair_batch_size=4,
        include_subvolume_mean=False,
        stencil_width=stencil_width,
        block_shape_kji=(2, 3, 4),
    )
    serial = compute_finite_domain_structure_functions(
        _cube(),
        displacements,
        config=config,
        q_names=("B", "u"),
        support_displacements_ijk=displacements,
    )
    parallel = reduce_finite_domain_shards(
        compute_finite_domain_structure_functions_parallel(
            _cube(),
            displacements,
            config=config,
            q_names=("B", "u"),
            worker_count=2,
            support_displacements_ijk=displacements,
        )
    )
    _assert_moments_match(serial, parallel)
    assert serial.support_displacements_sha256 == parallel.support_displacements_sha256
    assert np.array_equal(serial.block_counts, parallel.block_counts)
    assert np.array_equal(serial.block_sampled_origins, parallel.block_sampled_origins)
    assert np.array_equal(serial.block_eligible_origins, parallel.block_eligible_origins)
    for name in (
        "intrinsic_eligible_origins_per_displacement",
        "boundary_excluded_origins_per_displacement",
        "support_policy_excluded_origins_per_displacement",
    ):
        serial_by_offset = {
            tuple(offset): int(value)
            for offset, value in zip(serial.displacements_ijk, getattr(serial, name))
        }
        parallel_by_offset = {
            tuple(offset): int(value)
            for offset, value in zip(parallel.displacements_ijk, getattr(parallel, name))
        }
        assert serial_by_offset == parallel_by_offset
