"""Focused synthetic tests for the disjoint Phase 3a analysis helpers."""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from scripts.phase3a import run_phase3a_sampler as phase3a_runner
from sfunctor.analysis.phase3a import (
    block_bootstrap_moment_uncertainty,
    block_jackknife_moment_uncertainty,
    load_finite_domain_partial_npz,
    local_log_slope,
    reduce_finite_domain_shards,
    save_finite_domain_partial_npz,
)
from sfunctor.core.finite_domain import (
    FiniteDomainConfig,
    FiniteDomainResult,
    compute_finite_domain_structure_functions,
)


def _cube_data(shape: tuple[int, int, int] = (4, 5, 6)) -> dict[str, np.ndarray]:
    """Return a finite synthetic KJI cube with a guide field and curved velocity."""

    k, j, i = np.indices(shape, dtype=float)
    return {
        "rho": np.ones(shape),
        "B_x": 1.0 + 0.01 * j,
        "B_y": 0.1 + 0.02 * k,
        "B_z": 0.05 + 0.01 * i,
        "v_x": 0.12 * i**2 + 0.04 * j - 0.03 * k,
        "v_y": 0.07 * j**2 - 0.02 * i + 0.05 * k,
        "v_z": 0.03 * k**2 + 0.01 * i * j,
    }


def _calculate(
    offsets: np.ndarray,
    *,
    pair_mode: str = "all_valid_origins",
    stencil_width: int = 2,
    block_shape_kji: tuple[int, int, int] = (2, 2, 2),
    support_offsets: np.ndarray | None = None,
) -> FiniteDomainResult:
    if support_offsets is None:
        support_offsets = np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 0]])
    return compute_finite_domain_structure_functions(
        _cube_data(),
        np.asarray(offsets),
        config=FiniteDomainConfig(
            np.array([0.5, 1.5]),
            pair_mode=pair_mode,
            include_subvolume_mean=False,
            stencil_width=stencil_width,
            block_shape_kji=block_shape_kji,
        ),
        q_names=("u",),
        support_displacements_ijk=support_offsets,
    )


def _assert_result_arrays_equal(left: FiniteDomainResult, right: FiniteDomainResult) -> None:
    for field in fields(FiniteDomainResult):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            assert np.array_equal(left_value, right_value, equal_nan=True), field.name
        elif field.name == "elapsed_seconds":
            assert right_value == pytest.approx(left_value)
        elif isinstance(left_value, float) and np.isnan(left_value):
            assert np.isnan(right_value), field.name
        else:
            assert left_value == right_value, field.name


def test_reducer_is_order_invariant_and_concatenates_sorted_offset_support():
    shards = (
        _calculate(np.array([[0, 1, 0]])),
        _calculate(np.array([[-1, 0, 0]])),
        _calculate(np.array([[1, 0, 0]])),
    )

    forward = reduce_finite_domain_shards(shards)
    reverse = reduce_finite_domain_shards(reversed(shards))

    _assert_result_arrays_equal(forward, reverse)
    assert np.array_equal(
        forward.displacements_ijk,
        np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 0]]),
    )
    assert np.array_equal(
        forward.sampled_pairs_per_displacement,
        np.array(
            [
                shards[1].sampled_pairs_per_displacement[0],
                shards[0].sampled_pairs_per_displacement[0],
                shards[2].sampled_pairs_per_displacement[0],
            ]
        ),
    )
    assert np.array_equal(
        forward.block_counts,
        sum((shard.block_counts for shard in shards)),
    )
    assert np.allclose(
        forward.elapsed_seconds_per_ell_bin,
        sum((shard.elapsed_seconds_per_ell_bin for shard in shards)),
    )


def test_reducer_rejects_overlapping_offsets_and_missing_or_duplicate_shard_ids():
    left = _calculate(np.array([[1, 0, 0]]))
    right = _calculate(np.array([[1, 0, 0]]))

    with pytest.raises(ValueError, match="overlapping displacement offsets"):
        reduce_finite_domain_shards((left, right))
    with pytest.raises(ValueError, match="inventory mismatch"):
        reduce_finite_domain_shards(
            {"part-a": left},
            expected_shard_ids=("part-a", "part-b"),
        )
    with pytest.raises(ValueError, match="overlapping shard identifiers"):
        reduce_finite_domain_shards(
            (left, _calculate(np.array([[-1, 0, 0]]))),
            shard_ids=("part-a", "part-a"),
        )


@pytest.mark.parametrize(
    ("right", "metadata_name"),
    [
        (_calculate(np.array([[-1, 0, 0]]), stencil_width=3), "stencil_width"),
        (_calculate(np.array([[-1, 0, 0]]), pair_mode="shell_local"), "pair_mode"),
        (_calculate(np.array([[-1, 0, 0]]), block_shape_kji=(1, 2, 2)), "block_shape_kji"),
    ],
    ids=["mixed-stencil", "mixed-support-mode", "mixed-block-layout"],
)
def test_reducer_rejects_incompatible_phase3a_metadata(right, metadata_name):
    left = _calculate(np.array([[1, 0, 0]]))

    with pytest.raises(ValueError, match=metadata_name):
        reduce_finite_domain_shards((left, right))


def test_reducer_rejects_mixed_frozen_support_censuses():
    left = _calculate(np.array([[1, 0, 0]]))
    right = _calculate(
        np.array([[-1, 0, 0]]),
        support_offsets=np.array([[-1, 0, 0], [1, 0, 0]]),
    )

    with pytest.raises(ValueError, match="support_displacements_sha256"):
        reduce_finite_domain_shards((left, right))


def test_block_uncertainty_is_spatial_and_bootstrap_is_seed_deterministic():
    result = reduce_finite_domain_shards(
        (
            _calculate(np.array([[-1, 0, 0]])),
            _calculate(np.array([[1, 0, 0]])),
            _calculate(np.array([[0, 1, 0]])),
        )
    )

    jackknife = block_jackknife_moment_uncertainty(result)
    first = block_bootstrap_moment_uncertainty(
        result, n_resamples=40, seed=1729, return_replicates=True
    )
    second = block_bootstrap_moment_uncertainty(
        result, n_resamples=40, seed=1729, return_replicates=True
    )

    assert np.array_equal(jackknife.estimate, result.moments, equal_nan=True)
    assert np.any(np.isfinite(jackknife.standard_error))
    assert np.array_equal(first.standard_error, second.standard_error, equal_nan=True)
    assert np.array_equal(first.interval_low, second.interval_low, equal_nan=True)
    assert np.array_equal(first.interval_high, second.interval_high, equal_nan=True)
    assert np.array_equal(first.replicate_moments, second.replicate_moments, equal_nan=True)
    assert first.seed == second.seed == 1729
    assert first.n_resamples == second.n_resamples == 40
    expected_effective_blocks = np.divide(
        np.square(result.block_counts.sum(axis=0)),
        np.square(result.block_counts).sum(axis=0),
        out=np.zeros_like(result.counts, dtype=float),
        where=np.square(result.block_counts).sum(axis=0) > 0,
    )
    assert np.allclose(jackknife.effective_blocks, expected_effective_blocks)
    assert np.allclose(first.effective_blocks, expected_effective_blocks)
    assert jackknife.geometric_block_count == first.geometric_block_count == result.block_counts.shape[0]
    assert (
        jackknife.resampling_population
        == first.resampling_population
        == "fixed_geometric_layout_including_empty_blocks"
    )


def test_local_log_slope_uses_complete_centered_regression_window():
    ell = np.geomspace(1.0, 64.0, 7)
    moments = np.stack((3.0 * ell**2, 5.0 * ell**0.5))

    slopes = local_log_slope(ell, moments, window=3)

    assert np.all(np.isnan(slopes[:, (0, -1)]))
    assert np.allclose(slopes[0, 1:-1], 2.0)
    assert np.allclose(slopes[1, 1:-1], 0.5)
    moments[0, 3] = np.nan
    slopes = local_log_slope(ell, moments, window=3)
    assert np.all(np.isnan(slopes[0, 2:5]))


def test_partial_npz_round_trip_loads_without_pickle_and_reduces(tmp_path):
    left = _calculate(np.array([[1, 0, 0]]))
    right = _calculate(np.array([[-1, 0, 0]]))
    path = tmp_path / "phase3a_partial.npz"

    save_finite_domain_partial_npz(path, left)
    with np.load(path, allow_pickle=False) as payload:
        assert all(payload[name].dtype.kind != "O" for name in payload.files)
        for name in payload.files:
            payload[name]
    loaded = load_finite_domain_partial_npz(path)

    _assert_result_arrays_equal(left, loaded)
    reduced = reduce_finite_domain_shards(
        {"left": loaded, "right": right},
        expected_shard_ids=("left", "right"),
    )
    assert np.array_equal(reduced.displacements_ijk, np.array([[-1, 0, 0], [1, 0, 0]]))


def test_runner_relative_difference_reports_finite_mask_mismatches():
    diagnostic = phase3a_runner._relative_difference(
        np.asarray((1.0, np.nan, 3.0)),
        np.asarray((1.0, 2.0, np.nan)),
    )

    assert diagnostic["compared"] == 1
    assert diagnostic["finite_mask_mismatch_count"] == 2
    assert diagnostic["equivalent"] is False


def test_runner_relative_difference_marks_exactly_matching_arrays_equivalent():
    diagnostic = phase3a_runner._relative_difference(
        np.asarray((1.0, np.nan, 3.0)),
        np.asarray((1.0, np.nan, 3.0)),
    )

    assert diagnostic["compared"] == 2
    assert diagnostic["finite_mask_mismatch_count"] == 0
    assert diagnostic["maximum_relative_difference"] == 0.0
    assert diagnostic["equivalent"] is True
