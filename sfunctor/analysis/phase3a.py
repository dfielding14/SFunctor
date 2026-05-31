"""Strict reducers and uncertainty helpers for Phase 3a finite-domain shards.

The core estimator returns additive :class:`FiniteDomainResult` objects.  A
parallel worker may evaluate only a subset of the displacement manifest, so
the reducer concatenates offset-resolved support arrays while adding shell and
block accumulators.  Offsets are sorted lexicographically in IJK order to make
the reduced representation independent of shard arrival order.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from os import PathLike
from typing import Any

import numpy as np

from sfunctor.core.finite_domain import FiniteDomainResult

__all__ = [
    "MomentUncertainty",
    "block_bootstrap_moment_uncertainty",
    "block_jackknife_moment_uncertainty",
    "finite_domain_partial_from_npz_payload",
    "finite_domain_partial_to_npz_payload",
    "load_finite_domain_partial_npz",
    "local_log_slope",
    "merge_finite_domain_shards",
    "reduce_finite_domain_shards",
    "save_finite_domain_partial_npz",
]


_NPZ_FORMAT = "sfunctor.phase3a.finite_domain_partial"
_NPZ_VERSION = 2

_AGGREGATE_ARRAY_NAMES = (
    "counts",
    "sums",
    "sums_sq",
    "exclusions",
    "sampled_pairs",
    "eligible_pairs",
    "cube_candidate_pairs",
    "excluded_boundary_pairs",
    "displacements_per_bin",
    "elapsed_seconds_per_ell_bin",
    "intrinsic_eligible_origins",
    "boundary_excluded_origins",
    "support_policy_excluded_origins",
)
_BLOCK_ARRAY_NAMES = (
    "block_counts",
    "block_sums",
    "block_sums_sq",
    "block_sampled_origins",
    "block_eligible_origins",
    "block_exclusions",
)
_OFFSET_ARRAY_NAMES = (
    "displacements_ijk",
    "ell_bin_index_per_displacement",
    "sampled_pairs_per_displacement",
    "eligible_pairs_per_displacement",
    "cube_candidate_pairs_per_displacement",
    "excluded_boundary_pairs_per_displacement",
    "intrinsic_eligible_origins_per_displacement",
    "boundary_excluded_origins_per_displacement",
    "support_policy_excluded_origins_per_displacement",
)
_LABEL_ARRAY_NAMES = (
    "q_names",
    "density_conventions",
    "geometry_names",
    "measurement_names",
    "direction_names",
    "exclusion_names",
)
_COMPATIBLE_METADATA_NAMES = (
    "q_names",
    "density_conventions",
    "geometry_names",
    "measurement_names",
    "direction_names",
    "exclusion_names",
    "ell_bin_edges",
    "p_values",
    "pair_mode",
    "cube_shape_kji",
    "nested_core_bounds_kji",
    "rho0",
    "rho0_provenance",
    "cell_sizes",
    "angle_limits",
    "sample_count",
    "pair_batch_size",
    "seed",
    "stencil_width",
    "shell_core_bounds_kji",
    "block_shape_kji",
    "support_displacements_sha256",
    "support_displacement_count",
    "block_assignment",
)


@dataclass(frozen=True)
class MomentUncertainty:
    """Spatial-block uncertainty for every moment in a finite-domain result."""

    method: str
    estimate: np.ndarray
    standard_error: np.ndarray
    contributing_blocks: np.ndarray
    effective_blocks: np.ndarray
    geometric_block_count: int
    resampling_population: str
    interval_low: np.ndarray | None = None
    interval_high: np.ndarray | None = None
    valid_resamples: np.ndarray | None = None
    seed: int | None = None
    n_resamples: int | None = None
    confidence_level: float | None = None
    replicate_moments: np.ndarray | None = None


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape != right_array.shape:
            return False
        if left_array.dtype.kind in "fc" or right_array.dtype.kind in "fc":
            return bool(np.array_equal(left_array, right_array, equal_nan=True))
        return bool(np.array_equal(left_array, right_array))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _metadata_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        return bool(left == right or (np.isnan(left) and np.isnan(right)))
    return bool(left == right)


def _ell_bin_indices(result: FiniteDomainResult) -> np.ndarray:
    vectors = result.displacements_ijk * np.asarray(result.cell_sizes, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    indices = np.searchsorted(result.ell_bin_edges, lengths, side="right") - 1
    indices[lengths == result.ell_bin_edges[-1]] = result.ell_bin_edges.size - 2
    return np.where(
        (indices >= 0) & (indices < result.ell_bin_edges.size - 1),
        indices,
        -1,
    ).astype(np.int64)


def _sum_by_ell_bin(values: np.ndarray, ell_bin_indices: np.ndarray, n_ell: int) -> np.ndarray:
    output = np.zeros(n_ell, dtype=values.dtype)
    in_range = ell_bin_indices >= 0
    np.add.at(output, ell_bin_indices[in_range], values[in_range])
    return output


def _validate_nonnegative_integer_array(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative integers")


def _validate_result(result: FiniteDomainResult) -> None:
    """Reject malformed shard objects before they enter an additive reduction."""

    if not isinstance(result, FiniteDomainResult):
        raise TypeError("shards must contain FiniteDomainResult objects")
    edges = np.asarray(result.ell_bin_edges)
    if (
        edges.ndim != 1
        or edges.size < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("ell_bin_edges must be finite and strictly increasing")
    n_ell = edges.size - 1
    moment_shape = (
        len(result.q_names),
        len(result.geometry_names),
        len(result.measurement_names),
        len(result.direction_names),
        len(result.p_values),
        n_ell,
    )
    exclusions_shape = (
        len(result.q_names),
        len(result.geometry_names),
        len(result.exclusion_names),
        n_ell,
    )
    expected_shapes = {
        "counts": moment_shape,
        "sums": moment_shape,
        "sums_sq": moment_shape,
        "exclusions": exclusions_shape,
        "sampled_pairs": (n_ell,),
        "eligible_pairs": (n_ell,),
        "cube_candidate_pairs": (n_ell,),
        "excluded_boundary_pairs": (n_ell,),
        "displacements_per_bin": (n_ell,),
        "elapsed_seconds_per_ell_bin": (n_ell,),
        "intrinsic_eligible_origins": (n_ell,),
        "boundary_excluded_origins": (n_ell,),
        "support_policy_excluded_origins": (n_ell,),
    }
    integer_names = {
        "counts",
        "exclusions",
        "sampled_pairs",
        "eligible_pairs",
        "cube_candidate_pairs",
        "excluded_boundary_pairs",
        "displacements_per_bin",
        "intrinsic_eligible_origins",
        "boundary_excluded_origins",
        "support_policy_excluded_origins",
    }
    for name, expected_shape in expected_shapes.items():
        array = np.asarray(getattr(result, name))
        if array.shape != expected_shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {expected_shape}")
        if name in integer_names:
            _validate_nonnegative_integer_array(array, name)
        elif np.any(~np.isfinite(array)) or (
            name == "elapsed_seconds_per_ell_bin" and np.any(array < 0.0)
        ):
            raise ValueError(f"{name} must contain finite non-negative values")

    offsets = np.asarray(result.displacements_ijk)
    if offsets.ndim != 2 or offsets.shape[1:] != (3,):
        raise ValueError("displacements_ijk must have shape (n, 3)")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("displacements_ijk must contain integers")
    if np.any(np.all(offsets == 0, axis=1)):
        raise ValueError("zero displacement offsets are not permitted")
    if len({tuple(row) for row in offsets.tolist()}) != len(offsets):
        raise ValueError("a shard must not contain duplicate displacement offsets")
    offset_count = len(offsets)
    for name in _OFFSET_ARRAY_NAMES[1:]:
        array = np.asarray(getattr(result, name))
        if array.shape != (offset_count,):
            raise ValueError(f"{name} must have one entry per displacement")
        if name != "ell_bin_index_per_displacement":
            _validate_nonnegative_integer_array(array, name)

    ell_bin_indices = np.asarray(result.ell_bin_index_per_displacement)
    if not np.issubdtype(ell_bin_indices.dtype, np.integer):
        raise ValueError("ell_bin_index_per_displacement must contain integers")
    expected_ell_bin_indices = _ell_bin_indices(result)
    if not np.array_equal(ell_bin_indices, expected_ell_bin_indices):
        raise ValueError("ell_bin_index_per_displacement is inconsistent with the offsets")
    if result.out_of_range_displacements != int(np.count_nonzero(ell_bin_indices < 0)):
        raise ValueError("out_of_range_displacements is inconsistent with the offsets")
    expected_displacements = _sum_by_ell_bin(
        np.ones(offset_count, dtype=np.int64), ell_bin_indices, n_ell
    )
    if not np.array_equal(result.displacements_per_bin, expected_displacements):
        raise ValueError("displacements_per_bin is inconsistent with offset metadata")
    for aggregate_name in (
        "sampled_pairs",
        "eligible_pairs",
        "cube_candidate_pairs",
        "excluded_boundary_pairs",
    ):
        per_displacement = np.asarray(getattr(result, f"{aggregate_name}_per_displacement"))
        expected = _sum_by_ell_bin(per_displacement, ell_bin_indices, n_ell)
        if not np.array_equal(getattr(result, aggregate_name), expected):
            raise ValueError(f"{aggregate_name} is inconsistent with offset support")
    if not np.array_equal(
        result.cube_candidate_pairs_per_displacement - result.eligible_pairs_per_displacement,
        result.excluded_boundary_pairs_per_displacement,
    ):
        raise ValueError("offset boundary exclusions are inconsistent with eligible support")
    if np.any(result.sampled_pairs_per_displacement > result.eligible_pairs_per_displacement):
        raise ValueError("sampled offset support must not exceed eligible support")
    assert result.intrinsic_eligible_origins_per_displacement is not None
    assert result.boundary_excluded_origins_per_displacement is not None
    assert result.support_policy_excluded_origins_per_displacement is not None
    assert result.intrinsic_eligible_origins is not None
    assert result.boundary_excluded_origins is not None
    assert result.support_policy_excluded_origins is not None
    if not np.array_equal(
        result.cube_candidate_pairs_per_displacement
        - result.intrinsic_eligible_origins_per_displacement,
        result.boundary_excluded_origins_per_displacement,
    ):
        raise ValueError("intrinsic boundary accounting is inconsistent")
    if not np.array_equal(
        result.intrinsic_eligible_origins_per_displacement
        - result.eligible_pairs_per_displacement,
        result.support_policy_excluded_origins_per_displacement,
    ):
        raise ValueError("support-policy accounting is inconsistent")
    for aggregate_name in (
        "intrinsic_eligible_origins",
        "boundary_excluded_origins",
        "support_policy_excluded_origins",
    ):
        per_displacement = np.asarray(getattr(result, f"{aggregate_name}_per_displacement"))
        expected = _sum_by_ell_bin(per_displacement, ell_bin_indices, n_ell)
        if not np.array_equal(getattr(result, aggregate_name), expected):
            raise ValueError(f"{aggregate_name} is inconsistent with offset support")

    block_arrays = tuple(getattr(result, name) for name in _BLOCK_ARRAY_NAMES)
    if result.block_shape_kji is None:
        if any(array is not None for array in block_arrays):
            raise ValueError("block accumulators require block_shape_kji metadata")
        return
    if any(array is None for array in block_arrays):
        raise ValueError("block_shape_kji requires all block accumulators")
    block_count = int(
        np.prod(
            [
                (size + block - 1) // block
                for size, block in zip(result.cube_shape_kji, result.block_shape_kji)
            ],
            dtype=np.int64,
        )
    )
    expected_block_shapes = {
        "block_counts": (block_count, *moment_shape),
        "block_sums": (block_count, *moment_shape),
        "block_sums_sq": (block_count, *moment_shape),
        "block_sampled_origins": (block_count, n_ell),
        "block_eligible_origins": (block_count, n_ell),
        "block_exclusions": (block_count, *exclusions_shape),
    }
    for name, array in zip(_BLOCK_ARRAY_NAMES, block_arrays):
        assert array is not None
        expected_block_shape = expected_block_shapes[name]
        if array.shape != expected_block_shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {expected_block_shape}")
        if name in {"block_counts", "block_sampled_origins", "block_eligible_origins", "block_exclusions"}:
            _validate_nonnegative_integer_array(array, name)
        elif np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values")
    assert result.block_counts is not None
    assert result.block_sums is not None
    assert result.block_sums_sq is not None
    if not np.array_equal(result.block_counts.sum(axis=0), result.counts):
        raise ValueError("block_counts do not reproduce aggregate counts")
    if not np.allclose(result.block_sums.sum(axis=0), result.sums, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("block_sums do not reproduce aggregate sums")
    if not np.allclose(
        result.block_sums_sq.sum(axis=0), result.sums_sq, rtol=1.0e-12, atol=1.0e-12
    ):
        raise ValueError("block_sums_sq do not reproduce aggregate sums_sq")
    assert result.block_sampled_origins is not None
    assert result.block_eligible_origins is not None
    assert result.block_exclusions is not None
    if not np.array_equal(result.block_sampled_origins.sum(axis=0), result.sampled_pairs):
        raise ValueError("block_sampled_origins do not reproduce sampled_pairs")
    if not np.array_equal(result.block_eligible_origins.sum(axis=0), result.eligible_pairs):
        raise ValueError("block_eligible_origins do not reproduce eligible_pairs")
    if not np.array_equal(result.block_exclusions.sum(axis=0), result.exclusions):
        raise ValueError("block_exclusions do not reproduce exclusions")


def _canonical_shard_key(result: FiniteDomainResult) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted(tuple(int(value) for value in row) for row in result.displacements_ijk))


def _sum_arrays(shards: Sequence[FiniteDomainResult], name: str) -> np.ndarray:
    output = np.zeros_like(np.asarray(getattr(shards[0], name)))
    for shard in shards:
        output += np.asarray(getattr(shard, name))
    return output


def _validated_shard_ids(
    shard_ids: Iterable[Hashable] | None,
    expected_shard_ids: Iterable[Hashable] | None,
    shard_count: int,
) -> None:
    if shard_ids is None:
        if expected_shard_ids is not None:
            raise ValueError("expected_shard_ids requires supplied shard identifiers")
        return
    actual = tuple(shard_ids)
    if len(actual) != shard_count:
        raise ValueError("shard_ids must contain one identifier per shard")
    try:
        actual_set = set(actual)
    except TypeError as error:
        raise ValueError("shard identifiers must be hashable") from error
    if len(actual_set) != len(actual):
        raise ValueError("overlapping shard identifiers are not permitted")
    if expected_shard_ids is None:
        return
    expected = tuple(expected_shard_ids)
    try:
        expected_set = set(expected)
    except TypeError as error:
        raise ValueError("expected shard identifiers must be hashable") from error
    if len(expected_set) != len(expected):
        raise ValueError("expected_shard_ids must not contain duplicates")
    missing = expected_set - actual_set
    unexpected = actual_set - expected_set
    if missing or unexpected:
        raise ValueError(
            f"shard identifier inventory mismatch: missing={sorted(map(repr, missing))}, "
            f"unexpected={sorted(map(repr, unexpected))}"
        )


def reduce_finite_domain_shards(
    shards: Iterable[FiniteDomainResult] | Mapping[Hashable, FiniteDomainResult],
    *,
    shard_ids: Iterable[Hashable] | None = None,
    expected_shard_ids: Iterable[Hashable] | None = None,
) -> FiniteDomainResult:
    """Strictly reduce additive finite-domain displacement shards.

    ``shard_ids`` are optional for in-memory reductions.  Supply both
    ``shard_ids`` and ``expected_shard_ids`` (or pass a mapping plus
    ``expected_shard_ids``) when reducing a restartable manifest so omitted,
    duplicate, and unexpected partials are rejected.
    """

    if isinstance(shards, Mapping):
        if shard_ids is not None:
            raise ValueError("shard_ids must not be supplied when shards is a mapping")
        shard_ids = tuple(shards)
        shard_results = tuple(shards.values())
    else:
        shard_results = tuple(shards)
    if not shard_results:
        raise ValueError("at least one finite-domain shard is required")
    _validated_shard_ids(shard_ids, expected_shard_ids, len(shard_results))
    for shard in shard_results:
        _validate_result(shard)

    ordered = tuple(sorted(shard_results, key=_canonical_shard_key))
    reference = ordered[0]
    for shard in ordered[1:]:
        for name in _COMPATIBLE_METADATA_NAMES:
            if not _metadata_equal(getattr(reference, name), getattr(shard, name)):
                raise ValueError(f"incompatible shard metadata: {name}")
        for name in _AGGREGATE_ARRAY_NAMES:
            if np.asarray(getattr(reference, name)).shape != np.asarray(getattr(shard, name)).shape:
                raise ValueError(f"incompatible shard accumulator shape: {name}")

    seen_offsets: set[tuple[int, int, int]] = set()
    for shard in ordered:
        offsets = {tuple(int(value) for value in row) for row in shard.displacements_ijk}
        overlap = seen_offsets & offsets
        if overlap:
            raise ValueError(f"overlapping displacement offsets are not permitted: {sorted(overlap)}")
        seen_offsets.update(offsets)

    reduced = deepcopy(reference)
    for name in _AGGREGATE_ARRAY_NAMES:
        setattr(reduced, name, _sum_arrays(ordered, name))
    reduced.out_of_range_displacements = sum(
        shard.out_of_range_displacements for shard in ordered
    )
    reduced.elapsed_seconds = sum(shard.elapsed_seconds for shard in ordered)
    if reference.block_counts is not None:
        for name in _BLOCK_ARRAY_NAMES:
            setattr(reduced, name, _sum_arrays(ordered, name))

    offsets = np.concatenate([shard.displacements_ijk for shard in ordered], axis=0)
    order = np.lexsort((offsets[:, 2], offsets[:, 1], offsets[:, 0]))
    for name in _OFFSET_ARRAY_NAMES:
        values = np.concatenate([np.asarray(getattr(shard, name)) for shard in ordered], axis=0)
        setattr(reduced, name, values[order].copy())
    _validate_result(reduced)
    return reduced


def merge_finite_domain_shards(
    left: FiniteDomainResult,
    right: FiniteDomainResult,
) -> FiniteDomainResult:
    """Merge two compatible, disjoint finite-domain displacement shards."""

    return reduce_finite_domain_shards((left, right))


def _metadata_payload(result: FiniteDomainResult) -> dict[str, Any]:
    return {
        "format": _NPZ_FORMAT,
        "version": _NPZ_VERSION,
        "pair_mode": result.pair_mode,
        "cube_shape_kji": result.cube_shape_kji,
        "nested_core_bounds_kji": result.nested_core_bounds_kji,
        "rho0": result.rho0,
        "rho0_provenance": result.rho0_provenance,
        "cell_sizes": result.cell_sizes,
        "angle_limits": result.angle_limits,
        "sample_count": result.sample_count,
        "pair_batch_size": result.pair_batch_size,
        "seed": result.seed,
        "elapsed_seconds": result.elapsed_seconds,
        "out_of_range_displacements": result.out_of_range_displacements,
        "stencil_width": result.stencil_width,
        "shell_core_bounds_kji": result.shell_core_bounds_kji,
        "block_shape_kji": result.block_shape_kji,
        "support_displacements_sha256": result.support_displacements_sha256,
        "support_displacement_count": result.support_displacement_count,
        "block_assignment": result.block_assignment,
        "has_block_accumulators": result.block_counts is not None,
    }


def finite_domain_partial_to_npz_payload(result: FiniteDomainResult) -> dict[str, np.ndarray]:
    """Return a pickle-free NPZ payload for one validated Phase 3a partial."""

    _validate_result(result)
    payload = {
        name: np.asarray(getattr(result, name)) for name in _AGGREGATE_ARRAY_NAMES
    }
    payload.update({name: np.asarray(getattr(result, name)) for name in _OFFSET_ARRAY_NAMES})
    payload.update(
        {name: np.asarray(getattr(result, name), dtype=str) for name in _LABEL_ARRAY_NAMES}
    )
    payload["ell_bin_edges"] = np.asarray(result.ell_bin_edges)
    payload["p_values"] = np.asarray(result.p_values)
    payload["metadata_json"] = np.asarray(json.dumps(_metadata_payload(result), sort_keys=True))
    if result.block_counts is not None:
        payload.update({name: np.asarray(getattr(result, name)) for name in _BLOCK_ARRAY_NAMES})
    if any(array.dtype.kind == "O" for array in payload.values()):
        raise ValueError("Phase 3a NPZ payloads must not contain object arrays")
    return payload


def _required_payload_array(payload: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"finite-domain partial payload is missing {name}")
    array = np.asarray(payload[name])
    if array.dtype.kind == "O":
        raise ValueError(f"finite-domain partial payload contains object array {name}")
    return array.copy()


def _string_tuple(payload: Mapping[str, Any], name: str) -> tuple[str, ...]:
    values = _required_payload_array(payload, name)
    if values.ndim != 1 or values.dtype.kind not in "SU":
        raise ValueError(f"{name} must be a one-dimensional string array")
    return tuple(str(value) for value in values.tolist())


def _bounds_from_json(
    values: list[list[int]] | None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if values is None:
        return None
    if len(values) != 3 or any(len(bound) != 2 for bound in values):
        raise ValueError("invalid KJI bounds in finite-domain metadata")
    return tuple(tuple(int(value) for value in bound) for bound in values)  # type: ignore[return-value]


def finite_domain_partial_from_npz_payload(payload: Mapping[str, Any]) -> FiniteDomainResult:
    """Reconstruct and validate one Phase 3a partial from a pickle-free payload."""

    raw_metadata = _required_payload_array(payload, "metadata_json")
    if raw_metadata.shape != () or raw_metadata.dtype.kind not in "SU":
        raise ValueError("metadata_json must be one string scalar")
    metadata = json.loads(str(raw_metadata.item()))
    if metadata.get("format") != _NPZ_FORMAT or metadata.get("version") != _NPZ_VERSION:
        raise ValueError("unsupported finite-domain partial NPZ format")
    shell_bounds_json = metadata["shell_core_bounds_kji"]
    shell_bounds = (
        tuple(_bounds_from_json(bounds) for bounds in shell_bounds_json)
        if shell_bounds_json is not None
        else None
    )
    has_blocks = bool(metadata["has_block_accumulators"])
    block_arrays = {
        name: _required_payload_array(payload, name) if has_blocks else None
        for name in _BLOCK_ARRAY_NAMES
    }
    result = FiniteDomainResult(
        q_names=_string_tuple(payload, "q_names"),
        density_conventions=_string_tuple(payload, "density_conventions"),
        geometry_names=_string_tuple(payload, "geometry_names"),
        measurement_names=_string_tuple(payload, "measurement_names"),
        direction_names=_string_tuple(payload, "direction_names"),
        exclusion_names=_string_tuple(payload, "exclusion_names"),
        ell_bin_edges=_required_payload_array(payload, "ell_bin_edges"),
        p_values=tuple(float(value) for value in _required_payload_array(payload, "p_values")),
        counts=_required_payload_array(payload, "counts"),
        sums=_required_payload_array(payload, "sums"),
        sums_sq=_required_payload_array(payload, "sums_sq"),
        exclusions=_required_payload_array(payload, "exclusions"),
        sampled_pairs=_required_payload_array(payload, "sampled_pairs"),
        eligible_pairs=_required_payload_array(payload, "eligible_pairs"),
        cube_candidate_pairs=_required_payload_array(payload, "cube_candidate_pairs"),
        excluded_boundary_pairs=_required_payload_array(payload, "excluded_boundary_pairs"),
        displacements_per_bin=_required_payload_array(payload, "displacements_per_bin"),
        displacements_ijk=_required_payload_array(payload, "displacements_ijk"),
        ell_bin_index_per_displacement=_required_payload_array(
            payload, "ell_bin_index_per_displacement"
        ),
        sampled_pairs_per_displacement=_required_payload_array(
            payload, "sampled_pairs_per_displacement"
        ),
        eligible_pairs_per_displacement=_required_payload_array(
            payload, "eligible_pairs_per_displacement"
        ),
        cube_candidate_pairs_per_displacement=_required_payload_array(
            payload, "cube_candidate_pairs_per_displacement"
        ),
        excluded_boundary_pairs_per_displacement=_required_payload_array(
            payload, "excluded_boundary_pairs_per_displacement"
        ),
        intrinsic_eligible_origins=_required_payload_array(payload, "intrinsic_eligible_origins"),
        boundary_excluded_origins=_required_payload_array(payload, "boundary_excluded_origins"),
        support_policy_excluded_origins=_required_payload_array(payload, "support_policy_excluded_origins"),
        intrinsic_eligible_origins_per_displacement=_required_payload_array(
            payload, "intrinsic_eligible_origins_per_displacement"
        ),
        boundary_excluded_origins_per_displacement=_required_payload_array(
            payload, "boundary_excluded_origins_per_displacement"
        ),
        support_policy_excluded_origins_per_displacement=_required_payload_array(
            payload, "support_policy_excluded_origins_per_displacement"
        ),
        out_of_range_displacements=int(metadata["out_of_range_displacements"]),
        pair_mode=str(metadata["pair_mode"]),
        cube_shape_kji=tuple(int(value) for value in metadata["cube_shape_kji"]),
        nested_core_bounds_kji=_bounds_from_json(metadata["nested_core_bounds_kji"]),
        rho0=float(metadata["rho0"]),
        rho0_provenance=str(metadata["rho0_provenance"]),
        cell_sizes=tuple(float(value) for value in metadata["cell_sizes"]),
        angle_limits={str(name): float(value) for name, value in metadata["angle_limits"].items()},
        sample_count=(
            int(metadata["sample_count"]) if metadata["sample_count"] is not None else None
        ),
        pair_batch_size=int(metadata["pair_batch_size"]),
        seed=int(metadata["seed"]),
        elapsed_seconds=float(metadata["elapsed_seconds"]),
        elapsed_seconds_per_ell_bin=_required_payload_array(
            payload, "elapsed_seconds_per_ell_bin"
        ),
        stencil_width=int(metadata["stencil_width"]),
        shell_core_bounds_kji=shell_bounds,
        block_shape_kji=(
            tuple(int(value) for value in metadata["block_shape_kji"])
            if metadata["block_shape_kji"] is not None
            else None
        ),
        support_displacements_sha256=str(metadata["support_displacements_sha256"]),
        support_displacement_count=int(metadata["support_displacement_count"]),
        block_assignment=(
            str(metadata["block_assignment"])
            if metadata["block_assignment"] is not None
            else None
        ),
        **block_arrays,
    )
    _validate_result(result)
    return result


def save_finite_domain_partial_npz(
    path: str | PathLike[str],
    result: FiniteDomainResult,
) -> None:
    """Write one validated partial as a compressed, pickle-free NPZ file."""

    np.savez_compressed(path, **finite_domain_partial_to_npz_payload(result))


def load_finite_domain_partial_npz(path: str | PathLike[str]) -> FiniteDomainResult:
    """Load one partial with ``allow_pickle=False`` and reconstruct its result."""

    with np.load(path, allow_pickle=False) as payload:
        return finite_domain_partial_from_npz_payload(payload)


def _block_accumulators(result: FiniteDomainResult) -> tuple[np.ndarray, np.ndarray]:
    _validate_result(result)
    if result.block_counts is None or result.block_sums is None:
        raise ValueError("spatial block uncertainty requires block accumulators")
    return result.block_counts, result.block_sums


def _effective_block_counts(block_counts: np.ndarray) -> np.ndarray:
    """Return Kish effective block counts from accepted-sample populations."""

    counts = np.asarray(block_counts, dtype=float)
    numerator = np.square(counts.sum(axis=0))
    denominator = np.square(counts).sum(axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def block_jackknife_moment_uncertainty(result: FiniteDomainResult) -> MomentUncertainty:
    """Estimate moment uncertainty by deleting each fixed-layout origin block.

    Empty blocks remain represented in the shared geometric layout, but they
    are not delete-one replicates for a cell to which they contributed
    nothing. Cells with fewer than two contributing delete-one replicates
    receive ``NaN`` uncertainty.
    """

    block_counts, block_sums = _block_accumulators(result)
    remaining_counts = result.counts[None, ...] - block_counts
    remaining_sums = result.sums[None, ...] - block_sums
    contributing = block_counts > 0
    replicate_valid = contributing & (remaining_counts > 0)
    replicates = np.full_like(block_sums, np.nan, dtype=float)
    np.divide(remaining_sums, remaining_counts, out=replicates, where=replicate_valid)
    contributing_blocks = np.count_nonzero(contributing, axis=0)
    replicate_counts = np.count_nonzero(replicate_valid, axis=0)
    replicate_mean = np.divide(
        np.where(replicate_valid, replicates, 0.0).sum(axis=0),
        replicate_counts,
        out=np.full_like(result.sums, np.nan, dtype=float),
        where=replicate_counts > 0,
    )
    squared_deviation = np.where(
        replicate_valid, np.square(replicates - replicate_mean[None, ...]), 0.0
    ).sum(axis=0)
    standard_error = np.sqrt(
        np.divide(
            (replicate_counts - 1) * squared_deviation,
            replicate_counts,
            out=np.full_like(result.sums, np.nan, dtype=float),
            where=replicate_counts > 1,
        )
    )
    standard_error[contributing_blocks < 2] = np.nan
    return MomentUncertainty(
        method="spatial_block_jackknife",
        estimate=result.moments.copy(),
        standard_error=standard_error,
        contributing_blocks=contributing_blocks,
        effective_blocks=_effective_block_counts(block_counts),
        geometric_block_count=block_counts.shape[0],
        resampling_population="delete_contributing_blocks_only",
    )


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def block_bootstrap_moment_uncertainty(
    result: FiniteDomainResult,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence_level: float = 0.95,
    return_replicates: bool = False,
) -> MomentUncertainty:
    """Estimate uncertainty with a deterministic fixed-layout block bootstrap.

    Each replicate samples ``n_blocks`` block identifiers with replacement
    from the complete 3-D block layout, including empty blocks. One seeded
    schedule is shared across all output cells so cross-curve covariance is
    retained. Effective accepted-block counts are reported separately so a
    nominally large layout cannot hide weak spatial support.
    """

    block_counts, block_sums = _block_accumulators(result)
    n_resamples = _positive_int(n_resamples, "n_resamples")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    block_count = block_counts.shape[0]
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(block_count, 1.0 / block_count)
    replicates = np.full((n_resamples, *result.sums.shape), np.nan, dtype=float)
    for index in range(n_resamples):
        multiplicity = rng.multinomial(block_count, probabilities)
        sampled_counts = np.tensordot(multiplicity, block_counts, axes=(0, 0))
        sampled_sums = np.tensordot(multiplicity, block_sums, axes=(0, 0))
        np.divide(
            sampled_sums,
            sampled_counts,
            out=replicates[index],
            where=sampled_counts > 0,
        )

    contributing_blocks = np.count_nonzero(block_counts > 0, axis=0)
    standard_error = np.full_like(result.sums, np.nan, dtype=float)
    interval_low = np.full_like(result.sums, np.nan, dtype=float)
    interval_high = np.full_like(result.sums, np.nan, dtype=float)
    valid_resamples = np.count_nonzero(np.isfinite(replicates), axis=0)
    alpha = 0.5 * (1.0 - confidence_level)
    for index in np.ndindex(result.sums.shape):
        if contributing_blocks[index] < 2:
            continue
        finite = replicates[(slice(None), *index)]
        finite = finite[np.isfinite(finite)]
        if finite.size < 2:
            continue
        standard_error[index] = np.std(finite, ddof=1)
        interval_low[index], interval_high[index] = np.quantile(finite, (alpha, 1.0 - alpha))
    return MomentUncertainty(
        method="spatial_block_bootstrap",
        estimate=result.moments.copy(),
        standard_error=standard_error,
        contributing_blocks=contributing_blocks,
        effective_blocks=_effective_block_counts(block_counts),
        geometric_block_count=block_count,
        resampling_population="fixed_geometric_layout_including_empty_blocks",
        interval_low=interval_low,
        interval_high=interval_high,
        valid_resamples=valid_resamples,
        seed=int(seed),
        n_resamples=n_resamples,
        confidence_level=float(confidence_level),
        replicate_moments=replicates if return_replicates else None,
    )


def local_log_slope(
    ell: np.ndarray,
    moments: np.ndarray,
    *,
    window: int = 5,
) -> np.ndarray:
    """Return centered local regressions of ``d log(moment) / d log(ell)``.

    The documented smoothing rule is an ordinary least-squares fit over
    exactly ``window`` contiguous bins centered on each reported bin.
    ``window`` must be odd and at least three.  Edge bins without a complete
    centered window, and windows containing non-positive or non-finite
    moments, are returned as ``NaN``.
    """

    ell = np.asarray(ell, dtype=float)
    moments = np.asarray(moments, dtype=float)
    if ell.ndim != 1 or ell.size < 3:
        raise ValueError("ell must be a one-dimensional array with at least three values")
    if np.any(~np.isfinite(ell)) or np.any(ell <= 0.0) or np.any(np.diff(ell) <= 0.0):
        raise ValueError("ell must contain finite, positive, strictly increasing values")
    if moments.ndim < 1 or moments.shape[-1] != ell.size:
        raise ValueError("the final moments axis must match ell")
    window = _positive_int(window, "window")
    if window < 3 or window % 2 == 0 or window > ell.size:
        raise ValueError("window must be odd, at least three, and no larger than ell")
    output = np.full_like(moments, np.nan, dtype=float)
    radius = window // 2
    log_ell = np.log(ell)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_moments = np.log(moments)
    for center in range(radius, ell.size - radius):
        x = log_ell[center - radius : center + radius + 1]
        x_centered = x - np.mean(x)
        y = log_moments[..., center - radius : center + radius + 1]
        valid = np.all(np.isfinite(y), axis=-1)
        slope = np.sum(y * x_centered, axis=-1) / np.sum(np.square(x_centered))
        output[..., center] = np.where(valid, slope, np.nan)
    return output
