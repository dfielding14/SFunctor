"""Readable finite-domain 3-D oracle for small synthetic cubes.

This module intentionally favors explicit loops. It exists to validate the
batched implementation in :mod:`sfunctor.core.finite_domain`.
"""
from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Sequence

import numpy as np

from sfunctor.core.directional import DIRECTION_NAMES, EXCLUSION_NAMES, QField
from sfunctor.core.finite_domain import (
    FINITE_DOMAIN_GEOMETRY_NAMES,
    MEASUREMENT_NAMES,
    FiniteDomainConfig,
    FiniteDomainResult,
    build_cube_q_variants,
    cube_offset_to_vector,
    nested_core_bounds_kji,
    stencil_definition,
    _block_population_for_bounds,
    _ell_bin_index as _fast_ell_bin_index,
    _origin_block_ids,
    _require_integer_displacements,
    _require_signed_closure,
    _shell_seed,
    valid_origin_bounds_kji,
)

__all__ = ["compute_finite_domain_structure_functions_reference"]


def _ell_bin_index(value: float, edges: np.ndarray) -> int:
    if value == edges[-1]:
        return edges.size - 2
    for index in range(edges.size - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return -1


def _bounds_size(bounds: Sequence[tuple[int, int]]) -> int:
    return int(np.prod([max(0, stop - start) for start, stop in bounds], dtype=np.int64))


def _origins(
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    sample_count: int | None,
    seed: int,
) -> list[tuple[int, int, int]]:
    output = [
        (k, j, i)
        for k in range(*bounds[0])
        for j in range(*bounds[1])
        for i in range(*bounds[2])
    ]
    if sample_count is None or sample_count >= len(output):
        return output
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    while len(selected) < sample_count:
        selected.add(int(rng.integers(0, len(output))))
    return [output[index] for index in sorted(selected)]


def _offset_seed(seed: int, displacement: Sequence[int]) -> int:
    import hashlib
    import json

    payload = json.dumps([int(seed), *(int(value) for value in displacement)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _folded_angle(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return np.nan
    cosine = abs(float(np.dot(left, right))) / denominator
    return float(np.arccos(np.clip(cosine, 0.0, 1.0)))


def _empty_result(
    q_fields: dict[str, QField],
    geometry_names: tuple[str, ...],
    config: FiniteDomainConfig,
    cube_shape_kji: tuple[int, int, int],
    core_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None,
    rho0: float,
    rho0_provenance: str,
    displacements_ijk: np.ndarray,
    support_displacements_ijk: np.ndarray,
) -> FiniteDomainResult:
    n_ell = config.ell_bin_edges.size - 1
    displacement_count = len(displacements_ijk)
    shape = (
        len(q_fields),
        len(geometry_names),
        len(MEASUREMENT_NAMES),
        len(DIRECTION_NAMES),
        len(config.p_values),
        n_ell,
    )
    exclusions_shape = (len(q_fields), len(geometry_names), len(EXCLUSION_NAMES), n_ell)
    block_count = (
        int(
            np.prod(
                [
                    (size + block - 1) // block
                    for size, block in zip(cube_shape_kji, config.block_shape_kji)
                ],
                dtype=np.int64,
            )
        )
        if config.block_shape_kji is not None
        else 0
    )
    return FiniteDomainResult(
        q_names=tuple(q_fields),
        density_conventions=tuple(field.density_convention for field in q_fields.values()),
        geometry_names=geometry_names,
        measurement_names=MEASUREMENT_NAMES,
        direction_names=DIRECTION_NAMES,
        exclusion_names=EXCLUSION_NAMES,
        ell_bin_edges=config.ell_bin_edges.copy(),
        p_values=config.p_values,
        counts=np.zeros(shape, dtype=np.int64),
        sums=np.zeros(shape, dtype=float),
        sums_sq=np.zeros(shape, dtype=float),
        exclusions=np.zeros(exclusions_shape, dtype=np.int64),
        sampled_pairs=np.zeros(n_ell, dtype=np.int64),
        eligible_pairs=np.zeros(n_ell, dtype=np.int64),
        cube_candidate_pairs=np.zeros(n_ell, dtype=np.int64),
        excluded_boundary_pairs=np.zeros(n_ell, dtype=np.int64),
        displacements_per_bin=np.zeros(n_ell, dtype=np.int64),
        displacements_ijk=displacements_ijk.copy(),
        ell_bin_index_per_displacement=np.full(displacement_count, -1, dtype=np.int64),
        sampled_pairs_per_displacement=np.zeros(displacement_count, dtype=np.int64),
        eligible_pairs_per_displacement=np.zeros(displacement_count, dtype=np.int64),
        cube_candidate_pairs_per_displacement=np.zeros(displacement_count, dtype=np.int64),
        excluded_boundary_pairs_per_displacement=np.zeros(displacement_count, dtype=np.int64),
        out_of_range_displacements=0,
        pair_mode=config.pair_mode,
        cube_shape_kji=cube_shape_kji,
        nested_core_bounds_kji=core_bounds,
        rho0=rho0,
        rho0_provenance=rho0_provenance,
        cell_sizes=tuple(config.cell_sizes),
        angle_limits={
            "theta_parallel_max": config.theta_parallel_max,
            "theta_perpendicular_min": config.theta_perpendicular_min,
            "phi_xi_max": config.phi_xi_max,
            "phi_lambda_min": config.phi_lambda_min,
        },
        sample_count=config.sample_count,
        pair_batch_size=config.pair_batch_size,
        seed=config.seed,
        elapsed_seconds=0.0,
        elapsed_seconds_per_ell_bin=np.zeros(n_ell, dtype=float),
        stencil_width=config.stencil_width,
        block_shape_kji=config.block_shape_kji,
        block_counts=np.zeros((block_count, *shape), dtype=np.int64) if block_count else None,
        block_sums=np.zeros((block_count, *shape), dtype=float) if block_count else None,
        block_sums_sq=np.zeros((block_count, *shape), dtype=float) if block_count else None,
        support_displacements_sha256=hashlib.sha256(
            support_displacements_ijk.tobytes()
        ).hexdigest(),
        support_displacement_count=len(support_displacements_ijk),
        block_assignment="stencil_midpoint" if block_count else None,
        block_sampled_origins=np.zeros((block_count, n_ell), dtype=np.int64) if block_count else None,
        block_eligible_origins=np.zeros((block_count, n_ell), dtype=np.int64) if block_count else None,
        block_exclusions=np.zeros((block_count, *exclusions_shape), dtype=np.int64) if block_count else None,
        intrinsic_eligible_origins=np.zeros(n_ell, dtype=np.int64),
        boundary_excluded_origins=np.zeros(n_ell, dtype=np.int64),
        support_policy_excluded_origins=np.zeros(n_ell, dtype=np.int64),
        intrinsic_eligible_origins_per_displacement=np.zeros(displacement_count, dtype=np.int64),
        boundary_excluded_origins_per_displacement=np.zeros(displacement_count, dtype=np.int64),
        support_policy_excluded_origins_per_displacement=np.zeros(displacement_count, dtype=np.int64),
    )


def _record(
    result: FiniteDomainResult,
    q_index: int,
    geometry_index: int,
    measurement_index: int,
    direction: str,
    ell_index: int,
    magnitude: float,
    block_id: int | None = None,
) -> None:
    direction_index = result.direction_names.index(direction)
    for p_index, p_value in enumerate(result.p_values):
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.power(magnitude, p_value)
        if not np.isfinite(value):
            raise FloatingPointError("non-finite powered structure-function increment")
        index = (q_index, geometry_index, measurement_index, direction_index, p_index, ell_index)
        with np.errstate(over="ignore", invalid="ignore"):
            value_sq = value * value
            cumulative_sum = result.sums[index] + value
            cumulative_sum_sq = result.sums_sq[index] + value_sq
        if not np.isfinite(value_sq):
            raise FloatingPointError("non-finite structure-function accumulator contribution")
        if not np.isfinite(cumulative_sum) or not np.isfinite(cumulative_sum_sq):
            raise FloatingPointError("non-finite cumulative structure-function accumulator")
        result.counts[index] += 1
        result.sums[index] = cumulative_sum
        result.sums_sq[index] = cumulative_sum_sq
        if block_id is not None:
            assert result.block_counts is not None
            assert result.block_sums is not None
            assert result.block_sums_sq is not None
            result.block_counts[(block_id, *index)] += 1
            result.block_sums[(block_id, *index)] += value
            result.block_sums_sq[(block_id, *index)] += value_sq


def _record_block_exclusion(
    result: FiniteDomainResult,
    block_id: int | None,
    q_index: int,
    geometry_index: int,
    exclusion_index: int,
    ell_index: int,
) -> None:
    if block_id is None:
        return
    assert result.block_exclusions is not None
    result.block_exclusions[block_id, q_index, geometry_index, exclusion_index, ell_index] += 1


def compute_finite_domain_structure_functions_reference(
    cube_data: dict[str, np.ndarray],
    displacements_ijk: np.ndarray,
    *,
    config: FiniteDomainConfig,
    rho0: float | None = None,
    rho0_provenance: str | None = None,
    rho_floor: float = 0.0,
    q_names: Sequence[str] | None = None,
    support_displacements_ijk: np.ndarray | None = None,
) -> FiniteDomainResult:
    """Compute finite-domain moments with explicit point-pair loops."""

    started = perf_counter()
    displacements = _require_integer_displacements(displacements_ijk)
    support_displacements = (
        displacements
        if support_displacements_ijk is None
        else _require_integer_displacements(support_displacements_ijk)
    )
    if np.any(np.all(displacements == 0, axis=1)):
        raise ValueError("zero displacement is not permitted")
    if np.any(np.all(support_displacements == 0, axis=1)):
        raise ValueError("zero support displacement is not permitted")
    if len({tuple(row) for row in displacements.tolist()}) != len(displacements):
        raise ValueError("displacements_ijk must not contain duplicates")
    if len({tuple(row) for row in support_displacements.tolist()}) != len(
        support_displacements
    ):
        raise ValueError("support_displacements_ijk must not contain duplicates")
    measured_offsets = {tuple(row) for row in displacements.tolist()}
    support_offsets = {tuple(row) for row in support_displacements.tolist()}
    if not measured_offsets <= support_offsets:
        raise ValueError("every measured displacement must belong to support_displacements_ijk")
    if config.pair_mode == "nested_core":
        _require_signed_closure(support_displacements)

    inferred_rho0 = rho0 is None
    B, q_fields, rho0 = build_cube_q_variants(
        cube_data, q_names=q_names, rho0=rho0, rho_floor=rho_floor
    )
    if rho0_provenance is None:
        rho0_provenance = (
            "subvolume finite positive-density mean"
            if inferred_rho0 and np.isfinite(rho0)
            else "explicit configuration value"
            if np.isfinite(rho0)
            else "not applicable for requested q variants"
        )
    cube_shape = tuple(int(value) for value in B.shape[1:])
    geometry_names = (
        FINITE_DOMAIN_GEOMETRY_NAMES
        if config.include_subvolume_mean
        else FINITE_DOMAIN_GEOMETRY_NAMES[:1]
    )
    in_range_displacements = np.asarray(
        [
            displacement
            for displacement in displacements
            if _fast_ell_bin_index(
                float(np.linalg.norm(cube_offset_to_vector(displacement, config.cell_sizes))),
                config.ell_bin_edges,
            )
            >= 0
        ],
        dtype=np.int64,
    ).reshape((-1, 3))
    in_range_support_displacements = np.asarray(
        [
            displacement
            for displacement in support_displacements
            if _fast_ell_bin_index(
                float(np.linalg.norm(cube_offset_to_vector(displacement, config.cell_sizes))),
                config.ell_bin_edges,
            )
            >= 0
        ],
        dtype=np.int64,
    ).reshape((-1, 3))
    if config.pair_mode == "nested_core" and not in_range_displacements.size:
        raise ValueError("no in-range displacements remain for nested_core")
    core_bounds = (
        nested_core_bounds_kji(
            cube_shape, in_range_support_displacements, config.stencil_width
        )
        if config.pair_mode == "nested_core"
        else None
    )
    shell_bounds: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
    ] = [None] * (config.ell_bin_edges.size - 1)
    if config.pair_mode == "shell_local":
        for ell_index in range(len(shell_bounds)):
            members = np.asarray(
                [
                    displacement
                    for displacement in in_range_support_displacements
                    if _ell_bin_index(
                        float(np.linalg.norm(cube_offset_to_vector(displacement, config.cell_sizes))),
                        config.ell_bin_edges,
                    )
                    == ell_index
                ],
                dtype=np.int64,
            ).reshape((-1, 3))
            if not members.size:
                continue
            try:
                shell_bounds[ell_index] = nested_core_bounds_kji(
                    cube_shape, members, config.stencil_width
                )
            except ValueError:
                shell_bounds[ell_index] = ((0, 0), (0, 0), (0, 0))
    result = _empty_result(
        q_fields,
        geometry_names,
        config,
        cube_shape,
        core_bounds,
        rho0,
        rho0_provenance,
        displacements,
        support_displacements,
    )
    result.shell_core_bounds_kji = tuple(shell_bounds) if config.pair_mode == "shell_local" else None
    finite_B = np.all(np.isfinite(B), axis=0)
    B_mean_sub = np.mean(B[:, finite_B], axis=1) if np.any(finite_B) else np.full(3, np.nan)
    cube_cells = int(np.prod(cube_shape))
    exclusion_index = {name: index for index, name in enumerate(EXCLUSION_NAMES)}
    shared_origins = _origins(core_bounds, config.sample_count, config.seed) if core_bounds else None
    shell_origins = [
        _origins(bounds, config.sample_count, _shell_seed(config.seed, ell_index))
        if bounds is not None
        else None
        for ell_index, bounds in enumerate(shell_bounds)
    ]
    multipliers, increment_weights, local_B_weights = stencil_definition(config.stencil_width)

    for displacement_index, displacement in enumerate(displacements):
        displacement_started = perf_counter()
        r = cube_offset_to_vector(displacement, config.cell_sizes)
        ell_index = _ell_bin_index(float(np.linalg.norm(r)), config.ell_bin_edges)
        result.ell_bin_index_per_displacement[displacement_index] = ell_index
        if ell_index < 0:
            result.out_of_range_displacements += 1
            continue
        result.displacements_per_bin[ell_index] += 1
        intrinsic_bounds = valid_origin_bounds_kji(
            cube_shape, displacement, config.stencil_width
        )
        bounds = (
            core_bounds
            or shell_bounds[ell_index]
            or intrinsic_bounds
        )
        eligible = _bounds_size(bounds)
        intrinsic_eligible = _bounds_size(intrinsic_bounds)
        boundary_excluded = cube_cells - intrinsic_eligible
        support_policy_excluded = intrinsic_eligible - eligible
        result.eligible_pairs[ell_index] += eligible
        result.cube_candidate_pairs[ell_index] += cube_cells
        result.excluded_boundary_pairs[ell_index] += cube_cells - eligible
        result.eligible_pairs_per_displacement[displacement_index] = eligible
        result.cube_candidate_pairs_per_displacement[displacement_index] = cube_cells
        result.excluded_boundary_pairs_per_displacement[displacement_index] = cube_cells - eligible
        assert result.intrinsic_eligible_origins is not None
        assert result.boundary_excluded_origins is not None
        assert result.support_policy_excluded_origins is not None
        assert result.intrinsic_eligible_origins_per_displacement is not None
        assert result.boundary_excluded_origins_per_displacement is not None
        assert result.support_policy_excluded_origins_per_displacement is not None
        result.intrinsic_eligible_origins[ell_index] += intrinsic_eligible
        result.boundary_excluded_origins[ell_index] += boundary_excluded
        result.support_policy_excluded_origins[ell_index] += support_policy_excluded
        result.intrinsic_eligible_origins_per_displacement[displacement_index] = intrinsic_eligible
        result.boundary_excluded_origins_per_displacement[displacement_index] = boundary_excluded
        result.support_policy_excluded_origins_per_displacement[displacement_index] = support_policy_excluded
        block_eligible = _block_population_for_bounds(
            bounds,
            cube_shape,
            config.block_shape_kji,
            displacement,
            config.stencil_width,
        )
        if block_eligible is not None:
            assert result.block_eligible_origins is not None
            result.block_eligible_origins[:, ell_index] += block_eligible
        origins = (
            shared_origins
            if shared_origins is not None
            else shell_origins[ell_index]
            if shell_origins[ell_index] is not None
            else _origins(bounds, config.sample_count, _offset_seed(config.seed, displacement))
        )
        result.sampled_pairs[ell_index] += len(origins)
        result.sampled_pairs_per_displacement[displacement_index] = len(origins)
        di, dj, dk = (int(value) for value in displacement)

        for k0, j0, i0 in origins:
            block_ids = _origin_block_ids(
                (np.asarray((k0,)), np.asarray((j0,)), np.asarray((i0,))),
                cube_shape,
                config.block_shape_kji,
                displacement,
                config.stencil_width,
            )
            block_id = int(block_ids[0]) if block_ids is not None else None
            if block_id is not None:
                assert result.block_sampled_origins is not None
                result.block_sampled_origins[block_id, ell_index] += 1
            points = tuple(
                (k0 + multiplier * dk, j0 + multiplier * dj, i0 + multiplier * di)
                for multiplier in multipliers
            )
            if any(
                not (0 <= k < cube_shape[0] and 0 <= j < cube_shape[1] and 0 <= i < cube_shape[2])
                for k, j, i in points
            ):
                raise RuntimeError("finite-domain origin construction produced an invalid stencil point")
            B_points = tuple(B[:, k, j, i] for k, j, i in points)
            B_loc = sum(weight * value for weight, value in zip(local_B_weights, B_points))

            for geometry_index, geometry_name in enumerate(geometry_names):
                B_direction = B_loc if geometry_name == "pair_local" else B_mean_sub
                valid_B = bool(np.all(np.isfinite(B_direction)))
                if geometry_name == "pair_local":
                    valid_B &= bool(all(np.all(np.isfinite(value)) for value in B_points))
                B_mag = float(np.linalg.norm(B_direction))
                valid_parallel = valid_B and B_mag > config.B_epsilon
                e_parallel = B_direction / B_mag if valid_parallel else np.full(3, np.nan)
                theta = _folded_angle(r, e_parallel)
                r_perp = r - np.dot(r, e_parallel) * e_parallel
                r_perp_mag = float(np.linalg.norm(r_perp))

                for q_index, q_field in enumerate(q_fields.values()):
                    if not valid_B:
                        result.exclusions[q_index, geometry_index, exclusion_index["invalid_B"], ell_index] += 1
                        _record_block_exclusion(result, block_id, q_index, geometry_index, exclusion_index["invalid_B"], ell_index)
                    elif not valid_parallel:
                        result.exclusions[q_index, geometry_index, exclusion_index["weak_B_direction"], ell_index] += 1
                        _record_block_exclusion(result, block_id, q_index, geometry_index, exclusion_index["weak_B_direction"], ell_index)
                    q_points = tuple(q_field.values[:, k, j, i] for k, j, i in points)
                    valid_q = bool(all(q_field.valid[k, j, i] for k, j, i in points))
                    valid_q &= bool(all(np.all(np.isfinite(value)) for value in q_points))
                    if valid_parallel and not valid_q:
                        result.exclusions[q_index, geometry_index, exclusion_index["invalid_q"], ell_index] += 1
                        _record_block_exclusion(result, block_id, q_index, geometry_index, exclusion_index["invalid_q"], ell_index)
                    if not valid_parallel or not valid_q:
                        continue

                    delta_q = sum(
                        weight * value for weight, value in zip(increment_weights, q_points)
                    )
                    total_mag = float(np.linalg.norm(delta_q))
                    delta_q_perp = delta_q - np.dot(delta_q, e_parallel) * e_parallel
                    q_perp_mag = float(np.linalg.norm(delta_q_perp))
                    magnitudes = (total_mag, q_perp_mag)
                    for measurement_index, magnitude in enumerate(magnitudes):
                        _record(result, q_index, geometry_index, measurement_index, "all", ell_index, magnitude, block_id)
                        if theta <= config.theta_parallel_max:
                            _record(result, q_index, geometry_index, measurement_index, "parallel", ell_index, magnitude, block_id)
                        if theta >= config.theta_perpendicular_min:
                            _record(result, q_index, geometry_index, measurement_index, "perpendicular", ell_index, magnitude, block_id)

                    valid_q_perp = np.isfinite(q_perp_mag) and q_perp_mag > config.q_perp_epsilon
                    if not valid_q_perp:
                        result.exclusions[q_index, geometry_index, exclusion_index["weak_q_perp_for_phi"], ell_index] += 1
                        _record_block_exclusion(result, block_id, q_index, geometry_index, exclusion_index["weak_q_perp_for_phi"], ell_index)
                        continue
                    valid_r_perp = np.isfinite(r_perp_mag) and r_perp_mag > config.r_perp_epsilon
                    if not valid_r_perp:
                        result.exclusions[q_index, geometry_index, exclusion_index["weak_r_perp_for_phi"], ell_index] += 1
                        _record_block_exclusion(result, block_id, q_index, geometry_index, exclusion_index["weak_r_perp_for_phi"], ell_index)
                        continue
                    phi = _folded_angle(r_perp, delta_q_perp / q_perp_mag)
                    for measurement_index, magnitude in enumerate(magnitudes):
                        if theta >= config.theta_perpendicular_min and phi <= config.phi_xi_max:
                            _record(result, q_index, geometry_index, measurement_index, "xi", ell_index, magnitude, block_id)
                        if theta >= config.theta_perpendicular_min and phi >= config.phi_lambda_min:
                            _record(result, q_index, geometry_index, measurement_index, "lambda", ell_index, magnitude, block_id)
        result.elapsed_seconds_per_ell_bin[ell_index] += perf_counter() - displacement_started
    result.elapsed_seconds = perf_counter() - started
    return result
