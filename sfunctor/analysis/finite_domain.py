"""Reducers and serialization for finite-domain 3-D structure functions."""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from sfunctor.core.finite_domain import FiniteDomainResult

__all__ = [
    "constant_sp_shapes",
    "coverage_rows",
    "fit_directional_slopes",
    "fit_interval_sensitivity_rows",
    "geometric_bin_centers",
    "offset_support_rows",
    "result_to_npz_payload",
    "summary_rows",
]


def geometric_bin_centers(edges: np.ndarray) -> np.ndarray:
    """Return geometric centers for positive bins and arithmetic centers otherwise."""

    edges = np.asarray(edges, dtype=float)
    if np.any(edges <= 0.0):
        return 0.5 * (edges[:-1] + edges[1:])
    return np.sqrt(edges[:-1] * edges[1:])


def _indices(
    result: FiniteDomainResult,
    q_name: str,
    geometry_name: str,
    measurement_name: str,
    direction_name: str,
    p_value: float,
) -> tuple[int, int, int, int, int]:
    return (
        result.q_names.index(q_name),
        result.geometry_names.index(geometry_name),
        result.measurement_names.index(measurement_name),
        result.direction_names.index(direction_name),
        result.p_values.index(float(p_value)),
    )


def coverage_rows(
    result: FiniteDomainResult,
    *,
    min_count: int = 100,
    min_sampled_fraction: float = 0.0,
) -> list[dict[str, object]]:
    """Report angular occupancy and finite-domain support for every output bin."""

    if min_count < 1:
        raise ValueError("min_count must be positive")
    if not 0.0 <= min_sampled_fraction <= 1.0:
        raise ValueError("min_sampled_fraction must be in [0, 1]")
    ell = geometric_bin_centers(result.ell_bin_edges)
    rows: list[dict[str, object]] = []
    for qi, q_name in enumerate(result.q_names):
        for gi, geometry_name in enumerate(result.geometry_names):
            for mi, measurement_name in enumerate(result.measurement_names):
                for di, direction_name in enumerate(result.direction_names):
                    for pi, p_value in enumerate(result.p_values):
                        for ell_index, ell_center in enumerate(ell):
                            accepted = int(result.counts[qi, gi, mi, di, pi, ell_index])
                            sampled = int(result.sampled_pairs[ell_index])
                            fraction = accepted / sampled if sampled else 0.0
                            candidates = int(result.cube_candidate_pairs[ell_index])
                            eligible = int(result.eligible_pairs[ell_index])
                            rows.append(
                                {
                                    "q": q_name,
                                    "density_convention": result.density_conventions[qi],
                                    "geometry": geometry_name,
                                    "measurement": measurement_name,
                                    "direction": direction_name,
                                    "p": p_value,
                                    "ell_bin_index": ell_index,
                                    "ell_center": float(ell_center),
                                    "sampled_geometry_valid_pairs": sampled,
                                    "accepted": accepted,
                                    "accepted_per_sampled_pair": fraction,
                                    "full_origin_population": candidates,
                                    "valid_origin_population": eligible,
                                    "outside_or_nested_core_excluded_population": int(
                                        result.excluded_boundary_pairs[ell_index]
                                    ),
                                    "valid_origin_fraction": eligible / candidates if candidates else 0.0,
                                    "fit_eligible": accepted >= min_count and fraction >= min_sampled_fraction,
                                    "excluded_pair_counts": {
                                        name: int(result.exclusions[qi, gi, ei, ell_index])
                                        for ei, name in enumerate(result.exclusion_names)
                                    },
                                }
                            )
    return rows


def offset_support_rows(result: FiniteDomainResult) -> list[dict[str, object]]:
    """Report finite-domain support separately for every signed displacement."""

    rows = []
    for index, displacement in enumerate(result.displacements_ijk):
        vector = displacement.astype(float) * np.asarray(result.cell_sizes)
        candidates = int(result.cube_candidate_pairs_per_displacement[index])
        eligible = int(result.eligible_pairs_per_displacement[index])
        rows.append(
            {
                "displacement_ijk": tuple(int(value) for value in displacement),
                "r_vector": tuple(float(value) for value in vector),
                "ell": float(np.linalg.norm(vector)),
                "ell_bin_index": int(result.ell_bin_index_per_displacement[index]),
                "sampled_pairs": int(result.sampled_pairs_per_displacement[index]),
                "eligible_pairs": eligible,
                "cube_candidate_pairs": candidates,
                "excluded_boundary_pairs": int(result.excluded_boundary_pairs_per_displacement[index]),
                "eligible_origin_fraction": eligible / candidates if candidates else 0.0,
            }
        )
    return rows


def _fit_one(
    ell: np.ndarray,
    moments: np.ndarray,
    counts: np.ndarray,
    sampled_pairs: np.ndarray,
    fit_interval: tuple[float, float],
    min_count: int,
    min_sampled_fraction: float,
    min_bins: int,
) -> dict[str, float | int | str]:
    fraction = np.divide(
        counts,
        sampled_pairs,
        out=np.zeros_like(counts, dtype=float),
        where=sampled_pairs > 0,
    )
    mask = np.isfinite(moments) & (moments > 0.0) & (counts >= min_count)
    mask &= fraction >= min_sampled_fraction
    mask &= (ell >= fit_interval[0]) & (ell <= fit_interval[1])
    n_bins = int(np.count_nonzero(mask))
    if n_bins < min_bins:
        return {"slope": np.nan, "intercept": np.nan, "n_bins": n_bins, "quality": "insufficient_coverage"}
    slope, intercept = np.polyfit(np.log(ell[mask]), np.log(moments[mask]), 1)
    return {"slope": float(slope), "intercept": float(intercept), "n_bins": n_bins, "quality": "ok"}


def fit_directional_slopes(
    result: FiniteDomainResult,
    fit_interval: tuple[float, float],
    *,
    min_count: int = 1,
    min_sampled_fraction: float = 0.0,
    min_bins: int = 2,
) -> dict[str, object]:
    """Fit log-log slopes for every q, frame, increment kind, direction, and order."""

    if fit_interval[0] <= 0.0 or fit_interval[1] <= fit_interval[0]:
        raise ValueError("fit_interval must be a positive increasing pair")
    if min_count < 1 or min_bins < 2:
        raise ValueError("min_count must be positive and min_bins must be at least 2")
    if not 0.0 <= min_sampled_fraction <= 1.0:
        raise ValueError("min_sampled_fraction must be in [0, 1]")
    ell = geometric_bin_centers(result.ell_bin_edges)
    output: dict[str, object] = {}
    for q_name in result.q_names:
        output[q_name] = {}
        for geometry_name in result.geometry_names:
            output[q_name][geometry_name] = {}
            for measurement_name in result.measurement_names:
                output[q_name][geometry_name][measurement_name] = {}
                for p_value in result.p_values:
                    by_direction = {}
                    for direction_name in result.direction_names:
                        indices = _indices(
                            result, q_name, geometry_name, measurement_name, direction_name, p_value
                        )
                        by_direction[direction_name] = _fit_one(
                            ell,
                            result.moments[indices],
                            result.counts[indices],
                            result.sampled_pairs,
                            fit_interval,
                            min_count,
                            min_sampled_fraction,
                            min_bins,
                        )
                    output[q_name][geometry_name][measurement_name][str(p_value)] = by_direction
    return output


def _crossing_scale(
    ell: np.ndarray,
    moment: np.ndarray,
    error: np.ndarray,
    counts: np.ndarray,
    target: float,
    *,
    min_count: int,
    ell_interval: tuple[float, float] | None,
) -> tuple[float, float, str]:
    valid = np.isfinite(ell) & np.isfinite(moment) & (ell > 0.0) & (moment > 0.0)
    valid &= counts >= min_count
    if ell_interval is not None:
        valid &= (ell >= ell_interval[0]) & (ell <= ell_interval[1])
    valid_indices = np.flatnonzero(valid)
    exact_indices = [
        index
        for index in valid_indices
        if np.isclose(target, moment[index], rtol=1.0e-14, atol=0.0)
    ]
    if len(exact_indices) > 1:
        return np.nan, np.nan, "multiple_crossings"
    if exact_indices:
        index = exact_indices[0]
        relative_error = error[index] / moment[index]
        sigma = float(ell[index] * relative_error) if np.isfinite(relative_error) else np.nan
        return float(ell[index]), sigma, "ok"
    crossings: list[tuple[float, float]] = []
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        if right != left + 1:
            continue
        y0, y1 = moment[left], moment[right]
        if (target - y0) * (target - y1) >= 0.0 or y0 == y1:
            continue
        log_slope = (np.log(y1) - np.log(y0)) / (np.log(ell[right]) - np.log(ell[left]))
        fraction = (np.log(target) - np.log(y0)) / (np.log(y1) - np.log(y0))
        scale = float(np.exp(np.log(ell[left]) + fraction * (np.log(ell[right]) - np.log(ell[left]))))
        relative_errors = np.asarray([error[left] / y0, error[right] / y1])
        finite_errors = relative_errors[np.isfinite(relative_errors)]
        sigma = float(scale * np.max(finite_errors) / abs(log_slope)) if finite_errors.size and log_slope else np.nan
        crossings.append((scale, sigma))
    if not crossings:
        return np.nan, np.nan, "no_crossing"
    if len(crossings) > 1:
        return np.nan, np.nan, "multiple_crossings"
    return crossings[0][0], crossings[0][1], "ok"


def constant_sp_shapes(
    result: FiniteDomainResult,
    q_name: str,
    p_value: float,
    targets: Iterable[float],
    *,
    geometry_name: str = "pair_local",
    measurement_name: str = "perpendicular",
    min_count: int = 1,
    ell_interval: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Infer eddy dimensions by matching directional curves at fixed ``S_p``.

    Multiple crossings are rejected rather than silently selecting one branch.
    Returned uncertainties propagate sampling standard errors only; spatially
    correlated physical uncertainties require a later block analysis.
    """

    targets_array = np.asarray(tuple(targets), dtype=float)
    if targets_array.ndim != 1 or np.any(~np.isfinite(targets_array)) or np.any(targets_array <= 0.0):
        raise ValueError("targets must contain finite positive values")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    if ell_interval is not None and (ell_interval[0] <= 0.0 or ell_interval[1] <= ell_interval[0]):
        raise ValueError("ell_interval must be a positive increasing pair")
    ell = geometric_bin_centers(result.ell_bin_edges)
    output: dict[str, object] = {"S_p": targets_array, "p": float(p_value)}
    for direction_name in ("parallel", "xi", "lambda"):
        indices = _indices(result, q_name, geometry_name, measurement_name, direction_name, p_value)
        values, errors, reasons = [], [], []
        for target in targets_array:
            value, error, reason = _crossing_scale(
                ell,
                result.moments[indices],
                result.standard_error[indices],
                result.counts[indices],
                float(target),
                min_count=min_count,
                ell_interval=ell_interval,
            )
            values.append(value)
            errors.append(error)
            reasons.append(reason)
        output[direction_name] = np.asarray(values)
        output[f"{direction_name}_sampling_error"] = np.asarray(errors)
        output[f"{direction_name}_quality"] = np.asarray(reasons)
    with np.errstate(divide="ignore", invalid="ignore"):
        output["xi_over_lambda"] = output["xi"] / output["lambda"]
        output["parallel_over_lambda"] = output["parallel"] / output["lambda"]
    return output


def summary_rows(
    result: FiniteDomainResult,
    fit_interval: tuple[float, float],
    *,
    min_count: int = 1,
    min_sampled_fraction: float = 0.0,
    min_bins: int = 2,
) -> list[dict[str, object]]:
    """Build compact report rows with slopes, counts, exclusions, and cost."""

    slopes = fit_directional_slopes(
        result,
        fit_interval,
        min_count=min_count,
        min_sampled_fraction=min_sampled_fraction,
        min_bins=min_bins,
    )
    rows: list[dict[str, object]] = []
    for qi, q_name in enumerate(result.q_names):
        for gi, geometry_name in enumerate(result.geometry_names):
            for mi, measurement_name in enumerate(result.measurement_names):
                for pi, p_value in enumerate(result.p_values):
                    by_direction = slopes[q_name][geometry_name][measurement_name][str(p_value)]
                    rows.append(
                        {
                            "q": q_name,
                            "density_convention": result.density_conventions[qi],
                            "geometry": geometry_name,
                            "measurement": measurement_name,
                            "p": p_value,
                            "fit_interval": tuple(fit_interval),
                            "slopes": {name: values["slope"] for name, values in by_direction.items()},
                            "fit_quality": {name: values["quality"] for name, values in by_direction.items()},
                            "fit_bin_counts": {name: values["n_bins"] for name, values in by_direction.items()},
                            "sample_counts": {
                                name: int(result.counts[qi, gi, mi, di, pi].sum())
                                for di, name in enumerate(result.direction_names)
                            },
                            "excluded_pair_counts": {
                                name: int(result.exclusions[qi, gi, ei].sum())
                                for ei, name in enumerate(result.exclusion_names)
                            },
                            "wall_clock_seconds": result.elapsed_seconds,
                        }
                    )
    return rows


def fit_interval_sensitivity_rows(
    result: FiniteDomainResult,
    fit_intervals: Iterable[tuple[float, float]],
    *,
    min_count: int = 1,
    min_sampled_fraction: float = 0.0,
    min_bins: int = 2,
) -> list[dict[str, object]]:
    """Return slope rows for an explicit set of alternate fit intervals."""

    rows = []
    for interval in fit_intervals:
        interval = tuple(float(value) for value in interval)
        for row in summary_rows(
            result,
            interval,
            min_count=min_count,
            min_sampled_fraction=min_sampled_fraction,
            min_bins=min_bins,
        ):
            rows.append(row)
    return rows


def result_to_npz_payload(result: FiniteDomainResult) -> dict[str, object]:
    """Return an NPZ-friendly payload retaining finite-domain conventions."""

    metadata = {
        "pair_mode": result.pair_mode,
        "stencil_width": result.stencil_width,
        "cube_shape_kji": result.cube_shape_kji,
        "nested_core_bounds_kji": result.nested_core_bounds_kji,
        "shell_core_bounds_kji": result.shell_core_bounds_kji,
        "rho0": result.rho0,
        "rho0_provenance": result.rho0_provenance,
        "cell_sizes": result.cell_sizes,
        "angle_limits": result.angle_limits,
        "sample_count": result.sample_count,
        "pair_batch_size": result.pair_batch_size,
        "seed": result.seed,
        "elapsed_seconds": result.elapsed_seconds,
        "out_of_range_displacements": result.out_of_range_displacements,
        "block_shape_kji": result.block_shape_kji,
        "block_assignment": result.block_assignment,
        "support_displacements_sha256": result.support_displacements_sha256,
        "support_displacement_count": result.support_displacement_count,
        "uncertainty_note": "standard_error is pair-sampling noise only; use block accumulators for spatial uncertainty",
    }
    return {
        "q_names": np.asarray(result.q_names),
        "density_conventions": np.asarray(result.density_conventions),
        "geometry_names": np.asarray(result.geometry_names),
        "measurement_names": np.asarray(result.measurement_names),
        "direction_names": np.asarray(result.direction_names),
        "exclusion_names": np.asarray(result.exclusion_names),
        "ell_bin_edges": result.ell_bin_edges,
        "p_values": np.asarray(result.p_values),
        "counts": result.counts,
        "sums": result.sums,
        "sums_sq": result.sums_sq,
        "moments": result.moments,
        "standard_error": result.standard_error,
        "exclusions": result.exclusions,
        "sampled_pairs": result.sampled_pairs,
        "eligible_pairs": result.eligible_pairs,
        "cube_candidate_pairs": result.cube_candidate_pairs,
        "excluded_boundary_pairs": result.excluded_boundary_pairs,
        "displacements_per_bin": result.displacements_per_bin,
        "displacements_ijk": result.displacements_ijk,
        "ell_bin_index_per_displacement": result.ell_bin_index_per_displacement,
        "sampled_pairs_per_displacement": result.sampled_pairs_per_displacement,
        "eligible_pairs_per_displacement": result.eligible_pairs_per_displacement,
        "cube_candidate_pairs_per_displacement": result.cube_candidate_pairs_per_displacement,
        "excluded_boundary_pairs_per_displacement": result.excluded_boundary_pairs_per_displacement,
        "intrinsic_eligible_origins": result.intrinsic_eligible_origins,
        "boundary_excluded_origins": result.boundary_excluded_origins,
        "support_policy_excluded_origins": result.support_policy_excluded_origins,
        "intrinsic_eligible_origins_per_displacement": result.intrinsic_eligible_origins_per_displacement,
        "boundary_excluded_origins_per_displacement": result.boundary_excluded_origins_per_displacement,
        "support_policy_excluded_origins_per_displacement": result.support_policy_excluded_origins_per_displacement,
        **(
            {
                "block_counts": result.block_counts,
                "block_sums": result.block_sums,
                "block_sums_sq": result.block_sums_sq,
                "block_sampled_origins": result.block_sampled_origins,
                "block_eligible_origins": result.block_eligible_origins,
                "block_exclusions": result.block_exclusions,
            }
            if result.block_counts is not None
            else {}
        ),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
