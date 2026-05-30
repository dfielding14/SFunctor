"""Post-processing helpers for strict pairwise directional structure functions."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from sfunctor.core.directional import DirectionalResult

__all__ = [
    "constant_s2_shapes",
    "coverage_rows",
    "fit_directional_slopes",
    "geometric_bin_centers",
    "result_to_npz_payload",
    "summary_rows",
]


def geometric_bin_centers(edges: np.ndarray) -> np.ndarray:
    """Return geometric centers for positive logarithmic separation bins."""

    edges = np.asarray(edges, dtype=float)
    if np.any(edges <= 0.0):
        return 0.5 * (edges[:-1] + edges[1:])
    return np.sqrt(edges[:-1] * edges[1:])


def _indices(result: DirectionalResult, q_name: str, geometry_name: str, direction_name: str) -> tuple[int, int, int]:
    return (
        result.q_names.index(q_name),
        result.geometry_names.index(geometry_name),
        result.direction_names.index(direction_name),
    )


def _fit_one(
    ell: np.ndarray,
    s2: np.ndarray,
    counts: np.ndarray,
    attempts: np.ndarray,
    fit_interval: tuple[float, float],
    min_count: int,
    min_accepted_fraction: float,
    min_bins: int,
) -> dict[str, float | int]:
    accepted_fraction = np.divide(
        counts,
        attempts,
        out=np.zeros_like(counts, dtype=float),
        where=attempts > 0,
    )
    mask = np.isfinite(s2) & (s2 > 0.0) & (counts >= min_count)
    mask &= accepted_fraction >= min_accepted_fraction
    mask &= (ell >= fit_interval[0]) & (ell <= fit_interval[1])
    n_bins = int(np.count_nonzero(mask))
    if n_bins < min_bins:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "n_bins": n_bins,
            "quality": "insufficient_coverage",
        }
    slope, intercept = np.polyfit(np.log(ell[mask]), np.log(s2[mask]), 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_bins": n_bins,
        "quality": "ok",
    }


def fit_directional_slopes(
    result: DirectionalResult,
    fit_interval: tuple[float, float],
    *,
    min_count: int = 1,
    min_accepted_fraction: float = 0.0,
    min_bins: int = 2,
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    """Fit log-log ``S2`` slopes for every field, frame, and direction."""

    if fit_interval[0] <= 0.0 or fit_interval[1] <= fit_interval[0]:
        raise ValueError("fit_interval must be a positive increasing pair")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    if not 0.0 <= min_accepted_fraction <= 1.0:
        raise ValueError("min_accepted_fraction must be in [0, 1]")
    if min_bins < 2:
        raise ValueError("min_bins must be at least 2")
    ell = geometric_bin_centers(result.ell_bin_edges)
    output: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for q_name in result.q_names:
        output[q_name] = {}
        for geometry_name in result.geometry_names:
            output[q_name][geometry_name] = {}
            for direction_name in result.direction_names:
                qi, gi, di = _indices(result, q_name, geometry_name, direction_name)
                output[q_name][geometry_name][direction_name] = _fit_one(
                    ell,
                    result.s2[qi, gi, di],
                    result.counts[qi, gi, di],
                    result.attempts,
                    fit_interval,
                    min_count,
                    min_accepted_fraction,
                    min_bins,
                )
    return output


def coverage_rows(
    result: DirectionalResult,
    *,
    min_count: int = 100,
    min_accepted_fraction: float = 0.0,
) -> list[dict[str, object]]:
    """Report slice-conditioned occupancy and quality gates for every bin."""

    if min_count < 1:
        raise ValueError("min_count must be positive")
    if not 0.0 <= min_accepted_fraction <= 1.0:
        raise ValueError("min_accepted_fraction must be in [0, 1]")
    ell = geometric_bin_centers(result.ell_bin_edges)
    rows = []
    for qi, q_name in enumerate(result.q_names):
        for gi, geometry_name in enumerate(result.geometry_names):
            for di, direction_name in enumerate(result.direction_names):
                for ell_index, ell_center in enumerate(ell):
                    accepted = int(result.counts[qi, gi, di, ell_index])
                    attempts = int(result.attempts[ell_index])
                    fraction = accepted / attempts if attempts else 0.0
                    rows.append(
                        {
                            "q": q_name,
                            "density_convention": result.density_conventions[qi],
                            "geometry": geometry_name,
                            "direction": direction_name,
                            "ell_bin_index": ell_index,
                            "ell_center": float(ell_center),
                            "attempts": attempts,
                            "accepted": accepted,
                            "accepted_fraction": fraction,
                            "fit_eligible": accepted >= min_count and fraction >= min_accepted_fraction,
                            "excluded_pair_counts": {
                                name: int(result.exclusions[qi, gi, ei, ell_index])
                                for ei, name in enumerate(result.exclusion_names)
                            },
                        }
                    )
    return rows


def _crossing_scale(ell: np.ndarray, s2: np.ndarray, target: float) -> float:
    valid = np.isfinite(ell) & np.isfinite(s2) & (ell > 0.0) & (s2 > 0.0)
    ell, s2 = ell[valid], s2[valid]
    for index in range(len(ell) - 1):
        y0, y1 = s2[index], s2[index + 1]
        if np.isclose(target, y0, rtol=1.0e-14, atol=0.0):
            return float(ell[index])
        if np.isclose(target, y1, rtol=1.0e-14, atol=0.0):
            return float(ell[index + 1])
        if (target - y0) * (target - y1) > 0.0:
            continue
        if y0 == y1:
            return float(ell[index])
        fraction = (np.log(target) - np.log(y0)) / (np.log(y1) - np.log(y0))
        return float(np.exp(np.log(ell[index]) + fraction * (np.log(ell[index + 1]) - np.log(ell[index]))))
    return np.nan


def constant_s2_shapes(
    result: DirectionalResult,
    q_name: str,
    targets: Iterable[float],
    *,
    geometry_name: str = "local",
) -> dict[str, np.ndarray]:
    """Return constant-``S2`` eddy scales and aspect ratios.

    Each scale uses the first adjacent log-space crossing in the measured
    curve.  Callers should inspect the source curves and restrict targets to a
    monotonic inertial-range branch when interpreting noisy simulation data.
    """

    ell = geometric_bin_centers(result.ell_bin_edges)
    targets = np.asarray(tuple(targets), dtype=float)
    if targets.ndim != 1 or np.any(~np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("targets must contain finite positive S2 levels")
    output = {"S2": targets}
    for direction_name in ("parallel", "xi", "lambda"):
        qi, gi, di = _indices(result, q_name, geometry_name, direction_name)
        curve = result.s2[qi, gi, di]
        output[direction_name] = np.array([_crossing_scale(ell, curve, target) for target in targets])
    with np.errstate(divide="ignore", invalid="ignore"):
        output["xi_over_lambda"] = output["xi"] / output["lambda"]
        output["parallel_over_lambda"] = output["parallel"] / output["lambda"]
    return output


def summary_rows(
    result: DirectionalResult,
    fit_interval: tuple[float, float],
    *,
    min_count: int = 1,
    min_accepted_fraction: float = 0.0,
    min_bins: int = 2,
) -> list[dict[str, object]]:
    """Build compact report rows with slopes, counts, exclusions, and cost."""

    slopes = fit_directional_slopes(
        result,
        fit_interval,
        min_count=min_count,
        min_accepted_fraction=min_accepted_fraction,
        min_bins=min_bins,
    )
    rows = []
    for qi, q_name in enumerate(result.q_names):
        for gi, geometry_name in enumerate(result.geometry_names):
            rows.append(
                {
                    "q": q_name,
                    "density_convention": result.density_conventions[qi],
                    "geometry": geometry_name,
                    "theta_parallel_max_deg": np.rad2deg(result.angle_limits["theta_parallel_max"]),
                    "theta_perpendicular_min_deg": np.rad2deg(result.angle_limits["theta_perpendicular_min"]),
                    "phi_xi_max_deg": np.rad2deg(result.angle_limits["phi_xi_max"]),
                    "phi_lambda_min_deg": np.rad2deg(result.angle_limits["phi_lambda_min"]),
                    "fit_interval": tuple(fit_interval),
                    "slopes": {name: values["slope"] for name, values in slopes[q_name][geometry_name].items()},
                    "fit_quality": {
                        name: values["quality"]
                        for name, values in slopes[q_name][geometry_name].items()
                    },
                    "fit_bin_counts": {
                        name: values["n_bins"]
                        for name, values in slopes[q_name][geometry_name].items()
                    },
                    "sample_counts": {
                        name: int(result.counts[qi, gi, di].sum())
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


def result_to_npz_payload(result: DirectionalResult) -> dict[str, object]:
    """Return an NPZ-friendly payload retaining conventions and exclusions."""

    metadata = {
        "rho0": result.rho0,
        "cell_sizes": result.cell_sizes,
        "angle_limits": result.angle_limits,
        "sample_count": result.sample_count,
        "seed": result.seed,
        "elapsed_seconds": result.elapsed_seconds,
        "out_of_range_displacements": result.out_of_range_displacements,
        "out_of_range_attempts": result.out_of_range_attempts,
    }
    return {
        "q_names": np.asarray(result.q_names),
        "density_conventions": np.asarray(result.density_conventions),
        "geometry_names": np.asarray(result.geometry_names),
        "direction_names": np.asarray(result.direction_names),
        "exclusion_names": np.asarray(result.exclusion_names),
        "ell_bin_edges": result.ell_bin_edges,
        "counts": result.counts,
        "sums": result.sums,
        "sums_sq": result.sums_sq,
        "s2": result.s2,
        "standard_error": result.standard_error,
        "exclusions": result.exclusions,
        "attempts": result.attempts,
        "metadata": metadata,
    }
