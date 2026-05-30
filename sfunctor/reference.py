"""Readable reference implementations for validation on small problems.

These routines favor explicit loops over speed.  Production code must agree
with them on small synthetic slices before optimization results are trusted.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from sfunctor.core.directional import (
    DIRECTION_NAMES,
    EXCLUSION_NAMES,
    GEOMETRY_NAMES,
    DirectionalConfig,
    DirectionalResult,
    QField,
    build_q_variants,
    slice_offset_to_vector,
)
from sfunctor.core.histograms import Channel

__all__ = [
    "compute_directional_structure_functions_reference",
    "compute_standard_s2_reference",
    "compute_unified_channel_values_reference",
]


def _ell_bin_index(value: float, edges: np.ndarray) -> int:
    """Return the left-inclusive bin index, retaining the final right edge."""

    if value == edges[-1]:
        return len(edges) - 2
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return -1


def _folded_angle(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return np.nan
    cosine = abs(float(np.dot(left, right))) / denominator
    return float(np.arccos(np.clip(cosine, 0.0, 1.0)))


def _origin_indices(shape: tuple[int, int], sample_count: int | None, seed: int) -> np.ndarray:
    n_cells = shape[0] * shape[1]
    if sample_count is None:
        return np.arange(n_cells, dtype=np.int64)
    if sample_count > n_cells:
        raise ValueError(f"sample_count={sample_count} exceeds the number of slice cells ({n_cells})")
    return np.random.default_rng(seed).choice(n_cells, size=sample_count, replace=False)


def _empty_result(
    q_fields: Mapping[str, QField],
    geometry_names: tuple[str, ...],
    config: DirectionalConfig,
    rho0: float,
) -> DirectionalResult:
    shape = (len(q_fields), len(geometry_names), len(DIRECTION_NAMES), len(config.ell_bin_edges) - 1)
    exclusions_shape = (len(q_fields), len(geometry_names), len(EXCLUSION_NAMES), shape[-1])
    return DirectionalResult(
        q_names=tuple(q_fields),
        density_conventions=tuple(field.density_convention for field in q_fields.values()),
        geometry_names=geometry_names,
        direction_names=DIRECTION_NAMES,
        exclusion_names=EXCLUSION_NAMES,
        ell_bin_edges=config.ell_bin_edges.copy(),
        counts=np.zeros(shape, dtype=np.int64),
        sums=np.zeros(shape, dtype=float),
        sums_sq=np.zeros(shape, dtype=float),
        exclusions=np.zeros(exclusions_shape, dtype=np.int64),
        attempts=np.zeros(shape[-1], dtype=np.int64),
        out_of_range_displacements=0,
        out_of_range_attempts=0,
        rho0=rho0,
        cell_sizes=tuple(config.cell_sizes),
        angle_limits={
            "theta_parallel_max": config.theta_parallel_max,
            "theta_perpendicular_min": config.theta_perpendicular_min,
            "phi_xi_max": config.phi_xi_max,
            "phi_lambda_min": config.phi_lambda_min,
        },
        sample_count=config.sample_count,
        seed=config.seed,
        elapsed_seconds=0.0,
    )


def _record(result: DirectionalResult, q: int, geometry: int, direction: str, ell: int, value: float) -> None:
    index = (q, geometry, result.direction_names.index(direction), ell)
    result.counts[index] += 1
    result.sums[index] += value
    result.sums_sq[index] += value * value


def compute_standard_s2_reference(
    field: np.ndarray,
    displacements: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Return full-domain periodic ``<|delta q|^2>`` for each 2-D offset."""

    field = np.asarray(field, dtype=float)
    if field.ndim != 3 or field.shape[0] != 3:
        raise ValueError("field must have shape (3, ny, nx)")
    _, ny, nx = field.shape
    output: dict[tuple[int, int], float] = {}
    for raw_delta_i, raw_delta_j in np.asarray(displacements):
        delta_i, delta_j = int(raw_delta_i), int(raw_delta_j)
        values = []
        for y0 in range(ny):
            for x0 in range(nx):
                y1 = (y0 + delta_j) % ny
                x1 = (x0 + delta_i) % nx
                delta = field[:, y1, x1] - field[:, y0, x0]
                values.append(float(np.dot(delta, delta)))
        output[(delta_i, delta_j)] = float(np.mean(values))
    return output


def compute_unified_channel_values_reference(
    increments: Mapping[str, np.ndarray | float],
    B_loc: np.ndarray,
) -> dict[str, float]:
    """Evaluate the 26 legacy channel values for one already-formed increment.

    This readable oracle validates channel definitions independently of Numba,
    bin search, periodic lookup, and stencil construction.
    """

    B_loc = np.asarray(B_loc, dtype=float)
    B_mag = float(np.linalg.norm(B_loc))
    if B_loc.shape != (3,) or not np.isfinite(B_mag) or B_mag <= 0.0:
        raise ValueError("B_loc must be a finite nonzero Cartesian 3-vector")
    e_parallel = B_loc / B_mag

    def vector(name: str) -> np.ndarray:
        value = np.asarray(increments[name], dtype=float)
        if value.shape != (3,):
            raise ValueError(f"increments[{name!r}] must have shape (3,)")
        return value

    def magnitude(name: str) -> float:
        return float(np.linalg.norm(vector(name)))

    def perpendicular(name: str) -> np.ndarray:
        value = vector(name)
        return value - np.dot(value, e_parallel) * e_parallel

    def alignment(left: str, right: str) -> tuple[float, float, float | None]:
        left_perp, right_perp = perpendicular(left), perpendicular(right)
        cross = float(np.linalg.norm(np.cross(left_perp, right_perp)))
        product = float(np.linalg.norm(left_perp) * np.linalg.norm(right_perp))
        return cross, product, min(1.0, cross / product) if product > 0.0 else None

    output = {
        Channel.D_V.name: magnitude("v"),
        Channel.D_B.name: magnitude("B"),
        Channel.D_RHO.name: abs(float(increments["rho"])),
        Channel.D_VA.name: magnitude("vA"),
        Channel.D_ZPLUS.name: magnitude("zp"),
        Channel.D_ZMINUS.name: magnitude("zm"),
        Channel.D_OMEGA.name: magnitude("omega"),
        Channel.D_J.name: magnitude("j"),
        Channel.D_CURV.name: magnitude("curv"),
        Channel.D_GRAD_RHO.name: magnitude("grad_rho"),
        Channel.D_B_over_Bmean_loc.name: magnitude("B") / B_mag,
    }
    pairs = (
        ("v", "B", Channel.D_Vperp_CROSS_D_Bperp, Channel.D_Vperp_D_Bperp_MAG, Channel.D_Vperp_D_Bperp_CROSS_MAG_RATIO),
        ("v", "omega", Channel.D_Vperp_CROSS_D_Omegaperp, Channel.D_Vperp_D_Omegaperp_MAG, Channel.D_Vperp_D_Omegaperp_CROSS_MAG_RATIO),
        ("B", "j", Channel.D_Bperp_CROSS_D_Jperp, Channel.D_Bperp_D_Jperp_MAG, Channel.D_Bperp_D_Jperp_CROSS_MAG_RATIO),
        ("omega", "j", Channel.D_Omegaperp_CROSS_D_Jperp, Channel.D_Omegaperp_D_Jperp_MAG, Channel.D_Omegaperp_D_Jperp_CROSS_MAG_RATIO),
        ("zp", "zm", Channel.D_Zplusperp_CROSS_D_Zminusperp, Channel.D_Zplusperp_D_Zminusperp_MAG, Channel.D_Zplusperp_D_Zminusperp_CROSS_MAG_RATIO),
    )
    for left, right, cross_channel, product_channel, ratio_channel in pairs:
        cross, product, ratio = alignment(left, right)
        output[cross_channel.name] = cross
        output[product_channel.name] = product
        if ratio is not None:
            output[ratio_channel.name] = ratio
    return output


def compute_directional_structure_functions_reference(
    slice_data: Mapping[str, np.ndarray],
    displacements: np.ndarray,
    *,
    slice_axis: int,
    config: DirectionalConfig,
    rho0: float | None = None,
    rho_floor: float = 0.0,
    q_names: Sequence[str] | None = None,
) -> DirectionalResult:
    """Compute directional ``S2`` with a deliberately explicit point-pair loop."""

    B, available, rho0 = build_q_variants(slice_data, rho0=rho0, rho_floor=rho_floor)
    if q_names is None:
        q_names = tuple(available)
    unknown = set(q_names) - available.keys()
    if unknown:
        raise ValueError(f"Unknown q variants: {sorted(unknown)}")
    q_fields = {name: available[name] for name in q_names}
    geometry_names = GEOMETRY_NAMES if config.include_global else GEOMETRY_NAMES[:1]
    result = _empty_result(q_fields, geometry_names, config, rho0)
    exclusion_index = {name: index for index, name in enumerate(EXCLUSION_NAMES)}

    _, ny, nx = B.shape
    origins = _origin_indices((ny, nx), config.sample_count, config.seed)
    finite_B_points = np.all(np.isfinite(B), axis=0)
    global_B = (
        np.mean(B[:, finite_B_points], axis=1)
        if np.any(finite_B_points)
        else np.full(3, np.nan)
    )

    for raw_delta_i, raw_delta_j in np.asarray(displacements):
        delta_i, delta_j = int(raw_delta_i), int(raw_delta_j)
        r = slice_offset_to_vector(slice_axis, (delta_i, delta_j), config.cell_sizes)
        ell_index = _ell_bin_index(float(np.linalg.norm(r)), config.ell_bin_edges)
        if ell_index < 0:
            result.out_of_range_displacements += 1
            result.out_of_range_attempts += origins.size
            continue
        result.attempts[ell_index] += origins.size

        for origin in origins:
            y0, x0 = int(origin // nx), int(origin % nx)
            y1 = (y0 + delta_j) % ny
            x1 = (x0 + delta_i) % nx
            B_left = B[:, y0, x0]
            B_right = B[:, y1, x1]
            B_loc = 0.5 * (B_left + B_right)

            for geometry_index, geometry_name in enumerate(geometry_names):
                B_direction = B_loc if geometry_name == "local" else global_B
                valid_B = bool(np.all(np.isfinite(B_direction)))
                if geometry_name == "local":
                    valid_B = valid_B and bool(np.all(np.isfinite(B_left))) and bool(np.all(np.isfinite(B_right)))
                B_mag = float(np.linalg.norm(B_direction))
                valid_parallel = valid_B and B_mag > config.B_epsilon
                e_parallel = B_direction / B_mag if valid_parallel else np.full(3, np.nan)
                theta = _folded_angle(r, e_parallel)
                r_perp = r - np.dot(r, e_parallel) * e_parallel
                r_perp_mag = float(np.linalg.norm(r_perp))

                for q_index, q_field in enumerate(q_fields.values()):
                    if not valid_B:
                        result.exclusions[q_index, geometry_index, exclusion_index["invalid_B"], ell_index] += 1
                    elif not valid_parallel:
                        result.exclusions[q_index, geometry_index, exclusion_index["weak_B_direction"], ell_index] += 1

                    q_left = q_field.values[:, y0, x0]
                    q_right = q_field.values[:, y1, x1]
                    valid_q = bool(q_field.valid[y0, x0] and q_field.valid[y1, x1])
                    valid_q = valid_q and bool(np.all(np.isfinite(q_left))) and bool(np.all(np.isfinite(q_right)))
                    if valid_parallel and not valid_q:
                        result.exclusions[q_index, geometry_index, exclusion_index["invalid_q"], ell_index] += 1
                    if not valid_parallel or not valid_q:
                        continue

                    delta_q = q_right - q_left
                    delta_q_perp = delta_q - np.dot(delta_q, e_parallel) * e_parallel
                    q_perp_mag = float(np.linalg.norm(delta_q_perp))
                    s2 = q_perp_mag * q_perp_mag
                    _record(result, q_index, geometry_index, "all", ell_index, s2)
                    if theta <= config.theta_parallel_max:
                        _record(result, q_index, geometry_index, "parallel", ell_index, s2)
                    if theta >= config.theta_perpendicular_min:
                        _record(result, q_index, geometry_index, "perpendicular", ell_index, s2)

                    valid_q_perp = np.isfinite(q_perp_mag) and q_perp_mag > config.q_perp_epsilon
                    if not valid_q_perp:
                        result.exclusions[q_index, geometry_index, exclusion_index["weak_q_perp_for_phi"], ell_index] += 1
                        continue
                    valid_r_perp = np.isfinite(r_perp_mag) and r_perp_mag > config.r_perp_epsilon
                    if not valid_r_perp:
                        result.exclusions[q_index, geometry_index, exclusion_index["weak_r_perp_for_phi"], ell_index] += 1
                        continue

                    e_xi = delta_q_perp / q_perp_mag
                    phi = _folded_angle(r_perp, e_xi)
                    if theta >= config.theta_perpendicular_min and phi <= config.phi_xi_max:
                        _record(result, q_index, geometry_index, "xi", ell_index, s2)
                    if theta >= config.theta_perpendicular_min and phi >= config.phi_lambda_min:
                        _record(result, q_index, geometry_index, "lambda", ell_index, s2)
    return result
