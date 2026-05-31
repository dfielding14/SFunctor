"""Finite-domain three-direction structure functions for extracted 3-D cubes.

AthenaK cube arrays remain in native KJI order:

``array[k, j, i] == field[x3, x2, x1]``.

Displacements are always supplied in Cartesian index order ``(di, dj, dk)``.
This module never wraps endpoints across an extracted-cube boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from sfunctor.core.directional import DIRECTION_NAMES, EXCLUSION_NAMES, QField, folded_angle

__all__ = [
    "FINITE_DOMAIN_GEOMETRY_NAMES",
    "MEASUREMENT_NAMES",
    "PAIR_MODES",
    "FiniteDomainConfig",
    "FiniteDomainResult",
    "build_cube_q_variants",
    "compute_finite_domain_structure_functions",
    "cube_offset_to_vector",
    "generate_fibonacci_displacements",
    "nested_core_bounds_kji",
    "valid_origin_bounds_kji",
]


FINITE_DOMAIN_GEOMETRY_NAMES = ("pair_local", "subvolume_mean")
MEASUREMENT_NAMES = ("total", "perpendicular")
PAIR_MODES = ("nested_core", "all_valid_pairs")


@dataclass(frozen=True)
class FiniteDomainConfig:
    """Configuration for non-periodic 3-D conditional structure functions."""

    ell_bin_edges: np.ndarray
    p_values: tuple[float, ...] = (2.0,)
    cell_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pair_mode: str = "nested_core"
    sample_count: int | None = None
    pair_batch_size: int = 8192
    seed: int = 0
    theta_parallel_max: float = np.deg2rad(15.0)
    theta_perpendicular_min: float = np.deg2rad(75.0)
    phi_xi_max: float = np.deg2rad(15.0)
    phi_lambda_min: float = np.deg2rad(75.0)
    B_epsilon: float = 1.0e-12
    q_perp_epsilon: float = 1.0e-12
    r_perp_epsilon: float = 1.0e-12
    include_subvolume_mean: bool = True

    def __post_init__(self) -> None:
        edges = np.asarray(self.ell_bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("ell_bin_edges must be a 1-D array with at least two edges")
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
            raise ValueError("ell_bin_edges must contain finite, strictly increasing values")
        p_values = tuple(float(value) for value in self.p_values)
        if not p_values or any(not np.isfinite(value) or value <= 0.0 for value in p_values):
            raise ValueError("p_values must contain positive finite values")
        if len(set(p_values)) != len(p_values):
            raise ValueError("p_values must not contain duplicates")
        if len(self.cell_sizes) != 3 or not np.all(np.isfinite(self.cell_sizes)):
            raise ValueError("cell_sizes must contain three finite Cartesian spacings")
        if np.any(np.asarray(self.cell_sizes) <= 0.0):
            raise ValueError("cell_sizes must be positive")
        if self.pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}")
        if self.sample_count is not None and (
            isinstance(self.sample_count, (bool, np.bool_))
            or not isinstance(self.sample_count, (int, np.integer))
            or self.sample_count <= 0
        ):
            raise ValueError("sample_count must be positive or None")
        if (
            isinstance(self.pair_batch_size, (bool, np.bool_))
            or not isinstance(self.pair_batch_size, (int, np.integer))
            or self.pair_batch_size <= 0
        ):
            raise ValueError("pair_batch_size must be positive")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        for name in (
            "theta_parallel_max",
            "theta_perpendicular_min",
            "phi_xi_max",
            "phi_lambda_min",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= np.pi / 2.0:
                raise ValueError(f"{name} must be in [0, pi/2]")
        if self.theta_parallel_max > self.theta_perpendicular_min:
            raise ValueError("parallel and perpendicular theta windows must not overlap")
        if self.phi_xi_max > self.phi_lambda_min:
            raise ValueError("xi and lambda phi windows must not overlap")
        epsilons = np.asarray([self.B_epsilon, self.q_perp_epsilon, self.r_perp_epsilon], dtype=float)
        if np.any(~np.isfinite(epsilons)) or np.any(epsilons < 0.0):
            raise ValueError("epsilon values must be finite and non-negative")
        object.__setattr__(self, "ell_bin_edges", edges)
        object.__setattr__(self, "p_values", p_values)


@dataclass
class FiniteDomainResult:
    """Measured moments, pair accounting, exclusions, and conventions."""

    q_names: tuple[str, ...]
    density_conventions: tuple[str, ...]
    geometry_names: tuple[str, ...]
    measurement_names: tuple[str, ...]
    direction_names: tuple[str, ...]
    exclusion_names: tuple[str, ...]
    ell_bin_edges: np.ndarray
    p_values: tuple[float, ...]
    counts: np.ndarray
    sums: np.ndarray
    sums_sq: np.ndarray
    exclusions: np.ndarray
    sampled_pairs: np.ndarray
    eligible_pairs: np.ndarray
    cube_candidate_pairs: np.ndarray
    excluded_boundary_pairs: np.ndarray
    displacements_per_bin: np.ndarray
    displacements_ijk: np.ndarray
    ell_bin_index_per_displacement: np.ndarray
    sampled_pairs_per_displacement: np.ndarray
    eligible_pairs_per_displacement: np.ndarray
    cube_candidate_pairs_per_displacement: np.ndarray
    excluded_boundary_pairs_per_displacement: np.ndarray
    out_of_range_displacements: int
    pair_mode: str
    cube_shape_kji: tuple[int, int, int]
    nested_core_bounds_kji: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
    rho0: float
    rho0_provenance: str
    cell_sizes: tuple[float, float, float]
    angle_limits: dict[str, float]
    sample_count: int | None
    pair_batch_size: int
    seed: int
    elapsed_seconds: float

    @property
    def moments(self) -> np.ndarray:
        """Return ``<|delta q|^p>``-style moments, with empty bins as NaN."""

        out = np.full_like(self.sums, np.nan, dtype=float)
        return np.divide(self.sums, self.counts, out=out, where=self.counts > 0)

    @property
    def standard_error(self) -> np.ndarray:
        """Return sampling standard errors, not physical uncertainty bars."""

        out = np.full_like(self.sums, np.nan, dtype=float)
        mask = self.counts > 1
        mean_square_term = self.sums[mask] ** 2 / self.counts[mask]
        numerator = self.sums_sq[mask] - mean_square_term
        roundoff_floor = 64.0 * np.finfo(float).eps * (
            np.abs(self.sums_sq[mask]) + np.abs(mean_square_term)
        )
        numerator[np.abs(numerator) <= roundoff_floor] = 0.0
        numerator = np.maximum(numerator, 0.0)
        out[mask] = np.sqrt(numerator / (self.counts[mask] * (self.counts[mask] - 1)))
        return out


def _as_cube_vector(cube_data: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    """Return Cartesian components with shape ``(3, nk, nj, ni)``."""

    arrays = [np.asarray(cube_data[f"{prefix}_{component}"]) for component in "xyz"]
    if any(array.ndim != 3 for array in arrays):
        raise ValueError(f"{prefix} components must be 3-D KJI cube arrays")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError(f"{prefix} components must have matching shapes")
    return np.stack(arrays, axis=0)


def _finite_vector(field: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(field), axis=0)


def build_cube_q_variants(
    cube_data: Mapping[str, np.ndarray],
    *,
    q_names: Sequence[str] | None = None,
    rho0: float | None = None,
    rho_floor: float = 0.0,
) -> tuple[np.ndarray, dict[str, QField], float]:
    """Build requested 3-D MHD vector variants without unnecessary arrays.

    ``rho0`` defaults to the finite positive-density subvolume mean. Reference
    variants are explicitly diagnostics based on that fixed density.
    """

    requested = tuple(q_names or ("B", "u", "vA", "vA_ref", "z_plus", "z_minus", "z_plus_ref", "z_minus_ref"))
    known = {"B", "u", "vA", "vA_ref", "z_plus", "z_minus", "z_plus_ref", "z_minus_ref"}
    unknown = set(requested) - known
    if unknown:
        raise ValueError(f"Unknown q variants: {sorted(unknown)}")
    if len(set(requested)) != len(requested):
        raise ValueError("q_names must not contain duplicates")
    if not np.isfinite(rho_floor) or rho_floor < 0.0:
        raise ValueError("rho_floor must be finite and non-negative")

    B = _as_cube_vector(cube_data, "B")
    u = _as_cube_vector(cube_data, "v")
    if u.shape != B.shape:
        raise ValueError("B and v components must share one KJI cube shape")
    valid_B = _finite_vector(B)
    valid_u = _finite_vector(u)
    need_rho0 = bool(set(requested) & {"vA_ref", "z_plus_ref", "z_minus_ref"})
    need_rho = bool(set(requested) & {"vA", "z_plus", "z_minus"}) or (
        need_rho0 and rho0 is None
    )
    if need_rho:
        rho = np.asarray(cube_data["rho"])
        if rho.ndim != 3 or rho.shape != B.shape[1:]:
            raise ValueError("rho, B, and v must share one KJI cube shape")
        valid_rho = np.isfinite(rho) & (rho > rho_floor)
    else:
        rho = None
        valid_rho = None
    if rho0 is None and need_rho0:
        assert rho is not None and valid_rho is not None
        if not np.any(valid_rho):
            raise ValueError("rho0 cannot be inferred: no finite density exceeds rho_floor")
        rho0 = float(np.mean(rho[valid_rho], dtype=np.float64))
    if rho0 is not None and (not np.isfinite(rho0) or rho0 <= rho_floor):
        raise ValueError("rho0 must be finite and greater than rho_floor")
    rho0_value = float(rho0) if rho0 is not None else np.nan

    need_vA = bool(set(requested) & {"vA", "z_plus", "z_minus"})
    need_vA_ref = bool(set(requested) & {"vA_ref", "z_plus_ref", "z_minus_ref"})
    vA = None
    if need_vA:
        assert rho is not None and valid_rho is not None
        vA = np.full(B.shape, np.nan, dtype=np.result_type(B.dtype, np.float32))
        vA[:, valid_rho] = B[:, valid_rho] / np.sqrt(rho[valid_rho])
    vA_ref = B / np.sqrt(rho0_value) if need_vA_ref else None

    output: dict[str, QField] = {}
    for name in requested:
        if name == "B":
            output[name] = QField(name, B, valid_B, "not applicable")
        elif name == "u":
            output[name] = QField(name, u, valid_u, "not applicable")
        elif name == "vA":
            assert vA is not None and valid_rho is not None
            output[name] = QField(name, vA, valid_B & valid_rho, "pointwise rho")
        elif name == "vA_ref":
            assert vA_ref is not None
            output[name] = QField(name, vA_ref, valid_B, f"fixed rho0={rho0_value:.16g}")
        elif name in ("z_plus", "z_minus"):
            assert vA is not None and valid_rho is not None
            sign = 1.0 if name == "z_plus" else -1.0
            output[name] = QField(name, u + sign * vA, valid_u & valid_B & valid_rho, "pointwise rho")
        else:
            assert vA_ref is not None
            sign = 1.0 if name == "z_plus_ref" else -1.0
            output[name] = QField(name, u + sign * vA_ref, valid_u & valid_B, f"fixed rho0={rho0_value:.16g}")
    return B, output, rho0_value


def _require_integer_triplet(
    values: Sequence[int],
    name: str,
    *,
    positive: bool = False,
) -> tuple[int, int, int]:
    raw = np.asarray(values)
    if raw.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must contain finite integer values")
    if np.any(~np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite integer values")
    rounded = np.rint(raw)
    if not np.array_equal(raw, rounded):
        raise ValueError(f"{name} must contain exact integer values")
    output = tuple(int(value) for value in rounded)
    if positive and any(value <= 0 for value in output):
        raise ValueError(f"{name} must contain positive integer values")
    return output


def cube_offset_to_vector(
    displacement_ijk: Sequence[int],
    cell_sizes: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Convert integer ``(di, dj, dk)`` into Cartesian ``(x1, x2, x3)``."""

    if len(cell_sizes) != 3:
        raise ValueError("cell_sizes must contain three values")
    di, dj, dk = _require_integer_triplet(displacement_ijk, "displacement_ijk")
    dx1, dx2, dx3 = (float(value) for value in cell_sizes)
    if not np.all(np.isfinite((dx1, dx2, dx3))) or np.any(np.asarray((dx1, dx2, dx3)) <= 0.0):
        raise ValueError("cell_sizes must contain three positive finite values")
    return np.array([di * dx1, dj * dx2, dk * dx3], dtype=float)


def valid_origin_bounds_kji(
    shape_kji: Sequence[int],
    displacement_ijk: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return half-open KJI origin bounds whose displaced endpoints are valid."""

    nk, nj, ni = _require_integer_triplet(shape_kji, "shape_kji", positive=True)
    di, dj, dk = _require_integer_triplet(displacement_ijk, "displacement_ijk")
    bounds = (
        (max(0, -dk), min(nk, nk - dk)),
        (max(0, -dj), min(nj, nj - dj)),
        (max(0, -di), min(ni, ni - di)),
    )
    if any(stop < start for start, stop in bounds):
        return ((0, 0), (0, 0), (0, 0))
    return bounds


def nested_core_bounds_kji(
    shape_kji: Sequence[int],
    displacements_ijk: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return one KJI core whose origins are valid for every displacement."""

    displacements = _require_integer_displacements(displacements_ijk)
    nk, nj, ni = _require_integer_triplet(shape_kji, "shape_kji", positive=True)
    di, dj, dk = displacements.T
    bounds = (
        (max(0, int(-dk.min())), min(nk, int(nk - dk.max()))),
        (max(0, int(-dj.min())), min(nj, int(nj - dj.max()))),
        (max(0, int(-di.min())), min(ni, int(ni - di.max()))),
    )
    if any(stop <= start for start, stop in bounds):
        raise ValueError("displacements leave no non-empty nested core")
    return bounds


def _require_integer_displacements(displacements_ijk: np.ndarray) -> np.ndarray:
    """Return exact integer displacements or reject silent truncation."""

    raw = np.asarray(displacements_ijk)
    if raw.ndim != 2 or raw.shape[1] != 3 or not raw.size:
        raise ValueError("displacements_ijk must have shape (n, 3) with n > 0")
    if not np.issubdtype(raw.dtype, np.number) or np.any(~np.isfinite(raw)):
        raise ValueError("displacements_ijk must contain finite integer values")
    rounded = np.rint(raw)
    if not np.array_equal(raw, rounded):
        raise ValueError("displacements_ijk must contain exact integer values")
    return rounded.astype(np.int64)


def _require_signed_closure(displacements_ijk: np.ndarray) -> None:
    offsets = {tuple(int(value) for value in row) for row in displacements_ijk}
    missing = [offset for offset in offsets if tuple(-value for value in offset) not in offsets]
    if missing:
        raise ValueError("nested_core displacements must contain signed closure")


def _bounds_size(bounds: Sequence[tuple[int, int]]) -> int:
    return int(np.prod([max(0, stop - start) for start, stop in bounds], dtype=np.int64))


def _sample_unique_linear(size: int, sample_count: int | None, seed: int) -> np.ndarray:
    """Sample linear indices without allocating an array proportional to ``size``."""

    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if sample_count is None or sample_count >= size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    while len(chosen) < sample_count:
        chosen.add(int(rng.integers(0, size)))
    return np.fromiter(sorted(chosen), dtype=np.int64)


def _origin_arrays(
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    sample_count: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    widths = tuple(stop - start for start, stop in bounds)
    linear = _sample_unique_linear(_bounds_size(bounds), sample_count, seed)
    k_rel, remainder = divmod(linear, widths[1] * widths[2])
    j_rel, i_rel = divmod(remainder, widths[2])
    return k_rel + bounds[0][0], j_rel + bounds[1][0], i_rel + bounds[2][0]


def _offset_seed(seed: int, displacement_ijk: Sequence[int]) -> int:
    payload = json.dumps([int(seed), *(int(value) for value in displacement_ijk)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _ell_bin_index(value: float, edges: np.ndarray) -> int:
    if value == edges[-1]:
        return edges.size - 2
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return index if 0 <= index < edges.size - 1 else -1


def _allocate_result(
    q_fields: Mapping[str, QField],
    geometry_names: tuple[str, ...],
    config: FiniteDomainConfig,
    cube_shape_kji: tuple[int, int, int],
    core_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None,
    rho0: float,
    rho0_provenance: str,
    displacements_ijk: np.ndarray,
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
        cell_sizes=tuple(float(value) for value in config.cell_sizes),
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
    )


def _accumulate(
    magnitudes: np.ndarray,
    mask: np.ndarray,
    p_values: tuple[float, ...],
    result: FiniteDomainResult,
    index_prefix: tuple[int, int, int, int],
    ell_index: int,
) -> None:
    selected = magnitudes[mask]
    if not selected.size:
        return
    for p_index, p_value in enumerate(p_values):
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.power(selected, p_value)
        if np.any(~np.isfinite(values)):
            raise FloatingPointError("non-finite powered structure-function increment")
        index = (*index_prefix, p_index, ell_index)
        with np.errstate(over="ignore", invalid="ignore"):
            batch_sum = values.sum(dtype=np.float64)
            batch_sum_sq = np.square(values).sum(dtype=np.float64)
            cumulative_sum = result.sums[index] + batch_sum
            cumulative_sum_sq = result.sums_sq[index] + batch_sum_sq
        if not np.isfinite(batch_sum) or not np.isfinite(batch_sum_sq):
            raise FloatingPointError("non-finite structure-function accumulator contribution")
        if not np.isfinite(cumulative_sum) or not np.isfinite(cumulative_sum_sq):
            raise FloatingPointError("non-finite cumulative structure-function accumulator")
        result.counts[index] += values.size
        result.sums[index] = cumulative_sum
        result.sums_sq[index] = cumulative_sum_sq


def compute_finite_domain_structure_functions(
    cube_data: Mapping[str, np.ndarray],
    displacements_ijk: np.ndarray,
    *,
    config: FiniteDomainConfig,
    rho0: float | None = None,
    rho0_provenance: str | None = None,
    rho_floor: float = 0.0,
    q_names: Sequence[str] | None = None,
) -> FiniteDomainResult:
    """Compute non-periodic 3-D conditional structure functions.

    Nested-core mode samples one common inner origin set that is valid for all
    configured offsets. All-valid-pairs mode uses the largest valid origin box
    for each individual offset. Neither mode applies periodic wrapping.
    """

    started = perf_counter()
    displacements = _require_integer_displacements(displacements_ijk)
    if np.any(np.all(displacements == 0, axis=1)):
        raise ValueError("zero displacement is not permitted")
    if len({tuple(row) for row in displacements.tolist()}) != len(displacements):
        raise ValueError("displacements_ijk must not contain duplicates")
    if config.pair_mode == "nested_core":
        _require_signed_closure(displacements)

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
    cube_shape_kji = tuple(int(value) for value in B.shape[1:])
    geometry_names = (
        FINITE_DOMAIN_GEOMETRY_NAMES
        if config.include_subvolume_mean
        else FINITE_DOMAIN_GEOMETRY_NAMES[:1]
    )
    in_range_displacements = np.asarray(
        [
            displacement
            for displacement in displacements
            if _ell_bin_index(
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
        nested_core_bounds_kji(cube_shape_kji, in_range_displacements)
        if config.pair_mode == "nested_core"
        else None
    )
    result = _allocate_result(
        q_fields,
        geometry_names,
        config,
        cube_shape_kji,
        core_bounds,
        rho0,
        rho0_provenance,
        displacements,
    )
    finite_B_points = _finite_vector(B)
    B_mean_sub = (
        np.mean(B[:, finite_B_points], axis=1, dtype=np.float64)
        if np.any(finite_B_points)
        else np.full(3, np.nan)
    )
    cube_cell_count = int(np.prod(cube_shape_kji, dtype=np.int64))
    exclusion_index = {name: index for index, name in enumerate(EXCLUSION_NAMES)}
    shared_origins = _origin_arrays(core_bounds, config.sample_count, config.seed) if core_bounds else None

    for displacement_index, displacement in enumerate(displacements):
        r_vector = cube_offset_to_vector(displacement, config.cell_sizes)
        ell_index = _ell_bin_index(float(np.linalg.norm(r_vector)), config.ell_bin_edges)
        result.ell_bin_index_per_displacement[displacement_index] = ell_index
        if ell_index < 0:
            result.out_of_range_displacements += 1
            continue
        result.displacements_per_bin[ell_index] += 1
        bounds = core_bounds or valid_origin_bounds_kji(cube_shape_kji, displacement)
        eligible = _bounds_size(bounds)
        result.eligible_pairs[ell_index] += eligible
        result.cube_candidate_pairs[ell_index] += cube_cell_count
        result.excluded_boundary_pairs[ell_index] += cube_cell_count - eligible
        result.eligible_pairs_per_displacement[displacement_index] = eligible
        result.cube_candidate_pairs_per_displacement[displacement_index] = cube_cell_count
        result.excluded_boundary_pairs_per_displacement[displacement_index] = cube_cell_count - eligible
        origins = (
            shared_origins
            if shared_origins is not None
            else _origin_arrays(bounds, config.sample_count, _offset_seed(config.seed, displacement))
        )
        sampled = origins[0].size
        result.sampled_pairs[ell_index] += sampled
        result.sampled_pairs_per_displacement[displacement_index] = sampled
        di, dj, dk = (int(value) for value in displacement)

        for begin in range(0, sampled, config.pair_batch_size):
            stop = min(sampled, begin + config.pair_batch_size)
            k0, j0, i0 = (values[begin:stop] for values in origins)
            k1, j1, i1 = k0 + dk, j0 + dj, i0 + di
            if (
                np.any(k1 < 0) or np.any(k1 >= cube_shape_kji[0])
                or np.any(j1 < 0) or np.any(j1 >= cube_shape_kji[1])
                or np.any(i1 < 0) or np.any(i1 >= cube_shape_kji[2])
            ):
                raise RuntimeError("finite-domain origin construction produced an invalid endpoint")

            B_left = B[:, k0, j0, i0].T
            B_right = B[:, k1, j1, i1].T
            finite_B_pair = np.all(np.isfinite(B_left), axis=1) & np.all(np.isfinite(B_right), axis=1)
            B_loc = 0.5 * (B_left + B_right)

            for geometry_index, geometry_name in enumerate(geometry_names):
                B_direction = B_loc if geometry_name == "pair_local" else np.broadcast_to(B_mean_sub, B_loc.shape)
                valid_B = np.all(np.isfinite(B_direction), axis=1)
                if geometry_name == "pair_local":
                    valid_B &= finite_B_pair
                B_mag = np.linalg.norm(B_direction, axis=1)
                valid_parallel = valid_B & (B_mag > config.B_epsilon)
                e_parallel = np.full_like(B_direction, np.nan, dtype=float)
                e_parallel[valid_parallel] = B_direction[valid_parallel] / B_mag[valid_parallel, None]

                r_rows = np.broadcast_to(r_vector, B_loc.shape)
                theta = folded_angle(r_rows, e_parallel)
                r_perp = r_rows - np.einsum("ij,ij->i", r_rows, e_parallel)[:, None] * e_parallel
                r_perp_mag = np.linalg.norm(r_perp, axis=1)

                for q_index, q_field in enumerate(q_fields.values()):
                    q_left = q_field.values[:, k0, j0, i0].T
                    q_right = q_field.values[:, k1, j1, i1].T
                    valid_q_pair = q_field.valid[k0, j0, i0] & q_field.valid[k1, j1, i1]
                    valid_q_pair &= np.all(np.isfinite(q_left), axis=1) & np.all(np.isfinite(q_right), axis=1)

                    result.exclusions[q_index, geometry_index, exclusion_index["invalid_B"], ell_index] += np.count_nonzero(~valid_B)
                    result.exclusions[q_index, geometry_index, exclusion_index["weak_B_direction"], ell_index] += np.count_nonzero(valid_B & ~valid_parallel)
                    result.exclusions[q_index, geometry_index, exclusion_index["invalid_q"], ell_index] += np.count_nonzero(valid_parallel & ~valid_q_pair)

                    valid = valid_parallel & valid_q_pair
                    delta_q = q_right - q_left
                    total_mag = np.linalg.norm(delta_q, axis=1)
                    delta_q_perp = delta_q - np.einsum("ij,ij->i", delta_q, e_parallel)[:, None] * e_parallel
                    q_perp_mag = np.linalg.norm(delta_q_perp, axis=1)
                    masks = {
                        "all": valid,
                        "parallel": valid & (theta <= config.theta_parallel_max),
                        "perpendicular": valid & (theta >= config.theta_perpendicular_min),
                    }
                    magnitudes = (total_mag, q_perp_mag)
                    for measurement_index, magnitude in enumerate(magnitudes):
                        for direction_index, direction_name in enumerate(DIRECTION_NAMES[:3]):
                            _accumulate(
                                magnitude,
                                masks[direction_name],
                                config.p_values,
                                result,
                                (q_index, geometry_index, measurement_index, direction_index),
                                ell_index,
                            )

                    valid_q_perp = valid & np.isfinite(q_perp_mag) & (q_perp_mag > config.q_perp_epsilon)
                    valid_r_perp = valid_q_perp & np.isfinite(r_perp_mag) & (r_perp_mag > config.r_perp_epsilon)
                    result.exclusions[q_index, geometry_index, exclusion_index["weak_q_perp_for_phi"], ell_index] += np.count_nonzero(valid & ~valid_q_perp)
                    result.exclusions[q_index, geometry_index, exclusion_index["weak_r_perp_for_phi"], ell_index] += np.count_nonzero(valid_q_perp & ~valid_r_perp)
                    e_xi = np.full_like(delta_q_perp, np.nan, dtype=float)
                    e_xi[valid_r_perp] = delta_q_perp[valid_r_perp] / q_perp_mag[valid_r_perp, None]
                    phi = folded_angle(r_perp, e_xi)
                    perpendicular = valid_r_perp & (theta >= config.theta_perpendicular_min)
                    directional_masks = (
                        perpendicular & (phi <= config.phi_xi_max),
                        perpendicular & (phi >= config.phi_lambda_min),
                    )
                    for measurement_index, magnitude in enumerate(magnitudes):
                        for relative_index, mask in enumerate(directional_masks):
                            _accumulate(
                                magnitude,
                                mask,
                                config.p_values,
                                result,
                                (q_index, geometry_index, measurement_index, relative_index + 3),
                                ell_index,
                            )
    result.elapsed_seconds = perf_counter() - started
    return result


def generate_fibonacci_displacements(
    radii: Sequence[float],
    *,
    directions_per_radius: int = 24,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate deterministic approximately spherical integer IJK offsets."""

    if directions_per_radius < 6:
        raise ValueError("directions_per_radius must be at least 6")
    radii = tuple(float(value) for value in radii)
    if not radii or any(not np.isfinite(value) or value <= 0.0 for value in radii):
        raise ValueError("radii must contain positive finite values")
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    offsets: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    if directions_per_radius % 2:
        raise ValueError("directions_per_radius must be even for signed closure")
    for radius in radii:
        for index in range(directions_per_radius // 2):
            z = (index + 0.5) / (directions_per_radius // 2)
            radial = np.sqrt(max(0.0, 1.0 - z * z))
            angle = phase + index * golden_angle
            vector = radius * np.array([radial * np.cos(angle), radial * np.sin(angle), z])
            offset = tuple(int(value) for value in np.rint(vector))
            for signed_offset in (offset, tuple(-value for value in offset)):
                if signed_offset == (0, 0, 0) or signed_offset in seen:
                    continue
                seen.add(signed_offset)
                offsets.append(signed_offset)
    if not offsets:
        raise ValueError("radii produced no nonzero integer displacements")
    return np.asarray(offsets, dtype=np.int64)
