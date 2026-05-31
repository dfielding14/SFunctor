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
    "STENCIL_WIDTHS",
    "FiniteDomainConfig",
    "FiniteDomainResult",
    "build_cube_q_variants",
    "compute_finite_domain_structure_functions",
    "cube_offset_to_vector",
    "generate_fibonacci_displacements",
    "nested_core_bounds_kji",
    "stencil_definition",
    "valid_origin_bounds_kji",
]


FINITE_DOMAIN_GEOMETRY_NAMES = ("pair_local", "subvolume_mean")
MEASUREMENT_NAMES = ("total", "perpendicular")
PAIR_MODES = ("nested_core", "shell_local", "all_valid_pairs", "all_valid_origins")
STENCIL_WIDTHS = (2, 3, 5)


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
    stencil_width: int = 2
    block_shape_kji: tuple[int, int, int] | None = None

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
        if self.stencil_width not in STENCIL_WIDTHS:
            raise ValueError(f"stencil_width must be one of {STENCIL_WIDTHS}")
        if self.pair_mode == "all_valid_pairs" and self.stencil_width != 2:
            raise ValueError("all_valid_pairs is a historical 2-point compatibility label")
        if self.block_shape_kji is not None:
            block_shape = _require_integer_triplet(
                self.block_shape_kji, "block_shape_kji", positive=True
            )
            object.__setattr__(self, "block_shape_kji", block_shape)
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
    elapsed_seconds_per_ell_bin: np.ndarray
    stencil_width: int = 2
    shell_core_bounds_kji: tuple[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None, ...
    ] | None = None
    block_shape_kji: tuple[int, int, int] | None = None
    block_counts: np.ndarray | None = None
    block_sums: np.ndarray | None = None
    block_sums_sq: np.ndarray | None = None
    support_displacements_sha256: str = ""
    support_displacement_count: int = 0
    block_assignment: str | None = None
    block_sampled_origins: np.ndarray | None = None
    block_eligible_origins: np.ndarray | None = None
    block_exclusions: np.ndarray | None = None
    intrinsic_eligible_origins: np.ndarray | None = None
    boundary_excluded_origins: np.ndarray | None = None
    support_policy_excluded_origins: np.ndarray | None = None
    intrinsic_eligible_origins_per_displacement: np.ndarray | None = None
    boundary_excluded_origins_per_displacement: np.ndarray | None = None
    support_policy_excluded_origins_per_displacement: np.ndarray | None = None

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

    @property
    def sampled_origins(self) -> np.ndarray:
        """Return sampled valid stencil origins; ``sampled_pairs`` is the compatibility name."""

        return self.sampled_pairs

    @property
    def eligible_origins(self) -> np.ndarray:
        """Return selected support-policy origins; ``eligible_pairs`` is the compatibility name."""

        return self.eligible_pairs


def _as_cube_vector(cube_data: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    """Return Cartesian components with shape ``(3, nk, nj, ni)``."""

    prebuilt = cube_data.get(f"_{prefix}_vector")
    if prebuilt is not None:
        vector = np.asarray(prebuilt)
        if vector.ndim != 4 or vector.shape[0] != 3:
            raise ValueError(f"_{prefix}_vector must have shape (3, nk, nj, ni)")
        component_shape = np.asarray(cube_data[f"{prefix}_x"]).shape
        if vector.shape[1:] != component_shape:
            raise ValueError(
                f"_{prefix}_vector spatial shape must match {prefix} component arrays"
            )
        return vector
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
    stencil_width: int = 2,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return half-open KJI origin bounds whose stencil points are valid."""

    nk, nj, ni = _require_integer_triplet(shape_kji, "shape_kji", positive=True)
    di, dj, dk = _require_integer_triplet(displacement_ijk, "displacement_ijk")
    multipliers, _, _ = stencil_definition(stencil_width)
    bounds = (
        (max(0, max(-multiplier * dk for multiplier in multipliers)),
         min(nk, min(nk - multiplier * dk for multiplier in multipliers))),
        (max(0, max(-multiplier * dj for multiplier in multipliers)),
         min(nj, min(nj - multiplier * dj for multiplier in multipliers))),
        (max(0, max(-multiplier * di for multiplier in multipliers)),
         min(ni, min(ni - multiplier * di for multiplier in multipliers))),
    )
    if any(stop < start for start, stop in bounds):
        return ((0, 0), (0, 0), (0, 0))
    return bounds


def nested_core_bounds_kji(
    shape_kji: Sequence[int],
    displacements_ijk: np.ndarray,
    stencil_width: int = 2,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return one KJI core whose origins are valid for every displacement."""

    displacements = _require_integer_displacements(displacements_ijk)
    per_offset = [
        valid_origin_bounds_kji(shape_kji, displacement, stencil_width)
        for displacement in displacements
    ]
    bounds = tuple(
        (
            max(bounds_for_offset[axis][0] for bounds_for_offset in per_offset),
            min(bounds_for_offset[axis][1] for bounds_for_offset in per_offset),
        )
        for axis in range(3)
    )
    if any(stop <= start for start, stop in bounds):
        raise ValueError("displacements leave no non-empty nested core")
    return bounds


def stencil_definition(stencil_width: int) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Return stencil multipliers, normalized increment weights, and local-B weights."""

    if stencil_width == 2:
        return (0, 1), np.asarray((-1.0, 1.0)), np.asarray((0.5, 0.5))
    if stencil_width == 3:
        return (
            (-1, 0, 1),
            np.asarray((1.0, -2.0, 1.0)) / np.sqrt(3.0),
            np.asarray((1.0, 1.0, 1.0)) / 3.0,
        )
    if stencil_width == 5:
        return (
            (-2, -1, 0, 1, 2),
            np.asarray((1.0, -4.0, 6.0, -4.0, 1.0)) / np.sqrt(35.0),
            np.asarray((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0,
        )
    raise ValueError(f"stencil_width must be one of {STENCIL_WIDTHS}")


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


def _origin_block_ids(
    origins_kji: tuple[np.ndarray, np.ndarray, np.ndarray],
    cube_shape_kji: tuple[int, int, int],
    block_shape_kji: tuple[int, int, int] | None,
    displacement_ijk: Sequence[int],
    stencil_width: int,
) -> np.ndarray | None:
    """Assign every tuple by stencil midpoint, preserving signed-offset symmetry."""

    if block_shape_kji is None:
        return None
    block_grid = tuple(
        (size + block - 1) // block
        for size, block in zip(cube_shape_kji, block_shape_kji)
    )
    di, dj, dk = _require_integer_triplet(displacement_ijk, "displacement_ijk")
    k, j, i = origins_kji
    if stencil_width == 2:
        block_k = (2 * k + dk) // (2 * block_shape_kji[0])
        block_j = (2 * j + dj) // (2 * block_shape_kji[1])
        block_i = (2 * i + di) // (2 * block_shape_kji[2])
    else:
        block_k = k // block_shape_kji[0]
        block_j = j // block_shape_kji[1]
        block_i = i // block_shape_kji[2]
    return (block_k * block_grid[1] + block_j) * block_grid[2] + block_i


def _block_population_for_bounds(
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    cube_shape_kji: tuple[int, int, int],
    block_shape_kji: tuple[int, int, int] | None,
    displacement_ijk: Sequence[int],
    stencil_width: int,
) -> np.ndarray | None:
    """Return exact origin populations by midpoint block for one bounds box."""

    if block_shape_kji is None:
        return None
    block_grid = tuple(
        (size + block - 1) // block
        for size, block in zip(cube_shape_kji, block_shape_kji)
    )
    axis_origins = tuple(np.arange(start, stop, dtype=np.int64) for start, stop in bounds)
    di, dj, dk = _require_integer_triplet(displacement_ijk, "displacement_ijk")
    shifts_kji = (dk, dj, di)
    populations = []
    for axis, values in enumerate(axis_origins):
        assigned = (
            (2 * values + shifts_kji[axis]) // (2 * block_shape_kji[axis])
            if stencil_width == 2
            else values // block_shape_kji[axis]
        )
        populations.append(np.bincount(assigned, minlength=block_grid[axis]))
    return np.einsum("k,j,i->kji", *populations).reshape(-1)


def _offset_seed(seed: int, displacement_ijk: Sequence[int]) -> int:
    payload = json.dumps([int(seed), *(int(value) for value in displacement_ijk)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _shell_seed(seed: int, ell_index: int) -> int:
    payload = json.dumps([int(seed), "shell", int(ell_index)]).encode()
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
        elapsed_seconds_per_ell_bin=np.zeros(n_ell, dtype=float),
        stencil_width=config.stencil_width,
        block_shape_kji=config.block_shape_kji,
        block_counts=np.zeros((block_count, *shape), dtype=np.int64) if block_count else None,
        block_sums=np.zeros((block_count, *shape), dtype=float) if block_count else None,
        block_sums_sq=np.zeros((block_count, *shape), dtype=float) if block_count else None,
        support_displacements_sha256=hashlib.sha256(support_displacements_ijk.tobytes()).hexdigest(),
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


def _accumulate(
    magnitudes: np.ndarray,
    mask: np.ndarray,
    p_values: tuple[float, ...],
    result: FiniteDomainResult,
    index_prefix: tuple[int, int, int, int],
    ell_index: int,
    block_ids: np.ndarray | None = None,
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
        if block_ids is not None:
            assert result.block_counts is not None
            assert result.block_sums is not None
            assert result.block_sums_sq is not None
            selected_blocks = block_ids[mask]
            block_count = result.block_counts.shape[0]
            result.block_counts[(slice(None), *index)] += np.bincount(
                selected_blocks, minlength=block_count
            )
            result.block_sums[(slice(None), *index)] += np.bincount(
                selected_blocks, weights=values, minlength=block_count
            )
            result.block_sums_sq[(slice(None), *index)] += np.bincount(
                selected_blocks, weights=np.square(values), minlength=block_count
            )


def _accumulate_block_exclusion(
    result: FiniteDomainResult,
    block_ids: np.ndarray | None,
    mask: np.ndarray,
    q_index: int,
    geometry_index: int,
    exclusion_index: int,
    ell_index: int,
) -> None:
    if block_ids is None:
        return
    assert result.block_exclusions is not None
    result.block_exclusions[:, q_index, geometry_index, exclusion_index, ell_index] += np.bincount(
        block_ids[mask], minlength=result.block_exclusions.shape[0]
    )


def compute_finite_domain_structure_functions(
    cube_data: Mapping[str, np.ndarray],
    displacements_ijk: np.ndarray,
    *,
    config: FiniteDomainConfig,
    rho0: float | None = None,
    rho0_provenance: str | None = None,
    rho_floor: float = 0.0,
    q_names: Sequence[str] | None = None,
    support_displacements_ijk: np.ndarray | None = None,
) -> FiniteDomainResult:
    """Compute non-periodic 3-D conditional structure functions.

    Nested-core mode samples one common inner origin set that is valid for all
    configured offsets. Shell-local mode uses one common origin box and sample
    per separation bin. All-valid-origin modes use the largest valid origin
    box for each individual offset. No mode applies periodic wrapping.
    """

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
    if len({tuple(row) for row in support_displacements.tolist()}) != len(support_displacements):
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
    in_range_support_displacements = np.asarray(
        [
            displacement
            for displacement in support_displacements
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
        nested_core_bounds_kji(
            cube_shape_kji, in_range_support_displacements, config.stencil_width
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
                    cube_shape_kji, members, config.stencil_width
                )
            except ValueError:
                shell_bounds[ell_index] = ((0, 0), (0, 0), (0, 0))
    result = _allocate_result(
        q_fields,
        geometry_names,
        config,
        cube_shape_kji,
        core_bounds,
        rho0,
        rho0_provenance,
        displacements,
        support_displacements,
    )
    result.shell_core_bounds_kji = tuple(shell_bounds) if config.pair_mode == "shell_local" else None
    finite_B_points = _finite_vector(B)
    B_mean_sub = (
        np.mean(B[:, finite_B_points], axis=1, dtype=np.float64)
        if np.any(finite_B_points)
        else np.full(3, np.nan)
    )
    cube_cell_count = int(np.prod(cube_shape_kji, dtype=np.int64))
    exclusion_index = {name: index for index, name in enumerate(EXCLUSION_NAMES)}
    shared_origins = _origin_arrays(core_bounds, config.sample_count, config.seed) if core_bounds else None
    shell_origins = [
        _origin_arrays(bounds, config.sample_count, _shell_seed(config.seed, ell_index))
        if bounds is not None
        else None
        for ell_index, bounds in enumerate(shell_bounds)
    ]
    multipliers, increment_weights, local_B_weights = stencil_definition(config.stencil_width)

    for displacement_index, displacement in enumerate(displacements):
        displacement_started = perf_counter()
        r_vector = cube_offset_to_vector(displacement, config.cell_sizes)
        ell_index = _ell_bin_index(float(np.linalg.norm(r_vector)), config.ell_bin_edges)
        result.ell_bin_index_per_displacement[displacement_index] = ell_index
        if ell_index < 0:
            result.out_of_range_displacements += 1
            continue
        result.displacements_per_bin[ell_index] += 1
        intrinsic_bounds = valid_origin_bounds_kji(
            cube_shape_kji, displacement, config.stencil_width
        )
        bounds = (
            core_bounds
            or shell_bounds[ell_index]
            or intrinsic_bounds
        )
        eligible = _bounds_size(bounds)
        intrinsic_eligible = _bounds_size(intrinsic_bounds)
        boundary_excluded = cube_cell_count - intrinsic_eligible
        support_policy_excluded = intrinsic_eligible - eligible
        if support_policy_excluded < 0:
            raise RuntimeError("support policy retained more origins than intrinsic stencil geometry")
        result.eligible_pairs[ell_index] += eligible
        result.cube_candidate_pairs[ell_index] += cube_cell_count
        result.excluded_boundary_pairs[ell_index] += cube_cell_count - eligible
        result.eligible_pairs_per_displacement[displacement_index] = eligible
        result.cube_candidate_pairs_per_displacement[displacement_index] = cube_cell_count
        result.excluded_boundary_pairs_per_displacement[displacement_index] = cube_cell_count - eligible
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
            cube_shape_kji,
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
            else _origin_arrays(bounds, config.sample_count, _offset_seed(config.seed, displacement))
        )
        assert origins is not None
        sampled = origins[0].size
        result.sampled_pairs[ell_index] += sampled
        result.sampled_pairs_per_displacement[displacement_index] = sampled
        di, dj, dk = (int(value) for value in displacement)

        for begin in range(0, sampled, config.pair_batch_size):
            stop = min(sampled, begin + config.pair_batch_size)
            k0, j0, i0 = (values[begin:stop] for values in origins)
            block_ids = _origin_block_ids(
                (k0, j0, i0),
                cube_shape_kji,
                config.block_shape_kji,
                displacement,
                config.stencil_width,
            )
            if block_ids is not None:
                assert result.block_sampled_origins is not None
                result.block_sampled_origins[:, ell_index] += np.bincount(
                    block_ids, minlength=result.block_sampled_origins.shape[0]
                )
            points = tuple(
                (k0 + multiplier * dk, j0 + multiplier * dj, i0 + multiplier * di)
                for multiplier in multipliers
            )
            for k_point, j_point, i_point in points:
                if (
                    np.any(k_point < 0) or np.any(k_point >= cube_shape_kji[0])
                    or np.any(j_point < 0) or np.any(j_point >= cube_shape_kji[1])
                    or np.any(i_point < 0) or np.any(i_point >= cube_shape_kji[2])
                ):
                    raise RuntimeError("finite-domain origin construction produced an invalid stencil point")

            B_points = tuple(B[:, k_point, j_point, i_point].T for k_point, j_point, i_point in points)
            finite_B_stencil = np.logical_and.reduce(
                [np.all(np.isfinite(values), axis=1) for values in B_points]
            )
            B_loc = sum(
                weight * values for weight, values in zip(local_B_weights, B_points)
            )

            for geometry_index, geometry_name in enumerate(geometry_names):
                B_direction = B_loc if geometry_name == "pair_local" else np.broadcast_to(B_mean_sub, B_loc.shape)
                valid_B = np.all(np.isfinite(B_direction), axis=1)
                if geometry_name == "pair_local":
                    valid_B &= finite_B_stencil
                B_mag = np.linalg.norm(B_direction, axis=1)
                valid_parallel = valid_B & (B_mag > config.B_epsilon)
                e_parallel = np.full_like(B_direction, np.nan, dtype=float)
                e_parallel[valid_parallel] = B_direction[valid_parallel] / B_mag[valid_parallel, None]

                r_rows = np.broadcast_to(r_vector, B_loc.shape)
                theta = folded_angle(r_rows, e_parallel)
                r_perp = r_rows - np.einsum("ij,ij->i", r_rows, e_parallel)[:, None] * e_parallel
                r_perp_mag = np.linalg.norm(r_perp, axis=1)

                for q_index, q_field in enumerate(q_fields.values()):
                    q_points = tuple(
                        q_field.values[:, k_point, j_point, i_point].T
                        for k_point, j_point, i_point in points
                    )
                    valid_q_stencil = np.logical_and.reduce(
                        [q_field.valid[k_point, j_point, i_point] for k_point, j_point, i_point in points]
                    )
                    valid_q_stencil &= np.logical_and.reduce(
                        [np.all(np.isfinite(values), axis=1) for values in q_points]
                    )

                    result.exclusions[q_index, geometry_index, exclusion_index["invalid_B"], ell_index] += np.count_nonzero(~valid_B)
                    result.exclusions[q_index, geometry_index, exclusion_index["weak_B_direction"], ell_index] += np.count_nonzero(valid_B & ~valid_parallel)
                    result.exclusions[q_index, geometry_index, exclusion_index["invalid_q"], ell_index] += np.count_nonzero(valid_parallel & ~valid_q_stencil)
                    _accumulate_block_exclusion(result, block_ids, ~valid_B, q_index, geometry_index, exclusion_index["invalid_B"], ell_index)
                    _accumulate_block_exclusion(result, block_ids, valid_B & ~valid_parallel, q_index, geometry_index, exclusion_index["weak_B_direction"], ell_index)
                    _accumulate_block_exclusion(result, block_ids, valid_parallel & ~valid_q_stencil, q_index, geometry_index, exclusion_index["invalid_q"], ell_index)

                    valid = valid_parallel & valid_q_stencil
                    delta_q = sum(
                        weight * values for weight, values in zip(increment_weights, q_points)
                    )
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
                                block_ids,
                            )

                    valid_q_perp = valid & np.isfinite(q_perp_mag) & (q_perp_mag > config.q_perp_epsilon)
                    valid_r_perp = valid_q_perp & np.isfinite(r_perp_mag) & (r_perp_mag > config.r_perp_epsilon)
                    result.exclusions[q_index, geometry_index, exclusion_index["weak_q_perp_for_phi"], ell_index] += np.count_nonzero(valid & ~valid_q_perp)
                    result.exclusions[q_index, geometry_index, exclusion_index["weak_r_perp_for_phi"], ell_index] += np.count_nonzero(valid_q_perp & ~valid_r_perp)
                    _accumulate_block_exclusion(result, block_ids, valid & ~valid_q_perp, q_index, geometry_index, exclusion_index["weak_q_perp_for_phi"], ell_index)
                    _accumulate_block_exclusion(result, block_ids, valid_q_perp & ~valid_r_perp, q_index, geometry_index, exclusion_index["weak_r_perp_for_phi"], ell_index)
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
                                block_ids,
                            )
        result.elapsed_seconds_per_ell_bin[ell_index] += perf_counter() - displacement_started
    result.elapsed_seconds = perf_counter() - started
    return result


def generate_fibonacci_displacements(
    radii: Sequence[float],
    *,
    directions_per_radius: int = 24,
    phase: float = 0.0,
    return_accounting: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    """Generate deterministic approximately spherical integer IJK offsets.

    The optional accounting result separates integer-rounding losses so dense
    displacement manifests can distinguish zero offsets from duplicates.
    """

    if directions_per_radius < 6:
        raise ValueError("directions_per_radius must be at least 6")
    radii = tuple(float(value) for value in radii)
    if not radii or any(not np.isfinite(value) or value <= 0.0 for value in radii):
        raise ValueError("radii must contain positive finite values")
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    offsets: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    zero_offset_count = 0
    duplicate_offset_count = 0
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
                if signed_offset == (0, 0, 0):
                    zero_offset_count += 1
                    continue
                if signed_offset in seen:
                    duplicate_offset_count += 1
                    continue
                seen.add(signed_offset)
                offsets.append(signed_offset)
    if not offsets:
        raise ValueError("radii produced no nonzero integer displacements")
    output = np.asarray(offsets, dtype=np.int64)
    if return_accounting:
        return output, {
            "post_rounding_zero_offset_removed": zero_offset_count,
            "post_rounding_duplicate_offset_removed": duplicate_offset_count,
        }
    return output
