"""Strict pairwise three-direction structure functions for 2-D slices.

This module implements the local basis used for the Chen/Mallet-style
conditional structure functions.  It is intentionally separate from the
legacy unified histogram kernels: those kernels support higher-order stencils
and a magnetic-fluctuation azimuth, while this path uses a two-point increment
and lets each measured field define its own fluctuation direction.

AthenaK stores arrays in KJI order.  The input arrays remain in their native
2-D slice layout; :func:`slice_offset_to_vector` is the only place where a
slice-native ``(delta_i, delta_j)`` offset becomes a Cartesian vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "DIRECTION_NAMES",
    "EXCLUSION_NAMES",
    "GEOMETRY_NAMES",
    "QField",
    "DirectionalConfig",
    "DirectionalResult",
    "build_local_basis",
    "build_q_variants",
    "compute_directional_structure_functions",
    "folded_angle",
    "slice_offset_to_vector",
]


DIRECTION_NAMES = ("all", "parallel", "perpendicular", "xi", "lambda")
GEOMETRY_NAMES = ("local", "global")
EXCLUSION_NAMES = (
    "invalid_B",
    "weak_B_direction",
    "invalid_q",
    "weak_q_perp_for_phi",
    "weak_r_perp_for_phi",
)


@dataclass(frozen=True)
class QField:
    """A vector field variant and its pointwise validity mask."""

    name: str
    values: np.ndarray
    valid: np.ndarray
    density_convention: str


@dataclass(frozen=True)
class DirectionalConfig:
    """Configuration for strict pairwise directional structure functions.

    Angles are folded onto ``[0, pi/2]`` because an eddy axis is unoriented:
    parallel and anti-parallel separations are identified.
    """

    ell_bin_edges: np.ndarray
    cell_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    theta_parallel_max: float = np.deg2rad(15.0)
    theta_perpendicular_min: float = np.deg2rad(75.0)
    phi_xi_max: float = np.deg2rad(15.0)
    phi_lambda_min: float = np.deg2rad(75.0)
    B_epsilon: float = 1.0e-12
    q_perp_epsilon: float = 1.0e-12
    r_perp_epsilon: float = 1.0e-12
    sample_count: int | None = None
    seed: int = 0
    include_global: bool = True

    def __post_init__(self) -> None:
        edges = np.asarray(self.ell_bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("ell_bin_edges must be a 1-D array with at least two edges")
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
            raise ValueError("ell_bin_edges must contain finite, strictly increasing values")
        if len(self.cell_sizes) != 3 or not np.all(np.isfinite(self.cell_sizes)):
            raise ValueError("cell_sizes must contain three finite Cartesian spacings")
        if np.any(np.asarray(self.cell_sizes) <= 0.0):
            raise ValueError("cell_sizes must be positive")
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
        if self.B_epsilon < 0.0 or self.q_perp_epsilon < 0.0 or self.r_perp_epsilon < 0.0:
            raise ValueError("epsilon values must be non-negative")
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive or None")
        object.__setattr__(self, "ell_bin_edges", edges)


@dataclass
class DirectionalResult:
    """Directional sums, counts, exclusions, and reproducibility metadata."""

    q_names: tuple[str, ...]
    density_conventions: tuple[str, ...]
    geometry_names: tuple[str, ...]
    direction_names: tuple[str, ...]
    exclusion_names: tuple[str, ...]
    ell_bin_edges: np.ndarray
    counts: np.ndarray
    sums: np.ndarray
    sums_sq: np.ndarray
    exclusions: np.ndarray
    attempts: np.ndarray
    out_of_range_displacements: int
    out_of_range_attempts: int
    rho0: float
    cell_sizes: tuple[float, float, float]
    angle_limits: dict[str, float]
    sample_count: int | None
    seed: int
    elapsed_seconds: float

    @property
    def s2(self) -> np.ndarray:
        """Return ``<|delta q_perp|^2>`` with empty bins represented by NaN."""

        out = np.full_like(self.sums, np.nan, dtype=float)
        return np.divide(self.sums, self.counts, out=out, where=self.counts > 0)

    @property
    def standard_error(self) -> np.ndarray:
        """Return the sample standard error of the mean in populated bins.

        Spatial pairs are correlated, so this is a sampling-noise diagnostic,
        not a complete physical uncertainty estimate.
        """

        out = np.full_like(self.sums, np.nan, dtype=float)
        mask = self.counts > 1
        numerator = self.sums_sq[mask] - self.sums[mask] ** 2 / self.counts[mask]
        numerator = np.maximum(numerator, 0.0)
        out[mask] = np.sqrt(numerator / (self.counts[mask] * (self.counts[mask] - 1)))
        return out


def _as_vector_field(slice_data: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    """Return Cartesian components with shape ``(3, ny, nx)``."""

    arrays = [np.asarray(slice_data[f"{prefix}_{component}"], dtype=float) for component in "xyz"]
    if any(arr.ndim != 2 for arr in arrays):
        raise ValueError(f"{prefix} components must be 2-D slice arrays")
    if any(arr.shape != arrays[0].shape for arr in arrays[1:]):
        raise ValueError(f"{prefix} components must have matching shapes")
    return np.stack(arrays, axis=0)


def _finite_vector(field: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(field), axis=0)


def build_q_variants(
    slice_data: Mapping[str, np.ndarray],
    *,
    rho0: float | None = None,
    rho_floor: float = 0.0,
) -> tuple[np.ndarray, dict[str, QField], float]:
    """Build the requested compressible-MHD field variants.

    ``rho0`` defaults to the volume mean over finite positive-density cells.
    This is a documented reference convention, not a claim that compressible
    full MHD has one uniquely preferred Elsasser generalization.
    """

    B = _as_vector_field(slice_data, "B")
    u = _as_vector_field(slice_data, "v")
    rho = np.asarray(slice_data["rho"], dtype=float)
    if rho.ndim != 2 or rho.shape != B.shape[1:] or u.shape[1:] != B.shape[1:]:
        raise ValueError("rho, B, and v must share one 2-D slice shape")
    if not np.isfinite(rho_floor) or rho_floor < 0.0:
        raise ValueError("rho_floor must be finite and non-negative")

    valid_B = _finite_vector(B)
    valid_u = _finite_vector(u)
    valid_rho = np.isfinite(rho) & (rho > rho_floor)
    if rho0 is None:
        if not np.any(valid_rho):
            raise ValueError("rho0 cannot be inferred: no finite density exceeds rho_floor")
        rho0 = float(np.mean(rho[valid_rho]))
    if not np.isfinite(rho0) or rho0 <= rho_floor:
        raise ValueError("rho0 must be finite and greater than rho_floor")

    vA = np.full_like(B, np.nan)
    vA[:, valid_rho] = B[:, valid_rho] / np.sqrt(rho[valid_rho])
    vA_ref = B / np.sqrt(rho0)

    valid_vA = valid_B & valid_rho
    valid_vA_ref = valid_B
    valid_z = valid_u & valid_vA
    valid_z_ref = valid_u & valid_vA_ref

    q_fields = {
        "z_plus": QField("z_plus", u + vA, valid_z, "pointwise rho"),
        "z_minus": QField("z_minus", u - vA, valid_z, "pointwise rho"),
        "z_plus_ref": QField("z_plus_ref", u + vA_ref, valid_z_ref, f"fixed rho0={rho0:.16g}"),
        "z_minus_ref": QField("z_minus_ref", u - vA_ref, valid_z_ref, f"fixed rho0={rho0:.16g}"),
        "B": QField("B", B, valid_B, "not applicable"),
        "vA": QField("vA", vA, valid_vA, "pointwise rho"),
        "vA_ref": QField("vA_ref", vA_ref, valid_vA_ref, f"fixed rho0={rho0:.16g}"),
        "u": QField("u", u, valid_u, "not applicable"),
    }
    return B, q_fields, rho0


def slice_offset_to_vector(
    slice_axis: int,
    displacement: Sequence[int],
    cell_sizes: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Convert slice-native ``(delta_i, delta_j)`` to Cartesian ``(x1,x2,x3)``.

    In AthenaK KJI storage, the first 2-D array index is the row and the
    second is the column.  ``delta_i`` is the column offset and ``delta_j`` is
    the row offset, matching the historical SFunctor displacement files.
    """

    if slice_axis not in (1, 2, 3):
        raise ValueError("slice_axis must be 1, 2, or 3")
    delta_i, delta_j = (int(displacement[0]), int(displacement[1]))
    dx1, dx2, dx3 = (float(value) for value in cell_sizes)
    if slice_axis == 1:  # array plane (k=x3, j=x2)
        return np.array([0.0, delta_i * dx2, delta_j * dx3])
    if slice_axis == 2:  # array plane (k=x3, i=x1)
        return np.array([delta_i * dx1, 0.0, delta_j * dx3])
    return np.array([delta_i * dx1, delta_j * dx2, 0.0])  # plane (j=x2, i=x1)


def folded_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the unoriented angle between rows of vectors in ``[0, pi/2]``."""

    numerator = np.abs(np.einsum("ij,ij->i", left, right))
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
    return np.arccos(np.clip(cosine, 0.0, 1.0))


def build_local_basis(
    B_left: np.ndarray,
    B_right: np.ndarray,
    delta_q: np.ndarray,
    *,
    B_epsilon: float = 1.0e-12,
    q_perp_epsilon: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct ``(e_parallel, e_xi, e_lambda)`` for vectorized point pairs.

    Inputs have shape ``(n_pairs, 3)``.  Invalid or degenerate rows are filled
    with NaN and marked false in the returned validity mask.
    """

    B_loc = 0.5 * (np.asarray(B_left, dtype=float) + np.asarray(B_right, dtype=float))
    delta_q = np.asarray(delta_q, dtype=float)
    if B_loc.ndim != 2 or B_loc.shape[1] != 3 or delta_q.shape != B_loc.shape:
        raise ValueError("B_left, B_right, and delta_q must all have shape (n_pairs, 3)")
    B_mag = np.linalg.norm(B_loc, axis=1)
    finite = np.all(np.isfinite(B_loc), axis=1) & np.all(np.isfinite(delta_q), axis=1)
    valid_B = finite & (B_mag > B_epsilon)

    e_parallel = np.full_like(B_loc, np.nan)
    e_parallel[valid_B] = B_loc[valid_B] / B_mag[valid_B, None]
    delta_q_perp = delta_q - np.einsum("ij,ij->i", delta_q, e_parallel)[:, None] * e_parallel
    q_perp_mag = np.linalg.norm(delta_q_perp, axis=1)
    valid = valid_B & np.isfinite(q_perp_mag) & (q_perp_mag > q_perp_epsilon)

    e_xi = np.full_like(B_loc, np.nan)
    e_xi[valid] = delta_q_perp[valid] / q_perp_mag[valid, None]
    e_lambda = np.full_like(B_loc, np.nan)
    e_lambda[valid] = np.cross(e_parallel[valid], e_xi[valid])
    return e_parallel, e_xi, e_lambda, valid


def _ell_bin_index(value: float, edges: np.ndarray) -> int:
    if value == edges[-1]:
        return edges.size - 2
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return index if 0 <= index < edges.size - 1 else -1


def _origin_indices(shape: tuple[int, int], sample_count: int | None, seed: int) -> np.ndarray:
    n_cells = shape[0] * shape[1]
    if sample_count is None:
        return np.arange(n_cells, dtype=np.int64)
    if sample_count > n_cells:
        raise ValueError(f"sample_count={sample_count} exceeds the number of slice cells ({n_cells})")
    return np.random.default_rng(seed).choice(n_cells, size=sample_count, replace=False)


def _allocate_result(
    q_fields: Mapping[str, QField],
    geometry_names: tuple[str, ...],
    config: DirectionalConfig,
    rho0: float,
) -> DirectionalResult:
    shape = (len(q_fields), len(geometry_names), len(DIRECTION_NAMES), config.ell_bin_edges.size - 1)
    exclusion_shape = (len(q_fields), len(geometry_names), len(EXCLUSION_NAMES), config.ell_bin_edges.size - 1)
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
    exclusions=np.zeros(exclusion_shape, dtype=np.int64),
        attempts=np.zeros(shape[-1], dtype=np.int64),
        out_of_range_displacements=0,
        out_of_range_attempts=0,
        rho0=rho0,
        cell_sizes=tuple(float(value) for value in config.cell_sizes),
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


def _accumulate(values: np.ndarray, mask: np.ndarray, result: DirectionalResult, index: tuple[int, int, int, int]) -> None:
    selected = values[mask]
    result.counts[index] += selected.size
    result.sums[index] += selected.sum()
    result.sums_sq[index] += np.square(selected).sum()


def compute_directional_structure_functions(
    slice_data: Mapping[str, np.ndarray],
    displacements: np.ndarray,
    *,
    slice_axis: int,
    config: DirectionalConfig,
    rho0: float | None = None,
    rho_floor: float = 0.0,
    q_names: Sequence[str] | None = None,
) -> DirectionalResult:
    """Compute strict pairwise ``<|delta q_perp|^2>`` directional statistics.

    The local calculation uses ``B_loc = (B(x) + B(x+r)) / 2``.  The optional
    global calculation is retained only as a sanity check.  Periodicity is
    explicit in both in-plane array dimensions.
    """

    start = perf_counter()
    B, available, rho0 = build_q_variants(slice_data, rho0=rho0, rho_floor=rho_floor)
    if q_names is None:
        q_names = tuple(available)
    unknown = set(q_names) - available.keys()
    if unknown:
        raise ValueError(f"Unknown q variants: {sorted(unknown)}")
    q_fields = {name: available[name] for name in q_names}
    geometry_names = GEOMETRY_NAMES if config.include_global else GEOMETRY_NAMES[:1]
    result = _allocate_result(q_fields, geometry_names, config, rho0)

    displacements = np.asarray(displacements)
    if displacements.ndim != 2 or displacements.shape[1] != 2:
        raise ValueError("displacements must have shape (n, 2)")
    if slice_axis not in (1, 2, 3):
        raise ValueError("slice_axis must be 1, 2, or 3")

    ny, nx = B.shape[1:]
    origins = _origin_indices((ny, nx), config.sample_count, config.seed)
    y0, x0 = origins // nx, origins % nx
    finite_B_points = _finite_vector(B)
    global_B = (
        np.mean(B[:, finite_B_points], axis=1)
        if np.any(finite_B_points)
        else np.full(3, np.nan)
    )
    exclusion_index = {name: index for index, name in enumerate(EXCLUSION_NAMES)}

    for displacement in displacements:
        delta_i, delta_j = (int(displacement[0]), int(displacement[1]))
        r_vector = slice_offset_to_vector(slice_axis, (delta_i, delta_j), config.cell_sizes)
        ell = float(np.linalg.norm(r_vector))
        ell_index = _ell_bin_index(ell, config.ell_bin_edges)
        if ell_index < 0:
            result.out_of_range_displacements += 1
            result.out_of_range_attempts += origins.size
            continue
        result.attempts[ell_index] += origins.size
        x1 = (x0 + delta_i) % nx
        y1 = (y0 + delta_j) % ny

        B_left = B[:, y0, x0].T
        B_right = B[:, y1, x1].T
        finite_B_pair = np.all(np.isfinite(B_left), axis=1) & np.all(np.isfinite(B_right), axis=1)
        B_loc = 0.5 * (B_left + B_right)

        for geometry_index, geometry_name in enumerate(geometry_names):
            B_direction = B_loc if geometry_name == "local" else np.broadcast_to(global_B, B_loc.shape)
            B_mag = np.linalg.norm(B_direction, axis=1)
            valid_B = np.all(np.isfinite(B_direction), axis=1)
            if geometry_name == "local":
                valid_B &= finite_B_pair
            valid_parallel = valid_B & (B_mag > config.B_epsilon)
            e_parallel = np.full_like(B_direction, np.nan)
            e_parallel[valid_parallel] = B_direction[valid_parallel] / B_mag[valid_parallel, None]

            r_rows = np.broadcast_to(r_vector, B_loc.shape)
            theta = folded_angle(r_rows, e_parallel)
            r_perp = r_rows - np.einsum("ij,ij->i", r_rows, e_parallel)[:, None] * e_parallel
            r_perp_mag = np.linalg.norm(r_perp, axis=1)

            for q_index, q_field in enumerate(q_fields.values()):
                q_left = q_field.values[:, y0, x0].T
                q_right = q_field.values[:, y1, x1].T
                valid_q_pair = q_field.valid[y0, x0] & q_field.valid[y1, x1]
                valid_q_pair &= np.all(np.isfinite(q_left), axis=1) & np.all(np.isfinite(q_right), axis=1)

                result.exclusions[q_index, geometry_index, exclusion_index["invalid_B"], ell_index] += np.count_nonzero(~valid_B)
                result.exclusions[q_index, geometry_index, exclusion_index["weak_B_direction"], ell_index] += np.count_nonzero(valid_B & ~valid_parallel)
                result.exclusions[q_index, geometry_index, exclusion_index["invalid_q"], ell_index] += np.count_nonzero(valid_parallel & ~valid_q_pair)

                valid = valid_parallel & valid_q_pair
                delta_q = q_right - q_left
                delta_q_perp = delta_q - np.einsum("ij,ij->i", delta_q, e_parallel)[:, None] * e_parallel
                q_perp_mag = np.linalg.norm(delta_q_perp, axis=1)
                s2 = np.square(q_perp_mag)

                masks = {
                    "all": valid,
                    "parallel": valid & (theta <= config.theta_parallel_max),
                    "perpendicular": valid & (theta >= config.theta_perpendicular_min),
                }
                for direction_index, direction_name in enumerate(DIRECTION_NAMES[:3]):
                    _accumulate(s2, masks[direction_name], result, (q_index, geometry_index, direction_index, ell_index))

                valid_q_perp = valid & np.isfinite(q_perp_mag) & (q_perp_mag > config.q_perp_epsilon)
                valid_r_perp = valid_q_perp & np.isfinite(r_perp_mag) & (r_perp_mag > config.r_perp_epsilon)
                result.exclusions[q_index, geometry_index, exclusion_index["weak_q_perp_for_phi"], ell_index] += np.count_nonzero(valid & ~valid_q_perp)
                result.exclusions[q_index, geometry_index, exclusion_index["weak_r_perp_for_phi"], ell_index] += np.count_nonzero(valid_q_perp & ~valid_r_perp)

                e_xi = np.full_like(delta_q_perp, np.nan)
                e_xi[valid_r_perp] = delta_q_perp[valid_r_perp] / q_perp_mag[valid_r_perp, None]
                phi = folded_angle(r_perp, e_xi)
                perpendicular = valid_r_perp & (theta >= config.theta_perpendicular_min)
                xi_mask = perpendicular & (phi <= config.phi_xi_max)
                lambda_mask = perpendicular & (phi >= config.phi_lambda_min)
                _accumulate(s2, xi_mask, result, (q_index, geometry_index, 3, ell_index))
                _accumulate(s2, lambda_mask, result, (q_index, geometry_index, 4, ell_index))

    result.elapsed_seconds = perf_counter() - start
    return result
