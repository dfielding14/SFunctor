"""Dense displacement design and node-local parallelism for Phase 3a."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from typing import Mapping, Sequence

import numpy as np

from sfunctor.core.finite_domain import (
    FiniteDomainConfig,
    FiniteDomainResult,
    _require_integer_displacements,
    compute_finite_domain_structure_functions,
    generate_fibonacci_displacements,
)

__all__ = [
    "dense_separation_centers",
    "dense_displacement_manifest",
    "displacement_shard",
    "compute_finite_domain_structure_functions_parallel",
    "prepare_cube_for_parallel",
]

_APPROVED_ELL_MAX = {2: 320, 3: 160, 5: 80}


def dense_separation_centers(ell_max: int, bin_count: int) -> np.ndarray:
    """Return a deterministic hybrid integer separation design.

    The design keeps consecutive integer radii at small scales, then switches
    to geometric spacing.  This resolves the dissipative range without forcing
    physical fits to use it and retains dense large-scale coverage.
    """

    if ell_max < 32:
        raise ValueError("ell_max must be at least 32 cells")
    if bin_count < 1 or bin_count > ell_max:
        raise ValueError("bin_count must be in [1, ell_max]")
    linear_count = min(16, bin_count)
    centers = list(range(1, linear_count + 1))
    if bin_count > linear_count:
        candidates = np.geomspace(linear_count + 1, ell_max, 8 * (bin_count - linear_count))
        for value in np.rint(candidates).astype(int):
            if value > centers[-1] and value <= ell_max:
                centers.append(int(value))
            if len(centers) == bin_count:
                break
    if len(centers) < bin_count:
        for value in range(centers[-1] + 1, ell_max + 1):
            centers.append(value)
            if len(centers) == bin_count:
                break
    centers[-1] = ell_max
    output = np.asarray(sorted(set(centers)), dtype=np.int64)
    if output.size != bin_count or output[-1] != ell_max:
        raise RuntimeError("unable to construct requested hybrid separation design")
    return output


def _bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    interior = 0.5 * (centers[:-1].astype(float) + centers[1:].astype(float))
    return np.concatenate(([0.5], interior, [float(centers[-1]) + 0.5]))


def _json_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def dense_displacement_manifest(
    *,
    stencil_width: int,
    ell_max: int,
    bin_count: int,
    directions_per_bin: int,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Generate signed, integer, source-bound offsets and their manifest."""

    if stencil_width not in _APPROVED_ELL_MAX:
        raise ValueError("stencil_width must be 2, 3, or 5")
    if ell_max > _APPROVED_ELL_MAX[stencil_width]:
        raise ValueError(
            f"ell_max exceeds the approved {stencil_width}-point limit "
            f"of {_APPROVED_ELL_MAX[stencil_width]}"
        )
    centers = dense_separation_centers(ell_max, bin_count)
    edges = _bin_edges_from_centers(centers)
    raw, rounding_accounting = generate_fibonacci_displacements(
        centers,
        directions_per_radius=directions_per_bin,
        phase=phase,
        return_accounting=True,
    )
    requested_candidate_count = len(centers) * directions_per_bin
    ell = np.linalg.norm(raw.astype(float), axis=1)
    retained = raw[ell <= ell_max]
    retained_ell = np.linalg.norm(retained.astype(float), axis=1)
    ell_bin = np.searchsorted(edges, retained_ell, side="right") - 1
    in_bins = (ell_bin >= 0) & (ell_bin < bin_count)
    retained, retained_ell, ell_bin = retained[in_bins], retained_ell[in_bins], ell_bin[in_bins]
    counts = np.bincount(ell_bin, minlength=bin_count)
    offsets = {tuple(int(value) for value in row) for row in retained}
    if any(tuple(-value for value in offset) not in offsets for offset in offsets):
        raise RuntimeError("dense displacement design lost signed closure")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "stencil_width": int(stencil_width),
        "ell_max_cells": int(ell_max),
        "requested_bin_count": int(bin_count),
        "requested_directions_per_bin": int(directions_per_bin),
        "requested_centers_cells": centers.tolist(),
        "ell_bin_edges_cells": edges.tolist(),
        "realized_offset_count": int(len(retained)),
        "stable_offset_ids": list(range(len(retained))),
        "stable_offset_id_sha256": hashlib.sha256(
            np.arange(len(retained), dtype=np.int64).tobytes()
        ).hexdigest(),
        "realized_offsets_per_bin": counts.tolist(),
        "realized_empty_bin_count": int(np.count_nonzero(counts == 0)),
        "realized_minimum_ell_cells": float(retained_ell.min()),
        "realized_maximum_ell_cells": float(retained_ell.max()),
        "requested_candidate_count": int(requested_candidate_count),
        **rounding_accounting,
        "post_rounding_zero_or_duplicate_removed": int(
            rounding_accounting["post_rounding_zero_offset_removed"]
            + rounding_accounting["post_rounding_duplicate_offset_removed"]
        ),
        "post_rounding_out_of_range_removed": int(len(raw) - len(retained)),
        "signed_closure": True,
        "offsets_sha256": hashlib.sha256(retained.tobytes()).hexdigest(),
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    return retained, edges, manifest


def displacement_shard(
    displacements_ijk: np.ndarray,
    shard_index: int,
    shard_count: int,
) -> np.ndarray:
    """Return one stable round-robin displacement shard."""

    displacements = _require_integer_displacements(displacements_ijk)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    shard = displacements[shard_index::shard_count]
    if not len(shard):
        raise ValueError("requested displacement shard is empty")
    return shard


_GLOBAL_CUBE: Mapping[str, np.ndarray] | None = None
_GLOBAL_SUPPORT_DISPLACEMENTS: np.ndarray | None = None
_GLOBAL_CONFIG: FiniteDomainConfig | None = None
_GLOBAL_Q_NAMES: tuple[str, ...] | None = None


def _init_fork_worker(
    cube_data: Mapping[str, np.ndarray],
    support_displacements: np.ndarray,
    config: FiniteDomainConfig,
    q_names: tuple[str, ...],
) -> None:
    global _GLOBAL_CUBE, _GLOBAL_SUPPORT_DISPLACEMENTS, _GLOBAL_CONFIG, _GLOBAL_Q_NAMES
    _GLOBAL_CUBE = cube_data
    _GLOBAL_SUPPORT_DISPLACEMENTS = support_displacements
    _GLOBAL_CONFIG = config
    _GLOBAL_Q_NAMES = q_names


def _compute_worker(displacements: np.ndarray) -> FiniteDomainResult:
    assert _GLOBAL_CUBE is not None
    assert _GLOBAL_SUPPORT_DISPLACEMENTS is not None
    assert _GLOBAL_CONFIG is not None
    assert _GLOBAL_Q_NAMES is not None
    return compute_finite_domain_structure_functions(
        _GLOBAL_CUBE,
        displacements,
        config=_GLOBAL_CONFIG,
        q_names=_GLOBAL_Q_NAMES,
        support_displacements_ijk=_GLOBAL_SUPPORT_DISPLACEMENTS,
    )


def prepare_cube_for_parallel(cube_data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Stack required baseline vectors once so fork workers share read-only pages."""

    output = dict(cube_data)
    if "_B_vector" not in output:
        output["_B_vector"] = np.stack([np.asarray(cube_data[f"B_{axis}"]) for axis in "xyz"])
    if "_v_vector" not in output:
        output["_v_vector"] = np.stack([np.asarray(cube_data[f"v_{axis}"]) for axis in "xyz"])
    return output


def compute_finite_domain_structure_functions_parallel(
    cube_data: Mapping[str, np.ndarray],
    displacements_ijk: np.ndarray,
    *,
    config: FiniteDomainConfig,
    q_names: Sequence[str] = ("B", "u"),
    worker_count: int = 1,
    support_displacements_ijk: np.ndarray | None = None,
) -> list[FiniteDomainResult]:
    """Compute deterministic offset shards using shared copy-on-write cube pages.

    The returned shard results remain separate so the strict Phase 3a reducer
    can validate overlap, completeness, metadata, and reduction ordering.
    """

    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    displacements = _require_integer_displacements(displacements_ijk)
    support_displacements = (
        displacements
        if support_displacements_ijk is None
        else _require_integer_displacements(support_displacements_ijk)
    )
    if worker_count == 1:
        return [
            compute_finite_domain_structure_functions(
                cube_data,
                displacements,
                config=config,
                q_names=tuple(q_names),
                support_displacements_ijk=support_displacements,
            )
        ]
    worker_count = min(worker_count, len(displacements))
    shards = [displacement_shard(displacements, index, worker_count) for index in range(worker_count)]
    shared_cube = prepare_cube_for_parallel(cube_data)
    context = mp.get_context("fork")
    with context.Pool(
        processes=worker_count,
        initializer=_init_fork_worker,
        initargs=(shared_cube, support_displacements, replace(config), tuple(q_names)),
    ) as pool:
        return pool.map(_compute_worker, shards)
