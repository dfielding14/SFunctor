#!/usr/bin/env python3
"""Run the bounded Phase 4 Batch B matched-support tail diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3 import run_phase3_sampler as phase3
from scripts.phase3a import run_phase3a_sampler as inherited
from sfunctor.core.directional import folded_angle
from sfunctor.core.directional import QField
from sfunctor.core.finite_domain import (
    _bounds_size,
    _offset_seed,
    _origin_arrays,
    _origin_block_ids,
    _shell_seed,
    build_cube_q_variants,
    cube_offset_to_vector,
    nested_core_bounds_kji,
    valid_origin_bounds_kji,
)
from sfunctor.core.phase3a import dense_displacement_manifest, prepare_cube_for_parallel


SCHEMA_VERSION = 1
CUBE_IDS = ("L640_sub02822", "L640_sub03026", "L640_sub02602", "L640_sub00738")
Q_NAMES = ("B", "u")
DIRECTIONS = ("parallel", "xi", "lambda")
P_VALUES = (2.0, 4.0, 6.0)
SAMPLE_COUNTS = (2048, 8192)
SEEDS = (20260530, 20260531, 20260532)
SCIENCE_SCALE_MINIMUM = 32.0
SELECTED_OFFSET_MAXIMUM = 128
BLOCK_SHAPE_KJI = (80, 80, 80)
PAIR_BATCH_SIZE = 1024
ELL_MAX = 320
BIN_COUNT = 64
DIRECTIONS_PER_BIN = 24
SUMMARY_FILENAME = "phase4_batch_b_tail_diagnostic_summary.json"
MARKER_FILENAME = "PHASE4_BATCH_B_TAIL_DIAGNOSTIC_COMPLETE.json"
REPRESENTATIVE_SUMMARY_FILENAME = "phase4_batch_b_representative_summary.json"
REPRESENTATIVE_MARKER_FILENAME = "PHASE4_BATCH_B_REPRESENTATIVE_COMPLETE.json"


def _json_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_builtin(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_builtin(value.tolist())
    if isinstance(value, np.generic):
        return _json_builtin(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(_json_builtin(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "sfunctor" / "core" / "finite_domain.py",
        root / "sfunctor" / "core" / "phase3a.py",
        root / "scripts" / "phase3" / "run_phase3_sampler.py",
        root / "scripts" / "phase3a" / "run_phase3a_sampler.py",
        root / "job_scripts" / "phase4" / "run_phase4_batch_b_tail_diagnostic_andes.sh",
        root / "config" / "phase4_batch_b_representative_review_decision.json",
        root / "PHASE4_BATCH_B_REPRESENTATIVE_STATUS_UPDATE.md",
    )
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in paths}
    return {
        "implementation_source_hashes": hashes,
        "implementation_sha256": inherited._mapping_sha256(hashes),
    }


def _input_identity(phase2_root: Path, representative_release_root: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    decision_path = repository / "config" / "phase4_batch_b_representative_review_decision.json"
    decision = json.loads(decision_path.read_text())
    review_summary = repository / decision["review_summary"]
    summary_path = representative_release_root / REPRESENTATIVE_SUMMARY_FILENAME
    marker_path = representative_release_root / REPRESENTATIVE_MARKER_FILENAME
    campaign_path = representative_release_root / "manifests" / "campaign.json"
    summary = json.loads(summary_path.read_text())
    marker = json.loads(marker_path.read_text())
    campaign = json.loads(campaign_path.read_text())
    if (
        decision.get("schema_version") != 1
        or decision.get("status") != "hold_all21_batch_b_expansion_pending_tail_diagnostic"
        or Path(decision.get("probe_release", "")).resolve() != representative_release_root.resolve()
        or decision.get("review_summary_sha256") != file_sha256(review_summary)
        or marker.get("schema_version") != 1
        or marker.get("status") != "release_aggregation_complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or summary.get("phase") != "phase4_batch_b_bounded_8_cube_2point_p1_to_p6"
        or Path(campaign.get("phase2_root", "")).resolve() != phase2_root.resolve()
    ):
        raise RuntimeError("invalid or stale representative Batch B diagnostic predecessor")
    return {
        "phase2_root": str(phase2_root.resolve()),
        "representative_release_root": str(representative_release_root.resolve()),
        "representative_summary_sha256": file_sha256(summary_path),
        "representative_marker_sha256": file_sha256(marker_path),
        "representative_campaign_sha256": file_sha256(campaign_path),
        "representative_review_decision_sha256": file_sha256(decision_path),
        "representative_review_summary_sha256": file_sha256(review_summary),
    }


def _linear_to_origins(
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    linear: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    widths = tuple(stop - start for start, stop in bounds)
    k_rel, remainder = divmod(linear, widths[1] * widths[2])
    j_rel, i_rel = divmod(remainder, widths[2])
    return k_rel + bounds[0][0], j_rel + bounds[1][0], i_rel + bounds[2][0]


def _inside_bounds(
    origins: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    return np.logical_and.reduce(
        [
            (values >= start) & (values < stop)
            for values, (start, stop) in zip(origins, bounds)
        ]
    )


def _sample_exterior_origins(
    intrinsic_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    interior_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample uniformly without replacement from intrinsic minus interior bounds."""

    intrinsic_size = _bounds_size(intrinsic_bounds)
    exterior_size = intrinsic_size - _bounds_size(interior_bounds)
    if exterior_size <= 0:
        return tuple(np.empty(0, dtype=np.int64) for _ in range(3))  # type: ignore[return-value]
    requested = min(sample_count, exterior_size)
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    while len(chosen) < requested:
        batch_size = max(256, 2 * (requested - len(chosen)))
        candidates = rng.integers(0, intrinsic_size, size=batch_size)
        origins = _linear_to_origins(intrinsic_bounds, candidates)
        for value, inside in zip(candidates, _inside_bounds(origins, interior_bounds)):
            if not inside:
                chosen.add(int(value))
                if len(chosen) == requested:
                    break
    return _linear_to_origins(intrinsic_bounds, np.fromiter(sorted(chosen), dtype=np.int64))


def _measurement(
    B: np.ndarray,
    q_fields: Mapping[str, QField],
    origins: tuple[np.ndarray, np.ndarray, np.ndarray],
    displacement: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return accepted counts, powered sums, and spatial-block powered sums."""

    count = origins[0].size
    sums = np.zeros((len(Q_NAMES), len(DIRECTIONS), len(P_VALUES)))
    counts = np.zeros_like(sums)
    block_sums = np.zeros((512, *sums.shape))
    if not count:
        return counts, sums, block_sums
    di, dj, dk = (int(value) for value in displacement)
    r_vector = cube_offset_to_vector(displacement)
    for begin in range(0, count, PAIR_BATCH_SIZE):
        stop = min(count, begin + PAIR_BATCH_SIZE)
        k0, j0, i0 = (values[begin:stop] for values in origins)
        k1, j1, i1 = k0 + dk, j0 + dj, i0 + di
        block_ids = _origin_block_ids(
            (k0, j0, i0), tuple(int(value) for value in B.shape[1:]), BLOCK_SHAPE_KJI, displacement, 2
        )
        assert block_ids is not None
        B0 = B[:, k0, j0, i0].T
        B1 = B[:, k1, j1, i1].T
        B_local = 0.5 * (B0 + B1)
        B_mag = np.linalg.norm(B_local, axis=1)
        valid_B = np.all(np.isfinite(B0), axis=1) & np.all(np.isfinite(B1), axis=1)
        valid_parallel = valid_B & np.all(np.isfinite(B_local), axis=1) & (B_mag > 1.0e-12)
        e_parallel = np.full_like(B_local, np.nan)
        e_parallel[valid_parallel] = B_local[valid_parallel] / B_mag[valid_parallel, None]
        r_rows = np.broadcast_to(r_vector, B_local.shape)
        theta = folded_angle(r_rows, e_parallel)
        r_perp = r_rows - np.einsum("ij,ij->i", r_rows, e_parallel)[:, None] * e_parallel
        r_perp_mag = np.linalg.norm(r_perp, axis=1)
        for q_index, q_field in enumerate(q_fields.values()):
            q0 = q_field.values[:, k0, j0, i0].T
            q1 = q_field.values[:, k1, j1, i1].T
            valid = (
                valid_parallel
                & q_field.valid[k0, j0, i0]
                & q_field.valid[k1, j1, i1]
                & np.all(np.isfinite(q0), axis=1)
                & np.all(np.isfinite(q1), axis=1)
            )
            delta_q = q1 - q0
            delta_q_perp = delta_q - np.einsum("ij,ij->i", delta_q, e_parallel)[:, None] * e_parallel
            q_perp_mag = np.linalg.norm(delta_q_perp, axis=1)
            valid_q_perp = valid & np.isfinite(q_perp_mag) & (q_perp_mag > 1.0e-12)
            valid_r_perp = valid_q_perp & np.isfinite(r_perp_mag) & (r_perp_mag > 1.0e-12)
            e_xi = np.full_like(delta_q_perp, np.nan)
            e_xi[valid_r_perp] = delta_q_perp[valid_r_perp] / q_perp_mag[valid_r_perp, None]
            phi = folded_angle(r_perp, e_xi)
            masks = (
                valid & (theta <= np.deg2rad(15.0)),
                valid_r_perp & (theta >= np.deg2rad(75.0)) & (phi <= np.deg2rad(15.0)),
                valid_r_perp & (theta >= np.deg2rad(75.0)) & (phi >= np.deg2rad(75.0)),
            )
            for direction_index, mask in enumerate(masks):
                selected_blocks = block_ids[mask]
                values = q_perp_mag[mask]
                for p_index, p_value in enumerate(P_VALUES):
                    powered = np.power(values, p_value)
                    counts[q_index, direction_index, p_index] += powered.size
                    sums[q_index, direction_index, p_index] += powered.sum()
                    block_sums[:, q_index, direction_index, p_index] += np.bincount(
                        selected_blocks, weights=powered, minlength=512
                    )
    return counts, sums, block_sums


def _selected_offsets() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    offsets, edges, manifest = dense_displacement_manifest(
        stencil_width=2,
        ell_max=ELL_MAX,
        bin_count=BIN_COUNT,
        directions_per_bin=DIRECTIONS_PER_BIN,
    )
    ell = np.linalg.norm(offsets.astype(float), axis=1)
    science = offsets[ell >= SCIENCE_SCALE_MINIMUM]
    selected = inherited._stratified_offset_subset(science, edges, maximum=SELECTED_OFFSET_MAXIMUM)
    return offsets, edges, selected, manifest


def _shell_bounds(
    cube_shape: tuple[int, int, int], offsets: np.ndarray, edges: np.ndarray
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None, ...]:
    ell = np.linalg.norm(offsets.astype(float), axis=1)
    bins = np.searchsorted(edges, ell, side="right") - 1
    output = []
    for index in range(len(edges) - 1):
        members = offsets[bins == index]
        output.append(nested_core_bounds_kji(cube_shape, members, 2) if len(members) else None)
    return tuple(output)


def _weighted_add(
    destination_counts: np.ndarray,
    destination_sums: np.ndarray,
    destination_blocks: np.ndarray,
    measured: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    population: int,
    sampled: int,
    ell_index: int,
) -> None:
    if sampled <= 0 or population <= 0:
        return
    counts, sums, blocks = measured
    weight = population / sampled
    destination_counts[..., ell_index] += weight * counts
    destination_sums[..., ell_index] += weight * sums
    destination_blocks[..., ell_index] += weight * blocks


def _concentration(block_sums: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(block_sums, axis=0)[::-1]
    total = ordered.sum(axis=0)

    def fraction(count: int) -> np.ndarray:
        return np.divide(
            ordered[:count].sum(axis=0),
            total,
            out=np.full_like(total, np.nan),
            where=total > 0.0,
        )

    return fraction(1), fraction(5), fraction(10)


def _run_scenario(
    B: np.ndarray,
    q_fields: Mapping[str, QField],
    offsets: np.ndarray,
    edges: np.ndarray,
    selected: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    shape = (len(Q_NAMES), len(DIRECTIONS), len(P_VALUES), len(edges) - 1)
    components = ("interior", "exterior", "recomposed", "direct_intrinsic")
    counts = {name: np.zeros(shape) for name in components}
    sums = {name: np.zeros(shape) for name in components}
    blocks = {name: np.zeros((512, *shape)) for name in components}
    sampled = {name: 0 for name in components}
    cube_shape = tuple(int(value) for value in B.shape[1:])
    shell_bounds = _shell_bounds(cube_shape, offsets, edges)
    for displacement in selected:
        ell = float(np.linalg.norm(displacement.astype(float)))
        ell_index = int(np.searchsorted(edges, ell, side="right") - 1)
        interior_bounds = shell_bounds[ell_index]
        if interior_bounds is None:
            continue
        intrinsic_bounds = valid_origin_bounds_kji(cube_shape, displacement, 2)
        interior_origins = _origin_arrays(interior_bounds, sample_count, _shell_seed(seed, ell_index))
        exterior_origins = _sample_exterior_origins(
            intrinsic_bounds,
            interior_bounds,
            sample_count,
            _offset_seed(seed + 1000003, displacement),
        )
        direct_origins = _origin_arrays(intrinsic_bounds, sample_count, _offset_seed(seed, displacement))
        interior_population = _bounds_size(interior_bounds)
        intrinsic_population = _bounds_size(intrinsic_bounds)
        exterior_population = intrinsic_population - interior_population
        measurements = {
            "interior": (_measurement(B, q_fields, interior_origins, displacement), interior_population, len(interior_origins[0])),
            "exterior": (_measurement(B, q_fields, exterior_origins, displacement), exterior_population, len(exterior_origins[0])),
            "direct_intrinsic": (_measurement(B, q_fields, direct_origins, displacement), intrinsic_population, len(direct_origins[0])),
        }
        for name, (measurement, population, origin_count) in measurements.items():
            _weighted_add(
                counts[name], sums[name], blocks[name], measurement,
                population=population, sampled=origin_count, ell_index=ell_index,
            )
            sampled[name] += origin_count
        counts["recomposed"][..., ell_index] = counts["interior"][..., ell_index] + counts["exterior"][..., ell_index]
        sums["recomposed"][..., ell_index] = sums["interior"][..., ell_index] + sums["exterior"][..., ell_index]
        blocks["recomposed"][..., ell_index] = blocks["interior"][..., ell_index] + blocks["exterior"][..., ell_index]
    payload: dict[str, np.ndarray] = {
        "ell_bin_edges": edges,
        "selected_displacements_ijk": selected,
        "sample_count": np.asarray(sample_count),
        "seed": np.asarray(seed),
    }
    for name in components:
        payload[f"{name}_estimated_accepted_counts"] = counts[name]
        payload[f"{name}_estimated_sums"] = sums[name]
        payload[f"{name}_moments"] = np.divide(
            sums[name], counts[name], out=np.full_like(sums[name], np.nan), where=counts[name] > 0.0
        )
        payload[f"{name}_sampled_origins"] = np.asarray(sampled[name])
        one, five, ten = _concentration(blocks[name])
        payload[f"{name}_largest_block_fraction"] = one
        payload[f"{name}_largest_5_blocks_fraction"] = five
        payload[f"{name}_largest_10_blocks_fraction"] = ten
    payload["recomposed_over_direct_ratio"] = np.divide(
        payload["recomposed_moments"],
        payload["direct_intrinsic_moments"],
        out=np.full_like(payload["recomposed_moments"], np.nan),
        where=np.isfinite(payload["direct_intrinsic_moments"]) & (payload["direct_intrinsic_moments"] > 0.0),
    )
    return payload


def work(phase2_root: Path, representative_release_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    input_identity = _input_identity(phase2_root, representative_release_root)
    procid = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    offsets, edges, selected, manifest = _selected_offsets()
    rows = []
    for cube_id in CUBE_IDS[procid::ntasks]:
        cube = prepare_cube_for_parallel(phase3._load_cube(phase2_root, cube_id))
        B, q_fields, _ = build_cube_q_variants(cube, q_names=Q_NAMES)
        for sample_count in SAMPLE_COUNTS:
            for seed in SEEDS:
                started = time.perf_counter()
                payload = _run_scenario(
                    B, q_fields, offsets, edges, selected, sample_count=sample_count, seed=seed
                )
                artifact = output_root / "scenarios" / f"{cube_id}_samples_{sample_count}_seed_{seed}.npz"
                _atomic_write_npz(artifact, **payload)
                rows.append(
                    {
                        "cube_id": cube_id,
                        "sample_count": sample_count,
                        "seed": seed,
                        "artifact_relative_path": str(artifact.relative_to(output_root)),
                        "artifact_sha256": file_sha256(artifact),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
    record = output_root / "work_records" / f"task_{procid:04d}.json"
    _atomic_write_json(
        record,
        {
            "schema_version": SCHEMA_VERSION,
            "procid": procid,
            "ntasks": ntasks,
            "rows": rows,
            "source_version": _source_version(),
            "input_identity": input_identity,
            "manifest_sha256": manifest["manifest_sha256"],
            "selected_offsets_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        },
    )
    return {"procid": procid, "published_scenarios": len(rows), "record": str(record)}


def summarize(phase2_root: Path, representative_release_root: Path, output_root: Path) -> dict[str, Any]:
    offsets, _, selected, manifest = _selected_offsets()
    input_identity = _input_identity(phase2_root, representative_release_root)
    records = sorted((output_root / "work_records").glob("task_*.json"))
    rows = []
    for record in records:
        payload = json.loads(record.read_text())
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("source_version") != _source_version()
            or payload.get("input_identity") != input_identity
            or payload.get("manifest_sha256") != manifest["manifest_sha256"]
            or payload.get("selected_offsets_sha256") != hashlib.sha256(selected.tobytes()).hexdigest()
        ):
            raise RuntimeError(f"invalid tail-diagnostic work record: {record}")
        rows.extend(payload["rows"])
    expected = {(cube_id, samples, seed) for cube_id in CUBE_IDS for samples in SAMPLE_COUNTS for seed in SEEDS}
    observed = {(row["cube_id"], row["sample_count"], row["seed"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("tail-diagnostic scenario inventory is incomplete or duplicated")
    for row in rows:
        path = output_root / row["artifact_relative_path"]
        if row["artifact_sha256"] != file_sha256(path):
            raise RuntimeError(f"tail-diagnostic scenario changed after publication: {path}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "phase": "phase4_batch_b_bounded_matched_support_tail_diagnostic",
        "cube_ids": CUBE_IDS,
        "q_names": Q_NAMES,
        "directions": DIRECTIONS,
        "p_values": P_VALUES,
        "sample_counts": SAMPLE_COUNTS,
        "seeds": SEEDS,
        "science_scale_minimum_cells": SCIENCE_SCALE_MINIMUM,
        "selected_offset_maximum": SELECTED_OFFSET_MAXIMUM,
        "full_displacement_count": len(offsets),
        "selected_displacement_count": len(selected),
        "displacement_manifest_sha256": manifest["manifest_sha256"],
        "selected_offsets_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "scenario_rows": rows,
        "source_version": _source_version(),
        "input_identity": input_identity,
        "interpretation": {
            "interior": "shared shell-local interior sampled with the production shell seed",
            "exterior": "intrinsic-valid origins outside the shared shell-local interior",
            "recomposed": "population-weighted interior plus exterior estimate",
            "direct_intrinsic": "independent direct intrinsic-valid estimate sampled with the production offset seed",
        },
    }
    summary_path = output_root / SUMMARY_FILENAME
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(
        output_root / MARKER_FILENAME,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "summary_sha256": file_sha256(summary_path),
            "implementation_sha256": summary["source_version"]["implementation_sha256"],
        },
    )
    return verify(phase2_root, representative_release_root, output_root)


def verify(phase2_root: Path, representative_release_root: Path, output_root: Path) -> dict[str, Any]:
    summary_path = output_root / SUMMARY_FILENAME
    marker = json.loads((output_root / MARKER_FILENAME).read_text())
    summary = json.loads(summary_path.read_text())
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or marker.get("implementation_sha256") != _source_version()["implementation_sha256"]
        or summary.get("source_version") != _source_version()
        or summary.get("input_identity") != _input_identity(phase2_root, representative_release_root)
        or summary.get("phase") != "phase4_batch_b_bounded_matched_support_tail_diagnostic"
    ):
        raise RuntimeError("invalid or stale Phase 4 Batch B tail diagnostic")
    for row in summary["scenario_rows"]:
        path = output_root / row["artifact_relative_path"]
        if row["artifact_sha256"] != file_sha256(path):
            raise RuntimeError(f"tail-diagnostic artifact changed after publication: {path}")
    return {"status": "passed", "scenario_count": len(summary["scenario_rows"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("work", "summarize", "verify"))
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--representative-release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    actions = {
        "work": lambda: work(args.phase2_root, args.representative_release_root, args.output_root),
        "summarize": lambda: summarize(args.phase2_root, args.representative_release_root, args.output_root),
        "verify": lambda: verify(args.phase2_root, args.representative_release_root, args.output_root),
    }
    print(json.dumps(_json_builtin(actions[args.action]()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
