#!/usr/bin/env python3
"""Plan, execute, reduce, and verify the bounded Phase 3a validation campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3.run_phase3_sampler import (
    BENCHMARK_CUBE_IDS,
    DEFAULT_PHASE2_ROOT,
    SCHEMA_VERSION as PHASE3_SCHEMA_VERSION,
    _load_cube,
    _phase2_source_identity,
)
from sfunctor.analysis.phase3a import (
    block_bootstrap_moment_uncertainty,
    block_jackknife_moment_uncertainty,
    load_finite_domain_partial_npz,
    local_log_slope,
    reduce_finite_domain_shards,
    save_finite_domain_partial_npz,
)
from sfunctor.core.finite_domain import (
    FiniteDomainConfig,
    compute_finite_domain_structure_functions,
)
from sfunctor.core.phase3a import (
    compute_finite_domain_structure_functions_parallel,
    dense_displacement_manifest,
    prepare_cube_for_parallel,
)

SCHEMA_VERSION = 1
PRODUCTION_SEED = 20260530
BOOTSTRAP_SEED = 20260531
BOOTSTRAP_N_RESAMPLES = 200
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
MINIMUM_LOCAL_SLOPE_EFFECTIVE_BLOCKS = 8.0
MINIMUM_LOCAL_SLOPE_VALID_BOOTSTRAP_FRACTION = 0.9
PARALLEL_EQUIVALENCE_RTOL = 5.0e-13
PRODUCTION_SAMPLE_COUNT = 2048
PRODUCTION_PAIR_BATCH_SIZE = 1024
PRODUCTION_BLOCK_SHAPE_KJI = (80, 80, 80)
OFFSETS_PER_SHARD = 480
Q_NAMES = ("B", "u")
SUPPORT_MODES = ("shell_local", "all_valid_origins")
DIAGNOSTIC_SUPPORT_MODES = (*SUPPORT_MODES, "nested_core")
STENCIL_SPECS = {
    2: {"label": "2-point", "ell_max": 320, "bin_count": 64, "directions_per_bin": 24},
    3: {"label": "3-point", "ell_max": 160, "bin_count": 64, "directions_per_bin": 24},
    5: {"label": "5-point", "ell_max": 80, "bin_count": 64, "directions_per_bin": 24},
}
PHASE3_ANCHOR_ROOT = Path(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase3_sampler/"
    "smoke_remediated_verified_release_primary_20260531T001131Z"
)


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


def _publish_json_diagnostic(
    output_root: Path,
    filename: str,
    marker_filename: str,
    payload: Mapping[str, Any],
) -> None:
    output_path = output_root / filename
    _atomic_write_json(output_path, payload)
    _atomic_write_json(
        output_root / marker_filename,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "payload_relative_path": filename,
            "payload_sha256": file_sha256(output_path),
            "implementation_sha256": payload["source_version"]["implementation_sha256"],
            "published_unix_seconds": time.time(),
        },
    )


def _verify_json_diagnostic(
    output_root: Path,
    filename: str,
    marker_filename: str,
    *,
    phase2_root: Path | None = None,
) -> dict[str, Any]:
    """Reject stale or source-mixed bounded diagnostic publications."""

    payload_path = output_root / filename
    marker = json.loads((output_root / marker_filename).read_text())
    payload = json.loads(payload_path.read_text())
    current_source = _source_version()
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "complete"
        or marker.get("payload_relative_path") != filename
        or marker.get("payload_sha256") != file_sha256(payload_path)
        or marker.get("implementation_sha256")
        != current_source["implementation_sha256"]
        or payload.get("source_version", {}).get("implementation_sha256")
        != current_source["implementation_sha256"]
        or payload.get("operational_status") != "complete"
    ):
        raise RuntimeError(f"invalid or stale Phase 3a diagnostic: {output_root}")
    if phase2_root is not None:
        cube_id = payload.get("cube_id")
        if not isinstance(cube_id, str) or payload.get("phase2_source") != _phase2_source_identity(
            phase2_root, cube_id, verify_arrays=True
        ):
            raise RuntimeError(f"Phase 2 source changed after diagnostic publication: {output_root}")
    for row in payload.get("rows", ()):
        artifact_relative_path = row.get("artifact_relative_path")
        artifact_sha256 = row.get("artifact_sha256")
        if (artifact_relative_path is None) != (artifact_sha256 is None):
            raise RuntimeError(f"incomplete diagnostic artifact binding: {output_root}")
        if artifact_relative_path is None:
            continue
        artifact_path = output_root / artifact_relative_path
        if (
            artifact_path.resolve().parent != (output_root / "scenarios").resolve()
            or artifact_sha256 != file_sha256(artifact_path)
        ):
            raise RuntimeError(f"invalid diagnostic artifact binding: {artifact_path}")
    for binding in payload.get("artifact_bindings", ()):
        relative_path = binding.get("relative_path")
        sha256 = binding.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(sha256, str):
            raise RuntimeError(f"incomplete supplemental artifact binding: {output_root}")
        artifact_path = output_root / relative_path
        if (
            output_root.resolve() not in artifact_path.resolve().parents
            or sha256 != file_sha256(artifact_path)
        ):
            raise RuntimeError(f"invalid supplemental artifact binding: {artifact_path}")
    if "resource_record_bindings" in payload:
        _verify_multinode_resource_record_bindings(output_root, payload)
    return payload


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_builtin(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "sfunctor" / "core" / "directional.py",
        root / "sfunctor" / "core" / "finite_domain.py",
        root / "sfunctor" / "core" / "phase3a.py",
        root / "sfunctor" / "analysis" / "finite_domain.py",
        root / "sfunctor" / "analysis" / "phase3a.py",
        root / "sfunctor" / "reference_3d.py",
        root / "scripts" / "phase3" / "run_phase3_sampler.py",
        root / "scripts" / "phase1" / "cbin_tools.py",
        root / "job_scripts" / "phase3a" / "run_phase3a_sampler_andes.sh",
    )
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in paths}
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"), cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {
        "commit": commit,
        "dirty": dirty,
        "implementation_source_hashes": hashes,
        "implementation_sha256": _mapping_sha256(hashes),
    }


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _tree_size_accounting(root: Path) -> dict[str, int]:
    """Return logical and allocated regular-file bytes below one root."""

    logical_bytes = allocated_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        logical_bytes += stat.st_size
        allocated_bytes += stat.st_blocks * 512
    return {
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
    }


def _cube_memory_footprint(cube: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Return mapped-input and parent-stacked-vector byte footprints."""

    return {
        "input_mmap_bytes": sum(
            np.asarray(values).nbytes for name, values in cube.items() if not name.startswith("_")
        ),
        "parent_stacked_vector_bytes": sum(
            np.asarray(values).nbytes for name, values in cube.items() if name.startswith("_")
        ),
    }


def _publish_work_resource_record(
    output_root: Path,
    *,
    action: str,
    workers: int,
    started: float,
    rows: list[dict[str, Any]],
    cube_footprints: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Persist task-local execution accounting without changing science markers."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "action": action,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_procid": int(os.environ.get("SLURM_PROCID", "0")),
        "slurm_ntasks": int(os.environ.get("SLURM_NTASKS", "1")),
        "workers": workers,
        "action_wall_seconds": time.perf_counter() - started,
        "published_count": sum(not row["reused"] for row in rows),
        "reused_count": sum(row["reused"] for row in rows),
        "cube_footprints": cube_footprints,
        "rows": rows,
        "parent_process_peak_rss_kib": _peak_rss_kib(),
        "source_version": _source_version(),
    }
    root = output_root / "work_resource_records"
    path = root / (
        f"{action}_job_{payload['slurm_job_id']}_proc_{payload['slurm_procid']:04d}_"
        f"{time.time_ns()}.json"
    )
    _atomic_write_json(path, payload)
    return {
        "resource_record_relative_path": str(path.relative_to(output_root)),
        "resource_record_sha256": file_sha256(path),
        **{name: payload[name] for name in ("published_count", "reused_count", "action_wall_seconds")},
    }


def _manifest_paths(output_root: Path, stencil_width: int) -> tuple[Path, Path]:
    root = output_root / "manifests" / "displacements"
    return root / f"stencil_{stencil_width}point.json", root / f"stencil_{stencil_width}point.npz"


def _group_id(cube_id: str, stencil_width: int, support_mode: str) -> str:
    return f"{cube_id}/stencil_{stencil_width}point/{support_mode}"


def _shard_id(group_id: str, shard_index: int) -> str:
    return f"{group_id}/shard_{shard_index:04d}"


def _shard_paths(output_root: Path, shard_id: str) -> tuple[Path, Path, Path]:
    root = output_root / "shards" / shard_id
    return root, root / "partial.npz", root / "COMPLETE.json"


def _reduction_paths(output_root: Path, group_id: str) -> tuple[Path, Path, Path, Path]:
    root = output_root / "reductions" / group_id
    return root, root / "result.npz", root / "uncertainty.npz", root / "COMPLETE.json"


def _load_displacement_manifest(output_root: Path, stencil_width: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    json_path, npz_path = _manifest_paths(output_root, stencil_width)
    metadata = json.loads(json_path.read_text())
    if metadata.get("manifest_sha256") != _mapping_sha256(
        {key: value for key, value in metadata.items() if key != "manifest_sha256"}
    ):
        raise RuntimeError(f"stale displacement manifest: {json_path}")
    if metadata.get("npz_sha256") != file_sha256(npz_path):
        raise RuntimeError(f"displacement NPZ checksum mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        displacements = payload["displacements_ijk"].copy()
        ell_bin_edges = payload["ell_bin_edges"].copy()
        offset_ids = payload["offset_ids"].copy()
    if not np.array_equal(offset_ids, np.arange(len(displacements), dtype=np.int64)):
        raise RuntimeError(f"invalid stable offset IDs: {npz_path}")
    if metadata.get("offsets_sha256") != hashlib.sha256(displacements.tobytes()).hexdigest():
        raise RuntimeError(f"displacement checksum mismatch: {npz_path}")
    return metadata, displacements, ell_bin_edges


def _campaign_configuration() -> dict[str, Any]:
    return {
        "q_names": Q_NAMES,
        "p_values": (2.0,),
        "support_modes": SUPPORT_MODES,
        "production_seed": PRODUCTION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sample_count_per_displacement": PRODUCTION_SAMPLE_COUNT,
        "pair_batch_size": PRODUCTION_PAIR_BATCH_SIZE,
        "block_shape_kji": PRODUCTION_BLOCK_SHAPE_KJI,
        "block_assignment": "stencil_midpoint",
        "offsets_per_shard": OFFSETS_PER_SHARD,
        "stencils": STENCIL_SPECS,
    }


def _planned_shards(output_root: Path) -> list[dict[str, Any]]:
    """Return the canonical exact-once shard inventory for frozen manifests."""

    shards = []
    for cube_id in BENCHMARK_CUBE_IDS:
        for stencil_width in sorted(STENCIL_SPECS):
            _, displacements, _ = _load_displacement_manifest(output_root, stencil_width)
            for support_mode in SUPPORT_MODES:
                group_id = _group_id(cube_id, stencil_width, support_mode)
                for shard_index, start in enumerate(range(0, len(displacements), OFFSETS_PER_SHARD)):
                    stop = min(len(displacements), start + OFFSETS_PER_SHARD)
                    shards.append(
                        {
                            "group_id": group_id,
                            "shard_id": _shard_id(group_id, shard_index),
                            "cube_id": cube_id,
                            "stencil_width": stencil_width,
                            "support_mode": support_mode,
                            "shard_index": shard_index,
                            "offset_start": start,
                            "offset_stop": stop,
                        }
                    )
    return shards


def plan(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    """Freeze input, displacement, and shard manifests for the bounded campaign."""

    marker_path = output_root / "PLAN_COMPLETE.json"
    if marker_path.exists():
        return _verify_plan(phase2_root, output_root, verify_arrays=False)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to plan into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_version = _source_version()
    input_sources = {
        cube_id: _phase2_source_identity(phase2_root, cube_id, verify_arrays=True)
        for cube_id in BENCHMARK_CUBE_IDS
    }
    displacement_rows = {}
    for stencil_width, spec in STENCIL_SPECS.items():
        displacements, edges, metadata = dense_displacement_manifest(
            stencil_width=stencil_width,
            ell_max=spec["ell_max"],
            bin_count=spec["bin_count"],
            directions_per_bin=spec["directions_per_bin"],
        )
        json_path, npz_path = _manifest_paths(output_root, stencil_width)
        _atomic_write_npz(
            npz_path,
            displacements_ijk=displacements,
            ell_bin_edges=edges,
            offset_ids=np.arange(len(displacements), dtype=np.int64),
        )
        metadata = {**metadata, "npz_sha256": file_sha256(npz_path)}
        metadata["manifest_sha256"] = _mapping_sha256(
            {key: value for key, value in metadata.items() if key != "manifest_sha256"}
        )
        _atomic_write_json(json_path, metadata)
        displacement_rows[str(stencil_width)] = {
            "json_relative_path": str(json_path.relative_to(output_root)),
            "json_sha256": file_sha256(json_path),
            "npz_relative_path": str(npz_path.relative_to(output_root)),
            "npz_sha256": file_sha256(npz_path),
            "manifest_sha256": metadata["manifest_sha256"],
            "offset_count": len(displacements),
        }
    shards = _planned_shards(output_root)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "phase2_root": str(phase2_root),
        "source_version": source_version,
        "configuration": _campaign_configuration(),
        "configuration_sha256": _mapping_sha256(_campaign_configuration()),
        "phase2_sources": input_sources,
        "displacement_manifests": displacement_rows,
        "shard_count": len(shards),
    }
    _atomic_write_json(output_root / "manifests" / "campaign.json", campaign)
    _atomic_write_json(output_root / "manifests" / "shards.json", {"shards": shards})
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "campaign_sha256": file_sha256(output_root / "manifests" / "campaign.json"),
        "shards_sha256": file_sha256(output_root / "manifests" / "shards.json"),
        "implementation_sha256": source_version["implementation_sha256"],
        "published_unix_seconds": time.time(),
    }
    _atomic_write_json(marker_path, marker)
    return _verify_plan(phase2_root, output_root, verify_arrays=False)


def _verify_plan(phase2_root: Path, output_root: Path, *, verify_arrays: bool) -> dict[str, Any]:
    marker = json.loads((output_root / "PLAN_COMPLETE.json").read_text())
    campaign_path = output_root / "manifests" / "campaign.json"
    shards_path = output_root / "manifests" / "shards.json"
    campaign = json.loads(campaign_path.read_text())
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "passed"
        or marker.get("campaign_sha256") != file_sha256(campaign_path)
        or marker.get("shards_sha256") != file_sha256(shards_path)
        or campaign.get("configuration_sha256") != _mapping_sha256(campaign.get("configuration", {}))
    ):
        raise RuntimeError("invalid Phase 3a plan marker")
    current_source = _source_version()
    if marker.get("implementation_sha256") != current_source["implementation_sha256"]:
        raise RuntimeError("Phase 3a plan was produced by a different implementation")
    if campaign.get("source_version", {}).get("implementation_sha256") != marker["implementation_sha256"]:
        raise RuntimeError("Phase 3a campaign source binding mismatch")
    for cube_id in BENCHMARK_CUBE_IDS:
        if campaign["phase2_sources"][cube_id] != _phase2_source_identity(
            phase2_root, cube_id, verify_arrays=verify_arrays
        ):
            raise RuntimeError(f"Phase 2 source changed after planning: {cube_id}")
    for stencil_width in STENCIL_SPECS:
        metadata, _, _ = _load_displacement_manifest(output_root, stencil_width)
        row = campaign["displacement_manifests"][str(stencil_width)]
        json_path, npz_path = _manifest_paths(output_root, stencil_width)
        if (
            row.get("json_relative_path") != str(json_path.relative_to(output_root))
            or row.get("json_sha256") != file_sha256(json_path)
            or row.get("npz_relative_path") != str(npz_path.relative_to(output_root))
            or row.get("npz_sha256") != file_sha256(npz_path)
            or row.get("manifest_sha256") != metadata["manifest_sha256"]
            or row.get("offset_count") != metadata["realized_offset_count"]
        ):
            raise RuntimeError(f"campaign lost displacement-manifest binding: stencil {stencil_width}")
    shards = json.loads(shards_path.read_text())["shards"]
    if len(shards) != campaign["shard_count"]:
        raise RuntimeError("Phase 3a shard inventory count mismatch")
    if len({row["shard_id"] for row in shards}) != len(shards):
        raise RuntimeError("Phase 3a shard inventory contains duplicates")
    if shards != _planned_shards(output_root):
        raise RuntimeError("Phase 3a shard inventory is not the canonical exact-once coverage")
    return campaign


def _result_configuration(row: Mapping[str, Any], edges: np.ndarray) -> FiniteDomainConfig:
    return FiniteDomainConfig(
        ell_bin_edges=edges,
        p_values=(2.0,),
        pair_mode=str(row["support_mode"]),
        sample_count=PRODUCTION_SAMPLE_COUNT,
        pair_batch_size=PRODUCTION_PAIR_BATCH_SIZE,
        seed=PRODUCTION_SEED,
        stencil_width=int(row["stencil_width"]),
        block_shape_kji=PRODUCTION_BLOCK_SHAPE_KJI,
    )


def _sampling_schedule_sha256(row: Mapping[str, Any]) -> str:
    return _mapping_sha256(
        {
            "group_id": row["group_id"],
            "shard_id": row["shard_id"],
            "offset_start": row["offset_start"],
            "offset_stop": row["offset_stop"],
            "sample_count": PRODUCTION_SAMPLE_COUNT,
            "pair_batch_size": PRODUCTION_PAIR_BATCH_SIZE,
            "seed": PRODUCTION_SEED,
            "block_shape_kji": PRODUCTION_BLOCK_SHAPE_KJI,
            "block_assignment": "stencil_midpoint",
        }
    )


def _verify_shard(output_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    root, partial_path, marker_path = _shard_paths(output_root, str(row["shard_id"]))
    marker = json.loads(marker_path.read_text())
    campaign = json.loads((output_root / "manifests" / "campaign.json").read_text())
    metadata, displacements, _ = _load_displacement_manifest(output_root, int(row["stencil_width"]))
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "passed"
        or marker.get("shard") != dict(row)
        or marker.get("partial_sha256") != file_sha256(partial_path)
        or marker.get("implementation_sha256")
        != campaign["source_version"]["implementation_sha256"]
        or marker.get("displacement_manifest_sha256") != metadata["manifest_sha256"]
        or marker.get("support_displacements_sha256") != metadata["offsets_sha256"]
        or marker.get("phase2_source") != campaign["phase2_sources"][row["cube_id"]]
        or marker.get("sampling_schedule_sha256") != _sampling_schedule_sha256(row)
        or any(
            not isinstance(marker.get(name), int) or marker[name] < 0
            for name in (
                "staging_logical_bytes_before_marker",
                "staging_allocated_bytes_before_marker",
            )
        )
    ):
        raise RuntimeError(f"invalid Phase 3a shard marker: {root}")
    result = load_finite_domain_partial_npz(partial_path)
    expected = displacements[int(row["offset_start"]) : int(row["offset_stop"])]
    order = np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))
    expected = expected[order]
    if not np.array_equal(result.displacements_ijk, expected):
        raise RuntimeError(f"Phase 3a shard has wrong offsets: {root}")
    if result.support_displacements_sha256 != metadata["offsets_sha256"]:
        raise RuntimeError(f"Phase 3a shard has wrong frozen support census: {root}")
    if (
        result.stencil_width != int(row["stencil_width"])
        or result.pair_mode != row["support_mode"]
        or result.sample_count != PRODUCTION_SAMPLE_COUNT
        or result.pair_batch_size != PRODUCTION_PAIR_BATCH_SIZE
        or result.seed != PRODUCTION_SEED
        or result.block_shape_kji != PRODUCTION_BLOCK_SHAPE_KJI
        or result.block_assignment != "stencil_midpoint"
    ):
        raise RuntimeError(f"Phase 3a shard has wrong sampling configuration: {root}")
    return marker


def work(phase2_root: Path, output_root: Path, *, workers: int) -> dict[str, Any]:
    """Compute assigned fixed shards and atomically publish validated partials."""

    work_started = time.perf_counter()
    campaign = _verify_plan(phase2_root, output_root, verify_arrays=False)
    rows = json.loads((output_root / "manifests" / "shards.json").read_text())["shards"]
    procid = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    assigned = [row for index, row in enumerate(rows) if index % ntasks == procid]
    cache_cube_id = None
    cube = None
    published = reused = 0
    resource_rows: list[dict[str, Any]] = []
    cube_footprints: dict[str, dict[str, int]] = {}
    for row in assigned:
        root, _, marker_path = _shard_paths(output_root, row["shard_id"])
        if marker_path.exists():
            reuse_started = time.perf_counter()
            _verify_shard(output_root, row)
            resource_rows.append(
                {
                    "shard_id": row["shard_id"],
                    "cube_id": row["cube_id"],
                    "stencil_width": row["stencil_width"],
                    "support_mode": row["support_mode"],
                    "reused": True,
                    "reuse_verification_wall_seconds": time.perf_counter() - reuse_started,
                }
            )
            reused += 1
            continue
        if root.exists():
            raise RuntimeError(f"incomplete authoritative shard directory exists: {root}")
        metadata, displacements, edges = _load_displacement_manifest(
            output_root, int(row["stencil_width"])
        )
        if cache_cube_id != row["cube_id"]:
            if campaign["phase2_sources"][row["cube_id"]] != _phase2_source_identity(
                phase2_root, row["cube_id"], verify_arrays=True
            ):
                raise RuntimeError(f"Phase 2 arrays changed before work: {row['cube_id']}")
            cube = prepare_cube_for_parallel(_load_cube(phase2_root, row["cube_id"]))
            cube_footprints[row["cube_id"]] = _cube_memory_footprint(cube)
            cache_cube_id = row["cube_id"]
        assert cube is not None
        start, stop = int(row["offset_start"]), int(row["offset_stop"])
        shard_offsets = displacements[start:stop]
        compute_started = time.perf_counter()
        partials = compute_finite_domain_structure_functions_parallel(
            cube,
            shard_offsets,
            config=_result_configuration(row, edges),
            q_names=Q_NAMES,
            worker_count=workers,
            support_displacements_ijk=displacements,
        )
        result = reduce_finite_domain_shards(partials)
        compute_wall_seconds = time.perf_counter() - compute_started
        if result.support_displacements_sha256 != metadata["offsets_sha256"]:
            raise RuntimeError("computed shard lost frozen support census binding")
        attempt = output_root / "attempts" / f"proc_{procid}" / f"{row['shard_id'].replace('/', '__')}_{time.time_ns()}"
        attempt.mkdir(parents=True)
        partial_path = attempt / "partial.npz"
        serialization_started = time.perf_counter()
        save_finite_domain_partial_npz(partial_path, result)
        serialization_wall_seconds = time.perf_counter() - serialization_started
        staging_bytes = _tree_size_accounting(attempt)
        _atomic_write_json(
            attempt / "COMPLETE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "shard": row,
                "implementation_sha256": campaign["source_version"]["implementation_sha256"],
                "displacement_manifest_sha256": metadata["manifest_sha256"],
                "support_displacements_sha256": metadata["offsets_sha256"],
                "phase2_source": campaign["phase2_sources"][row["cube_id"]],
                "sampling_schedule_sha256": _sampling_schedule_sha256(row),
                "partial_sha256": file_sha256(partial_path),
                "staging_logical_bytes_before_marker": staging_bytes["logical_bytes"],
                "staging_allocated_bytes_before_marker": staging_bytes["allocated_bytes"],
                "published_unix_seconds": time.time(),
            },
        )
        publish_started = time.perf_counter()
        root.parent.mkdir(parents=True, exist_ok=True)
        attempt.replace(root)
        _verify_shard(output_root, row)
        publish_verification_wall_seconds = time.perf_counter() - publish_started
        resource_rows.append(
            {
                "shard_id": row["shard_id"],
                "cube_id": row["cube_id"],
                "stencil_width": row["stencil_width"],
                "support_mode": row["support_mode"],
                "reused": False,
                "compute_wall_seconds": compute_wall_seconds,
                "estimator_elapsed_seconds_sum": result.elapsed_seconds,
                "serialization_wall_seconds": serialization_wall_seconds,
                "publish_verification_wall_seconds": publish_verification_wall_seconds,
                "staging_logical_bytes_before_marker": staging_bytes["logical_bytes"],
                "staging_allocated_bytes_before_marker": staging_bytes["allocated_bytes"],
            }
        )
        published += 1
    return {
        "procid": procid,
        "ntasks": ntasks,
        "assigned": len(assigned),
        "published": published,
        "reused": reused,
        **_publish_work_resource_record(
            output_root,
            action="work",
            workers=workers,
            started=work_started,
            rows=resource_rows,
            cube_footprints=cube_footprints,
        ),
    }


def _group_rows(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows = json.loads((output_root / "manifests" / "shards.json").read_text())["shards"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    return groups


def _finite_distribution_summary(
    values: np.ndarray,
    *,
    minimum_valid_count: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return finite sample SD, central interval, and counts along axis zero."""

    values = np.asarray(values, dtype=float)
    if minimum_valid_count < 2:
        raise ValueError("minimum_valid_count must be at least two")
    output_shape = values.shape[1:]
    standard_error = np.full(output_shape, np.nan, dtype=float)
    interval_low = np.full(output_shape, np.nan, dtype=float)
    interval_high = np.full(output_shape, np.nan, dtype=float)
    valid_count = np.count_nonzero(np.isfinite(values), axis=0)
    for index in np.ndindex(output_shape):
        finite = values[(slice(None), *index)]
        finite = finite[np.isfinite(finite)]
        if finite.size < minimum_valid_count:
            continue
        standard_error[index] = np.std(finite, ddof=1)
        interval_low[index], interval_high[index] = np.quantile(
            finite,
            (0.5 * (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL), 0.5 * (1.0 + BOOTSTRAP_CONFIDENCE_LEVEL)),
        )
    return standard_error, interval_low, interval_high, valid_count


def _local_slope_support_mask(
    contributing_blocks: np.ndarray,
    effective_blocks: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Return cells whose complete local-slope window has spatial support."""

    contributing = np.asarray(contributing_blocks)
    effective = np.asarray(effective_blocks, dtype=float)
    if contributing.shape != effective.shape or contributing.ndim < 1:
        raise ValueError("local-slope support arrays must have matching non-scalar shapes")
    if window < 3 or window % 2 == 0 or window > contributing.shape[-1]:
        raise ValueError("local-slope support window must be odd, at least three, and fit the bins")
    supported_bins = (contributing >= 2) & (
        effective >= MINIMUM_LOCAL_SLOPE_EFFECTIVE_BLOCKS
    )
    supported_slopes = np.zeros_like(supported_bins, dtype=bool)
    radius = window // 2
    for center in range(radius, contributing.shape[-1] - radius):
        supported_slopes[..., center] = np.all(
            supported_bins[..., center - radius : center + radius + 1],
            axis=-1,
        )
    return supported_slopes


def _supported_local_log_slopes(
    ell: np.ndarray,
    moments: np.ndarray,
    contributing_blocks: np.ndarray,
    effective_blocks: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local slopes masked by a documented spatial-support gate."""

    support_mask = _local_slope_support_mask(
        contributing_blocks,
        effective_blocks,
        window=window,
    )
    slopes = local_log_slope(ell, moments, window=window)
    return np.where(support_mask, slopes, np.nan), support_mask


def _uncertainty_payload(
    result,
    jackknife,
    bootstrap,
) -> dict[str, np.ndarray]:
    """Return reproducible moment and local-slope uncertainty products."""

    if bootstrap.replicate_moments is None:
        raise ValueError("bootstrap replicates are required for local-slope uncertainty")
    ell = np.sqrt(result.ell_bin_edges[:-1] * result.ell_bin_edges[1:])
    local_slope, local_slope_support = _supported_local_log_slopes(
        ell,
        result.moments,
        bootstrap.contributing_blocks,
        bootstrap.effective_blocks,
        window=5,
    )
    local_slope_replicates = np.where(
        local_slope_support[None, ...],
        local_log_slope(ell, bootstrap.replicate_moments, window=5),
        np.nan,
    )
    slope_se, slope_low, slope_high, slope_valid = _finite_distribution_summary(
        local_slope_replicates,
        minimum_valid_count=int(
            np.ceil(
                MINIMUM_LOCAL_SLOPE_VALID_BOOTSTRAP_FRACTION
                * bootstrap.n_resamples
            )
        ),
    )
    assert result.block_sampled_origins is not None
    assert result.block_eligible_origins is not None
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_method": bootstrap.method,
        "bootstrap_seed": bootstrap.seed,
        "bootstrap_n_resamples": bootstrap.n_resamples,
        "bootstrap_confidence_level": bootstrap.confidence_level,
        "jackknife_method": jackknife.method,
        "jackknife_resampling_population": jackknife.resampling_population,
        "resampling_population": bootstrap.resampling_population,
        "geometric_block_count": bootstrap.geometric_block_count,
        "block_shape_kji": result.block_shape_kji,
        "block_assignment": result.block_assignment,
        "local_slope_window_bins": 5,
        "minimum_local_slope_effective_blocks": MINIMUM_LOCAL_SLOPE_EFFECTIVE_BLOCKS,
        "minimum_local_slope_valid_bootstrap_fraction": (
            MINIMUM_LOCAL_SLOPE_VALID_BOOTSTRAP_FRACTION
        ),
    }
    return {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "moments": result.moments,
        "pair_sampling_standard_error": result.standard_error,
        "block_jackknife_standard_error": jackknife.standard_error,
        "block_bootstrap_standard_error": bootstrap.standard_error,
        "block_bootstrap_interval_low": bootstrap.interval_low,
        "block_bootstrap_interval_high": bootstrap.interval_high,
        "accepted_contributing_blocks": bootstrap.contributing_blocks,
        "accepted_effective_blocks": bootstrap.effective_blocks,
        "sampled_blocks_per_shell": np.count_nonzero(result.block_sampled_origins > 0, axis=0),
        "eligible_blocks_per_shell": np.count_nonzero(result.block_eligible_origins > 0, axis=0),
        "valid_bootstrap_resamples": bootstrap.valid_resamples,
        "local_log_slope": local_slope,
        "local_log_slope_support_mask": local_slope_support,
        "local_log_slope_bootstrap_standard_error": slope_se,
        "local_log_slope_bootstrap_interval_low": slope_low,
        "local_log_slope_bootstrap_interval_high": slope_high,
        "local_log_slope_valid_bootstrap_resamples": slope_valid,
    }


def _verify_uncertainty_payload(path: Path, result) -> None:
    """Reject incomplete or inconsistent uncertainty publications."""

    required = {
        "metadata_json",
        "moments",
        "pair_sampling_standard_error",
        "block_jackknife_standard_error",
        "block_bootstrap_standard_error",
        "block_bootstrap_interval_low",
        "block_bootstrap_interval_high",
        "accepted_contributing_blocks",
        "accepted_effective_blocks",
        "sampled_blocks_per_shell",
        "eligible_blocks_per_shell",
        "valid_bootstrap_resamples",
        "local_log_slope",
        "local_log_slope_support_mask",
        "local_log_slope_bootstrap_standard_error",
        "local_log_slope_bootstrap_interval_low",
        "local_log_slope_bootstrap_interval_high",
        "local_log_slope_valid_bootstrap_resamples",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != required:
            raise RuntimeError(f"invalid Phase 3a uncertainty array inventory: {path}")
        metadata_array = payload["metadata_json"]
        if metadata_array.shape != () or metadata_array.dtype.kind not in "SU":
            raise RuntimeError(f"invalid Phase 3a uncertainty metadata scalar: {path}")
        metadata = json.loads(str(metadata_array.item()))
        if (
            metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("bootstrap_method") != "spatial_block_bootstrap"
            or metadata.get("bootstrap_seed") != BOOTSTRAP_SEED
            or metadata.get("bootstrap_n_resamples") != BOOTSTRAP_N_RESAMPLES
            or metadata.get("bootstrap_confidence_level") != BOOTSTRAP_CONFIDENCE_LEVEL
            or metadata.get("jackknife_resampling_population")
            != "delete_contributing_blocks_only"
            or metadata.get("resampling_population")
            != "fixed_geometric_layout_including_empty_blocks"
            or tuple(metadata.get("block_shape_kji", ())) != result.block_shape_kji
            or metadata.get("block_assignment") != result.block_assignment
            or metadata.get("local_slope_window_bins") != 5
            or metadata.get("minimum_local_slope_effective_blocks")
            != MINIMUM_LOCAL_SLOPE_EFFECTIVE_BLOCKS
            or metadata.get("minimum_local_slope_valid_bootstrap_fraction")
            != MINIMUM_LOCAL_SLOPE_VALID_BOOTSTRAP_FRACTION
        ):
            raise RuntimeError(f"invalid Phase 3a uncertainty metadata: {path}")
        if not np.array_equal(payload["moments"], result.moments, equal_nan=True):
            raise RuntimeError(f"Phase 3a uncertainty moments do not match result: {path}")
        if payload["accepted_effective_blocks"].shape != result.sums.shape:
            raise RuntimeError(f"Phase 3a uncertainty block shape mismatch: {path}")
        if (
            payload["local_log_slope_support_mask"].shape != result.sums.shape
            or payload["local_log_slope_support_mask"].dtype.kind != "b"
        ):
            raise RuntimeError(f"Phase 3a local-slope support mask mismatch: {path}")


def _verify_reduction(output_root: Path, group_id: str) -> dict[str, Any]:
    root, result_path, uncertainty_path, marker_path = _reduction_paths(output_root, group_id)
    marker = json.loads(marker_path.read_text())
    reduction_manifest_path = root / "reduction_manifest.json"
    reduction_manifest = json.loads(reduction_manifest_path.read_text())
    campaign = json.loads((output_root / "manifests" / "campaign.json").read_text())
    rows = _group_rows(output_root)[group_id]
    expected = tuple(row["shard_id"] for row in rows)
    timing_names = (
        "reduction_elapsed_seconds",
        "jackknife_elapsed_seconds",
        "bootstrap_elapsed_seconds",
        "staging_logical_bytes_before_marker",
        "staging_allocated_bytes_before_marker",
    )
    marker_hashes = {}
    for row in rows:
        _verify_shard(output_root, row)
        marker_hashes[row["shard_id"]] = file_sha256(
            _shard_paths(output_root, row["shard_id"])[2]
        )
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "passed"
        or marker.get("group_id") != group_id
        or marker.get("result_sha256") != file_sha256(result_path)
        or marker.get("uncertainty_sha256") != file_sha256(uncertainty_path)
        or marker.get("reduction_manifest_sha256") != file_sha256(reduction_manifest_path)
        or reduction_manifest.get("group_id") != group_id
        or tuple(reduction_manifest.get("ordered_shard_ids", ())) != expected
        or reduction_manifest.get("ordered_shard_marker_sha256") != marker_hashes
        or reduction_manifest.get("implementation_sha256")
        != campaign["source_version"]["implementation_sha256"]
        or any(
            not isinstance(marker.get(name), (int, float))
            or not np.isfinite(marker[name])
            or marker[name] < 0
            for name in timing_names
        )
    ):
        raise RuntimeError(f"invalid Phase 3a reduction marker: {root}")
    result = load_finite_domain_partial_npz(result_path)
    _verify_uncertainty_payload(uncertainty_path, result)
    return marker


def reduce(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    """Reduce exactly the planned shard inventory and publish uncertainty products."""

    campaign = _verify_plan(phase2_root, output_root, verify_arrays=True)
    published = reused = 0
    for group_id, rows in sorted(_group_rows(output_root).items()):
        root, _, _, marker_path = _reduction_paths(output_root, group_id)
        if marker_path.exists():
            _verify_reduction(output_root, group_id)
            reused += 1
            continue
        if root.exists():
            raise RuntimeError(f"incomplete authoritative reduction directory exists: {root}")
        partials = {}
        marker_hashes = {}
        for row in rows:
            marker = _verify_shard(output_root, row)
            partials[row["shard_id"]] = load_finite_domain_partial_npz(
                _shard_paths(output_root, row["shard_id"])[1]
            )
            marker_hashes[row["shard_id"]] = file_sha256(
                _shard_paths(output_root, row["shard_id"])[2]
            )
            if marker["implementation_sha256"] != campaign["source_version"]["implementation_sha256"]:
                raise RuntimeError(f"mixed source hash in shard: {row['shard_id']}")
        expected = tuple(row["shard_id"] for row in rows)
        reduction_started = time.perf_counter()
        result = reduce_finite_domain_shards(partials, expected_shard_ids=expected)
        reduction_elapsed_seconds = time.perf_counter() - reduction_started
        jackknife_started = time.perf_counter()
        jackknife = block_jackknife_moment_uncertainty(result)
        jackknife_elapsed_seconds = time.perf_counter() - jackknife_started
        bootstrap_started = time.perf_counter()
        bootstrap = block_bootstrap_moment_uncertainty(
            result,
            n_resamples=BOOTSTRAP_N_RESAMPLES,
            seed=BOOTSTRAP_SEED,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            return_replicates=True,
        )
        bootstrap_elapsed_seconds = time.perf_counter() - bootstrap_started
        attempt = output_root / "attempts" / f"reduction_{group_id.replace('/', '__')}_{time.time_ns()}"
        attempt.mkdir(parents=True)
        result_path = attempt / "result.npz"
        uncertainty_path = attempt / "uncertainty.npz"
        save_finite_domain_partial_npz(result_path, result)
        _atomic_write_npz(uncertainty_path, **_uncertainty_payload(result, jackknife, bootstrap))
        _atomic_write_json(
            attempt / "reduction_manifest.json",
            {
                "group_id": group_id,
                "ordered_shard_ids": expected,
                "ordered_shard_marker_sha256": marker_hashes,
                "implementation_sha256": campaign["source_version"]["implementation_sha256"],
            },
        )
        staging_bytes = _tree_size_accounting(attempt)
        _atomic_write_json(
            attempt / "COMPLETE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "group_id": group_id,
                "result_sha256": file_sha256(result_path),
                "uncertainty_sha256": file_sha256(uncertainty_path),
                "reduction_manifest_sha256": file_sha256(attempt / "reduction_manifest.json"),
                "reduction_elapsed_seconds": reduction_elapsed_seconds,
                "jackknife_elapsed_seconds": jackknife_elapsed_seconds,
                "bootstrap_elapsed_seconds": bootstrap_elapsed_seconds,
                "staging_logical_bytes_before_marker": staging_bytes["logical_bytes"],
                "staging_allocated_bytes_before_marker": staging_bytes["allocated_bytes"],
                "published_unix_seconds": time.time(),
            },
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        attempt.replace(root)
        _verify_reduction(output_root, group_id)
        published += 1
    return {"published": published, "reused": reused}


def verify(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    """Recheck the complete plan, partial, and reduction marker chain."""

    _verify_plan(phase2_root, output_root, verify_arrays=True)
    shard_count = 0
    for rows in _group_rows(output_root).values():
        for row in rows:
            _verify_shard(output_root, row)
            shard_count += 1
    groups = sorted(_group_rows(output_root))
    for group_id in groups:
        _verify_reduction(output_root, group_id)
    return {"status": "passed", "verified_shards": shard_count, "verified_reductions": len(groups)}


def _relative_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape": left.shape,
            "other_shape": right.shape,
            "finite_mask_mismatch_count": None,
            "compared": 0,
            "median_relative_difference": None,
            "maximum_relative_difference": None,
            "equivalent": False,
        }
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    valid = left_finite & right_finite
    relative = np.abs(left - right) / np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-30)
    comparison = {
        "shape": left.shape,
        "finite_mask_mismatch_count": int(np.count_nonzero(left_finite ^ right_finite)),
        "compared": int(np.count_nonzero(valid)),
        "median_relative_difference": float(np.median(relative[valid])) if np.any(valid) else None,
        "maximum_relative_difference": float(np.max(relative[valid])) if np.any(valid) else None,
    }
    comparison["equivalent"] = bool(
        comparison["finite_mask_mismatch_count"] == 0
        and comparison["compared"] > 0
        and comparison["maximum_relative_difference"] is not None
        and comparison["maximum_relative_difference"] <= PARALLEL_EQUIVALENCE_RTOL
    )
    return comparison


def _require_equivalent(
    comparison: Mapping[str, Any],
    label: str,
    *,
    rtol: float = PARALLEL_EQUIVALENCE_RTOL,
) -> None:
    """Require populated finite support and floating-point-equivalent values."""

    if (
        not comparison.get("equivalent")
        or comparison.get("maximum_relative_difference") is None
        or comparison["maximum_relative_difference"] > rtol
    ):
        raise RuntimeError(f"{label} failed equivalence tolerance {rtol}: {comparison}")


def _crop_cube(cube: Mapping[str, np.ndarray], side: int) -> dict[str, np.ndarray]:
    """Return a true cubic crop without carrying parent prebuilt vector stacks."""

    return {
        name: np.asarray(values)[:side, :side, :side]
        for name, values in cube.items()
        if not name.startswith("_")
    }


def _exact_control_offsets() -> np.ndarray:
    """Return signed small, intermediate, and large Cartesian/oblique offsets."""

    return np.asarray(
        (
            (1, 0, 0),
            (-1, 0, 0),
            (4, 0, 0),
            (-4, 0, 0),
            (3, 4, 0),
            (-3, -4, 0),
            (6, 4, 2),
            (-6, -4, -2),
        ),
        dtype=np.int64,
    )


def _result_comparison(left, right) -> dict[str, Any]:
    """Compare additive statistics and derived moments for two results."""

    return {
        "counts_equal": bool(np.array_equal(left.counts, right.counts)),
        "sums": _relative_difference(left.sums, right.sums),
        "sums_sq": _relative_difference(left.sums_sq, right.sums_sq),
        "moments": _relative_difference(left.moments, right.moments),
    }


def _require_result_equivalent(comparison: Mapping[str, Any], label: str) -> None:
    if not comparison.get("counts_equal"):
        raise RuntimeError(f"{label} produced different accepted counts")
    for name in ("sums", "sums_sq", "moments"):
        _require_equivalent(comparison[name], f"{label} {name}")


def _stratified_offset_subset(
    displacements: np.ndarray,
    ell_bin_edges: np.ndarray,
    *,
    maximum: int,
) -> np.ndarray:
    """Select a deterministic shell-stratified offset subset for bounded diagnostics."""

    offsets = np.asarray(displacements, dtype=np.int64)
    lengths = np.linalg.norm(offsets.astype(float), axis=1)
    bins = np.searchsorted(ell_bin_edges, lengths, side="right") - 1
    selected: list[np.ndarray] = []
    occupied = tuple(sorted(set(int(value) for value in bins if value >= 0)))
    per_bin = max(1, maximum // max(1, len(occupied)))
    for ell_index in occupied:
        members = offsets[bins == ell_index]
        if len(members) <= per_bin:
            selected.extend(members)
            continue
        indices = np.linspace(0, len(members) - 1, per_bin, dtype=int)
        selected.extend(members[indices])
    if len(selected) > maximum:
        selected = selected[:maximum]
    return np.asarray(selected, dtype=np.int64).reshape((-1, 3))


def _convergence_offset_subset(
    family: str,
    displacements: np.ndarray,
    ell_bin_edges: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Retain the evidence needed for each bounded convergence question."""

    if family in {"bins", "directions", "directions_all_valid", "support"}:
        return np.asarray(displacements, dtype=np.int64), "full_displacement_census"
    if family in {"origins", "origin_seeds"}:
        return (
            _stratified_offset_subset(displacements, ell_bin_edges, maximum=256),
            "shell_stratified_maximum_256_offsets",
        )
    return (
        _stratified_offset_subset(displacements, ell_bin_edges, maximum=96),
        "shell_stratified_maximum_96_offsets",
    )


def _convergence_uncertainty_payload(result) -> dict[str, np.ndarray]:
    """Return bounded block uncertainty products for one convergence scenario."""

    bootstrap = block_bootstrap_moment_uncertainty(
        result,
        n_resamples=100,
        seed=BOOTSTRAP_SEED,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        return_replicates=True,
    )
    jackknife = block_jackknife_moment_uncertainty(result)
    if bootstrap.replicate_moments is None:
        raise RuntimeError("convergence bootstrap did not retain replicates")
    ell = np.sqrt(result.ell_bin_edges[:-1] * result.ell_bin_edges[1:])
    slopes, slope_support = _supported_local_log_slopes(
        ell,
        result.moments,
        bootstrap.contributing_blocks,
        bootstrap.effective_blocks,
        window=5,
    )
    slope_replicates = np.where(
        slope_support[None, ...],
        local_log_slope(ell, bootstrap.replicate_moments, window=5),
        np.nan,
    )
    slope_se, slope_low, slope_high, slope_valid = _finite_distribution_summary(
        slope_replicates,
        minimum_valid_count=int(
            np.ceil(
                MINIMUM_LOCAL_SLOPE_VALID_BOOTSTRAP_FRACTION
                * bootstrap.n_resamples
            )
        ),
    )
    return {
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "q_names": result.q_names,
                    "density_conventions": result.density_conventions,
                    "geometry_names": result.geometry_names,
                    "measurement_names": result.measurement_names,
                    "direction_names": result.direction_names,
                    "p_values": result.p_values,
                    "bootstrap_method": bootstrap.method,
                    "bootstrap_seed": bootstrap.seed,
                    "bootstrap_n_resamples": bootstrap.n_resamples,
                    "bootstrap_confidence_level": bootstrap.confidence_level,
                    "jackknife_method": jackknife.method,
                    "jackknife_resampling_population": (
                        jackknife.resampling_population
                    ),
                    "resampling_population": bootstrap.resampling_population,
                    "geometric_block_count": bootstrap.geometric_block_count,
                    "block_shape_kji": result.block_shape_kji,
                    "block_assignment": result.block_assignment,
                    "local_slope_window_bins": 5,
                    "minimum_local_slope_effective_blocks": (
                        MINIMUM_LOCAL_SLOPE_EFFECTIVE_BLOCKS
                    ),
                    "minimum_local_slope_valid_bootstrap_fraction": (
                        MINIMUM_LOCAL_SLOPE_VALID_BOOTSTRAP_FRACTION
                    ),
                },
                sort_keys=True,
            )
        ),
        "ell_bin_edges": result.ell_bin_edges,
        "moments": result.moments,
        "counts": result.counts,
        "pair_sampling_standard_error": result.standard_error,
        "block_jackknife_standard_error": jackknife.standard_error,
        "block_bootstrap_standard_error": bootstrap.standard_error,
        "block_bootstrap_interval_low": bootstrap.interval_low,
        "block_bootstrap_interval_high": bootstrap.interval_high,
        "accepted_contributing_blocks": bootstrap.contributing_blocks,
        "accepted_effective_blocks": bootstrap.effective_blocks,
        "valid_bootstrap_resamples": bootstrap.valid_resamples,
        "local_log_slope": slopes,
        "local_log_slope_support_mask": slope_support,
        "local_log_slope_bootstrap_standard_error": slope_se,
        "local_log_slope_bootstrap_interval_low": slope_low,
        "local_log_slope_bootstrap_interval_high": slope_high,
        "local_log_slope_valid_bootstrap_resamples": slope_valid,
        "sampled_origins": result.sampled_pairs,
        "eligible_origins": result.eligible_pairs,
        "intrinsic_eligible_origins": result.intrinsic_eligible_origins,
        "boundary_excluded_origins": result.boundary_excluded_origins,
        "support_policy_excluded_origins": result.support_policy_excluded_origins,
        "displacements_per_bin": result.displacements_per_bin,
        "displacements_ijk": result.displacements_ijk,
        "ell_bin_index_per_displacement": result.ell_bin_index_per_displacement,
        "sampled_origins_per_displacement": result.sampled_pairs_per_displacement,
        "eligible_origins_per_displacement": result.eligible_pairs_per_displacement,
        "intrinsic_eligible_origins_per_displacement": (
            result.intrinsic_eligible_origins_per_displacement
        ),
        "boundary_excluded_origins_per_displacement": (
            result.boundary_excluded_origins_per_displacement
        ),
        "support_policy_excluded_origins_per_displacement": (
            result.support_policy_excluded_origins_per_displacement
        ),
        "elapsed_seconds_per_ell_bin": result.elapsed_seconds_per_ell_bin,
    }


def _occupied_effective_block_summary(
    effective_blocks: np.ndarray,
) -> dict[str, float | int | None]:
    """Summarize non-empty Kish effective-block populations only."""

    values = np.asarray(effective_blocks, dtype=float)
    occupied = values[np.isfinite(values) & (values > 0.0)]
    return {
        "minimum_accepted_effective_blocks": (
            float(np.min(occupied)) if occupied.size else None
        ),
        "median_accepted_effective_blocks": (
            float(np.median(occupied)) if occupied.size else None
        ),
        "occupied_effective_block_cells": int(occupied.size),
    }


def _phase3_anchor_regression(
    phase2_root: Path,
    cube: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Replay the retained real-cube Phase 3 smoke settings through the current core."""

    cube_id = BENCHMARK_CUBE_IDS[0]
    cube_root = PHASE3_ANCHOR_ROOT / cube_id
    marker = json.loads((cube_root / "COMPLETE.json").read_text())
    if (
        marker.get("schema_version") != PHASE3_SCHEMA_VERSION
        or marker.get("status") != "passed"
        or marker.get("cube_id") != cube_id
        or marker.get("phase2_source")
        != _phase2_source_identity(phase2_root, cube_id, verify_arrays=True)
    ):
        raise RuntimeError("retained Phase 3 anchor marker is stale relative to Phase 2 inputs")
    rows = []
    for mode in ("nested_core", "all_valid_pairs"):
        path = cube_root / f"{mode}.npz"
        if marker.get("result_sha256", {}).get(path.name) != file_sha256(path):
            raise RuntimeError(f"retained Phase 3 anchor checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            angle_limits = metadata["angle_limits"]
            displacements = payload["displacements_ijk"].copy()
            current = compute_finite_domain_structure_functions(
                cube,
                displacements,
                config=FiniteDomainConfig(
                    payload["ell_bin_edges"].copy(),
                    p_values=tuple(float(value) for value in payload["p_values"]),
                    pair_mode=mode,
                    sample_count=int(metadata["sample_count"]),
                    pair_batch_size=int(metadata["pair_batch_size"]),
                    seed=int(metadata["seed"]),
                    cell_sizes=tuple(float(value) for value in metadata["cell_sizes"]),
                    theta_parallel_max=float(angle_limits["theta_parallel_max"]),
                    theta_perpendicular_min=float(angle_limits["theta_perpendicular_min"]),
                    phi_xi_max=float(angle_limits["phi_xi_max"]),
                    phi_lambda_min=float(angle_limits["phi_lambda_min"]),
                ),
                q_names=tuple(str(value) for value in payload["q_names"].tolist()),
                support_displacements_ijk=displacements,
            )
            comparisons = {
                "counts_equal": bool(np.array_equal(payload["counts"], current.counts)),
                "sampled_origins_equal": bool(
                    np.array_equal(payload["sampled_pairs"], current.sampled_pairs)
                ),
                "eligible_origins_equal": bool(
                    np.array_equal(payload["eligible_pairs"], current.eligible_pairs)
                ),
                "sums": _relative_difference(payload["sums"], current.sums),
                "sums_sq": _relative_difference(payload["sums_sq"], current.sums_sq),
                "moments": _relative_difference(payload["moments"], current.moments),
            }
        if not all(
            comparisons[name]
            for name in ("counts_equal", "sampled_origins_equal", "eligible_origins_equal")
        ):
            raise RuntimeError(f"Phase 3 anchor discrete accounting changed for {mode}")
        for name in ("sums", "sums_sq", "moments"):
            _require_equivalent(comparisons[name], f"Phase 3 anchor {mode} {name}")
        rows.append({"mode": mode, "artifact_sha256": file_sha256(path), **comparisons})
    return {
        "anchor_root": str(PHASE3_ANCHOR_ROOT),
        "cube_id": cube_id,
        "validation_status": "passed",
        "rows": rows,
    }


def controls(phase2_root: Path, output_root: Path, *, workers: int) -> dict[str, Any]:
    """Run one-cube serial, fork-worker, and cropped exhaustive controls."""

    output_root.mkdir(parents=True, exist_ok=True)
    cube = prepare_cube_for_parallel(_load_cube(phase2_root, BENCHMARK_CUBE_IDS[0]))
    rows = []
    for stencil_width, ell_max in ((2, 128), (3, 80), (5, 40)):
        displacements, edges, manifest = dense_displacement_manifest(
            stencil_width=stencil_width, ell_max=ell_max, bin_count=32, directions_per_bin=12
        )
        selected = displacements[:: max(1, len(displacements) // 48)][:48]
        config = FiniteDomainConfig(
            edges,
            pair_mode="shell_local",
            sample_count=256,
            pair_batch_size=128,
            stencil_width=stencil_width,
            block_shape_kji=(160, 160, 160),
        )
        started = time.perf_counter()
        serial = compute_finite_domain_structure_functions(
            cube, selected, config=config, q_names=Q_NAMES, support_displacements_ijk=displacements
        )
        serial_wall_seconds = time.perf_counter() - started
        worker_sweep = []
        parallel = None
        for worker_count in sorted({1, 2, workers}):
            started = time.perf_counter()
            measured = reduce_finite_domain_shards(
                compute_finite_domain_structure_functions_parallel(
                    cube,
                    selected,
                    config=config,
                    q_names=Q_NAMES,
                    worker_count=worker_count,
                    support_displacements_ijk=displacements,
                )
            )
            wall_seconds = time.perf_counter() - started
            worker_sweep.append(
                {
                    "worker_count": worker_count,
                    "wall_seconds": wall_seconds,
                    "elapsed_seconds_sum": measured.elapsed_seconds,
                    "serial_vs_parallel": _relative_difference(serial.moments, measured.moments),
                }
            )
            _require_equivalent(
                worker_sweep[-1]["serial_vs_parallel"],
                f"{stencil_width}-point serial-versus-{worker_count}-worker",
            )
            if worker_count == workers:
                parallel = measured
        assert parallel is not None
        crop = _crop_cube(cube, 32)
        crop_offsets = _exact_control_offsets()
        crop_edges = np.asarray((0.5, 2.0, 4.5, 6.0, 8.5))
        cropped_controls = []
        for support_mode in SUPPORT_MODES:
            exact_config = FiniteDomainConfig(
                crop_edges,
                pair_mode=support_mode,
                sample_count=None,
                stencil_width=stencil_width,
                block_shape_kji=(8, 8, 8),
            )
            exhaustive = compute_finite_domain_structure_functions(
                crop,
                crop_offsets,
                config=exact_config,
                q_names=Q_NAMES,
                support_displacements_ijk=crop_offsets,
            )
            exact_parallel = reduce_finite_domain_shards(
                compute_finite_domain_structure_functions_parallel(
                    crop,
                    crop_offsets,
                    config=exact_config,
                    q_names=Q_NAMES,
                    worker_count=max(2, workers),
                    support_displacements_ijk=crop_offsets,
                )
            )
            exact_serial_vs_parallel = _result_comparison(exhaustive, exact_parallel)
            _require_result_equivalent(
                exact_serial_vs_parallel,
                f"{stencil_width}-point {support_mode} cropped exact serial-versus-parallel",
            )
            cropped_sample_ladder = []
            for sample_count in (2048, 8192, 32768):
                sampled = compute_finite_domain_structure_functions(
                    crop,
                    crop_offsets,
                    config=FiniteDomainConfig(
                        crop_edges,
                        pair_mode=support_mode,
                        sample_count=sample_count,
                        stencil_width=stencil_width,
                        block_shape_kji=(8, 8, 8),
                    ),
                    q_names=Q_NAMES,
                    support_displacements_ijk=crop_offsets,
                )
                cropped_sample_ladder.append(
                    {
                        "requested_sample_count": sample_count,
                        "sampled_origins": int(sampled.sampled_pairs.sum()),
                        "exact_vs_sampled": _result_comparison(exhaustive, sampled),
                    }
                )
            cropped_controls.append(
                {
                    "support_mode": support_mode,
                    "cube_shape_kji": exhaustive.cube_shape_kji,
                    "offsets_ijk": crop_offsets,
                    "serial_vs_parallel_exact": exact_serial_vs_parallel,
                    "sample_ladder": cropped_sample_ladder,
                }
            )
        one_worker_wall_seconds = next(
            row["wall_seconds"] for row in worker_sweep if row["worker_count"] == 1
        )
        for row in worker_sweep:
            row["speedup_vs_one_worker_wall"] = one_worker_wall_seconds / row["wall_seconds"]
        rows.append(
            {
                "stencil_width": stencil_width,
                "manifest_sha256": manifest["manifest_sha256"],
                "serial_elapsed_seconds": serial.elapsed_seconds,
                "serial_wall_seconds": serial_wall_seconds,
                "parallel_elapsed_seconds_sum": parallel.elapsed_seconds,
                "parallel_wall_seconds": next(
                    row["wall_seconds"] for row in worker_sweep if row["worker_count"] == workers
                ),
                "serial_vs_parallel": _relative_difference(serial.moments, parallel.moments),
                "worker_sweep": worker_sweep,
                "cropped_exact_controls": cropped_controls,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operational_status": "complete",
        "validation_status": "passed",
        "scientific_acceptance": "diagnostic_only",
        "workers": workers,
        "cube_id": BENCHMARK_CUBE_IDS[0],
        "phase2_source": _phase2_source_identity(
            phase2_root, BENCHMARK_CUBE_IDS[0], verify_arrays=True
        ),
        "parent_process_peak_rss_kib": _peak_rss_kib(),
        "phase3_anchor_regression": _phase3_anchor_regression(phase2_root, cube),
        "rows": rows,
        "source_version": _source_version(),
    }
    _publish_json_diagnostic(output_root, "controls.json", "CONTROLS_COMPLETE.json", payload)
    return _verify_json_diagnostic(
        output_root,
        "controls.json",
        "CONTROLS_COMPLETE.json",
        phase2_root=phase2_root,
    )


def _multinode_design() -> tuple[np.ndarray, np.ndarray, np.ndarray, FiniteDomainConfig]:
    displacements, edges, _ = dense_displacement_manifest(
        stencil_width=2, ell_max=128, bin_count=32, directions_per_bin=12
    )
    selected = displacements[:: max(1, len(displacements) // 96)][:96]
    config = FiniteDomainConfig(
        edges,
        pair_mode="shell_local",
        sample_count=256,
        pair_batch_size=128,
        stencil_width=2,
        block_shape_kji=(160, 160, 160),
    )
    return displacements, edges, selected, config


def _multinode_task_metadata(
    phase2_root: Path,
    task_offsets: np.ndarray,
    support_displacements: np.ndarray,
    config: FiniteDomainConfig,
    *,
    procid: int,
    ntasks: int,
    workers: int,
) -> dict[str, Any]:
    """Return the complete provenance binding for one multi-node control task."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "procid": procid,
        "ntasks": ntasks,
        "workers": workers,
        "implementation_sha256": _source_version()["implementation_sha256"],
        "phase2_source": _phase2_source_identity(
            phase2_root, BENCHMARK_CUBE_IDS[0], verify_arrays=True
        ),
        "support_displacements_sha256": hashlib.sha256(
            support_displacements.tobytes()
        ).hexdigest(),
        "offsets_sha256": hashlib.sha256(task_offsets.tobytes()).hexdigest(),
        "sampling_configuration_sha256": _mapping_sha256(
            {
                "ell_bin_edges": config.ell_bin_edges,
                "pair_mode": config.pair_mode,
                "sample_count": config.sample_count,
                "pair_batch_size": config.pair_batch_size,
                "seed": config.seed,
                "stencil_width": config.stencil_width,
                "block_shape_kji": config.block_shape_kji,
            }
        ),
    }


def _verify_multinode_task(
    phase2_root: Path,
    output_root: Path,
    *,
    procid: int,
    ntasks: int,
    workers: int,
    task_offsets: np.ndarray,
    support_displacements: np.ndarray,
    config: FiniteDomainConfig,
) -> dict[str, Any]:
    task_root = output_root / "multinode_control" / f"task_{procid:04d}"
    marker = json.loads((task_root / "COMPLETE.json").read_text())
    expected = _multinode_task_metadata(
        phase2_root,
        task_offsets,
        support_displacements,
        config,
        procid=procid,
        ntasks=ntasks,
        workers=workers,
    )
    if (
        {key: marker.get(key) for key in expected} != expected
        or marker.get("partial_sha256") != file_sha256(task_root / "partial.npz")
        or any(
            not isinstance(marker.get(name), int) or marker[name] < 0
            for name in (
                "staging_logical_bytes_before_marker",
                "staging_allocated_bytes_before_marker",
            )
        )
    ):
        raise RuntimeError(f"invalid multi-node control task: {task_root}")
    result = load_finite_domain_partial_npz(task_root / "partial.npz")
    ordered_offsets = task_offsets[
        np.lexsort((task_offsets[:, 2], task_offsets[:, 1], task_offsets[:, 0]))
    ]
    if (
        not np.array_equal(result.displacements_ijk, ordered_offsets)
        or result.support_displacements_sha256
        != expected["support_displacements_sha256"]
    ):
        raise RuntimeError(f"multi-node control task has wrong offset coverage: {task_root}")
    return marker


def multinode_work(phase2_root: Path, output_root: Path, *, workers: int) -> dict[str, Any]:
    """Publish one deterministic multi-node control partial per Slurm task."""

    work_started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    procid = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    displacements, _, selected, config = _multinode_design()
    task_offsets = selected[procid::ntasks]
    if not len(task_offsets):
        raise RuntimeError("multi-node control assigned an empty task offset set")
    task_root = output_root / "multinode_control" / f"task_{procid:04d}"
    if task_root.exists():
        reuse_started = time.perf_counter()
        _verify_multinode_task(
            phase2_root,
            output_root,
            procid=procid,
            ntasks=ntasks,
            workers=workers,
            task_offsets=task_offsets,
            support_displacements=displacements,
            config=config,
        )
        return {
            "reused": True,
            "procid": procid,
            "ntasks": ntasks,
            **_publish_work_resource_record(
                output_root,
                action="multinode_work",
                workers=workers,
                started=work_started,
                rows=[
                    {
                        "shard_id": f"task_{procid:04d}",
                        "cube_id": BENCHMARK_CUBE_IDS[0],
                        "stencil_width": config.stencil_width,
                        "support_mode": config.pair_mode,
                        "reused": True,
                        "reuse_verification_wall_seconds": time.perf_counter() - reuse_started,
                    }
                ],
                cube_footprints={},
            ),
        }
    cube = prepare_cube_for_parallel(_load_cube(phase2_root, BENCHMARK_CUBE_IDS[0]))
    cube_footprint = _cube_memory_footprint(cube)
    compute_started = time.perf_counter()
    result = reduce_finite_domain_shards(
        compute_finite_domain_structure_functions_parallel(
            cube,
            task_offsets,
            config=config,
            q_names=Q_NAMES,
            worker_count=workers,
            support_displacements_ijk=displacements,
        )
    )
    compute_wall_seconds = time.perf_counter() - compute_started
    attempt = output_root / "attempts" / f"multinode_task_{procid}_{time.time_ns()}"
    attempt.mkdir(parents=True)
    serialization_started = time.perf_counter()
    save_finite_domain_partial_npz(attempt / "partial.npz", result)
    serialization_wall_seconds = time.perf_counter() - serialization_started
    staging_bytes = _tree_size_accounting(attempt)
    _atomic_write_json(
        attempt / "COMPLETE.json",
        {
            **_multinode_task_metadata(
                phase2_root,
                task_offsets,
                displacements,
                config,
                procid=procid,
                ntasks=ntasks,
                workers=workers,
            ),
            "partial_sha256": file_sha256(attempt / "partial.npz"),
            "staging_logical_bytes_before_marker": staging_bytes["logical_bytes"],
            "staging_allocated_bytes_before_marker": staging_bytes["allocated_bytes"],
        },
    )
    publish_started = time.perf_counter()
    task_root.parent.mkdir(parents=True, exist_ok=True)
    attempt.replace(task_root)
    _verify_multinode_task(
        phase2_root,
        output_root,
        procid=procid,
        ntasks=ntasks,
        workers=workers,
        task_offsets=task_offsets,
        support_displacements=displacements,
        config=config,
    )
    return {
        "reused": False,
        "procid": procid,
        "ntasks": ntasks,
        "offset_count": len(task_offsets),
        **_publish_work_resource_record(
            output_root,
            action="multinode_work",
            workers=workers,
            started=work_started,
            rows=[
                {
                    "shard_id": f"task_{procid:04d}",
                    "cube_id": BENCHMARK_CUBE_IDS[0],
                    "stencil_width": config.stencil_width,
                    "support_mode": config.pair_mode,
                    "reused": False,
                    "compute_wall_seconds": compute_wall_seconds,
                    "estimator_elapsed_seconds_sum": result.elapsed_seconds,
                    "serialization_wall_seconds": serialization_wall_seconds,
                    "publish_verification_wall_seconds": time.perf_counter() - publish_started,
                    "staging_logical_bytes_before_marker": staging_bytes["logical_bytes"],
                    "staging_allocated_bytes_before_marker": staging_bytes["allocated_bytes"],
                }
            ],
            cube_footprints={BENCHMARK_CUBE_IDS[0]: cube_footprint},
        ),
    }


def _select_fresh_multinode_resource_records(
    output_root: Path,
    *,
    task_count: int,
    workers: int,
) -> list[dict[str, str]]:
    """Bind exactly one complete fresh multi-node work allocation."""

    by_job: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = {}
    current_source = _source_version()["implementation_sha256"]
    for path in sorted((output_root / "work_resource_records").glob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload.get("rows", ())
        if (
            payload.get("action") != "multinode_work"
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("source_version", {}).get("implementation_sha256")
            != current_source
            or payload.get("slurm_ntasks") != task_count
            or payload.get("workers") != workers
            or payload.get("published_count") != 1
            or payload.get("reused_count") != 0
            or len(rows) != 1
            or rows[0].get("reused") is not False
        ):
            continue
        procid = payload.get("slurm_procid")
        if (
            not isinstance(procid, int)
            or not 0 <= procid < task_count
            or rows[0].get("shard_id") != f"task_{procid:04d}"
        ):
            continue
        job_id = str(payload.get("slurm_job_id"))
        if procid in by_job.setdefault(job_id, {}):
            raise RuntimeError(f"duplicate fresh multi-node resource record: {path}")
        by_job[job_id][procid] = (path, payload)
    candidates = [
        (max(path.stat().st_mtime_ns for path, _ in records.values()), job_id, records)
        for job_id, records in by_job.items()
        if set(records) == set(range(task_count))
    ]
    if not candidates:
        raise RuntimeError("no complete fresh multi-node resource allocation is retained")
    _, _, selected = max(candidates)
    return [
        {
            "relative_path": str(selected[procid][0].relative_to(output_root)),
            "sha256": file_sha256(selected[procid][0]),
        }
        for procid in range(task_count)
    ]


def _verify_multinode_resource_record_bindings(
    output_root: Path,
    payload: Mapping[str, Any],
) -> None:
    """Recheck the top-level resource-record inventory for one profile."""

    task_count = payload.get("task_count")
    workers = payload.get("workers_per_task")
    bindings = payload.get("resource_record_bindings")
    if (
        not isinstance(task_count, int)
        or task_count < 1
        or not isinstance(workers, int)
        or workers < 1
        or not isinstance(bindings, list)
        or len(bindings) != task_count
    ):
        raise RuntimeError("invalid multi-node resource-record binding inventory")
    observed_procids = set()
    observed_jobs = set()
    for binding in bindings:
        path = output_root / str(binding.get("relative_path"))
        if binding.get("sha256") != file_sha256(path):
            raise RuntimeError(f"invalid multi-node resource-record checksum: {path}")
        record = json.loads(path.read_text())
        rows = record.get("rows", ())
        if (
            record.get("action") != "multinode_work"
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("slurm_ntasks") != task_count
            or record.get("workers") != workers
            or record.get("published_count") != 1
            or record.get("reused_count") != 0
            or len(rows) != 1
            or rows[0].get("reused") is not False
        ):
            raise RuntimeError(f"invalid multi-node resource record: {path}")
        procid = record.get("slurm_procid")
        if (
            not isinstance(procid, int)
            or rows[0].get("shard_id") != f"task_{procid:04d}"
        ):
            raise RuntimeError(f"invalid multi-node resource-record task binding: {path}")
        observed_procids.add(procid)
        observed_jobs.add(str(record.get("slurm_job_id")))
    if observed_procids != set(range(task_count)) or len(observed_jobs) != 1:
        raise RuntimeError("multi-node resource records do not describe one complete allocation")


def multinode_reduce(
    phase2_root: Path,
    output_root: Path,
    *,
    workers: int,
    task_count: int,
) -> dict[str, Any]:
    """Reduce multi-node control partials and compare with a serial calculation."""

    if task_count < 1:
        raise ValueError("multi-node control requires at least one Slurm task")
    displacements, _, selected, config = _multinode_design()
    partials = {}
    for procid in range(task_count):
        task_offsets = selected[procid::task_count]
        _verify_multinode_task(
            phase2_root,
            output_root,
            procid=procid,
            ntasks=task_count,
            workers=workers,
            task_offsets=task_offsets,
            support_displacements=displacements,
            config=config,
        )
        task_root = output_root / "multinode_control" / f"task_{procid:04d}"
        partials[f"task_{procid:04d}"] = load_finite_domain_partial_npz(task_root / "partial.npz")
    expected = tuple(sorted(partials))
    missing_shard_rejected = None
    if task_count > 1:
        missing_shard_rejected = False
        try:
            reduce_finite_domain_shards(
                {key: value for key, value in partials.items() if key != expected[-1]},
                expected_shard_ids=expected,
            )
        except ValueError:
            missing_shard_rejected = True
    reduced = reduce_finite_domain_shards(partials, expected_shard_ids=expected)
    reversed_reduced = reduce_finite_domain_shards(
        dict(reversed(tuple(partials.items()))), expected_shard_ids=expected
    )
    cube = prepare_cube_for_parallel(_load_cube(phase2_root, BENCHMARK_CUBE_IDS[0]))
    serial = compute_finite_domain_structure_functions(
        cube,
        selected,
        config=config,
        q_names=Q_NAMES,
        support_displacements_ijk=displacements,
    )
    serial_vs_multinode = _relative_difference(serial.moments, reduced.moments)
    forward_vs_reverse = _relative_difference(reduced.moments, reversed_reduced.moments)
    if task_count > 1 and not missing_shard_rejected:
        raise RuntimeError("multi-node reducer accepted an intentionally omitted shard")
    _require_equivalent(serial_vs_multinode, "serial-versus-multinode control")
    _require_equivalent(forward_vs_reverse, "forward-versus-reverse multi-node reduction")
    resource_record_bindings = _select_fresh_multinode_resource_records(
        output_root,
        task_count=task_count,
        workers=workers,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operational_status": "complete",
        "validation_status": "passed",
        "task_count": task_count,
        "workers_per_task": workers,
        "cube_id": BENCHMARK_CUBE_IDS[0],
        "phase2_source": _phase2_source_identity(
            phase2_root, BENCHMARK_CUBE_IDS[0], verify_arrays=True
        ),
        "selected_offset_count": len(selected),
        "missing_shard_rejected": missing_shard_rejected,
        "serial_vs_multinode": serial_vs_multinode,
        "forward_vs_reverse_reduction": forward_vs_reverse,
        "parent_process_peak_rss_kib": _peak_rss_kib(),
        "artifact_bindings": [],
        "resource_record_bindings": resource_record_bindings,
        "source_version": _source_version(),
    }
    for procid in range(task_count):
        task_root = output_root / "multinode_control" / f"task_{procid:04d}"
        for path in (task_root / "COMPLETE.json", task_root / "partial.npz"):
            payload["artifact_bindings"].append(
                {
                    "relative_path": str(path.relative_to(output_root)),
                    "sha256": file_sha256(path),
                }
            )
    payload["artifact_bindings"].extend(resource_record_bindings)
    _verify_multinode_resource_record_bindings(output_root, payload)
    _publish_json_diagnostic(
        output_root,
        "multinode_control.json",
        "MULTINODE_CONTROL_COMPLETE.json",
        payload,
    )
    return _verify_json_diagnostic(
        output_root,
        "multinode_control.json",
        "MULTINODE_CONTROL_COMPLETE.json",
        phase2_root=phase2_root,
    )


def convergence(phase2_root: Path, output_root: Path, *, workers: int) -> dict[str, Any]:
    """Measure the bounded representative-cube design sensitivity matrix."""

    output_root.mkdir(parents=True, exist_ok=True)
    phase2_source = _phase2_source_identity(
        phase2_root, BENCHMARK_CUBE_IDS[0], verify_arrays=True
    )
    cube = prepare_cube_for_parallel(_load_cube(phase2_root, BENCHMARK_CUBE_IDS[0]))
    scenarios = []
    for bin_count in (32, 64, 128):
        scenarios.append(("bins", 2, 320, bin_count, 12, 256, PRODUCTION_SEED, "shell_local", (160, 160, 160)))
    for directions in (12, 24, 48):
        scenarios.append(("directions", 2, 320, 64, directions, 256, PRODUCTION_SEED, "shell_local", (160, 160, 160)))
    for directions in (12, 24, 48):
        scenarios.append(("directions_all_valid", 2, 320, 64, directions, 256, PRODUCTION_SEED, "all_valid_origins", (160, 160, 160)))
    for samples in (256, 1024, 2048, 4096):
        scenarios.append(("origins", 2, 320, 64, 12, samples, PRODUCTION_SEED, "shell_local", (160, 160, 160)))
    for seed in (PRODUCTION_SEED, PRODUCTION_SEED + 1):
        scenarios.append(("origin_seeds", 2, 320, 64, 12, 2048, seed, "shell_local", (160, 160, 160)))
    for ell_max in (128, 192, 256, 320):
        scenarios.append(("ell_max", 2, ell_max, 64, 12, 256, PRODUCTION_SEED, "shell_local", (160, 160, 160)))
    for block_shape in ((80, 80, 80), (160, 160, 160), (320, 320, 320)):
        scenarios.append(("blocks", 2, 320, 64, 12, 256, PRODUCTION_SEED, "shell_local", block_shape))
    for support_mode in DIAGNOSTIC_SUPPORT_MODES:
        scenarios.append(("support", 2, 320, 64, 12, 256, PRODUCTION_SEED, support_mode, (160, 160, 160)))
    for stencil_width, ell_max in ((3, 80), (3, 160), (5, 40), (5, 80)):
        scenarios.append(("stencils", stencil_width, ell_max, 32, 12, 256, PRODUCTION_SEED, "shell_local", (160, 160, 160)))
    for stencil_width, ell_max in ((3, 160), (5, 80)):
        for support_mode in ("all_valid_origins", "nested_core"):
            scenarios.append(("stencil_support", stencil_width, ell_max, 32, 12, 256, PRODUCTION_SEED, support_mode, (160, 160, 160)))
    rows = []
    for index, scenario in enumerate(scenarios):
        family, stencil_width, ell_max, bins, directions, samples, seed, support_mode, block_shape = scenario
        displacements, edges, manifest = dense_displacement_manifest(
            stencil_width=stencil_width,
            ell_max=ell_max,
            bin_count=bins,
            directions_per_bin=directions,
        )
        selected, selection_policy = _convergence_offset_subset(family, displacements, edges)
        config = FiniteDomainConfig(
            edges,
            pair_mode=support_mode,
            sample_count=samples,
            pair_batch_size=256,
            seed=seed,
            stencil_width=stencil_width,
            block_shape_kji=block_shape,
        )
        try:
            result = reduce_finite_domain_shards(
                compute_finite_domain_structure_functions_parallel(
                    cube,
                    selected,
                    config=config,
                    q_names=Q_NAMES,
                    worker_count=workers,
                    support_displacements_ijk=displacements,
                )
            )
        except ValueError as error:
            if support_mode != "nested_core":
                raise
            rows.append(
                {
                    "scenario_index": index,
                    "family": family,
                    "operational_status": "unavailable",
                    "reason": str(error),
                    "stencil_width": stencil_width,
                    "ell_max": ell_max,
                    "bin_count": bins,
                    "directions_per_bin": directions,
                    "sample_count": samples,
                    "seed": seed,
                    "support_mode": support_mode,
                    "block_shape_kji": block_shape,
                    "selection_policy": selection_policy,
                }
            )
            continue
        artifact_path = output_root / "scenarios" / f"scenario_{index:03d}.npz"
        artifact_payload = _convergence_uncertainty_payload(result)
        ell_centers = np.sqrt(edges[:-1] * edges[1:])
        artifact_payload["local_log_slope_window_3"], artifact_payload[
            "local_log_slope_window_3_support_mask"
        ] = _supported_local_log_slopes(
            ell_centers,
            result.moments,
            artifact_payload["accepted_contributing_blocks"],
            artifact_payload["accepted_effective_blocks"],
            window=3,
        )
        artifact_payload["local_log_slope_window_7"], artifact_payload[
            "local_log_slope_window_7_support_mask"
        ] = _supported_local_log_slopes(
            ell_centers,
            result.moments,
            artifact_payload["accepted_contributing_blocks"],
            artifact_payload["accepted_effective_blocks"],
            window=7,
        )
        _atomic_write_npz(artifact_path, **artifact_payload)
        rows.append(
            {
                "scenario_index": index,
                "family": family,
                "operational_status": "complete",
                "stencil_width": stencil_width,
                "ell_max": ell_max,
                "bin_count": bins,
                "directions_per_bin": directions,
                "sample_count": samples,
                "seed": seed,
                "support_mode": support_mode,
                "block_shape_kji": block_shape,
                "manifest_sha256": manifest["manifest_sha256"],
                "realized_offset_count": manifest["realized_offset_count"],
                "measured_subset_offset_count": len(selected),
                "selection_policy": selection_policy,
                "measured_subset_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                "elapsed_seconds_sum": result.elapsed_seconds,
                "sampled_pairs": int(result.sampled_pairs.sum()),
                "artifact_relative_path": str(artifact_path.relative_to(output_root)),
                "artifact_sha256": file_sha256(artifact_path),
                **_occupied_effective_block_summary(
                    artifact_payload["accepted_effective_blocks"]
                ),
                "minimum_shell_valid_fraction": float(
                    np.min(
                        np.divide(
                            result.eligible_pairs,
                            result.cube_candidate_pairs,
                            out=np.zeros_like(result.eligible_pairs, dtype=float),
                            where=result.cube_candidate_pairs > 0,
                        )
                    )
                ),
                "parent_process_peak_rss_kib": _peak_rss_kib(),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operational_status": "complete",
        "scientific_acceptance": "pending_interpretation",
        "cube_id": BENCHMARK_CUBE_IDS[0],
        "phase2_source": phase2_source,
        "workers": workers,
        "rows": rows,
        "source_version": _source_version(),
    }
    _publish_json_diagnostic(
        output_root, "convergence.json", "CONVERGENCE_COMPLETE.json", payload
    )
    return _verify_json_diagnostic(
        output_root,
        "convergence.json",
        "CONVERGENCE_COMPLETE.json",
        phase2_root=phase2_root,
    )


def summarize(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    """Publish the verified four-cube Phase 3a campaign summary."""

    verification = verify(phase2_root, output_root)
    groups = []
    for group_id in sorted(_group_rows(output_root)):
        result = load_finite_domain_partial_npz(_reduction_paths(output_root, group_id)[1])
        groups.append(
            {
                "group_id": group_id,
                "stencil_width": result.stencil_width,
                "support_mode": result.pair_mode,
                "offset_count": len(result.displacements_ijk),
                "sampled_origins": int(result.sampled_pairs.sum()),
                "minimum_valid_origin_fraction": float(
                    np.min(result.eligible_pairs / np.maximum(result.cube_candidate_pairs, 1))
                ),
                "elapsed_seconds_sum": result.elapsed_seconds,
                "elapsed_seconds_per_ell_bin": result.elapsed_seconds_per_ell_bin,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operational_status": "release_aggregation_complete",
        "scientific_acceptance": "pending_report_level_gate",
        "verification": verification,
        "groups": groups,
        "parent_process_peak_rss_kib": _peak_rss_kib(),
        "settled_release_bytes_before_summary_marker": _tree_size_accounting(output_root),
        "source_version": _source_version(),
    }
    summary_path = output_root / "phase3a_summary.json"
    _atomic_write_json(summary_path, payload)
    _atomic_write_json(
        output_root / "PHASE3A_RELEASE_COMPLETE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "release_aggregation_complete",
            "summary_sha256": file_sha256(summary_path),
            "implementation_sha256": payload["source_version"]["implementation_sha256"],
            "published_unix_seconds": time.time(),
        },
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "work", "reduce", "verify", "controls", "multinode-work", "multinode-reduce", "convergence", "summarize"))
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    actions = {
        "plan": lambda: plan(args.phase2_root, args.output_root),
        "work": lambda: work(args.phase2_root, args.output_root, workers=args.workers),
        "reduce": lambda: reduce(args.phase2_root, args.output_root),
        "verify": lambda: verify(args.phase2_root, args.output_root),
        "controls": lambda: controls(args.phase2_root, args.output_root, workers=args.workers),
        "multinode-work": lambda: multinode_work(args.phase2_root, args.output_root, workers=args.workers),
        "multinode-reduce": lambda: multinode_reduce(
            args.phase2_root, args.output_root, workers=args.workers, task_count=args.task_count
        ),
        "convergence": lambda: convergence(args.phase2_root, args.output_root, workers=args.workers),
        "summarize": lambda: summarize(args.phase2_root, args.output_root),
    }
    print(json.dumps(_json_builtin(actions[args.action]()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
