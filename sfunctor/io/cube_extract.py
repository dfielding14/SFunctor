"""Restart-safe selected-cube extraction from Athena full primitive shards.

The Phase 2 workflow extracts only explicitly selected regions.  Output arrays
use KJI ordering: ``array[k, j, i] == field[x3, x2, x1]``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import resource
import shutil
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.phase1.cbin_tools import (
    MESH_BLOCK_CELLS,
    PRIMARY_LABELS,
    file_sha256,
    morton_rank,
    parse_binary_shard,
    read_record_fields,
    shard_identity,
    shard_path,
)
from scripts.phase1.validate_reconstruction import (
    _conserved_fields,
    cbin_region_means,
    comparison_rows,
    magnetic_diagnostics,
    retained_primary_diagnostics,
)

PRIMITIVE_FIELDS = (
    "dens",
    "velx",
    "vely",
    "velz",
    "eint",
    "bcc1",
    "bcc2",
    "bcc3",
)
PRIMARY_FIELDS = ("dens", "mom1", "mom2", "mom3", "ener", "bcc1", "bcc2", "bcc3")
BENCHMARK_CUBE_IDS = (
    "L640_sub00370",
    "L640_sub03942",
    "L640_sub00579",
    "L640_sub00738",
)
AXIS_ORDER = "KJI: array axes are (k=x3, j=x2, i=x1)"


@dataclass(frozen=True)
class CubeSelection:
    """One half-open selected region in global IJK cell coordinates."""

    cube_id: str
    bounds_ijk: tuple[int, int, int, int, int, int]
    required_rank_ids: tuple[int, ...] = ()
    role: str = ""
    lsub: int | None = None

    @property
    def shape_kji(self) -> tuple[int, int, int]:
        i0, i1, j0, j1, k0, k1 = self.bounds_ijk
        return (k1 - k0, j1 - j0, i1 - i0)

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.shape_kji))


@dataclass(frozen=True)
class BlockCopyPlan:
    """One source-block intersection copied into a target KJI cube."""

    rank_id: int
    logical_ijk: tuple[int, int, int]
    source_slices_kji: tuple[slice, slice, slice]
    target_slices_kji: tuple[slice, slice, slice]
    cell_count: int


class CubeExtractionError(RuntimeError):
    """Raised when preflight, extraction, or verification rejects a cube."""


def metadata_probe_selections() -> dict[str, CubeSelection]:
    """Deterministic header-only probes outside the four extraction cubes."""

    probes = (
        CubeSelection("probe_domain_origin", (0, 160, 0, 320, 0, 320)),
        CubeSelection("probe_domain_upper", (10080, 10240, 9920, 10240, 9920, 10240)),
        CubeSelection("probe_remote_morton_cross", (9600, 9920, 9280, 9760, 8960, 9440)),
        CubeSelection("probe_randomized_stratified_20260530", (4960, 5280, 3520, 4000, 6720, 7200)),
    )
    return {probe.cube_id: probe for probe in probes}


def stream_validation_probe_selections() -> dict[str, CubeSelection]:
    """Small real-payload probes for held-out direct-versus-cbin validation."""

    probes = (
        CubeSelection("stream_domain_origin", (0, 160, 0, 320, 0, 320)),
        CubeSelection("stream_domain_upper", (10080, 10240, 9920, 10240, 9920, 10240)),
        CubeSelection("stream_remote_morton", (9600, 9760, 9280, 9600, 8960, 9280)),
        CubeSelection("stream_randomized_stratified_20260530", (4960, 5120, 3520, 3840, 6720, 7040)),
    )
    return {probe.cube_id: probe for probe in probes}


def _slice_triplet_json(slices: Sequence[slice]) -> list[list[int]]:
    return [[int(item.start), int(item.stop)] for item in slices]


def _slice_triplet_from_json(values: Sequence[Sequence[int]]) -> tuple[slice, slice, slice]:
    return tuple(slice(int(item[0]), int(item[1])) for item in values)  # type: ignore[return-value]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _array_sha256(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _stat_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def _require_stat_fingerprint(path: Path, expected: Mapping[str, Any]) -> None:
    observed = _stat_fingerprint(path)
    if observed != expected:
        raise CubeExtractionError(f"source shard changed after preflight: {path}")


def _require_file_sha256(path: Path, expected: str, *, label: str) -> None:
    if not expected or file_sha256(path) != expected:
        raise CubeExtractionError(f"{label} fingerprint mismatch: {path}")


def _require_simple_basename(basename: str, *, label: str) -> None:
    if not basename or Path(basename).is_absolute() or Path(basename).name != basename:
        raise CubeExtractionError(f"{label} must be one simple basename: {basename}")


def _require_contained_path(path: Path, root: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise CubeExtractionError(f"{label} path escapes its declared root: {path}") from error


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_version(repo_root: Path | None = None) -> dict[str, Any]:
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    source_paths = (
        Path(__file__).resolve(),
        cwd / "scripts" / "phase2" / "run_phase2_extraction.py",
        cwd / "job_scripts" / "phase2" / "run_phase2_extract_andes.sh",
        cwd / "scripts" / "phase1" / "cbin_tools.py",
        cwd / "scripts" / "phase1" / "validate_reconstruction.py",
    )
    source_hashes = {
        str(path.relative_to(cwd)): file_sha256(path)
        for path in source_paths
    }
    aggregate = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "commit": commit,
        "dirty": dirty,
        "implementation_source_hashes": source_hashes,
        "implementation_sha256": aggregate,
    }


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _rank_path(data_root: Path, rank_id: int, basename: str) -> Path:
    return data_root / "bin" / f"rank_{rank_id:08d}" / basename


def load_pilot_selections(trusted_run: Path) -> dict[str, CubeSelection]:
    """Load the trusted Phase 1 pilot CSV into explicit cube selections."""

    verify_trusted_run(trusted_run)
    path = trusted_run / "analysis" / "pilot_sample.csv"
    rows: dict[str, CubeSelection] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cube_id = row["pilot_id"]
            if cube_id in rows:
                raise CubeExtractionError(f"duplicate pilot ID in {path}: {cube_id}")
            ranks = tuple(int(value) for value in json.loads(row["required_rank_ids_json"]))
            if len(ranks) != int(row["required_rank_count"]) or len(ranks) != len(set(ranks)):
                raise CubeExtractionError(f"{cube_id} has inconsistent required rank IDs")
            rows[cube_id] = CubeSelection(
                cube_id=cube_id,
                bounds_ijk=tuple(
                    int(row[name])
                    for name in ("cell_i0", "cell_i1", "cell_j0", "cell_j1", "cell_k0", "cell_k1")
                ),
                required_rank_ids=ranks,
                role=row.get("role", ""),
                lsub=int(row["L_sub"]),
            )
    if not rows:
        raise CubeExtractionError(f"no pilot selections found in {path}")
    return rows


def load_snapshot_identity(trusted_run: Path) -> dict[str, Any]:
    """Load the Phase 1 full-resolution snapshot identity used for validation."""

    path = trusted_run / "validation" / "VALIDATION_COMPLETE.json"
    payload = json.loads(path.read_text())
    required = ("full_resolution_basename", "target_time", "cycle")
    missing = [name for name in required if name not in payload]
    if missing:
        raise CubeExtractionError(f"{path} is missing snapshot identity keys: {missing}")
    if not payload.get("passed"):
        raise CubeExtractionError(f"{path} is not a passed validation token")
    return {
        "data_root": payload["data_root"],
        "full_resolution_basename": payload["full_resolution_basename"],
        "target_time": payload["target_time"],
        "target_cycle": payload["cycle"],
        "primary_cbin_scale": 40,
        "primary_cbin_basename": payload["primary_snapshot_identities"]["40"]["basename"],
        "primary_cbin_identity": payload["primary_snapshot_identities"]["40"],
        "full_snapshot_identity": payload["full_snapshot_identity"],
    }


def _trusted_artifact_graph(trusted_run: Path, build: Mapping[str, Any]) -> dict[str, Any]:
    catalogs_dir = trusted_run / "catalogs"
    cache_dir = trusted_run / "cache"
    return {
        "validation_token_sha256": file_sha256(
            trusted_run / "validation" / "VALIDATION_COMPLETE.json"
        ),
        "rank_map_sha256": file_sha256(cache_dir / "rank_map.npy"),
        "source_hashes": build["source_hashes"],
        "raw_caches": {
            "primary": {
                "cache_sha256": file_sha256(cache_dir / "raw_mhd_u_bcc_80.npz"),
                "manifest_sha256": file_sha256(cache_dir / "raw_mhd_u_bcc_80_manifest.json"),
            },
        },
        "catalogs": {
            str(lsub): {
                "catalog_sha256": file_sha256(catalogs_dir / f"catalog_L{lsub}.npz"),
                "manifest_sha256": file_sha256(catalogs_dir / f"catalog_L{lsub}_manifest.json"),
                "complete_sha256": file_sha256(catalogs_dir / f"catalog_L{lsub}.complete"),
            }
            for lsub in (80, 160, 320, 640, 1280)
        },
    }


def verify_trusted_run(trusted_run: Path) -> dict[str, Any]:
    """Verify the lightweight Phase 1 completion chain used by extraction."""

    validation_path = trusted_run / "validation" / "VALIDATION_COMPLETE.json"
    build_path = trusted_run / "BUILD_COMPLETE.json"
    verify_path = trusted_run / "verification" / "VERIFY_COMPLETE.json"
    analysis_manifest_path = trusted_run / "analysis" / "analysis_manifest.json"
    analysis_complete_path = trusted_run / "analysis" / "ANALYSIS_COMPLETE.json"
    required = (
        validation_path,
        build_path,
        verify_path,
        analysis_manifest_path,
        analysis_complete_path,
        trusted_run / "analysis" / "pilot_sample.csv",
        trusted_run / "analysis" / "pilot_sample_metadata.json",
        trusted_run / "cache" / "rank_map.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CubeExtractionError(f"trusted Phase 1 run is incomplete: {missing}")
    validation = json.loads(validation_path.read_text())
    build = json.loads(build_path.read_text())
    verified = json.loads(verify_path.read_text())
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    analysis_complete = json.loads(analysis_complete_path.read_text())
    if not validation.get("passed") or not verified.get("passed"):
        raise CubeExtractionError("trusted Phase 1 validation or independent verification did not pass")
    if build.get("validation_token_sha256") != file_sha256(validation_path):
        raise CubeExtractionError("trusted Phase 1 validation token changed after build")
    if verified.get("build_complete_sha256") != file_sha256(build_path):
        raise CubeExtractionError("trusted Phase 1 build marker changed after independent verification")
    if analysis_complete.get("analysis_manifest_sha256") != file_sha256(analysis_manifest_path):
        raise CubeExtractionError("trusted Phase 1 analysis manifest changed after publication")
    if analysis_complete.get("verification_marker_sha256") != file_sha256(verify_path):
        raise CubeExtractionError("trusted Phase 1 verification marker changed after analysis")
    for relative in ("pilot_sample.csv", "pilot_sample_metadata.json"):
        path = trusted_run / "analysis" / relative
        if analysis_manifest["generated_files"].get(relative) != file_sha256(path):
            raise CubeExtractionError(f"trusted Phase 1 analysis artifact changed: {path}")
    rank_map_path = trusted_run / "cache" / "rank_map.npy"
    expected_rank_hash = build["primary_cache_metadata"].get("rank_map_sha256")
    if expected_rank_hash != file_sha256(rank_map_path):
        raise CubeExtractionError("trusted Phase 1 rank map changed after build")
    graph = _trusted_artifact_graph(trusted_run, build)
    graph_hash = hashlib.sha256(
        json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_graph_hash = verified.get("artifact_graph_sha256")
    if graph_hash != expected_graph_hash or analysis_complete.get("artifact_graph_sha256") != graph_hash:
        raise CubeExtractionError("trusted Phase 1 artifact graph changed after independent verification")
    return {
        "trusted_run": str(trusted_run),
        "validation_token_sha256": file_sha256(validation_path),
        "build_complete_sha256": file_sha256(build_path),
        "verification_marker_sha256": file_sha256(verify_path),
        "analysis_manifest_sha256": file_sha256(analysis_manifest_path),
        "artifact_graph_sha256": graph_hash,
        "pilot_sample_sha256": file_sha256(trusted_run / "analysis" / "pilot_sample.csv"),
        "pilot_sample_metadata_sha256": file_sha256(
            trusted_run / "analysis" / "pilot_sample_metadata.json"
        ),
        "rank_map_sha256": expected_rank_hash,
    }


def load_rank_map(trusted_run: Path) -> np.ndarray:
    verify_trusted_run(trusted_run)
    rank_map = np.load(trusted_run / "cache" / "rank_map.npy", allow_pickle=False)
    if rank_map.ndim != 3:
        raise CubeExtractionError(f"rank map must be 3D KJI, got shape {rank_map.shape}")
    return rank_map


def _logical_ranges(
    bounds_ijk: Sequence[int], mesh_block_cells: Sequence[int]
) -> tuple[range, range, range]:
    i0, i1, j0, j1, k0, k1 = (int(value) for value in bounds_ijk)
    nx, ny, nz = (int(value) for value in mesh_block_cells)
    if min(i0, j0, k0) < 0 or i1 <= i0 or j1 <= j0 or k1 <= k0:
        raise CubeExtractionError(f"invalid half-open bounds: {tuple(bounds_ijk)}")
    return (
        range(i0 // nx, (i1 - 1) // nx + 1),
        range(j0 // ny, (j1 - 1) // ny + 1),
        range(k0 // nz, (k1 - 1) // nz + 1),
    )


def _rank_for_logical(rank_map: np.ndarray | None, rx: int, ry: int, rz: int) -> int:
    if rank_map is None:
        return int(morton_rank(rx, ry, rz))
    try:
        rank_id = int(rank_map[rz, ry, rx])
    except IndexError as error:
        raise CubeExtractionError(
            f"logical block {(rx, ry, rz)} lies outside rank map {rank_map.shape}"
        ) from error
    if rank_id < 0:
        raise CubeExtractionError(f"logical block {(rx, ry, rz)} has no rank owner")
    return rank_id


def _boxes_overlap(left: BlockCopyPlan, right: BlockCopyPlan) -> bool:
    for lhs, rhs in zip(left.target_slices_kji, right.target_slices_kji):
        if lhs.stop <= rhs.start or rhs.stop <= lhs.start:
            return False
    return True


def plan_cube_blocks(
    selection: CubeSelection,
    *,
    rank_map: np.ndarray | None = None,
    mesh_block_cells: Sequence[int] = MESH_BLOCK_CELLS,
) -> list[BlockCopyPlan]:
    """Create a disjoint, complete source-to-target KJI copy plan."""

    i0, i1, j0, j1, k0, k1 = selection.bounds_ijk
    nx, ny, nz = (int(value) for value in mesh_block_cells)
    ranges = _logical_ranges(selection.bounds_ijk, mesh_block_cells)
    blocks: list[BlockCopyPlan] = []
    for rz in ranges[2]:
        for ry in ranges[1]:
            for rx in ranges[0]:
                rank_id = _rank_for_logical(rank_map, rx, ry, rz)
                gi0, gi1 = max(i0, rx * nx), min(i1, (rx + 1) * nx)
                gj0, gj1 = max(j0, ry * ny), min(j1, (ry + 1) * ny)
                gk0, gk1 = max(k0, rz * nz), min(k1, (rz + 1) * nz)
                source = (
                    slice(gk0 - rz * nz, gk1 - rz * nz),
                    slice(gj0 - ry * ny, gj1 - ry * ny),
                    slice(gi0 - rx * nx, gi1 - rx * nx),
                )
                target = (
                    slice(gk0 - k0, gk1 - k0),
                    slice(gj0 - j0, gj1 - j0),
                    slice(gi0 - i0, gi1 - i0),
                )
                cell_count = int((gi1 - gi0) * (gj1 - gj0) * (gk1 - gk0))
                blocks.append(
                    BlockCopyPlan(
                        rank_id=rank_id,
                        logical_ijk=(rx, ry, rz),
                        source_slices_kji=source,
                        target_slices_kji=target,
                        cell_count=cell_count,
                    )
                )

    for index, left in enumerate(blocks):
        for right in blocks[index + 1 :]:
            if _boxes_overlap(left, right):
                raise CubeExtractionError(
                    f"overlapping target coverage from ranks {left.rank_id} and {right.rank_id}"
                )
    planned_cells = sum(block.cell_count for block in blocks)
    if planned_cells != selection.cell_count:
        raise CubeExtractionError(
            f"incomplete target coverage: planned {planned_cells}, expected {selection.cell_count}"
        )
    planned_ranks = tuple(sorted({block.rank_id for block in blocks}))
    if selection.required_rank_ids and planned_ranks != tuple(sorted(selection.required_rank_ids)):
        raise CubeExtractionError(
            f"{selection.cube_id} rank mismatch: planned {planned_ranks}, "
            f"trusted {tuple(sorted(selection.required_rank_ids))}"
        )
    return blocks


def _validate_geometry(
    geometry: Sequence[float],
    logical_ijk: Sequence[int],
    global_cells: Sequence[int],
    mesh_block_cells: Sequence[int],
    domain_bounds: Sequence[tuple[float, float]] | None,
) -> None:
    if len(geometry) != 6 or not np.all(np.isfinite(geometry)):
        raise CubeExtractionError(f"invalid physical geometry: {geometry}")
    for lower, upper in zip(geometry[::2], geometry[1::2]):
        if not lower < upper:
            raise CubeExtractionError(f"non-positive physical extent: {geometry}")
    if domain_bounds is None:
        return
    expected: list[float] = []
    for logical, block, total, (lower, upper) in zip(
        logical_ijk, mesh_block_cells, global_cells, domain_bounds
    ):
        width = upper - lower
        expected.extend(
            [
                lower + width * logical * block / total,
                lower + width * (logical + 1) * block / total,
            ]
        )
    if not np.allclose(geometry, expected, rtol=0.0, atol=1.0e-12):
        raise CubeExtractionError(f"physical geometry mismatch: {geometry} != {expected}")


def _physical_cube_bounds(
    bounds_ijk: Sequence[int],
    global_cells: Sequence[int],
    domain_bounds: Sequence[tuple[float, float]] | None,
) -> list[list[float]] | None:
    if domain_bounds is None:
        return None
    output: list[list[float]] = []
    for lower_cell, upper_cell, total, (lower, upper) in zip(
        bounds_ijk[::2], bounds_ijk[1::2], global_cells, domain_bounds
    ):
        width = upper - lower
        output.append(
            [
                lower + width * lower_cell / total,
                lower + width * upper_cell / total,
            ]
        )
    return output


def preflight_selection(
    selection: CubeSelection,
    *,
    data_root: Path,
    basename: str,
    rank_map: np.ndarray | None = None,
    expected_time: float | None = None,
    expected_cycle: int | None = None,
    expected_fields: Sequence[str] = PRIMITIVE_FIELDS,
    domain_bounds: Sequence[tuple[float, float]] | None = None,
    mesh_block_cells: Sequence[int] = MESH_BLOCK_CELLS,
    expected_header_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate headers, geometry, rank ownership, and coverage without payload I/O."""

    started = time.perf_counter()
    _require_simple_basename(basename, label="primitive source basename")
    blocks = plan_cube_blocks(
        selection, rank_map=rank_map, mesh_block_cells=mesh_block_cells
    )
    source_rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for block in blocks:
        path = _rank_path(data_root, block.rank_id, basename)
        if not path.is_file():
            raise CubeExtractionError(f"missing source shard: {path}")
        _require_contained_path(path, data_root, label="primitive source")
        shard = parse_binary_shard(path, expect_cbin=False)
        if len(shard.records) != 1:
            raise CubeExtractionError(f"{path} has {len(shard.records)} records; expected one")
        record = shard.records[0]
        expected_logical = (*block.logical_ijk, 0)
        if tuple(record.logical) != expected_logical:
            raise CubeExtractionError(
                f"{path} logical block {record.logical} != expected {expected_logical}"
            )
        if tuple(shard.mesh_block_cells) != tuple(mesh_block_cells):
            raise CubeExtractionError(
                f"{path} mesh-block cells {shard.mesh_block_cells} != {tuple(mesh_block_cells)}"
            )
        expected_shape = tuple(reversed(shard.mesh_block_cells))
        if tuple(record.shape_kji) != expected_shape:
            raise CubeExtractionError(
                f"{path} active KJI shape {record.shape_kji} != expected {expected_shape}"
            )
        if tuple(record.indices) != (
            0,
            shard.mesh_block_cells[0] - 1,
            0,
            shard.mesh_block_cells[1] - 1,
            0,
            shard.mesh_block_cells[2] - 1,
        ):
            raise CubeExtractionError(f"{path} has unexpected ghost-adjusted indices {record.indices}")
        _validate_geometry(
            record.geometry,
            block.logical_ijk,
            shard.full_grid_cells,
            shard.mesh_block_cells,
            domain_bounds,
        )
        observed = {
            "var_names": list(shard.var_names),
            "mesh_block_cells": list(shard.mesh_block_cells),
            "full_grid_cells": list(shard.full_grid_cells),
            "variable_dtype": shard.variable_dtype.str,
            "location_dtype": shard.location_dtype.str,
            "nghost": int(shard.nghost),
            "time": float(shard.time),
            "cycle": int(shard.cycle),
        }
        if any(
            upper > total
            for upper, total in zip(selection.bounds_ijk[1::2], shard.full_grid_cells)
        ):
            raise CubeExtractionError(
                f"{selection.cube_id} bounds {selection.bounds_ijk} exceed {shard.full_grid_cells}"
            )
        if tuple(shard.var_names) != tuple(expected_fields):
            raise CubeExtractionError(f"{path} fields {shard.var_names} != {tuple(expected_fields)}")
        if shard.variable_dtype != np.dtype("<f4"):
            raise CubeExtractionError(f"{path} variable dtype {shard.variable_dtype} != float32")
        if expected_time is not None and not np.isclose(shard.time, expected_time, atol=1.0e-12):
            raise CubeExtractionError(f"{path} time {shard.time} != {expected_time}")
        if expected_cycle is not None and shard.cycle != expected_cycle:
            raise CubeExtractionError(f"{path} cycle {shard.cycle} != {expected_cycle}")
        if baseline is None:
            baseline = observed
            if expected_header_identity is not None:
                mismatches = {
                    key: (observed.get(key), expected)
                    for key, expected in expected_header_identity.items()
                    if key in observed and observed.get(key) != expected
                }
                if mismatches:
                    raise CubeExtractionError(f"{path} snapshot identity mismatch: {mismatches}")
        elif observed != baseline:
            raise CubeExtractionError(f"{path} header differs from the first selected source shard")
        source_rows.append(
            {
                "rank_id": block.rank_id,
                "logical_ijk": list(block.logical_ijk),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "stat_fingerprint": _stat_fingerprint(path),
                "payload_offset": record.payload_offset,
                "source_slices_kji": _slice_triplet_json(block.source_slices_kji),
                "target_slices_kji": _slice_triplet_json(block.target_slices_kji),
                "cell_count": block.cell_count,
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "cube_id": selection.cube_id,
        "axis_order": AXIS_ORDER,
        "bounds_ijk_half_open": list(selection.bounds_ijk),
        "shape_kji": list(selection.shape_kji),
        "physical_bounds_x1_x2_x3": _physical_cube_bounds(
            selection.bounds_ijk,
            baseline["full_grid_cells"] if baseline is not None else (),
            domain_bounds,
        ),
        "cell_count": selection.cell_count,
        "source_file_count": len(source_rows),
        "unique_referenced_rank_ids": sorted({int(row["rank_id"]) for row in source_rows}),
        "coverage": {
            "expected_cells": selection.cell_count,
            "copied_cells": sum(int(row["cell_count"]) for row in source_rows),
            "holes": 0,
            "overlaps": 0,
        },
        "referenced_source_file_bytes": sum(row["size_bytes"] for row in source_rows),
        "expected_payload_bytes_touched": selection.cell_count
        * len(expected_fields)
        * np.dtype(np.float32).itemsize,
        "expected_materialized_payload_bytes": selection.cell_count
        * len(expected_fields)
        * np.dtype(np.float32).itemsize,
        "header_identity": baseline,
        "source_blocks": source_rows,
        "source_fingerprint_sha256": _source_fingerprint_sha256(source_rows),
        "source_plan_sha256": _source_plan_sha256(source_rows),
        "preflight_wall_seconds": time.perf_counter() - started,
    }


def _empty_raw_sums() -> dict[str, float]:
    return {
        f"{name}_{power}{suffix}": 0.0
        for name in PRIMARY_FIELDS
        for power, suffix in ((1, "st"), (2, "nd"), (3, "rd"), (4, "th"))
    }


def _accumulate_primary_raw_sums(
    totals: dict[str, float], primitive: Mapping[str, np.ndarray]
) -> None:
    conserved = _conserved_fields(primitive)
    for name, values in conserved.items():
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise CubeExtractionError(f"non-finite reconstructed primary field values in {name}")
        totals[f"{name}_1st"] += float(np.sum(array, dtype=np.float64))
        square = array * array
        totals[f"{name}_2nd"] += float(np.sum(square, dtype=np.float64))
        totals[f"{name}_3rd"] += float(np.sum(square * array, dtype=np.float64))
        totals[f"{name}_4th"] += float(np.sum(square * square, dtype=np.float64))


def _catalog_row(trusted_run: Path, selection: CubeSelection) -> dict[str, Any]:
    path = trusted_run / "catalogs" / f"catalog_L{selection.lsub}.npz"
    if selection.lsub is None or not path.is_file():
        raise CubeExtractionError(f"missing trusted catalog for {selection.cube_id}: {path}")
    subvolume_id = int(selection.cube_id.split("_sub")[-1])
    with np.load(path, allow_pickle=False) as catalog:
        ids = np.asarray(catalog["subvolume_id"]).reshape(-1)
        matches = np.flatnonzero(ids == subvolume_id)
        if len(matches) != 1:
            raise CubeExtractionError(f"{path} has {len(matches)} matches for subvolume {subvolume_id}")
        index = int(matches[0])
        row = {
            name: np.asarray(catalog[name]).reshape(-1)[index].item()
            for name in catalog.files
            if np.asarray(catalog[name]).size == ids.size
        }
    for name, expected in zip(
        ("cell_i0", "cell_i1", "cell_j0", "cell_j1", "cell_k0", "cell_k1"),
        selection.bounds_ijk,
    ):
        if name in row and int(row[name]) != expected:
            raise CubeExtractionError(f"{selection.cube_id} catalog {name}={row[name]} != {expected}")
    return row


def _comparison_row(
    name: str,
    extracted: float,
    catalog: float,
    *,
    atol: float,
    rtol: float,
    precision_limited: bool = False,
) -> dict[str, Any]:
    absolute_error = abs(extracted - catalog) if np.isfinite(extracted) and np.isfinite(catalog) else None
    allowed = atol + rtol * abs(catalog) if np.isfinite(catalog) else None
    if np.isnan(extracted) and np.isnan(catalog):
        passed = True
    elif precision_limited and not np.isfinite(catalog):
        passed = True
    else:
        passed = bool(
            np.isfinite(extracted)
            and np.isfinite(catalog)
            and absolute_error is not None
            and allowed is not None
            and absolute_error <= allowed
        )
    return {
        "quantity": name,
        "extracted": extracted,
        "catalog": catalog,
        "absolute_error": absolute_error,
        "allowed_error": allowed,
        "precision_limited": precision_limited,
        "passed": passed,
    }


def _magnetic_complements(moments: Mapping[str, float]) -> dict[str, float]:
    bmean_sq = sum(moments[f"bcc{axis}_1st"] ** 2 for axis in (1, 2, 3))
    b2mean = sum(moments[f"bcc{axis}_2nd"] for axis in (1, 2, 3))
    delta_sq = max(0.0, b2mean - bmean_sq)
    return {
        "B_mean_sq_over_B2_mean": bmean_sq / b2mean if b2mean > 0.0 else float("nan"),
        "deltaB_sq_over_B2_mean": delta_sq / b2mean if b2mean > 0.0 else float("nan"),
    }


def compare_catalog_summaries(
    trusted_run: Path,
    selection: CubeSelection,
    raw_moments: Mapping[str, float],
) -> dict[str, Any]:
    """Compare supported extracted diagnostics against the trusted Phase 1 catalog."""

    catalog = _catalog_row(trusted_run, selection)
    rows: list[dict[str, Any]] = []
    for name, extracted in raw_moments.items():
        if name in catalog:
            rows.append(
                _comparison_row(name, extracted, float(catalog[name]), atol=2.0e-5, rtol=2.0e-4)
            )

    direct = dict(magnetic_diagnostics(raw_moments))
    direct.update(_magnetic_complements(raw_moments))
    direct.update(retained_primary_diagnostics(raw_moments))
    standardized = ("_skewness", "_kurtosis")
    for name, extracted in direct.items():
        if name not in catalog or not np.isscalar(catalog[name]):
            continue
        base = name.split("_", 1)[0]
        limited = name.endswith(standardized) and bool(catalog.get(f"{base}_moment_flags", 0))
        rows.append(
            _comparison_row(
                name,
                float(extracted),
                float(catalog[name]),
                atol=5.0e-4,
                rtol=5.0e-3,
                precision_limited=limited,
            )
        )
    failures = [row for row in rows if not row["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "comparison_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
    }


def compare_cbin(
    data_root: Path,
    trusted_run: Path,
    selection: CubeSelection,
    raw_moments: Mapping[str, float],
    *,
    rank_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare extracted raw moments and supported summaries to retained cbin."""

    verify_trusted_run(trusted_run)
    snapshot = load_snapshot_identity(trusted_run)
    if data_root.resolve() != Path(snapshot["data_root"]).resolve():
        raise CubeExtractionError(
            f"data root {data_root} does not match trusted cbin oracle root {snapshot['data_root']}"
        )
    scale = int(snapshot["primary_cbin_scale"])
    cbin_identity = snapshot["primary_cbin_identity"]
    ranges = _logical_ranges(selection.bounds_ijk, MESH_BLOCK_CELLS)
    cbin_source_files: list[dict[str, Any]] = []
    for rz in ranges[2]:
        for ry in ranges[1]:
            for rx in ranges[0]:
                rank_id = _rank_for_logical(rank_map, rx, ry, rz)
                if rank_id != morton_rank(rx, ry, rz):
                    raise CubeExtractionError(
                        "selected cbin reader requires the verified production Morton rank layout"
                    )
                path = shard_path(
                    data_root,
                    "mhd_u_bcc",
                    scale,
                    rank_id,
                    snapshot["primary_cbin_basename"],
                )
                shard = parse_binary_shard(path, expect_cbin=True)
                if len(shard.records) != 1 or tuple(shard.records[0].logical) != (rx, ry, rz, 0):
                    raise CubeExtractionError(f"{path} has unexpected cbin logical metadata")
                observed = shard_identity(shard)
                mismatches = {
                    name: (observed.get(name), expected)
                    for name, expected in cbin_identity.items()
                    if observed.get(name) != expected
                }
                if mismatches:
                    raise CubeExtractionError(f"{path} cbin snapshot identity mismatch: {mismatches}")
                cbin_source_files.append(
                    {
                        "rank_id": rank_id,
                        "logical_ijk": [rx, ry, rz],
                        "path": str(path),
                        "stat_fingerprint": _stat_fingerprint(path),
                        "sha256": file_sha256(path),
                    }
                )
    reconstructed, voxels, error_bounds = cbin_region_means(
        data_root,
        "mhd_u_bcc",
        scale,
        snapshot["primary_cbin_basename"],
        selection.bounds_ijk,
        PRIMARY_LABELS,
        return_error_bounds=True,
    )
    for source in cbin_source_files:
        path = Path(source["path"])
        _require_stat_fingerprint(path, source["stat_fingerprint"])
        if file_sha256(path) != source["sha256"]:
            raise CubeExtractionError(f"cbin source shard changed while reading: {path}")
    rows = comparison_rows(
        f"{selection.cube_id}:raw",
        raw_moments,
        reconstructed,
        atol=2.0e-5,
        rtol=2.0e-4,
    )
    direct_magnetic = dict(magnetic_diagnostics(raw_moments))
    direct_magnetic.update(_magnetic_complements(raw_moments))
    reconstructed_magnetic = dict(magnetic_diagnostics(reconstructed))
    reconstructed_magnetic.update(_magnetic_complements(reconstructed))
    rows.extend(
        comparison_rows(
            f"{selection.cube_id}:magnetic",
            direct_magnetic,
            reconstructed_magnetic,
            atol=2.0e-5,
            rtol=2.0e-4,
        )
    )
    limited: set[str] = set()
    diagnostic_bounds: dict[str, float] = {}
    direct_retained = retained_primary_diagnostics(raw_moments)
    reconstructed_retained = retained_primary_diagnostics(
        reconstructed,
        raw_moment_error_bounds=error_bounds,
        precision_limited_unavailable=limited,
        diagnostic_error_bounds=diagnostic_bounds,
    )
    rows.extend(
        comparison_rows(
            f"{selection.cube_id}:retained-primary",
            direct_retained,
            reconstructed_retained,
            atol=5.0e-4,
            rtol=5.0e-3,
            precision_limited_unavailable=limited,
            paired_nan_allowed_names=limited,
            reconstructed_error_bounds=diagnostic_bounds,
        )
    )
    failures = [row for row in rows if not row["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "cbin_product": "mhd_u_bcc",
        "cbin_scale": scale,
        "cbin_basename": snapshot["primary_cbin_basename"],
        "cbin_voxels": voxels,
        "comparison_count": len(rows),
        "failure_count": len(failures),
        "precision_limited_unavailable": sorted(limited),
        "cbin_source_files": cbin_source_files,
        "rows": rows,
    }


def stream_validate_selection(
    selection: CubeSelection,
    *,
    data_root: Path,
    basename: str,
    trusted_run: Path,
    rank_map: np.ndarray | None = None,
    expected_time: float | None = None,
    expected_cycle: int | None = None,
    domain_bounds: Sequence[tuple[float, float]] | None = None,
    mesh_block_cells: Sequence[int] = MESH_BLOCK_CELLS,
    expected_header_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream one held-out primitive region and compare supported cbin quantities."""

    started = time.perf_counter()
    preflight = preflight_selection(
        selection,
        data_root=data_root,
        basename=basename,
        rank_map=rank_map,
        expected_time=expected_time,
        expected_cycle=expected_cycle,
        domain_bounds=domain_bounds,
        mesh_block_cells=mesh_block_cells,
        expected_header_identity=expected_header_identity,
    )
    _bind_primitive_source_hashes(preflight)
    raw_sums = _empty_raw_sums()
    for row in preflight["source_blocks"]:
        path = Path(row["path"])
        _require_stat_fingerprint(path, row["stat_fingerprint"])
        shard = parse_binary_shard(path, expect_cbin=False)
        if tuple(shard.records[0].logical) != (*tuple(row["logical_ijk"]), 0):
            raise CubeExtractionError(f"source shard logical metadata changed: {path}")
        primitive = read_record_fields(shard, PRIMITIVE_FIELDS, copy=False)
        source_slices = _slice_triplet_from_json(row["source_slices_kji"])
        selected = {name: np.asarray(values[source_slices]) for name, values in primitive.items()}
        _accumulate_primary_raw_sums(raw_sums, selected)
        _require_stat_fingerprint(path, row["stat_fingerprint"])
        _require_file_sha256(path, row["sha256"], label="primitive source")
        _require_stat_fingerprint(path, row["stat_fingerprint"])
    raw_moments = {name: value / selection.cell_count for name, value in raw_sums.items()}
    cbin_validation = compare_cbin(
        data_root, trusted_run, selection, raw_moments, rank_map=rank_map
    )
    return {
        "schema_version": 1,
        "status": cbin_validation["status"],
        "cube_id": selection.cube_id,
        "axis_order": AXIS_ORDER,
        "bounds_ijk_half_open": list(selection.bounds_ijk),
        "shape_kji": list(selection.shape_kji),
        "cell_count": selection.cell_count,
        "preflight": preflight,
        "cbin_validation": cbin_validation,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": _peak_rss_kib(),
    }


def _comparison_csv_text(validation: Mapping[str, Any]) -> str:
    rows = list(validation["rows"])
    fieldnames = tuple(sorted({name for row in rows for name in row}))
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _write_comparison_csv(path: Path, validation: Mapping[str, Any]) -> None:
    path.write_text(_comparison_csv_text(validation))


def _require_passed_comparisons(
    validation: Mapping[str, Any],
    *,
    label: str,
    expected_count: int | None = None,
) -> None:
    rows = validation.get("rows")
    if (
        not isinstance(rows, list)
        or validation.get("status") != "passed"
        or validation.get("failure_count") != 0
        or len(rows) != validation.get("comparison_count")
        or (expected_count is not None and len(rows) != expected_count)
        or any(row.get("passed") is not True for row in rows)
    ):
        raise CubeExtractionError(f"{label} comparisons are missing, failed, or incoherent")


def _completion_path(cube_dir: Path) -> Path:
    return cube_dir / "COMPLETE.json"


def _open_npy_writer(path: Path, shape_kji: Sequence[int]):
    handle = path.open("wb")
    np.lib.format.write_array_header_1_0(
        handle,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": tuple(int(value) for value in shape_kji),
        },
    )
    return handle


def _source_fingerprint_sha256(source_blocks: Sequence[Mapping[str, Any]]) -> str:
    source_files = [
        {
            "rank_id": int(block["rank_id"]),
            "path": str(block["path"]),
            "stat_fingerprint": block["stat_fingerprint"],
            "sha256": block.get("sha256"),
        }
        for block in source_blocks
    ]
    return _mapping_sha256({"source_files": source_files})


def _source_plan_sha256(source_blocks: Sequence[Mapping[str, Any]]) -> str:
    return _mapping_sha256(
        {
            "source_blocks": [
                {
                    "rank_id": int(block["rank_id"]),
                    "logical_ijk": list(block["logical_ijk"]),
                    "path": str(block["path"]),
                    "source_slices_kji": list(block["source_slices_kji"]),
                    "target_slices_kji": list(block["target_slices_kji"]),
                    "cell_count": int(block["cell_count"]),
                    "stat_fingerprint": block["stat_fingerprint"],
                    "sha256": block.get("sha256"),
                }
                for block in source_blocks
            ]
        }
    )


def _bind_primitive_source_hashes(preflight: dict[str, Any]) -> None:
    for block in preflight["source_blocks"]:
        path = Path(block["path"])
        _require_stat_fingerprint(path, block["stat_fingerprint"])
        block["sha256"] = file_sha256(path)
        _require_stat_fingerprint(path, block["stat_fingerprint"])
    preflight["source_fingerprint_sha256"] = _source_fingerprint_sha256(
        preflight["source_blocks"]
    )
    preflight["source_plan_sha256"] = _source_plan_sha256(preflight["source_blocks"])


def _expected_request(
    selection: CubeSelection,
    *,
    data_root: Path,
    basename: str,
    trusted_run: Path | None,
    rank_map: np.ndarray | None,
    expected_time: float | None,
    expected_cycle: int | None,
    domain_bounds: Sequence[tuple[float, float]] | None,
    mesh_block_cells: Sequence[int],
    expected_header_identity: Mapping[str, Any] | None,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cube_id": selection.cube_id,
        "role": selection.role,
        "Lsub": selection.lsub,
        "bounds_ijk_half_open": list(selection.bounds_ijk),
        "shape_kji": list(selection.shape_kji),
        "cell_count": selection.cell_count,
        "source_basename": basename,
        "source_root": str(data_root),
        "trusted_phase1_artifacts": (
            verify_trusted_run(trusted_run) if trusted_run is not None else None
        ),
        "extraction_configuration": {
            "rank_map_sha256": _array_sha256(rank_map),
            "expected_time": expected_time,
            "expected_cycle": expected_cycle,
            "domain_bounds": (
                [list(bounds) for bounds in domain_bounds] if domain_bounds is not None else None
            ),
            "mesh_block_cells": list(mesh_block_cells),
            "expected_header_identity": expected_header_identity,
            "source_fingerprint_sha256": _source_fingerprint_sha256(
                preflight["source_blocks"]
            ),
            "source_plan_sha256": _source_plan_sha256(preflight["source_blocks"]),
        },
    }


def _validate_preflight_source_blocks(
    preflight: Mapping[str, Any],
    *,
    source_root: Path,
    source_basename: str,
) -> None:
    _require_simple_basename(source_basename, label="primitive source basename")
    blocks = preflight.get("source_blocks")
    if not isinstance(blocks, list) or not blocks:
        raise CubeExtractionError("preflight has no source blocks")
    target_blocks: list[BlockCopyPlan] = []
    copied_cells = 0
    source_shape = tuple(reversed(preflight.get("header_identity", {}).get("mesh_block_cells", ())))
    target_shape = tuple(preflight.get("shape_kji", ()))
    if len(source_shape) != 3 or len(target_shape) != 3:
        raise CubeExtractionError("preflight is missing source or target shape metadata")
    for block in blocks:
        try:
            source = _slice_triplet_from_json(block["source_slices_kji"])
            target = _slice_triplet_from_json(block["target_slices_kji"])
            rank_id = int(block["rank_id"])
            logical_ijk = tuple(int(value) for value in block["logical_ijk"])
            stat_fingerprint = block["stat_fingerprint"]
            source_sha256 = block["sha256"]
            path = Path(block["path"])
        except (KeyError, TypeError, ValueError) as error:
            raise CubeExtractionError("malformed preflight source-block metadata") from error
        for name, slices, limits in (
            ("source", source, source_shape),
            ("target", target, target_shape),
        ):
            if any(
                item.start is None
                or item.stop is None
                or item.start < 0
                or item.stop <= item.start
                or item.stop > limit
                for item, limit in zip(slices, limits)
            ):
                raise CubeExtractionError(f"preflight {name} slices escape declared shape")
        source_cells = int(np.prod([item.stop - item.start for item in source]))
        target_cells = int(np.prod([item.stop - item.start for item in target]))
        if source_cells <= 0 or target_cells != source_cells or target_cells != block.get("cell_count"):
            raise CubeExtractionError("incoherent source-block slice accounting")
        copied_cells += target_cells
        expected_path = _rank_path(source_root, rank_id, source_basename)
        if path != expected_path:
            raise CubeExtractionError(f"primitive source path is not canonical: {path}")
        _require_contained_path(path, source_root, label="primitive source")
        target_blocks.append(
            BlockCopyPlan(
                rank_id=rank_id,
                logical_ijk=logical_ijk,
                source_slices_kji=source,
                target_slices_kji=target,
                cell_count=target_cells,
            )
        )
        _require_stat_fingerprint(path, stat_fingerprint)
        _require_file_sha256(path, source_sha256, label="primitive source")
        _require_stat_fingerprint(path, stat_fingerprint)
    for index, left in enumerate(target_blocks):
        for right in target_blocks[index + 1 :]:
            if _boxes_overlap(left, right):
                raise CubeExtractionError("preflight target slices overlap")
    if copied_cells != preflight.get("cell_count"):
        raise CubeExtractionError("preflight target slices contain holes")
    if _source_fingerprint_sha256(blocks) != preflight.get("source_fingerprint_sha256"):
        raise CubeExtractionError("preflight source fingerprint summary mismatch")
    if _source_plan_sha256(blocks) != preflight.get("source_plan_sha256"):
        raise CubeExtractionError("preflight source plan fingerprint mismatch")


def _verify_cbin_source_files(
    validation: Mapping[str, Any],
    *,
    source_blocks: Sequence[Mapping[str, Any]],
    data_root: Path,
    expected_basename: str,
    expected_identity: Mapping[str, Any],
) -> None:
    source_files = validation.get("cbin_source_files")
    if not isinstance(source_files, list) or not source_files:
        raise CubeExtractionError("cbin validation has no source fingerprints")
    if validation.get("cbin_product") != "mhd_u_bcc" or validation.get("cbin_scale") != 40:
        raise CubeExtractionError("cbin validation does not use the retained factor-40 primary oracle")
    if (
        validation.get("cbin_basename") != expected_basename
    ):
        raise CubeExtractionError("cbin validation basename does not match the trusted primary oracle")
    _require_simple_basename(expected_basename, label="cbin source basename")
    cbin_root = data_root / "cbin_mhd_u_bcc_40"
    expected = {
        (
            int(block["rank_id"]),
            tuple(int(value) for value in block["logical_ijk"]),
            str(
                shard_path(
                    data_root,
                    "mhd_u_bcc",
                    40,
                    int(block["rank_id"]),
                    validation["cbin_basename"],
                )
            ),
        )
        for block in source_blocks
    }
    observed = {
        (
            int(source["rank_id"]),
            tuple(int(value) for value in source["logical_ijk"]),
            str(Path(source["path"])),
        )
        for source in source_files
    }
    if observed != expected or len(source_files) != len(expected):
        raise CubeExtractionError("cbin validation source set does not match the primitive source plan")
    for source in source_files:
        path = Path(source["path"])
        _require_contained_path(path, cbin_root, label="cbin source")
        _require_stat_fingerprint(path, source["stat_fingerprint"])
        _require_file_sha256(path, source["sha256"], label="cbin source")
        _require_stat_fingerprint(path, source["stat_fingerprint"])
        shard = parse_binary_shard(path, expect_cbin=True)
        if len(shard.records) != 1 or tuple(shard.records[0].logical) != (
            *tuple(int(value) for value in source["logical_ijk"]),
            0,
        ):
            raise CubeExtractionError(f"cbin source logical metadata mismatch: {path}")
        observed_identity = shard_identity(shard)
        mismatches = {
            name: (observed_identity.get(name), expected)
            for name, expected in expected_identity.items()
            if observed_identity.get(name) != expected
        }
        if mismatches:
            raise CubeExtractionError(f"cbin source snapshot identity mismatch for {path}: {mismatches}")


def _verify_manifest_structure(
    cube_dir: Path,
    manifest: Mapping[str, Any],
    *,
    require_validations: bool,
) -> None:
    if manifest.get("completion_state") != "complete":
        raise CubeExtractionError(f"manifest completion state is not complete for {cube_dir}")
    if manifest.get("axis_order") != AXIS_ORDER:
        raise CubeExtractionError(f"unexpected axis convention for {cube_dir}")
    if manifest.get("cube_id") != cube_dir.name:
        raise CubeExtractionError(f"cube directory name does not match manifest ID for {cube_dir}")
    fields = manifest.get("output_fields", {})
    if set(fields) != set(PRIMITIVE_FIELDS):
        raise CubeExtractionError(f"unexpected output field inventory for {cube_dir}")
    relative_paths = [metadata.get("relative_path") for metadata in fields.values()]
    expected_paths = [f"fields/{field}.npy" for field in fields]
    if relative_paths != expected_paths or len(set(relative_paths)) != len(relative_paths):
        raise CubeExtractionError(f"non-canonical or duplicate output field paths for {cube_dir}")
    shape_kji = manifest.get("shape_kji")
    bounds = manifest.get("bounds_ijk_half_open")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 6
        or any(not isinstance(value, int) for value in bounds)
        or min(bounds[::2]) < 0
        or any(upper <= lower for lower, upper in zip(bounds[::2], bounds[1::2]))
    ):
        raise CubeExtractionError(f"invalid manifest half-open bounds for {cube_dir}")
    expected_shape = [bounds[5] - bounds[4], bounds[3] - bounds[2], bounds[1] - bounds[0]]
    if (
        not isinstance(shape_kji, list)
        or shape_kji != expected_shape
        or any(not isinstance(value, int) or value <= 0 for value in shape_kji)
        or int(np.prod(shape_kji)) != manifest.get("cell_count")
    ):
        raise CubeExtractionError(f"invalid manifest shape or cell count for {cube_dir}")
    preflight = manifest.get("preflight", {})
    coverage = preflight.get("coverage", {})
    if (
        preflight.get("status") != "passed"
        or preflight.get("cube_id") != manifest.get("cube_id")
        or preflight.get("axis_order") != AXIS_ORDER
        or preflight.get("bounds_ijk_half_open") != bounds
        or preflight.get("shape_kji") != shape_kji
        or preflight.get("cell_count") != manifest.get("cell_count")
        or coverage.get("expected_cells") != manifest.get("cell_count")
        or coverage.get("copied_cells") != manifest.get("cell_count")
        or coverage.get("holes") != 0
        or coverage.get("overlaps") != 0
    ):
        raise CubeExtractionError(f"incoherent preflight coverage metadata for {cube_dir}")
    trusted_artifacts = manifest.get("trusted_phase1_artifacts")
    if not isinstance(trusted_artifacts, Mapping) or not trusted_artifacts.get("trusted_run"):
        if require_validations:
            raise CubeExtractionError(f"missing trusted Phase 1 artifact binding for {cube_dir}")
        snapshot = None
    else:
        trusted_run = Path(trusted_artifacts["trusted_run"])
        if verify_trusted_run(trusted_run) != trusted_artifacts:
            raise CubeExtractionError(f"trusted Phase 1 artifact graph changed for {cube_dir}")
        snapshot = load_snapshot_identity(trusted_run)
        if (
            Path(manifest.get("source_root", "")).resolve() != Path(snapshot["data_root"]).resolve()
            or manifest.get("source_basename") != snapshot["full_resolution_basename"]
        ):
            raise CubeExtractionError(f"primitive source snapshot binding changed for {cube_dir}")
    _validate_preflight_source_blocks(
        preflight,
        source_root=Path(manifest["source_root"]),
        source_basename=manifest["source_basename"],
    )
    extraction_configuration = manifest.get("extraction_configuration", {})
    if (
        not extraction_configuration
        or extraction_configuration.get("source_fingerprint_sha256")
        != preflight.get("source_fingerprint_sha256")
        or extraction_configuration.get("source_plan_sha256")
        != preflight.get("source_plan_sha256")
    ):
        raise CubeExtractionError(f"incoherent extraction configuration for {cube_dir}")
    position_validation = manifest.get("position_validation", {})
    if (
        position_validation.get("status") != "passed"
        or not position_validation.get("sample_count")
        or position_validation.get("failure_count") != 0
        or len(position_validation.get("rows", ())) != position_validation.get("sample_count")
    ):
        raise CubeExtractionError(f"missing or failed positional validation for {cube_dir}")
    exact_validation = manifest.get("exact_validation", {})
    if (
        exact_validation.get("status") != "passed"
        or exact_validation.get("failure_count") != 0
        or exact_validation.get("comparison_count")
        != len(preflight["source_blocks"]) * len(PRIMITIVE_FIELDS)
        or len(exact_validation.get("rows", ())) != exact_validation.get("comparison_count")
        or exact_validation.get("primary_raw_moments") != manifest.get("primary_raw_moments")
    ):
        raise CubeExtractionError(f"missing or failed exhaustive source validation for {cube_dir}")
    for name in ("cbin_validation", "catalog_validation"):
        validation = manifest.get(name)
        if require_validations and validation is None:
            raise CubeExtractionError(f"{name} is missing for {cube_dir}")
        if validation is not None:
            _require_passed_comparisons(validation, label=f"{name} for {cube_dir}")
    expected_comparison_counts = {"cbin_validation": 79, "catalog_validation": 47}
    for name, expected_count in expected_comparison_counts.items():
        validation = manifest.get(name)
        if (
            require_validations
            and validation is not None
            and validation.get("comparison_count") != expected_count
        ):
            raise CubeExtractionError(
                f"{name} comparison count is not the expected {expected_count} for {cube_dir}"
            )
    if manifest.get("cbin_validation") is not None:
        if snapshot is None:
            raise CubeExtractionError(f"cbin validation lacks a trusted snapshot binding for {cube_dir}")
        _verify_cbin_source_files(
            manifest["cbin_validation"],
            source_blocks=preflight["source_blocks"],
            data_root=Path(manifest["source_root"]),
            expected_basename=snapshot["primary_cbin_basename"],
            expected_identity=snapshot["primary_cbin_identity"],
        )
    code_version = manifest.get("code_version", {})
    if (
        not code_version.get("implementation_sha256")
        or not code_version.get("implementation_source_hashes")
    ):
        raise CubeExtractionError(f"missing implementation source hashes for {cube_dir}")
    if _mapping_sha256(code_version["implementation_source_hashes"]) != code_version["implementation_sha256"]:
        raise CubeExtractionError(f"incoherent implementation source hashes for {cube_dir}")


def _verify_requested_reuse(cube_dir: Path, manifest: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = {
        name: (manifest.get(name), value)
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if mismatches:
        raise CubeExtractionError(f"completed cube request mismatch for {cube_dir}: {mismatches}")


def validate_output_positions(
    fields_dir: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare sampled output cells directly to source-shard positions."""

    outputs = {
        field: np.load(fields_dir / f"{field}.npy", mmap_mode="r", allow_pickle=False)
        for field in PRIMITIVE_FIELDS
    }
    rows: list[dict[str, Any]] = []
    for block in preflight["source_blocks"]:
        path = Path(block["path"])
        _require_stat_fingerprint(path, block["stat_fingerprint"])
        shard = parse_binary_shard(path, expect_cbin=False)
        if tuple(shard.records[0].logical) != (*tuple(block["logical_ijk"]), 0):
            raise CubeExtractionError(f"source shard logical metadata changed: {path}")
        primitive = read_record_fields(shard, PRIMITIVE_FIELDS, copy=False)
        source = _slice_triplet_from_json(block["source_slices_kji"])
        target = _slice_triplet_from_json(block["target_slices_kji"])
        shape = tuple(item.stop - item.start for item in target)
        offsets = {
            (0, 0, 0),
            tuple(size - 1 for size in shape),
            tuple(min(size - 1, offset) for size, offset in zip(shape, (1, 2, 3))),
            tuple(size // divisor for size, divisor in zip(shape, (3, 2, 4))),
            tuple(size // divisor for size, divisor in zip(shape, (2, 3, 5))),
        }
        target_samples = tuple(
            tuple(target_slice.start + offset for target_slice, offset in zip(target, sample))
            for sample in sorted(offsets)
        )
        for target_kji in target_samples:
            source_kji = tuple(
                source_slice.start + target_index - target_slice.start
                for source_slice, target_slice, target_index in zip(source, target, target_kji)
            )
            for field in PRIMITIVE_FIELDS:
                extracted = float(outputs[field][target_kji])
                original = float(primitive[field][source_kji])
                rows.append(
                    {
                        "rank_id": int(block["rank_id"]),
                        "field": field,
                        "target_kji": list(target_kji),
                        "source_kji": list(source_kji),
                        "extracted": extracted,
                        "source": original,
                        "passed": extracted == original,
                    }
                )
        _require_stat_fingerprint(path, block["stat_fingerprint"])
    failures = [row for row in rows if not row["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "sample_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
    }


def validate_output_exact(
    fields_dir: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare every output voxel to its primitive source and recompute raw moments."""

    outputs = {
        field: np.load(fields_dir / f"{field}.npy", mmap_mode="r", allow_pickle=False)
        for field in PRIMITIVE_FIELDS
    }
    raw_sums = _empty_raw_sums()
    rows: list[dict[str, Any]] = []
    for block in preflight["source_blocks"]:
        path = Path(block["path"])
        _require_stat_fingerprint(path, block["stat_fingerprint"])
        shard = parse_binary_shard(path, expect_cbin=False)
        if tuple(shard.records[0].logical) != (*tuple(block["logical_ijk"]), 0):
            raise CubeExtractionError(f"source shard logical metadata changed: {path}")
        primitive = read_record_fields(shard, PRIMITIVE_FIELDS, copy=False)
        source = _slice_triplet_from_json(block["source_slices_kji"])
        target = _slice_triplet_from_json(block["target_slices_kji"])
        selected = {field: np.asarray(primitive[field][source]) for field in PRIMITIVE_FIELDS}
        _accumulate_primary_raw_sums(raw_sums, selected)
        for field in PRIMITIVE_FIELDS:
            passed = bool(np.array_equal(outputs[field][target], selected[field]))
            rows.append(
                {
                    "rank_id": int(block["rank_id"]),
                    "field": field,
                    "source_slices_kji": block["source_slices_kji"],
                    "target_slices_kji": block["target_slices_kji"],
                    "passed": passed,
                }
            )
        _require_stat_fingerprint(path, block["stat_fingerprint"])
    failures = [row for row in rows if not row["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "comparison_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
        "primary_raw_moments": {
            name: value / preflight["cell_count"] for name, value in raw_sums.items()
        },
    }


def verify_cube_output(
    cube_dir: Path,
    *,
    verify_hashes: bool = True,
    expected_request: Mapping[str, Any] | None = None,
    require_validations: bool = False,
) -> dict[str, Any]:
    """Reject incomplete, stale, malformed, or checksum-corrupted cube outputs."""

    started = time.perf_counter()
    completion_path = _completion_path(cube_dir)
    manifest_path = cube_dir / "manifest.json"
    if not cube_dir.is_dir() or not completion_path.is_file() or not manifest_path.is_file():
        raise CubeExtractionError(f"incomplete cube output: {cube_dir}")
    completion = json.loads(completion_path.read_text())
    manifest_hash = file_sha256(manifest_path)
    if completion.get("manifest_sha256") != manifest_hash:
        raise CubeExtractionError(f"stale completion marker for {cube_dir}")
    manifest = json.loads(manifest_path.read_text())
    if completion.get("cube_id") != manifest.get("cube_id"):
        raise CubeExtractionError(f"completion cube ID mismatch for {cube_dir}")
    _verify_manifest_structure(cube_dir, manifest, require_validations=require_validations)
    if expected_request is not None:
        _verify_requested_reuse(cube_dir, manifest, expected_request)
    shape_kji = manifest["shape_kji"]
    for field, metadata in manifest["output_fields"].items():
        path = cube_dir / metadata["relative_path"]
        try:
            path.resolve().relative_to(cube_dir.resolve())
        except ValueError as error:
            raise CubeExtractionError(f"output path escapes cube directory: {path}") from error
        if not path.is_file() or path.stat().st_size != metadata["size_bytes"]:
            raise CubeExtractionError(f"missing or resized {field} output: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            metadata["shape_kji"] != shape_kji
            or metadata["dtype"] != str(np.dtype(np.float32))
            or list(array.shape) != metadata["shape_kji"]
            or str(array.dtype) != metadata["dtype"]
        ):
            raise CubeExtractionError(f"shape or dtype mismatch for {path}")
        if verify_hashes and file_sha256(path) != metadata["sha256"]:
            raise CubeExtractionError(f"checksum mismatch for {path}")
    position_validation = validate_output_positions(cube_dir / "fields", manifest["preflight"])
    if (
        position_validation["status"] != "passed"
        or position_validation["rows"] != manifest["position_validation"]["rows"]
    ):
        raise CubeExtractionError(f"recomputed positional validation mismatch for {cube_dir}")
    exact_validation = validate_output_exact(cube_dir / "fields", manifest["preflight"])
    if (
        exact_validation["status"] != "passed"
        or exact_validation["rows"] != manifest["exact_validation"]["rows"]
        or exact_validation["primary_raw_moments"] != manifest["primary_raw_moments"]
    ):
        raise CubeExtractionError(f"recomputed exhaustive source validation mismatch for {cube_dir}")
    settled_allocated_output_bytes = sum(
        (cube_dir / metadata["relative_path"]).stat().st_blocks * 512
        for metadata in manifest["output_fields"].values()
    )
    settled_allocated_storage_limit_bytes = max(
        1.5 * sum(metadata["size_bytes"] for metadata in manifest["output_fields"].values()),
        sum(metadata["size_bytes"] for metadata in manifest["output_fields"].values()) + 64 * 1024**2,
    )
    if settled_allocated_output_bytes > settled_allocated_storage_limit_bytes:
        raise CubeExtractionError(f"settled allocated-storage amplification exceeds 1.5 for {cube_dir}")
    return {
        "status": "passed",
        "cube_id": manifest["cube_id"],
        "verify_hashes": verify_hashes,
        "verify_wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": _peak_rss_kib(),
        "settled_allocated_output_bytes": settled_allocated_output_bytes,
        "settled_allocated_storage_limit_bytes": settled_allocated_storage_limit_bytes,
        "settled_storage_amplification_ratio": settled_allocated_output_bytes
        / sum(metadata["size_bytes"] for metadata in manifest["output_fields"].values()),
    }


def verify_pilot_cube_output(
    cube_dir: Path,
    selection: CubeSelection,
    *,
    data_root: Path,
    basename: str,
    trusted_run: Path,
    rank_map: np.ndarray | None = None,
    expected_time: float | None = None,
    expected_cycle: int | None = None,
    domain_bounds: Sequence[tuple[float, float]] | None = None,
    mesh_block_cells: Sequence[int] = MESH_BLOCK_CELLS,
    expected_header_identity: Mapping[str, Any] | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Strictly verify one published pilot cube against its current source request."""

    preflight = preflight_selection(
        selection,
        data_root=data_root,
        basename=basename,
        rank_map=rank_map,
        expected_time=expected_time,
        expected_cycle=expected_cycle,
        domain_bounds=domain_bounds,
        mesh_block_cells=mesh_block_cells,
        expected_header_identity=expected_header_identity,
    )
    _bind_primitive_source_hashes(preflight)
    expected_request = _expected_request(
        selection,
        data_root=data_root,
        basename=basename,
        trusted_run=trusted_run,
        rank_map=rank_map,
        expected_time=expected_time,
        expected_cycle=expected_cycle,
        domain_bounds=domain_bounds,
        mesh_block_cells=mesh_block_cells,
        expected_header_identity=expected_header_identity,
        preflight=preflight,
    )
    verification = verify_cube_output(
        cube_dir,
        verify_hashes=verify_hashes,
        expected_request=expected_request,
        require_validations=True,
    )
    manifest = json.loads((cube_dir / "manifest.json").read_text())
    expected_cbin = compare_cbin(
        data_root, trusted_run, selection, manifest["primary_raw_moments"], rank_map=rank_map
    )
    expected_catalog = compare_catalog_summaries(
        trusted_run, selection, manifest["primary_raw_moments"]
    )
    _require_passed_comparisons(expected_cbin, label=f"recomputed cbin oracle for {selection.cube_id}", expected_count=79)
    _require_passed_comparisons(
        expected_catalog,
        label=f"recomputed catalog oracle for {selection.cube_id}",
        expected_count=47,
    )
    if expected_cbin["rows"] != manifest["cbin_validation"]["rows"]:
        raise CubeExtractionError(f"recomputed cbin oracle rows changed for {selection.cube_id}")
    if expected_catalog["rows"] != manifest["catalog_validation"]["rows"]:
        raise CubeExtractionError(f"recomputed catalog oracle rows changed for {selection.cube_id}")
    return verification


def _extract_cube_unlocked(
    selection: CubeSelection,
    *,
    data_root: Path,
    basename: str,
    output_root: Path,
    rank_map: np.ndarray | None = None,
    trusted_run: Path | None = None,
    expected_time: float | None = None,
    expected_cycle: int | None = None,
    domain_bounds: Sequence[tuple[float, float]] | None = None,
    mesh_block_cells: Sequence[int] = MESH_BLOCK_CELLS,
    expected_header_identity: Mapping[str, Any] | None = None,
    clean_partial: bool = False,
    clean_incomplete: bool = False,
    keep_failed_partial: bool = False,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Extract one cube atomically and write checksummed provenance."""

    started = time.perf_counter()
    cube_dir = output_root / selection.cube_id
    partial_dir = output_root / f".{selection.cube_id}.partial"
    failure_path = output_root / f"{selection.cube_id}.failed.json"
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = preflight_selection(
        selection,
        data_root=data_root,
        basename=basename,
        rank_map=rank_map,
        expected_time=expected_time,
        expected_cycle=expected_cycle,
        domain_bounds=domain_bounds,
        mesh_block_cells=mesh_block_cells,
        expected_header_identity=expected_header_identity,
    )
    _bind_primitive_source_hashes(preflight)
    expected_request = _expected_request(
        selection,
        data_root=data_root,
        basename=basename,
        trusted_run=trusted_run,
        rank_map=rank_map,
        expected_time=expected_time,
        expected_cycle=expected_cycle,
        domain_bounds=domain_bounds,
        mesh_block_cells=mesh_block_cells,
        expected_header_identity=expected_header_identity,
        preflight=preflight,
    )
    if cube_dir.exists():
        if _completion_path(cube_dir).is_file():
            try:
                result = verify_cube_output(
                    cube_dir,
                    verify_hashes=verify_hashes,
                    expected_request=expected_request,
                    require_validations=trusted_run is not None,
                )
                result["status"] = "reused_complete_output"
                return result
            except CubeExtractionError:
                if not clean_incomplete:
                    raise
                shutil.rmtree(cube_dir)
        elif not clean_incomplete:
            raise CubeExtractionError(f"incomplete final output blocks restart: {cube_dir}")
        elif cube_dir.exists():
            shutil.rmtree(cube_dir)
    if partial_dir.exists():
        if not clean_partial:
            raise CubeExtractionError(f"partial output blocks restart: {partial_dir}")
        shutil.rmtree(partial_dir)
    failure_path.unlink(missing_ok=True)

    partial_dir.mkdir(parents=True)
    fields_dir = partial_dir / "fields"
    fields_dir.mkdir()
    raw_sums = _empty_raw_sums()
    payload_started = time.perf_counter()
    source_validation_wall_seconds = 0.0
    moment_wall_seconds = 0.0
    sequential_write_wall_seconds = 0.0
    try:
        source_blocks: list[dict[str, Any]] = []
        for row in preflight["source_blocks"]:
            path = Path(row["path"])
            _require_stat_fingerprint(path, row["stat_fingerprint"])
            shard = parse_binary_shard(path, expect_cbin=False)
            if tuple(shard.records[0].logical) != (*tuple(row["logical_ijk"]), 0):
                raise CubeExtractionError(f"source shard logical metadata changed: {path}")
            primitive = read_record_fields(shard, PRIMITIVE_FIELDS, copy=False)
            source_slices = _slice_triplet_from_json(row["source_slices_kji"])
            target_slices = _slice_triplet_from_json(row["target_slices_kji"])
            selected = {name: np.asarray(values[source_slices]) for name, values in primitive.items()}
            read_started = time.perf_counter()
            for name, values in selected.items():
                if not np.all(np.isfinite(values)):
                    raise CubeExtractionError(f"non-finite primitive values in {path}:{name}")
            source_validation_wall_seconds += time.perf_counter() - read_started
            moment_started = time.perf_counter()
            _accumulate_primary_raw_sums(raw_sums, selected)
            moment_wall_seconds += time.perf_counter() - moment_started
            _require_stat_fingerprint(path, row["stat_fingerprint"])
            _require_file_sha256(path, row["sha256"], label="primitive source")
            _require_stat_fingerprint(path, row["stat_fingerprint"])
            source_blocks.append(
                {
                    "primitive": primitive,
                    "source_slices_kji": source_slices,
                    "target_slices_kji": target_slices,
                }
            )

        write_started = time.perf_counter()
        with ExitStack() as stack:
            handles = {
                field: stack.enter_context(_open_npy_writer(fields_dir / f"{field}.npy", selection.shape_kji))
                for field in PRIMITIVE_FIELDS
            }
            for target_k in range(selection.shape_kji[0]):
                blocks = [
                    block
                    for block in source_blocks
                    if block["target_slices_kji"][0].start
                    <= target_k
                    < block["target_slices_kji"][0].stop
                ]
                occupancy = np.zeros(selection.shape_kji[1:], dtype=bool)
                for block in blocks:
                    target_kji = block["target_slices_kji"]
                    if np.any(occupancy[target_kji[1], target_kji[2]]):
                        raise CubeExtractionError(
                            f"{selection.cube_id} assembly overlap at target k={target_k}"
                        )
                    occupancy[target_kji[1], target_kji[2]] = True
                if not np.all(occupancy):
                    raise CubeExtractionError(
                        f"{selection.cube_id} assembly hole at target k={target_k}"
                    )
                for field in PRIMITIVE_FIELDS:
                    plane = np.empty(selection.shape_kji[1:], dtype=np.float32)
                    for block in blocks:
                        source_kji = block["source_slices_kji"]
                        target_kji = block["target_slices_kji"]
                        source_k = source_kji[0].start + target_k - target_kji[0].start
                        plane[target_kji[1], target_kji[2]] = block["primitive"][field][
                            source_k, source_kji[1], source_kji[2]
                        ]
                    handles[field].write(plane.tobytes(order="C"))
        sequential_write_wall_seconds = time.perf_counter() - write_started
        del source_blocks
        payload_wall_seconds = time.perf_counter() - payload_started
        position_validation = validate_output_positions(fields_dir, preflight)
        if position_validation["status"] != "passed":
            raise CubeExtractionError(
                f"{selection.cube_id} failed {position_validation['failure_count']} "
                "source-position comparisons"
            )
        raw_moments = {name: value / selection.cell_count for name, value in raw_sums.items()}
        exact_validation = validate_output_exact(fields_dir, preflight)
        if (
            exact_validation["status"] != "passed"
            or exact_validation["primary_raw_moments"] != raw_moments
        ):
            raise CubeExtractionError(
                f"{selection.cube_id} failed exhaustive source-to-output validation"
            )
        cbin_validation: dict[str, Any] | None = None
        catalog_validation: dict[str, Any] | None = None
        if trusted_run is not None:
            cbin_validation = compare_cbin(
                data_root, trusted_run, selection, raw_moments, rank_map=rank_map
            )
            _write_comparison_csv(partial_dir / "cbin_comparison.csv", cbin_validation)
            if cbin_validation["status"] != "passed":
                raise CubeExtractionError(
                    f"{selection.cube_id} failed {cbin_validation['failure_count']} cbin comparisons"
                )
            catalog_validation = compare_catalog_summaries(trusted_run, selection, raw_moments)
            _write_comparison_csv(partial_dir / "catalog_comparison.csv", catalog_validation)
            if catalog_validation["status"] != "passed":
                raise CubeExtractionError(
                    f"{selection.cube_id} failed {catalog_validation['failure_count']} catalog comparisons"
                )

        hash_started = time.perf_counter()
        output_fields: dict[str, dict[str, Any]] = {}
        for field in PRIMITIVE_FIELDS:
            path = fields_dir / f"{field}.npy"
            output_fields[field] = {
                "relative_path": str(path.relative_to(partial_dir)),
                "shape_kji": list(selection.shape_kji),
                "dtype": str(np.dtype(np.float32)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        hash_wall_seconds = time.perf_counter() - hash_started
        output_bytes = sum(item["size_bytes"] for item in output_fields.values())
        allocated_output_bytes = sum(
            (fields_dir / f"{field}.npy").stat().st_blocks * 512
            for field in PRIMITIVE_FIELDS
        )
        storage_amplification_ratio = allocated_output_bytes / output_bytes
        allocated_storage_limit = max(1.5 * output_bytes, output_bytes + 64 * 1024**2)
        if allocated_output_bytes > allocated_storage_limit:
            raise CubeExtractionError(
                f"{selection.cube_id} allocated-storage amplification "
                f"{storage_amplification_ratio:.3f} exceeds 1.5"
            )
        manifest = {
            "schema_version": 1,
            "cube_id": selection.cube_id,
            "role": selection.role,
            "Lsub": selection.lsub,
            "axis_order": AXIS_ORDER,
            "bounds_ijk_half_open": list(selection.bounds_ijk),
            "shape_kji": list(selection.shape_kji),
            "cell_count": selection.cell_count,
            "physical_bounds_x1_x2_x3": preflight["physical_bounds_x1_x2_x3"],
            "units": "AthenaK simulation code units",
            "source_basename": basename,
            "source_root": str(data_root),
            "trusted_phase1_artifacts": (
                verify_trusted_run(trusted_run) if trusted_run is not None else None
            ),
            "extraction_configuration": expected_request["extraction_configuration"],
            "preflight": preflight,
            "output_fields": output_fields,
            "primary_raw_moments": raw_moments,
            "cbin_validation": cbin_validation,
            "catalog_validation": catalog_validation,
            "position_validation": position_validation,
            "exact_validation": exact_validation,
            "performance": {
                "preflight_wall_seconds": preflight["preflight_wall_seconds"],
                "payload_wall_seconds": payload_wall_seconds,
                "source_validation_wall_seconds": source_validation_wall_seconds,
                "moment_wall_seconds": moment_wall_seconds,
                "sequential_write_wall_seconds": sequential_write_wall_seconds,
                "read_timing_note": (
                    "Physical primitive read time is not isolated from validation, moment "
                    "accumulation, SHA-256 passes, or shared-filesystem cache effects."
                ),
                "hash_wall_seconds": hash_wall_seconds,
                "total_wall_seconds_before_publish": time.perf_counter() - started,
                "logical_source_bytes_touched": preflight["expected_payload_bytes_touched"],
                "referenced_source_file_bytes": preflight["referenced_source_file_bytes"],
                "output_bytes": output_bytes,
                "allocated_output_bytes": allocated_output_bytes,
                "allocated_storage_limit_bytes": allocated_storage_limit,
                "storage_amplification_ratio": storage_amplification_ratio,
                "peak_rss_kib": _peak_rss_kib(),
            },
            "code_version": _git_version(),
            "completion_state": "complete",
        }
        _atomic_write_json(partial_dir / "manifest.json", manifest)
        manifest_hash = file_sha256(partial_dir / "manifest.json")
        partial_dir.replace(cube_dir)
        _atomic_write_json(
            _completion_path(cube_dir),
            {
                "schema_version": 1,
                "cube_id": selection.cube_id,
                "manifest_sha256": manifest_hash,
                "published_unix_seconds": time.time(),
            },
        )
        result = verify_cube_output(
            cube_dir,
            verify_hashes=verify_hashes,
            require_validations=trusted_run is not None,
        )
        result["status"] = "extracted"
        result["manifest_path"] = str(cube_dir / "manifest.json")
        return result
    except Exception as error:
        _atomic_write_json(
            failure_path,
            {
                "schema_version": 1,
                "cube_id": selection.cube_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "partial_output_retained": keep_failed_partial,
                "failed_unix_seconds": time.time(),
            },
        )
        if partial_dir.exists() and not keep_failed_partial:
            shutil.rmtree(partial_dir)
        raise


def extract_cube(
    selection: CubeSelection,
    *,
    clean_stale_lock: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract one cube while preventing concurrent output-root writers."""

    output_root = Path(kwargs["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    lock_dir = output_root / f".{selection.cube_id}.extract.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        if not clean_stale_lock:
            raise CubeExtractionError(f"cube extraction lock already exists: {lock_dir}") from error
        shutil.rmtree(lock_dir)
        lock_dir.mkdir()
    _atomic_write_json(
        lock_dir / "owner.json",
        {
            "schema_version": 1,
            "cube_id": selection.cube_id,
            "pid": os.getpid(),
            "created_unix_seconds": time.time(),
            "note": "Remove only after confirming no active extraction owns this lock.",
        },
    )
    try:
        return _extract_cube_unlocked(selection, **kwargs)
    finally:
        shutil.rmtree(lock_dir)


def write_inspection_figure(cube_dir: Path, output_path: Path) -> None:
    """Write a KJI-aware midplane figure for visual axis-order inspection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = {
        name: np.load(cube_dir / "fields" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in PRIMITIVE_FIELDS
    }
    k_mid = fields["dens"].shape[0] // 2
    density = np.asarray(fields["dens"][k_mid])
    speed = np.sqrt(sum(np.asarray(fields[name][k_mid]) ** 2 for name in ("velx", "vely", "velz")))
    bmag = np.sqrt(sum(np.asarray(fields[name][k_mid]) ** 2 for name in ("bcc1", "bcc2", "bcc3")))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, values, title in zip(
        axes, (density, speed, bmag), ("density", "speed", "magnetic-field magnitude")
    ):
        image = axis.imshow(values, origin="lower", interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("i = x1 cell")
        axis.set_ylabel("j = x2 cell")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(f"{cube_dir.name}: k-midplane (KJI storage)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _verified_stream_dependencies(
    output_root: Path,
    *,
    data_root: Path,
    basename: str,
    trusted_run: Path,
    rank_map: np.ndarray | None,
    expected_time: float | None,
    expected_cycle: int | None,
    domain_bounds: Sequence[tuple[float, float]] | None,
    expected_header_identity: Mapping[str, Any] | None,
) -> dict[str, str]:
    dependencies: dict[str, str] = {}
    snapshot = load_snapshot_identity(trusted_run)
    for cube_id, selection in stream_validation_probe_selections().items():
        json_path = output_root / "stream_validation" / f"{cube_id}.json"
        csv_path = output_root / "stream_validation" / f"{cube_id}_cbin.csv"
        if not json_path.is_file() or not csv_path.is_file():
            raise CubeExtractionError(f"missing held-out stream validation artifacts for {cube_id}")
        payload = json.loads(json_path.read_text())
        validation = payload.get("cbin_validation", {})
        if payload.get("status") != "passed" or payload.get("cube_id") != cube_id:
            raise CubeExtractionError(f"invalid held-out stream validation artifact for {cube_id}")
        _require_passed_comparisons(
            validation, label=f"held-out stream artifact for {cube_id}", expected_count=79
        )
        expected = stream_validate_selection(
            selection,
            data_root=data_root,
            basename=basename,
            trusted_run=trusted_run,
            rank_map=rank_map,
            expected_time=expected_time,
            expected_cycle=expected_cycle,
            domain_bounds=domain_bounds,
            expected_header_identity=expected_header_identity,
        )
        _require_passed_comparisons(
            expected["cbin_validation"],
            label=f"recomputed held-out stream oracle for {cube_id}",
            expected_count=79,
        )
        if (
            expected.get("status") != "passed"
            or payload.get("bounds_ijk_half_open") != list(selection.bounds_ijk)
            or payload.get("shape_kji") != list(selection.shape_kji)
            or payload["preflight"].get("source_plan_sha256")
            != expected["preflight"].get("source_plan_sha256")
            or validation["rows"] != expected["cbin_validation"]["rows"]
            or csv_path.read_bytes() != _comparison_csv_text(validation).encode()
        ):
            raise CubeExtractionError(f"held-out stream validation semantics changed for {cube_id}")
        _validate_preflight_source_blocks(
            payload["preflight"],
            source_root=data_root,
            source_basename=basename,
        )
        _verify_cbin_source_files(
            validation,
            source_blocks=payload["preflight"]["source_blocks"],
            data_root=data_root,
            expected_basename=snapshot["primary_cbin_basename"],
            expected_identity=snapshot["primary_cbin_identity"],
        )
        dependencies[str(json_path.relative_to(output_root))] = file_sha256(json_path)
        dependencies[str(csv_path.relative_to(output_root))] = file_sha256(csv_path)
    return dependencies


def summarize_benchmark(
    output_root: Path,
    cube_ids: Sequence[str] = BENCHMARK_CUBE_IDS,
    *,
    campaign_count: int = 21,
    trusted_run: Path | None = None,
    data_root: Path | None = None,
    basename: str | None = None,
    rank_map: np.ndarray | None = None,
    expected_time: float | None = None,
    expected_cycle: int | None = None,
    domain_bounds: Sequence[tuple[float, float]] | None = None,
    expected_header_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize four-cube measurements and scale them to the Phase 2 campaign."""

    summary_json_path = output_root / "phase2_benchmark_summary.json"
    summary_markdown_path = output_root / "phase2_benchmark_summary.md"
    completion_path = output_root / "PHASE2_BENCHMARK_COMPLETE.json"
    for path in (summary_json_path, summary_markdown_path, completion_path):
        path.unlink(missing_ok=True)
    if tuple(cube_ids) != BENCHMARK_CUBE_IDS or campaign_count != 21:
        raise CubeExtractionError("official Phase 2 summary requires the exact four benchmark IDs and 21-cube campaign")
    if trusted_run is None or data_root is None or basename is None:
        raise CubeExtractionError("official Phase 2 summary requires trusted Phase 1 artifacts and source paths")
    selections = load_pilot_selections(trusted_run)
    verification: list[dict[str, Any]] = []
    for cube_id in cube_ids:
        verification.append(
            verify_pilot_cube_output(
                output_root / cube_id,
                selections[cube_id],
                data_root=data_root,
                basename=basename,
                trusted_run=trusted_run,
                rank_map=rank_map,
                expected_time=expected_time,
                expected_cycle=expected_cycle,
                domain_bounds=domain_bounds,
                expected_header_identity=expected_header_identity,
                verify_hashes=True,
            )
        )
    manifests = [
        json.loads((output_root / cube_id / "manifest.json").read_text()) for cube_id in cube_ids
    ]
    implementation_hashes = {
        manifest["code_version"]["implementation_sha256"] for manifest in manifests
    }
    if len(implementation_hashes) != 1:
        raise CubeExtractionError("benchmark cubes were not produced by one implementation")
    stream_dependencies = _verified_stream_dependencies(
        output_root,
        data_root=data_root,
        basename=basename,
        trusted_run=trusted_run,
        rank_map=rank_map,
        expected_time=expected_time,
        expected_cycle=expected_cycle,
        domain_bounds=domain_bounds,
        expected_header_identity=expected_header_identity,
    )
    cube_completion_dependencies = {
        str((_completion_path(output_root / cube_id)).relative_to(output_root)): file_sha256(
            _completion_path(output_root / cube_id)
        )
        for cube_id in cube_ids
    }
    verification_by_cube = {item["cube_id"]: item for item in verification}
    for manifest in manifests:
        cube_id = manifest["cube_id"]
        _write_comparison_csv(output_root / cube_id / "cbin_comparison.csv", manifest["cbin_validation"])
        _write_comparison_csv(
            output_root / cube_id / "catalog_comparison.csv", manifest["catalog_validation"]
        )
        _atomic_write_json(
            output_root / "restart_checks" / f"{cube_id}.json",
            verification_by_cube[cube_id],
        )
        write_inspection_figure(
            output_root / cube_id,
            output_root / "inspection" / f"{cube_id}_midplanes.png",
        )
    report_artifact_paths = [
        relative
        for cube_id in cube_ids
        for relative in (
            f"{cube_id}/cbin_comparison.csv",
            f"{cube_id}/catalog_comparison.csv",
            f"restart_checks/{cube_id}.json",
            f"inspection/{cube_id}_midplanes.png",
        )
    ]
    missing_report_artifacts = [
        relative for relative in report_artifact_paths if not (output_root / relative).is_file()
    ]
    if missing_report_artifacts:
        raise CubeExtractionError(f"missing report artifact dependencies: {missing_report_artifacts}")
    report_artifact_dependencies = {
        relative: file_sha256(output_root / relative) for relative in report_artifact_paths
    }
    additive_keys = (
        "preflight_wall_seconds",
        "payload_wall_seconds",
        "source_validation_wall_seconds",
        "moment_wall_seconds",
        "sequential_write_wall_seconds",
        "hash_wall_seconds",
        "total_wall_seconds_before_publish",
        "logical_source_bytes_touched",
        "referenced_source_file_bytes",
        "output_bytes",
        "allocated_output_bytes",
    )
    measured = {
        key: float(sum(manifest["performance"][key] for manifest in manifests))
        for key in additive_keys
    }
    factor = campaign_count / len(manifests)
    forecast = {key: value * factor for key, value in measured.items()}
    settled_allocated_output_bytes = float(
        sum(item["settled_allocated_output_bytes"] for item in verification)
    )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "benchmark_cube_ids": list(cube_ids),
        "measured_cube_count": len(manifests),
        "campaign_cube_count": campaign_count,
        "forecast_scale_factor": factor,
        "measured_totals": measured,
        "forecast_campaign_totals": forecast,
        "strict_verification": verification,
        "artifact_dependencies": {
            "cube_completion_sha256": cube_completion_dependencies,
            "stream_validation_sha256": stream_dependencies,
            "report_artifact_sha256": report_artifact_dependencies,
        },
        "trusted_phase1_artifacts": verify_trusted_run(trusted_run),
        "implementation_sha256": next(iter(implementation_hashes)),
        "cache_state": "first-touch within allocation; shared filesystem cache state uncontrolled",
        "peak_rss_kib_max": max(
            int(manifest["performance"]["peak_rss_kib"]) for manifest in manifests
        ),
        "storage_amplification_ratio_max": max(
            float(manifest["performance"]["storage_amplification_ratio"])
            for manifest in manifests
        ),
        "settled_allocated_output_bytes": settled_allocated_output_bytes,
        "forecast_settled_allocated_output_bytes": settled_allocated_output_bytes * factor,
        "settled_storage_amplification_ratio_max": max(
            float(item["settled_storage_amplification_ratio"]) for item in verification
        ),
        "forecast_node_hours_before_publish": forecast["total_wall_seconds_before_publish"]
        / 3600.0,
    }
    _atomic_write_json(summary_json_path, summary)
    lines = [
        "# Phase 2 four-cube benchmark summary",
        "",
        f"Measured cubes: `{len(manifests)}`. Forecast campaign cubes: `{campaign_count}`.",
        "",
        "| Metric | Four-cube measurement | 21-cube linear forecast |",
        "| --- | ---: | ---: |",
    ]
    for key in additive_keys:
        lines.append(f"| `{key}` | {measured[key]:.6g} | {forecast[key]:.6g} |")
    lines.extend(
        [
            "",
            f"Peak extractor RSS: `{summary['peak_rss_kib_max']} KiB`.",
            f"Maximum allocated-storage amplification: `{summary['storage_amplification_ratio_max']:.6f}`.",
            f"Settled allocated output bytes: `{summary['settled_allocated_output_bytes']:.6g}`.",
            f"Forecast settled allocated output bytes: `{summary['forecast_settled_allocated_output_bytes']:.6g}`.",
            f"Maximum settled allocated-storage amplification: `{summary['settled_storage_amplification_ratio_max']:.6f}`.",
            f"Linear extraction node-hour forecast before publish: `{summary['forecast_node_hours_before_publish']:.6f}`.",
        ]
    )
    summary_markdown_path.write_text("\n".join(lines) + "\n")
    _atomic_write_json(
        completion_path,
        {
            "schema_version": 1,
            "status": "passed",
            "benchmark_summary_sha256": file_sha256(summary_json_path),
            "benchmark_summary_markdown_sha256": file_sha256(summary_markdown_path),
            "cube_ids": list(cube_ids),
            "implementation_sha256": summary["implementation_sha256"],
            "trusted_phase1_artifacts": summary["trusted_phase1_artifacts"],
            "cube_completion_sha256": cube_completion_dependencies,
            "stream_validation_sha256": stream_dependencies,
            "report_artifact_sha256": report_artifact_dependencies,
            "strict_array_reverification_required_after_artifact_changes": True,
        },
    )
    return summary
