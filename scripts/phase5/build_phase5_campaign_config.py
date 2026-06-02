#!/usr/bin/env python3
"""Generate the frozen Phase 5 cross-scale extraction campaign config."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import DOMAIN_CELLS, file_sha256, required_rank_ids
from sfunctor.io.cube_extract import CubeExtractionError, load_snapshot_identity, verify_trusted_run

SCHEMA_VERSION = 1
CONFIG_PHASE = "phase5_cross_scale_extraction_campaign"
CONFIGURED_SCALES = (640, 320, 160, 80)
PROHIBITED_EXTRACTION_SCALES = (1280,)
L640_PARENT_PILOT_IDS = (
    "L640_sub00370",
    "L640_sub02822",
    "L640_sub03026",
    "L640_sub02615",
    "L640_sub03942",
    "L640_sub00957",
    "L640_sub03356",
    "L640_sub01582",
    "L640_sub00579",
    "L640_sub00886",
    "L640_sub00032",
    "L640_sub02279",
    "L640_sub01088",
    "L640_sub02297",
    "L640_sub00732",
    "L640_sub02000",
    "L640_sub02602",
    "L640_sub02249",
    "L640_sub00738",
    "L640_sub01591",
    "L640_sub01651",
)
SMOKE_ANCHOR_ROLE_SPECS = (
    ("low_dBB", "representative:low_dBB"),
    ("median_dBB", "representative:near_median_dBB"),
    ("high_dBB", "representative:high_dBB"),
    ("small_B_mean_outlier", "outlier:small_B_mean"),
    ("large_deltaB_outlier", "outlier:large_deltaB"),
    ("large_dBB_outlier", "outlier:large_dBB"),
)
L320_MATCHED_SMOKE_SPECS = (
    (1, "low", "L320_sub20629", "L640_sub02602"),
    (1, "high", "L320_sub17363", "L640_sub02297"),
    (2, "low", "L320_sub06008", "L640_sub00732"),
    (2, "high", "L320_sub01152", "L640_sub00032"),
    (3, "low", "L320_sub24421", "L640_sub03026"),
    (3, "high", "L320_sub16224", "L640_sub02000"),
)
PHASE4_PREREQUISITE_DIR = (
    Path(__file__).resolve().parents[2] / "figures" / "phase4_completion_supplement_v2"
)
PHASE4_PREREQUISITE_EXPECTED_SHA256 = {
    "phase4_completion_supplement_summary.json": (
        "80a8ed7d0eb0c27d0e0871b7acb550ba1c5941044688bf320f28152083609c38"
    ),
    "figure_manifest.json": "b2093e759dc6ab3b19069a07676e4abd39e605b69e74d392e7f882704a52ce32",
}
PHASE4_P2_REPRODUCTION_GATE_EXPECTED = {
    "excluded_metadata": "timing_and_staging_only",
    "p_value": 2.0,
    "status": "passed",
    "verification_mode": "exact_arrays_equal_nan",
    "verified_group_count": 42,
}
MAGNETIC_FIELDS = (
    "dBB",
    "B_mean",
    "deltaB",
    "B_rms",
    "B_mean_sq_over_B2_mean",
    "deltaB_sq_over_B2_mean",
)
MAGNETIC_FLAG_FIELDS = (
    "deltaB_flags",
    "dBB_flags",
    "B_mean_fraction_flags",
    "deltaB_fraction_flags",
)
CATALOG_CONTROL_FIELDS = (
    "dens_mean",
    "rho_sigma_over_mean",
    "B_rms",
    "deltaB",
    "mom1_sigma",
    "mom2_sigma",
    "mom3_sigma",
)
CATALOG_FIELDS = (
    "subvolume_id",
    "L_sub",
    "parent_L_sub",
    "parent_subvolume_id",
    "cell_i0",
    "cell_i1",
    "cell_j0",
    "cell_j1",
    "cell_k0",
    "cell_k1",
    "required_rank_count",
    "wraps_periodic_boundary",
    *MAGNETIC_FIELDS,
    *MAGNETIC_FLAG_FIELDS,
    *CATALOG_CONTROL_FIELDS,
    "catalog_validity_flags",
)


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = Path(__file__).resolve()
    hashes = {str(path.relative_to(root)): file_sha256(path)}
    return {
        "implementation_source_hashes": hashes,
        "implementation_sha256": _mapping_sha256(hashes),
    }


def _artifact_identity(path: Path, trusted_run: Path) -> dict[str, str]:
    return {
        "relative_path": str(path.relative_to(trusted_run)),
        "sha256": file_sha256(path),
    }


def _trusted_source_artifacts(trusted_run: Path) -> dict[str, Any]:
    catalogs = trusted_run / "catalogs"
    return {
        "pilot_sample": _artifact_identity(trusted_run / "analysis" / "pilot_sample.csv", trusted_run),
        "pilot_sample_metadata": _artifact_identity(
            trusted_run / "analysis" / "pilot_sample_metadata.json", trusted_run
        ),
        "rank_map": _artifact_identity(trusted_run / "cache" / "rank_map.npy", trusted_run),
        "catalogs": {
            str(scale): {
                "catalog": _artifact_identity(catalogs / f"catalog_L{scale}.npz", trusted_run),
                "manifest": _artifact_identity(
                    catalogs / f"catalog_L{scale}_manifest.json", trusted_run
                ),
                "complete": _artifact_identity(
                    catalogs / f"catalog_L{scale}.complete", trusted_run
                ),
            }
            for scale in CONFIGURED_SCALES
        },
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CubeExtractionError(f"cannot read required retained Phase 4 artifact: {path}") from error
    if not isinstance(payload, dict):
        raise CubeExtractionError(f"required retained Phase 4 artifact is not a JSON object: {path}")
    return payload


def _phase4_prerequisite_proof() -> dict[str, Any]:
    """Require the immutable retained Phase 4 supplement and exact p=2 gate."""

    artifacts: dict[str, dict[str, str]] = {}
    repo_root = Path(__file__).resolve().parents[2]
    for filename, expected_sha256 in PHASE4_PREREQUISITE_EXPECTED_SHA256.items():
        path = PHASE4_PREREQUISITE_DIR / filename
        try:
            actual_sha256 = file_sha256(path)
        except OSError as error:
            raise CubeExtractionError(f"missing required retained Phase 4 artifact: {path}") from error
        if actual_sha256 != expected_sha256:
            raise CubeExtractionError(f"required retained Phase 4 artifact changed: {path}")
        try:
            repo_relative_path = str(path.relative_to(repo_root))
        except ValueError:
            repo_relative_path = str(path.resolve())
        artifacts[filename] = {
            "repo_relative_path": repo_relative_path,
            "sha256": actual_sha256,
        }

    summary = _load_json_object(PHASE4_PREREQUISITE_DIR / "phase4_completion_supplement_summary.json")
    _load_json_object(PHASE4_PREREQUISITE_DIR / "figure_manifest.json")
    strict_verification = summary.get("strict_verification")
    gate = (
        strict_verification.get("batch_a_to_all21_batch_b_p2_reproduction")
        if isinstance(strict_verification, Mapping)
        else None
    )
    if gate != PHASE4_P2_REPRODUCTION_GATE_EXPECTED:
        raise CubeExtractionError("retained Phase 4 exact p=2 reproduction gate is absent or changed")
    return {
        "status": "passed",
        "retained_repo_artifacts": artifacts,
        "exact_p2_reproduction_gate": dict(PHASE4_P2_REPRODUCTION_GATE_EXPECTED),
        "required_result": "42/42 exact p=2 reproduction groups passed",
    }


def _load_rank_map(trusted_run: Path) -> np.ndarray:
    rank_map = np.load(trusted_run / "cache" / "rank_map.npy", allow_pickle=False)
    if rank_map.ndim != 3:
        raise CubeExtractionError(f"trusted rank map must be 3D KJI, got {rank_map.shape}")
    return rank_map


def _load_catalog_columns(trusted_run: Path, scale: int) -> dict[str, np.ndarray]:
    path = trusted_run / "catalogs" / f"catalog_L{scale}.npz"
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(CATALOG_FIELDS) - set(payload.files))
        if missing:
            raise CubeExtractionError(f"{path} is missing Phase 5 selection columns: {missing}")
        columns = {name: np.asarray(payload[name]).reshape(-1) for name in CATALOG_FIELDS}
    row_count = len(columns["subvolume_id"])
    if any(len(values) != row_count for values in columns.values()):
        raise CubeExtractionError(f"{path} has inconsistent Phase 5 selection-column lengths")
    if not np.array_equal(columns["subvolume_id"], np.arange(row_count)):
        raise CubeExtractionError(f"{path} subvolume IDs are not contiguous row indexes")
    if not np.all(columns["L_sub"] == scale):
        raise CubeExtractionError(f"{path} contains rows with the wrong L_sub")
    return columns


def _subvolume_id(cube_id: str, *, expected_scale: int) -> int:
    prefix = f"L{expected_scale}_sub"
    if not cube_id.startswith(prefix):
        raise CubeExtractionError(f"invalid L{expected_scale} cube ID: {cube_id}")
    try:
        return int(cube_id[len(prefix) :])
    except ValueError as error:
        raise CubeExtractionError(f"invalid cube ID: {cube_id}") from error


def _cube_id(scale: int, subvolume_id: int) -> str:
    return f"L{scale}_sub{subvolume_id:05d}"


def _catalog_row(columns: Mapping[str, np.ndarray], subvolume_id: int) -> dict[str, Any]:
    ids = columns["subvolume_id"]
    if subvolume_id < 0 or subvolume_id >= len(ids) or int(ids[subvolume_id]) != subvolume_id:
        raise CubeExtractionError(f"catalog has no row for subvolume ID {subvolume_id}")
    return {name: np.asarray(values)[subvolume_id].item() for name, values in columns.items()}


def _magnetic_values(row: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in MAGNETIC_FIELDS}


def _magnetic_flags(row: Mapping[str, Any]) -> dict[str, int]:
    return {name: int(row[name]) for name in MAGNETIC_FLAG_FIELDS}


def _is_magnetic_valid(row: Mapping[str, Any]) -> bool:
    return (
        all(np.isfinite(float(row[name])) for name in MAGNETIC_FIELDS)
        and all(int(row[name]) == 0 for name in MAGNETIC_FLAG_FIELDS)
        and int(row["wraps_periodic_boundary"]) == 0
    )


def _bounds(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
    return tuple(
        int(row[name])
        for name in ("cell_i0", "cell_i1", "cell_j0", "cell_j1", "cell_k0", "cell_k1")
    )


def _physical_bounds(bounds: Sequence[int]) -> list[list[float]]:
    return [
        [-0.5 + int(lower) / DOMAIN_CELLS, -0.5 + int(upper) / DOMAIN_CELLS]
        for lower, upper in zip(bounds[::2], bounds[1::2])
    ]


def _selected_rank_ids(row: Mapping[str, Any], rank_map: np.ndarray) -> tuple[int, ...]:
    bounds = _bounds(row)
    try:
        ranks = tuple(required_rank_ids(bounds, rank_map))
    except (IndexError, ValueError) as error:
        raise CubeExtractionError(f"cannot derive trusted rank IDs for bounds {bounds}") from error
    if len(ranks) != len(set(ranks)) or len(ranks) != int(row["required_rank_count"]):
        raise CubeExtractionError(f"catalog rank-count mismatch for bounds {bounds}")
    return ranks


def _catalog_parent_link(row: Mapping[str, Any]) -> dict[str, int] | None:
    scale = int(row["parent_L_sub"])
    subvolume_id = int(row["parent_subvolume_id"])
    if scale < 0 or subvolume_id < 0:
        if scale != -1 or subvolume_id != -1:
            raise CubeExtractionError("catalog parent link is only partially absent")
        return None
    return {"L_sub": scale, "subvolume_id": subvolume_id}


def _selection_payload(
    row: Mapping[str, Any],
    rank_map: np.ndarray,
    *,
    scale: int,
    root_cube_id: str,
    root_role: str,
    smoke_anchor_role: str | None,
    parent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not _is_magnetic_valid(row):
        raise CubeExtractionError(f"selected {_cube_id(scale, int(row['subvolume_id']))} is not magnetic-valid")
    bounds = _bounds(row)
    if any(upper - lower != scale for lower, upper in zip(bounds[::2], bounds[1::2])):
        raise CubeExtractionError(f"L{scale} row has wrong bounds: {bounds}")
    ranks = _selected_rank_ids(row, rank_map)
    cube_id = _cube_id(scale, int(row["subvolume_id"]))
    roles = [f"phase1_parent_role:{root_role}"]
    roles.append("phase5_l640_parent" if parent is None else "phase5_nearest_dBB_child")
    if smoke_anchor_role is not None:
        roles.append(f"phase5_smoke_anchor:{smoke_anchor_role}")
    selected_parent_link = None
    nearest_dbb = None
    if parent is not None:
        selected_parent_link = {
            "cube_id": parent["cube_id"],
            "L_sub": int(parent["L_sub"]),
            "subvolume_id": int(parent["subvolume_id"]),
        }
        nearest_dbb = {
            "reference_parent_cube_id": parent["cube_id"],
            "reference_parent_dBB": float(parent["catalog_magnetic_values"]["dBB"]),
            "absolute_dBB_difference": abs(
                float(row["dBB"]) - float(parent["catalog_magnetic_values"]["dBB"])
            ),
            "tie_break": "minimum (absolute_dBB_difference, child_subvolume_id)",
        }
    return {
        "cube_id": cube_id,
        "L_sub": scale,
        "subvolume_id": int(row["subvolume_id"]),
        "roles": roles,
        "root_L640_cube_id": root_cube_id,
        "root_phase1_role": root_role,
        "smoke_anchor": smoke_anchor_role is not None,
        "smoke_anchor_role": smoke_anchor_role,
        "selected_parent_link": selected_parent_link,
        "catalog_parent_link": _catalog_parent_link(row),
        "selection_method": (
            "trusted_phase1_frozen_L640_parent"
            if parent is None
            else "nearest_dBB_magnetic_valid_catalog_child"
        ),
        "nearest_dBB_selection": nearest_dbb,
        "bounds_ijk_half_open": list(bounds),
        "physical_bounds_x1_x2_x3": _physical_bounds(bounds),
        "shape_kji": [scale, scale, scale],
        "required_rank_count": len(ranks),
        "required_rank_ids": list(ranks),
        "catalog_magnetic_values": _magnetic_values(row),
        "catalog_magnetic_flags": _magnetic_flags(row),
        "catalog_controls": {
            name: float(row[name])
            for name in CATALOG_CONTROL_FIELDS
        },
        "catalog_validity_flags": int(row["catalog_validity_flags"]),
        "magnetic_selection_valid": True,
    }


def _pilot_rows(trusted_run: Path, rank_map: np.ndarray) -> list[dict[str, Any]]:
    path = trusted_run / "analysis" / "pilot_sample.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cube_ids = tuple(row.get("pilot_id", "") for row in rows)
    if cube_ids != L640_PARENT_PILOT_IDS:
        raise CubeExtractionError("Phase 5 requires the exact frozen 21-row Phase 1 L640 pilot")
    for row in rows:
        if int(row["L_sub"]) != 640:
            raise CubeExtractionError(f"Phase 5 parent pilot is not L640: {row['pilot_id']}")
        bounds = tuple(
            int(row[name])
            for name in ("cell_i0", "cell_i1", "cell_j0", "cell_j1", "cell_k0", "cell_k1")
        )
        derived = tuple(required_rank_ids(bounds, rank_map))
        recorded = tuple(int(rank) for rank in json.loads(row["required_rank_ids_json"]))
        if derived != tuple(sorted(recorded)) or len(derived) != int(row["required_rank_count"]):
            raise CubeExtractionError(f"trusted pilot rank IDs changed for {row['pilot_id']}")
    return rows


def _smoke_anchor_roles(pilot_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for smoke_role, phase1_role in SMOKE_ANCHOR_ROLE_SPECS:
        matching = [row["pilot_id"] for row in pilot_rows if row["role"] == phase1_role]
        if not matching:
            raise CubeExtractionError(f"trusted L640 pilot lacks smoke-anchor role {phase1_role}")
        anchors[str(matching[0])] = smoke_role
    if len(anchors) != len(SMOKE_ANCHOR_ROLE_SPECS):
        raise CubeExtractionError("Phase 5 smoke anchors must be six distinct L640 parents")
    return anchors


def _require_nested_bounds(child: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    child_bounds = child["bounds_ijk_half_open"]
    parent_bounds = parent["bounds_ijk_half_open"]
    if any(
        child_lower < parent_lower or child_upper > parent_upper
        for child_lower, child_upper, parent_lower, parent_upper in zip(
            child_bounds[::2], child_bounds[1::2], parent_bounds[::2], parent_bounds[1::2]
        )
    ):
        raise CubeExtractionError(f"{child['cube_id']} is not nested inside {parent['cube_id']}")


def _nearest_child(
    columns: Mapping[str, np.ndarray],
    rank_map: np.ndarray,
    *,
    scale: int,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    parent_subvolume_id = int(parent["subvolume_id"])
    candidate_indexes = np.flatnonzero(columns["parent_subvolume_id"] == parent_subvolume_id)
    if len(candidate_indexes) != 8 or not np.all(columns["parent_L_sub"][candidate_indexes] == scale * 2):
        raise CubeExtractionError(f"{parent['cube_id']} does not have exactly eight L{scale} children")
    valid_indexes = [
        int(index)
        for index in candidate_indexes
        if _is_magnetic_valid(_catalog_row(columns, int(index)))
    ]
    if not valid_indexes:
        raise CubeExtractionError(f"{parent['cube_id']} has no magnetic-valid L{scale} child")
    parent_dbb = float(parent["catalog_magnetic_values"]["dBB"])
    selected_index = min(
        valid_indexes,
        key=lambda index: (abs(float(columns["dBB"][index]) - parent_dbb), index),
    )
    child = _selection_payload(
        _catalog_row(columns, selected_index),
        rank_map,
        scale=scale,
        root_cube_id=str(parent["root_L640_cube_id"]),
        root_role=str(parent["root_phase1_role"]),
        smoke_anchor_role=parent.get("smoke_anchor_role"),
        parent=parent,
    )
    if child["catalog_parent_link"] != {
        "L_sub": int(parent["L_sub"]),
        "subvolume_id": parent_subvolume_id,
    }:
        raise CubeExtractionError(f"{child['cube_id']} catalog parent link is inconsistent")
    _require_nested_bounds(child, parent)
    return child


def _matched_smoke_rows(
    columns: Mapping[str, np.ndarray],
    rank_map: np.ndarray,
    *,
    parents_by_cube_id: Mapping[str, Mapping[str, Any]],
    nearest_l320: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nearest_ids = {str(row["cube_id"]) for row in nearest_l320}
    for pair_id, matched_role, cube_id, parent_cube_id in L320_MATCHED_SMOKE_SPECS:
        parent = parents_by_cube_id.get(parent_cube_id)
        if parent is None:
            raise CubeExtractionError(f"matched_smoke parent is not in the frozen L640 pilot: {parent_cube_id}")
        row = _catalog_row(columns, _subvolume_id(cube_id, expected_scale=320))
        child = _selection_payload(
            row,
            rank_map,
            scale=320,
            root_cube_id=parent_cube_id,
            root_role=str(parent["root_phase1_role"]),
            smoke_anchor_role=None,
            parent=parent,
        )
        if child["cube_id"] != cube_id:
            raise CubeExtractionError(f"matched_smoke catalog ID changed while selecting {cube_id}")
        if child["catalog_parent_link"] != {
            "L_sub": 640,
            "subvolume_id": int(parent["subvolume_id"]),
        }:
            raise CubeExtractionError(
                f"matched_smoke cube {cube_id} is not a child of stated parent {parent_cube_id}"
            )
        _require_nested_bounds(child, parent)
        child["selection_method"] = "fixed_L320_matched_smoke_observational_control"
        child["nearest_dBB_selection"] = None
        child["roles"] = [
            role
            for role in child["roles"]
            if role != "phase5_nearest_dBB_child"
        ]
        child["roles"].extend(
            (
                "phase5_fixed_L320_matched_smoke_observational_child",
                "phase5_observational_control_set:matched_smoke",
                f"phase5_matched_pair:{pair_id}:{matched_role}",
            )
        )
        child["overlaps_nearest_dBB_lineage"] = cube_id in nearest_ids
        child["observational_control_set"] = {
            "label": "matched_smoke",
            "interpretation": "separately labeled observational control set",
            "matched_pair_id": pair_id,
            "matched_role": matched_role,
            "stated_L640_parent_cube_id": parent_cube_id,
        }
        rows.append(child)
    cube_ids = [row["cube_id"] for row in rows]
    if len(cube_ids) != 6 or len(cube_ids) != len(set(cube_ids)):
        raise CubeExtractionError("Phase 5 matched_smoke must contain six distinct L320 cubes")
    return rows


def build_campaign_config(trusted_run: Path) -> dict[str, Any]:
    """Build the deterministic Phase 5 selection config from trusted Phase 1 artifacts."""

    trusted_run = trusted_run.resolve()
    phase4_prerequisite_proof = _phase4_prerequisite_proof()
    trusted_artifacts = verify_trusted_run(trusted_run)
    snapshot = load_snapshot_identity(trusted_run)
    rank_map = _load_rank_map(trusted_run)
    pilot_rows = _pilot_rows(trusted_run, rank_map)
    smoke_anchor_roles = _smoke_anchor_roles(pilot_rows)
    catalogs = {scale: _load_catalog_columns(trusted_run, scale) for scale in CONFIGURED_SCALES}
    parent_rows_by_id = {str(row["pilot_id"]): row for row in pilot_rows}

    selections_by_scale: dict[str, list[dict[str, Any]]] = {}
    parents: list[dict[str, Any]] = []
    for cube_id in L640_PARENT_PILOT_IDS:
        pilot_row = parent_rows_by_id[cube_id]
        subvolume_id = _subvolume_id(cube_id, expected_scale=640)
        parent = _selection_payload(
            _catalog_row(catalogs[640], subvolume_id),
            rank_map,
            scale=640,
            root_cube_id=cube_id,
            root_role=str(pilot_row["role"]),
            smoke_anchor_role=smoke_anchor_roles.get(cube_id),
            parent=None,
        )
        if parent["cube_id"] != cube_id:
            raise CubeExtractionError(f"trusted pilot ID changed while loading catalog: {cube_id}")
        parents.append(parent)
    selections_by_scale["640"] = parents

    l320 = [
        _nearest_child(catalogs[320], rank_map, scale=320, parent=parent)
        for parent in parents
    ]
    selections_by_scale["320"] = l320
    matched_smoke_l320 = _matched_smoke_rows(
        catalogs[320],
        rank_map,
        parents_by_cube_id={str(parent["cube_id"]): parent for parent in parents},
        nearest_l320=l320,
    )
    smoke_parents = [selection for selection in l320 if selection["smoke_anchor"]]
    for scale in (160, 80):
        smoke_parents = [
            _nearest_child(catalogs[scale], rank_map, scale=scale, parent=parent)
            for parent in smoke_parents
        ]
        selections_by_scale[str(scale)] = smoke_parents

    expected_counts = {"640": 21, "320": 21, "160": 6, "80": 6}
    actual_counts = {scale: len(rows) for scale, rows in selections_by_scale.items()}
    if actual_counts != expected_counts:
        raise CubeExtractionError(f"unexpected Phase 5 selection counts: {actual_counts}")
    for scale, rows in selections_by_scale.items():
        cube_ids = [row["cube_id"] for row in rows]
        if len(cube_ids) != len(set(cube_ids)):
            raise CubeExtractionError(f"Phase 5 L{scale} selections contain duplicate cube IDs")

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": CONFIG_PHASE,
        "status": "frozen",
        "phase4_prerequisite_proof": phase4_prerequisite_proof,
        "trusted_run": str(trusted_run),
        "trusted_phase1_artifacts": trusted_artifacts,
        "trusted_snapshot_identity": snapshot,
        "source_artifacts": _trusted_source_artifacts(trusted_run),
        "source_version": _source_version(),
        "q_names": ["B", "u"],
        "density_conventions": ["not applicable", "not applicable"],
        "sgs_channels_authorized": False,
        "lsub1280_authorized": False,
        "selection_policy": {
            "configured_scales": list(CONFIGURED_SCALES),
            "prohibited_extraction_scales": list(PROHIBITED_EXTRACTION_SCALES),
            "L640_parent_policy": "exact trusted Phase 1 frozen 21-row pilot",
            "child_policy": (
                "recursively choose one magnetic-valid child minimizing "
                "(absolute child-parent dBB difference, child subvolume ID)"
            ),
            "magnetic_validity_policy": {
                "finite_fields": list(MAGNETIC_FIELDS),
                "zero_flag_fields": list(MAGNETIC_FLAG_FIELDS),
                "require_nonperiodic_catalog_row": True,
                "catalog_validity_flags_recorded_but_not_used": (
                    "aggregate flags include unrelated standardized-moment availability"
                ),
            },
            "L320_policy": "one nearest-dBB magnetic-valid child for every L640 parent",
            "L320_matched_smoke_policy": (
                "six exact magnetic-valid nested catalog children retained as a separately "
                "labeled observational control set; IDs may overlap nearest-dBB lineages but "
                "membership remains separately labeled"
            ),
            "selection_set_overlap_policy": (
                "cube IDs must be distinct within each named set; reuse across separately "
                "labeled named sets is permitted and recorded"
            ),
            "L160_L80_policy": "nearest-dBB magnetic-valid recursion for six smoke lineages only",
            "L1280_policy": "not configured; separate explicit approval required before extraction",
        },
        "smoke_anchor_L640_cube_ids": list(smoke_anchor_roles),
        "smoke_anchor_roles": smoke_anchor_roles,
        "selection_counts_by_scale": expected_counts,
        "selection_set_counts": {
            "all": expected_counts,
            "smoke": {"640": 6, "320": 6, "160": 6, "80": 6},
            "matched_smoke": {"320": 6},
        },
        "selections_by_scale": selections_by_scale,
        "matched_smoke_L320_selections": matched_smoke_l320,
        "matched_smoke_nearest_dBB_lineage_overlap_cube_ids": [
            row["cube_id"]
            for row in matched_smoke_l320
            if row["overlaps_nearest_dBB_lineage"]
        ],
    }


def write_campaign_config(trusted_run: Path, output: Path) -> dict[str, Any]:
    """Write one new frozen config without overwriting any prior artifact."""

    if output.exists():
        raise CubeExtractionError(f"refusing to overwrite frozen Phase 5 campaign config: {output}")
    payload = build_campaign_config(trusted_run)
    _atomic_write_json(output, payload)
    return payload


def verify_campaign_config_file(trusted_run: Path, path: Path) -> dict[str, Any]:
    """Require a config to match a fresh deterministic build from trusted Phase 1."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload != build_campaign_config(trusted_run):
        raise CubeExtractionError(f"invalid or stale Phase 5 campaign config: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--trusted-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.action == "build":
        payload = write_campaign_config(args.trusted_run, args.output)
    else:
        payload = verify_campaign_config_file(args.trusted_run, args.output)
    print(
        json.dumps(
            {
                "status": "passed",
                "action": args.action,
                "output": str(args.output),
                "config_sha256": file_sha256(args.output),
                "selection_counts_by_scale": payload["selection_counts_by_scale"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
