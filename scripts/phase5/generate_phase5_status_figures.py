#!/usr/bin/env python3
"""Generate an immutable, hash-bound Phase 5 cross-scale report package.

This is a report-only adapter for retained sampler releases.  It does not plan
or run a campaign, discover additional outputs, or publish fitted exponents.
The explicit campaign config is the bounded report scope and must bind Phase 1
catalog artifacts plus the selected cubes at each environmental scale.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from sfunctor.analysis.phase3a import load_finite_domain_partial_npz


SCHEMA_VERSION = 1
SUMMARY_FILENAME = "phase5_cross_scale_report_summary.json"
HASH_MANIFEST_FILENAME = "phase5_hash_manifest.json"
LEDGER_SNAPSHOT_FILENAME = "phase5_compute_ledger_summary_snapshot.md"
DECISION_SNAPSHOT_FILENAME = "phase5_execution_decision_snapshot.json"
SELECTION_JSON_FILENAME = "phase5_scale_selection_lineage_environment_census.json"
SELECTION_CSV_FILENAME = "phase5_scale_selection_lineage_environment_census.csv"
SHELL_JSON_FILENAME = "phase5_shell_support_effective_block_diagnostics.json"
SHELL_CSV_FILENAME = "phase5_shell_support_effective_block_diagnostics.csv"
POLICY_JSON_FILENAME = "phase5_all_valid_vs_shell_local_policy_factor.json"
POLICY_CSV_FILENAME = "phase5_all_valid_vs_shell_local_policy_factor.csv"
SLOPE_JSON_FILENAME = "phase5_local_slope_uncertainty_across_scale.json"
SLOPE_CSV_FILENAME = "phase5_local_slope_uncertainty_across_scale.csv"
STENCIL_JSON_FILENAME = "phase5_2point_vs_3point_comparison.json"
STENCIL_CSV_FILENAME = "phase5_2point_vs_3point_comparison.csv"
RUNTIME_JSON_FILENAME = "phase5_runtime_storage_summary.json"
RUNTIME_CSV_FILENAME = "phase5_runtime_storage_summary.csv"
OUTLIER_JSON_FILENAME = "phase5_outlier_panel.json"
OUTLIER_CSV_FILENAME = "phase5_outlier_panel.csv"

PRIMARY_SUPPORT_MODE = "all_valid_origins"
OVERLAY_SUPPORT_MODE = "shell_local"
CURVE_SHELL_LOCAL_MINIMUM = 0.05
SLOPE_SHELL_LOCAL_MINIMUM = 0.10
MINIMUM_ACCEPTED_MEASUREMENTS = 2
MINIMUM_CONTRIBUTING_BLOCKS = 2
MINIMUM_EFFECTIVE_BLOCKS = 8.0
MINIMUM_VALID_BOOTSTRAP_FRACTION = 0.90
SCIENCE_DIRECTIONS = ("parallel", "xi", "lambda")
ENVIRONMENT_FIELDS = (
    "dBB",
    "B_mean",
    "deltaB",
    "B_rms",
    "B_mean_sq_over_B2_mean",
    "deltaB_sq_over_B2_mean",
)
FIGURE_FILENAMES = (
    "phase5_workflow_schematic.png",
    "phase5_scale_selection_lineage_environment_census.png",
    "phase5_representative_p2_rooted_sf_curves.png",
    "phase5_support_fraction_vs_ell.png",
    "phase5_effective_block_retention_diagnostics.png",
    "phase5_policy_factor_by_scale_order.png",
    "phase5_local_slope_uncertainty_across_scale.png",
    "phase5_2point_vs_3point_comparison.png",
    "phase5_runtime_storage_summary.png",
    "phase5_outlier_panel.png",
)
MATRIX_AVAILABILITY_BY_SCALE = {
    320: ("baseline", "orders", "3point", "5point"),
    160: ("baseline", "orders", "3point"),
    80: ("baseline", "orders"),
}
DEFAULT_PHASE4_BASELINE_REPRESENTATIVES = (
    ("L640_sub00370", "representative:low_dBB"),
    ("L640_sub03942", "representative:near_median_dBB"),
    ("L640_sub00579", "representative:high_dBB"),
    ("L640_sub00738", "outlier:weak_mean_field"),
)


@dataclass(frozen=True)
class Selection:
    """One explicit scale-specific catalog selection frozen by the report config."""

    release_label: str
    selection_set: str
    matrix: str
    cube_id: str
    l_sub: int
    role: str
    physical_region_id: str
    matched_group: str | None
    observational_control_set: dict[str, Any] | None
    representative: bool
    outlier: bool
    environment: dict[str, float]
    source_row: dict[str, Any]


@dataclass(frozen=True)
class Group:
    """One verified reduction and uncertainty publication."""

    release_label: str
    scale: int
    cube_id: str
    stencil_width: int
    support_mode: str
    group_id: str
    result: Any
    uncertainty: dict[str, np.ndarray]
    reduction_marker: dict[str, Any]


@dataclass(frozen=True)
class VerifiedRelease:
    """One retained sampler release verified through its marker graph."""

    label: str
    root: Path
    scale: int
    selection_set: str
    matrix: str
    campaign: dict[str, Any]
    summary: dict[str, Any]
    phase2_sources: dict[str, Any]
    groups: dict[tuple[str, int, str], Group]
    verification: dict[str, Any]


class InputHashes:
    """Collect direct retained input bindings for the published report."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}

    def add(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"required retained artifact is missing: {resolved}")
        observed = file_sha256(resolved)
        self._hashes[str(resolved)] = observed
        return observed

    def bind(self, path: Path, expected: Any) -> str:
        resolved = path.resolve()
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"invalid retained SHA-256 binding for {resolved}")
        observed = self.add(resolved)
        if observed != expected:
            raise RuntimeError(
                f"retained artifact checksum mismatch: {resolved}: "
                f"expected {expected}, observed {observed}"
            )
        return observed

    def add_many(self, paths: Iterable[Path]) -> None:
        for path in paths:
            self.add(path)

    def as_dict(self) -> dict[str, str]:
        return dict(sorted(self._hashes.items()))


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _native(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required retained JSON artifact is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"retained JSON artifact must contain an object: {path}")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(_native(payload), indent=2, sort_keys=True) + "\n")


def _csv_cell(value: Any) -> Any:
    value = _native(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({name for row in rows for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: _csv_cell(value) for name, value in row.items()} for row in rows
        )
    temporary.replace(path)


def _write_table_pair(
    output_dir: Path,
    *,
    json_filename: str,
    csv_filename: str,
    description: str,
    rows: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    json_path = output_dir / json_filename
    csv_path = output_dir / csv_filename
    _atomic_write_json(
        json_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "description": description,
            "row_count": len(rows),
            **dict(extra or {}),
            "rows": list(rows),
        },
    )
    _atomic_write_csv(csv_path, rows)
    return {
        "json_relative_path": json_path.name,
        "json_sha256": file_sha256(json_path),
        "csv_relative_path": csv_path.name,
        "csv_sha256": file_sha256(csv_path),
        "row_count": len(rows),
    }


def _contained(root: Path, path: Path, *, label: str) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes retained root {root}: {path}") from error
    return path


def _relative_file(root: Path, relative: Any, *, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
    ):
        raise RuntimeError(f"{label} must use one non-empty relative path")
    path = _contained(root, root / relative, label=label)
    if not path.is_file():
        raise RuntimeError(f"required {label} is missing: {path}")
    return path


def _finite_float(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be finite numeric data") from error
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} must be finite numeric data")
    return numeric


def _centers(edges: Any) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise RuntimeError("reduction has invalid separation-bin edges")
    if np.all(values > 0.0):
        return np.sqrt(values[:-1] * values[1:])
    return 0.5 * (values[:-1] + values[1:])


def _support_fraction(result: Any) -> np.ndarray:
    return np.divide(
        result.eligible_pairs,
        result.cube_candidate_pairs,
        out=np.zeros_like(result.eligible_pairs, dtype=float),
        where=np.asarray(result.cube_candidate_pairs) > 0,
    )


def _moment_index(
    result: Any,
    q_name: str,
    direction: str,
    *,
    p_value: float,
) -> tuple[int, int, int, int, int]:
    geometry = "pair_local" if "pair_local" in result.geometry_names else result.geometry_names[0]
    measurement = (
        "perpendicular"
        if "perpendicular" in result.measurement_names
        else result.measurement_names[0]
    )
    return (
        result.q_names.index(q_name),
        result.geometry_names.index(geometry),
        result.measurement_names.index(measurement),
        result.direction_names.index(direction),
        result.p_values.index(p_value),
    )


def _uncertainty_metadata(uncertainty: Mapping[str, np.ndarray]) -> dict[str, Any]:
    metadata = uncertainty["metadata_json"]
    if metadata.shape != () or metadata.dtype.kind not in "SU":
        raise RuntimeError("uncertainty metadata_json must be one string scalar")
    payload = json.loads(str(metadata.item()))
    if not isinstance(payload, dict):
        raise RuntimeError("uncertainty metadata_json must decode to one object")
    return payload


def _curve_support_mask(group: Group, index: tuple[int, ...]) -> np.ndarray:
    uncertainty = group.uncertainty
    metadata = _uncertainty_metadata(uncertainty)
    n_resamples = int(metadata.get("bootstrap_n_resamples", 200))
    minimum_valid = int(math.ceil(MINIMUM_VALID_BOOTSTRAP_FRACTION * n_resamples))
    return (
        np.isfinite(group.result.moments[index])
        & (group.result.counts[index] >= MINIMUM_ACCEPTED_MEASUREMENTS)
        & (
            uncertainty["accepted_contributing_blocks"][index]
            >= MINIMUM_CONTRIBUTING_BLOCKS
        )
        & (uncertainty["accepted_effective_blocks"][index] >= MINIMUM_EFFECTIVE_BLOCKS)
        & (uncertainty["valid_bootstrap_resamples"][index] >= minimum_valid)
        & np.isfinite(uncertainty["block_bootstrap_interval_low"][index])
        & np.isfinite(uncertainty["block_bootstrap_interval_high"][index])
    )


def _source_identity_coherent(source_version: Any, *, label: str) -> str:
    if not isinstance(source_version, Mapping):
        raise RuntimeError(f"{label} lacks a source-version object")
    implementation = source_version.get("implementation_sha256")
    if not isinstance(implementation, str) or len(implementation) != 64:
        raise RuntimeError(f"{label} lacks a valid implementation SHA-256")
    hashes = source_version.get("implementation_source_hashes")
    if hashes is not None and (
        not isinstance(hashes, Mapping) or _mapping_sha256(hashes) != implementation
    ):
        raise RuntimeError(f"{label} has an incoherent implementation source map")
    return implementation


def _binding_base(payload: Mapping[str, Any], inherited: Path) -> Path:
    candidates = [
        Path(value).resolve()
        for key, value in payload.items()
        if key.endswith("_root")
        and isinstance(value, str)
        and value
        and Path(value).is_absolute()
        and Path(value).is_dir()
    ]
    return candidates[0] if len(candidates) == 1 else inherited


def _verify_nested_relative_bindings(
    payload: Any,
    *,
    base_root: Path,
    input_hashes: InputHashes,
) -> None:
    """Rehash nested ``*_relative_path``/``*_sha256`` identity pairs."""

    if isinstance(payload, Mapping):
        local_base = _binding_base(payload, base_root)
        for key, value in payload.items():
            if not key.endswith("_relative_path") or not isinstance(value, str):
                continue
            digest_key = f"{key[:-len('_relative_path')]}_sha256"
            expected = payload.get(digest_key)
            if expected is None:
                continue
            path = _relative_file(local_base, value, label=key)
            input_hashes.bind(path, expected)
        for value in payload.values():
            _verify_nested_relative_bindings(
                value, base_root=local_base, input_hashes=input_hashes
            )
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _verify_nested_relative_bindings(
                value, base_root=base_root, input_hashes=input_hashes
            )


def _verify_phase2_source(
    phase2_root: Path,
    cube_id: str,
    source: Any,
    input_hashes: InputHashes,
) -> None:
    if not isinstance(source, Mapping) or source.get("cube_id") != cube_id:
        raise RuntimeError(f"campaign has an invalid Phase 2 source binding: {cube_id}")
    if Path(str(source.get("phase2_root", ""))).resolve() != phase2_root:
        raise RuntimeError(f"campaign Phase 2 root mismatch: {cube_id}")
    completion_path = _relative_file(
        phase2_root,
        source.get("completion_relative_path"),
        label=f"{cube_id} Phase 2 completion marker",
    )
    manifest_path = _relative_file(
        phase2_root,
        source.get("manifest_relative_path"),
        label=f"{cube_id} Phase 2 manifest",
    )
    input_hashes.bind(completion_path, source.get("completion_sha256"))
    input_hashes.bind(manifest_path, source.get("manifest_sha256"))
    field_hashes = source.get("analysis_field_sha256")
    if not isinstance(field_hashes, Mapping) or not field_hashes:
        raise RuntimeError(f"campaign Phase 2 source lacks analysis-array bindings: {cube_id}")
    cube_root = _contained(phase2_root, phase2_root / cube_id, label=f"{cube_id} Phase 2 cube")
    for relative, expected in field_hashes.items():
        path = _relative_file(cube_root, relative, label=f"{cube_id} analysis field")
        input_hashes.bind(path, expected)
    _verify_nested_relative_bindings(
        source, base_root=phase2_root, input_hashes=input_hashes
    )


def _verify_uncertainty(path: Path, result: Any) -> dict[str, np.ndarray]:
    required = {
        "metadata_json",
        "moments",
        "block_bootstrap_interval_low",
        "block_bootstrap_interval_high",
        "accepted_contributing_blocks",
        "accepted_effective_blocks",
        "valid_bootstrap_resamples",
        "local_log_slope",
        "local_log_slope_support_mask",
        "local_log_slope_bootstrap_interval_low",
        "local_log_slope_bootstrap_interval_high",
    }
    with np.load(path, allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            missing = sorted(required - set(payload.files))
            raise RuntimeError(f"uncertainty publication is incomplete: {path}: {missing}")
        uncertainty = {name: payload[name].copy() for name in payload.files}
    _uncertainty_metadata(uncertainty)
    if not np.array_equal(uncertainty["moments"], result.moments, equal_nan=True):
        raise RuntimeError(f"uncertainty moments differ from reduction result: {path}")
    for name in required - {"metadata_json"}:
        if uncertainty[name].shape != result.moments.shape:
            raise RuntimeError(f"uncertainty array shape mismatch for {name}: {path}")
    if uncertainty["local_log_slope_support_mask"].dtype.kind != "b":
        raise RuntimeError(f"local-slope support mask is not boolean: {path}")
    return uncertainty


def _group_path(root: Path, group_id: str) -> Path:
    return _contained(root, root / "reductions" / group_id, label="reduction")


def _verify_release(
    label: str,
    root: Path,
    metadata: Mapping[str, Any],
    input_hashes: InputHashes,
) -> VerifiedRelease:
    """Verify a retained Phase 3a-style sampler release without live-source replay."""

    root = root.resolve()
    scale = int(metadata.get("L_sub", metadata.get("scale", 0)))
    if scale <= 0:
        raise RuntimeError(f"release {label!r} must declare a positive L_sub")
    campaign_path = root / "manifests" / "campaign.json"
    shards_path = root / "manifests" / "shards.json"
    plan_path = root / "PLAN_COMPLETE.json"
    summary_filename = str(metadata.get("summary_filename", "phase3a_summary.json"))
    marker_filename = str(
        metadata.get("summary_marker_filename", "PHASE3A_RELEASE_COMPLETE.json")
    )
    summary_path = _relative_file(root, summary_filename, label=f"{label} release summary")
    summary_marker_path = _relative_file(
        root, marker_filename, label=f"{label} release summary marker"
    )
    campaign = _load_json(campaign_path)
    shard_payload = _load_json(shards_path)
    plan = _load_json(plan_path)
    summary = _load_json(summary_path)
    summary_marker = _load_json(summary_marker_path)
    implementation = _source_identity_coherent(
        campaign.get("source_version"), label=f"{label} campaign"
    )
    expected_summary_phase = metadata.get("summary_phase")
    expected_campaign_config = metadata.get("phase5_sampler_campaign_config")
    expected_selection_set = metadata.get("selection_set")
    expected_matrix = metadata.get("matrix")
    if (
        campaign.get("schema_version") != SCHEMA_VERSION
        or campaign.get("status") != "planned"
        or campaign.get("configuration_sha256")
        != _mapping_sha256(campaign.get("configuration", {}))
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("status") != "passed"
        or plan.get("campaign_sha256") != file_sha256(campaign_path)
        or plan.get("shards_sha256") != file_sha256(shards_path)
        or plan.get("implementation_sha256") != implementation
        or summary_marker.get("schema_version") != SCHEMA_VERSION
        or summary_marker.get("status") != "release_aggregation_complete"
        or summary_marker.get("summary_sha256") != file_sha256(summary_path)
        or summary_marker.get("implementation_sha256") != implementation
        or summary.get("source_version", {}).get("implementation_sha256")
        != implementation
        or (
            expected_summary_phase is not None
            and summary.get("phase") != expected_summary_phase
        )
        or (
            expected_campaign_config is not None
            and summary.get("phase5_sampler_campaign_config")
            != expected_campaign_config
        )
        or (
            expected_selection_set is not None
            and summary.get("selection_set") != expected_selection_set
        )
        or (expected_matrix is not None and summary.get("matrix") != expected_matrix)
    ):
        raise RuntimeError(f"invalid plan/campaign/summary publication for release {label!r}")
    input_hashes.add_many(
        (campaign_path, shards_path, plan_path, summary_path, summary_marker_path)
    )
    if (summary_filename, marker_filename) != (
        "phase3a_summary.json",
        "PHASE3A_RELEASE_COMPLETE.json",
    ):
        inherited_summary_path = _relative_file(
            root, "phase3a_summary.json", label=f"{label} inherited release summary"
        )
        inherited_marker_path = _relative_file(
            root,
            "PHASE3A_RELEASE_COMPLETE.json",
            label=f"{label} inherited release summary marker",
        )
        inherited_summary = _load_json(inherited_summary_path)
        inherited_marker = _load_json(inherited_marker_path)
        if (
            inherited_marker.get("schema_version") != SCHEMA_VERSION
            or inherited_marker.get("status") != "release_aggregation_complete"
            or inherited_marker.get("summary_sha256")
            != file_sha256(inherited_summary_path)
            or inherited_marker.get("implementation_sha256") != implementation
            or inherited_summary.get("source_version", {}).get(
                "implementation_sha256"
            )
            != implementation
        ):
            raise RuntimeError(f"invalid inherited summary publication for release {label!r}")
        input_hashes.add_many((inherited_summary_path, inherited_marker_path))
    phase2_root = Path(str(campaign.get("phase2_root", ""))).resolve()
    if not phase2_root.is_dir():
        raise RuntimeError(f"release {label!r} has no retained Phase 2 root: {phase2_root}")
    phase2_sources = campaign.get("phase2_sources")
    if not isinstance(phase2_sources, dict) or not phase2_sources:
        raise RuntimeError(f"release {label!r} has no campaign Phase 2 source bindings")
    for cube_id, source in phase2_sources.items():
        _verify_phase2_source(phase2_root, str(cube_id), source, input_hashes)

    displacement_rows = campaign.get("displacement_manifests")
    if not isinstance(displacement_rows, Mapping) or not displacement_rows:
        raise RuntimeError(f"release {label!r} has no displacement-manifest bindings")
    offset_counts: dict[int, int] = {}
    for width_text, row in displacement_rows.items():
        if not isinstance(row, Mapping):
            raise RuntimeError(f"invalid displacement binding for release {label!r}")
        width = int(width_text)
        json_path = _relative_file(
            root, row.get("json_relative_path"), label=f"{label} displacement JSON"
        )
        npz_path = _relative_file(
            root, row.get("npz_relative_path"), label=f"{label} displacement NPZ"
        )
        input_hashes.bind(json_path, row.get("json_sha256"))
        input_hashes.bind(npz_path, row.get("npz_sha256"))
        offset_counts[width] = int(row.get("offset_count", -1))
        if offset_counts[width] <= 0:
            raise RuntimeError(f"invalid displacement offset count for {label} stencil {width}")

    shard_rows = shard_payload.get("shards")
    if (
        not isinstance(shard_rows, list)
        or len(shard_rows) != campaign.get("shard_count")
        or not shard_rows
    ):
        raise RuntimeError(f"release {label!r} has an incoherent shard inventory")
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    shard_marker_sha256: dict[str, str] = {}
    shard_staging_bytes = 0
    for row in shard_rows:
        if (
            not isinstance(row, dict)
            or row.get("cube_id") not in phase2_sources
            or int(row.get("stencil_width", 0)) not in offset_counts
        ):
            raise RuntimeError(f"release {label!r} contains an invalid shard row")
        shard_id = str(row.get("shard_id", ""))
        group_id = str(row.get("group_id", ""))
        shard_root = _contained(root, root / "shards" / shard_id, label="shard")
        marker_path = _relative_file(shard_root, "COMPLETE.json", label=f"{shard_id} marker")
        partial_path = _relative_file(shard_root, "partial.npz", label=f"{shard_id} partial")
        marker = _load_json(marker_path)
        if (
            marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("status") != "passed"
            or marker.get("shard") != row
            or marker.get("implementation_sha256") != implementation
            or marker.get("phase2_source") != phase2_sources[row["cube_id"]]
        ):
            raise RuntimeError(f"invalid retained shard marker: {shard_id}")
        input_hashes.bind(partial_path, marker.get("partial_sha256"))
        shard_marker_sha256[shard_id] = input_hashes.add(marker_path)
        shard_staging_bytes += int(marker.get("staging_logical_bytes_before_marker", 0))
        rows_by_group.setdefault(group_id, []).append(row)

    summary_rows = summary.get("groups")
    summary_group_ids = {
        str(row.get("group_id"))
        for row in summary_rows
        if isinstance(row, Mapping)
    } if isinstance(summary_rows, list) else set()
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("operational_status") != "release_aggregation_complete"
        or summary_group_ids != set(rows_by_group)
    ):
        raise RuntimeError(f"release {label!r} has an incoherent summary group inventory")

    groups: dict[tuple[str, int, str], Group] = {}
    reduction_staging_bytes = 0
    reduction_elapsed_seconds = 0.0
    for group_id, rows in sorted(rows_by_group.items()):
        group_root = _group_path(root, group_id)
        result_path = _relative_file(group_root, "result.npz", label=f"{group_id} result")
        uncertainty_path = _relative_file(
            group_root, "uncertainty.npz", label=f"{group_id} uncertainty"
        )
        manifest_path = _relative_file(
            group_root, "reduction_manifest.json", label=f"{group_id} reduction manifest"
        )
        marker_path = _relative_file(group_root, "COMPLETE.json", label=f"{group_id} marker")
        marker = _load_json(marker_path)
        manifest = _load_json(manifest_path)
        ordered_shard_ids = tuple(str(row["shard_id"]) for row in rows)
        if (
            marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("status") != "passed"
            or marker.get("group_id") != group_id
            or manifest.get("group_id") != group_id
            or tuple(manifest.get("ordered_shard_ids", ())) != ordered_shard_ids
            or manifest.get("ordered_shard_marker_sha256")
            != {shard_id: shard_marker_sha256[shard_id] for shard_id in ordered_shard_ids}
            or manifest.get("implementation_sha256") != implementation
        ):
            raise RuntimeError(f"invalid retained reduction publication: {group_id}")
        input_hashes.bind(result_path, marker.get("result_sha256"))
        input_hashes.bind(uncertainty_path, marker.get("uncertainty_sha256"))
        input_hashes.bind(manifest_path, marker.get("reduction_manifest_sha256"))
        input_hashes.add(marker_path)
        result = load_finite_domain_partial_npz(result_path)
        stencil_width = int(rows[0]["stencil_width"])
        support_mode = str(rows[0]["support_mode"])
        cube_id = str(rows[0]["cube_id"])
        if (
            any(
                int(row["stencil_width"]) != stencil_width
                or str(row["support_mode"]) != support_mode
                or str(row["cube_id"]) != cube_id
                for row in rows
            )
            or result.stencil_width != stencil_width
            or result.pair_mode != support_mode
            or int(result.support_displacement_count) != offset_counts[stencil_width]
        ):
            raise RuntimeError(f"reduction matrix metadata mismatch: {group_id}")
        uncertainty = _verify_uncertainty(uncertainty_path, result)
        key = (cube_id, stencil_width, support_mode)
        if key in groups:
            raise RuntimeError(f"release {label!r} repeats one reduction matrix: {key}")
        groups[key] = Group(
            release_label=label,
            scale=scale,
            cube_id=cube_id,
            stencil_width=stencil_width,
            support_mode=support_mode,
            group_id=group_id,
            result=result,
            uncertainty=uncertainty,
            reduction_marker=marker,
        )
        reduction_staging_bytes += int(marker.get("staging_logical_bytes_before_marker", 0))
        reduction_elapsed_seconds += float(marker.get("reduction_elapsed_seconds", 0.0))
    settled = summary.get("settled_release_bytes_before_summary_marker", {})
    return VerifiedRelease(
        label=label,
        root=root,
        scale=scale,
        selection_set=str(
            metadata.get(
                "report_selection_set",
                summary.get("selection_set", metadata.get("selection_set", "lineage")),
            )
        ),
        matrix=str(
            metadata.get(
                "report_matrix",
                summary.get("matrix", metadata.get("matrix", "unknown")),
            )
        ),
        campaign=campaign,
        summary=summary,
        phase2_sources=phase2_sources,
        groups=groups,
        verification={
            "status": "passed",
            "verification_mode": "retained_marker_graph_without_live_source_replay",
            "historical_implementation_sha256": implementation,
            "verified_phase2_sources": len(phase2_sources),
            "verified_shards": len(shard_rows),
            "verified_reductions": len(groups),
            "shard_staging_logical_bytes_before_markers": shard_staging_bytes,
            "reduction_staging_logical_bytes_before_markers": reduction_staging_bytes,
            "reduction_elapsed_seconds": reduction_elapsed_seconds,
            "settled_release_logical_bytes_before_summary_marker": int(
                settled.get("logical_bytes", 0)
            )
            if isinstance(settled, Mapping)
            else 0,
            "settled_release_allocated_bytes_before_summary_marker": int(
                settled.get("allocated_bytes", 0)
            )
            if isinstance(settled, Mapping)
            else 0,
        },
    )


def _release_specs(values: Sequence[str]) -> dict[str, Path]:
    releases: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label)
            or not raw_path
        ):
            raise ValueError(f"--release must use label=path with a stable label: {value!r}")
        if label in releases:
            raise ValueError(f"duplicate --release label: {label}")
        releases[label] = Path(raw_path)
    if not releases:
        raise ValueError("at least one --release label=path is required")
    return releases


def _verify_phase1_bindings(
    phase1_root: Path,
    config: Mapping[str, Any],
    input_hashes: InputHashes,
) -> None:
    phase1_root = phase1_root.resolve()
    if not phase1_root.is_dir():
        raise RuntimeError(f"Phase 1 root is not a retained directory: {phase1_root}")
    artifacts = config.get("phase1_artifacts")
    if isinstance(artifacts, list) and artifacts:
        for row in artifacts:
            if not isinstance(row, Mapping):
                raise RuntimeError("campaign config has an invalid Phase 1 artifact binding")
            path = _relative_file(
                phase1_root, row.get("relative_path"), label="Phase 1 catalog artifact"
            )
            input_hashes.bind(path, row.get("sha256"))
        return
    source_artifacts = config.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or not source_artifacts:
        raise RuntimeError("campaign config must bind retained Phase 1 catalog artifacts")

    verified = 0

    def bind_tree(value: Any) -> None:
        nonlocal verified
        if isinstance(value, Mapping):
            if "relative_path" in value or "sha256" in value:
                path = _relative_file(
                    phase1_root,
                    value.get("relative_path"),
                    label="Phase 1 catalog artifact",
                )
                input_hashes.bind(path, value.get("sha256"))
                verified += 1
            for item in value.values():
                bind_tree(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                bind_tree(item)

    bind_tree(source_artifacts)
    if verified == 0:
        raise RuntimeError("campaign config has no usable Phase 1 artifact bindings")


def _environment(row: Mapping[str, Any], *, label: str) -> dict[str, float]:
    source = row.get("environment", row)
    if not isinstance(source, Mapping):
        raise RuntimeError(f"{label} lacks an environment object")
    values: dict[str, float] = {}
    for name in ("B_mean", "deltaB", "B_rms"):
        values[name] = _finite_float(source.get(name), label=f"{label} {name}")
    values["dBB"] = _finite_float(
        source.get("dBB", values["deltaB"] / values["B_mean"]),
        label=f"{label} dBB",
    )
    if values["B_rms"] <= 0.0:
        raise RuntimeError(f"{label} B_rms must be positive")
    values["B_mean_sq_over_B2_mean"] = _finite_float(
        source.get(
            "B_mean_sq_over_B2_mean",
            values["B_mean"] ** 2 / values["B_rms"] ** 2,
        ),
        label=f"{label} B_mean_sq_over_B2_mean",
    )
    values["deltaB_sq_over_B2_mean"] = _finite_float(
        source.get(
            "deltaB_sq_over_B2_mean",
            values["deltaB"] ** 2 / values["B_rms"] ** 2,
        ),
        label=f"{label} deltaB_sq_over_B2_mean",
    )
    return values


def _normalize_selections(
    config: Mapping[str, Any],
    release_metadata: Mapping[str, Mapping[str, Any]],
    release_sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Selection, ...]:
    rows = config.get("selections")
    if not isinstance(rows, list) or not rows:
        def configured_rows(value: Any) -> list[dict[str, Any]]:
            found: list[dict[str, Any]] = []
            if isinstance(value, Mapping):
                if (
                    isinstance(value.get("cube_id"), str)
                    and isinstance(value.get("catalog_magnetic_values"), Mapping)
                ):
                    found.append(dict(value))
                for item in value.values():
                    found.extend(configured_rows(item))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    found.extend(configured_rows(item))
            return found

        candidates = configured_rows(config)
        rows = []
        for release_label, metadata in release_metadata.items():
            scale = int(metadata.get("L_sub", metadata.get("scale", 0)))
            selection_set = str(metadata.get("selection_set", "lineage"))
            matrix = str(metadata.get("matrix", "unknown"))
            source_ids = set((release_sources or {}).get(release_label, {}))
            selected_by_cube = {
                str(row.get("cube_id")): row
                for row in candidates
                if int(row.get("L_sub", row.get("scale", -1))) == scale
                and str(row.get("cube_id", "")) in source_ids
            }
            missing = sorted(source_ids - set(selected_by_cube))
            if missing:
                raise RuntimeError(
                    f"campaign config lacks {selection_set!r} L{scale} selection rows "
                    f"for release {release_label!r}: {missing}"
                )
            rows.extend(
                {
                    **dict(row),
                    "release": release_label,
                    "selection_set": selection_set,
                    "matrix": matrix,
                    "role": str(
                        (
                            f"observational_control:matched_smoke:"
                            f"{row['observational_control_set'].get('matched_pair_id')}:"
                            f"{row['observational_control_set'].get('matched_role')}"
                        )
                        if isinstance(row.get("observational_control_set"), Mapping)
                        else row.get(
                            "root_phase1_role",
                            ";".join(str(value) for value in row.get("roles", ())),
                        )
                    ),
                    "physical_region_id": str(
                        row.get("root_L640_cube_id", row.get("cube_id", ""))
                    ),
                    "matched_group": (
                        f"matched_smoke:{row['observational_control_set'].get('matched_pair_id')}"
                        if isinstance(row.get("observational_control_set"), Mapping)
                        else str(row.get("root_phase1_role"))
                        if str(row.get("root_phase1_role", "")).startswith("matched:")
                        else None
                    ),
                    "representative": bool(
                        row.get("smoke_anchor")
                        or str(row.get("root_phase1_role", "")).startswith(
                            "representative:"
                        )
                    ),
                    "outlier": bool(
                        "outlier" in str(row.get("root_phase1_role", "")).lower()
                        or "outlier" in str(row.get("smoke_anchor_role", "")).lower()
                    ),
                    "environment": row.get("catalog_magnetic_values"),
                }
                for row in selected_by_cube.values()
            )
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"selection {index} must be an object")
        release_label = str(row.get("release", ""))
        cube_id = str(row.get("cube_id", ""))
        if release_label not in release_metadata or not cube_id:
            raise RuntimeError(f"selection {index} has an unknown release or empty cube ID")
        l_sub = int(row.get("L_sub", row.get("scale", 0)))
        expected_scale = int(
            release_metadata[release_label].get(
                "L_sub", release_metadata[release_label].get("scale", 0)
            )
        )
        if l_sub <= 0 or l_sub != expected_scale:
            raise RuntimeError(f"selection {index} scale differs from release metadata")
        role = str(row.get("role", "selected"))
        physical_region = str(row.get("physical_region_id", cube_id))
        output.append(
            Selection(
                release_label=release_label,
                selection_set=str(row.get("selection_set", "lineage")),
                matrix=str(row.get("matrix", "unknown")),
                cube_id=cube_id,
                l_sub=l_sub,
                role=role,
                physical_region_id=physical_region,
                matched_group=(
                    str(row["matched_group"])
                    if row.get("matched_group") is not None
                    else None
                ),
                observational_control_set=(
                    dict(row["observational_control_set"])
                    if isinstance(row.get("observational_control_set"), Mapping)
                    else None
                ),
                representative=bool(
                    row.get("representative", role.startswith("representative:"))
                ),
                outlier=bool(row.get("outlier", "outlier" in role.lower())),
                environment=_environment(row, label=f"selection {index} {cube_id}"),
                source_row=dict(row),
            )
        )
    return tuple(output)


def _phase5_release_metadata(config_path: Path, release_root: Path) -> dict[str, Any]:
    """Derive report metadata from one retained Phase 5 sampler publication."""

    summary = _load_json(release_root / "phase5_sampler_summary.json")
    scale = int(summary.get("scale", 0))
    if (
        summary.get("phase") != "phase5_cross_scale_sampler"
        or scale <= 0
        or summary.get("phase5_sampler_campaign_config")
        != {
            "campaign_config_path": str(config_path.resolve()),
            "campaign_config_sha256": file_sha256(config_path),
        }
    ):
        raise RuntimeError(f"retained Phase 5 sampler summary has stale scope: {release_root}")
    return {
        "L_sub": scale,
        "summary_filename": "phase5_sampler_summary.json",
        "summary_marker_filename": "PHASE5_SAMPLER_COMPLETE.json",
        "summary_phase": "phase5_cross_scale_sampler",
        "phase5_sampler_campaign_config": summary["phase5_sampler_campaign_config"],
        "selection_set": summary.get("selection_set"),
        "matrix": summary.get("matrix"),
    }


def _verify_config(
    config_path: Path,
    phase1_root: Path,
    release_paths: Mapping[str, Path],
    input_hashes: InputHashes,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    config = _load_json(config_path)
    input_hashes.add(config_path)
    release_metadata = config.get("releases")
    phase = config.get("phase")
    if config.get("schema_version") != SCHEMA_VERSION or phase not in {
        "phase5_cross_scale_report",
        "phase5_cross_scale_extraction_campaign",
    }:
        raise RuntimeError("campaign config is not a supported frozen Phase 5 config")
    if phase == "phase5_cross_scale_extraction_campaign":
        if (
            config.get("status") != "frozen"
            or Path(str(config.get("trusted_run", ""))).resolve()
            != phase1_root.resolve()
        ):
            raise RuntimeError("frozen Phase 5 extraction config lost its trusted-run binding")
        release_metadata = {
            label: _phase5_release_metadata(config_path, path)
            for label, path in release_paths.items()
        }
    if (
        not isinstance(release_metadata, dict)
        or set(release_metadata) != set(release_paths)
        or any(not isinstance(value, Mapping) for value in release_metadata.values())
    ):
        raise RuntimeError(
            "campaign config scope must resolve the same explicit release labels "
            "supplied on the CLI"
        )
    _verify_phase1_bindings(phase1_root, config, input_hashes)
    return config, release_metadata


def _bind_ledger_summary(path: Path, input_hashes: InputHashes) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise RuntimeError(f"required compute-ledger summary is missing: {path}")
    snapshot = path.read_bytes()
    input_hashes.bind(path, hashlib.sha256(snapshot).hexdigest())
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("compute-ledger summary must be UTF-8 text") from error
    labels = (
        "Workflow budget",
        "Consumed allocated runtime",
        "Remaining budget",
        "Pending maximum additional exposure",
        "Projected remaining after pending maximum",
    )
    metrics: dict[str, float] = {}
    for label in labels:
        match = re.search(
            rf"^\| {re.escape(label)} \| `?([-+0-9.eE]+)`? \|$",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"compute-ledger summary lacks required metric {label!r}")
        metrics[label.lower().replace(" ", "_")] = float(match.group(1))
    if metrics["pending_maximum_additional_exposure"] != 0.0:
        raise RuntimeError("compute-ledger summary must report zero pending maximum exposure")
    updated = re.search(r"^Updated: `([^`]+)`$", text, flags=re.MULTILINE)
    return {
        "source_path": str(path.resolve()),
        "source_sha256": hashlib.sha256(snapshot).hexdigest(),
        "updated": updated.group(1) if updated else None,
        "metrics_node_hours": metrics,
        "pending_exposure_gate": "passed_zero_pending_maximum_additional_exposure",
    }, snapshot


def _bind_execution_decision(
    path: Path,
    *,
    campaign_config: Path,
    release_paths: Mapping[str, Path],
    input_hashes: InputHashes,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError(f"required Phase 5 execution decision is missing: {path}")
    decision = _load_json(path)
    input_hashes.add(path)
    labels = decision.get("retained_report_release_labels")
    if (
        decision.get("schema_version") != SCHEMA_VERSION
        or decision.get("phase") != "phase5_execution_decision"
        or decision.get("status") != "retained_closeout_acquisition_scope"
        or decision.get("campaign_config_sha256") != file_sha256(campaign_config)
        or not isinstance(labels, list)
        or sorted(labels) != sorted(release_paths)
    ):
        raise RuntimeError("Phase 5 execution decision does not bind the supplied release matrix")
    return decision, path.read_text()


def _selection_rows(selections: Sequence[Selection]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for selection in selections:
        key = (selection.l_sub, selection.selection_set, selection.cube_id)
        if key not in rows_by_key:
            rows_by_key[key] = {
                "release_label": selection.release_label,
                "release_labels": [selection.release_label],
                "selection_set": selection.selection_set,
                "matrices": [selection.matrix],
                "cube_id": selection.cube_id,
                "L_sub_cells": selection.l_sub,
                "role": selection.role,
                "physical_region_id": selection.physical_region_id,
                "matched_group": selection.matched_group,
                "observational_control_set": selection.observational_control_set,
                "representative": selection.representative,
                "outlier": selection.outlier,
                **selection.environment,
                "source_selection": selection.source_row,
            }
        elif selection.release_label not in rows_by_key[key]["release_labels"]:
            rows_by_key[key]["release_labels"].append(selection.release_label)
        if selection.matrix not in rows_by_key[key]["matrices"]:
            rows_by_key[key]["matrices"].append(selection.matrix)
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def _science_axes(result: Any) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    q_names = tuple(name for name in ("B", "u") if name in result.q_names)
    p_values = tuple(float(value) for value in result.p_values)
    directions = tuple(name for name in SCIENCE_DIRECTIONS if name in result.direction_names)
    if not q_names or not p_values or not directions:
        raise RuntimeError("reduction lacks required B/u directional science axes")
    return q_names, p_values, directions


def _shell_rows(releases: Sequence[VerifiedRelease]) -> list[dict[str, Any]]:
    rows = []
    for release in releases:
        for group in release.groups.values():
            result = group.result
            ell = _centers(result.ell_bin_edges)
            fraction = _support_fraction(result)
            q_names, p_values, directions = _science_axes(result)
            for q_name in q_names:
                for p_value in p_values:
                    for direction in directions:
                        index = _moment_index(result, q_name, direction, p_value=p_value)
                        curve_support = _curve_support_mask(group, index)
                        for shell, ell_cells in enumerate(ell):
                            rows.append(
                                {
                                    "release_label": release.label,
                                    "selection_set": release.selection_set,
                                    "matrix": release.matrix,
                                    "L_sub_cells": release.scale,
                                    "cube_id": group.cube_id,
                                    "stencil_width": group.stencil_width,
                                    "stencil_label": f"{group.stencil_width}-point",
                                    "support_mode": group.support_mode,
                                    "q_name": q_name,
                                    "p_value": p_value,
                                    "direction": direction,
                                    "shell_index": shell,
                                    "ell_cells": float(ell_cells),
                                    "eligible_origin_fraction": float(fraction[shell]),
                                    "sampled_origins": int(result.sampled_pairs[shell]),
                                    "eligible_origins": int(result.eligible_pairs[shell]),
                                    "cube_candidate_origins": int(
                                        result.cube_candidate_pairs[shell]
                                    ),
                                    "accepted_measurements": int(
                                        result.counts[index][shell]
                                    ),
                                    "accepted_contributing_blocks": int(
                                        group.uncertainty[
                                            "accepted_contributing_blocks"
                                        ][index][shell]
                                    ),
                                    "accepted_effective_blocks": float(
                                        group.uncertainty["accepted_effective_blocks"][
                                            index
                                        ][shell]
                                    ),
                                    "valid_bootstrap_resamples": int(
                                        group.uncertainty["valid_bootstrap_resamples"][
                                            index
                                        ][shell]
                                    ),
                                    "curve_support_gate_passes": bool(
                                        curve_support[shell]
                                    ),
                                    "shell_local_curve_geometry_gate_passes": (
                                        bool(fraction[shell] >= CURVE_SHELL_LOCAL_MINIMUM)
                                        if group.support_mode == OVERLAY_SUPPORT_MODE
                                        else None
                                    ),
                                    "curve_value": float(result.moments[index][shell]),
                                }
                            )
    return rows


def _policy_rows(releases: Sequence[VerifiedRelease]) -> list[dict[str, Any]]:
    rows = []
    for release in releases:
        keys = {
            (cube_id, width)
            for cube_id, width, mode in release.groups
            if mode == PRIMARY_SUPPORT_MODE
            and (cube_id, width, OVERLAY_SUPPORT_MODE) in release.groups
        }
        for cube_id, width in sorted(keys):
            primary = release.groups[(cube_id, width, PRIMARY_SUPPORT_MODE)]
            shell = release.groups[(cube_id, width, OVERLAY_SUPPORT_MODE)]
            if not np.array_equal(primary.result.ell_bin_edges, shell.result.ell_bin_edges):
                raise RuntimeError(f"support-policy shell axes differ: {release.label} {cube_id}")
            ell = _centers(primary.result.ell_bin_edges)
            shell_fraction = _support_fraction(shell.result)
            q_names, p_values, directions = _science_axes(primary.result)
            for q_name in q_names:
                for p_value in p_values:
                    for direction in directions:
                        index = _moment_index(
                            primary.result, q_name, direction, p_value=p_value
                        )
                        primary_supported = _curve_support_mask(primary, index)
                        shell_supported = _curve_support_mask(shell, index)
                        for shell_index, ell_cells in enumerate(ell):
                            left = float(primary.result.moments[index][shell_index])
                            right = float(shell.result.moments[index][shell_index])
                            ratio = left / right if right != 0.0 else float("nan")
                            finite_positive = math.isfinite(ratio) and ratio > 0.0
                            supported = bool(
                                shell_fraction[shell_index] >= CURVE_SHELL_LOCAL_MINIMUM
                                and primary_supported[shell_index]
                                and shell_supported[shell_index]
                                and finite_positive
                            )
                            rows.append(
                                {
                                    "release_label": release.label,
                                    "selection_set": release.selection_set,
                                    "matrix": release.matrix,
                                    "L_sub_cells": release.scale,
                                    "cube_id": cube_id,
                                    "stencil_width": width,
                                    "stencil_label": f"{width}-point",
                                    "q_name": q_name,
                                    "p_value": p_value,
                                    "direction": direction,
                                    "shell_index": shell_index,
                                    "ell_cells": float(ell_cells),
                                    "shell_local_eligible_origin_fraction": float(
                                        shell_fraction[shell_index]
                                    ),
                                    "all_valid_origins_value": left,
                                    "shell_local_value": right,
                                    "all_valid_over_shell_local_ratio": (
                                        ratio if finite_positive else None
                                    ),
                                    "policy_factor": (
                                        max(ratio, 1.0 / ratio) if supported else None
                                    ),
                                    "supported_policy_diagnostic": supported,
                                    "interpretation": (
                                        "finite-support policy sensitivity diagnostic; "
                                        "not a correction"
                                    ),
                                }
                            )
    return rows


def _slope_rows(releases: Sequence[VerifiedRelease]) -> list[dict[str, Any]]:
    rows = []
    for release in releases:
        for group in release.groups.values():
            result = group.result
            ell = _centers(result.ell_bin_edges)
            shell_fraction = _support_fraction(result)
            q_names, p_values, directions = _science_axes(result)
            for q_name in q_names:
                for p_value in p_values:
                    for direction in directions:
                        index = _moment_index(result, q_name, direction, p_value=p_value)
                        slope_support = group.uncertainty["local_log_slope_support_mask"][
                            index
                        ].copy()
                        slope_support &= (
                            np.isfinite(group.uncertainty["local_log_slope"][index])
                            & np.isfinite(
                                group.uncertainty[
                                    "local_log_slope_bootstrap_interval_low"
                                ][index]
                            )
                            & np.isfinite(
                                group.uncertainty[
                                    "local_log_slope_bootstrap_interval_high"
                                ][index]
                            )
                        )
                        if group.support_mode == OVERLAY_SUPPORT_MODE:
                            slope_support &= shell_fraction >= SLOPE_SHELL_LOCAL_MINIMUM
                        for shell_index, ell_cells in enumerate(ell):
                            rows.append(
                                {
                                    "release_label": release.label,
                                    "selection_set": release.selection_set,
                                    "matrix": release.matrix,
                                    "L_sub_cells": release.scale,
                                    "cube_id": group.cube_id,
                                    "stencil_width": group.stencil_width,
                                    "stencil_label": f"{group.stencil_width}-point",
                                    "support_mode": group.support_mode,
                                    "q_name": q_name,
                                    "p_value": p_value,
                                    "direction": direction,
                                    "shell_index": shell_index,
                                    "ell_cells": float(ell_cells),
                                    "supported_local_slope_diagnostic": bool(
                                        slope_support[shell_index]
                                    ),
                                    "local_log_slope": float(
                                        group.uncertainty["local_log_slope"][index][
                                            shell_index
                                        ]
                                    ),
                                    "local_log_slope_bootstrap_interval_low": float(
                                        group.uncertainty[
                                            "local_log_slope_bootstrap_interval_low"
                                        ][index][shell_index]
                                    ),
                                    "local_log_slope_bootstrap_interval_high": float(
                                        group.uncertainty[
                                            "local_log_slope_bootstrap_interval_high"
                                        ][index][shell_index]
                                    ),
                                    "interpretation": (
                                        "centered-window local-slope uncertainty diagnostic; "
                                        "not a fitted exponent"
                                    ),
                                }
                            )
    return rows


def _selection_groups(
    selections: Sequence[Selection],
    releases: Sequence[VerifiedRelease],
) -> dict[tuple[str, int, str], list[Group]]:
    grouped: dict[tuple[str, int, str], list[Group]] = {}
    by_label = {release.label: release for release in releases}
    for selection in selections:
        release = by_label[selection.release_label]
        for (cube_id, _width, mode), group in release.groups.items():
            if cube_id == selection.cube_id and mode == PRIMARY_SUPPORT_MODE:
                grouped.setdefault(
                    (
                        selection.selection_set,
                        selection.l_sub,
                        selection.physical_region_id,
                    ),
                    [],
                ).append(group)
    return grouped


def _stencil_rows(
    selections: Sequence[Selection],
    releases: Sequence[VerifiedRelease],
) -> list[dict[str, Any]]:
    rows = []
    for (selection_set, scale, physical_region), groups in _selection_groups(
        selections, releases
    ).items():
        by_width: dict[int, Group] = {}
        for group in sorted(groups, key=lambda value: len(value.result.p_values)):
            by_width.setdefault(group.stencil_width, group)
        if 2 not in by_width or 3 not in by_width:
            continue
        two, three = by_width[2], by_width[3]
        ell_two = _centers(two.result.ell_bin_edges)
        ell_three = _centers(three.result.ell_bin_edges)
        q_names, _p_values, directions = _science_axes(two.result)
        if 2.0 not in two.result.p_values or 2.0 not in three.result.p_values:
            continue
        for q_name in q_names:
            if q_name not in three.result.q_names:
                continue
            for direction in directions:
                if direction not in three.result.direction_names:
                    continue
                two_index = _moment_index(two.result, q_name, direction, p_value=2.0)
                three_index = _moment_index(three.result, q_name, direction, p_value=2.0)
                two_supported = _curve_support_mask(two, two_index)
                three_supported = _curve_support_mask(three, three_index)
                for shell_index, ell_cells in enumerate(ell_three):
                    nearest = int(np.argmin(np.abs(ell_two - ell_cells)))
                    if not np.isclose(ell_two[nearest], ell_cells, rtol=0.08, atol=0.0):
                        continue
                    two_value = float(two.result.moments[two_index][nearest])
                    three_value = float(three.result.moments[three_index][shell_index])
                    supported = bool(
                        two_supported[nearest]
                        and three_supported[shell_index]
                        and two_value > 0.0
                        and three_value > 0.0
                    )
                    rows.append(
                        {
                            "selection_set": selection_set,
                            "L_sub_cells": scale,
                            "physical_region_id": physical_region,
                            "2point_release_label": two.release_label,
                            "3point_release_label": three.release_label,
                            "2point_cube_id": two.cube_id,
                            "3point_cube_id": three.cube_id,
                            "q_name": q_name,
                            "p_value": 2.0,
                            "direction": direction,
                            "ell_cells": float(ell_cells),
                            "2point_value": two_value,
                            "3point_value": three_value,
                            "3point_over_2point_ratio": (
                                three_value / two_value if supported else None
                            ),
                            "supported_labeled_stencil_diagnostic": supported,
                            "interpretation": (
                                "labeled 2-point versus 3-point comparison; "
                                "the products remain distinct statistics"
                            ),
                        }
                    )
    return rows


def _runtime_rows(releases: Sequence[VerifiedRelease]) -> list[dict[str, Any]]:
    rows = []
    for release in releases:
        verification = release.verification
        rows.append(
            {
                "release_label": release.label,
                "selection_set": release.selection_set,
                "matrix": release.matrix,
                "L_sub_cells": release.scale,
                "cube_count": len(release.phase2_sources),
                "verified_shards": verification["verified_shards"],
                "verified_reductions": verification["verified_reductions"],
                "estimator_elapsed_seconds_sum": float(
                    sum(group.result.elapsed_seconds for group in release.groups.values())
                ),
                "reduction_elapsed_seconds_sum": verification[
                    "reduction_elapsed_seconds"
                ],
                "shard_staging_logical_bytes_before_markers": verification[
                    "shard_staging_logical_bytes_before_markers"
                ],
                "reduction_staging_logical_bytes_before_markers": verification[
                    "reduction_staging_logical_bytes_before_markers"
                ],
                "settled_release_logical_bytes_before_summary_marker": verification[
                    "settled_release_logical_bytes_before_summary_marker"
                ],
                "settled_release_allocated_bytes_before_summary_marker": verification[
                    "settled_release_allocated_bytes_before_summary_marker"
                ],
                "interpretation": (
                    "marker-recorded sampler accounting; scheduler node-hours remain "
                    "separate in the bound ledger snapshot"
                ),
            }
        )
    return rows


def _matrix_availability_rows(
    releases: Sequence[VerifiedRelease],
) -> list[dict[str, Any]]:
    all_matrices = ("baseline", "orders", "3point", "5point")
    output = []
    for selection_set, scale in sorted(
        {(release.selection_set, release.scale) for release in releases}
    ):
        supplied = sorted(
            {
                release.matrix
                for release in releases
                if release.selection_set == selection_set and release.scale == scale
            }
        )
        allowed = tuple(MATRIX_AVAILABILITY_BY_SCALE.get(scale, ()))
        output.append(
            {
                "selection_set": selection_set,
                "L_sub_cells": scale,
                "allowed_matrices": list(allowed),
                "supplied_retained_matrices": supplied,
                "allowed_but_not_supplied_matrices": sorted(set(allowed) - set(supplied)),
                "unavailable_by_scale_policy": sorted(set(all_matrices) - set(allowed)),
                "interpretation": (
                    "not-supplied and unavailable matrices remain explicitly unclaimed"
                ),
            }
        )
    return output


def _outlier_rows(selections: Sequence[Selection]) -> list[dict[str, Any]]:
    return [
        row for row in _selection_rows(selections) if row["outlier"]
    ]


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def workflow_schematic(output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(12.6, 3.5))
    axis.set_axis_off()
    boxes = (
        (0.01, "trusted Phase 1\nscale catalogs"),
        (0.18, "approved Phase 5\nselection lineage"),
        (0.35, "retained Phase 2\nprimitive cubes"),
        (0.52, "hash-bound sampler\nreductions + blocks"),
        (0.69, "cross-scale\nreport diagnostics"),
        (0.86, "immutable\nreview package"),
    )
    for left, text in boxes:
        axis.add_patch(
            plt.Rectangle(
                (left, 0.36),
                0.13,
                0.33,
                facecolor="#e7f0fa",
                edgecolor="#2a5c8a",
                linewidth=1.3,
            )
        )
        axis.text(left + 0.065, 0.525, text, ha="center", va="center", fontsize=9)
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(right[0] - 0.01, 0.525),
            xytext=(left[0] + 0.14, 0.525),
            arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#444444"},
        )
    axis.text(
        0.5,
        0.13,
        "Schematic only. Quantitative figures are derived from retained hash-verified artifacts.",
        ha="center",
        fontsize=9.5,
    )
    return _save(figure, output_dir, FIGURE_FILENAMES[0])


def environment_census_figure(selections: Sequence[Selection], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    scales = sorted({selection.l_sub for selection in selections})
    colors = {scale: plt.cm.viridis(index / max(1, len(scales) - 1)) for index, scale in enumerate(scales)}
    selection_sets = sorted({selection.selection_set for selection in selections})
    markers = {
        selection_set: ("s" if selection_set == "matched_smoke" else ("o", "^", "D")[index % 3])
        for index, selection_set in enumerate(selection_sets)
    }
    for selection in selections:
        env = selection.environment
        marker = "*" if selection.outlier else markers[selection.selection_set]
        kwargs = {
            "color": colors[selection.l_sub],
            "marker": marker,
            "s": 76 if selection.outlier else 34,
            "alpha": 0.84,
        }
        axes[0].scatter(env["dBB"], env["B_mean"], **kwargs)
        axes[1].scatter(env["dBB"], env["deltaB"], **kwargs)
        axes[2].scatter(
            env["B_mean_sq_over_B2_mean"],
            env["deltaB_sq_over_B2_mean"],
            **kwargs,
        )
    for scale in scales:
        axes[0].scatter([], [], color=colors[scale], label=f"L={scale}")
    for selection_set in selection_sets:
        axes[1].scatter(
            [], [], color="#555555", marker=markers[selection_set], label=selection_set
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("dBB")
    axes[1].set_xlabel("dBB")
    axes[0].set_ylabel("B_mean")
    axes[1].set_ylabel("deltaB")
    axes[2].set_xlabel(r"$B_{\rm mean}^2 / \langle B^2\rangle_V$")
    axes[2].set_ylabel(r"$\delta B^2 / \langle B^2\rangle_V$")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Scale-specific selection lineage and magnetic environment census")
    return _save(figure, output_dir, FIGURE_FILENAMES[1])


def _representative_groups(
    selections: Sequence[Selection],
    releases: Sequence[VerifiedRelease],
) -> list[tuple[str, str, Group]]:
    by_label = {release.label: release for release in releases}
    output: dict[tuple[str, int, str], tuple[str, str, Group]] = {}
    selected = [selection for selection in selections if selection.representative]
    if not selected:
        selected = list(selections)
    for selection in selected:
        release = by_label[selection.release_label]
        key = (selection.cube_id, 2, PRIMARY_SUPPORT_MODE)
        if key in release.groups:
            group = release.groups[key]
            group_key = (
                selection.selection_set,
                selection.l_sub,
                selection.physical_region_id,
            )
            candidate = (selection.selection_set, selection.role, group)
            current = output.get(group_key)
            if current is None or len(group.result.p_values) < len(
                current[2].result.p_values
            ):
                output[group_key] = candidate
    return [output[key] for key in sorted(output)]


def representative_curves_figure(
    selections: Sequence[Selection],
    releases: Sequence[VerifiedRelease],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    groups = _representative_groups(selections, releases)
    scales = sorted({group.scale for _, _, group in groups})
    cohorts = sorted({selection_set for selection_set, _, _ in groups})
    colors = {
        scale: plt.cm.viridis(index / max(1, len(scales) - 1))
        for index, scale in enumerate(scales)
    }
    line_styles = {
        cohort: ("-", "--", ":", "-.")[index % 4]
        for index, cohort in enumerate(cohorts)
    }
    for selection_set, _role, group in groups:
        result = group.result
        if 2.0 not in result.p_values:
            continue
        direction = "lambda" if "lambda" in result.direction_names else _science_axes(result)[2][0]
        ell = _centers(result.ell_bin_edges)
        for axis, q_name in zip(axes, ("B", "u")):
            if q_name not in result.q_names:
                continue
            index = _moment_index(result, q_name, direction, p_value=2.0)
            supported = _curve_support_mask(group, index)
            values = np.sqrt(np.where(supported, result.moments[index], np.nan))
            low_values = group.uncertainty["block_bootstrap_interval_low"][index]
            high_values = group.uncertainty["block_bootstrap_interval_high"][index]
            low = np.sqrt(
                np.where(
                    supported & (low_values > 0.0),
                    low_values,
                    np.nan,
                )
            )
            high = np.sqrt(
                np.where(
                    supported & (high_values > 0.0),
                    high_values,
                    np.nan,
                )
            )
            valid = np.isfinite(values) & (values > 0.0)
            axis.plot(
                ell[valid],
                values[valid],
                color=colors[group.scale],
                linestyle=line_styles[selection_set],
                linewidth=1.45,
                alpha=0.72,
            )
            axis.fill_between(
                ell[valid],
                low[valid],
                high[valid],
                color=colors[group.scale],
                alpha=0.10,
                linewidth=0.0,
            )
    for axis, q_name in zip(axes, ("B", "u")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(rf"$[S_2^{{{q_name}}}(\ell)]^{{1/2}}$")
        axis.grid(alpha=0.22)
    if groups:
        for scale in scales:
            axes[0].plot([], [], color=colors[scale], label=f"L={scale}")
        for cohort in cohorts:
            axes[0].plot(
                [],
                [],
                color="#555555",
                linestyle=line_styles[cohort],
                label=cohort,
            )
        axes[0].legend(fontsize=7.2, ncol=2)
    else:
        axes[0].text(0.5, 0.5, "No retained representative 2-point p=2 curves", ha="center")
    figure.suptitle("Representative rooted p=2 curves across retained scales; lambda direction")
    return _save(figure, output_dir, FIGURE_FILENAMES[2])


def support_fraction_figure(releases: Sequence[VerifiedRelease], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True, sharey=True)
    for axis, mode in zip(axes, (PRIMARY_SUPPORT_MODE, OVERLAY_SUPPORT_MODE)):
        for release in releases:
            for group in release.groups.values():
                if group.support_mode != mode:
                    continue
                axis.plot(
                    _centers(group.result.ell_bin_edges),
                    _support_fraction(group.result),
                    linewidth=0.8,
                    alpha=0.32,
                    label=(
                        f"L={release.scale} {group.stencil_width}-point"
                        f" {release.selection_set}/{release.matrix}"
                        if group.cube_id == next(iter(release.phase2_sources))
                        else ""
                    ),
                )
        if mode == OVERLAY_SUPPORT_MODE:
            axis.axhline(CURVE_SHELL_LOCAL_MINIMUM, color="#e45756", linestyle="--", label="5% curve gate")
            axis.axhline(SLOPE_SHELL_LOCAL_MINIMUM, color="#777777", linestyle=":", label="10% slope gate")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_title(mode.replace("_", " "))
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("eligible-origin fraction")
    figure.suptitle("Non-periodic finite-support fraction versus separation")
    return _save(figure, output_dir, FIGURE_FILENAMES[3])


def effective_block_figure(shell_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    keys = sorted(
        {
            (
                row["L_sub_cells"],
                row["selection_set"],
                row["matrix"],
                row["stencil_width"],
                row["support_mode"],
            )
            for row in shell_rows
        }
    )
    labels = []
    medians = []
    retentions = []
    for scale, selection_set, matrix, width, mode in keys:
        selected = [
            row
            for row in shell_rows
            if (
                row["L_sub_cells"],
                row["selection_set"],
                row["matrix"],
                row["stencil_width"],
                row["support_mode"],
            )
            == (scale, selection_set, matrix, width, mode)
        ]
        labels.append(
            f"L{scale}\n{selection_set}/{matrix}\n{width}pt {mode.replace('_origins', '')}"
        )
        medians.append(np.median([row["accepted_effective_blocks"] for row in selected]))
        retentions.append(np.mean([row["curve_support_gate_passes"] for row in selected]))
    x = np.arange(len(labels))
    axes[0].bar(x, medians, color="#4c78a8")
    axes[1].bar(x, retentions, color="#54a24b")
    axes[0].axhline(MINIMUM_EFFECTIVE_BLOCKS, color="#e45756", linestyle="--")
    axes[0].set_ylabel("median accepted effective blocks")
    axes[1].set_ylabel("curve-bin retention fraction")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Effective-block support and retained curve-bin diagnostics")
    return _save(figure, output_dir, FIGURE_FILENAMES[4])


def policy_factor_figure(policy_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    retained = [row for row in policy_rows if row["supported_policy_diagnostic"]]
    cohorts = sorted({row["selection_set"] for row in retained})
    markers = {cohort: ("s" if cohort == "matched_smoke" else "o") for cohort in cohorts}
    for p_value, cohort in sorted(
        {(row["p_value"], row["selection_set"]) for row in retained}
    ):
        rows = [
            row
            for row in retained
            if row["p_value"] == p_value and row["selection_set"] == cohort
        ]
        axis.scatter(
            [row["L_sub_cells"] for row in rows],
            [row["policy_factor"] for row in rows],
            s=13,
            alpha=0.24,
            marker=markers[cohort],
            label=f"{cohort} p={p_value:g}",
        )
    axis.axhline(1.0, color="#777777", linestyle="--")
    axis.set_xlabel(r"$L_{\rm sub}$ [cells]")
    axis.set_ylabel("all-valid vs shell-local policy factor")
    axis.set_yscale("log")
    axis.grid(alpha=0.22)
    if retained:
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "No policy-factor rows pass support gates", ha="center")
    axis.set_title("Supported finite-support policy sensitivity by scale and order")
    return _save(figure, output_dir, FIGURE_FILENAMES[5])


def slope_figure(slope_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True, sharey=True)
    retained = [
        row
        for row in slope_rows
        if row["supported_local_slope_diagnostic"]
        and row["support_mode"] == PRIMARY_SUPPORT_MODE
        and row["stencil_width"] == 2
        and row["p_value"] == 2.0
        and row["direction"] == "lambda"
        and row["q_name"] in {"B", "u"}
    ]
    scales = sorted({row["L_sub_cells"] for row in retained})
    cohorts = sorted({row["selection_set"] for row in retained})
    colors = {
        scale: plt.cm.viridis(index / max(1, len(scales) - 1))
        for index, scale in enumerate(scales)
    }
    line_styles = {
        cohort: ("-", "--", ":", "-.")[index % 4]
        for index, cohort in enumerate(cohorts)
    }
    for axis, q_name in zip(axes, ("B", "u")):
        selected = [
            row
            for row in retained
            if row["q_name"] == q_name
        ]
        for (selection_set, scale, cube_id) in sorted(
            {
                (row["selection_set"], row["L_sub_cells"], row["cube_id"])
                for row in selected
            }
        ):
            rows = [
                row
                for row in selected
                if row["selection_set"] == selection_set
                and row["L_sub_cells"] == scale
                and row["cube_id"] == cube_id
            ]
            axis.plot(
                [row["ell_cells"] for row in rows],
                [row["local_log_slope"] for row in rows],
                linewidth=1.0,
                alpha=0.54,
                color=colors[scale],
                linestyle=line_styles[selection_set],
            )
            axis.fill_between(
                [row["ell_cells"] for row in rows],
                [row["local_log_slope_bootstrap_interval_low"] for row in rows],
                [row["local_log_slope_bootstrap_interval_high"] for row in rows],
                color=colors[scale],
                alpha=0.10,
            )
        axis.set_xscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_title(q_name)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("centered-window local log slope")
    if axes[0].lines:
        for scale in scales:
            axes[0].plot([], [], color=colors[scale], label=f"L={scale}")
        for cohort in cohorts:
            axes[0].plot(
                [],
                [],
                color="#555555",
                linestyle=line_styles[cohort],
                label=cohort,
            )
        axes[0].legend(fontsize=7.2, ncol=2)
    figure.suptitle("Local-slope uncertainty support across scale; diagnostics only, not fitted exponents")
    return _save(figure, output_dir, FIGURE_FILENAMES[6])


def stencil_figure(stencil_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), constrained_layout=True)
    retained = [row for row in stencil_rows if row["supported_labeled_stencil_diagnostic"]]
    for q_name, axis in zip(("B", "u"), axes):
        rows = [row for row in retained if row["q_name"] == q_name]
        if rows:
            plotted = axis.scatter(
                [row["ell_cells"] for row in rows],
                [row["3point_over_2point_ratio"] for row in rows],
                c=[row["L_sub_cells"] for row in rows],
                cmap="viridis",
                s=17,
                alpha=0.68,
            )
            figure.colorbar(plotted, ax=axis, label=r"$L_{\rm sub}$ [cells]")
        else:
            axis.text(0.5, 0.5, "No supplied comparable 3-point product", ha="center")
        axis.axhline(1.0, color="#777777", linestyle="--")
        axis.set_xscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel("3-point / 2-point")
        axis.set_title(q_name)
        axis.grid(alpha=0.22)
    figure.suptitle("Labeled stencil comparison where supplied; products remain distinct")
    return _save(figure, output_dir, FIGURE_FILENAMES[7])


def runtime_figure(
    runtime_rows: Sequence[Mapping[str, Any]],
    ledger_summary: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    labels = [row["release_label"] for row in runtime_rows]
    x = np.arange(len(labels))
    axes[0].bar(
        x,
        [row["estimator_elapsed_seconds_sum"] / 3600.0 for row in runtime_rows],
        label="estimator",
        color="#4c78a8",
    )
    axes[0].bar(
        x,
        [row["reduction_elapsed_seconds_sum"] / 3600.0 for row in runtime_rows],
        bottom=[row["estimator_elapsed_seconds_sum"] / 3600.0 for row in runtime_rows],
        label="reduction",
        color="#f58518",
    )
    axes[1].bar(
        x,
        [
            row["settled_release_logical_bytes_before_summary_marker"] / 1024**3
            for row in runtime_rows
        ],
        color="#54a24b",
    )
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("summed marker-recorded hours")
    axes[1].set_ylabel("settled logical release GiB")
    axes[0].legend(fontsize=8)
    metrics = ledger_summary["metrics_node_hours"]
    figure.suptitle(
        "Runtime/storage summary; ledger node-hours: "
        f"consumed={metrics['consumed_allocated_runtime']:.3f}, "
        f"remaining={metrics['remaining_budget']:.3f}"
    )
    return _save(figure, output_dir, FIGURE_FILENAMES[8])


def outlier_figure(
    selections: Sequence[Selection],
    releases: Sequence[VerifiedRelease],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), constrained_layout=True)
    outlier_by_cube: dict[tuple[str, int, str], Selection] = {}
    for selection in selections:
        if not selection.outlier:
            continue
        key = (selection.selection_set, selection.l_sub, selection.cube_id)
        current = outlier_by_cube.get(key)
        if current is None or selection.matrix == "baseline":
            outlier_by_cube[key] = selection
    outliers = [outlier_by_cube[key] for key in sorted(outlier_by_cube)]
    scales = sorted({selection.l_sub for selection in outliers})
    colors = {
        scale: plt.cm.viridis(index / max(1, len(scales) - 1))
        for index, scale in enumerate(scales)
    }
    by_label = {release.label: release for release in releases}
    for selection in outliers:
        env = selection.environment
        color = colors[selection.l_sub]
        axes[0].scatter(env["dBB"], env["B_mean"], color=color, s=48)
        release = by_label[selection.release_label]
        key = (selection.cube_id, 2, PRIMARY_SUPPORT_MODE)
        if key not in release.groups:
            continue
        group = release.groups[key]
        result = group.result
        if 2.0 not in result.p_values:
            continue
        direction = "lambda" if "lambda" in result.direction_names else _science_axes(result)[2][0]
        ell = _centers(result.ell_bin_edges)
        for axis, q_name in zip(axes[1:], ("B", "u")):
            if q_name not in result.q_names:
                continue
            index = _moment_index(result, q_name, direction, p_value=2.0)
            supported = _curve_support_mask(group, index)
            values = np.sqrt(np.where(supported, result.moments[index], np.nan))
            axis.plot(ell, values, color=color, linewidth=1.4, alpha=0.78)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("dBB")
    axes[0].set_ylabel("B_mean")
    for axis, q_name in zip(axes[1:], ("B", "u")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(rf"$[S_2^{{{q_name}}}]^{{1/2}}$")
    for axis in axes:
        axis.grid(alpha=0.22)
    if outliers:
        for scale in scales:
            axes[0].scatter([], [], color=colors[scale], label=f"L={scale}")
        axes[0].legend(fontsize=7.5, title=r"$L_{\rm sub}$")
    else:
        axes[0].text(0.5, 0.5, "No configured outlier selections", ha="center")
    figure.suptitle("Explicitly labeled outlier panel; diagnostic context only")
    return _save(figure, output_dir, FIGURE_FILENAMES[9])


def _phase4_baseline_selections(
    baseline: VerifiedRelease | None,
) -> tuple[Selection, ...]:
    if baseline is None:
        return ()
    rows = []
    for cube_id, role in DEFAULT_PHASE4_BASELINE_REPRESENTATIVES:
        if (cube_id, 2, PRIMARY_SUPPORT_MODE) not in baseline.groups:
            continue
        rows.append(
            Selection(
                release_label=baseline.label,
                selection_set="phase4_baseline",
                matrix="baseline",
                cube_id=cube_id,
                l_sub=640,
                role=f"Phase4 baseline {role}",
                physical_region_id=cube_id,
                matched_group=None,
                observational_control_set=None,
                representative=True,
                outlier="outlier" in role,
                environment={name: float("nan") for name in ENVIRONMENT_FIELDS},
                source_row={"source": "retained Phase 4 Batch A baseline defaults"},
            )
        )
    return tuple(rows)


def _assert_output_available(output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise RuntimeError(f"refusing to publish into non-empty output: {output_dir}")


def _publish_atomic_directory(temporary: Path, output_dir: Path) -> None:
    _assert_output_available(output_dir)
    if output_dir.exists():
        output_dir.rmdir()
    temporary.replace(output_dir)


def _write_hash_manifest(
    output_dir: Path,
    *,
    input_hashes: InputHashes,
    release_roots: Mapping[str, str],
) -> None:
    artifacts = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    _atomic_write_json(
        output_dir / HASH_MANIFEST_FILENAME,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "publication_policy": "immutable_atomic_directory_publish_refuse_nonempty_output",
            "generator_sha256": file_sha256(Path(__file__).resolve()),
            "release_roots": dict(release_roots),
            "input_sha256": input_hashes.as_dict(),
            "generated_artifacts_before_manifest": artifacts,
            "generated_artifact_sha256": {
                name: file_sha256(output_dir / name) for name in artifacts
            },
        },
    )


def generate_report(
    *,
    campaign_config: Path,
    phase1_root: Path,
    release_paths: Mapping[str, Path],
    decision_record_path: Path,
    ledger_summary_path: Path,
    output_dir: Path,
    phase4_batch_a_root: Path | None = None,
) -> None:
    output_dir = output_dir.resolve()
    _assert_output_available(output_dir)
    input_hashes = InputHashes()
    config, release_metadata = _verify_config(
        campaign_config, phase1_root, release_paths, input_hashes
    )
    execution_decision, decision_snapshot = _bind_execution_decision(
        decision_record_path,
        campaign_config=campaign_config,
        release_paths=release_paths,
        input_hashes=input_hashes,
    )
    ledger_summary, ledger_snapshot = _bind_ledger_summary(
        ledger_summary_path, input_hashes
    )
    releases = [
        _verify_release(label, release_paths[label], release_metadata[label], input_hashes)
        for label in sorted(release_paths)
    ]
    selections = _normalize_selections(
        config,
        release_metadata,
        {release.label: release.phase2_sources for release in releases},
    )
    release_by_label = {release.label: release for release in releases}
    for selection in selections:
        if selection.cube_id not in release_by_label[selection.release_label].phase2_sources:
            raise RuntimeError(
                f"selection {selection.cube_id} is absent from release "
                f"{selection.release_label!r} Phase 2 source bindings"
            )
    baseline = None
    if phase4_batch_a_root is not None:
        baseline = _verify_release(
            "phase4_batch_a_baseline",
            phase4_batch_a_root,
            {
                "L_sub": 640,
                "summary_filename": "phase4_batch_a_summary.json",
                "summary_marker_filename": "PHASE4_BATCH_A_COMPLETE.json",
                "report_selection_set": "phase4_baseline",
                "report_matrix": "baseline",
            },
            input_hashes,
        )
    all_releases = [*releases, *([baseline] if baseline is not None else [])]
    curve_selections = (*selections, *_phase4_baseline_selections(baseline))

    selection_rows = _selection_rows(selections)
    shell_rows = _shell_rows(all_releases)
    policy_rows = _policy_rows(all_releases)
    slope_rows = _slope_rows(all_releases)
    stencil_rows = _stencil_rows(selections, releases)
    runtime_rows = _runtime_rows(all_releases)
    matrix_availability_rows = _matrix_availability_rows(releases)
    matrix_availability_policy = {
        str(scale): list(matrices)
        for scale, matrices in sorted(MATRIX_AVAILABILITY_BY_SCALE.items())
    }
    outlier_rows = _outlier_rows(selections)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _atomic_write_text(temporary / LEDGER_SNAPSHOT_FILENAME, ledger_snapshot.decode())
        _atomic_write_text(temporary / DECISION_SNAPSHOT_FILENAME, decision_snapshot)
        tables = {
            "scale_selection_lineage_environment_census": _write_table_pair(
                temporary,
                json_filename=SELECTION_JSON_FILENAME,
                csv_filename=SELECTION_CSV_FILENAME,
                description=(
                    "Scale-specific selector lineage with dBB, B_mean, deltaB, B_rms, "
                    "and bounded magnetic complements reported beside one another."
                ),
                rows=selection_rows,
                extra={
                    "matrix_availability_policy_by_scale": matrix_availability_policy,
                    "retained_release_matrix_availability": matrix_availability_rows,
                },
            ),
            "shell_support_effective_block_diagnostics": _write_table_pair(
                temporary,
                json_filename=SHELL_JSON_FILENAME,
                csv_filename=SHELL_CSV_FILENAME,
                description=(
                    "Shell-resolved finite-support, accepted-count, effective-block, and "
                    "retention diagnostics."
                ),
                rows=shell_rows,
            ),
            "all_valid_vs_shell_local_policy_factor": _write_table_pair(
                temporary,
                json_filename=POLICY_JSON_FILENAME,
                csv_filename=POLICY_CSV_FILENAME,
                description=(
                    "Supported all-valid-origins versus shell-local factors by scale and "
                    "order. Factors are diagnostics, not corrections."
                ),
                rows=policy_rows,
            ),
            "local_slope_uncertainty_across_scale": _write_table_pair(
                temporary,
                json_filename=SLOPE_JSON_FILENAME,
                csv_filename=SLOPE_CSV_FILENAME,
                description=(
                    "Centered-window local-slope values and bootstrap intervals where "
                    "supported. No fitted exponent is published."
                ),
                rows=slope_rows,
            ),
            "labeled_2point_vs_3point_comparison": _write_table_pair(
                temporary,
                json_filename=STENCIL_JSON_FILENAME,
                csv_filename=STENCIL_CSV_FILENAME,
                description=(
                    "Labeled 2-point versus 3-point comparison where both products were "
                    "supplied for one scale-specific physical-region lineage."
                ),
                rows=stencil_rows,
                extra={
                    "comparison_available": bool(stencil_rows),
                    "matrix_availability_policy_by_scale": matrix_availability_policy,
                    "retained_release_matrix_availability": matrix_availability_rows,
                },
            ),
            "runtime_storage_summary": _write_table_pair(
                temporary,
                json_filename=RUNTIME_JSON_FILENAME,
                csv_filename=RUNTIME_CSV_FILENAME,
                description=(
                    "Marker-recorded runtime and storage accounting. Scheduler node-hours "
                    "remain separate in the bound ledger snapshot."
                ),
                rows=runtime_rows,
            ),
            "outlier_panel": _write_table_pair(
                temporary,
                json_filename=OUTLIER_JSON_FILENAME,
                csv_filename=OUTLIER_CSV_FILENAME,
                description="Explicitly labeled outlier selection census.",
                rows=outlier_rows,
            ),
        }
        workflow_schematic(temporary)
        environment_census_figure(selections, temporary)
        representative_curves_figure(curve_selections, all_releases, temporary)
        support_fraction_figure(all_releases, temporary)
        effective_block_figure(shell_rows, temporary)
        policy_factor_figure(policy_rows, temporary)
        slope_figure(slope_rows, temporary)
        stencil_figure(stencil_rows, temporary)
        runtime_figure(runtime_rows, ledger_summary, temporary)
        outlier_figure(selections, releases, temporary)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "phase5_cross_scale_report_package_generated",
            "decision_scope": "diagnostic report for explicitly supplied retained releases only",
            "publication_policy": "immutable_atomic_directory_publish_refuse_nonempty_output",
            "campaign_config": {
                "source_path": str(campaign_config.resolve()),
                "source_sha256": file_sha256(campaign_config),
                "bounded_campaign_label": config.get("campaign_label"),
            },
            "execution_decision": {
                "source_path": str(decision_record_path.resolve()),
                "source_sha256": file_sha256(decision_record_path),
                "snapshot_relative_path": DECISION_SNAPSHOT_FILENAME,
                "snapshot_sha256": file_sha256(temporary / DECISION_SNAPSHOT_FILENAME),
                "scope": execution_decision,
            },
            "phase1_root": str(phase1_root.resolve()),
            "retained_release_roots": {
                release.label: str(release.root) for release in all_releases
            },
            "strict_release_verification": {
                release.label: release.verification for release in all_releases
            },
            "executed_scales_cells": sorted({release.scale for release in releases}),
            "phase4_L640_baseline_included": baseline is not None,
            "configured_unique_selection_count": len(selection_rows),
            "retained_release_selection_binding_count": len(selections),
            "outlier_selection_count": len(outlier_rows),
            "selection_sets_reported_separately": sorted(
                {selection.selection_set for selection in selections}
            ),
            "matrix_availability": matrix_availability_rows,
            "matrix_availability_policy_by_scale": matrix_availability_policy,
            "observational_control_policy": {
                "matched_smoke_selection_set_is_separate": True,
                "matched_smoke_is_not_merged_with_nearest_descendant_lineages": True,
            },
            "optional_extensions_reported": {
                "labeled_3point_comparison": bool(stencil_rows),
                "phase4_L640_baseline": baseline is not None,
            },
            "tables": tables,
            "generated_figures": list(FIGURE_FILENAMES),
            "compute_ledger_summary_snapshot": {
                **ledger_summary,
                "snapshot_relative_path": LEDGER_SNAPSHOT_FILENAME,
                "snapshot_sha256": file_sha256(temporary / LEDGER_SNAPSHOT_FILENAME),
            },
            "interpretation_policy": {
                "statements_are_diagnostic": True,
                "directional_fitted_slopes_published": False,
                "fitted_exponents_published": False,
                "local_slopes_are_centered_window_uncertainty_diagnostics_only": True,
                "support_policy_factors_are_diagnostics_not_corrections": True,
                "different_L_sub_values_remain_distinct_environmental_labels": True,
            },
            "known_limits": [
                "This package reports only releases explicitly listed on the command line and in the campaign config.",
                "Deferred scales, shifted tilings, additional snapshots, and absent diagnostics are not claimed as validated.",
                "Local slopes are diagnostic centered-window products; no fitted exponent claim is made.",
                "Labeled 2-point and 3-point statistics remain distinct products.",
                "Measured sampler timings and marker-recorded bytes remain distinct from scheduler node-hours in the bound ledger snapshot.",
                "The L320 matched_smoke observational-control cohort remains separately labeled and is not merged with nearest-descendant lineages.",
            ],
            "input_sha256": input_hashes.as_dict(),
        }
        _atomic_write_json(temporary / SUMMARY_FILENAME, summary)
        _write_hash_manifest(
            temporary,
            input_hashes=input_hashes,
            release_roots={release.label: str(release.root) for release in all_releases},
        )
        _publish_atomic_directory(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        f"Wrote immutable Phase 5 cross-scale report package: {output_dir} "
        f"({len(FIGURE_FILENAMES)} figures)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an immutable retained-artifact Phase 5 cross-scale report."
    )
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--decision-record", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument(
        "--release",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="retained Phase 5 sampler release; repeat for every approved release",
    )
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase4-batch-a-root",
        type=Path,
        help="optional retained L640 Phase 4 Batch A baseline release",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        releases = _release_specs(args.release)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    generate_report(
        campaign_config=args.campaign_config,
        decision_record_path=args.decision_record,
        phase1_root=args.phase1_root,
        release_paths=releases,
        ledger_summary_path=args.ledger_summary,
        output_dir=args.output_dir,
        phase4_batch_a_root=args.phase4_batch_a_root,
    )


if __name__ == "__main__":
    main()
