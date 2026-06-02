#!/usr/bin/env python3
"""Publish the immutable Phase 4 completion supplement after all-21 Batch B.

The supplement is deliberately report-only.  It replays retained publication
markers, reduction bindings, and the strict historical Batch A verifier before
deriving any quantitative output.  Equal-SF inverse scales are exploratory
diagnostics with explicit support gates; this generator never publishes
directional fitted slopes or fitted exponents.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import json
import math
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
from scripts.phase3a import generate_phase3a_status_figures as figures
from scripts.phase3a import run_phase3a_sampler as runner
from scripts.phase4 import generate_phase4_batch_a_status_figures as batch_a_report
from scripts.phase4 import generate_phase4_batch_a2_review as batch_a2_review
from scripts.phase4 import generate_phase4_batch_b_representative_review as batch_b_review
from sfunctor.analysis import finite_domain as finite_domain_analysis
from sfunctor.core.finite_domain import FiniteDomainResult


DEFAULT_OUTPUT_DIR = Path("figures/phase4_completion_supplement")
SUMMARY_FILENAME = "phase4_completion_supplement_summary.json"
LEDGER_SNAPSHOT_FILENAME = "phase4_completion_compute_ledger_summary_snapshot.md"
FIGURE_MANIFEST_FILENAME = "figure_manifest.json"

CONDITIONING_JSON_FILENAME = "phase4_completion_conditioning_comparison.json"
CONDITIONING_CSV_FILENAME = "phase4_completion_conditioning_comparison.csv"
ASPECT_JSON_FILENAME = "phase4_completion_equal_sf_aspect_ratio.json"
ASPECT_CSV_FILENAME = "phase4_completion_equal_sf_aspect_ratio_quality_census.csv"
ORDER_JSON_FILENAME = "phase4_completion_batch_b_order_sensitivity.json"
ORDER_CSV_FILENAME = "phase4_completion_batch_b_order_sensitivity.csv"
ENVIRONMENT_JSON_FILENAME = "phase4_completion_sf_environment.json"
ENVIRONMENT_CSV_FILENAME = "phase4_completion_sf_environment.csv"
STENCIL_JSON_FILENAME = "phase4_completion_labeled_stencil_comparison.json"
STENCIL_CSV_FILENAME = "phase4_completion_labeled_stencil_comparison.csv"
RUNTIME_JSON_FILENAME = "phase4_completion_runtime_storage_ledger.json"
RUNTIME_CSV_FILENAME = "phase4_completion_runtime_storage_ledger.csv"

ALL21_3POINT_SUMMARY_FILENAME = "phase4_batch_a2_3point_extension_summary.json"
ALL21_3POINT_MARKER_FILENAME = "PHASE4_BATCH_A2_3POINT_EXTENSION_COMPLETE.json"
ALL21_BATCH_B_SUMMARY_FILENAME = "phase4_batch_b_all21_extension_summary.json"
ALL21_BATCH_B_MARKER_FILENAME = "PHASE4_BATCH_B_ALL21_EXTENSION_COMPLETE.json"

SUPPORT_MODES = batch_a2_review.SUPPORT_MODES
Q_NAMES = batch_a2_review.Q_NAMES
DIRECTIONS = batch_a2_review.DIRECTIONS
BATCH_B_P_VALUES = batch_b_review.P_VALUES
PRIMARY_SUPPORT_MODE = "all_valid_origins"
P2 = 2.0
SCIENCE_SCALE_MINIMUM = 32.0
EQUAL_SF_INVERSION_ELL_INTERVAL = (32.0, 128.0)
EQUAL_SF_TARGET_LOG_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
SF_ENVIRONMENT_SCALE_TARGETS = (32.0, 64.0, 128.0)
SF_ENVIRONMENT_FIGURE_SCALE_TARGET = 64.0
DENOMINATOR_OUTLIER_DBB_MAXIMUM = 5.0
STENCIL_COMPARISON_TARGETS = (32.0, 64.0, 128.0)
MAGNETIC_COVARIATES = (
    "dBB",
    "B_mean",
    "deltaB",
    "B_rms",
    "B_mean_sq_over_B2_mean",
    "deltaB_sq_over_B2_mean",
)
SF_ENVIRONMENT_FIGURE_VARIABLES = (
    *MAGNETIC_COVARIATES,
    "accepted_measurements",
    "directional_excluded_measurements_sum",
)

CONDITIONING_FIGURE_FILENAME = (
    "phase4_completion_pair_local_vs_subvolume_mean_conditioning.png"
)
ASPECT_FIGURE_FILENAME = "phase4_completion_equal_sf_inverse_scale_aspect_ratio.png"
ORDER_FIGURE_FILENAME = (
    "phase4_completion_batch_b_order_sensitivity_vs_magnetic_census.png"
)
ENVIRONMENT_FIGURE_FILENAME = "phase4_completion_rooted_sf_environment.png"
STENCIL_FIGURE_FILENAME = "phase4_completion_labeled_3point_vs_2point.png"
RUNTIME_FIGURE_FILENAME = "phase4_completion_runtime_storage_supplement.png"
FIGURE_FILENAMES = (
    CONDITIONING_FIGURE_FILENAME,
    ASPECT_FIGURE_FILENAME,
    ORDER_FIGURE_FILENAME,
    ENVIRONMENT_FIGURE_FILENAME,
    STENCIL_FIGURE_FILENAME,
    RUNTIME_FIGURE_FILENAME,
)

REPRODUCTION_RESULT_P_AXIS_ARRAY_NAMES = (
    "counts",
    "sums",
    "sums_sq",
    "block_counts",
    "block_sums",
    "block_sums_sq",
    "moments",
    "standard_error",
)
REPRODUCTION_RESULT_EXCLUDED_NAMES = (
    "p_values",
    "elapsed_seconds",
    "elapsed_seconds_per_ell_bin",
)
REPRODUCTION_RESULT_FULL_VALUE_NAMES = tuple(
    field.name
    for field in fields(FiniteDomainResult)
    if field.name
    not in {
        *REPRODUCTION_RESULT_P_AXIS_ARRAY_NAMES,
        *REPRODUCTION_RESULT_EXCLUDED_NAMES,
    }
)
REPRODUCTION_UNCERTAINTY_P_AXIS_ARRAY_NAMES = (
    "moments",
    "pair_sampling_standard_error",
    "block_jackknife_standard_error",
    "block_bootstrap_standard_error",
    "block_bootstrap_interval_low",
    "block_bootstrap_interval_high",
    "accepted_contributing_blocks",
    "accepted_effective_blocks",
    "valid_bootstrap_resamples",
    "local_log_slope",
    "local_log_slope_support_mask",
    "local_log_slope_bootstrap_standard_error",
    "local_log_slope_bootstrap_interval_low",
    "local_log_slope_bootstrap_interval_high",
    "local_log_slope_valid_bootstrap_resamples",
)
REPRODUCTION_UNCERTAINTY_FULL_ARRAY_NAMES = (
    "sampled_blocks_per_shell",
    "eligible_blocks_per_shell",
)
REPRODUCTION_UNCERTAINTY_NAMES = {
    "metadata_json",
    *REPRODUCTION_UNCERTAINTY_P_AXIS_ARRAY_NAMES,
    *REPRODUCTION_UNCERTAINTY_FULL_ARRAY_NAMES,
}


@dataclass(frozen=True)
class MarkerBoundRelease:
    """One post-Batch-A release verified through its retained marker graph."""

    root: Path
    summary: dict[str, Any]
    campaign: dict[str, Any]
    cube_ids: tuple[str, ...]
    stencil_width: int
    groups: dict[tuple[str, str], batch_a2_review.Group]
    shard_rows: tuple[dict[str, Any], ...]
    verification: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return batch_a_report._load_json(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    batch_a_report._write_json(path, payload)


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _native(value: Any) -> Any:
    return batch_a_report.inherited._json_builtin(value)


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _positive_or_none(value: Any) -> float | None:
    numeric = _finite_or_none(value)
    return numeric if numeric is not None and numeric > 0.0 else None


def _factor_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    return {
        "count": int(array.size),
        "median": float(np.median(array)) if array.size else None,
        "p90": float(np.quantile(array, 0.90)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
    }


def _summary_by(
    rows: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str],
    value_name: str,
) -> list[dict[str, Any]]:
    keys = sorted({tuple(row[name] for name in dimensions) for row in rows})
    output = []
    for key in keys:
        selected = [
            row
            for row in rows
            if tuple(row[name] for name in dimensions) == key
            and _positive_or_none(row.get(value_name)) is not None
        ]
        output.append(
            {
                **dict(zip(dimensions, key)),
                **_factor_summary(float(row[value_name]) for row in selected),
            }
        )
    return output


def _csv_cell(value: Any) -> Any:
    value = _native(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_table_pair(
    output_dir: Path,
    *,
    json_filename: str,
    csv_filename: str,
    payload: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    json_path = output_dir / json_filename
    csv_path = output_dir / csv_filename
    _write_json(json_path, payload)
    fieldnames = sorted({name for row in csv_rows for name in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: _csv_cell(value) for name, value in row.items()}
            for row in csv_rows
        )
    return json_path, csv_path


def _expected_group_id(cube_id: str, stencil_width: int, support_mode: str) -> str:
    return f"{cube_id}/stencil_{stencil_width}point/{support_mode}"


def _canonical_shards(
    cube_ids: Sequence[str],
    *,
    stencil_width: int,
    support_modes: Sequence[str],
    displacement_count: int,
    offsets_per_shard: int,
) -> list[dict[str, Any]]:
    rows = []
    for cube_id in cube_ids:
        for support_mode in support_modes:
            group_id = _expected_group_id(cube_id, stencil_width, support_mode)
            for shard_index, start in enumerate(range(0, displacement_count, offsets_per_shard)):
                rows.append(
                    {
                        "group_id": group_id,
                        "shard_id": f"{group_id}/shard_{shard_index:04d}",
                        "cube_id": cube_id,
                        "stencil_width": stencil_width,
                        "support_mode": support_mode,
                        "shard_index": shard_index,
                        "offset_start": start,
                        "offset_stop": min(displacement_count, start + offsets_per_shard),
                    }
                )
    return rows


def _sampling_schedule_sha256(
    row: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> str:
    return runner._mapping_sha256(
        {
            "group_id": row["group_id"],
            "shard_id": row["shard_id"],
            "offset_start": row["offset_start"],
            "offset_stop": row["offset_stop"],
            "sample_count": configuration["sample_count_per_displacement"],
            "pair_batch_size": configuration["pair_batch_size"],
            "seed": configuration["production_seed"],
            "block_shape_kji": configuration["block_shape_kji"],
            "block_assignment": configuration["block_assignment"],
            "q_names": configuration["q_names"],
            "p_values": configuration["p_values"],
            "density_conventions": configuration["density_conventions"],
        }
    )


def _validate_result_matrix(
    result: Any,
    *,
    label: str,
    stencil_width: int,
    support_mode: str,
    q_names: Sequence[str],
    p_values: Sequence[float],
    density_conventions: Sequence[str],
    configuration: Mapping[str, Any],
    displacement_metadata: Mapping[str, Any],
) -> None:
    if (
        result.stencil_width != stencil_width
        or result.pair_mode != support_mode
        or tuple(result.q_names) != tuple(q_names)
        or tuple(result.p_values) != tuple(p_values)
        or tuple(result.density_conventions) != tuple(density_conventions)
        or not {"pair_local", "subvolume_mean"}.issubset(result.geometry_names)
        or not set(DIRECTIONS).issubset(result.direction_names)
        or result.sample_count != configuration["sample_count_per_displacement"]
        or result.pair_batch_size != configuration["pair_batch_size"]
        or result.seed != configuration["production_seed"]
        or tuple(result.block_shape_kji) != tuple(configuration["block_shape_kji"])
        or result.block_assignment != configuration["block_assignment"]
        or result.support_displacements_sha256
        != displacement_metadata["offsets_sha256"]
        or result.support_displacement_count
        != displacement_metadata["realized_offset_count"]
    ):
        raise RuntimeError(f"retained release result matrix mismatch: {label}")


def _verify_marker_bound_release(
    *,
    root: Path,
    extraction_root: Path,
    summary_filename: str,
    marker_filename: str,
    cube_ids: tuple[str, ...],
    stencil_width: int,
    q_names: tuple[str, ...],
    p_values: tuple[float, ...],
    density_conventions: tuple[str, ...],
    input_hashes: batch_a_report.InputHashes,
    groups: Mapping[tuple[str, str], batch_a2_review.Group] | None = None,
) -> MarkerBoundRelease:
    """Replay one current-style Phase 3a marker graph without live source replay."""

    root = root.resolve()
    extraction_root = extraction_root.resolve()
    campaign_path = root / "manifests" / "campaign.json"
    shards_path = root / "manifests" / "shards.json"
    plan_marker_path = root / "PLAN_COMPLETE.json"
    inherited_summary_path = root / "phase3a_summary.json"
    inherited_marker_path = root / "PHASE3A_RELEASE_COMPLETE.json"
    campaign = _load_json(campaign_path)
    shard_payload = _load_json(shards_path)
    plan_marker = _load_json(plan_marker_path)
    inherited_summary = _load_json(inherited_summary_path)
    inherited_marker = _load_json(inherited_marker_path)
    summary = batch_a2_review._verify_release_marker(
        root,
        summary_filename=summary_filename,
        marker_filename=marker_filename,
        input_hashes=input_hashes,
    )
    input_hashes.add_many(
        (
            campaign_path,
            shards_path,
            plan_marker_path,
            inherited_summary_path,
            inherited_marker_path,
        )
    )
    implementation_sha256 = campaign.get("source_version", {}).get("implementation_sha256")
    configuration = campaign.get("configuration", {})
    expected_stencils = {str(stencil_width): dict(runner.STENCIL_SPECS[stencil_width])}
    if (
        not isinstance(implementation_sha256, str)
        or len(implementation_sha256) != 64
        or campaign.get("schema_version") != runner.SCHEMA_VERSION
        or campaign.get("status") != "planned"
        or Path(str(campaign.get("phase2_root", ""))).resolve() != extraction_root
        or campaign.get("configuration_sha256")
        != runner._mapping_sha256(configuration)
        or tuple(configuration.get("q_names", ())) != q_names
        or tuple(configuration.get("p_values", ())) != p_values
        or tuple(configuration.get("density_conventions", ())) != density_conventions
        or tuple(configuration.get("support_modes", ())) != tuple(SUPPORT_MODES)
        or configuration.get("stencils") != expected_stencils
        or set(campaign.get("phase2_sources", {})) != set(cube_ids)
        or plan_marker.get("schema_version") != runner.SCHEMA_VERSION
        or plan_marker.get("status") != "passed"
        or plan_marker.get("campaign_sha256") != file_sha256(campaign_path)
        or plan_marker.get("shards_sha256") != file_sha256(shards_path)
        or plan_marker.get("implementation_sha256") != implementation_sha256
        or inherited_marker.get("schema_version") != runner.SCHEMA_VERSION
        or inherited_marker.get("status") != "release_aggregation_complete"
        or inherited_marker.get("summary_sha256") != file_sha256(inherited_summary_path)
        or inherited_marker.get("implementation_sha256") != implementation_sha256
        or inherited_summary.get("source_version", {}).get("implementation_sha256")
        != implementation_sha256
        or summary.get("source_version", {}).get("implementation_sha256")
        != implementation_sha256
    ):
        raise RuntimeError(f"invalid or stale marker-bound retained release: {root}")

    displacement_metadata, displacements, _ = runner._load_displacement_manifest(
        root, stencil_width
    )
    displacement_json, displacement_npz = runner._manifest_paths(root, stencil_width)
    input_hashes.add_many((displacement_json, displacement_npz))
    displacement_binding = campaign.get("displacement_manifests", {}).get(
        str(stencil_width), {}
    )
    if (
        displacement_binding.get("json_relative_path")
        != str(displacement_json.relative_to(root))
        or displacement_binding.get("json_sha256") != file_sha256(displacement_json)
        or displacement_binding.get("npz_relative_path")
        != str(displacement_npz.relative_to(root))
        or displacement_binding.get("npz_sha256") != file_sha256(displacement_npz)
        or displacement_binding.get("manifest_sha256")
        != displacement_metadata["manifest_sha256"]
        or displacement_binding.get("offset_count") != len(displacements)
    ):
        raise RuntimeError(f"retained release lost displacement-manifest binding: {root}")

    shard_rows = shard_payload.get("shards")
    expected_shards = _canonical_shards(
        cube_ids,
        stencil_width=stencil_width,
        support_modes=SUPPORT_MODES,
        displacement_count=len(displacements),
        offsets_per_shard=int(configuration["offsets_per_shard"]),
    )
    if (
        not isinstance(shard_rows, list)
        or shard_rows != expected_shards
        or campaign.get("shard_count") != len(expected_shards)
    ):
        raise RuntimeError(f"retained release shard inventory is not canonical: {root}")
    shard_marker_sha256: dict[str, str] = {}
    for row in shard_rows:
        _, partial_path, shard_marker_path = runner._shard_paths(root, row["shard_id"])
        shard_marker = _load_json(shard_marker_path)
        if (
            shard_marker.get("schema_version") != runner.SCHEMA_VERSION
            or shard_marker.get("status") != "passed"
            or shard_marker.get("shard") != row
            or shard_marker.get("implementation_sha256") != implementation_sha256
            or shard_marker.get("displacement_manifest_sha256")
            != displacement_metadata["manifest_sha256"]
            or shard_marker.get("support_displacements_sha256")
            != displacement_metadata["offsets_sha256"]
            or shard_marker.get("phase2_source")
            != campaign["phase2_sources"][row["cube_id"]]
            or shard_marker.get("sampling_schedule_sha256")
            != _sampling_schedule_sha256(row, configuration)
            or any(
                not isinstance(shard_marker.get(name), int) or shard_marker[name] < 0
                for name in (
                    "staging_logical_bytes_before_marker",
                    "staging_allocated_bytes_before_marker",
                )
            )
        ):
            raise RuntimeError(f"invalid retained shard marker: {row['shard_id']}")
        input_hashes.bind_verified(
            partial_path, str(shard_marker.get("partial_sha256", ""))
        )
        input_hashes.add(shard_marker_path)
        shard_marker_sha256[row["shard_id"]] = file_sha256(shard_marker_path)

    loaded_groups = dict(groups or {})
    expected_group_ids = {
        _expected_group_id(cube_id, stencil_width, support_mode)
        for cube_id in cube_ids
        for support_mode in SUPPORT_MODES
    }
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in shard_rows:
        rows_by_group.setdefault(row["group_id"], []).append(row)
    if set(rows_by_group) != expected_group_ids:
        raise RuntimeError(f"retained release reduction inventory is not canonical: {root}")
    for cube_id in cube_ids:
        for support_mode in SUPPORT_MODES:
            key = (cube_id, support_mode)
            group = loaded_groups.get(key)
            if group is None:
                group = batch_a2_review._load_group(
                    root, cube_id, stencil_width, support_mode, input_hashes
                )
                loaded_groups[key] = group
            group_id = _expected_group_id(cube_id, stencil_width, support_mode)
            group_root = batch_a2_review._group_root(
                root, cube_id, stencil_width, support_mode
            )
            reduction_manifest_path = group_root / "reduction_manifest.json"
            uncertainty_path = group_root / "uncertainty.npz"
            reduction_manifest = _load_json(reduction_manifest_path)
            ordered_shard_ids = tuple(row["shard_id"] for row in rows_by_group[group_id])
            if (
                reduction_manifest.get("group_id") != group_id
                or tuple(reduction_manifest.get("ordered_shard_ids", ()))
                != ordered_shard_ids
                or reduction_manifest.get("ordered_shard_marker_sha256")
                != {
                    shard_id: shard_marker_sha256[shard_id]
                    for shard_id in ordered_shard_ids
                }
                or reduction_manifest.get("implementation_sha256")
                != implementation_sha256
            ):
                raise RuntimeError(f"invalid retained reduction manifest: {group_id}")
            _validate_result_matrix(
                group.result,
                label=group_id,
                stencil_width=stencil_width,
                support_mode=support_mode,
                q_names=q_names,
                p_values=p_values,
                density_conventions=density_conventions,
                configuration=configuration,
                displacement_metadata=displacement_metadata,
            )
            batch_a_report._verify_uncertainty_payload(
                uncertainty_path, group.result, configuration
            )

    expected_summary_rows = {
        group_id: batch_a_report._summary_group_row(
            loaded_groups[
                (
                    group_id.split("/", 1)[0],
                    group_id.rsplit("/", 1)[1],
                )
            ].result,
            group_id,
        )
        for group_id in expected_group_ids
    }
    for candidate, label in (
        (inherited_summary, "inherited"),
        (summary, "adapter"),
    ):
        rows = candidate.get("groups")
        by_group = {
            str(row.get("group_id")): row for row in rows
        } if isinstance(rows, list) else {}
        if (
            candidate.get("schema_version") != runner.SCHEMA_VERSION
            or candidate.get("operational_status") != "release_aggregation_complete"
            or candidate.get("scientific_acceptance") != "pending_report_level_gate"
            or by_group != expected_summary_rows
        ):
            raise RuntimeError(f"invalid {label} retained release summary inventory: {root}")
    return MarkerBoundRelease(
        root=root,
        summary=summary,
        campaign=campaign,
        cube_ids=cube_ids,
        stencil_width=stencil_width,
        groups=loaded_groups,
        shard_rows=tuple(shard_rows),
        verification={
            "status": "passed",
            "verification_mode": "strict_retained_marker_graph_without_live_source_replay",
            "verified_shards": len(shard_rows),
            "verified_reductions": len(expected_group_ids),
            "historical_implementation_sha256": implementation_sha256,
        },
    )


def _verify_dependency_bindings(
    *,
    batch_a_summary: Mapping[str, Any],
    extension_summary: Mapping[str, Any],
    batch_b_summary: Mapping[str, Any],
    batch_a_root: Path,
    extension_root: Path,
    cube_ids: tuple[str, ...],
) -> None:
    """Require the staged completion products to retain their predecessor roots."""

    if (
        batch_a_summary.get("phase") != "phase4_batch_a_bounded_21_cube_2point"
        or tuple(batch_a_summary.get("pilot_cube_ids", ())) != cube_ids
        or extension_summary.get("phase") != "phase4_batch_a2_all21_3point_extension"
        or tuple(extension_summary.get("pilot_cube_ids", ())) != cube_ids
        or Path(str(extension_summary.get("batch_a_reference_root", ""))).resolve()
        != batch_a_root.resolve()
        or batch_b_summary.get("phase")
        != "phase4_batch_b_all21_2point_p1_to_p6_extension"
        or tuple(batch_b_summary.get("pilot_cube_ids", ())) != cube_ids
        or Path(str(batch_b_summary.get("batch_a_reference_root", ""))).resolve()
        != batch_a_root.resolve()
        or Path(
            str(batch_b_summary.get("all21_3point_extension_reference_root", ""))
        ).resolve()
        != extension_root.resolve()
    ):
        raise RuntimeError("completion supplement inputs are not one bound staged release chain")


def _raise_reproduction_mismatch(label: str) -> None:
    raise RuntimeError(
        f"strict Batch-A to all21-Batch-B p=2 reproduction mismatch: {label}"
    )


def _exact_equal(left: Any, right: Any) -> bool:
    """Return exact equality while treating paired NaNs as equal."""

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape != right_array.shape:
            return False
        if left_array.dtype.kind in "fc" or right_array.dtype.kind in "fc":
            return bool(np.array_equal(left_array, right_array, equal_nan=True))
        return bool(np.array_equal(left_array, right_array))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _exact_equal(left[name], right[name]) for name in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _exact_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    if isinstance(left, (float, np.floating)) and isinstance(
        right, (float, np.floating)
    ):
        return bool(left == right or (np.isnan(left) and np.isnan(right)))
    return bool(left == right)


def _p_slice(
    value: Any,
    *,
    p_index: int,
    p_count: int,
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[-2] != p_count:
        _raise_reproduction_mismatch(f"{label} has no canonical p axis")
    return np.take(array, (p_index,), axis=-2)


def _uncertainty_metadata(
    uncertainty: Mapping[str, np.ndarray],
    *,
    expected_p_values: tuple[float, ...],
    label: str,
) -> dict[str, Any]:
    raw = np.asarray(uncertainty["metadata_json"])
    if raw.shape != () or raw.dtype.kind not in "SU":
        _raise_reproduction_mismatch(f"{label} uncertainty metadata scalar")
    try:
        metadata = json.loads(str(raw.item()))
    except json.JSONDecodeError:
        _raise_reproduction_mismatch(f"{label} uncertainty metadata JSON")
    if (
        not isinstance(metadata, dict)
        or tuple(metadata.get("p_values", ())) != expected_p_values
    ):
        _raise_reproduction_mismatch(f"{label} uncertainty metadata p-values")
    return {**metadata, "p_values": (P2,)}


def _verify_batch_a_to_all21_batch_b_p2_reproduction(
    *,
    cube_ids: tuple[str, ...],
    batch_a_groups: Mapping[tuple[str, str], Any],
    batch_b_groups: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    """Require exact Batch-B p=2 reproduction of every retained Batch-A group."""

    if cube_ids != tuple(batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS):
        _raise_reproduction_mismatch("cube census is not the frozen all-21 inventory")
    expected_keys = {
        (cube_id, support_mode)
        for cube_id in cube_ids
        for support_mode in SUPPORT_MODES
    }
    if set(batch_a_groups) != expected_keys or set(batch_b_groups) != expected_keys:
        _raise_reproduction_mismatch("group inventory")

    batch_a_p_values = (P2,)
    batch_b_p_values = tuple(BATCH_B_P_VALUES)
    batch_b_p2_index = batch_b_p_values.index(P2)
    for cube_id in cube_ids:
        for support_mode in SUPPORT_MODES:
            label = f"{cube_id}/{support_mode}"
            batch_a = batch_a_groups[(cube_id, support_mode)]
            batch_b = batch_b_groups[(cube_id, support_mode)]
            if tuple(batch_a.result.p_values) != batch_a_p_values:
                _raise_reproduction_mismatch(f"{label} Batch A p-values")
            if tuple(batch_b.result.p_values) != batch_b_p_values:
                _raise_reproduction_mismatch(f"{label} Batch B p-values")

            for name in REPRODUCTION_RESULT_FULL_VALUE_NAMES:
                if not _exact_equal(
                    getattr(batch_a.result, name), getattr(batch_b.result, name)
                ):
                    _raise_reproduction_mismatch(f"{label} result.{name}")
            for name in REPRODUCTION_RESULT_P_AXIS_ARRAY_NAMES:
                if not _exact_equal(
                    _p_slice(
                        getattr(batch_a.result, name),
                        p_index=0,
                        p_count=len(batch_a_p_values),
                        label=f"{label} Batch A result.{name}",
                    ),
                    _p_slice(
                        getattr(batch_b.result, name),
                        p_index=batch_b_p2_index,
                        p_count=len(batch_b_p_values),
                        label=f"{label} Batch B result.{name}",
                    ),
                ):
                    _raise_reproduction_mismatch(f"{label} result.{name}")

            if (
                set(batch_a.uncertainty) != REPRODUCTION_UNCERTAINTY_NAMES
                or set(batch_b.uncertainty) != REPRODUCTION_UNCERTAINTY_NAMES
            ):
                _raise_reproduction_mismatch(f"{label} uncertainty inventory")
            if not _exact_equal(
                _uncertainty_metadata(
                    batch_a.uncertainty,
                    expected_p_values=batch_a_p_values,
                    label=f"{label} Batch A",
                ),
                _uncertainty_metadata(
                    batch_b.uncertainty,
                    expected_p_values=batch_b_p_values,
                    label=f"{label} Batch B",
                ),
            ):
                _raise_reproduction_mismatch(f"{label} uncertainty metadata")
            for name in REPRODUCTION_UNCERTAINTY_FULL_ARRAY_NAMES:
                if not _exact_equal(
                    batch_a.uncertainty[name], batch_b.uncertainty[name]
                ):
                    _raise_reproduction_mismatch(f"{label} uncertainty.{name}")
            for name in REPRODUCTION_UNCERTAINTY_P_AXIS_ARRAY_NAMES:
                if not _exact_equal(
                    _p_slice(
                        batch_a.uncertainty[name],
                        p_index=0,
                        p_count=len(batch_a_p_values),
                        label=f"{label} Batch A uncertainty.{name}",
                    ),
                    _p_slice(
                        batch_b.uncertainty[name],
                        p_index=batch_b_p2_index,
                        p_count=len(batch_b_p_values),
                        label=f"{label} Batch B uncertainty.{name}",
                    ),
                ):
                    _raise_reproduction_mismatch(f"{label} uncertainty.{name}")

    return {
        "status": "passed",
        "verification_mode": "exact_arrays_equal_nan",
        "p_value": P2,
        "verified_group_count": len(expected_keys),
        "excluded_metadata": "timing_and_staging_only",
    }


def _catalog_values(
    catalog_rows: Mapping[str, Mapping[str, Any]], cube_id: str
) -> dict[str, float]:
    return {
        name: float(catalog_rows[cube_id]["catalog"][name])
        for name in MAGNETIC_COVARIATES
    }


def _curve_support_mask(group: Any, index: tuple[int, ...]) -> np.ndarray:
    return batch_a_report._curve_uncertainty_support_mask(group, index)


def _conditioning_rows(
    cube_ids: Sequence[str],
    groups: Mapping[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    rows = []
    representative_ids = set(batch_a2_review.REPRESENTATIVE_CUBES)
    for cube_id in cube_ids:
        group = groups[(cube_id, PRIMARY_SUPPORT_MODE)]
        result = group.result
        ell = figures._centers(result.ell_bin_edges)
        for q_name in Q_NAMES:
            for direction in DIRECTIONS:
                pair_index = figures._moment_index(
                    result, q_name, direction, geometry_name="pair_local", p_value=P2
                )
                mean_index = figures._moment_index(
                    result,
                    q_name,
                    direction,
                    geometry_name="subvolume_mean",
                    p_value=P2,
                )
                supported = (
                    (ell >= SCIENCE_SCALE_MINIMUM)
                    & _curve_support_mask(group, pair_index)
                    & _curve_support_mask(group, mean_index)
                )
                for shell_index in np.flatnonzero(ell >= SCIENCE_SCALE_MINIMUM):
                    pair_value = _positive_or_none(result.moments[pair_index][shell_index])
                    mean_value = _positive_or_none(result.moments[mean_index][shell_index])
                    ratio = (
                        pair_value / mean_value
                        if supported[shell_index]
                        and pair_value is not None
                        and mean_value is not None
                        else None
                    )
                    rows.append(
                        {
                            "cube_id": cube_id,
                            "representative_cube": cube_id in representative_ids,
                            "q_name": q_name,
                            "direction": direction,
                            "ell_cells": float(ell[shell_index]),
                            "pair_local_curve_value": pair_value,
                            "subvolume_mean_curve_value": mean_value,
                            "pair_local_over_subvolume_mean": ratio,
                            "conditioning_factor": (
                                max(ratio, 1.0 / ratio)
                                if ratio is not None and ratio > 0.0
                                else None
                            ),
                            "supported_comparison": bool(supported[shell_index]),
                            "pair_local_accepted_measurements": int(
                                result.counts[pair_index][shell_index]
                            ),
                            "subvolume_mean_accepted_measurements": int(
                                result.counts[mean_index][shell_index]
                            ),
                        }
                    )
    return rows


def _common_equal_sf_targets(
    curves: Mapping[str, np.ndarray],
    support_masks: Mapping[str, np.ndarray],
    *,
    target_log_fractions: Sequence[float] = EQUAL_SF_TARGET_LOG_FRACTIONS,
) -> dict[str, Any]:
    """Define common equal-SF levels from the supported directional overlap."""

    ranges = {}
    for direction in DIRECTIONS:
        values = np.asarray(curves[direction], dtype=float)
        mask = np.asarray(support_masks[direction], dtype=bool)
        selected = values[mask & np.isfinite(values) & (values > 0.0)]
        if selected.size < 2:
            return {
                "status": "insufficient_supported_directional_range",
                "targets": np.asarray([], dtype=float),
                "directional_supported_ranges": ranges,
            }
        ranges[direction] = {
            "minimum": float(np.min(selected)),
            "maximum": float(np.max(selected)),
            "supported_bin_count": int(selected.size),
        }
    lower = max(row["minimum"] for row in ranges.values())
    upper = min(row["maximum"] for row in ranges.values())
    fractions = np.asarray(tuple(target_log_fractions), dtype=float)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
        or fractions.ndim != 1
        or not fractions.size
        or np.any(~np.isfinite(fractions))
        or np.any((fractions <= 0.0) | (fractions >= 1.0))
    ):
        return {
            "status": "no_positive_common_supported_directional_overlap",
            "targets": np.asarray([], dtype=float),
            "directional_supported_ranges": ranges,
            "common_overlap_minimum": _finite_or_none(lower),
            "common_overlap_maximum": _finite_or_none(upper),
        }
    targets = np.exp(np.log(lower) + fractions * (np.log(upper) - np.log(lower)))
    return {
        "status": "passed",
        "targets": targets,
        "target_log_fractions": fractions,
        "directional_supported_ranges": ranges,
        "common_overlap_minimum": float(lower),
        "common_overlap_maximum": float(upper),
    }


def _supported_inverse_scale(
    group: Any,
    index: tuple[int, ...],
    target: float,
    *,
    ell_interval: tuple[float, float] = EQUAL_SF_INVERSION_ELL_INTERVAL,
) -> tuple[float | None, float | None, str]:
    """Invert one curve only across contiguous bins passing the report support gate."""

    result = group.result
    ell = figures._centers(result.ell_bin_edges)
    support = (
        _curve_support_mask(group, index)
        & (ell >= ell_interval[0])
        & (ell <= ell_interval[1])
    )
    return _invert_supported_curve(
        ell,
        result.moments[index],
        group.uncertainty["block_bootstrap_standard_error"][index],
        result.counts[index],
        support,
        target,
        ell_interval=ell_interval,
    )


def _invert_supported_curve(
    ell: np.ndarray,
    curve: np.ndarray,
    error: np.ndarray,
    counts: np.ndarray,
    support: np.ndarray,
    target: float,
    *,
    ell_interval: tuple[float, float],
) -> tuple[float | None, float | None, str]:
    """Invert one retained curve with unique-crossing and contiguous-bin checks."""

    scale, crossing_error, quality = finite_domain_analysis._crossing_scale(
        ell,
        np.where(support, curve, np.nan),
        np.where(support, error, np.nan),
        np.where(support, counts, 0),
        float(target),
        min_count=batch_a_report.MINIMUM_CURVE_ACCEPTED_MEASUREMENTS,
        ell_interval=ell_interval,
    )
    return _positive_or_none(scale), _finite_or_none(crossing_error), quality


def _supported_inverse_scale_interval(
    group: Any,
    index: tuple[int, ...],
    target: float,
    *,
    ell_interval: tuple[float, float] = EQUAL_SF_INVERSION_ELL_INTERVAL,
) -> dict[str, Any]:
    """Invert central and retained block-bootstrap envelope curves fail-closed."""

    result = group.result
    ell = figures._centers(result.ell_bin_edges)
    support = (
        _curve_support_mask(group, index)
        & (ell >= ell_interval[0])
        & (ell <= ell_interval[1])
    )
    counts = result.counts[index]
    standard_error = group.uncertainty["block_bootstrap_standard_error"][index]
    central = _invert_supported_curve(
        ell,
        result.moments[index],
        standard_error,
        counts,
        support,
        target,
        ell_interval=ell_interval,
    )
    low = _invert_supported_curve(
        ell,
        group.uncertainty["block_bootstrap_interval_low"][index],
        standard_error,
        counts,
        support,
        target,
        ell_interval=ell_interval,
    )
    high = _invert_supported_curve(
        ell,
        group.uncertainty["block_bootstrap_interval_high"][index],
        standard_error,
        counts,
        support,
        target,
        ell_interval=ell_interval,
    )
    envelope_scales = [value for value in (low[0], high[0]) if value is not None]
    block_stable = (
        central[2] == "ok"
        and central[0] is not None
        and low[2] == "ok"
        and high[2] == "ok"
        and len(envelope_scales) == 2
    )
    return {
        "scale_cells": central[0],
        "crossing_error_cells": central[1],
        "central_quality": central[2],
        "block_bootstrap_low_curve_crossing_quality": low[2],
        "block_bootstrap_high_curve_crossing_quality": high[2],
        "block_bootstrap_curve_envelope_crossing_interval_low_cells": (
            min(envelope_scales) if block_stable else None
        ),
        "block_bootstrap_curve_envelope_crossing_interval_high_cells": (
            max(envelope_scales) if block_stable else None
        ),
        "block_stable_unique_crossing": block_stable,
    }


def _equal_sf_aspect_rows(
    cube_ids: Sequence[str],
    groups: Mapping[tuple[str, str], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = []
    quality_census = []
    for cube_id in cube_ids:
        group = groups[(cube_id, PRIMARY_SUPPORT_MODE)]
        result = group.result
        ell = figures._centers(result.ell_bin_edges)
        for q_name in Q_NAMES:
            indices = {
                direction: figures._moment_index(
                    result, q_name, direction, geometry_name="pair_local", p_value=P2
                )
                for direction in DIRECTIONS
            }
            support_masks = {
                direction: (
                    _curve_support_mask(group, index)
                    & (ell >= EQUAL_SF_INVERSION_ELL_INTERVAL[0])
                    & (ell <= EQUAL_SF_INVERSION_ELL_INTERVAL[1])
                )
                for direction, index in indices.items()
            }
            definition = _common_equal_sf_targets(
                {
                    direction: result.moments[index]
                    for direction, index in indices.items()
                },
                support_masks,
            )
            definitions.append(
                {
                    "cube_id": cube_id,
                    "q_name": q_name,
                    **definition,
                }
            )
            for target_index, target in enumerate(definition["targets"]):
                inverse = {
                    direction: _supported_inverse_scale_interval(
                        group, index, float(target)
                    )
                    for direction, index in indices.items()
                }
                scales = {
                    direction: inverse[direction]["scale_cells"]
                    for direction in DIRECTIONS
                }
                scale_intervals = {
                    direction: (
                        inverse[direction][
                            "block_bootstrap_curve_envelope_crossing_interval_low_cells"
                        ],
                        inverse[direction][
                            "block_bootstrap_curve_envelope_crossing_interval_high_cells"
                        ],
                    )
                    for direction in DIRECTIONS
                }
                all_ok = all(
                    inverse[direction]["block_stable_unique_crossing"]
                    for direction in DIRECTIONS
                )
                lambda_scale = scales["lambda"]
                lambda_interval = scale_intervals["lambda"]
                xi_interval = scale_intervals["xi"]
                parallel_interval = scale_intervals["parallel"]
                quality_census.append(
                    {
                        "cube_id": cube_id,
                        "q_name": q_name,
                        "target_index": target_index,
                        "equal_sf_target": float(target),
                        "target_definition": (
                            "interior log-spaced level within the common supported "
                            "parallel/xi/lambda curve-value overlap"
                        ),
                        "common_overlap_minimum": definition["common_overlap_minimum"],
                        "common_overlap_maximum": definition["common_overlap_maximum"],
                        "parallel_scale_cells": scales["parallel"],
                        "xi_scale_cells": scales["xi"],
                        "lambda_scale_cells": lambda_scale,
                        "parallel_scale_crossing_error_cells": inverse["parallel"][
                            "crossing_error_cells"
                        ],
                        "xi_scale_crossing_error_cells": inverse["xi"][
                            "crossing_error_cells"
                        ],
                        "lambda_scale_crossing_error_cells": inverse["lambda"][
                            "crossing_error_cells"
                        ],
                        "parallel_scale_block_bootstrap_curve_envelope_interval_low_cells": (
                            parallel_interval[0]
                        ),
                        "parallel_scale_block_bootstrap_curve_envelope_interval_high_cells": (
                            parallel_interval[1]
                        ),
                        "xi_scale_block_bootstrap_curve_envelope_interval_low_cells": (
                            xi_interval[0]
                        ),
                        "xi_scale_block_bootstrap_curve_envelope_interval_high_cells": (
                            xi_interval[1]
                        ),
                        "lambda_scale_block_bootstrap_curve_envelope_interval_low_cells": (
                            lambda_interval[0]
                        ),
                        "lambda_scale_block_bootstrap_curve_envelope_interval_high_cells": (
                            lambda_interval[1]
                        ),
                        "parallel_central_quality": inverse["parallel"]["central_quality"],
                        "xi_central_quality": inverse["xi"]["central_quality"],
                        "lambda_central_quality": inverse["lambda"]["central_quality"],
                        "parallel_block_bootstrap_low_curve_crossing_quality": inverse[
                            "parallel"
                        ]["block_bootstrap_low_curve_crossing_quality"],
                        "parallel_block_bootstrap_high_curve_crossing_quality": inverse[
                            "parallel"
                        ]["block_bootstrap_high_curve_crossing_quality"],
                        "xi_block_bootstrap_low_curve_crossing_quality": inverse["xi"][
                            "block_bootstrap_low_curve_crossing_quality"
                        ],
                        "xi_block_bootstrap_high_curve_crossing_quality": inverse["xi"][
                            "block_bootstrap_high_curve_crossing_quality"
                        ],
                        "lambda_block_bootstrap_low_curve_crossing_quality": inverse[
                            "lambda"
                        ]["block_bootstrap_low_curve_crossing_quality"],
                        "lambda_block_bootstrap_high_curve_crossing_quality": inverse[
                            "lambda"
                        ]["block_bootstrap_high_curve_crossing_quality"],
                        "aspect_table_eligible": all_ok,
                        "xi_over_lambda": (
                            scales["xi"] / lambda_scale
                            if all_ok and lambda_scale is not None and lambda_scale > 0.0
                            else None
                        ),
                        "xi_over_lambda_block_bootstrap_curve_envelope_interval_low": (
                            xi_interval[0] / lambda_interval[1] if all_ok else None
                        ),
                        "xi_over_lambda_block_bootstrap_curve_envelope_interval_high": (
                            xi_interval[1] / lambda_interval[0] if all_ok else None
                        ),
                        "parallel_over_lambda": (
                            scales["parallel"] / lambda_scale
                            if all_ok and lambda_scale is not None and lambda_scale > 0.0
                            else None
                        ),
                        "parallel_over_lambda_block_bootstrap_curve_envelope_interval_low": (
                            parallel_interval[0] / lambda_interval[1]
                            if all_ok
                            else None
                        ),
                        "parallel_over_lambda_block_bootstrap_curve_envelope_interval_high": (
                            parallel_interval[1] / lambda_interval[0]
                            if all_ok
                            else None
                        ),
                    }
                )
    rows = [row for row in quality_census if row["aspect_table_eligible"]]
    return definitions, quality_census, rows


def _order_sensitivity_rows(
    retained_rows: Sequence[Mapping[str, Any]],
    cube_ids: Sequence[str],
    catalog_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate Batch B support-policy sensitivity by cube, order, and channel."""

    rows = []
    for cube_id in cube_ids:
        catalog = _catalog_values(catalog_rows, cube_id)
        for p_value in BATCH_B_P_VALUES:
            for q_name in Q_NAMES:
                for direction in DIRECTIONS:
                    selected = [
                        row
                        for row in retained_rows
                        if row["cube_id"] == cube_id
                        and row["p_value"] == p_value
                        and row["q_name"] == q_name
                        and row["direction"] == direction
                    ]
                    rows.append(
                        {
                            "cube_id": cube_id,
                            "p_value": p_value,
                            "q_name": q_name,
                            "direction": direction,
                            **catalog,
                            **_factor_summary(
                                float(row["policy_factor"]) for row in selected
                            ),
                        }
                    )
    return rows


def _order_sensitivity_correlations(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    correlations = []
    for p_value in BATCH_B_P_VALUES:
        for q_name in Q_NAMES:
            for direction in DIRECTIONS:
                selected = [
                    row
                    for row in rows
                    if row["p_value"] == p_value
                    and row["q_name"] == q_name
                    and row["direction"] == direction
                    and row["p90"] is not None
                ]
                for covariate in MAGNETIC_COVARIATES:
                    correlations.append(
                        {
                            "p_value": p_value,
                            "q_name": q_name,
                            "direction": direction,
                            "magnetic_covariate": covariate,
                            "cube_count": len(selected),
                            "spearman_rho": batch_a_report._spearman(
                                [float(row[covariate]) for row in selected],
                                [float(row["p90"]) for row in selected],
                            ),
                            "interpretation": (
                                "exploratory rank correlation of per-cube p90 supported "
                                "all-valid-origin versus shell-local policy factor"
                            ),
                        }
                    )
    return correlations


def _sf_environment_rows(
    cube_ids: Sequence[str],
    groups: Mapping[tuple[str, str], Any],
    catalog_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Publish rooted amplitudes at bounded supported scales while retaining raw moments."""

    rows = []
    for cube_id in cube_ids:
        group = groups[(cube_id, PRIMARY_SUPPORT_MODE)]
        result = group.result
        ell = figures._centers(result.ell_bin_edges)
        catalog = _catalog_values(catalog_rows, cube_id)
        for p_value in BATCH_B_P_VALUES:
            for q_name in Q_NAMES:
                for direction in DIRECTIONS:
                    index = figures._moment_index(
                        result, q_name, direction, p_value=p_value
                    )
                    q_index, geometry_index = index[:2]
                    supported_bins = _curve_support_mask(group, index)
                    for requested_ell in SF_ENVIRONMENT_SCALE_TARGETS:
                        shell_index = _closest_shell_index(ell, requested_ell)
                        raw_moment = _positive_or_none(
                            result.moments[index][shell_index]
                        )
                        supported = bool(
                            supported_bins[shell_index] and raw_moment is not None
                        )
                        exclusions = {
                            str(name): int(
                                result.exclusions[
                                    q_index, geometry_index, exclusion_index, shell_index
                                ]
                            )
                            for exclusion_index, name in enumerate(result.exclusion_names)
                        }
                        rows.append(
                            {
                                "cube_id": cube_id,
                                "p_value": p_value,
                                "q_name": q_name,
                                "direction": direction,
                                "requested_ell_cells": requested_ell,
                                "actual_shell_center_cells": float(ell[shell_index]),
                                **catalog,
                                "supported_rooted_amplitude": supported,
                                "raw_moment_S_p": raw_moment,
                                "rooted_amplitude_A_p": (
                                    raw_moment ** (1.0 / p_value)
                                    if supported and raw_moment is not None
                                    else None
                                ),
                                "accepted_measurements": int(
                                    result.counts[index][shell_index]
                                ),
                                "directional_excluded_measurements_sum": sum(
                                    exclusions.values()
                                ),
                                "directional_excluded_measurements": exclusions,
                            }
                        )
    return rows


def _sf_environment_correlations(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Correlate supported rooted amplitudes with environment and count dependence."""

    dependence_variables = (
        *MAGNETIC_COVARIATES,
        "accepted_measurements",
        "directional_excluded_measurements_sum",
    )
    correlations = []
    for p_value in BATCH_B_P_VALUES:
        for q_name in Q_NAMES:
            for direction in DIRECTIONS:
                for requested_ell in SF_ENVIRONMENT_SCALE_TARGETS:
                    selected = [
                        row
                        for row in rows
                        if row["p_value"] == p_value
                        and row["q_name"] == q_name
                        and row["direction"] == direction
                        and row["requested_ell_cells"] == requested_ell
                        and row["rooted_amplitude_A_p"] is not None
                    ]
                    without_denominator_outliers = [
                        row
                        for row in selected
                        if float(row["dBB"]) <= DENOMINATOR_OUTLIER_DBB_MAXIMUM
                    ]
                    for variable in dependence_variables:
                        correlations.append(
                            {
                                "p_value": p_value,
                                "q_name": q_name,
                                "direction": direction,
                                "requested_ell_cells": requested_ell,
                                "actual_shell_center_cells": (
                                    selected[0]["actual_shell_center_cells"]
                                    if selected
                                    else None
                                ),
                                "dependence_variable": variable,
                                "full_census_cube_count": len(selected),
                                "full_census_spearman_rho": batch_a_report._spearman(
                                    [float(row[variable]) for row in selected],
                                    [
                                        float(row["rooted_amplitude_A_p"])
                                        for row in selected
                                    ],
                                ),
                                "denominator_outlier_policy": (
                                    f"exclude dBB > {DENOMINATOR_OUTLIER_DBB_MAXIMUM:g}"
                                ),
                                "without_denominator_outliers_cube_count": len(
                                    without_denominator_outliers
                                ),
                                "without_denominator_outliers_spearman_rho": (
                                    batch_a_report._spearman(
                                        [
                                            float(row[variable])
                                            for row in without_denominator_outliers
                                        ],
                                        [
                                            float(row["rooted_amplitude_A_p"])
                                            for row in without_denominator_outliers
                                        ],
                                    )
                                ),
                                "interpretation": (
                                    "exploratory rank correlation of supported rooted "
                                    "amplitude A_p=S_p^(1/p); not a fitted scaling claim"
                                ),
                            }
                        )
    return correlations


def _closest_shell_index(ell: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(ell, dtype=float) - float(target))))


def _stencil_comparison_rows(
    cube_ids: Sequence[str],
    batch_a_groups: Mapping[tuple[str, str], Any],
    extension_groups: Mapping[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    rows = []
    representative_ids = set(batch_a2_review.REPRESENTATIVE_CUBES)
    for cube_id in cube_ids:
        two = batch_a_groups[(cube_id, PRIMARY_SUPPORT_MODE)]
        three = extension_groups[(cube_id, PRIMARY_SUPPORT_MODE)]
        ell_two = figures._centers(two.result.ell_bin_edges)
        ell_three = figures._centers(three.result.ell_bin_edges)
        for q_name in Q_NAMES:
            for direction in DIRECTIONS:
                index_two = figures._moment_index(two.result, q_name, direction, p_value=P2)
                index_three = figures._moment_index(
                    three.result, q_name, direction, p_value=P2
                )
                supported_two = _curve_support_mask(two, index_two)
                supported_three = _curve_support_mask(three, index_three)
                for target in STENCIL_COMPARISON_TARGETS:
                    shell_two = _closest_shell_index(ell_two, target)
                    shell_three = _closest_shell_index(ell_three, target)
                    value_two = _positive_or_none(two.result.moments[index_two][shell_two])
                    value_three = _positive_or_none(
                        three.result.moments[index_three][shell_three]
                    )
                    supported = bool(
                        supported_two[shell_two]
                        and supported_three[shell_three]
                        and value_two is not None
                        and value_three is not None
                    )
                    ratio = value_three / value_two if supported else None
                    rows.append(
                        {
                            "cube_id": cube_id,
                            "scope": (
                                "representative_overlap"
                                if cube_id in representative_ids
                                else "all21_extension_only"
                            ),
                            "q_name": q_name,
                            "direction": direction,
                            "requested_ell_cells": target,
                            "two_point_actual_shell_center_cells": float(ell_two[shell_two]),
                            "three_point_actual_shell_center_cells": float(ell_three[shell_three]),
                            "two_point_curve_value": value_two,
                            "three_point_curve_value": value_three,
                            "three_point_over_two_point": ratio,
                            "labeled_stencil_factor": (
                                max(ratio, 1.0 / ratio)
                                if ratio is not None and ratio > 0.0
                                else None
                            ),
                            "supported_comparison": supported,
                            "interpretation": (
                                "comparison of distinctly labeled products; not an "
                                "interchangeability or correction claim"
                            ),
                        }
                    )
    return rows


def _release_runtime_storage_row(
    *,
    label: str,
    root: Path,
    summary: Mapping[str, Any],
    shard_rows: Sequence[Mapping[str, Any]],
    input_hashes: batch_a_report.InputHashes,
) -> dict[str, Any]:
    estimator = sum(float(row["elapsed_seconds_sum"]) for row in summary["groups"])
    reduction = jackknife = bootstrap = reduction_bytes = 0.0
    for row in summary["groups"]:
        marker_path = runner._reduction_paths(root, str(row["group_id"]))[3]
        marker = _load_json(marker_path)
        input_hashes.add(marker_path)
        reduction += float(marker["reduction_elapsed_seconds"])
        jackknife += float(marker["jackknife_elapsed_seconds"])
        bootstrap += float(marker["bootstrap_elapsed_seconds"])
        reduction_bytes += float(marker["staging_logical_bytes_before_marker"])
    shard_bytes = 0.0
    for row in shard_rows:
        marker_path = runner._shard_paths(root, str(row["shard_id"]))[2]
        marker = _load_json(marker_path)
        input_hashes.add(marker_path)
        shard_bytes += float(marker["staging_logical_bytes_before_marker"])
    settled = summary.get("settled_release_bytes_before_summary_marker")
    return {
        "release_label": label,
        "root": str(root.resolve()),
        "group_count": len(summary["groups"]),
        "shard_count": len(shard_rows),
        "estimator_elapsed_seconds_sum": estimator,
        "reduction_elapsed_seconds_sum": reduction,
        "jackknife_elapsed_seconds_sum": jackknife,
        "bootstrap_elapsed_seconds_sum": bootstrap,
        "published_shard_staging_logical_bytes": int(shard_bytes),
        "published_reduction_staging_logical_bytes": int(reduction_bytes),
        "settled_release_bytes_before_summary_marker": (
            int(settled["logical_bytes"])
            if isinstance(settled, Mapping) and "logical_bytes" in settled
            else None
        ),
    }


def _write_ledger_snapshot(
    output_dir: Path,
    ledger_summary: Mapping[str, Any],
    snapshot: bytes,
) -> Path:
    path = output_dir / LEDGER_SNAPSHOT_FILENAME
    path.write_bytes(snapshot)
    if file_sha256(path) != ledger_summary["source_sha256"]:
        raise RuntimeError("completion supplement ledger snapshot changed during assembly")
    return path


def conditioning_figure(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True, sharey=True)
    for axis, q_name in zip(axes, Q_NAMES):
        plotted = False
        for direction in DIRECTIONS:
            selected = [
                row
                for row in rows
                if row["q_name"] == q_name
                and row["direction"] == direction
                and row["conditioning_factor"] is not None
            ]
            by_ell: dict[float, list[float]] = {}
            for row in selected:
                by_ell.setdefault(float(row["ell_cells"]), []).append(
                    float(row["conditioning_factor"])
                )
            ell = np.asarray(sorted(by_ell))
            if ell.size:
                plotted = True
                values = [by_ell[value] for value in ell]
                axis.plot(
                    ell,
                    [np.median(value) for value in values],
                    color=figures.DIRECTION_COLORS[direction],
                    label=direction,
                )
                axis.fill_between(
                    ell,
                    [np.quantile(value, 0.10) for value in values],
                    [np.quantile(value, 0.90) for value in values],
                    color=figures.DIRECTION_COLORS[direction],
                    alpha=0.14,
                )
            representative = [
                row
                for row in selected
                if row["representative_cube"]
            ]
            axis.scatter(
                [row["ell_cells"] for row in representative],
                [row["conditioning_factor"] for row in representative],
                color=figures.DIRECTION_COLORS[direction],
                s=13,
                alpha=0.38,
            )
        axis.axhline(1.0, color="#777777", linestyle="--")
        axis.set_xscale("log")
        if plotted:
            axis.set_yscale("log")
        else:
            axis.text(0.5, 0.5, "no supported rows", ha="center", va="center")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_title(q_name)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("conditioning factor")
    axes[0].legend(fontsize=8)
    figure.suptitle("Batch A p=2 pair-local versus subvolume-mean magnetic conditioning")
    return _save(figure, output_dir, CONDITIONING_FIGURE_FILENAME)


def equal_sf_aspect_figure(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for column, q_name in enumerate(Q_NAMES):
        selected = [
            row
            for row in rows
            if row["q_name"] == q_name and row["aspect_table_eligible"]
        ]
        for direction in DIRECTIONS:
            axes[0, column].scatter(
                [row["equal_sf_target"] for row in selected],
                [row[f"{direction}_scale_cells"] for row in selected],
                s=12,
                alpha=0.30,
                color=figures.DIRECTION_COLORS[direction],
                label=direction,
            )
        for ratio_name, color in (
            ("xi_over_lambda", "#f58518"),
            ("parallel_over_lambda", "#4c78a8"),
        ):
            axes[1, column].scatter(
                [row["equal_sf_target"] for row in selected],
                [row[ratio_name] for row in selected],
                s=12,
                alpha=0.34,
                color=color,
                label=ratio_name.replace("_", " "),
            )
        axes[0, column].set_ylabel("inverse scale [cells]")
        axes[1, column].set_ylabel("equal-SF aspect ratio")
        axes[1, column].axhline(1.0, color="#777777", linestyle="--")
        axes[0, column].set_title(q_name)
        for axis in axes[:, column]:
            if selected:
                axis.set_xscale("log")
                axis.set_yscale("log")
            else:
                axis.text(0.5, 0.5, "aspect table withheld", ha="center", va="center")
            axis.set_xlabel(r"equal-$S_2$ target")
            axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    eligible = sum(bool(row["aspect_table_eligible"]) for row in rows)
    figure.suptitle(
        "Exploratory equal-SF inverse scales; block-stable unique crossings only; "
        f"aspect rows published={eligible}/{len(rows)}; no fitted exponent"
    )
    return _save(figure, output_dir, ASPECT_FIGURE_FILENAME)


def order_sensitivity_figure(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(
        2,
        len(MAGNETIC_COVARIATES),
        figsize=(3.7 * len(MAGNETIC_COVARIATES), 8.0),
        constrained_layout=True,
    )
    for row_index, q_name in enumerate(Q_NAMES):
        for column, covariate in enumerate(MAGNETIC_COVARIATES):
            axis = axes[row_index, column]
            plotted = False
            for p_value in BATCH_B_P_VALUES:
                selected = [
                    row
                    for row in rows
                    if row["q_name"] == q_name
                    and row["p_value"] == p_value
                    and row["p90"] is not None
                ]
                axis.scatter(
                    [row[covariate] for row in selected],
                    [row["p90"] for row in selected],
                    s=14,
                    alpha=0.33,
                    label=f"p={int(p_value)}",
                )
                plotted |= bool(selected)
            axis.set_xlabel(covariate)
            axis.set_ylabel("per-cube p90 policy factor")
            if plotted:
                axis.set_yscale("log")
            else:
                axis.text(0.5, 0.5, "no supported rows", ha="center", va="center")
            if plotted and covariate == "dBB":
                axis.set_xscale("log")
            axis.grid(alpha=0.22)
            axis.set_title(q_name)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        "All-21 Batch B order-sensitive policy factors versus Phase 1 magnetic complements"
    )
    return _save(figure, output_dir, ORDER_FIGURE_FILENAME)


def sf_environment_figure(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(
        2,
        len(SF_ENVIRONMENT_FIGURE_VARIABLES),
        figsize=(3.6 * len(SF_ENVIRONMENT_FIGURE_VARIABLES), 7.8),
        constrained_layout=True,
    )
    for row_index, q_name in enumerate(Q_NAMES):
        for column, covariate in enumerate(SF_ENVIRONMENT_FIGURE_VARIABLES):
            axis = axes[row_index, column]
            plotted = False
            for p_value in BATCH_B_P_VALUES:
                selected = [
                    row
                    for row in rows
                    if row["q_name"] == q_name
                    and row["p_value"] == p_value
                    and row["requested_ell_cells"] == SF_ENVIRONMENT_FIGURE_SCALE_TARGET
                    and row["rooted_amplitude_A_p"] is not None
                ]
                axis.scatter(
                    [row[covariate] for row in selected],
                    [row["rooted_amplitude_A_p"] for row in selected],
                    s=16,
                    alpha=0.42,
                    label=f"p={int(p_value)}",
                )
                plotted |= bool(selected)
            if covariate in {
                "dBB",
                "B_mean",
                "deltaB",
                "B_rms",
                "accepted_measurements",
                "directional_excluded_measurements_sum",
            }:
                axis.set_xscale("log")
            if plotted:
                axis.set_yscale("log")
            else:
                axis.text(0.5, 0.5, "no supported rows", ha="center", va="center")
            axis.set_xlabel(covariate)
            axis.set_ylabel(r"$A_p=S_p^{1/p}$")
            axis.set_title(q_name)
            axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        "Supported rooted SF amplitudes versus Phase 1 magnetic environment at nearest "
        f"{SF_ENVIRONMENT_FIGURE_SCALE_TARGET:g}-cell shell; directions pooled as points; "
        "raw moments retained in JSON"
    )
    return _save(figure, output_dir, ENVIRONMENT_FIGURE_FILENAME)


def labeled_stencil_figure(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True, sharey=True)
    for axis, q_name in zip(axes, Q_NAMES):
        plotted = False
        for direction in DIRECTIONS:
            selected = [
                row
                for row in rows
                if row["q_name"] == q_name
                and row["direction"] == direction
                and row["labeled_stencil_factor"] is not None
            ]
            axis.scatter(
                [row["requested_ell_cells"] for row in selected],
                [row["labeled_stencil_factor"] for row in selected],
                s=16,
                alpha=0.32,
                color=figures.DIRECTION_COLORS[direction],
                label=direction,
            )
            plotted |= bool(selected)
        axis.axhline(1.0, color="#777777", linestyle="--")
        axis.set_xscale("log")
        if plotted:
            axis.set_yscale("log")
        else:
            axis.text(0.5, 0.5, "no supported rows", ha="center", va="center")
        axis.set_xlabel("requested nearest-shell scale [cells]")
        axis.set_title(q_name)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("labeled 3-point versus 2-point factor")
    axes[0].legend(fontsize=8)
    figure.suptitle("All-21 p=2 labeled-stencil comparison; products remain distinct")
    return _save(figure, output_dir, STENCIL_FIGURE_FILENAME)


def runtime_storage_figure(
    rows: Sequence[Mapping[str, Any]],
    ledger_summary: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    labels = [str(row["release_label"]) for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), constrained_layout=True)
    axes[0].bar(x, [row["estimator_elapsed_seconds_sum"] for row in rows])
    axes[0].set_ylabel("estimator elapsed-seconds sum")
    bottom = np.zeros(len(rows))
    for name, label in (
        ("reduction_elapsed_seconds_sum", "strict reduction"),
        ("jackknife_elapsed_seconds_sum", "block jackknife"),
        ("bootstrap_elapsed_seconds_sum", "block bootstrap"),
    ):
        values = np.asarray([row[name] for row in rows], dtype=float)
        axes[1].bar(x, values, bottom=bottom, label=label)
        bottom += values
    width = 0.36
    axes[2].bar(
        x - width / 2,
        [row["published_shard_staging_logical_bytes"] / 1024**3 for row in rows],
        width=width,
        label="shards",
    )
    axes[2].bar(
        x + width / 2,
        [row["published_reduction_staging_logical_bytes"] / 1024**3 for row in rows],
        width=width,
        label="reductions",
    )
    axes[1].set_ylabel("post-processing wall time sum [s]")
    axes[2].set_ylabel("published staging logical GiB")
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.22)
    metrics = ledger_summary["metrics_node_hours"]
    figure.suptitle(
        "Verified runtime/storage supplement; hash-bound ledger: "
        f"consumed={metrics['consumed_allocated_runtime']:.3f}, "
        f"remaining={metrics['remaining_budget']:.3f}, "
        f"pending={metrics['pending_maximum_additional_exposure']:.3f} node-hours"
    )
    return _save(figure, output_dir, RUNTIME_FIGURE_FILENAME)


def _directional_fit_policy() -> dict[str, Any]:
    return {
        "directional_fitted_slopes_published": False,
        "directional_fitted_exponents_published": False,
        "directional_zeta_published": False,
        "status": "explicitly_withheld",
        "reason": (
            "Completion supplement publishes curve values, policy factors, and exploratory "
            "equal-SF inverse scales only. It does not promote local or directional "
            "diagnostics to fitted scaling claims."
        ),
    }


def _publish_atomic_directory(temporary_output: Path, output_dir: Path) -> None:
    batch_a_report._publish_atomic_directory(temporary_output, output_dir)


def _write_manifest(
    output_dir: Path,
    *,
    input_hashes: batch_a_report.InputHashes,
    input_roots: Mapping[str, str],
) -> Path:
    figures_out = sorted(path.name for path in output_dir.glob("*.png"))
    artifacts = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.name != FIGURE_MANIFEST_FILENAME
    )
    path = output_dir / FIGURE_MANIFEST_FILENAME
    _write_json(
        path,
        {
            "schema_version": 1,
            "status": "passed",
            "publication_policy": (
                "immutable_one_time_publish_temp_directory_rename_refuse_existing_output"
            ),
            "generator_sha256": file_sha256(Path(__file__).resolve()),
            "input_roots": dict(input_roots),
            "input_sha256": input_hashes.as_dict(),
            "generated_figures": figures_out,
            "figure_sha256": {
                name: file_sha256(output_dir / name) for name in figures_out
            },
            "generated_artifacts_before_manifest": artifacts,
            "artifact_sha256": {
                name: file_sha256(output_dir / name) for name in artifacts
            },
        },
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the immutable Phase 4 all-21 completion supplement."
    )
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--batch-a-root", type=Path, required=True)
    parser.add_argument("--all21-3point-extension-root", type=Path, required=True)
    parser.add_argument("--all21-batch-b-root", type=Path, required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
    input_hashes = batch_a_report.InputHashes()
    ledger_summary, ledger_snapshot = batch_a_report._bind_ledger_summary_snapshot(
        args.ledger_summary, input_hashes
    )
    verified_batch_a, _ = batch_a_report.verify_inputs(
        phase1_root=args.phase1_root,
        extraction_root=args.extraction_root,
        release_root=args.batch_a_root,
        ledger_summary=ledger_summary,
        input_hashes=input_hashes,
    )
    batch_a_groups = batch_a_report._load_groups(verified_batch_a, input_hashes)
    catalog_rows, catalog_metadata = batch_a_report._phase1_catalog_rows(
        verified_batch_a, input_hashes
    )
    cube_ids = tuple(verified_batch_a.cube_ids)

    extension = _verify_marker_bound_release(
        root=args.all21_3point_extension_root,
        extraction_root=args.extraction_root,
        summary_filename=ALL21_3POINT_SUMMARY_FILENAME,
        marker_filename=ALL21_3POINT_MARKER_FILENAME,
        cube_ids=cube_ids,
        stencil_width=3,
        q_names=Q_NAMES,
        p_values=(P2,),
        density_conventions=("not applicable", "not applicable"),
        input_hashes=input_hashes,
    )
    batch_b_preloaded = batch_b_review._load_groups(
        args.all21_batch_b_root, cube_ids, input_hashes
    )
    batch_b = _verify_marker_bound_release(
        root=args.all21_batch_b_root,
        extraction_root=args.extraction_root,
        summary_filename=ALL21_BATCH_B_SUMMARY_FILENAME,
        marker_filename=ALL21_BATCH_B_MARKER_FILENAME,
        cube_ids=cube_ids,
        stencil_width=2,
        q_names=Q_NAMES,
        p_values=BATCH_B_P_VALUES,
        density_conventions=("not applicable", "not applicable"),
        input_hashes=input_hashes,
        groups=batch_b_preloaded,
    )
    _verify_dependency_bindings(
        batch_a_summary=verified_batch_a.release_summary,
        extension_summary=extension.summary,
        batch_b_summary=batch_b.summary,
        batch_a_root=args.batch_a_root,
        extension_root=args.all21_3point_extension_root,
        cube_ids=cube_ids,
    )
    batch_a_to_batch_b_p2_reproduction = (
        _verify_batch_a_to_all21_batch_b_p2_reproduction(
            cube_ids=cube_ids,
            batch_a_groups=batch_a_groups,
            batch_b_groups=batch_b.groups,
        )
    )

    conditioning_rows = _conditioning_rows(cube_ids, batch_a_groups)
    target_definitions, aspect_quality_census, aspect_rows = _equal_sf_aspect_rows(
        cube_ids, batch_a_groups
    )
    retained_batch_b_rows, batch_b_census = batch_b_review._curve_rows(
        batch_b.groups, cube_ids
    )
    order_rows = _order_sensitivity_rows(retained_batch_b_rows, cube_ids, catalog_rows)
    order_correlations = _order_sensitivity_correlations(order_rows)
    sf_environment_rows = _sf_environment_rows(cube_ids, batch_b.groups, catalog_rows)
    sf_environment_correlations = _sf_environment_correlations(sf_environment_rows)
    stencil_rows = _stencil_comparison_rows(cube_ids, batch_a_groups, extension.groups)
    batch_a_shards = tuple(
        _load_json(verified_batch_a.release_root / "manifests" / "shards.json")["shards"]
    )
    runtime_rows = [
        _release_runtime_storage_row(
            label="Batch A: 2-point p=2",
            root=verified_batch_a.release_root,
            summary=verified_batch_a.release_summary,
            shard_rows=batch_a_shards,
            input_hashes=input_hashes,
        ),
        _release_runtime_storage_row(
            label="all21: labeled 3-point p=2",
            root=extension.root,
            summary=extension.summary,
            shard_rows=extension.shard_rows,
            input_hashes=input_hashes,
        ),
        _release_runtime_storage_row(
            label="all21 Batch B: 2-point p=1..6",
            root=batch_b.root,
            summary=batch_b.summary,
            shard_rows=batch_b.shard_rows,
            input_hashes=input_hashes,
        ),
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        ledger_path = _write_ledger_snapshot(
            temporary_output, ledger_summary, ledger_snapshot
        )
        conditioning_paths = _write_table_pair(
            temporary_output,
            json_filename=CONDITIONING_JSON_FILENAME,
            csv_filename=CONDITIONING_CSV_FILENAME,
            payload={
                "schema_version": 1,
                "status": "passed",
                "metric_definition": (
                    "factor=max(pair-local/subvolume-mean, subvolume-mean/pair-local) "
                    "for Batch A p=2 all-valid-origin products"
                ),
                "science_scale_minimum_cells": SCIENCE_SCALE_MINIMUM,
                "representative_cube_ids": tuple(batch_a2_review.REPRESENTATIVE_CUBES),
                "summary_representative_by_channel": _summary_by(
                    [
                        row
                        for row in conditioning_rows
                        if row["representative_cube"]
                    ],
                    ("q_name", "direction"),
                    "conditioning_factor",
                ),
                "summary_by_channel": _summary_by(
                    conditioning_rows, ("q_name", "direction"), "conditioning_factor"
                ),
                "rows": conditioning_rows,
            },
            csv_rows=conditioning_rows,
        )
        aspect_paths = _write_table_pair(
            temporary_output,
            json_filename=ASPECT_JSON_FILENAME,
            csv_filename=ASPECT_CSV_FILENAME,
            payload={
                "schema_version": 1,
                "status": (
                    "explicitly_exploratory_block_stable_aspect_rows_published"
                    if aspect_rows
                    else "aspect_table_withheld_quality_census_only"
                ),
                "target_definition": (
                    "For each cube and q, use five interior log-spaced equal-SF levels "
                    "inside the common positive parallel/xi/lambda p=2 pair-local curve-value "
                    "overlap after support gates within 32..128 cells."
                ),
                "support_gates": {
                    "ell_interval_cells": EQUAL_SF_INVERSION_ELL_INTERVAL,
                    "minimum_accepted_measurements": (
                        batch_a_report.MINIMUM_CURVE_ACCEPTED_MEASUREMENTS
                    ),
                    "minimum_contributing_blocks": (
                        batch_a_report.MINIMUM_CURVE_CONTRIBUTING_BLOCKS
                    ),
                    "minimum_effective_blocks": (
                        batch_a_report.MINIMUM_CURVE_EFFECTIVE_BLOCKS
                    ),
                    "minimum_valid_bootstrap_fraction": (
                        batch_a_report.MINIMUM_CURVE_VALID_BOOTSTRAP_FRACTION
                    ),
                    "finite_block_bootstrap_interval_required": True,
                    "crossing_policy": (
                        "central and retained block-bootstrap interval-envelope curves use "
                        "contiguous supported bins only; reject missing or multiple crossings"
                    ),
                },
                "fitted_exponent_claimed": False,
                "aspect_table_publication_gate": (
                    "Publish an aspect row only when central, lower-envelope, and "
                    "upper-envelope crossings are unique for parallel, xi, and lambda."
                ),
                "target_definitions": target_definitions,
                "quality_census_rows": aspect_quality_census,
                "aspect_ratio_rows": aspect_rows,
            },
            csv_rows=aspect_quality_census,
        )
        order_paths = _write_table_pair(
            temporary_output,
            json_filename=ORDER_JSON_FILENAME,
            csv_filename=ORDER_CSV_FILENAME,
            payload={
                "schema_version": 1,
                "status": "passed",
                "metric_definition": (
                    "Per-cube distribution of supported all-valid-origin versus shell-local "
                    "policy factors by Batch B order, q, and direction. This is a retained "
                    "policy-sensitivity diagnostic, not a correction."
                ),
                "magnetic_covariates": MAGNETIC_COVARIATES,
                "retention_by_order": batch_b_review._summarize_census(batch_b_census),
                "rows": order_rows,
                "exploratory_rank_correlations": order_correlations,
                "supported_policy_ratio_rows": retained_batch_b_rows,
            },
            csv_rows=order_rows,
        )
        sf_environment_paths = _write_table_pair(
            temporary_output,
            json_filename=ENVIRONMENT_JSON_FILENAME,
            csv_filename=ENVIRONMENT_CSV_FILENAME,
            payload={
                "schema_version": 1,
                "status": "passed",
                "classification": "exploratory_sf_environment_diagnostic",
                "amplitude_definition": "A_p = S_p^(1/p)",
                "raw_moments_retained": True,
                "support_policy": (
                    "all-valid-origin Batch B curve bins passing accepted-measurement, "
                    "contributing-block, effective-block, valid-bootstrap, and "
                    "finite-block-bootstrap-interval gates"
                ),
                "requested_bounded_scales_cells": SF_ENVIRONMENT_SCALE_TARGETS,
                "magnetic_covariates": MAGNETIC_COVARIATES,
                "figure_dependence_variables": SF_ENVIRONMENT_FIGURE_VARIABLES,
                "count_dependence_fields": (
                    "accepted_measurements",
                    "directional_excluded_measurements_sum",
                    "directional_excluded_measurements",
                ),
                "denominator_outlier_sensitivity": (
                    f"publish full census and dBB <= {DENOMINATOR_OUTLIER_DBB_MAXIMUM:g} "
                    "rank correlations"
                ),
                "rows": sf_environment_rows,
                "exploratory_rank_correlations": sf_environment_correlations,
            },
            csv_rows=sf_environment_rows,
        )
        stencil_paths = _write_table_pair(
            temporary_output,
            json_filename=STENCIL_JSON_FILENAME,
            csv_filename=STENCIL_CSV_FILENAME,
            payload={
                "schema_version": 1,
                "status": "passed",
                "comparison_definition": (
                    "Nearest-shell all-valid-origin p=2 pair-local comparison of distinctly "
                    "labeled 2-point Batch A and 3-point all-21 extension products."
                ),
                "stencil_labels_remain_distinct": True,
                "summary_representative_overlap": _summary_by(
                    [
                        row
                        for row in stencil_rows
                        if row["scope"] == "representative_overlap"
                    ],
                    ("q_name", "direction", "requested_ell_cells"),
                    "labeled_stencil_factor",
                ),
                "summary_all21": _summary_by(
                    stencil_rows,
                    ("q_name", "direction", "requested_ell_cells"),
                    "labeled_stencil_factor",
                ),
                "rows": stencil_rows,
            },
            csv_rows=stencil_rows,
        )
        runtime_paths = _write_table_pair(
            temporary_output,
            json_filename=RUNTIME_JSON_FILENAME,
            csv_filename=RUNTIME_CSV_FILENAME,
            payload={
                "schema_version": 1,
                "status": "passed",
                "measurement_note": (
                    "Retained estimator elapsed-second sums, per-group post-processing wall "
                    "times, and marker-recorded staging bytes remain distinct from scheduler "
                    "node-hours in the copied ledger snapshot."
                ),
                "compute_ledger_summary_snapshot": {
                    **ledger_summary,
                    "snapshot_relative_path": ledger_path.name,
                    "snapshot_sha256": file_sha256(ledger_path),
                },
                "rows": runtime_rows,
            },
            csv_rows=runtime_rows,
        )

        conditioning_figure(conditioning_rows, temporary_output)
        equal_sf_aspect_figure(aspect_quality_census, temporary_output)
        order_sensitivity_figure(order_rows, temporary_output)
        sf_environment_figure(sf_environment_rows, temporary_output)
        labeled_stencil_figure(stencil_rows, temporary_output)
        runtime_storage_figure(runtime_rows, ledger_summary, temporary_output)
        table_pairs = {
            "conditioning_comparison": conditioning_paths,
            "equal_sf_inverse_scale_aspect_ratio": aspect_paths,
            "batch_b_order_sensitivity": order_paths,
            "sf_environment_rooted_amplitudes": sf_environment_paths,
            "labeled_stencil_comparison": stencil_paths,
            "runtime_storage_ledger": runtime_paths,
        }
        input_roots = {
            "phase1_root": str(verified_batch_a.phase1_root),
            "extraction_root": str(verified_batch_a.extraction_root),
            "batch_a_root": str(verified_batch_a.release_root),
            "all21_3point_extension_root": str(extension.root),
            "all21_batch_b_root": str(batch_b.root),
        }
        summary = {
            "schema_version": 1,
            "status": "phase4_completion_supplement_generated",
            "publication_policy": (
                "immutable_one_time_publish_temp_directory_rename_refuse_existing_output"
            ),
            "decision_scope": (
                "post-acquisition Phase 4 completion evidence only; no later-phase GO claim"
            ),
            "input_roots": input_roots,
            "cube_ids": cube_ids,
            "strict_verification": {
                "batch_a": verified_batch_a.verification,
                "all21_3point_extension": extension.verification,
                "all21_batch_b": batch_b.verification,
                "staged_dependency_bindings": "passed",
                "batch_a_to_all21_batch_b_p2_reproduction": (
                    batch_a_to_batch_b_p2_reproduction
                ),
            },
            "phase1_catalog_metadata": catalog_metadata,
            "directional_fit_policy": _directional_fit_policy(),
            "equal_sf_inverse_scale_policy": {
                "status": (
                    "explicitly_exploratory_block_stable_aspect_rows_published"
                    if aspect_rows
                    else "aspect_table_withheld_quality_census_only"
                ),
                "fitted_exponent_claimed": False,
                "quality_census_row_count": len(aspect_quality_census),
                "aspect_ratio_row_count": len(aspect_rows),
                "target_definition": (
                    "common supported directional overlap with interior log-spaced targets"
                ),
            },
            "tables": {
                name: {
                    "json_relative_path": paths[0].name,
                    "json_sha256": file_sha256(paths[0]),
                    "csv_relative_path": paths[1].name,
                    "csv_sha256": file_sha256(paths[1]),
                }
                for name, paths in table_pairs.items()
            },
            "compute_ledger_summary_snapshot": {
                **ledger_summary,
                "snapshot_relative_path": ledger_path.name,
                "snapshot_sha256": file_sha256(ledger_path),
            },
            "known_limits": [
                "Directional fitted slopes and fitted exponents are explicitly withheld.",
                "Directional zeta values are explicitly withheld.",
                "Equal-SF inverse scales are exploratory diagnostics, not fitted scaling laws.",
                (
                    "Equal-SF aspect rows are withheld unless central and retained "
                    "block-bootstrap envelope curves all have unique supported crossings."
                ),
                "Labeled 2-point and 3-point products remain distinct statistics.",
                "Batch B policy factors are diagnostics, not corrections.",
                "This completion supplement does not claim or authorize a later-phase GO.",
            ],
            "input_sha256": input_hashes.as_dict(),
        }
        _write_json(temporary_output / SUMMARY_FILENAME, summary)
        _write_manifest(
            temporary_output,
            input_hashes=input_hashes,
            input_roots=input_roots,
        )
        _publish_atomic_directory(temporary_output, output_dir)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    print(f"Wrote immutable Phase 4 completion supplement: {output_dir}")


if __name__ == "__main__":
    main()
