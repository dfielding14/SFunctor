#!/usr/bin/env python3
"""Generate the hash-bound Phase 4 Batch A review package.

Every quantitative figure is derived from the strict Phase 4 extraction and
Batch A publication chain.  The only schematic figure is labeled explicitly.
Primitive-only environmental diagnostics are exploratory additions computed
from mmap-backed extracted arrays; they are not Phase 1 catalog validations.
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
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3a import generate_phase3a_status_figures as phase3_figures
from scripts.phase3a import run_phase3a_sampler as inherited
from scripts.phase4 import run_phase4_extraction as extraction
from sfunctor.analysis.phase3a import load_finite_domain_partial_npz


DEFAULT_OUTPUT_DIR = Path("figures/phase4_batch_a_status")
SUMMARY_FILENAME = "phase4_batch_a_report_summary.json"
ENVIRONMENT_JSON_FILENAME = "phase4_batch_a_environmental_table.json"
ENVIRONMENT_CSV_FILENAME = "phase4_batch_a_environmental_table.csv"
SCALE_DIAGNOSTICS_JSON_FILENAME = "phase4_batch_a_scale_resolved_diagnostics.json"
SCALE_DIAGNOSTICS_CSV_FILENAME = "phase4_batch_a_scale_resolved_diagnostics.csv"
LEDGER_SUMMARY_SNAPSHOT_FILENAME = "phase4_batch_a_compute_ledger_summary_snapshot.md"
PRIMARY_SUPPORT_MODE = "all_valid_origins"
OVERLAY_SUPPORT_MODE = "shell_local"
SUPPORT_MODES = (PRIMARY_SUPPORT_MODE, OVERLAY_SUPPORT_MODE)
APPROVED_OVERLAY_SUPPORT_FRACTION = 0.05
SLOPE_CANDIDATE_SUPPORT_FRACTION = 0.10
MINIMUM_CURVE_ACCEPTED_MEASUREMENTS = 2
MINIMUM_CURVE_CONTRIBUTING_BLOCKS = 2
MINIMUM_CURVE_EFFECTIVE_BLOCKS = 8.0
MINIMUM_CURVE_VALID_BOOTSTRAP_FRACTION = 0.90
BOOTSTRAP_N_RESAMPLES = 200
SCIENCE_SCALE_TARGETS = (32.0, 64.0, 128.0)
Q_NAMES = ("B", "u")
DIRECTION_NAMES = ("parallel", "xi", "lambda")
EXPECTED_RESULT_DIRECTION_NAMES = ("all", "parallel", "perpendicular", "xi", "lambda")
FROZEN_BATCH_A_SOURCE_COMMIT = "0c8a7ab0d9dd2c10b125af610d25c6c2fbd1fe5a"
FROZEN_PHASE4_PILOT_CUBE_IDS = (
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
EXTRACTION_PLAN_FILENAME = "phase4_extraction_plan.json"
EXTRACTION_PLAN_MARKER_FILENAME = "PHASE4_EXTRACTION_PLAN_COMPLETE.json"
PHASE3A_PLAN_MARKER_FILENAME = "PLAN_COMPLETE.json"
CAMPAIGN_FILENAME = "manifests/campaign.json"
SHARDS_FILENAME = "manifests/shards.json"
INHERITED_SUMMARY_FILENAME = "phase3a_summary.json"
INHERITED_SUMMARY_MARKER_FILENAME = "PHASE3A_RELEASE_COMPLETE.json"
PHASE4_SUMMARY_FILENAME = "phase4_batch_a_summary.json"
PHASE4_SUMMARY_MARKER_FILENAME = "PHASE4_BATCH_A_COMPLETE.json"
EXPECTED_SHARD_COUNT = 168
EXPECTED_REDUCTION_COUNT = 42
EXPECTED_EXTRACTION_FIELDS = (
    "dens",
    "velx",
    "vely",
    "velz",
    "eint",
    "bcc1",
    "bcc2",
    "bcc3",
)
ANALYSIS_FIELD_NAMES = {
    "rho": "dens",
    "v_x": "velx",
    "v_y": "vely",
    "v_z": "velz",
    "B_x": "bcc1",
    "B_y": "bcc2",
    "B_z": "bcc3",
}
EXPECTED_CAMPAIGN_CONFIGURATION = {
    "q_names": ["B", "u"],
    "p_values": [2.0],
    "support_modes": list(SUPPORT_MODES),
    "production_seed": 20260530,
    "bootstrap_seed": 20260531,
    "sample_count_per_displacement": 2048,
    "pair_batch_size": 1024,
    "block_shape_kji": [80, 80, 80],
    "block_assignment": "stencil_midpoint",
    "offsets_per_shard": 480,
    "stencils": {
        "2": {
            "label": "2-point",
            "ell_max": 320,
            "bin_count": 64,
            "directions_per_bin": 24,
        }
    },
}
SUPPORT_MODE_COLORS = {
    PRIMARY_SUPPORT_MODE: "#4c78a8",
    OVERLAY_SUPPORT_MODE: "#e45756",
}
EXAMPLE_COLORS = {
    "low_dBB": "#4c78a8",
    "median_dBB": "#54a24b",
    "high_dBB": "#f58518",
    "weak_mean_field": "#e45756",
}
PHASE1_CATALOG_FIELDS = (
    "subvolume_id",
    "L_sub",
    "cell_i0",
    "cell_i1",
    "cell_j0",
    "cell_j1",
    "cell_k0",
    "cell_k1",
    "dBB",
    "B_mean",
    "deltaB",
    "B_rms",
    "B2_mean",
    "B_mean_sq_over_B2_mean",
    "deltaB_sq_over_B2_mean",
    "deltaB_flags",
    "dBB_flags",
    "B_mean_fraction_flags",
    "deltaB_fraction_flags",
    "magnetic_energy_mean",
    "vA_mean_proxy",
    "vA_rms_like_proxy",
    "u_mass_weighted_mean_x",
    "u_mass_weighted_mean_y",
    "u_mass_weighted_mean_z",
    "dens_mean",
    "dens_sigma",
    "rho_sigma_over_mean",
    "dens_skewness",
    "dens_kurtosis",
    "dens_moment_flags",
    "mom1_mean",
    "mom1_sigma",
    "mom1_skewness",
    "mom1_kurtosis",
    "mom1_moment_flags",
    "mom2_mean",
    "mom2_sigma",
    "mom2_skewness",
    "mom2_kurtosis",
    "mom2_moment_flags",
    "mom3_mean",
    "mom3_sigma",
    "mom3_skewness",
    "mom3_kurtosis",
    "mom3_moment_flags",
    "ener_mean",
    "ener_sigma",
    "ener_skewness",
    "ener_kurtosis",
    "ener_moment_flags",
    "catalog_validity_flags",
)


@dataclass(frozen=True)
class VerifiedInputs:
    """Strictly verified retained roots and publication metadata."""

    phase1_root: Path
    extraction_root: Path
    release_root: Path
    extraction_plan: dict[str, Any]
    campaign: dict[str, Any]
    release_summary: dict[str, Any]
    verification: dict[str, Any]
    cube_ids: tuple[str, ...]
    implementation_sha256: str
    ledger_summary: dict[str, Any]


class InputHashes:
    """Collect hashes for direct report inputs and adapter-verified array bindings."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}

    def add(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            raise RuntimeError(f"required retained artifact is missing: {path}")
        self._hashes[str(path)] = file_sha256(path)

    def add_many(self, paths: Iterable[Path]) -> None:
        for path in paths:
            self.add(path)

    def bind_verified(self, path: Path, sha256: str) -> None:
        """Recompute, validate, and record one retained artifact checksum."""

        path = path.resolve()
        if not path.is_file():
            raise RuntimeError(f"required retained artifact is missing: {path}")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise RuntimeError(f"invalid retained SHA-256 binding for {path}")
        observed = file_sha256(path)
        if observed != sha256:
            raise RuntimeError(
                f"retained artifact checksum mismatch: {path}: "
                f"expected {sha256}, observed {observed}"
            )
        self._hashes[str(path)] = observed

    def as_dict(self) -> dict[str, str]:
        return dict(sorted(self._hashes.items()))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required retained JSON artifact is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"retained JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(inherited._json_builtin(payload), indent=2, sort_keys=True) + "\n"
    )


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        inherited._json_builtin(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contained_path(root: Path, path: Path, *, label: str) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} path escapes retained root {root}: {path}") from error
    return path


def _relative_path(root: Path, relative_path: Any, *, label: str) -> Path:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
    ):
        raise RuntimeError(f"{label} must use one non-empty relative path")
    return _contained_path(root, root / relative_path, label=label)


def _relative_file(root: Path, relative_path: Any, *, label: str) -> Path:
    path = _relative_path(root, relative_path, label=label)
    if not path.is_file():
        raise RuntimeError(f"required {label} is missing: {path}")
    return path


def _contained_cbin_source(data_root: Path, path: Path) -> Path:
    """Require one retained cbin source under a trusted cbin-prefixed data subtree."""

    path = _contained_path(data_root, path, label="cbin source")
    relative = path.relative_to(data_root.resolve())
    if not relative.parts or not relative.parts[0].startswith("cbin"):
        raise RuntimeError(f"cbin source is outside a trusted cbin subtree: {path}")
    return path


def _fixed_file(root: Path, relative_path: str, *, label: str) -> Path:
    return _relative_file(root, relative_path, label=label)


def _historical_implementation_sha256(
    source_version: Mapping[str, Any],
    *,
    label: str,
) -> str:
    hashes = source_version.get("implementation_source_hashes")
    sha256 = source_version.get("implementation_sha256")
    if (
        not isinstance(hashes, Mapping)
        or not hashes
        or source_version.get("commit") != FROZEN_BATCH_A_SOURCE_COMMIT
        or source_version.get("dirty") is not False
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or sha256 != sha256.lower()
        or any(
            not isinstance(relative_path, str)
            or not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or source_sha256 != source_sha256.lower()
            for relative_path, source_sha256 in hashes.items()
        )
        or _mapping_sha256(hashes) != sha256
    ):
        raise RuntimeError(f"{label} has an incoherent historical implementation identity")
    return sha256


def _require_matching_source_hashes(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    relative_paths: Sequence[str],
    *,
    label: str,
) -> None:
    """Require exact retained digests for a shared historical source inventory."""

    reference_hashes = reference.get("implementation_source_hashes", {})
    candidate_hashes = candidate.get("implementation_source_hashes", {})
    if (
        not isinstance(reference_hashes, Mapping)
        or not isinstance(candidate_hashes, Mapping)
        or any(
            reference_hashes.get(relative_path) != candidate_hashes.get(relative_path)
            for relative_path in relative_paths
        )
    ):
        raise RuntimeError(f"{label} has mismatched shared historical source hashes")


def _phase1_pilot_projection(phase1_root: Path) -> tuple[dict[str, Any], ...]:
    """Load the exact trusted Phase 1 pilot fields frozen into the extraction plan."""

    pilot_path = phase1_root / "analysis" / "pilot_sample.csv"
    with pilot_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    projection = []
    for row in rows:
        projection.append(
            {
                "cube_id": str(row["pilot_id"]),
                "role": str(row["role"]),
                "L_sub": int(row["L_sub"]),
                "bounds_ijk_half_open": [
                    int(row[name])
                    for name in (
                        "cell_i0",
                        "cell_i1",
                        "cell_j0",
                        "cell_j1",
                        "cell_k0",
                        "cell_k1",
                    )
                ],
                "shape_kji": [640, 640, 640],
                "required_rank_ids": json.loads(row["required_rank_ids_json"]),
            }
        )
    return tuple(projection)


def _bound_identity_file(
    root: Path,
    binding: Mapping[str, Any],
    *,
    relative_key: str,
    sha256_key: str,
    expected_relative_path: str,
    input_hashes: InputHashes,
    label: str,
) -> Path:
    if binding.get(relative_key) != expected_relative_path:
        raise RuntimeError(f"{label} has a non-canonical retained path")
    path = _relative_file(root, binding.get(relative_key), label=label)
    input_hashes.bind_verified(path, str(binding.get(sha256_key, "")))
    return path


def _require_passed_comparison_rows(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_count: int | None = None,
) -> None:
    rows = payload.get("rows")
    if (
        payload.get("status") != "passed"
        or payload.get("failure_count") != 0
        or not isinstance(rows, list)
        or payload.get("comparison_count") != len(rows)
        or (expected_count is not None and len(rows) != expected_count)
        or any(row.get("passed") is not True for row in rows)
    ):
        raise RuntimeError(f"{label} is incomplete, failed, or incoherent")


def _extraction_plan_identity(
    phase1_root: Path,
    extraction_root: Path,
    input_hashes: InputHashes,
) -> tuple[dict[str, Any], dict[str, str]]:
    plan_path = _fixed_file(
        extraction_root, EXTRACTION_PLAN_FILENAME, label="Phase 4 extraction plan"
    )
    marker_path = _fixed_file(
        extraction_root,
        EXTRACTION_PLAN_MARKER_FILENAME,
        label="Phase 4 extraction plan marker",
    )
    plan = _load_json(plan_path)
    marker = _load_json(marker_path)
    implementation_sha256 = _historical_implementation_sha256(
        plan.get("source_version", {}),
        label="Phase 4 extraction plan",
    )
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("plan_sha256") != file_sha256(plan_path)
        or marker.get("implementation_sha256") != implementation_sha256
        or plan.get("schema_version") != 1
        or plan.get("phase") != "phase4_bounded_21_cube_extraction"
        or plan.get("status") != "planned"
        or Path(str(plan.get("trusted_run", ""))).resolve() != phase1_root
        or Path(str(plan.get("output_root", ""))).resolve() != extraction_root
        or plan.get("pilot_cube_count") != len(FROZEN_PHASE4_PILOT_CUBE_IDS)
        or tuple(plan.get("pilot_cube_ids", ())) != FROZEN_PHASE4_PILOT_CUBE_IDS
    ):
        raise RuntimeError("invalid frozen Phase 4 extraction plan publication")
    selection_rows = list(plan.get("selections", ()))
    if (
        len(selection_rows) != len(FROZEN_PHASE4_PILOT_CUBE_IDS)
        or tuple(str(row.get("cube_id")) for row in selection_rows)
        != FROZEN_PHASE4_PILOT_CUBE_IDS
        or any(
            row.get("L_sub") != 640
            or row.get("shape_kji") != [640, 640, 640]
            or len(row.get("bounds_ijk_half_open", ())) != 6
            or len(row.get("required_rank_ids", ())) == 0
            for row in selection_rows
        )
    ):
        raise RuntimeError("Phase 4 extraction plan lost the exact frozen 21-cube selection")
    if tuple(selection_rows) != _phase1_pilot_projection(phase1_root):
        raise RuntimeError("Phase 4 extraction plan differs from the trusted Phase 1 pilot rows")
    input_hashes.add_many((plan_path, marker_path))
    return plan, {
        "plan_relative_path": EXTRACTION_PLAN_FILENAME,
        "plan_sha256": file_sha256(plan_path),
        "marker_relative_path": EXTRACTION_PLAN_MARKER_FILENAME,
        "marker_sha256": file_sha256(marker_path),
    }


def _cube_publication_identity(
    extraction_root: Path,
    cube_id: str,
    input_hashes: InputHashes,
) -> tuple[dict[str, str], dict[str, Any]]:
    cube_root = _contained_path(extraction_root, extraction_root / cube_id, label="cube")
    completion_path = _fixed_file(cube_root, "COMPLETE.json", label=f"{cube_id} completion marker")
    manifest_path = _fixed_file(cube_root, "manifest.json", label=f"{cube_id} manifest")
    completion = _load_json(completion_path)
    manifest = _load_json(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    if (
        completion.get("schema_version") != 1
        or completion.get("cube_id") != cube_id
        or completion.get("manifest_sha256") != manifest_sha256
    ):
        raise RuntimeError(f"invalid retained cube completion marker: {completion_path}")
    input_hashes.add_many((completion_path, manifest_path))
    return {
        "completion_relative_path": f"{cube_id}/COMPLETE.json",
        "completion_sha256": file_sha256(completion_path),
        "manifest_relative_path": f"{cube_id}/manifest.json",
        "manifest_sha256": manifest_sha256,
    }, manifest


def _verify_extraction_manifest(
    extraction_root: Path,
    cube_id: str,
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    input_hashes: InputHashes,
) -> None:
    cube_root = _contained_path(extraction_root, extraction_root / cube_id, label="cube")
    shape = [640, 640, 640]
    bounds = list(selection["bounds_ijk_half_open"])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("completion_state") != "complete"
        or manifest.get("cube_id") != cube_id
        or manifest.get("role") != selection.get("role")
        or manifest.get("Lsub") != 640
        or manifest.get("bounds_ijk_half_open") != bounds
        or manifest.get("shape_kji") != shape
        or manifest.get("cell_count") != 640**3
        or manifest.get("trusted_phase1_artifacts") != plan.get("trusted_phase1_artifacts")
        or Path(str(manifest.get("source_root", ""))).resolve()
        != Path(str(plan.get("data_root", ""))).resolve()
        or manifest.get("source_basename") != plan.get("source_basename")
    ):
        raise RuntimeError(f"retained extraction manifest identity mismatch: {cube_id}")
    preflight = manifest.get("preflight", {})
    coverage = preflight.get("coverage", {})
    if (
        preflight.get("status") != "passed"
        or preflight.get("cube_id") != cube_id
        or preflight.get("bounds_ijk_half_open") != bounds
        or preflight.get("shape_kji") != shape
        or preflight.get("cell_count") != 640**3
        or coverage
        != {
            "expected_cells": 640**3,
            "copied_cells": 640**3,
            "holes": 0,
            "overlaps": 0,
        }
    ):
        raise RuntimeError(f"retained extraction coverage metadata mismatch: {cube_id}")
    data_root = Path(str(plan["data_root"])).resolve()
    for source in preflight.get("source_blocks", ()):
        _contained_path(data_root, Path(str(source.get("path", ""))), label="primitive source")
    fields = manifest.get("output_fields", {})
    if set(fields) != set(EXPECTED_EXTRACTION_FIELDS):
        raise RuntimeError(f"retained extraction field inventory mismatch: {cube_id}")
    for field in EXPECTED_EXTRACTION_FIELDS:
        metadata = fields[field]
        relative_path = f"fields/{field}.npy"
        if metadata.get("relative_path") != relative_path:
            raise RuntimeError(f"non-canonical retained extraction array path: {cube_id}/{field}")
        path = _relative_file(cube_root, relative_path, label=f"{cube_id} {field} array")
        input_hashes.bind_verified(path, str(metadata.get("sha256", "")))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            metadata.get("shape_kji") != shape
            or metadata.get("dtype") != str(np.dtype(np.float32))
            or list(array.shape) != shape
            or str(array.dtype) != str(np.dtype(np.float32))
            or path.stat().st_size != metadata.get("size_bytes")
        ):
            raise RuntimeError(f"retained extraction array metadata mismatch: {path}")
    _require_passed_comparison_rows(
        manifest.get("cbin_validation", {}),
        label=f"{cube_id} extraction-versus-cbin validation",
        expected_count=79,
    )
    _require_passed_comparison_rows(
        manifest.get("catalog_validation", {}),
        label=f"{cube_id} extraction-versus-catalog validation",
        expected_count=47,
    )
    position = manifest.get("position_validation", {})
    exact = manifest.get("exact_validation", {})
    if (
        position.get("status") != "passed"
        or position.get("failure_count") != 0
        or len(position.get("rows", ())) != position.get("sample_count")
        or exact.get("status") != "passed"
        or exact.get("failure_count") != 0
        or len(exact.get("rows", ())) != exact.get("comparison_count")
        or exact.get("primary_raw_moments") != manifest.get("primary_raw_moments")
    ):
        raise RuntimeError(f"retained exact extraction validation is incoherent: {cube_id}")
    for source in manifest["cbin_validation"].get("cbin_source_files", ()):
        _contained_cbin_source(data_root, Path(str(source.get("path", ""))))
    _historical_implementation_sha256(
        manifest.get("code_version", {}),
        label=f"{cube_id} extraction manifest",
    )
    _require_matching_source_hashes(
        plan.get("source_version", {}),
        manifest.get("code_version", {}),
        extraction.CORE_EXTRACTOR_SOURCE_PATHS,
        label=f"{cube_id} extraction plan-to-manifest provenance",
    )


def _verify_extraction_sources(
    extraction_root: Path,
    campaign: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, str],
    input_hashes: InputHashes,
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    selections = {str(row["cube_id"]): row for row in plan["selections"]}
    sources = campaign.get("phase2_sources", {})
    if (
        not isinstance(sources, Mapping)
        or len(sources) != len(FROZEN_PHASE4_PILOT_CUBE_IDS)
        or set(sources) != set(FROZEN_PHASE4_PILOT_CUBE_IDS)
    ):
        raise RuntimeError("campaign extraction-source inventory differs from the frozen pilot")
    for cube_id in FROZEN_PHASE4_PILOT_CUBE_IDS:
        source = sources[cube_id]
        if (
            source.get("cube_id") != cube_id
            or Path(str(source.get("phase2_root", ""))).resolve() != extraction_root
            or source.get("phase4_extraction_plan") != plan_identity
        ):
            raise RuntimeError(f"invalid frozen extraction source identity: {cube_id}")
        publication_identity, manifest = _cube_publication_identity(
            extraction_root, cube_id, input_hashes
        )
        if (
            source.get("completion_relative_path")
            != publication_identity["completion_relative_path"]
            or source.get("completion_sha256") != publication_identity["completion_sha256"]
            or source.get("manifest_relative_path")
            != publication_identity["manifest_relative_path"]
            or source.get("manifest_sha256") != publication_identity["manifest_sha256"]
        ):
            raise RuntimeError(f"campaign cube-publication binding mismatch: {cube_id}")
        _verify_extraction_manifest(
            extraction_root,
            cube_id,
            manifest,
            plan=plan,
            selection=selections[cube_id],
            input_hashes=input_hashes,
        )
        analysis_hashes = source.get("analysis_field_sha256")
        expected_analysis_hashes = {
            str(manifest["output_fields"][field]["relative_path"]): str(
                manifest["output_fields"][field]["sha256"]
            )
            for field in ANALYSIS_FIELD_NAMES.values()
        }
        if analysis_hashes != expected_analysis_hashes:
            raise RuntimeError(f"campaign analysis-array binding mismatch: {cube_id}")
        materialization = source.get("phase4_materialization_record", {})
        materialization_path = _bound_identity_file(
            extraction_root,
            materialization,
            relative_key="materialization_record_relative_path",
            sha256_key="materialization_record_sha256",
            expected_relative_path=f"phase4_materialization_records/{cube_id}.json",
            input_hashes=input_hashes,
            label=f"{cube_id} Phase 4 materialization record",
        )
        materialization_payload = _load_json(materialization_path)
        if materialization_payload != {
            "schema_version": 1,
            "status": "phase4_fresh_extraction",
            "cube_id": cube_id,
            "phase4_extraction_plan": dict(plan_identity),
            "cube_publication": publication_identity,
        }:
            raise RuntimeError(f"invalid retained Phase 4 materialization record: {cube_id}")
        restart = source.get("phase4_restart_check", {})
        restart_path = _bound_identity_file(
            extraction_root,
            restart,
            relative_key="restart_check_relative_path",
            sha256_key="restart_check_sha256",
            expected_relative_path=f"restart_checks/{cube_id}.json",
            input_hashes=input_hashes,
            label=f"{cube_id} strict restart record",
        )
        restart_payload = _load_json(restart_path)
        if (
            restart_payload.get("status") != "passed"
            or restart_payload.get("cube_id") != cube_id
            or restart_payload.get("verify_hashes") is not True
            or restart_payload.get("phase4_extraction_plan") != plan_identity
            or restart_payload.get("phase4_materialization_record") != materialization
            or restart_payload.get("cube_publication") != publication_identity
        ):
            raise RuntimeError(f"invalid retained strict restart verification: {cube_id}")
        manifests[cube_id] = manifest
    return manifests


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    return phase3_figures._save(figure, output_dir, filename)


def _support_fraction(result: Any) -> np.ndarray:
    return np.divide(
        result.eligible_pairs,
        result.cube_candidate_pairs,
        out=np.zeros_like(result.eligible_pairs, dtype=float),
        where=result.cube_candidate_pairs > 0,
    )


def _curve_uncertainty_support_mask(
    group: phase3_figures.ReleaseGroup,
    index: tuple[int, ...],
) -> np.ndarray:
    """Return curve bins with enough accepted spatial support for science-facing use."""

    result = group.result
    uncertainty = group.uncertainty
    minimum_valid_resamples = int(
        math.ceil(MINIMUM_CURVE_VALID_BOOTSTRAP_FRACTION * BOOTSTRAP_N_RESAMPLES)
    )
    return (
        np.isfinite(result.moments[index])
        & (result.counts[index] >= MINIMUM_CURVE_ACCEPTED_MEASUREMENTS)
        & (
            uncertainty["accepted_contributing_blocks"][index]
            >= MINIMUM_CURVE_CONTRIBUTING_BLOCKS
        )
        & (
            uncertainty["accepted_effective_blocks"][index]
            >= MINIMUM_CURVE_EFFECTIVE_BLOCKS
        )
        & (uncertainty["valid_bootstrap_resamples"][index] >= minimum_valid_resamples)
        & np.isfinite(uncertainty["block_bootstrap_interval_low"][index])
        & np.isfinite(uncertainty["block_bootstrap_interval_high"][index])
    )


def _complete_centered_window_mask(mask: np.ndarray, *, window: int = 5) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or window < 3 or window % 2 == 0 or window > mask.size:
        raise ValueError("centered-window mask requires one fitting odd-width shell axis")
    output = np.zeros_like(mask)
    radius = window // 2
    for center in range(radius, mask.size - radius):
        output[center] = bool(np.all(mask[center - radius : center + radius + 1]))
    return output


def _group_id(cube_id: str, support_mode: str) -> str:
    return phase3_figures._group_id(cube_id, 2, support_mode)


def _summary_group_row(result: Any, group_id: str) -> dict[str, Any]:
    return inherited._json_builtin(
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


def _verify_summary_inventory(
    release_root: Path,
    summary: Mapping[str, Any],
    cube_ids: Sequence[str],
    *,
    require_phase4_wrapper: bool,
) -> None:
    expected_group_ids = sorted(
        _group_id(cube_id, support_mode)
        for cube_id in cube_ids
        for support_mode in SUPPORT_MODES
    )
    rows = list(summary.get("groups", ()))
    if (
        summary.get("schema_version") != 1
        or summary.get("operational_status") != "release_aggregation_complete"
        or summary.get("scientific_acceptance") != "pending_report_level_gate"
        or sorted(str(row.get("group_id")) for row in rows) != expected_group_ids
        or (
            require_phase4_wrapper
            and (
                summary.get("phase") != "phase4_batch_a_bounded_21_cube_2point"
                or tuple(summary.get("pilot_cube_ids", ())) != tuple(cube_ids)
            )
        )
    ):
        raise RuntimeError("Phase 4 Batch A release summary has an invalid group inventory")
    by_group = {str(row["group_id"]): row for row in rows}
    for group_id in expected_group_ids:
        result = load_finite_domain_partial_npz(
            _reduction_paths(release_root, group_id)[1]
        )
        if by_group[group_id] != _summary_group_row(result, group_id):
            raise RuntimeError(f"Phase 4 Batch A release summary row is stale: {group_id}")


def _displacement_paths(release_root: Path, stencil_width: int) -> tuple[Path, Path]:
    root = release_root / "manifests" / "displacements"
    return (
        _contained_path(
            release_root,
            root / f"stencil_{stencil_width}point.json",
            label="displacement JSON manifest",
        ),
        _contained_path(
            release_root,
            root / f"stencil_{stencil_width}point.npz",
            label="displacement NPZ manifest",
        ),
    )


def _shard_paths(release_root: Path, shard_id: str) -> tuple[Path, Path, Path]:
    root = _contained_path(release_root, release_root / "shards" / shard_id, label="shard")
    return root, root / "partial.npz", root / "COMPLETE.json"


def _reduction_paths(release_root: Path, group_id: str) -> tuple[Path, Path, Path, Path]:
    root = _contained_path(
        release_root, release_root / "reductions" / group_id, label="reduction"
    )
    return root, root / "result.npz", root / "uncertainty.npz", root / "COMPLETE.json"


def _load_frozen_displacement_manifest(
    release_root: Path,
    campaign: Mapping[str, Any],
    stencil_width: int,
    input_hashes: InputHashes,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    row = campaign.get("displacement_manifests", {}).get(str(stencil_width), {})
    json_path, npz_path = _displacement_paths(release_root, stencil_width)
    expected_json_relative = str(json_path.relative_to(release_root))
    expected_npz_relative = str(npz_path.relative_to(release_root))
    if (
        row.get("json_relative_path") != expected_json_relative
        or row.get("npz_relative_path") != expected_npz_relative
    ):
        raise RuntimeError(f"non-canonical displacement manifest path: stencil {stencil_width}")
    input_hashes.bind_verified(json_path, str(row.get("json_sha256", "")))
    input_hashes.bind_verified(npz_path, str(row.get("npz_sha256", "")))
    metadata = _load_json(json_path)
    if (
        metadata.get("manifest_sha256")
        != _mapping_sha256(
            {key: value for key, value in metadata.items() if key != "manifest_sha256"}
        )
        or metadata.get("npz_sha256") != file_sha256(npz_path)
        or row.get("manifest_sha256") != metadata.get("manifest_sha256")
    ):
        raise RuntimeError(f"stale frozen displacement manifest: stencil {stencil_width}")
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"displacements_ijk", "ell_bin_edges", "offset_ids"}:
            raise RuntimeError(f"invalid displacement NPZ inventory: {npz_path}")
        displacements = payload["displacements_ijk"].copy()
        ell_bin_edges = payload["ell_bin_edges"].copy()
        offset_ids = payload["offset_ids"].copy()
    if (
        displacements.ndim != 2
        or displacements.shape[1:] != (3,)
        or not np.issubdtype(displacements.dtype, np.integer)
        or ell_bin_edges.ndim != 1
        or ell_bin_edges.size < 33
        or np.any(~np.isfinite(ell_bin_edges))
        or np.any(np.diff(ell_bin_edges) <= 0.0)
        or not np.array_equal(offset_ids, np.arange(len(displacements), dtype=np.int64))
        or metadata.get("offsets_sha256")
        != hashlib.sha256(displacements.tobytes()).hexdigest()
        or metadata.get("realized_offset_count") != len(displacements)
        or row.get("offset_count") != len(displacements)
    ):
        raise RuntimeError(f"invalid frozen displacement census: stencil {stencil_width}")
    return metadata, displacements, ell_bin_edges


def _planned_shards(
    displacements: Mapping[int, np.ndarray],
    configuration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    offsets_per_shard = int(configuration["offsets_per_shard"])
    for cube_id in FROZEN_PHASE4_PILOT_CUBE_IDS:
        for stencil_width in sorted(displacements):
            for support_mode in SUPPORT_MODES:
                group_id = _group_id(cube_id, support_mode)
                for shard_index, start in enumerate(
                    range(0, len(displacements[stencil_width]), offsets_per_shard)
                ):
                    rows.append(
                        {
                            "group_id": group_id,
                            "shard_id": f"{group_id}/shard_{shard_index:04d}",
                            "cube_id": cube_id,
                            "stencil_width": stencil_width,
                            "support_mode": support_mode,
                            "shard_index": shard_index,
                            "offset_start": start,
                            "offset_stop": min(
                                len(displacements[stencil_width]), start + offsets_per_shard
                            ),
                        }
                    )
    return rows


def _sampling_schedule_sha256(
    row: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> str:
    return _mapping_sha256(
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
        }
    )


def _verify_result_metadata(
    result: Any,
    *,
    expected_offsets: np.ndarray,
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    configuration: Mapping[str, Any],
    label: str,
) -> None:
    order = np.lexsort((expected_offsets[:, 2], expected_offsets[:, 1], expected_offsets[:, 0]))
    expected_offsets = expected_offsets[order]
    if (
        not np.array_equal(result.displacements_ijk, expected_offsets)
        or result.support_displacements_sha256 != metadata.get("offsets_sha256")
        or result.support_displacement_count != metadata.get("realized_offset_count")
        or result.stencil_width != row["stencil_width"]
        or result.pair_mode != row["support_mode"]
        or result.sample_count != configuration["sample_count_per_displacement"]
        or result.pair_batch_size != configuration["pair_batch_size"]
        or result.seed != configuration["production_seed"]
        or list(result.block_shape_kji) != configuration["block_shape_kji"]
        or result.block_assignment != configuration["block_assignment"]
        or result.cube_shape_kji != (640, 640, 640)
        or tuple(result.q_names) != Q_NAMES
        or tuple(result.direction_names) != EXPECTED_RESULT_DIRECTION_NAMES
        or tuple(result.p_values) != (2.0,)
    ):
        raise RuntimeError(f"frozen sampler result metadata mismatch: {label}")


def _verify_shard(
    release_root: Path,
    campaign: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    displacement_metadata: Mapping[int, Mapping[str, Any]],
    displacements: Mapping[int, np.ndarray],
    input_hashes: InputHashes,
) -> str:
    root, partial_path, marker_path = _shard_paths(release_root, str(row["shard_id"]))
    partial_path = _relative_file(root, "partial.npz", label=f"{row['shard_id']} partial")
    marker_path = _relative_file(root, "COMPLETE.json", label=f"{row['shard_id']} marker")
    marker = _load_json(marker_path)
    configuration = campaign["configuration"]
    stencil_width = int(row["stencil_width"])
    metadata = displacement_metadata[stencil_width]
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("shard") != dict(row)
        or marker.get("implementation_sha256")
        != campaign["source_version"]["implementation_sha256"]
        or marker.get("displacement_manifest_sha256") != metadata["manifest_sha256"]
        or marker.get("support_displacements_sha256") != metadata["offsets_sha256"]
        or marker.get("phase2_source") != campaign["phase2_sources"][row["cube_id"]]
        or marker.get("sampling_schedule_sha256")
        != _sampling_schedule_sha256(row, configuration)
        or any(
            not isinstance(marker.get(name), int) or marker[name] < 0
            for name in (
                "staging_logical_bytes_before_marker",
                "staging_allocated_bytes_before_marker",
            )
        )
    ):
        raise RuntimeError(f"invalid frozen shard marker: {row['shard_id']}")
    input_hashes.bind_verified(partial_path, str(marker.get("partial_sha256", "")))
    input_hashes.add(marker_path)
    result = load_finite_domain_partial_npz(partial_path)
    _verify_result_metadata(
        result,
        expected_offsets=displacements[stencil_width][
            int(row["offset_start"]) : int(row["offset_stop"])
        ],
        metadata=metadata,
        row=row,
        configuration=configuration,
        label=str(row["shard_id"]),
    )
    return file_sha256(marker_path)


def _verify_uncertainty_payload(path: Path, result: Any, configuration: Mapping[str, Any]) -> None:
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
            raise RuntimeError(f"invalid frozen uncertainty inventory: {path}")
        metadata_array = payload["metadata_json"]
        if metadata_array.shape != () or metadata_array.dtype.kind not in "SU":
            raise RuntimeError(f"invalid frozen uncertainty metadata scalar: {path}")
        metadata = json.loads(str(metadata_array.item()))
        if (
            metadata.get("schema_version") != 1
            or metadata.get("bootstrap_method") != "spatial_block_bootstrap"
            or metadata.get("bootstrap_seed") != configuration["bootstrap_seed"]
            or metadata.get("bootstrap_n_resamples") != BOOTSTRAP_N_RESAMPLES
            or metadata.get("bootstrap_confidence_level") != 0.95
            or metadata.get("jackknife_resampling_population")
            != "delete_contributing_blocks_only"
            or metadata.get("resampling_population")
            != "fixed_geometric_layout_including_empty_blocks"
            or metadata.get("block_shape_kji") != configuration["block_shape_kji"]
            or metadata.get("block_assignment") != configuration["block_assignment"]
            or metadata.get("local_slope_window_bins") != 5
            or metadata.get("minimum_local_slope_effective_blocks") != 8.0
            or metadata.get("minimum_local_slope_valid_bootstrap_fraction") != 0.9
        ):
            raise RuntimeError(f"invalid frozen uncertainty metadata: {path}")
        if not np.array_equal(payload["moments"], result.moments, equal_nan=True):
            raise RuntimeError(f"uncertainty moments do not match frozen result: {path}")
        for name in required - {"metadata_json", "sampled_blocks_per_shell", "eligible_blocks_per_shell"}:
            if payload[name].shape != result.sums.shape:
                raise RuntimeError(f"frozen uncertainty array shape mismatch for {name}: {path}")
        for name in ("sampled_blocks_per_shell", "eligible_blocks_per_shell"):
            if payload[name].shape != result.sampled_pairs.shape:
                raise RuntimeError(f"frozen uncertainty shell-array shape mismatch for {name}: {path}")
        if payload["local_log_slope_support_mask"].dtype.kind != "b":
            raise RuntimeError(f"frozen uncertainty slope-support mask is not boolean: {path}")


def _group_rows(shard_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in shard_rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    return groups


def _verify_reduction(
    release_root: Path,
    campaign: Mapping[str, Any],
    group_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    displacement_metadata: Mapping[int, Mapping[str, Any]],
    displacements: Mapping[int, np.ndarray],
    shard_marker_sha256: Mapping[str, str],
    input_hashes: InputHashes,
) -> None:
    root, result_path, uncertainty_path, marker_path = _reduction_paths(release_root, group_id)
    result_path = _relative_file(root, "result.npz", label=f"{group_id} reduction result")
    uncertainty_path = _relative_file(
        root, "uncertainty.npz", label=f"{group_id} uncertainty result"
    )
    manifest_path = _relative_file(
        root, "reduction_manifest.json", label=f"{group_id} reduction manifest"
    )
    marker_path = _relative_file(root, "COMPLETE.json", label=f"{group_id} reduction marker")
    marker = _load_json(marker_path)
    manifest = _load_json(manifest_path)
    ordered_shard_ids = tuple(str(row["shard_id"]) for row in rows)
    expected_marker_hashes = {
        shard_id: shard_marker_sha256[shard_id] for shard_id in ordered_shard_ids
    }
    timing_names = (
        "reduction_elapsed_seconds",
        "jackknife_elapsed_seconds",
        "bootstrap_elapsed_seconds",
        "staging_logical_bytes_before_marker",
        "staging_allocated_bytes_before_marker",
    )
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("group_id") != group_id
        or manifest.get("group_id") != group_id
        or tuple(manifest.get("ordered_shard_ids", ())) != ordered_shard_ids
        or manifest.get("ordered_shard_marker_sha256") != expected_marker_hashes
        or manifest.get("implementation_sha256")
        != campaign["source_version"]["implementation_sha256"]
        or any(
            not isinstance(marker.get(name), (int, float))
            or not np.isfinite(marker[name])
            or marker[name] < 0
            for name in timing_names
        )
    ):
        raise RuntimeError(f"invalid frozen reduction marker: {group_id}")
    input_hashes.bind_verified(result_path, str(marker.get("result_sha256", "")))
    input_hashes.bind_verified(
        uncertainty_path, str(marker.get("uncertainty_sha256", ""))
    )
    input_hashes.bind_verified(
        manifest_path, str(marker.get("reduction_manifest_sha256", ""))
    )
    input_hashes.add(marker_path)
    result = load_finite_domain_partial_npz(result_path)
    stencil_width = int(rows[0]["stencil_width"])
    _verify_result_metadata(
        result,
        expected_offsets=displacements[stencil_width],
        metadata=displacement_metadata[stencil_width],
        row=rows[0],
        configuration=campaign["configuration"],
        label=group_id,
    )
    _verify_uncertainty_payload(uncertainty_path, result, campaign["configuration"])


def _verify_summary_publication(
    release_root: Path,
    *,
    summary_filename: str,
    marker_filename: str,
    campaign: Mapping[str, Any],
    verification: Mapping[str, Any],
    input_hashes: InputHashes,
    require_phase4_wrapper: bool,
) -> dict[str, Any]:
    summary_path = _fixed_file(release_root, summary_filename, label=summary_filename)
    marker_path = _fixed_file(release_root, marker_filename, label=marker_filename)
    summary = _load_json(summary_path)
    marker = _load_json(marker_path)
    implementation_sha256 = campaign["source_version"]["implementation_sha256"]
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "release_aggregation_complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or marker.get("implementation_sha256") != implementation_sha256
        or summary.get("source_version") != campaign.get("source_version")
        or summary.get("verification") != verification
    ):
        raise RuntimeError(f"invalid frozen summary publication: {summary_filename}")
    _historical_implementation_sha256(
        summary.get("source_version", {}),
        label=summary_filename,
    )
    _verify_summary_inventory(
        release_root,
        summary,
        FROZEN_PHASE4_PILOT_CUBE_IDS,
        require_phase4_wrapper=require_phase4_wrapper,
    )
    input_hashes.add_many((summary_path, marker_path))
    return summary


def _verify_frozen_release(
    extraction_root: Path,
    release_root: Path,
    input_hashes: InputHashes,
    *,
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    campaign_path = _fixed_file(release_root, CAMPAIGN_FILENAME, label="campaign manifest")
    shards_path = _fixed_file(release_root, SHARDS_FILENAME, label="shard manifest")
    plan_marker_path = _fixed_file(
        release_root, PHASE3A_PLAN_MARKER_FILENAME, label="sampler campaign marker"
    )
    campaign = _load_json(campaign_path)
    shard_payload = _load_json(shards_path)
    plan_marker = _load_json(plan_marker_path)
    implementation_sha256 = _historical_implementation_sha256(
        campaign.get("source_version", {}),
        label="Phase 4 Batch A campaign",
    )
    if (
        plan_marker.get("schema_version") != 1
        or plan_marker.get("status") != "passed"
        or plan_marker.get("campaign_sha256") != file_sha256(campaign_path)
        or plan_marker.get("shards_sha256") != file_sha256(shards_path)
        or plan_marker.get("implementation_sha256") != implementation_sha256
        or campaign.get("schema_version") != 1
        or campaign.get("status") != "planned"
        or Path(str(campaign.get("phase2_root", ""))).resolve() != extraction_root
        or campaign.get("configuration") != EXPECTED_CAMPAIGN_CONFIGURATION
        or campaign.get("configuration_sha256")
        != _mapping_sha256(campaign.get("configuration", {}))
    ):
        raise RuntimeError("invalid frozen Phase 4 Batch A campaign publication")
    _require_matching_source_hashes(
        plan.get("source_version", {}),
        campaign.get("source_version", {}),
        (
            "job_scripts/phase4/run_phase4_extract_andes.sh",
            "scripts/phase1/cbin_tools.py",
            "scripts/phase4/run_phase4_extraction.py",
        ),
        label="Phase 4 extraction-plan-to-sampler-campaign provenance",
    )
    input_hashes.add_many((campaign_path, shards_path, plan_marker_path))
    manifests = _verify_extraction_sources(
        extraction_root,
        campaign,
        plan=plan,
        plan_identity=plan_identity,
        input_hashes=input_hashes,
    )
    displacement_metadata: dict[int, Mapping[str, Any]] = {}
    displacements: dict[int, np.ndarray] = {}
    for stencil_width in (2,):
        metadata, offsets, _ = _load_frozen_displacement_manifest(
            release_root, campaign, stencil_width, input_hashes
        )
        displacement_metadata[stencil_width] = metadata
        displacements[stencil_width] = offsets
    shard_rows = shard_payload.get("shards")
    expected_shards = _planned_shards(displacements, campaign["configuration"])
    if (
        not isinstance(shard_rows, list)
        or shard_rows != expected_shards
        or campaign.get("shard_count") != len(expected_shards)
        or len(expected_shards) != EXPECTED_SHARD_COUNT
    ):
        raise RuntimeError("Phase 4 Batch A shard inventory is not canonical exact-once coverage")
    shard_marker_sha256 = {
        str(row["shard_id"]): _verify_shard(
            release_root,
            campaign,
            row,
            displacement_metadata=displacement_metadata,
            displacements=displacements,
            input_hashes=input_hashes,
        )
        for row in shard_rows
    }
    groups = _group_rows(shard_rows)
    if len(groups) != EXPECTED_REDUCTION_COUNT:
        raise RuntimeError("Phase 4 Batch A reduction inventory is not the expected 42 groups")
    for group_id, rows in sorted(groups.items()):
        _verify_reduction(
            release_root,
            campaign,
            group_id,
            rows,
            displacement_metadata=displacement_metadata,
            displacements=displacements,
            shard_marker_sha256=shard_marker_sha256,
            input_hashes=input_hashes,
        )
    verification = {
        "status": "passed",
        "verification_mode": "strict_frozen_artifact_chain_without_live_source_replay",
        "historical_implementation_sha256": implementation_sha256,
        "verified_extraction_plans": 1,
        "verified_cubes": len(FROZEN_PHASE4_PILOT_CUBE_IDS),
        "verified_materialization_records": len(FROZEN_PHASE4_PILOT_CUBE_IDS),
        "verified_restart_records": len(FROZEN_PHASE4_PILOT_CUBE_IDS),
        "verified_shards": len(shard_rows),
        "verified_reductions": len(groups),
    }
    inherited_verification = {
        "status": "passed",
        "verified_shards": len(shard_rows),
        "verified_reductions": len(groups),
    }
    _verify_summary_publication(
        release_root,
        summary_filename=INHERITED_SUMMARY_FILENAME,
        marker_filename=INHERITED_SUMMARY_MARKER_FILENAME,
        campaign=campaign,
        verification=inherited_verification,
        input_hashes=input_hashes,
        require_phase4_wrapper=False,
    )
    summary = _verify_summary_publication(
        release_root,
        summary_filename=PHASE4_SUMMARY_FILENAME,
        marker_filename=PHASE4_SUMMARY_MARKER_FILENAME,
        campaign=campaign,
        verification=inherited_verification,
        input_hashes=input_hashes,
        require_phase4_wrapper=True,
    )
    return campaign, summary, verification, manifests


def verify_inputs(
    *,
    phase1_root: Path,
    extraction_root: Path,
    release_root: Path,
    ledger_summary: Mapping[str, Any],
    input_hashes: InputHashes,
) -> tuple[VerifiedInputs, dict[str, dict[str, Any]]]:
    """Verify the exact frozen publication chain without live source replay."""

    phase1_root = phase1_root.resolve()
    extraction_root = extraction_root.resolve()
    release_root = release_root.resolve()
    trusted_artifacts = extraction.verify_trusted_run(phase1_root)
    input_hashes.add_many(
        (
            phase1_root / "validation" / "VALIDATION_COMPLETE.json",
            phase1_root / "BUILD_COMPLETE.json",
            phase1_root / "verification" / "VERIFY_COMPLETE.json",
            phase1_root / "analysis" / "analysis_manifest.json",
            phase1_root / "analysis" / "ANALYSIS_COMPLETE.json",
            phase1_root / "analysis" / "pilot_sample.csv",
            phase1_root / "analysis" / "pilot_sample_metadata.json",
            phase1_root / "cache" / "rank_map.npy",
        )
    )
    plan, plan_identity = _extraction_plan_identity(
        phase1_root, extraction_root, input_hashes
    )
    if plan.get("trusted_phase1_artifacts") != trusted_artifacts:
        raise RuntimeError("frozen extraction plan lost its trusted Phase 1 artifact graph")
    campaign, summary, verification, manifests = _verify_frozen_release(
        extraction_root,
        release_root,
        input_hashes,
        plan=plan,
        plan_identity=plan_identity,
    )
    verified = VerifiedInputs(
        phase1_root=phase1_root,
        extraction_root=extraction_root,
        release_root=release_root,
        extraction_plan=plan,
        campaign=campaign,
        release_summary=summary,
        verification=verification,
        cube_ids=FROZEN_PHASE4_PILOT_CUBE_IDS,
        implementation_sha256=str(campaign["source_version"]["implementation_sha256"]),
        ledger_summary=dict(ledger_summary),
    )
    return verified, manifests


def _load_groups(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
) -> dict[tuple[str, str], phase3_figures.ReleaseGroup]:
    groups: dict[tuple[str, str], phase3_figures.ReleaseGroup] = {}
    for cube_id in verified.cube_ids:
        for support_mode in SUPPORT_MODES:
            group_id = _group_id(cube_id, support_mode)
            root, result_path, uncertainty_path, marker_path = _reduction_paths(
                verified.release_root, group_id
            )
            result = load_finite_domain_partial_npz(result_path)
            with np.load(uncertainty_path, allow_pickle=False) as payload:
                uncertainty = {name: payload[name].copy() for name in payload.files}
            if result.stencil_width != 2 or result.pair_mode != support_mode:
                raise RuntimeError(f"Phase 4 reduction metadata mismatch: {group_id}")
            input_hashes.add_many(
                (marker_path, root / "reduction_manifest.json", result_path, uncertainty_path)
            )
            groups[(cube_id, support_mode)] = phase3_figures.ReleaseGroup(
                group_id=group_id,
                result=result,
                uncertainty=uncertainty,
            )
    _assert_matching_policy_axis_metadata(verified, groups)
    return groups


def _same_metadata(left: Any, right: Any) -> bool:
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
            _same_metadata(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _same_metadata(lhs, rhs) for lhs, rhs in zip(left, right)
        )
    return bool(left == right)


def _assert_matching_policy_axis_metadata(
    verified: VerifiedInputs,
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
) -> None:
    """Require one shared plotted axis contract across cubes and support policies."""

    axis_names = (
        "q_names",
        "density_conventions",
        "geometry_names",
        "measurement_names",
        "direction_names",
        "exclusion_names",
        "ell_bin_edges",
        "p_values",
        "cell_sizes",
        "angle_limits",
        "cube_shape_kji",
        "stencil_width",
        "support_displacements_sha256",
        "support_displacement_count",
    )
    reference = groups[(verified.cube_ids[0], PRIMARY_SUPPORT_MODE)].result
    for cube_id in verified.cube_ids:
        primary = groups[(cube_id, PRIMARY_SUPPORT_MODE)].result
        overlay = groups[(cube_id, OVERLAY_SUPPORT_MODE)].result
        for name in axis_names:
            if not _same_metadata(getattr(primary, name), getattr(overlay, name)):
                raise RuntimeError(
                    f"support-policy axis metadata mismatch for {cube_id}: {name}"
                )
            if not _same_metadata(getattr(reference, name), getattr(primary, name)):
                raise RuntimeError(f"Phase 4 Batch A plotted axis varies across cubes: {name}")


def _native_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _phase1_catalog_rows(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load exact L=640 catalog rows so validity flags remain visible."""

    pilot_path = verified.phase1_root / "analysis" / "pilot_sample.csv"
    metadata_path = verified.phase1_root / "analysis" / "pilot_sample_metadata.json"
    catalog_path = verified.phase1_root / "catalogs" / "catalog_L640.npz"
    catalog_manifest_path = verified.phase1_root / "catalogs" / "catalog_L640_manifest.json"
    catalog_marker_path = verified.phase1_root / "catalogs" / "catalog_L640.complete"
    cache_manifest_path = verified.phase1_root / "cache" / "raw_mhd_u_bcc_80_manifest.json"
    validation_path = verified.phase1_root / "validation" / "VALIDATION_COMPLETE.json"
    input_hashes.add_many(
        (
            pilot_path,
            metadata_path,
            catalog_manifest_path,
            catalog_marker_path,
            cache_manifest_path,
        )
    )
    catalog_sha256 = file_sha256(catalog_path)
    input_hashes.bind_verified(catalog_path, catalog_sha256)
    catalog_manifest = _load_json(catalog_manifest_path)
    cache_manifest = _load_json(cache_manifest_path)
    if (
        catalog_manifest.get("L_sub") != 640
        or catalog_manifest.get("catalog_sha256") != catalog_sha256
        or catalog_manifest.get("primary_cache_sha256") != cache_manifest.get("cache_sha256")
        or catalog_manifest.get("validation_token_sha256") != file_sha256(validation_path)
    ):
        raise RuntimeError("Phase 1 L=640 catalog manifest is missing or stale")
    with pilot_path.open(newline="") as stream:
        pilot_rows = list(csv.DictReader(stream))
    pilot_by_id = {str(row["pilot_id"]): row for row in pilot_rows}
    if tuple(pilot_by_id) != verified.cube_ids:
        raise RuntimeError("Phase 1 pilot CSV membership differs from the frozen Phase 4 plan")
    with np.load(catalog_path, allow_pickle=False) as payload:
        missing = sorted(set(PHASE1_CATALOG_FIELDS) - set(payload.files))
        if missing:
            raise RuntimeError(f"Phase 1 L=640 catalog lacks required report fields: {missing}")
        catalog_metadata = json.loads(str(payload["__metadata_json__"].item()))
        subvolume_ids = np.asarray(payload["subvolume_id"], dtype=int)
        index_by_id = {int(value): index for index, value in enumerate(subvolume_ids)}
        rows: dict[str, dict[str, Any]] = {}
        for cube_id in verified.cube_ids:
            subvolume_id = int(cube_id.rsplit("sub", 1)[1])
            if subvolume_id not in index_by_id:
                raise RuntimeError(f"Phase 1 L=640 catalog lacks {cube_id}")
            index = index_by_id[subvolume_id]
            catalog = {
                name: _native_scalar(payload[name][index]) for name in PHASE1_CATALOG_FIELDS
            }
            pilot = pilot_by_id[cube_id]
            rows[cube_id] = {
                "cube_id": cube_id,
                "role": str(pilot["role"]),
                "bounds_ijk_half_open": [
                    int(pilot[name])
                    for name in (
                        "cell_i0",
                        "cell_i1",
                        "cell_j0",
                        "cell_j1",
                        "cell_k0",
                        "cell_k1",
                    )
                ],
                "required_rank_ids": json.loads(pilot["required_rank_ids_json"]),
                "estimated_primitive_eight_field_bytes": int(
                    pilot["primitive_eight_field_bytes"]
                ),
                "estimated_full_primitive_rank_file_read_gib": float(
                    pilot["full_primitive_rank_file_read_gib"]
                ),
                "catalog": catalog,
            }
    return rows, {
        "pilot_sample_metadata": _load_json(metadata_path),
        "catalog_metadata": catalog_metadata,
    }


def _pressure_convention(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return the explicit primitive pressure convention or a reportable gap."""

    gaps: list[str] = []
    build = _load_json(verified.phase1_root / "BUILD_COMPLETE.json")
    validation_path = verified.phase1_root / "validation" / "VALIDATION_COMPLETE.json"
    validation = _load_json(validation_path)
    gamma = build.get("gamma")
    snapshot_fields = set(
        verified.extraction_plan.get("snapshot_identity", {})
        .get("full_snapshot_identity", {})
        .get("var_names", ())
    )
    try:
        gamma_value = float(gamma)
        validated_gamma = float(validation["gamma"])
    except (KeyError, TypeError, ValueError):
        gaps.append("pressure omitted: trusted Phase 1 metadata does not retain one numeric gamma")
        return None, gaps
    if (
        not math.isfinite(gamma_value)
        or gamma_value <= 1.0
        or gamma_value != validated_gamma
        or "eint" not in snapshot_fields
    ):
        gaps.append(
            "pressure omitted: the trusted primitive eint field and validated ideal-MHD gamma "
            "assumption are not jointly available"
        )
        return None, gaps
    input_file = Path(str(build.get("input_file", "")))
    input_sha256 = validation.get("input_file_sha256")
    if not input_file.is_file() or file_sha256(input_file) != input_sha256:
        gaps.append(
            "pressure omitted: the Athena input file no longer matches the validated gamma binding"
        )
        return None, gaps
    input_hashes.bind_verified(input_file, str(input_sha256))
    return {
        "status": "supported_exploratory_primitive_only",
        "gamma_assumption": gamma_value,
        "athena_input_file": str(input_file.resolve()),
        "internal_energy_field": "eint",
        "total_mhd_energy_convention": (
            "E_total = eint + 0.5 * rho * |u|^2 + 0.5 * |B|^2"
        ),
        "pressure_convention": "p = (gamma - 1) * eint",
        "sound_speed_convention": "c_s,V,rms = sqrt(<gamma * p / rho>_V)",
    }, gaps


def _primitive_diagnostics(
    cube_root: Path,
    manifest: Mapping[str, Any],
    *,
    chunk_k: int,
    pressure_convention: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compute exploratory primitive diagnostics without stacking full cubes."""

    if chunk_k < 1:
        raise ValueError("--chunk-k must be positive")
    arrays = {
        name: np.load(
            cube_root / str(manifest["output_fields"][name]["relative_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3")
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"primitive mmap arrays have inconsistent shapes under {cube_root}")
    shape = next(iter(shapes))
    cell_count = int(np.prod(shape))
    gamma = (
        float(pressure_convention["gamma_assumption"])
        if pressure_convention is not None
        else None
    )
    sums = {
        "rho": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "speed2": 0.0,
        "kinetic": 0.0,
        "magnetic": 0.0,
        "internal": 0.0,
        "pressure": 0.0,
        "cs2": 0.0,
        "va2": 0.0,
    }
    density_min = float("inf")
    density_max = float("-inf")
    pressure_min = float("inf")
    pressure_max = float("-inf")
    invalid_finite_cells = 0
    nonpositive_density_cells = 0
    negative_internal_energy_cells = 0
    pressure_valid_cells = 0
    va_valid_cells = 0
    for start in range(0, shape[0], chunk_k):
        stop = min(shape[0], start + chunk_k)
        block = slice(start, stop)
        rho = np.asarray(arrays["dens"][block], dtype=np.float64)
        vx = np.asarray(arrays["velx"][block], dtype=np.float64)
        vy = np.asarray(arrays["vely"][block], dtype=np.float64)
        vz = np.asarray(arrays["velz"][block], dtype=np.float64)
        eint = np.asarray(arrays["eint"][block], dtype=np.float64)
        bx = np.asarray(arrays["bcc1"][block], dtype=np.float64)
        by = np.asarray(arrays["bcc2"][block], dtype=np.float64)
        bz = np.asarray(arrays["bcc3"][block], dtype=np.float64)
        finite = (
            np.isfinite(rho)
            & np.isfinite(vx)
            & np.isfinite(vy)
            & np.isfinite(vz)
            & np.isfinite(eint)
            & np.isfinite(bx)
            & np.isfinite(by)
            & np.isfinite(bz)
        )
        invalid_finite_cells += int(finite.size - np.count_nonzero(finite))
        if not np.all(finite):
            raise RuntimeError(f"non-finite extracted primitive value under {cube_root}")
        positive_rho = rho > 0.0
        nonnegative_eint = eint >= 0.0
        nonpositive_density_cells += int(rho.size - np.count_nonzero(positive_rho))
        negative_internal_energy_cells += int(eint.size - np.count_nonzero(nonnegative_eint))
        speed2 = vx * vx + vy * vy + vz * vz
        b2 = bx * bx + by * by + bz * bz
        kinetic = 0.5 * rho * speed2
        magnetic = 0.5 * b2
        sums["rho"] += float(np.sum(rho, dtype=np.float64))
        sums["vx"] += float(np.sum(vx, dtype=np.float64))
        sums["vy"] += float(np.sum(vy, dtype=np.float64))
        sums["vz"] += float(np.sum(vz, dtype=np.float64))
        sums["speed2"] += float(np.sum(speed2, dtype=np.float64))
        sums["kinetic"] += float(np.sum(kinetic, dtype=np.float64))
        sums["magnetic"] += float(np.sum(magnetic, dtype=np.float64))
        sums["internal"] += float(np.sum(eint, dtype=np.float64))
        density_min = min(density_min, float(np.min(rho)))
        density_max = max(density_max, float(np.max(rho)))
        va_mask = positive_rho
        va_valid_cells += int(np.count_nonzero(va_mask))
        sums["va2"] += float(np.sum(b2[va_mask] / rho[va_mask], dtype=np.float64))
        if gamma is not None:
            pressure = (gamma - 1.0) * eint
            pressure_mask = positive_rho & nonnegative_eint
            pressure_valid_cells += int(np.count_nonzero(pressure_mask))
            sums["pressure"] += float(np.sum(pressure[pressure_mask], dtype=np.float64))
            sums["cs2"] += float(
                np.sum(gamma * pressure[pressure_mask] / rho[pressure_mask], dtype=np.float64)
            )
            if np.any(pressure_mask):
                pressure_min = min(pressure_min, float(np.min(pressure[pressure_mask])))
                pressure_max = max(pressure_max, float(np.max(pressure[pressure_mask])))
    velocity_mean = np.asarray([sums["vx"], sums["vy"], sums["vz"]]) / cell_count
    delta_u_sq = max(sums["speed2"] / cell_count - float(np.sum(velocity_mean**2)), 0.0)
    delta_u = math.sqrt(delta_u_sq)
    va_rms = math.sqrt(sums["va2"] / va_valid_cells) if va_valid_cells else None
    kinetic_mean = sums["kinetic"] / cell_count
    magnetic_mean = sums["magnetic"] / cell_count
    result: dict[str, Any] = {
        "classification": "newly_computed_full_resolution_primitive_exploratory",
        "units": "AthenaK simulation code units",
        "array_traversal": f"numpy mmap KJI traversal in chunks of at most {chunk_k} k-planes",
        "cell_count": cell_count,
        "nonfinite_primitive_cell_count": invalid_finite_cells,
        "nonpositive_density_cell_count": nonpositive_density_cells,
        "negative_internal_energy_density_cell_count": negative_internal_energy_cells,
        "velocity_weighting": "volume",
        "velocity_volume_mean_x": float(velocity_mean[0]),
        "velocity_volume_mean_y": float(velocity_mean[1]),
        "velocity_volume_mean_z": float(velocity_mean[2]),
        "delta_u_volume_rms": delta_u,
        "delta_u_convention": "sqrt(<|u - <u>_V|^2>_V)",
        "density_volume_mean": sums["rho"] / cell_count,
        "density_minimum": density_min,
        "density_maximum": density_max,
        "internal_energy_density_volume_mean": sums["internal"] / cell_count,
        "kinetic_energy_density_volume_mean": kinetic_mean,
        "kinetic_energy_convention": "<0.5 * rho * |u|^2>_V",
        "magnetic_energy_density_volume_mean": magnetic_mean,
        "magnetic_energy_convention": "<0.5 * |B|^2>_V",
        "magnetic_to_kinetic_energy_ratio": (
            magnetic_mean / kinetic_mean if kinetic_mean > 0.0 else None
        ),
        "alfven_speed_volume_rms": va_rms,
        "alfven_speed_convention": "sqrt(<|B|^2 / rho>_V) over rho > 0 cells",
        "alfven_valid_cell_count": va_valid_cells,
        "alfven_mach_delta_u_over_alfven_speed_volume_rms": (
            delta_u / va_rms if va_rms is not None and va_rms > 0.0 else None
        ),
    }
    if gamma is None:
        result["pressure_diagnostic_status"] = "omitted_without_supported_convention"
    else:
        sound_speed_rms = (
            math.sqrt(sums["cs2"] / pressure_valid_cells) if pressure_valid_cells else None
        )
        result.update(
            {
                "pressure_diagnostic_status": "derived_from_extracted_eint",
                "gamma_assumption": gamma,
                "pressure_convention": "p = (gamma - 1) * eint",
                "pressure_valid_cell_count": pressure_valid_cells,
                "pressure_conditional_mean_over_valid_cells": (
                    sums["pressure"] / pressure_valid_cells if pressure_valid_cells else None
                ),
                "pressure_conditional_mean_note": (
                    "Mean pressure over rho > 0 and eint >= 0 cells; this is a "
                    "valid-cell conditional mean, not an unconditional volume mean."
                ),
                "pressure_minimum": pressure_min if pressure_valid_cells else None,
                "pressure_maximum": pressure_max if pressure_valid_cells else None,
                "sound_speed_volume_rms": sound_speed_rms,
                "sound_speed_convention": "sqrt(<gamma * p / rho>_V) over rho > 0, eint >= 0 cells",
                "sonic_mach_delta_u_over_sound_speed_volume_rms": (
                    delta_u / sound_speed_rms
                    if sound_speed_rms is not None and sound_speed_rms > 0.0
                    else None
                ),
            }
        )
    return result


def _exclusion_summary(result: Any) -> dict[str, int]:
    return {
        str(name): int(result.exclusions[:, :, index, :].sum())
        for index, name in enumerate(result.exclusion_names)
    }


def _sampler_resource_summary(
    verified: VerifiedInputs,
    shard_rows: Sequence[Mapping[str, Any]],
    input_hashes: InputHashes,
) -> tuple[dict[str, Any], dict[str, int | None], list[str]]:
    """Bind retained fresh work records and summarize their task-local parent RSS."""

    expected = {str(row["shard_id"]): str(row["cube_id"]) for row in shard_rows}
    covered: dict[str, Path] = {}
    rss_by_cube: dict[str, list[int]] = {cube_id: [] for cube_id in verified.cube_ids}
    records = []
    for path in sorted((verified.release_root / "work_resource_records").glob("*.json")):
        payload = _load_json(path)
        if (
            payload.get("schema_version") != inherited.SCHEMA_VERSION
            or payload.get("action") != "work"
            or payload.get("source_version", {}).get("implementation_sha256")
            != verified.implementation_sha256
        ):
            continue
        fresh_rows = [
            row
            for row in payload.get("rows", ())
            if row.get("reused") is False and str(row.get("shard_id")) in expected
        ]
        if not fresh_rows:
            continue
        input_hashes.add(path)
        parent_rss = int(payload["parent_process_peak_rss_kib"])
        touched_cubes = sorted({str(row["cube_id"]) for row in fresh_rows})
        for cube_id in touched_cubes:
            rss_by_cube[cube_id].append(parent_rss)
        for row in fresh_rows:
            shard_id = str(row["shard_id"])
            if shard_id in covered:
                raise RuntimeError(
                    f"multiple fresh sampler resource records claim shard {shard_id}: "
                    f"{covered[shard_id]}, {path}"
                )
            covered[shard_id] = path
        records.append(
            {
                "relative_path": str(path.relative_to(verified.release_root)),
                "sha256": file_sha256(path),
                "slurm_job_id": str(payload.get("slurm_job_id")),
                "slurm_procid": int(payload.get("slurm_procid", 0)),
                "slurm_ntasks": int(payload.get("slurm_ntasks", 1)),
                "workers": int(payload.get("workers", 0)),
                "action_wall_seconds": float(payload["action_wall_seconds"]),
                "parent_process_peak_rss_kib": parent_rss,
                "fresh_published_shard_count": len(fresh_rows),
            }
        )
    missing = sorted(set(expected) - set(covered))
    gaps = []
    if missing:
        gaps.append(
            "sampler resource-record coverage is incomplete: "
            f"{len(missing)} of {len(expected)} published shards lack retained fresh work records; "
            "sampler parent-process RSS summaries are partial"
        )
    per_cube = {
        cube_id: max(values) if values else None for cube_id, values in rss_by_cube.items()
    }
    return {
        "status": "complete" if not missing else "partial",
        "expected_published_shard_count": len(expected),
        "fresh_resource_record_covered_shard_count": len(covered),
        "missing_fresh_resource_record_shard_ids": missing,
        "sampler_parent_process_peak_rss_kib_max": (
            max((row["parent_process_peak_rss_kib"] for row in records), default=None)
        ),
        "resource_records": records,
        "rss_scope": (
            "task-local sampler parent process peak RSS from retained work resource records; "
            "not a multi-process tree peak"
        ),
    }, per_cube, gaps


def _operational_rows(
    verified: VerifiedInputs,
    manifests: Mapping[str, Mapping[str, Any]],
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
    input_hashes: InputHashes,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    shard_rows = _load_json(verified.release_root / "manifests" / "shards.json")["shards"]
    sampler_resources, sampler_rss_by_cube, resource_gaps = _sampler_resource_summary(
        verified, shard_rows, input_hashes
    )
    shard_bytes: dict[str, int] = {cube_id: 0 for cube_id in verified.cube_ids}
    for row in shard_rows:
        marker_path = _shard_paths(verified.release_root, str(row["shard_id"]))[2]
        marker = _load_json(marker_path)
        input_hashes.add(marker_path)
        shard_bytes[str(row["cube_id"])] += int(marker["staging_logical_bytes_before_marker"])
    rows: dict[str, dict[str, Any]] = {}
    for cube_id in verified.cube_ids:
        manifest = manifests[cube_id]
        restart_path = verified.extraction_root / str(
            verified.campaign["phase2_sources"][cube_id]["phase4_restart_check"][
                "restart_check_relative_path"
            ]
        )
        restart = _load_json(restart_path)
        support_modes: dict[str, Any] = {}
        reduction_bytes = 0
        for support_mode in SUPPORT_MODES:
            group = groups[(cube_id, support_mode)]
            result = group.result
            marker_path = _reduction_paths(
                verified.release_root, group.group_id
            )[3]
            marker = _load_json(marker_path)
            input_hashes.add(marker_path)
            reduction_bytes += int(marker["staging_logical_bytes_before_marker"])
            support_modes[support_mode] = {
                "estimator_elapsed_seconds_sum": float(result.elapsed_seconds),
                "sampled_origins": int(result.sampled_pairs.sum()),
                "eligible_origins": int(result.eligible_pairs.sum()),
                "cube_candidate_origins": int(result.cube_candidate_pairs.sum()),
                "excluded_boundary_origins": int(result.excluded_boundary_pairs.sum()),
                "support_policy_excluded_origins": int(
                    np.sum(result.support_policy_excluded_origins)
                    if result.support_policy_excluded_origins is not None
                    else 0
                ),
                "accepted_measurement_samples_across_channels": int(result.counts.sum()),
                "directional_exclusion_counts_across_channels": _exclusion_summary(result),
                "reduction_elapsed_seconds": float(marker["reduction_elapsed_seconds"]),
                "jackknife_elapsed_seconds": float(marker["jackknife_elapsed_seconds"]),
                "bootstrap_elapsed_seconds": float(marker["bootstrap_elapsed_seconds"]),
                "reduction_staging_logical_bytes_before_marker": int(
                    marker["staging_logical_bytes_before_marker"]
                ),
            }
        performance = manifest["performance"]
        rows[cube_id] = {
            "extraction_payload_wall_seconds": float(performance["payload_wall_seconds"]),
            "extraction_hash_wall_seconds": float(performance["hash_wall_seconds"]),
            "extraction_total_wall_seconds_before_publish": float(
                performance["total_wall_seconds_before_publish"]
            ),
            "extraction_peak_rss_kib": int(performance["peak_rss_kib"]),
            "extraction_output_bytes": int(performance["output_bytes"]),
            "extraction_allocated_output_bytes": int(performance["allocated_output_bytes"]),
            "strict_restart_verify_wall_seconds": float(restart["verify_wall_seconds"]),
            "sampler_parent_process_peak_rss_kib_max_for_records_touching_cube": (
                sampler_rss_by_cube[cube_id]
            ),
            "sampler_shard_staging_logical_bytes_before_markers": shard_bytes[cube_id],
            "sampler_reduction_staging_logical_bytes_before_markers": reduction_bytes,
            "support_modes": support_modes,
        }
    return rows, sampler_resources, resource_gaps


def _write_environment_tables(
    output_dir: Path,
    verified: VerifiedInputs,
    catalog_rows: Mapping[str, Mapping[str, Any]],
    primitive_rows: Mapping[str, Mapping[str, Any]],
    operational_rows: Mapping[str, Mapping[str, Any]],
    *,
    catalog_metadata: Mapping[str, Any],
    pressure_convention: Mapping[str, Any] | None,
    gaps: Sequence[str],
) -> tuple[Path, Path]:
    rows = [
        {
            "cube_id": cube_id,
            "role": catalog_rows[cube_id]["role"],
            "phase1_catalog_supported": dict(catalog_rows[cube_id]),
            "full_resolution_primitive_exploratory": dict(primitive_rows[cube_id]),
            "operational": dict(operational_rows[cube_id]),
        }
        for cube_id in verified.cube_ids
    ]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "cube_count": len(rows),
        "classification_note": (
            "Phase 1 catalog-supported fields remain distinct from newly computed "
            "full-resolution primitive exploratory diagnostics."
        ),
        "phase1_catalog_metadata": catalog_metadata,
        "primitive_pressure_convention": pressure_convention,
        "gaps": list(gaps),
        "rows": rows,
    }
    json_path = output_dir / ENVIRONMENT_JSON_FILENAME
    _write_json(json_path, payload)
    flattened = []
    for row in rows:
        flat: dict[str, Any] = {"cube_id": row["cube_id"], "role": row["role"]}

        def add_flat(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for name, item in value.items():
                    add_flat(f"{prefix}_{name}", item)
            else:
                flat[prefix] = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, tuple))
                    else value
                )

        for prefix, category in (
            ("phase1", row["phase1_catalog_supported"]),
            ("primitive", row["full_resolution_primitive_exploratory"]),
            ("operational", row["operational"]),
        ):
            for name, value in category.items():
                if name in {"cube_id", "role"}:
                    continue
                add_flat(f"{prefix}_{name}", value)
        flattened.append(inherited._json_builtin(flat))
    csv_path = output_dir / ENVIRONMENT_CSV_FILENAME
    fieldnames = sorted({name for row in flattened for name in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    return json_path, csv_path


def _write_scale_diagnostics_tables(
    output_dir: Path,
    verified: VerifiedInputs,
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
) -> tuple[Path, Path]:
    """Publish shell-resolved support, exclusion, and uncertainty diagnostics."""

    rows = []
    for cube_id in verified.cube_ids:
        for support_mode in SUPPORT_MODES:
            group = groups[(cube_id, support_mode)]
            result = group.result
            uncertainty = group.uncertainty
            ell = phase3_figures._centers(result.ell_bin_edges)
            support_fraction = _support_fraction(result)
            for q_name in Q_NAMES:
                for direction in DIRECTION_NAMES:
                    moment_index = phase3_figures._moment_index(result, q_name, direction)
                    q_index, geometry_index = moment_index[:2]
                    for ell_index, center in enumerate(ell):
                        exclusion_counts = {
                            str(name): int(result.exclusions[q_index, geometry_index, index, ell_index])
                            for index, name in enumerate(result.exclusion_names)
                        }
                        is_shell_local = support_mode == OVERLAY_SUPPORT_MODE
                        curve_uncertainty_supported = bool(
                            _curve_uncertainty_support_mask(group, moment_index)[ell_index]
                        )
                        geometry_supported = bool(
                            is_shell_local
                            and support_fraction[ell_index]
                            >= APPROVED_OVERLAY_SUPPORT_FRACTION
                        )
                        science_facing_overlay_supported = bool(
                            geometry_supported and curve_uncertainty_supported
                        )
                        rows.append(
                            {
                                "cube_id": cube_id,
                                "support_mode": support_mode,
                                "q_name": q_name,
                                "direction": direction,
                                "shell_index": ell_index,
                                "ell_lower_cells": float(result.ell_bin_edges[ell_index]),
                                "ell_center_cells": float(center),
                                "ell_upper_cells": float(result.ell_bin_edges[ell_index + 1]),
                                "displacement_count": int(result.displacements_per_bin[ell_index]),
                                "sampled_origins": int(result.sampled_pairs[ell_index]),
                                "eligible_origins": int(result.eligible_pairs[ell_index]),
                                "cube_candidate_origins": int(
                                    result.cube_candidate_pairs[ell_index]
                                ),
                                "intrinsic_eligible_origins": int(
                                    result.intrinsic_eligible_origins[ell_index]
                                ),
                                "boundary_excluded_origins": int(
                                    result.boundary_excluded_origins[ell_index]
                                ),
                                "support_policy_excluded_origins": int(
                                    result.support_policy_excluded_origins[ell_index]
                                ),
                                "eligible_origin_fraction": float(support_fraction[ell_index]),
                                "shell_local_below_5pct_curve_overlay_threshold": (
                                    not geometry_supported if is_shell_local else None
                                ),
                                "shell_local_meets_5pct_curve_overlay_threshold": (
                                    geometry_supported if is_shell_local else None
                                ),
                                "shell_local_meets_10pct_slope_candidate_geometry_threshold": (
                                    bool(
                                        support_fraction[ell_index]
                                        >= SLOPE_CANDIDATE_SUPPORT_FRACTION
                                    )
                                    if is_shell_local
                                    else None
                                ),
                                "curve_uncertainty_support_passes": (
                                    curve_uncertainty_supported
                                ),
                                "shell_local_science_facing_overlay_supported": (
                                    science_facing_overlay_supported if is_shell_local else None
                                ),
                                "weak_shell_local_value_retained_as_flagged_diagnostic": (
                                    bool(is_shell_local and not science_facing_overlay_supported)
                                ),
                                "accepted_measurement_samples": int(
                                    result.counts[moment_index][ell_index]
                                ),
                                "curve_value": float(result.moments[moment_index][ell_index]),
                                "pair_sampling_standard_error": float(
                                    uncertainty["pair_sampling_standard_error"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "block_bootstrap_standard_error": float(
                                    uncertainty["block_bootstrap_standard_error"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "block_bootstrap_interval_low": float(
                                    uncertainty["block_bootstrap_interval_low"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "block_bootstrap_interval_high": float(
                                    uncertainty["block_bootstrap_interval_high"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "accepted_contributing_blocks": int(
                                    uncertainty["accepted_contributing_blocks"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "accepted_effective_blocks": float(
                                    uncertainty["accepted_effective_blocks"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "valid_bootstrap_resamples": int(
                                    uncertainty["valid_bootstrap_resamples"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "local_log_slope_diagnostic": float(
                                    uncertainty["local_log_slope"][moment_index][ell_index]
                                ),
                                "local_log_slope_support_mask": bool(
                                    uncertainty["local_log_slope_support_mask"][moment_index][
                                        ell_index
                                    ]
                                ),
                                "directional_exclusion_counts_across_channels": exclusion_counts,
                            }
                        )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "row_count": len(rows),
        "classification_note": (
            "Shell-local rows remain science-facing overlays only when they pass the 5% "
            "geometry threshold plus accepted-measurement, contributing-block, effective-block, "
            "valid-bootstrap, and finite-interval gates. Other rows remain visibly flagged "
            "diagnostics. Thresholds are reporting gates, not corrections."
        ),
        "exclusion_note": (
            "Directional exclusion counts are shell- and q-resolved counts across directional "
            "classification channels; they are repeated beside each direction-specific curve row."
        ),
        "rows": rows,
    }
    json_path = output_dir / SCALE_DIAGNOSTICS_JSON_FILENAME
    _write_json(json_path, payload)
    flattened = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key != "directional_exclusion_counts_across_channels"
        }
        exclusions = row["directional_exclusion_counts_across_channels"]
        flat["directional_exclusion_counts_across_channels_json"] = json.dumps(
            exclusions, sort_keys=True
        )
        flat.update(
            {f"directional_exclusion_{name}": count for name, count in exclusions.items()}
        )
        flattened.append(inherited._json_builtin(flat))
    csv_path = output_dir / SCALE_DIAGNOSTICS_CSV_FILENAME
    fieldnames = sorted({name for row in flattened for name in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    return json_path, csv_path


def _bind_ledger_summary_snapshot(
    path: Path,
    input_hashes: InputHashes,
) -> tuple[dict[str, Any], bytes]:
    """Bind one refreshed compute-budget summary for immutable report inclusion."""

    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"required compute-ledger summary is missing: {path}")
    snapshot = path.read_bytes()
    sha256 = hashlib.sha256(snapshot).hexdigest()
    input_hashes.bind_verified(path, sha256)
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"compute-ledger summary is not UTF-8 text: {path}") from error
    updated = re.search(r"^Updated: `([^`]+)`$", text, flags=re.MULTILINE)
    metric_labels = (
        "Workflow budget",
        "Consumed allocated runtime",
        "Remaining budget",
        "Pending maximum additional exposure",
        "Projected remaining after pending maximum",
    )
    metrics: dict[str, float] = {}
    for label in metric_labels:
        match = re.search(
            rf"^\| {re.escape(label)} \| `?([-+0-9.eE]+)`? \|$",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"compute-ledger summary lacks required metric {label!r}: {path}")
        metrics[label.lower().replace(" ", "_")] = float(match.group(1))
    if metrics["pending_maximum_additional_exposure"] != 0.0:
        raise RuntimeError("compute-ledger summary must report zero pending maximum exposure")
    return {
        "source_path": str(path),
        "source_sha256": sha256,
        "updated": updated.group(1) if updated is not None else None,
        "metrics_node_hours": metrics,
        "pending_exposure_gate": "passed_zero_pending_maximum_additional_exposure",
    }, snapshot


def _write_ledger_summary_snapshot(
    output_dir: Path,
    ledger_summary: Mapping[str, Any],
    snapshot: bytes,
) -> Path:
    path = output_dir / LEDGER_SUMMARY_SNAPSHOT_FILENAME
    path.write_bytes(snapshot)
    if file_sha256(path) != ledger_summary["source_sha256"]:
        raise RuntimeError("published compute-ledger summary snapshot changed during report assembly")
    return path


def _select_examples(catalog_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    def catalog(cube_id: str, name: str) -> float:
        return float(catalog_rows[cube_id]["catalog"][name])

    cube_ids = tuple(catalog_rows)
    role_groups = {
        "low_dBB": [
            cube_id
            for cube_id in cube_ids
            if catalog_rows[cube_id]["role"] == "representative:low_dBB"
        ],
        "median_dBB": [
            cube_id
            for cube_id in cube_ids
            if catalog_rows[cube_id]["role"] == "representative:near_median_dBB"
        ],
        "high_dBB": [
            cube_id
            for cube_id in cube_ids
            if catalog_rows[cube_id]["role"] == "representative:high_dBB"
        ],
    }
    selected: dict[str, str] = {}
    for name, candidates in role_groups.items():
        if not candidates:
            raise RuntimeError(f"Phase 1 pilot has no {name} representative")
        values = np.asarray([catalog(cube_id, "dBB") for cube_id in candidates])
        selected[name] = candidates[int(np.argmin(np.abs(values - np.median(values))))]
    selected["weak_mean_field"] = min(cube_ids, key=lambda cube_id: catalog(cube_id, "B_mean"))
    if len(set(selected.values())) != 4:
        raise RuntimeError("Phase 4 representative example selection collapsed to duplicate cubes")
    return selected


def workflow_schematic(output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(12.2, 3.5))
    axis.set_axis_off()
    boxes = (
        (0.01, "verified Phase 1\n21-cube pilot"),
        (0.18, "fresh Phase 4\nprimitive extraction"),
        (0.35, "strict restart +\nprovenance checks"),
        (0.52, "2-point Batch A\nfixed shards"),
        (0.69, "block bootstrap\ncurve products"),
        (0.86, "hash-bound\nreview package"),
    )
    for x, label in boxes:
        axis.add_patch(
            plt.Rectangle(
                (x, 0.35),
                0.13,
                0.34,
                facecolor="#e7f0fa",
                edgecolor="#2a5c8a",
                linewidth=1.4,
            )
        )
        axis.text(x + 0.065, 0.52, label, ha="center", va="center", fontsize=9.5)
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(right[0] - 0.01, 0.52),
            xytext=(left[0] + 0.14, 0.52),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444444"},
        )
    axis.text(
        0.5,
        0.12,
        "Schematic: Phase 4 Batch A review workflow; all other figures are retained-artifact quantitative products",
        ha="center",
        fontsize=9.5,
    )
    return _save(figure, output_dir, "phase4_batch_a_workflow_schematic.png")


def dbb_census(
    catalog_rows: Mapping[str, Mapping[str, Any]],
    examples: Mapping[str, str],
    output_dir: Path,
) -> Path:
    cube_ids = tuple(catalog_rows)
    d_bb = np.asarray([catalog_rows[cube_id]["catalog"]["dBB"] for cube_id in cube_ids])
    b_mean = np.asarray([catalog_rows[cube_id]["catalog"]["B_mean"] for cube_id in cube_ids])
    delta_b = np.asarray([catalog_rows[cube_id]["catalog"]["deltaB"] for cube_id in cube_ids])
    b_fraction = np.asarray(
        [catalog_rows[cube_id]["catalog"]["B_mean_sq_over_B2_mean"] for cube_id in cube_ids]
    )
    delta_fraction = np.asarray(
        [catalog_rows[cube_id]["catalog"]["deltaB_sq_over_B2_mean"] for cube_id in cube_ids]
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
    axes[0].scatter(d_bb, b_mean, color="#a7b4c3", s=34)
    axes[1].scatter(d_bb, delta_b, color="#a7b4c3", s=34)
    order = np.argsort(d_bb)
    axes[2].plot(d_bb[order], b_fraction[order], "o-", label=r"$B_{\rm mean}^2/\langle B^2\rangle$")
    axes[2].plot(
        d_bb[order],
        delta_fraction[order],
        "o-",
        label=r"$\delta B^2/\langle B^2\rangle$",
    )
    for label, cube_id in examples.items():
        index = cube_ids.index(cube_id)
        color = EXAMPLE_COLORS[label]
        axes[0].scatter([d_bb[index]], [b_mean[index]], color=color, edgecolor="black", s=72)
        axes[1].scatter([d_bb[index]], [delta_b[index]], color=color, edgecolor="black", s=72)
        axes[0].annotate(label.replace("_", " "), (d_bb[index], b_mean[index]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set_ylabel(r"$B_{\rm mean}$")
    axes[1].set_ylabel(r"$\delta B$")
    axes[2].set_ylabel("bounded magnetic complement")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel(r"$\mathrm{dBB} = \delta B/B_{\rm mean}$")
        axis.set_xscale("log")
        axis.grid(alpha=0.24)
    figure.suptitle("Phase 4 Batch A frozen 21-cube dBB census with magnetic complements")
    return _save(figure, output_dir, "phase4_batch_a_21cube_dbb_census.png")


def extraction_slice_montage(
    verified: VerifiedInputs,
    manifests: Mapping[str, Mapping[str, Any]],
    examples: Mapping[str, str],
    output_dir: Path,
) -> Path:
    images = []
    for label, cube_id in examples.items():
        metadata = manifests[cube_id]["output_fields"]["dens"]
        density = np.load(
            verified.extraction_root / cube_id / str(metadata["relative_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        image = np.asarray(density[density.shape[0] // 2], dtype=float)
        if image.ndim != 2 or np.any(~np.isfinite(image)) or np.any(image <= 0.0):
            raise RuntimeError(f"invalid density midplane for montage: {cube_id}")
        images.append((label, cube_id, image))
    values = np.concatenate([image.ravel() for _, _, image in images])
    vmin, vmax = np.quantile(values, (0.01, 0.99))
    if not 0.0 < vmin < vmax:
        raise RuntimeError("representative density montage has an invalid shared display range")
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), constrained_layout=True)
    plotted = None
    for axis, (label, cube_id, image) in zip(axes.flat, images):
        plotted = axis.imshow(
            image,
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=float(vmin), vmax=float(vmax)),
            interpolation="nearest",
        )
        axis.set_title(f"{label.replace('_', ' ')}: {cube_id}")
        axis.set_xlabel("$i$")
        axis.set_ylabel("$j$")
    assert plotted is not None
    colorbar = figure.colorbar(plotted, ax=axes, shrink=0.86)
    colorbar.set_label(r"$\rho$ (shared 1st-99th percentile display range)")
    figure.suptitle(r"Actual extracted primitive density on each representative cube's $k=320$ midplane")
    return _save(figure, output_dir, "phase4_batch_a_representative_extraction_slice_montage.png")


def support_vs_ell(
    verified: VerifiedInputs,
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), constrained_layout=True, sharey=True)
    for axis, support_mode in zip(axes, SUPPORT_MODES):
        rows = []
        ell_reference = None
        for cube_id in verified.cube_ids:
            result = groups[(cube_id, support_mode)].result
            ell = phase3_figures._centers(result.ell_bin_edges)
            if ell_reference is None:
                ell_reference = ell
            elif not np.array_equal(ell_reference, ell):
                raise RuntimeError("Phase 4 Batch A ell grid varies across cubes")
            rows.append(_support_fraction(result))
        values = np.asarray(rows)
        axis.fill_between(
            ell_reference,
            np.min(values, axis=0),
            np.max(values, axis=0),
            color=SUPPORT_MODE_COLORS[support_mode],
            alpha=0.16,
            label="21-cube range",
        )
        axis.plot(
            ell_reference,
            np.median(values, axis=0),
            color=SUPPORT_MODE_COLORS[support_mode],
            linewidth=1.8,
            label="21-cube median",
        )
        if support_mode == OVERLAY_SUPPORT_MODE:
            axis.axhline(APPROVED_OVERLAY_SUPPORT_FRACTION, color="#777777", linestyle="--", label="5% curve-overlay minimum")
            axis.axhline(SLOPE_CANDIDATE_SUPPORT_FRACTION, color="#777777", linestyle=":", label="10% slope-candidate minimum")
        axis.set_xscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_title(support_mode.replace("_", " "))
        axis.grid(alpha=0.24)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("eligible-origin fraction")
    figure.suptitle("Phase 4 Batch A support versus separation: primary and robustness-overlay policies")
    return _save(figure, output_dir, "phase4_batch_a_support_vs_ell.png")


def representative_curves_with_bands(
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
    examples: Mapping[str, str],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        len(examples),
        len(Q_NAMES),
        figsize=(12.4, 12.2),
        constrained_layout=True,
        sharex=True,
    )
    for row, (label, cube_id) in enumerate(examples.items()):
        primary = groups[(cube_id, PRIMARY_SUPPORT_MODE)]
        overlay = groups[(cube_id, OVERLAY_SUPPORT_MODE)]
        ell = phase3_figures._centers(primary.result.ell_bin_edges)
        overlay_support = _support_fraction(overlay.result)
        weak = overlay_support < APPROVED_OVERLAY_SUPPORT_FRACTION
        for column, q_name in enumerate(Q_NAMES):
            axis = axes[row, column]
            for direction in DIRECTION_NAMES:
                index = phase3_figures._moment_index(primary.result, q_name, direction)
                color = phase3_figures.DIRECTION_COLORS[direction]
                primary_supported = _curve_uncertainty_support_mask(primary, index)
                overlay_supported = (
                    ~weak & _curve_uncertainty_support_mask(overlay, index)
                )
                phase3_figures._plot_positive_band(
                    axis,
                    ell,
                    np.where(
                        primary_supported,
                        primary.uncertainty["block_bootstrap_interval_low"][index],
                        np.nan,
                    ),
                    np.where(
                        primary_supported,
                        primary.uncertainty["block_bootstrap_interval_high"][index],
                        np.nan,
                    ),
                    color=color,
                )
                phase3_figures._plot_positive_curve(
                    axis,
                    ell,
                    np.where(primary_supported, primary.result.moments[index], np.nan),
                    label=f"{direction}: primary",
                    color=color,
                )
                shell_values = overlay.result.moments[index]
                phase3_figures._plot_positive_curve(
                    axis,
                    ell,
                    np.where(overlay_supported, shell_values, np.nan),
                    label=f"{direction}: shell overlay" if row == 0 else "",
                    color=color,
                    linestyle="--",
                )
                flagged = ~overlay_supported & np.isfinite(shell_values) & (shell_values > 0.0)
                axis.plot(
                    ell[flagged],
                    shell_values[flagged],
                    color="#888888",
                    linestyle="none",
                    marker="x",
                    markersize=2.8,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(
                f"{label.replace('_', ' ')}: {cube_id}\n" + rf"$S_{{2,\perp}}^{{{q_name}}}(\ell)$"
            )
            axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    figure.suptitle(
        "Representative B/u curves: science-facing supported overlays; gray x: flagged shell-local diagnostics"
    )
    return _save(figure, output_dir, "phase4_batch_a_representative_B_u_curves_with_block_bands.png")


def curve_ratio_census(
    verified: VerifiedInputs,
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True, sharey=True)
    for axis, q_name in zip(axes, Q_NAMES):
        axis_supported_rows = []
        axis_flagged_geometry_rows = []
        for direction in DIRECTION_NAMES:
            retained_rows = []
            ell_reference = None
            for cube_index, cube_id in enumerate(verified.cube_ids):
                primary_group = groups[(cube_id, PRIMARY_SUPPORT_MODE)]
                overlay_group = groups[(cube_id, OVERLAY_SUPPORT_MODE)]
                primary = primary_group.result
                overlay = overlay_group.result
                ell = phase3_figures._centers(primary.ell_bin_edges)
                ell_reference = ell if ell_reference is None else ell_reference
                support = _support_fraction(overlay)
                index = phase3_figures._moment_index(primary, q_name, direction)
                ratio = np.divide(
                    primary.moments[index],
                    overlay.moments[index],
                    out=np.full_like(primary.moments[index], np.nan, dtype=float),
                    where=np.isfinite(overlay.moments[index]) & (overlay.moments[index] != 0.0),
                )
                geometry_supported = support >= APPROVED_OVERLAY_SUPPORT_FRACTION
                science_supported = (
                    geometry_supported
                    & _curve_uncertainty_support_mask(primary_group, index)
                    & _curve_uncertainty_support_mask(overlay_group, index)
                    & np.isfinite(ratio)
                    & (ratio > 0.0)
                )
                flagged_visible = ~science_supported & np.isfinite(ratio) & (ratio > 0.0)
                axis.plot(
                    ell[flagged_visible],
                    ratio[flagged_visible],
                    color="#888888",
                    linestyle="none",
                    marker="x",
                    markersize=2.8,
                    alpha=0.32,
                    label=(
                        "retained flagged diagnostics"
                        if direction == DIRECTION_NAMES[0] and cube_index == 0
                        else ""
                    ),
                )
                retained_rows.append(np.where(science_supported, ratio, np.nan))
                for ell_index in np.flatnonzero(science_supported):
                    axis_supported_rows.append(
                        (float(ratio[ell_index]), cube_id, direction, int(ell_index))
                    )
                geometry_only = geometry_supported & ~science_supported
                for ell_index in np.flatnonzero(geometry_only & np.isfinite(ratio) & (ratio > 0.0)):
                    axis_flagged_geometry_rows.append(
                        (
                            float(ratio[ell_index]),
                            cube_id,
                            direction,
                            int(ell_index),
                            int(overlay.counts[index][ell_index]),
                            float(
                                overlay_group.uncertainty["accepted_effective_blocks"][index][
                                    ell_index
                                ]
                            ),
                        )
                    )
            values = np.asarray(retained_rows)
            color = phase3_figures.DIRECTION_COLORS[direction]
            axis.fill_between(
                ell_reference,
                np.nanquantile(values, 0.10, axis=0),
                np.nanquantile(values, 0.90, axis=0),
                color=color,
                alpha=0.13,
            )
            axis.plot(
                ell_reference,
                np.nanmedian(values, axis=0),
                color=color,
                label=f"{direction}: median and 10-90%",
            )
        if axis_supported_rows:
            supported = max(
                axis_supported_rows,
                key=lambda row: max(row[0], 1.0 / row[0]),
            )
            supported_factor = max(supported[0], 1.0 / supported[0])
        else:
            supported_factor = float("nan")
        if axis_flagged_geometry_rows:
            flagged = max(
                axis_flagged_geometry_rows,
                key=lambda row: max(row[0], 1.0 / row[0]),
            )
            flagged_factor = max(flagged[0], 1.0 / flagged[0])
            flagged_note = (
                f"flagged geometry-only max={flagged_factor:.2f}x\n"
                f"{flagged[1]} {flagged[2]}: n={flagged[4]}, "
                f"$N_{{\\rm eff}}$={flagged[5]:.1f}"
            )
        else:
            flagged_note = "no flagged geometry-only rows"
        axis.text(
            0.03,
            0.97,
            f"supported max={supported_factor:.2f}x\n{flagged_note}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.82},
        )
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(rf"$S_{{2,\perp}}^{{{q_name},\mathrm{{all}}}} / S_{{2,\perp}}^{{{q_name},\mathrm{{shell}}}}$")
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=7)
    figure.suptitle(
        "21-cube support-policy curve-ratio census; gray x retain flagged diagnostics; sensitivities, not corrections"
    )
    return _save(figure, output_dir, "phase4_batch_a_support_policy_curve_ratio_census.png")


def matched_pair_curve_comparison(
    catalog_rows: Mapping[str, Mapping[str, Any]],
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
    output_dir: Path,
) -> Path:
    """Show Phase 1 matched low/high-dBB curve contrasts without fitting slopes."""

    pairs: dict[int, dict[str, str]] = {}
    for cube_id, catalog in catalog_rows.items():
        match = re.fullmatch(r"matched:(\d+):(low|high)", str(catalog["role"]))
        if match is not None:
            pairs.setdefault(int(match.group(1)), {})[match.group(2)] = cube_id
    if not pairs or any(set(pair) != {"low", "high"} for pair in pairs.values()):
        raise RuntimeError("Phase 1 pilot matched-pair roles are incomplete")
    ordered = sorted(pairs.items())
    figure, axes = plt.subplots(
        len(Q_NAMES),
        len(ordered),
        figsize=(4.25 * len(ordered), 7.0),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes).reshape(len(Q_NAMES), len(ordered))
    for column, (pair_id, pair) in enumerate(ordered):
        low = groups[(pair["low"], PRIMARY_SUPPORT_MODE)].result
        high = groups[(pair["high"], PRIMARY_SUPPORT_MODE)].result
        ell = phase3_figures._centers(low.ell_bin_edges)
        for row, q_name in enumerate(Q_NAMES):
            axis = axes[row, column]
            for direction in DIRECTION_NAMES:
                index = phase3_figures._moment_index(low, q_name, direction)
                ratio = np.divide(
                    high.moments[index],
                    low.moments[index],
                    out=np.full_like(high.moments[index], np.nan, dtype=float),
                    where=np.isfinite(low.moments[index]) & (low.moments[index] != 0.0),
                )
                phase3_figures._plot_positive_curve(
                    axis,
                    ell,
                    ratio,
                    label=direction,
                    color=phase3_figures.DIRECTION_COLORS[direction],
                )
            axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(
                rf"$S_{{2,\perp}}^{{{q_name},\mathrm{{high}}}}/"
                rf"S_{{2,\perp}}^{{{q_name},\mathrm{{low}}}}$"
            )
            axis.grid(alpha=0.22)
        axes[0, column].set_title(
            f"matched pair {pair_id}\n{pair['high']} / {pair['low']}",
            fontsize=8,
        )
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(
        "Phase 1 matched-pair all-valid-origin curve contrasts; exploratory ratios, not fitted scaling laws"
    )
    return _save(figure, output_dir, "phase4_batch_a_matched_pair_curve_comparison.png")


def local_slope_effective_block_diagnostic(
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
    examples: Mapping[str, str],
    output_dir: Path,
) -> Path:
    cube_id = examples["median_dBB"]
    primary = groups[(cube_id, PRIMARY_SUPPORT_MODE)]
    overlay = groups[(cube_id, OVERLAY_SUPPORT_MODE)]
    ell = phase3_figures._centers(primary.result.ell_bin_edges)
    shell_window = _complete_centered_window_mask(
        _support_fraction(overlay.result) >= SLOPE_CANDIDATE_SUPPORT_FRACTION
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), constrained_layout=True, sharex="col")
    for column, q_name in enumerate(Q_NAMES):
        for direction in DIRECTION_NAMES:
            color = phase3_figures.DIRECTION_COLORS[direction]
            primary_index = phase3_figures._moment_index(primary.result, q_name, direction)
            overlay_index = phase3_figures._moment_index(overlay.result, q_name, direction)
            low = primary.uncertainty["local_log_slope_bootstrap_interval_low"][primary_index]
            high = primary.uncertainty["local_log_slope_bootstrap_interval_high"][primary_index]
            band = np.isfinite(ell) & np.isfinite(low) & np.isfinite(high)
            axes[0, column].fill_between(ell[band], low[band], high[band], color=color, alpha=0.12)
            axes[0, column].plot(
                ell,
                primary.uncertainty["local_log_slope"][primary_index],
                color=color,
                label=f"{direction}: primary",
            )
            shell_mask = (
                shell_window
                & overlay.uncertainty["local_log_slope_support_mask"][overlay_index]
            )
            axes[0, column].plot(
                ell,
                np.where(
                    shell_mask,
                    overlay.uncertainty["local_log_slope"][overlay_index],
                    np.nan,
                ),
                color=color,
                linestyle="--",
            )
            axes[1, column].plot(
                ell,
                primary.uncertainty["accepted_effective_blocks"][primary_index],
                color=color,
                label=direction,
            )
        axes[0, column].axhline(0.0, color="#777777", linewidth=0.8)
        axes[1, column].axhline(
            inherited.MINIMUM_LOCAL_SLOPE_EFFECTIVE_BLOCKS,
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label="minimum effective blocks",
        )
        axes[0, column].set_ylabel(rf"diagnostic local $\alpha_{{{q_name},\perp}}(\ell)$")
        axes[1, column].set_ylabel("Kish effective accepted blocks")
        axes[1, column].set_xlabel(r"$\ell$ [cells]")
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    figure.suptitle(
        f"{cube_id}: centered 5-bin local-slope diagnostic and effective blocks; dashed shell slopes are display-eligible diagnostics, not fits"
    )
    return _save(figure, output_dir, "phase4_batch_a_local_slope_effective_block_diagnostic.png")


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    cursor = 0
    while cursor < values.size:
        stop = cursor + 1
        while stop < values.size and values[order[stop]] == values[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1) + 1.0
        cursor = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if np.count_nonzero(valid) < 3:
        return None
    left_rank = _rankdata(left_array[valid])
    right_rank = _rankdata(right_array[valid])
    if np.ptp(left_rank) == 0.0 or np.ptp(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _curve_scale_rows(
    verified: VerifiedInputs,
    catalog_rows: Mapping[str, Mapping[str, Any]],
    groups: Mapping[tuple[str, str], phase3_figures.ReleaseGroup],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for cube_id in verified.cube_ids:
        group = groups[(cube_id, PRIMARY_SUPPORT_MODE)]
        ell = phase3_figures._centers(group.result.ell_bin_edges)
        for requested_ell in SCIENCE_SCALE_TARGETS:
            ell_index = int(np.argmin(np.abs(ell - requested_ell)))
            for q_name in Q_NAMES:
                for direction in DIRECTION_NAMES:
                    index = phase3_figures._moment_index(group.result, q_name, direction)
                    rows.append(
                        {
                            "cube_id": cube_id,
                            "dBB": float(catalog_rows[cube_id]["catalog"]["dBB"]),
                            "B_mean": float(catalog_rows[cube_id]["catalog"]["B_mean"]),
                            "deltaB": float(catalog_rows[cube_id]["catalog"]["deltaB"]),
                            "B_mean_sq_over_B2_mean": float(
                                catalog_rows[cube_id]["catalog"]["B_mean_sq_over_B2_mean"]
                            ),
                            "q_name": q_name,
                            "direction": direction,
                            "requested_ell_cells": requested_ell,
                            "actual_shell_center_cells": float(ell[ell_index]),
                            "primary_curve_value": float(group.result.moments[index][ell_index]),
                            "block_bootstrap_interval_low": float(
                                group.uncertainty["block_bootstrap_interval_low"][index][ell_index]
                            ),
                            "block_bootstrap_interval_high": float(
                                group.uncertainty["block_bootstrap_interval_high"][index][ell_index]
                            ),
                        }
                    )
    correlations = []
    for requested_ell in SCIENCE_SCALE_TARGETS:
        for q_name in Q_NAMES:
            for direction in DIRECTION_NAMES:
                selected = [
                    row
                    for row in rows
                    if row["requested_ell_cells"] == requested_ell
                    and row["q_name"] == q_name
                    and row["direction"] == direction
                ]
                correlations.append(
                    {
                        "q_name": q_name,
                        "direction": direction,
                        "requested_ell_cells": requested_ell,
                        "actual_shell_center_cells": selected[0]["actual_shell_center_cells"],
                        "spearman_rho_dBB_vs_primary_curve_value": _spearman(
                            [row["dBB"] for row in selected],
                            [row["primary_curve_value"] for row in selected],
                        ),
                        "spearman_rho_dBB_vs_primary_curve_value_excluding_dBB_gt_5": _spearman(
                            [row["dBB"] for row in selected if row["dBB"] <= 5.0],
                            [
                                row["primary_curve_value"]
                                for row in selected
                                if row["dBB"] <= 5.0
                            ],
                        ),
                        "spearman_rho_deltaB_vs_primary_curve_value": _spearman(
                            [row["deltaB"] for row in selected],
                            [row["primary_curve_value"] for row in selected],
                        ),
                        "spearman_rho_deltaB_vs_primary_curve_value_excluding_dBB_gt_5": _spearman(
                            [row["deltaB"] for row in selected if row["dBB"] <= 5.0],
                            [
                                row["primary_curve_value"]
                                for row in selected
                                if row["dBB"] <= 5.0
                            ],
                        ),
                        "interpretation": "exploratory 21-cube rank correlation; not a fitted scaling law",
                    }
                )
    return rows, correlations


def environment_trend_summary(
    curve_scale_rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(4, 3, figsize=(13.3, 14.0), constrained_layout=True)
    for q_row, q_name in enumerate(Q_NAMES):
        for column, requested_ell in enumerate(SCIENCE_SCALE_TARGETS):
            lambda_correlation = next(
                item
                for item in correlations
                if item["q_name"] == q_name
                and item["direction"] == "lambda"
                and item["requested_ell_cells"] == requested_ell
            )
            actual_ell = lambda_correlation["actual_shell_center_cells"]
            rho_dbb = lambda_correlation["spearman_rho_dBB_vs_primary_curve_value"]
            rho_dbb_zoom = lambda_correlation[
                "spearman_rho_dBB_vs_primary_curve_value_excluding_dBB_gt_5"
            ]
            rho_delta_b_zoom = lambda_correlation[
                "spearman_rho_deltaB_vs_primary_curve_value_excluding_dBB_gt_5"
            ]
            for zoom_index, zoom in enumerate((False, True)):
                axis = axes[2 * q_row + zoom_index, column]
                for direction in DIRECTION_NAMES:
                    selected = [
                        item
                        for item in curve_scale_rows
                        if item["q_name"] == q_name
                        and item["direction"] == direction
                        and item["requested_ell_cells"] == requested_ell
                        and (not zoom or item["dBB"] <= 5.0)
                    ]
                    axis.scatter(
                        [item["dBB"] for item in selected],
                        [item["primary_curve_value"] for item in selected],
                        s=25,
                        color=phase3_figures.DIRECTION_COLORS[direction],
                        label=direction,
                        alpha=0.84,
                    )
                    if not zoom and direction == "lambda":
                        for item in selected:
                            if item["dBB"] > 5.0:
                                axis.annotate(
                                    item["cube_id"],
                                    (item["dBB"], item["primary_curve_value"]),
                                    xytext=(4, 3),
                                    textcoords="offset points",
                                    fontsize=6,
                                )
                axis.set_title(
                    (
                        rf"$\ell={actual_ell:.1f}$; full range; "
                        rf"$\rho_s^{{\rm dBB}}(\lambda)={rho_dbb:.2f}$"
                    )
                    if not zoom
                    else (
                        rf"$\ell={actual_ell:.1f}$; $\mathrm{{dBB}}\leq5$; "
                        rf"$\rho_s^{{\rm dBB}}={rho_dbb_zoom:.2f}$, "
                        rf"$\rho_s^{{\delta B}}={rho_delta_b_zoom:.2f}$"
                    ),
                    fontsize=8,
                )
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set_xlabel(r"$\mathrm{dBB}$")
                axis.set_ylabel(rf"$S_{{2,\perp}}^{{{q_name}}}(\ell)$")
                axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(
        "Exploratory all-valid-origin curves versus Phase 1 dBB: full log range and no-denominator-outlier zooms"
    )
    return _save(figure, output_dir, "phase4_batch_a_environment_trend_summary.png")


def runtime_storage_summary(
    verified: VerifiedInputs,
    catalog_rows: Mapping[str, Mapping[str, Any]],
    operational_rows: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    cube_ids = tuple(
        sorted(verified.cube_ids, key=lambda cube_id: catalog_rows[cube_id]["catalog"]["dBB"])
    )
    labels = [cube_id.replace("L640_", "") for cube_id in cube_ids]
    x = np.arange(len(cube_ids))
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.1), constrained_layout=True, sharex=True)
    axes[0, 0].plot(
        x,
        [operational_rows[cube_id]["extraction_payload_wall_seconds"] for cube_id in cube_ids],
        "o-",
        label="extraction payload assembly",
    )
    axes[0, 0].plot(
        x,
        [operational_rows[cube_id]["strict_restart_verify_wall_seconds"] for cube_id in cube_ids],
        "o-",
        label="strict restart verification",
    )
    for support_mode in SUPPORT_MODES:
        axes[0, 1].plot(
            x,
            [
                operational_rows[cube_id]["support_modes"][support_mode][
                    "estimator_elapsed_seconds_sum"
                ]
                for cube_id in cube_ids
            ],
            "o-",
            label=support_mode.replace("_", " "),
            color=SUPPORT_MODE_COLORS[support_mode],
        )
    axes[1, 0].plot(
        x,
        [
            operational_rows[cube_id]["extraction_allocated_output_bytes"] / 1024**3
            for cube_id in cube_ids
        ],
        "o-",
        label="extraction allocated output",
    )
    axes[1, 0].plot(
        x,
        [
            (
                operational_rows[cube_id]["sampler_shard_staging_logical_bytes_before_markers"]
                + operational_rows[cube_id][
                    "sampler_reduction_staging_logical_bytes_before_markers"
                ]
            )
            / 1024**3
            for cube_id in cube_ids
        ],
        "o-",
        label="sampler published staging",
    )
    axes[1, 1].plot(
        x,
        [operational_rows[cube_id]["extraction_peak_rss_kib"] / 1024**2 for cube_id in cube_ids],
        "o-",
        color="#9467bd",
        label="extractor-process manifest peak RSS",
    )
    sampler_rss = [
        operational_rows[cube_id][
            "sampler_parent_process_peak_rss_kib_max_for_records_touching_cube"
        ]
        for cube_id in cube_ids
    ]
    if any(value is not None for value in sampler_rss):
        axes[1, 1].plot(
            x,
            [float(value) / 1024**2 if value is not None else np.nan for value in sampler_rss],
            "o-",
            color="#8c564b",
            label="sampler task-parent RSS for retained records only; not process-tree peak",
        )
    axes[0, 0].set_ylabel("wall time [s]")
    axes[0, 1].set_ylabel("estimator elapsed-seconds sum")
    axes[1, 0].set_ylabel("storage [GiB]")
    axes[1, 1].set_ylabel("peak RSS [GiB]")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=58, ha="right", fontsize=7)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7)
    axes[1, 0].set_xlabel("cube ID ordered by dBB")
    axes[1, 1].set_xlabel("cube ID ordered by dBB")
    figure.suptitle(
        "Verified runtime and storage components; sampler RSS is record-covered task-parent RSS and may be partial"
    )
    return _save(figure, output_dir, "phase4_batch_a_runtime_storage_summary.png")


def _runtime_summary(
    verified: VerifiedInputs,
    operational_rows: Mapping[str, Mapping[str, Any]],
    sampler_resources: Mapping[str, Any],
) -> dict[str, Any]:
    estimator = {
        support_mode: sum(
            float(row["support_modes"][support_mode]["estimator_elapsed_seconds_sum"])
            for row in operational_rows.values()
        )
        for support_mode in SUPPORT_MODES
    }
    return {
        "measured_components": {
            "extraction_payload_wall_seconds_sum": sum(
                float(row["extraction_payload_wall_seconds"])
                for row in operational_rows.values()
            ),
            "strict_restart_verify_wall_seconds_sum": sum(
                float(row["strict_restart_verify_wall_seconds"])
                for row in operational_rows.values()
            ),
            "estimator_elapsed_seconds_sum_by_support_mode": estimator,
            "extractor_manifest_peak_rss_kib_max": max(
                int(row["extraction_peak_rss_kib"]) for row in operational_rows.values()
            ),
            "sampler_resources": dict(sampler_resources),
            "extraction_allocated_output_bytes_sum": sum(
                int(row["extraction_allocated_output_bytes"])
                for row in operational_rows.values()
            ),
            "sampler_published_staging_logical_bytes_sum": sum(
                int(row["sampler_shard_staging_logical_bytes_before_markers"])
                + int(row["sampler_reduction_staging_logical_bytes_before_markers"])
                for row in operational_rows.values()
            ),
            "settled_release_bytes_before_summary_marker": verified.release_summary.get(
                "settled_release_bytes_before_summary_marker"
            ),
        },
        "launch_decision_planning_proxies_node_hours": {
            "21_cube_extraction_conservative_wrapper_proxy": 1.7558,
            "21_cube_2point_all_valid_origins_estimator_allocation_share_proxy": 0.684,
            "21_cube_2point_shell_local_estimator_allocation_share_proxy": 0.727,
            "21_cube_two_policy_2point_estimator_allocation_share_proxy": 1.411,
        },
        "comparison_note": (
            "Measured component sums are retained wall-time or estimator elapsed-second "
            "accounting, not end-to-end scheduler node-hour measurements. Do not compare "
            "them numerically to allocation-share planning proxies without the ledger."
        ),
    }


def write_manifest(
    output_dir: Path,
    verified: VerifiedInputs,
    input_hashes: InputHashes,
) -> None:
    figures = sorted(path.name for path in output_dir.glob("*.png"))
    artifacts = sorted(
        (
            SUMMARY_FILENAME,
            ENVIRONMENT_JSON_FILENAME,
            ENVIRONMENT_CSV_FILENAME,
            SCALE_DIAGNOSTICS_JSON_FILENAME,
            SCALE_DIAGNOSTICS_CSV_FILENAME,
            LEDGER_SUMMARY_SNAPSHOT_FILENAME,
            *figures,
        )
    )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "publication_policy": "immutable_one_time_publish_refuse_existing_output_directory",
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_sha256": verified.implementation_sha256,
        "input_roots": {
            "phase1_root": str(verified.phase1_root),
            "extraction_root": str(verified.extraction_root),
            "release_root": str(verified.release_root),
        },
        "input_sha256": input_hashes.as_dict(),
        "generated_figures": figures,
        "figure_sha256": {name: file_sha256(output_dir / name) for name in figures},
        "generated_artifacts": artifacts,
        "generated_artifact_sha256": {
            name: file_sha256(output_dir / name) for name in artifacts
        },
        "report_summary_sha256": file_sha256(output_dir / SUMMARY_FILENAME),
        "environmental_table_sha256": {
            ENVIRONMENT_JSON_FILENAME: file_sha256(output_dir / ENVIRONMENT_JSON_FILENAME),
            ENVIRONMENT_CSV_FILENAME: file_sha256(output_dir / ENVIRONMENT_CSV_FILENAME),
        },
        "scale_resolved_diagnostics_sha256": {
            SCALE_DIAGNOSTICS_JSON_FILENAME: file_sha256(
                output_dir / SCALE_DIAGNOSTICS_JSON_FILENAME
            ),
            SCALE_DIAGNOSTICS_CSV_FILENAME: file_sha256(
                output_dir / SCALE_DIAGNOSTICS_CSV_FILENAME
            ),
        },
        "compute_ledger_summary_snapshot_sha256": file_sha256(
            output_dir / LEDGER_SUMMARY_SNAPSHOT_FILENAME
        ),
    }
    _write_json(output_dir / "figure_manifest.json", payload)


def _publish_atomic_directory(temporary_output: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
    temporary_output.rename(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the hash-bound Phase 4 Batch A mandatory-review package."
    )
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument(
        "--ledger-summary",
        type=Path,
        required=True,
        help="refreshed compute_budget_summary.md snapshot with zero pending exposure",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--chunk-k",
        type=int,
        default=8,
        help="maximum mmap-backed primitive k-planes traversed per diagnostic chunk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {args.output_dir}")
    input_hashes = InputHashes()
    ledger_summary, ledger_snapshot = _bind_ledger_summary_snapshot(
        args.ledger_summary, input_hashes
    )
    verified, manifests = verify_inputs(
        phase1_root=args.phase1_root,
        extraction_root=args.extraction_root,
        release_root=args.release_root,
        ledger_summary=ledger_summary,
        input_hashes=input_hashes,
    )
    groups = _load_groups(verified, input_hashes)
    catalog_rows, phase1_metadata = _phase1_catalog_rows(verified, input_hashes)
    pressure_convention, gaps = _pressure_convention(verified, input_hashes)
    primitive_rows = {
        cube_id: _primitive_diagnostics(
            verified.extraction_root / cube_id,
            manifests[cube_id],
            chunk_k=args.chunk_k,
            pressure_convention=pressure_convention,
        )
        for cube_id in verified.cube_ids
    }
    operational_rows, sampler_resources, resource_gaps = _operational_rows(
        verified, manifests, groups, input_hashes
    )
    examples = _select_examples(catalog_rows)
    curve_scale_rows, correlations = _curve_scale_rows(verified, catalog_rows, groups)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        environment_json, environment_csv = _write_environment_tables(
            temporary_output,
            verified,
            catalog_rows,
            primitive_rows,
            operational_rows,
            catalog_metadata=phase1_metadata,
            pressure_convention=pressure_convention,
            gaps=(*gaps, *resource_gaps),
        )
        scale_diagnostics_json, scale_diagnostics_csv = _write_scale_diagnostics_tables(
            temporary_output, verified, groups
        )
        ledger_snapshot_path = _write_ledger_summary_snapshot(
            temporary_output, verified.ledger_summary, ledger_snapshot
        )
        workflow_schematic(temporary_output)
        dbb_census(catalog_rows, examples, temporary_output)
        extraction_slice_montage(verified, manifests, examples, temporary_output)
        support_vs_ell(verified, groups, temporary_output)
        representative_curves_with_bands(groups, examples, temporary_output)
        curve_ratio_census(verified, groups, temporary_output)
        matched_pair_curve_comparison(catalog_rows, groups, temporary_output)
        local_slope_effective_block_diagnostic(groups, examples, temporary_output)
        environment_trend_summary(curve_scale_rows, correlations, temporary_output)
        runtime_storage_summary(verified, catalog_rows, operational_rows, temporary_output)
        known_gaps = [
            *gaps,
            *resource_gaps,
            (
                "No directional fit slope or fitted exponent is published by this generator. "
                "Local slopes are retained diagnostics only."
            ),
            (
                "Primitive-only environmental quantities are exploratory post-extraction "
                "covariates and are not independent Phase 1 cbin validation targets."
            ),
            (
                "Measured report components remain distinct from scheduler node-hours. "
                "The refreshed zero-pending compute-ledger summary is copied and hash-bound."
            ),
            (
                "This Batch A review package does not claim or authorize Phase 5 GO."
            ),
        ]
        summary = {
            "schema_version": 1,
            "status": "phase4_batch_a_review_package_generated",
            "decision_scope": "mandatory intermediate Batch A review only",
            "phase5_go_claimed": False,
            "batch_a2_data_claimed": False,
            "publication_policy": "immutable_one_time_publish_refuse_existing_output_directory",
            "input_roots": {
                "phase1_root": str(verified.phase1_root),
                "extraction_root": str(verified.extraction_root),
                "release_root": str(verified.release_root),
            },
            "strict_verification": verified.verification,
            "implementation_sha256": verified.implementation_sha256,
            "cube_count": len(verified.cube_ids),
            "cube_ids": verified.cube_ids,
            "representative_examples": examples,
            "support_policy": {
                "primary_curve_product": PRIMARY_SUPPORT_MODE,
                "required_directional_robustness_overlay": OVERLAY_SUPPORT_MODE,
                "shell_local_curve_overlay_minimum_fraction": APPROVED_OVERLAY_SUPPORT_FRACTION,
                "curve_minimum_accepted_measurements": MINIMUM_CURVE_ACCEPTED_MEASUREMENTS,
                "curve_minimum_contributing_blocks": MINIMUM_CURVE_CONTRIBUTING_BLOCKS,
                "curve_minimum_effective_blocks": MINIMUM_CURVE_EFFECTIVE_BLOCKS,
                "curve_minimum_valid_bootstrap_fraction": (
                    MINIMUM_CURVE_VALID_BOOTSTRAP_FRACTION
                ),
                "curve_requires_finite_block_bootstrap_interval": True,
                "shell_local_directional_slope_candidate_minimum_fraction": (
                    SLOPE_CANDIDATE_SUPPORT_FRACTION
                ),
                "directional_fit_slopes_published": False,
            },
            "primitive_pressure_convention": pressure_convention,
            "environmental_table": {
                "json_relative_path": environment_json.name,
                "json_sha256": file_sha256(environment_json),
                "csv_relative_path": environment_csv.name,
                "csv_sha256": file_sha256(environment_csv),
            },
            "scale_resolved_support_exclusion_uncertainty_table": {
                "json_relative_path": scale_diagnostics_json.name,
                "json_sha256": file_sha256(scale_diagnostics_json),
                "csv_relative_path": scale_diagnostics_csv.name,
                "csv_sha256": file_sha256(scale_diagnostics_csv),
                "unsupported_shell_local_values_retained_and_flagged": True,
            },
            "compute_ledger_summary_snapshot": {
                **verified.ledger_summary,
                "snapshot_relative_path": ledger_snapshot_path.name,
                "snapshot_sha256": file_sha256(ledger_snapshot_path),
            },
            "curve_scale_values": curve_scale_rows,
            "curve_scale_exploratory_rank_correlations": correlations,
            "runtime_storage_summary": _runtime_summary(
                verified, operational_rows, sampler_resources
            ),
            "known_gaps": known_gaps,
            "input_sha256": input_hashes.as_dict(),
        }
        _write_json(temporary_output / SUMMARY_FILENAME, summary)
        write_manifest(temporary_output, verified, input_hashes)
        _publish_atomic_directory(temporary_output, args.output_dir)
        print(
            f"Wrote Phase 4 Batch A review package: {args.output_dir.resolve()} "
            f"({len(list(args.output_dir.glob('*.png')))} figures)"
        )
        if gaps:
            print("Known primitive-diagnostic gaps: " + "; ".join(gaps))
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)


if __name__ == "__main__":
    main()
