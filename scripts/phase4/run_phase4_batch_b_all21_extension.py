#!/usr/bin/env python3
"""Run the explicitly authorized all-21 Phase 4 Batch B extension."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3 import run_phase3_sampler as phase3
from scripts.phase3a import run_phase3a_sampler as inherited
from scripts.phase4 import run_phase4_batch_b_representative_sampler as representative_batch_b
from scripts.phase4 import run_phase4_extraction as extraction


PHASE4_BATCH_B_ALL21_CUBE_IDS = extraction.PHASE4_PILOT_CUBE_IDS
PHASE4_BATCH_B_ALL21_STENCIL_SPECS = {
    2: {
        "label": "2-point",
        "ell_max": 320,
        "bin_count": 64,
        "directions_per_bin": 24,
    }
}
PHASE4_BATCH_B_ALL21_SUPPORT_MODES = ("all_valid_origins", "shell_local")
PHASE4_BATCH_B_ALL21_P_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
PHASE4_BATCH_B_ALL21_DENSITY_CONVENTIONS = ("not applicable", "not applicable")
DECISION_RELATIVE_PATH = "config/phase4_batch_b_all21_go_decision.json"
DECISION_REPORT_RELATIVE_PATH = "PHASE4_BATCH_B_ALL21_GO_DECISION.md"
REPRESENTATIVE_REVIEW_DECISION_RELATIVE_PATH = (
    "config/phase4_batch_b_representative_review_decision.json"
)
REPRESENTATIVE_STATUS_RELATIVE_PATH = "PHASE4_BATCH_B_REPRESENTATIVE_STATUS_UPDATE.md"
REPRESENTATIVE_REVIEW_SUMMARY_RELATIVE_PATH = (
    "figures/phase4_batch_b_representative_review/"
    "phase4_batch_b_representative_review_summary.json"
)
REPRESENTATIVE_FIGURE_MANIFEST_RELATIVE_PATH = (
    "figures/phase4_batch_b_representative_review/figure_manifest.json"
)
REPRESENTATIVE_BATCH_B_RELEASE = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase4/"
    "batch_b_representative_primary_20260601T185131Z"
)
TAIL_DIAGNOSTIC_RELEASE = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase4/"
    "batch_b_tail_diagnostic_v2_primary_20260602T003701Z"
)
TAIL_DIAGNOSTIC_SUMMARY_FILENAME = "phase4_batch_b_tail_diagnostic_summary.json"
TAIL_DIAGNOSTIC_MARKER_FILENAME = "PHASE4_BATCH_B_TAIL_DIAGNOSTIC_COMPLETE.json"
SUMMARY_FILENAME = "phase4_batch_b_all21_extension_summary.json"
SUMMARY_MARKER_FILENAME = "PHASE4_BATCH_B_ALL21_EXTENSION_COMPLETE.json"
_INHERITED_SOURCE_VERSION = inherited._source_version
_INHERITED_PHASE2_SOURCE_IDENTITY = inherited._phase2_source_identity
_configured_batch_a_root: Path | None = None
_configured_representative_a2_root: Path | None = None
_configured_all21_3point_extension_root: Path | None = None
_configured_representative_batch_b_root: Path | None = None
_configured_tail_diagnostic_root: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_file_sha256(relative_path: str) -> str:
    return file_sha256(_repo_root() / relative_path)


def _require_file_sha256(root: Path, relative_path: str, expected_sha256: str) -> None:
    path = root / relative_path
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"stale Phase 4 Batch A reference artifact: {path}")


def _representative_review_identity(
    representative_batch_b_root: Path | None = None,
) -> dict[str, str]:
    root = _repo_root()
    path = root / REPRESENTATIVE_REVIEW_DECISION_RELATIVE_PATH
    payload = _load_json(path)
    reviewed_scope = payload.get("reviewed_scope", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "hold_all21_batch_b_expansion_pending_tail_diagnostic"
        or payload.get("evidence_checkpoint") != REPRESENTATIVE_STATUS_RELATIVE_PATH
        or payload.get("probe_release") != REPRESENTATIVE_BATCH_B_RELEASE
        or payload.get("review_summary") != REPRESENTATIVE_REVIEW_SUMMARY_RELATIVE_PATH
        or payload.get("review_summary_sha256")
        != _repo_file_sha256(REPRESENTATIVE_REVIEW_SUMMARY_RELATIVE_PATH)
        or payload.get("figure_manifest") != REPRESENTATIVE_FIGURE_MANIFEST_RELATIVE_PATH
        or payload.get("figure_manifest_sha256")
        != _repo_file_sha256(REPRESENTATIVE_FIGURE_MANIFEST_RELATIVE_PATH)
        or reviewed_scope.get("cube_count") != len(representative_batch_b.PHASE4_BATCH_B_CUBE_IDS)
        or tuple(reviewed_scope.get("q_names", ())) != ("B", "u")
        or tuple(reviewed_scope.get("p_values", ())) != PHASE4_BATCH_B_ALL21_P_VALUES
        or tuple(reviewed_scope.get("stencil_widths", ())) != (2,)
        or tuple(reviewed_scope.get("support_modes", ()))
        != PHASE4_BATCH_B_ALL21_SUPPORT_MODES
        or payload.get("all21_batch_b_expansion_authorized") is not False
        or payload.get("five_point_expansion_authorized") is not False
        or payload.get("fitted_directional_exponents_authorized") is not False
        or payload.get("batch_c_authorized") is not False
        or payload.get("sgs_channels_authorized") is not False
        or payload.get("lsub1280_authorized") is not False
        or (
            representative_batch_b_root is not None
            and representative_batch_b_root.resolve()
            != Path(REPRESENTATIVE_BATCH_B_RELEASE).resolve()
        )
    ):
        raise RuntimeError("invalid Phase 4 Batch B representative review evidence")
    return {
        "review_decision_relative_path": REPRESENTATIVE_REVIEW_DECISION_RELATIVE_PATH,
        "review_decision_sha256": file_sha256(path),
        "status_relative_path": REPRESENTATIVE_STATUS_RELATIVE_PATH,
        "status_sha256": _repo_file_sha256(REPRESENTATIVE_STATUS_RELATIVE_PATH),
        "review_summary_relative_path": REPRESENTATIVE_REVIEW_SUMMARY_RELATIVE_PATH,
        "review_summary_sha256": _repo_file_sha256(
            REPRESENTATIVE_REVIEW_SUMMARY_RELATIVE_PATH
        ),
        "figure_manifest_relative_path": REPRESENTATIVE_FIGURE_MANIFEST_RELATIVE_PATH,
        "figure_manifest_sha256": _repo_file_sha256(
            REPRESENTATIVE_FIGURE_MANIFEST_RELATIVE_PATH
        ),
    }


def _tail_diagnostic_identity(tail_diagnostic_root: Path | None = None) -> dict[str, Any]:
    root = tail_diagnostic_root or Path(TAIL_DIAGNOSTIC_RELEASE)
    summary_path = root / TAIL_DIAGNOSTIC_SUMMARY_FILENAME
    marker_path = root / TAIL_DIAGNOSTIC_MARKER_FILENAME
    summary = _load_json(summary_path)
    marker = _load_json(marker_path)
    implementation_sha256 = summary.get("source_version", {}).get("implementation_sha256")
    input_identity = summary.get("input_identity", {})
    if (
        root.resolve() != Path(TAIL_DIAGNOSTIC_RELEASE).resolve()
        or marker.get("schema_version") != 2
        or marker.get("status") != "complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or not implementation_sha256
        or marker.get("implementation_sha256") != implementation_sha256
        or summary.get("schema_version") != 2
        or summary.get("status") != "complete"
        or summary.get("phase") != "phase4_batch_b_bounded_matched_origin_tail_diagnostic"
        or tuple(summary.get("cube_ids", ()))
        != ("L640_sub02822", "L640_sub03026", "L640_sub02602", "L640_sub00738")
        or tuple(summary.get("q_names", ())) != ("B", "u")
        or tuple(summary.get("directions", ())) != ("parallel", "xi", "lambda")
        or tuple(summary.get("p_values", ())) != (2.0, 4.0, 6.0)
        or tuple(
            (row.get("sample_count"), row.get("seed"))
            for row in summary.get("scenarios", ())
        )
        != (
            (2048, 20260530),
            (8192, 20260530),
            (32768, 20260530),
            (8192, 20260531),
            (8192, 20260532),
        )
        or summary.get("selected_displacement_count") != 898
        or len(summary.get("scenario_rows", ())) != 20
        or Path(input_identity.get("representative_release_root", "")).resolve()
        != Path(REPRESENTATIVE_BATCH_B_RELEASE).resolve()
    ):
        raise RuntimeError("invalid or stale Phase 4 Batch B tail diagnostic evidence")
    return {
        "tail_diagnostic_root": str(root.resolve()),
        "summary_relative_path": TAIL_DIAGNOSTIC_SUMMARY_FILENAME,
        "summary_sha256": file_sha256(summary_path),
        "marker_relative_path": TAIL_DIAGNOSTIC_MARKER_FILENAME,
        "marker_sha256": file_sha256(marker_path),
        "historical_implementation_sha256": implementation_sha256,
    }


def _decision_identity() -> dict[str, Any]:
    root = _repo_root()
    path = root / DECISION_RELATIVE_PATH
    report_path = root / DECISION_REPORT_RELATIVE_PATH
    payload = _load_json(path)
    review_identity = _representative_review_identity()
    tail_identity = _tail_diagnostic_identity()
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "post_diagnostic_user_authorized_all21_batch_b_extension"
        or payload.get("decision_date") != "2026-06-02"
        or payload.get("authorization_source") != "explicit_user_request"
        or payload.get("decision_scope")
        != "exact all-21 L640 2-point Batch B p=1..6 extension only"
        or payload.get("representative_review_status")
        != "hold_all21_batch_b_expansion_pending_tail_diagnostic"
        or payload.get("representative_review_hold_acknowledged") is not True
        or payload.get("representative_probe_release") != REPRESENTATIVE_BATCH_B_RELEASE
        or payload.get("representative_review_decision")
        != review_identity["review_decision_relative_path"]
        or payload.get("representative_review_decision_sha256")
        != review_identity["review_decision_sha256"]
        or payload.get("representative_status_evidence")
        != review_identity["status_relative_path"]
        or payload.get("representative_status_evidence_sha256")
        != review_identity["status_sha256"]
        or payload.get("representative_review_summary")
        != review_identity["review_summary_relative_path"]
        or payload.get("representative_review_summary_sha256")
        != review_identity["review_summary_sha256"]
        or payload.get("representative_figure_manifest")
        != review_identity["figure_manifest_relative_path"]
        or payload.get("representative_figure_manifest_sha256")
        != review_identity["figure_manifest_sha256"]
        or payload.get("tail_diagnostic_root") != tail_identity["tail_diagnostic_root"]
        or payload.get("tail_diagnostic_summary") != tail_identity["summary_relative_path"]
        or payload.get("tail_diagnostic_summary_sha256") != tail_identity["summary_sha256"]
        or payload.get("tail_diagnostic_marker") != tail_identity["marker_relative_path"]
        or payload.get("tail_diagnostic_marker_sha256") != tail_identity["marker_sha256"]
        or payload.get("tail_diagnostic_status") != "complete_review_acquisition_authorized"
        or payload.get("tail_diagnostic_interpretation")
        != "tail sensitivity remains real and imperfect; acquire all-21 review evidence without treating p=6 as converged"
        or tuple(payload.get("cube_ids", ())) != PHASE4_BATCH_B_ALL21_CUBE_IDS
        or tuple(payload.get("q_names", ())) != ("B", "u")
        or tuple(payload.get("p_values", ())) != PHASE4_BATCH_B_ALL21_P_VALUES
        or payload.get("stencils")
        != {
            str(width): dict(spec)
            for width, spec in PHASE4_BATCH_B_ALL21_STENCIL_SPECS.items()
        }
        or tuple(payload.get("support_modes", ())) != PHASE4_BATCH_B_ALL21_SUPPORT_MODES
        or tuple(payload.get("density_conventions", ()))
        != PHASE4_BATCH_B_ALL21_DENSITY_CONVENTIONS
        or payload.get("all21_batch_b_expansion_authorized") is not True
        or payload.get("density_weighting_authorized") is not False
        or payload.get("five_point_expansion_authorized") is not False
        or payload.get("fitted_directional_exponents_authorized") is not False
        or payload.get("batch_c_authorized") is not False
        or payload.get("sgs_channels_authorized") is not False
        or payload.get("lsub1280_authorized") is not False
    ):
        raise RuntimeError("invalid Phase 4 Batch B all-21 GO decision artifact")
    return {
        "decision_relative_path": DECISION_RELATIVE_PATH,
        "decision_sha256": file_sha256(path),
        "decision_report_relative_path": DECISION_REPORT_RELATIVE_PATH,
        "decision_report_sha256": file_sha256(report_path),
        "representative_review_identity": review_identity,
        "tail_diagnostic_identity": tail_identity,
    }


def _source_version() -> dict[str, Any]:
    root = _repo_root()
    payload = _INHERITED_SOURCE_VERSION()
    hashes = dict(payload["implementation_source_hashes"])
    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "phase4" / "run_phase4_extraction.py",
        root / "scripts" / "phase4" / "run_phase4_batch_a2_sampler.py",
        root / "scripts" / "phase4" / "run_phase4_batch_a2_3point_extension.py",
        root / "scripts" / "phase4" / "run_phase4_batch_b_representative_sampler.py",
        root / "job_scripts" / "phase4" / "run_phase4_batch_b_all21_extension_andes.sh",
        root / DECISION_RELATIVE_PATH,
        root / DECISION_REPORT_RELATIVE_PATH,
        root / REPRESENTATIVE_REVIEW_DECISION_RELATIVE_PATH,
        root / REPRESENTATIVE_STATUS_RELATIVE_PATH,
        root / REPRESENTATIVE_REVIEW_SUMMARY_RELATIVE_PATH,
        root / REPRESENTATIVE_FIGURE_MANIFEST_RELATIVE_PATH,
    )
    hashes.update({str(path.relative_to(root)): file_sha256(path) for path in paths})
    return {
        **payload,
        "implementation_source_hashes": hashes,
        "implementation_sha256": inherited._mapping_sha256(hashes),
    }


def _batch_a_reference_sources(
    phase2_root: Path,
    batch_a_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_sources, reference_identity = representative_batch_b._batch_a_reference_sources(
        phase2_root, batch_a_root
    )
    if (
        len(frozen_sources) != len(PHASE4_BATCH_B_ALL21_CUBE_IDS)
        or set(frozen_sources) != set(PHASE4_BATCH_B_ALL21_CUBE_IDS)
    ):
        raise RuntimeError("immutable Phase 4 Batch A reference is not the exact frozen all-21 set")
    return frozen_sources, reference_identity


def _representative_batch_b_reference(
    phase2_root: Path,
    batch_a_root: Path,
    representative_a2_root: Path,
    all21_3point_extension_root: Path,
    representative_batch_b_root: Path,
) -> dict[str, Any]:
    prerequisite_reference = representative_batch_b._all21_3point_extension_reference(
        batch_a_root, representative_a2_root, all21_3point_extension_root
    )
    campaign_path = representative_batch_b_root / "manifests" / "campaign.json"
    plan_marker_path = representative_batch_b_root / "PLAN_COMPLETE.json"
    inherited_summary_path = representative_batch_b_root / "phase3a_summary.json"
    inherited_marker_path = representative_batch_b_root / "PHASE3A_RELEASE_COMPLETE.json"
    summary_path = representative_batch_b_root / representative_batch_b.SUMMARY_FILENAME
    summary_marker_path = (
        representative_batch_b_root / representative_batch_b.SUMMARY_MARKER_FILENAME
    )
    campaign = _load_json(campaign_path)
    plan_marker = _load_json(plan_marker_path)
    inherited_marker = _load_json(inherited_marker_path)
    summary = _load_json(summary_path)
    summary_marker = _load_json(summary_marker_path)
    implementation_sha256 = campaign.get("source_version", {}).get("implementation_sha256")
    configuration = campaign.get("configuration", {})
    if (
        representative_batch_b_root.resolve() != Path(REPRESENTATIVE_BATCH_B_RELEASE).resolve()
        or plan_marker.get("schema_version") != inherited.SCHEMA_VERSION
        or plan_marker.get("status") != "passed"
        or plan_marker.get("campaign_sha256") != file_sha256(campaign_path)
        or inherited_marker.get("schema_version") != inherited.SCHEMA_VERSION
        or inherited_marker.get("status") != "release_aggregation_complete"
        or inherited_marker.get("summary_sha256") != file_sha256(inherited_summary_path)
        or summary_marker.get("schema_version") != inherited.SCHEMA_VERSION
        or summary_marker.get("status") != "release_aggregation_complete"
        or summary_marker.get("summary_sha256") != file_sha256(summary_path)
        or not implementation_sha256
        or plan_marker.get("implementation_sha256") != implementation_sha256
        or inherited_marker.get("implementation_sha256") != implementation_sha256
        or summary_marker.get("implementation_sha256") != implementation_sha256
        or summary.get("source_version", {}).get("implementation_sha256")
        != implementation_sha256
        or summary.get("phase") != "phase4_batch_b_bounded_8_cube_2point_p1_to_p6"
        or tuple(summary.get("representative_cube_ids", ()))
        != representative_batch_b.PHASE4_BATCH_B_CUBE_IDS
        or Path(summary.get("batch_a_reference_root", "")).resolve() != batch_a_root.resolve()
        or Path(summary.get("representative_a2_reference_root", "")).resolve()
        != representative_a2_root.resolve()
        or Path(summary.get("all21_3point_extension_reference_root", "")).resolve()
        != all21_3point_extension_root.resolve()
        or summary.get("decision_identity") != representative_batch_b._decision_identity()
        or Path(campaign.get("phase2_root", "")).resolve() != phase2_root.resolve()
        or tuple(configuration.get("q_names", ())) != ("B", "u")
        or tuple(configuration.get("p_values", ())) != PHASE4_BATCH_B_ALL21_P_VALUES
        or tuple(configuration.get("density_conventions", ()))
        != PHASE4_BATCH_B_ALL21_DENSITY_CONVENTIONS
        or configuration.get("stencils")
        != {
            str(width): dict(spec)
            for width, spec in PHASE4_BATCH_B_ALL21_STENCIL_SPECS.items()
        }
        or tuple(configuration.get("support_modes", ()))
        != PHASE4_BATCH_B_ALL21_SUPPORT_MODES
        or set(campaign.get("phase2_sources", {}))
        != set(representative_batch_b.PHASE4_BATCH_B_CUBE_IDS)
    ):
        raise RuntimeError("invalid or stale representative Phase 4 Batch B release")
    return {
        "representative_batch_b_root": str(representative_batch_b_root.resolve()),
        "campaign_relative_path": "manifests/campaign.json",
        "campaign_sha256": file_sha256(campaign_path),
        "plan_marker_relative_path": "PLAN_COMPLETE.json",
        "plan_marker_sha256": file_sha256(plan_marker_path),
        "summary_relative_path": representative_batch_b.SUMMARY_FILENAME,
        "summary_sha256": file_sha256(summary_path),
        "summary_marker_relative_path": representative_batch_b.SUMMARY_MARKER_FILENAME,
        "summary_marker_sha256": file_sha256(summary_marker_path),
        "historical_implementation_sha256": implementation_sha256,
        "all21_3point_extension_reference": prerequisite_reference,
    }


def _phase2_source_identity(
    phase2_root: Path,
    cube_id: str,
    *,
    verify_arrays: bool = True,
) -> dict[str, Any]:
    if (
        _configured_batch_a_root is None
        or _configured_representative_a2_root is None
        or _configured_all21_3point_extension_root is None
        or _configured_representative_batch_b_root is None
        or _configured_tail_diagnostic_root is None
    ):
        raise RuntimeError("Phase 4 Batch B all-21 reference roots are not configured")
    if cube_id not in PHASE4_BATCH_B_ALL21_CUBE_IDS:
        raise ValueError(f"Phase 4 Batch B all-21 extension rejects unapproved cube ID: {cube_id}")
    frozen_sources, batch_a_reference = _batch_a_reference_sources(
        phase2_root, _configured_batch_a_root
    )
    frozen = frozen_sources[cube_id]
    observed = _INHERITED_PHASE2_SOURCE_IDENTITY(
        phase2_root, cube_id, verify_arrays=verify_arrays
    )
    for key in (
        "cube_id",
        "phase2_root",
        "completion_relative_path",
        "completion_sha256",
        "manifest_relative_path",
        "manifest_sha256",
        "analysis_field_sha256",
    ):
        if observed.get(key) != frozen.get(key):
            raise RuntimeError(
                f"Phase 4 Batch B all-21 input differs from immutable Batch A: {cube_id}"
            )
    for identity_key, path_key, sha_key in (
        ("phase4_extraction_plan", "plan_relative_path", "plan_sha256"),
        ("phase4_extraction_plan", "marker_relative_path", "marker_sha256"),
        (
            "phase4_materialization_record",
            "materialization_record_relative_path",
            "materialization_record_sha256",
        ),
        ("phase4_restart_check", "restart_check_relative_path", "restart_check_sha256"),
    ):
        identity = frozen[identity_key]
        _require_file_sha256(phase2_root, identity[path_key], identity[sha_key])
    representative_reference = _representative_batch_b_reference(
        phase2_root,
        _configured_batch_a_root,
        _configured_representative_a2_root,
        _configured_all21_3point_extension_root,
        _configured_representative_batch_b_root,
    )
    return {
        **frozen,
        "phase4_batch_a_reference": batch_a_reference,
        "phase4_all21_3point_extension_reference": representative_reference[
            "all21_3point_extension_reference"
        ],
        "phase4_representative_batch_b_reference": representative_reference,
        "phase4_representative_batch_b_review": _representative_review_identity(
            _configured_representative_batch_b_root
        ),
        "phase4_batch_b_tail_diagnostic": _tail_diagnostic_identity(
            _configured_tail_diagnostic_root
        ),
        "phase4_batch_b_all21_decision": _decision_identity(),
    }


@contextmanager
def _configured_runner(
    batch_a_root: Path,
    representative_a2_root: Path,
    all21_3point_extension_root: Path,
    representative_batch_b_root: Path,
    tail_diagnostic_root: Path,
) -> Iterator[None]:
    """Apply the user-authorized all-21 Batch B constants for one inherited call."""

    global _configured_batch_a_root, _configured_representative_a2_root
    global _configured_all21_3point_extension_root, _configured_representative_batch_b_root
    global _configured_tail_diagnostic_root
    original = {
        "phase3_cube_ids": phase3.BENCHMARK_CUBE_IDS,
        "cube_ids": inherited.BENCHMARK_CUBE_IDS,
        "q_names": inherited.Q_NAMES,
        "stencils": inherited.STENCIL_SPECS,
        "support_modes": inherited.SUPPORT_MODES,
        "p_values": inherited.P_VALUES,
        "density_conventions": inherited.DENSITY_CONVENTIONS,
        "source_version": inherited._source_version,
        "phase2_source_identity": inherited._phase2_source_identity,
        "batch_a_root": _configured_batch_a_root,
        "representative_a2_root": _configured_representative_a2_root,
        "all21_3point_extension_root": _configured_all21_3point_extension_root,
        "representative_batch_b_root": _configured_representative_batch_b_root,
        "tail_diagnostic_root": _configured_tail_diagnostic_root,
    }
    phase3.BENCHMARK_CUBE_IDS = PHASE4_BATCH_B_ALL21_CUBE_IDS
    inherited.BENCHMARK_CUBE_IDS = PHASE4_BATCH_B_ALL21_CUBE_IDS
    inherited.Q_NAMES = ("B", "u")
    inherited.STENCIL_SPECS = PHASE4_BATCH_B_ALL21_STENCIL_SPECS
    inherited.SUPPORT_MODES = PHASE4_BATCH_B_ALL21_SUPPORT_MODES
    inherited.P_VALUES = PHASE4_BATCH_B_ALL21_P_VALUES
    inherited.DENSITY_CONVENTIONS = PHASE4_BATCH_B_ALL21_DENSITY_CONVENTIONS
    inherited._source_version = _source_version
    inherited._phase2_source_identity = _phase2_source_identity
    _configured_batch_a_root = batch_a_root
    _configured_representative_a2_root = representative_a2_root
    _configured_all21_3point_extension_root = all21_3point_extension_root
    _configured_representative_batch_b_root = representative_batch_b_root
    _configured_tail_diagnostic_root = tail_diagnostic_root
    try:
        configuration = inherited._campaign_configuration()
        if (
            tuple(extraction.PHASE4_PILOT_CUBE_IDS) != PHASE4_BATCH_B_ALL21_CUBE_IDS
            or configuration["q_names"] != ("B", "u")
            or configuration["p_values"] != PHASE4_BATCH_B_ALL21_P_VALUES
            or configuration["density_conventions"]
            != PHASE4_BATCH_B_ALL21_DENSITY_CONVENTIONS
            or configuration["stencils"] != PHASE4_BATCH_B_ALL21_STENCIL_SPECS
            or configuration["support_modes"] != PHASE4_BATCH_B_ALL21_SUPPORT_MODES
            or tuple(inherited.BENCHMARK_CUBE_IDS) != PHASE4_BATCH_B_ALL21_CUBE_IDS
        ):
            raise RuntimeError("inherited sampler no longer matches approved all-21 Batch B matrix")
        _decision_identity()
        yield
    finally:
        phase3.BENCHMARK_CUBE_IDS = original["phase3_cube_ids"]
        inherited.BENCHMARK_CUBE_IDS = original["cube_ids"]
        inherited.Q_NAMES = original["q_names"]
        inherited.STENCIL_SPECS = original["stencils"]
        inherited.SUPPORT_MODES = original["support_modes"]
        inherited.P_VALUES = original["p_values"]
        inherited.DENSITY_CONVENTIONS = original["density_conventions"]
        inherited._source_version = original["source_version"]
        inherited._phase2_source_identity = original["phase2_source_identity"]
        _configured_batch_a_root = original["batch_a_root"]
        _configured_representative_a2_root = original["representative_a2_root"]
        _configured_all21_3point_extension_root = original["all21_3point_extension_root"]
        _configured_representative_batch_b_root = original["representative_batch_b_root"]
        _configured_tail_diagnostic_root = original["tail_diagnostic_root"]


def _summarize(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    payload = {
        **inherited.summarize(phase2_root, output_root),
        "phase": "phase4_batch_b_all21_2point_p1_to_p6_extension",
        "pilot_cube_ids": PHASE4_BATCH_B_ALL21_CUBE_IDS,
        "batch_a_reference_root": str(_configured_batch_a_root.resolve()),
        "representative_a2_reference_root": str(_configured_representative_a2_root.resolve()),
        "all21_3point_extension_reference_root": str(
            _configured_all21_3point_extension_root.resolve()
        ),
        "representative_batch_b_reference_root": str(
            _configured_representative_batch_b_root.resolve()
        ),
        "tail_diagnostic_identity": _tail_diagnostic_identity(
            _configured_tail_diagnostic_root
        ),
        "representative_review_identity": _representative_review_identity(
            _configured_representative_batch_b_root
        ),
        "decision_identity": _decision_identity(),
    }
    summary_path = output_root / SUMMARY_FILENAME
    inherited._atomic_write_json(summary_path, payload)
    inherited._atomic_write_json(
        output_root / SUMMARY_MARKER_FILENAME,
        {
            "schema_version": inherited.SCHEMA_VERSION,
            "status": "release_aggregation_complete",
            "summary_sha256": file_sha256(summary_path),
            "implementation_sha256": payload["source_version"]["implementation_sha256"],
            "published_unix_seconds": time.time(),
        },
    )
    return payload


def _verify(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    verification = inherited.verify(phase2_root, output_root)
    marker_path = output_root / SUMMARY_MARKER_FILENAME
    summary_path = output_root / SUMMARY_FILENAME
    if marker_path.exists() != summary_path.exists():
        raise RuntimeError("orphaned Phase 4 Batch B all-21 summary publication")
    if not marker_path.exists():
        return verification
    marker = _load_json(marker_path)
    summary = _load_json(summary_path)
    if (
        marker.get("schema_version") != inherited.SCHEMA_VERSION
        or marker.get("status") != "release_aggregation_complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or marker.get("implementation_sha256") != _source_version()["implementation_sha256"]
        or summary.get("source_version", {}).get("implementation_sha256")
        != marker["implementation_sha256"]
        or summary.get("phase") != "phase4_batch_b_all21_2point_p1_to_p6_extension"
        or tuple(summary.get("pilot_cube_ids", ())) != PHASE4_BATCH_B_ALL21_CUBE_IDS
        or summary.get("batch_a_reference_root") != str(_configured_batch_a_root.resolve())
        or summary.get("representative_a2_reference_root")
        != str(_configured_representative_a2_root.resolve())
        or summary.get("all21_3point_extension_reference_root")
        != str(_configured_all21_3point_extension_root.resolve())
        or summary.get("representative_batch_b_reference_root")
        != str(_configured_representative_batch_b_root.resolve())
        or summary.get("tail_diagnostic_identity")
        != _tail_diagnostic_identity(_configured_tail_diagnostic_root)
        or summary.get("representative_review_identity")
        != _representative_review_identity(_configured_representative_batch_b_root)
        or summary.get("decision_identity") != _decision_identity()
    ):
        raise RuntimeError("invalid or stale Phase 4 Batch B all-21 summary marker")
    return {**verification, "phase4_batch_b_all21_extension_summary_status": "passed"}


def run(
    action: str,
    phase2_root: Path,
    batch_a_root: Path,
    representative_a2_root: Path,
    all21_3point_extension_root: Path,
    representative_batch_b_root: Path,
    tail_diagnostic_root: Path,
    output_root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    with _configured_runner(
        batch_a_root,
        representative_a2_root,
        all21_3point_extension_root,
        representative_batch_b_root,
        tail_diagnostic_root,
    ):
        actions = {
            "plan": lambda: inherited.plan(phase2_root, output_root),
            "work": lambda: inherited.work(phase2_root, output_root, workers=workers),
            "reduce": lambda: inherited.reduce(phase2_root, output_root),
            "verify": lambda: _verify(phase2_root, output_root),
            "summarize": lambda: _summarize(phase2_root, output_root),
        }
        return actions[action]()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "work", "reduce", "verify", "summarize"))
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--batch-a-root", type=Path, required=True)
    parser.add_argument("--representative-a2-root", type=Path, required=True)
    parser.add_argument("--all21-3point-extension-root", type=Path, required=True)
    parser.add_argument("--representative-batch-b-root", type=Path, required=True)
    parser.add_argument("--tail-diagnostic-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            inherited._json_builtin(
                run(
                    args.action,
                    args.phase2_root,
                    args.batch_a_root,
                    args.representative_a2_root,
                    args.all21_3point_extension_root,
                    args.representative_batch_b_root,
                    args.tail_diagnostic_root,
                    args.output_root,
                    workers=args.workers,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
