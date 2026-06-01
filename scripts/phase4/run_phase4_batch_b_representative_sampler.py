#!/usr/bin/env python3
"""Run the approved representative-cube Phase 4 Batch B order probe."""
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
from scripts.phase4 import run_phase4_batch_a2_3point_extension as all21_extension
from scripts.phase4 import run_phase4_batch_a2_sampler as representative_a2
from scripts.phase4 import run_phase4_extraction as extraction

PHASE4_BATCH_B_CUBE_IDS = (
    "L640_sub00370",
    "L640_sub03942",
    "L640_sub00579",
    "L640_sub00738",
    "L640_sub03026",
    "L640_sub00732",
    "L640_sub02822",
    "L640_sub02602",
)
PHASE4_BATCH_B_STENCIL_SPECS = {2: dict(inherited.STENCIL_SPECS[2])}
PHASE4_BATCH_B_SUPPORT_MODES = ("all_valid_origins", "shell_local")
PHASE4_BATCH_B_P_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
PHASE4_BATCH_B_DENSITY_CONVENTIONS = ("not applicable", "not applicable")
DECISION_RELATIVE_PATH = "config/phase4_batch_b_representative_go_decision.json"
DECISION_REPORT_RELATIVE_PATH = "PHASE4_BATCH_B_REPRESENTATIVE_GO_DECISION.md"
SUMMARY_FILENAME = "phase4_batch_b_representative_summary.json"
SUMMARY_MARKER_FILENAME = "PHASE4_BATCH_B_REPRESENTATIVE_COMPLETE.json"
_INHERITED_SOURCE_VERSION = inherited._source_version
_INHERITED_PHASE2_SOURCE_IDENTITY = inherited._phase2_source_identity
_configured_batch_a_root: Path | None = None
_configured_representative_a2_root: Path | None = None
_configured_all21_3point_extension_root: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require_file_sha256(root: Path, relative_path: str, expected_sha256: str) -> None:
    path = root / relative_path
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"stale Phase 4 Batch A reference artifact: {path}")


def _decision_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    path = root / DECISION_RELATIVE_PATH
    report_path = root / DECISION_REPORT_RELATIVE_PATH
    payload = _load_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "approved_bounded_representative_probe"
        or tuple(payload.get("cube_ids", ())) != PHASE4_BATCH_B_CUBE_IDS
        or tuple(payload.get("q_names", ())) != ("B", "u")
        or tuple(payload.get("p_values", ())) != PHASE4_BATCH_B_P_VALUES
        or payload.get("stencils")
        != {str(width): dict(spec) for width, spec in PHASE4_BATCH_B_STENCIL_SPECS.items()}
        or tuple(payload.get("support_modes", ())) != PHASE4_BATCH_B_SUPPORT_MODES
        or tuple(payload.get("density_conventions", ()))
        != PHASE4_BATCH_B_DENSITY_CONVENTIONS
        or payload.get("all21_batch_b_expansion_authorized") is not False
        or payload.get("five_point_expansion_authorized") is not False
        or payload.get("fitted_directional_exponents_authorized") is not False
        or payload.get("batch_c_authorized") is not False
        or payload.get("sgs_channels_authorized") is not False
        or payload.get("lsub1280_authorized") is not False
    ):
        raise RuntimeError("invalid Phase 4 Batch B representative-probe decision artifact")
    return {
        "decision_relative_path": DECISION_RELATIVE_PATH,
        "decision_sha256": file_sha256(path),
        "decision_report_relative_path": DECISION_REPORT_RELATIVE_PATH,
        "decision_report_sha256": file_sha256(report_path),
    }


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    payload = _INHERITED_SOURCE_VERSION()
    hashes = dict(payload["implementation_source_hashes"])
    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "phase4" / "run_phase4_extraction.py",
        root / "scripts" / "phase4" / "run_phase4_batch_a2_sampler.py",
        root / "scripts" / "phase4" / "run_phase4_batch_a2_3point_extension.py",
        root / "job_scripts" / "phase4" / "run_phase4_batch_b_representative_andes.sh",
        root / DECISION_RELATIVE_PATH,
        root / DECISION_REPORT_RELATIVE_PATH,
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
    """Bind Batch B to the immutable verified Batch A release without rewriting extraction."""

    campaign_path = batch_a_root / "manifests" / "campaign.json"
    plan_marker_path = batch_a_root / "PLAN_COMPLETE.json"
    inherited_summary_path = batch_a_root / "phase3a_summary.json"
    inherited_marker_path = batch_a_root / "PHASE3A_RELEASE_COMPLETE.json"
    summary_path = batch_a_root / "phase4_batch_a_summary.json"
    summary_marker_path = batch_a_root / "PHASE4_BATCH_A_COMPLETE.json"
    campaign = _load_json(campaign_path)
    plan_marker = _load_json(plan_marker_path)
    inherited_summary = _load_json(inherited_summary_path)
    inherited_marker = _load_json(inherited_marker_path)
    summary = _load_json(summary_path)
    summary_marker = _load_json(summary_marker_path)
    historical_implementation_sha256 = campaign.get("source_version", {}).get(
        "implementation_sha256"
    )
    if (
        plan_marker.get("schema_version") != inherited.SCHEMA_VERSION
        or plan_marker.get("status") != "passed"
        or plan_marker.get("campaign_sha256") != file_sha256(campaign_path)
        or inherited_marker.get("schema_version") != inherited.SCHEMA_VERSION
        or inherited_marker.get("status") != "release_aggregation_complete"
        or inherited_marker.get("summary_sha256") != file_sha256(inherited_summary_path)
        or summary_marker.get("schema_version") != inherited.SCHEMA_VERSION
        or summary_marker.get("status") != "release_aggregation_complete"
        or summary_marker.get("summary_sha256") != file_sha256(summary_path)
        or not historical_implementation_sha256
        or plan_marker.get("implementation_sha256") != historical_implementation_sha256
        or inherited_marker.get("implementation_sha256") != historical_implementation_sha256
        or summary_marker.get("implementation_sha256") != historical_implementation_sha256
        or summary.get("source_version", {}).get("implementation_sha256")
        != historical_implementation_sha256
        or summary.get("phase") != "phase4_batch_a_bounded_21_cube_2point"
        or tuple(summary.get("pilot_cube_ids", ())) != extraction.PHASE4_PILOT_CUBE_IDS
        or Path(campaign.get("phase2_root", "")).resolve() != phase2_root.resolve()
        or campaign.get("configuration", {}).get("stencils")
        != {"2": dict(PHASE4_BATCH_B_STENCIL_SPECS[2])}
        or tuple(campaign.get("configuration", {}).get("support_modes", ()))
        != PHASE4_BATCH_B_SUPPORT_MODES
        or not set(PHASE4_BATCH_B_CUBE_IDS).issubset(campaign.get("phase2_sources", {}))
    ):
        raise RuntimeError("invalid or stale immutable Phase 4 Batch A reference")
    reference_identity = {
        "batch_a_root": str(batch_a_root.resolve()),
        "campaign_relative_path": "manifests/campaign.json",
        "campaign_sha256": file_sha256(campaign_path),
        "plan_marker_relative_path": "PLAN_COMPLETE.json",
        "plan_marker_sha256": file_sha256(plan_marker_path),
        "summary_relative_path": "phase4_batch_a_summary.json",
        "summary_sha256": file_sha256(summary_path),
        "summary_marker_relative_path": "PHASE4_BATCH_A_COMPLETE.json",
        "summary_marker_sha256": file_sha256(summary_marker_path),
        "historical_implementation_sha256": historical_implementation_sha256,
    }
    return campaign["phase2_sources"], reference_identity


def _all21_3point_extension_reference(
    batch_a_root: Path,
    representative_a2_root: Path,
    all21_3point_extension_root: Path,
) -> dict[str, Any]:
    """Bind Batch B to the completed representative and all-21 A2 checkpoints."""

    representative_identity = all21_extension._representative_a2_reference(
        batch_a_root, representative_a2_root
    )
    campaign_path = all21_3point_extension_root / "manifests" / "campaign.json"
    plan_marker_path = all21_3point_extension_root / "PLAN_COMPLETE.json"
    inherited_summary_path = all21_3point_extension_root / "phase3a_summary.json"
    inherited_marker_path = all21_3point_extension_root / "PHASE3A_RELEASE_COMPLETE.json"
    summary_path = all21_3point_extension_root / all21_extension.SUMMARY_FILENAME
    summary_marker_path = all21_3point_extension_root / all21_extension.SUMMARY_MARKER_FILENAME
    campaign = _load_json(campaign_path)
    plan_marker = _load_json(plan_marker_path)
    inherited_marker = _load_json(inherited_marker_path)
    summary = _load_json(summary_path)
    summary_marker = _load_json(summary_marker_path)
    implementation_sha256 = campaign.get("source_version", {}).get("implementation_sha256")
    if (
        plan_marker.get("schema_version") != inherited.SCHEMA_VERSION
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
        or summary.get("phase") != "phase4_batch_a2_all21_3point_extension"
        or tuple(summary.get("pilot_cube_ids", ())) != extraction.PHASE4_PILOT_CUBE_IDS
        or Path(summary.get("batch_a_reference_root", "")).resolve() != batch_a_root.resolve()
        or Path(summary.get("representative_a2_reference_root", "")).resolve()
        != representative_a2_root.resolve()
        or campaign.get("configuration", {}).get("stencils")
        != {"3": dict(all21_extension.PHASE4_A2_EXTENSION_STENCIL_SPECS[3])}
        or tuple(campaign.get("configuration", {}).get("support_modes", ()))
        != PHASE4_BATCH_B_SUPPORT_MODES
    ):
        raise RuntimeError("invalid or stale all-21 Phase 4 Batch A2 3-point extension reference")
    return {
        "representative_a2_reference": representative_identity,
        "all21_3point_extension_root": str(all21_3point_extension_root.resolve()),
        "campaign_relative_path": "manifests/campaign.json",
        "campaign_sha256": file_sha256(campaign_path),
        "plan_marker_relative_path": "PLAN_COMPLETE.json",
        "plan_marker_sha256": file_sha256(plan_marker_path),
        "summary_relative_path": all21_extension.SUMMARY_FILENAME,
        "summary_sha256": file_sha256(summary_path),
        "summary_marker_relative_path": all21_extension.SUMMARY_MARKER_FILENAME,
        "summary_marker_sha256": file_sha256(summary_marker_path),
        "historical_implementation_sha256": implementation_sha256,
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
    ):
        raise RuntimeError("Phase 4 Batch B reference roots are not configured")
    frozen_sources, reference_identity = _batch_a_reference_sources(
        phase2_root, _configured_batch_a_root
    )
    if cube_id not in PHASE4_BATCH_B_CUBE_IDS:
        raise ValueError(f"Phase 4 Batch B is restricted to approved representative IDs: {cube_id}")
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
            raise RuntimeError(f"Phase 4 Batch B input differs from immutable Batch A: {cube_id}")
    for identity_key, path_key, sha_key in (
        ("phase4_extraction_plan", "plan_relative_path", "plan_sha256"),
        (
            "phase4_extraction_plan",
            "marker_relative_path",
            "marker_sha256",
        ),
        (
            "phase4_materialization_record",
            "materialization_record_relative_path",
            "materialization_record_sha256",
        ),
        ("phase4_restart_check", "restart_check_relative_path", "restart_check_sha256"),
    ):
        identity = frozen[identity_key]
        _require_file_sha256(phase2_root, identity[path_key], identity[sha_key])
    return {
        **frozen,
        "phase4_batch_a_reference": reference_identity,
        "phase4_all21_3point_extension_reference": _all21_3point_extension_reference(
            _configured_batch_a_root,
            _configured_representative_a2_root,
            _configured_all21_3point_extension_root,
        ),
        "phase4_batch_b_representative_decision": _decision_identity(),
    }


@contextmanager
def _configured_runner(
    batch_a_root: Path,
    representative_a2_root: Path,
    all21_3point_extension_root: Path,
) -> Iterator[None]:
    """Apply the approved representative Batch B constants for one inherited call."""

    global _configured_batch_a_root, _configured_representative_a2_root
    global _configured_all21_3point_extension_root
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
    }
    phase3.BENCHMARK_CUBE_IDS = PHASE4_BATCH_B_CUBE_IDS
    inherited.BENCHMARK_CUBE_IDS = PHASE4_BATCH_B_CUBE_IDS
    inherited.Q_NAMES = ("B", "u")
    inherited.STENCIL_SPECS = PHASE4_BATCH_B_STENCIL_SPECS
    inherited.SUPPORT_MODES = PHASE4_BATCH_B_SUPPORT_MODES
    inherited.P_VALUES = PHASE4_BATCH_B_P_VALUES
    inherited.DENSITY_CONVENTIONS = PHASE4_BATCH_B_DENSITY_CONVENTIONS
    inherited._source_version = _source_version
    inherited._phase2_source_identity = _phase2_source_identity
    _configured_batch_a_root = batch_a_root
    _configured_representative_a2_root = representative_a2_root
    _configured_all21_3point_extension_root = all21_3point_extension_root
    try:
        configuration = inherited._campaign_configuration()
        if (
            configuration["q_names"] != ("B", "u")
            or configuration["p_values"] != PHASE4_BATCH_B_P_VALUES
            or configuration["density_conventions"]
            != PHASE4_BATCH_B_DENSITY_CONVENTIONS
            or configuration["stencils"] != PHASE4_BATCH_B_STENCIL_SPECS
            or configuration["support_modes"] != PHASE4_BATCH_B_SUPPORT_MODES
            or tuple(inherited.BENCHMARK_CUBE_IDS) != PHASE4_BATCH_B_CUBE_IDS
        ):
            raise RuntimeError("inherited sampler no longer matches approved Phase 4 Batch B matrix")
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


def _summarize(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    payload = {
        **inherited.summarize(phase2_root, output_root),
        "phase": "phase4_batch_b_bounded_8_cube_2point_p1_to_p6",
        "representative_cube_ids": PHASE4_BATCH_B_CUBE_IDS,
        "batch_a_reference_root": str(_configured_batch_a_root.resolve()),
        "representative_a2_reference_root": str(_configured_representative_a2_root.resolve()),
        "all21_3point_extension_reference_root": str(
            _configured_all21_3point_extension_root.resolve()
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
        raise RuntimeError("orphaned Phase 4 Batch B summary publication")
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
        or summary.get("phase") != "phase4_batch_b_bounded_8_cube_2point_p1_to_p6"
        or tuple(summary.get("representative_cube_ids", ())) != PHASE4_BATCH_B_CUBE_IDS
        or summary.get("batch_a_reference_root") != str(_configured_batch_a_root.resolve())
        or summary.get("representative_a2_reference_root")
        != str(_configured_representative_a2_root.resolve())
        or summary.get("all21_3point_extension_reference_root")
        != str(_configured_all21_3point_extension_root.resolve())
        or summary.get("decision_identity") != _decision_identity()
    ):
        raise RuntimeError("invalid or stale Phase 4 Batch B summary marker")
    return {**verification, "phase4_batch_b_representative_summary_status": "passed"}


def run(
    action: str,
    phase2_root: Path,
    batch_a_root: Path,
    representative_a2_root: Path,
    all21_3point_extension_root: Path,
    output_root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    with _configured_runner(batch_a_root, representative_a2_root, all21_3point_extension_root):
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
