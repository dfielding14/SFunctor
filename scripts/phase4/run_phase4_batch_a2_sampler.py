#!/usr/bin/env python3
"""Run the approved representative-cube Phase 4 Batch A2 stencil comparison."""
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
from scripts.phase4 import run_phase4_extraction as extraction

PHASE4_A2_CUBE_IDS = (
    "L640_sub00370",
    "L640_sub03942",
    "L640_sub00579",
    "L640_sub00738",
)
BATCH_A_REFERENCE_STENCIL_SPECS = {2: dict(inherited.STENCIL_SPECS[2])}
PHASE4_A2_STENCIL_SPECS = {
    width: dict(inherited.STENCIL_SPECS[width]) for width in (3, 5)
}
PHASE4_A2_SUPPORT_MODES = ("all_valid_origins", "shell_local")
SUMMARY_FILENAME = "phase4_batch_a2_summary.json"
SUMMARY_MARKER_FILENAME = "PHASE4_BATCH_A2_COMPLETE.json"
_INHERITED_SOURCE_VERSION = inherited._source_version
_INHERITED_PHASE2_SOURCE_IDENTITY = inherited._phase2_source_identity
_configured_batch_a_root: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require_file_sha256(root: Path, relative_path: str, expected_sha256: str) -> None:
    path = root / relative_path
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"stale Phase 4 Batch A reference artifact: {path}")


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    payload = _INHERITED_SOURCE_VERSION()
    hashes = dict(payload["implementation_source_hashes"])
    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "phase4" / "run_phase4_extraction.py",
        root / "job_scripts" / "phase4" / "run_phase4_batch_a2_sampler_andes.sh",
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
    """Bind A2 to the immutable verified Batch A release without rewriting extraction."""

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
        != {"2": dict(BATCH_A_REFERENCE_STENCIL_SPECS[2])}
        or tuple(campaign.get("configuration", {}).get("support_modes", ()))
        != PHASE4_A2_SUPPORT_MODES
        or not set(PHASE4_A2_CUBE_IDS).issubset(campaign.get("phase2_sources", {}))
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


def _phase2_source_identity(
    phase2_root: Path,
    cube_id: str,
    *,
    verify_arrays: bool = True,
) -> dict[str, Any]:
    if _configured_batch_a_root is None:
        raise RuntimeError("Phase 4 Batch A2 reference root is not configured")
    frozen_sources, reference_identity = _batch_a_reference_sources(
        phase2_root, _configured_batch_a_root
    )
    if cube_id not in PHASE4_A2_CUBE_IDS:
        raise ValueError(f"Phase 4 Batch A2 is restricted to approved representative IDs: {cube_id}")
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
            raise RuntimeError(f"Phase 4 Batch A2 input differs from immutable Batch A: {cube_id}")
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
    return {**frozen, "phase4_batch_a_reference": reference_identity}


@contextmanager
def _configured_runner(batch_a_root: Path) -> Iterator[None]:
    """Apply the approved representative A2 constants only for one inherited call."""

    global _configured_batch_a_root
    original = {
        "phase3_cube_ids": phase3.BENCHMARK_CUBE_IDS,
        "cube_ids": inherited.BENCHMARK_CUBE_IDS,
        "stencils": inherited.STENCIL_SPECS,
        "support_modes": inherited.SUPPORT_MODES,
        "source_version": inherited._source_version,
        "phase2_source_identity": inherited._phase2_source_identity,
        "batch_a_root": _configured_batch_a_root,
    }
    phase3.BENCHMARK_CUBE_IDS = PHASE4_A2_CUBE_IDS
    inherited.BENCHMARK_CUBE_IDS = PHASE4_A2_CUBE_IDS
    inherited.STENCIL_SPECS = PHASE4_A2_STENCIL_SPECS
    inherited.SUPPORT_MODES = PHASE4_A2_SUPPORT_MODES
    inherited._source_version = _source_version
    inherited._phase2_source_identity = _phase2_source_identity
    _configured_batch_a_root = batch_a_root
    try:
        configuration = inherited._campaign_configuration()
        if (
            configuration["q_names"] != ("B", "u")
            or configuration["p_values"] != (2.0,)
            or configuration["stencils"] != PHASE4_A2_STENCIL_SPECS
            or configuration["support_modes"] != PHASE4_A2_SUPPORT_MODES
            or tuple(inherited.BENCHMARK_CUBE_IDS) != PHASE4_A2_CUBE_IDS
        ):
            raise RuntimeError("inherited sampler no longer matches approved Phase 4 Batch A2 matrix")
        yield
    finally:
        phase3.BENCHMARK_CUBE_IDS = original["phase3_cube_ids"]
        inherited.BENCHMARK_CUBE_IDS = original["cube_ids"]
        inherited.STENCIL_SPECS = original["stencils"]
        inherited.SUPPORT_MODES = original["support_modes"]
        inherited._source_version = original["source_version"]
        inherited._phase2_source_identity = original["phase2_source_identity"]
        _configured_batch_a_root = original["batch_a_root"]


def _summarize(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    payload = {
        **inherited.summarize(phase2_root, output_root),
        "phase": "phase4_batch_a2_bounded_4_cube_3point_5point",
        "representative_cube_ids": PHASE4_A2_CUBE_IDS,
        "batch_a_reference_root": str(_configured_batch_a_root.resolve()),
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
        raise RuntimeError("orphaned Phase 4 Batch A2 summary publication")
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
        or summary.get("phase") != "phase4_batch_a2_bounded_4_cube_3point_5point"
        or tuple(summary.get("representative_cube_ids", ())) != PHASE4_A2_CUBE_IDS
        or summary.get("batch_a_reference_root") != str(_configured_batch_a_root.resolve())
    ):
        raise RuntimeError("invalid or stale Phase 4 Batch A2 summary marker")
    return {**verification, "phase4_batch_a2_summary_status": "passed"}


def run(
    action: str,
    phase2_root: Path,
    batch_a_root: Path,
    output_root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    with _configured_runner(batch_a_root):
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
