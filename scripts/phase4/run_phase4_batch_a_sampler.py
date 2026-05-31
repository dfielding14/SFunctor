#!/usr/bin/env python3
"""Run the bounded Phase 4 Batch A matrix through the inherited Phase 3a sampler."""
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

PHASE4_PILOT_CUBE_IDS = extraction.PHASE4_PILOT_CUBE_IDS
EXTRACTION_PLAN_FILENAME = extraction.PLAN_FILENAME
EXTRACTION_PLAN_MARKER_FILENAME = extraction.PLAN_MARKER_FILENAME
PHASE4_STENCIL_SPECS = {2: dict(inherited.STENCIL_SPECS[2])}
PHASE4_SUPPORT_MODES = ("all_valid_origins", "shell_local")
SUMMARY_FILENAME = "phase4_batch_a_summary.json"
SUMMARY_MARKER_FILENAME = "PHASE4_BATCH_A_COMPLETE.json"
_INHERITED_SOURCE_VERSION = inherited._source_version
_INHERITED_PHASE2_SOURCE_IDENTITY = inherited._phase2_source_identity


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    payload = _INHERITED_SOURCE_VERSION()
    hashes = dict(payload["implementation_source_hashes"])
    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "phase4" / "run_phase4_extraction.py",
        root / "job_scripts" / "phase4" / "run_phase4_batch_a_sampler_andes.sh",
        root / "job_scripts" / "phase4" / "run_phase4_extract_andes.sh",
    )
    hashes.update({str(path.relative_to(root)): file_sha256(path) for path in paths})
    return {
        **payload,
        "implementation_source_hashes": hashes,
        "implementation_sha256": inherited._mapping_sha256(hashes),
    }


def _phase4_extraction_plan_identity(phase2_root: Path) -> dict[str, str]:
    """Bind Batch A inputs to one internally coherent Phase 4 extraction plan."""

    plan_path = phase2_root / EXTRACTION_PLAN_FILENAME
    marker_path = phase2_root / EXTRACTION_PLAN_MARKER_FILENAME
    plan = json.loads(plan_path.read_text())
    marker = json.loads(marker_path.read_text())
    current_extraction_source = extraction._source_version()
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("plan_sha256") != file_sha256(plan_path)
        or marker.get("implementation_sha256")
        != plan.get("source_version", {}).get("implementation_sha256")
        or marker.get("implementation_sha256")
        != current_extraction_source["implementation_sha256"]
        or plan.get("phase") != "phase4_bounded_21_cube_extraction"
        or plan.get("status") != "planned"
        or plan.get("output_root") != str(phase2_root.resolve())
        or plan.get("pilot_cube_count") != 21
        or tuple(plan.get("pilot_cube_ids", ())) != PHASE4_PILOT_CUBE_IDS
    ):
        raise RuntimeError("invalid Phase 4 extraction plan binding")
    try:
        verified_plan = extraction.verify_plan(
            Path(plan["trusted_run"]),
            Path(plan["data_root"]),
            phase2_root,
            basename=plan["source_basename"],
        )
    except Exception as error:
        raise RuntimeError("stale or untrusted Phase 4 extraction plan") from error
    if verified_plan != plan:
        raise RuntimeError("Phase 4 extraction plan changed during verification")
    return {
        "plan_relative_path": EXTRACTION_PLAN_FILENAME,
        "plan_sha256": file_sha256(plan_path),
        "marker_relative_path": EXTRACTION_PLAN_MARKER_FILENAME,
        "marker_sha256": file_sha256(marker_path),
    }


def _restart_check_identity(
    phase2_root: Path,
    cube_id: str,
    *,
    plan_identity: dict[str, str],
    materialization_identity: dict[str, str],
) -> dict[str, str]:
    """Require a recorded successful strict restart verification for one cube."""

    path = phase2_root / "restart_checks" / f"{cube_id}.json"
    payload = json.loads(path.read_text())
    publication_identity = extraction._cube_publication_identity(phase2_root, cube_id)
    if (
        payload.get("status") != "passed"
        or payload.get("cube_id") != cube_id
        or payload.get("verify_hashes") is not True
        or payload.get("phase4_extraction_plan") != plan_identity
        or payload.get("phase4_materialization_record") != materialization_identity
        or payload.get("cube_publication") != publication_identity
    ):
        raise RuntimeError(f"invalid Phase 4 restart check: {path}")
    return {
        "restart_check_relative_path": str(path.relative_to(phase2_root)),
        "restart_check_sha256": file_sha256(path),
    }


def _strict_phase4_cube_verification(phase2_root: Path, cube_id: str) -> None:
    """Recompute the full source, cbin, catalog, shape, and coverage checks."""

    plan = json.loads((phase2_root / EXTRACTION_PLAN_FILENAME).read_text())
    trusted_run = Path(plan["trusted_run"])
    data_root = Path(plan["data_root"])
    selections = extraction._load_frozen_pilot(trusted_run)
    snapshot = extraction._snapshot_identity(trusted_run, plan["source_basename"])
    extraction.verify_pilot_cube_output(
        phase2_root / cube_id,
        selections[cube_id],
        data_root=data_root,
        basename=snapshot["full_resolution_basename"],
        trusted_run=trusted_run,
        rank_map=extraction.load_rank_map(trusted_run),
        expected_time=snapshot["target_time"],
        expected_cycle=snapshot["target_cycle"],
        domain_bounds=extraction.DOMAIN_BOUNDS,
        expected_header_identity=snapshot["full_snapshot_identity"],
        verify_hashes=True,
    )


def _phase2_source_identity(
    phase2_root: Path,
    cube_id: str,
    *,
    verify_arrays: bool = True,
) -> dict[str, Any]:
    plan_identity = _phase4_extraction_plan_identity(phase2_root)
    materialization_identity = extraction._materialization_record_identity(phase2_root, cube_id)
    restart_identity = _restart_check_identity(
        phase2_root,
        cube_id,
        plan_identity=plan_identity,
        materialization_identity=materialization_identity,
    )
    if verify_arrays:
        _strict_phase4_cube_verification(phase2_root, cube_id)
    return {
        **_INHERITED_PHASE2_SOURCE_IDENTITY(
            phase2_root,
            cube_id,
            verify_arrays=verify_arrays,
        ),
        "phase4_extraction_plan": plan_identity,
        "phase4_materialization_record": materialization_identity,
        "phase4_restart_check": restart_identity,
    }


@contextmanager
def _configured_runner() -> Iterator[None]:
    """Apply Batch A constants only while invoking the inherited implementation."""

    original = {
        "phase3_cube_ids": phase3.BENCHMARK_CUBE_IDS,
        "cube_ids": inherited.BENCHMARK_CUBE_IDS,
        "stencils": inherited.STENCIL_SPECS,
        "support_modes": inherited.SUPPORT_MODES,
        "source_version": inherited._source_version,
        "phase2_source_identity": inherited._phase2_source_identity,
    }
    phase3.BENCHMARK_CUBE_IDS = PHASE4_PILOT_CUBE_IDS
    inherited.BENCHMARK_CUBE_IDS = PHASE4_PILOT_CUBE_IDS
    inherited.STENCIL_SPECS = PHASE4_STENCIL_SPECS
    inherited.SUPPORT_MODES = PHASE4_SUPPORT_MODES
    inherited._source_version = _source_version
    inherited._phase2_source_identity = _phase2_source_identity
    try:
        configuration = inherited._campaign_configuration()
        if (
            configuration["q_names"] != ("B", "u")
            or configuration["p_values"] != (2.0,)
            or configuration["stencils"] != PHASE4_STENCIL_SPECS
            or configuration["support_modes"] != PHASE4_SUPPORT_MODES
        ):
            raise RuntimeError("inherited sampler no longer matches the approved Phase 4 Batch A matrix")
        yield
    finally:
        phase3.BENCHMARK_CUBE_IDS = original["phase3_cube_ids"]
        inherited.BENCHMARK_CUBE_IDS = original["cube_ids"]
        inherited.STENCIL_SPECS = original["stencils"]
        inherited.SUPPORT_MODES = original["support_modes"]
        inherited._source_version = original["source_version"]
        inherited._phase2_source_identity = original["phase2_source_identity"]


def _campaign_configuration() -> dict[str, Any]:
    with _configured_runner():
        return inherited._campaign_configuration()


def _summarize(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    payload = {
        **inherited.summarize(phase2_root, output_root),
        "phase": "phase4_batch_a_bounded_21_cube_2point",
        "pilot_cube_ids": PHASE4_PILOT_CUBE_IDS,
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
        raise RuntimeError("orphaned Phase 4 Batch A summary publication")
    if not marker_path.exists():
        return verification
    marker = json.loads(marker_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (
        marker.get("schema_version") != inherited.SCHEMA_VERSION
        or marker.get("status") != "release_aggregation_complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or marker.get("implementation_sha256") != _source_version()["implementation_sha256"]
        or summary.get("source_version", {}).get("implementation_sha256")
        != marker["implementation_sha256"]
        or summary.get("phase") != "phase4_batch_a_bounded_21_cube_2point"
        or tuple(summary.get("pilot_cube_ids", ())) != PHASE4_PILOT_CUBE_IDS
    ):
        raise RuntimeError("invalid or stale Phase 4 Batch A summary marker")
    return {**verification, "phase4_summary_status": "passed"}


def run(action: str, phase2_root: Path, output_root: Path, *, workers: int) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    with _configured_runner():
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            inherited._json_builtin(
                run(args.action, args.phase2_root, args.output_root, workers=args.workers)
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
