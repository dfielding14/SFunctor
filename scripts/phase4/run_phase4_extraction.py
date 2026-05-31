#!/usr/bin/env python3
"""Plan and run extraction for the frozen 21-cube Phase 4 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from sfunctor.io.cube_extract import (
    CubeExtractionError,
    CubeSelection,
    extract_cube,
    load_pilot_selections,
    load_rank_map,
    load_snapshot_identity,
    verify_pilot_cube_output,
    verify_trusted_run,
    write_inspection_figure,
)

DOMAIN_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
PLAN_FILENAME = "phase4_extraction_plan.json"
PLAN_MARKER_FILENAME = "PHASE4_EXTRACTION_PLAN_COMPLETE.json"
MATERIALIZATION_ROOT = "phase4_materialization_records"
PHASE4_PILOT_CUBE_IDS = (
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
    paths = (
        Path(__file__).resolve(),
        root / "job_scripts" / "phase4" / "run_phase4_extract_andes.sh",
        root / "sfunctor" / "io" / "cube_extract.py",
        root / "scripts" / "phase2" / "run_phase2_extraction.py",
        root / "job_scripts" / "phase2" / "run_phase2_extract_andes.sh",
        root / "scripts" / "phase1" / "cbin_tools.py",
        root / "scripts" / "phase1" / "validate_reconstruction.py",
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


def _selection_payload(selection: CubeSelection) -> dict[str, Any]:
    return {
        "cube_id": selection.cube_id,
        "role": selection.role,
        "L_sub": selection.lsub,
        "bounds_ijk_half_open": list(selection.bounds_ijk),
        "shape_kji": list(selection.shape_kji),
        "required_rank_ids": list(selection.required_rank_ids),
    }


def _load_frozen_pilot(trusted_run: Path) -> dict[str, CubeSelection]:
    selections = load_pilot_selections(trusted_run)
    cube_ids = tuple(selections)
    if (
        len(cube_ids) != 21
        or cube_ids != PHASE4_PILOT_CUBE_IDS
        or any(selection.lsub != 640 for selection in selections.values())
        or any(selection.shape_kji != (640, 640, 640) for selection in selections.values())
    ):
        raise CubeExtractionError(
            "Phase 4 requires the exact frozen 21-selection L_sub=640 pilot from Phase 1"
        )
    return selections


def _snapshot_identity(trusted_run: Path, basename: str | None) -> dict[str, Any]:
    snapshot = load_snapshot_identity(trusted_run)
    if basename is not None and basename != snapshot["full_resolution_basename"]:
        raise CubeExtractionError("--basename must match the trusted Phase 1 full-resolution snapshot")
    return snapshot


def _plan_payload(
    trusted_run: Path,
    data_root: Path,
    output_root: Path,
    basename: str | None,
) -> dict[str, Any]:
    selections = _load_frozen_pilot(trusted_run)
    snapshot = _snapshot_identity(trusted_run, basename)
    return {
        "schema_version": 1,
        "phase": "phase4_bounded_21_cube_extraction",
        "status": "planned",
        "trusted_run": str(trusted_run.resolve()),
        "trusted_phase1_artifacts": verify_trusted_run(trusted_run),
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_basename": snapshot["full_resolution_basename"],
        "snapshot_identity": snapshot,
        "pilot_cube_count": len(selections),
        "pilot_cube_ids": list(selections),
        "selections": [_selection_payload(selection) for selection in selections.values()],
        "source_version": _source_version(),
    }


def plan(trusted_run: Path, data_root: Path, output_root: Path, *, basename: str | None) -> dict[str, Any]:
    """Freeze the trusted Phase 1 pilot and Phase 4 extraction implementation."""

    marker_path = output_root / PLAN_MARKER_FILENAME
    if marker_path.exists():
        return verify_plan(trusted_run, data_root, output_root, basename=basename)
    if output_root.exists() and any(output_root.iterdir()):
        raise CubeExtractionError(f"refusing to plan into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = _plan_payload(trusted_run, data_root, output_root, basename)
    plan_path = output_root / PLAN_FILENAME
    _atomic_write_json(plan_path, payload)
    _atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "status": "passed",
            "plan_sha256": file_sha256(plan_path),
            "implementation_sha256": payload["source_version"]["implementation_sha256"],
            "published_unix_seconds": time.time(),
        },
    )
    return verify_plan(trusted_run, data_root, output_root, basename=basename)


def verify_plan(
    trusted_run: Path,
    data_root: Path,
    output_root: Path,
    *,
    basename: str | None,
) -> dict[str, Any]:
    """Reject changed pilot membership, trusted sources, or adapter code."""

    plan_path = output_root / PLAN_FILENAME
    marker_path = output_root / PLAN_MARKER_FILENAME
    payload = json.loads(plan_path.read_text())
    marker = json.loads(marker_path.read_text())
    expected = _plan_payload(trusted_run, data_root, output_root, basename)
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("plan_sha256") != file_sha256(plan_path)
        or marker.get("implementation_sha256")
        != expected["source_version"]["implementation_sha256"]
        or payload != expected
    ):
        raise CubeExtractionError("invalid or stale Phase 4 extraction plan marker")
    return payload


def _plan_identity(output_root: Path) -> dict[str, str]:
    plan_path = output_root / PLAN_FILENAME
    marker_path = output_root / PLAN_MARKER_FILENAME
    return {
        "plan_relative_path": PLAN_FILENAME,
        "plan_sha256": file_sha256(plan_path),
        "marker_relative_path": PLAN_MARKER_FILENAME,
        "marker_sha256": file_sha256(marker_path),
    }


def _cube_publication_identity(output_root: Path, cube_id: str) -> dict[str, str]:
    cube_root = output_root / cube_id
    completion_path = cube_root / "COMPLETE.json"
    manifest_path = cube_root / "manifest.json"
    completion = json.loads(completion_path.read_text())
    manifest_sha256 = file_sha256(manifest_path)
    if (
        completion.get("cube_id") != cube_id
        or completion.get("manifest_sha256") != manifest_sha256
    ):
        raise CubeExtractionError(f"invalid cube publication marker: {completion_path}")
    return {
        "completion_relative_path": str(completion_path.relative_to(output_root)),
        "completion_sha256": file_sha256(completion_path),
        "manifest_relative_path": str(manifest_path.relative_to(output_root)),
        "manifest_sha256": manifest_sha256,
    }


def _materialization_record_path(output_root: Path, cube_id: str) -> Path:
    return output_root / MATERIALIZATION_ROOT / f"{cube_id}.json"


def _materialization_record_payload(output_root: Path, cube_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "phase4_fresh_extraction",
        "cube_id": cube_id,
        "phase4_extraction_plan": _plan_identity(output_root),
        "cube_publication": _cube_publication_identity(output_root, cube_id),
    }


def _publish_materialization_record(output_root: Path, cube_id: str) -> dict[str, str]:
    path = _materialization_record_path(output_root, cube_id)
    if path.exists():
        raise CubeExtractionError(f"refusing to overwrite Phase 4 materialization record: {path}")
    _atomic_write_json(path, _materialization_record_payload(output_root, cube_id))
    return _materialization_record_identity(output_root, cube_id)


def _materialization_record_identity(output_root: Path, cube_id: str) -> dict[str, str]:
    path = _materialization_record_path(output_root, cube_id)
    payload = json.loads(path.read_text())
    if payload != _materialization_record_payload(output_root, cube_id):
        raise CubeExtractionError(f"invalid or stale Phase 4 materialization record: {path}")
    return {
        "materialization_record_relative_path": str(path.relative_to(output_root)),
        "materialization_record_sha256": file_sha256(path),
    }


def _restart_check_payload(
    output_root: Path,
    cube_id: str,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **verification,
        "phase4_extraction_plan": _plan_identity(output_root),
        "phase4_materialization_record": _materialization_record_identity(output_root, cube_id),
        "cube_publication": _cube_publication_identity(output_root, cube_id),
    }


def _bind_extraction_materialization(
    output_root: Path,
    cube_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Publish a fresh record or require the existing Phase 4 restart record."""

    if payload["status"] == "extracted":
        payload["phase4_materialization_record"] = _publish_materialization_record(
            output_root,
            cube_id,
        )
    elif payload["status"] == "reused_complete_output":
        payload["phase4_materialization_record"] = _materialization_record_identity(
            output_root,
            cube_id,
        )
    else:
        raise CubeExtractionError(f"unexpected extraction status for {cube_id}: {payload}")
    return payload


def _selected_cube_ids(args: argparse.Namespace, selections: Mapping[str, CubeSelection]) -> tuple[str, ...]:
    if args.all_pilot and args.cube_id:
        raise SystemExit("--all-pilot cannot be combined with --cube-id")
    if not args.all_pilot and not args.cube_id:
        raise SystemExit("select cubes with --all-pilot or one or more --cube-id arguments")
    cube_ids = tuple(selections) if args.all_pilot else tuple(args.cube_id)
    if len(cube_ids) != len(set(cube_ids)):
        raise SystemExit("duplicate --cube-id values are not permitted")
    unknown = [cube_id for cube_id in cube_ids if cube_id not in selections]
    if unknown:
        raise SystemExit(f"Phase 4 extraction is restricted to the frozen pilot IDs: {unknown}")
    if args.all_pilot and (
        cube_ids != PHASE4_PILOT_CUBE_IDS
        or len(cube_ids) != 21
        or any(selection.lsub != 640 for selection in selections.values())
        or any(selection.shape_kji != (640, 640, 640) for selection in selections.values())
    ):
        raise SystemExit("--all-pilot requires exactly 21 frozen L_sub=640 selections")
    return cube_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "extract", "verify", "inspect"))
    parser.add_argument("--trusted-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cube-id", action="append")
    parser.add_argument("--all-pilot", action="store_true")
    parser.add_argument("--basename")
    parser.add_argument("--clean-partial", action="store_true")
    parser.add_argument("--clean-incomplete", action="store_true")
    parser.add_argument("--clean-stale-lock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.action == "plan":
        if args.all_pilot or args.cube_id:
            raise SystemExit("plan always freezes the complete pilot; do not pass cube selectors")
        print(
            json.dumps(
                plan(args.trusted_run, args.data_root, args.output_root, basename=args.basename),
                indent=2,
                sort_keys=True,
            )
        )
        return

    verify_plan(args.trusted_run, args.data_root, args.output_root, basename=args.basename)
    selections = _load_frozen_pilot(args.trusted_run)
    cube_ids = _selected_cube_ids(args, selections)
    if args.action == "inspect":
        for cube_id in cube_ids:
            output = args.output_root / "inspection" / f"{cube_id}_midplanes.png"
            write_inspection_figure(args.output_root / cube_id, output)
            print(output)
        return

    snapshot = _snapshot_identity(args.trusted_run, args.basename)
    rank_map = load_rank_map(args.trusted_run)
    common = {
        "data_root": args.data_root,
        "basename": snapshot["full_resolution_basename"],
        "trusted_run": args.trusted_run,
        "rank_map": rank_map,
        "expected_time": snapshot["target_time"],
        "expected_cycle": snapshot["target_cycle"],
        "domain_bounds": DOMAIN_BOUNDS,
        "expected_header_identity": snapshot["full_snapshot_identity"],
    }
    for cube_id in cube_ids:
        if args.action == "extract":
            payload = extract_cube(
                selections[cube_id],
                output_root=args.output_root,
                clean_partial=args.clean_partial,
                clean_incomplete=args.clean_incomplete,
                clean_stale_lock=args.clean_stale_lock,
                verify_hashes=True,
                **common,
            )
            payload = _bind_extraction_materialization(args.output_root, cube_id, payload)
        else:
            payload = verify_pilot_cube_output(
                args.output_root / cube_id,
                selections[cube_id],
                verify_hashes=True,
                **common,
            )
            payload = _restart_check_payload(args.output_root, cube_id, payload)
            report = args.output_root / "restart_checks" / f"{cube_id}.json"
            _atomic_write_json(report, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
