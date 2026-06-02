#!/usr/bin/env python3
"""Plan and run hash-bound Phase 5 cross-scale selected-cube extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase5 import build_phase5_campaign_config as campaign
from sfunctor.io.cube_extract import (
    CORE_EXTRACTOR_SOURCE_PATHS,
    CubeExtractionError,
    CubeSelection,
    extract_cube,
    load_rank_map,
    load_snapshot_identity,
    verify_pilot_cube_output,
    write_inspection_figure,
)

DOMAIN_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
EXTRACTABLE_SCALES = (320, 160, 80)
PLAN_FILENAME = "phase5_extraction_plan.json"
PLAN_MARKER_FILENAME = "PHASE5_EXTRACTION_PLAN_COMPLETE.json"
MATERIALIZATION_ROOT = "phase5_materialization_records"
RESTART_ROOT = "restart_checks"


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
        root / "scripts" / "phase5" / "build_phase5_campaign_config.py",
        root / "job_scripts" / "phase5" / "run_phase5_extract_andes.sh",
        *(root / relative_path for relative_path in CORE_EXTRACTOR_SOURCE_PATHS),
    )
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in paths}
    return {
        "implementation_source_hashes": hashes,
        "implementation_sha256": _mapping_sha256(hashes),
    }


def _snapshot_identity(trusted_run: Path, basename: str | None) -> dict[str, Any]:
    snapshot = load_snapshot_identity(trusted_run)
    if basename is not None and basename != snapshot["full_resolution_basename"]:
        raise CubeExtractionError("--basename must match the trusted Phase 1 full-resolution snapshot")
    return snapshot


def _config_identity(config_path: Path, trusted_run: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config_path = config_path.resolve()
    payload = campaign.verify_campaign_config_file(trusted_run, config_path)
    return payload, {
        "path": str(config_path),
        "sha256": file_sha256(config_path),
    }


def _extractor_role(row: Mapping[str, Any]) -> str:
    roles = row.get("roles")
    if not isinstance(roles, list) or not roles or any(not isinstance(role, str) for role in roles):
        raise CubeExtractionError(f"invalid Phase 5 role inventory for {row.get('cube_id')}")
    return ";".join(roles)


def _selection_from_config_row(row: Mapping[str, Any]) -> CubeSelection:
    try:
        cube_id = str(row["cube_id"])
        scale = int(row["L_sub"])
        bounds = tuple(int(value) for value in row["bounds_ijk_half_open"])
        required_rank_ids = tuple(int(value) for value in row["required_rank_ids"])
    except (KeyError, TypeError, ValueError) as error:
        raise CubeExtractionError("malformed Phase 5 campaign selection") from error
    if (
        scale not in EXTRACTABLE_SCALES
        or scale in campaign.PROHIBITED_EXTRACTION_SCALES
        or len(bounds) != 6
        or tuple(row.get("shape_kji", ())) != (scale, scale, scale)
        or len(required_rank_ids) != int(row.get("required_rank_count", -1))
        or len(required_rank_ids) != len(set(required_rank_ids))
        or row.get("magnetic_selection_valid") is not True
    ):
        raise CubeExtractionError(f"invalid configured Phase 5 selection: {cube_id}")
    return CubeSelection(
        cube_id=cube_id,
        bounds_ijk=bounds,  # type: ignore[arg-type]
        required_rank_ids=required_rank_ids,
        role=_extractor_role(row),
        lsub=scale,
    )


def _selected_config_rows(
    config: Mapping[str, Any],
    *,
    scale: int,
    subset: str,
) -> tuple[dict[str, Any], ...]:
    if scale not in EXTRACTABLE_SCALES or scale in campaign.PROHIBITED_EXTRACTION_SCALES:
        raise CubeExtractionError(f"Phase 5 extraction does not configure L_sub={scale}")
    if subset not in ("all", "smoke", "matched_smoke"):
        raise CubeExtractionError(f"unsupported Phase 5 subset: {subset}")
    if subset == "matched_smoke":
        if scale != 320:
            raise CubeExtractionError("Phase 5 matched_smoke is configured only for L_sub=320")
        raw_rows = config.get("matched_smoke_L320_selections")
        if not isinstance(raw_rows, list):
            raise CubeExtractionError("Phase 5 campaign config lacks L320 matched_smoke selections")
        rows = tuple(dict(row) for row in raw_rows)
    else:
        selections_by_scale = config.get("selections_by_scale")
        if not isinstance(selections_by_scale, Mapping):
            raise CubeExtractionError("Phase 5 campaign config lacks scale selections")
        raw_rows = selections_by_scale.get(str(scale))
        if not isinstance(raw_rows, list):
            raise CubeExtractionError(f"Phase 5 campaign config lacks L{scale} selections")
        rows = tuple(
            dict(row)
            for row in raw_rows
            if subset == "all" or row.get("smoke_anchor") is True
        )
    if not rows:
        raise CubeExtractionError(f"Phase 5 L{scale} subset {subset} is empty")
    cube_ids = tuple(str(row.get("cube_id")) for row in rows)
    if len(cube_ids) != len(set(cube_ids)):
        raise CubeExtractionError(f"Phase 5 L{scale} subset {subset} contains duplicate IDs")
    for row in rows:
        _selection_from_config_row(row)
    return rows


def _plan_payload(
    trusted_run: Path,
    data_root: Path,
    output_root: Path,
    campaign_config: Path,
    *,
    scale: int,
    subset: str,
    basename: str | None,
) -> dict[str, Any]:
    trusted_run = trusted_run.resolve()
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    config, config_identity = _config_identity(campaign_config, trusted_run)
    snapshot = _snapshot_identity(trusted_run, basename)
    if config.get("trusted_snapshot_identity") != snapshot:
        raise CubeExtractionError("Phase 5 config snapshot identity changed")
    if data_root != Path(snapshot["data_root"]).resolve():
        raise CubeExtractionError("--data-root must match the trusted Phase 1 primitive snapshot root")
    rows = _selected_config_rows(config, scale=scale, subset=subset)
    return {
        "schema_version": 1,
        "phase": "phase5_cross_scale_selected_cube_extraction",
        "status": "planned",
        "trusted_run": str(trusted_run),
        "trusted_phase1_artifacts": config["trusted_phase1_artifacts"],
        "data_root": str(data_root),
        "output_root": str(output_root),
        "source_basename": snapshot["full_resolution_basename"],
        "snapshot_identity": snapshot,
        "campaign_config": config_identity,
        "scope": {
            "L_sub": scale,
            "subset": subset,
            "cube_count": len(rows),
            "cube_ids": [row["cube_id"] for row in rows],
        },
        "selections": list(rows),
        "source_version": _source_version(),
    }


def plan(
    trusted_run: Path,
    data_root: Path,
    output_root: Path,
    campaign_config: Path,
    *,
    scale: int,
    subset: str,
    basename: str | None,
) -> dict[str, Any]:
    """Publish one new single-scale, single-subset plan root."""

    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise CubeExtractionError(f"refusing to plan into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = _plan_payload(
        trusted_run,
        data_root,
        output_root,
        campaign_config,
        scale=scale,
        subset=subset,
        basename=basename,
    )
    plan_path = output_root / PLAN_FILENAME
    marker_path = output_root / PLAN_MARKER_FILENAME
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
    return verify_plan(
        trusted_run,
        data_root,
        output_root,
        campaign_config,
        scale=scale,
        subset=subset,
        basename=basename,
    )


def verify_plan(
    trusted_run: Path,
    data_root: Path,
    output_root: Path,
    campaign_config: Path,
    *,
    scale: int,
    subset: str,
    basename: str | None,
) -> dict[str, Any]:
    """Reject stale config, snapshot, trusted-run, scope, or implementation bindings."""

    plan_path = output_root / PLAN_FILENAME
    marker_path = output_root / PLAN_MARKER_FILENAME
    try:
        payload = json.loads(plan_path.read_text())
        marker = json.loads(marker_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CubeExtractionError(f"incomplete Phase 5 extraction plan root: {output_root}") from error
    expected = _plan_payload(
        trusted_run,
        data_root,
        output_root,
        campaign_config,
        scale=scale,
        subset=subset,
        basename=basename,
    )
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("plan_sha256") != file_sha256(plan_path)
        or marker.get("implementation_sha256")
        != expected["source_version"]["implementation_sha256"]
        or payload != expected
    ):
        raise CubeExtractionError("invalid or stale Phase 5 extraction plan marker")
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
    if completion.get("cube_id") != cube_id or completion.get("manifest_sha256") != manifest_sha256:
        raise CubeExtractionError(f"invalid cube publication marker: {completion_path}")
    return {
        "completion_relative_path": str(completion_path.relative_to(output_root)),
        "completion_sha256": file_sha256(completion_path),
        "manifest_relative_path": str(manifest_path.relative_to(output_root)),
        "manifest_sha256": manifest_sha256,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_hash_bound_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    encoded = path.read_bytes()
    if not _is_sha256(expected_sha256) or hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise CubeExtractionError(f"{label} changed while validating provenance: {path}")
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise CubeExtractionError(f"{label} must contain one JSON object: {path}")
    return payload


def _plan_manifest_core_provenance(
    output_root: Path,
    cube_id: str,
    *,
    plan_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Require one exact plan-to-generic-extractor core-source binding."""

    plan_payload = _load_hash_bound_json(
        output_root / PLAN_FILENAME, plan_sha256, label="frozen Phase 5 extraction plan"
    )
    manifest = _load_hash_bound_json(
        output_root / cube_id / "manifest.json",
        manifest_sha256,
        label="published cube manifest",
    )
    plan_version = plan_payload.get("source_version")
    manifest_version = manifest.get("code_version")
    if not isinstance(plan_version, Mapping) or not isinstance(manifest_version, Mapping):
        raise CubeExtractionError("missing Phase 5 plan-to-manifest core provenance")
    plan_hashes = plan_version.get("implementation_source_hashes")
    manifest_hashes = manifest_version.get("implementation_source_hashes")
    if not isinstance(plan_hashes, Mapping) or not isinstance(manifest_hashes, Mapping):
        raise CubeExtractionError("malformed Phase 5 plan-to-manifest source hashes")
    if (
        not _is_sha256(plan_version.get("implementation_sha256"))
        or _mapping_sha256(plan_hashes) != plan_version["implementation_sha256"]
        or not _is_sha256(manifest_version.get("implementation_sha256"))
        or _mapping_sha256(manifest_hashes) != manifest_version["implementation_sha256"]
    ):
        raise CubeExtractionError("incoherent Phase 5 plan-to-manifest source hashes")
    try:
        core_hashes = {
            relative_path: plan_hashes[relative_path]
            for relative_path in CORE_EXTRACTOR_SOURCE_PATHS
        }
    except KeyError as error:
        raise CubeExtractionError("frozen Phase 5 plan lacks core extractor source hashes") from error
    if (
        any(not _is_sha256(value) for value in core_hashes.values())
        or dict(manifest_hashes) != core_hashes
        or manifest_version["implementation_sha256"] != _mapping_sha256(core_hashes)
    ):
        raise CubeExtractionError("Phase 5 plan-to-manifest core extractor provenance mismatch")
    return {
        "implementation_source_hashes": core_hashes,
        "implementation_sha256": _mapping_sha256(core_hashes),
        "manifest_git_commit": manifest_version.get("commit"),
        "manifest_git_dirty": manifest_version.get("dirty"),
    }


def _plan_selection(output_root: Path, cube_id: str) -> dict[str, Any]:
    plan_payload = json.loads((output_root / PLAN_FILENAME).read_text())
    rows = [
        row
        for row in plan_payload.get("selections", ())
        if isinstance(row, Mapping) and row.get("cube_id") == cube_id
    ]
    if len(rows) != 1:
        raise CubeExtractionError(f"frozen Phase 5 plan has {len(rows)} selections for {cube_id}")
    return dict(rows[0])


def _require_manifest_plan_binding(
    output_root: Path,
    cube_id: str,
    *,
    plan_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    plan_payload = _load_hash_bound_json(
        output_root / PLAN_FILENAME, plan_sha256, label="frozen Phase 5 extraction plan"
    )
    manifest = _load_hash_bound_json(
        output_root / cube_id / "manifest.json",
        manifest_sha256,
        label="published cube manifest",
    )
    row = _plan_selection(output_root, cube_id)
    selection = _selection_from_config_row(row)
    preflight = manifest.get("preflight", {})
    if (
        manifest.get("cube_id") != cube_id
        or manifest.get("Lsub") != selection.lsub
        or manifest.get("role") != selection.role
        or manifest.get("bounds_ijk_half_open") != list(selection.bounds_ijk)
        or manifest.get("shape_kji") != list(selection.shape_kji)
        or manifest.get("source_root") != plan_payload["data_root"]
        or manifest.get("source_basename") != plan_payload["source_basename"]
        or manifest.get("trusted_phase1_artifacts") != plan_payload["trusted_phase1_artifacts"]
        or preflight.get("unique_referenced_rank_ids") != sorted(selection.required_rank_ids)
    ):
        raise CubeExtractionError(f"published cube is not bound to the Phase 5 plan: {cube_id}")
    return {
        "selection_sha256": _mapping_sha256(row),
        "selection_cube_id": cube_id,
        "selection_L_sub": selection.lsub,
    }


def _materialization_record_path(output_root: Path, cube_id: str) -> Path:
    return output_root / MATERIALIZATION_ROOT / f"{cube_id}.json"


def _materialization_record_payload(output_root: Path, cube_id: str) -> dict[str, Any]:
    plan_identity = _plan_identity(output_root)
    cube_publication = _cube_publication_identity(output_root, cube_id)
    return {
        "schema_version": 1,
        "status": "phase5_materialized_extraction",
        "cube_id": cube_id,
        "phase5_extraction_plan": plan_identity,
        "campaign_config": json.loads((output_root / PLAN_FILENAME).read_text())["campaign_config"],
        "cube_publication": cube_publication,
        "plan_selection_binding": _require_manifest_plan_binding(
            output_root,
            cube_id,
            plan_sha256=plan_identity["plan_sha256"],
            manifest_sha256=cube_publication["manifest_sha256"],
        ),
        "plan_derived_core_extractor_provenance": _plan_manifest_core_provenance(
            output_root,
            cube_id,
            plan_sha256=plan_identity["plan_sha256"],
            manifest_sha256=cube_publication["manifest_sha256"],
        ),
    }


def _materialization_record_identity(output_root: Path, cube_id: str) -> dict[str, str]:
    path = _materialization_record_path(output_root, cube_id)
    payload = json.loads(path.read_text())
    if payload != _materialization_record_payload(output_root, cube_id):
        raise CubeExtractionError(f"invalid or stale Phase 5 materialization record: {path}")
    return {
        "materialization_record_relative_path": str(path.relative_to(output_root)),
        "materialization_record_sha256": file_sha256(path),
    }


def _publish_materialization_record(output_root: Path, cube_id: str) -> dict[str, str]:
    path = _materialization_record_path(output_root, cube_id)
    if not path.exists():
        _atomic_write_json(path, _materialization_record_payload(output_root, cube_id))
    return _materialization_record_identity(output_root, cube_id)


def _restart_record_path(output_root: Path, cube_id: str) -> Path:
    return output_root / RESTART_ROOT / f"{cube_id}.json"


def _restart_record_payload(output_root: Path, cube_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "passed",
        "cube_id": cube_id,
        "verification_contract": {
            "adapter": "verify_pilot_cube_output",
            "verify_hashes": True,
            "note": "strict source, cbin, catalog, output-array, publication, and provenance replay",
        },
        "phase5_extraction_plan": _plan_identity(output_root),
        "phase5_materialization_record": _materialization_record_identity(output_root, cube_id),
        "cube_publication": _cube_publication_identity(output_root, cube_id),
    }


def _restart_record_identity(output_root: Path, cube_id: str) -> dict[str, str]:
    path = _restart_record_path(output_root, cube_id)
    payload = json.loads(path.read_text())
    published = payload.pop("published_unix_seconds", None)
    if not isinstance(published, (float, int)) or payload != _restart_record_payload(output_root, cube_id):
        raise CubeExtractionError(f"invalid or stale Phase 5 restart record: {path}")
    return {
        "restart_record_relative_path": str(path.relative_to(output_root)),
        "restart_record_sha256": file_sha256(path),
    }


def _publish_restart_record(output_root: Path, cube_id: str) -> dict[str, str]:
    path = _restart_record_path(output_root, cube_id)
    if not path.exists():
        _atomic_write_json(
            path,
            {
                **_restart_record_payload(output_root, cube_id),
                "published_unix_seconds": time.time(),
            },
        )
    return _restart_record_identity(output_root, cube_id)


def _common_extraction_arguments(
    trusted_run: Path,
    data_root: Path,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "data_root": data_root,
        "basename": snapshot["full_resolution_basename"],
        "trusted_run": trusted_run,
        "rank_map": load_rank_map(trusted_run),
        "expected_time": snapshot["target_time"],
        "expected_cycle": snapshot["target_cycle"],
        "domain_bounds": DOMAIN_BOUNDS,
        "expected_header_identity": snapshot["full_snapshot_identity"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "extract", "verify", "inspect"))
    parser.add_argument("--trusted-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=EXTRACTABLE_SCALES, required=True)
    parser.add_argument("--subset", choices=("all", "smoke", "matched_smoke"), required=True)
    parser.add_argument("--basename")
    parser.add_argument("--clean-partial", action="store_true")
    parser.add_argument("--clean-incomplete", action="store_true")
    parser.add_argument("--clean-stale-lock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.action == "plan":
        payload = plan(
            args.trusted_run,
            args.data_root,
            args.output_root,
            args.campaign_config,
            scale=args.scale,
            subset=args.subset,
            basename=args.basename,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    plan_payload = verify_plan(
        args.trusted_run,
        args.data_root,
        args.output_root,
        args.campaign_config,
        scale=args.scale,
        subset=args.subset,
        basename=args.basename,
    )
    rows = tuple(dict(row) for row in plan_payload["selections"])
    selections = {row["cube_id"]: _selection_from_config_row(row) for row in rows}
    snapshot = _snapshot_identity(args.trusted_run.resolve(), args.basename)
    if args.action == "inspect":
        for cube_id in selections:
            materialization = _materialization_record_identity(args.output_root, cube_id)
            restart = _restart_record_identity(args.output_root, cube_id)
            output = args.output_root / "inspection" / f"{cube_id}_midplanes.png"
            write_inspection_figure(args.output_root / cube_id, output)
            print(
                json.dumps(
                    {
                        "status": "inspected",
                        "cube_id": cube_id,
                        "output": str(output),
                        "phase5_materialization_record": materialization,
                        "phase5_restart_record": restart,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return

    common = _common_extraction_arguments(args.trusted_run.resolve(), args.data_root.resolve(), snapshot)
    for cube_id, selection in selections.items():
        if args.action == "extract":
            payload = extract_cube(
                selection,
                output_root=args.output_root,
                clean_partial=args.clean_partial,
                clean_incomplete=args.clean_incomplete,
                clean_stale_lock=args.clean_stale_lock,
                verify_hashes=True,
                **common,
            )
            payload["phase5_materialization_record"] = _publish_materialization_record(
                args.output_root, cube_id
            )
        else:
            payload = verify_pilot_cube_output(
                args.output_root / cube_id,
                selection,
                verify_hashes=True,
                **common,
            )
            _publish_materialization_record(args.output_root, cube_id)
            payload["phase5_restart_record"] = _publish_restart_record(args.output_root, cube_id)
        payload["phase5_extraction_plan"] = _plan_identity(args.output_root)
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
