#!/usr/bin/env python3
"""Run one frozen Phase 5 cross-scale sampler matrix through the Phase 3a engine."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3 import run_phase3_sampler as phase3
from scripts.phase3a import run_phase3a_sampler as inherited

PHASE5_SCALES = (80, 160, 320)
PHASE5_BIN_COUNTS = {80: 32, 160: 48, 320: 64}
PHASE5_SUPPORT_MODES = ("all_valid_origins", "shell_local")
PHASE5_Q_NAMES = ("B", "u")
PHASE5_DENSITY_CONVENTIONS = ("not applicable", "not applicable")
PHASE5_SELECTION_SETS = ("all", "smoke", "matched_smoke")
PHASE5_MATRIX_P_VALUES = {
    "baseline": (2.0,),
    "orders": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    "3point": (2.0,),
    "5point": (2.0,),
}
PHASE5_MATRIX_STENCIL_WIDTHS = {
    "baseline": 2,
    "orders": 2,
    "3point": 3,
    "5point": 5,
}
PHASE5_ALLOWED_MATRICES_BY_SCALE = {
    320: ("baseline", "orders", "3point", "5point"),
    160: ("baseline", "orders", "3point"),
    80: ("baseline", "orders"),
}
SUMMARY_FILENAME = "phase5_sampler_summary.json"
SUMMARY_MARKER_FILENAME = "PHASE5_SAMPLER_COMPLETE.json"
_INHERITED_SOURCE_VERSION = inherited._source_version
_INHERITED_PHASE2_SOURCE_IDENTITY = inherited._phase2_source_identity
_EXTRACTION_MODULE: Any | None = None
_configured_campaign_config: Path | None = None
_configured_scale: int | None = None
_configured_selection_set: str | None = None
_configured_matrix: str | None = None
_configured_cube_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Phase5Selection:
    campaign_config: Path
    scale: int
    selection_set: str
    matrix: str
    cube_ids: tuple[str, ...]


def _extraction_module() -> Any:
    """Load the independently maintained Phase 5 extraction adapter on demand."""

    global _EXTRACTION_MODULE
    if _EXTRACTION_MODULE is None:
        _EXTRACTION_MODULE = importlib.import_module("scripts.phase5.run_phase5_extraction")
    return _EXTRACTION_MODULE


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _campaign_config_identity(path: Path) -> dict[str, str]:
    return {
        "campaign_config_path": str(path.resolve()),
        "campaign_config_sha256": file_sha256(path),
    }


def _extraction_campaign_config_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def _row_scale(row: Mapping[str, Any]) -> int | None:
    value = row.get("scale", row.get("L_sub"))
    return int(value) if value is not None else None


def _cube_ids_from_entry(entry: Any, *, scale: int, selection_set: str) -> tuple[str, ...] | None:
    """Read common approved-campaign selection layouts without changing their order."""

    if isinstance(entry, Mapping):
        entry_scale = _row_scale(entry)
        if entry_scale is not None and entry_scale != scale:
            return None
        for key in ("scales", "by_scale"):
            scales = entry.get(key)
            if isinstance(scales, Mapping):
                child = scales.get(str(scale), scales.get(scale))
                if child is not None:
                    return _cube_ids_from_entry(
                        child, scale=scale, selection_set=selection_set
                    )
        child = entry.get(str(scale), entry.get(scale))
        if child is not None:
            return _cube_ids_from_entry(child, scale=scale, selection_set=selection_set)
        for key in ("cube_ids", "cubes", "selections"):
            if key in entry:
                return _cube_ids_from_entry(
                    entry[key], scale=scale, selection_set=selection_set
                )
        cube_id = entry.get("cube_id")
        return (str(cube_id),) if isinstance(cube_id, str) and cube_id else None
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        cube_ids = []
        for item in entry:
            if isinstance(item, str):
                cube_ids.append(item)
                continue
            if not isinstance(item, Mapping):
                raise ValueError(f"invalid cube row in selection set {selection_set!r}")
            item_selection_set = item.get("selection_set", item.get("selection_set_id"))
            if item_selection_set is not None and str(item_selection_set) != selection_set:
                continue
            item_scale = _row_scale(item)
            if item_scale is not None and item_scale != scale:
                continue
            cube_id = item.get("cube_id")
            if not isinstance(cube_id, str) or not cube_id:
                raise ValueError(f"invalid cube row in selection set {selection_set!r}")
            cube_ids.append(cube_id)
        return tuple(cube_ids) if cube_ids else None
    return None


def _explicit_selection_set_entry(
    payload: Mapping[str, Any],
    *,
    scale: int,
    selection_set: str,
) -> Any | None:
    """Return one explicitly published per-scale selection set when present."""

    named_scale_entry = payload.get(f"{selection_set}_L{scale}_selections")
    if named_scale_entry is not None:
        return named_scale_entry
    for key in ("selection_sets_by_scale", "selection_sets"):
        selection_sets = payload.get(key)
        if not isinstance(selection_sets, Mapping):
            continue
        scale_sets = selection_sets.get(str(scale), selection_sets.get(scale))
        if isinstance(scale_sets, Mapping) and selection_set in scale_sets:
            return scale_sets[selection_set]
        selected = selection_sets.get(selection_set)
        if isinstance(selected, Mapping):
            scale_entry = selected.get(str(scale), selected.get(scale))
            if scale_entry is not None:
                return scale_entry
    selections_by_scale = payload.get("selections_by_scale")
    if isinstance(selections_by_scale, Mapping):
        scale_entry = selections_by_scale.get(str(scale), selections_by_scale.get(scale))
        if isinstance(scale_entry, Mapping):
            selection_sets = scale_entry.get("selection_sets", scale_entry)
            if isinstance(selection_sets, Mapping) and selection_set in selection_sets:
                return selection_sets[selection_set]
    scales = payload.get("scales")
    if isinstance(scales, Mapping):
        scale_entry = scales.get(str(scale), scales.get(scale))
        if isinstance(scale_entry, Mapping):
            selection_sets = scale_entry.get("selection_sets", scale_entry)
            if isinstance(selection_sets, Mapping) and selection_set in selection_sets:
                return selection_sets[selection_set]
    return None


def _selection_cube_ids(
    payload: Mapping[str, Any],
    *,
    scale: int,
    selection_set: str,
) -> tuple[str, ...]:
    if selection_set not in PHASE5_SELECTION_SETS:
        raise ValueError(f"canonical Phase 5 selection sets are {PHASE5_SELECTION_SETS}")
    explicit = _explicit_selection_set_entry(
        payload,
        scale=scale,
        selection_set=selection_set,
    )
    if explicit is not None:
        cube_ids = _cube_ids_from_entry(
            explicit,
            scale=scale,
            selection_set=selection_set,
        )
        if not cube_ids:
            raise ValueError(
                f"campaign config has no non-empty {selection_set!r} selection set for L{scale}"
            )
        if len(cube_ids) != len(set(cube_ids)):
            raise ValueError(f"duplicate cube IDs in selection set {selection_set!r}")
        return cube_ids
    selections_by_scale = payload.get("selections_by_scale")
    if isinstance(selections_by_scale, Mapping):
        rows = selections_by_scale.get(str(scale))
        if not isinstance(rows, list):
            raise ValueError(f"campaign config lacks L{scale} selections")
        if selection_set == "matched_smoke":
            raise ValueError(
                f"campaign config does not define {selection_set!r} for L{scale}"
            )
        selected = [
            row
            for row in rows
            if selection_set == "all"
            or isinstance(row, Mapping)
            and row.get("smoke_anchor") is True
        ]
        cube_ids = _cube_ids_from_entry(
            selected,
            scale=scale,
            selection_set=selection_set,
        )
        if not cube_ids:
            raise ValueError(
                f"campaign config has no non-empty {selection_set!r} selection set for L{scale}"
            )
        if len(cube_ids) != len(set(cube_ids)):
            raise ValueError(f"duplicate cube IDs in selection set {selection_set!r}")
        return cube_ids
    selection_sets = payload.get("selection_sets")
    candidates: list[Any] = []
    if isinstance(selection_sets, Mapping):
        if selection_set in selection_sets:
            candidates.append(selection_sets[selection_set])
        scale_sets = selection_sets.get(str(scale), selection_sets.get(scale))
        if isinstance(scale_sets, Mapping) and selection_set in scale_sets:
            candidates.append(scale_sets[selection_set])
    elif isinstance(selection_sets, Sequence) and not isinstance(selection_sets, (str, bytes)):
        for row in selection_sets:
            if not isinstance(row, Mapping):
                raise ValueError("campaign selection_sets rows must be JSON objects")
            name = row.get("selection_set", row.get("selection_set_id", row.get("name")))
            if name is not None and str(name) == selection_set:
                candidates.append(row)
    scales = payload.get("scales")
    if isinstance(scales, Mapping):
        scale_payload = scales.get(str(scale), scales.get(scale))
        if isinstance(scale_payload, Mapping):
            nested = scale_payload.get("selection_sets", scale_payload)
            if isinstance(nested, Mapping) and selection_set in nested:
                candidates.append(nested[selection_set])
    if "selections" in payload:
        candidates.append(payload["selections"])
    for candidate in candidates:
        cube_ids = _cube_ids_from_entry(
            candidate,
            scale=scale,
            selection_set=selection_set,
        )
        if cube_ids:
            if len(cube_ids) != len(set(cube_ids)):
                raise ValueError(f"duplicate cube IDs in selection set {selection_set!r}")
            return cube_ids
    raise ValueError(
        f"campaign config has no non-empty {selection_set!r} selection set for L{scale}"
    )


def _require_frozen_quantity_policy(payload: Mapping[str, Any]) -> None:
    q_names = payload.get("q_names")
    density_conventions = payload.get("density_conventions")
    if not isinstance(q_names, list) or tuple(q_names) != PHASE5_Q_NAMES:
        raise ValueError("Phase 5 sampler is restricted to B and u")
    if not isinstance(density_conventions, list) or tuple(density_conventions) != PHASE5_DENSITY_CONVENTIONS:
        raise ValueError("Phase 5 sampler density conventions must be not applicable")
    if payload.get("sgs_channels_authorized") is not False:
        raise ValueError("Phase 5 sampler does not permit SGS channels")


def _resolve_selection(
    campaign_config: Path,
    *,
    scale: int,
    selection_set: str,
    matrix: str,
) -> Phase5Selection:
    campaign_config = campaign_config.resolve()
    if scale not in PHASE5_SCALES:
        raise ValueError(f"--scale must be one of {PHASE5_SCALES}")
    if matrix not in PHASE5_MATRIX_P_VALUES:
        raise ValueError(f"--matrix must be one of {tuple(PHASE5_MATRIX_P_VALUES)}")
    if matrix not in PHASE5_ALLOWED_MATRICES_BY_SCALE[scale]:
        raise ValueError(f"--matrix {matrix} is not supported for L{scale}")
    if not selection_set:
        raise ValueError("--selection-set must be non-empty")
    payload = _load_json(campaign_config)
    _require_frozen_quantity_policy(payload)
    return Phase5Selection(
        campaign_config=campaign_config,
        scale=scale,
        selection_set=selection_set,
        matrix=matrix,
        cube_ids=_selection_cube_ids(payload, scale=scale, selection_set=selection_set),
    )


def _stencil_specs(scale: int, matrix: str) -> dict[int, dict[str, Any]]:
    if scale not in PHASE5_ALLOWED_MATRICES_BY_SCALE:
        raise ValueError(f"--scale must be one of {PHASE5_SCALES}")
    if matrix not in PHASE5_MATRIX_STENCIL_WIDTHS:
        raise ValueError(f"--matrix must be one of {tuple(PHASE5_MATRIX_P_VALUES)}")
    if matrix not in PHASE5_ALLOWED_MATRICES_BY_SCALE[scale]:
        raise ValueError(f"--matrix {matrix} is not supported for L{scale}")
    width = PHASE5_MATRIX_STENCIL_WIDTHS[matrix]
    ell_max = {2: scale // 2, 3: scale // 4, 5: scale // 8}[width]
    return {
        width: {
            "label": f"{width}-point",
            "ell_max": ell_max,
            "bin_count": min(PHASE5_BIN_COUNTS[scale], ell_max),
            "directions_per_bin": 24,
        }
    }


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if _configured_campaign_config is None:
        raise RuntimeError("Phase 5 campaign config is not configured")
    payload = _INHERITED_SOURCE_VERSION()
    hashes = dict(payload["implementation_source_hashes"])
    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "phase5" / "run_phase5_extraction.py",
        root / "job_scripts" / "phase5" / "run_phase5_sampler_andes.sh",
    )
    hashes.update({str(path.relative_to(root)): file_sha256(path) for path in paths})
    config_identity = _campaign_config_identity(_configured_campaign_config)
    hashes[f"phase5_campaign_config:{config_identity['campaign_config_path']}"] = (
        config_identity["campaign_config_sha256"]
    )
    return {
        **payload,
        "implementation_source_hashes": hashes,
        "implementation_sha256": inherited._mapping_sha256(hashes),
    }


def _phase5_extraction_plan_identity(phase2_root: Path) -> dict[str, str]:
    if (
        _configured_campaign_config is None
        or _configured_scale is None
        or _configured_selection_set is None
    ):
        raise RuntimeError("Phase 5 extraction scope is not configured")
    extraction = _extraction_module()
    identity = extraction._plan_identity(phase2_root)
    if not isinstance(identity, dict):
        raise RuntimeError("invalid Phase 5 extraction plan identity")
    plan = _load_json(phase2_root / identity["plan_relative_path"])
    scope = plan.get("scope")
    if (
        plan.get("phase") != "phase5_cross_scale_selected_cube_extraction"
        or plan.get("status") != "planned"
        or plan.get("campaign_config")
        != _extraction_campaign_config_identity(_configured_campaign_config)
        or not isinstance(scope, Mapping)
        or scope.get("L_sub") != _configured_scale
        or scope.get("subset") != _configured_selection_set
        or tuple(scope.get("cube_ids", ())) != _configured_cube_ids
        or scope.get("cube_count") != len(_configured_cube_ids)
    ):
        raise RuntimeError("invalid Phase 5 extraction plan scope binding")
    return identity


def _restart_record_identity(
    phase2_root: Path,
    cube_id: str,
) -> dict[str, str]:
    extraction = _extraction_module()
    identity = extraction._restart_record_identity(phase2_root, cube_id)
    if not isinstance(identity, dict):
        raise RuntimeError("invalid Phase 5 restart-record identity")
    return identity


def _phase2_source_identity(
    phase2_root: Path,
    cube_id: str,
    *,
    verify_arrays: bool = True,
) -> dict[str, Any]:
    if cube_id not in _configured_cube_ids:
        raise ValueError(f"cube is outside the configured Phase 5 selection set: {cube_id}")
    if _configured_campaign_config is None:
        raise RuntimeError("Phase 5 campaign config is not configured")
    extraction = _extraction_module()
    plan_identity = _phase5_extraction_plan_identity(phase2_root)
    materialization_identity = extraction._materialization_record_identity(
        phase2_root, cube_id
    )
    restart_identity = _restart_record_identity(phase2_root, cube_id)
    return {
        **_INHERITED_PHASE2_SOURCE_IDENTITY(
            phase2_root,
            cube_id,
            verify_arrays=verify_arrays,
        ),
        "phase5_sampler_campaign_config": _campaign_config_identity(
            _configured_campaign_config
        ),
        "phase5_extraction_plan": plan_identity,
        "phase5_materialization_record": materialization_identity,
        "phase5_restart_record": restart_identity,
    }


def _load_cube(phase2_root: Path, cube_id: str) -> dict[str, np.ndarray]:
    """Load one exact-scale mmap cube through its required manifest bindings."""

    if _configured_scale is None:
        raise RuntimeError("Phase 5 scale is not configured")
    cube_root = phase2_root / cube_id
    manifest = _load_json(cube_root / "manifest.json")
    output_fields = manifest.get("output_fields")
    if not isinstance(output_fields, Mapping):
        raise RuntimeError(f"Phase 5 cube manifest has no output-field bindings: {cube_id}")
    paths = {}
    for logical_name, relative_path in phase3.FIELD_PATHS.items():
        manifest_name = phase3.MANIFEST_FIELD_NAMES[logical_name]
        metadata = output_fields.get(manifest_name)
        expected_relative_path = f"fields/{relative_path}"
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("relative_path") != expected_relative_path
            or not isinstance(metadata.get("sha256"), str)
        ):
            raise RuntimeError(
                f"Phase 5 cube manifest lost required analysis binding "
                f"{manifest_name}={expected_relative_path}: {cube_id}"
            )
        paths[logical_name] = cube_root / expected_relative_path
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    expected = (_configured_scale,) * 3
    shapes = {array.shape for array in arrays.values()}
    if shapes != {expected}:
        raise RuntimeError(f"Phase 5 expected one {expected} KJI cube for {cube_id}, got {sorted(shapes)}")
    return arrays


@contextmanager
def _configured_runner(
    campaign_config: Path,
    *,
    scale: int,
    selection_set: str,
    matrix: str,
) -> Iterator[Phase5Selection]:
    """Install Phase 5 constants for one inherited call and always restore them."""

    global _configured_campaign_config, _configured_scale, _configured_selection_set
    global _configured_matrix, _configured_cube_ids
    selection = _resolve_selection(
        campaign_config,
        scale=scale,
        selection_set=selection_set,
        matrix=matrix,
    )
    stencils = _stencil_specs(scale, matrix)
    block_shape = (scale // 8,) * 3
    original = {
        "phase3_cube_ids": phase3.BENCHMARK_CUBE_IDS,
        "cube_ids": inherited.BENCHMARK_CUBE_IDS,
        "q_names": inherited.Q_NAMES,
        "p_values": inherited.P_VALUES,
        "density_conventions": inherited.DENSITY_CONVENTIONS,
        "stencils": inherited.STENCIL_SPECS,
        "support_modes": inherited.SUPPORT_MODES,
        "diagnostic_support_modes": inherited.DIAGNOSTIC_SUPPORT_MODES,
        "block_shape": inherited.PRODUCTION_BLOCK_SHAPE_KJI,
        "source_version": inherited._source_version,
        "phase2_source_identity": inherited._phase2_source_identity,
        "load_cube": inherited._load_cube,
        "campaign_config": _configured_campaign_config,
        "scale": _configured_scale,
        "selection_set": _configured_selection_set,
        "matrix": _configured_matrix,
        "configured_cube_ids": _configured_cube_ids,
    }
    phase3.BENCHMARK_CUBE_IDS = selection.cube_ids
    inherited.BENCHMARK_CUBE_IDS = selection.cube_ids
    inherited.Q_NAMES = PHASE5_Q_NAMES
    inherited.P_VALUES = PHASE5_MATRIX_P_VALUES[matrix]
    inherited.DENSITY_CONVENTIONS = PHASE5_DENSITY_CONVENTIONS
    inherited.STENCIL_SPECS = stencils
    inherited.SUPPORT_MODES = PHASE5_SUPPORT_MODES
    inherited.DIAGNOSTIC_SUPPORT_MODES = PHASE5_SUPPORT_MODES
    inherited.PRODUCTION_BLOCK_SHAPE_KJI = block_shape
    inherited._source_version = _source_version
    inherited._phase2_source_identity = _phase2_source_identity
    inherited._load_cube = _load_cube
    _configured_campaign_config = selection.campaign_config
    _configured_scale = scale
    _configured_selection_set = selection_set
    _configured_matrix = matrix
    _configured_cube_ids = selection.cube_ids
    try:
        configuration = inherited._campaign_configuration()
        if (
            configuration["q_names"] != PHASE5_Q_NAMES
            or configuration["p_values"] != PHASE5_MATRIX_P_VALUES[matrix]
            or configuration["density_conventions"] != PHASE5_DENSITY_CONVENTIONS
            or configuration["stencils"] != stencils
            or configuration["support_modes"] != PHASE5_SUPPORT_MODES
            or configuration["block_shape_kji"] != block_shape
            or tuple(inherited.BENCHMARK_CUBE_IDS) != selection.cube_ids
        ):
            raise RuntimeError("inherited sampler no longer matches the frozen Phase 5 matrix")
        yield selection
    finally:
        phase3.BENCHMARK_CUBE_IDS = original["phase3_cube_ids"]
        inherited.BENCHMARK_CUBE_IDS = original["cube_ids"]
        inherited.Q_NAMES = original["q_names"]
        inherited.P_VALUES = original["p_values"]
        inherited.DENSITY_CONVENTIONS = original["density_conventions"]
        inherited.STENCIL_SPECS = original["stencils"]
        inherited.SUPPORT_MODES = original["support_modes"]
        inherited.DIAGNOSTIC_SUPPORT_MODES = original["diagnostic_support_modes"]
        inherited.PRODUCTION_BLOCK_SHAPE_KJI = original["block_shape"]
        inherited._source_version = original["source_version"]
        inherited._phase2_source_identity = original["phase2_source_identity"]
        inherited._load_cube = original["load_cube"]
        _configured_campaign_config = original["campaign_config"]
        _configured_scale = original["scale"]
        _configured_selection_set = original["selection_set"]
        _configured_matrix = original["matrix"]
        _configured_cube_ids = original["configured_cube_ids"]


def _summary_context(selection: Phase5Selection) -> dict[str, Any]:
    return {
        "phase": "phase5_cross_scale_sampler",
        "phase5_sampler_campaign_config": _campaign_config_identity(
            selection.campaign_config
        ),
        "scale": selection.scale,
        "selection_set": selection.selection_set,
        "matrix": selection.matrix,
        "cube_ids": selection.cube_ids,
    }


def _verify_summary_marker(output_root: Path, selection: Phase5Selection) -> dict[str, Any]:
    marker_path = output_root / SUMMARY_MARKER_FILENAME
    summary_path = output_root / SUMMARY_FILENAME
    if marker_path.exists() != summary_path.exists():
        raise RuntimeError("orphaned Phase 5 sampler summary publication")
    if not marker_path.exists():
        return {"phase5_sampler_summary_status": "not published"}
    marker = _load_json(marker_path)
    summary = _load_json(summary_path)
    expected_context = _summary_context(selection)
    if (
        marker.get("schema_version") != inherited.SCHEMA_VERSION
        or marker.get("status") != "release_aggregation_complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or marker.get("implementation_sha256") != _source_version()["implementation_sha256"]
        or summary.get("source_version", {}).get("implementation_sha256")
        != marker["implementation_sha256"]
        or any(summary.get(key) != inherited._json_builtin(value) for key, value in expected_context.items())
    ):
        raise RuntimeError("invalid or stale Phase 5 sampler summary marker")
    return {"phase5_sampler_summary_status": "passed"}


def _summarize(
    phase2_root: Path,
    output_root: Path,
    selection: Phase5Selection,
) -> dict[str, Any]:
    payload = {
        **inherited.summarize(phase2_root, output_root),
        **_summary_context(selection),
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
    _verify_summary_marker(output_root, selection)
    return payload


def _verify(
    phase2_root: Path,
    output_root: Path,
    selection: Phase5Selection,
) -> dict[str, Any]:
    return {
        **inherited.verify(phase2_root, output_root),
        **_verify_summary_marker(output_root, selection),
    }


def run(
    action: str,
    campaign_config: Path,
    scale: int,
    selection_set: str,
    matrix: str,
    phase2_root: Path,
    output_root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    with _configured_runner(
        campaign_config,
        scale=scale,
        selection_set=selection_set,
        matrix=matrix,
    ) as selection:
        actions = {
            "plan": lambda: inherited.plan(phase2_root, output_root),
            "work": lambda: inherited.work(phase2_root, output_root, workers=workers),
            "reduce": lambda: inherited.reduce(phase2_root, output_root),
            "verify": lambda: _verify(phase2_root, output_root, selection),
            "summarize": lambda: _summarize(phase2_root, output_root, selection),
        }
        return actions[action]()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "work", "reduce", "verify", "summarize"))
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=PHASE5_SCALES, required=True)
    parser.add_argument("--selection-set", required=True)
    parser.add_argument("--matrix", choices=tuple(PHASE5_MATRIX_P_VALUES), required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
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
                    args.campaign_config,
                    args.scale,
                    args.selection_set,
                    args.matrix,
                    args.phase2_root,
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
