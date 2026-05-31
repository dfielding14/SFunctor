#!/usr/bin/env python3
"""Profile the two primitive-array traversals used by the Phase 4 Batch A report."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase4 import generate_phase4_batch_a_status_figures as report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.cube_id not in report.FROZEN_PHASE4_PILOT_CUBE_IDS:
        raise RuntimeError(f"cube is not in the frozen Phase 4 pilot: {args.cube_id}")
    extraction_root = args.extraction_root.resolve()
    cube_root = report._contained_path(
        extraction_root, extraction_root / args.cube_id, label="profile cube"
    )
    manifest_path = report._relative_file(
        cube_root, "manifest.json", label="profile cube manifest"
    )
    manifest = report._load_json(manifest_path)
    if (
        manifest.get("cube_id") != args.cube_id
        or manifest.get("shape_kji") != [640, 640, 640]
        or set(manifest.get("output_fields", {})) != set(report.EXPECTED_EXTRACTION_FIELDS)
    ):
        raise RuntimeError(f"invalid retained profile-cube manifest: {manifest_path}")

    input_hashes = report.InputHashes()
    ledger_summary, _ = report._bind_ledger_summary_snapshot(
        args.ledger_summary, input_hashes
    )
    input_hashes.add(manifest_path)

    started = time.perf_counter()
    hashed_bytes = 0
    hash_started = time.perf_counter()
    for field in report.EXPECTED_EXTRACTION_FIELDS:
        metadata = manifest["output_fields"][field]
        path = report._relative_file(
            cube_root, metadata.get("relative_path"), label=f"profile {field} array"
        )
        input_hashes.bind_verified(path, str(metadata.get("sha256", "")))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            list(array.shape) != [640, 640, 640]
            or str(array.dtype) != str(np.dtype(np.float32))
            or path.stat().st_size != metadata.get("size_bytes")
        ):
            raise RuntimeError(f"profile array metadata mismatch: {path}")
        hashed_bytes += path.stat().st_size
    hash_wall_seconds = time.perf_counter() - hash_started

    primitive_started = time.perf_counter()
    primitive = report._primitive_diagnostics(
        cube_root,
        manifest,
        chunk_k=args.chunk_k,
        pressure_convention=None,
    )
    primitive_wall_seconds = time.perf_counter() - primitive_started
    total_wall_seconds = time.perf_counter() - started
    profile_cube_count = 1
    full_cube_count = len(report.FROZEN_PHASE4_PILOT_CUBE_IDS)
    return {
        "schema_version": 1,
        "status": "passed",
        "purpose": (
            "Representative-cube smoke/profile for the two extraction-array traversals "
            "used by the Phase 4 Batch A report. This is an I/O-focused arithmetic "
            "lower bound without pressure diagnostics, not a science publication."
        ),
        "cube_id": args.cube_id,
        "extraction_root": str(extraction_root),
        "ledger_summary": ledger_summary,
        "chunk_k": args.chunk_k,
        "profile_cube_count": profile_cube_count,
        "full_report_cube_count": full_cube_count,
        "hash_scan_bytes": hashed_bytes,
        "primitive_traversal_bytes": hashed_bytes,
        "minimum_profile_payload_bytes": 2 * hashed_bytes,
        "minimum_full_report_payload_bytes_linear_projection": 2
        * hashed_bytes
        * full_cube_count,
        "hash_scan_wall_seconds": hash_wall_seconds,
        "primitive_traversal_wall_seconds": primitive_wall_seconds,
        "profile_total_wall_seconds": total_wall_seconds,
        "full_report_two_traversal_wall_seconds_linear_projection": total_wall_seconds
        * full_cube_count,
        "projection_caveat": (
            "I/O-focused lower bound: pressure, sound-speed, and sonic-Mach arithmetic "
            "are omitted because this profiler passes pressure_convention=None."
        ),
        "maximum_resident_set_size_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "primitive_diagnostics": primitive,
        "input_sha256": input_hashes.as_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one retained Phase 4 cube through the report I/O traversals."
    )
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--cube-id", required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--chunk-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_k < 1:
        raise ValueError("--chunk-k must be positive")
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite immutable profile output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _profile(args)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.", dir=output_path.parent) as tmp:
        temporary = Path(tmp) / output_path.name
        _write_json(temporary, payload)
        temporary.rename(output_path)
    print(f"Wrote Phase 4 Batch A report I/O profile: {output_path}")


if __name__ == "__main__":
    main()
