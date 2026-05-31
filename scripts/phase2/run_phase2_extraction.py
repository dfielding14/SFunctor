#!/usr/bin/env python3
"""CLI for bounded Phase 2 selected-cube preflight, extraction, and checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sfunctor.io.cube_extract import (
    BENCHMARK_CUBE_IDS,
    _write_comparison_csv,
    extract_cube,
    load_pilot_selections,
    load_rank_map,
    load_snapshot_identity,
    metadata_probe_selections,
    preflight_selection,
    stream_validate_selection,
    stream_validation_probe_selections,
    summarize_benchmark,
    verify_pilot_cube_output,
    write_inspection_figure,
)

DOMAIN_BOUNDS = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "probe",
            "stream-validate",
            "extract",
            "verify",
            "inspect",
            "summarize",
        ),
    )
    parser.add_argument("--trusted-run", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cube-id", action="append")
    parser.add_argument("--all-pilot", action="store_true")
    parser.add_argument("--basename")
    parser.add_argument("--clean-partial", action="store_true")
    parser.add_argument("--clean-incomplete", action="store_true")
    parser.add_argument("--clean-stale-lock", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def _cube_ids(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.cube_id or BENCHMARK_CUBE_IDS)


def _require_sources(args: argparse.Namespace) -> tuple[dict, dict, object]:
    if args.trusted_run is None or args.data_root is None:
        raise SystemExit("--trusted-run and --data-root are required for this action")
    selections = load_pilot_selections(args.trusted_run)
    snapshot = load_snapshot_identity(args.trusted_run)
    if args.basename:
        if args.basename != snapshot["full_resolution_basename"]:
            raise SystemExit("--basename must match the trusted Phase 1 full-resolution snapshot")
    rank_map = load_rank_map(args.trusted_run)
    return selections, snapshot, rank_map


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cube_ids = _cube_ids(args)
    if args.action in ("preflight", "probe", "stream-validate", "extract"):
        selections, snapshot, rank_map = _require_sources(args)
        if args.action == "probe":
            selections = metadata_probe_selections()
            cube_ids = tuple(selections)
        elif args.action == "stream-validate":
            selections = stream_validation_probe_selections()
            cube_ids = tuple(selections)
        elif args.all_pilot:
            if args.action == "extract":
                raise SystemExit("Phase 2 forbids extract --all-pilot; extract only explicit approved IDs")
            cube_ids = tuple(selections)
        if args.action == "extract" and any(cube_id not in BENCHMARK_CUBE_IDS for cube_id in cube_ids):
            raise SystemExit("Phase 2 extraction is restricted to the four approved benchmark IDs")
        for cube_id in cube_ids:
            selection = selections[cube_id]
            common = {
                "data_root": args.data_root,
                "basename": snapshot["full_resolution_basename"],
                "rank_map": rank_map,
                "expected_time": snapshot["target_time"],
                "expected_cycle": snapshot["target_cycle"],
                "domain_bounds": DOMAIN_BOUNDS,
                "expected_header_identity": snapshot["full_snapshot_identity"],
            }
            if args.action in ("preflight", "probe"):
                payload = preflight_selection(selection, **common)
                subdirectory = "probes" if args.action == "probe" else "preflight"
                path = args.output_root / subdirectory / f"{cube_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            elif args.action == "extract":
                payload = extract_cube(
                    selection,
                    output_root=args.output_root,
                    trusted_run=args.trusted_run,
                    clean_partial=args.clean_partial,
                    clean_incomplete=args.clean_incomplete,
                    clean_stale_lock=args.clean_stale_lock,
                    verify_hashes=not args.skip_hashes,
                    **common,
                )
            else:
                payload = stream_validate_selection(
                    selection,
                    trusted_run=args.trusted_run,
                    **common,
                )
                path = args.output_root / "stream_validation" / f"{cube_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                comparison_path = args.output_root / "stream_validation" / f"{cube_id}_cbin.csv"
                _write_comparison_csv(comparison_path, payload["cbin_validation"])
            print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.action == "verify":
        selections, snapshot, rank_map = _require_sources(args)
        for cube_id in cube_ids:
            payload = verify_pilot_cube_output(
                args.output_root / cube_id,
                selections[cube_id],
                data_root=args.data_root,
                basename=snapshot["full_resolution_basename"],
                trusted_run=args.trusted_run,
                rank_map=rank_map,
                expected_time=snapshot["target_time"],
                expected_cycle=snapshot["target_cycle"],
                domain_bounds=DOMAIN_BOUNDS,
                expected_header_identity=snapshot["full_snapshot_identity"],
                verify_hashes=not args.skip_hashes,
            )
            report = args.output_root / "restart_checks" / f"{cube_id}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.action == "inspect":
        for cube_id in cube_ids:
            output = args.output_root / "inspection" / f"{cube_id}_midplanes.png"
            write_inspection_figure(args.output_root / cube_id, output)
            print(output)
    else:
        _, snapshot, rank_map = _require_sources(args)
        print(
            json.dumps(
                summarize_benchmark(
                    args.output_root,
                    cube_ids,
                    trusted_run=args.trusted_run,
                    data_root=args.data_root,
                    basename=snapshot["full_resolution_basename"],
                    rank_map=rank_map,
                    expected_time=snapshot["target_time"],
                    expected_cycle=snapshot["target_cycle"],
                    domain_bounds=DOMAIN_BOUNDS,
                    expected_header_identity=snapshot["full_snapshot_identity"],
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
