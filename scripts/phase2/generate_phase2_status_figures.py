#!/usr/bin/env python3
"""Generate reproducible Phase 2 status-report figures from retained outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from sfunctor.io.cube_extract import stream_validation_probe_selections, verify_trusted_run

BENCHMARK_IDS = (
    "L640_sub00370",
    "L640_sub03942",
    "L640_sub00579",
    "L640_sub00738",
)
SHORT_LABELS = ("low dBB", "median dBB", "high dBB", "weak mean field")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_manifests(benchmark_root: Path) -> list[dict]:
    return [_load_json(benchmark_root / cube_id / "manifest.json") for cube_id in BENCHMARK_IDS]


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _verify_completion_marker(benchmark_root: Path) -> dict:
    completion = _load_json(benchmark_root / "PHASE2_BENCHMARK_COMPLETE.json")
    if (
        completion.get("status") != "passed"
        or tuple(completion.get("cube_ids", ())) != BENCHMARK_IDS
        or completion.get("benchmark_summary_sha256")
        != file_sha256(benchmark_root / "phase2_benchmark_summary.json")
        or completion.get("benchmark_summary_markdown_sha256")
        != file_sha256(benchmark_root / "phase2_benchmark_summary.md")
    ):
        raise RuntimeError("benchmark summary completion marker is missing or stale")
    expected_dependencies = {
        "cube_completion_sha256": {f"{cube_id}/COMPLETE.json" for cube_id in BENCHMARK_IDS},
        "stream_validation_sha256": {
            relative
            for cube_id in stream_validation_probe_selections()
            for relative in (
                f"stream_validation/{cube_id}.json",
                f"stream_validation/{cube_id}_cbin.csv",
            )
        },
        "report_artifact_sha256": {
            relative
            for cube_id in BENCHMARK_IDS
            for relative in (
                f"{cube_id}/cbin_comparison.csv",
                f"{cube_id}/catalog_comparison.csv",
                f"restart_checks/{cube_id}.json",
                f"inspection/{cube_id}_midplanes.png",
            )
        },
    }
    for dependency_group, expected_paths in expected_dependencies.items():
        dependencies = completion.get(dependency_group, {})
        if set(dependencies) != expected_paths:
            raise RuntimeError(f"benchmark marker has an invalid {dependency_group} inventory")
        for relative_path, expected_sha256 in dependencies.items():
            if file_sha256(benchmark_root / relative_path) != expected_sha256:
                raise RuntimeError(f"benchmark dependency changed: {relative_path}")
    for cube_id in BENCHMARK_IDS:
        cube_completion = _load_json(benchmark_root / cube_id / "COMPLETE.json")
        if (
            cube_completion.get("cube_id") != cube_id
            or cube_completion.get("manifest_sha256")
            != file_sha256(benchmark_root / cube_id / "manifest.json")
        ):
            raise RuntimeError(f"cube completion marker is stale: {cube_id}")
    return completion


def workflow_schematic(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 3.2))
    axis.set_axis_off()
    boxes = (
        (0.02, "trusted Phase 1\npilot + rank map"),
        (0.22, "selected-rank\nheader preflight"),
        (0.42, "primitive shards\nminimum rank set"),
        (0.62, "sequential KJI\n.npy writer"),
        (0.82, "cbin gate + hash\nCOMPLETE.json"),
    )
    for x, label in boxes:
        axis.add_patch(
            plt.Rectangle((x, 0.36), 0.15, 0.32, facecolor="#e7f0fa", edgecolor="#2a5c8a", lw=1.5)
        )
        axis.text(x + 0.075, 0.52, label, ha="center", va="center", fontsize=10)
    for index in range(len(boxes) - 1):
        axis.annotate(
            "",
            xy=(boxes[index + 1][0] - 0.01, 0.52),
            xytext=(boxes[index][0] + 0.16, 0.52),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#444444"},
        )
    axis.text(0.5, 0.11, "Schematic: Phase 2 reads only explicitly selected full-resolution ranks", ha="center")
    _save(figure, output_dir / "phase2_extraction_workflow_schematic.png")


def pilot_context(trusted_run: Path, output_dir: Path) -> None:
    with (trusted_run / "analysis" / "pilot_sample.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([float(row["B_mean"]) for row in rows])
    y = np.asarray([float(row["dBB"]) for row in rows])
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(x, y, c="#a7b4c3", s=28, alpha=0.75, label="21-region Phase 1 proposal")
    lookup = {row["pilot_id"]: row for row in rows}
    for cube_id, label in zip(BENCHMARK_IDS, SHORT_LABELS):
        row = lookup[cube_id]
        bx, d_bb = float(row["B_mean"]), float(row["dBB"])
        axis.scatter([bx], [d_bb], s=75, marker="o", edgecolor="black", label=label)
        axis.annotate(cube_id.replace("L640_", ""), (bx, d_bb), xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel(r"$B_{\rm mean}$")
    axis.set_ylabel(r"$\mathrm{dBB} = \delta B / B_{\rm mean}$")
    axis.set_title("Four-cube benchmark spans the intended magnetic regimes")
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=0.25)
    _save(figure, output_dir / "phase2_benchmark_selection_context.png")


def resource_summary(benchmark_root: Path, output_dir: Path) -> None:
    manifests = _load_manifests(benchmark_root)
    payload = [item["performance"]["payload_wall_seconds"] for item in manifests]
    checksum = [item["performance"]["hash_wall_seconds"] for item in manifests]
    restart = [
        _load_json(benchmark_root / "restart_checks" / f"{cube_id}.json")["verify_wall_seconds"]
        for cube_id in BENCHMARK_IDS
    ]
    rss = [item["performance"]["peak_rss_kib"] / 1024**2 for item in manifests]
    allocated = [
        sum(
            (benchmark_root / item["cube_id"] / metadata["relative_path"]).stat().st_blocks * 512
            for metadata in item["output_fields"].values()
        )
        / 1024**3
        for item in manifests
    ]
    x = np.arange(len(BENCHMARK_IDS))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    axes[0].bar(x - 0.25, payload, width=0.25, label="payload assembly")
    axes[0].bar(x, checksum, width=0.25, label="initial SHA-256")
    axes[0].bar(x + 0.25, restart, width=0.25, label="strict restart verify")
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_xticks(x, SHORT_LABELS, rotation=18, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - 0.18, rss, width=0.36, label="manifest peak RSS [GiB]")
    axes[1].bar(x + 0.18, allocated, width=0.36, label="allocated output [GiB]")
    axes[1].set_xticks(x, SHORT_LABELS, rotation=18, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase2_four_cube_resource_summary.png")


def cbin_residuals(benchmark_root: Path, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for index, cube_id in enumerate(BENCHMARK_IDS):
        with (benchmark_root / cube_id / "cbin_comparison.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = []
        for row in rows:
            try:
                value = float(row["relative_difference"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(max(value, 1.0e-16))
        axis.scatter(np.full(len(values), index), values, s=14, alpha=0.55)
    axis.set_yscale("log")
    axis.set_ylabel("relative difference")
    axis.set_xticks(np.arange(len(BENCHMARK_IDS)), SHORT_LABELS, rotation=18, ha="right")
    axis.set_title("All supported direct primitive-cube versus cbin checks pass")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase2_cbin_relative_residuals.png")


def stream_validation(benchmark_root: Path, output_dir: Path) -> None:
    paths = [
        benchmark_root / "stream_validation" / f"{cube_id}.json"
        for cube_id in stream_validation_probe_selections()
    ]
    labels, max_relative, wall = [], [], []
    for path in paths:
        payload = _load_json(path)
        values = [
            float(row["relative_difference"])
            for row in payload["cbin_validation"]["rows"]
            if math.isfinite(float(row["relative_difference"]))
        ]
        labels.append(path.stem.replace("stream_", "").replace("_20260530", ""))
        max_relative.append(max(values))
        wall.append(payload["wall_seconds"])
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(x, max_relative, color="#587da5")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("maximum relative difference")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, wall, color="#8d6e63")
    axes[1].set_ylabel("stream validation wall time [s]")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase2_heldout_stream_validation.png")


def storage_layout_fix(benchmark_root: Path, rejected_storage_log: Path, output_dir: Path) -> None:
    match = re.search(r"amplification ([0-9.]+)", rejected_storage_log.read_text())
    rejected_ratio = float(match.group(1)) if match else float("nan")
    manifests = _load_manifests(benchmark_root)
    accepted = [item["performance"]["storage_amplification_ratio"] for item in manifests]
    labels = ("rejected\nstrided mmap",) + tuple(label.replace(" ", "\n") for label in SHORT_LABELS)
    values = (rejected_ratio,) + tuple(accepted)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    colors = ("#b54c4c",) + ("#4f8a5b",) * len(accepted)
    axis.bar(np.arange(len(values)), values, color=colors)
    axis.axhline(1.5, color="black", ls="--", lw=1.2, label="publication limit")
    axis.set_ylabel("allocated / apparent output bytes")
    axis.set_xticks(np.arange(len(values)), labels)
    axis.set_title("Sequential output layout reduces Lustre allocation amplification below the guard")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase2_storage_layout_fix.png")


def inspection_montage(benchmark_root: Path, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, cube_id, label in zip(axes.reshape(-1), BENCHMARK_IDS, SHORT_LABELS):
        source = benchmark_root / "inspection" / f"{cube_id}_midplanes.png"
        destination = output_dir / source.name
        shutil.copyfile(source, destination)
        axis.imshow(plt.imread(source))
        axis.set_title(f"{label}: {cube_id}")
        axis.set_axis_off()
    _save(figure, output_dir / "phase2_midplane_inspection_montage.png")


def campaign_forecast(benchmark_root: Path, output_dir: Path) -> None:
    summary = _load_json(benchmark_root / "phase2_benchmark_summary.json")
    measured = summary["measured_totals"]
    forecast = summary["forecast_campaign_totals"]
    labels = ("logical source read", "apparent output", "allocated output")
    keys = ("logical_source_bytes_touched", "output_bytes", "allocated_output_bytes")
    measured_gib = [
        measured["logical_source_bytes_touched"] / 1024**3,
        measured["output_bytes"] / 1024**3,
        summary["settled_allocated_output_bytes"] / 1024**3,
    ]
    forecast_gib = [
        forecast["logical_source_bytes_touched"] / 1024**3,
        forecast["output_bytes"] / 1024**3,
        summary["forecast_settled_allocated_output_bytes"] / 1024**3,
    ]
    x = np.arange(len(keys))
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(x - 0.18, measured_gib, width=0.36, label="four cubes measured")
    axis.bar(x + 0.18, forecast_gib, width=0.36, label="21 cubes linear forecast")
    axis.set_ylabel("GiB")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase2_campaign_storage_io_forecast.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trusted-run", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--rejected-storage-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trusted_artifacts = verify_trusted_run(args.trusted_run)
    completion = _verify_completion_marker(args.benchmark_root)
    if completion.get("trusted_phase1_artifacts") != trusted_artifacts:
        raise RuntimeError("benchmark marker does not bind the requested trusted Phase 1 run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.output_dir.glob("*.png"):
        path.unlink()
    workflow_schematic(args.output_dir)
    pilot_context(args.trusted_run, args.output_dir)
    resource_summary(args.benchmark_root, args.output_dir)
    cbin_residuals(args.benchmark_root, args.output_dir)
    stream_validation(args.benchmark_root, args.output_dir)
    storage_layout_fix(args.benchmark_root, args.rejected_storage_log, args.output_dir)
    inspection_montage(args.benchmark_root, args.output_dir)
    campaign_forecast(args.benchmark_root, args.output_dir)
    generated = sorted(path.name for path in args.output_dir.glob("*.png"))
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "generated_figures": generated,
                "generated_figure_sha256": {
                    name: file_sha256(args.output_dir / name) for name in generated
                },
                "benchmark_completion_sha256": file_sha256(
                    args.benchmark_root / "PHASE2_BENCHMARK_COMPLETE.json"
                ),
                "generator_sha256": file_sha256(Path(__file__).resolve()),
                "rejected_storage_log": str(args.rejected_storage_log),
                "rejected_storage_log_sha256": file_sha256(args.rejected_storage_log),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
