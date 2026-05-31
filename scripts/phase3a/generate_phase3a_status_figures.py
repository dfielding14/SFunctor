#!/usr/bin/env python3
"""Generate reproducible PHASE3A_STATUS_UPDATE figures from retained artifacts.

The helper deliberately contains no synthetic science data.  Its two
schematic figures are labeled as schematics; every quantitative figure is
derived from hash-verified Phase 2 or Phase 3a publications.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3.run_phase3_sampler import BENCHMARK_CUBE_IDS
from scripts.phase3a import run_phase3a_sampler as runner
from sfunctor.analysis.phase3a import load_finite_domain_partial_npz


DEFAULT_OUTPUT_DIR = Path("figures/phase3a_status_update")
SHORT_LABELS = {
    "L640_sub00370": "low dBB",
    "L640_sub03942": "median dBB",
    "L640_sub00579": "high dBB",
    "L640_sub00738": "weak mean field",
}
STENCIL_LABELS = {2: "2-point", 3: "3-point", 5: "5-point"}
STENCIL_COLORS = {2: "#4c78a8", 3: "#f58518", 5: "#54a24b"}
DIRECTION_COLORS = {
    "parallel": "#4c78a8",
    "xi": "#f58518",
    "lambda": "#54a24b",
}
MODE_COLORS = {
    "shell_local": "#4c78a8",
    "all_valid_origins": "#54a24b",
    "nested_core": "#e45756",
}
REPRESENTATIVE_CUBE_ID = BENCHMARK_CUBE_IDS[0]
PRIMARY_STENCIL_WIDTH = 2
PRIMARY_SUPPORT_MODE = "shell_local"


@dataclass(frozen=True)
class VerifiedInputs:
    """Required retained inputs after strict runner-level verification."""

    phase2_root: Path
    controls_root: Path
    convergence_root: Path
    release_root: Path
    campaign: dict[str, Any]
    release_summary: dict[str, Any]
    controls: dict[str, Any]
    convergence: dict[str, Any]
    implementation_sha256: str


@dataclass(frozen=True)
class ReleaseGroup:
    """One verified release reduction and its block-uncertainty publication."""

    group_id: str
    result: Any
    uncertainty: dict[str, np.ndarray]


@dataclass(frozen=True)
class NodeProfile:
    """Measured task-local wall-time evidence for one retained node profile."""

    root: Path
    profile_kind: str
    node_count: int
    workers_per_node: int
    wall_seconds: float
    task_local_node_hour_proxy: float
    resource_records: tuple[Path, ...]
    workload_fingerprint: str


class InputHashes:
    """Collect hashes for every retained file used directly by this helper."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}

    def add(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            raise RuntimeError(f"required retained artifact is missing: {path}")
        self._hashes[str(path)] = file_sha256(path)

    def add_many(self, paths: Iterable[Path]) -> None:
        for path in paths:
            self.add(path)

    def as_dict(self) -> dict[str, str]:
        return dict(sorted(self._hashes.items()))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required retained JSON artifact is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"retained JSON artifact must contain an object: {path}")
    return payload


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if np.all(edges > 0.0):
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def _require_same_source(
    payload: Mapping[str, Any],
    implementation_sha256: str,
    *,
    label: str,
) -> None:
    observed = payload.get("source_version", {}).get("implementation_sha256")
    if observed != implementation_sha256:
        raise RuntimeError(
            f"{label} was produced by a different implementation: "
            f"expected {implementation_sha256}, observed {observed}"
        )


def _verify_release(
    phase2_root: Path,
    release_root: Path,
    input_hashes: InputHashes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete release chain, including its report-level marker."""

    campaign_path = release_root / "manifests" / "campaign.json"
    shards_path = release_root / "manifests" / "shards.json"
    summary_path = release_root / "phase3a_summary.json"
    completion_path = release_root / "PHASE3A_RELEASE_COMPLETE.json"
    required = (
        release_root / "PLAN_COMPLETE.json",
        campaign_path,
        shards_path,
        summary_path,
        completion_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "required Phase 3a release artifacts are missing: "
            + ", ".join(str(path) for path in missing)
        )
    try:
        verification = runner.verify(phase2_root, release_root)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"required Phase 3a release artifact chain is incomplete: {error.filename}"
        ) from error
    if verification.get("status") != "passed":
        raise RuntimeError("Phase 3a release verification did not pass")
    campaign = _load_json(campaign_path)
    summary = _load_json(summary_path)
    completion = _load_json(completion_path)
    implementation_sha256 = campaign.get("source_version", {}).get(
        "implementation_sha256"
    )
    if not isinstance(implementation_sha256, str) or not implementation_sha256:
        raise RuntimeError("Phase 3a campaign has no implementation source binding")
    if (
        completion.get("schema_version") != runner.SCHEMA_VERSION
        or completion.get("status") != "release_aggregation_complete"
        or completion.get("summary_sha256") != file_sha256(summary_path)
        or completion.get("implementation_sha256") != implementation_sha256
        or summary.get("schema_version") != runner.SCHEMA_VERSION
        or summary.get("operational_status") != "release_aggregation_complete"
        or summary.get("scientific_acceptance") != "pending_report_level_gate"
        or summary.get("verification") != verification
        or summary.get("source_version", {}).get("implementation_sha256")
        != implementation_sha256
    ):
        raise RuntimeError("Phase 3a release completion marker is missing or stale")
    summary_rows = list(summary.get("groups", ()))
    expected_group_ids = sorted(runner._group_rows(release_root))
    if (
        len(summary_rows) != len(expected_group_ids)
        or sorted(row.get("group_id") for row in summary_rows) != expected_group_ids
    ):
        raise RuntimeError("Phase 3a release summary has an invalid group inventory")
    for row in summary_rows:
        group_id = str(row["group_id"])
        result = load_finite_domain_partial_npz(
            runner._reduction_paths(release_root, group_id)[1]
        )
        expected = runner._json_builtin(
            {
                "group_id": group_id,
                "stencil_width": result.stencil_width,
                "support_mode": result.pair_mode,
                "offset_count": len(result.displacements_ijk),
                "sampled_origins": int(result.sampled_pairs.sum()),
                "minimum_valid_origin_fraction": float(
                    np.min(result.eligible_pairs / np.maximum(result.cube_candidate_pairs, 1))
                ),
                "elapsed_seconds_sum": result.elapsed_seconds,
                "elapsed_seconds_per_ell_bin": result.elapsed_seconds_per_ell_bin,
            }
        )
        if row != expected:
            raise RuntimeError(f"Phase 3a release summary row is stale: {group_id}")
    input_hashes.add_many(
        (
            release_root / "PLAN_COMPLETE.json",
            campaign_path,
            shards_path,
            summary_path,
            completion_path,
        )
    )
    for stencil_width in sorted(runner.STENCIL_SPECS):
        json_path, npz_path = runner._manifest_paths(release_root, stencil_width)
        input_hashes.add_many((json_path, npz_path))
    return campaign, summary


def _verify_required_diagnostic(
    phase2_root: Path,
    diagnostic_root: Path,
    *,
    filename: str,
    marker_filename: str,
    implementation_sha256: str,
    phase2_source: Mapping[str, Any],
    input_hashes: InputHashes,
) -> dict[str, Any]:
    missing = [
        diagnostic_root / relative
        for relative in (filename, marker_filename)
        if not (diagnostic_root / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            "required Phase 3a diagnostic artifacts are missing: "
            + ", ".join(str(path) for path in missing)
        )
    payload = runner._verify_json_diagnostic(
        diagnostic_root,
        filename,
        marker_filename,
        phase2_root=phase2_root,
    )
    _require_same_source(payload, implementation_sha256, label=filename)
    if payload.get("phase2_source") != phase2_source:
        raise RuntimeError(f"{filename} is bound to a different Phase 2 source")
    input_hashes.add_many((diagnostic_root / marker_filename, diagnostic_root / filename))
    for row in payload.get("rows", ()):
        relative = row.get("artifact_relative_path")
        if relative is not None:
            input_hashes.add(diagnostic_root / str(relative))
    for binding in payload.get("artifact_bindings", ()):
        input_hashes.add(diagnostic_root / str(binding["relative_path"]))
    return payload


def verify_inputs(
    *,
    phase2_root: Path,
    controls_root: Path,
    convergence_root: Path,
    release_root: Path,
    input_hashes: InputHashes,
) -> VerifiedInputs:
    """Verify required roots, checksums, and cross-root source consistency."""

    campaign, release_summary = _verify_release(phase2_root, release_root, input_hashes)
    implementation_sha256 = campaign["source_version"]["implementation_sha256"]
    phase2_source = campaign["phase2_sources"][REPRESENTATIVE_CUBE_ID]
    controls = _verify_required_diagnostic(
        phase2_root,
        controls_root,
        filename="controls.json",
        marker_filename="CONTROLS_COMPLETE.json",
        implementation_sha256=implementation_sha256,
        phase2_source=phase2_source,
        input_hashes=input_hashes,
    )
    if controls.get("validation_status") != "passed":
        raise RuntimeError("Phase 3a controls publication did not pass validation")
    convergence = _verify_required_diagnostic(
        phase2_root,
        convergence_root,
        filename="convergence.json",
        marker_filename="CONVERGENCE_COMPLETE.json",
        implementation_sha256=implementation_sha256,
        phase2_source=phase2_source,
        input_hashes=input_hashes,
    )
    return VerifiedInputs(
        phase2_root=phase2_root,
        controls_root=controls_root,
        convergence_root=convergence_root,
        release_root=release_root,
        campaign=campaign,
        release_summary=release_summary,
        controls=controls,
        convergence=convergence,
        implementation_sha256=implementation_sha256,
    )


def _group_id(cube_id: str, stencil_width: int, support_mode: str) -> str:
    return f"{cube_id}/stencil_{stencil_width}point/{support_mode}"


def _load_release_group(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    *,
    cube_id: str,
    stencil_width: int,
    support_mode: str,
) -> ReleaseGroup:
    group_id = _group_id(cube_id, stencil_width, support_mode)
    root, result_path, uncertainty_path, marker_path = runner._reduction_paths(
        verified.release_root, group_id
    )
    if not marker_path.is_file():
        raise RuntimeError(f"required release reduction is missing: {group_id}")
    result = load_finite_domain_partial_npz(result_path)
    with np.load(uncertainty_path, allow_pickle=False) as payload:
        uncertainty = {name: payload[name].copy() for name in payload.files}
    if result.stencil_width != stencil_width or result.pair_mode != support_mode:
        raise RuntimeError(f"release reduction metadata mismatch: {group_id}")
    input_hashes.add_many((marker_path, root / "reduction_manifest.json", result_path, uncertainty_path))
    return ReleaseGroup(group_id=group_id, result=result, uncertainty=uncertainty)


def _moment_index(
    result: Any,
    q_name: str,
    direction_name: str,
    *,
    geometry_name: str = "pair_local",
    measurement_name: str = "perpendicular",
    p_value: float = 2.0,
) -> tuple[int, int, int, int, int]:
    return (
        result.q_names.index(q_name),
        result.geometry_names.index(geometry_name),
        result.measurement_names.index(measurement_name),
        result.direction_names.index(direction_name),
        result.p_values.index(p_value),
    )


def _plot_positive_curve(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str | None = None,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    axis.plot(
        x[valid],
        y[valid],
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3.0,
        linewidth=1.5,
    )


def _plot_positive_band(
    axis: plt.Axes,
    x: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    color: str,
) -> None:
    valid = (
        np.isfinite(x)
        & np.isfinite(low)
        & np.isfinite(high)
        & (x > 0.0)
        & (low > 0.0)
        & (high > 0.0)
    )
    axis.fill_between(x[valid], low[valid], high[valid], color=color, alpha=0.14)


def workflow_schematic(output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(12.0, 3.4))
    axis.set_axis_off()
    boxes = (
        (0.01, "verified Phase 2\n$640^3$ cubes"),
        (0.18, "frozen signed\n3-D offsets"),
        (0.35, "restartable\nfixed shards"),
        (0.52, "node-local fork\nworkers"),
        (0.69, "strict reduction\n+ block bands"),
        (0.86, "source-bound\nreport figures"),
    )
    for x, label in boxes:
        axis.add_patch(
            plt.Rectangle(
                (x, 0.35),
                0.13,
                0.34,
                facecolor="#e7f0fa",
                edgecolor="#2a5c8a",
                linewidth=1.4,
            )
        )
        axis.text(x + 0.065, 0.52, label, ha="center", va="center", fontsize=10)
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(right[0] - 0.01, 0.52),
            xytext=(left[0] + 0.14, 0.52),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444444"},
        )
    axis.text(
        0.5,
        0.12,
        "Schematic: Phase 3a finite-domain publication workflow; quantitative timings are plotted separately",
        ha="center",
        fontsize=10,
    )
    return _save(figure, output_dir, "phase3a_workflow_schematic.png")


def stencil_schematic(output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), constrained_layout=True)
    rows = (
        (2, (0, 1), ("-1", "+1"), ("1", "1"), "1", "2", r"$\ell_{\max}=320$"),
        (3, (-1, 0, 1), ("+1", "-2", "+1"), ("1", "1", "1"), r"$\sqrt{3}$", "3", r"$\ell_{\max}=160$"),
        (5, (-2, -1, 0, 1, 2), ("+1", "-4", "+6", "-4", "+1"), ("1", "4", "6", "4", "1"), r"$\sqrt{35}$", "16", r"$\ell_{\max}=80$"),
    )
    for axis, (width, positions, increment, local_b, delta_denominator, b_denominator, ell_label) in zip(axes, rows):
        axis.axhline(0.0, color="#555555", linewidth=1.0)
        for position, delta_weight, b_weight in zip(positions, increment, local_b):
            axis.plot(position, 0.0, "o", color=STENCIL_COLORS[width], markersize=9)
            axis.text(position, 0.20, rf"$\Delta q$: {delta_weight}", ha="center", fontsize=9)
            axis.text(position, -0.23, rf"$B_{{loc}}$: {b_weight}", ha="center", fontsize=9)
        axis.set_xlim(-2.55, 2.55)
        axis.set_ylim(-0.55, 0.55)
        axis.set_xticks((-2, -1, 0, 1, 2), (r"$x-2r$", r"$x-r$", r"$x$", r"$x+r$", r"$x+2r$"))
        axis.set_yticks(())
        axis.set_title(f"{STENCIL_LABELS[width]} filter\n{ell_label}")
        axis.text(
            0.5,
            0.02,
            f"Schematic: increment denominator {delta_denominator}; "
            f"$B_{{loc}}$ denominator {b_denominator}",
            transform=axis.transAxes,
            ha="center",
            fontsize=8,
            color="#555555",
        )
    figure.suptitle("Schematic: labeled increment filters and local-field weights")
    return _save(figure, output_dir, "phase3a_stencil_schematic.png")


def input_cube_midplane_montage(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    images = []
    for cube_id in BENCHMARK_CUBE_IDS:
        manifest_path = verified.phase2_root / cube_id / "manifest.json"
        completion_path = verified.phase2_root / cube_id / "COMPLETE.json"
        manifest = _load_json(manifest_path)
        input_hashes.add_many((manifest_path, completion_path))
        relative = manifest["output_fields"]["dens"]["relative_path"]
        path = verified.phase2_root / cube_id / relative
        input_hashes.add(path)
        density = np.load(path, mmap_mode="r")
        if density.ndim != 3 or np.any(density[density.shape[0] // 2] <= 0.0):
            raise RuntimeError(f"Phase 2 density midplane is not a positive 3-D array: {path}")
        images.append(np.asarray(density[density.shape[0] // 2], dtype=float))
    values = np.concatenate([image.ravel() for image in images])
    vmin, vmax = np.quantile(values, (0.01, 0.99))
    if not 0.0 < vmin < vmax:
        raise RuntimeError("Phase 2 density montage has an invalid shared color range")
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True)
    plotted = None
    for axis, cube_id, image in zip(axes.flat, BENCHMARK_CUBE_IDS, images):
        plotted = axis.imshow(
            image,
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=float(vmin), vmax=float(vmax)),
            interpolation="nearest",
        )
        axis.set_title(f"{SHORT_LABELS[cube_id]}: {cube_id}")
        axis.set_xlabel("$i = x_1 / \\Delta x$")
        axis.set_ylabel("$j = x_2 / \\Delta x$")
    assert plotted is not None
    colorbar = figure.colorbar(plotted, ax=axes, shrink=0.87)
    colorbar.set_label(r"$\rho$ (shared 1st-99th percentile display range)")
    figure.suptitle(r"Verified Phase 2 inputs: density on each cube's $k=320$ midplane")
    return _save(figure, output_dir, "phase3a_input_cube_midplane_montage.png")


def manifest_occupancy(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    accounting = []
    for stencil_width in sorted(runner.STENCIL_SPECS):
        metadata, _, edges = runner._load_displacement_manifest(
            verified.release_root, stencil_width
        )
        json_path, npz_path = runner._manifest_paths(verified.release_root, stencil_width)
        input_hashes.add_many((json_path, npz_path))
        occupancy = np.asarray(metadata["realized_directional_occupancy_per_bin"], dtype=int)
        axes[0].plot(
            _centers(edges),
            occupancy,
            marker="o",
            markersize=3,
            color=STENCIL_COLORS[stencil_width],
            label=f"{STENCIL_LABELS[stencil_width]}: {metadata['realized_offset_count']:,} offsets",
        )
        accounting.append(
            (
                stencil_width,
                int(metadata["post_rounding_zero_offset_removed"]),
                int(metadata["post_rounding_duplicate_offset_removed"]),
                int(metadata["post_rounding_out_of_range_removed"]),
            )
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\ell$ [cells]")
    axes[0].set_ylabel("realized signed offsets per bin")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    labels = [STENCIL_LABELS[row[0]] for row in accounting]
    x = np.arange(len(labels))
    for offset, column, label in (
        (-0.24, 1, "rounded zero"),
        (0.0, 2, "rounded duplicate"),
        (0.24, 3, "strict scale exclusion"),
    ):
        axes[1].bar(x + offset, [row[column] for row in accounting], width=0.24, label=label)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("removed candidates")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Frozen displacement-manifest occupancy and post-rounding accounting")
    return _save(figure, output_dir, "phase3a_manifest_occupancy.png")


def worker_scaling(verified: VerifiedInputs, output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for row in verified.controls["rows"]:
        stencil_width = int(row["stencil_width"])
        sweep = sorted(row["worker_sweep"], key=lambda item: int(item["worker_count"]))
        workers = np.asarray([item["worker_count"] for item in sweep], dtype=int)
        wall = np.asarray([item["wall_seconds"] for item in sweep], dtype=float)
        speedup = np.asarray([item["speedup_vs_one_worker_wall"] for item in sweep], dtype=float)
        if np.any(workers < 1) or np.any(~np.isfinite(wall)) or np.any(wall <= 0.0):
            raise RuntimeError(f"invalid retained worker sweep for stencil {stencil_width}")
        axes[0].plot(workers, wall, "o-", color=STENCIL_COLORS[stencil_width], label=STENCIL_LABELS[stencil_width])
        axes[1].plot(workers, speedup, "o-", color=STENCIL_COLORS[stencil_width], label=STENCIL_LABELS[stencil_width])
    axes[0].set_xlabel("node-local worker processes")
    axes[0].set_ylabel("measured bounded-control wall time [s]")
    axes[1].set_xlabel("node-local worker processes")
    axes[1].set_ylabel("measured speedup versus one worker")
    axes[1].plot((1, max(axes[1].get_xlim()[1], 2)), (1, max(axes[1].get_xlim()[1], 2)), "--", color="#777777", linewidth=1.0, label="ideal")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Measured representative-cube worker scaling controls")
    return _save(figure, output_dir, "phase3a_worker_scaling.png")


def convergence_census(verified: VerifiedInputs, output_dir: Path) -> Path:
    rows = list(verified.convergence["rows"])
    complete = [row for row in rows if row.get("operational_status") == "complete"]
    if not complete:
        raise RuntimeError("convergence publication contains no completed scenarios")
    families = list(dict.fromkeys(str(row["family"]) for row in rows))
    family_colors = {
        family: plt.get_cmap("tab10")(index % 10) for index, family in enumerate(families)
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for family in families:
        selected = [row for row in complete if row["family"] == family]
        axes[0].scatter(
            [row["scenario_index"] for row in selected],
            [row["realized_offset_count"] for row in selected],
            color=family_colors[family],
            label=family,
            marker="o",
            s=32,
        )
        axes[0].scatter(
            [row["scenario_index"] for row in selected],
            [row["measured_subset_offset_count"] for row in selected],
            color=family_colors[family],
            marker="x",
            s=34,
        )
        axes[1].scatter(
            [row["scenario_index"] for row in selected],
            [row["elapsed_seconds_sum"] for row in selected],
            color=family_colors[family],
            label=family,
            s=32,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("retained scenario index")
    axes[0].set_ylabel("offset count")
    axes[0].text(0.02, 0.05, "circles: generated census\ncrosses: measured bounded subset", transform=axes[0].transAxes, fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("retained scenario index")
    axes[1].set_ylabel("estimator elapsed-seconds sum")
    unavailable = [row for row in rows if row.get("operational_status") == "unavailable"]
    if unavailable:
        text = "\n".join(
            f"scenario {row['scenario_index']}: {row['support_mode']} unavailable"
            for row in unavailable
        )
        axes[1].text(
            0.98,
            0.04,
            text,
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    figure.suptitle("Bounded convergence census: measured scenarios only; unavailable diagnostics are labeled")
    return _save(figure, output_dir, "phase3a_convergence_census.png")


def support_by_ell(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True, sharey=True)
    for axis, stencil_width in zip(axes, sorted(runner.STENCIL_SPECS)):
        groups = {
            mode: _load_release_group(
                verified,
                input_hashes,
                cube_id=REPRESENTATIVE_CUBE_ID,
                stencil_width=stencil_width,
                support_mode=mode,
            )
            for mode in runner.SUPPORT_MODES
        }
        for mode, group in groups.items():
            result = group.result
            ell = _centers(result.ell_bin_edges)
            fraction = np.divide(
                result.eligible_pairs,
                result.cube_candidate_pairs,
                out=np.zeros_like(result.eligible_pairs, dtype=float),
                where=result.cube_candidate_pairs > 0,
            )
            axis.plot(ell, fraction, marker="o", markersize=2.5, label=mode.replace("_", " "), color=MODE_COLORS[mode])
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.04)
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_title(STENCIL_LABELS[stencil_width])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("eligible-origin fraction")
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Verified release support by separation: {REPRESENTATIVE_CUBE_ID}")
    return _save(figure, output_dir, "phase3a_support_by_ell.png")


def offset_resolved_support_orientation(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        3, 2, figsize=(12.5, 10.4), constrained_layout=True, sharex="col", sharey=True
    )
    plotted = None
    for row, stencil_width in enumerate(sorted(runner.STENCIL_SPECS)):
        for column, support_mode in enumerate(runner.SUPPORT_MODES):
            axis = axes[row, column]
            result = _load_release_group(
                verified,
                input_hashes,
                cube_id=REPRESENTATIVE_CUBE_ID,
                stencil_width=stencil_width,
                support_mode=support_mode,
            ).result
            displacements = np.asarray(result.displacements_ijk, dtype=float)
            ell = np.linalg.norm(displacements, axis=1)
            orientation = np.max(np.abs(displacements), axis=1) / ell
            candidates = np.maximum(
                np.asarray(result.cube_candidate_pairs_per_displacement, dtype=float), 1.0
            )
            intrinsic = (
                np.asarray(result.intrinsic_eligible_origins_per_displacement, dtype=float)
                / candidates
            )
            selected = (
                np.asarray(result.eligible_pairs_per_displacement, dtype=float)
                / candidates
            )
            axis.scatter(
                ell,
                intrinsic,
                s=8,
                marker="x",
                linewidths=0.55,
                color="#b5b5b5",
                alpha=0.45,
                label="intrinsic non-periodic support",
            )
            plotted = axis.scatter(
                ell,
                selected,
                s=10,
                c=orientation,
                vmin=1.0 / np.sqrt(3.0),
                vmax=1.0,
                cmap="viridis",
                alpha=0.75,
                label="selected support-policy support",
            )
            axis.set_xscale("log")
            axis.set_ylim(-0.02, 1.03)
            axis.grid(alpha=0.22)
            if row == 0:
                axis.set_title(support_mode.replace("_", " "))
            if column == 0:
                axis.set_ylabel(f"{STENCIL_LABELS[stencil_width]}\neligible-origin fraction")
            if row == 2:
                axis.set_xlabel(r"$|\mathbf{r}|$ [cells]")
    assert plotted is not None
    axes[0, 0].legend(fontsize=7, loc="lower left")
    colorbar = figure.colorbar(plotted, ax=axes, shrink=0.86)
    colorbar.set_label(r"Cartesian alignment $\max_i |r_i| / |\mathbf{r}|$")
    figure.suptitle(
        f"{REPRESENTATIVE_CUBE_ID}: offset-resolved finite support and orientation"
    )
    return _save(figure, output_dir, "phase3a_offset_resolved_support_orientation.png")


def representative_curves_with_block_bands(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    group = _load_release_group(
        verified,
        input_hashes,
        cube_id=REPRESENTATIVE_CUBE_ID,
        stencil_width=PRIMARY_STENCIL_WIDTH,
        support_mode=PRIMARY_SUPPORT_MODE,
    )
    result, uncertainty = group.result, group.uncertainty
    ell = _centers(result.ell_bin_edges)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for axis, q_name in zip(axes, ("B", "u")):
        for direction in ("parallel", "xi", "lambda"):
            index = _moment_index(result, q_name, direction)
            _plot_positive_band(
                axis,
                ell,
                uncertainty["block_bootstrap_interval_low"][index],
                uncertainty["block_bootstrap_interval_high"][index],
                color=DIRECTION_COLORS[direction],
            )
            _plot_positive_curve(
                axis,
                ell,
                result.moments[index],
                label=direction,
                color=DIRECTION_COLORS[direction],
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(rf"$S_{{2,\perp}}^{{{q_name}}}(\ell)$")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        f"{REPRESENTATIVE_CUBE_ID}: 2-point shell-local curves with 95% spatial-block bootstrap bands"
    )
    return _save(figure, output_dir, "phase3a_representative_B_u_curves_with_block_bands.png")


def local_slopes_with_effective_blocks(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    group = _load_release_group(
        verified,
        input_hashes,
        cube_id=REPRESENTATIVE_CUBE_ID,
        stencil_width=PRIMARY_STENCIL_WIDTH,
        support_mode=PRIMARY_SUPPORT_MODE,
    )
    result, uncertainty = group.result, group.uncertainty
    ell = _centers(result.ell_bin_edges)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True, sharex="col")
    for column, q_name in enumerate(("B", "u")):
        for direction in ("parallel", "xi", "lambda"):
            index = _moment_index(result, q_name, direction)
            slope = uncertainty["local_log_slope"][index]
            low = uncertainty["local_log_slope_bootstrap_interval_low"][index]
            high = uncertainty["local_log_slope_bootstrap_interval_high"][index]
            valid_band = np.isfinite(ell) & np.isfinite(low) & np.isfinite(high)
            axes[0, column].fill_between(
                ell[valid_band],
                low[valid_band],
                high[valid_band],
                color=DIRECTION_COLORS[direction],
                alpha=0.14,
            )
            axes[0, column].plot(ell, slope, color=DIRECTION_COLORS[direction], label=direction)
            axes[1, column].plot(
                ell,
                uncertainty["accepted_effective_blocks"][index],
                color=DIRECTION_COLORS[direction],
                label=direction,
            )
        axes[0, column].set_ylabel(rf"local $\alpha_{{{q_name},\perp}} = d\log S_{{2,\perp}}^{{{q_name}}}/d\log\ell$")
        axes[1, column].set_ylabel("Kish effective accepted blocks")
        axes[1, column].set_xlabel(r"$\ell$ [cells]")
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"{REPRESENTATIVE_CUBE_ID}: centered 5-bin local slopes, 95% block bands, and effective blocks"
    )
    return _save(figure, output_dir, "phase3a_local_slopes_with_block_bands_and_effective_blocks.png")


def slope_window_sensitivity(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    candidates = [
        row
        for row in verified.convergence["rows"]
        if row.get("operational_status") == "complete"
        and row["family"] == "bins"
        and int(row["stencil_width"]) == 2
        and int(row["bin_count"]) == 64
        and row["support_mode"] == "shell_local"
    ]
    if len(candidates) != 1:
        raise RuntimeError("convergence publication lacks one 64-bin slope-window scenario")
    artifact = _convergence_scenario_artifact(verified, input_hashes, candidates[0])
    reference = _load_release_group(
        verified,
        input_hashes,
        cube_id=REPRESENTATIVE_CUBE_ID,
        stencil_width=PRIMARY_STENCIL_WIDTH,
        support_mode=PRIMARY_SUPPORT_MODE,
    ).result
    ell = _centers(artifact["ell_bin_edges"])
    windows = (
        (3, artifact["local_log_slope_window_3"], "#4c78a8"),
        (5, artifact["local_log_slope"], "#f58518"),
        (7, artifact["local_log_slope_window_7"], "#54a24b"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), constrained_layout=True, sharex=True)
    for row, q_name in enumerate(("B", "u")):
        for column, direction in enumerate(("parallel", "xi", "lambda")):
            axis = axes[row, column]
            index = _moment_index(reference, q_name, direction)
            for window, slopes, color in windows:
                axis.plot(ell, slopes[index], color=color, label=f"{window}-bin window")
            axis.set_xscale("log")
            axis.grid(alpha=0.22)
            axis.set_title(direction)
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(rf"$\alpha_{{{q_name},\perp}}(\ell)$")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"{REPRESENTATIVE_CUBE_ID}: supported local-slope sensitivity to centered regression width"
    )
    return _save(figure, output_dir, "phase3a_slope_window_sensitivity.png")


def four_cube_supported_slope_overview(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        len(BENCHMARK_CUBE_IDS),
        2,
        figsize=(12.0, 12.0),
        constrained_layout=True,
        sharex=True,
        sharey="col",
    )
    for row, cube_id in enumerate(BENCHMARK_CUBE_IDS):
        group = _load_release_group(
            verified,
            input_hashes,
            cube_id=cube_id,
            stencil_width=PRIMARY_STENCIL_WIDTH,
            support_mode=PRIMARY_SUPPORT_MODE,
        )
        result, uncertainty = group.result, group.uncertainty
        ell = _centers(result.ell_bin_edges)
        for column, q_name in enumerate(("B", "u")):
            axis = axes[row, column]
            for direction in ("parallel", "xi", "lambda"):
                index = _moment_index(result, q_name, direction)
                low = uncertainty["local_log_slope_bootstrap_interval_low"][index]
                high = uncertainty["local_log_slope_bootstrap_interval_high"][index]
                valid_band = np.isfinite(ell) & np.isfinite(low) & np.isfinite(high)
                axis.fill_between(
                    ell[valid_band],
                    low[valid_band],
                    high[valid_band],
                    color=DIRECTION_COLORS[direction],
                    alpha=0.10,
                )
                axis.plot(
                    ell,
                    uncertainty["local_log_slope"][index],
                    color=DIRECTION_COLORS[direction],
                    label=direction,
                )
            axis.set_xscale("log")
            axis.grid(alpha=0.22)
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(f"{SHORT_LABELS[cube_id]}\n" + rf"$\alpha_{{{q_name},\perp}}(\ell)$")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Four-cube 2-point shell-local supported local slopes with 95% block bands")
    return _save(figure, output_dir, "phase3a_four_cube_supported_slope_overview.png")


def _convergence_scenario_artifact(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    row: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    path = verified.convergence_root / str(row["artifact_relative_path"])
    input_hashes.add(path)
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _median_interpolated_relative_difference(
    reference_ell: np.ndarray,
    reference_values: np.ndarray,
    other_ell: np.ndarray,
    other_values: np.ndarray,
) -> float:
    reference_valid = (
        np.isfinite(reference_ell)
        & np.isfinite(reference_values)
        & (reference_ell > 0.0)
        & (reference_values > 0.0)
    )
    other_valid = (
        np.isfinite(other_ell)
        & np.isfinite(other_values)
        & (other_ell > 0.0)
        & (other_values > 0.0)
    )
    if np.count_nonzero(reference_valid) < 2 or np.count_nonzero(other_valid) < 2:
        return float("nan")
    lo = max(float(np.min(reference_ell[reference_valid])), float(np.min(other_ell[other_valid])))
    hi = min(float(np.max(reference_ell[reference_valid])), float(np.max(other_ell[other_valid])))
    shared = reference_valid & (reference_ell >= lo) & (reference_ell <= hi)
    if np.count_nonzero(shared) < 2:
        return float("nan")
    interpolated = np.exp(
        np.interp(
            np.log(reference_ell[shared]),
            np.log(other_ell[other_valid]),
            np.log(other_values[other_valid]),
        )
    )
    relative = np.abs(interpolated / reference_values[shared] - 1.0)
    return float(np.median(relative))


def convergence_science_differences(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    reference_result = _load_release_group(
        verified,
        input_hashes,
        cube_id=REPRESENTATIVE_CUBE_ID,
        stencil_width=PRIMARY_STENCIL_WIDTH,
        support_mode=PRIMARY_SUPPORT_MODE,
    ).result
    rows = [
        row
        for row in verified.convergence["rows"]
        if row.get("operational_status") == "complete"
        and row["family"]
        in {
            "bins",
            "directions",
            "directions_all_valid",
            "origins",
            "origin_seeds",
            "support",
        }
        and int(row["stencil_width"]) == 2
    ]
    if not rows:
        raise RuntimeError("convergence publication contains no comparable science scenarios")
    references: dict[str, Mapping[str, Any]] = {}
    preferred = {
        "bins": ("bin_count", 64),
        "directions": ("directions_per_bin", 24),
        "directions_all_valid": ("directions_per_bin", 24),
        "origins": ("sample_count", 2048),
        "origin_seeds": ("seed", runner.PRODUCTION_SEED),
        "support": ("support_mode", "shell_local"),
    }
    for family, (name, value) in preferred.items():
        candidates = [row for row in rows if row["family"] == family and row[name] == value]
        if len(candidates) != 1:
            raise RuntimeError(f"convergence family {family} lacks one preferred reference")
        references[family] = candidates[0]
    artifacts = {
        int(row["scenario_index"]): _convergence_scenario_artifact(verified, input_hashes, row)
        for row in rows
    }
    family_colors = {
        family: plt.get_cmap("tab10")(index)
        for index, family in enumerate(preferred)
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True, sharey=True)
    x_positions: list[int] = []
    x_labels: list[str] = []
    x_cursor = 0
    for axis, q_name in zip(axes, ("B", "u")):
        index = _moment_index(reference_result, q_name, "lambda")
        x_cursor = 0
        for family in preferred:
            selected = [row for row in rows if row["family"] == family]
            reference_artifact = artifacts[int(references[family]["scenario_index"])]
            reference_ell = _centers(reference_artifact["ell_bin_edges"])
            reference_values = reference_artifact["moments"][index]
            values = []
            for row in selected:
                artifact = artifacts[int(row["scenario_index"])]
                values.append(
                    _median_interpolated_relative_difference(
                        reference_ell,
                        reference_values,
                        _centers(artifact["ell_bin_edges"]),
                        artifact["moments"][index],
                    )
                )
            positions = list(range(x_cursor, x_cursor + len(selected)))
            if axis is axes[0]:
                x_positions.extend(positions)
                for row in selected:
                    if family in {"directions", "directions_all_valid"}:
                        value = row["directions_per_bin"]
                    elif family == "origins":
                        value = row["sample_count"]
                    elif family == "origin_seeds":
                        value = row["seed"]
                    elif family == "bins":
                        value = row["bin_count"]
                    else:
                        value = str(row["support_mode"]).replace("_", " ")
                    x_labels.append(f"{family}\n{value}")
            axis.scatter(
                positions,
                values,
                s=34,
                color=family_colors[family],
                label=family,
            )
            x_cursor += len(selected) + 1
        axis.set_yscale("symlog", linthresh=1.0e-6)
        axis.set_xticks(x_positions, x_labels, rotation=55, ha="right", fontsize=7)
        axis.set_xlabel("convergence family and tested value")
        axis.set_ylabel(rf"median relative difference in $\lambda$-wedge $S_{{2,\perp}}^{{{q_name}}}$")
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        "Bounded science-result convergence: each family is compared with its documented reference"
    )
    return _save(figure, output_dir, "phase3a_convergence_science_differences.png")


def convergence_scale_and_block_diagnostics(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    reference = _load_release_group(
        verified,
        input_hashes,
        cube_id=REPRESENTATIVE_CUBE_ID,
        stencil_width=PRIMARY_STENCIL_WIDTH,
        support_mode=PRIMARY_SUPPORT_MODE,
    ).result
    ell_rows = [
        row
        for row in verified.convergence["rows"]
        if row.get("operational_status") == "complete"
        and row["family"] == "ell_max"
        and int(row["stencil_width"]) == 2
    ]
    block_rows = [
        row
        for row in verified.convergence["rows"]
        if row.get("operational_status") == "complete"
        and row["family"] == "blocks"
        and int(row["stencil_width"]) == 2
    ]
    if len(ell_rows) != 4 or len(block_rows) != 3:
        raise RuntimeError("convergence publication lacks the expected ell_max or block matrix")
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
    ell_max_values = np.asarray([int(row["ell_max"]) for row in ell_rows])
    axes[0].plot(
        ell_max_values,
        [float(row["minimum_shell_valid_fraction"]) for row in ell_rows],
        "o-",
        color="#4c78a8",
    )
    axes[0].set_xlabel(r"configured 2-point $\ell_{\max}$ [cells]")
    axes[0].set_ylabel("minimum shell-local eligible-origin fraction")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.22)
    block_labels = []
    uncertainty_widths = []
    effective_blocks = []
    index = _moment_index(reference, "B", "lambda")
    for row in block_rows:
        artifact = _convergence_scenario_artifact(verified, input_hashes, row)
        ell = _centers(artifact["ell_bin_edges"])
        large_scale = np.isfinite(ell) & (ell >= 32.0) & (ell <= 160.0)
        moment = artifact["moments"][index]
        low = artifact["block_bootstrap_interval_low"][index]
        high = artifact["block_bootstrap_interval_high"][index]
        widths = np.divide(
            high - low,
            2.0 * moment,
            out=np.full_like(moment, np.nan, dtype=float),
            where=np.isfinite(moment) & (moment > 0.0),
        )
        block_labels.append(rf"${int(row['block_shape_kji'][0])}^3$")
        uncertainty_widths.append(float(np.nanmedian(widths[large_scale])))
        effective_blocks.append(
            float(np.nanmedian(artifact["accepted_effective_blocks"][index][large_scale]))
        )
    positions = np.arange(len(block_rows))
    axes[1].bar(positions, uncertainty_widths, color="#f58518")
    axes[1].set_xticks(positions, block_labels)
    axes[1].set_xlabel("spatial block side length [cells]")
    axes[1].set_ylabel(r"median 95% block-band fractional half-width, $32 \leq \ell \leq 160$")
    axes[1].grid(axis="y", alpha=0.22)
    axes[2].bar(positions, effective_blocks, color="#54a24b")
    axes[2].set_xticks(positions, block_labels)
    axes[2].set_xlabel("spatial block side length [cells]")
    axes[2].set_ylabel(r"median Kish effective blocks, $32 \leq \ell \leq 160$")
    axes[2].grid(axis="y", alpha=0.22)
    figure.suptitle(
        f"{REPRESENTATIVE_CUBE_ID}: explicit outer-scale support loss and block-layout uncertainty sensitivity"
    )
    return _save(figure, output_dir, "phase3a_convergence_scale_and_block_diagnostics.png")


def support_mode_comparison(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    rows = [
        row
        for row in verified.convergence["rows"]
        if row["family"] == "support" and int(row["stencil_width"]) == 2
    ]
    by_mode = {str(row["support_mode"]): row for row in rows}
    required = {"shell_local", "all_valid_origins", "nested_core"}
    if set(by_mode) != required:
        raise RuntimeError("convergence support-mode diagnostic inventory is incomplete")
    reference = _load_release_group(
        verified,
        input_hashes,
        cube_id=REPRESENTATIVE_CUBE_ID,
        stencil_width=PRIMARY_STENCIL_WIDTH,
        support_mode=PRIMARY_SUPPORT_MODE,
    ).result
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    unavailable = []
    for mode in ("shell_local", "all_valid_origins", "nested_core"):
        row = by_mode[mode]
        if row.get("operational_status") == "unavailable":
            unavailable.append(f"{mode}: {row.get('reason', 'unavailable')}")
            continue
        if row.get("operational_status") != "complete":
            raise RuntimeError(f"unexpected convergence support-mode status: {row}")
        artifact = _convergence_scenario_artifact(verified, input_hashes, row)
        ell = _centers(artifact["ell_bin_edges"])
        for axis, q_name in zip(axes, ("B", "u")):
            index = _moment_index(reference, q_name, "lambda")
            _plot_positive_curve(
                axis,
                ell,
                artifact["moments"][index],
                label=mode.replace("_", " "),
                color=MODE_COLORS[mode],
            )
    for axis, q_name in zip(axes, ("B", "u")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(rf"$S_{{2,\perp}}^{{{q_name}}}(\ell)$ in $\lambda$ wedge")
        axis.grid(alpha=0.25)
    if unavailable:
        axes[1].text(
            0.98,
            0.04,
            "\n".join(unavailable),
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
    axes[0].legend(fontsize=8)
    figure.suptitle("Bounded representative-cube support-mode comparison; unavailable diagnostics are labeled")
    return _save(figure, output_dir, "phase3a_support_mode_comparison.png")


def four_cube_support_mode_ratios(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        len(BENCHMARK_CUBE_IDS),
        2,
        figsize=(12.0, 12.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for row, cube_id in enumerate(BENCHMARK_CUBE_IDS):
        shell = _load_release_group(
            verified,
            input_hashes,
            cube_id=cube_id,
            stencil_width=PRIMARY_STENCIL_WIDTH,
            support_mode="shell_local",
        ).result
        all_valid = _load_release_group(
            verified,
            input_hashes,
            cube_id=cube_id,
            stencil_width=PRIMARY_STENCIL_WIDTH,
            support_mode="all_valid_origins",
        ).result
        ell = _centers(shell.ell_bin_edges)
        for column, q_name in enumerate(("B", "u")):
            axis = axes[row, column]
            for direction in ("parallel", "xi", "lambda"):
                index = _moment_index(shell, q_name, direction)
                ratio = np.divide(
                    all_valid.moments[index],
                    shell.moments[index],
                    out=np.full_like(shell.moments[index], np.nan, dtype=float),
                    where=np.isfinite(shell.moments[index]) & (shell.moments[index] != 0.0),
                )
                axis.plot(ell, ratio, color=DIRECTION_COLORS[direction], label=direction)
            axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
            axis.set_xscale("log")
            axis.grid(alpha=0.22)
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(
                f"{SHORT_LABELS[cube_id]}\n"
                + rf"$S_{{2,\perp}}^{{{q_name},\mathrm{{all}}}} / S_{{2,\perp}}^{{{q_name},\mathrm{{shell}}}}$"
            )
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Four-cube 2-point support-policy sensitivity; ratios are descriptive, not corrections")
    return _save(figure, output_dir, "phase3a_four_cube_support_mode_ratios.png")


def stencil_comparison(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    groups = {
        stencil_width: _load_release_group(
            verified,
            input_hashes,
            cube_id=REPRESENTATIVE_CUBE_ID,
            stencil_width=stencil_width,
            support_mode=PRIMARY_SUPPORT_MODE,
        )
        for stencil_width in sorted(runner.STENCIL_SPECS)
    }
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), constrained_layout=True, sharex=False)
    for row, q_name in enumerate(("B", "u")):
        for column, direction in enumerate(("parallel", "xi", "lambda")):
            axis = axes[row, column]
            for stencil_width, group in groups.items():
                result = group.result
                index = _moment_index(result, q_name, direction)
                _plot_positive_curve(
                    axis,
                    _centers(result.ell_bin_edges),
                    result.moments[index],
                    label=STENCIL_LABELS[stencil_width],
                    color=STENCIL_COLORS[stencil_width],
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(rf"$S_{{2,\perp}}^{{{q_name}}}(\ell)$")
            axis.set_title(direction)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"{REPRESENTATIVE_CUBE_ID}: shell-local labeled filter comparison; filters are distinct statistics"
    )
    return _save(figure, output_dir, "phase3a_stencil_comparison.png")


def runtime_storage_summary(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
) -> Path:
    group_rows = list(verified.release_summary["groups"])
    by_stencil: dict[int, dict[str, float]] = {
        stencil_width: {
            "estimator": 0.0,
            "reduce": 0.0,
            "jackknife": 0.0,
            "bootstrap": 0.0,
            "reduction_bytes": 0.0,
            "shard_bytes": 0.0,
        }
        for stencil_width in sorted(runner.STENCIL_SPECS)
    }
    for row in group_rows:
        stencil_width = int(row["stencil_width"])
        by_stencil[stencil_width]["estimator"] += float(row["elapsed_seconds_sum"])
        marker_path = runner._reduction_paths(verified.release_root, str(row["group_id"]))[3]
        marker = _load_json(marker_path)
        input_hashes.add(marker_path)
        by_stencil[stencil_width]["reduce"] += float(marker["reduction_elapsed_seconds"])
        by_stencil[stencil_width]["jackknife"] += float(marker["jackknife_elapsed_seconds"])
        by_stencil[stencil_width]["bootstrap"] += float(marker["bootstrap_elapsed_seconds"])
        by_stencil[stencil_width]["reduction_bytes"] += float(
            marker["staging_logical_bytes_before_marker"]
        )
    shards = _load_json(verified.release_root / "manifests" / "shards.json")["shards"]
    for row in shards:
        marker_path = runner._shard_paths(verified.release_root, str(row["shard_id"]))[2]
        marker = _load_json(marker_path)
        input_hashes.add(marker_path)
        by_stencil[int(row["stencil_width"])]["shard_bytes"] += float(
            marker["staging_logical_bytes_before_marker"]
        )
    stencils = sorted(by_stencil)
    labels = [STENCIL_LABELS[value] for value in stencils]
    x = np.arange(len(stencils))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), constrained_layout=True)
    axes[0].bar(x, [by_stencil[value]["estimator"] for value in stencils], color=[STENCIL_COLORS[value] for value in stencils])
    axes[0].set_ylabel("estimator elapsed-seconds sum")
    axes[0].set_xticks(x, labels)
    axes[1].bar(x, [by_stencil[value]["reduce"] for value in stencils], label="strict reduction")
    axes[1].bar(
        x,
        [by_stencil[value]["jackknife"] for value in stencils],
        bottom=[by_stencil[value]["reduce"] for value in stencils],
        label="block jackknife",
    )
    first_two = [
        by_stencil[value]["reduce"] + by_stencil[value]["jackknife"] for value in stencils
    ]
    axes[1].bar(
        x,
        [by_stencil[value]["bootstrap"] for value in stencils],
        bottom=first_two,
        label="block bootstrap",
    )
    axes[1].set_ylabel("summed per-group post-processing wall time [s]")
    axes[1].set_xticks(x, labels)
    axes[1].legend(fontsize=7)
    width = 0.36
    axes[2].bar(
        x - width / 2,
        [by_stencil[value]["shard_bytes"] / 1024**3 for value in stencils],
        width=width,
        label="published shard staging",
    )
    axes[2].bar(
        x + width / 2,
        [by_stencil[value]["reduction_bytes"] / 1024**3 for value in stencils],
        width=width,
        label="published reduction staging",
    )
    axes[2].set_ylabel("logical GiB recorded before markers")
    axes[2].set_xticks(x, labels)
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Verified release runtime and publication-storage accounting by labeled stencil")
    return _save(figure, output_dir, "phase3a_runtime_storage_summary.png")


def _profile_resource_records(root: Path, action: str) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for path in sorted((root / "work_resource_records").glob("*.json")):
        payload = _load_json(path)
        if payload.get("action") == action:
            records.append((path, payload))
    return records


def _select_resource_allocation(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    action: str,
    node_count: int,
    workers: int,
    implementation_sha256: str,
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...]]:
    by_job: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, payload in records:
        _require_same_source(payload, implementation_sha256, label=str(path))
        if (
            payload.get("action") != action
            or payload.get("schema_version") != runner.SCHEMA_VERSION
            or int(payload.get("slurm_ntasks", 0)) != node_count
            or int(payload.get("workers", 0)) != workers
            or not math.isfinite(float(payload.get("action_wall_seconds", float("nan"))))
            or float(payload["action_wall_seconds"]) <= 0.0
        ):
            continue
        by_job.setdefault(str(payload.get("slurm_job_id")), []).append((path, payload))
    candidates = []
    for job_id, rows in by_job.items():
        procids = {int(payload["slurm_procid"]) for _, payload in rows}
        if len(rows) != node_count or procids != set(range(node_count)):
            continue
        if any(any(bool(row.get("reused")) for row in payload["rows"]) for _, payload in rows):
            continue
        candidates.append((max(path.stat().st_mtime_ns for path, _ in rows), job_id, rows))
    if not candidates:
        raise RuntimeError(
            f"no complete fresh {node_count}-node {action} resource allocation is retained"
        )
    _, _, selected = max(candidates)
    selected.sort(key=lambda item: int(item[1]["slurm_procid"]))
    return tuple(path for path, _ in selected), tuple(payload for _, payload in selected)


def _require_multinode_resource_binding(
    root: Path,
    records: tuple[dict[str, Any], ...],
    *,
    node_count: int,
    workers: int,
) -> None:
    """Cross-check report-level profile records against verified task markers."""

    expected_tasks = {f"task_{procid:04d}" for procid in range(node_count)}
    observed_tasks: set[str] = set()
    job_ids = {str(payload["slurm_job_id"]) for payload in records}
    if len(job_ids) != 1:
        raise RuntimeError(f"multi-node profile resource rows mix Slurm jobs: {root}")
    for payload in records:
        if (
            int(payload["workers"]) != workers
            or int(payload["published_count"]) != 1
            or int(payload["reused_count"]) != 0
            or len(payload["rows"]) != 1
        ):
            raise RuntimeError(f"multi-node profile has invalid task resource summary: {root}")
        row = payload["rows"][0]
        task_id = str(row["shard_id"])
        marker = _load_json(root / "multinode_control" / task_id / "COMPLETE.json")
        if (
            bool(row.get("reused"))
            or int(row["staging_logical_bytes_before_marker"])
            != int(marker["staging_logical_bytes_before_marker"])
            or int(row["staging_allocated_bytes_before_marker"])
            != int(marker["staging_allocated_bytes_before_marker"])
        ):
            raise RuntimeError(f"multi-node profile resource row lost task-marker binding: {task_id}")
        observed_tasks.add(task_id)
    if observed_tasks != expected_tasks:
        raise RuntimeError(f"multi-node profile resource rows have wrong task inventory: {root}")


def _verify_node_profile(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    root: Path,
) -> NodeProfile:
    """Verify either a multi-node control profile or a full-work profile."""

    multinode_marker = root / "MULTINODE_CONTROL_COMPLETE.json"
    if multinode_marker.is_file():
        payload = _verify_required_diagnostic(
            verified.phase2_root,
            root,
            filename="multinode_control.json",
            marker_filename="MULTINODE_CONTROL_COMPLETE.json",
            implementation_sha256=verified.implementation_sha256,
            phase2_source=verified.campaign["phase2_sources"][REPRESENTATIVE_CUBE_ID],
            input_hashes=input_hashes,
        )
        if payload.get("validation_status") != "passed":
            raise RuntimeError(f"multi-node profile did not pass validation: {root}")
        node_count = int(payload["task_count"])
        workers = int(payload["workers_per_task"])
        displacements, _, selected_offsets, config = runner._multinode_design()
        for procid in range(node_count):
            runner._verify_multinode_task(
                verified.phase2_root,
                root,
                procid=procid,
                ntasks=node_count,
                workers=workers,
                task_offsets=selected_offsets[procid::node_count],
                support_displacements=displacements,
                config=config,
            )
            task_root = root / "multinode_control" / f"task_{procid:04d}"
            input_hashes.add_many((task_root / "COMPLETE.json", task_root / "partial.npz"))
        files, records = _select_resource_allocation(
            _profile_resource_records(root, "multinode_work"),
            action="multinode_work",
            node_count=node_count,
            workers=workers,
            implementation_sha256=verified.implementation_sha256,
        )
        _require_multinode_resource_binding(
            root, records, node_count=node_count, workers=workers
        )
        fingerprint = json.dumps(
            {
                "kind": "multinode_control",
                "cube_id": payload["cube_id"],
                "selected_offset_count": payload["selected_offset_count"],
                "workers_per_task": workers,
                "phase2_source": payload["phase2_source"],
            },
            sort_keys=True,
        )
        profile_kind = "multi-node equivalence control"
    else:
        if not (root / "PLAN_COMPLETE.json").is_file():
            raise RuntimeError(
                "optional profile root has neither a multi-node control marker "
                f"nor a full-work plan marker: {root}"
            )
        campaign = runner._verify_plan(verified.phase2_root, root, verify_arrays=False)
        if campaign["phase2_sources"] != verified.campaign["phase2_sources"]:
            raise RuntimeError(f"full-work profile is bound to different Phase 2 inputs: {root}")
        if (
            campaign["source_version"]["implementation_sha256"]
            != verified.implementation_sha256
        ):
            raise RuntimeError(f"full-work profile was produced by a different source: {root}")
        input_hashes.add_many(
            (
                root / "PLAN_COMPLETE.json",
                root / "manifests" / "campaign.json",
                root / "manifests" / "shards.json",
            )
        )
        records_all = _profile_resource_records(root, "work")
        retained_node_counts = sorted(
            {int(payload.get("slurm_ntasks", 0)) for _, payload in records_all}
        )
        if len(retained_node_counts) != 1 or retained_node_counts[0] < 1:
            raise RuntimeError(f"full-work profile does not identify one node count: {root}")
        node_count = retained_node_counts[0]
        fresh_worker_counts = {
            int(payload.get("workers", 0))
            for _, payload in records_all
            if not any(bool(row.get("reused")) for row in payload.get("rows", ()))
        }
        if len(fresh_worker_counts) != 1 or min(fresh_worker_counts) < 1:
            raise RuntimeError(f"full-work profile does not identify one fresh worker count: {root}")
        files, records = _select_resource_allocation(
            records_all,
            action="work",
            node_count=node_count,
            workers=fresh_worker_counts.pop(),
            implementation_sha256=verified.implementation_sha256,
        )
        workers_set = {int(payload["workers"]) for payload in records}
        if len(workers_set) != 1:
            raise RuntimeError(f"full-work profile mixes worker counts: {root}")
        workers = workers_set.pop()
        expected_shards = {
            row["shard_id"]
            for row in _load_json(root / "manifests" / "shards.json")["shards"]
        }
        observed_shards = [
            row["shard_id"] for payload in records for row in payload["rows"]
        ]
        if len(observed_shards) != len(set(observed_shards)) or set(observed_shards) != expected_shards:
            raise RuntimeError(f"full-work profile does not cover the exact shard inventory: {root}")
        for row in _load_json(root / "manifests" / "shards.json")["shards"]:
            runner._verify_shard(root, row)
            _, partial_path, marker_path = runner._shard_paths(root, str(row["shard_id"]))
            input_hashes.add_many((partial_path, marker_path))
        fingerprint = json.dumps(
            {
                "kind": "full_release_work",
                "configuration_sha256": campaign["configuration_sha256"],
                "displacement_manifests": campaign["displacement_manifests"],
                "shard_count": campaign["shard_count"],
                "phase2_sources": campaign["phase2_sources"],
                "workers_per_task": workers,
            },
            sort_keys=True,
        )
        profile_kind = "full release work"
    input_hashes.add_many(files)
    wall_seconds = max(float(payload["action_wall_seconds"]) for payload in records)
    return NodeProfile(
        root=root,
        profile_kind=profile_kind,
        node_count=node_count,
        workers_per_node=workers,
        wall_seconds=wall_seconds,
        task_local_node_hour_proxy=node_count * wall_seconds / 3600.0,
        resource_records=files,
        workload_fingerprint=fingerprint,
    )


def node_scaling(
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    output_dir: Path,
    *,
    one_node_root: Path,
    two_node_root: Path,
) -> tuple[Path, tuple[NodeProfile, NodeProfile]]:
    profiles = (
        _verify_node_profile(verified, input_hashes, one_node_root),
        _verify_node_profile(verified, input_hashes, two_node_root),
    )
    if tuple(profile.node_count for profile in profiles) != (1, 2):
        raise RuntimeError(
            "profile roots must retain one-node and two-node measurements respectively"
        )
    if (
        profiles[0].profile_kind != profiles[1].profile_kind
        or profiles[0].workload_fingerprint != profiles[1].workload_fingerprint
    ):
        raise RuntimeError("node profiles do not measure the same retained workload")
    wall = np.asarray([profile.wall_seconds for profile in profiles])
    speedup = wall[0] / wall
    node_hours = np.asarray([profile.task_local_node_hour_proxy for profile in profiles])
    x = np.asarray([profile.node_count for profile in profiles])
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
    axes[0].plot(x, wall, "o-", color="#4c78a8")
    axes[0].set_ylabel("maximum task-local action wall time [s]")
    axes[1].plot(x, speedup, "o-", color="#54a24b", label="measured")
    axes[1].plot(x, x, "--", color="#777777", label="ideal")
    axes[1].set_ylabel("speedup versus one node")
    axes[1].legend(fontsize=8)
    axes[2].plot(x, node_hours, "o-", color="#f58518")
    axes[2].set_ylabel("task-local node-hour proxy")
    for axis in axes:
        axis.set_xlabel("nodes")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Measured node scaling: {profiles[0].profile_kind}, "
        f"{profiles[0].workers_per_node} worker(s) per node"
    )
    return _save(figure, output_dir, "phase3a_node_scaling.png"), profiles


def write_manifest(
    output_dir: Path,
    verified: VerifiedInputs,
    input_hashes: InputHashes,
    *,
    omitted_figures: Mapping[str, str],
    profiles: tuple[NodeProfile, NodeProfile] | None,
) -> None:
    generated = sorted(path.name for path in output_dir.glob("*.png"))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_sha256": verified.implementation_sha256,
        "input_roots": {
            "phase2_root": str(verified.phase2_root),
            "controls_root": str(verified.controls_root),
            "convergence_root": str(verified.convergence_root),
            "release_root": str(verified.release_root),
        },
        "input_sha256": input_hashes.as_dict(),
        "generated_figures": generated,
        "figure_sha256": {
            name: file_sha256(output_dir / name) for name in generated
        },
        "omitted_figures": dict(sorted(omitted_figures.items())),
    }
    if profiles is not None:
        payload["node_profiles"] = [
            {
                "root": str(profile.root),
                "profile_kind": profile.profile_kind,
                "node_count": profile.node_count,
                "workers_per_node": profile.workers_per_node,
                "wall_seconds": profile.wall_seconds,
                "task_local_node_hour_proxy": profile.task_local_node_hour_proxy,
                "resource_record_sha256": {
                    str(path.resolve()): file_sha256(path) for path in profile.resource_records
                },
            }
            for profile in profiles
        ]
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hash-bound Phase 3a status-report figures from retained artifacts."
    )
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--profile-one-node-root", type=Path)
    parser.add_argument("--profile-two-node-root", type=Path)
    parser.add_argument("--convergence-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.profile_one_node_root is None) != (args.profile_two_node_root is None):
        raise SystemExit(
            "--profile-one-node-root and --profile-two-node-root must be supplied together"
        )
    input_hashes = InputHashes()
    verified = verify_inputs(
        phase2_root=args.phase2_root,
        controls_root=args.controls_root,
        convergence_root=args.convergence_root,
        release_root=args.release_root,
        input_hashes=input_hashes,
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        workflow_schematic(temporary_output)
        stencil_schematic(temporary_output)
        input_cube_midplane_montage(verified, input_hashes, temporary_output)
        manifest_occupancy(verified, input_hashes, temporary_output)
        worker_scaling(verified, temporary_output)
        convergence_census(verified, temporary_output)
        support_by_ell(verified, input_hashes, temporary_output)
        offset_resolved_support_orientation(verified, input_hashes, temporary_output)
        representative_curves_with_block_bands(verified, input_hashes, temporary_output)
        local_slopes_with_effective_blocks(verified, input_hashes, temporary_output)
        slope_window_sensitivity(verified, input_hashes, temporary_output)
        four_cube_supported_slope_overview(verified, input_hashes, temporary_output)
        convergence_science_differences(verified, input_hashes, temporary_output)
        convergence_scale_and_block_diagnostics(verified, input_hashes, temporary_output)
        support_mode_comparison(verified, input_hashes, temporary_output)
        four_cube_support_mode_ratios(verified, input_hashes, temporary_output)
        stencil_comparison(verified, input_hashes, temporary_output)
        runtime_storage_summary(verified, input_hashes, temporary_output)
        omitted_figures: dict[str, str] = {}
        profiles = None
        if args.profile_one_node_root is not None:
            _, profiles = node_scaling(
                verified,
                input_hashes,
                temporary_output,
                one_node_root=args.profile_one_node_root,
                two_node_root=args.profile_two_node_root,
            )
        else:
            omitted_figures["phase3a_node_scaling.png"] = (
                "optional one-node and two-node profile roots were not supplied; "
                "no node-scaling values were fabricated"
            )
        write_manifest(
            temporary_output,
            verified,
            input_hashes,
            omitted_figures=omitted_figures,
            profiles=profiles,
        )
        backup = args.output_dir.parent / f".{args.output_dir.name}.backup-{time.time_ns()}"
        if args.output_dir.exists():
            args.output_dir.replace(backup)
        try:
            temporary_output.replace(args.output_dir)
        except Exception:
            if backup.exists() and not args.output_dir.exists():
                backup.replace(args.output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)


if __name__ == "__main__":
    main()
