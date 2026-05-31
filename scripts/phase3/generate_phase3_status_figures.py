#!/usr/bin/env python3
"""Generate reproducible Phase 3 status-report figures from the retained smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3.run_phase3_sampler import (
    BENCHMARK_CUBE_IDS,
    _verify_cube_output,
    _verify_seed_robustness_output,
)


SHORT_LABELS = ("low dBB", "median dBB", "high dBB", "weak mean field")
DIRECTION_COLORS = {
    "parallel": "#4c78a8",
    "xi": "#f58518",
    "lambda": "#54a24b",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


def _load_mode(cube_root: Path, mode: str) -> dict[str, np.ndarray]:
    keys = (
        "q_names",
        "geometry_names",
        "measurement_names",
        "direction_names",
        "ell_bin_edges",
        "counts",
        "moments",
        "standard_error",
        "sampled_pairs",
        "eligible_pairs",
        "cube_candidate_pairs",
        "excluded_boundary_pairs",
        "displacements_per_bin",
    )
    with np.load(cube_root / f"{mode}.npz", allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in keys}


def _index(payload: dict[str, np.ndarray], q: str, geometry: str, measurement: str, direction: str):
    return (
        list(payload["q_names"]).index(q),
        list(payload["geometry_names"]).index(geometry),
        list(payload["measurement_names"]).index(measurement),
        list(payload["direction_names"]).index(direction),
        0,
    )


def _summary_row(summary: dict, mode: str, q: str, geometry: str, measurement: str) -> dict:
    rows = summary["mode_summaries"][mode]["summary_rows"]
    for row in rows:
        if row["q"] == q and row["geometry"] == geometry and row["measurement"] == measurement:
            return row
    raise KeyError((mode, q, geometry, measurement))


def verify_inputs(smoke_root: Path, phase2_root: Path) -> tuple[list[dict], dict, dict]:
    summaries = []
    for cube_id in BENCHMARK_CUBE_IDS:
        _verify_cube_output(smoke_root / cube_id, phase2_root, cube_id)
        summaries.append(_load_json(smoke_root / cube_id / "summary.json"))
    campaign = _load_json(smoke_root / "phase3_smoke_summary.json")
    completion = _load_json(smoke_root / "PHASE3_SMOKE_COMPLETE.json")
    synthetic = _load_json(smoke_root / "synthetic_validation.json")
    _verify_seed_robustness_output(
        smoke_root / "seed_robustness", phase2_root, BENCHMARK_CUBE_IDS[0]
    )
    if campaign.get("status") != "passed" or tuple(campaign.get("cube_ids", ())) != BENCHMARK_CUBE_IDS:
        raise RuntimeError("Phase 3 campaign summary is missing or stale")
    if (
        completion.get("status") != "passed"
        or tuple(completion.get("cube_ids", ())) != BENCHMARK_CUBE_IDS
        or completion.get("summary_sha256") != file_sha256(smoke_root / "phase3_smoke_summary.json")
        or completion.get("summary_markdown_sha256") != file_sha256(smoke_root / "phase3_smoke_summary.md")
        or completion.get("synthetic_validation_sha256") != file_sha256(smoke_root / "synthetic_validation.json")
        or completion.get("seed_robustness_marker_sha256")
        != file_sha256(smoke_root / "seed_robustness" / "ROBUSTNESS_COMPLETE.json")
    ):
        raise RuntimeError("Phase 3 campaign completion marker is missing or stale")
    if synthetic.get("status") != "passed":
        raise RuntimeError("Phase 3 synthetic gate did not pass")
    return summaries, campaign, synthetic


def workflow_schematic(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 3.2))
    axis.set_axis_off()
    boxes = (
        (0.01, "Phase 2\n$640^3$ cubes"),
        (0.18, "signed 3-D\ndisplacements"),
        (0.35, "non-periodic\npair sampler"),
        (0.52, "local-field\nconditioning"),
        (0.69, "$S_2^B$, $S_2^u$\nreducers"),
        (0.86, "hash-bound\npublication"),
    )
    for x, label in boxes:
        axis.add_patch(
            plt.Rectangle((x, 0.35), 0.13, 0.34, facecolor="#e7f0fa", edgecolor="#2a5c8a", lw=1.4)
        )
        axis.text(x + 0.065, 0.52, label, ha="center", va="center", fontsize=10)
    for index in range(len(boxes) - 1):
        axis.annotate(
            "",
            xy=(boxes[index + 1][0] - 0.01, 0.52),
            xytext=(boxes[index][0] + 0.14, 0.52),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444444"},
        )
    axis.text(0.5, 0.12, "Schematic: Phase 3 validates a bounded CPU-only sampler before the 21-region pilot", ha="center")
    _save(figure, output_dir / "phase3_workflow_schematic.png")


def pair_mode_schematic(output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, title in zip(axes, ("Primary: nested core", "Robustness: all valid pairs")):
        axis.set_aspect("equal")
        axis.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", lw=1.6))
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title)
    axes[0].add_patch(plt.Rectangle((0.22, 0.22), 0.56, 0.56, facecolor="#d9ead3", edgecolor="#3d7a45"))
    axes[0].annotate("", xy=(0.69, 0.66), xytext=(0.44, 0.40), arrowprops={"arrowstyle": "->", "lw": 2})
    axes[0].text(0.5, 0.08, "one shared origin region for every signed offset", ha="center", fontsize=9)
    origins = ((0.15, 0.18), (0.36, 0.30), (0.73, 0.68), (0.61, 0.22), (0.20, 0.74))
    offsets = ((0.22, 0.18), (0.27, 0.13), (-0.18, -0.15), (0.21, 0.25), (0.16, -0.20))
    for (x, y), (dx, dy) in zip(origins, offsets):
        axes[1].plot(x, y, "o", color="#4c78a8", ms=4)
        axes[1].annotate("", xy=(x + dx, y + dy), xytext=(x, y), arrowprops={"arrowstyle": "->", "lw": 1.5})
    axes[1].text(0.5, 0.08, "offset-specific origin regions\nuse more boundary volume", ha="center", va="center", fontsize=9)
    figure.suptitle("Schematic: neither mode wraps pairs across extracted-cube boundaries")
    _save(figure, output_dir / "phase3_pair_modes_schematic.png")


def synthetic_gate(synthetic: dict, output_dir: Path) -> None:
    checks = synthetic["science_checks"]
    labels = (
        "oracle:\nnested",
        "oracle:\nall-valid",
        "isotropic\nratio",
        "guide-field\nordering",
        "ribbon\nordering",
        "weak-$B$\nexclusion",
        "core\nshrinkage",
    )
    values = [float(item["passed"]) for item in synthetic["oracle_checks"]]
    values.extend(
        [
            float(checks["isotropic_total_parallel_over_perpendicular"]["passed"]),
            float(checks["guide_field_parallel_exceeds_perpendicular"]["passed"]),
            float(checks["ribbon_fixed_S2_scale_ordering"]["passed"]),
            float(checks["degenerate_local_B_is_excluded"]["passed"]),
            float(checks["nested_core_shrinks_with_ell_max"]["passed"]),
        ]
    )
    figure, axis = plt.subplots(figsize=(9, 3.8))
    axis.bar(np.arange(len(values)), values, color=["#4f8a5b" if value else "#b54c4c" for value in values])
    axis.set_ylim(0.0, 1.12)
    axis.set_ylabel("gate passed")
    axis.set_yticks((0, 1))
    axis.set_xticks(np.arange(len(labels)), labels)
    ratio = checks["isotropic_total_parallel_over_perpendicular"]["value"]
    axis.text(2, 1.025, f"{ratio:.4f}", ha="center", va="bottom", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase3_synthetic_validation_gate.png")


def finite_support(smoke_root: Path, output_dir: Path) -> None:
    cube_root = smoke_root / BENCHMARK_CUBE_IDS[0]
    nested = _load_mode(cube_root, "nested_core")
    all_valid = _load_mode(cube_root, "all_valid_pairs")
    ell = _centers(nested["ell_bin_edges"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for payload, label in ((nested, "nested core"), (all_valid, "all valid pairs")):
        fraction = payload["eligible_pairs"] / payload["cube_candidate_pairs"]
        axes[0].plot(ell, fraction, marker="o", label=label)
        axes[1].plot(ell, payload["sampled_pairs"], marker="o", label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\ell$ [cells]")
    axes[0].set_ylabel("eligible origin fraction")
    axes[0].set_ylim(0, 1.04)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\ell$ [cells]")
    axes[1].set_ylabel("sampled pairs per separation bin")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    _save(figure, output_dir / "phase3_finite_support_by_ell.png")


def angular_support(summaries: list[dict], output_dir: Path) -> None:
    """Show that all-valid finite support is retained per offset, not only per shell."""

    rows = summaries[0]["mode_summaries"]["all_valid_pairs"]["offset_support_rows"]
    ell = np.asarray([row["ell"] for row in rows], dtype=float)
    vectors = np.asarray([row["r_vector"] for row in rows], dtype=float)
    support = np.asarray([row["eligible_origin_fraction"] for row in rows], dtype=float)
    direction_cosines = np.abs(vectors) / ell[:, None]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True, sharey=True)
    for axis, column, label in zip(axes, range(3), ("$|r_{x1}| / |r|$", "$|r_{x2}| / |r|$", "$|r_{x3}| / |r|$")):
        image = axis.scatter(ell, direction_cosines[:, column], c=support, cmap="viridis", vmin=0.0, vmax=1.0, s=18)
        axis.set_xscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88)
    colorbar.set_label("all-valid eligible-origin fraction")
    figure.suptitle("Representative cube: offset-resolved support across separation and orientation")
    _save(figure, output_dir / "phase3_all_valid_support_by_separation_and_orientation.png")


def directional_curves(smoke_root: Path, output_dir: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(11, 14), constrained_layout=True)
    for row, (cube_id, short_label) in enumerate(zip(BENCHMARK_CUBE_IDS, SHORT_LABELS)):
        payload = _load_mode(smoke_root / cube_id, "nested_core")
        ell = _centers(payload["ell_bin_edges"])
        for column, q_name in enumerate(("B", "u")):
            axis = axes[row, column]
            for direction in ("parallel", "xi", "lambda"):
                index = _index(payload, q_name, "pair_local", "perpendicular", direction)
                axis.plot(ell, payload["moments"][index], marker="o", color=DIRECTION_COLORS[direction], label=direction)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(rf"$S_2^{{{q_name}}}(\ell)$")
            axis.set_title(f"{short_label}: {cube_id}")
            axis.grid(alpha=0.25)
            if row == 0:
                axis.legend(fontsize=8)
    _save(figure, output_dir / "phase3_directional_perpendicular_curves_nested_core.png")


def mode_difference(summaries: list[dict], output_dir: Path) -> None:
    median = [summary["nested_vs_all_valid"]["median_relative_difference"] for summary in summaries]
    maximum = [summary["nested_vs_all_valid"]["maximum_relative_difference"] for summary in summaries]
    x = np.arange(len(summaries))
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar(x - 0.18, median, width=0.36, label="median across populated products")
    axis.bar(x + 0.18, maximum, width=0.36, label="maximum across populated products")
    axis.set_ylabel("nested-core versus all-valid relative difference")
    axis.set_xticks(x, SHORT_LABELS, rotation=15, ha="right")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase3_nested_vs_all_valid_difference.png")


def slope_comparison(summaries: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    x = np.arange(len(summaries))
    for axis, q_name in zip(axes, ("B", "u")):
        for offset, direction in zip((-0.24, 0.0, 0.24), ("parallel", "xi", "lambda")):
            nested = [
                _summary_row(summary, "nested_core", q_name, "pair_local", "perpendicular")["slopes"][direction]
                for summary in summaries
            ]
            robust = [
                _summary_row(summary, "all_valid_pairs", q_name, "pair_local", "perpendicular")["slopes"][direction]
                for summary in summaries
            ]
            axis.plot(x + offset, nested, marker="o", ls="-", color=DIRECTION_COLORS[direction], label=f"{direction}: nested")
            axis.plot(x + offset, robust, marker="x", ls=":", color=DIRECTION_COLORS[direction], label=f"{direction}: all-valid")
        axis.set_ylabel(rf"fitted $S_2^{{{q_name}}}$ slope")
        axis.set_xticks(x, SHORT_LABELS, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    _save(figure, output_dir / "phase3_slope_comparison_nested_vs_all_valid.png")


def fit_window_sensitivity(summaries: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    directions = ("parallel", "xi", "lambda")
    x = np.arange(len(summaries))
    for axis, q_name in zip(axes, ("B", "u")):
        for offset, direction in zip((-0.22, 0.0, 0.22), directions):
            spreads = []
            for summary in summaries:
                rows = summary["mode_summaries"]["nested_core"]["fit_stability_rows"]
                match = next(
                    row for row in rows
                    if row["q"] == q_name
                    and row["geometry"] == "pair_local"
                    and row["measurement"] == "perpendicular"
                    and row["direction"] == direction
                )
                spreads.append(match["absolute_slope_spread"])
            axis.plot(x + offset, spreads, marker="o", ls="", color=DIRECTION_COLORS[direction], label=direction)
        axis.axhline(0.2, color="black", ls="--", lw=1.0, label="diagnostic tolerance")
        axis.set_xticks(x, SHORT_LABELS, rotation=15, ha="right")
        axis.set_ylabel(r"absolute slope spread across tested $\ell$ windows")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    _save(figure, output_dir / "phase3_fit_window_slope_sensitivity.png")


def repeat_seed_robustness(smoke_root: Path, output_dir: Path) -> None:
    summary = _load_json(smoke_root / "seed_robustness" / "seed_robustness_summary.json")
    seeds = sorted(summary["repeat_seed_vs_base"], key=int)
    median = [summary["repeat_seed_vs_base"][seed]["median_relative_difference"] for seed in seeds]
    maximum = [summary["repeat_seed_vs_base"][seed]["maximum_relative_difference"] for seed in seeds]
    x = np.arange(len(seeds))
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    axis.bar(x - 0.18, median, width=0.36, label="median across populated products")
    axis.bar(x + 0.18, maximum, width=0.36, label="maximum across populated products")
    axis.set_xticks(x, [f"seed {seed}\nversus base" for seed in seeds])
    axis.set_ylabel("all-valid relative difference")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase3_repeat_seed_robustness.png")


def conditioning_coverage(summaries: list[dict], output_dir: Path) -> None:
    directions = ("parallel", "xi", "lambda")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(directions))
    for axis, q_name in zip(axes, ("B", "u")):
        for offset, geometry in ((-0.18, "pair_local"), (0.18, "subvolume_mean")):
            samples = np.asarray(
                [
                    [
                        _summary_row(summary, "nested_core", q_name, geometry, "perpendicular")["sample_counts"][direction]
                        for direction in directions
                    ]
                    for summary in summaries
                ]
            )
            axis.bar(x + offset, np.median(samples, axis=0), width=0.36, label=geometry.replace("_", " "))
        axis.set_yscale("log")
        axis.set_xticks(x, directions)
        axis.set_ylabel(f"median accepted {q_name} pairs")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    _save(figure, output_dir / "phase3_conditioning_basis_coverage.png")


def resource_summary(summaries: list[dict], campaign: dict, output_dir: Path) -> None:
    wall = [summary["performance"]["total_wall_seconds"] for summary in summaries]
    rss = [summary["performance"]["peak_rss_kib"] / 1024**2 for summary in summaries]
    x = np.arange(len(summaries))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(x, wall, color="#587da5")
    axes[0].set_ylabel("sampler wall time [s]")
    axes[0].set_xticks(x, SHORT_LABELS, rotation=15, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, rss, color="#8d6e63", label="measured per cube")
    axes[1].axhline(max(rss), color="black", ls="--", lw=1.0, label="campaign peak")
    axes[1].set_ylabel("peak RSS [GiB]")
    axes[1].set_xticks(x, SHORT_LABELS, rotation=15, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Four cubes: "
        f"{campaign['measured_total_wall_seconds']:.1f} s measured sampler work; "
        f"{campaign['forecast_21_cube_sampler_wall_seconds']:.1f} s linear 21-cube forecast"
    )
    _save(figure, output_dir / "phase3_resource_summary_and_forecast.png")


def hardening_comparison(smoke_root: Path, prehardening_profile_root: Path, output_dir: Path) -> None:
    cube_id = BENCHMARK_CUBE_IDS[0]
    pre_summary = _load_json(prehardening_profile_root / cube_id / "summary.json")
    post_summary = _load_json(smoke_root / cube_id / "summary.json")
    pre_config = pre_summary["configuration"]
    post_config = post_summary["configuration"]
    pre_directions = pre_config["displacements"]["directions_per_radius"]
    post_directions = post_config["displacements"]["directions_per_radius"]
    pre_depth = pre_config["modes"]["all_valid_pairs"]["sample_count"]
    post_depth = post_config["modes"]["all_valid_pairs"]["sample_count"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)
    axes[0].bar(("pre-hardening", "retained"), (pre_summary["performance"]["total_wall_seconds"], post_summary["performance"]["total_wall_seconds"]))
    axes[0].set_ylabel("sampler wall time [s]")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(("pre-hardening", "retained"), (pre_directions, post_directions), color="#4f8a5b")
    axes[1].set_ylabel("directions per nominal radius")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(("pre-hardening", "retained"), (pre_depth, post_depth), color="#8d6e63")
    axes[2].set_ylabel("all-valid samples per offset")
    axes[2].grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "phase3_before_after_hardening.png")


def outlier_curves(smoke_root: Path, summaries: list[dict], output_dir: Path) -> None:
    index = int(np.argmax([summary["nested_vs_all_valid"]["maximum_relative_difference"] for summary in summaries]))
    cube_id = BENCHMARK_CUBE_IDS[index]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, q_name in zip(axes, ("B", "u")):
        for mode, marker, linestyle in (("nested_core", "o", "-"), ("all_valid_pairs", "x", ":")):
            payload = _load_mode(smoke_root / cube_id, mode)
            ell = _centers(payload["ell_bin_edges"])
            values = payload["moments"][_index(payload, q_name, "pair_local", "perpendicular", "lambda")]
            axis.plot(ell, values, marker=marker, ls=linestyle, label=mode.replace("_", " "))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(rf"$S_2^{{{q_name}}}$ in $\lambda$ wedge")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Largest aggregate mode discrepancy: {cube_id}")
    _save(figure, output_dir / "phase3_outlier_mode_comparison.png")


def copy_input_montage(phase2_montage: Path, output_dir: Path) -> None:
    shutil.copyfile(phase2_montage, output_dir / "phase3_input_cube_midplane_montage.png")


def write_manifest(
    output_dir: Path,
    smoke_root: Path,
    phase2_montage: Path,
    prehardening_profile_root: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "status": "passed",
        "generator_sha256": file_sha256(Path(__file__)),
        "campaign_completion_sha256": file_sha256(smoke_root / "PHASE3_SMOKE_COMPLETE.json"),
        "synthetic_validation_sha256": file_sha256(smoke_root / "synthetic_validation.json"),
        "seed_robustness_marker_sha256": file_sha256(smoke_root / "seed_robustness" / "ROBUSTNESS_COMPLETE.json"),
        "phase2_montage_sha256": file_sha256(phase2_montage),
        "prehardening_profile_summary_sha256": file_sha256(
            prehardening_profile_root / BENCHMARK_CUBE_IDS[0] / "summary.json"
        ),
        "figure_sha256": {
            path.name: file_sha256(path)
            for path in sorted(output_dir.glob("*.png"))
        },
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--prehardening-profile-root", type=Path, required=True)
    parser.add_argument("--phase2-montage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/phase3_status_update"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, campaign, synthetic = verify_inputs(args.smoke_root, args.phase2_root)
    workflow_schematic(args.output_dir)
    pair_mode_schematic(args.output_dir)
    synthetic_gate(synthetic, args.output_dir)
    finite_support(args.smoke_root, args.output_dir)
    angular_support(summaries, args.output_dir)
    directional_curves(args.smoke_root, args.output_dir)
    mode_difference(summaries, args.output_dir)
    slope_comparison(summaries, args.output_dir)
    fit_window_sensitivity(summaries, args.output_dir)
    repeat_seed_robustness(args.smoke_root, args.output_dir)
    conditioning_coverage(summaries, args.output_dir)
    resource_summary(summaries, campaign, args.output_dir)
    hardening_comparison(args.smoke_root, args.prehardening_profile_root, args.output_dir)
    outlier_curves(args.smoke_root, summaries, args.output_dir)
    copy_input_montage(args.phase2_montage, args.output_dir)
    write_manifest(
        args.output_dir,
        args.smoke_root,
        args.phase2_montage,
        args.prehardening_profile_root,
    )


if __name__ == "__main__":
    main()
