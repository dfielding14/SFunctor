#!/usr/bin/env python3
"""Publish the immutable Phase 4 all-21-cube 3-point extension review."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3a import generate_phase3a_status_figures as figures
from scripts.phase4 import generate_phase4_batch_a2_review as base


DEFAULT_OUTPUT_DIR = Path("figures/phase4_batch_a2_3point_extension_review")
SUMMARY_FILENAME = "phase4_batch_a2_3point_extension_review_summary.json"
LEDGER_SNAPSHOT_FILENAME = "phase4_batch_a2_3point_extension_compute_ledger_summary_snapshot.md"
WIDTHS = (2, 3)
WIDTH_COLORS = {2: "#4c78a8", 3: "#f58518"}
WIDTH_LABELS = {2: "2-point baseline", 3: "3-point extension"}
HIGHLIGHTED_CURVES = (
    ("L640_sub03026", "B", "parallel"),
    ("L640_sub00732", "B", "parallel"),
    ("L640_sub02822", "B", "lambda"),
    ("L640_sub02602", "u", "parallel"),
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _load_groups(
    batch_a_root: Path,
    extension_root: Path,
    cube_ids: tuple[str, ...],
    input_hashes: base.InputHashes,
) -> dict[tuple[str, int, str], base.Group]:
    groups: dict[tuple[str, int, str], base.Group] = {}
    for cube_id in cube_ids:
        for width, root in ((2, batch_a_root), (3, extension_root)):
            for support_mode in base.SUPPORT_MODES:
                groups[(cube_id, width, support_mode)] = base._load_group(
                    root, cube_id, width, support_mode, input_hashes
                )
    return groups


def _verify_representative_overlap(
    representative_a2_root: Path,
    extension_root: Path,
    input_hashes: base.InputHashes,
) -> dict[str, Any]:
    compared_groups = []
    for cube_id in base.REPRESENTATIVE_CUBES:
        for support_mode in base.SUPPORT_MODES:
            base._load_group(
                representative_a2_root, cube_id, 3, support_mode, input_hashes
            )
            representative_root = base._group_root(
                representative_a2_root, cube_id, 3, support_mode
            )
            extension_group_root = base._group_root(
                extension_root, cube_id, 3, support_mode
            )
            with (
                np.load(representative_root / "result.npz", allow_pickle=False) as left,
                np.load(extension_group_root / "result.npz", allow_pickle=False) as right,
            ):
                if left.files != right.files:
                    raise RuntimeError(f"representative overlap schema mismatch: {cube_id}")
                for name in left.files:
                    if name == "elapsed_seconds_per_ell_bin":
                        continue
                    if name == "metadata_json":
                        left_metadata = json.loads(str(left[name]))
                        right_metadata = json.loads(str(right[name]))
                        left_metadata.pop("elapsed_seconds", None)
                        right_metadata.pop("elapsed_seconds", None)
                        if left_metadata != right_metadata:
                            raise RuntimeError(
                                f"representative overlap metadata mismatch: {cube_id}"
                            )
                        continue
                    equal_nan = left[name].dtype.kind in "fc"
                    if not np.array_equal(left[name], right[name], equal_nan=equal_nan):
                        raise RuntimeError(
                            f"representative overlap result mismatch: {cube_id}/{support_mode}/{name}"
                        )
            if file_sha256(representative_root / "uncertainty.npz") != file_sha256(
                extension_group_root / "uncertainty.npz"
            ):
                raise RuntimeError(
                    f"representative overlap uncertainty mismatch: {cube_id}/{support_mode}"
                )
            compared_groups.append(f"{cube_id}/stencil_3point/{support_mode}")
    return {
        "status": "passed",
        "compared_groups": compared_groups,
        "nonruntime_result_fields_equal": True,
        "uncertainty_npz_byte_identical": True,
        "runtime_only_result_fields_ignored": [
            "elapsed_seconds_per_ell_bin",
            "metadata_json.elapsed_seconds",
        ],
    }


def _factor(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "median": float(np.median(array)) if array.size else None,
        "p90": float(np.quantile(array, 0.90)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
    }


def _curve_rows(
    groups: Mapping[tuple[str, int, str], base.Group],
    cube_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    census: list[dict[str, Any]] = []
    for width in WIDTHS:
        for cube_id in cube_ids:
            primary = groups[(cube_id, width, "all_valid_origins")]
            shell = groups[(cube_id, width, "shell_local")]
            ell = figures._centers(primary.result.ell_bin_edges)
            shell_support = base._support_fraction(shell.result)
            for q_name in base.Q_NAMES:
                for direction in base.DIRECTIONS:
                    index = figures._moment_index(primary.result, q_name, direction)
                    numerator = primary.result.moments[index]
                    denominator = shell.result.moments[index]
                    ratio = np.divide(
                        numerator,
                        denominator,
                        out=np.full_like(numerator, np.nan, dtype=float),
                        where=np.isfinite(denominator) & (denominator != 0.0),
                    )
                    science = ell >= base.SCIENCE_SCALE_MINIMUM
                    geometry = shell_support >= base.SHELL_CURVE_MINIMUM
                    uncertainty = (
                        base._curve_support_mask(primary, index)
                        & base._curve_support_mask(shell, index)
                    )
                    finite_ratio = np.isfinite(ratio) & (ratio > 0.0)
                    supported = science & geometry & uncertainty & finite_ratio
                    census.append(
                        {
                            "width": width,
                            "cube_id": cube_id,
                            "q_name": q_name,
                            "direction": direction,
                            "candidate_science_scale_bins": int(np.count_nonzero(science)),
                            "retained_bins": int(np.count_nonzero(supported)),
                            "excluded_shell_support_bins": int(
                                np.count_nonzero(science & ~geometry)
                            ),
                            "excluded_uncertainty_bins": int(
                                np.count_nonzero(science & geometry & ~uncertainty)
                            ),
                            "excluded_nonfinite_ratio_bins": int(
                                np.count_nonzero(
                                    science & geometry & uncertainty & ~finite_ratio
                                )
                            ),
                        }
                    )
                    for bin_index in np.flatnonzero(supported):
                        factor = max(float(ratio[bin_index]), 1.0 / float(ratio[bin_index]))
                        retained.append(
                            {
                                "width": width,
                                "cube_id": cube_id,
                                "q_name": q_name,
                                "direction": direction,
                                "ell_cells": float(ell[bin_index]),
                                "shell_local_eligible_origin_fraction": float(
                                    shell_support[bin_index]
                                ),
                                "primary_over_shell_ratio": float(ratio[bin_index]),
                                "policy_factor": factor,
                            }
                        )
    return retained, census


def _policy_summary(
    retained: list[dict[str, Any]],
    cube_ids: tuple[str, ...],
) -> dict[str, Any]:
    pooled = {
        str(width): _factor(
            [row["policy_factor"] for row in retained if row["width"] == width]
        )
        for width in WIDTHS
    }
    by_channel = []
    for width in WIDTHS:
        for q_name in base.Q_NAMES:
            for direction in base.DIRECTIONS:
                rows = [
                    row
                    for row in retained
                    if row["width"] == width
                    and row["q_name"] == q_name
                    and row["direction"] == direction
                ]
                by_channel.append(
                    {
                        "width": width,
                        "q_name": q_name,
                        "direction": direction,
                        **_factor([row["policy_factor"] for row in rows]),
                        "count_above_1p5": sum(
                            row["policy_factor"] > 1.5 for row in rows
                        ),
                    }
                )
    maxima = []
    for width in WIDTHS:
        for cube_id in cube_ids:
            rows = [
                row
                for row in retained
                if row["width"] == width and row["cube_id"] == cube_id
            ]
            if rows:
                maximum = max(rows, key=lambda row: row["policy_factor"])
                maxima.append(
                    {
                        "width": width,
                        "cube_id": cube_id,
                        "retained_bins": len(rows),
                        "maximum_policy_factor": maximum["policy_factor"],
                        "maximum_row": maximum,
                        "count_above_1p5": sum(
                            row["policy_factor"] > 1.5 for row in rows
                        ),
                    }
                )
    top_rows = sorted(
        retained, key=lambda row: row["policy_factor"], reverse=True
    )[:20]
    return {
        "pooled_supported_factor": pooled,
        "supported_factor_by_channel": by_channel,
        "per_cube_maximum": maxima,
        "top_supported_rows": top_rows,
    }


def _census_summary(census: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for width in WIDTHS:
        rows = [row for row in census if row["width"] == width]
        output[str(width)] = {
            name: int(sum(row[name] for row in rows))
            for name in (
                "candidate_science_scale_bins",
                "retained_bins",
                "excluded_shell_support_bins",
                "excluded_uncertainty_bins",
                "excluded_nonfinite_ratio_bins",
            )
        }
    return output


def support_vs_ell(
    groups: Mapping[tuple[str, int, str], base.Group],
    cube_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.4), constrained_layout=True, sharey=True
    )
    cube_id = cube_ids[0]
    for axis, mode in zip(axes, base.SUPPORT_MODES):
        for width in WIDTHS:
            result = groups[(cube_id, width, mode)].result
            ell = figures._centers(result.ell_bin_edges)
            axis.plot(
                ell,
                base._support_fraction(result),
                color=WIDTH_COLORS[width],
                linewidth=1.8,
                label=WIDTH_LABELS[width],
            )
        if mode == "shell_local":
            axis.axhline(
                base.SHELL_CURVE_MINIMUM,
                color="#777777",
                linestyle="--",
                label="5% curve overlay",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_title(mode.replace("_", " "))
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("eligible-origin fraction")
    figure.suptitle("Finite-domain support geometry for the retained two-policy products")
    return _save(figure, output_dir, "phase4_batch_a2_3point_extension_support_vs_ell.png")


def policy_sensitivity_census(
    retained: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.7), constrained_layout=True, sharey=True
    )
    for axis, width in zip(axes, WIDTHS):
        for q_name, marker in zip(base.Q_NAMES, ("o", "s")):
            rows = [
                row
                for row in retained
                if row["width"] == width and row["q_name"] == q_name
            ]
            axis.scatter(
                [row["ell_cells"] for row in rows],
                [row["policy_factor"] for row in rows],
                marker=marker,
                s=11,
                alpha=0.25,
                label=q_name,
            )
        axis.axhline(1.0, color="#777777", linestyle="--")
        axis.axhline(1.5, color="#777777", linestyle=":")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(WIDTH_LABELS[width])
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"$F_{\rm policy}$")
    figure.suptitle(
        r"All-21-cube supported shell-local sensitivity for $\ell \geq 32$ cells"
    )
    return _save(
        figure,
        output_dir,
        "phase4_batch_a2_3point_extension_policy_sensitivity_census.png",
    )


def per_cube_maximum(
    policy_summary: Mapping[str, Any],
    cube_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        2, 1, figsize=(13.2, 7.8), constrained_layout=True, sharex=True
    )
    x = np.arange(len(cube_ids))
    by_key = {
        (row["width"], row["cube_id"]): row
        for row in policy_summary["per_cube_maximum"]
    }
    for axis, width in zip(axes, WIDTHS):
        maxima = [by_key[(width, cube_id)]["maximum_policy_factor"] for cube_id in cube_ids]
        tail_counts = [by_key[(width, cube_id)]["count_above_1p5"] for cube_id in cube_ids]
        axis.scatter(x, maxima, color=WIDTH_COLORS[width], s=38)
        for index, count in enumerate(tail_counts):
            if count:
                axis.text(index, maxima[index] * 1.035, str(count), ha="center", fontsize=7)
        axis.axhline(1.5, color="#777777", linestyle=":")
        axis.set_yscale("log")
        axis.set_ylabel(r"maximum $F_{\rm policy}$")
        axis.set_title(f"{WIDTH_LABELS[width]}: labels above points count retained bins with factor > 1.5")
        axis.grid(alpha=0.22)
    axes[-1].set_xticks(x, cube_ids, rotation=70, ha="right", fontsize=7)
    return _save(
        figure,
        output_dir,
        "phase4_batch_a2_3point_extension_per_cube_maximum_policy_factor.png",
    )


def highlighted_curves(
    groups: Mapping[tuple[str, int, str], base.Group],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        2, 2, figsize=(12.8, 9.0), constrained_layout=True
    )
    for axis, (cube_id, q_name, direction) in zip(axes.flat, HIGHLIGHTED_CURVES):
        for mode, color, linestyle in (
            ("all_valid_origins", "#4c78a8", "-"),
            ("shell_local", "#f58518", "--"),
        ):
            group = groups[(cube_id, 3, mode)]
            ell = figures._centers(group.result.ell_bin_edges)
            index = figures._moment_index(group.result, q_name, direction)
            mask = base._curve_support_mask(group, index)
            if mode == "shell_local":
                mask &= base._support_fraction(group.result) >= base.SHELL_CURVE_MINIMUM
            values = np.where(mask, group.result.moments[index], np.nan)
            low = np.where(
                mask, group.uncertainty["block_bootstrap_interval_low"][index], np.nan
            )
            high = np.where(
                mask, group.uncertainty["block_bootstrap_interval_high"][index], np.nan
            )
            valid = np.isfinite(values) & (values > 0.0)
            axis.plot(
                ell[valid],
                values[valid],
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                label=mode.replace("_", " "),
            )
            band = valid & np.isfinite(low) & np.isfinite(high) & (low > 0.0)
            axis.fill_between(ell[band], low[band], high[band], color=color, alpha=0.15)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_ylabel(rf"$S_2^{{{q_name}}}(\ell)$")
        axis.set_title(f"{cube_id}: {q_name} {direction}")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    figure.suptitle("Highlighted 3-point policy-sensitive curves with block-bootstrap bands")
    return _save(
        figure,
        output_dir,
        "phase4_batch_a2_3point_extension_highlighted_curves_with_uncertainty.png",
    )


def retained_vs_excluded(
    census: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.5), constrained_layout=True, sharey=True
    )
    fields = (
        ("retained_bins", "#4c78a8", "retained"),
        ("excluded_shell_support_bins", "#f58518", "excluded: shell support"),
        ("excluded_uncertainty_bins", "#e45756", "excluded: uncertainty"),
        ("excluded_nonfinite_ratio_bins", "#72b7b2", "excluded: nonfinite ratio"),
    )
    for axis, width in zip(axes, WIDTHS):
        rows = [row for row in census if row["width"] == width]
        counts = Counter()
        for row in rows:
            for field, _, _ in fields:
                counts[field] += row[field]
        left = 0
        for field, color, label in fields:
            axis.barh([0], [counts[field]], left=left, color=color, label=label)
            left += counts[field]
        axis.set_title(WIDTH_LABELS[width])
        axis.set_xlabel("directional science-scale bins")
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.22)
        axis.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figure.suptitle(r"Audit census for candidate directional bins at $\ell \geq 32$ cells")
    return _save(
        figure,
        output_dir,
        "phase4_batch_a2_3point_extension_retained_vs_excluded_bins.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-a-root", type=Path, required=True)
    parser.add_argument("--representative-a2-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
    input_hashes = base.InputHashes()
    batch_a = base._verify_release_marker(
        args.batch_a_root,
        summary_filename="phase4_batch_a_summary.json",
        marker_filename="PHASE4_BATCH_A_COMPLETE.json",
        input_hashes=input_hashes,
    )
    representative_a2 = base._verify_release_marker(
        args.representative_a2_root,
        summary_filename="phase4_batch_a2_summary.json",
        marker_filename="PHASE4_BATCH_A2_COMPLETE.json",
        input_hashes=input_hashes,
    )
    extension = base._verify_release_marker(
        args.extension_root,
        summary_filename="phase4_batch_a2_3point_extension_summary.json",
        marker_filename="PHASE4_BATCH_A2_3POINT_EXTENSION_COMPLETE.json",
        input_hashes=input_hashes,
    )
    cube_ids = tuple(extension.get("pilot_cube_ids", ()))
    if (
        batch_a.get("phase") != "phase4_batch_a_bounded_21_cube_2point"
        or representative_a2.get("phase")
        != "phase4_batch_a2_bounded_4_cube_3point_5point"
        or extension.get("phase") != "phase4_batch_a2_all21_3point_extension"
        or tuple(batch_a.get("pilot_cube_ids", ())) != cube_ids
        or len(cube_ids) != 21
        or Path(extension.get("batch_a_reference_root", "")).resolve()
        != args.batch_a_root.resolve()
        or Path(extension.get("representative_a2_reference_root", "")).resolve()
        != args.representative_a2_root.resolve()
    ):
        raise RuntimeError("all-21 extension is not bound to the retained staged predecessors")
    input_hashes.add(args.ledger_summary)
    groups = _load_groups(args.batch_a_root, args.extension_root, cube_ids, input_hashes)
    overlap_audit = _verify_representative_overlap(
        args.representative_a2_root, args.extension_root, input_hashes
    )
    retained, census = _curve_rows(groups, cube_ids)
    policy_summary = _policy_summary(retained, cube_ids)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copyfile(args.ledger_summary, temporary_output / LEDGER_SNAPSHOT_FILENAME)
        support_vs_ell(groups, cube_ids, temporary_output)
        policy_sensitivity_census(retained, temporary_output)
        per_cube_maximum(policy_summary, cube_ids, temporary_output)
        highlighted_curves(groups, temporary_output)
        retained_vs_excluded(census, temporary_output)
        summary = {
            "schema_version": 1,
            "status": "phase4_batch_a2_all21_3point_extension_review_generated",
            "decision_scope": "all-21-cube 3-point acquisition review before Batch B",
            "automatic_batch_b_expansion_claimed": False,
            "batch_a_root": str(args.batch_a_root.resolve()),
            "representative_a2_root": str(args.representative_a2_root.resolve()),
            "extension_root": str(args.extension_root.resolve()),
            "cube_ids": cube_ids,
            "configuration": {
                "primary_curve_product": "all_valid_origins",
                "directional_robustness_overlay": "shell_local",
                "shell_local_curve_overlay_minimum_fraction": base.SHELL_CURVE_MINIMUM,
                "science_scale_minimum_cells_for_policy_census": base.SCIENCE_SCALE_MINIMUM,
                "directional_fitted_exponents_published": False,
                "stencil_labels_remain_distinct": True,
            },
            "supported_curve_policy_factor": policy_summary,
            "support_census": _census_summary(census),
            "support_census_by_curve": census,
            "highlighted_curves": HIGHLIGHTED_CURVES,
            "representative_overlap_audit": overlap_audit,
            "operational_result": {
                "strict_release_verification_completed_before_report": True,
                "verified_shards": 168,
                "verified_reductions": 42,
                "recommendation": (
                    "Review the all-21 3-point sensitivity tail before Batch B. Keep 5-point "
                    "bounded. Do not promote curve-level diagnostics to fitted exponents."
                ),
            },
            "input_sha256": input_hashes.as_dict(),
        }
        _write_json(temporary_output / SUMMARY_FILENAME, summary)
        figures_out = sorted(path.name for path in temporary_output.glob("*.png"))
        artifacts = sorted(path.name for path in temporary_output.iterdir())
        _write_json(
            temporary_output / "figure_manifest.json",
            {
                "schema_version": 1,
                "status": "passed",
                "publication_policy": "immutable_one_time_publish_refuse_existing_output_directory",
                "generator_sha256": file_sha256(Path(__file__).resolve()),
                "generated_figures": figures_out,
                "figure_sha256": {
                    name: file_sha256(temporary_output / name) for name in figures_out
                },
                "generated_artifacts_before_manifest": artifacts,
                "artifact_sha256": {
                    name: file_sha256(temporary_output / name) for name in artifacts
                },
                "input_sha256": input_hashes.as_dict(),
            },
        )
        temporary_output.rename(output_dir)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    print(f"Wrote immutable Phase 4 all-21 3-point extension review package: {output_dir}")


if __name__ == "__main__":
    main()
