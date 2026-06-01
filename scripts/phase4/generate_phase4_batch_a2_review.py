#!/usr/bin/env python3
"""Publish the immutable Phase 4 Batch A2 representative-stencil review."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
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
from sfunctor.analysis.phase3a import load_finite_domain_partial_npz


DEFAULT_OUTPUT_DIR = Path("figures/phase4_batch_a2_review")
SUMMARY_FILENAME = "phase4_batch_a2_review_summary.json"
LEDGER_SNAPSHOT_FILENAME = "phase4_batch_a2_compute_ledger_summary_snapshot.md"
REPRESENTATIVE_CUBES = {
    "L640_sub00370": "low dBB",
    "L640_sub03942": "median dBB",
    "L640_sub00579": "high dBB",
    "L640_sub00738": "weak mean field",
}
WIDTHS = (2, 3, 5)
A2_WIDTHS = (3, 5)
SUPPORT_MODES = ("all_valid_origins", "shell_local")
Q_NAMES = ("B", "u")
DIRECTIONS = ("parallel", "xi", "lambda")
WIDTH_COLORS = {2: "#4c78a8", 3: "#f58518", 5: "#54a24b"}
WIDTH_LABELS = {width: f"{width}-point" for width in WIDTHS}
SCIENCE_SCALE_MINIMUM = 32.0
SHELL_CURVE_MINIMUM = 0.05
SHELL_SLOPE_CANDIDATE_MINIMUM = 0.10
BOOTSTRAP_N_RESAMPLES = 200
MINIMUM_VALID_BOOTSTRAP_FRACTION = 0.90


@dataclass(frozen=True)
class Group:
    result: Any
    uncertainty: dict[str, np.ndarray]


class InputHashes:
    def __init__(self) -> None:
        self._rows: dict[str, str] = {}

    def add(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"required retained artifact is missing: {resolved}")
        self._rows[str(resolved)] = file_sha256(resolved)

    def as_dict(self) -> dict[str, str]:
        return dict(sorted(self._rows.items()))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _verify_release_marker(
    root: Path,
    *,
    summary_filename: str,
    marker_filename: str,
    input_hashes: InputHashes,
) -> dict[str, Any]:
    summary_path = root / summary_filename
    marker_path = root / marker_filename
    summary = _load_json(summary_path)
    marker = _load_json(marker_path)
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "release_aggregation_complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or summary.get("source_version", {}).get("implementation_sha256")
        != marker.get("implementation_sha256")
    ):
        raise RuntimeError(f"stale retained release publication: {root}")
    input_hashes.add(summary_path)
    input_hashes.add(marker_path)
    return summary


def _group_root(root: Path, cube_id: str, width: int, support_mode: str) -> Path:
    return root / "reductions" / cube_id / f"stencil_{width}point" / support_mode


def _load_group(
    root: Path,
    cube_id: str,
    width: int,
    support_mode: str,
    input_hashes: InputHashes,
) -> Group:
    group_root = _group_root(root, cube_id, width, support_mode)
    result_path = group_root / "result.npz"
    uncertainty_path = group_root / "uncertainty.npz"
    manifest_path = group_root / "reduction_manifest.json"
    marker_path = group_root / "COMPLETE.json"
    marker = _load_json(marker_path)
    expected_group_id = f"{cube_id}/stencil_{width}point/{support_mode}"
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("group_id") != expected_group_id
        or marker.get("result_sha256") != file_sha256(result_path)
        or marker.get("uncertainty_sha256") != file_sha256(uncertainty_path)
        or marker.get("reduction_manifest_sha256") != file_sha256(manifest_path)
    ):
        raise RuntimeError(f"stale retained reduction publication: {expected_group_id}")
    result = load_finite_domain_partial_npz(result_path)
    with np.load(uncertainty_path, allow_pickle=False) as payload:
        uncertainty = {name: payload[name].copy() for name in payload.files}
    if result.stencil_width != width or result.pair_mode != support_mode:
        raise RuntimeError(f"reduction metadata mismatch: {expected_group_id}")
    for path in (result_path, uncertainty_path, manifest_path, marker_path):
        input_hashes.add(path)
    return Group(result=result, uncertainty=uncertainty)


def _load_groups(
    batch_a_root: Path,
    batch_a2_root: Path,
    input_hashes: InputHashes,
) -> dict[tuple[str, int, str], Group]:
    groups: dict[tuple[str, int, str], Group] = {}
    for cube_id in REPRESENTATIVE_CUBES:
        for width in WIDTHS:
            root = batch_a_root if width == 2 else batch_a2_root
            for support_mode in SUPPORT_MODES:
                groups[(cube_id, width, support_mode)] = _load_group(
                    root, cube_id, width, support_mode, input_hashes
                )
    return groups


def _support_fraction(result: Any) -> np.ndarray:
    return np.divide(
        result.eligible_pairs,
        result.cube_candidate_pairs,
        out=np.zeros_like(result.eligible_pairs, dtype=float),
        where=result.cube_candidate_pairs > 0,
    )


def _curve_support_mask(group: Group, index: tuple[int, ...]) -> np.ndarray:
    uncertainty = group.uncertainty
    minimum_valid = int(
        math.ceil(MINIMUM_VALID_BOOTSTRAP_FRACTION * BOOTSTRAP_N_RESAMPLES)
    )
    return (
        np.isfinite(group.result.moments[index])
        & (group.result.counts[index] >= 2)
        & (uncertainty["accepted_contributing_blocks"][index] >= 2)
        & (uncertainty["accepted_effective_blocks"][index] >= 8.0)
        & (uncertainty["valid_bootstrap_resamples"][index] >= minimum_valid)
        & np.isfinite(uncertainty["block_bootstrap_interval_low"][index])
        & np.isfinite(uncertainty["block_bootstrap_interval_high"][index])
    )


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _factor(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "median": float(np.median(array)) if array.size else None,
        "p90": float(np.quantile(array, 0.90)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
    }


def _policy_factor_summary(
    groups: Mapping[tuple[str, int, str], Group],
) -> dict[str, Any]:
    output = {}
    for width in WIDTHS:
        factors: list[float] = []
        for cube_id in REPRESENTATIVE_CUBES:
            primary = groups[(cube_id, width, "all_valid_origins")]
            overlay = groups[(cube_id, width, "shell_local")]
            ell = figures._centers(primary.result.ell_bin_edges)
            shell_support = _support_fraction(overlay.result)
            for q_name in Q_NAMES:
                for direction in DIRECTIONS:
                    index = figures._moment_index(primary.result, q_name, direction)
                    numerator = primary.result.moments[index]
                    denominator = overlay.result.moments[index]
                    ratio = np.divide(
                        numerator,
                        denominator,
                        out=np.full_like(numerator, np.nan, dtype=float),
                        where=np.isfinite(denominator) & (denominator != 0.0),
                    )
                    mask = (
                        (ell >= SCIENCE_SCALE_MINIMUM)
                        & (shell_support >= SHELL_CURVE_MINIMUM)
                        & _curve_support_mask(primary, index)
                        & _curve_support_mask(overlay, index)
                        & np.isfinite(ratio)
                        & (ratio > 0.0)
                    )
                    factors.extend(np.maximum(ratio[mask], 1.0 / ratio[mask]).tolist())
        output[str(width)] = _factor(factors)
    return output


def _support_summary(
    groups: Mapping[tuple[str, int, str], Group],
) -> dict[str, Any]:
    output = {}
    cube_id = next(iter(REPRESENTATIVE_CUBES))
    for width in WIDTHS:
        shell = groups[(cube_id, width, "shell_local")].result
        ell = figures._centers(shell.ell_bin_edges)
        fraction = _support_fraction(shell)
        endpoints = {}
        for cutoff in (0.01, 0.05, 0.10):
            accepted = ell[fraction >= cutoff]
            endpoints[f"{cutoff:.2f}"] = float(np.max(accepted)) if accepted.size else None
        output[str(width)] = {
            "minimum_shell_local_fraction": float(np.min(fraction)),
            "maximum_shell_center_cells": float(np.max(ell)),
            "largest_shell_center_cells_by_minimum_fraction": endpoints,
        }
    return output


def support_vs_ell(
    groups: Mapping[tuple[str, int, str], Group], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True, sharey=True)
    cube_id = next(iter(REPRESENTATIVE_CUBES))
    for axis, mode in zip(axes, SUPPORT_MODES):
        for width in WIDTHS:
            result = groups[(cube_id, width, mode)].result
            ell = figures._centers(result.ell_bin_edges)
            axis.plot(
                ell,
                _support_fraction(result),
                color=WIDTH_COLORS[width],
                linewidth=1.8,
                label=WIDTH_LABELS[width],
            )
        if mode == "shell_local":
            axis.axhline(SHELL_CURVE_MINIMUM, color="#777777", linestyle="--", label="5% curve overlay")
            axis.axhline(SHELL_SLOPE_CANDIDATE_MINIMUM, color="#777777", linestyle=":", label="10% slope candidate")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.set_title(mode.replace("_", " "))
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("eligible-origin fraction")
    figure.suptitle("Representative finite-domain support by labeled stencil and policy")
    return _save(figure, output_dir, "phase4_batch_a2_support_vs_ell.png")


def representative_curves(
    groups: Mapping[tuple[str, int, str], Group], output_dir: Path, q_name: str
) -> Path:
    figure, axes = plt.subplots(
        len(REPRESENTATIVE_CUBES),
        len(DIRECTIONS),
        figsize=(13.3, 11.0),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )
    for row, (cube_id, role) in enumerate(REPRESENTATIVE_CUBES.items()):
        for column, direction in enumerate(DIRECTIONS):
            axis = axes[row, column]
            for width in WIDTHS:
                primary = groups[(cube_id, width, "all_valid_origins")]
                overlay = groups[(cube_id, width, "shell_local")]
                ell = figures._centers(primary.result.ell_bin_edges)
                index = figures._moment_index(primary.result, q_name, direction)
                primary_mask = _curve_support_mask(primary, index)
                overlay_mask = (
                    (_support_fraction(overlay.result) >= SHELL_CURVE_MINIMUM)
                    & _curve_support_mask(overlay, index)
                )
                figures._plot_positive_curve(
                    axis,
                    ell,
                    np.where(primary_mask, primary.result.moments[index], np.nan),
                    label=f"{width}-point primary",
                    color=WIDTH_COLORS[width],
                )
                figures._plot_positive_curve(
                    axis,
                    ell,
                    np.where(overlay_mask, overlay.result.moments[index], np.nan),
                    label=f"{width}-point shell overlay",
                    color=WIDTH_COLORS[width],
                    linestyle="--",
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(direction)
            axis.grid(alpha=0.22)
            axis.set_xlabel(r"$\ell$ [cells]")
            if column == 0:
                axis.set_ylabel(f"{role}: {cube_id}\n" + rf"$S_{{2,\perp}}^{{{q_name}}}(\ell)$")
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    figure.suptitle(
        rf"Representative labeled-stencil ${q_name}$ curves: solid primary; dashed supported shell overlay"
    )
    return _save(figure, output_dir, f"phase4_batch_a2_{q_name}_representative_curves.png")


def policy_ratio_census(
    groups: Mapping[tuple[str, int, str], Group], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(1, len(WIDTHS), figsize=(14.0, 4.5), constrained_layout=True, sharey=True)
    for axis, width in zip(axes, WIDTHS):
        for q_name, marker in zip(Q_NAMES, ("o", "s")):
            values = []
            for cube_id in REPRESENTATIVE_CUBES:
                primary = groups[(cube_id, width, "all_valid_origins")]
                overlay = groups[(cube_id, width, "shell_local")]
                ell = figures._centers(primary.result.ell_bin_edges)
                support = _support_fraction(overlay.result)
                for direction in DIRECTIONS:
                    index = figures._moment_index(primary.result, q_name, direction)
                    ratio = np.divide(
                        primary.result.moments[index],
                        overlay.result.moments[index],
                        out=np.full_like(ell, np.nan, dtype=float),
                        where=np.isfinite(overlay.result.moments[index])
                        & (overlay.result.moments[index] != 0.0),
                    )
                    mask = (
                        (ell >= SCIENCE_SCALE_MINIMUM)
                        & (support >= SHELL_CURVE_MINIMUM)
                        & _curve_support_mask(primary, index)
                        & _curve_support_mask(overlay, index)
                        & np.isfinite(ratio)
                        & (ratio > 0.0)
                    )
                    factor = np.maximum(ratio[mask], 1.0 / ratio[mask])
                    axis.scatter(
                        ell[mask],
                        factor,
                        marker=marker,
                        s=12,
                        alpha=0.28,
                        label=q_name if direction == DIRECTIONS[0] and cube_id == next(iter(REPRESENTATIVE_CUBES)) else "",
                    )
                    values.extend(factor.tolist())
            summary = _factor(values)
            axis.text(
                0.04,
                0.95 if q_name == "B" else 0.82,
                f"{q_name}: median={summary['median']:.3f}x; p90={summary['p90']:.3f}x",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
            )
        axis.axhline(1.0, color="#777777", linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(WIDTH_LABELS[width])
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel(r"$\max(S_{\rm primary}/S_{\rm shell}, S_{\rm shell}/S_{\rm primary})$")
    axes[0].legend(fontsize=8)
    figure.suptitle(r"Supported shell-local sensitivity for $\ell \geq 32$ cells; factors are diagnostics, not corrections")
    return _save(figure, output_dir, "phase4_batch_a2_policy_sensitivity_census.png")


def local_slope_diagnostic(
    groups: Mapping[tuple[str, int, str], Group], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(
        len(REPRESENTATIVE_CUBES),
        len(Q_NAMES),
        figsize=(12.0, 10.4),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for row, (cube_id, role) in enumerate(REPRESENTATIVE_CUBES.items()):
        for column, q_name in enumerate(Q_NAMES):
            axis = axes[row, column]
            for width in WIDTHS:
                for mode, linestyle in (("all_valid_origins", "-"), ("shell_local", "--")):
                    group = groups[(cube_id, width, mode)]
                    index = figures._moment_index(group.result, q_name, "lambda")
                    ell = figures._centers(group.result.ell_bin_edges)
                    support = group.uncertainty["local_log_slope_support_mask"][index]
                    if mode == "shell_local":
                        support = support & (
                            _support_fraction(group.result) >= SHELL_SLOPE_CANDIDATE_MINIMUM
                        )
                    slope = np.where(
                        support,
                        group.uncertainty["local_log_slope"][index],
                        np.nan,
                    )
                    valid = np.isfinite(ell) & np.isfinite(slope) & (ell > 0.0)
                    axis.plot(
                        ell[valid],
                        slope[valid],
                        label=f"{width}-point {mode.replace('_origins', '').replace('_', ' ')}",
                        color=WIDTH_COLORS[width],
                        linestyle=linestyle,
                        linewidth=1.5,
                    )
            axis.set_xscale("log")
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.grid(alpha=0.22)
            if row == 0:
                axis.set_title(q_name)
            if column == 0:
                axis.set_ylabel(f"{role}: {cube_id}\nlocal log slope")
    axes[0, 0].legend(fontsize=6.3, ncol=2)
    figure.suptitle(r"$\lambda$-wedge centered five-bin local slopes: diagnostics only, not fitted exponents")
    return _save(figure, output_dir, "phase4_batch_a2_lambda_local_slope_diagnostics.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-a-root", type=Path, required=True)
    parser.add_argument("--batch-a2-root", type=Path, required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
    input_hashes = InputHashes()
    batch_a = _verify_release_marker(
        args.batch_a_root,
        summary_filename="phase4_batch_a_summary.json",
        marker_filename="PHASE4_BATCH_A_COMPLETE.json",
        input_hashes=input_hashes,
    )
    batch_a2 = _verify_release_marker(
        args.batch_a2_root,
        summary_filename="phase4_batch_a2_summary.json",
        marker_filename="PHASE4_BATCH_A2_COMPLETE.json",
        input_hashes=input_hashes,
    )
    if (
        batch_a.get("phase") != "phase4_batch_a_bounded_21_cube_2point"
        or batch_a2.get("phase") != "phase4_batch_a2_bounded_4_cube_3point_5point"
        or tuple(batch_a2.get("representative_cube_ids", ())) != tuple(REPRESENTATIVE_CUBES)
        or Path(batch_a2.get("batch_a_reference_root", "")).resolve()
        != args.batch_a_root.resolve()
    ):
        raise RuntimeError("Batch A2 release is not bound to the expected Batch A baseline")
    input_hashes.add(args.ledger_summary)
    groups = _load_groups(args.batch_a_root, args.batch_a2_root, input_hashes)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copyfile(args.ledger_summary, temporary_output / LEDGER_SNAPSHOT_FILENAME)
        support_vs_ell(groups, temporary_output)
        for q_name in Q_NAMES:
            representative_curves(groups, temporary_output, q_name)
        policy_ratio_census(groups, temporary_output)
        local_slope_diagnostic(groups, temporary_output)
        summary = {
            "schema_version": 1,
            "status": "phase4_batch_a2_representative_review_generated",
            "decision_scope": "bounded representative 3-point and 5-point comparison only",
            "automatic_expansion_claimed": False,
            "batch_a_root": str(args.batch_a_root.resolve()),
            "batch_a2_root": str(args.batch_a2_root.resolve()),
            "representative_cubes": REPRESENTATIVE_CUBES,
            "configuration": {
                "primary_curve_product": "all_valid_origins",
                "directional_robustness_overlay": "shell_local",
                "shell_local_curve_overlay_minimum_fraction": SHELL_CURVE_MINIMUM,
                "shell_local_slope_candidate_minimum_fraction": SHELL_SLOPE_CANDIDATE_MINIMUM,
                "science_scale_minimum_cells_for_policy_census": SCIENCE_SCALE_MINIMUM,
                "stencil_labels_remain_distinct": True,
                "directional_fitted_exponents_published": False,
            },
            "support_geometry": _support_summary(groups),
            "supported_curve_policy_factor": _policy_factor_summary(groups),
            "operational_result": {
                "strict_release_verification_completed_before_report": True,
                "verified_shards": 64,
                "verified_reductions": 16,
                "recommendation": (
                    "The representative 3-point comparison is ready for a separate human "
                    "decision on 21-cube expansion. Keep 5-point bounded to representative "
                    "cubes. Do not promote local slopes to fitted exponents."
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
                "input_sha256": input_hashes.as_dict(),
            },
        )
        temporary_output.rename(output_dir)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    print(f"Wrote immutable Phase 4 Batch A2 review package: {output_dir}")


if __name__ == "__main__":
    main()
