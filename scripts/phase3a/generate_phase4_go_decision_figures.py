#!/usr/bin/env python3
"""Generate the Phase 4 launch-GO evidence package from retained Phase 3a artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3.run_phase3_sampler import BENCHMARK_CUBE_IDS
from scripts.phase3a import generate_phase3a_status_figures as phase3_figures


DEFAULT_OUTPUT_DIR = Path("figures/phase4_go_decision")
CANDIDATE_SHELL_SUPPORT_FRACTIONS = (0.01, 0.05, 0.10)
APPROVED_SHELL_CURVE_OVERLAY_FRACTION = 0.05
MINIMUM_DIRECTIONAL_SLOPE_CANDIDATE_FRACTION = 0.10
SCIENCE_SCALE_MINIMUM = 32.0
Q_NAMES = ("B", "u")
DIRECTION_NAMES = ("parallel", "xi", "lambda")
SUPPORT_MODE_LABELS = {
    "shell_local": "shell-local",
    "all_valid_origins": "all valid origins",
}
SUPPORT_MODE_COLORS = {
    "shell_local": "#4c78a8",
    "all_valid_origins": "#54a24b",
}


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _support_fraction(result: Any) -> np.ndarray:
    return np.divide(
        result.eligible_pairs,
        result.cube_candidate_pairs,
        out=np.zeros_like(result.eligible_pairs, dtype=float),
        where=result.cube_candidate_pairs > 0,
    )


def _positive_factor(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    return np.maximum(values[valid], 1.0 / values[valid])


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def _load_groups(
    verified: phase3_figures.VerifiedInputs,
    input_hashes: phase3_figures.InputHashes,
) -> dict[tuple[str, int, str], phase3_figures.ReleaseGroup]:
    groups = {}
    for cube_id in BENCHMARK_CUBE_IDS:
        for stencil_width in (2, 3, 5):
            for support_mode in ("shell_local", "all_valid_origins"):
                key = (cube_id, stencil_width, support_mode)
                groups[key] = phase3_figures._load_release_group(
                    verified,
                    input_hashes,
                    cube_id=cube_id,
                    stencil_width=stencil_width,
                    support_mode=support_mode,
                )
    return groups


def _cutoff_endpoint(
    ell: np.ndarray,
    support: np.ndarray,
    cutoff: float,
) -> float | None:
    selected = np.isfinite(ell) & np.isfinite(support) & (support >= cutoff)
    return float(np.max(ell[selected])) if np.any(selected) else None


def _complete_centered_window_mask(mask: np.ndarray, *, window: int = 5) -> np.ndarray:
    """Return centers whose complete odd-width window satisfies ``mask``."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or window < 3 or window % 2 == 0 or window > mask.size:
        raise ValueError("window mask requires a fitting odd width for one shell axis")
    output = np.zeros_like(mask)
    radius = window // 2
    for center in range(radius, mask.size - radius):
        output[center] = bool(np.all(mask[center - radius : center + radius + 1]))
    return output


def _last_masked_endpoint(ell: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.isfinite(ell) & np.asarray(mask, dtype=bool)
    return float(np.max(ell[selected])) if np.any(selected) else None


def build_summary(
    verified: phase3_figures.VerifiedInputs,
    groups: dict[tuple[str, int, str], phase3_figures.ReleaseGroup],
) -> dict[str, Any]:
    representative_cube = BENCHMARK_CUBE_IDS[0]
    geometry_summary = {}
    for stencil_width in (2, 3, 5):
        reference = groups[(representative_cube, stencil_width, "shell_local")].result
        ell = phase3_figures._centers(reference.ell_bin_edges)
        support = _support_fraction(reference)
        for cube_id in BENCHMARK_CUBE_IDS[1:]:
            other = groups[(cube_id, stencil_width, "shell_local")].result
            if not np.array_equal(reference.ell_bin_edges, other.ell_bin_edges):
                raise RuntimeError(f"shell-local ell grid varies by cube for stencil {stencil_width}")
            if not np.array_equal(reference.eligible_pairs, other.eligible_pairs):
                raise RuntimeError(f"shell-local eligible origins vary by cube for stencil {stencil_width}")
            if not np.array_equal(reference.cube_candidate_pairs, other.cube_candidate_pairs):
                raise RuntimeError(f"shell-local candidate origins vary by cube for stencil {stencil_width}")
        geometry_summary[str(stencil_width)] = {
            "minimum_shell_local_support_fraction": float(np.min(support)),
            "candidate_cutoff_endpoints_cells": {
                f"{cutoff:.2f}": _cutoff_endpoint(ell, support, cutoff)
                for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS
            },
            "candidate_cutoff_full_five_bin_slope_window_endpoints_cells": {
                f"{cutoff:.2f}": _last_masked_endpoint(
                    ell,
                    _complete_centered_window_mask(support >= cutoff),
                )
                for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS
            },
        }

    cutoff_metrics = {}
    for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS:
        curve_factors = []
        slope_differences = []
        for cube_id in BENCHMARK_CUBE_IDS:
            shell = groups[(cube_id, 2, "shell_local")]
            all_valid = groups[(cube_id, 2, "all_valid_origins")]
            ell = phase3_figures._centers(shell.result.ell_bin_edges)
            support = _support_fraction(shell.result)
            scale_mask = (
                np.isfinite(ell)
                & np.isfinite(support)
                & (ell >= SCIENCE_SCALE_MINIMUM)
                & (support >= cutoff)
            )
            slope_geometric_mask = _complete_centered_window_mask(support >= cutoff)
            for q_name in Q_NAMES:
                for direction_name in DIRECTION_NAMES:
                    index = phase3_figures._moment_index(
                        shell.result,
                        q_name,
                        direction_name,
                    )
                    ratio = np.divide(
                        all_valid.result.moments[index],
                        shell.result.moments[index],
                        out=np.full_like(shell.result.moments[index], np.nan, dtype=float),
                        where=np.isfinite(shell.result.moments[index])
                        & (shell.result.moments[index] != 0.0),
                    )
                    curve_factors.extend(_positive_factor(ratio[scale_mask]).tolist())
                    shell_slope = shell.uncertainty["local_log_slope"][index]
                    all_valid_slope = all_valid.uncertainty["local_log_slope"][index]
                    slope_mask = (
                        (ell >= SCIENCE_SCALE_MINIMUM)
                        & slope_geometric_mask
                        & shell.uncertainty["local_log_slope_support_mask"][index]
                        & all_valid.uncertainty["local_log_slope_support_mask"][index]
                        & np.isfinite(shell_slope)
                        & np.isfinite(all_valid_slope)
                    )
                    slope_differences.extend(
                        np.abs(all_valid_slope[slope_mask] - shell_slope[slope_mask]).tolist()
                    )
        cutoff_metrics[f"{cutoff:.2f}"] = {
            "curve_policy_factor": _finite_summary(np.asarray(curve_factors)),
            "absolute_local_slope_difference": _finite_summary(
                np.asarray(slope_differences)
            ),
        }

    return {
        "schema_version": 1,
        "status": "phase4_batch_a_launch_go",
        "decision_scope": "Phase 4 Batch A launch and bounded downstream staging only",
        "implementation_sha256": verified.implementation_sha256,
        "retained_release_root": str(verified.release_root),
        "science_scale_minimum_cells_for_policy_census": SCIENCE_SCALE_MINIMUM,
        "candidate_shell_support_fractions": list(CANDIDATE_SHELL_SUPPORT_FRACTIONS),
        "approved_policy": {
            "primary_curve_support_mode": "all_valid_origins",
            "required_directional_robustness_overlay": "shell_local",
            "minimum_shell_local_fraction_for_science_facing_curve_overlay": (
                APPROVED_SHELL_CURVE_OVERLAY_FRACTION
            ),
            "minimum_shell_local_fraction_for_directional_slope_table_candidate": (
                MINIMUM_DIRECTIONAL_SLOPE_CANDIDATE_FRACTION
            ),
            "directional_slope_tables_authorized_for_batch_a": False,
            "reporting_mode": "curve_first",
            "support_cutoffs_are": "conservative reporting-policy choices from bounded equal-bin diagnostics, not physical admissibility thresholds",
            "weak_shell_local_values": "retain as visibly flagged diagnostics only",
        },
        "approved_staged_matrix": {
            "batch_a": {
                "cube_count": 21,
                "q_names": list(Q_NAMES),
                "p_values": [2],
                "stencils": [2],
                "support_modes": ["all_valid_origins", "shell_local"],
            },
            "batch_a2_initial_planned_after_separate_post_batch_a_review": {
                "authorized_for_submission": False,
                "requires": "separate post-Batch-A review and explicit human approval",
                "cube_ids": list(BENCHMARK_CUBE_IDS),
                "q_names": list(Q_NAMES),
                "p_values": [2],
                "stencils": [3, 5],
                "support_modes": ["all_valid_origins", "shell_local"],
            },
            "batch_a2_conditional_expansion": {
                "authorized_for_submission": False,
                "stencils": [3],
                "cube_count": 21,
                "condition": "expand only if representative-cube review finds an informative and stable labeled comparison",
            },
            "five_point_expansion": {
                "authorized_for_submission": False,
                "condition": "keep bounded to representative cases until a later explicit review",
            },
        },
        "shell_local_geometry": geometry_summary,
        "candidate_cutoff_metrics": cutoff_metrics,
    }


def support_cutoff_diagnostic(
    groups: dict[tuple[str, int, str], phase3_figures.ReleaseGroup],
    output_dir: Path,
) -> Path:
    representative_cube = BENCHMARK_CUBE_IDS[0]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    for stencil_width in (2, 3, 5):
        result = groups[(representative_cube, stencil_width, "shell_local")].result
        ell = phase3_figures._centers(result.ell_bin_edges)
        support = _support_fraction(result)
        axes[0].plot(
            ell,
            support,
            color=phase3_figures.STENCIL_COLORS[stencil_width],
            label=phase3_figures.STENCIL_LABELS[stencil_width],
        )
        endpoints = [
            _cutoff_endpoint(ell, support, cutoff)
            for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS
        ]
        axes[1].plot(
            np.asarray(CANDIDATE_SHELL_SUPPORT_FRACTIONS) * 100.0,
            endpoints,
            "o-",
            color=phase3_figures.STENCIL_COLORS[stencil_width],
            label=phase3_figures.STENCIL_LABELS[stencil_width],
        )
    for cutoff, linestyle in zip(CANDIDATE_SHELL_SUPPORT_FRACTIONS, (":", "--", "-.")):
        axes[0].axhline(
            cutoff,
            color="#777777",
            linestyle=linestyle,
            linewidth=1.0,
            label=f"{cutoff:.0%} candidate cutoff",
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\ell$ [cells]")
    axes[0].set_ylabel("minimum shell-local eligible-origin fraction")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("candidate shell-local support cutoff [%]")
    axes[1].set_ylabel("largest retained shell center [cells]")
    axes[1].set_xticks(np.asarray(CANDIDATE_SHELL_SUPPORT_FRACTIONS) * 100.0)
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)
    figure.suptitle(
        "Phase 4 launch gate: shell-local support cutoff census from verified four-cube geometry"
    )
    return _save(figure, output_dir, "phase4_go_shell_local_cutoff_census.png")


def curve_policy_ratio_diagnostic(
    groups: dict[tuple[str, int, str], phase3_figures.ReleaseGroup],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        len(BENCHMARK_CUBE_IDS),
        len(Q_NAMES),
        figsize=(12.0, 12.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for row, cube_id in enumerate(BENCHMARK_CUBE_IDS):
        shell = groups[(cube_id, 2, "shell_local")].result
        all_valid = groups[(cube_id, 2, "all_valid_origins")].result
        ell = phase3_figures._centers(shell.ell_bin_edges)
        support = _support_fraction(shell)
        weak = support < APPROVED_SHELL_CURVE_OVERLAY_FRACTION
        for column, q_name in enumerate(Q_NAMES):
            axis = axes[row, column]
            for direction_name in DIRECTION_NAMES:
                index = phase3_figures._moment_index(shell, q_name, direction_name)
                ratio = np.divide(
                    all_valid.moments[index],
                    shell.moments[index],
                    out=np.full_like(shell.moments[index], np.nan, dtype=float),
                    where=np.isfinite(shell.moments[index]) & (shell.moments[index] != 0.0),
                )
                axis.plot(
                    ell,
                    np.where(weak, np.nan, ratio),
                    color=phase3_figures.DIRECTION_COLORS[direction_name],
                    label=direction_name,
                )
                axis.plot(
                    ell[weak],
                    ratio[weak],
                    color="#888888",
                    linestyle="none",
                    marker="x",
                    markersize=3.0,
                )
            axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(alpha=0.22)
            axis.set_xlabel(r"$\ell$ [cells]")
            axis.set_ylabel(
                f"{phase3_figures.SHORT_LABELS[cube_id]}\n"
                + rf"$S_{{2,\perp}}^{{{q_name},\mathrm{{all}}}} / S_{{2,\perp}}^{{{q_name},\mathrm{{shell}}}}$"
            )
    axes[0, 0].legend(fontsize=7)
    figure.suptitle(
        "Phase 4 launch gate: 2-point curve-policy sensitivity; gray x: shell-local support <5%"
    )
    return _save(figure, output_dir, "phase4_go_2point_curve_policy_ratios.png")


def slope_policy_comparison(
    groups: dict[tuple[str, int, str], phase3_figures.ReleaseGroup],
    output_dir: Path,
    *,
    q_name: str,
) -> Path:
    figure, axes = plt.subplots(
        len(BENCHMARK_CUBE_IDS),
        len(DIRECTION_NAMES),
        figsize=(13.0, 11.0),
        constrained_layout=True,
        sharex=True,
    )
    for row, cube_id in enumerate(BENCHMARK_CUBE_IDS):
        shell = groups[(cube_id, 2, "shell_local")]
        all_valid = groups[(cube_id, 2, "all_valid_origins")]
        ell = phase3_figures._centers(shell.result.ell_bin_edges)
        support = _support_fraction(shell.result)
        slope_geometric_support = _complete_centered_window_mask(
            support >= APPROVED_SHELL_CURVE_OVERLAY_FRACTION
        )
        last_supported = _last_masked_endpoint(ell, slope_geometric_support)
        first_weak = (
            float(np.min(ell[ell > last_supported]))
            if last_supported is not None and np.any(ell > last_supported)
            else None
        )
        for column, direction_name in enumerate(DIRECTION_NAMES):
            axis = axes[row, column]
            index = phase3_figures._moment_index(shell.result, q_name, direction_name)
            for group, support_mode, linestyle in (
                (all_valid, "all_valid_origins", "--"),
                (shell, "shell_local", "-"),
            ):
                uncertainty = group.uncertainty
                slope = uncertainty["local_log_slope"][index]
                low = uncertainty["local_log_slope_bootstrap_interval_low"][index]
                high = uncertainty["local_log_slope_bootstrap_interval_high"][index]
                valid_band = np.isfinite(ell) & np.isfinite(low) & np.isfinite(high)
                if support_mode == "shell_local":
                    valid_band &= slope_geometric_support
                    slope = np.where(slope_geometric_support, slope, np.nan)
                axis.fill_between(
                    ell[valid_band],
                    low[valid_band],
                    high[valid_band],
                    color=SUPPORT_MODE_COLORS[support_mode],
                    alpha=0.10,
                )
                axis.plot(
                    ell,
                    slope,
                    color=SUPPORT_MODE_COLORS[support_mode],
                    linestyle=linestyle,
                    label=SUPPORT_MODE_LABELS[support_mode],
                )
            if first_weak is not None:
                axis.axvspan(first_weak, float(np.max(ell)), color="#dddddd", alpha=0.45)
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_xscale("log")
            axis.grid(alpha=0.22)
            axis.set_xlabel(r"$\ell$ [cells]")
            if row == 0:
                axis.set_title(direction_name)
            if column == 0:
                axis.set_ylabel(
                    f"{phase3_figures.SHORT_LABELS[cube_id]}\n"
                    + rf"$\alpha_{{{q_name},\perp}}(\ell)$"
                )
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"Phase 4 launch gate: 2-point {q_name} local slopes by support policy; "
        "gray region: full 5-bin shell-local >=5% slope window unavailable"
    )
    return _save(
        figure,
        output_dir,
        f"phase4_go_2point_{q_name}_local_slope_policy_comparison.png",
    )


def cutoff_sensitivity_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    cutoff_values = np.asarray(CANDIDATE_SHELL_SUPPORT_FRACTIONS) * 100.0
    rows = summary["candidate_cutoff_metrics"]
    curve_p90 = [rows[f"{cutoff:.2f}"]["curve_policy_factor"]["p90"] for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS]
    curve_max = [rows[f"{cutoff:.2f}"]["curve_policy_factor"]["maximum"] for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS]
    slope_median = [rows[f"{cutoff:.2f}"]["absolute_local_slope_difference"]["median"] for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS]
    slope_p90 = [rows[f"{cutoff:.2f}"]["absolute_local_slope_difference"]["p90"] for cutoff in CANDIDATE_SHELL_SUPPORT_FRACTIONS]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(cutoff_values, curve_p90, "o-", color="#4c78a8", label="90th percentile")
    axes[0].plot(cutoff_values, curve_max, "o--", color="#e45756", label="maximum")
    axes[0].axhline(1.0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel(r"folded curve sensitivity factor $\max(r, 1/r)$")
    axes[0].legend(fontsize=8)
    axes[1].plot(cutoff_values, slope_median, "o-", color="#4c78a8", label="median")
    axes[1].plot(cutoff_values, slope_p90, "o--", color="#e45756", label="90th percentile")
    axes[1].set_ylabel(r"$|\alpha_{\mathrm{all}}-\alpha_{\mathrm{shell}}|$")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("candidate shell-local support cutoff [%]")
        axis.set_xticks(cutoff_values)
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Phase 4 launch gate: a support cutoff controls weak tails but does not make slope policies interchangeable"
    )
    return _save(figure, output_dir, "phase4_go_cutoff_sensitivity_summary.png")


def write_manifest(
    output_dir: Path,
    verified: phase3_figures.VerifiedInputs,
    input_hashes: phase3_figures.InputHashes,
) -> None:
    figures = sorted(path.name for path in output_dir.glob("*.png"))
    summary_path = output_dir / "phase4_go_policy_summary.json"
    payload = {
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
        "policy_summary_sha256": file_sha256(summary_path),
        "generated_figures": figures,
        "figure_sha256": {
            name: file_sha256(output_dir / name) for name in figures
        },
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the hash-bound Phase 4 launch-GO evidence package."
    )
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--convergence-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_hashes = phase3_figures.InputHashes()
    verified = phase3_figures.verify_inputs(
        phase2_root=args.phase2_root,
        controls_root=args.controls_root,
        convergence_root=args.convergence_root,
        release_root=args.release_root,
        input_hashes=input_hashes,
    )
    groups = _load_groups(verified, input_hashes)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        summary = build_summary(verified, groups)
        (temporary_output / "phase4_go_policy_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        support_cutoff_diagnostic(groups, temporary_output)
        curve_policy_ratio_diagnostic(groups, temporary_output)
        slope_policy_comparison(groups, temporary_output, q_name="B")
        slope_policy_comparison(groups, temporary_output, q_name="u")
        cutoff_sensitivity_summary(summary, temporary_output)
        write_manifest(temporary_output, verified, input_hashes)
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        temporary_output.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
