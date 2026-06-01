#!/usr/bin/env python3
"""Publish the immutable Phase 4 Batch B representative p=1..6 tail review."""
from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT_DIR = Path("figures/phase4_batch_b_representative_review")
SUMMARY_FILENAME = "phase4_batch_b_representative_review_summary.json"
LEDGER_SNAPSHOT_FILENAME = "phase4_batch_b_representative_compute_ledger_summary_snapshot.md"
P_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
P6 = 6.0
SCIENCE_SCALE_MINIMUM = 32.0
SHELL_CURVE_MINIMUM = 0.05
TAIL_FACTOR_THRESHOLD = 1.5
FOCUS_CUBE = "L640_sub03026"
P_COLORS = {
    1.0: "#4c78a8",
    2.0: "#f58518",
    3.0: "#54a24b",
    4.0: "#e45756",
    5.0: "#72b7b2",
    6.0: "#b279a2",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _load_groups(
    release_root: Path,
    cube_ids: tuple[str, ...],
    input_hashes: base.InputHashes,
) -> dict[tuple[str, str], base.Group]:
    groups = {}
    for cube_id in cube_ids:
        for support_mode in base.SUPPORT_MODES:
            group = base._load_group(release_root, cube_id, 2, support_mode, input_hashes)
            result = group.result
            if (
                result.q_names != base.Q_NAMES
                or result.p_values != P_VALUES
                or result.density_conventions != ("not applicable", "not applicable")
                or not np.isnan(result.rho0)
                or result.rho0_provenance != "not applicable for requested q variants"
            ):
                raise RuntimeError(f"unexpected Batch B quantity matrix: {cube_id}/{support_mode}")
            groups[(cube_id, support_mode)] = group
    return groups


def _curve_rows(
    groups: Mapping[tuple[str, str], base.Group],
    cube_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained = []
    census = []
    for cube_id in cube_ids:
        primary = groups[(cube_id, "all_valid_origins")]
        shell = groups[(cube_id, "shell_local")]
        ell = figures._centers(primary.result.ell_bin_edges)
        shell_support = base._support_fraction(shell.result)
        for p_value in P_VALUES:
            for q_name in base.Q_NAMES:
                for direction in base.DIRECTIONS:
                    index = figures._moment_index(
                        primary.result, q_name, direction, p_value=p_value
                    )
                    primary_values = primary.result.moments[index]
                    shell_values = shell.result.moments[index]
                    ratio = np.divide(
                        primary_values,
                        shell_values,
                        out=np.full_like(primary_values, np.nan, dtype=float),
                        where=np.isfinite(shell_values) & (shell_values != 0.0),
                    )
                    science = ell >= SCIENCE_SCALE_MINIMUM
                    geometry = shell_support >= SHELL_CURVE_MINIMUM
                    uncertainty = (
                        base._curve_support_mask(primary, index)
                        & base._curve_support_mask(shell, index)
                    )
                    finite_ratio = np.isfinite(ratio) & (ratio > 0.0)
                    supported = science & geometry & uncertainty & finite_ratio
                    census.append(
                        {
                            "cube_id": cube_id,
                            "p_value": p_value,
                            "q_name": q_name,
                            "direction": direction,
                            "candidate_science_scale_bins": int(np.count_nonzero(science)),
                            "retained_bins": int(np.count_nonzero(supported)),
                            "excluded_shell_support_bins": int(np.count_nonzero(science & ~geometry)),
                            "excluded_uncertainty_bins": int(
                                np.count_nonzero(science & geometry & ~uncertainty)
                            ),
                            "excluded_nonfinite_ratio_bins": int(
                                np.count_nonzero(science & geometry & uncertainty & ~finite_ratio)
                            ),
                        }
                    )
                    for bin_index in np.flatnonzero(supported):
                        value = float(ratio[bin_index])
                        retained.append(
                            {
                                "cube_id": cube_id,
                                "p_value": p_value,
                                "q_name": q_name,
                                "direction": direction,
                                "ell_cells": float(ell[bin_index]),
                                "shell_local_eligible_origin_fraction": float(
                                    shell_support[bin_index]
                                ),
                                "primary_over_shell_ratio": value,
                                "policy_factor": max(value, 1.0 / value),
                                "primary_count": int(primary.result.counts[index][bin_index]),
                                "shell_count": int(shell.result.counts[index][bin_index]),
                                "primary_effective_blocks": float(
                                    primary.uncertainty["accepted_effective_blocks"][index][bin_index]
                                ),
                                "shell_effective_blocks": float(
                                    shell.uncertainty["accepted_effective_blocks"][index][bin_index]
                                ),
                                "primary_valid_bootstrap_resamples": int(
                                    primary.uncertainty["valid_bootstrap_resamples"][index][bin_index]
                                ),
                                "shell_valid_bootstrap_resamples": int(
                                    shell.uncertainty["valid_bootstrap_resamples"][index][bin_index]
                                ),
                            }
                        )
    return retained, census


def _summarize_census(census: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    fields = (
        "candidate_science_scale_bins",
        "retained_bins",
        "excluded_shell_support_bins",
        "excluded_uncertainty_bins",
        "excluded_nonfinite_ratio_bins",
    )
    for p_value in P_VALUES:
        rows = [row for row in census if row["p_value"] == p_value]
        payload = {"p_value": p_value, **{name: sum(row[name] for row in rows) for name in fields}}
        payload["retained_fraction"] = (
            payload["retained_bins"] / payload["candidate_science_scale_bins"]
        )
        output.append(payload)
    return output


def _focus_tail_rows(
    groups: Mapping[tuple[str, str], base.Group],
) -> list[dict[str, Any]]:
    primary = groups[(FOCUS_CUBE, "all_valid_origins")]
    shell = groups[(FOCUS_CUBE, "shell_local")]
    ell = figures._centers(primary.result.ell_bin_edges)
    shell_support = base._support_fraction(shell.result)
    output = []
    for q_name in base.Q_NAMES:
        for direction in base.DIRECTIONS:
            index = figures._moment_index(primary.result, q_name, direction, p_value=P6)
            primary_values = primary.result.moments[index]
            shell_values = shell.result.moments[index]
            ratio = np.divide(
                primary_values,
                shell_values,
                out=np.full_like(primary_values, np.nan, dtype=float),
                where=np.isfinite(shell_values) & (shell_values != 0.0),
            )
            primary_valid = base._curve_support_mask(primary, index)
            shell_valid = base._curve_support_mask(shell, index)
            for bin_index in np.flatnonzero(ell >= SCIENCE_SCALE_MINIMUM):
                value = float(ratio[bin_index]) if np.isfinite(ratio[bin_index]) else None
                output.append(
                    {
                        "q_name": q_name,
                        "direction": direction,
                        "ell_cells": float(ell[bin_index]),
                        "shell_local_eligible_origin_fraction": float(shell_support[bin_index]),
                        "primary_over_shell_ratio": value,
                        "policy_factor": max(value, 1.0 / value) if value is not None and value > 0.0 else None,
                        "primary_supported": bool(primary_valid[bin_index]),
                        "shell_supported": bool(
                            shell_valid[bin_index] & (shell_support[bin_index] >= SHELL_CURVE_MINIMUM)
                        ),
                        "primary_count": int(primary.result.counts[index][bin_index]),
                        "shell_count": int(shell.result.counts[index][bin_index]),
                        "primary_effective_blocks": float(
                            primary.uncertainty["accepted_effective_blocks"][index][bin_index]
                        ),
                        "shell_effective_blocks": float(
                            shell.uncertainty["accepted_effective_blocks"][index][bin_index]
                        ),
                        "primary_valid_bootstrap_resamples": int(
                            primary.uncertainty["valid_bootstrap_resamples"][index][bin_index]
                        ),
                        "shell_valid_bootstrap_resamples": int(
                            shell.uncertainty["valid_bootstrap_resamples"][index][bin_index]
                        ),
                    }
                )
    return output


def order_retention(census: list[dict[str, Any]], output_dir: Path) -> Path:
    summary = _summarize_census(census)
    figure, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    x = np.arange(len(summary))
    retained = np.asarray([row["retained_bins"] for row in summary])
    shell = np.asarray([row["excluded_shell_support_bins"] for row in summary])
    uncertainty = np.asarray([row["excluded_uncertainty_bins"] for row in summary])
    nonfinite = np.asarray([row["excluded_nonfinite_ratio_bins"] for row in summary])
    axis.bar(x, retained, label="retained", color="#4c78a8")
    axis.bar(x, shell, bottom=retained, label="excluded: shell support", color="#f58518")
    axis.bar(x, uncertainty, bottom=retained + shell, label="excluded: uncertainty", color="#e45756")
    axis.bar(
        x,
        nonfinite,
        bottom=retained + shell + uncertainty,
        label="excluded: nonfinite ratio",
        color="#72b7b2",
    )
    axis.set_xticks(x, [f"$p={int(row['p_value'])}$" for row in summary])
    axis.set_ylabel("directional science-scale bins")
    axis.set_title(r"Representative retention audit for $\ell \geq 32$ cells")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8)
    return _save(figure, output_dir, "phase4_batch_b_order_retention_audit.png")


def p6_policy_ratios(
    retained: list[dict[str, Any]],
    cube_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True, sharex=True)
    for axis, (q_name, direction) in zip(
        axes.flat, [(q_name, direction) for q_name in base.Q_NAMES for direction in base.DIRECTIONS]
    ):
        for cube_id in cube_ids:
            rows = [
                row
                for row in retained
                if row["p_value"] == P6
                and row["q_name"] == q_name
                and row["direction"] == direction
                and row["cube_id"] == cube_id
            ]
            axis.plot(
                [row["ell_cells"] for row in rows],
                [row["primary_over_shell_ratio"] for row in rows],
                marker="o",
                markersize=2.5,
                linewidth=0.9,
                alpha=0.8,
                label=cube_id,
            )
        axis.axhline(1.0, color="#777777", linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{q_name}: {direction}")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel(r"$S_{6,\rm primary}/S_{6,\rm shell}$")
    axes[1, 0].set_ylabel(r"$S_{6,\rm primary}/S_{6,\rm shell}$")
    axes[0, -1].legend(fontsize=5.7, ncol=2)
    figure.suptitle(r"Supported signed policy sensitivity for the representative $p=6$ tails")
    return _save(figure, output_dir, "phase4_batch_b_p6_signed_policy_ratios.png")


def p6_retention_heatmap(
    census: list[dict[str, Any]],
    cube_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    channels = [(q_name, direction) for q_name in base.Q_NAMES for direction in base.DIRECTIONS]
    values = np.zeros((len(cube_ids), len(channels)))
    for row_index, cube_id in enumerate(cube_ids):
        for column, (q_name, direction) in enumerate(channels):
            row = next(
                item
                for item in census
                if item["p_value"] == P6
                and item["cube_id"] == cube_id
                and item["q_name"] == q_name
                and item["direction"] == direction
            )
            values[row_index, column] = row["retained_bins"] / row["candidate_science_scale_bins"]
    figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(channels)), [f"{q_name} {direction}" for q_name, direction in channels])
    axis.set_yticks(np.arange(len(cube_ids)), cube_ids)
    axis.tick_params(axis="x", rotation=35)
    for row, column in np.ndindex(values.shape):
        axis.text(column, row, f"{values[row, column]:.0%}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="retained science-scale fraction")
    axis.set_title(r"$p=6$ retention after shell-support and block-uncertainty gates")
    return _save(figure, output_dir, "phase4_batch_b_p6_retention_heatmap.png")


def focus_tail(
    groups: Mapping[tuple[str, str], base.Group],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True, sharex=True)
    for axis, (q_name, direction) in zip(
        axes.flat, [(q_name, direction) for q_name in base.Q_NAMES for direction in base.DIRECTIONS]
    ):
        for mode, color, linestyle in (
            ("all_valid_origins", "#4c78a8", "-"),
            ("shell_local", "#f58518", "--"),
        ):
            group = groups[(FOCUS_CUBE, mode)]
            ell = figures._centers(group.result.ell_bin_edges)
            index = figures._moment_index(group.result, q_name, direction, p_value=P6)
            mask = base._curve_support_mask(group, index)
            if mode == "shell_local":
                mask &= base._support_fraction(group.result) >= SHELL_CURVE_MINIMUM
            values = np.where(mask, group.result.moments[index], np.nan)
            low = np.where(mask, group.uncertainty["block_bootstrap_interval_low"][index], np.nan)
            high = np.where(mask, group.uncertainty["block_bootstrap_interval_high"][index], np.nan)
            valid = np.isfinite(values) & (values > 0.0)
            axis.plot(ell[valid], values[valid], color=color, linestyle=linestyle, label=mode)
            band = valid & np.isfinite(low) & np.isfinite(high) & (low > 0.0)
            axis.fill_between(ell[band], low[band], high[band], color=color, alpha=0.15)
        axis.axvline(SCIENCE_SCALE_MINIMUM, color="#777777", linestyle=":")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{q_name}: {direction}")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel(r"$S_6(\ell)$")
    axes[1, 0].set_ylabel(r"$S_6(\ell)$")
    axes[0, -1].legend(fontsize=7)
    figure.suptitle(f"{FOCUS_CUBE}: complete supported $p=6$ tail with block-bootstrap bands")
    return _save(figure, output_dir, "phase4_batch_b_sub03026_p6_tail.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
    input_hashes = base.InputHashes()
    release = base._verify_release_marker(
        args.release_root,
        summary_filename="phase4_batch_b_representative_summary.json",
        marker_filename="PHASE4_BATCH_B_REPRESENTATIVE_COMPLETE.json",
        input_hashes=input_hashes,
    )
    cube_ids = tuple(release.get("representative_cube_ids", ()))
    if (
        release.get("phase") != "phase4_batch_b_bounded_8_cube_2point_p1_to_p6"
        or len(cube_ids) != 8
        or len(set(cube_ids)) != 8
    ):
        raise RuntimeError("Batch B release does not match the bounded representative matrix")
    input_hashes.add(args.ledger_summary)
    groups = _load_groups(args.release_root, cube_ids, input_hashes)
    retained, census = _curve_rows(groups, cube_ids)
    p6_rows = [row for row in retained if row["p_value"] == P6]
    top_p6_rows = sorted(p6_rows, key=lambda row: row["policy_factor"], reverse=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copyfile(args.ledger_summary, temporary_output / LEDGER_SNAPSHOT_FILENAME)
        order_retention(census, temporary_output)
        p6_policy_ratios(retained, cube_ids, temporary_output)
        p6_retention_heatmap(census, cube_ids, temporary_output)
        focus_tail(groups, temporary_output)
        summary = {
            "schema_version": 1,
            "status": "phase4_batch_b_representative_review_generated",
            "decision_scope": "bounded eight-cube 2-point p=1..6 review only",
            "automatic_all21_batch_b_expansion_claimed": False,
            "five_point_expansion_claimed": False,
            "release_root": str(args.release_root.resolve()),
            "cube_ids": cube_ids,
            "configuration": {
                "q_names": base.Q_NAMES,
                "p_values": P_VALUES,
                "stencil_width": 2,
                "primary_curve_product": "all_valid_origins",
                "directional_robustness_overlay": "shell_local",
                "science_scale_minimum_cells": SCIENCE_SCALE_MINIMUM,
                "shell_local_curve_overlay_minimum_fraction": SHELL_CURVE_MINIMUM,
                "tail_factor_threshold": TAIL_FACTOR_THRESHOLD,
                "directional_fitted_exponents_published": False,
            },
            "retention_by_order": _summarize_census(census),
            "p6_supported_row_count": len(p6_rows),
            "p6_count_above_factor_threshold": sum(
                row["policy_factor"] > TAIL_FACTOR_THRESHOLD for row in p6_rows
            ),
            "p6_top_supported_rows": top_p6_rows[:40],
            "sub03026_complete_p6_science_tail": _focus_tail_rows(groups),
            "operational_result": {
                "review_package_generated": True,
                "release_marker_replayed": True,
                "review_gate": "human review required before any all-21 Batch B expansion",
            },
            "input_sha256": input_hashes.as_dict(),
        }
        _write_json(temporary_output / SUMMARY_FILENAME, summary)
        figure_names = sorted(path.name for path in temporary_output.glob("*.png"))
        artifacts = sorted(path.name for path in temporary_output.iterdir())
        _write_json(
            temporary_output / "figure_manifest.json",
            {
                "schema_version": 1,
                "status": "passed",
                "publication_policy": "immutable_one_time_publish_refuse_existing_output_directory",
                "generator_sha256": file_sha256(Path(__file__).resolve()),
                "generated_figures": figure_names,
                "figure_sha256": {
                    name: file_sha256(temporary_output / name) for name in figure_names
                },
                "generated_artifacts_before_manifest": artifacts,
                "input_sha256": input_hashes.as_dict(),
            },
        )
        temporary_output.rename(output_dir)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    print(f"Wrote immutable Phase 4 Batch B representative review package: {output_dir}")


if __name__ == "__main__":
    main()
