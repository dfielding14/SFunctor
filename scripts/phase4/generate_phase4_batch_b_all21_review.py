#!/usr/bin/env python3
"""Publish the immutable all-21 Phase 4 Batch B p=1..6 review diagnostics."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase4 import generate_phase4_batch_a2_review as base
from scripts.phase4 import generate_phase4_batch_a_status_figures as batch_a_figures
from scripts.phase4 import generate_phase4_batch_b_representative_review as representative


DEFAULT_OUTPUT_DIR = Path("figures/phase4_batch_b_all21_review")
SUMMARY_FILENAME = "phase4_batch_b_all21_review_summary.json"
LEDGER_SNAPSHOT_FILENAME = "phase4_batch_b_all21_compute_ledger_summary_snapshot.md"
RELEASE_SUMMARY_FILENAME = "phase4_batch_b_all21_extension_summary.json"
RELEASE_MARKER_FILENAME = "PHASE4_BATCH_B_ALL21_EXTENSION_COMPLETE.json"
EXPECTED_RELEASE_PHASE = "phase4_batch_b_all21_2point_p1_to_p6_extension"
ALL21_CUBE_IDS = batch_a_figures.FROZEN_PHASE4_PILOT_CUBE_IDS
P_VALUES = representative.P_VALUES
P6 = representative.P6
SCIENCE_SCALE_MINIMUM = representative.SCIENCE_SCALE_MINIMUM
SHELL_CURVE_MINIMUM = representative.SHELL_CURVE_MINIMUM
TAIL_FACTOR_THRESHOLD = representative.TAIL_FACTOR_THRESHOLD
CATALOG_FIELDS = ("dBB", "B_mean", "deltaB")
POLICY_FACTOR_STATISTICS = ("median", "p90", "maximum")
ORDER_SENSITIVITY_STATISTICS = (
    "p6_over_p1_median_policy_factor",
    "p6_minus_p1_median_policy_factor",
    "spearman_rho_order_vs_median_policy_factor",
)
REVIEW_DIAGNOSTIC_INTERPRETATION = (
    "exploratory all-21 review diagnostic only; not a fitted scaling law or fitted exponent"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _verify_all21_release(
    release_root: Path,
    input_hashes: batch_a_figures.InputHashes,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Verify the extension marker and reject any release other than the frozen all-21 matrix."""

    release = base._verify_release_marker(
        release_root,
        summary_filename=RELEASE_SUMMARY_FILENAME,
        marker_filename=RELEASE_MARKER_FILENAME,
        input_hashes=input_hashes,
    )
    cube_ids = tuple(release.get("pilot_cube_ids", ()))
    if (
        release.get("phase") != EXPECTED_RELEASE_PHASE
        or cube_ids != ALL21_CUBE_IDS
        or len(cube_ids) != 21
        or len(set(cube_ids)) != 21
    ):
        raise RuntimeError("Batch B all-21 release is not the exact frozen 21-cube extension")
    return release, cube_ids


def _load_phase1_catalog_rows(
    phase1_root: Path,
    cube_ids: tuple[str, ...],
    input_hashes: batch_a_figures.InputHashes,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reuse the strict Phase 1 L=640 catalog loader without loading Batch A report inputs."""

    verified = SimpleNamespace(phase1_root=phase1_root.resolve(), cube_ids=cube_ids)
    return batch_a_figures._phase1_catalog_rows(verified, input_hashes)


def _factor_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if np.isfinite(value) and value > 0.0]
    return base._factor(finite)


def _cube_order_policy_factor_rows(
    retained: Sequence[Mapping[str, Any]],
    cube_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for cube_id in cube_ids:
        for p_value in P_VALUES:
            values = [
                float(row["policy_factor"])
                for row in retained
                if row["cube_id"] == cube_id and row["p_value"] == p_value
            ]
            rows.append(
                {
                    "cube_id": cube_id,
                    "p_value": p_value,
                    "supported_policy_factor": _factor_summary(values),
                    "interpretation": REVIEW_DIAGNOSTIC_INTERPRETATION,
                }
            )
    return rows


def _finite_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if (
        numerator is None
        or denominator is None
        or not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return None
    return float(numerator / denominator)


def _finite_difference(left: float | None, right: float | None) -> float | None:
    if (
        left is None
        or right is None
        or not np.isfinite(left)
        or not np.isfinite(right)
    ):
        return None
    return float(left - right)


def _cube_order_sensitivity_rows(
    factor_rows: Sequence[Mapping[str, Any]],
    cube_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for cube_id in cube_ids:
        selected = [row for row in factor_rows if row["cube_id"] == cube_id]
        median_by_order = {
            float(row["p_value"]): row["supported_policy_factor"]["median"] for row in selected
        }
        supported_orders = [
            p_value for p_value in P_VALUES if median_by_order.get(p_value) is not None
        ]
        medians = [median_by_order[p_value] for p_value in supported_orders]
        rows.append(
            {
                "cube_id": cube_id,
                "supported_order_count": len(supported_orders),
                "supported_median_policy_factor_by_order": {
                    str(int(p_value)): median_by_order.get(p_value) for p_value in P_VALUES
                },
                "p6_over_p1_median_policy_factor": _finite_ratio(
                    median_by_order.get(P6), median_by_order.get(1.0)
                ),
                "p6_minus_p1_median_policy_factor": _finite_difference(
                    median_by_order.get(P6), median_by_order.get(1.0)
                ),
                "spearman_rho_order_vs_median_policy_factor": batch_a_figures._spearman(
                    supported_orders, medians
                ),
                "interpretation": REVIEW_DIAGNOSTIC_INTERPRETATION,
            }
        )
    return rows


def _catalog_correlation_row(
    *,
    diagnostic: str,
    values_by_cube: Mapping[str, float | None],
    catalog_rows: Mapping[str, Mapping[str, Any]],
    p_value: float | None = None,
) -> dict[str, Any]:
    selected = [
        cube_id
        for cube_id, value in values_by_cube.items()
        if value is not None and np.isfinite(value)
    ]
    payload = {
        "diagnostic": diagnostic,
        "finite_cube_count": len(selected),
        "spearman_rho_vs_phase1_catalog": {
            field: batch_a_figures._spearman(
                [float(catalog_rows[cube_id]["catalog"][field]) for cube_id in selected],
                [float(values_by_cube[cube_id]) for cube_id in selected],
            )
            for field in CATALOG_FIELDS
        },
        "interpretation": REVIEW_DIAGNOSTIC_INTERPRETATION,
    }
    if p_value is not None:
        payload["p_value"] = p_value
    return payload


def _catalog_review_correlations(
    factor_rows: Sequence[Mapping[str, Any]],
    order_sensitivity_rows: Sequence[Mapping[str, Any]],
    catalog_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for p_value in P_VALUES:
        selected = [row for row in factor_rows if row["p_value"] == p_value]
        for statistic in POLICY_FACTOR_STATISTICS:
            output.append(
                _catalog_correlation_row(
                    diagnostic=f"supported_policy_factor_{statistic}",
                    p_value=p_value,
                    values_by_cube={
                        str(row["cube_id"]): row["supported_policy_factor"][statistic]
                        for row in selected
                    },
                    catalog_rows=catalog_rows,
                )
            )
    for statistic in ORDER_SENSITIVITY_STATISTICS:
        output.append(
            _catalog_correlation_row(
                diagnostic=statistic,
                values_by_cube={
                    str(row["cube_id"]): row[statistic] for row in order_sensitivity_rows
                },
                catalog_rows=catalog_rows,
            )
        )
    return output


def _top_p6_rows(
    retained: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in retained if row["p_value"] == P6]
    return sorted(rows, key=lambda row: float(row["policy_factor"]), reverse=True)[:limit]


def order_retention(census: list[dict[str, Any]], output_dir: Path) -> Path:
    summary = representative._summarize_census(census)
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
    axis.set_title(r"All-21 order-retention review diagnostic for $\ell \geq 32$ cells")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8)
    return representative._save(
        figure, output_dir, "phase4_batch_b_all21_order_retention_diagnostic.png"
    )


def p6_policy_sensitivity(
    retained: list[dict[str, Any]],
    cube_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    channels = [(q_name, direction) for q_name in base.Q_NAMES for direction in base.DIRECTIONS]
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(cube_ids)))
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True, sharex=True)
    for axis, (q_name, direction) in zip(axes.flat, channels):
        for cube_id, color in zip(cube_ids, colors):
            rows = [
                row
                for row in retained
                if row["p_value"] == P6
                and row["q_name"] == q_name
                and row["direction"] == direction
                and row["cube_id"] == cube_id
            ]
            if not rows:
                continue
            axis.plot(
                [row["ell_cells"] for row in rows],
                [row["primary_over_shell_ratio"] for row in rows],
                marker="o",
                markersize=2.0,
                linewidth=0.75,
                alpha=0.72,
                color=color,
            )
            axis.fill_between(
                [row["ell_cells"] for row in rows],
                [row["coupled_ratio_bootstrap_interval_low"] for row in rows],
                [row["coupled_ratio_bootstrap_interval_high"] for row in rows],
                alpha=0.035,
                color=color,
            )
        axis.axhline(1.0, color="#777777", linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{q_name}: {direction}")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel(r"$S_{6,\rm primary}/S_{6,\rm shell}$")
    axes[1, 0].set_ylabel(r"$S_{6,\rm primary}/S_{6,\rm shell}$")
    figure.suptitle(r"All-21 supported $p=6$ policy-sensitivity review diagnostic")
    return representative._save(
        figure, output_dir, "phase4_batch_b_all21_p6_policy_sensitivity_diagnostic.png"
    )


def p6_moment_concentration(
    retained: list[dict[str, Any]],
    cube_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    channels = [(q_name, direction) for q_name in base.Q_NAMES for direction in base.DIRECTIONS]
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 7.8), constrained_layout=True)
    for axis, policy in zip(axes, ("primary", "shell")):
        values = np.full((len(cube_ids), len(channels)), np.nan)
        for row_index, cube_id in enumerate(cube_ids):
            for column, (q_name, direction) in enumerate(channels):
                selected = [
                    float(row[f"{policy}_largest_5_blocks_fraction"])
                    for row in retained
                    if row["p_value"] == P6
                    and row["cube_id"] == cube_id
                    and row["q_name"] == q_name
                    and row["direction"] == direction
                ]
                if selected:
                    values[row_index, column] = max(selected)
        image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
        axis.set_xticks(
            np.arange(len(channels)),
            [f"{q_name} {direction}" for q_name, direction in channels],
        )
        axis.set_yticks(np.arange(len(cube_ids)), cube_ids)
        axis.tick_params(axis="x", rotation=35)
        for row, column in np.ndindex(values.shape):
            label = f"{values[row, column]:.0%}" if np.isfinite(values[row, column]) else "-"
            axis.text(column, row, label, ha="center", va="center", fontsize=6)
        axis.set_title(f"{policy}: maximum top-5 block share")
    figure.colorbar(image, ax=axes, label=r"largest 5 blocks / total $S_6$ contribution")
    figure.suptitle(r"All-21 $p=6$ moment-concentration review diagnostic")
    return representative._save(
        figure, output_dir, "phase4_batch_b_all21_p6_concentration_diagnostic.png"
    )


def p6_retention(
    census: list[dict[str, Any]],
    cube_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    channels = [(q_name, direction) for q_name in base.Q_NAMES for direction in base.DIRECTIONS]
    values = np.full((len(cube_ids), len(channels)), np.nan)
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
            candidate = int(row["candidate_science_scale_bins"])
            if candidate:
                values[row_index, column] = float(row["retained_bins"]) / candidate
    figure, axis = plt.subplots(figsize=(9.0, 7.8), constrained_layout=True)
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(channels)), [f"{q_name} {direction}" for q_name, direction in channels])
    axis.set_yticks(np.arange(len(cube_ids)), cube_ids)
    axis.tick_params(axis="x", rotation=35)
    for row, column in np.ndindex(values.shape):
        label = f"{values[row, column]:.0%}" if np.isfinite(values[row, column]) else "-"
        axis.text(column, row, label, ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=axis, label="retained science-scale fraction")
    axis.set_title(r"All-21 $p=6$ retention review diagnostic after support and block gates")
    return representative._save(
        figure, output_dir, "phase4_batch_b_all21_p6_retention_diagnostic.png"
    )


def p6_top_tail(
    retained: list[dict[str, Any]],
    output_dir: Path,
    *,
    limit: int = 30,
) -> Path:
    rows = _top_p6_rows(retained, limit=limit)
    figure, axis = plt.subplots(figsize=(11.0, max(4.8, 0.24 * len(rows))), constrained_layout=True)
    if not rows:
        axis.text(0.5, 0.5, "No supported p=6 rows passed the review gates", ha="center", va="center")
        axis.set_axis_off()
    else:
        y = np.arange(len(rows))
        ratio = np.asarray([row["primary_over_shell_ratio"] for row in rows], dtype=float)
        low = np.asarray([row["coupled_ratio_bootstrap_interval_low"] for row in rows], dtype=float)
        high = np.asarray([row["coupled_ratio_bootstrap_interval_high"] for row in rows], dtype=float)
        interval = np.isfinite(low) & np.isfinite(high) & (low > 0.0) & (high > 0.0)
        colors = [
            "#e45756" if row["policy_factor"] > TAIL_FACTOR_THRESHOLD else "#4c78a8"
            for row in rows
        ]
        axis.hlines(y[interval], low[interval], high[interval], color="#777777", linewidth=1.0)
        axis.scatter(ratio, y, c=colors, s=24, zorder=3)
        axis.axvline(1.0, color="#777777", linestyle="--")
        axis.set_xscale("log")
        axis.set_yticks(
            y,
            [
                f"{row['cube_id']} {row['q_name']} {row['direction']} ell={row['ell_cells']:.1f}"
                for row in rows
            ],
            fontsize=7,
        )
        axis.invert_yaxis()
        axis.set_xlabel(r"$S_{6,\rm primary}/S_{6,\rm shell}$ with coupled block-bootstrap interval")
        axis.grid(axis="x", alpha=0.22)
    axis.set_title("All-21 strongest supported p=6 top-tail review diagnostics")
    return representative._save(
        figure, output_dir, "phase4_batch_b_all21_p6_top_tail_diagnostic.png"
    )


def _write_ledger_summary_snapshot(
    output_dir: Path,
    ledger_summary: Mapping[str, Any],
    snapshot: bytes,
) -> Path:
    path = output_dir / LEDGER_SNAPSHOT_FILENAME
    path.write_bytes(snapshot)
    if file_sha256(path) != ledger_summary["source_sha256"]:
        raise RuntimeError("published compute-ledger summary snapshot changed during report assembly")
    return path


def _write_manifest(
    output_dir: Path,
    input_hashes: batch_a_figures.InputHashes,
) -> None:
    figures = sorted(path.name for path in output_dir.glob("*.png"))
    artifacts = sorted(path.name for path in output_dir.iterdir())
    _write_json(
        output_dir / "figure_manifest.json",
        {
            "schema_version": 1,
            "status": "passed",
            "publication_policy": "immutable_one_time_publish_refuse_existing_output_directory",
            "generator_sha256": file_sha256(Path(__file__).resolve()),
            "generated_figures": figures,
            "figure_sha256": {
                name: file_sha256(output_dir / name) for name in figures
            },
            "generated_artifacts_before_manifest": artifacts,
            "artifact_sha256": {
                name: file_sha256(output_dir / name) for name in artifacts
            },
            "input_sha256": input_hashes.as_dict(),
        },
    )


def generate(
    *,
    phase1_root: Path,
    release_root: Path,
    ledger_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
    input_hashes = batch_a_figures.InputHashes()
    release, cube_ids = _verify_all21_release(release_root, input_hashes)
    ledger_identity, ledger_snapshot = batch_a_figures._bind_ledger_summary_snapshot(
        ledger_summary, input_hashes
    )
    groups = representative._load_groups(release_root, cube_ids, input_hashes)
    retained, census = representative._curve_rows(groups, cube_ids)
    catalog_rows, catalog_metadata = _load_phase1_catalog_rows(
        phase1_root, cube_ids, input_hashes
    )
    factor_rows = _cube_order_policy_factor_rows(retained, cube_ids)
    order_sensitivity_rows = _cube_order_sensitivity_rows(factor_rows, cube_ids)
    catalog_correlations = _catalog_review_correlations(
        factor_rows, order_sensitivity_rows, catalog_rows
    )
    p6_rows = [row for row in retained if row["p_value"] == P6]
    top_p6_rows = _top_p6_rows(retained, limit=60)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        ledger_snapshot_path = _write_ledger_summary_snapshot(
            temporary_output, ledger_identity, ledger_snapshot
        )
        order_retention(census, temporary_output)
        p6_policy_sensitivity(retained, cube_ids, temporary_output)
        p6_moment_concentration(retained, cube_ids, temporary_output)
        p6_retention(census, cube_ids, temporary_output)
        p6_top_tail(retained, temporary_output)
        summary = {
            "schema_version": 1,
            "status": "phase4_batch_b_all21_review_diagnostics_generated",
            "decision_scope": "exact all-21 L640 2-point Batch B p=1..6 extension review only",
            "publication_policy": "immutable_one_time_publish_refuse_existing_output_directory",
            "conclusion_scope": REVIEW_DIAGNOSTIC_INTERPRETATION,
            "all_conclusions_are_review_diagnostics": True,
            "automatic_follow_on_expansion_claimed": False,
            "five_point_expansion_claimed": False,
            "fitted_directional_exponents_published": False,
            "input_roots": {
                "phase1_root": str(phase1_root.resolve()),
                "release_root": str(release_root.resolve()),
            },
            "verified_release": {
                "phase": release["phase"],
                "marker_replayed": True,
                "exact_frozen_all21_cube_inventory": True,
                "cube_count": len(cube_ids),
                "cube_ids": cube_ids,
            },
            "configuration": {
                "q_names": base.Q_NAMES,
                "p_values": P_VALUES,
                "stencil_width": 2,
                "primary_curve_product": "all_valid_origins",
                "directional_robustness_overlay": "shell_local",
                "science_scale_minimum_cells": SCIENCE_SCALE_MINIMUM,
                "shell_local_curve_overlay_minimum_fraction": SHELL_CURVE_MINIMUM,
                "curve_minimum_accepted_measurements": 2,
                "curve_minimum_contributing_blocks": 2,
                "curve_minimum_effective_blocks": 8.0,
                "curve_minimum_valid_bootstrap_fraction": base.MINIMUM_VALID_BOOTSTRAP_FRACTION,
                "curve_requires_finite_block_bootstrap_interval": True,
                "tail_factor_threshold": TAIL_FACTOR_THRESHOLD,
                "coupled_ratio_bootstrap_seed": representative.COUPLED_RATIO_BOOTSTRAP_SEED,
                "coupled_ratio_bootstrap_n_resamples": (
                    representative.COUPLED_RATIO_BOOTSTRAP_N_RESAMPLES
                ),
                "coupled_ratio_bootstrap_confidence_level": (
                    representative.COUPLED_RATIO_BOOTSTRAP_CONFIDENCE_LEVEL
                ),
                "directional_fitted_exponents_published": False,
            },
            "compute_ledger_summary_snapshot": {
                **ledger_identity,
                "snapshot_relative_path": ledger_snapshot_path.name,
                "snapshot_sha256": file_sha256(ledger_snapshot_path),
            },
            "retention_by_order": representative._summarize_census(census),
            "supported_policy_factor_by_cube_and_order": factor_rows,
            "order_sensitivity_by_cube": order_sensitivity_rows,
            "phase1_catalog_review_diagnostic_correlations": catalog_correlations,
            "phase1_catalog": {
                "fields_used": CATALOG_FIELDS,
                "metadata": catalog_metadata,
            },
            "p6_supported_row_count": len(p6_rows),
            "p6_supported_policy_factor": _factor_summary(
                [row["policy_factor"] for row in p6_rows]
            ),
            "p6_count_above_factor_threshold": sum(
                row["policy_factor"] > TAIL_FACTOR_THRESHOLD for row in p6_rows
            ),
            "p6_top_supported_rows": top_p6_rows,
            "operational_result": {
                "review_package_generated": True,
                "all21_release_marker_replayed": True,
                "exact_21_cube_release_verified": True,
                "review_gate": (
                    "human review required before any further expansion or scientific "
                    "publication claim"
                ),
            },
            "input_sha256": input_hashes.as_dict(),
        }
        _write_json(temporary_output / SUMMARY_FILENAME, summary)
        _write_manifest(temporary_output, input_hashes)
        batch_a_figures._publish_atomic_directory(temporary_output, output_dir)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the immutable Phase 4 Batch B all-21 review diagnostics."
    )
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        phase1_root=args.phase1_root,
        release_root=args.release_root,
        ledger_summary=args.ledger_summary,
        output_dir=args.output_dir,
    )
    print(f"Wrote immutable Phase 4 Batch B all-21 review diagnostics: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
