from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase4 import generate_phase4_batch_b_all21_review as review


def _ledger_summary() -> str:
    return """# Compute budget summary

Updated: `2026-06-02T00:00:00Z`

| Workflow budget | `10` |
| Consumed allocated runtime | `4` |
| Remaining budget | `6` |
| Pending maximum additional exposure | `0` |
| Projected remaining after pending maximum | `6` |
"""


def _retained_rows(cube_ids: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {
            "cube_id": cube_id,
            "p_value": p_value,
            "policy_factor": 1.0 + cube_index * p_value,
        }
        for cube_index, cube_id in enumerate(cube_ids)
        for p_value in review.P_VALUES
    ]


def _census_rows() -> list[dict[str, object]]:
    return [
        {
            "p_value": p_value,
            "candidate_science_scale_bins": 10,
            "retained_bins": 7,
            "excluded_shell_support_bins": 1,
            "excluded_uncertainty_bins": 1,
            "excluded_nonfinite_ratio_bins": 1,
        }
        for p_value in review.P_VALUES
    ]


def _catalog_rows(cube_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        cube_id: {
            "catalog": {
                "dBB": float(index),
                "B_mean": float(len(cube_ids) - index),
                "deltaB": float(2 * index),
            }
        }
        for index, cube_id in enumerate(cube_ids, start=1)
    }


def test_verify_all21_release_requires_exact_ordered_frozen_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "phase": review.EXPECTED_RELEASE_PHASE,
        "pilot_cube_ids": list(review.ALL21_CUBE_IDS),
    }
    monkeypatch.setattr(review.base, "_verify_release_marker", lambda *args, **kwargs: payload)

    release, cube_ids = review._verify_all21_release(
        Path("/release"), review.batch_a_figures.InputHashes()
    )

    assert release is payload
    assert cube_ids == review.ALL21_CUBE_IDS

    payload["pilot_cube_ids"] = list(reversed(review.ALL21_CUBE_IDS))
    with pytest.raises(RuntimeError, match="exact frozen 21-cube"):
        review._verify_all21_release(Path("/release"), review.batch_a_figures.InputHashes())


def test_catalog_correlations_cover_policy_factors_and_order_sensitivity() -> None:
    cube_ids = ("cube-a", "cube-b", "cube-c")
    factor_rows = review._cube_order_policy_factor_rows(_retained_rows(cube_ids), cube_ids)
    sensitivity_rows = review._cube_order_sensitivity_rows(factor_rows, cube_ids)

    correlations = review._catalog_review_correlations(
        factor_rows, sensitivity_rows, _catalog_rows(cube_ids)
    )

    p6_median = next(
        row
        for row in correlations
        if row["diagnostic"] == "supported_policy_factor_median"
        and row["p_value"] == 6.0
    )
    assert p6_median["finite_cube_count"] == 3
    assert p6_median["spearman_rho_vs_phase1_catalog"] == pytest.approx(
        {"dBB": 1.0, "B_mean": -1.0, "deltaB": 1.0}
    )
    order_delta = next(
        row
        for row in correlations
        if row["diagnostic"] == "p6_minus_p1_median_policy_factor"
    )
    assert order_delta["spearman_rho_vs_phase1_catalog"] == pytest.approx(
        {"dBB": 1.0, "B_mean": -1.0, "deltaB": 1.0}
    )
    assert all("not a fitted scaling law or fitted exponent" in row["interpretation"] for row in correlations)


def test_generate_publishes_immutable_review_package_without_production_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_summary = tmp_path / "ledger_summary.md"
    ledger_summary.write_text(_ledger_summary())
    output_dir = tmp_path / "review"
    cube_ids = review.ALL21_CUBE_IDS
    retained = _retained_rows(cube_ids)
    census = _census_rows()
    figure_names = (
        "phase4_batch_b_all21_order_retention_diagnostic.png",
        "phase4_batch_b_all21_p6_policy_sensitivity_diagnostic.png",
        "phase4_batch_b_all21_p6_concentration_diagnostic.png",
        "phase4_batch_b_all21_p6_retention_diagnostic.png",
        "phase4_batch_b_all21_p6_top_tail_diagnostic.png",
    )

    monkeypatch.setattr(
        review,
        "_verify_all21_release",
        lambda release_root, input_hashes: (
            {"phase": review.EXPECTED_RELEASE_PHASE},
            cube_ids,
        ),
    )
    monkeypatch.setattr(review.representative, "_load_groups", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        review.representative,
        "_curve_rows",
        lambda groups, selected_cube_ids: (retained, census),
    )
    monkeypatch.setattr(
        review,
        "_load_phase1_catalog_rows",
        lambda phase1_root, selected_cube_ids, input_hashes: (
            _catalog_rows(selected_cube_ids),
            {"catalog": "synthetic"},
        ),
    )

    def write_png(output: Path, name: str) -> Path:
        path = output / name
        path.write_bytes(b"synthetic png")
        return path

    monkeypatch.setattr(
        review,
        "order_retention",
        lambda selected_census, output: write_png(output, figure_names[0]),
    )
    monkeypatch.setattr(
        review,
        "p6_policy_sensitivity",
        lambda selected_retained, selected_cube_ids, output: write_png(output, figure_names[1]),
    )
    monkeypatch.setattr(
        review,
        "p6_moment_concentration",
        lambda selected_retained, selected_cube_ids, output: write_png(output, figure_names[2]),
    )
    monkeypatch.setattr(
        review,
        "p6_retention",
        lambda selected_census, selected_cube_ids, output: write_png(output, figure_names[3]),
    )
    monkeypatch.setattr(
        review,
        "p6_top_tail",
        lambda selected_retained, output: write_png(output, figure_names[4]),
    )

    summary = review.generate(
        phase1_root=tmp_path / "phase1",
        release_root=tmp_path / "release",
        ledger_summary=ledger_summary,
        output_dir=output_dir,
    )

    manifest = json.loads((output_dir / "figure_manifest.json").read_text())
    published_summary = json.loads((output_dir / review.SUMMARY_FILENAME).read_text())
    assert output_dir.is_dir()
    assert (output_dir / review.LEDGER_SNAPSHOT_FILENAME).read_text() == _ledger_summary()
    assert tuple(manifest["generated_figures"]) == tuple(sorted(figure_names))
    assert published_summary["verified_release"]["cube_count"] == 21
    assert published_summary["all_conclusions_are_review_diagnostics"] is True
    assert published_summary["fitted_directional_exponents_published"] is False
    assert summary["operational_result"]["exact_21_cube_release_verified"] is True


def test_generate_refuses_existing_output_before_reading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "already-published"
    output_dir.mkdir()
    monkeypatch.setattr(
        review,
        "_verify_all21_release",
        lambda *args, **kwargs: pytest.fail("input verification must not run"),
    )

    with pytest.raises(RuntimeError, match="refusing to overwrite immutable report output"):
        review.generate(
            phase1_root=tmp_path / "phase1",
            release_root=tmp_path / "release",
            ledger_summary=tmp_path / "ledger.md",
            output_dir=output_dir,
        )


def test_diagnostic_renderers_write_five_pngs_without_production_output(
    tmp_path: Path,
) -> None:
    cube_ids = ("cube-a", "cube-b")
    retained = []
    census = []
    for cube_index, cube_id in enumerate(cube_ids):
        for p_value in review.P_VALUES:
            for q_name in review.base.Q_NAMES:
                for direction in review.base.DIRECTIONS:
                    census.append(
                        {
                            "cube_id": cube_id,
                            "p_value": p_value,
                            "q_name": q_name,
                            "direction": direction,
                            "candidate_science_scale_bins": 2,
                            "retained_bins": 1,
                            "excluded_shell_support_bins": 1,
                            "excluded_uncertainty_bins": 0,
                            "excluded_nonfinite_ratio_bins": 0,
                        }
                    )
                    if p_value == review.P6:
                        ratio = 1.1 + 0.1 * cube_index
                        retained.append(
                            {
                                "cube_id": cube_id,
                                "p_value": p_value,
                                "q_name": q_name,
                                "direction": direction,
                                "ell_cells": 48.0,
                                "primary_over_shell_ratio": ratio,
                                "policy_factor": ratio,
                                "coupled_ratio_bootstrap_interval_low": ratio - 0.05,
                                "coupled_ratio_bootstrap_interval_high": ratio + 0.05,
                                "primary_largest_5_blocks_fraction": 0.4,
                                "shell_largest_5_blocks_fraction": 0.5,
                            }
                        )

    review.order_retention(census, tmp_path)
    review.p6_policy_sensitivity(retained, cube_ids, tmp_path)
    review.p6_moment_concentration(retained, cube_ids, tmp_path)
    review.p6_retention(census, cube_ids, tmp_path)
    review.p6_top_tail(retained, tmp_path)

    assert len(list(tmp_path.glob("*.png"))) == 5
