"""Focused tests for the future Phase 4 completion-supplement generator."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.phase4 import generate_phase4_completion_supplement as supplement


def _curve_group(
    *,
    pair_local_values: tuple[float, ...] = (1.0, 4.0, 16.0),
    subvolume_mean_values: tuple[float, ...] = (0.5, 2.0, 8.0),
    effective_blocks: tuple[float, ...] = (12.0, 12.0, 12.0),
) -> SimpleNamespace:
    ell_edges = np.sqrt(2.0) * np.asarray((16.0, 32.0, 64.0, 128.0))
    shape = (2, 2, 2, 5, 1, 3)
    moments = np.ones(shape, dtype=float)
    counts = np.full(shape, 64, dtype=np.int64)
    exclusions = np.zeros((2, 2, 2, 3), dtype=np.int64)
    exclusions[:, :, 0] = 3
    exclusions[:, :, 1] = 5
    direction_names = ("all", "parallel", "perpendicular", "xi", "lambda")
    geometry_names = ("pair_local", "subvolume_mean")
    measurement_names = ("total", "perpendicular")
    for qi in range(2):
        for direction in ("parallel", "xi", "lambda"):
            di = direction_names.index(direction)
            moments[qi, 0, 1, di, 0] = np.asarray(pair_local_values)
            moments[qi, 1, 1, di, 0] = np.asarray(subvolume_mean_values)
    uncertainty = {
        "accepted_contributing_blocks": np.full(shape, 16, dtype=np.int64),
        "accepted_effective_blocks": np.broadcast_to(
            np.asarray(effective_blocks, dtype=float), shape
        ).copy(),
        "valid_bootstrap_resamples": np.full(shape, 200, dtype=np.int64),
        "block_bootstrap_interval_low": 0.8 * moments,
        "block_bootstrap_interval_high": 1.2 * moments,
        "block_bootstrap_standard_error": 0.1 * moments,
    }
    return SimpleNamespace(
        result=SimpleNamespace(
            ell_bin_edges=ell_edges,
            q_names=("B", "u"),
            geometry_names=geometry_names,
            measurement_names=measurement_names,
            direction_names=direction_names,
            p_values=(2.0,),
            moments=moments,
            counts=counts,
            exclusion_names=("weak_B_direction", "invalid_q"),
            exclusions=exclusions,
        ),
        uncertainty=uncertainty,
    )


def _catalog_rows(cube_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        cube_id: {
            "catalog": {
                "dBB": float(index + 1),
                "B_mean": float(10 - index),
                "deltaB": float(2 + index),
                "B_rms": float(11 + index),
                "B_mean_sq_over_B2_mean": float(0.8 - 0.1 * index),
                "deltaB_sq_over_B2_mean": float(0.2 + 0.1 * index),
            }
        }
        for index, cube_id in enumerate(cube_ids)
    }


def _environment_group() -> SimpleNamespace:
    group = _curve_group()
    p_count = len(supplement.BATCH_B_P_VALUES)
    group.result.p_values = supplement.BATCH_B_P_VALUES
    group.result.moments = np.repeat(group.result.moments, p_count, axis=4)
    group.result.counts = np.repeat(group.result.counts, p_count, axis=4)
    for name, values in tuple(group.uncertainty.items()):
        group.uncertainty[name] = np.repeat(values, p_count, axis=4)
    return group


def _reproduction_group(
    p_values: tuple[float, ...],
    *,
    elapsed_seconds: float = 1.0,
    staging_logical_bytes: int = 100,
) -> SimpleNamespace:
    p_count = len(p_values)

    def expanded(values: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(values), p_count, axis=-2)

    moment_values = np.asarray([[[[[[1.0, np.nan, 3.0]]]]]])
    count_values = np.asarray([[[[[[2, 0, 3]]]]]], dtype=np.int64)
    sum_values = np.asarray([[[[[[2.0, 0.0, 9.0]]]]]])
    sum_sq_values = np.asarray([[[[[[2.0, 0.0, 27.0]]]]]])
    block_values = np.asarray(
        [
            [[[[[[1.0, 0.0, 4.0]]]]]],
            [[[[[[1.0, 0.0, 5.0]]]]]],
        ]
    )
    result = SimpleNamespace(
        q_names=("B",),
        density_conventions=("not applicable",),
        geometry_names=("pair_local",),
        measurement_names=("total",),
        direction_names=("parallel",),
        exclusion_names=("invalid_q",),
        ell_bin_edges=np.asarray((1.0, 2.0, 4.0, 8.0)),
        p_values=p_values,
        counts=expanded(count_values),
        sums=expanded(sum_values),
        sums_sq=expanded(sum_sq_values),
        moments=expanded(moment_values),
        standard_error=expanded(np.asarray([[[[[[0.0, np.nan, 0.0]]]]]])),
        exclusions=np.asarray([[[[0, 1, 2]]]], dtype=np.int64),
        sampled_pairs=np.asarray((2, 2, 3), dtype=np.int64),
        eligible_pairs=np.asarray((10, 20, 30), dtype=np.int64),
        cube_candidate_pairs=np.asarray((12, 24, 36), dtype=np.int64),
        excluded_boundary_pairs=np.asarray((2, 4, 6), dtype=np.int64),
        displacements_per_bin=np.asarray((1, 1, 1), dtype=np.int64),
        displacements_ijk=np.asarray(((0, 0, 1), (0, 1, 0), (1, 0, 0))),
        ell_bin_index_per_displacement=np.asarray((0, 1, 2), dtype=np.int64),
        sampled_pairs_per_displacement=np.asarray((2, 2, 3), dtype=np.int64),
        eligible_pairs_per_displacement=np.asarray((10, 20, 30), dtype=np.int64),
        cube_candidate_pairs_per_displacement=np.asarray(
            (12, 24, 36), dtype=np.int64
        ),
        excluded_boundary_pairs_per_displacement=np.asarray(
            (2, 4, 6), dtype=np.int64
        ),
        out_of_range_displacements=0,
        pair_mode="all_valid_origins",
        cube_shape_kji=(640, 640, 640),
        nested_core_bounds_kji=None,
        rho0=float("nan"),
        rho0_provenance="not applicable for requested q variants",
        cell_sizes=(1.0, 1.0, 1.0),
        angle_limits={"parallel": 0.25},
        sample_count=2048,
        pair_batch_size=1024,
        seed=20260530,
        elapsed_seconds=elapsed_seconds,
        elapsed_seconds_per_ell_bin=np.asarray((0.1, 0.2, 0.3)),
        stencil_width=2,
        shell_core_bounds_kji=None,
        block_shape_kji=(80, 80, 80),
        block_counts=expanded(block_values.astype(np.int64)),
        block_sums=expanded(block_values),
        block_sums_sq=expanded(block_values),
        support_displacements_sha256="0" * 64,
        support_displacement_count=3,
        block_assignment="stencil_midpoint",
        block_sampled_origins=np.asarray(((1, 1, 1), (1, 1, 2)), dtype=np.int64),
        block_eligible_origins=np.asarray(((5, 10, 15), (5, 10, 15)), dtype=np.int64),
        block_exclusions=np.asarray([[[[[0, 1, 2]]]], [[[[0, 0, 0]]]]]),
        intrinsic_eligible_origins=np.asarray((10, 20, 30), dtype=np.int64),
        boundary_excluded_origins=np.asarray((2, 4, 6), dtype=np.int64),
        support_policy_excluded_origins=np.asarray((0, 0, 0), dtype=np.int64),
        intrinsic_eligible_origins_per_displacement=np.asarray(
            (10, 20, 30), dtype=np.int64
        ),
        boundary_excluded_origins_per_displacement=np.asarray(
            (2, 4, 6), dtype=np.int64
        ),
        support_policy_excluded_origins_per_displacement=np.asarray(
            (0, 0, 0), dtype=np.int64
        ),
    )
    uncertainty = {
        name: expanded(moment_values)
        for name in supplement.REPRODUCTION_UNCERTAINTY_P_AXIS_ARRAY_NAMES
    }
    uncertainty["metadata_json"] = np.asarray(
        json.dumps(
            {
                "schema_version": 1,
                "p_values": p_values,
                "q_names": result.q_names,
                "density_conventions": result.density_conventions,
                "rho0": result.rho0,
                "rho0_provenance": result.rho0_provenance,
                "bootstrap_seed": 20260531,
            },
            sort_keys=True,
        )
    )
    uncertainty["sampled_blocks_per_shell"] = np.asarray((2, 2, 2), dtype=np.int64)
    uncertainty["eligible_blocks_per_shell"] = np.asarray((2, 2, 2), dtype=np.int64)
    uncertainty["local_log_slope_support_mask"] = expanded(
        np.asarray([[[[[[True, False, True]]]]]])
    )
    return SimpleNamespace(
        result=result,
        uncertainty=uncertainty,
        staging_logical_bytes_before_marker=staging_logical_bytes,
    )


def _reproduction_groups(
    p_values: tuple[float, ...],
) -> dict[tuple[str, str], SimpleNamespace]:
    return {
        (cube_id, support_mode): _reproduction_group(p_values)
        for cube_id in supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS
        for support_mode in supplement.SUPPORT_MODES
    }


def test_common_equal_sf_targets_use_supported_directional_overlap() -> None:
    curves = {
        "parallel": np.asarray((1.0, 4.0, 16.0)),
        "xi": np.asarray((2.0, 8.0, 32.0)),
        "lambda": np.asarray((0.5, 2.0, 8.0)),
    }
    masks = {direction: np.ones(3, dtype=bool) for direction in supplement.DIRECTIONS}

    definition = supplement._common_equal_sf_targets(curves, masks)

    assert definition["status"] == "passed"
    assert definition["common_overlap_minimum"] == pytest.approx(2.0)
    assert definition["common_overlap_maximum"] == pytest.approx(8.0)
    assert definition["targets"] == pytest.approx(
        np.exp(
            np.log(2.0)
            + np.asarray(supplement.EQUAL_SF_TARGET_LOG_FRACTIONS)
            * (np.log(8.0) - np.log(2.0))
        )
    )


def test_supported_inverse_scale_does_not_bridge_failed_support_bin() -> None:
    supported = _curve_group()
    sparse = _curve_group(effective_blocks=(12.0, 1.0, 12.0))
    index = supplement.figures._moment_index(
        supported.result, "B", "parallel", p_value=2.0
    )

    scale, _, quality = supplement._supported_inverse_scale(
        supported, index, 8.0, ell_interval=(32.0, 128.0)
    )
    sparse_scale, _, sparse_quality = supplement._supported_inverse_scale(
        sparse, index, 8.0, ell_interval=(32.0, 128.0)
    )

    assert quality == "ok"
    assert scale == pytest.approx(64.0 * np.sqrt(2.0))
    assert sparse_quality == "no_crossing"
    assert sparse_scale is None


def test_supported_inverse_scale_interval_requires_block_stable_unique_crossings() -> None:
    group = _curve_group()
    index = supplement.figures._moment_index(
        group.result, "B", "parallel", p_value=2.0
    )

    interval = supplement._supported_inverse_scale_interval(
        group, index, 4.0, ell_interval=(32.0, 128.0)
    )

    assert interval["central_quality"] == "ok"
    assert interval["block_bootstrap_low_curve_crossing_quality"] == "ok"
    assert interval["block_bootstrap_high_curve_crossing_quality"] == "ok"
    assert interval["block_stable_unique_crossing"] is True
    assert (
        interval["block_bootstrap_curve_envelope_crossing_interval_low_cells"]
        < interval["scale_cells"]
        < interval["block_bootstrap_curve_envelope_crossing_interval_high_cells"]
    )

    group.uncertainty["block_bootstrap_interval_low"][index] = np.asarray(
        (1.0, 8.0, 1.0)
    )
    unstable = supplement._supported_inverse_scale_interval(
        group, index, 4.0, ell_interval=(32.0, 128.0)
    )

    assert unstable["block_bootstrap_low_curve_crossing_quality"] == "multiple_crossings"
    assert unstable["block_stable_unique_crossing"] is False
    assert unstable["block_bootstrap_curve_envelope_crossing_interval_low_cells"] is None


def test_equal_sf_aspect_rows_publish_only_block_stable_unique_crossings() -> None:
    cube_id = "cube"
    group = _curve_group()

    definitions, quality_census, aspects = supplement._equal_sf_aspect_rows(
        (cube_id,),
        {(cube_id, supplement.PRIMARY_SUPPORT_MODE): group},
    )

    assert len(definitions) == 2
    assert len(quality_census) == 2 * len(supplement.EQUAL_SF_TARGET_LOG_FRACTIONS)
    assert len(aspects) == len(quality_census)
    assert all(row["aspect_table_eligible"] for row in aspects)
    assert all(
        row["xi_over_lambda_block_bootstrap_curve_envelope_interval_low"]
        <= row["xi_over_lambda"]
        <= row["xi_over_lambda_block_bootstrap_curve_envelope_interval_high"]
        for row in aspects
    )

    index = supplement.figures._moment_index(
        group.result, "B", "parallel", p_value=2.0
    )
    group.uncertainty["block_bootstrap_interval_low"][index] = np.asarray(
        (1.0, 8.0, 1.0)
    )
    _, unstable_census, unstable_aspects = supplement._equal_sf_aspect_rows(
        (cube_id,),
        {(cube_id, supplement.PRIMARY_SUPPORT_MODE): group},
    )

    assert len(unstable_aspects) < len(unstable_census)
    assert any(not row["aspect_table_eligible"] for row in unstable_census)


def test_conditioning_rows_compare_pair_local_and_subvolume_mean_products() -> None:
    cube_id = "cube"
    rows = supplement._conditioning_rows(
        (cube_id,),
        {(cube_id, supplement.PRIMARY_SUPPORT_MODE): _curve_group()},
    )

    assert len(rows) == 2 * 3 * 3
    assert {row["supported_comparison"] for row in rows} == {True}
    assert {row["pair_local_over_subvolume_mean"] for row in rows} == {2.0}
    assert {row["conditioning_factor"] for row in rows} == {2.0}


def test_sf_environment_rows_publish_rooted_amplitude_raw_moment_and_counts() -> None:
    cube_id = "cube0"
    rows = supplement._sf_environment_rows(
        (cube_id,),
        {(cube_id, supplement.PRIMARY_SUPPORT_MODE): _environment_group()},
        _catalog_rows((cube_id,)),
    )
    selected = next(
        row
        for row in rows
        if row["p_value"] == 2.0
        and row["q_name"] == "B"
        and row["direction"] == "parallel"
        and row["requested_ell_cells"] == 64.0
    )

    assert len(rows) == 6 * 2 * 3 * 3
    assert selected["raw_moment_S_p_perpendicular"] == pytest.approx(4.0)
    assert selected["rooted_amplitude_A_p_perpendicular"] == pytest.approx(2.0)
    assert selected["accepted_measurements"] == 64
    assert selected["pre_wedge_shell_exclusion_total"] == 8
    assert selected["pre_wedge_shell_exclusions"] == {
        "weak_B_direction": 3,
        "invalid_q": 5,
    }
    assert selected["B_rms"] == pytest.approx(11.0)


def test_sf_environment_correlations_report_denominator_outlier_sensitivity() -> None:
    rows = [
        {
            "p_value": 2.0,
            "q_name": "B",
            "direction": "parallel",
            "requested_ell_cells": 64.0,
            "actual_shell_center_cells": 64.0,
            "rooted_amplitude_A_p_perpendicular": amplitude,
            "dBB": dbb,
            "B_mean": 10.0 - index,
            "deltaB": 2.0 + index,
            "B_rms": 11.0 + index,
            "B_mean_sq_over_B2_mean": 0.8 - 0.1 * index,
            "deltaB_sq_over_B2_mean": 0.2 + 0.1 * index,
            "accepted_measurements": 100 + index,
            "pre_wedge_shell_exclusion_total": 10 + index,
        }
        for index, (dbb, amplitude) in enumerate(
            ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (12.0, 0.5))
        )
    ]

    correlations = supplement._sf_environment_correlations(rows)
    selected = next(
        row
        for row in correlations
        if row["p_value"] == 2.0
        and row["q_name"] == "B"
        and row["direction"] == "parallel"
        and row["requested_ell_cells"] == 64.0
        and row["dependence_variable"] == "dBB"
    )

    assert selected["full_census_cube_count"] == 4
    assert selected["without_denominator_outliers_cube_count"] == 3
    assert selected["without_denominator_outliers_spearman_rho"] == pytest.approx(1.0)


def test_refined_environment_and_order_figures_render_full_variable_inventory(
    tmp_path: Path,
) -> None:
    cube_id = "cube0"
    catalog_rows = _catalog_rows((cube_id,))
    environment_rows = supplement._sf_environment_rows(
        (cube_id,),
        {(cube_id, supplement.PRIMARY_SUPPORT_MODE): _environment_group()},
        catalog_rows,
    )
    retained = [
        {
            "cube_id": cube_id,
            "p_value": 1.0,
            "q_name": "B",
            "direction": "parallel",
            "policy_factor": 1.2,
        }
    ]
    order_rows = supplement._order_sensitivity_rows(retained, (cube_id,), catalog_rows)

    assert supplement.sf_environment_figure(environment_rows, tmp_path).is_file()
    assert supplement.order_sensitivity_figure(order_rows, tmp_path).is_file()


def test_stencil_rows_keep_labeled_products_distinct() -> None:
    cube_id = next(iter(supplement.batch_a2_review.REPRESENTATIVE_CUBES))
    two = _curve_group(pair_local_values=(1.0, 2.0, 4.0))
    three = _curve_group(pair_local_values=(2.0, 4.0, 8.0))

    rows = supplement._stencil_comparison_rows(
        (cube_id,),
        {(cube_id, supplement.PRIMARY_SUPPORT_MODE): two},
        {(cube_id, supplement.PRIMARY_SUPPORT_MODE): three},
    )

    assert len(rows) == 2 * 3 * 3
    assert {row["scope"] for row in rows} == {"representative_overlap"}
    assert {row["three_point_over_two_point"] for row in rows} == {2.0}
    assert {row["labeled_stencil_factor"] for row in rows} == {2.0}
    assert all("distinctly labeled" in row["interpretation"] for row in rows)


def test_order_sensitivity_rows_join_phase1_magnetic_complements() -> None:
    cube_ids = ("cube0", "cube1", "cube2")
    retained = [
        {
            "cube_id": cube_id,
            "p_value": 1.0,
            "q_name": "B",
            "direction": "parallel",
            "policy_factor": factor,
        }
        for cube_id, factor in zip(cube_ids, (1.1, 1.3, 1.7))
    ]

    rows = supplement._order_sensitivity_rows(
        retained, cube_ids, _catalog_rows(cube_ids)
    )
    selected = [
        row
        for row in rows
        if row["p_value"] == 1.0
        and row["q_name"] == "B"
        and row["direction"] == "parallel"
    ]
    correlations = supplement._order_sensitivity_correlations(rows)
    dbb = next(
        row
        for row in correlations
        if row["p_value"] == 1.0
        and row["q_name"] == "B"
        and row["direction"] == "parallel"
        and row["magnetic_covariate"] == "dBB"
    )

    assert len(rows) == len(cube_ids) * 6 * 2 * 3
    assert [row["deltaB_sq_over_B2_mean"] for row in selected] == pytest.approx(
        (0.2, 0.3, 0.4)
    )
    assert dbb["cube_count"] == 3
    assert dbb["spearman_rho"] == pytest.approx(1.0)


def test_staged_dependency_binding_rejects_wrong_extension_root(tmp_path: Path) -> None:
    cube_ids = ("cube0", "cube1")
    batch_a_root = tmp_path / "batch_a"
    extension_root = tmp_path / "extension"
    batch_a = {
        "phase": "phase4_batch_a_bounded_21_cube_2point",
        "pilot_cube_ids": cube_ids,
    }
    extension = {
        "phase": "phase4_batch_a2_all21_3point_extension",
        "pilot_cube_ids": cube_ids,
        "batch_a_reference_root": str(batch_a_root),
    }
    batch_b = {
        "phase": "phase4_batch_b_all21_2point_p1_to_p6_extension",
        "pilot_cube_ids": cube_ids,
        "batch_a_reference_root": str(batch_a_root),
        "all21_3point_extension_reference_root": str(extension_root),
    }

    supplement._verify_dependency_bindings(
        batch_a_summary=batch_a,
        extension_summary=extension,
        batch_b_summary=batch_b,
        batch_a_root=batch_a_root,
        extension_root=extension_root,
        cube_ids=cube_ids,
    )
    batch_b["all21_3point_extension_reference_root"] = str(tmp_path / "wrong")

    with pytest.raises(RuntimeError, match="not one bound staged release chain"):
        supplement._verify_dependency_bindings(
            batch_a_summary=batch_a,
            extension_summary=extension,
            batch_b_summary=batch_b,
            batch_a_root=batch_a_root,
            extension_root=extension_root,
            cube_ids=cube_ids,
        )


def test_campaign_density_conventions_binding_accepts_explicit_labels() -> None:
    assert supplement._campaign_density_conventions_binding(
        {"density_conventions": ("not applicable", "not applicable")},
        ("not applicable", "not applicable"),
    ) == "explicit_campaign_manifest_and_every_reduction"


def test_campaign_density_conventions_binding_accepts_legacy_not_applicable_omission() -> None:
    assert supplement._campaign_density_conventions_binding(
        {},
        ("not applicable", "not applicable"),
    ) == "legacy_manifest_omission_revalidated_from_every_reduction"


def test_campaign_density_conventions_binding_rejects_ambiguous_or_wrong_labels() -> None:
    assert supplement._campaign_density_conventions_binding({}, ("pointwise",)) is None
    assert (
        supplement._campaign_density_conventions_binding(
            {"density_conventions": ("pointwise",)},
            ("not applicable",),
        )
        is None
    )


def test_sampling_schedule_hash_can_replay_legacy_base_schema() -> None:
    row = {
        "group_id": "cube/stencil_3point/all_valid_origins",
        "shard_id": "cube/stencil_3point/all_valid_origins/shard_0000",
        "offset_start": 0,
        "offset_stop": 10,
    }
    configuration = {
        "sample_count_per_displacement": 2048,
        "pair_batch_size": 1024,
        "production_seed": 20260530,
        "block_shape_kji": (80, 80, 80),
        "block_assignment": "stencil_midpoint",
        "q_names": ("B", "u"),
        "p_values": (2.0,),
        "density_conventions": ("not applicable", "not applicable"),
    }

    legacy = supplement._sampling_schedule_sha256(
        row, configuration, include_quantity_axes=False
    )
    explicit = supplement._sampling_schedule_sha256(row, configuration)

    assert legacy != explicit


def test_strict_batch_a_to_all21_batch_b_p2_reproduction_passes() -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS

    verification = supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
        cube_ids=cube_ids,
        batch_a_groups=_reproduction_groups((2.0,)),
        batch_b_groups=_reproduction_groups(supplement.BATCH_B_P_VALUES),
    )

    assert verification == {
        "status": "passed",
        "verification_mode": "exact_arrays_equal_nan",
        "p_value": 2.0,
        "verified_group_count": 42,
        "excluded_metadata": "timing_and_staging_only",
    }


def test_strict_batch_a_to_all21_batch_b_p2_reproduction_allows_legacy_batch_a_uncertainty_metadata() -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS
    batch_a_groups = _reproduction_groups((2.0,))
    for group in batch_a_groups.values():
        metadata = json.loads(str(group.uncertainty["metadata_json"].item()))
        for name in (
            "p_values",
            "q_names",
            "density_conventions",
            "rho0",
            "rho0_provenance",
        ):
            metadata.pop(name)
        group.uncertainty["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )

    verification = supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
        cube_ids=cube_ids,
        batch_a_groups=batch_a_groups,
        batch_b_groups=_reproduction_groups(supplement.BATCH_B_P_VALUES),
    )

    assert verification["status"] == "passed"


def test_strict_batch_a_to_all21_batch_b_p2_reproduction_rejects_modern_batch_b_uncertainty_metadata_omission() -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS
    batch_b_groups = _reproduction_groups(supplement.BATCH_B_P_VALUES)
    first = batch_b_groups[(cube_ids[0], supplement.SUPPORT_MODES[0])]
    metadata = json.loads(str(first.uncertainty["metadata_json"].item()))
    metadata.pop("p_values")
    first.uncertainty["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )

    with pytest.raises(RuntimeError, match="quantity-axis inventory"):
        supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
            cube_ids=cube_ids,
            batch_a_groups=_reproduction_groups((2.0,)),
            batch_b_groups=batch_b_groups,
        )


def test_strict_batch_a_to_all21_batch_b_p2_reproduction_rejects_result_mismatch() -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS
    batch_b_groups = _reproduction_groups(supplement.BATCH_B_P_VALUES)
    first = batch_b_groups[(cube_ids[0], supplement.SUPPORT_MODES[0])]
    first.result.sums[..., 1, 0] += 1.0

    with pytest.raises(RuntimeError, match=r"result\.sums"):
        supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
            cube_ids=cube_ids,
            batch_a_groups=_reproduction_groups((2.0,)),
            batch_b_groups=batch_b_groups,
        )


def test_strict_batch_a_to_all21_batch_b_p2_reproduction_rejects_per_displacement_support_mismatch() -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS
    batch_b_groups = _reproduction_groups(supplement.BATCH_B_P_VALUES)
    first = batch_b_groups[(cube_ids[0], supplement.SUPPORT_MODES[0])]
    first.result.support_policy_excluded_origins_per_displacement[0] += 1

    with pytest.raises(
        RuntimeError, match="result.support_policy_excluded_origins_per_displacement"
    ):
        supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
            cube_ids=cube_ids,
            batch_a_groups=_reproduction_groups((2.0,)),
            batch_b_groups=batch_b_groups,
        )


def test_strict_batch_a_to_all21_batch_b_p2_reproduction_rejects_uncertainty_mismatch() -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS
    batch_b_groups = _reproduction_groups(supplement.BATCH_B_P_VALUES)
    first = batch_b_groups[(cube_ids[0], supplement.SUPPORT_MODES[0])]
    first.uncertainty["block_bootstrap_interval_low"][..., 1, 0] += 1.0

    with pytest.raises(RuntimeError, match=r"uncertainty\.block_bootstrap_interval_low"):
        supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
            cube_ids=cube_ids,
            batch_a_groups=_reproduction_groups((2.0,)),
            batch_b_groups=batch_b_groups,
        )


def test_strict_batch_a_to_all21_batch_b_p2_reproduction_allows_timing_differences() -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS
    batch_b_groups = _reproduction_groups(supplement.BATCH_B_P_VALUES)
    first = batch_b_groups[(cube_ids[0], supplement.SUPPORT_MODES[0])]
    first.result.elapsed_seconds = 999.0
    first.result.elapsed_seconds_per_ell_bin[:] = 999.0
    first.staging_logical_bytes_before_marker = 999

    verification = supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
        cube_ids=cube_ids,
        batch_a_groups=_reproduction_groups((2.0,)),
        batch_b_groups=batch_b_groups,
    )

    assert verification["status"] == "passed"


@pytest.mark.parametrize(
    "p_values",
    (
        (1.0, 2.0, 3.0, 4.0, 5.0),
        (1.0, 2.0, 2.0, 4.0, 5.0, 6.0),
        (2.0, 1.0, 3.0, 4.0, 5.0, 6.0),
    ),
    ids=("missing", "duplicate", "reordered"),
)
def test_strict_batch_a_to_all21_batch_b_p2_reproduction_rejects_noncanonical_batch_b_p_values(
    p_values: tuple[float, ...],
) -> None:
    cube_ids = supplement.batch_a_report.FROZEN_PHASE4_PILOT_CUBE_IDS

    with pytest.raises(RuntimeError, match="Batch B p-values"):
        supplement._verify_batch_a_to_all21_batch_b_p2_reproduction(
            cube_ids=cube_ids,
            batch_a_groups=_reproduction_groups((2.0,)),
            batch_b_groups=_reproduction_groups(p_values),
        )


def test_directional_fit_policy_explicitly_withholds_slopes_and_exponents() -> None:
    policy = supplement._directional_fit_policy()

    assert policy["status"] == "explicitly_withheld"
    assert policy["directional_fitted_slopes_published"] is False
    assert policy["directional_fitted_exponents_published"] is False
    assert policy["directional_zeta_published"] is False


def test_atomic_directory_publish_refuses_existing_output(tmp_path: Path) -> None:
    first = tmp_path / "first"
    destination = tmp_path / "published"
    first.mkdir()
    (first / "artifact.txt").write_text("first\n")

    supplement._publish_atomic_directory(first, destination)

    assert (destination / "artifact.txt").read_text() == "first\n"
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        supplement._publish_atomic_directory(second, destination)
    assert second.is_dir()
