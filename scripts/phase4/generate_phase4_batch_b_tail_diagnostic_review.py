#!/usr/bin/env python3
"""Publish an immutable review of the bounded Phase 4 Batch B tail diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase4 import run_phase4_batch_b_tail_diagnostic as diagnostic


DEFAULT_OUTPUT_DIR = Path("figures/phase4_batch_b_tail_diagnostic_review")
SUMMARY_FILENAME = "phase4_batch_b_tail_diagnostic_review_summary.json"
FIGURE_MANIFEST_FILENAME = "figure_manifest.json"
MINIMUM_DIRECTIONAL_COUNT = 2
RARE_EVENT_LIMIT = 40
TOP_DEVIATION_LIMIT = 40
P6 = 6.0
P_COLORS = {
    2.0: "#4c78a8",
    4.0: "#f58518",
    6.0: "#e45756",
}
COMPONENT_COLORS = {
    "direct_intrinsic": "#4c78a8",
    "stratified_recomposition_overlay": "#f58518",
    "direct_interior": "#54a24b",
    "shell_interior": "#e45756",
    "direct_exterior": "#72b7b2",
    "exterior_overlay": "#b279a2",
    "stratified_shell_interior": "#ff9da6",
    "stratified_exterior_overlay": "#9d755d",
}
CONCENTRATION_COMPONENTS = (
    "direct_intrinsic",
    "stratified_recomposition_overlay",
    "direct_interior",
    "shell_interior",
)
TOP_EVENT_COMPONENTS = (
    *diagnostic.RAW_COMPONENTS,
    "stratified_shell_interior",
    "stratified_exterior_overlay",
)
TOP_EVENT_COUNT_COMPONENT = {
    "stratified_shell_interior": "shell_interior",
    "stratified_exterior_overlay": "exterior_overlay",
}


@dataclass(frozen=True)
class Scenario:
    cube_id: str
    sample_count: int
    seed: int
    artifact_path: Path
    top_events_path: Path
    arrays: Mapping[str, np.ndarray]
    top_events: tuple[dict[str, Any], ...]

    @property
    def key(self) -> tuple[str, int, int]:
        return self.cube_id, self.sample_count, self.seed

    @property
    def ell_bin_edges(self) -> np.ndarray:
        return self.arrays["ell_bin_edges"]

    @property
    def ell_bin_centers(self) -> np.ndarray:
        return 0.5 * (self.ell_bin_edges[:-1] + self.ell_bin_edges[1:])


@dataclass(frozen=True)
class DiagnosticRelease:
    root: Path
    summary: Mapping[str, Any]
    scenarios: Mapping[tuple[str, int, int], Scenario]
    input_sha256: Mapping[str, str]


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


def _json_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_builtin(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_builtin(value.tolist())
    if isinstance(value, np.generic):
        return _json_builtin(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_builtin(payload), indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _resolved_relative_artifact(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str):
        raise RuntimeError("diagnostic scenario artifact path must be a relative string")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise RuntimeError(f"diagnostic scenario artifact path must be relative: {candidate}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeError(f"diagnostic scenario artifact escapes release root: {candidate}")
    return resolved


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def _expected_npz_fields() -> set[str]:
    fields = {
        "ell_bin_edges",
        "selected_displacements_ijk",
        "sample_count",
        "seed",
        "stratified_recomposition_over_direct_ratio",
    }
    for component in diagnostic.COMPONENTS:
        fields.update(
            f"{component}_{name}" for name in diagnostic.ACCUMULATOR_NAMES
        )
        fields.update(
            (
                f"{component}_moments",
                f"{component}_largest_block_fraction",
                f"{component}_largest_5_blocks_fraction",
                f"{component}_largest_10_blocks_fraction",
            )
        )
    fields.update(
        f"{component}_sampled_origins" for component in diagnostic.RAW_COMPONENTS
    )
    return fields


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12, equal_nan=True):
        raise RuntimeError(f"tail-diagnostic derived array mismatch: {label}")


def _concentration(block_sums: np.ndarray, count: int) -> np.ndarray:
    ordered = np.sort(block_sums, axis=0)[::-1]
    total = ordered.sum(axis=0)
    return np.divide(
        ordered[:count].sum(axis=0),
        total,
        out=np.full_like(total, np.nan, dtype=float),
        where=total > 0.0,
    )


def _verify_direct_partition(arrays: Mapping[str, np.ndarray], label: str) -> None:
    for accumulator_name in diagnostic.ACCUMULATOR_NAMES:
        expected = (
            arrays[f"direct_interior_{accumulator_name}"]
            + arrays[f"direct_exterior_{accumulator_name}"]
        )
        actual = arrays[f"direct_intrinsic_{accumulator_name}"]
        if not np.array_equal(actual, expected, equal_nan=True):
            raise RuntimeError(
                "direct intrinsic partition mismatch for "
                f"{label}/{accumulator_name}: expected direct_interior + direct_exterior"
            )


def _verify_derived_arrays(arrays: Mapping[str, np.ndarray], label: str) -> None:
    for component in diagnostic.COMPONENTS:
        counts = arrays[f"{component}_counts"]
        sums = arrays[f"{component}_sums"]
        expected_moments = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=float),
            where=counts > 0.0,
        )
        _assert_close(
            arrays[f"{component}_moments"],
            expected_moments,
            f"{label}/{component}_moments",
        )
        block_sums = arrays[f"{component}_block_sums"]
        for count, suffix in (
            (1, "largest_block_fraction"),
            (5, "largest_5_blocks_fraction"),
            (10, "largest_10_blocks_fraction"),
        ):
            _assert_close(
                arrays[f"{component}_{suffix}"],
                _concentration(block_sums, count),
                f"{label}/{component}_{suffix}",
            )
    expected_ratio = np.divide(
        arrays["stratified_recomposition_overlay_moments"],
        arrays["direct_intrinsic_moments"],
        out=np.full_like(arrays["direct_intrinsic_moments"], np.nan, dtype=float),
        where=np.isfinite(arrays["direct_intrinsic_moments"])
        & (arrays["direct_intrinsic_moments"] > 0.0),
    )
    _assert_close(
        arrays["stratified_recomposition_over_direct_ratio"],
        expected_ratio,
        f"{label}/stratified_recomposition_over_direct_ratio",
    )


def _validate_scenario_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
    label: str,
) -> None:
    if set(arrays) != _expected_npz_fields():
        missing = sorted(_expected_npz_fields() - set(arrays))
        unexpected = sorted(set(arrays) - _expected_npz_fields())
        raise RuntimeError(
            f"tail-diagnostic NPZ schema mismatch for {label}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    edges = arrays["ell_bin_edges"]
    selected = arrays["selected_displacements_ijk"]
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
        or selected.ndim != 2
        or selected.shape[1] != 3
        or int(np.asarray(arrays["sample_count"]).item()) != sample_count
        or int(np.asarray(arrays["seed"]).item()) != seed
    ):
        raise RuntimeError(f"tail-diagnostic geometry metadata mismatch for {label}")
    shape = (
        len(diagnostic.Q_NAMES),
        len(diagnostic.DIRECTIONS),
        len(diagnostic.P_VALUES),
        len(edges) - 1,
    )
    block_count: int | None = None
    for component in diagnostic.COMPONENTS:
        for accumulator_name in diagnostic.ACCUMULATOR_NAMES:
            values = arrays[f"{component}_{accumulator_name}"]
            if accumulator_name.startswith("block_"):
                if values.ndim != len(shape) + 1 or values.shape[1:] != shape:
                    raise RuntimeError(
                        f"tail-diagnostic block accumulator shape mismatch: "
                        f"{label}/{component}_{accumulator_name}"
                    )
                if block_count is None:
                    block_count = values.shape[0]
                elif values.shape[0] != block_count:
                    raise RuntimeError(
                        f"tail-diagnostic block layout mismatch: "
                        f"{label}/{component}_{accumulator_name}"
                    )
            elif values.shape != shape:
                raise RuntimeError(
                    f"tail-diagnostic accumulator shape mismatch: "
                    f"{label}/{component}_{accumulator_name}"
                )
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise RuntimeError(
                    f"tail-diagnostic accumulator must be finite and non-negative: "
                    f"{label}/{component}_{accumulator_name}"
                )
        for suffix in (
            "moments",
            "largest_block_fraction",
            "largest_5_blocks_fraction",
            "largest_10_blocks_fraction",
        ):
            if arrays[f"{component}_{suffix}"].shape != shape:
                raise RuntimeError(
                    f"tail-diagnostic derived array shape mismatch: "
                    f"{label}/{component}_{suffix}"
                )
    if block_count is None or block_count <= 0:
        raise RuntimeError(f"tail-diagnostic block layout is empty: {label}")
    if arrays["stratified_recomposition_over_direct_ratio"].shape != shape:
        raise RuntimeError(f"tail-diagnostic ratio shape mismatch: {label}")
    for component in diagnostic.RAW_COMPONENTS:
        sampled = arrays[f"{component}_sampled_origins"]
        if sampled.shape != () or not np.isfinite(sampled) or float(sampled) < 0.0:
            raise RuntimeError(
                f"tail-diagnostic sampled-origin count mismatch: {label}/{component}"
            )
    _verify_direct_partition(arrays, label)
    _verify_derived_arrays(arrays, label)


def _validate_top_events(
    payload: Mapping[str, Any],
    *,
    block_count: int,
    label: str,
) -> tuple[dict[str, Any], ...]:
    if payload.get("schema_version") != diagnostic.SCHEMA_VERSION:
        raise RuntimeError(f"tail-diagnostic top-event schema mismatch: {label}")
    events = payload.get("top_events")
    if not isinstance(events, list):
        raise RuntimeError(f"tail-diagnostic top-event list is missing: {label}")
    validated = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise RuntimeError(f"tail-diagnostic top event must be an object: {label}/{index}")
        component = event.get("component")
        q_name = event.get("q_name")
        direction = event.get("direction")
        block_id = event.get("block_id")
        origin = event.get("origin_kji")
        displacement = event.get("displacement_ijk")
        q_perp = event.get("q_perp_magnitude")
        p6 = event.get("p6_powered_contribution")
        weighted_p6 = event.get("weighted_p6_contribution")
        if (
            component not in TOP_EVENT_COMPONENTS
            or q_name not in diagnostic.Q_NAMES
            or direction not in diagnostic.DIRECTIONS
            or not isinstance(block_id, int)
            or not 0 <= block_id < block_count
            or not isinstance(origin, (tuple, list))
            or len(origin) != 3
            or not all(isinstance(value, int) for value in origin)
            or not isinstance(displacement, (tuple, list))
            or len(displacement) != 3
            or not all(isinstance(value, int) for value in displacement)
            or not isinstance(q_perp, (int, float))
            or not np.isfinite(q_perp)
            or q_perp <= 0.0
            or not isinstance(p6, (int, float))
            or not np.isfinite(p6)
            or p6 <= 0.0
            or not isinstance(weighted_p6, (int, float))
            or not np.isfinite(weighted_p6)
            or weighted_p6 < 0.0
            or not np.isclose(p6, float(q_perp) ** 6, rtol=1.0e-12, atol=1.0e-12)
        ):
            raise RuntimeError(f"invalid tail-diagnostic top event: {label}/{index}")
        validated.append(dict(event))
    return tuple(validated)


def _load_scenario(
    root: Path,
    row: Mapping[str, Any],
    input_hashes: InputHashes,
) -> Scenario:
    cube_id = row.get("cube_id")
    sample_count = row.get("sample_count")
    seed = row.get("seed")
    if (
        cube_id not in diagnostic.CUBE_IDS
        or not isinstance(sample_count, int)
        or not isinstance(seed, int)
    ):
        raise RuntimeError(f"invalid tail-diagnostic scenario row: {row}")
    artifact_path = _resolved_relative_artifact(root, row.get("artifact_relative_path"))
    events_path = _resolved_relative_artifact(root, row.get("top_events_relative_path"))
    if row.get("artifact_sha256") != file_sha256(artifact_path):
        raise RuntimeError(f"tail-diagnostic scenario artifact changed after publication: {artifact_path}")
    if row.get("top_events_sha256") != file_sha256(events_path):
        raise RuntimeError(f"tail-diagnostic top-event sidecar changed after publication: {events_path}")
    with np.load(artifact_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    label = f"{cube_id}/samples_{sample_count}/seed_{seed}"
    _validate_scenario_arrays(arrays, sample_count=sample_count, seed=seed, label=label)
    block_count = arrays["direct_intrinsic_block_sums"].shape[0]
    top_events = _validate_top_events(
        _load_json(events_path), block_count=block_count, label=label
    )
    input_hashes.add(artifact_path)
    input_hashes.add(events_path)
    return Scenario(
        cube_id=cube_id,
        sample_count=sample_count,
        seed=seed,
        artifact_path=artifact_path,
        top_events_path=events_path,
        arrays=arrays,
        top_events=top_events,
    )


def _load_diagnostic_release(root: Path) -> DiagnosticRelease:
    root = root.resolve()
    summary_path = root / diagnostic.SUMMARY_FILENAME
    marker_path = root / diagnostic.MARKER_FILENAME
    summary = _load_json(summary_path)
    marker = _load_json(marker_path)
    implementation_sha256 = summary.get("source_version", {}).get("implementation_sha256")
    expected_scenarios = [
        {"sample_count": sample_count, "seed": seed}
        for sample_count, seed in diagnostic.SCENARIOS
    ]
    if (
        marker.get("schema_version") != diagnostic.SCHEMA_VERSION
        or marker.get("status") != "complete"
        or marker.get("summary_sha256") != file_sha256(summary_path)
        or marker.get("implementation_sha256") != implementation_sha256
        or summary.get("schema_version") != diagnostic.SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("phase") != "phase4_batch_b_bounded_matched_origin_tail_diagnostic"
        or tuple(summary.get("cube_ids", ())) != diagnostic.CUBE_IDS
        or tuple(summary.get("q_names", ())) != diagnostic.Q_NAMES
        or tuple(summary.get("directions", ())) != diagnostic.DIRECTIONS
        or tuple(summary.get("p_values", ())) != diagnostic.P_VALUES
        or summary.get("scenarios") != expected_scenarios
        or summary.get("science_scale_minimum_cells") != diagnostic.SCIENCE_SCALE_MINIMUM
    ):
        raise RuntimeError(f"invalid or stale Phase 4 Batch B tail diagnostic: {root}")
    input_hashes = InputHashes()
    input_hashes.add(summary_path)
    input_hashes.add(marker_path)
    rows = summary.get("scenario_rows")
    if not isinstance(rows, list):
        raise RuntimeError("tail-diagnostic scenario inventory is missing")
    expected_inventory = {
        (cube_id, sample_count, seed)
        for cube_id in diagnostic.CUBE_IDS
        for sample_count, seed in diagnostic.SCENARIOS
    }
    scenarios: dict[tuple[str, int, int], Scenario] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("tail-diagnostic scenario row must be an object")
        scenario = _load_scenario(root, row, input_hashes)
        if scenario.key in scenarios:
            raise RuntimeError(f"duplicate tail-diagnostic scenario: {scenario.key}")
        scenarios[scenario.key] = scenario
    if set(scenarios) != expected_inventory:
        raise RuntimeError("tail-diagnostic scenario inventory is incomplete or unexpected")
    reference = next(iter(scenarios.values()))
    expected_offsets_sha256 = summary.get("selected_offsets_sha256")
    if _array_sha256(reference.arrays["selected_displacements_ijk"]) != expected_offsets_sha256:
        raise RuntimeError("tail-diagnostic selected-offset hash mismatch")
    for scenario in scenarios.values():
        if (
            not np.array_equal(scenario.ell_bin_edges, reference.ell_bin_edges)
            or not np.array_equal(
                scenario.arrays["selected_displacements_ijk"],
                reference.arrays["selected_displacements_ijk"],
            )
        ):
            raise RuntimeError(f"tail-diagnostic scenario geometry changed: {scenario.key}")
    return DiagnosticRelease(
        root=root,
        summary=summary,
        scenarios=scenarios,
        input_sha256=input_hashes.as_dict(),
    )


def _screen_mask(
    ell: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_counts: np.ndarray,
    denominator_counts: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    if not (
        ell.shape
        == numerator.shape
        == denominator.shape
        == numerator_counts.shape
        == denominator_counts.shape
    ):
        raise RuntimeError("screened diagnostic arrays must share one shape")
    science = np.isfinite(ell) & (ell >= diagnostic.SCIENCE_SCALE_MINIMUM)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = numerator / denominator
    finite_positive = (
        np.isfinite(numerator)
        & (numerator > 0.0)
        & np.isfinite(denominator)
        & (denominator > 0.0)
        & np.isfinite(ratio)
        & (ratio > 0.0)
    )
    minimum_count = (
        np.isfinite(numerator_counts)
        & (numerator_counts >= MINIMUM_DIRECTIONAL_COUNT)
        & np.isfinite(denominator_counts)
        & (denominator_counts >= MINIMUM_DIRECTIONAL_COUNT)
    )
    retained = science & finite_positive & minimum_count
    return retained, {
        "all_bins": int(ell.size),
        "excluded_below_science_scale": int(np.count_nonzero(~science)),
        "science_scale_candidate_bins": int(np.count_nonzero(science)),
        "excluded_nonfinite_or_nonpositive": int(
            np.count_nonzero(science & ~finite_positive)
        ),
        "excluded_below_minimum_count": int(
            np.count_nonzero(science & finite_positive & ~minimum_count)
        ),
        "retained_bins": int(np.count_nonzero(retained)),
    }


def _empty_census() -> dict[str, int]:
    return {
        "all_bins": 0,
        "excluded_below_science_scale": 0,
        "science_scale_candidate_bins": 0,
        "excluded_nonfinite_or_nonpositive": 0,
        "excluded_below_minimum_count": 0,
        "retained_bins": 0,
    }


def _merge_census(destination: dict[str, int], source: Mapping[str, int]) -> None:
    for name in source:
        destination[name] += int(source[name])


def _ratio_rows(
    comparisons: Iterable[
        tuple[Scenario, Scenario, str, str, Mapping[str, Any]]
    ],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    census = _empty_census()
    for numerator_scenario, denominator_scenario, numerator_component, denominator_component, context in comparisons:
        if not np.array_equal(
            numerator_scenario.ell_bin_edges, denominator_scenario.ell_bin_edges
        ):
            raise RuntimeError("comparison scenarios do not share one ell grid")
        ell = numerator_scenario.ell_bin_centers
        for q_index, q_name in enumerate(diagnostic.Q_NAMES):
            for direction_index, direction in enumerate(diagnostic.DIRECTIONS):
                for p_index, p_value in enumerate(diagnostic.P_VALUES):
                    index = (q_index, direction_index, p_index)
                    numerator = numerator_scenario.arrays[
                        f"{numerator_component}_moments"
                    ][index]
                    denominator = denominator_scenario.arrays[
                        f"{denominator_component}_moments"
                    ][index]
                    numerator_counts = numerator_scenario.arrays[
                        f"{numerator_component}_counts"
                    ][index]
                    denominator_counts = denominator_scenario.arrays[
                        f"{denominator_component}_counts"
                    ][index]
                    retained, curve_census = _screen_mask(
                        ell, numerator, denominator, numerator_counts, denominator_counts
                    )
                    _merge_census(census, curve_census)
                    ratios = np.divide(
                        numerator,
                        denominator,
                        out=np.full_like(numerator, np.nan, dtype=float),
                        where=np.isfinite(denominator) & (denominator > 0.0),
                    )
                    for bin_index in np.flatnonzero(retained):
                        ratio = float(ratios[bin_index])
                        if not np.isfinite(ratio) or ratio <= 0.0:
                            raise RuntimeError("screened diagnostic ratio is not finite-positive")
                        rows.append(
                            {
                                **context,
                                "cube_id": numerator_scenario.cube_id,
                                "q_name": q_name,
                                "direction": direction,
                                "p_value": p_value,
                                "ell_cells": float(ell[bin_index]),
                                "numerator_component": numerator_component,
                                "denominator_component": denominator_component,
                                "numerator_count": float(numerator_counts[bin_index]),
                                "denominator_count": float(denominator_counts[bin_index]),
                                "ratio": ratio,
                                "diagnostic_factor": max(ratio, 1.0 / ratio),
                            }
                        )
    return rows, census


def _factor_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios = np.asarray([row["ratio"] for row in rows], dtype=float)
    factors = np.asarray([row["diagnostic_factor"] for row in rows], dtype=float)
    return {
        "count": int(len(rows)),
        "ratio_median": float(np.median(ratios)) if ratios.size else None,
        "ratio_p10": float(np.quantile(ratios, 0.10)) if ratios.size else None,
        "ratio_p90": float(np.quantile(ratios, 0.90)) if ratios.size else None,
        "diagnostic_factor_median": float(np.median(factors)) if factors.size else None,
        "diagnostic_factor_p90": float(np.quantile(factors, 0.90)) if factors.size else None,
        "diagnostic_factor_maximum": float(np.max(factors)) if factors.size else None,
    }


def _comparison_summary(
    rows: list[dict[str, Any]],
    census: Mapping[str, int],
    *,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "interpretation": interpretation,
        "screen_census": dict(census),
        "screened_ratio_summary": _factor_summary(rows),
        "top_diagnostic_deviations": sorted(
            rows, key=lambda row: row["diagnostic_factor"], reverse=True
        )[:TOP_DEVIATION_LIMIT],
    }


def _depth_comparisons(
    release: DiagnosticRelease,
) -> Iterable[tuple[Scenario, Scenario, str, str, Mapping[str, Any]]]:
    for cube_id in diagnostic.CUBE_IDS:
        reference = release.scenarios[(cube_id, 32768, 20260530)]
        for sample_count in (2048, 8192):
            yield (
                release.scenarios[(cube_id, sample_count, 20260530)],
                reference,
                "direct_intrinsic",
                "direct_intrinsic",
                {
                    "comparison": "depth_sensitivity_diagnostic",
                    "seed": 20260530,
                    "sample_count": sample_count,
                    "reference_sample_count": 32768,
                },
            )


def _seed_comparisons(
    release: DiagnosticRelease,
) -> Iterable[tuple[Scenario, Scenario, str, str, Mapping[str, Any]]]:
    for cube_id in diagnostic.CUBE_IDS:
        reference = release.scenarios[(cube_id, 8192, 20260530)]
        for seed in (20260531, 20260532):
            yield (
                release.scenarios[(cube_id, 8192, seed)],
                reference,
                "direct_intrinsic",
                "direct_intrinsic",
                {
                    "comparison": "seed_sensitivity_diagnostic",
                    "sample_count": 8192,
                    "seed": seed,
                    "reference_seed": 20260530,
                },
            )


def _same_scenario_comparisons(
    release: DiagnosticRelease,
    numerator_component: str,
    denominator_component: str,
    comparison: str,
) -> Iterable[tuple[Scenario, Scenario, str, str, Mapping[str, Any]]]:
    for scenario in release.scenarios.values():
        yield (
            scenario,
            scenario,
            numerator_component,
            denominator_component,
            {
                "comparison": comparison,
                "sample_count": scenario.sample_count,
                "seed": scenario.seed,
            },
        )


def _concentration_rows(
    release: DiagnosticRelease,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    census = _empty_census()
    census["excluded_nonfinite_or_out_of_range_concentration"] = 0
    census["retained_bins_after_concentration_check"] = 0
    p_index = diagnostic.P_VALUES.index(P6)
    for scenario in release.scenarios.values():
        ell = scenario.ell_bin_centers
        for component in CONCENTRATION_COMPONENTS:
            for q_index, q_name in enumerate(diagnostic.Q_NAMES):
                for direction_index, direction in enumerate(diagnostic.DIRECTIONS):
                    index = (q_index, direction_index, p_index)
                    moments = scenario.arrays[f"{component}_moments"][index]
                    counts = scenario.arrays[f"{component}_counts"][index]
                    retained, curve_census = _screen_mask(
                        ell, moments, moments, counts, counts
                    )
                    _merge_census(census, curve_census)
                    concentration = scenario.arrays[
                        f"{component}_largest_5_blocks_fraction"
                    ][index]
                    finite_concentration = (
                        np.isfinite(concentration)
                        & (concentration >= 0.0)
                        & (concentration <= 1.0)
                    )
                    census["excluded_nonfinite_or_out_of_range_concentration"] += int(
                        np.count_nonzero(retained & ~finite_concentration)
                    )
                    retained &= finite_concentration
                    census["retained_bins_after_concentration_check"] += int(
                        np.count_nonzero(retained)
                    )
                    for bin_index in np.flatnonzero(retained):
                        rows.append(
                            {
                                "cube_id": scenario.cube_id,
                                "sample_count": scenario.sample_count,
                                "seed": scenario.seed,
                                "component": component,
                                "q_name": q_name,
                                "direction": direction,
                                "p_value": P6,
                                "ell_cells": float(ell[bin_index]),
                                "count": float(counts[bin_index]),
                                "largest_5_blocks_fraction": float(
                                    concentration[bin_index]
                                ),
                            }
                        )
    return rows, census


def _concentration_summary(
    rows: list[dict[str, Any]], census: Mapping[str, int]
) -> dict[str, Any]:
    by_component = {}
    for component in CONCENTRATION_COMPONENTS:
        values = np.asarray(
            [
                row["largest_5_blocks_fraction"]
                for row in rows
                if row["component"] == component
            ],
            dtype=float,
        )
        by_component[component] = {
            "count": int(values.size),
            "median": float(np.median(values)) if values.size else None,
            "p90": float(np.quantile(values, 0.90)) if values.size else None,
            "maximum": float(np.max(values)) if values.size else None,
        }
    return {
        "interpretation": (
            "Diagnostic p=6 concentration only: fraction of the screened moment sum "
            "contributed by the five largest retained spatial blocks."
        ),
        "screen_census": dict(census),
        "screened_row_count": len(rows),
        "screened_by_component": by_component,
        "top_diagnostic_concentrations": sorted(
            rows, key=lambda row: row["largest_5_blocks_fraction"], reverse=True
        )[:TOP_DEVIATION_LIMIT],
    }


def _rare_event_rows(
    release: DiagnosticRelease,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    retained = []
    census = {
        "loaded_events": 0,
        "excluded_below_science_scale": 0,
        "excluded_nonfinite_or_nonpositive": 0,
        "excluded_below_minimum_count": 0,
        "retained_events_before_selection_limit": 0,
        "selected_events": 0,
    }
    p_index = diagnostic.P_VALUES.index(P6)
    for scenario in release.scenarios.values():
        ell = scenario.ell_bin_centers
        edges = scenario.ell_bin_edges
        for event in scenario.top_events:
            census["loaded_events"] += 1
            displacement = np.asarray(event["displacement_ijk"], dtype=float)
            displacement_ell = float(np.linalg.norm(displacement))
            bin_index = int(np.searchsorted(edges, displacement_ell, side="right") - 1)
            if not 0 <= bin_index < len(ell) or ell[bin_index] < diagnostic.SCIENCE_SCALE_MINIMUM:
                census["excluded_below_science_scale"] += 1
                continue
            weighted_p6 = float(event["weighted_p6_contribution"])
            if not np.isfinite(weighted_p6) or weighted_p6 <= 0.0:
                census["excluded_nonfinite_or_nonpositive"] += 1
                continue
            component = str(event["component"])
            count_component = TOP_EVENT_COUNT_COMPONENT.get(component, component)
            q_index = diagnostic.Q_NAMES.index(str(event["q_name"]))
            direction_index = diagnostic.DIRECTIONS.index(str(event["direction"]))
            count = float(
                scenario.arrays[f"{count_component}_counts"][
                    q_index, direction_index, p_index, bin_index
                ]
            )
            if not np.isfinite(count) or count < MINIMUM_DIRECTIONAL_COUNT:
                census["excluded_below_minimum_count"] += 1
                continue
            retained.append(
                {
                    **event,
                    "cube_id": scenario.cube_id,
                    "sample_count": scenario.sample_count,
                    "seed": scenario.seed,
                    "ell_cells": float(ell[bin_index]),
                    "displacement_ell_cells": displacement_ell,
                    "screen_count_component": count_component,
                    "screen_count": count,
                }
            )
    retained.sort(key=lambda row: row["weighted_p6_contribution"], reverse=True)
    census["retained_events_before_selection_limit"] = len(retained)
    selected = retained[:RARE_EVENT_LIMIT]
    census["selected_events"] = len(selected)
    return selected, census


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_ratio_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    filename: str,
    title: str,
    ylabel: str,
) -> Path:
    channels = [
        (q_name, direction)
        for q_name in diagnostic.Q_NAMES
        for direction in diagnostic.DIRECTIONS
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=(13.2, 7.8), constrained_layout=True, sharex=True, sharey=True
    )
    for axis, (q_name, direction) in zip(axes.flat, channels):
        for p_value in diagnostic.P_VALUES:
            selected = [
                row
                for row in rows
                if row["q_name"] == q_name
                and row["direction"] == direction
                and row["p_value"] == p_value
            ]
            axis.scatter(
                [row["ell_cells"] for row in selected],
                [row["ratio"] for row in selected],
                color=P_COLORS[p_value],
                s=8,
                alpha=0.24,
                label=f"p={int(p_value)}",
            )
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{q_name}: {direction}")
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[0, -1].legend(fontsize=7)
    figure.suptitle(title)
    return _save(figure, output_dir, filename)


def _plot_concentration(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.0), constrained_layout=True, sharex=True, sharey=True
    )
    for axis, component in zip(axes.flat, CONCENTRATION_COMPONENTS):
        selected = [row for row in rows if row["component"] == component]
        axis.scatter(
            [row["ell_cells"] for row in selected],
            [row["largest_5_blocks_fraction"] for row in selected],
            color=COMPONENT_COLORS[component],
            s=8,
            alpha=0.24,
        )
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(component.replace("_", " "))
        axis.set_xlabel(r"$\ell$ [cells]")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel("largest 5 blocks / p=6 sum")
    axes[1, 0].set_ylabel("largest 5 blocks / p=6 sum")
    figure.suptitle("Diagnostic p=6 top-5 block concentration after explicit screens")
    return _save(
        figure, output_dir, "phase4_batch_b_tail_diagnostic_p6_top5_block_concentration.png"
    )


def _plot_rare_events(
    selected_events: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    components = sorted({str(row["component"]) for row in selected_events})
    for component in components:
        rows = [row for row in selected_events if row["component"] == component]
        axes[0].scatter(
            [row["ell_cells"] for row in rows],
            [row["weighted_p6_contribution"] for row in rows],
            color=COMPONENT_COLORS[component],
            s=18,
            alpha=0.7,
            label=component,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\ell$ [cells]")
    axes[0].set_ylabel("weighted p=6 contribution")
    axes[0].set_title("selected screened events by scale")
    axes[0].grid(alpha=0.22)
    if components:
        axes[0].legend(fontsize=6)
    ranked = sorted(
        selected_events, key=lambda row: row["weighted_p6_contribution"], reverse=True
    )
    axes[1].scatter(
        np.arange(1, len(ranked) + 1),
        [row["weighted_p6_contribution"] for row in ranked],
        color=[COMPONENT_COLORS[str(row["component"])] for row in ranked],
        s=18,
        alpha=0.75,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("selected-event rank")
    axes[1].set_ylabel("weighted p=6 contribution")
    axes[1].set_title("selected screened rare-event ranking")
    axes[1].grid(alpha=0.22)
    figure.suptitle("Selected rare p=6 events: bounded diagnostic records, not converged claims")
    return _save(
        figure, output_dir, "phase4_batch_b_tail_diagnostic_selected_rare_p6_events.png"
    )


def _write_figure_manifest(
    output_dir: Path,
    *,
    input_sha256: Mapping[str, str],
) -> None:
    figures = sorted(path.name for path in output_dir.glob("*.png"))
    artifacts = sorted(path.name for path in output_dir.iterdir())
    _write_json(
        output_dir / FIGURE_MANIFEST_FILENAME,
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
            "input_sha256": dict(input_sha256),
        },
    )


def generate_review(diagnostic_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
    release = _load_diagnostic_release(diagnostic_root)
    depth_rows, depth_census = _ratio_rows(_depth_comparisons(release))
    seed_rows, seed_census = _ratio_rows(_seed_comparisons(release))
    overlay_rows, overlay_census = _ratio_rows(
        _same_scenario_comparisons(
            release,
            "stratified_recomposition_overlay",
            "direct_intrinsic",
            "stratified_overlay_over_direct_diagnostic",
        )
    )
    shell_rows, shell_census = _ratio_rows(
        _same_scenario_comparisons(
            release,
            "shell_interior",
            "direct_interior",
            "shell_interior_over_direct_interior_schedule_diagnostic",
        )
    )
    concentration_rows, concentration_census = _concentration_rows(release)
    rare_events, rare_event_census = _rare_event_rows(release)
    summary = {
        "schema_version": 1,
        "status": "phase4_batch_b_tail_diagnostic_review_generated",
        "decision_scope": (
            "bounded corrected-v2 matched-origin tail diagnostic review only; "
            "results are diagnostics rather than converged science claims"
        ),
        "converged_science_claimed": False,
        "automatic_expansion_claimed": False,
        "diagnostic_root": str(release.root),
        "configuration": {
            "diagnostic_release_schema_version": diagnostic.SCHEMA_VERSION,
            "q_names": diagnostic.Q_NAMES,
            "directions": diagnostic.DIRECTIONS,
            "p_values": diagnostic.P_VALUES,
            "science_scale_minimum_cells": diagnostic.SCIENCE_SCALE_MINIMUM,
            "science_scale_definition": "ell-bin center",
            "finite_positive_screen": True,
            "minimum_directional_count": MINIMUM_DIRECTIONAL_COUNT,
            "rare_event_selection_limit": RARE_EVENT_LIMIT,
            "directional_fitted_exponents_published": False,
        },
        "verification": {
            "diagnostic_marker_summary_sha_chain_verified": True,
            "scenario_npz_and_top_event_sha256_verified": True,
            "scenario_count": len(release.scenarios),
            "expected_scenario_count": len(diagnostic.CUBE_IDS) * len(diagnostic.SCENARIOS),
            "direct_partition_verified": True,
            "direct_partition_equation": (
                "direct_intrinsic == direct_interior + direct_exterior "
                "for every raw accumulator array"
            ),
            "verified_raw_accumulators": diagnostic.ACCUMULATOR_NAMES,
            "derived_moments_concentrations_and_overlay_ratio_recomputed": True,
        },
        "depth_sensitivity_diagnostic": _comparison_summary(
            depth_rows,
            depth_census,
            interpretation=(
                "At seed 20260530, screened direct-intrinsic moments for sample_count "
                "2048 and 8192 divided by the corresponding 32768-depth moments."
            ),
        ),
        "seed_sensitivity_diagnostic": _comparison_summary(
            seed_rows,
            seed_census,
            interpretation=(
                "At sample_count 8192, screened direct-intrinsic moments for seeds "
                "20260531 and 20260532 divided by the corresponding seed-20260530 moments."
            ),
        ),
        "stratified_overlay_over_direct_diagnostic": _comparison_summary(
            overlay_rows,
            overlay_census,
            interpretation=(
                "Screened stratified-recomposition-overlay moments divided by matched "
                "direct-intrinsic moments. This is a diagnostic overlay, not a replacement estimator."
            ),
        ),
        "shell_interior_over_direct_interior_schedule_diagnostic": _comparison_summary(
            shell_rows,
            shell_census,
            interpretation=(
                "Screened shell-interior moments divided by direct-interior moments to "
                "inspect schedule sensitivity within the interior stratum."
            ),
        ),
        "p6_top5_block_concentration_diagnostic": _concentration_summary(
            concentration_rows, concentration_census
        ),
        "selected_rare_p6_events_diagnostic": {
            "interpretation": (
                "Highest screened weighted p=6 event records retained for inspection. "
                "Selection is bounded and does not establish converged tail statistics."
            ),
            "screen_census": rare_event_census,
            "events": rare_events,
        },
        "input_sha256": dict(release.input_sha256),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _plot_ratio_diagnostic(
            depth_rows,
            temporary_output,
            filename="phase4_batch_b_tail_diagnostic_depth_sensitivity.png",
            title="Depth sensitivity diagnostic at seed 20260530",
            ylabel=r"$S_p(N) / S_p(N=32768)$",
        )
        _plot_ratio_diagnostic(
            seed_rows,
            temporary_output,
            filename="phase4_batch_b_tail_diagnostic_seed_sensitivity.png",
            title="Seed sensitivity diagnostic at sample_count 8192",
            ylabel=r"$S_p(\mathrm{seed}) / S_p(20260530)$",
        )
        _plot_ratio_diagnostic(
            overlay_rows,
            temporary_output,
            filename="phase4_batch_b_tail_diagnostic_stratified_overlay_over_direct.png",
            title="Stratified-overlay versus direct diagnostic ratios",
            ylabel=r"$S_{p,\rm overlay} / S_{p,\rm direct}$",
        )
        _plot_ratio_diagnostic(
            shell_rows,
            temporary_output,
            filename="phase4_batch_b_tail_diagnostic_shell_interior_schedule_ratio.png",
            title="Shell-interior versus direct-interior schedule diagnostic ratios",
            ylabel=r"$S_{p,\rm shell\ interior} / S_{p,\rm direct\ interior}$",
        )
        _plot_concentration(concentration_rows, temporary_output)
        _plot_rare_events(rare_events, temporary_output)
        _write_json(temporary_output / SUMMARY_FILENAME, summary)
        _write_figure_manifest(
            temporary_output, input_sha256=release.input_sha256
        )
        if output_dir.exists():
            raise RuntimeError(f"refusing to overwrite immutable report output: {output_dir}")
        temporary_output.rename(output_dir)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    generate_review(args.diagnostic_root, args.output_dir)
    print(f"Wrote immutable Phase 4 Batch B tail diagnostic review package: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
