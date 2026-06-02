from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase4 import generate_phase4_batch_b_tail_diagnostic_review as review
from scripts.phase4 import run_phase4_batch_b_tail_diagnostic as diagnostic


SHAPE = (
    len(diagnostic.Q_NAMES),
    len(diagnostic.DIRECTIONS),
    len(diagnostic.P_VALUES),
    3,
)
BLOCK_WEIGHTS = np.asarray([0.34, 0.24, 0.16, 0.12, 0.08, 0.06])


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _measurement(count: float, moment: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.full(SHAPE, count)
    sums = counts * moment
    sums_sq = counts * np.square(moment)
    return {
        "counts": counts,
        "sums": sums,
        "sums_sq": sums_sq,
        "block_counts": BLOCK_WEIGHTS[:, None, None, None, None] * counts,
        "block_sums": BLOCK_WEIGHTS[:, None, None, None, None] * sums,
        "block_sums_sq": BLOCK_WEIGHTS[:, None, None, None, None] * sums_sq,
    }


def _add_measurements(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        name: left[name] + right[name] for name in diagnostic.ACCUMULATOR_NAMES
    }


def _concentration(block_sums: np.ndarray, count: int) -> np.ndarray:
    ordered = np.sort(block_sums, axis=0)[::-1]
    total = ordered.sum(axis=0)
    return np.divide(
        ordered[:count].sum(axis=0),
        total,
        out=np.full_like(total, np.nan),
        where=total > 0.0,
    )


def _scenario_payload(sample_count: int, seed: int) -> dict[str, np.ndarray]:
    q = np.arange(SHAPE[0])[:, None, None, None]
    direction = np.arange(SHAPE[1])[None, :, None, None]
    p = np.arange(SHAPE[2])[None, None, :, None]
    ell = np.arange(SHAPE[3])[None, None, None, :]
    depth_scale = 1.0 + sample_count / 100000.0
    seed_scale = 1.0 + (seed - 20260530) / 1000.0
    moment = depth_scale * seed_scale * (1.0 + 0.10 * q + 0.04 * direction + 0.02 * p + 0.03 * ell)
    components = {
        "direct_interior": _measurement(4.0, moment),
        "direct_exterior": _measurement(6.0, 1.02 * moment),
        "shell_interior": _measurement(5.0, 0.98 * moment),
        "exterior_overlay": _measurement(7.0, 1.04 * moment),
    }
    components["direct_intrinsic"] = _add_measurements(
        components["direct_interior"], components["direct_exterior"]
    )
    components["stratified_recomposition_overlay"] = _add_measurements(
        components["shell_interior"], components["exterior_overlay"]
    )
    payload: dict[str, np.ndarray] = {
        "ell_bin_edges": np.asarray([0.0, 32.0, 64.0, 96.0]),
        "selected_displacements_ijk": np.asarray([[40, 0, 0], [72, 0, 0]], dtype=np.int64),
        "sample_count": np.asarray(sample_count),
        "seed": np.asarray(seed),
    }
    for component, arrays in components.items():
        for name, values in arrays.items():
            payload[f"{component}_{name}"] = values
        payload[f"{component}_moments"] = arrays["sums"] / arrays["counts"]
        payload[f"{component}_largest_block_fraction"] = _concentration(
            arrays["block_sums"], 1
        )
        payload[f"{component}_largest_5_blocks_fraction"] = _concentration(
            arrays["block_sums"], 5
        )
        payload[f"{component}_largest_10_blocks_fraction"] = _concentration(
            arrays["block_sums"], 10
        )
    for component in diagnostic.RAW_COMPONENTS:
        payload[f"{component}_sampled_origins"] = np.asarray(sample_count)
    payload["stratified_recomposition_over_direct_ratio"] = (
        payload["stratified_recomposition_overlay_moments"]
        / payload["direct_intrinsic_moments"]
    )
    return payload


def _create_release(root: Path) -> Path:
    rows = []
    selected = np.asarray([[40, 0, 0], [72, 0, 0]], dtype=np.int64)
    for cube_id in diagnostic.CUBE_IDS:
        for sample_count, seed in diagnostic.SCENARIOS:
            stem = f"{cube_id}_samples_{sample_count}_seed_{seed}"
            artifact = root / "scenarios" / f"{stem}.npz"
            events = root / "scenarios" / f"{stem}.top_events.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact, **_scenario_payload(sample_count, seed))
            _write_json(
                events,
                {
                    "schema_version": diagnostic.SCHEMA_VERSION,
                    "top_events": [
                        {
                            "component": "direct_interior",
                            "q_name": "B",
                            "direction": "parallel",
                            "block_id": 0,
                            "origin_kji": [1, 2, 3],
                            "displacement_ijk": [72, 0, 0],
                            "q_perp_magnitude": 2.0,
                            "p6_powered_contribution": 64.0,
                            "weighted_p6_contribution": 64.0,
                            "theta_degrees": 0.0,
                            "phi_degrees": None,
                        }
                    ],
                },
            )
            rows.append(
                {
                    "cube_id": cube_id,
                    "sample_count": sample_count,
                    "seed": seed,
                    "artifact_relative_path": str(artifact.relative_to(root)),
                    "artifact_sha256": file_sha256(artifact),
                    "top_events_relative_path": str(events.relative_to(root)),
                    "top_events_sha256": file_sha256(events),
                    "elapsed_seconds": 0.01,
                }
            )
    summary = {
        "schema_version": diagnostic.SCHEMA_VERSION,
        "status": "complete",
        "phase": "phase4_batch_b_bounded_matched_origin_tail_diagnostic",
        "cube_ids": diagnostic.CUBE_IDS,
        "q_names": diagnostic.Q_NAMES,
        "directions": diagnostic.DIRECTIONS,
        "p_values": diagnostic.P_VALUES,
        "scenarios": [
            {"sample_count": sample_count, "seed": seed}
            for sample_count, seed in diagnostic.SCENARIOS
        ],
        "science_scale_minimum_cells": diagnostic.SCIENCE_SCALE_MINIMUM,
        "selected_offsets_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "source_version": {"implementation_sha256": "synthetic-v2"},
        "scenario_rows": rows,
    }
    summary_path = root / diagnostic.SUMMARY_FILENAME
    _write_json(summary_path, summary)
    _write_json(
        root / diagnostic.MARKER_FILENAME,
        {
            "schema_version": diagnostic.SCHEMA_VERSION,
            "status": "complete",
            "summary_sha256": file_sha256(summary_path),
            "implementation_sha256": "synthetic-v2",
        },
    )
    return root


def test_screen_mask_applies_science_finite_positive_and_count_gates() -> None:
    retained, census = review._screen_mask(
        np.asarray([16.0, 48.0, 64.0, 80.0, 96.0]),
        np.asarray([1.0, np.nan, 1.0, 1.0, 0.0]),
        np.ones(5),
        np.asarray([10.0, 10.0, 1.0, 10.0, 10.0]),
        np.full(5, 10.0),
    )

    assert retained.tolist() == [False, False, False, True, False]
    assert census == {
        "all_bins": 5,
        "excluded_below_science_scale": 1,
        "science_scale_candidate_bins": 4,
        "excluded_nonfinite_or_nonpositive": 2,
        "excluded_below_minimum_count": 1,
        "retained_bins": 1,
    }


def test_direct_partition_verifier_rejects_numerical_mismatch() -> None:
    arrays = {}
    for name in diagnostic.ACCUMULATOR_NAMES:
        arrays[f"direct_interior_{name}"] = np.asarray([1.0, 2.0])
        arrays[f"direct_exterior_{name}"] = np.asarray([3.0, 4.0])
        arrays[f"direct_intrinsic_{name}"] = np.asarray([4.0, 6.0])

    review._verify_direct_partition(arrays, "synthetic")
    arrays["direct_intrinsic_sums"] = np.asarray([4.0, 6.5])

    with pytest.raises(RuntimeError, match="direct intrinsic partition mismatch"):
        review._verify_direct_partition(arrays, "synthetic")


def test_release_loader_rejects_broken_marker_summary_sha_chain(tmp_path: Path) -> None:
    release = _create_release(tmp_path / "release")
    summary_path = release / diagnostic.SUMMARY_FILENAME
    summary_path.write_text(summary_path.read_text() + "\n")

    with pytest.raises(RuntimeError, match="invalid or stale"):
        review._load_diagnostic_release(release)


def test_generate_review_publishes_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    release = _create_release(tmp_path / "release")
    output = tmp_path / "review"

    summary = review.generate_review(release, output)

    assert summary["converged_science_claimed"] is False
    assert summary["verification"]["direct_partition_verified"] is True
    assert summary["verification"]["scenario_count"] == 20
    manifest = json.loads((output / review.FIGURE_MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "passed"
    assert len(manifest["generated_figures"]) == 6
    assert set(manifest["figure_sha256"]) == set(manifest["generated_figures"])
    assert (output / review.SUMMARY_FILENAME).is_file()
    assert not list(tmp_path.glob(".review.*"))

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        review.generate_review(release, output)
