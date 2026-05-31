"""Fast synthetic tests for Phase 3a runner provenance and crop helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.phase3a import run_phase3a_sampler as runner


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _build_bound_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build a minimal source-bound plan without touching Phase 2 or Lustre."""

    output_root = tmp_path / "phase3a"
    phase2_root = tmp_path / "phase2"
    source_version = {"implementation_sha256": "source-sha256"}
    phase2_identity = {"cube_id": "cube-a", "arrays_sha256": "phase2-sha256"}
    metadata = {
        "manifest_sha256": "manifest-sha256",
        "realized_offset_count": 3,
    }
    displacements = np.asarray(((1, 0, 0), (0, 1, 0), (-1, 0, 0)), dtype=np.int64)
    edges = np.asarray((0.5, 1.5, 2.5))

    monkeypatch.setattr(runner, "BENCHMARK_CUBE_IDS", ("cube-a",))
    monkeypatch.setattr(runner, "STENCIL_SPECS", {2: {}})
    monkeypatch.setattr(runner, "SUPPORT_MODES", ("shell_local",))
    monkeypatch.setattr(runner, "OFFSETS_PER_SHARD", 2)
    monkeypatch.setattr(runner, "_source_version", lambda: source_version)
    monkeypatch.setattr(
        runner,
        "_phase2_source_identity",
        lambda phase2_root, cube_id, *, verify_arrays: phase2_identity,
    )
    monkeypatch.setattr(
        runner,
        "_load_displacement_manifest",
        lambda output_root, stencil_width: (metadata, displacements, edges),
    )

    json_path, npz_path = runner._manifest_paths(output_root, 2)
    _write_json(json_path, {"synthetic": True})
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, synthetic=np.asarray((1,)))

    configuration = {"synthetic": True}
    displacement_row = {
        "json_relative_path": str(json_path.relative_to(output_root)),
        "json_sha256": runner.file_sha256(json_path),
        "npz_relative_path": str(npz_path.relative_to(output_root)),
        "npz_sha256": runner.file_sha256(npz_path),
        "manifest_sha256": metadata["manifest_sha256"],
        "offset_count": len(displacements),
    }
    shards = runner._planned_shards(output_root)
    campaign = {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "planned",
        "source_version": source_version,
        "configuration": configuration,
        "configuration_sha256": runner._mapping_sha256(configuration),
        "phase2_sources": {"cube-a": phase2_identity},
        "displacement_manifests": {"2": displacement_row},
        "shard_count": len(shards),
    }
    campaign_path = output_root / "manifests" / "campaign.json"
    shards_path = output_root / "manifests" / "shards.json"
    _write_json(campaign_path, campaign)
    _write_json(shards_path, {"shards": shards})
    _write_json(
        output_root / "PLAN_COMPLETE.json",
        {
            "schema_version": runner.SCHEMA_VERSION,
            "status": "passed",
            "campaign_sha256": runner.file_sha256(campaign_path),
            "shards_sha256": runner.file_sha256(shards_path),
            "implementation_sha256": source_version["implementation_sha256"],
        },
    )
    return phase2_root, output_root


def test_verify_json_diagnostic_rejects_stale_source_hash(tmp_path, monkeypatch):
    output_root = tmp_path / "diagnostic"
    old_source = {"implementation_sha256": "old-source"}
    monkeypatch.setattr(runner, "_source_version", lambda: old_source)
    runner._publish_json_diagnostic(
        output_root,
        "controls.json",
        "CONTROLS_COMPLETE.json",
        {
            "operational_status": "complete",
            "source_version": old_source,
        },
    )

    assert (
        runner._verify_json_diagnostic(
            output_root, "controls.json", "CONTROLS_COMPLETE.json"
        )["operational_status"]
        == "complete"
    )

    monkeypatch.setattr(
        runner,
        "_source_version",
        lambda: {"implementation_sha256": "new-source"},
    )
    with pytest.raises(RuntimeError, match="invalid or stale"):
        runner._verify_json_diagnostic(
            output_root, "controls.json", "CONTROLS_COMPLETE.json"
        )


def test_verify_json_diagnostic_rejects_changed_phase2_source_and_artifact(tmp_path, monkeypatch):
    output_root = tmp_path / "diagnostic"
    artifact_path = output_root / "scenarios" / "scenario_000.npz"
    artifact_path.parent.mkdir(parents=True)
    np.savez(artifact_path, value=np.asarray((1,)))
    source = {"implementation_sha256": "source"}
    phase2_identity = {"cube_id": "cube-a", "arrays_sha256": "phase2"}
    monkeypatch.setattr(runner, "_source_version", lambda: source)
    monkeypatch.setattr(
        runner,
        "_phase2_source_identity",
        lambda phase2_root, cube_id, *, verify_arrays: phase2_identity,
    )
    runner._publish_json_diagnostic(
        output_root,
        "convergence.json",
        "CONVERGENCE_COMPLETE.json",
        {
            "operational_status": "complete",
            "source_version": source,
            "cube_id": "cube-a",
            "phase2_source": phase2_identity,
            "rows": [
                {
                    "artifact_relative_path": "scenarios/scenario_000.npz",
                    "artifact_sha256": runner.file_sha256(artifact_path),
                }
            ],
        },
    )

    assert runner._verify_json_diagnostic(
        output_root,
        "convergence.json",
        "CONVERGENCE_COMPLETE.json",
        phase2_root=tmp_path / "phase2",
    )
    artifact_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact binding"):
        runner._verify_json_diagnostic(
            output_root,
            "convergence.json",
            "CONVERGENCE_COMPLETE.json",
            phase2_root=tmp_path / "phase2",
        )


def test_convergence_design_uses_full_census_only_for_bin_and_direction_evidence():
    displacements = np.column_stack(
        (np.arange(1, 121, dtype=np.int64), np.zeros(120, dtype=np.int64), np.zeros(120, dtype=np.int64))
    )
    edges = np.asarray((0.5, 40.5, 80.5, 120.5))

    full, full_policy = runner._convergence_offset_subset("directions", displacements, edges)
    bounded, bounded_policy = runner._convergence_offset_subset("origins", displacements, edges)

    assert np.array_equal(full, displacements)
    assert full_policy == "full_displacement_census"
    assert len(bounded) == 96
    assert bounded_policy == "shell_stratified_maximum_96_offsets"


def test_planned_shards_are_canonical_disjoint_exact_coverage(monkeypatch, tmp_path):
    lengths = {2: 5, 5: 3}
    monkeypatch.setattr(runner, "BENCHMARK_CUBE_IDS", ("cube-a",))
    monkeypatch.setattr(runner, "STENCIL_SPECS", {2: {}, 5: {}})
    monkeypatch.setattr(runner, "SUPPORT_MODES", ("shell_local", "all_valid_origins"))
    monkeypatch.setattr(runner, "OFFSETS_PER_SHARD", 2)
    monkeypatch.setattr(
        runner,
        "_load_displacement_manifest",
        lambda output_root, stencil_width: (
            {},
            np.zeros((lengths[stencil_width], 3), dtype=np.int64),
            np.asarray((0.5, 1.5)),
        ),
    )

    shards = runner._planned_shards(tmp_path)

    for stencil_width, offset_count in lengths.items():
        for support_mode in runner.SUPPORT_MODES:
            group = [
                row
                for row in shards
                if row["stencil_width"] == stencil_width
                and row["support_mode"] == support_mode
            ]
            assert [row["shard_index"] for row in group] == list(range(len(group)))
            assert [(row["offset_start"], row["offset_stop"]) for row in group] == [
                (start, min(offset_count, start + 2))
                for start in range(0, offset_count, 2)
            ]
            covered = [
                offset
                for row in group
                for offset in range(row["offset_start"], row["offset_stop"])
            ]
            assert covered == list(range(offset_count))
    assert len({row["shard_id"] for row in shards}) == len(shards)


def test_verify_plan_rejects_rechecksummed_noncanonical_shard_inventory(tmp_path, monkeypatch):
    phase2_root, output_root = _build_bound_plan(tmp_path, monkeypatch)
    assert runner._verify_plan(phase2_root, output_root, verify_arrays=False)

    shards_path = output_root / "manifests" / "shards.json"
    shards = json.loads(shards_path.read_text())
    shards["shards"][0]["offset_stop"] = 1
    _write_json(shards_path, shards)
    marker_path = output_root / "PLAN_COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["shards_sha256"] = runner.file_sha256(shards_path)
    _write_json(marker_path, marker)

    with pytest.raises(RuntimeError, match="canonical exact-once coverage"):
        runner._verify_plan(phase2_root, output_root, verify_arrays=False)


def test_verify_plan_rejects_rechecksummed_campaign_displacement_hash(tmp_path, monkeypatch):
    phase2_root, output_root = _build_bound_plan(tmp_path, monkeypatch)
    assert runner._verify_plan(phase2_root, output_root, verify_arrays=False)

    campaign_path = output_root / "manifests" / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign["displacement_manifests"]["2"]["json_sha256"] = "modified"
    _write_json(campaign_path, campaign)
    marker_path = output_root / "PLAN_COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["campaign_sha256"] = runner.file_sha256(campaign_path)
    _write_json(marker_path, marker)

    with pytest.raises(RuntimeError, match="lost displacement-manifest binding"):
        runner._verify_plan(phase2_root, output_root, verify_arrays=False)


def test_require_equivalent_rejects_finite_nan_mismatch():
    comparison = runner._relative_difference(
        np.asarray((1.0, np.nan)),
        np.asarray((1.0, 2.0)),
    )

    assert comparison["finite_mask_mismatch_count"] == 1
    assert comparison["equivalent"] is False
    with pytest.raises(RuntimeError, match="failed equivalence tolerance"):
        runner._require_equivalent(comparison, "finite-mask control")


def test_crop_cube_drops_prebuilt_vectors_and_returns_true_spatial_cube():
    spatial_shape = (5, 6, 7)
    cube = {
        "rho": np.arange(np.prod(spatial_shape)).reshape(spatial_shape),
        "B_x": np.ones(spatial_shape),
        "v_x": np.ones(spatial_shape),
        "_B_vector": np.ones((3, *spatial_shape)),
        "_v_vector": np.ones((3, *spatial_shape)),
    }

    cropped = runner._crop_cube(cube, 4)

    assert set(cropped) == {"rho", "B_x", "v_x"}
    assert {values.shape for values in cropped.values()} == {(4, 4, 4)}
    assert np.array_equal(cropped["rho"], cube["rho"][:4, :4, :4])
