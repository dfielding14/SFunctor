"""Restart and provenance tests for the bounded Phase 3 sampler driver."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.phase1.cbin_tools import file_sha256
from scripts.phase3 import run_phase3_sampler as phase3


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _phase2_cube(tmp_path: Path, cube_id: str = phase3.BENCHMARK_CUBE_IDS[0]) -> tuple[Path, dict]:
    root = tmp_path / "phase2"
    cube = root / cube_id
    output_fields = {}
    for manifest_name in phase3.MANIFEST_FIELD_NAMES.values():
        path = cube / "fields" / f"{manifest_name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{manifest_name}-payload".encode())
        output_fields[manifest_name] = {
            "relative_path": f"fields/{manifest_name}.npy",
            "sha256": file_sha256(path),
        }
    manifest = {"output_fields": output_fields}
    _write_json(cube / "manifest.json", manifest)
    _write_json(
        cube / "COMPLETE.json",
        {"cube_id": cube_id, "manifest_sha256": file_sha256(cube / "manifest.json")},
    )
    return root, phase3._phase2_source_identity(root, cube_id)


def _phase3_cube(
    tmp_path: Path,
    phase2_source: dict,
    source_version: dict,
    *,
    cube_id: str = phase3.BENCHMARK_CUBE_IDS[0],
    configuration_sha256: str | None = None,
    configuration: dict | None = None,
) -> Path:
    root = tmp_path / "phase3" / cube_id
    full_configuration = {
        "cube_id": cube_id,
        "setting": "shared",
        "q_names": ["B", "u"],
        "fit_interval_cells": [8.0, 96.0],
        "fit_min_count": 100,
    }
    full_configuration.update(configuration or {})
    configuration = full_configuration
    configuration_sha256 = configuration_sha256 or phase3._mapping_sha256(configuration)
    result_paths = {
        "nested_core.npz": root / "nested_core.npz",
        "all_valid_pairs.npz": root / "all_valid_pairs.npz",
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, result in result_paths.items():
        result.write_bytes(f"{name}-payload".encode())
    _write_json(
        root / "summary.json",
        {
            "schema_version": phase3.SCHEMA_VERSION,
            "status": "passed",
            "cube_id": cube_id,
            "phase2_source": phase2_source,
            "configuration": configuration,
            "configuration_sha256": configuration_sha256,
            "source_version": source_version,
            "nested_vs_all_valid": {
                "sampling_depth_matched": True,
            },
            "mode_summaries": {
                "nested_core": {
                    "coverage_rows": [
                        {
                            "q": q_name,
                            "geometry": "subvolume_mean",
                            "measurement": "perpendicular",
                            "direction": "parallel",
                            "p": 2.0,
                            "ell_center": ell,
                            "accepted": 100,
                        }
                        for q_name in ("B", "u")
                        for ell in (8.0, 16.0, 32.0, 64.0, 96.0)
                    ],
                },
            },
            "performance": {
                "total_wall_seconds": 1.25,
                "peak_rss_kib": 4096,
            },
        },
    )
    _write_json(
        root / "COMPLETE.json",
        {
            "schema_version": phase3.SCHEMA_VERSION,
            "status": "passed",
            "cube_id": cube_id,
            "phase2_source": phase2_source,
            "implementation_sha256": source_version["implementation_sha256"],
            "configuration_sha256": configuration_sha256,
            "summary_sha256": file_sha256(root / "summary.json"),
            "result_sha256": {
                name: file_sha256(result)
                for name, result in result_paths.items()
            },
        },
    )
    return root


def _seed_robustness(
    output_root: Path,
    phase2_source: dict,
    source_version: dict,
    *,
    cube_id: str = phase3.BENCHMARK_CUBE_IDS[0],
) -> Path:
    root = output_root / "seed_robustness"
    configuration = {"cube_id": cube_id, "setting": "matched-depth"}
    configuration_sha256 = phase3._mapping_sha256(configuration)
    result_paths = {
        f"all_valid_pairs_seed{seed}.npz": root / f"all_valid_pairs_seed{seed}.npz"
        for seed in phase3.ROBUSTNESS_SEEDS[1:]
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, result in result_paths.items():
        result.write_bytes(f"{name}-payload".encode())
    _write_json(
        root / "seed_robustness_summary.json",
        {
            "schema_version": phase3.SCHEMA_VERSION,
            "status": "passed",
            "cube_id": cube_id,
            "phase2_source": phase2_source,
            "source_version": source_version,
            "configuration": configuration,
            "base_result_sha256": file_sha256(output_root / cube_id / "all_valid_pairs.npz"),
        },
    )
    _write_json(
        root / "ROBUSTNESS_COMPLETE.json",
        {
            "schema_version": phase3.SCHEMA_VERSION,
            "status": "passed",
            "cube_id": cube_id,
            "phase2_source": phase2_source,
            "implementation_sha256": source_version["implementation_sha256"],
            "configuration_sha256": configuration_sha256,
            "summary_sha256": file_sha256(root / "seed_robustness_summary.json"),
            "result_sha256": {
                name: file_sha256(result)
                for name, result in result_paths.items()
            },
        },
    )
    return root


def _synthetic_validation(output_root: Path, source_version: dict, *, status: str = "passed") -> Path:
    path = output_root / "synthetic_validation.json"
    _write_json(
        path,
        {
            "schema_version": phase3.SCHEMA_VERSION,
            "status": status,
            "source_version": source_version,
        },
    )
    return path


def _complete_campaign(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict]:
    source_version = {"implementation_sha256": "implementation-a"}
    monkeypatch.setattr(phase3, "_source_version", lambda: source_version)
    phase2_root = tmp_path / "phase2"
    output_root = tmp_path / "phase3"
    for cube_id in phase3.BENCHMARK_CUBE_IDS:
        _, phase2_source = _phase2_cube(tmp_path, cube_id)
        _phase3_cube(tmp_path, phase2_source, source_version, cube_id=cube_id)
        if cube_id == phase3.BENCHMARK_CUBE_IDS[0]:
            _seed_robustness(output_root, phase2_source, source_version)
    return phase2_root, output_root, source_version


def test_phase2_source_identity_rehashes_actual_analysis_inputs(tmp_path):
    root, source = _phase2_cube(tmp_path)
    assert len(source["analysis_field_sha256"]) == 7

    field = root / phase3.BENCHMARK_CUBE_IDS[0] / "fields" / "velx.npy"
    field.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="analysis input checksum mismatch"):
        phase3._phase2_source_identity(root, phase3.BENCHMARK_CUBE_IDS[0])


def test_phase3_verify_rejects_stale_result_configuration_and_implementation(tmp_path, monkeypatch):
    phase2_root, phase2_source = _phase2_cube(tmp_path)
    source_version = {"implementation_sha256": "implementation-a"}
    cube = _phase3_cube(tmp_path, phase2_source, source_version)
    monkeypatch.setattr(phase3, "_source_version", lambda: source_version)
    configuration_sha256 = json.loads((cube / "summary.json").read_text())["configuration_sha256"]

    passed = phase3._verify_cube_output(
        cube,
        phase2_root,
        phase3.BENCHMARK_CUBE_IDS[0],
        expected_configuration_sha256=configuration_sha256,
    )
    assert passed["status"] == "passed"

    with pytest.raises(RuntimeError, match="requested run"):
        phase3._verify_cube_output(
            cube,
            phase2_root,
            phase3.BENCHMARK_CUBE_IDS[0],
            expected_configuration_sha256="different-configuration",
        )

    monkeypatch.setattr(phase3, "_source_version", lambda: {"implementation_sha256": "implementation-b"})
    with pytest.raises(RuntimeError, match="different implementation"):
        phase3._verify_cube_output(cube, phase2_root, phase3.BENCHMARK_CUBE_IDS[0])

    monkeypatch.setattr(phase3, "_source_version", lambda: source_version)
    (cube / "nested_core.npz").write_bytes(b"changed-result")
    with pytest.raises(RuntimeError, match="result checksum mismatch"):
        phase3._verify_cube_output(cube, phase2_root, phase3.BENCHMARK_CUBE_IDS[0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda marker, summary: marker.update(status="failed"), "completion marker"),
        (lambda marker, summary: marker.update(result_sha256={}), "result inventory"),
        (lambda marker, summary: summary.update(status="failed"), "summary identity"),
        (lambda marker, summary: summary.update(cube_id="WRONG"), "summary identity"),
        (lambda marker, summary: summary.update(phase2_source={}), "summary identity"),
        (lambda marker, summary: summary["configuration"].update(setting="changed"), "configuration checksum"),
    ],
)
def test_phase3_verify_rejects_incomplete_or_failed_publications(tmp_path, monkeypatch, mutation, message):
    phase2_root, phase2_source = _phase2_cube(tmp_path)
    source_version = {"implementation_sha256": "implementation-a"}
    cube = _phase3_cube(tmp_path, phase2_source, source_version)
    monkeypatch.setattr(phase3, "_source_version", lambda: source_version)
    marker_path, summary_path = cube / "COMPLETE.json", cube / "summary.json"
    marker, summary = json.loads(marker_path.read_text()), json.loads(summary_path.read_text())
    mutation(marker, summary)
    _write_json(summary_path, summary)
    marker["summary_sha256"] = file_sha256(summary_path)
    _write_json(marker_path, marker)

    with pytest.raises(RuntimeError, match=message):
        phase3._verify_cube_output(cube, phase2_root, phase3.BENCHMARK_CUBE_IDS[0])


@pytest.mark.parametrize(
    "override",
    [
        {"ell_max": 161},
        {"directions_per_radius": 129},
        {"sample_count": 131073},
        {"robustness_sample_count": 32769},
        {"pair_batch_size": 32769},
    ],
)
def test_run_cube_rejects_parameters_outside_bounded_phase3_scope(tmp_path, override):
    kwargs = {
        "phase2_root": tmp_path / "phase2",
        "output_root": tmp_path / "phase3",
        "cube_id": phase3.BENCHMARK_CUBE_IDS[0],
        "ell_max": 128,
        "directions_per_radius": 24,
        "sample_count": 32768,
        "robustness_sample_count": 32768,
        "pair_batch_size": 8192,
        "seed": 20260530,
        "fit_interval": (8.0, 96.0),
        "fit_min_count": 100,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match="Phase 3"):
        phase3.run_cube(**kwargs)


def test_displacement_payload_enforces_strict_ell_max_and_signed_closure():
    displacements, _, metadata = phase3._displacement_payload(128, 24)
    ell = np.linalg.norm(displacements.astype(float), axis=1)
    offsets = {tuple(row) for row in displacements.tolist()}
    assert ell.max() <= 128.0
    assert metadata["maximum_actual_ell_cells"] <= 128.0
    assert all(tuple(-value for value in offset) in offsets for offset in offsets)


def test_summarize_rejects_mixed_campaign_configuration(tmp_path, monkeypatch):
    source_version = {"implementation_sha256": "implementation-a"}
    monkeypatch.setattr(phase3, "_source_version", lambda: source_version)
    phase2_root = tmp_path / "phase2"
    output_root = tmp_path / "phase3"
    first_phase2_source = None
    for index, cube_id in enumerate(phase3.BENCHMARK_CUBE_IDS):
        _, phase2_source = _phase2_cube(tmp_path, cube_id)
        if first_phase2_source is None:
            first_phase2_source = phase2_source
        _phase3_cube(
            tmp_path,
            phase2_source,
            source_version,
            cube_id=cube_id,
            configuration={"cube_id": cube_id, "setting": "changed" if index == 3 else "shared"},
        )
    assert first_phase2_source is not None
    _synthetic_validation(output_root, source_version)
    _seed_robustness(output_root, first_phase2_source, source_version)

    with pytest.raises(RuntimeError, match="campaign configuration"):
        phase3.summarize(phase2_root, output_root)


def test_smoke_cli_defaults_use_matched_pair_mode_sampling_depth(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_phase3_sampler.py", "smoke", "--output-root", str(tmp_path / "phase3")],
    )
    args = phase3._parse_args()
    assert args.directions_per_radius == 96
    assert args.sample_count == args.robustness_sample_count


@pytest.mark.parametrize("status", [None, "failed"], ids=["missing", "failed"])
def test_summarize_requires_passed_synthetic_validation(tmp_path, monkeypatch, status):
    phase2_root, output_root, source_version = _complete_campaign(tmp_path, monkeypatch)
    if status is not None:
        _synthetic_validation(output_root, source_version, status=status)

    with pytest.raises((FileNotFoundError, RuntimeError), match="synthetic|synthetic_validation"):
        phase3.summarize(phase2_root, output_root)


def test_campaign_completion_marker_binds_synthetic_validation(tmp_path, monkeypatch):
    phase2_root, output_root, source_version = _complete_campaign(tmp_path, monkeypatch)
    synthetic = _synthetic_validation(output_root, source_version)
    phase3.summarize(phase2_root, output_root)

    marker = json.loads((output_root / "PHASE3_SMOKE_COMPLETE.json").read_text())
    assert marker["synthetic_validation_sha256"] == file_sha256(synthetic)


def test_seed_robustness_verifier_rejects_tampered_base_result(tmp_path, monkeypatch):
    phase2_root, output_root, source_version = _complete_campaign(tmp_path, monkeypatch)
    base = output_root / phase3.BENCHMARK_CUBE_IDS[0] / "all_valid_pairs.npz"
    base.write_bytes(b"tampered-base-result")

    with pytest.raises(RuntimeError, match="base result checksum"):
        phase3._verify_seed_robustness_output(
            output_root / "seed_robustness", phase2_root, phase3.BENCHMARK_CUBE_IDS[0]
        )


def test_control_profile_reuse_rejects_changed_phase2_inputs(tmp_path, monkeypatch):
    phase2_root, phase2_source = _phase2_cube(tmp_path)
    output_root = tmp_path / "profile"
    output_root.mkdir()
    source_version = {"implementation_sha256": "implementation-a"}
    monkeypatch.setattr(phase3, "_source_version", lambda: source_version)
    profile_path = output_root / "control_profile.json"
    _write_json(
        profile_path,
        {
            "schema_version": phase3.SCHEMA_VERSION,
            "status": "passed",
            "cube_id": phase3.BENCHMARK_CUBE_IDS[0],
            "phase2_source": phase2_source,
            "source_version": source_version,
        },
    )
    _write_json(
        output_root / "CONTROL_PROFILE_COMPLETE.json",
        {
            "schema_version": phase3.SCHEMA_VERSION,
            "status": "passed",
            "cube_id": phase3.BENCHMARK_CUBE_IDS[0],
            "implementation_sha256": source_version["implementation_sha256"],
            "profile_sha256": file_sha256(profile_path),
        },
    )
    assert phase3.profile_controls(phase2_root, output_root)["reused"]

    field = phase2_root / phase3.BENCHMARK_CUBE_IDS[0] / "fields" / "velx.npy"
    field.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="analysis input checksum mismatch"):
        phase3.profile_controls(phase2_root, output_root)


def test_summarize_rejects_underpowered_subvolume_parallel_fit_bin(tmp_path, monkeypatch):
    phase2_root, output_root, source_version = _complete_campaign(tmp_path, monkeypatch)
    _synthetic_validation(output_root, source_version)
    summary_path = output_root / phase3.BENCHMARK_CUBE_IDS[0] / "summary.json"
    marker_path = output_root / phase3.BENCHMARK_CUBE_IDS[0] / "COMPLETE.json"
    summary = json.loads(summary_path.read_text())
    summary["mode_summaries"]["nested_core"]["coverage_rows"][0]["accepted"] = 99
    _write_json(summary_path, summary)
    marker = json.loads(marker_path.read_text())
    marker["summary_sha256"] = file_sha256(summary_path)
    _write_json(marker_path, marker)

    with pytest.raises(RuntimeError, match="subvolume-mean parallel coverage"):
        phase3.summarize(phase2_root, output_root)


def test_mode_summary_reports_slope_stability_over_multiple_fit_intervals():
    from tests.test_finite_domain import _controlled_reducer_result

    fit_interval = (1.0, 8.0)
    summary = phase3._mode_summary(
        _controlled_reducer_result(),
        fit_interval=fit_interval,
        min_count=1,
    )
    assert "fit_interval_sensitivity_rows" in summary
    intervals = {
        tuple(row["fit_interval"])
        for row in summary["fit_interval_sensitivity_rows"]
    }
    assert intervals == set(phase3.FIT_SENSITIVITY_INTERVALS)
    assert len(intervals) >= 2
    assert "fit_stability_rows" in summary
    assert summary["fit_stability_rows"]
    assert all(
        row["absolute_slope_spread_tolerance"] == phase3.FIT_STABILITY_ABSOLUTE_TOLERANCE
        for row in summary["fit_stability_rows"]
    )
