"""Focused synthetic tests for the Phase 5 cross-scale report generator."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.phase5 import generate_phase5_status_figures as report


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "job_scripts" / "phase5" / "run_phase5_report_andes.sh"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents.lstrip())
    path.chmod(0o755)


def _ledger(path: Path) -> None:
    path.write_text(
        """# Compute Budget Summary

Updated: `2026-06-02T12:00:00Z`

| Metric | Node-hours |
| --- | ---: |
| Workflow budget | 5000 |
| Consumed allocated runtime | 25 |
| Remaining budget | 4975 |
| Pending maximum additional exposure | 0 |
| Projected remaining after pending maximum | 4975 |
"""
    )


def _decision_record(path: Path, campaign_config: Path, labels: tuple[str, ...]) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "phase": "phase5_execution_decision",
            "status": "retained_closeout_acquisition_scope",
            "campaign_config_sha256": report.file_sha256(campaign_config),
            "retained_report_release_labels": list(labels),
        },
    )


def _result(
    *,
    stencil_width: int,
    support_mode: str,
    value: float,
) -> SimpleNamespace:
    shape = (2, 1, 1, 1, 1, 3)
    return SimpleNamespace(
        stencil_width=stencil_width,
        pair_mode=support_mode,
        support_displacement_count=1,
        q_names=("B", "u"),
        geometry_names=("pair_local",),
        measurement_names=("perpendicular",),
        direction_names=("lambda",),
        p_values=(2.0,),
        ell_bin_edges=np.asarray((1.0, 4.0, 16.0, 64.0)),
        eligible_pairs=np.asarray((90, 70, 50), dtype=np.int64),
        cube_candidate_pairs=np.asarray((100, 100, 100), dtype=np.int64),
        sampled_pairs=np.asarray((64, 64, 64), dtype=np.int64),
        moments=np.full(shape, value, dtype=float),
        counts=np.full(shape, 64, dtype=np.int64),
        elapsed_seconds=12.5,
    )


def _uncertainty(result: SimpleNamespace) -> dict[str, np.ndarray]:
    shape = result.moments.shape
    return {
        "metadata_json": np.asarray(
            json.dumps({"bootstrap_n_resamples": 10}, sort_keys=True)
        ),
        "moments": result.moments.copy(),
        "block_bootstrap_interval_low": 0.8 * result.moments,
        "block_bootstrap_interval_high": 1.2 * result.moments,
        "accepted_contributing_blocks": np.full(shape, 16, dtype=np.int64),
        "accepted_effective_blocks": np.full(shape, 12.0),
        "valid_bootstrap_resamples": np.full(shape, 10, dtype=np.int64),
        "local_log_slope": np.full(shape, 0.75),
        "local_log_slope_support_mask": np.ones(shape, dtype=bool),
        "local_log_slope_bootstrap_interval_low": np.full(shape, 0.6),
        "local_log_slope_bootstrap_interval_high": np.full(shape, 0.9),
    }


def _group(
    *,
    release_label: str,
    cube_id: str,
    stencil_width: int,
    support_mode: str,
    value: float,
) -> report.Group:
    result = _result(
        stencil_width=stencil_width, support_mode=support_mode, value=value
    )
    return report.Group(
        release_label=release_label,
        scale=320,
        cube_id=cube_id,
        stencil_width=stencil_width,
        support_mode=support_mode,
        group_id=f"{cube_id}/stencil_{stencil_width}point/{support_mode}",
        result=result,
        uncertainty=_uncertainty(result),
        reduction_marker={},
    )


def _verified_release(label: str = "L320") -> report.VerifiedRelease:
    cube_id = "L320_sub00001"
    groups = {}
    for width, primary_value, shell_value in ((2, 4.0, 2.0), (3, 6.0, 3.0)):
        for mode, value in (
            (report.PRIMARY_SUPPORT_MODE, primary_value),
            (report.OVERLAY_SUPPORT_MODE, shell_value),
        ):
            group = _group(
                release_label=label,
                cube_id=cube_id,
                stencil_width=width,
                support_mode=mode,
                value=value,
            )
            groups[(cube_id, width, mode)] = group
    return report.VerifiedRelease(
        label=label,
        root=Path(f"/retained/{label}"),
        scale=320,
        selection_set="lineage",
        matrix="baseline",
        campaign={},
        summary={},
        phase2_sources={cube_id: {}},
        groups=groups,
        verification={
            "status": "passed",
            "verified_shards": 4,
            "verified_reductions": 4,
            "reduction_elapsed_seconds": 2.0,
            "shard_staging_logical_bytes_before_markers": 100,
            "reduction_staging_logical_bytes_before_markers": 200,
            "settled_release_logical_bytes_before_summary_marker": 300,
            "settled_release_allocated_bytes_before_summary_marker": 400,
        },
    )


def _campaign_config(path: Path, phase1_root: Path) -> None:
    artifact = phase1_root / "catalog.json"
    _write_json(artifact, {"status": "passed"})
    _write_json(
        path,
        {
            "schema_version": 1,
            "phase": "phase5_cross_scale_report",
            "campaign_label": "synthetic",
            "phase1_artifacts": [
                {
                    "relative_path": "catalog.json",
                    "sha256": report.file_sha256(artifact),
                }
            ],
            "releases": {"L320": {"L_sub": 320}},
            "selections": [
                {
                    "release": "L320",
                    "cube_id": "L320_sub00001",
                    "L_sub": 320,
                    "role": "representative:high_dBB_outlier",
                    "physical_region_id": "region-a",
                    "representative": True,
                    "outlier": True,
                    "environment": {
                        "dBB": 2.0,
                        "B_mean": 3.0,
                        "deltaB": 6.0,
                        "B_rms": 7.0,
                        "B_mean_sq_over_B2_mean": 9.0 / 49.0,
                        "deltaB_sq_over_B2_mean": 36.0 / 49.0,
                    },
                }
            ],
        },
    )


def test_release_specs_require_stable_unique_labels() -> None:
    assert report._release_specs(("L320=/retained/a", "L160=/retained/b")) == {
        "L320": Path("/retained/a"),
        "L160": Path("/retained/b"),
    }
    with pytest.raises(ValueError, match="duplicate"):
        report._release_specs(("L320=/retained/a", "L320=/retained/b"))
    with pytest.raises(ValueError, match="label=path"):
        report._release_specs(("not-a-binding",))


def test_matched_smoke_is_not_merged_with_nearest_descendant_lineage() -> None:
    def row(cube_id: str, root: str) -> dict[str, Any]:
        return {
            "cube_id": cube_id,
            "L_sub": 320,
            "root_L640_cube_id": root,
            "root_phase1_role": "representative:low_dBB",
            "catalog_magnetic_values": {
                "dBB": 1.0,
                "B_mean": 2.0,
                "deltaB": 2.0,
                "B_rms": 3.0,
                "B_mean_sq_over_B2_mean": 4.0 / 9.0,
                "deltaB_sq_over_B2_mean": 4.0 / 9.0,
            },
        }

    config = {
        "selections_by_scale": {"320": [row("lineage-a", "root-a")]},
        "matched_smoke_L320_selections": [
            {
                **row("matched-a", "matched-root-a"),
                "observational_control_set": {
                    "label": "matched_smoke",
                    "matched_pair_id": "pair-1",
                    "matched_role": "low",
                },
            }
        ],
    }
    metadata = {
        "lineage-baseline": {
            "L_sub": 320,
            "selection_set": "lineage",
            "matrix": "baseline",
        },
        "matched-baseline": {
            "L_sub": 320,
            "selection_set": "matched_smoke",
            "matrix": "baseline",
        },
    }
    selections = report._normalize_selections(
        config,
        metadata,
        {
            "lineage-baseline": {"lineage-a": {}},
            "matched-baseline": {"matched-a": {}},
        },
    )
    rows = report._selection_rows(selections)

    assert {selection.selection_set for selection in selections} == {
        "lineage",
        "matched_smoke",
    }
    assert {(row["selection_set"], row["cube_id"]) for row in rows} == {
        ("lineage", "lineage-a"),
        ("matched_smoke", "matched-a"),
    }
    matched = next(selection for selection in selections if selection.selection_set == "matched_smoke")
    assert matched.matched_group == "matched_smoke:pair-1"
    assert matched.role == "observational_control:matched_smoke:pair-1:low"
    availability = report._matrix_availability_rows(
        (
            SimpleNamespace(selection_set="lineage", scale=320, matrix="baseline"),
            SimpleNamespace(selection_set="lineage", scale=160, matrix="3point"),
            SimpleNamespace(selection_set="lineage", scale=80, matrix="orders"),
        )
    )
    l80 = next(row for row in availability if row["L_sub_cells"] == 80)
    assert l80["allowed_matrices"] == ["baseline", "orders"]
    assert l80["unavailable_by_scale_policy"] == ["3point", "5point"]


def test_phase2_source_binding_rehashes_analysis_arrays(tmp_path: Path) -> None:
    phase2_root = tmp_path / "phase2"
    cube_root = phase2_root / "cube-a"
    array_path = cube_root / "fields" / "B.npy"
    array_path.parent.mkdir(parents=True)
    np.save(array_path, np.asarray((1.0, 2.0)))
    manifest_path = cube_root / "manifest.json"
    completion_path = cube_root / "COMPLETE.json"
    _write_json(manifest_path, {"status": "complete"})
    _write_json(completion_path, {"cube_id": "cube-a"})
    source = {
        "cube_id": "cube-a",
        "phase2_root": str(phase2_root),
        "completion_relative_path": "cube-a/COMPLETE.json",
        "completion_sha256": report.file_sha256(completion_path),
        "manifest_relative_path": "cube-a/manifest.json",
        "manifest_sha256": report.file_sha256(manifest_path),
        "analysis_field_sha256": {"fields/B.npy": report.file_sha256(array_path)},
    }

    report._verify_phase2_source(phase2_root, "cube-a", source, report.InputHashes())

    array_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        report._verify_phase2_source(
            phase2_root, "cube-a", source, report.InputHashes()
        )


def _retained_release(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    release_root = tmp_path / "release"
    phase2_root = tmp_path / "phase2"
    cube_id = "cube-a"
    group_id = f"{cube_id}/stencil_2point/{report.PRIMARY_SUPPORT_MODE}"
    shard_id = f"{group_id}/shard_0000"
    source_version = {"implementation_sha256": "a" * 64}

    cube_root = phase2_root / cube_id
    field_path = cube_root / "fields" / "B.npy"
    field_path.parent.mkdir(parents=True)
    np.save(field_path, np.asarray((1.0, 2.0)))
    manifest_path = cube_root / "manifest.json"
    completion_path = cube_root / "COMPLETE.json"
    _write_json(manifest_path, {"status": "complete"})
    _write_json(completion_path, {"cube_id": cube_id})
    phase2_source = {
        "cube_id": cube_id,
        "phase2_root": str(phase2_root),
        "completion_relative_path": f"{cube_id}/COMPLETE.json",
        "completion_sha256": report.file_sha256(completion_path),
        "manifest_relative_path": f"{cube_id}/manifest.json",
        "manifest_sha256": report.file_sha256(manifest_path),
        "analysis_field_sha256": {"fields/B.npy": report.file_sha256(field_path)},
    }

    displacement_json = release_root / "manifests" / "displacements" / "stencil_2point.json"
    displacement_npz = release_root / "manifests" / "displacements" / "stencil_2point.npz"
    _write_json(displacement_json, {"status": "passed"})
    displacement_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(displacement_npz, offsets=np.asarray(((1, 0, 0),)))
    shard_row = {
        "group_id": group_id,
        "shard_id": shard_id,
        "cube_id": cube_id,
        "stencil_width": 2,
        "support_mode": report.PRIMARY_SUPPORT_MODE,
        "shard_index": 0,
        "offset_start": 0,
        "offset_stop": 1,
    }
    campaign = {
        "schema_version": 1,
        "status": "planned",
        "phase2_root": str(phase2_root),
        "source_version": source_version,
        "configuration": {"synthetic": True},
        "phase2_sources": {cube_id: phase2_source},
        "displacement_manifests": {
            "2": {
                "json_relative_path": str(displacement_json.relative_to(release_root)),
                "json_sha256": report.file_sha256(displacement_json),
                "npz_relative_path": str(displacement_npz.relative_to(release_root)),
                "npz_sha256": report.file_sha256(displacement_npz),
                "offset_count": 1,
            }
        },
        "shard_count": 1,
    }
    campaign["configuration_sha256"] = report._mapping_sha256(
        campaign["configuration"]
    )
    campaign_path = release_root / "manifests" / "campaign.json"
    shards_path = release_root / "manifests" / "shards.json"
    _write_json(campaign_path, campaign)
    _write_json(shards_path, {"shards": [shard_row]})
    _write_json(
        release_root / "PLAN_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "passed",
            "campaign_sha256": report.file_sha256(campaign_path),
            "shards_sha256": report.file_sha256(shards_path),
            "implementation_sha256": source_version["implementation_sha256"],
        },
    )

    partial_path = release_root / "shards" / shard_id / "partial.npz"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(b"synthetic partial")
    shard_marker_path = partial_path.parent / "COMPLETE.json"
    _write_json(
        shard_marker_path,
        {
            "schema_version": 1,
            "status": "passed",
            "shard": shard_row,
            "implementation_sha256": source_version["implementation_sha256"],
            "phase2_source": phase2_source,
            "partial_sha256": report.file_sha256(partial_path),
            "staging_logical_bytes_before_marker": 10,
        },
    )

    result = _result(
        stencil_width=2, support_mode=report.PRIMARY_SUPPORT_MODE, value=4.0
    )
    reduction_root = release_root / "reductions" / group_id
    result_path = reduction_root / "result.npz"
    uncertainty_path = reduction_root / "uncertainty.npz"
    reduction_manifest_path = reduction_root / "reduction_manifest.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"synthetic result")
    np.savez(uncertainty_path, **_uncertainty(result))
    _write_json(
        reduction_manifest_path,
        {
            "group_id": group_id,
            "ordered_shard_ids": [shard_id],
            "ordered_shard_marker_sha256": {
                shard_id: report.file_sha256(shard_marker_path)
            },
            "implementation_sha256": source_version["implementation_sha256"],
        },
    )
    _write_json(
        reduction_root / "COMPLETE.json",
        {
            "schema_version": 1,
            "status": "passed",
            "group_id": group_id,
            "result_sha256": report.file_sha256(result_path),
            "uncertainty_sha256": report.file_sha256(uncertainty_path),
            "reduction_manifest_sha256": report.file_sha256(reduction_manifest_path),
            "reduction_elapsed_seconds": 1.0,
            "staging_logical_bytes_before_marker": 20,
        },
    )
    summary_path = release_root / "phase3a_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "operational_status": "release_aggregation_complete",
            "source_version": source_version,
            "groups": [{"group_id": group_id}],
        },
    )
    _write_json(
        release_root / "PHASE3A_RELEASE_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "release_aggregation_complete",
            "summary_sha256": report.file_sha256(summary_path),
            "implementation_sha256": source_version["implementation_sha256"],
        },
    )
    return release_root, result


def test_release_verifier_replays_marker_hash_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_root, result = _retained_release(tmp_path)
    monkeypatch.setattr(report, "load_finite_domain_partial_npz", lambda path: result)

    release = report._verify_release(
        "L320", release_root, {"L_sub": 320}, report.InputHashes()
    )

    assert release.verification["verified_phase2_sources"] == 1
    assert release.verification["verified_shards"] == 1
    assert release.verification["verified_reductions"] == 1

    (
        release_root
        / "reductions"
        / "cube-a"
        / "stencil_2point"
        / report.PRIMARY_SUPPORT_MODE
        / "result.npz"
    ).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        report._verify_release(
            "L320", release_root, {"L_sub": 320}, report.InputHashes()
        )


def test_report_package_publishes_quantitative_tables_figures_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase1_root = tmp_path / "phase1"
    phase1_root.mkdir()
    config_path = tmp_path / "campaign.json"
    ledger_path = tmp_path / "ledger.md"
    decision_path = tmp_path / "decision.json"
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    _campaign_config(config_path, phase1_root)
    _ledger(ledger_path)
    _decision_record(decision_path, config_path, ("L320",))
    synthetic = _verified_release()
    monkeypatch.setattr(report, "_verify_release", lambda *args, **kwargs: synthetic)

    report.generate_report(
        campaign_config=config_path,
        phase1_root=phase1_root,
        release_paths={"L320": tmp_path / "retained"},
        decision_record_path=decision_path,
        ledger_summary_path=ledger_path,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / report.SUMMARY_FILENAME).read_text())
    manifest = json.loads((output_dir / report.HASH_MANIFEST_FILENAME).read_text())
    policy = json.loads((output_dir / report.POLICY_JSON_FILENAME).read_text())
    stencil = json.loads((output_dir / report.STENCIL_JSON_FILENAME).read_text())
    assert summary["interpretation_policy"]["fitted_exponents_published"] is False
    assert len(list(output_dir.glob("*.png"))) == len(report.FIGURE_FILENAMES)
    assert report.SUMMARY_FILENAME in manifest["generated_artifact_sha256"]
    assert any(row["policy_factor"] == pytest.approx(2.0) for row in policy["rows"])
    assert any(
        row["3point_over_2point_ratio"] == pytest.approx(1.5)
        for row in stencil["rows"]
    )

    with pytest.raises(RuntimeError, match="non-empty output"):
        report.generate_report(
            campaign_config=config_path,
            phase1_root=phase1_root,
            release_paths={"L320": tmp_path / "retained"},
            decision_record_path=decision_path,
            ledger_summary_path=ledger_path,
            output_dir=output_dir,
        )


@pytest.fixture
def fake_andes(tmp_path: Path) -> tuple[Path, Path]:
    sfunctor_dir = tmp_path / "sfunctor"
    fake_bin = tmp_path / "bin"
    _write_executable(
        sfunctor_dir / "venv_sfunctor" / "bin" / "activate",
        "#!/usr/bin/env bash\n",
    )
    _write_executable(
        sfunctor_dir / "venv_sfunctor" / "bin" / "python",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_PYTHON_LOG}"
""",
    )
    _write_executable(fake_bin / "module", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "srun",
        """#!/usr/bin/env bash
set -euo pipefail
while (( "$#" )); do
  case "$1" in
    -N|-n|--ntasks|--ntasks-per-node|--cpus-per-task)
      shift 2
      ;;
    --*=*)
      shift
      ;;
    *)
      exec "$@"
      ;;
  esac
done
""",
    )
    _write_executable(
        fake_bin / "sacct",
        """#!/usr/bin/env bash
printf '9001|1|1|1|1024K|node|0|512K|2048K|COMPLETED|0:0\n'
""",
    )
    return sfunctor_dir, fake_bin


def _run_wrapper(
    fake_andes: tuple[Path, Path],
    tmp_path: Path,
    run_dir: Path,
    **updates: str,
) -> subprocess.CompletedProcess[str]:
    sfunctor_dir, fake_bin = fake_andes
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SFUNCTOR_DIR": str(sfunctor_dir),
            "CAMPAIGN_CONFIG": str(tmp_path / "campaign.json"),
            "PHASE5_DECISION_RECORD": str(tmp_path / "decision.json"),
            "PHASE1_ROOT": str(tmp_path / "phase1"),
            "PHASE5_RELEASES": "L320=/retained/L320 L160=/retained/L160",
            "LEDGER_SUMMARY": str(tmp_path / "ledger.md"),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "RUN_DIR": str(run_dir),
            "SLURM_JOB_ID": "9001",
            "FAKE_PYTHON_LOG": str(tmp_path / "python.log"),
        }
    )
    env.update(updates)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_wrapper_rejects_existing_run_dir_before_writes(
    tmp_path: Path, fake_andes: tuple[Path, Path]
) -> None:
    sfunctor_dir, _ = fake_andes
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = _run_wrapper(fake_andes, tmp_path, run_dir)

    assert result.returncode == 3
    assert "allocation directory already exists" in result.stderr
    assert not (sfunctor_dir / "logs").exists()
    assert not (tmp_path / "output").exists()


def test_wrapper_passes_explicit_releases_optional_baseline_and_archives_sacct(
    tmp_path: Path, fake_andes: tuple[Path, Path]
) -> None:
    run_dir = tmp_path / "run"

    result = _run_wrapper(
        fake_andes,
        tmp_path,
        run_dir,
        PHASE4_BATCH_A_ROOT="/retained/phase4",
    )

    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "python.log").read_text()
    assert "--campaign-config" in arguments
    assert "--decision-record" in arguments
    assert "--release L320=/retained/L320" in arguments
    assert "--release L160=/retained/L160" in arguments
    assert "--phase4-batch-a-root /retained/phase4" in arguments
    assert (run_dir / "resources" / "sacct_9001.psv").is_file()


def test_wrapper_has_logs_no_email_and_cpu_only_shape() -> None:
    text = WRAPPER.read_text()

    assert "#SBATCH -o logs/" in text
    assert "#SBATCH -e logs/" in text
    assert "--mail" not in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert 'if ! mkdir "${RUN_DIR}"; then' in text
    assert text.index('if ! mkdir "${RUN_DIR}"; then') < text.index('mkdir "${RUN_DIR}/logs"')
