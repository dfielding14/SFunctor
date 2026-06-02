"""Focused tests for the Phase 5 cross-scale extraction adapter."""
from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
from typing import Any

import numpy as np
import pytest

from scripts.phase5 import build_phase5_campaign_config as builder
from scripts.phase5 import run_phase5_extraction as extraction
from sfunctor.io.cube_extract import CORE_EXTRACTOR_SOURCE_PATHS, CubeExtractionError

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "job_scripts" / "phase5" / "run_phase5_extract_andes.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents).lstrip())
    path.chmod(0o755)


def _phase1_roles() -> dict[str, str]:
    cube_ids = builder.L640_PARENT_PILOT_IDS
    roles = {}
    for index, cube_id in enumerate(cube_ids):
        if index < 4:
            role = "representative:low_dBB"
        elif index < 8:
            role = "representative:near_median_dBB"
        elif index < 12:
            role = "representative:high_dBB"
        elif index < 18:
            role = f"matched:{(index - 12) // 2 + 1}:{'low' if index % 2 == 0 else 'high'}"
        else:
            role = (
                "outlier:small_B_mean",
                "outlier:large_deltaB",
                "outlier:large_dBB",
            )[index - 18]
        roles[cube_id] = role
    return roles


def _blank_catalog(scale: int, count: int) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for name in builder.CATALOG_FIELDS:
        if name in builder.MAGNETIC_FIELDS or name in builder.CATALOG_CONTROL_FIELDS:
            columns[name] = np.ones(count, dtype=np.float64)
        else:
            columns[name] = np.zeros(count, dtype=np.int64)
    columns["subvolume_id"] = np.arange(count, dtype=np.int64)
    columns["L_sub"][:] = scale
    columns["parent_L_sub"][:] = scale * 2
    columns["parent_subvolume_id"][:] = -1
    columns["catalog_validity_flags"][:] = 7
    return columns


def _set_catalog_row(
    columns: dict[str, np.ndarray],
    rank_map: np.ndarray,
    *,
    subvolume_id: int,
    scale: int,
    bounds: tuple[int, int, int, int, int, int],
    dbb: float,
    parent_subvolume_id: int,
    invalid_dbb: bool = False,
) -> None:
    row = subvolume_id
    for name, value in zip(
        ("cell_i0", "cell_i1", "cell_j0", "cell_j1", "cell_k0", "cell_k1"),
        bounds,
    ):
        columns[name][row] = value
    columns["parent_L_sub"][row] = scale * 2
    columns["parent_subvolume_id"][row] = parent_subvolume_id
    columns["required_rank_count"][row] = len(builder.required_rank_ids(bounds, rank_map))
    columns["dBB"][row] = dbb
    columns["B_mean"][row] = 2.0
    columns["deltaB"][row] = 2.0 * dbb
    columns["B_rms"][row] = np.sqrt(4.0 + (2.0 * dbb) ** 2)
    columns["B_mean_sq_over_B2_mean"][row] = 1.0 / (1.0 + dbb**2)
    columns["deltaB_sq_over_B2_mean"][row] = dbb**2 / (1.0 + dbb**2)
    columns["dBB_flags"][row] = int(invalid_dbb)


def _child_bounds(
    parent_bounds: tuple[int, int, int, int, int, int],
    ordinal: int,
) -> tuple[int, int, int, int, int, int]:
    offsets = tuple(itertools.product((0, 1), repeat=3))[ordinal]
    output = []
    for lower, upper, offset in zip(parent_bounds[::2], parent_bounds[1::2], offsets):
        midpoint = (lower + upper) // 2
        output.extend((lower, midpoint) if offset == 0 else (midpoint, upper))
    return tuple(output)  # type: ignore[return-value]


def _touch_trusted_sources(trusted_run: Path) -> None:
    (trusted_run / "analysis").mkdir(parents=True)
    (trusted_run / "cache").mkdir()
    (trusted_run / "catalogs").mkdir()
    (trusted_run / "analysis" / "pilot_sample_metadata.json").write_text("{}\n")
    for scale in builder.CONFIGURED_SCALES:
        (trusted_run / "catalogs" / f"catalog_L{scale}.npz").write_bytes(f"L{scale}".encode())
        (trusted_run / "catalogs" / f"catalog_L{scale}_manifest.json").write_text("{}\n")
        (trusted_run / "catalogs" / f"catalog_L{scale}.complete").write_text("complete\n")


def _synthetic_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    trusted_run = tmp_path / "trusted"
    data_root = tmp_path / "data"
    data_root.mkdir()
    _touch_trusted_sources(trusted_run)
    rank_map = np.arange(16, dtype=np.int64).reshape(2, 2, 4)
    np.save(trusted_run / "cache" / "rank_map.npy", rank_map)
    parent_bounds = (0, 640, 0, 640, 0, 640)
    parent_ranks = builder.required_rank_ids(parent_bounds, rank_map)
    roles = _phase1_roles()
    pilot_path = trusted_run / "analysis" / "pilot_sample.csv"
    with pilot_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pilot_id",
                "role",
                "L_sub",
                "cell_i0",
                "cell_i1",
                "cell_j0",
                "cell_j1",
                "cell_k0",
                "cell_k1",
                "required_rank_count",
                "required_rank_ids_json",
            ),
        )
        writer.writeheader()
        for cube_id in builder.L640_PARENT_PILOT_IDS:
            writer.writerow(
                {
                    "pilot_id": cube_id,
                    "role": roles[cube_id],
                    "L_sub": 640,
                    "cell_i0": 0,
                    "cell_i1": 640,
                    "cell_j0": 0,
                    "cell_j1": 640,
                    "cell_k0": 0,
                    "cell_k1": 640,
                    "required_rank_count": len(parent_ranks),
                    "required_rank_ids_json": json.dumps(parent_ranks),
                }
            )

    catalogs = {
        640: _blank_catalog(640, 4096),
        320: _blank_catalog(
            320,
            max(
                int(cube_id.split("_sub")[1])
                for _, _, cube_id, _ in builder.L320_MATCHED_SMOKE_SPECS
            )
            + 1,
        ),
        160: _blank_catalog(160, 6 * 8),
        80: _blank_catalog(80, 6 * 8),
    }
    parent_descriptors = []
    for index, cube_id in enumerate(builder.L640_PARENT_PILOT_IDS):
        subvolume_id = int(cube_id.split("_sub")[1])
        dbb = 0.5 + index
        _set_catalog_row(
            catalogs[640],
            rank_map,
            subvolume_id=subvolume_id,
            scale=640,
            bounds=parent_bounds,
            dbb=dbb,
            parent_subvolume_id=0,
        )
        parent_descriptors.append((subvolume_id, parent_bounds, dbb, cube_id in {
            "L640_sub00370",
            "L640_sub03942",
            "L640_sub00579",
            "L640_sub00738",
            "L640_sub01591",
            "L640_sub01651",
        }))

    next_descriptors = []
    fixed_l320_by_parent_id = {
        int(parent_cube_id.split("_sub")[1]): (
            int(cube_id.split("_sub")[1]),
            1 if cube_id == "L320_sub17363" else 7,
        )
        for _, _, cube_id, parent_cube_id in builder.L320_MATCHED_SMOKE_SPECS
    }
    for parent_index, (parent_id, bounds, dbb, smoke) in enumerate(parent_descriptors):
        for ordinal in range(8):
            fixed_child = fixed_l320_by_parent_id.get(parent_id)
            child_id = (
                fixed_child[0]
                if fixed_child is not None and ordinal == fixed_child[1]
                else parent_index * 8 + ordinal
            )
            _set_catalog_row(
                catalogs[320],
                rank_map,
                subvolume_id=child_id,
                scale=320,
                bounds=_child_bounds(bounds, ordinal),
                dbb=dbb + (0.001 if ordinal == 0 else 0.01 * ordinal),
                parent_subvolume_id=parent_id,
                invalid_dbb=ordinal == 0,
            )
        if smoke:
            next_descriptors.append((parent_index * 8 + 1, _child_bounds(bounds, 1), dbb + 0.01, True))

    for scale in (160, 80):
        following = []
        for parent_index, (parent_id, bounds, dbb, smoke) in enumerate(next_descriptors):
            for ordinal in range(8):
                child_id = parent_index * 8 + ordinal
                _set_catalog_row(
                    catalogs[scale],
                    rank_map,
                    subvolume_id=child_id,
                    scale=scale,
                    bounds=_child_bounds(bounds, ordinal),
                    dbb=dbb + (0.001 if ordinal == 0 else 0.01 * ordinal),
                    parent_subvolume_id=parent_id,
                    invalid_dbb=ordinal == 0,
                )
            following.append((parent_index * 8 + 1, _child_bounds(bounds, 1), dbb + 0.01, smoke))
        next_descriptors = following

    snapshot = {
        "data_root": str(data_root.resolve()),
        "full_resolution_basename": "full.bin",
        "target_time": 6.0,
        "target_cycle": 24,
        "full_snapshot_identity": {"cycle": 24},
    }
    trusted_artifacts = {
        "trusted_run": str(trusted_run.resolve()),
        "artifact_graph_sha256": "trusted-graph",
    }
    monkeypatch.setattr(builder, "verify_trusted_run", lambda _: trusted_artifacts)
    monkeypatch.setattr(builder, "load_snapshot_identity", lambda _: snapshot)
    monkeypatch.setattr(builder, "_load_catalog_columns", lambda _, scale: catalogs[scale])
    config = builder.build_campaign_config(trusted_run)
    config_path = tmp_path / "phase5_campaign.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return trusted_run, data_root, config, catalogs


def test_builder_freezes_nearest_valid_cross_scale_lineages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, config, _ = _synthetic_campaign(tmp_path, monkeypatch)

    assert config["selection_counts_by_scale"] == {"640": 21, "320": 21, "160": 6, "80": 6}
    assert config["selection_policy"]["prohibited_extraction_scales"] == [1280]
    assert config["q_names"] == ["B", "u"]
    assert config["density_conventions"] == ["not applicable", "not applicable"]
    assert config["sgs_channels_authorized"] is False
    assert config["lsub1280_authorized"] is False
    assert "1280" not in config["selections_by_scale"]
    assert config["smoke_anchor_L640_cube_ids"] == [
        "L640_sub00370",
        "L640_sub03942",
        "L640_sub00579",
        "L640_sub00738",
        "L640_sub01591",
        "L640_sub01651",
    ]

    l320 = config["selections_by_scale"]["320"]
    assert all(row["selection_method"] == "nearest_dBB_magnetic_valid_catalog_child" for row in l320)
    assert all(row["nearest_dBB_selection"]["absolute_dBB_difference"] == pytest.approx(0.01) for row in l320)
    assert all(row["catalog_validity_flags"] == 7 for row in l320)
    assert all(row["required_rank_ids"] for row in l320)
    assert all(row["magnetic_selection_valid"] is True for row in l320)
    assert len(config["selections_by_scale"]["160"]) == 6
    assert len(config["selections_by_scale"]["80"]) == 6
    expected_l320_ids = [index * 8 + 1 for index in range(21)]
    expected_l320_ids[builder.L640_PARENT_PILOT_IDS.index("L640_sub02297")] = 17363
    assert [row["subvolume_id"] for row in l320] == expected_l320_ids

    matched = config["matched_smoke_L320_selections"]
    assert [row["cube_id"] for row in matched] == [
        cube_id for _, _, cube_id, _ in builder.L320_MATCHED_SMOKE_SPECS
    ]
    assert all(row["magnetic_selection_valid"] is True for row in matched)
    assert all(row["required_rank_ids"] for row in matched)
    assert all(row["catalog_controls"] for row in matched)
    assert [
        (
            row["observational_control_set"]["matched_pair_id"],
            row["observational_control_set"]["matched_role"],
            row["observational_control_set"]["stated_L640_parent_cube_id"],
        )
        for row in matched
    ] == [
        (pair_id, matched_role, parent_cube_id)
        for pair_id, matched_role, _, parent_cube_id in builder.L320_MATCHED_SMOKE_SPECS
    ]
    assert [
        row["catalog_parent_link"]["subvolume_id"]
        for row in matched
    ] == [
        int(parent_cube_id.split("_sub")[1])
        for _, _, _, parent_cube_id in builder.L320_MATCHED_SMOKE_SPECS
    ]
    assert all(
        row["observational_control_set"]["interpretation"]
        == "separately labeled observational control set"
        for row in matched
    )
    assert all("phase5_nearest_dBB_child" not in row["roles"] for row in matched)
    assert [row["cube_id"] for row in matched if row["overlaps_nearest_dBB_lineage"]] == [
        "L320_sub17363"
    ]
    assert config["matched_smoke_nearest_dBB_lineage_overlap_cube_ids"] == [
        "L320_sub17363"
    ]


def test_phase4_prerequisite_hashes_and_exact_p2_gate_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prerequisite_dir = tmp_path / "phase4_completion_supplement_v2"
    prerequisite_dir.mkdir()
    for filename in builder.PHASE4_PREREQUISITE_EXPECTED_SHA256:
        shutil.copyfile(builder.PHASE4_PREREQUISITE_DIR / filename, prerequisite_dir / filename)
    monkeypatch.setattr(builder, "PHASE4_PREREQUISITE_DIR", prerequisite_dir)

    proof = builder._phase4_prerequisite_proof()
    assert proof["required_result"] == "42/42 exact p=2 reproduction groups passed"

    summary_path = prerequisite_dir / "phase4_completion_supplement_summary.json"
    summary_path.write_text(summary_path.read_text() + "\n")
    with pytest.raises(CubeExtractionError, match="artifact changed"):
        builder._phase4_prerequisite_proof()

    shutil.copyfile(
        REPO_ROOT
        / "figures"
        / "phase4_completion_supplement_v2"
        / "phase4_completion_supplement_summary.json",
        summary_path,
    )
    summary = json.loads(summary_path.read_text())
    del summary["strict_verification"]["batch_a_to_all21_batch_b_p2_reproduction"]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    expected_sha256 = dict(builder.PHASE4_PREREQUISITE_EXPECTED_SHA256)
    expected_sha256[summary_path.name] = builder.file_sha256(summary_path)
    monkeypatch.setattr(builder, "PHASE4_PREREQUISITE_EXPECTED_SHA256", expected_sha256)
    with pytest.raises(CubeExtractionError, match="gate is absent or changed"):
        builder._phase4_prerequisite_proof()


def test_plan_is_single_scope_hash_bound_and_refuses_nonempty_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_run, data_root, config, _ = _synthetic_campaign(tmp_path, monkeypatch)
    config_path = tmp_path / "phase5_campaign.json"
    monkeypatch.setattr(
        extraction.campaign,
        "verify_campaign_config_file",
        lambda _, path: config if path == config_path.resolve() else None,
    )
    monkeypatch.setattr(extraction, "load_snapshot_identity", lambda _: config["trusted_snapshot_identity"])

    output_root = tmp_path / "plan"
    payload = extraction.plan(
        trusted_run,
        data_root,
        output_root,
        config_path,
        scale=320,
        subset="smoke",
        basename=None,
    )
    assert payload["scope"]["L_sub"] == 320
    assert payload["scope"]["subset"] == "smoke"
    assert payload["scope"]["cube_count"] == 6
    assert extraction.verify_plan(
        trusted_run,
        data_root,
        output_root,
        config_path,
        scale=320,
        subset="smoke",
        basename=None,
    ) == payload
    with pytest.raises(CubeExtractionError, match="non-empty output root"):
        extraction.plan(
            trusted_run,
            data_root,
            output_root,
            config_path,
            scale=320,
            subset="smoke",
            basename=None,
        )
    with pytest.raises(CubeExtractionError, match="does not configure"):
        extraction._selected_config_rows(config, scale=1280, subset="all")

    matched_output_root = tmp_path / "matched_plan"
    matched_payload = extraction.plan(
        trusted_run,
        data_root,
        matched_output_root,
        config_path,
        scale=320,
        subset="matched_smoke",
        basename=None,
    )
    assert matched_payload["scope"]["subset"] == "matched_smoke"
    assert matched_payload["scope"]["cube_ids"] == [
        cube_id for _, _, cube_id, _ in builder.L320_MATCHED_SMOKE_SPECS
    ]
    with pytest.raises(CubeExtractionError, match="configured only for L_sub=320"):
        extraction._selected_config_rows(config, scale=160, subset="matched_smoke")


def test_materialization_and_restart_records_are_immutable_and_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_run, data_root, config, _ = _synthetic_campaign(tmp_path, monkeypatch)
    config_path = tmp_path / "phase5_campaign.json"
    monkeypatch.setattr(extraction.campaign, "verify_campaign_config_file", lambda *_: config)
    monkeypatch.setattr(extraction, "load_snapshot_identity", lambda _: config["trusted_snapshot_identity"])
    output_root = tmp_path / "plan"
    plan = extraction.plan(
        trusted_run,
        data_root,
        output_root,
        config_path,
        scale=320,
        subset="smoke",
        basename=None,
    )
    row = plan["selections"][0]
    cube_id = row["cube_id"]
    cube_root = output_root / cube_id
    cube_root.mkdir()
    plan_hashes = plan["source_version"]["implementation_source_hashes"]
    core_hashes = {path: plan_hashes[path] for path in CORE_EXTRACTOR_SOURCE_PATHS}
    manifest = {
        "cube_id": cube_id,
        "Lsub": row["L_sub"],
        "role": extraction._extractor_role(row),
        "bounds_ijk_half_open": row["bounds_ijk_half_open"],
        "shape_kji": row["shape_kji"],
        "source_root": str(data_root.resolve()),
        "source_basename": "full.bin",
        "trusted_phase1_artifacts": config["trusted_phase1_artifacts"],
        "preflight": {"unique_referenced_rank_ids": sorted(row["required_rank_ids"])},
        "code_version": {
            "commit": "test",
            "dirty": False,
            "implementation_source_hashes": core_hashes,
            "implementation_sha256": extraction._mapping_sha256(core_hashes),
        },
    }
    manifest_path = cube_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (cube_root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "cube_id": cube_id,
                "manifest_sha256": builder.file_sha256(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    first_materialization = extraction._publish_materialization_record(output_root, cube_id)
    assert extraction._publish_materialization_record(output_root, cube_id) == first_materialization
    first_restart = extraction._publish_restart_record(output_root, cube_id)
    assert extraction._publish_restart_record(output_root, cube_id) == first_restart

    restart_path = extraction._restart_record_path(output_root, cube_id)
    restart = json.loads(restart_path.read_text())
    restart["verification_contract"]["verify_hashes"] = False
    restart_path.write_text(json.dumps(restart, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CubeExtractionError, match="invalid or stale Phase 5 restart record"):
        extraction._restart_record_identity(output_root, cube_id)


@pytest.fixture
def fake_andes(tmp_path: Path) -> tuple[Path, Path]:
    sfunctor_dir = tmp_path / "sfunctor"
    fake_bin = tmp_path / "bin"
    _write_executable(sfunctor_dir / "venv_sfunctor" / "bin" / "activate", "#!/usr/bin/env bash\n")
    _write_executable(
        sfunctor_dir / "venv_sfunctor" / "bin" / "python",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\n' "$*" >> "${FAKE_PYTHON_LOG}"
        """,
    )
    _write_executable(fake_bin / "module", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "srun",
        """
        #!/usr/bin/env bash
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
        """
        #!/usr/bin/env bash
        printf '9001|1|1|32|1024K|node|0|512K|2048K|COMPLETED|0:0\n'
        """,
    )
    _write_executable(
        fake_bin / "squeue",
        """
        #!/usr/bin/env bash
        if [[ -n "${FAKE_SQUEUE_ROWS:-}" ]]; then
          printf '%s\n' "${FAKE_SQUEUE_ROWS}"
        fi
        """,
    )
    return sfunctor_dir, fake_bin


def _run_wrapper(
    fake_andes: tuple[Path, Path],
    output_root: Path,
    run_dir: Path,
    *,
    action: str = "plan",
    **environment: str,
) -> subprocess.CompletedProcess[str]:
    sfunctor_dir, fake_bin = fake_andes
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SFUNCTOR_DIR": str(sfunctor_dir),
            "CAMPAIGN_CONFIG": str(output_root.parent / "campaign.json"),
            "OUTPUT_ROOT": str(output_root),
            "RUN_DIR": str(run_dir),
            "ACTION": action,
            "SCALE": "320",
            "SUBSET": "smoke",
            "SLURM_JOB_ID": "9001",
            "FAKE_PYTHON_LOG": str(run_dir / "python.log"),
        }
    )
    env.update(environment)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=sfunctor_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_wrapper_rejects_reused_run_dir_before_writes(
    tmp_path: Path,
    fake_andes: tuple[Path, Path],
) -> None:
    sfunctor_dir, _ = fake_andes
    run_dir = tmp_path / "allocation"
    run_dir.mkdir()
    output_root = tmp_path / "new_parent" / "output"

    result = _run_wrapper(fake_andes, output_root, run_dir)

    assert result.returncode == 3
    assert "allocation directory already exists" in result.stderr
    assert not (sfunctor_dir / "logs").exists()
    assert not output_root.parent.exists()


def test_wrapper_passes_scope_archives_resources_and_releases_lock(
    tmp_path: Path,
    fake_andes: tuple[Path, Path],
) -> None:
    output_root = tmp_path / "output"
    run_dir = tmp_path / "allocation"

    result = _run_wrapper(
        fake_andes,
        output_root,
        run_dir,
        action="extract",
        SUBSET="matched_smoke",
    )

    assert result.returncode == 0, result.stderr
    arguments = (run_dir / "python.log").read_text()
    assert "run_phase5_extraction.py extract" in arguments
    assert "--scale 320 --subset matched_smoke" in arguments
    assert (run_dir / "resources" / "sacct_9001.psv").is_file()
    assert not (tmp_path / ".output.phase5_extract_action_lock").exists()


def test_wrapper_recovers_verified_inactive_action_lock(
    tmp_path: Path,
    fake_andes: tuple[Path, Path],
) -> None:
    output_root = tmp_path / "output"
    lock_dir = tmp_path / ".output.phase5_extract_action_lock"
    lock_dir.mkdir()
    (lock_dir / "owner.txt").write_text("job_id=8123\naction=extract\ntoken=prior\n")

    result = _run_wrapper(
        fake_andes,
        output_root,
        tmp_path / "allocation",
        FAKE_SQUEUE_ROWS="7000",
    )

    assert result.returncode == 0, result.stderr
    assert "Recovered stale Phase 5 extraction action lock" in result.stderr
    assert not lock_dir.exists()


def test_wrapper_preserves_active_action_lock(
    tmp_path: Path,
    fake_andes: tuple[Path, Path],
) -> None:
    output_root = tmp_path / "output"
    lock_dir = tmp_path / ".output.phase5_extract_action_lock"
    lock_dir.mkdir()
    (lock_dir / "owner.txt").write_text("job_id=8123\naction=extract\ntoken=prior\n")
    run_dir = tmp_path / "allocation"

    result = _run_wrapper(
        fake_andes,
        output_root,
        run_dir,
        FAKE_SQUEUE_ROWS="8123",
    )

    assert result.returncode == 3
    assert "another Phase 5 extraction action holds" in result.stderr
    assert lock_dir.is_dir()
    assert not run_dir.exists()


def test_wrapper_is_cpu_only_logs_locally_and_has_no_email_directives() -> None:
    contents = WRAPPER.read_text()
    assert "#SBATCH -p batch" in contents
    assert "#SBATCH --cpus-per-task=32" in contents
    assert "#SBATCH -o logs/" in contents
    assert "#SBATCH -e logs/" in contents
    assert "--mail" not in contents
    assert "gpu" not in contents.lower()
