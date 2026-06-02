"""Focused tests for the generic Phase 5 sampler adapter and Andes wrapper."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.phase5 import run_phase5_sampler as sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "job_scripts" / "phase5" / "run_phase5_sampler_andes.sh"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _campaign_config(tmp_path: Path, *, scale: int = 320) -> Path:
    path = tmp_path / "phase5_campaign.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "q_names": ["B", "u"],
            "density_conventions": ["not applicable", "not applicable"],
            "sgs_channels_authorized": False,
            "selections_by_scale": {
                str(scale): [
                    {"L_sub": scale, "cube_id": f"L{scale}_cube_a", "smoke_anchor": True},
                    {"L_sub": scale, "cube_id": f"L{scale}_cube_b", "smoke_anchor": True},
                ]
            },
        },
    )
    return path


@pytest.mark.parametrize(
    ("scale", "matrix", "stencil_width", "ell_max", "bin_count", "p_values"),
    [
        (320, "baseline", 2, 160, 64, (2.0,)),
        (320, "orders", 2, 160, 64, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        (320, "5point", 5, 40, 40, (2.0,)),
        (160, "3point", 3, 40, 40, (2.0,)),
        (80, "baseline", 2, 40, 32, (2.0,)),
    ],
)
def test_configured_runner_freezes_scale_aware_matrix(
    tmp_path: Path,
    scale: int,
    matrix: str,
    stencil_width: int,
    ell_max: int,
    bin_count: int,
    p_values: tuple[float, ...],
) -> None:
    campaign_config = _campaign_config(tmp_path, scale=scale)

    with sampler._configured_runner(
        campaign_config,
        scale=scale,
        selection_set="smoke",
        matrix=matrix,
    ) as selection:
        configuration = sampler.inherited._campaign_configuration()

        assert selection.cube_ids == (f"L{scale}_cube_a", f"L{scale}_cube_b")
        assert configuration["q_names"] == ("B", "u")
        assert configuration["p_values"] == p_values
        assert configuration["density_conventions"] == (
            "not applicable",
            "not applicable",
        )
        assert configuration["support_modes"] == (
            "all_valid_origins",
            "shell_local",
        )
        assert configuration["block_shape_kji"] == (scale // 8,) * 3
        assert configuration["stencils"] == {
            stencil_width: {
                "label": f"{stencil_width}-point",
                "ell_max": ell_max,
                "bin_count": bin_count,
                "directions_per_bin": 24,
            }
        }


@pytest.mark.parametrize(
    ("scale", "matrix", "expected_bin_count"),
    [
        (320, "baseline", 64),
        (320, "orders", 64),
        (320, "3point", 64),
        (320, "5point", 40),
        (160, "baseline", 48),
        (160, "orders", 48),
        (160, "3point", 40),
        (80, "baseline", 32),
        (80, "orders", 32),
    ],
)
def test_stencil_specs_cap_bins_to_ell_max(
    scale: int,
    matrix: str,
    expected_bin_count: int,
) -> None:
    spec = next(iter(sampler._stencil_specs(scale, matrix).values()))

    assert spec["bin_count"] == expected_bin_count
    assert spec["bin_count"] <= spec["ell_max"]
    sampler.inherited.dense_displacement_manifest(
        stencil_width=sampler.PHASE5_MATRIX_STENCIL_WIDTHS[matrix],
        ell_max=spec["ell_max"],
        bin_count=spec["bin_count"],
        directions_per_bin=spec["directions_per_bin"],
    )


@pytest.mark.parametrize(
    ("scale", "matrix"),
    [
        (160, "5point"),
        (80, "3point"),
        (80, "5point"),
    ],
)
def test_rejects_matrix_when_stencil_ell_max_is_below_generator_guard(
    tmp_path: Path,
    scale: int,
    matrix: str,
) -> None:
    with pytest.raises(ValueError, match="not supported"):
        sampler._resolve_selection(
            _campaign_config(tmp_path, scale=scale),
            scale=scale,
            selection_set="smoke",
            matrix=matrix,
        )
    with pytest.raises(ValueError, match="not supported"):
        sampler._stencil_specs(scale, matrix)


def test_matched_smoke_requires_explicit_per_scale_selection_set(tmp_path: Path) -> None:
    campaign_config = _campaign_config(tmp_path, scale=320)
    payload = json.loads(campaign_config.read_text())
    payload["matched_smoke_L320_selections"] = [
        {"L_sub": 320, "cube_id": "L320_cube_b"},
    ]
    _write_json(campaign_config, payload)

    selection = sampler._resolve_selection(
        campaign_config,
        scale=320,
        selection_set="matched_smoke",
        matrix="baseline",
    )

    assert selection.cube_ids == ("L320_cube_b",)
    for scale in (160, 80):
        with pytest.raises(ValueError, match="does not define 'matched_smoke'"):
            sampler._resolve_selection(
                _campaign_config(tmp_path, scale=scale),
                scale=scale,
                selection_set="matched_smoke",
                matrix="baseline",
            )
    campaign_config = _campaign_config(tmp_path, scale=160)
    payload = json.loads(campaign_config.read_text())
    payload["matched_smoke_L160_selections"] = [
        {"L_sub": 160, "cube_id": "L160_cube_b"},
    ]
    _write_json(campaign_config, payload)

    selection = sampler._resolve_selection(
        campaign_config,
        scale=160,
        selection_set="matched_smoke",
        matrix="baseline",
    )

    assert selection.cube_ids == ("L160_cube_b",)


def test_configured_runner_restores_every_inherited_override_after_failure(tmp_path: Path) -> None:
    campaign_config = _campaign_config(tmp_path)
    original = {
        "phase3_cube_ids": sampler.phase3.BENCHMARK_CUBE_IDS,
        "cube_ids": sampler.inherited.BENCHMARK_CUBE_IDS,
        "q_names": sampler.inherited.Q_NAMES,
        "p_values": sampler.inherited.P_VALUES,
        "density_conventions": sampler.inherited.DENSITY_CONVENTIONS,
        "stencils": sampler.inherited.STENCIL_SPECS,
        "support_modes": sampler.inherited.SUPPORT_MODES,
        "diagnostic_support_modes": sampler.inherited.DIAGNOSTIC_SUPPORT_MODES,
        "block_shape": sampler.inherited.PRODUCTION_BLOCK_SHAPE_KJI,
        "source_version": sampler.inherited._source_version,
        "phase2_source_identity": sampler.inherited._phase2_source_identity,
        "load_cube": sampler.inherited._load_cube,
    }

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with sampler._configured_runner(
            campaign_config,
            scale=320,
            selection_set="smoke",
            matrix="baseline",
        ):
            raise RuntimeError("synthetic failure")

    assert sampler.phase3.BENCHMARK_CUBE_IDS is original["phase3_cube_ids"]
    assert sampler.inherited.BENCHMARK_CUBE_IDS is original["cube_ids"]
    assert sampler.inherited.Q_NAMES is original["q_names"]
    assert sampler.inherited.P_VALUES is original["p_values"]
    assert sampler.inherited.DENSITY_CONVENTIONS is original["density_conventions"]
    assert sampler.inherited.STENCIL_SPECS is original["stencils"]
    assert sampler.inherited.SUPPORT_MODES is original["support_modes"]
    assert sampler.inherited.DIAGNOSTIC_SUPPORT_MODES is original["diagnostic_support_modes"]
    assert sampler.inherited.PRODUCTION_BLOCK_SHAPE_KJI is original["block_shape"]
    assert sampler.inherited._source_version is original["source_version"]
    assert sampler.inherited._phase2_source_identity is original["phase2_source_identity"]
    assert sampler.inherited._load_cube is original["load_cube"]


def test_scoped_loader_accepts_non640_mmaps_with_required_analysis_bindings(
    tmp_path: Path,
) -> None:
    campaign_config = _campaign_config(tmp_path, scale=160)
    cube_id = "L160_cube_a"
    cube_root = tmp_path / "extract" / cube_id
    output_fields = {}
    for logical_name, relative_path in sampler.phase3.FIELD_PATHS.items():
        manifest_name = sampler.phase3.MANIFEST_FIELD_NAMES[logical_name]
        path = cube_root / "fields" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(160, 160, 160))
        del array
        output_fields[manifest_name] = {
            "relative_path": f"fields/{relative_path}",
            "sha256": "synthetic-sha256",
        }
    _write_json(cube_root / "manifest.json", {"output_fields": output_fields})

    with sampler._configured_runner(
        campaign_config,
        scale=160,
        selection_set="smoke",
        matrix="baseline",
    ):
        arrays = sampler.inherited._load_cube(tmp_path / "extract", cube_id)

    assert set(arrays) == set(sampler.phase3.FIELD_PATHS)
    assert {array.shape for array in arrays.values()} == {(160, 160, 160)}
    assert all(isinstance(array, np.memmap) for array in arrays.values())


def test_phase2_source_identity_binds_phase5_plan_materialization_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_config = _campaign_config(tmp_path)
    phase2_root = tmp_path / "extract"
    cube_id = "L320_cube_a"
    plan_identity = {
        "plan_relative_path": "phase5_extraction_plan.json",
        "plan_sha256": "plan-sha256",
        "marker_relative_path": "PHASE5_EXTRACTION_PLAN_COMPLETE.json",
        "marker_sha256": "marker-sha256",
    }
    materialization_identity = {
        "materialization_record_relative_path": f"phase5_materialization_records/{cube_id}.json",
        "materialization_record_sha256": "materialization-sha256",
    }
    restart_identity = {
        "restart_record_relative_path": f"restart_checks/{cube_id}.json",
        "restart_record_sha256": "restart-sha256",
    }
    _write_json(
        phase2_root / "phase5_extraction_plan.json",
        {
            "phase": "phase5_cross_scale_selected_cube_extraction",
            "status": "planned",
            "campaign_config": sampler._extraction_campaign_config_identity(campaign_config),
            "scope": {
                "L_sub": 320,
                "subset": "smoke",
                "cube_count": 2,
                "cube_ids": ["L320_cube_a", "L320_cube_b"],
            },
        },
    )
    fake_extraction = SimpleNamespace(
        _plan_identity=lambda root: plan_identity,
        _materialization_record_identity=lambda root, selected_cube_id: materialization_identity,
        _restart_record_identity=lambda root, selected_cube_id: restart_identity,
    )
    monkeypatch.setattr(sampler, "_EXTRACTION_MODULE", fake_extraction)
    monkeypatch.setattr(
        sampler,
        "_INHERITED_PHASE2_SOURCE_IDENTITY",
        lambda root, selected_cube_id, *, verify_arrays: {
            "cube_id": selected_cube_id,
            "phase2_root": str(root),
            "arrays_verified": verify_arrays,
        },
    )

    with sampler._configured_runner(
        campaign_config,
        scale=320,
        selection_set="smoke",
        matrix="baseline",
    ):
        identity = sampler._phase2_source_identity(phase2_root, cube_id)

    assert identity["phase5_extraction_plan"] == plan_identity
    assert identity["phase5_materialization_record"] == materialization_identity
    assert identity["phase5_restart_record"]["restart_record_relative_path"] == (
        f"restart_checks/{cube_id}.json"
    )
    assert identity["phase5_sampler_campaign_config"] == sampler._campaign_config_identity(
        campaign_config
    )


def test_summary_marker_is_published_and_tamper_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_config = _campaign_config(tmp_path)
    output_root = tmp_path / "sampler"
    source = {"implementation_sha256": "phase5-source"}
    monkeypatch.setattr(sampler, "_source_version", lambda: source)
    monkeypatch.setattr(
        sampler.inherited,
        "summarize",
        lambda phase2_root, selected_output_root: {"source_version": source},
    )

    with sampler._configured_runner(
        campaign_config,
        scale=320,
        selection_set="smoke",
        matrix="baseline",
    ) as selection:
        sampler._summarize(tmp_path / "extract", output_root, selection)
        assert sampler._verify_summary_marker(output_root, selection) == {
            "phase5_sampler_summary_status": "passed"
        }
        summary_path = output_root / sampler.SUMMARY_FILENAME
        summary_path.write_text(summary_path.read_text() + "\n")
        with pytest.raises(RuntimeError, match="invalid or stale"):
            sampler._verify_summary_marker(output_root, selection)


def test_rejects_unapproved_scale_and_sgs_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--scale"):
        sampler._resolve_selection(
            _campaign_config(tmp_path),
            scale=640,
            selection_set="smoke",
            matrix="baseline",
        )

    campaign_config = _campaign_config(tmp_path)
    payload = json.loads(campaign_config.read_text())
    payload["sgs_channels_authorized"] = True
    _write_json(campaign_config, payload)
    with pytest.raises(ValueError, match="SGS"):
        sampler._resolve_selection(
            campaign_config,
            scale=320,
            selection_set="smoke",
            matrix="baseline",
        )


@pytest.mark.parametrize(
    "missing_key",
    ("q_names", "density_conventions", "sgs_channels_authorized"),
)
def test_rejects_missing_frozen_quantity_policy_key(
    tmp_path: Path,
    missing_key: str,
) -> None:
    campaign_config = _campaign_config(tmp_path)
    payload = json.loads(campaign_config.read_text())
    del payload[missing_key]
    _write_json(campaign_config, payload)

    with pytest.raises(ValueError):
        sampler._resolve_selection(
            campaign_config,
            scale=320,
            selection_set="smoke",
            matrix="baseline",
        )


def test_wrapper_has_cpu_multinode_work_and_no_email_directives() -> None:
    text = WRAPPER.read_text()

    assert "#SBATCH -p batch" in text
    assert "#SBATCH --cpus-per-task=32" in text
    assert "#SBATCH -o logs/" in text
    assert "#SBATCH -e logs/" in text
    assert "--mail" not in text
    assert 'if [[ -e "${RUN_DIR}" ]]; then' in text
    assert text.index('if [[ -e "${RUN_DIR}" ]]; then') < text.index("mkdir -p logs")
    assert '--ntasks="${SLURM_JOB_NUM_NODES:-1}"' in text
    assert "--ntasks-per-node=1" in text


def test_wrapper_rejects_existing_run_dir_before_writes(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing_run"
    run_dir.mkdir()
    output_root = tmp_path / "new_parent" / "sampler"
    env = {
        **os.environ,
        "SFUNCTOR_DIR": str(tmp_path / "missing_sfunctor"),
        "CAMPAIGN_CONFIG": str(tmp_path / "campaign.json"),
        "SCALE": "320",
        "SELECTION_SET": "smoke",
        "MATRIX": "baseline",
        "PHASE2_ROOT": str(tmp_path / "extract"),
        "OUTPUT_ROOT": str(output_root),
        "RUN_DIR": str(run_dir),
        "ACTION": "plan",
    }

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    assert "allocation directory already exists" in result.stderr
    assert not output_root.parent.exists()
    assert not (tmp_path / "logs").exists()
