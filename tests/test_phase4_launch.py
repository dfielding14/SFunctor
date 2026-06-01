"""Focused tests for the bounded Phase 4 operational adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from scripts.phase4 import run_phase4_batch_a_sampler as batch_a
from scripts.phase4 import run_phase4_batch_a2_3point_extension as batch_a2_extension
from scripts.phase4 import run_phase4_batch_a2_sampler as batch_a2
from scripts.phase4 import run_phase4_batch_b_representative_sampler as batch_b
from scripts.phase4 import run_phase4_extraction as extraction
from sfunctor.io.cube_extract import CubeExtractionError, CubeSelection


def _selections(
    *,
    cube_ids: tuple[str, ...] = extraction.PHASE4_PILOT_CUBE_IDS,
    lsub: int = 640,
) -> dict[str, CubeSelection]:
    return {
        cube_id: CubeSelection(
            cube_id,
            (0, 640, 0, 640, 0, 640),
            required_rank_ids=(index,),
            role=f"role-{index}",
            lsub=lsub,
        )
        for index, cube_id in enumerate(cube_ids)
    }


def _core_source_hashes() -> dict[str, str]:
    return {
        relative_path: f"{index + 1:064x}"
        for index, relative_path in enumerate(extraction.CORE_EXTRACTOR_SOURCE_PATHS)
    }


def _source_version(hashes: dict[str, str], *, dirty: bool = False) -> dict[str, object]:
    return {
        "commit": "commit",
        "dirty": dirty,
        "implementation_source_hashes": hashes,
        "implementation_sha256": extraction._mapping_sha256(hashes),
    }


def _materialization_versions() -> tuple[dict[str, object], dict[str, object]]:
    core_hashes = _core_source_hashes()
    plan_hashes = {
        "scripts/phase4/run_phase4_extraction.py": "a" * 64,
        "job_scripts/phase4/run_phase4_extract_andes.sh": "b" * 64,
        **core_hashes,
    }
    return _source_version(plan_hashes), _source_version(core_hashes)


def _write_materialization_inputs(
    output_root: Path,
    cube_id: str,
    *,
    plan_version: dict[str, object] | None = None,
    manifest_version: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, Path]:
    default_plan_version, default_manifest_version = _materialization_versions()
    plan_path = output_root / extraction.PLAN_FILENAME
    marker_path = output_root / extraction.PLAN_MARKER_FILENAME
    manifest_path = output_root / cube_id / "manifest.json"
    completion_path = output_root / cube_id / "COMPLETE.json"
    extraction._atomic_write_json(
        plan_path,
        {"source_version": default_plan_version if plan_version is None else plan_version},
    )
    extraction._atomic_write_json(marker_path, {"marker": True})
    extraction._atomic_write_json(
        manifest_path,
        {"code_version": default_manifest_version if manifest_version is None else manifest_version},
    )
    extraction._atomic_write_json(
        completion_path,
        {
            "cube_id": cube_id,
            "manifest_sha256": extraction.file_sha256(manifest_path),
        },
    )
    return plan_path, marker_path, manifest_path, completion_path


def test_frozen_pilot_requires_exact_21_lsub640_members(monkeypatch):
    monkeypatch.setattr(extraction, "load_pilot_selections", lambda trusted_run: _selections())
    assert tuple(extraction._load_frozen_pilot(Path("/trusted"))) == extraction.PHASE4_PILOT_CUBE_IDS

    monkeypatch.setattr(
        extraction,
        "load_pilot_selections",
        lambda trusted_run: _selections(cube_ids=extraction.PHASE4_PILOT_CUBE_IDS[:-1]),
    )
    with pytest.raises(CubeExtractionError, match="exact frozen 21-selection"):
        extraction._load_frozen_pilot(Path("/trusted"))

    monkeypatch.setattr(
        extraction,
        "load_pilot_selections",
        lambda trusted_run: _selections(lsub=1280),
    )
    with pytest.raises(CubeExtractionError, match="exact frozen 21-selection"):
        extraction._load_frozen_pilot(Path("/trusted"))


def test_all_pilot_guard_requires_exact_21_selections():
    args = argparse.Namespace(all_pilot=True, cube_id=None)
    with pytest.raises(SystemExit, match="exactly 21"):
        extraction._selected_cube_ids(args, _selections(cube_ids=extraction.PHASE4_PILOT_CUBE_IDS[:-1]))
    with pytest.raises(SystemExit, match="exactly 21"):
        extraction._selected_cube_ids(args, _selections(lsub=1280))

    assert extraction._selected_cube_ids(args, _selections()) == extraction.PHASE4_PILOT_CUBE_IDS


def test_extraction_plan_marker_is_source_bound(tmp_path, monkeypatch):
    source = {
        "commit": "commit",
        "dirty": False,
        "implementation_source_hashes": {"adapter": "sha"},
        "implementation_sha256": "implementation-sha",
    }
    snapshot = {
        "full_resolution_basename": "snapshot.bin",
        "target_time": 6.0,
        "target_cycle": 42,
        "full_snapshot_identity": {"sha256": "snapshot-sha"},
    }
    monkeypatch.setattr(extraction, "_load_frozen_pilot", lambda trusted_run: _selections())
    monkeypatch.setattr(extraction, "load_snapshot_identity", lambda trusted_run: snapshot)
    monkeypatch.setattr(extraction, "verify_trusted_run", lambda trusted_run: {"pilot": "sha"})
    monkeypatch.setattr(extraction, "_source_version", lambda: source)
    output_root = tmp_path / "extract"

    payload = extraction.plan(Path("/trusted"), Path("/data"), output_root, basename=None)

    assert payload["pilot_cube_count"] == 21
    assert json.loads((output_root / extraction.PLAN_MARKER_FILENAME).read_text())["status"] == "passed"
    monkeypatch.setattr(
        extraction,
        "_source_version",
        lambda: {**source, "implementation_sha256": "changed"},
    )
    with pytest.raises(CubeExtractionError, match="stale"):
        extraction.verify_plan(Path("/trusted"), Path("/data"), output_root, basename=None)


def test_batch_a_configures_only_exact_approved_matrix_and_restores_inherited_module():
    original_ids = batch_a.inherited.BENCHMARK_CUBE_IDS
    original_modes = batch_a.inherited.SUPPORT_MODES
    original_stencils = batch_a.inherited.STENCIL_SPECS

    with batch_a._configured_runner():
        configuration = batch_a.inherited._campaign_configuration()
        assert batch_a.inherited.BENCHMARK_CUBE_IDS == extraction.PHASE4_PILOT_CUBE_IDS
        assert batch_a.phase3.BENCHMARK_CUBE_IDS == extraction.PHASE4_PILOT_CUBE_IDS
        assert configuration["q_names"] == ("B", "u")
        assert configuration["p_values"] == (2.0,)
        assert configuration["support_modes"] == ("all_valid_origins", "shell_local")
        assert set(configuration["stencils"]) == {2}
        assert configuration["stencils"][2]["ell_max"] == 320

    assert batch_a.inherited.BENCHMARK_CUBE_IDS == original_ids
    assert batch_a.inherited.SUPPORT_MODES == original_modes
    assert batch_a.inherited.STENCIL_SPECS == original_stencils


def test_batch_a2_configures_only_exact_approved_matrix_and_restores_inherited_module(tmp_path):
    original_ids = batch_a2.inherited.BENCHMARK_CUBE_IDS
    original_modes = batch_a2.inherited.SUPPORT_MODES
    original_stencils = batch_a2.inherited.STENCIL_SPECS

    with batch_a2._configured_runner(tmp_path / "batch_a"):
        configuration = batch_a2.inherited._campaign_configuration()
        assert batch_a2.inherited.BENCHMARK_CUBE_IDS == batch_a2.PHASE4_A2_CUBE_IDS
        assert batch_a2.phase3.BENCHMARK_CUBE_IDS == batch_a2.PHASE4_A2_CUBE_IDS
        assert configuration["q_names"] == ("B", "u")
        assert configuration["p_values"] == (2.0,)
        assert configuration["support_modes"] == ("all_valid_origins", "shell_local")
        assert set(configuration["stencils"]) == {3, 5}
        assert configuration["stencils"][3]["ell_max"] == 160
        assert configuration["stencils"][5]["ell_max"] == 80

    assert batch_a2.inherited.BENCHMARK_CUBE_IDS == original_ids
    assert batch_a2.inherited.SUPPORT_MODES == original_modes
    assert batch_a2.inherited.STENCIL_SPECS == original_stencils


def test_batch_a2_extension_configures_only_exact_approved_matrix_and_restores_inherited_module(
    tmp_path,
):
    original_ids = batch_a2_extension.inherited.BENCHMARK_CUBE_IDS
    original_modes = batch_a2_extension.inherited.SUPPORT_MODES
    original_stencils = batch_a2_extension.inherited.STENCIL_SPECS

    with batch_a2_extension._configured_runner(tmp_path / "batch_a", tmp_path / "batch_a2"):
        configuration = batch_a2_extension.inherited._campaign_configuration()
        assert (
            batch_a2_extension.inherited.BENCHMARK_CUBE_IDS
            == extraction.PHASE4_PILOT_CUBE_IDS
        )
        assert batch_a2_extension.phase3.BENCHMARK_CUBE_IDS == extraction.PHASE4_PILOT_CUBE_IDS
        assert configuration["q_names"] == ("B", "u")
        assert configuration["p_values"] == (2.0,)
        assert configuration["support_modes"] == ("all_valid_origins", "shell_local")
        assert set(configuration["stencils"]) == {3}
        assert configuration["stencils"][3]["ell_max"] == 160

    assert batch_a2_extension.inherited.BENCHMARK_CUBE_IDS == original_ids
    assert batch_a2_extension.inherited.SUPPORT_MODES == original_modes
    assert batch_a2_extension.inherited.STENCIL_SPECS == original_stencils


def test_batch_b_configures_only_exact_approved_matrix_and_restores_inherited_module(tmp_path):
    original_ids = batch_b.inherited.BENCHMARK_CUBE_IDS
    original_q_names = batch_b.inherited.Q_NAMES
    original_p_values = batch_b.inherited.P_VALUES
    original_density_conventions = batch_b.inherited.DENSITY_CONVENTIONS
    original_modes = batch_b.inherited.SUPPORT_MODES
    original_stencils = batch_b.inherited.STENCIL_SPECS

    with batch_b._configured_runner(
        tmp_path / "batch_a", tmp_path / "batch_a2", tmp_path / "batch_a2_all21"
    ):
        configuration = batch_b.inherited._campaign_configuration()
        assert batch_b.inherited.BENCHMARK_CUBE_IDS == batch_b.PHASE4_BATCH_B_CUBE_IDS
        assert batch_b.phase3.BENCHMARK_CUBE_IDS == batch_b.PHASE4_BATCH_B_CUBE_IDS
        assert configuration["q_names"] == ("B", "u")
        assert configuration["p_values"] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert configuration["density_conventions"] == ("not applicable", "not applicable")
        assert configuration["support_modes"] == ("all_valid_origins", "shell_local")
        assert set(configuration["stencils"]) == {2}
        assert configuration["stencils"][2]["ell_max"] == 320

    assert batch_b.inherited.BENCHMARK_CUBE_IDS == original_ids
    assert batch_b.inherited.Q_NAMES == original_q_names
    assert batch_b.inherited.P_VALUES == original_p_values
    assert batch_b.inherited.DENSITY_CONVENTIONS == original_density_conventions
    assert batch_b.inherited.SUPPORT_MODES == original_modes
    assert batch_b.inherited.STENCIL_SPECS == original_stencils


def test_batch_b_restores_inherited_module_after_exception(tmp_path):
    original = (
        batch_b.phase3.BENCHMARK_CUBE_IDS,
        batch_b.inherited.BENCHMARK_CUBE_IDS,
        batch_b.inherited.Q_NAMES,
        batch_b.inherited.P_VALUES,
        batch_b.inherited.DENSITY_CONVENTIONS,
        batch_b.inherited.STENCIL_SPECS,
        batch_b.inherited.SUPPORT_MODES,
    )

    with pytest.raises(RuntimeError, match="synthetic forced exit"):
        with batch_b._configured_runner(
            tmp_path / "batch_a", tmp_path / "batch_a2", tmp_path / "batch_a2_all21"
        ):
            raise RuntimeError("synthetic forced exit")

    assert (
        batch_b.phase3.BENCHMARK_CUBE_IDS,
        batch_b.inherited.BENCHMARK_CUBE_IDS,
        batch_b.inherited.Q_NAMES,
        batch_b.inherited.P_VALUES,
        batch_b.inherited.DENSITY_CONVENTIONS,
        batch_b.inherited.STENCIL_SPECS,
        batch_b.inherited.SUPPORT_MODES,
    ) == original


def test_batch_b_decision_artifact_rejects_expansion_authorization_mutation(tmp_path, monkeypatch):
    source_path = Path(__file__).resolve().parents[1] / batch_b.DECISION_RELATIVE_PATH
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(source_path.read_text())
    monkeypatch.setattr(batch_b, "DECISION_RELATIVE_PATH", str(decision_path))

    assert batch_b._decision_identity()["decision_sha256"] == batch_b.file_sha256(decision_path)
    decision = json.loads(decision_path.read_text())
    decision["all21_batch_b_expansion_authorized"] = True
    batch_b.inherited._atomic_write_json(decision_path, decision)

    with pytest.raises(RuntimeError, match="invalid Phase 4 Batch B"):
        batch_b._decision_identity()


def test_batch_b_rejects_cube_outside_exact_representative_set(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_b, "_configured_batch_a_root", tmp_path / "batch_a")
    monkeypatch.setattr(batch_b, "_configured_representative_a2_root", tmp_path / "batch_a2")
    monkeypatch.setattr(
        batch_b, "_configured_all21_3point_extension_root", tmp_path / "batch_a2_all21"
    )
    monkeypatch.setattr(
        batch_b,
        "_batch_a_reference_sources",
        lambda phase2_root, batch_a_root: ({}, {"batch_a": "bound"}),
    )

    with pytest.raises(ValueError, match="restricted to approved representative IDs"):
        batch_b._phase2_source_identity(tmp_path, "L640_unapproved")


def test_batch_a_source_binding_and_phase4_summary_marker(tmp_path, monkeypatch):
    source = batch_a._source_version()
    hashes = source["implementation_source_hashes"]
    assert "scripts/phase4/run_phase4_batch_a_sampler.py" in hashes
    assert "scripts/phase3a/run_phase3a_sampler.py" in hashes
    assert "sfunctor/core/phase3a.py" in hashes

    monkeypatch.setattr(
        batch_a.inherited,
        "summarize",
        lambda phase2_root, output_root: {
            "source_version": {"implementation_sha256": "phase4-sha"},
            "verification": {"status": "passed"},
        },
    )
    payload = batch_a._summarize(tmp_path / "extract", tmp_path / "sampler")
    marker = json.loads((tmp_path / "sampler" / batch_a.SUMMARY_MARKER_FILENAME).read_text())

    assert payload["phase"] == "phase4_batch_a_bounded_21_cube_2point"
    assert marker["status"] == "release_aggregation_complete"
    assert marker["implementation_sha256"] == "phase4-sha"
    monkeypatch.setattr(batch_a.inherited, "verify", lambda phase2_root, output_root: {"status": "passed"})
    monkeypatch.setattr(batch_a, "_source_version", lambda: {"implementation_sha256": "phase4-sha"})
    assert batch_a._verify(tmp_path / "extract", tmp_path / "sampler")["phase4_summary_status"] == "passed"
    (tmp_path / "sampler" / batch_a.SUMMARY_FILENAME).write_text('{"tampered": true}\n')
    with pytest.raises(RuntimeError, match="summary marker"):
        batch_a._verify(tmp_path / "extract", tmp_path / "sampler")


def test_batch_a2_source_binding_and_phase4_summary_marker(tmp_path, monkeypatch):
    source = batch_a2._source_version()
    hashes = source["implementation_source_hashes"]
    assert "scripts/phase4/run_phase4_batch_a2_sampler.py" in hashes
    assert "scripts/phase3a/run_phase3a_sampler.py" in hashes
    assert "sfunctor/core/phase3a.py" in hashes

    monkeypatch.setattr(
        batch_a2.inherited,
        "summarize",
        lambda phase2_root, output_root: {
            "source_version": {"implementation_sha256": "phase4-a2-sha"},
            "verification": {"status": "passed"},
        },
    )
    with batch_a2._configured_runner(tmp_path / "batch_a"):
        payload = batch_a2._summarize(tmp_path / "extract", tmp_path / "sampler")
    marker = json.loads((tmp_path / "sampler" / batch_a2.SUMMARY_MARKER_FILENAME).read_text())

    assert payload["phase"] == "phase4_batch_a2_bounded_4_cube_3point_5point"
    assert marker["status"] == "release_aggregation_complete"
    assert marker["implementation_sha256"] == "phase4-a2-sha"
    monkeypatch.setattr(batch_a2.inherited, "verify", lambda phase2_root, output_root: {"status": "passed"})
    monkeypatch.setattr(batch_a2, "_source_version", lambda: {"implementation_sha256": "phase4-a2-sha"})
    with batch_a2._configured_runner(tmp_path / "batch_a"):
        assert batch_a2._verify(tmp_path / "extract", tmp_path / "sampler")[
            "phase4_batch_a2_summary_status"
        ] == "passed"
    (tmp_path / "sampler" / batch_a2.SUMMARY_FILENAME).write_text('{"tampered": true}\n')
    with batch_a2._configured_runner(tmp_path / "batch_a"):
        with pytest.raises(RuntimeError, match="summary marker"):
            batch_a2._verify(tmp_path / "extract", tmp_path / "sampler")


def test_batch_a2_extension_source_binding_and_phase4_summary_marker(tmp_path, monkeypatch):
    source = batch_a2_extension._source_version()
    hashes = source["implementation_source_hashes"]
    assert "scripts/phase4/run_phase4_batch_a2_3point_extension.py" in hashes
    assert "scripts/phase3a/run_phase3a_sampler.py" in hashes
    assert "sfunctor/core/phase3a.py" in hashes

    monkeypatch.setattr(
        batch_a2_extension.inherited,
        "summarize",
        lambda phase2_root, output_root: {
            "source_version": {"implementation_sha256": "phase4-a2-extension-sha"},
            "verification": {"status": "passed"},
        },
    )
    with batch_a2_extension._configured_runner(tmp_path / "batch_a", tmp_path / "batch_a2"):
        payload = batch_a2_extension._summarize(tmp_path / "extract", tmp_path / "sampler")
    marker = json.loads(
        (tmp_path / "sampler" / batch_a2_extension.SUMMARY_MARKER_FILENAME).read_text()
    )

    assert payload["phase"] == "phase4_batch_a2_all21_3point_extension"
    assert marker["status"] == "release_aggregation_complete"
    assert marker["implementation_sha256"] == "phase4-a2-extension-sha"
    monkeypatch.setattr(
        batch_a2_extension.inherited,
        "verify",
        lambda phase2_root, output_root: {"status": "passed"},
    )
    monkeypatch.setattr(
        batch_a2_extension,
        "_source_version",
        lambda: {"implementation_sha256": "phase4-a2-extension-sha"},
    )
    with batch_a2_extension._configured_runner(tmp_path / "batch_a", tmp_path / "batch_a2"):
        assert batch_a2_extension._verify(tmp_path / "extract", tmp_path / "sampler")[
            "phase4_batch_a2_3point_extension_summary_status"
        ] == "passed"
    (tmp_path / "sampler" / batch_a2_extension.SUMMARY_FILENAME).write_text(
        '{"tampered": true}\n'
    )
    with batch_a2_extension._configured_runner(tmp_path / "batch_a", tmp_path / "batch_a2"):
        with pytest.raises(RuntimeError, match="summary marker"):
            batch_a2_extension._verify(tmp_path / "extract", tmp_path / "sampler")


@pytest.mark.parametrize(
    "mutated_relative_path",
    ("plan.json", "plan_marker.json", "materialization.json", "restart_check.json"),
)
def test_batch_a2_extension_replays_frozen_extraction_sidecar_hashes(
    tmp_path, monkeypatch, mutated_relative_path
):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    sidecars = {
        "plan.json": '{"plan": true}\n',
        "plan_marker.json": '{"plan_marker": true}\n',
        "materialization.json": '{"materialization": true}\n',
        "restart_check.json": '{"restart_check": true}\n',
    }
    for relative_path, text in sidecars.items():
        (tmp_path / relative_path).write_text(text)
    frozen = {
        "cube_id": cube_id,
        "phase2_root": str(tmp_path.resolve()),
        "completion_relative_path": f"{cube_id}/COMPLETE.json",
        "completion_sha256": "completion-sha",
        "manifest_relative_path": f"{cube_id}/manifest.json",
        "manifest_sha256": "manifest-sha",
        "analysis_field_sha256": "analysis-field-sha",
        "phase4_extraction_plan": {
            "plan_relative_path": "plan.json",
            "plan_sha256": extraction.file_sha256(tmp_path / "plan.json"),
            "marker_relative_path": "plan_marker.json",
            "marker_sha256": extraction.file_sha256(tmp_path / "plan_marker.json"),
        },
        "phase4_materialization_record": {
            "materialization_record_relative_path": "materialization.json",
            "materialization_record_sha256": extraction.file_sha256(
                tmp_path / "materialization.json"
            ),
        },
        "phase4_restart_check": {
            "restart_check_relative_path": "restart_check.json",
            "restart_check_sha256": extraction.file_sha256(tmp_path / "restart_check.json"),
        },
    }
    observed = {
        key: frozen[key]
        for key in (
            "cube_id",
            "phase2_root",
            "completion_relative_path",
            "completion_sha256",
            "manifest_relative_path",
            "manifest_sha256",
            "analysis_field_sha256",
        )
    }
    monkeypatch.setattr(
        batch_a2_extension.representative,
        "_batch_a_reference_sources",
        lambda phase2_root, batch_a_root: ({cube_id: frozen}, {"batch_a": "bound"}),
    )
    monkeypatch.setattr(
        batch_a2_extension,
        "_INHERITED_PHASE2_SOURCE_IDENTITY",
        lambda phase2_root, requested_cube_id, verify_arrays: observed,
    )
    monkeypatch.setattr(
        batch_a2_extension,
        "_representative_a2_reference",
        lambda batch_a_root, representative_a2_root: {"representative_a2": "bound"},
    )
    monkeypatch.setattr(batch_a2_extension, "_configured_batch_a_root", tmp_path / "batch_a")
    monkeypatch.setattr(
        batch_a2_extension,
        "_configured_representative_a2_root",
        tmp_path / "representative_a2",
    )

    identity = batch_a2_extension._phase2_source_identity(tmp_path, cube_id)
    assert identity["phase4_batch_a_reference"] == {"batch_a": "bound"}

    (tmp_path / mutated_relative_path).write_text('{"mutated": true}\n')
    with pytest.raises(RuntimeError, match="stale Phase 4 Batch A reference artifact"):
        batch_a2_extension._phase2_source_identity(tmp_path, cube_id)


def test_batch_a_binds_phase4_extraction_plan_marker(tmp_path, monkeypatch):
    plan_path = tmp_path / extraction.PLAN_FILENAME
    marker_path = tmp_path / extraction.PLAN_MARKER_FILENAME
    plan = {
        "phase": "phase4_bounded_21_cube_extraction",
        "status": "planned",
        "trusted_run": "/trusted",
        "data_root": "/data",
        "output_root": str(tmp_path.resolve()),
        "source_basename": "snapshot.bin",
        "pilot_cube_count": 21,
        "pilot_cube_ids": extraction.PHASE4_PILOT_CUBE_IDS,
        "source_version": {"implementation_sha256": "extraction-sha"},
    }
    extraction._atomic_write_json(plan_path, plan)
    extraction._atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "status": "passed",
            "plan_sha256": extraction.file_sha256(plan_path),
            "implementation_sha256": "extraction-sha",
        },
    )
    monkeypatch.setattr(
        extraction,
        "_source_version",
        lambda: {"implementation_sha256": "extraction-sha"},
    )
    monkeypatch.setattr(
        extraction,
        "verify_plan",
        lambda trusted_run, data_root, output_root, basename: json.loads(plan_path.read_text()),
    )

    identity = batch_a._phase4_extraction_plan_identity(tmp_path)

    assert identity["plan_sha256"] == extraction.file_sha256(plan_path)
    monkeypatch.setattr(
        extraction,
        "_source_version",
        lambda: {"implementation_sha256": "changed"},
    )
    with pytest.raises(RuntimeError, match="invalid Phase 4 extraction plan binding"):
        batch_a._phase4_extraction_plan_identity(tmp_path)
    monkeypatch.setattr(
        extraction,
        "_source_version",
        lambda: {"implementation_sha256": "extraction-sha"},
    )
    marker_path.write_text('{"status": "changed"}\n')
    with pytest.raises(RuntimeError, match="invalid Phase 4 extraction plan binding"):
        batch_a._phase4_extraction_plan_identity(tmp_path)


def test_batch_a_rejects_extraction_plan_cloned_to_another_root(tmp_path, monkeypatch):
    output_root = tmp_path / "clone"
    output_root.mkdir()
    plan_path = output_root / extraction.PLAN_FILENAME
    marker_path = output_root / extraction.PLAN_MARKER_FILENAME
    extraction._atomic_write_json(
        plan_path,
        {
            "phase": "phase4_bounded_21_cube_extraction",
            "status": "planned",
            "trusted_run": "/trusted",
            "data_root": "/data",
            "output_root": "/original/phase4/root",
            "source_basename": "snapshot.bin",
            "pilot_cube_count": 21,
            "pilot_cube_ids": extraction.PHASE4_PILOT_CUBE_IDS,
            "source_version": {"implementation_sha256": "extraction-sha"},
        },
    )
    extraction._atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "status": "passed",
            "plan_sha256": extraction.file_sha256(plan_path),
            "implementation_sha256": "extraction-sha",
        },
    )
    monkeypatch.setattr(
        extraction,
        "_source_version",
        lambda: {"implementation_sha256": "extraction-sha"},
    )

    with pytest.raises(RuntimeError, match="invalid Phase 4 extraction plan binding"):
        batch_a._phase4_extraction_plan_identity(output_root)


def test_batch_a_source_identity_requires_restart_check_and_strict_verification(tmp_path, monkeypatch):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    plan_identity = {"plan": "sha"}
    materialization_identity = {"materialization": "sha"}
    publication_identity = {"publication": "sha"}
    monkeypatch.setattr(batch_a, "_phase4_extraction_plan_identity", lambda phase2_root: plan_identity)
    monkeypatch.setattr(
        extraction,
        "_materialization_record_identity",
        lambda phase2_root, requested_cube_id: materialization_identity,
    )
    monkeypatch.setattr(
        extraction,
        "_cube_publication_identity",
        lambda phase2_root, requested_cube_id: publication_identity,
    )
    monkeypatch.setattr(
        batch_a,
        "_INHERITED_PHASE2_SOURCE_IDENTITY",
        lambda phase2_root, requested_cube_id, verify_arrays: {"cube_id": requested_cube_id},
    )
    strict_calls = []
    monkeypatch.setattr(
        batch_a,
        "_strict_phase4_cube_verification",
        lambda phase2_root, requested_cube_id: strict_calls.append(requested_cube_id),
    )

    with pytest.raises(FileNotFoundError):
        batch_a._phase2_source_identity(tmp_path, cube_id, verify_arrays=True)

    extraction._atomic_write_json(
        tmp_path / "restart_checks" / f"{cube_id}.json",
        {
            "status": "passed",
            "cube_id": cube_id,
            "verify_hashes": True,
            "phase4_extraction_plan": plan_identity,
            "phase4_materialization_record": materialization_identity,
            "cube_publication": publication_identity,
        },
    )
    identity = batch_a._phase2_source_identity(tmp_path, cube_id, verify_arrays=True)

    assert strict_calls == [cube_id]
    assert identity["phase4_extraction_plan"] == {"plan": "sha"}
    assert identity["phase4_restart_check"]["restart_check_relative_path"] == (
        f"restart_checks/{cube_id}.json"
    )
    extraction._atomic_write_json(
        tmp_path / "restart_checks" / f"{cube_id}.json",
        {"status": "passed", "cube_id": cube_id, "verify_hashes": True},
    )
    with pytest.raises(RuntimeError, match="invalid Phase 4 restart check"):
        batch_a._phase2_source_identity(tmp_path, cube_id, verify_arrays=False)


def test_batch_a_source_identity_rejects_failed_strict_verification(tmp_path, monkeypatch):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    plan_identity = {"plan": "sha"}
    materialization_identity = {"materialization": "sha"}
    publication_identity = {"publication": "sha"}
    extraction._atomic_write_json(
        tmp_path / "restart_checks" / f"{cube_id}.json",
        {
            "status": "passed",
            "cube_id": cube_id,
            "verify_hashes": True,
            "phase4_extraction_plan": plan_identity,
            "phase4_materialization_record": materialization_identity,
            "cube_publication": publication_identity,
        },
    )
    monkeypatch.setattr(batch_a, "_phase4_extraction_plan_identity", lambda phase2_root: plan_identity)
    monkeypatch.setattr(
        extraction,
        "_materialization_record_identity",
        lambda phase2_root, requested_cube_id: materialization_identity,
    )
    monkeypatch.setattr(
        extraction,
        "_cube_publication_identity",
        lambda phase2_root, requested_cube_id: publication_identity,
    )
    monkeypatch.setattr(
        batch_a,
        "_INHERITED_PHASE2_SOURCE_IDENTITY",
        lambda phase2_root, requested_cube_id, verify_arrays: {"cube_id": requested_cube_id},
    )
    monkeypatch.setattr(
        batch_a,
        "_strict_phase4_cube_verification",
        lambda phase2_root, requested_cube_id: (_ for _ in ()).throw(
            CubeExtractionError("synthetic cube rejected")
        ),
    )

    with pytest.raises(CubeExtractionError, match="synthetic cube rejected"):
        batch_a._phase2_source_identity(tmp_path, cube_id, verify_arrays=True)


def test_phase4_materialization_record_rejects_unrecorded_or_changed_cube(tmp_path):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    _write_materialization_inputs(tmp_path, cube_id)
    cube_root = tmp_path / cube_id

    with pytest.raises(FileNotFoundError):
        extraction._materialization_record_identity(tmp_path, cube_id)
    with pytest.raises(FileNotFoundError):
        extraction._bind_extraction_materialization(
            tmp_path,
            cube_id,
            {"status": "reused_complete_output"},
        )

    fresh = extraction._bind_extraction_materialization(
        tmp_path,
        cube_id,
        {"status": "extracted"},
    )
    assert fresh["phase4_materialization_record"]["materialization_record_relative_path"] == (
        f"{extraction.MATERIALIZATION_ROOT}/{cube_id}.json"
    )
    reused = extraction._bind_extraction_materialization(
        tmp_path,
        cube_id,
        {"status": "reused_complete_output"},
    )
    assert reused["phase4_materialization_record"] == fresh["phase4_materialization_record"]
    (cube_root / "manifest.json").write_text('{"changed": true}\n')
    with pytest.raises(CubeExtractionError, match="publication marker"):
        extraction._materialization_record_identity(tmp_path, cube_id)


@pytest.mark.parametrize("mutation", ("aggregate", "source_digest", "missing_source"))
def test_phase4_materialization_guard_rejects_mismatched_core_provenance(tmp_path, mutation):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    plan_version, manifest_version = _materialization_versions()
    if mutation == "aggregate":
        manifest_version["implementation_sha256"] = "f" * 64
    else:
        manifest_hashes = dict(manifest_version["implementation_source_hashes"])
        core_path = extraction.CORE_EXTRACTOR_SOURCE_PATHS[0]
        if mutation == "source_digest":
            manifest_hashes[core_path] = "f" * 64
        else:
            manifest_hashes.pop(core_path)
        manifest_version = _source_version(manifest_hashes)
    _write_materialization_inputs(
        tmp_path,
        cube_id,
        plan_version=plan_version,
        manifest_version=manifest_version,
    )

    with pytest.raises(CubeExtractionError, match="core provenance mismatch"):
        extraction._materialization_record_payload(tmp_path, cube_id)


@pytest.mark.parametrize("dirty_record", ("plan", "manifest"))
def test_phase4_materialization_guard_requires_clean_provenance(tmp_path, dirty_record):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    plan_version, manifest_version = _materialization_versions()
    if dirty_record == "plan":
        plan_version["dirty"] = True
    else:
        manifest_version["dirty"] = True
    _write_materialization_inputs(
        tmp_path,
        cube_id,
        plan_version=plan_version,
        manifest_version=manifest_version,
    )

    with pytest.raises(CubeExtractionError, match="explicitly clean"):
        extraction._materialization_record_payload(tmp_path, cube_id)


def test_phase4_materialization_guard_fails_closed_for_missing_plan_core_source(tmp_path):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    plan_version, manifest_version = _materialization_versions()
    plan_hashes = dict(plan_version["implementation_source_hashes"])
    plan_hashes.pop(extraction.CORE_EXTRACTOR_SOURCE_PATHS[0])
    _write_materialization_inputs(
        tmp_path,
        cube_id,
        plan_version=_source_version(plan_hashes),
        manifest_version=manifest_version,
    )

    with pytest.raises(CubeExtractionError, match="missing core extractor source hashes"):
        extraction._materialization_record_payload(tmp_path, cube_id)


def test_phase4_materialization_identity_rejects_coherently_repinned_core_provenance(tmp_path):
    cube_id = extraction.PHASE4_PILOT_CUBE_IDS[0]
    plan_path, marker_path, manifest_path, completion_path = _write_materialization_inputs(
        tmp_path,
        cube_id,
    )
    extraction._bind_extraction_materialization(tmp_path, cube_id, {"status": "extracted"})

    plan = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    core_path = extraction.CORE_EXTRACTOR_SOURCE_PATHS[0]
    plan["source_version"]["implementation_source_hashes"][core_path] = "f" * 64
    plan["source_version"]["implementation_sha256"] = extraction._mapping_sha256(
        plan["source_version"]["implementation_source_hashes"]
    )
    manifest["code_version"]["implementation_source_hashes"][core_path] = "f" * 64
    manifest["code_version"]["implementation_sha256"] = extraction._mapping_sha256(
        manifest["code_version"]["implementation_source_hashes"]
    )
    extraction._atomic_write_json(plan_path, plan)
    extraction._atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "status": "passed",
            "plan_sha256": extraction.file_sha256(plan_path),
            "implementation_sha256": plan["source_version"]["implementation_sha256"],
        },
    )
    extraction._atomic_write_json(manifest_path, manifest)
    extraction._atomic_write_json(
        completion_path,
        {
            "cube_id": cube_id,
            "manifest_sha256": extraction.file_sha256(manifest_path),
        },
    )

    with pytest.raises(CubeExtractionError, match="invalid or stale"):
        extraction._materialization_record_identity(tmp_path, cube_id)


def test_batch_a_verify_rejects_orphaned_phase4_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_a.inherited, "verify", lambda phase2_root, output_root: {"status": "passed"})
    output_root = tmp_path / "sampler"
    output_root.mkdir()
    (output_root / batch_a.SUMMARY_FILENAME).write_text('{"orphaned": true}\n')

    with pytest.raises(RuntimeError, match="orphaned"):
        batch_a._verify(tmp_path / "extract", output_root)


@pytest.mark.parametrize(
    ("relative_path", "required_roots"),
    [
        (
            "job_scripts/phase4/run_phase4_extract_andes.sh",
            ("OUTPUT_ROOT", "RUN_DIR"),
        ),
        (
            "job_scripts/phase4/run_phase4_batch_a_sampler_andes.sh",
            ("PHASE2_ROOT", "OUTPUT_ROOT", "RUN_DIR"),
        ),
        (
            "job_scripts/phase4/run_phase4_batch_a2_sampler_andes.sh",
            ("PHASE2_ROOT", "BATCH_A_ROOT", "OUTPUT_ROOT", "RUN_DIR"),
        ),
        (
            "job_scripts/phase4/run_phase4_batch_a2_3point_extension_andes.sh",
            ("PHASE2_ROOT", "BATCH_A_ROOT", "REPRESENTATIVE_A2_ROOT", "OUTPUT_ROOT", "RUN_DIR"),
        ),
        (
            "job_scripts/phase4/run_phase4_batch_b_representative_andes.sh",
            (
                "PHASE2_ROOT",
                "BATCH_A_ROOT",
                "REPRESENTATIVE_A2_ROOT",
                "ALL21_3POINT_EXTENSION_ROOT",
                "OUTPUT_ROOT",
                "RUN_DIR",
            ),
        ),
    ],
)
def test_andes_wrappers_are_cpu_batch_policy_compliant(relative_path, required_roots):
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    text = path.read_text()

    subprocess.run(("bash", "-n", str(path)), check=True)
    assert "#SBATCH -A AST207" in text
    assert "#SBATCH -p batch" in text
    assert "#SBATCH -o logs/" in text
    assert "#SBATCH -e logs/" in text
    assert "#SBATCH --cpus-per-task=" in text
    assert "--mail" not in text
    assert "sgs" not in text.lower()
    assert "sbatch " not in text
    assert "recover_stale_lock" in text
    assert "LOCK_OWNER=" in text
    assert "archive_slurm_resources" in text
    assert "sacct -j" in text
    for root_name in required_roots:
        assert f': "${{{root_name}:?' in text


def test_extraction_wrapper_exposes_cleanup_flags():
    root = Path(__file__).resolve().parents[1]
    text = (root / "job_scripts" / "phase4" / "run_phase4_extract_andes.sh").read_text()

    assert "PHASE4_CLEAN_PARTIAL" in text
    assert "PHASE4_CLEAN_INCOMPLETE" in text
    assert "PHASE4_CLEAN_STALE_LOCK" in text
    assert "EXTRACT_FLAGS+=(--clean-partial)" in text
    assert "EXTRACT_FLAGS+=(--clean-incomplete)" in text
    assert "EXTRACT_FLAGS+=(--clean-stale-lock)" in text


def test_batch_b_review_wrapper_is_cpu_batch_policy_compliant():
    root = Path(__file__).resolve().parents[1]
    path = root / "job_scripts" / "phase4" / "run_phase4_batch_b_representative_review_andes.sh"
    text = path.read_text()

    subprocess.run(("bash", "-n", str(path)), check=True)
    assert "#SBATCH -A AST207" in text
    assert "#SBATCH -p batch" in text
    assert "#SBATCH -o logs/" in text
    assert "#SBATCH -e logs/" in text
    assert "#SBATCH --cpus-per-task=" in text
    assert "--mail" not in text
    assert "sgs" not in text.lower()
    assert "sbatch " not in text
    assert "archive_slurm_resources" in text
    assert "sacct -j" in text
    assert "generate_phase4_batch_b_representative_review.py" in text
    for root_name in ("RELEASE_ROOT", "LEDGER_SUMMARY", "OUTPUT_DIR", "RUN_DIR"):
        assert f': "${{{root_name}:?' in text
