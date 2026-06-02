"""Focused tests for the guarded Phase 4 Batch B all-21 extension."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase4 import run_phase4_batch_b_all21_extension as all21
from scripts.phase4 import run_phase4_extraction as extraction


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "job_scripts" / "phase4" / "run_phase4_batch_b_all21_extension_andes.sh"


def test_all21_go_artifact_freezes_exact_authorized_scope() -> None:
    payload = json.loads((REPO_ROOT / all21.DECISION_RELATIVE_PATH).read_text())

    assert payload["status"] == "post_diagnostic_user_authorized_all21_batch_b_extension"
    assert payload["authorization_source"] == "explicit_user_request"
    assert tuple(payload["cube_ids"]) == extraction.PHASE4_PILOT_CUBE_IDS
    assert tuple(payload["q_names"]) == ("B", "u")
    assert tuple(payload["p_values"]) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert payload["stencils"] == {
        "2": {
            "label": "2-point",
            "ell_max": 320,
            "bin_count": 64,
            "directions_per_bin": 24,
        }
    }
    assert tuple(payload["support_modes"]) == ("all_valid_origins", "shell_local")
    assert tuple(payload["density_conventions"]) == ("not applicable", "not applicable")
    assert payload["all21_batch_b_expansion_authorized"] is True
    for forbidden_expansion in (
        "density_weighting_authorized",
        "five_point_expansion_authorized",
        "fitted_directional_exponents_authorized",
        "batch_c_authorized",
        "sgs_channels_authorized",
        "lsub1280_authorized",
    ):
        assert payload[forbidden_expansion] is False


def test_all21_configures_exact_matrix_and_restores_inherited_module(tmp_path: Path) -> None:
    original = (
        all21.phase3.BENCHMARK_CUBE_IDS,
        all21.inherited.BENCHMARK_CUBE_IDS,
        all21.inherited.Q_NAMES,
        all21.inherited.P_VALUES,
        all21.inherited.DENSITY_CONVENTIONS,
        all21.inherited.STENCIL_SPECS,
        all21.inherited.SUPPORT_MODES,
        all21.inherited._source_version,
        all21.inherited._phase2_source_identity,
        all21._configured_tail_diagnostic_root,
    )

    with all21._configured_runner(
        tmp_path / "batch_a",
        tmp_path / "representative_a2",
        tmp_path / "all21_3point",
        tmp_path / "representative_batch_b",
        tmp_path / "tail_diagnostic",
    ):
        configuration = all21.inherited._campaign_configuration()
        assert all21.PHASE4_BATCH_B_ALL21_CUBE_IDS == extraction.PHASE4_PILOT_CUBE_IDS
        assert all21.inherited.BENCHMARK_CUBE_IDS == extraction.PHASE4_PILOT_CUBE_IDS
        assert all21.phase3.BENCHMARK_CUBE_IDS == extraction.PHASE4_PILOT_CUBE_IDS
        assert configuration["q_names"] == ("B", "u")
        assert configuration["p_values"] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert configuration["density_conventions"] == ("not applicable", "not applicable")
        assert configuration["support_modes"] == ("all_valid_origins", "shell_local")
        assert configuration["stencils"] == {
            2: {
                "label": "2-point",
                "ell_max": 320,
                "bin_count": 64,
                "directions_per_bin": 24,
            }
        }

    assert (
        all21.phase3.BENCHMARK_CUBE_IDS,
        all21.inherited.BENCHMARK_CUBE_IDS,
        all21.inherited.Q_NAMES,
        all21.inherited.P_VALUES,
        all21.inherited.DENSITY_CONVENTIONS,
        all21.inherited.STENCIL_SPECS,
        all21.inherited.SUPPORT_MODES,
        all21.inherited._source_version,
        all21.inherited._phase2_source_identity,
        all21._configured_tail_diagnostic_root,
    ) == original


def test_all21_restores_inherited_module_after_exception(tmp_path: Path) -> None:
    original = (
        all21.phase3.BENCHMARK_CUBE_IDS,
        all21.inherited.BENCHMARK_CUBE_IDS,
        all21.inherited.Q_NAMES,
        all21.inherited.P_VALUES,
        all21.inherited.DENSITY_CONVENTIONS,
        all21.inherited.STENCIL_SPECS,
        all21.inherited.SUPPORT_MODES,
    )

    with pytest.raises(RuntimeError, match="synthetic forced exit"):
        with all21._configured_runner(
            tmp_path / "batch_a",
            tmp_path / "representative_a2",
            tmp_path / "all21_3point",
            tmp_path / "representative_batch_b",
            tmp_path / "tail_diagnostic",
        ):
            raise RuntimeError("synthetic forced exit")

    assert (
        all21.phase3.BENCHMARK_CUBE_IDS,
        all21.inherited.BENCHMARK_CUBE_IDS,
        all21.inherited.Q_NAMES,
        all21.inherited.P_VALUES,
        all21.inherited.DENSITY_CONVENTIONS,
        all21.inherited.STENCIL_SPECS,
        all21.inherited.SUPPORT_MODES,
    ) == original


def test_all21_decision_rejects_scope_expansion_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = REPO_ROOT / all21.DECISION_RELATIVE_PATH
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(source_path.read_text())
    monkeypatch.setattr(all21, "DECISION_RELATIVE_PATH", str(decision_path))

    assert all21._decision_identity()["decision_sha256"] == all21.file_sha256(decision_path)
    decision = json.loads(decision_path.read_text())
    decision["five_point_expansion_authorized"] = True
    all21.inherited._atomic_write_json(decision_path, decision)

    with pytest.raises(RuntimeError, match="invalid Phase 4 Batch B all-21 GO decision"):
        all21._decision_identity()


def test_all21_review_evidence_rejects_authorization_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = REPO_ROOT / all21.REPRESENTATIVE_REVIEW_DECISION_RELATIVE_PATH
    decision_path = tmp_path / "representative_review_decision.json"
    decision_path.write_text(source_path.read_text())
    monkeypatch.setattr(
        all21,
        "REPRESENTATIVE_REVIEW_DECISION_RELATIVE_PATH",
        str(decision_path),
    )

    assert all21._representative_review_identity()["review_decision_sha256"] == (
        all21.file_sha256(decision_path)
    )
    decision = json.loads(decision_path.read_text())
    decision["all21_batch_b_expansion_authorized"] = True
    all21.inherited._atomic_write_json(decision_path, decision)

    with pytest.raises(RuntimeError, match="invalid Phase 4 Batch B representative review"):
        all21._representative_review_identity()


def test_all21_rejects_cube_outside_exact_frozen_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(all21, "_configured_batch_a_root", tmp_path / "batch_a")
    monkeypatch.setattr(all21, "_configured_representative_a2_root", tmp_path / "representative_a2")
    monkeypatch.setattr(
        all21,
        "_configured_all21_3point_extension_root",
        tmp_path / "all21_3point",
    )
    monkeypatch.setattr(
        all21,
        "_configured_representative_batch_b_root",
        tmp_path / "representative_batch_b",
    )
    monkeypatch.setattr(
        all21,
        "_configured_tail_diagnostic_root",
        tmp_path / "tail_diagnostic",
    )

    with pytest.raises(ValueError, match="rejects unapproved cube ID"):
        all21._phase2_source_identity(tmp_path, "L640_unapproved")


def test_all21_requires_exact_batch_a_source_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        all21.representative_batch_b,
        "_batch_a_reference_sources",
        lambda phase2_root, batch_a_root: (
            {cube_id: {} for cube_id in all21.PHASE4_BATCH_B_ALL21_CUBE_IDS[:-1]},
            {"batch_a": "bound"},
        ),
    )

    with pytest.raises(RuntimeError, match="exact frozen all-21 set"):
        all21._batch_a_reference_sources(Path("/extract"), Path("/batch_a"))


def test_all21_source_version_binds_adapter_wrapper_and_decisions() -> None:
    hashes = all21._source_version()["implementation_source_hashes"]

    assert "scripts/phase4/run_phase4_batch_b_all21_extension.py" in hashes
    assert "job_scripts/phase4/run_phase4_batch_b_all21_extension_andes.sh" in hashes
    assert all21.DECISION_RELATIVE_PATH in hashes
    assert all21.DECISION_REPORT_RELATIVE_PATH in hashes
    assert all21.REPRESENTATIVE_REVIEW_DECISION_RELATIVE_PATH in hashes
    assert all21.REPRESENTATIVE_STATUS_RELATIVE_PATH in hashes
    assert all21.REPRESENTATIVE_REVIEW_SUMMARY_RELATIVE_PATH in hashes
    assert all21.REPRESENTATIVE_FIGURE_MANIFEST_RELATIVE_PATH in hashes


def test_all21_tail_diagnostic_identity_rejects_marker_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path(all21.TAIL_DIAGNOSTIC_RELEASE)
    for name in (
        all21.TAIL_DIAGNOSTIC_SUMMARY_FILENAME,
        all21.TAIL_DIAGNOSTIC_MARKER_FILENAME,
    ):
        (tmp_path / name).write_text((source_root / name).read_text())
    monkeypatch.setattr(all21, "TAIL_DIAGNOSTIC_RELEASE", str(tmp_path))

    assert all21._tail_diagnostic_identity()["tail_diagnostic_root"] == str(tmp_path.resolve())
    marker_path = tmp_path / all21.TAIL_DIAGNOSTIC_MARKER_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["status"] = "mutated"
    all21.inherited._atomic_write_json(marker_path, marker)

    with pytest.raises(RuntimeError, match="invalid or stale Phase 4 Batch B tail diagnostic"):
        all21._tail_diagnostic_identity()


def test_all21_wrapper_has_static_staged_lock_and_no_email_policy() -> None:
    text = WRAPPER.read_text()

    assert "#SBATCH -o logs/" in text
    assert "#SBATCH -e logs/" in text
    assert "${RUN_DIR}/logs/${ACTION}.log" in text
    assert "${RUN_DIR}/logs/work.log" in text
    assert "phase4_batch_b_all21_extension_action_lock" in text
    assert "recover_stale_lock" in text
    assert "plan|reduce|verify|summarize" in text
    assert "--representative-batch-b-root" in text
    assert "--tail-diagnostic-root" in text
    assert "mail" not in text.lower()
