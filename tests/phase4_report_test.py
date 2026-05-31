"""Focused checks for the Phase 4 Batch A historical report verifier."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.phase4 import generate_phase4_batch_a_status_figures as report


class StopAfterInventory(RuntimeError):
    """Expected test sentinel after the cube-source inventory gate."""


def _check_inventory(monkeypatch: pytest.MonkeyPatch, sources: Mapping[str, object]) -> None:
    monkeypatch.setattr(
        report,
        "_cube_publication_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterInventory()),
    )
    with pytest.raises(StopAfterInventory):
        report._verify_extraction_sources(
            Path("/extract"),
            {"phase2_sources": sources},
            plan={
                "selections": [
                    {"cube_id": cube_id}
                    for cube_id in report.FROZEN_PHASE4_PILOT_CUBE_IDS
                ]
            },
            plan_identity={},
            input_hashes=report.InputHashes(),
        )


def test_frozen_cube_source_inventory_accepts_json_key_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _check_inventory(
        monkeypatch,
        {
            cube_id: {
                "cube_id": cube_id,
                "phase2_root": "/extract",
                "phase4_extraction_plan": {},
            }
            for cube_id in reversed(report.FROZEN_PHASE4_PILOT_CUBE_IDS)
        },
    )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_frozen_cube_source_inventory_rejects_membership_changes(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    sources = {cube_id: {} for cube_id in report.FROZEN_PHASE4_PILOT_CUBE_IDS}
    if mutation == "missing":
        sources.pop(report.FROZEN_PHASE4_PILOT_CUBE_IDS[0])
    else:
        sources["unexpected"] = {}
    with pytest.raises(
        RuntimeError, match="campaign extraction-source inventory differs from the frozen pilot"
    ):
        report._verify_extraction_sources(
            Path("/extract"),
            {"phase2_sources": sources},
            plan={
                "selections": [
                    {"cube_id": cube_id}
                    for cube_id in report.FROZEN_PHASE4_PILOT_CUBE_IDS
                ]
            },
            plan_identity={},
            input_hashes=report.InputHashes(),
        )


def test_cbin_source_containment_accepts_prefixed_trusted_subtree(tmp_path: Path) -> None:
    path = tmp_path / "cbin_mhd_u_bcc_40" / "rank_00000001" / "sample.cbin"

    assert report._contained_cbin_source(tmp_path, path) == path.resolve()


@pytest.mark.parametrize("relative_path", ("raw/sample.cbin", "../outside/sample.cbin"))
def test_cbin_source_containment_rejects_non_cbin_or_escaped_subtree(
    tmp_path: Path,
    relative_path: str,
) -> None:
    with pytest.raises(RuntimeError, match="cbin source"):
        report._contained_cbin_source(tmp_path, tmp_path / relative_path)


def test_result_metadata_accepts_full_sampler_axis_for_selected_direction_plots() -> None:
    offsets = np.asarray(((1, 0, 0), (0, 1, 0)), dtype=np.int64)
    result = SimpleNamespace(
        displacements_ijk=offsets[np.lexsort((offsets[:, 2], offsets[:, 1], offsets[:, 0]))],
        support_displacements_sha256="a" * 64,
        support_displacement_count=2,
        stencil_width=2,
        pair_mode="all_valid_origins",
        sample_count=2048,
        pair_batch_size=1024,
        seed=20260530,
        block_shape_kji=(80, 80, 80),
        block_assignment="stencil_midpoint",
        cube_shape_kji=(640, 640, 640),
        q_names=("B", "u"),
        direction_names=("all", "parallel", "perpendicular", "xi", "lambda"),
        p_values=(2.0,),
    )

    report._verify_result_metadata(
        result,
        expected_offsets=offsets,
        metadata={"offsets_sha256": "a" * 64, "realized_offset_count": 2},
        row={"stencil_width": 2, "support_mode": "all_valid_origins"},
        configuration={
            "sample_count_per_displacement": 2048,
            "pair_batch_size": 1024,
            "production_seed": 20260530,
            "block_shape_kji": [80, 80, 80],
            "block_assignment": "stencil_midpoint",
        },
        label="synthetic",
    )


@pytest.mark.parametrize("mutation", ("dirty", "commit"))
def test_historical_identity_requires_clean_frozen_commit(mutation: str) -> None:
    hashes = {"source.py": "a" * 64}
    identity = {
        "commit": report.FROZEN_BATCH_A_SOURCE_COMMIT,
        "dirty": False,
        "implementation_source_hashes": hashes,
        "implementation_sha256": report._mapping_sha256(hashes),
    }
    if mutation == "dirty":
        identity["dirty"] = True
    else:
        identity["commit"] = "not-the-frozen-commit"

    with pytest.raises(RuntimeError, match="incoherent historical implementation identity"):
        report._historical_implementation_sha256(identity, label="synthetic")


def test_curve_uncertainty_support_mask_rejects_sparse_directional_rows() -> None:
    shape = (3,)
    index = (0,)
    group = SimpleNamespace(
        result=SimpleNamespace(
            moments=np.asarray([[1.0, 1.0, 1.0]]),
            counts=np.asarray([[64, 4, 64]]),
        ),
        uncertainty={
            "accepted_contributing_blocks": np.asarray([[16, 2, 16]]),
            "accepted_effective_blocks": np.asarray([[12.0, 1.6, 12.0]]),
            "valid_bootstrap_resamples": np.asarray([[200, 170, 200]]),
            "block_bootstrap_interval_low": np.full((1, *shape), 0.8),
            "block_bootstrap_interval_high": np.full((1, *shape), 1.2),
        },
    )

    assert report._curve_uncertainty_support_mask(group, index).tolist() == [
        True,
        False,
        True,
    ]
