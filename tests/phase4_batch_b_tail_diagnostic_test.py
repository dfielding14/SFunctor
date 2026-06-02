from __future__ import annotations

import numpy as np

from scripts.phase4 import run_phase4_batch_b_tail_diagnostic as diagnostic
from sfunctor.core.directional import QField


def test_exterior_sampler_is_deterministic_and_excludes_interior() -> None:
    intrinsic = ((0, 8), (0, 8), (0, 8))
    interior = ((2, 6), (2, 6), (2, 6))

    left = diagnostic._sample_exterior_origins(intrinsic, interior, 128, 17)
    right = diagnostic._sample_exterior_origins(intrinsic, interior, 128, 17)

    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    assert len(left[0]) == 128
    assert not np.any(diagnostic._inside_bounds(left, interior))
    assert np.all(diagnostic._inside_bounds(left, intrinsic))


def test_concentration_reports_expected_shares() -> None:
    blocks = np.asarray([[6.0, 2.0], [3.0, 2.0], [1.0, 2.0]])

    one, five, ten = diagnostic._concentration(blocks)

    assert np.allclose(one, [0.6, 1.0 / 3.0])
    assert np.allclose(five, [1.0, 1.0])
    assert np.allclose(ten, [1.0, 1.0])


def test_selected_offsets_use_report_scale_bin_centers() -> None:
    _, edges, selected, _ = diagnostic._selected_offsets()
    ell = np.linalg.norm(selected.astype(float), axis=1)
    indices = np.searchsorted(edges, ell, side="right") - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    assert len(selected) == 898
    assert np.all(centers[indices] >= diagnostic.SCIENCE_SCALE_MINIMUM)


def test_direct_origin_partition_recomposes_exact_measurement() -> None:
    shape = (4, 4, 4)
    B = np.zeros((3, *shape))
    u = np.zeros_like(B)
    i = np.indices(shape)[2]
    B[0] = 0.1 * i
    B[2] = 1.0
    u[0] = i
    valid = np.ones(shape, dtype=bool)
    q_fields = {
        "B": QField("B", B, valid, "not applicable"),
        "u": QField("u", u, valid, "not applicable"),
    }
    displacement = (1, 0, 0)
    intrinsic = ((0, 4), (0, 4), (0, 3))
    interior = ((0, 4), (0, 4), (1, 2))
    origins = diagnostic._origin_arrays(intrinsic, None, 7)
    inside = diagnostic._inside_bounds(origins, interior)

    complete = diagnostic._measurement(B, q_fields, origins, displacement)
    left = diagnostic._measurement(
        B, q_fields, diagnostic._subset_origins(origins, inside), displacement
    )
    right = diagnostic._measurement(
        B, q_fields, diagnostic._subset_origins(origins, ~inside), displacement
    )

    for name in diagnostic.ACCUMULATOR_NAMES:
        assert np.allclose(complete[name], left[name] + right[name])


def test_stratum_weight_targets_equal_displacement_origin_count() -> None:
    interior = diagnostic._stratum_weight(2048, 60, 100, 2048)
    exterior = diagnostic._stratum_weight(2048, 40, 100, 2048)

    assert np.isclose(interior * 2048 + exterior * 2048, 2048)


def test_source_version_binds_directional_kernel_and_wrapper() -> None:
    hashes = diagnostic._source_version()["implementation_source_hashes"]

    assert "sfunctor/core/directional.py" in hashes
    assert "job_scripts/phase4/run_phase4_batch_b_tail_diagnostic_andes.sh" in hashes
