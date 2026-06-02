from __future__ import annotations

import numpy as np

from scripts.phase4 import run_phase4_batch_b_tail_diagnostic as diagnostic


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
