from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.phase4 import generate_phase4_batch_b_representative_review as review


def _group() -> SimpleNamespace:
    block_counts = np.asarray(
        [
            [[2, 2]],
            [[2, 2]],
            [[2, 2]],
        ],
        dtype=np.int64,
    )
    block_sums = np.asarray(
        [
            [[6.0, 2.0]],
            [[3.0, 2.0]],
            [[1.0, 2.0]],
        ]
    )
    result = SimpleNamespace(
        block_counts=block_counts,
        block_sums=block_sums,
        block_shape_kji=(80, 80, 80),
        block_assignment="stencil_midpoint",
        sums=block_sums.sum(axis=0),
    )
    return SimpleNamespace(result=result)


def test_moment_concentration_reports_largest_block_shares() -> None:
    concentration = review._moment_concentration(_group().result, (0,))

    assert concentration["largest_block_fraction"] == pytest.approx([0.6, 1.0 / 3.0])
    assert concentration["largest_5_blocks_fraction"] == pytest.approx([1.0, 1.0])
    assert concentration["largest_10_blocks_fraction"] == pytest.approx([1.0, 1.0])


def test_coupled_ratio_bootstrap_is_exact_for_identical_products() -> None:
    group = _group()

    bootstrap = review._coupled_ratio_bootstrap(group, group, (0,))

    assert bootstrap["valid_resamples"].tolist() == [
        review.COUPLED_RATIO_BOOTSTRAP_N_RESAMPLES,
        review.COUPLED_RATIO_BOOTSTRAP_N_RESAMPLES,
    ]
    assert bootstrap["interval_low"] == pytest.approx([1.0, 1.0])
    assert bootstrap["median"] == pytest.approx([1.0, 1.0])
    assert bootstrap["interval_high"] == pytest.approx([1.0, 1.0])
