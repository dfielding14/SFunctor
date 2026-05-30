"""Regression tests for count-first legacy histogram reducers."""
from __future__ import annotations

import numpy as np
import pytest

from plotting_scripts.plot_structure_functions import (
    angular_average,
    build_anisotropic_masks,
    compute_anisotropic_s2,
)


def test_build_anisotropic_masks_honors_requested_wedge_widths():
    L, perpendicular, xi, lam = build_anisotropic_masks(8, 7, 3, 2)
    assert L.sum() == 3 * 7
    assert perpendicular.sum() == 3 * 7
    assert xi.sum() == 3 * 2
    assert lam.sum() == 3 * 2
    assert L[:3].all()
    assert perpendicular[-3:].all()
    assert xi[-3:, :2].all()
    assert lam[-3:, -2:].all()


@pytest.mark.parametrize("theta_width,phi_width", [(0, 1), (9, 1), (1, 0), (1, 8)])
def test_build_anisotropic_masks_rejects_invalid_widths(theta_width, phi_width):
    with pytest.raises(ValueError):
        build_anisotropic_masks(8, 7, theta_width, phi_width)


def test_angular_average_ignores_empty_nan_cells():
    s2 = np.array([[2.0, np.nan], [np.nan, 8.0]])
    counts = np.array([[3, 0], [0, 1]])
    assert angular_average(s2, counts, np.ones_like(s2, dtype=bool)) == pytest.approx(3.5)


def test_anisotropic_s2_retains_populated_cells_in_sparse_wedge():
    hist = np.zeros((1, 2, 4, 4, 2), dtype=np.int64)
    hist[0, 0, 0, 0, 0] = 3
    hist[0, 0, 1, 1, 1] = 1
    delta_edges = [np.array([1.0, 2.0, 4.0])]
    result = compute_anisotropic_s2(
        hist,
        delta_edges,
        np.linspace(0.0, np.pi / 2.0, 5),
        np.linspace(0.0, np.pi / 2.0, 5),
        theta_wedge_bins=2,
        phi_wedge_bins=2,
    )[0]
    # Geometric delta-bin centers are sqrt(2) and sqrt(8), so the weighted S2
    # is (3*2 + 1*8) / 4 = 3.5 rather than NaN from empty angular cells.
    assert result["L"][0] == pytest.approx(3.5)
