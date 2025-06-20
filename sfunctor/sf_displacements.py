"""Utility functions for generating displacement vectors used in SF histograms."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "find_ell_bin_edges",
    "build_displacement_list",
]


def find_ell_bin_edges(r_min: float, r_max: float, n_ell_bins: int) -> np.ndarray:
    """Return logarithmically spaced *integer* ℓ-bin edges (length = n_ell_bins+1)."""
    n_points_low = n_ell_bins + 1
    n_points_high = 3 * n_ell_bins

    while n_points_low <= n_points_high:
        mid = (n_points_low + n_points_high) // 2
        edges = np.unique(np.around(np.geomspace(r_min, r_max, mid)).astype(int))
        if len(edges) < n_ell_bins + 1:
            n_points_low = mid + 1
        elif len(edges) > n_ell_bins + 1:
            n_points_high = mid - 1
        else:
            break
    if len(edges) != n_ell_bins + 1:
        print(
            f"[sf_displacements] Warning: requested {n_ell_bins+1} edges, got {len(edges)}."
        )
    return edges.astype(float)


def build_displacement_list(
    ell_bin_edges: np.ndarray,
    n_disp_total: int,
) -> np.ndarray:
    """Generate displacement vectors grouped per ℓ-bin then de-duplicated.

    Parameters
    ----------
    ell_bin_edges
        Output from :func:`find_ell_bin_edges`.
    n_disp_total
        Total number of displacement vectors desired (before mirror removal).

    Returns
    -------
    np.ndarray
        Array of shape *(N, 2)* with integer *(dx, dy)* displacements sorted by |Δ|.
    """
    n_ell_bins = len(ell_bin_edges) - 1
    n_per_bin = n_disp_total // n_ell_bins

    disp_list = []
    for i in range(n_ell_bins):
        r_low = ell_bin_edges[i]
        r_high = ell_bin_edges[i + 1]
        r_values = np.geomspace(r_low, r_high, n_per_bin)
        angles = np.random.uniform(0, np.pi, n_per_bin)
        dx = np.rint(r_values * np.cos(angles)).astype(np.int32)
        dy = np.rint(r_values * np.sin(angles)).astype(np.int32)
        disp_bin = np.unique(np.column_stack((dx, dy)), axis=0)
        disp_list.append(disp_bin)

    displacements = np.vstack(disp_list)

    # Remove mirror duplicates: (dx,dy) and (-dx,-dy)
    mask = (displacements[:, 0] < 0) | (
        (displacements[:, 0] == 0) & (displacements[:, 1] < 0)
    )
    canonical = np.where(mask[:, None], -displacements, displacements)
    unique_canonical = np.unique(canonical, axis=0)

    # Sort by absolute length
    idx_sort = np.argsort(
        np.sqrt(unique_canonical[:, 0] ** 2 + unique_canonical[:, 1] ** 2)
    )
    return unique_canonical[idx_sort]
