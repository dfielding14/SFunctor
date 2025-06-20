"""Shared-memory multiprocessing utilities for SF analysis.

This module avoids duplicate RAM usage by placing large read-only arrays in
`multiprocessing.shared_memory` so that each worker process maps the same
underlying pages.  All top-level functions are pickle-able so they can be used
with :pyclass:`multiprocessing.Pool`.
"""

from __future__ import annotations

import contextlib
from multiprocessing import Pool, cpu_count, shared_memory
from typing import Dict, Sequence, Tuple

import numpy as np

from .sf_histograms import (
    N_CHANNELS,
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
    compute_histogram_for_disp_2D,
)

__all__ = [
    "compute_histograms_shared",
]

# -----------------------------------------------------------------------------
# Shared arrays initialisation --------------------------------------------------
# -----------------------------------------------------------------------------

_GLOBAL_FIELDS: Dict[str, np.ndarray] = {}


def _init_worker(shm_meta: Dict[str, Tuple[str, Tuple[int, ...], str]]) -> None:
    """Pool initializer that attaches numpy views to shared-memory segments."""
    global _GLOBAL_FIELDS  # modify module-level dict
    for name, (shm_name, shape, dtype_str) in shm_meta.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _GLOBAL_FIELDS[name] = np.ndarray(
            shape, dtype=np.dtype(dtype_str), buffer=shm.buf
        )


# -----------------------------------------------------------------------------
# Worker function --------------------------------------------------------------
# -----------------------------------------------------------------------------


def _process_batch(
    batch_indices: Sequence[int],
    displacements: np.ndarray,
    axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
    stencil_width: int,
    n_ell_bins: int,
    n_theta_bins: int,
    n_phi_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram for a batch of displacement indices using globals."""
    vx = _GLOBAL_FIELDS["v_x"]
    vy = _GLOBAL_FIELDS["v_y"]
    vz = _GLOBAL_FIELDS["v_z"]
    bx = _GLOBAL_FIELDS["B_x"]
    by = _GLOBAL_FIELDS["B_y"]
    bz = _GLOBAL_FIELDS["B_z"]
    rho = _GLOBAL_FIELDS["rho"]

    vAx = _GLOBAL_FIELDS["vA_x"]
    vAy = _GLOBAL_FIELDS["vA_y"]
    vAz = _GLOBAL_FIELDS["vA_z"]

    zpx = _GLOBAL_FIELDS["zp_x"]
    zpy = _GLOBAL_FIELDS["zp_y"]
    zpz = _GLOBAL_FIELDS["zp_z"]

    zmx = _GLOBAL_FIELDS["zm_x"]
    zmy = _GLOBAL_FIELDS["zm_y"]
    zmz = _GLOBAL_FIELDS["zm_z"]

    omegax = _GLOBAL_FIELDS["omega_x"]
    omegay = _GLOBAL_FIELDS["omega_y"]
    omegaz = _GLOBAL_FIELDS["omega_z"]

    jx = _GLOBAL_FIELDS["j_x"]
    jy = _GLOBAL_FIELDS["j_y"]
    jz = _GLOBAL_FIELDS["j_z"]

    hist_mag = np.zeros(
        (
            N_MAG_CHANNELS,
            n_ell_bins,
            n_theta_bins,
            n_phi_bins,
            sf_bin_edges.shape[0] - 1,
        ),
        dtype=np.int64,
    )

    hist_other = np.zeros(
        (
            N_OTHER_CHANNELS,
            n_ell_bins,
            sf_bin_edges.shape[0] - 1,
        ),
        dtype=np.int64,
    )

    for idx in batch_indices:
        dx, dy = displacements[idx]
        hm_part, ho_part = compute_histogram_for_disp_2D(
            vx,
            vy,
            vz,
            bx,
            by,
            bz,
            rho,
            vAx,
            vAy,
            vAz,
            zpx,
            zpy,
            zpz,
            zmx,
            zmy,
            zmz,
            omegax,
            omegay,
            omegaz,
            jx,
            jy,
            jz,
            int(dx),
            int(dy),
            axis,
            N_random_subsamples,
            ell_bin_edges,
            theta_bin_edges,
            phi_bin_edges,
            sf_bin_edges,
            product_bin_edges,
            stencil_width,
        )

        hist_mag += hm_part
        hist_other += ho_part

    return hist_mag, hist_other


# -----------------------------------------------------------------------------
# Public helper ----------------------------------------------------------------
# -----------------------------------------------------------------------------


def compute_histograms_shared(
    fields: Dict[str, np.ndarray],
    displacements: np.ndarray,
    *,
    axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_bin_edges: np.ndarray,
    product_bin_edges: np.ndarray,
    stencil_width: int = 2,
    n_processes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histograms using shared-memory Pool.

    Parameters
    ----------
    fields
        Dict with keys "v_x", "v_y", "v_z", "B_x", "B_y", "B_z", "rho", "j_x", "j_y", "j_z".
    displacements
        Array *(N,2)* of integer offsets.
    axis
        Slice orientation (1,2,3).
    n_processes
        Worker count; defaults to *cpu_count() - 2*.
    """
    required = {
        "v_x",
        "v_y",
        "v_z",
        "B_x",
        "B_y",
        "B_z",
        "rho",
        "vA_x",
        "vA_y",
        "vA_z",
        "zp_x",
        "zp_y",
        "zp_z",
        "zm_x",
        "zm_y",
        "zm_z",
        "omega_x",
        "omega_y",
        "omega_z",
        "j_x",
        "j_y",
        "j_z",
    }
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"compute_histograms_shared missing fields: {missing}")

    n_processes = n_processes or max(1, cpu_count() - 2)

    # Create shared-memory segments ---------------------------------------
    shm_objects: Dict[str, shared_memory.SharedMemory] = {}
    shm_meta: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}
    try:
        for key in required:
            arr = np.ascontiguousarray(fields[key])
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[:] = arr  # copy once
            shm_objects[key] = shm
            shm_meta[key] = (shm.name, arr.shape, str(arr.dtype))

        # Prepare batching --------------------------------------------------
        batch_size = max(1, displacements.shape[0] // n_processes)
        batches = [
            range(i, min(i + batch_size, displacements.shape[0]))
            for i in range(0, displacements.shape[0], batch_size)
        ]

        n_ell_bins = ell_bin_edges.shape[0] - 1
        n_theta_bins = theta_bin_edges.shape[0] - 1
        n_phi_bins = phi_bin_edges.shape[0] - 1

        with Pool(
            processes=n_processes, initializer=_init_worker, initargs=(shm_meta,)
        ) as pool:
            results = pool.starmap(
                _process_batch,
                [
                    (
                        batch,
                        displacements,
                        axis,
                        N_random_subsamples,
                        ell_bin_edges,
                        theta_bin_edges,
                        phi_bin_edges,
                        sf_bin_edges,
                        product_bin_edges,
                        stencil_width,
                        n_ell_bins,
                        n_theta_bins,
                        n_phi_bins,
                    )
                    for batch in batches
                ],
            )

        # Aggregate results -------------------------------------------------
        hist_mag_total = np.zeros(
            (
                N_MAG_CHANNELS,
                n_ell_bins,
                n_theta_bins,
                n_phi_bins,
                sf_bin_edges.shape[0] - 1,
            ),
            dtype=np.int64,
        )

        hist_other_total = np.zeros(
            (
                N_OTHER_CHANNELS,
                n_ell_bins,
                sf_bin_edges.shape[0] - 1,
            ),
            dtype=np.int64,
        )

        for hm, ho in results:
            hist_mag_total += hm
            hist_other_total += ho

        return hist_mag_total, hist_other_total
    finally:
        # Cleanup shared memory -------------------------------------------
        for shm in shm_objects.values():
            with contextlib.suppress(FileNotFoundError):
                shm.close()
                shm.unlink()
