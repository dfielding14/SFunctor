"""Shared-memory multiprocessing utilities for SF analysis.

This module avoids duplicate RAM usage by placing large read-only arrays in
`multiprocessing.shared_memory` so that each worker process maps the same
underlying pages.  All top-level functions are pickle-able so they can be used
with :pyclass:`multiprocessing.Pool`.
"""
from __future__ import annotations

import contextlib
from multiprocessing import Pool, cpu_count, shared_memory
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from sfunctor.core.histograms import (
    compute_histogram_for_disp_2D,
    N_CENSOR_KINDS,
    N_CHANNELS,
)

__all__ = [
    "compute_histograms_shared",
]

# -----------------------------------------------------------------------------
# Shared arrays initialisation --------------------------------------------------
# -----------------------------------------------------------------------------

_GLOBAL_FIELDS: Dict[str, np.ndarray] = {}


def _init_worker(shm_meta: Dict[str, Tuple[str, Tuple[int, ...], str]]) -> None:
    """Pool initializer that attaches numpy views to shared-memory segments.

    This function is called once per worker process to set up access to
    shared memory arrays. It creates numpy array views that map to the
    same underlying memory, avoiding data duplication across processes.

    Parameters
    ----------
    shm_meta : dict
        Metadata dictionary mapping field names to tuples of:
        (shared_memory_name, array_shape, dtype_string)

    Notes
    -----
    Modifies the global _GLOBAL_FIELDS dictionary to store array references.
    These arrays are read-only views of the shared memory segments.
    """
    global _GLOBAL_FIELDS  # modify module-level dict
    for name, (shm_name, shape, dtype_str) in shm_meta.items():
        try:
            # Attach to existing shared memory
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            # Create array view - ensure it's read-only to prevent accidental modification
            arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
            arr.flags.writeable = False  # Make array read-only
            _GLOBAL_FIELDS[name] = arr
            # Store shared memory object to prevent garbage collection
            if not hasattr(_init_worker, '_shm_objects'):
                _init_worker._shm_objects = {}
            _init_worker._shm_objects[name] = shm
        except Exception as e:
            print(f"Error attaching to shared memory for {name}: {e}")
            raise


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
    delta_bin_edges: Sequence[np.ndarray],
    stencil_width: int,
    n_ell_bins: int,
    n_theta_bins: int,
    n_phi_bins: int,
    n_delta_bins: int,
    cell_sizes: Tuple[float, float, float],
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
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

    curvx = _GLOBAL_FIELDS["curv_x"]
    curvy = _GLOBAL_FIELDS["curv_y"]
    curvz = _GLOBAL_FIELDS["curv_z"]

    gradrhox = _GLOBAL_FIELDS["grad_rho_x"]
    gradrhoy = _GLOBAL_FIELDS["grad_rho_y"]
    gradrhoz = _GLOBAL_FIELDS["grad_rho_z"]

    hist = np.zeros(
        (
            N_CHANNELS,
            n_ell_bins,
            n_theta_bins,
            n_phi_bins,
            n_delta_bins,
        ),
        dtype=np.int64,
    )
    hist_censoring = np.zeros(
        (
            N_CHANNELS,
            n_ell_bins,
            n_theta_bins,
            n_phi_bins,
            N_CENSOR_KINDS,
        ),
        dtype=np.int64,
    )

    for idx in batch_indices:
        dx, dy = displacements[idx]
        ell_idx = _ell_bin_index(_physical_ell(int(dx), int(dy), axis, cell_sizes), ell_bin_edges)
        if ell_idx < 0:
            continue
        hist_part, censoring_part = compute_histogram_for_disp_2D(
            vx,
            vy,
            vz,
            bx,
            by,
            bz,
            rho,
            vAx, vAy, vAz,
            zpx, zpy, zpz,
            zmx, zmy, zmz,
            omegax, omegay, omegaz,
            jx, jy, jz,
            curvx, curvy, curvz,
            gradrhox, gradrhoy, gradrhoz,
            int(dx),
            int(dy),
            axis,
            N_random_subsamples,
            ell_bin_edges,
            theta_bin_edges,
            phi_bin_edges,
            delta_bin_edges,
            stencil_width,
            cell_sizes,
            _seed_for_displacement(random_seed, int(dx), int(dy), axis, stencil_width),
            True,
            True,
        )

        hist[:, ell_idx] += hist_part[:, 0]
        hist_censoring[:, ell_idx] += censoring_part[:, 0]

    return hist, hist_censoring


def _process_batch_args(args) -> tuple[np.ndarray, np.ndarray]:
    """Pool-compatible wrapper for streaming unordered batch reduction."""

    return _process_batch(*args)


def _physical_ell(delta_i: int, delta_j: int, axis: int, cell_sizes: Tuple[float, float, float]) -> float:
    """Return physical separation for a slice-native KJI offset."""

    if axis == 1:
        components = (delta_i * cell_sizes[1], delta_j * cell_sizes[2])
    elif axis == 2:
        components = (delta_i * cell_sizes[0], delta_j * cell_sizes[2])
    else:
        components = (delta_i * cell_sizes[0], delta_j * cell_sizes[1])
    return float(np.hypot(*components))


def _ell_bin_index(value: float, edges: np.ndarray) -> int:
    if value == edges[-1]:
        return len(edges) - 2
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return index if 0 <= index < len(edges) - 1 else -1


def _seed_for_displacement(base_seed: int | None, delta_i: int, delta_j: int, axis: int, stencil_width: int) -> int | None:
    """Derive a stable per-offset seed independent of worker scheduling."""

    if base_seed is None:
        return None
    seed = int(base_seed) & 0xFFFFFFFF
    seed ^= (int(delta_i) * 0x9E3779B1) & 0xFFFFFFFF
    seed ^= (int(delta_j) * 0x85EBCA77) & 0xFFFFFFFF
    seed ^= (int(axis) * 0xC2B2AE3D) & 0xFFFFFFFF
    seed ^= (int(stencil_width) * 0x27D4EB2F) & 0xFFFFFFFF
    return seed & 0x7FFFFFFF


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
    delta_bin_edges: Union[Sequence[np.ndarray], np.ndarray],
    stencil_width: int = 2,
    n_processes: int | None = None,
    cell_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    random_seed: int | None = None,
    return_censoring: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute unified histograms using a shared-memory process pool.

    ``cell_sizes`` are effective Cartesian spacings for the already-loaded
    slice.  ``random_seed`` makes spatial Monte Carlo schedules invariant to
    worker count and displacement ordering.
    """
    if axis not in (1, 2, 3):
        raise ValueError("axis must be 1, 2, or 3")

    required = {
        "v_x", "v_y", "v_z",
        "B_x", "B_y", "B_z",
        "rho",
        "vA_x", "vA_y", "vA_z",
        "zp_x", "zp_y", "zp_z",
        "zm_x", "zm_y", "zm_z",
        "omega_x", "omega_y", "omega_z",
        "j_x", "j_y", "j_z",
        "curv_x", "curv_y", "curv_z",
        "grad_rho_x", "grad_rho_y", "grad_rho_z",
    }
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"compute_histograms_shared missing fields: {missing}")

    if not isinstance(N_random_subsamples, int) or N_random_subsamples <= 0:
        raise ValueError("N_random_subsamples must be a positive integer")
    displacements = np.asarray(displacements)
    if displacements.ndim != 2 or displacements.shape[1] != 2:
        raise ValueError("displacements must have shape (n, 2)")
    if np.any(np.all(displacements == 0, axis=1)):
        raise ValueError("zero displacement is not a valid structure-function offset")
    cell_sizes = tuple(float(value) for value in cell_sizes)
    if len(cell_sizes) != 3 or not np.all(np.isfinite(cell_sizes)) or np.any(np.asarray(cell_sizes) <= 0.0):
        raise ValueError("cell_sizes must contain three finite positive Cartesian spacings")
    if n_processes is None or n_processes == 0:
        n_processes = max(1, cpu_count() - 2)
    if n_processes < 1:
        raise ValueError("n_processes must be positive or zero for auto-detection")

    reference_shape = fields["rho"].shape
    if len(reference_shape) != 2:
        raise ValueError("fields must contain 2-D slice arrays")
    for name in required:
        if fields[name].shape != reference_shape:
            raise ValueError(f"field {name} has shape {fields[name].shape}, expected {reference_shape}")

    for name, edges in (
        ("ell_bin_edges", ell_bin_edges),
        ("theta_bin_edges", theta_bin_edges),
        ("phi_bin_edges", phi_bin_edges),
    ):
        edges = np.asarray(edges)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
            raise ValueError(f"{name} must be finite and strictly increasing")

    # Normalize Δ bin edges to a list-of-arrays format.
    if isinstance(delta_bin_edges, np.ndarray):
        delta_bins_prepped = [np.ascontiguousarray(delta_bin_edges)] * N_CHANNELS
    else:
        delta_bins_prepped = [np.ascontiguousarray(arr) for arr in delta_bin_edges]
    if len(delta_bins_prepped) != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} Δ bin arrays, got {len(delta_bins_prepped)}"
        )
    n_delta_bins = delta_bins_prepped[0].shape[0] - 1
    for edges in delta_bins_prepped:
        if edges.ndim != 1 or edges.shape[0] != n_delta_bins + 1:
            raise ValueError("all delta_bin_edges arrays must have the same one-dimensional shape")
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
            raise ValueError("delta_bin_edges arrays must be finite and strictly increasing")

    # Special case: single process execution without multiprocessing overhead
    if n_processes == 1:
        hist_total = np.zeros(
            (
                N_CHANNELS,
                ell_bin_edges.shape[0] - 1,
                theta_bin_edges.shape[0] - 1,
                phi_bin_edges.shape[0] - 1,
                n_delta_bins,
            ),
            dtype=np.int64,
        )
        hist_censoring_total = np.zeros(
            (
                N_CHANNELS,
                ell_bin_edges.shape[0] - 1,
                theta_bin_edges.shape[0] - 1,
                phi_bin_edges.shape[0] - 1,
                N_CENSOR_KINDS,
            ),
            dtype=np.int64,
        )

        for idx in range(displacements.shape[0]):
            dx, dy = displacements[idx]
            ell_idx = _ell_bin_index(_physical_ell(int(dx), int(dy), axis, cell_sizes), ell_bin_edges)
            if ell_idx < 0:
                continue
            hist_part, censoring_part = compute_histogram_for_disp_2D(
                fields["v_x"], fields["v_y"], fields["v_z"],
                fields["B_x"], fields["B_y"], fields["B_z"],
                fields["rho"],
                fields["vA_x"], fields["vA_y"], fields["vA_z"],
                fields["zp_x"], fields["zp_y"], fields["zp_z"],
                fields["zm_x"], fields["zm_y"], fields["zm_z"],
                fields["omega_x"], fields["omega_y"], fields["omega_z"],
                fields["j_x"], fields["j_y"], fields["j_z"],
                fields["curv_x"], fields["curv_y"], fields["curv_z"],
                fields["grad_rho_x"], fields["grad_rho_y"], fields["grad_rho_z"],
            int(dx), int(dy), axis,
            N_random_subsamples,
            ell_bin_edges, theta_bin_edges, phi_bin_edges,
            tuple(delta_bins_prepped),
            stencil_width,
            cell_sizes,
            _seed_for_displacement(random_seed, int(dx), int(dy), axis, stencil_width),
            True,
            True,
        )
            hist_total[:, ell_idx] += hist_part[:, 0]
            hist_censoring_total[:, ell_idx] += censoring_part[:, 0]

        return (hist_total, hist_censoring_total) if return_censoring else hist_total

    # Create shared-memory segments ---------------------------------------
    shm_objects: Dict[str, shared_memory.SharedMemory] = {}
    shm_meta: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}
    try:
        import time
        required_for_shm = required.copy()
        for key in required_for_shm:
            arr = np.ascontiguousarray(fields[key])
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            np.copyto(shm_arr, arr, casting='no')
            if not np.array_equal(shm_arr, arr, equal_nan=True):
                raise RuntimeError(f"Failed to copy {key} to shared memory")
            shm_objects[key] = shm
            shm_meta[key] = (shm.name, arr.shape, str(arr.dtype))

        time.sleep(0.01)

        # Prepare batching --------------------------------------------------
        batch_size = max(1, displacements.shape[0] // n_processes)
        batches = [
            range(i, min(i + batch_size, displacements.shape[0]))
            for i in range(0, displacements.shape[0], batch_size)
        ]

        n_ell_bins = ell_bin_edges.shape[0] - 1
        n_theta_bins = theta_bin_edges.shape[0] - 1
        n_phi_bins = phi_bin_edges.shape[0] - 1

        work_items = [
                    (
                        batch,
                        displacements,
                        axis,
                        N_random_subsamples,
                        ell_bin_edges,
                        theta_bin_edges,
                        phi_bin_edges,
                        delta_bins_prepped,
                        stencil_width,
                        n_ell_bins,
                        n_theta_bins,
                        n_phi_bins,
                        n_delta_bins,
                        cell_sizes,
                        random_seed,
                    )
                    for batch in batches
                ]

        # Aggregate results as batches complete so the parent does not retain
        # one dense histogram per worker until the final reduction.
        hist_total = np.zeros(
            (
                N_CHANNELS,
                n_ell_bins,
                n_theta_bins,
                n_phi_bins,
                n_delta_bins,
            ),
            dtype=np.int64,
        )
        hist_censoring_total = np.zeros(
            (
                N_CHANNELS,
                n_ell_bins,
                n_theta_bins,
                n_phi_bins,
                N_CENSOR_KINDS,
            ),
            dtype=np.int64,
        )

        with Pool(processes=n_processes, initializer=_init_worker, initargs=(shm_meta,)) as pool:
            for hist_part, censoring_part in pool.imap_unordered(_process_batch_args, work_items):
                hist_total += hist_part
                hist_censoring_total += censoring_part

        return (hist_total, hist_censoring_total) if return_censoring else hist_total
    finally:
        # Cleanup shared memory -------------------------------------------
        for shm in shm_objects.values():
            with contextlib.suppress(FileNotFoundError):
                shm.close()
                shm.unlink()
