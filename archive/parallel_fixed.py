"""Fixed version of parallel.py that handles channel-specific bin edges correctly."""

from __future__ import annotations

import contextlib
from multiprocessing import Pool, cpu_count, shared_memory
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sfunctor.core.histograms import (
    compute_histogram_for_disp_2D,
    N_CHANNELS,
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
)

# -----------------------------------------------------------------------------
# Shared arrays initialisation
# -----------------------------------------------------------------------------

_GLOBAL_FIELDS: Dict[str, np.ndarray] = {}


def _init_worker(shm_meta: Dict[str, Tuple[str, Tuple[int, ...], str]]) -> None:
    """Pool initializer that attaches numpy views to shared-memory segments."""
    global _GLOBAL_FIELDS
    for name, (shm_name, shape, dtype_str) in shm_meta.items():
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
            arr.flags.writeable = False
            _GLOBAL_FIELDS[name] = arr
            if not hasattr(_init_worker, '_shm_objects'):
                _init_worker._shm_objects = {}
            _init_worker._shm_objects[name] = shm
        except Exception as e:
            print(f"Error attaching to shared memory for {name}: {e}")
            raise


# -----------------------------------------------------------------------------
# Worker function
# -----------------------------------------------------------------------------

def _process_batch(
    batch_indices: Sequence[int],
    displacements: np.ndarray,
    axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_channel_bin_edges: List[np.ndarray],  # FIXED: Now a list
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
    
    curvx = _GLOBAL_FIELDS["curv_x"]
    curvy = _GLOBAL_FIELDS["curv_y"]
    curvz = _GLOBAL_FIELDS["curv_z"]
    
    gradrhox = _GLOBAL_FIELDS["grad_rho_x"]
    gradrhoy = _GLOBAL_FIELDS["grad_rho_y"]
    gradrhoz = _GLOBAL_FIELDS["grad_rho_z"]

    hist_mag = np.zeros(
        (
            N_MAG_CHANNELS,
            n_ell_bins,
            n_theta_bins,
            n_phi_bins,
            sf_channel_bin_edges[0].shape[0] - 1,  # FIXED: Use list indexing
        ),
        dtype=np.int64,
    )

    hist_other = np.zeros(
        (
            N_OTHER_CHANNELS,
            n_ell_bins,
            product_bin_edges.shape[0] - 1,
        ),
        dtype=np.int64,
    )

    for idx in batch_indices:
        dx, dy = displacements[idx]
        hm_part, ho_part = compute_histogram_for_disp_2D(
            vx, vy, vz, bx, by, bz, rho,
            vAx, vAy, vAz, zpx, zpy, zpz,
            zmx, zmy, zmz, omegax, omegay, omegaz,
            jx, jy, jz, curvx, curvy, curvz,
            gradrhox, gradrhoy, gradrhoz,
            int(dx), int(dy), axis,
            N_random_subsamples,
            ell_bin_edges, theta_bin_edges, phi_bin_edges,
            sf_channel_bin_edges,  # FIXED: Pass as list
            product_bin_edges,
            stencil_width,
        )

        hist_mag += hm_part
        hist_other += ho_part

    return hist_mag, hist_other


# -----------------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------------

def compute_histograms_shared_fixed(
    fields: Dict[str, np.ndarray],
    displacements: np.ndarray,
    *,
    axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_channel_bin_edges: List[np.ndarray],  # FIXED: Now accepts list
    product_bin_edges: np.ndarray,
    stencil_width: int = 2,
    n_processes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed version that handles channel-specific bin edges correctly."""
    
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

    n_processes = n_processes or max(1, cpu_count() - 2)
    
    # Special case: single process execution without shared memory
    if n_processes == 1:
        hist_mag_total = np.zeros(
            (
                N_MAG_CHANNELS,
                ell_bin_edges.shape[0] - 1,
                theta_bin_edges.shape[0] - 1,
                phi_bin_edges.shape[0] - 1,
                sf_channel_bin_edges[0].shape[0] - 1,  # FIXED
            ),
            dtype=np.int64,
        )
        hist_other_total = np.zeros(
            (
                N_OTHER_CHANNELS,
                ell_bin_edges.shape[0] - 1,
                product_bin_edges.shape[0] - 1,
            ),
            dtype=np.int64,
        )
        
        for idx in range(displacements.shape[0]):
            dx, dy = displacements[idx]
            hm_part, ho_part = compute_histogram_for_disp_2D(
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
                sf_channel_bin_edges,  # FIXED: Pass as list
                product_bin_edges,
                stencil_width,
            )
            hist_mag_total += hm_part
            hist_other_total += ho_part
        
        return hist_mag_total, hist_other_total

    # Create shared-memory segments
    shm_objects: Dict[str, shared_memory.SharedMemory] = {}
    shm_meta: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}
    try:
        import time
        for key in required:
            arr = np.ascontiguousarray(fields[key])
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            np.copyto(shm_arr, arr, casting='no')
            if not np.array_equal(shm_arr, arr):
                raise RuntimeError(f"Failed to copy {key} to shared memory")
            shm_objects[key] = shm
            shm_meta[key] = (shm.name, arr.shape, str(arr.dtype))
        
        time.sleep(0.01)

        # Prepare batching
        batch_size = max(1, displacements.shape[0] // n_processes)
        batches = [
            range(i, min(i + batch_size, displacements.shape[0]))
            for i in range(0, displacements.shape[0], batch_size)
        ]

        n_ell_bins = ell_bin_edges.shape[0] - 1
        n_theta_bins = theta_bin_edges.shape[0] - 1
        n_phi_bins = phi_bin_edges.shape[0] - 1

        with Pool(processes=n_processes, initializer=_init_worker, initargs=(shm_meta,)) as pool:
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
                        sf_channel_bin_edges,  # FIXED: Pass as list
                        product_bin_edges,
                        stencil_width,
                        n_ell_bins,
                        n_theta_bins,
                        n_phi_bins,
                    )
                    for batch in batches
                ],
            )

        # Aggregate results
        hist_mag_total = np.zeros(
            (
                N_MAG_CHANNELS,
                n_ell_bins,
                n_theta_bins,
                n_phi_bins,
                sf_channel_bin_edges[0].shape[0] - 1,  # FIXED
            ),
            dtype=np.int64,
        )

        hist_other_total = np.zeros(
            (
                N_OTHER_CHANNELS,
                n_ell_bins,
                product_bin_edges.shape[0] - 1,
            ),
            dtype=np.int64,
        )

        for hm, ho in results:
            hist_mag_total += hm
            hist_other_total += ho

        return hist_mag_total, hist_other_total
    finally:
        # Cleanup shared memory
        for shm in shm_objects.values():
            with contextlib.suppress(FileNotFoundError):
                shm.close()
                shm.unlink()