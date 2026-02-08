#!/usr/bin/env python3
"""MPI-capable batch processing for structure function analysis.

This module provides the main entry point for processing multiple slices
in parallel using MPI. It handles work distribution across ranks and
aggregates results.

Usage:
    python -m sfunctor.analysis.batch --file_name slice.npz --stride 2
    mpirun -n 64 python -m sfunctor.analysis.batch --slice_list slices.txt
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np

# MPI import with graceful fallback
try:
    from mpi4py import MPI  # type: ignore
    _mpi_enabled = True
except ModuleNotFoundError:
    _mpi_enabled = False

    class _SerialComm:
        """Single-rank drop-in replacement for mpi4py COMM_WORLD."""

        def Get_rank(self) -> int:  # noqa: N802
            return 0

        def Get_size(self) -> int:  # noqa: N802
            return 1

        def bcast(self, obj, root: int = 0):  # noqa: D401, ANN001
            return obj

        def Reduce(self, sendbuf, recvbuf, op=None, root: int = 0):  # noqa: ANN001, N802
            recvbuf[...] = sendbuf

        def Barrier(self):  # noqa: N802
            pass

    class _FakeMPI:  # pylint: disable=too-few-public-methods
        """Minimal MPI module replacement for single-node execution."""
        COMM_WORLD = _SerialComm()
        SUM = None  # placeholder so "op=MPI.SUM" is still valid

    MPI = _FakeMPI()  # type: ignore

from sfunctor.utils.cli import parse_cli
from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.utils.displacements import find_ell_bin_edges, build_displacement_list
from sfunctor.core.histograms import (
    Channel,
    N_CHANNELS,
)
from sfunctor.core.parallel import compute_histograms_shared

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main() -> None:
    """Main entry point for batch structure function analysis.
    
    Orchestrates the full analysis pipeline:
    1. Parses command-line arguments
    2. Distributes work across MPI ranks (if available)
    3. Processes assigned slices
    4. Aggregates results across ranks
    5. Saves final histograms
    
    The function handles both single-node and multi-node execution
    transparently, using MPI when available or falling back to serial
    processing.
    """
    cfg = parse_cli()

    # Build list of slice paths and decide which ones this rank will run
    if cfg.slice_list:
        if rank == 0:
            with open(cfg.slice_list) as f:
                all_paths = [Path(line.strip()) for line in f if line.strip()]
        else:
            all_paths = None  # type: ignore[assignment]

        # Every rank needs the full list for bookkeeping / logging
        all_paths = comm.bcast(all_paths, root=0)

        if size > 1:
            # Keep the original one-slice-per-rank semantics
            if rank >= len(all_paths):
                # More ranks than slices – nothing to do for this rank
                return
            slice_paths_my_rank = [all_paths[rank]]
        else:
            # Single-rank run → process all slices sequentially
            slice_paths_my_rank = all_paths
    else:
        # Single-slice mode. In MPI runs, avoid rank collisions on output files by
        # letting rank 0 process the slice and having all other ranks exit.
        if size > 1 and rank != 0:
            if rank == 1:
                print(
                    "[sfunctor.batch] MPI + --file_name detected: only rank 0 will "
                    "process the slice to avoid duplicate writes."
                )
            return
        slice_paths_my_rank = [cfg.file_name]

    if rank == 0:
        mode = "MPI" if _mpi_enabled and size > 1 else "serial"
        print(f"[sfunctor.batch] Starting analysis in {mode} mode with {size} rank(s)")

    # Loop over the slice(s) assigned to this rank
    for slice_path in slice_paths_my_rank:
        _process_single_slice(slice_path, cfg)


def _process_single_slice(slice_path: Path, cfg) -> None:  # noqa: ANN001
    """Process a single 2-D slice and save results.
    
    This function performs the core analysis for one slice:
    1. Loads the slice data from disk
    2. Computes derived fields (Alfvén velocity, Elsasser variables)
    3. Generates displacement vectors
    4. Computes structure function histograms
    5. Saves output
    
    Parameters
    ----------
    slice_path : Path
        Path to the .npz file containing the 2D slice data.
    cfg : RunConfig
        Configuration object with analysis parameters.
    
    Notes
    -----
    This function is called once per slice assigned to the current MPI rank.
    """
    # Load slice
    axis, beta = parse_slice_metadata(slice_path)
    slice_data = load_slice_npz(slice_path, stride=cfg.stride)

    rho = slice_data["rho"]
    B_x = slice_data["B_x"]
    B_y = slice_data["B_y"]
    B_z = slice_data["B_z"]
    v_x = slice_data["v_x"]
    v_y = slice_data["v_y"]
    v_z = slice_data["v_z"]

    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (z_plus_x, z_plus_y, z_plus_z), (z_minus_x, z_minus_y, z_minus_z) = compute_z_plus_minus(
        v_x, v_y, v_z, vA_x, vA_y, vA_z
    )

    # Displacements
    N_res = rho.shape[0]
    if cfg.stencil_width == 2:
        ell_max = N_res // 2
    elif cfg.stencil_width == 3:
        ell_max = N_res // 4
    else:
        ell_max = N_res // 8  # 5-point stencil

    ell_bin_edges = find_ell_bin_edges(1.0, ell_max, cfg.n_ell_bins)
    displacements = build_displacement_list(ell_bin_edges, cfg.n_disp_total)

    # Histogram bin setup
    n_theta_bins = 18
    theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)
    n_phi_bins = 16  # keep phi resolution similar to theta
    # Phi only needs to cover 0–90° because the angle is built from |cos phi|
    phi_bin_edges = np.linspace(0, np.pi / 2, n_phi_bins + 1)
    
    delta_bin_edges = [np.logspace(-4, 1, 128) for _ in range(N_CHANNELS)]

    # Compute histograms
    fields = {
        "v_x": v_x, "v_y": v_y, "v_z": v_z,
        "B_x": B_x, "B_y": B_y, "B_z": B_z,
        "rho": rho,
        "vA_x": vA_x, "vA_y": vA_y, "vA_z": vA_z,
        "zp_x": z_plus_x, "zp_y": z_plus_y, "zp_z": z_plus_z,
        "zm_x": z_minus_x, "zm_y": z_minus_y, "zm_z": z_minus_z,
        "omega_x": slice_data.get("omega_x", np.full_like(rho, np.nan)),
        "omega_y": slice_data.get("omega_y", np.full_like(rho, np.nan)),
        "omega_z": slice_data.get("omega_z", np.full_like(rho, np.nan)),
        "j_x": slice_data.get("j_x", np.full_like(rho, np.nan)),
        "j_y": slice_data.get("j_y", np.full_like(rho, np.nan)),
        "j_z": slice_data.get("j_z", np.full_like(rho, np.nan)),
        "curv_x": slice_data.get("curv_x", np.full_like(rho, np.nan)),
        "curv_y": slice_data.get("curv_y", np.full_like(rho, np.nan)),
        "curv_z": slice_data.get("curv_z", np.full_like(rho, np.nan)),
        "grad_rho_x": slice_data.get("grad_rho_x", np.full_like(rho, np.nan)),
        "grad_rho_y": slice_data.get("grad_rho_y", np.full_like(rho, np.nan)),
        "grad_rho_z": slice_data.get("grad_rho_z", np.full_like(rho, np.nan)),
    }

    hist = compute_histograms_shared(
        fields,
        displacements,
        axis=axis,
        N_random_subsamples=cfg.N_random_subsamples,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        delta_bin_edges=delta_bin_edges,
        stencil_width=cfg.stencil_width,
        n_processes=cfg.n_processes,
    )

    # Save results.
    # When running with MPI, each rank processes a different slice (one line from
    # --slice_list), so we do not reduce across ranks.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"sf_results_{slice_path.stem}_{timestamp}.npz"
        
    np.savez_compressed(
        output_file,
        hist=hist,
        channels=[ch.name for ch in Channel],
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        delta_bin_edges=np.array(delta_bin_edges, dtype=object),
        displacements=displacements,
        metadata={
            "slice": str(slice_path),
            "stride": cfg.stride,
            "stencil_width": cfg.stencil_width,
            "N_random_subsamples": cfg.N_random_subsamples,
            "axis": axis,
            "beta": beta,
            "mpi_size": size,
            "mpi_rank": rank,
        },
    )
    print(f"[sfunctor.batch] Results saved to {output_file}")


if __name__ == "__main__":
    main()
