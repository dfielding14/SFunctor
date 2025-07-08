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

import sys
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
        
        def allgather(self, obj):  # noqa: ANN001
            return [obj]

    class _FakeMPI:  # pylint: disable=too-few-public-methods
        """Minimal MPI module replacement for single-node execution."""
        COMM_WORLD = _SerialComm()
        SUM = None  # placeholder so "op=MPI.SUM" is still valid
        
        @staticmethod
        def Get_processor_name() -> str:  # noqa: N802
            import socket
            return socket.gethostname()

    MPI = _FakeMPI()  # type: ignore

from sfunctor.utils.cli import parse_cli
from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.utils.displacements import find_ell_bin_edges, build_displacement_list
from sfunctor.core.histograms import (
    Channel,
    MAG_CHANNELS,
    OTHER_CHANNELS,
)
from sfunctor.core.parallel import compute_histograms_shared

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def is_multi_node_run() -> bool:
    """Detect if MPI ranks span multiple nodes.
    
    Returns
    -------
    bool
        True if MPI ranks are distributed across multiple nodes, False otherwise.
    """
    if size == 1:
        return False
    
    processor_name = MPI.Get_processor_name()
    all_names = comm.allgather(processor_name)
    unique_nodes = len(set(all_names))
    
    if rank == 0 and unique_nodes > 1:
        print(f"[sfunctor.batch] Detected {unique_nodes} unique nodes in MPI run")
        node_counts = {}
        for name in all_names:
            node_counts[name] = node_counts.get(name, 0) + 1
        for node, count in sorted(node_counts.items()):
            print(f"  - {node}: {count} rank(s)")
    
    return unique_nodes > 1


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
    
    # Detect multi-node execution early
    is_multi_node = is_multi_node_run() if size > 1 else False

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
        slice_paths_my_rank = [cfg.file_name]

    if rank == 0:
        mode = "MPI" if _mpi_enabled and size > 1 else "serial"
        print(f"[sfunctor.batch] Starting analysis in {mode} mode with {size} rank(s)")

    # Loop over the slice(s) assigned to this rank
    for slice_path in slice_paths_my_rank:
        _process_single_slice(slice_path, cfg, is_multi_node)


def _process_single_slice(slice_path: Path, cfg, is_multi_node: bool = False) -> None:  # noqa: ANN001
    """Process a single 2-D slice and save results.
    
    This function performs the core analysis for one slice:
    1. Loads the slice data from disk
    2. Computes derived fields (Alfvén velocity, Elsasser variables)
    3. Generates displacement vectors
    4. Computes structure function histograms
    5. Reduces results across MPI ranks
    6. Saves output (rank 0 only)
    
    Parameters
    ----------
    slice_path : Path
        Path to the .npz file containing the 2D slice data.
    cfg : RunConfig
        Configuration object with analysis parameters.
    is_multi_node : bool, optional
        Whether MPI ranks span multiple nodes. Default is False.
    
    Notes
    -----
    This function is called once per slice assigned to the current MPI rank.
    Results are automatically aggregated across ranks using MPI reductions.
    When running across multiple nodes, multiprocessing is disabled to avoid
    shared memory issues.
    """
    # Load slice
    axis, beta = parse_slice_metadata(slice_path)
    slice_data = load_slice_npz(slice_path, stride=cfg.stride)

    rho = slice_data["rho"]
    B_x = slice_data["B_x"]; B_y = slice_data["B_y"]; B_z = slice_data["B_z"]
    v_x = slice_data["v_x"]; v_y = slice_data["v_y"]; v_z = slice_data["v_z"]

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
    n_phi_bins = 18
    phi_bin_edges = np.linspace(0, np.pi, n_phi_bins + 1)
    
    sf_bin_edges = np.logspace(-4, 1, 128)
    product_bin_edges = np.logspace(-5, 5, 128)

    # Determine n_processes based on multi-node status
    effective_n_processes = cfg.n_processes
    if is_multi_node and cfg.n_processes != 1:
        if rank == 0:
            print(f"[sfunctor.batch] WARNING: Multi-node MPI detected. Disabling multiprocessing (n_processes=1)")
            print(f"[sfunctor.batch] Original n_processes was {cfg.n_processes}")
            print(f"[sfunctor.batch] For better performance, consider using more MPI ranks")
        effective_n_processes = 1
    
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

    hist_mag, hist_other = compute_histograms_shared(
        fields,
        displacements,
        axis=axis,
        N_random_subsamples=cfg.N_random_subsamples,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges,
        product_bin_edges=product_bin_edges,
        stencil_width=cfg.stencil_width,
        n_processes=effective_n_processes,
    )

    # MPI reduction
    if size > 1:
        recv_mag = np.zeros_like(hist_mag) if rank == 0 else None
        recv_other = np.zeros_like(hist_other) if rank == 0 else None
        comm.Reduce(hist_mag, recv_mag, op=MPI.SUM, root=0)
        comm.Reduce(hist_other, recv_other, op=MPI.SUM, root=0)
        if rank == 0:
            hist_mag = recv_mag
            hist_other = recv_other

    # Save results (rank 0 only)
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sf_results_{timestamp}.npz"
        
        np.savez_compressed(
            output_file,
            hist_mag=hist_mag,
            hist_other=hist_other,
            mag_channels=[ch.name for ch in MAG_CHANNELS],
            other_channels=[ch.name for ch in OTHER_CHANNELS],
            ell_bin_edges=ell_bin_edges,
            theta_bin_edges=theta_bin_edges,
            phi_bin_edges=phi_bin_edges,
            sf_bin_edges=sf_bin_edges,
            product_bin_edges=product_bin_edges,
            displacements=displacements,
            metadata={
                "n_slices": 1,  # Single slice processed in this function
                "stride": cfg.stride,
                "stencil_width": cfg.stencil_width,
                "N_random_subsamples": cfg.N_random_subsamples,
                "axis": axis,
                "beta": beta,
            },
        )
        print(f"[sfunctor.batch] Results saved to {output_file}")


if __name__ == "__main__":
    main()