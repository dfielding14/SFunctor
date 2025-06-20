#!/usr/bin/env python3
"""Driver for 2-D slice structure-function analysis with multiprocessing and optional MPI support.

This script supports three modes of operation:
1. Single slice processing
2. Multiple slices with multiprocessing (default for --slice_list)
3. Multiple slices with MPI (when run with mpirun/mpiexec)

The script automatically detects whether MPI is available and whether it's being
run under MPI. When processing multiple slices without MPI, it uses Python's
multiprocessing module for parallel execution on a single node.

Usage (single slice):
    python run_sf.py --file_name slice_x1_-0.375_....npz --stride 2

Usage (multiple slices with multiprocessing):
    python run_sf.py --slice_list slices.txt --stride 2 --n_disp_total 200000

Usage (cluster with MPI):
    mpirun -n 64 python run_sf.py --slice_list slices.txt --stride 2 --n_disp_total 200000
"""
from __future__ import annotations

import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from mpi4py import MPI

    HAS_MPI = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    HAS_MPI = False
    comm = None
    rank = 0
    size = 1

from sfunctor.sf_cli import RunConfig, parse_cli
from sfunctor.sf_displacements import build_displacement_list, find_ell_bin_edges
from sfunctor.sf_histograms import (
    MAG_CHANNELS,
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
    OTHER_CHANNELS,
    Channel,
)
from sfunctor.sf_io import load_slice_npz, parse_slice_metadata
from sfunctor.sf_parallel import compute_histograms_shared
from sfunctor.sf_physics import compute_vA, compute_z_plus_minus


def process_single_slice(
    args: Tuple[Path, RunConfig],
) -> Tuple[Path, np.ndarray, np.ndarray, Dict]:
    """Process a single slice and return histograms.

    Returns:
        Tuple of (slice_path, hist_mag, hist_other, metadata)
    """
    slice_path, cfg = args

    print(f"[process_slice] Processing {slice_path.name}")

    # Load slice
    slice_data = load_slice_npz(slice_path, stride=cfg.stride)
    axis, beta = parse_slice_metadata(slice_path)

    rho = slice_data["rho"]
    B_x = slice_data["B_x"]
    B_y = slice_data["B_y"]
    B_z = slice_data["B_z"]
    v_x = slice_data["v_x"]
    v_y = slice_data["v_y"]
    v_z = slice_data["v_z"]

    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (z_plus_x, z_plus_y, z_plus_z), (z_minus_x, z_minus_y, z_minus_z) = (
        compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
    )

    # Displacements
    N_res = rho.shape[0]
    if cfg.stencil_width == 2:
        ell_max = N_res // 2
    elif cfg.stencil_width == 3:
        ell_max = N_res // 4
    else:  # 5-point
        ell_max = N_res // 8

    ell_bin_edges = find_ell_bin_edges(1.0, ell_max, cfg.n_ell_bins)
    displacements = build_displacement_list(ell_bin_edges, cfg.n_disp_total)

    # Histogram bin definitions
    n_theta_bins = 18
    theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)
    n_phi_bins = 18
    phi_bin_edges = np.linspace(0, np.pi, n_phi_bins + 1)

    sf_bin_edges = np.logspace(-6, 1, 501)
    product_bin_edges = np.logspace(-10, 1, 501)

    fields_for_parallel = dict(
        v_x=v_x,
        v_y=v_y,
        v_z=v_z,
        B_x=B_x,
        B_y=B_y,
        B_z=B_z,
        rho=rho,
        j_x=slice_data["j_x"],
        j_y=slice_data["j_y"],
        j_z=slice_data["j_z"],
        vA_x=vA_x,
        vA_y=vA_y,
        vA_z=vA_z,
        zp_x=z_plus_x,
        zp_y=z_plus_y,
        zp_z=z_plus_z,
        zm_x=z_minus_x,
        zm_y=z_minus_y,
        zm_z=z_minus_z,
        omega_x=slice_data["omega_x"],
        omega_y=slice_data["omega_y"],
        omega_z=slice_data["omega_z"],
    )

    hist_mag, hist_other = compute_histograms_shared(
        fields_for_parallel,
        displacements,
        axis=axis,
        N_random_subsamples=cfg.N_random_subsamples,
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges,
        product_bin_edges=product_bin_edges,
        stencil_width=cfg.stencil_width,
        n_processes=cfg.n_processes,
    )

    metadata = dict(
        ell_bin_edges=ell_bin_edges,
        theta_bin_edges=theta_bin_edges,
        phi_bin_edges=phi_bin_edges,
        sf_bin_edges=sf_bin_edges,
        product_bin_edges=product_bin_edges,
        axis=axis,
        beta=beta,
    )

    return slice_path, hist_mag, hist_other, metadata


def save_results(
    slice_path: Path,
    hist_mag: np.ndarray,
    hist_other: np.ndarray,
    metadata: Dict,
    cfg: RunConfig,
) -> None:
    """Save histogram results to file."""
    out_name = (
        Path("slice_data")
        / f"hist_st{cfg.stencil_width}_{slice_path.stem}_ndisp{cfg.n_disp_total}_Nsub{cfg.N_random_subsamples}.npz"
    )
    out_name.parent.mkdir(parents=True, exist_ok=True)

    # Build dict combining mag (θ-φ resolved) and other (θ-φ collapsed)
    hist_dict = {}
    for idx, ch in enumerate(MAG_CHANNELS):
        hist_dict[ch.name] = hist_mag[idx]
    for idx, ch in enumerate(OTHER_CHANNELS):
        hist_dict[ch.name] = hist_other[idx]

    np.savez(
        out_name,
        **hist_dict,
        **metadata,
        meta=dict(
            slice=str(slice_path),
            axis=metadata["axis"],
            beta=metadata["beta"],
            stride=cfg.stride,
            stencil_width=cfg.stencil_width,
            date=datetime.utcnow().isoformat(),
        ),
    )
    print(f"[run_sf] Done. Output written to {out_name}")


def main() -> None:
    cfg = parse_cli()

    # ---------------------------------------------------------------------
    # Handle different execution modes -------------------------------------
    # ---------------------------------------------------------------------

    if cfg.slice_list:
        # Read slice list
        with open(cfg.slice_list) as f:
            all_paths = [Path(line.strip()) for line in f if line.strip()]

        if HAS_MPI and size > 1:
            # MPI mode: each rank processes different slices
            print(f"[run_sf] Using MPI with {size} ranks")

            # Distribute slices among ranks
            my_slices = []
            for i, path in enumerate(all_paths):
                if i % size == rank:
                    my_slices.append(path)

            # Process assigned slices
            for slice_path in my_slices:
                _, hist_mag, hist_other, metadata = process_single_slice(
                    (slice_path, cfg)
                )
                if (
                    rank == 0 or not HAS_MPI
                ):  # In MPI mode, optionally only rank 0 saves
                    save_results(slice_path, hist_mag, hist_other, metadata, cfg)

        else:
            # Multiprocessing mode: process slices in parallel on single node
            n_workers = min(cfg.n_processes or cpu_count() - 1, len(all_paths))
            print(f"[run_sf] Using multiprocessing with {n_workers} workers")

            # Process slices in parallel
            with Pool(processes=n_workers) as pool:
                args = [(path, cfg) for path in all_paths]
                results = pool.map(process_single_slice, args)

            # Save all results
            for slice_path, hist_mag, hist_other, metadata in results:
                save_results(slice_path, hist_mag, hist_other, metadata, cfg)

    else:
        # Single file mode
        if rank == 0:
            print(f"[run_sf] Processing single file")

        slice_path, hist_mag, hist_other, metadata = process_single_slice(
            (cfg.file_name, cfg)
        )

        if rank == 0:  # Only rank 0 saves in MPI mode
            save_results(slice_path, hist_mag, hist_other, metadata, cfg)


if __name__ == "__main__":
    main()
