#!/usr/bin/env python3
"""MPI-capable driver for 2-D slice structure-function analysis.

Usage (single node):
    python run_sf.py --file_name slice_x1_-0.375_....npz --stride 2

Usage (cluster):
    mpirun -n 64 python run_sf.py --slice_list slices.txt --stride 2 --n_disp_total 200000
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from mpi4py import MPI

from sf_cli import parse_cli
from sf_io import load_slice_npz, parse_slice_metadata
from sf_physics import compute_vA, compute_z_plus_minus
from sf_displacements import find_ell_bin_edges, build_displacement_list
from sf_histograms import (
    Channel,
    MAG_CHANNELS,
    OTHER_CHANNELS,
)
from sf_parallel import compute_histograms_shared

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main() -> None:
    cfg = parse_cli()

    # ---------------------------------------------------------------------
    # Determine slice path for this rank -----------------------------------
    # ---------------------------------------------------------------------
    if cfg.slice_list:
        if rank == 0:
            with open(cfg.slice_list) as f:
                all_paths = [Path(line.strip()) for line in f if line.strip()]
        else:
            all_paths = None
        all_paths = comm.bcast(all_paths, root=0)
        if rank >= len(all_paths):
            if rank == 0:
                print(f"[run_sf] Warning: size {size} > number of slices {len(all_paths)}")
            return  # idle ranks exit early
        slice_path = all_paths[rank]
    else:
        slice_path = cfg.file_name

    if rank == 0:
        print(f"[run_sf] Starting analysis with {size} MPI ranks")

    # ---------------------------------------------------------------------
    # Load slice -----------------------------------------------------------
    # ---------------------------------------------------------------------
    slice_data = load_slice_npz(slice_path, stride=cfg.stride)
    axis, beta = parse_slice_metadata(slice_path)

    rho = slice_data["rho"]
    B_x = slice_data["B_x"]; B_y = slice_data["B_y"]; B_z = slice_data["B_z"]
    v_x = slice_data["v_x"]; v_y = slice_data["v_y"]; v_z = slice_data["v_z"]

    vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
    (z_plus_x, z_plus_y, z_plus_z), (z_minus_x, z_minus_y, z_minus_z) = compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)

    # ---------------------------------------------------------------------
    # Displacements --------------------------------------------------------
    # ---------------------------------------------------------------------
    N_res = rho.shape[0]
    # Adjust ℓ-max depending on stencil width -----------------------------
    if cfg.stencil_width == 2:
        ell_max = N_res // 2
    elif cfg.stencil_width == 3:
        ell_max = N_res // 4
    else:  # 5-point
        ell_max = N_res // 8

    ell_bin_edges = find_ell_bin_edges(1.0, ell_max, cfg.n_ell_bins)
    displacements = build_displacement_list(ell_bin_edges, cfg.n_disp_total)

    # Histogram bin definitions (reuse from legacy script) -----------------
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

    local_mag, local_other = compute_histograms_shared(
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

    # MPI reduce -----------------------------------------------------------
    global_mag = np.zeros_like(local_mag)
    global_other = np.zeros_like(local_other)
    comm.Reduce(local_mag, global_mag, op=MPI.SUM, root=0)
    comm.Reduce(local_other, global_other, op=MPI.SUM, root=0)

    if rank == 0:
        out_name = Path("slice_data") / f"hist_st{cfg.stencil_width}_{slice_path.stem}_ndisp{cfg.n_disp_total}_Nsub{cfg.N_random_subsamples}.npz"
        out_name.parent.mkdir(parents=True, exist_ok=True)

        # Build dict combining mag (θ-φ resolved) and other (θ-φ collapsed) --
        hist_dict = {}
        for idx, ch in enumerate(MAG_CHANNELS):
            hist_dict[ch.name] = global_mag[idx]
        for idx, ch in enumerate(OTHER_CHANNELS):
            hist_dict[ch.name] = global_other[idx]

        np.savez(
            out_name,
            **hist_dict,
            ell_bin_edges=ell_bin_edges,
            theta_bin_edges=theta_bin_edges,
            phi_bin_edges=phi_bin_edges,
            sf_bin_edges=sf_bin_edges,
            product_bin_edges=product_bin_edges,
            meta=dict(
                slice=str(slice_path),
                axis=axis,
                beta=beta,
                stride=cfg.stride,
                stencil_width=cfg.stencil_width,
                date=datetime.utcnow().isoformat(),
            ),
        )
        print(f"[run_sf] Done. Output written to {out_name}")


if __name__ == "__main__":
    main() 