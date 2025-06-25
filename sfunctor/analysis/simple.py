#!/usr/bin/env python3
"""Simplified structure function analysis without Numba for testing purposes.

This module provides a pure Python implementation of structure function
calculations without dependencies on Numba or MPI. It's designed for:
- Testing and validation of the main pipeline
- Quick exploratory analysis on small datasets  
- Debugging structure function algorithms
- Educational purposes to understand the core concepts

The simplified version computes basic structure functions (velocity, magnetic
field, density) but omits the full histogram binning and angle-resolved
analysis of the production pipeline.

Usage:
    python simple_sf_analysis.py --file_name slice.npz --stride 2
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np

from sfunctor.utils.cli import parse_cli
from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.utils.displacements import find_ell_bin_edges, build_displacement_list


def compute_structure_functions_simple(
    v_x, v_y, v_z, B_x, B_y, B_z, rho,
    delta_i, delta_j, N_random_subsamples=100
):
    """Simplified structure function computation without Numba.
    
    Computes structure functions for a single displacement vector by
    randomly sampling spatial positions and computing field differences.
    
    Parameters
    ----------
    v_x, v_y, v_z : np.ndarray
        Velocity field components (2D arrays).
    B_x, B_y, B_z : np.ndarray
        Magnetic field components (2D arrays).
    rho : np.ndarray
        Density field (2D array).
    delta_i : int
        Displacement in x-direction (pixels).
    delta_j : int  
        Displacement in y-direction (pixels).
    N_random_subsamples : int, optional
        Number of random positions to sample. Default is 100.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays containing:
        - dv_values: Velocity structure function values |δv|
        - dB_values: Magnetic field structure function values |δB|
        - drho_values: Density structure function values |δρ|
    
    Notes
    -----
    Uses periodic boundary conditions for displacements that wrap around
    the domain edges.
    """
    N, M = v_x.shape
    
    # Random spatial samples
    flat_indices = np.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M
    random_points_x = flat_indices % M
    
    # Storage for structure function values
    dv_values = []
    dB_values = []
    drho_values = []
    
    for idx in range(N_random_subsamples):
        i = random_points_x[idx]
        j = random_points_y[idx]
        ip = (i + delta_i) % M
        jp = (j + delta_j) % N
        
        # Compute differences using 2-point stencil
        dvx = v_x[jp, ip] - v_x[j, i]
        dvy = v_y[jp, ip] - v_y[j, i]
        dvz = v_z[jp, ip] - v_z[j, i]
        
        dBx = B_x[jp, ip] - B_x[j, i]
        dBy = B_y[jp, ip] - B_y[j, i]
        dBz = B_z[jp, ip] - B_z[j, i]
        
        drho = rho[jp, ip] - rho[j, i]
        
        # Compute magnitudes
        dv = np.sqrt(dvx**2 + dvy**2 + dvz**2)
        dB = np.sqrt(dBx**2 + dBy**2 + dBz**2)
        drho_abs = abs(drho)
        
        dv_values.append(dv)
        dB_values.append(dB)
        drho_values.append(drho_abs)
    
    return np.array(dv_values), np.array(dB_values), np.array(drho_values)


def main():
    cfg = parse_cli()
    
    print(f"[simple_sf] Starting simplified structure function analysis")
    
    # Determine slice paths to process
    if cfg.slice_list:
        with open(cfg.slice_list) as f:
            slice_paths = [Path(line.strip()) for line in f if line.strip()]
        print(f"[simple_sf] Found {len(slice_paths)} slices in {cfg.slice_list}")
    else:
        slice_paths = [cfg.file_name]
        print(f"[simple_sf] Processing single slice: {cfg.file_name}")
    
    # Process each slice
    for i, slice_path in enumerate(slice_paths):
        print(f"[simple_sf] Processing slice {i+1}/{len(slice_paths)}: {slice_path}")
        
        try:
            # Load slice
            slice_data = load_slice_npz(slice_path, stride=cfg.stride)
            axis, beta = parse_slice_metadata(slice_path)
            
            rho = slice_data["rho"]
            B_x = slice_data["B_x"]; B_y = slice_data["B_y"]; B_z = slice_data["B_z"]
            v_x = slice_data["v_x"]; v_y = slice_data["v_y"]; v_z = slice_data["v_z"]
            
            print(f"[simple_sf] Loaded slice with shape: {rho.shape}")
            
            # Set up displacements
            N_res = rho.shape[0]
            ell_max = N_res // 4  # Conservative choice
            ell_bin_edges = find_ell_bin_edges(1.0, ell_max, cfg.n_ell_bins)
            displacements = build_displacement_list(ell_bin_edges, cfg.n_disp_total)
            
            print(f"[simple_sf] Grid size: {N_res}x{N_res}, ell_max: {ell_max}")
            print(f"[simple_sf] Number of displacements: {len(displacements)}")
            
            # Storage for results
            all_displacements = []
            all_dv_values = []
            all_dB_values = []
            all_drho_values = []
            
            # Process displacements
            for disp_idx, (dx, dy) in enumerate(displacements[:20]):  # Limit to first 20 for speed
                if disp_idx % 5 == 0:
                    print(f"[simple_sf] Processing displacement {disp_idx}/20")
                
                dv_vals, dB_vals, drho_vals = compute_structure_functions_simple(
                    v_x, v_y, v_z, B_x, B_y, B_z, rho,
                    int(dx), int(dy), cfg.N_random_subsamples
                )
                
                # Store displacement magnitude
                r = np.sqrt(dx**2 + dy**2)
                all_displacements.extend([r] * len(dv_vals))
                all_dv_values.extend(dv_vals)
                all_dB_values.extend(dB_vals)
                all_drho_values.extend(drho_vals)
            
            # Convert to arrays
            all_displacements = np.array(all_displacements)
            all_dv_values = np.array(all_dv_values)
            all_dB_values = np.array(all_dB_values)
            all_drho_values = np.array(all_drho_values)
            
            # Save results
            out_name = Path("slice_data") / f"simple_sf_{slice_path.stem}.npz"
            out_name.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                out_name,
                displacements=all_displacements,
                dv_values=all_dv_values,
                dB_values=all_dB_values,
                drho_values=all_drho_values,
                meta=dict(
                    slice=str(slice_path),
                    axis=axis,
                    beta=beta,
                    stride=cfg.stride,
                    date=datetime.utcnow().isoformat(),
                ),
            )
            
            print(f"[simple_sf] Saved results: {out_name}")
            
            # Print some basic statistics
            print(f"[simple_sf] Total samples: {len(all_dv_values)}")
            print(f"[simple_sf] Velocity SF range: {all_dv_values.min():.6f} - {all_dv_values.max():.6f}")
            print(f"[simple_sf] Magnetic SF range: {all_dB_values.min():.6f} - {all_dB_values.max():.6f}")
            print(f"[simple_sf] Density SF range: {all_drho_values.min():.6f} - {all_drho_values.max():.6f}")
            
        except Exception as e:
            print(f"[simple_sf] Failed to process {slice_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[simple_sf] Analysis complete!")


if __name__ == "__main__":
    main() 