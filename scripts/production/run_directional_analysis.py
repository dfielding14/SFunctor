#!/usr/bin/env python3
"""Run strict pairwise Chen/Mallet-style directional S2 analysis on one slice."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sfunctor.analysis.directional import (
    constant_s2_shapes,
    coverage_rows,
    result_to_npz_payload,
    summary_rows,
)
from sfunctor.core.directional import DirectionalConfig, compute_directional_structure_functions
from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True, help="Extracted 2-D slice NPZ")
    parser.add_argument("--displacements", required=True, help="NPZ containing displacements and ell_bin_edges")
    parser.add_argument("--output", required=True, help="Output NPZ")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cell_sizes", type=float, nargs=3, metavar=("DX1", "DX2", "DX3"), default=(1.0, 1.0, 1.0),
                        help="Effective Cartesian spacings after stride")
    parser.add_argument("--sample_count", type=int, default=None,
                        help="Origins per displacement; default enumerates the full slice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho0", type=float, default=None,
                        help="Fixed reference density; default is the finite positive slice mean")
    parser.add_argument("--rho_floor", type=float, default=0.0,
                        help="Pointwise-density cells at or below this value are excluded")
    parser.add_argument("--theta_parallel_max", type=float, default=15.0, help="Parallel wedge half-width in degrees")
    parser.add_argument("--theta_perpendicular_min", type=float, default=75.0, help="Perpendicular wedge lower edge in degrees")
    parser.add_argument("--phi_xi_max", type=float, default=15.0, help="xi wedge half-width in degrees")
    parser.add_argument("--phi_lambda_min", type=float, default=75.0, help="lambda wedge lower edge in degrees")
    parser.add_argument("--no_global", action="store_true", help="Omit global-mean-field sanity-check frame")
    parser.add_argument("--fit_interval", type=float, nargs=2, metavar=("ELL_MIN", "ELL_MAX"), default=None)
    parser.add_argument("--fit_min_count", type=int, default=100,
                        help="Minimum accepted pairs in each separation bin used for a fit")
    parser.add_argument("--fit_min_accepted_fraction", type=float, default=0.0,
                        help="Minimum accepted/attempted fraction in each separation bin used for a fit")
    parser.add_argument("--fit_min_bins", type=int, default=2,
                        help="Minimum eligible separation bins required to report a slope")
    parser.add_argument("--shape_s2_levels", type=float, nargs="*", default=())
    parser.add_argument("--q", nargs="+", default=None,
                        help="Subset of q variants; default: z_plus z_minus z_plus_ref z_minus_ref B vA vA_ref u")
    args = parser.parse_args(argv)

    slice_path = Path(args.slice)
    slice_axis, _ = parse_slice_metadata(slice_path)
    slice_data = load_slice_npz(slice_path, stride=args.stride)
    with np.load(args.displacements) as disp_npz:
        displacements = np.asarray(disp_npz["displacements"])
        ell_bin_edges = np.asarray(disp_npz["ell_bin_edges"], dtype=float)

    config = DirectionalConfig(
        ell_bin_edges=ell_bin_edges,
        cell_sizes=tuple(args.cell_sizes),
        theta_parallel_max=np.deg2rad(args.theta_parallel_max),
        theta_perpendicular_min=np.deg2rad(args.theta_perpendicular_min),
        phi_xi_max=np.deg2rad(args.phi_xi_max),
        phi_lambda_min=np.deg2rad(args.phi_lambda_min),
        sample_count=args.sample_count,
        seed=args.seed,
        include_global=not args.no_global,
    )
    result = compute_directional_structure_functions(
        slice_data,
        displacements,
        slice_axis=slice_axis,
        config=config,
        rho0=args.rho0,
        rho_floor=args.rho_floor,
        q_names=args.q,
    )
    payload = result_to_npz_payload(result)
    payload["slice_axis"] = slice_axis
    payload["slice"] = str(slice_path)
    payload["coverage_rows"] = np.asarray(
        coverage_rows(
            result,
            min_count=args.fit_min_count,
            min_accepted_fraction=args.fit_min_accepted_fraction,
        ),
        dtype=object,
    )
    if args.fit_interval:
        payload["summary_rows"] = np.asarray(
            summary_rows(
                result,
                tuple(args.fit_interval),
                min_count=args.fit_min_count,
                min_accepted_fraction=args.fit_min_accepted_fraction,
                min_bins=args.fit_min_bins,
            ),
            dtype=object,
        )
    if args.shape_s2_levels:
        for q_name in result.q_names:
            payload[f"shapes_{q_name}"] = constant_s2_shapes(result, q_name, args.shape_s2_levels)
    np.savez_compressed(args.output, **payload)
    print(f"Saved strict directional result to {args.output}")
    print(f"Elapsed seconds: {result.elapsed_seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
