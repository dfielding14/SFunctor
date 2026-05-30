#!/usr/bin/env python3
"""Compare pointwise-density and fixed-rho0 Elsasser-like directional S2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sfunctor.analysis.directional import coverage_rows, fit_directional_slopes
from sfunctor.core.directional import DirectionalConfig, compute_directional_structure_functions
from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata


Q_NAMES = ("z_plus", "z_minus", "z_plus_ref", "z_minus_ref")
COMPARISONS = (("z_plus", "z_plus_ref"), ("z_minus", "z_minus_ref"))


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def _coverage_summary(result, *, min_count, min_accepted_fraction):
    rows = coverage_rows(
        result,
        min_count=min_count,
        min_accepted_fraction=min_accepted_fraction,
    )
    summary = []
    for q_name in result.q_names:
        for geometry in result.geometry_names:
            for direction in result.direction_names:
                selected = [
                    row for row in rows
                    if row["q"] == q_name
                    and row["geometry"] == geometry
                    and row["direction"] == direction
                ]
                accepted = sum(row["accepted"] for row in selected)
                attempts = sum(row["attempts"] for row in selected)
                summary.append(
                    {
                        "q": q_name,
                        "geometry": geometry,
                        "direction": direction,
                        "accepted": accepted,
                        "attempts": attempts,
                        "accepted_fraction": accepted / attempts if attempts else 0.0,
                        "eligible_bins": sum(bool(row["fit_eligible"]) for row in selected),
                        "total_bins": len(selected),
                    }
                )
    return summary


def _comparison_rows(result, slopes):
    rows = []
    for pointwise, fixed in COMPARISONS:
        pointwise_index = result.q_names.index(pointwise)
        fixed_index = result.q_names.index(fixed)
        for geometry_index, geometry in enumerate(result.geometry_names):
            for direction_index, direction in enumerate(result.direction_names):
                left = result.s2[pointwise_index, geometry_index, direction_index]
                right = result.s2[fixed_index, geometry_index, direction_index]
                valid = np.isfinite(left) & np.isfinite(right) & ((np.abs(left) + np.abs(right)) > 0.0)
                difference = np.abs(left[valid] - right[valid]) / (0.5 * (np.abs(left[valid]) + np.abs(right[valid])))
                left_fit = slopes[pointwise][geometry][direction]
                right_fit = slopes[fixed][geometry][direction]
                slope_difference = (
                    left_fit["slope"] - right_fit["slope"]
                    if np.isfinite(left_fit["slope"]) and np.isfinite(right_fit["slope"])
                    else np.nan
                )
                rows.append(
                    {
                        "pointwise_q": pointwise,
                        "fixed_q": fixed,
                        "geometry": geometry,
                        "direction": direction,
                        "common_s2_bins": int(np.count_nonzero(valid)),
                        "median_symmetric_relative_s2_difference": _finite_or_none(np.median(difference)) if difference.size else None,
                        "max_symmetric_relative_s2_difference": _finite_or_none(np.max(difference)) if difference.size else None,
                        "pointwise_slope": _finite_or_none(left_fit["slope"]),
                        "fixed_slope": _finite_or_none(right_fit["slope"]),
                        "slope_difference": _finite_or_none(slope_difference),
                        "pointwise_fit_quality": left_fit["quality"],
                        "fixed_fit_quality": right_fit["quality"],
                    }
                )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", nargs="+", required=True, help="Extracted 2-D slice NPZ files")
    parser.add_argument("--displacements", required=True, help="NPZ containing displacements and ell_bin_edges")
    parser.add_argument("--output", required=True, help="JSON comparison report")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--sample_count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho0", type=float, default=None,
                        help="Fixed reference density; pass the initial density for production comparisons")
    parser.add_argument("--rho_floor", type=float, default=0.0)
    parser.add_argument("--cell_sizes", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--angular_widths", type=float, nargs="+", default=(10.0, 15.0, 20.0),
                        help="Symmetric theta and phi wedge half-widths in degrees")
    parser.add_argument("--fit_interval", type=float, nargs=2, required=True)
    parser.add_argument("--fit_min_count", type=int, default=100)
    parser.add_argument("--fit_min_accepted_fraction", type=float, default=0.0)
    parser.add_argument("--fit_min_bins", type=int, default=2)
    args = parser.parse_args(argv)

    with np.load(args.displacements) as disp_npz:
        displacements = np.asarray(disp_npz["displacements"])
        ell_bin_edges = np.asarray(disp_npz["ell_bin_edges"], dtype=float)
    report = {
        "config": {
            "displacements": str(args.displacements),
            "stride": args.stride,
            "sample_count": args.sample_count,
            "seed": args.seed,
            "rho0_requested": args.rho0,
            "rho_floor": args.rho_floor,
            "cell_sizes": tuple(args.cell_sizes),
            "angular_widths_deg": tuple(args.angular_widths),
            "fit_interval": tuple(args.fit_interval),
            "fit_min_count": args.fit_min_count,
            "fit_min_accepted_fraction": args.fit_min_accepted_fraction,
            "fit_min_bins": args.fit_min_bins,
        },
        "runs": [],
    }
    for slice_name in args.slice:
        slice_path = Path(slice_name)
        slice_axis, _ = parse_slice_metadata(slice_path)
        slice_data = load_slice_npz(slice_path, stride=args.stride)
        for width in args.angular_widths:
            config = DirectionalConfig(
                ell_bin_edges=ell_bin_edges,
                cell_sizes=tuple(args.cell_sizes),
                theta_parallel_max=np.deg2rad(width),
                theta_perpendicular_min=np.deg2rad(90.0 - width),
                phi_xi_max=np.deg2rad(width),
                phi_lambda_min=np.deg2rad(90.0 - width),
                sample_count=args.sample_count,
                seed=args.seed,
                include_global=True,
            )
            result = compute_directional_structure_functions(
                slice_data,
                displacements,
                slice_axis=slice_axis,
                config=config,
                rho0=args.rho0,
                rho_floor=args.rho_floor,
                q_names=Q_NAMES,
            )
            slopes = fit_directional_slopes(
                result,
                tuple(args.fit_interval),
                min_count=args.fit_min_count,
                min_accepted_fraction=args.fit_min_accepted_fraction,
                min_bins=args.fit_min_bins,
            )
            report["runs"].append(
                {
                    "slice": str(slice_path),
                    "slice_axis": slice_axis,
                    "angular_width_deg": width,
                    "rho0_used": result.rho0,
                    "elapsed_seconds": result.elapsed_seconds,
                    "coverage": _coverage_summary(
                        result,
                        min_count=args.fit_min_count,
                        min_accepted_fraction=args.fit_min_accepted_fraction,
                    ),
                    "comparisons": _comparison_rows(result, slopes),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Saved density-convention comparison to {output}")
    print(f"Completed {len(report['runs'])} slice/wedge runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
