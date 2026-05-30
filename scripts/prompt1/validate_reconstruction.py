#!/usr/bin/env python3
"""Validate cbin reconstruction against limited full-resolution primitive regions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbin_tools import (  # noqa: E402
    MESH_BLOCK_CELLS,
    PRIMARY_BASES,
    PRIMARY_LABELS,
    _rounding_gamma,
    discover_snapshot,
    file_sha256,
    morton_rank,
    parse_binary_shard,
    read_record_fields,
    raw_moment_statistics,
    robust_nonnegative,
    shard_identity,
    shard_path,
    source_raw_moment_error_bounds,
)


DATA_ROOT_DEFAULT = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "data/data_Turb_10240_beta25_dedt025_plm"
)
GAMMA = 1.00001
INPUT_FILE_DEFAULT = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "inputs/Turb_10240_beta25_dedt025_plm.athinput"
)
FULL_FIELDS = ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3")
PRIMARY_STANDARDIZED_MOMENT_NAMES = frozenset(
    f"{base}_{stat}"
    for base in PRIMARY_BASES
    for stat in ("skewness", "kurtosis")
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _rank_coordinates(bounds: Sequence[int]) -> list[tuple[int, int, int]]:
    i0, i1, j0, j1, k0, k1 = bounds
    output = []
    for rz in range(k0 // MESH_BLOCK_CELLS[2], math.ceil(k1 / MESH_BLOCK_CELLS[2])):
        for ry in range(j0 // MESH_BLOCK_CELLS[1], math.ceil(j1 / MESH_BLOCK_CELLS[1])):
            for rx in range(i0 // MESH_BLOCK_CELLS[0], math.ceil(i1 / MESH_BLOCK_CELLS[0])):
                output.append((rx, ry, rz))
    return output


def _intersection_slices(
    bounds: Sequence[int],
    rank_coordinates: Sequence[int],
    cell_sizes: Sequence[int],
) -> tuple[slice, slice, slice] | None:
    i0, i1, j0, j1, k0, k1 = bounds
    rx, ry, rz = rank_coordinates
    block_i0, block_j0, block_k0 = rx * cell_sizes[0], ry * cell_sizes[1], rz * cell_sizes[2]
    block_i1, block_j1, block_k1 = block_i0 + cell_sizes[0], block_j0 + cell_sizes[1], block_k0 + cell_sizes[2]
    gi0, gi1 = max(i0, block_i0), min(i1, block_i1)
    gj0, gj1 = max(j0, block_j0), min(j1, block_j1)
    gk0, gk1 = max(k0, block_k0), min(k1, block_k1)
    if gi0 >= gi1 or gj0 >= gj1 or gk0 >= gk1:
        return None
    return (
        slice(gk0 - block_k0, gk1 - block_k0),
        slice(gj0 - block_j0, gj1 - block_j0),
        slice(gi0 - block_i0, gi1 - block_i0),
    )


def _full_rank_path(data_root: Path, rank: int, basename: str) -> Path:
    return data_root / "bin" / f"rank_{rank:08d}" / basename


def _discover_full_snapshot(data_root: Path, target_time: float, cycle: int) -> object:
    pattern = str(data_root / "bin" / "rank_00000000" / "Turb.full_mhd_w_bcc.*.bin")
    candidates = [parse_binary_shard(path, expect_cbin=False) for path in sorted(glob.glob(pattern))]
    matches = [
        shard for shard in candidates
        if math.isclose(shard.time, target_time, abs_tol=5.0e-6) and shard.cycle == cycle
    ]
    if len(matches) != 1:
        found = [(Path(shard.path).name, shard.time, shard.cycle) for shard in candidates]
        raise ValueError(f"expected one full snapshot at t={target_time}, cycle={cycle}, found {found}")
    return matches[0]


def _read_gamma(path: Path) -> float:
    section = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("<") and line.endswith(">"):
            section = line[1:-1]
        elif section == "mhd" and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key == "gamma":
                return float(value)
    raise ValueError(f"{path}: missing <mhd> gamma")


def _conserved_fields(primitive: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    rho = np.asarray(primitive["dens"], dtype=np.float64)
    vx = np.asarray(primitive["velx"], dtype=np.float64)
    vy = np.asarray(primitive["vely"], dtype=np.float64)
    vz = np.asarray(primitive["velz"], dtype=np.float64)
    bx = np.asarray(primitive["bcc1"], dtype=np.float64)
    by = np.asarray(primitive["bcc2"], dtype=np.float64)
    bz = np.asarray(primitive["bcc3"], dtype=np.float64)
    kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    magnetic = 0.5 * (bx * bx + by * by + bz * bz)
    return {
        "dens": rho,
        "mom1": rho * vx,
        "mom2": rho * vy,
        "mom3": rho * vz,
        "ener": np.asarray(primitive["eint"], dtype=np.float64) + kinetic + magnetic,
        "bcc1": bx,
        "bcc2": by,
        "bcc3": bz,
    }


def direct_region_means(
    data_root: Path,
    full_basename: str,
    bounds: Sequence[int],
    transform: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
) -> tuple[dict[str, float], int]:
    """Stream intersecting primitive slices and average transformed fields."""

    sums: defaultdict[str, float] = defaultdict(float)
    total_cells = 0
    for coordinates in _rank_coordinates(bounds):
        rank = morton_rank(*coordinates)
        path = _full_rank_path(data_root, rank, full_basename)
        shard = parse_binary_shard(path, expect_cbin=False)
        if len(shard.records) != 1 or tuple(shard.records[0].logical[:3]) != tuple(coordinates):
            raise ValueError(f"{path}: unexpected rank metadata {shard.records[0].logical}")
        slices = _intersection_slices(bounds, coordinates, MESH_BLOCK_CELLS)
        if slices is None:
            continue
        primitive = read_record_fields(shard, FULL_FIELDS, copy=False)
        primitive_slice = {name: values[slices] for name, values in primitive.items()}
        fields = transform(primitive_slice)
        count = int(next(iter(fields.values())).size)
        total_cells += count
        for name, values in fields.items():
            sums[name] += float(np.sum(values, dtype=np.float64))
    expected_cells = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
    if total_cells != expected_cells:
        raise ValueError(f"direct coverage mismatch {total_cells} != {expected_cells}")
    return {name: value / total_cells for name, value in sums.items()}, total_cells


def direct_primary_moments(data_root: Path, full_basename: str, bounds: Sequence[int]) -> tuple[dict[str, float], int]:
    def transform(primitive: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        conserved = _conserved_fields(primitive)
        return {
            f"{base}_{suffix}": values**power
            for base, values in conserved.items()
            for power, suffix in enumerate(("1st", "2nd", "3rd", "4th"), start=1)
        }

    return direct_region_means(data_root, full_basename, bounds, transform)


def cbin_region_means(
    data_root: Path,
    product: str,
    scale: int,
    basename: str,
    bounds: Sequence[int],
    fields: Sequence[str],
    *,
    return_error_bounds: bool = False,
) -> tuple[dict[str, float], int] | tuple[dict[str, float], int, dict[str, float]]:
    """Average selected cbin voxels over one aligned fine-cell region."""

    if any(value % scale for value in bounds):
        raise ValueError(f"bounds {bounds} are not aligned to cbin factor {scale}")
    coarse_bounds = tuple(value // scale for value in bounds)
    sums: defaultdict[str, float] = defaultdict(float)
    error_bound_sums: defaultdict[str, float] = defaultdict(float)
    raw_abs_sums: defaultdict[str, float] = defaultdict(float)
    total_voxels = 0
    for coordinates in _rank_coordinates(bounds):
        rank = morton_rank(*coordinates)
        path = shard_path(data_root, product, scale, rank, basename)
        shard = parse_binary_shard(path, expect_cbin=True)
        if len(shard.records) != 1 or tuple(shard.records[0].logical[:3]) != tuple(coordinates):
            raise ValueError(f"{path}: unexpected rank metadata {shard.records[0].logical}")
        coarse_block = tuple(size // scale for size in MESH_BLOCK_CELLS)
        slices = _intersection_slices(coarse_bounds, coordinates, coarse_block)
        if slices is None:
            continue
        payload = read_record_fields(shard, fields, copy=False)
        count = int(next(iter(payload.values()))[slices].size)
        total_voxels += count
        for name, values in payload.items():
            sums[name] += float(np.sum(values[slices], dtype=np.float64))
        if return_error_bounds:
            for base in PRIMARY_BASES:
                bounds_by_power = source_raw_moment_error_bounds(*(
                    payload[f"{base}_{suffix}"][slices]
                    for suffix in ("1st", "2nd", "3rd", "4th")
                ), source_cells=scale**3)
                for suffix, error_bound in zip(("1st", "2nd", "3rd", "4th"), bounds_by_power):
                    name = f"{base}_{suffix}"
                    error_bound_sums[name] += float(np.sum(error_bound, dtype=np.float64))
                    raw_abs_sums[name] += float(np.sum(np.abs(payload[name][slices]), dtype=np.float64))
    expected_voxels = (
        (bounds[1] - bounds[0]) // scale
        * (bounds[3] - bounds[2]) // scale
        * (bounds[5] - bounds[4]) // scale
    )
    if total_voxels != expected_voxels:
        raise ValueError(f"cbin coverage mismatch {total_voxels} != {expected_voxels}")
    means = {name: value / total_voxels for name, value in sums.items()}
    if return_error_bounds:
        return means, total_voxels, _regional_error_bounds(error_bound_sums, raw_abs_sums, total_voxels)
    return means, total_voxels


def magnetic_diagnostics(moments: Mapping[str, float]) -> dict[str, float]:
    bmean_sq = sum(moments[f"bcc{axis}_1st"] ** 2 for axis in (1, 2, 3))
    b2mean = sum(moments[f"bcc{axis}_2nd"] for axis in (1, 2, 3))
    delta_sq_array, flags = robust_nonnegative(np.asarray([b2mean - bmean_sq]), np.asarray([b2mean]))
    delta_sq = float(delta_sq_array[0])
    bmean = math.sqrt(bmean_sq)
    delta = math.sqrt(delta_sq)
    return {
        "B_mean": bmean,
        "B_rms": math.sqrt(b2mean),
        "deltaB": delta,
        "dBB": delta / bmean if bmean > 0.0 else float("nan"),
        "deltaB_flags": int(flags[0]),
    }


def retained_primary_diagnostics(
    moments: Mapping[str, float],
    *,
    raw_moment_error_bounds: Mapping[str, float] | None = None,
    precision_limited_unavailable: set[str] | None = None,
    diagnostic_error_bounds: dict[str, float] | None = None,
) -> dict[str, float]:
    """Calculate retained primary-only summaries after raw-moment comparison."""

    output: dict[str, float] = {}
    for base in PRIMARY_BASES:
        raw_moments = tuple(
            np.asarray([moments[f"{base}_{suffix}"]])
            for suffix in ("1st", "2nd", "3rd", "4th")
        )
        bounds = None if raw_moment_error_bounds is None else tuple(
            np.asarray([raw_moment_error_bounds[f"{base}_{suffix}"]])
            for suffix in ("1st", "2nd", "3rd", "4th")
        )
        stats = raw_moment_statistics(*raw_moments, abs_error_bounds=bounds)
        for name in ("variance", "sigma", "skewness", "kurtosis"):
            output[f"{base}_{name}"] = float(stats[name][0])
        if diagnostic_error_bounds is not None and bounds is not None:
            diagnostic_error_bounds[f"{base}_skewness"] = float(stats["skewness_error_bound"][0])
            diagnostic_error_bounds[f"{base}_kurtosis"] = float(stats["kurtosis_error_bound"][0])
        if precision_limited_unavailable is not None and bounds is not None:
            flags = int(stats["moment_flags"][0])
            if not flags & (0x02 | 0x08 | 0x80):
                if math.isnan(output[f"{base}_skewness"]) and flags & (0x04 | 0x10):
                    precision_limited_unavailable.add(f"{base}_skewness")
                if math.isnan(output[f"{base}_kurtosis"]) and flags & (0x04 | 0x40):
                    precision_limited_unavailable.add(f"{base}_kurtosis")
    rho = moments["dens_1st"]
    b2_mean = sum(moments[f"bcc{axis}_2nd"] for axis in (1, 2, 3))
    b_mean = math.sqrt(sum(moments[f"bcc{axis}_1st"] ** 2 for axis in (1, 2, 3)))
    output["rho_sigma_over_mean"] = output["dens_sigma"] / rho if rho > 0.0 else float("nan")
    for axis, coordinate in enumerate("xyz", start=1):
        output[f"u_mass_weighted_mean_{coordinate}"] = moments[f"mom{axis}_1st"] / rho if rho > 0.0 else float("nan")
    output["ener_mean"] = moments["ener_1st"]
    output["magnetic_energy_mean"] = 0.5 * b2_mean
    output["vA_mean_proxy"] = b_mean / math.sqrt(rho) if rho > 0.0 else float("nan")
    output["vA_rms_like_proxy"] = math.sqrt(b2_mean / rho) if rho > 0.0 else float("nan")
    return output


def _regional_error_bounds(
    error_bound_sums: Mapping[str, float],
    raw_abs_sums: Mapping[str, float],
    total_voxels: int,
) -> dict[str, float]:
    reduction_bound = _rounding_gamma(max(1, total_voxels))
    return {
        name: (value + reduction_bound * raw_abs_sums[name]) / total_voxels
        for name, value in error_bound_sums.items()
    }


def comparison_rows(
    case: str,
    direct: Mapping[str, float],
    reconstructed: Mapping[str, float],
    *,
    atol: float,
    rtol: float,
    precision_limited_unavailable: set[str] | None = None,
    paired_nan_allowed_names: set[str] | None = None,
    reconstructed_error_bounds: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in sorted(direct):
        expected = float(direct[name])
        actual = float(reconstructed[name])
        paired_nan = (
            paired_nan_allowed_names is not None
            and name in paired_nan_allowed_names
            and math.isnan(expected)
            and math.isnan(actual)
        )
        precision_limited = (
            precision_limited_unavailable is not None
            and name in precision_limited_unavailable
            and math.isfinite(expected)
            and math.isnan(actual)
        )
        reconstructed_error_bound = None if reconstructed_error_bounds is None else reconstructed_error_bounds.get(name)
        supported_error_bound = (
            reconstructed_error_bound is not None
            and name in PRIMARY_STANDARDIZED_MOMENT_NAMES
            and math.isfinite(reconstructed_error_bound)
            and reconstructed_error_bound >= 0.0
        )
        writer_uncertainty_bound_applied = supported_error_bound and math.isfinite(expected) and math.isfinite(actual)
        allowed_absolute_difference = atol + rtol * abs(expected)
        if writer_uncertainty_bound_applied:
            allowed_absolute_difference += reconstructed_error_bound
        absolute = 0.0 if paired_nan else abs(actual - expected)
        relative = 0.0 if paired_nan else absolute / max(abs(expected), atol)
        rows.append({
            "case": case,
            "quantity": name,
            "direct": expected,
            "cbin": actual,
            "absolute_difference": absolute,
            "relative_difference": relative,
            "paired_nan_unavailable": paired_nan,
            "precision_limited_unavailable": precision_limited,
            "reconstructed_error_bound": reconstructed_error_bound,
            "writer_uncertainty_bound_applied": writer_uncertainty_bound_applied,
            "passed": paired_nan or precision_limited or absolute <= allowed_absolute_difference,
        })
    return rows


def run_validation(data_root: Path, output_dir: Path, target_time: float, input_file: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "VALIDATION_COMPLETE.json").unlink(missing_ok=True)
    primary_snapshots = {
        scale: discover_snapshot(data_root, "mhd_u_bcc", scale, target_time=target_time)
        for scale in (40, 80, 160)
    }
    cycles = {snapshot.cycle for snapshot in primary_snapshots.values()}
    times = {snapshot.time for snapshot in primary_snapshots.values()}
    if len(cycles) != 1 or any(not math.isclose(value, target_time, abs_tol=5.0e-6) for value in times):
        raise ValueError("cbin validation snapshots are not synchronized")
    cycle = next(iter(cycles))
    full_snapshot = _discover_full_snapshot(data_root, target_time, cycle)
    full_basename = Path(full_snapshot.path).name
    gamma = _read_gamma(input_file)
    cases = (
        ("single_40", 40, (80, 120, 120, 160, 200, 240)),
        ("single_80", 80, (80, 160, 160, 240, 160, 240)),
        ("single_160", 160, (0, 160, 160, 320, 160, 320)),
        ("cross_x_rank_boundary", 80, (80, 240, 80, 160, 80, 160)),
        ("cross_y_rank_boundary", 80, (0, 160, 240, 400, 80, 160)),
        ("cross_z_rank_boundary", 80, (0, 160, 80, 160, 240, 400)),
        ("merged_cross_rank", 40, (80, 400, 240, 560, 400, 720)),
    )
    all_rows: list[dict[str, object]] = []
    case_reports: list[dict[str, object]] = []
    direct_primary: dict[str, dict[str, float]] = {}
    reconstructed_primary: dict[str, dict[str, float]] = {}
    for case, scale, bounds in cases:
        log(f"validating {case}: scale={scale}, bounds={bounds}")
        direct, cells = direct_primary_moments(data_root, full_basename, bounds)
        reconstructed, voxels, reconstructed_error_bounds = cbin_region_means(
            data_root,
            "mhd_u_bcc",
            scale,
            Path(primary_snapshots[scale].path).name,
            bounds,
            PRIMARY_LABELS,
            return_error_bounds=True,
        )
        rows = comparison_rows(case, direct, reconstructed, atol=2.0e-5, rtol=2.0e-4)
        all_rows.extend(rows)
        direct_primary[case] = direct
        reconstructed_primary[case] = reconstructed
        direct_mag = magnetic_diagnostics(direct)
        reconstructed_mag = magnetic_diagnostics(reconstructed)
        magnetic_rows = comparison_rows(f"{case}:magnetic", direct_mag, reconstructed_mag, atol=2.0e-5, rtol=2.0e-4)
        all_rows.extend(magnetic_rows)
        direct_retained = retained_primary_diagnostics(direct)
        reconstructed_precision_limited: set[str] = set()
        reconstructed_diagnostic_error_bounds: dict[str, float] = {}
        reconstructed_retained = retained_primary_diagnostics(
            reconstructed,
            raw_moment_error_bounds=reconstructed_error_bounds,
            precision_limited_unavailable=reconstructed_precision_limited,
            diagnostic_error_bounds=reconstructed_diagnostic_error_bounds,
        )
        retained_rows = comparison_rows(
            f"{case}:retained-primary",
            direct_retained,
            reconstructed_retained,
            atol=5.0e-4,
            rtol=5.0e-3,
            precision_limited_unavailable=reconstructed_precision_limited,
            paired_nan_allowed_names=reconstructed_precision_limited,
            reconstructed_error_bounds=reconstructed_diagnostic_error_bounds,
        )
        all_rows.extend(retained_rows)
        case_reports.append({
            "case": case,
            "scale": scale,
            "bounds": list(bounds),
            "direct_cells": cells,
            "cbin_voxels": voxels,
            "max_relative_difference": max(float(row["relative_difference"]) for row in rows),
            "passed": all(bool(row["passed"]) for row in rows + magnetic_rows + retained_rows),
        })

    hierarchy_bounds = (0, 160, 0, 160, 0, 160)
    hierarchy = {
        scale: cbin_region_means(
            data_root,
            "mhd_u_bcc",
            scale,
            Path(snapshot.path).name,
            hierarchy_bounds,
            PRIMARY_LABELS,
        )[0]
        for scale, snapshot in primary_snapshots.items()
    }
    hierarchy_rows: list[dict[str, object]] = []
    for scale in (80, 160):
        hierarchy_rows.extend(comparison_rows(
            f"hierarchy_40_vs_{scale}",
            hierarchy[40],
            hierarchy[scale],
            atol=2.0e-5,
            rtol=2.0e-4,
        ))
    all_rows.extend(hierarchy_rows)

    failed = [row for row in all_rows if not bool(row["passed"])]
    payload = {
        "data_root": str(data_root),
        "target_time": target_time,
        "cycle": cycle,
        "gamma": gamma,
        "input_file": str(input_file),
        "full_resolution_basename": full_basename,
        "cases": case_reports,
        "comparison_count": len(all_rows),
        "failed_comparison_count": len(failed),
        "precision_limited_unavailable_count": sum(
            bool(row["precision_limited_unavailable"])
            for row in all_rows
        ),
        "writer_uncertainty_bounded_comparison_count": sum(
            bool(row["writer_uncertainty_bound_applied"])
            for row in all_rows
        ),
        "passed": not failed,
        "explicitly_unavailable": [
            "volume-weighted primitive mean velocity",
            "velocity dispersion",
            "sonic Mach number",
            "Alfvenic Mach number",
            "kinetic-energy partitions",
            "pressure",
            "mixed-field statistics",
        ],
        "rows": all_rows,
    }
    (output_dir / "validation_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# cbin Reconstruction Validation",
        "",
        f"Overall result: **{'PASS' if payload['passed'] else 'FAIL'}**",
        "",
        "| Case | Scale | Direct cells | cbin voxels | Max raw-moment relative difference | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in case_reports:
        lines.append(
            f"| {case['case']} | {case['scale']} | {case['direct_cells']} | {case['cbin_voxels']} "
            f"| {case['max_relative_difference']:.6g} | {'PASS' if case['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "Raw first-through-fourth moments are compared before central-moment conversion. "
        "The row-level results then compare retained primary-only variances, standard deviations, "
        "resolved skewnesses, resolved kurtoses, density contrast, mass-weighted mean velocities, "
        "magnetic energy, and labeled Alfven-speed proxies. "
        "The full-resolution primitive fields are converted to conserved momentum and total MHD "
        "energy before comparison. The merged cross-rank case crosses mesh-block boundaries in "
        "all three coordinate directions.",
        "",
        "Cancellation-dominated standardized moments are explicitly reported as precision-limited "
        "unavailable when propagated float32 writer bounds do not support a stable value.",
        "",
        "Finite skewness and kurtosis differences are accepted only when they fit within the "
        "field-specific interval-normalized float32 writer uncertainty bound, including numerator "
        "and variance-denominator uncertainty.",
        "",
        "The row-level results also compare factor-40 cbin averages against factor-80 and "
        "factor-160 cbin values over the same aligned region.",
        "",
        "Volume-weighted primitive velocity means, velocity dispersions, Mach numbers, kinetic "
        "partitions, pressure, and mixed-field statistics remain explicitly unavailable from the "
        "trusted primary cbin schema.",
        "",
        "See `validation_results.json` for absolute and relative differences for every quantity.",
    ])
    (output_dir / "validation_report.md").write_text("\n".join(lines) + "\n")
    if failed:
        raise RuntimeError(f"validation failed for {len(failed)} comparisons")
    scripts_dir = Path(__file__).resolve().parent
    token = {
        "passed": True,
        "data_root": str(data_root.resolve()),
        "target_time": target_time,
        "cycle": cycle,
        "full_resolution_basename": full_basename,
        "primary_snapshot_identities": {
            str(scale): shard_identity(snapshot)
            for scale, snapshot in primary_snapshots.items()
        },
        "full_snapshot_identity": shard_identity(full_snapshot),
        "gamma": gamma,
        "input_file_sha256": file_sha256(input_file),
        "source_hashes": {
            "cbin_tools.py": file_sha256(scripts_dir / "cbin_tools.py"),
            "validate_reconstruction.py": file_sha256(Path(__file__)),
        },
        "validation_results_sha256": file_sha256(output_dir / "validation_results.json"),
    }
    temporary = output_dir / "VALIDATION_COMPLETE.json.tmp"
    temporary.write_text(json.dumps(token, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_dir / "VALIDATION_COMPLETE.json")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time", type=float, default=6.0)
    parser.add_argument("--input-file", default=INPUT_FILE_DEFAULT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_validation(Path(args.data_root), Path(args.output_dir), args.time, Path(args.input_file))
    log(f"validation passed with {payload['comparison_count']} comparisons")


if __name__ == "__main__":
    main()
