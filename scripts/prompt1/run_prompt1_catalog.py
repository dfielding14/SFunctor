#!/usr/bin/env python3
"""Build, analyze, and summarize the Prompt 1 environmental catalogs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbin_tools import (  # noqa: E402
    DOMAIN_CELLS,
    EXPECTED_RANKS,
    MESH_BLOCK_CELLS,
    PRIMARY_BASES,
    PRIMARY_LABELS,
    aggregate_blocks,
    aggregate_raw_moment_error_bounds,
    assemble_product,
    discover_snapshot,
    file_sha256,
    finite_quantiles,
    flatten_columns,
    load_array_cache,
    raw_moment_statistics,
    required_rank_ids,
    robust_nonnegative,
    save_array_cache,
    shard_identity,
    source_shard_inventory_sha256,
    source_raw_moment_error_bounds,
)


DATA_ROOT_DEFAULT = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "data/data_Turb_10240_beta25_dedt025_plm"
)
RESULTS_ROOT_DEFAULT = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "prompt1_catalog"
)
L_SUB_VALUES = (80, 160, 320, 640, 1280)
SOURCE_SCALE = 80
GAMMA = 1.00001
INPUT_FILE_DEFAULT = (
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "inputs/Turb_10240_beta25_dedt025_plm.athinput"
)
QUANTILES = (0.0, 0.01, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.99, 1.0)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: Mapping[str, object] | Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    scripts_dir = Path(__file__).resolve().parent
    return {
        "cbin_tools.py": file_sha256(scripts_dir / "cbin_tools.py"),
        "run_prompt1_catalog.py": file_sha256(scripts_dir / "run_prompt1_catalog.py"),
    }


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


def _require_validation_token(
    run_dir: Path,
    target_time: float,
    input_file: Path,
    data_root: Path,
) -> dict[str, object]:
    path = run_dir / "validation" / "VALIDATION_COMPLETE.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing passing validation token {path}")
    token = json.loads(path.read_text())
    if not token.get("passed"):
        raise ValueError(f"{path}: validation did not pass")
    if not math.isclose(float(token["target_time"]), target_time, abs_tol=5.0e-6):
        raise ValueError(f"{path}: validation target-time mismatch")
    if token.get("data_root") != str(data_root.resolve()):
        raise ValueError(f"{path}: validated data-root mismatch")
    if token.get("input_file_sha256") != file_sha256(input_file):
        raise ValueError(f"{path}: validated simulation input hash is stale")
    if not math.isclose(float(token["gamma"]), _read_gamma(input_file), abs_tol=0.0):
        raise ValueError(f"{path}: validated gamma is stale")
    results_path = run_dir / "validation" / "validation_results.json"
    if token.get("validation_results_sha256") != file_sha256(results_path):
        raise ValueError(f"{path}: validation-results hash mismatch")
    validator = Path(__file__).resolve().parent / "validate_reconstruction.py"
    expected = {
        "cbin_tools.py": file_sha256(Path(__file__).resolve().parent / "cbin_tools.py"),
        "validate_reconstruction.py": file_sha256(validator),
    }
    if token.get("source_hashes") != expected:
        raise ValueError(f"{path}: validation source hashes are stale")
    return token


def _require_manifest(path: Path, expected: Mapping[str, object]) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing cache manifest {path}")
    manifest = json.loads(path.read_text())
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: stale cache manifest fields {mismatches}")
    return manifest


def artifact_graph(run_dir: Path) -> dict[str, object]:
    """Hash the complete validated cache and catalog graph used by analysis."""

    paths = _cache_paths(run_dir)
    catalogs_dir = run_dir / "catalogs"
    return {
        "validation_token_sha256": file_sha256(run_dir / "validation" / "VALIDATION_COMPLETE.json"),
        "rank_map_sha256": file_sha256(paths["rank_map"]),
        "source_hashes": _source_hashes(),
        "raw_caches": {
            "primary": {
                "cache_sha256": file_sha256(paths["primary"]),
                "manifest_sha256": file_sha256(paths["primary_manifest"]),
            },
        },
        "catalogs": {
            str(l_sub): {
                "catalog_sha256": file_sha256(catalogs_dir / f"catalog_L{l_sub}.npz"),
                "manifest_sha256": file_sha256(catalogs_dir / f"catalog_L{l_sub}_manifest.json"),
                "complete_sha256": file_sha256(catalogs_dir / f"catalog_L{l_sub}.complete"),
            }
            for l_sub in L_SUB_VALUES
        },
    }


def artifact_graph_sha256(graph: Mapping[str, object]) -> str:
    encoded = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_paths(run_dir: Path) -> dict[str, Path]:
    cache_dir = run_dir / "cache"
    return {
        "primary": cache_dir / "raw_mhd_u_bcc_80.npz",
        "rank_map": cache_dir / "rank_map.npy",
        "primary_manifest": cache_dir / "raw_mhd_u_bcc_80_manifest.json",
    }


def command_probe(args: argparse.Namespace) -> None:
    """Read representative rank-zero headers before any full scan."""

    data_root = Path(args.data_root)
    payload: dict[str, object] = {
        "data_root": str(data_root),
        "target_time": args.time,
        "expected_ranks": EXPECTED_RANKS,
        "domain_cells": DOMAIN_CELLS,
        "mesh_block_cells_x1_x2_x3": list(MESH_BLOCK_CELLS),
        "products": {},
    }
    products: dict[str, object] = {}
    for product, scales in (("mhd_u_bcc", (40, 80, 160)),):
        for scale in scales:
            shard = discover_snapshot(data_root, product, scale, target_time=args.time)
            products[f"{product}_{scale}"] = {
                "rank0_path": shard.path,
                "time": shard.time,
                "cycle": shard.cycle,
                "factor": shard.coarsen_factor,
                "moments": shard.number_of_moments,
                "var_names": list(shard.var_names),
                "coarse_grid_cells_x1_x2_x3": list(shard.coarse_grid_cells),
                "coarse_block_cells_x1_x2_x3": list(shard.coarse_block_cells),
                "local_shape_kji": list(shard.records[0].shape_kji),
                "rank0_logical": list(shard.records[0].logical),
                "rank0_geometry": list(shard.records[0].geometry),
                "rank0_file_size": shard.file_size,
            }
    payload["products"] = products
    write_json(Path(args.output), payload)
    log(f"wrote representative data-layout probe to {args.output}")


def ensure_raw_caches(
    data_root: Path,
    run_dir: Path,
    target_time: float,
    validation_token: Mapping[str, object],
    validation_token_sha256: str,
) -> tuple[Path, Path]:
    paths = _cache_paths(run_dir)
    rank_map_path = paths["rank_map"]
    rank_map_path.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes()
    primary_rank0 = discover_snapshot(data_root, "mhd_u_bcc", SOURCE_SCALE, target_time=target_time)
    if validation_token["primary_snapshot_identities"][str(SOURCE_SCALE)] != shard_identity(primary_rank0):
        raise ValueError("validated primary snapshot identity does not match census input")
    if primary_rank0.var_names != PRIMARY_LABELS:
        raise ValueError(f"unexpected primary schema: {primary_rank0.var_names}")
    if int(validation_token["cycle"]) != primary_rank0.cycle:
        raise ValueError("validation token cycle does not match census inputs")
    common = {
        "data_root": str(data_root.resolve()),
        "scale": SOURCE_SCALE,
        "time": primary_rank0.time,
        "cycle": primary_rank0.cycle,
        "source_hashes": source_hashes,
        "validation_token_sha256": validation_token_sha256,
    }
    primary_expected = {
        **common,
        "product": "mhd_u_bcc",
        "basename": Path(primary_rank0.path).name,
        "fields": list(PRIMARY_LABELS),
        "var_names": list(PRIMARY_LABELS),
    }

    if not paths["primary"].is_file():
        log(f"assembling strict primary cache from {Path(primary_rank0.path).name}")
        arrays, rank_map, manifest = assemble_product(
            data_root,
            "mhd_u_bcc",
            SOURCE_SCALE,
            Path(primary_rank0.path).name,
            PRIMARY_LABELS,
            expected_time=primary_rank0.time,
            expected_cycle=primary_rank0.cycle,
            expected_var_names=PRIMARY_LABELS,
        )
        np.save(rank_map_path, rank_map)
        manifest.update(primary_expected)
        manifest["rank_map_sha256"] = file_sha256(rank_map_path)
        save_array_cache(paths["primary"], arrays, manifest)
        manifest["cache_sha256"] = file_sha256(paths["primary"])
        write_json(paths["primary_manifest"], manifest)
        log(f"wrote {paths['primary']}")
        del arrays, rank_map
        gc.collect()
    else:
        manifest = _require_manifest(paths["primary_manifest"], primary_expected)
        if manifest.get("cache_sha256") != file_sha256(paths["primary"]):
            raise ValueError(f"{paths['primary']}: cache hash mismatch")
        if not rank_map_path.is_file() or manifest.get("rank_map_sha256") != file_sha256(rank_map_path):
            raise ValueError(f"{rank_map_path}: rank-map hash mismatch")
        current_inventory = source_shard_inventory_sha256(
            data_root, "mhd_u_bcc", SOURCE_SCALE, Path(primary_rank0.path).name
        )
        if manifest.get("source_shard_inventory_sha256") != current_inventory:
            raise ValueError(f"{paths['primary']}: source-shard inventory changed")
        log(f"reusing {paths['primary']}")

    if not rank_map_path.is_file():
        raise FileNotFoundError(f"missing expected rank map {rank_map_path}")
    return paths["primary"], rank_map_path


def _compact(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "f":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(values, dtype=np.float32)
    return np.asarray(values)


def _grid_columns(l_sub: int) -> dict[str, np.ndarray]:
    n = DOMAIN_CELLS // l_sub
    if DOMAIN_CELLS % l_sub:
        raise ValueError(f"L_sub={l_sub} does not tile domain={DOMAIN_CELLS}")
    k, j, i = np.indices((n, n, n), dtype=np.int16)
    i0, j0, k0 = i.astype(np.int32) * l_sub, j.astype(np.int32) * l_sub, k.astype(np.int32) * l_sub
    i1, j1, k1 = i0 + l_sub, j0 + l_sub, k0 + l_sub
    source_factor = l_sub // SOURCE_SCALE
    columns = {
        "subvolume_id": np.arange(n**3, dtype=np.int32).reshape(n, n, n),
        "L_sub": np.full((n, n, n), l_sub, dtype=np.int16),
        "grid_i": i,
        "grid_j": j,
        "grid_k": k,
        "cell_i0": i0,
        "cell_i1": i1,
        "cell_j0": j0,
        "cell_j1": j1,
        "cell_k0": k0,
        "cell_k1": k1,
        "center_i": (i0 + i1).astype(np.float32) * 0.5,
        "center_j": (j0 + j1).astype(np.float32) * 0.5,
        "center_k": (k0 + k1).astype(np.float32) * 0.5,
        "center_x1": -0.5 + (i0 + i1).astype(np.float32) / (2.0 * DOMAIN_CELLS),
        "center_x2": -0.5 + (j0 + j1).astype(np.float32) / (2.0 * DOMAIN_CELLS),
        "center_x3": -0.5 + (k0 + k1).astype(np.float32) / (2.0 * DOMAIN_CELLS),
        "x1_min": -0.5 + i0.astype(np.float32) / DOMAIN_CELLS,
        "x1_max": -0.5 + i1.astype(np.float32) / DOMAIN_CELLS,
        "x2_min": -0.5 + j0.astype(np.float32) / DOMAIN_CELLS,
        "x2_max": -0.5 + j1.astype(np.float32) / DOMAIN_CELLS,
        "x3_min": -0.5 + k0.astype(np.float32) / DOMAIN_CELLS,
        "x3_max": -0.5 + k1.astype(np.float32) / DOMAIN_CELLS,
        "source_cbin_scale": np.full((n, n, n), SOURCE_SCALE, dtype=np.int16),
        "source_cbin_i0": i.astype(np.int16) * source_factor,
        "source_cbin_i1": (i.astype(np.int16) + 1) * source_factor,
        "source_cbin_j0": j.astype(np.int16) * source_factor,
        "source_cbin_j1": (j.astype(np.int16) + 1) * source_factor,
        "source_cbin_k0": k.astype(np.int16) * source_factor,
        "source_cbin_k1": (k.astype(np.int16) + 1) * source_factor,
        "rank_x1_0": (i0 // MESH_BLOCK_CELLS[0]).astype(np.int16),
        "rank_x1_1": np.ceil(i1 / MESH_BLOCK_CELLS[0]).astype(np.int16),
        "rank_x2_0": (j0 // MESH_BLOCK_CELLS[1]).astype(np.int16),
        "rank_x2_1": np.ceil(j1 / MESH_BLOCK_CELLS[1]).astype(np.int16),
        "rank_x3_0": (k0 // MESH_BLOCK_CELLS[2]).astype(np.int16),
        "rank_x3_1": np.ceil(k1 / MESH_BLOCK_CELLS[2]).astype(np.int16),
        "wraps_periodic_boundary": np.zeros((n, n, n), dtype=np.uint8),
    }
    if l_sub < max(L_SUB_VALUES):
        parent_n = DOMAIN_CELLS // (2 * l_sub)
        columns["parent_L_sub"] = np.full((n, n, n), 2 * l_sub, dtype=np.int16)
        columns["parent_subvolume_id"] = (
            (k.astype(np.int32) // 2 * parent_n + j.astype(np.int32) // 2) * parent_n
            + i.astype(np.int32) // 2
        ).astype(np.int32)
    else:
        columns["parent_L_sub"] = np.full((n, n, n), -1, dtype=np.int16)
        columns["parent_subvolume_id"] = np.full((n, n, n), -1, dtype=np.int32)
    columns["required_rank_count"] = (
        (columns["rank_x1_1"] - columns["rank_x1_0"])
        * (columns["rank_x2_1"] - columns["rank_x2_0"])
        * (columns["rank_x3_1"] - columns["rank_x3_0"])
    ).astype(np.int16)
    return columns


def _aggregate_named(raw: Mapping[str, np.ndarray], factor: int) -> dict[str, np.ndarray]:
    return {name: aggregate_blocks(values, factor) for name, values in raw.items()}


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    denominator_error_bound: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    flags = np.zeros(numerator.shape, dtype=np.uint8)
    threshold = 64.0 * np.finfo(np.float32).eps * np.maximum(np.abs(denominator), 1.0)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > threshold)
    if denominator_error_bound is not None:
        propagated_bound = np.asarray(denominator_error_bound, dtype=np.float64)
        if propagated_bound.shape != denominator.shape:
            raise ValueError("denominator absolute-error bound must match denominator")
        unresolved = np.isfinite(denominator) & (np.abs(denominator) <= propagated_bound)
        flags[unresolved] |= 2
        valid &= ~unresolved
    result[valid] = numerator[valid] / denominator[valid]
    flags[~valid] |= 1
    return result, flags


def derive_catalog(
    raw_primary: Mapping[str, np.ndarray],
    l_sub: int,
    *,
    gamma: float = GAMMA,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Derive exact and explicitly named proxy quantities for one L_sub."""

    if l_sub not in L_SUB_VALUES:
        raise ValueError(f"unsupported L_sub={l_sub}")
    factor = l_sub // SOURCE_SCALE
    columns = _grid_columns(l_sub)
    primary = _aggregate_named(raw_primary, factor)
    metadata: dict[str, object] = {
        "L_sub": l_sub,
        "source_cbin_scale": SOURCE_SCALE,
        "constituent_voxels_per_axis": factor,
        "constituent_voxels_per_subvolume": factor**3,
        "subvolume_count": (DOMAIN_CELLS // l_sub) ** 3,
        "tiled_domain_fraction": 1.0,
        "gamma": gamma,
        "array_order": "[k,j,i] = [x3,x2,x1]",
        "physics_caveat": (
            "Primary cbin stores conserved momentum, not primitive velocity. "
            "SGS products are intentionally excluded. Velocity dispersions, Mach numbers, "
            "kinetic partitions, pressure, and mixed-field statistics are unavailable."
        ),
    }

    primary_stats: dict[str, dict[str, np.ndarray]] = {}
    raw_error_bounds_by_base: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for base in PRIMARY_BASES:
        raw_moments = tuple(
            raw_primary[f"{base}_{suffix}"]
            for suffix in ("1st", "2nd", "3rd", "4th")
        )
        source_bounds = source_raw_moment_error_bounds(
            *raw_moments,
            source_cells=SOURCE_SCALE**3,
        )
        aggregated_bounds = aggregate_raw_moment_error_bounds(raw_moments, source_bounds, factor)
        raw_error_bounds_by_base[base] = aggregated_bounds
        stats = raw_moment_statistics(
            primary[f"{base}_1st"],
            primary[f"{base}_2nd"],
            primary[f"{base}_3rd"],
            primary[f"{base}_4th"],
            abs_error_bounds=aggregated_bounds,
        )
        primary_stats[base] = stats
        for stat in (
            "mean", "variance", "variance_error_bound", "sigma", "central_moment_3", "central_moment_4",
            "central_moment_3_error_bound", "central_moment_4_error_bound",
            "skewness", "skewness_error_bound", "kurtosis", "kurtosis_error_bound",
            "excess_kurtosis", "moment_flags",
        ):
            columns[f"{base}_{stat}"] = _compact(stats[stat])

    bx, by, bz = (primary[f"bcc{axis}_1st"] for axis in (1, 2, 3))
    bx2, by2, bz2 = (primary[f"bcc{axis}_2nd"] for axis in (1, 2, 3))
    b_mean_sq = bx * bx + by * by + bz * bz
    b2_mean = bx2 + by2 + bz2
    delta_b_sq_error_bound = sum(
        primary_stats[f"bcc{axis}"]["variance_error_bound"]
        for axis in (1, 2, 3)
    )
    delta_b_sq, delta_b_flags = robust_nonnegative(
        b2_mean - b_mean_sq,
        b2_mean,
        abs_error_bound=delta_b_sq_error_bound,
    )
    b_mean = np.sqrt(b_mean_sq)
    b_mean_error_bound = np.sqrt(sum(
        raw_error_bounds_by_base[f"bcc{axis}"][0] ** 2
        for axis in (1, 2, 3)
    ))
    delta_b = np.sqrt(delta_b_sq)
    dBB, dBB_flags = _safe_ratio(delta_b, b_mean, denominator_error_bound=b_mean_error_bound)
    bmean_fraction, bmean_fraction_flags = _safe_ratio(b_mean_sq, b2_mean)
    delta_b_fraction, delta_b_fraction_flags = _safe_ratio(delta_b_sq, b2_mean)
    columns.update({
        "B_mean": _compact(b_mean),
        "B_mean_error_bound": _compact(b_mean_error_bound),
        "B_rms": _compact(np.sqrt(b2_mean)),
        "B2_mean": _compact(b2_mean),
        "deltaB": _compact(delta_b),
        "deltaB_sq": _compact(delta_b_sq),
        "deltaB_sq_error_bound": _compact(delta_b_sq_error_bound),
        "dBB": _compact(dBB),
        "B_mean_sq_over_B2_mean": _compact(bmean_fraction),
        "deltaB_sq_over_B2_mean": _compact(delta_b_fraction),
        "deltaB_flags": delta_b_flags,
        "dBB_flags": dBB_flags,
        "B_mean_fraction_flags": bmean_fraction_flags,
        "deltaB_fraction_flags": delta_b_fraction_flags,
    })
    rho_cv, rho_cv_flags = _safe_ratio(columns["dens_sigma"].astype(np.float64), columns["dens_mean"].astype(np.float64))
    columns["rho_sigma_over_mean"] = _compact(rho_cv)
    columns["rho_sigma_over_mean_flags"] = rho_cv_flags

    magnetic_energy = 0.5 * b2_mean
    rho = columns["dens_mean"].astype(np.float64)
    positive_rho = np.where(np.isfinite(rho) & (rho > 0.0), rho, np.nan)
    u_mass_weighted = [
        primary[f"mom{axis}_1st"] / positive_rho
        for axis in (1, 2, 3)
    ]
    vA_mean_proxy = b_mean / np.sqrt(positive_rho)
    vA_rms_like_proxy = np.sqrt(b2_mean / positive_rho)
    for axis, values in zip(("x", "y", "z"), u_mass_weighted):
        columns[f"u_mass_weighted_mean_{axis}"] = _compact(values)
    columns.update({
        "magnetic_energy_mean": _compact(magnetic_energy),
        "vA_mean_proxy": _compact(vA_mean_proxy),
        "vA_rms_like_proxy": _compact(vA_rms_like_proxy),
    })
    flag_columns = (
        *(f"{base}_moment_flags" for base in PRIMARY_BASES),
        "deltaB_flags",
        "dBB_flags",
        "B_mean_fraction_flags",
        "deltaB_fraction_flags",
        "rho_sigma_over_mean_flags",
    )
    catalog_validity_flags = np.zeros_like(columns["dBB_flags"], dtype=np.uint32)
    metadata["catalog_validity_flag_bits"] = {}
    for bit, name in enumerate(flag_columns):
        catalog_validity_flags |= (np.asarray(columns[name]) != 0).astype(np.uint32) << bit
        metadata["catalog_validity_flag_bits"][name] = bit
    columns["catalog_validity_flags"] = catalog_validity_flags
    return flatten_columns(columns), metadata


def command_build(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_marker in (
        run_dir / "BUILD_COMPLETE.json",
        run_dir / "verification" / "VERIFY_COMPLETE.json",
        run_dir / "analysis" / "ANALYSIS_COMPLETE",
        run_dir / "analysis" / "ANALYSIS_COMPLETE.json",
        run_dir / "analysis" / "analysis_manifest.json",
    ):
        stale_marker.unlink(missing_ok=True)
    input_file = Path(args.input_file)
    validation_token_path = run_dir / "validation" / "VALIDATION_COMPLETE.json"
    validation_token = _require_validation_token(run_dir, args.time, input_file, data_root)
    validation_token_sha256 = file_sha256(validation_token_path)
    gamma = _read_gamma(input_file)
    primary_path, _ = ensure_raw_caches(
        data_root,
        run_dir,
        args.time,
        validation_token,
        validation_token_sha256,
    )
    raw_primary, primary_metadata = load_array_cache(primary_path)
    if set(raw_primary) != set(PRIMARY_LABELS):
        raise ValueError(f"{primary_path}: raw cache fields do not match primary schema")
    run_metadata: dict[str, object] = {
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "target_time": args.time,
        "gamma": gamma,
        "input_file": args.input_file,
        "domain_cells": DOMAIN_CELLS,
        "L_sub_values": list(L_SUB_VALUES),
        "source_hashes": _source_hashes(),
        "validation_token_sha256": validation_token_sha256,
        "primary_cache_metadata": primary_metadata,
    }
    catalogs_dir = run_dir / "catalogs"
    catalogs_dir.mkdir(parents=True, exist_ok=True)
    primary_manifest = json.loads(_cache_paths(run_dir)["primary_manifest"].read_text())
    for l_sub in L_SUB_VALUES:
        output = catalogs_dir / f"catalog_L{l_sub}.npz"
        marker = catalogs_dir / f"catalog_L{l_sub}.complete"
        manifest_path = catalogs_dir / f"catalog_L{l_sub}_manifest.json"
        expected_catalog_manifest = {
            "L_sub": l_sub,
            "primary_cache_sha256": primary_manifest["cache_sha256"],
            "source_hashes": _source_hashes(),
            "validation_token_sha256": run_metadata["validation_token_sha256"],
        }
        if output.is_file() and marker.is_file():
            catalog_manifest = _require_manifest(manifest_path, expected_catalog_manifest)
            if catalog_manifest.get("catalog_sha256") != file_sha256(output):
                raise ValueError(f"{output}: catalog hash mismatch")
            log(f"reusing complete catalog {output}")
            continue
        log(f"deriving L_sub={l_sub} catalog")
        columns, metadata = derive_catalog(raw_primary, l_sub, gamma=gamma)
        metadata["primary_cache"] = str(primary_path)
        metadata.update(expected_catalog_manifest)
        save_array_cache(output, columns, metadata)
        write_json(manifest_path, {**expected_catalog_manifest, "catalog_sha256": file_sha256(output)})
        marker.write_text(f"complete {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log(f"wrote {output} with {metadata['subvolume_count']} subvolumes")
        del columns
        gc.collect()
    graph = artifact_graph(run_dir)
    run_metadata["artifact_graph_sha256"] = artifact_graph_sha256(graph)
    write_json(run_dir / "run_metadata.json", run_metadata)
    write_json(run_dir / "BUILD_COMPLETE.json", run_metadata)
    log("catalog build complete")


SUMMARY_PROPERTIES = (
    "dBB",
    "B_mean",
    "B_rms",
    "deltaB",
    "B_mean_sq_over_B2_mean",
    "deltaB_sq_over_B2_mean",
    "magnetic_energy_mean",
    "vA_mean_proxy",
    "vA_rms_like_proxy",
    "u_mass_weighted_mean_x",
    "u_mass_weighted_mean_y",
    "u_mass_weighted_mean_z",
    "dens_mean",
    "rho_sigma_over_mean",
    "dens_skewness",
    "dens_kurtosis",
    "bcc1_skewness",
    "bcc1_kurtosis",
    "mom1_sigma",
    "mom2_sigma",
    "mom3_sigma",
    "ener_mean",
    "ener_sigma",
)
CORRELATION_PROPERTIES = (
    "dBB",
    "B_mean",
    "deltaB",
    "B_rms",
    "magnetic_energy_mean",
    "vA_mean_proxy",
    "vA_rms_like_proxy",
    "u_mass_weighted_mean_x",
    "u_mass_weighted_mean_y",
    "u_mass_weighted_mean_z",
    "dens_mean",
    "rho_sigma_over_mean",
    "dens_skewness",
    "dens_kurtosis",
    "mom1_sigma",
    "mom2_sigma",
    "mom3_sigma",
    "ener_mean",
)
PILOT_MATCH_PROPERTIES = (
    "dens_mean",
    "rho_sigma_over_mean",
    "B_rms",
    "deltaB",
    "mom1_sigma",
    "mom2_sigma",
    "mom3_sigma",
)
FULL_PRIMITIVE_RANK_BYTES = 524_352_934


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _histogram_edges(values: np.ndarray, count: int = 80) -> tuple[np.ndarray | None, bool]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None, False
    low, high = np.quantile(values, (0.001, 0.999))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return None, False
    log_scale = low > 0.0 and high / low > 30.0
    edges = np.geomspace(low, high, count) if log_scale else np.linspace(low, high, count)
    return edges, log_scale


def _load_catalog(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    return load_array_cache(path)


def _spearman_matrix(columns: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    matrix = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    for row, row_name in enumerate(names):
        for column, column_name in enumerate(names):
            matrix[row, column] = _safe_spearman(columns[row_name], columns[column_name])
    return matrix


def _analysis_preflight(catalogs_dir: Path) -> list[dict[str, object]]:
    required = set(SUMMARY_PROPERTIES) | set(CORRELATION_PROPERTIES) | {
        "subvolume_id", "catalog_validity_flags", "dBB_flags",
    }
    rows: list[dict[str, object]] = []
    for l_sub in L_SUB_VALUES:
        path = catalogs_dir / f"catalog_L{l_sub}.npz"
        marker = catalogs_dir / f"catalog_L{l_sub}.complete"
        if not path.is_file() or not marker.is_file():
            raise FileNotFoundError(f"missing completed catalog artifacts for L_sub={l_sub}")
        columns, metadata = _load_catalog(path)
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"{path}: missing analysis columns {missing}")
        expected = (DOMAIN_CELLS // l_sub) ** 3
        if int(metadata["subvolume_count"]) != expected:
            raise ValueError(f"{path}: metadata row count mismatch")
        for name in required:
            if columns[name].size != expected:
                raise ValueError(f"{path}: {name} row count mismatch")
        for name in sorted(set(SUMMARY_PROPERTIES) | set(CORRELATION_PROPERTIES)):
            values = columns[name]
            finite = values[np.isfinite(values)]
            rows.append({
                "L_sub": l_sub,
                "property": name,
                "row_count": expected,
                "finite_count": int(finite.size),
                "finite_fraction": float(finite.size / expected),
                "finite_min": float(np.min(finite)) if finite.size else float("nan"),
                "finite_max": float(np.max(finite)) if finite.size else float("nan"),
                "flagged_row_count": int(np.count_nonzero(columns["catalog_validity_flags"])),
                "constant_when_finite": bool(finite.size > 0 and np.ptp(finite) == 0.0),
                "analysis_status": "available" if finite.size >= 2 and np.ptp(finite) > 0.0 else "skipped",
            })
    dbb_rows = [row for row in rows if row["property"] == "dBB"]
    if any(row["analysis_status"] != "available" for row in dbb_rows):
        raise ValueError(f"dBB is unavailable or constant in analysis preflight: {dbb_rows}")
    return rows


def _make_summary_rows(catalogs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for l_sub in L_SUB_VALUES:
        columns, _ = _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")
        for name in SUMMARY_PROPERTIES:
            values = columns[name]
            quantiles = finite_quantiles(values, QUANTILES)
            finite = values[np.isfinite(values)]
            row: dict[str, object] = {
                "L_sub": l_sub,
                "property": name,
                "count": int(values.size),
                "finite_count": int(finite.size),
                "finite_fraction": float(finite.size / values.size),
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite)) if finite.size else float("nan"),
            }
            for quantile, value in zip(QUANTILES, quantiles):
                row[f"q{quantile:g}"] = value
            rows.append(row)
    return rows


def _set_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rc("font", family="serif")
    matplotlib.rc("mathtext", fontset="cm")
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True


def _plot_distributions(catalogs_dir: Path, plots_dir: Path) -> None:
    _set_plot_style()
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    groups = (
        ("magnetic", ("dBB", "B_mean", "B_rms", "deltaB", "B_mean_sq_over_B2_mean", "deltaB_sq_over_B2_mean")),
        ("density_magnetic", ("dens_mean", "rho_sigma_over_mean", "magnetic_energy_mean",
                              "vA_mean_proxy", "vA_rms_like_proxy", "ener_mean")),
        ("higher_conserved", ("dens_skewness", "dens_kurtosis", "bcc1_skewness", "bcc1_kurtosis",
                              "mom1_sigma", "ener_sigma")),
    )
    catalogs = {l_sub: _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")[0] for l_sub in L_SUB_VALUES}
    for group, properties in groups:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        for axis, name in zip(axes.flat, properties):
            for l_sub in L_SUB_VALUES:
                values = catalogs[l_sub][name]
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                edges, log_scale = _histogram_edges(values)
                if edges is None:
                    axis.text(0.05, 0.9, f"L={l_sub}: constant/skipped", transform=axis.transAxes, fontsize=7)
                    continue
                if log_scale:
                    axis.set_xscale("log")
                hist, edges = np.histogram(values, bins=edges, density=True)
                axis.plot(0.5 * (edges[:-1] + edges[1:]), hist, label=str(l_sub))
            axis.set_xlabel(name)
            axis.set_ylabel("PDF")
            axis.set_yscale("log")
        axes.flat[0].legend(title=r"$L_{\rm sub}/\Delta x$", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"distributions_{group}.png", dpi=180)
        plt.close(fig)


def _plot_dbb_correlations(catalogs_dir: Path, plots_dir: Path) -> None:
    _set_plot_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plots_dir.mkdir(parents=True, exist_ok=True)
    properties = tuple(name for name in CORRELATION_PROPERTIES if name != "dBB")
    catalogs = {l_sub: _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")[0] for l_sub in L_SUB_VALUES}
    for name in properties:
        fig, axes = plt.subplots(1, len(L_SUB_VALUES), figsize=(16, 3.1), sharey=True)
        for axis, l_sub in zip(axes, L_SUB_VALUES):
            x = catalogs[l_sub][name]
            y = catalogs[l_sub]["dBB"]
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if not x.size:
                continue
            xlo, xhi = np.quantile(x, (0.002, 0.998))
            ylo, yhi = np.quantile(y, (0.002, 0.998))
            if xlo == xhi or ylo == yhi:
                axis.text(0.05, 0.9, "constant/skipped", transform=axis.transAxes, fontsize=7)
                continue
            positive_x = xlo > 0 and xhi / xlo > 30
            positive_y = ylo > 0 and yhi / ylo > 30
            xedges = np.geomspace(xlo, xhi, 80) if positive_x else np.linspace(xlo, xhi, 80)
            yedges = np.geomspace(ylo, yhi, 80) if positive_y else np.linspace(ylo, yhi, 80)
            histogram, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            axis.pcolormesh(xedges, yedges, histogram.T, norm=LogNorm(vmin=1), shading="auto")
            if positive_x:
                axis.set_xscale("log")
            if positive_y:
                axis.set_yscale("log")
            axis.set_xlabel(name)
            axis.set_title(f"L={l_sub}")
        axes[0].set_ylabel("dBB")
        fig.tight_layout()
        fig.savefig(plots_dir / f"dBB_vs_{name}.png", dpi=180)
        plt.close(fig)


def _plot_correlation_matrices(catalogs_dir: Path, plots_dir: Path) -> list[dict[str, object]]:
    _set_plot_style()
    import matplotlib.pyplot as plt

    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, len(L_SUB_VALUES), figsize=(19, 4.2))
    for axis, l_sub in zip(axes, L_SUB_VALUES):
        columns, _ = _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")
        matrix = _spearman_matrix(columns, CORRELATION_PROPERTIES)
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_title(f"L={l_sub}")
        axis.set_xticks(range(len(CORRELATION_PROPERTIES)))
        axis.set_yticks(range(len(CORRELATION_PROPERTIES)))
        axis.set_xticklabels(CORRELATION_PROPERTIES, rotation=90, fontsize=6)
        axis.set_yticklabels(CORRELATION_PROPERTIES, fontsize=6)
        for row_index, row_name in enumerate(CORRELATION_PROPERTIES):
            for column_index, column_name in enumerate(CORRELATION_PROPERTIES):
                rows.append({
                    "L_sub": l_sub,
                    "property_1": row_name,
                    "property_2": column_name,
                    "spearman_r": float(matrix[row_index, column_index]),
                })
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.65, label="Spearman rank correlation")
    fig.savefig(plots_dir / "spearman_correlation_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rows


def _cross_scale_rows(catalogs_dir: Path) -> list[dict[str, object]]:
    base, _ = _load_catalog(catalogs_dir / "catalog_L80.npz")
    base_dbb = base["dBB"].reshape((128, 128, 128))
    rows: list[dict[str, object]] = []
    for l_sub in L_SUB_VALUES[1:]:
        coarse, _ = _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")
        factor = l_sub // 80
        child_mean = aggregate_blocks(base_dbb, factor).reshape(-1)
        child_blocks = base_dbb.reshape(128 // factor, factor, 128 // factor, factor, 128 // factor, factor)
        child_median = np.median(child_blocks, axis=(1, 3, 5)).reshape(-1)
        coarse_dbb = coarse["dBB"]
        valid = np.isfinite(coarse_dbb) & np.isfinite(child_mean) & np.isfinite(child_median)
        if not np.any(valid):
            raise ValueError(f"L_sub={l_sub}: no valid cross-scale dBB rows")
        rows.append({
            "L_sub": l_sub,
            "subvolume_count": int(np.count_nonzero(valid)),
            "coarse_dBB_vs_child_mean_spearman": _safe_spearman(coarse_dbb[valid], child_mean[valid]),
            "coarse_dBB_vs_child_median_spearman": _safe_spearman(coarse_dbb[valid], child_median[valid]),
            "coarse_dBB_median": float(np.median(coarse_dbb[valid])),
            "child_dBB_mean_median": float(np.median(child_mean[valid])),
            "child_dBB_median_median": float(np.median(child_median[valid])),
        })
    return rows


def _cross_scale_transition_rows(catalogs_dir: Path) -> list[dict[str, object]]:
    base, _ = _load_catalog(catalogs_dir / "catalog_L80.npz")
    base_dbb = base["dBB"].reshape((128, 128, 128))
    rows: list[dict[str, object]] = []
    labels = ("low", "middle", "high")
    for l_sub in L_SUB_VALUES[1:]:
        coarse, _ = _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")
        factor = l_sub // 80
        blocks = base_dbb.reshape(128 // factor, factor, 128 // factor, factor, 128 // factor, factor)
        child_median = np.nanmedian(blocks, axis=(1, 3, 5)).reshape(-1)
        coarse_dbb = coarse["dBB"]
        valid = np.isfinite(child_median) & np.isfinite(coarse_dbb)
        if not np.any(valid):
            raise ValueError(f"L_sub={l_sub}: no valid cross-scale transition rows")
        child_bounds = np.quantile(child_median[valid], (1.0 / 3.0, 2.0 / 3.0))
        coarse_bounds = np.quantile(coarse_dbb[valid], (1.0 / 3.0, 2.0 / 3.0))
        child_class = np.digitize(child_median[valid], child_bounds)
        coarse_class = np.digitize(coarse_dbb[valid], coarse_bounds)
        for child_index, child_label in enumerate(labels):
            for coarse_index, coarse_label in enumerate(labels):
                rows.append({
                    "L_sub": l_sub,
                    "child_median_regime": child_label,
                    "coarse_regime": coarse_label,
                    "count": int(np.count_nonzero((child_class == child_index) & (coarse_class == coarse_index))),
                    "valid_subvolume_count": int(np.count_nonzero(valid)),
                    "fraction_of_child_regime": float(
                        np.count_nonzero((child_class == child_index) & (coarse_class == coarse_index))
                        / max(1, np.count_nonzero(child_class == child_index))
                    ),
                })
    return rows


def _conditional_quantile_rows(catalogs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    properties = tuple(name for name in CORRELATION_PROPERTIES if name != "dBB")
    for l_sub in L_SUB_VALUES:
        columns, _ = _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")
        dBB = columns["dBB"]
        for name in properties:
            values = columns[name]
            finite = np.isfinite(dBB) & np.isfinite(values)
            if not np.any(finite):
                continue
            bounds = np.unique(np.quantile(values[finite], np.linspace(0.0, 1.0, 6)))
            if bounds.size < 2:
                continue
            for index in range(bounds.size - 1):
                inclusive_upper = index == bounds.size - 2
                selected = finite & (values >= bounds[index])
                selected &= values <= bounds[index + 1] if inclusive_upper else values < bounds[index + 1]
                selected_dbb = dBB[selected]
                rows.append({
                    "L_sub": l_sub,
                    "property": name,
                    "property_quantile_bin": index,
                    "property_min": float(bounds[index]),
                    "property_max": float(bounds[index + 1]),
                    "count": int(selected_dbb.size),
                    "dBB_q16": float(np.quantile(selected_dbb, 0.16)) if selected_dbb.size else float("nan"),
                    "dBB_median": float(np.median(selected_dbb)) if selected_dbb.size else float("nan"),
                    "dBB_q84": float(np.quantile(selected_dbb, 0.84)) if selected_dbb.size else float("nan"),
                })
    return rows


def _feature_matrix(columns: Mapping[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.column_stack([columns[name][indices] for name in PILOT_MATCH_PROPERTIES]).astype(np.float64)


def _scaled_feature_matrices(
    columns: Mapping[str, np.ndarray],
    low_indices: np.ndarray,
    high_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low = _feature_matrix(columns, low_indices)
    high = _feature_matrix(columns, high_indices)
    combined = np.vstack((low, high))
    median = np.median(combined, axis=0)
    mad = np.median(np.abs(combined - median), axis=0)
    scale = np.maximum(mad, 1.0e-12)
    return (low - median) / scale, (high - median) / scale


def _select_spatially_separated(
    columns: Mapping[str, np.ndarray],
    candidates: np.ndarray,
    *,
    count: int,
    minimum_distance_cells: float,
    existing: Sequence[int] = (),
) -> list[int]:
    centers = np.column_stack((columns["center_i"], columns["center_j"], columns["center_k"]))
    selected: list[int] = list(existing)
    added: list[int] = []
    for index in candidates:
        if int(index) in selected:
            continue
        if not selected:
            selected.append(int(index))
            added.append(int(index))
        else:
            delta = np.abs(centers[selected] - centers[index])
            delta = np.minimum(delta, DOMAIN_CELLS - delta)
            if np.all(np.sqrt(np.sum(delta * delta, axis=1)) >= minimum_distance_cells):
                selected.append(int(index))
                added.append(int(index))
        if len(added) >= count:
            break
    return added


def _pilot_rows(catalogs_dir: Path, rank_map: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select representative, matched, and outlier pilot candidates."""

    l_sub = 640
    columns, _ = _load_catalog(catalogs_dir / f"catalog_L{l_sub}.npz")
    dBB = columns["dBB"]
    pilot_valid = np.isfinite(dBB) & (columns["dBB_flags"] == 0)
    finite_indices = np.flatnonzero(pilot_valid)
    if not finite_indices.size:
        raise ValueError("no unflagged finite L_sub=640 dBB rows for pilot selection")
    q20, q40, q60, q80 = np.quantile(dBB[finite_indices], (0.2, 0.4, 0.6, 0.8))
    regime_masks = {
        "low_dBB": dBB <= q20,
        "near_median_dBB": (dBB >= q40) & (dBB <= q60),
        "high_dBB": dBB >= q80,
    }
    rows: list[dict[str, object]] = []
    chosen: set[int] = set()
    for regime, mask in regime_masks.items():
        target = float(np.median(dBB[mask]))
        candidates = np.flatnonzero(mask & pilot_valid)
        candidates = candidates[np.argsort(np.abs(dBB[candidates] - target))]
        selected = _select_spatially_separated(
            columns,
            candidates,
            count=4,
            minimum_distance_cells=1280.0,
            existing=sorted(chosen),
        )
        for index in selected:
            chosen.add(index)
            rows.append(_pilot_row(columns, rank_map, index, l_sub, f"representative:{regime}"))

    match_valid = pilot_valid.copy()
    for name in PILOT_MATCH_PROPERTIES:
        match_valid &= np.isfinite(columns[name])
    low = np.flatnonzero(regime_masks["low_dBB"] & match_valid)
    high = np.flatnonzero(regime_masks["high_dBB"] & match_valid)
    if not low.size or not high.size:
        raise ValueError("no finite low/high dBB candidates for matched pilot comparisons")
    # Match a bounded candidate pool to keep the pilot selector cheap and deterministic.
    low_pool = low[np.linspace(0, len(low) - 1, min(1000, len(low)), dtype=np.int64)]
    high_pool = high[np.linspace(0, len(high) - 1, min(1000, len(high)), dtype=np.int64)]
    low_matrix, high_matrix = _scaled_feature_matrices(columns, low_pool, high_pool)
    distances = np.sum((low_matrix[:, None, :] - high_matrix[None, :, :]) ** 2, axis=2)
    match_score_limit = 1.0
    accepted_pairs = 0
    for flat_index in np.argsort(distances, axis=None):
        low_pos, high_pos = np.unravel_index(flat_index, distances.shape)
        pair = (int(low_pool[low_pos]), int(high_pool[high_pos]))
        score = float(math.sqrt(distances[low_pos, high_pos] / len(PILOT_MATCH_PROPERTIES)))
        if score > match_score_limit:
            break
        separated = _select_spatially_separated(
            columns,
            np.asarray(pair),
            count=2,
            minimum_distance_cells=1280.0,
            existing=sorted(chosen),
        )
        if separated != list(pair):
            continue
        accepted_pairs += 1
        pair_id = accepted_pairs
        differences = {
            name: float(columns[name][pair[1]] - columns[name][pair[0]])
            for name in PILOT_MATCH_PROPERTIES
        }
        for index, side in zip(pair, ("low", "high")):
            chosen.add(index)
            rows.append(_pilot_row(
                columns,
                rank_map,
                index,
                l_sub,
                f"matched:{pair_id}:{side}",
                match_score=score,
                matched_pair_id=pair_id,
                match_differences=differences,
            ))
        if pair_id >= 3:
            break

    outlier_categories = {
        "outlier:small_B_mean": np.argsort(columns["B_mean"]),
        "outlier:large_deltaB": np.argsort(columns["deltaB"])[::-1],
        "outlier:large_dBB": np.argsort(np.nan_to_num(dBB, nan=-np.inf))[::-1],
    }
    for role, candidates in outlier_categories.items():
        for index in candidates:
            index = int(index)
            property_name = {
                "outlier:small_B_mean": "B_mean",
                "outlier:large_deltaB": "deltaB",
                "outlier:large_dBB": "dBB",
            }[role]
            if not pilot_valid[index] or not np.isfinite(columns[property_name][index]):
                continue
            selected = _select_spatially_separated(
                columns,
                np.asarray([index]),
                count=1,
                minimum_distance_cells=1280.0,
                existing=sorted(chosen),
            )
            if selected:
                chosen.add(index)
                rows.append(_pilot_row(columns, rank_map, index, l_sub, role))
                break
    metadata = {
        "primary_pilot_L_sub": l_sub,
        "regime_method": "L=640 dBB quantile regimes after census",
        "regime_bounds": {
            "q20": float(q20),
            "q40": float(q40),
            "q60": float(q60),
            "q80": float(q80),
        },
        "matched_pair_count": accepted_pairs,
        "matched_pair_shortfall": max(0, 3 - accepted_pairs),
        "match_properties": list(PILOT_MATCH_PROPERTIES),
        "match_score_limit_rms_mad_units": match_score_limit,
        "spatial_separation_cells": 1280.0,
        "eligibility": "finite dBB with dBB_flags == 0; matched pairs additionally require finite confounders",
        "considered_L_sub_values": list(L_SUB_VALUES),
        "scale_choice_rationale": (
            "The first extraction benchmark uses L_sub=640 because it balances environmental "
            "contrast, 4096 census candidates, and a tractable 16 primitive-rank-file read per "
            "region. The cbin census retains all five scales; expand the extraction pilot only "
            "after measuring cold-read cost for this initial size."
        ),
        "note": "Pilot proposal only. Full-resolution extraction and structure functions were not run.",
    }
    return rows, metadata


def _pilot_row(
    columns: Mapping[str, np.ndarray],
    rank_map: np.ndarray,
    index: int,
    l_sub: int,
    role: str,
    *,
    match_score: float | None = None,
    matched_pair_id: int | None = None,
    match_differences: Mapping[str, float] | None = None,
) -> dict[str, object]:
    bounds = [int(columns[name][index]) for name in ("cell_i0", "cell_i1", "cell_j0", "cell_j1", "cell_k0", "cell_k1")]
    ranks = required_rank_ids(bounds, rank_map)
    cells = l_sub**3
    bytes_for_eight_float32_fields = cells * 8 * 4
    names = (
        "dBB", "B_mean", "deltaB", "B_rms", "magnetic_energy_mean",
        "vA_mean_proxy", "vA_rms_like_proxy", "u_mass_weighted_mean_x",
        "u_mass_weighted_mean_y", "u_mass_weighted_mean_z", "dens_mean", "rho_sigma_over_mean",
        "dens_skewness", "dens_kurtosis", "mom1_sigma", "mom2_sigma", "mom3_sigma",
        "ener_mean", "ener_sigma",
    )
    row: dict[str, object] = {
        "pilot_id": f"L{l_sub}_sub{int(columns['subvolume_id'][index]):05d}",
        "role": role,
        "L_sub": l_sub,
        "cell_i0": bounds[0],
        "cell_i1": bounds[1],
        "cell_j0": bounds[2],
        "cell_j1": bounds[3],
        "cell_k0": bounds[4],
        "cell_k1": bounds[5],
        "center_i": float(columns["center_i"][index]),
        "center_j": float(columns["center_j"][index]),
        "center_k": float(columns["center_k"][index]),
        "required_rank_count": len(ranks),
        "required_rank_ids_json": json.dumps(ranks),
        "primitive_eight_field_bytes": bytes_for_eight_float32_fields,
        "primitive_eight_field_gib": bytes_for_eight_float32_fields / 1024**3,
        "full_primitive_rank_file_read_gib": len(ranks) * FULL_PRIMITIVE_RANK_BYTES / 1024**3,
        "estimated_rank_read_seconds_at_1_gib_s": len(ranks) * FULL_PRIMITIVE_RANK_BYTES / 1024**3,
        "estimated_rank_read_seconds_at_250_mib_s": len(ranks) * FULL_PRIMITIVE_RANK_BYTES / 1024**2 / 250.0,
        "estimated_rank_read_seconds_at_500_mib_s": len(ranks) * FULL_PRIMITIVE_RANK_BYTES / 1024**2 / 500.0,
        "estimated_rank_read_seconds_at_1000_mib_s": len(ranks) * FULL_PRIMITIVE_RANK_BYTES / 1024**2 / 1000.0,
        "matched_pair_id": "" if matched_pair_id is None else matched_pair_id,
        "match_score_rms_mad_units": "" if match_score is None else match_score,
        "match_differences_json": "" if match_differences is None else json.dumps(match_differences, sort_keys=True),
        "extraction_caveat": "rank IDs use verified metadata rank_map; later cube must remain nonperiodic for structure-function pairs",
    }
    for name in names:
        row[name] = float(columns[name][index])
    return row


def _analysis_generated_files(analysis_dir: Path) -> dict[str, str]:
    excluded = {"ANALYSIS_COMPLETE", "ANALYSIS_COMPLETE.json", "analysis_manifest.json"}
    return {
        str(path.relative_to(analysis_dir)): file_sha256(path)
        for path in sorted(analysis_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _write_analysis_completion(
    analysis_dir: Path,
    *,
    artifact_graph_sha256_value: str,
    build_marker: Path,
    verify_marker: Path,
) -> None:
    generated_files = _analysis_generated_files(analysis_dir)
    analysis_manifest = {
        "artifact_graph_sha256": artifact_graph_sha256_value,
        "build_complete_sha256": file_sha256(build_marker),
        "verification_marker_sha256": file_sha256(verify_marker),
        "analysis_source_hashes": _source_hashes(),
        "generated_files": generated_files,
    }
    analysis_manifest_path = analysis_dir / "analysis_manifest.json"
    write_json(analysis_manifest_path, analysis_manifest)
    write_json(analysis_dir / "ANALYSIS_COMPLETE.json", {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_graph_sha256": artifact_graph_sha256_value,
        "verification_marker_sha256": analysis_manifest["verification_marker_sha256"],
        "analysis_manifest_sha256": file_sha256(analysis_manifest_path),
        "generated_file_count": len(generated_files),
    })


def _require_analysis_completion(
    analysis_dir: Path,
    *,
    artifact_graph_sha256_value: str,
    build_marker: Path,
    verify_marker: Path,
) -> dict[str, object]:
    manifest_path = analysis_dir / "analysis_manifest.json"
    complete_path = analysis_dir / "ANALYSIS_COMPLETE.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    expected = {
        "artifact_graph_sha256": artifact_graph_sha256_value,
        "build_complete_sha256": file_sha256(build_marker),
        "verification_marker_sha256": file_sha256(verify_marker),
        "analysis_source_hashes": _source_hashes(),
        "generated_files": _analysis_generated_files(analysis_dir),
    }
    if manifest != expected:
        raise ValueError(f"{manifest_path}: stale or incomplete analysis manifest")
    if complete.get("artifact_graph_sha256") != artifact_graph_sha256_value:
        raise ValueError(f"{complete_path}: artifact graph mismatch")
    if complete.get("verification_marker_sha256") != expected["verification_marker_sha256"]:
        raise ValueError(f"{complete_path}: verification marker mismatch")
    if complete.get("analysis_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError(f"{complete_path}: analysis-manifest hash mismatch")
    if int(complete.get("generated_file_count", -1)) != len(expected["generated_files"]):
        raise ValueError(f"{complete_path}: generated-file count mismatch")
    return manifest


def command_analyze(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    build_marker = run_dir / "BUILD_COMPLETE.json"
    verify_marker = run_dir / "verification" / "VERIFY_COMPLETE.json"
    if not build_marker.is_file():
        raise FileNotFoundError(f"missing build marker {build_marker}")
    if not verify_marker.is_file():
        raise FileNotFoundError(f"missing independent verification marker {verify_marker}")
    verification = json.loads(verify_marker.read_text())
    if not verification.get("passed") or verification.get("build_complete_sha256") != file_sha256(build_marker):
        raise ValueError(f"{verify_marker}: stale or failed independent verification")
    current_graph_sha256 = artifact_graph_sha256(artifact_graph(run_dir))
    if verification.get("artifact_graph_sha256") != current_graph_sha256:
        raise ValueError(f"{verify_marker}: artifact graph changed after independent verification")
    build_metadata = json.loads(build_marker.read_text())
    if build_metadata.get("artifact_graph_sha256") != current_graph_sha256:
        raise ValueError(f"{build_marker}: artifact graph does not match build marker")
    catalogs_dir = run_dir / "catalogs"
    analysis_dir = run_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    preflight_rows = _analysis_preflight(catalogs_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for stale_marker in (
        analysis_dir / "ANALYSIS_COMPLETE",
        analysis_dir / "ANALYSIS_COMPLETE.json",
        analysis_dir / "analysis_manifest.json",
    ):
        stale_marker.unlink(missing_ok=True)
    write_csv(analysis_dir / "analysis_preflight.csv", preflight_rows, list(preflight_rows[0]))

    summary_rows = _make_summary_rows(catalogs_dir)
    summary_fields = list(summary_rows[0])
    write_csv(analysis_dir / "distribution_quantiles.csv", summary_rows, summary_fields)
    _plot_distributions(catalogs_dir, plots_dir)
    _plot_dbb_correlations(catalogs_dir, plots_dir)
    correlation_rows = _plot_correlation_matrices(catalogs_dir, plots_dir)
    write_csv(analysis_dir / "spearman_correlations.csv", correlation_rows, list(correlation_rows[0]))
    cross_scale = _cross_scale_rows(catalogs_dir)
    write_csv(analysis_dir / "cross_scale_dBB.csv", cross_scale, list(cross_scale[0]))
    transition_rows = _cross_scale_transition_rows(catalogs_dir)
    write_csv(analysis_dir / "cross_scale_dBB_transitions.csv", transition_rows, list(transition_rows[0]))
    conditional_rows = _conditional_quantile_rows(catalogs_dir)
    if conditional_rows:
        write_csv(analysis_dir / "conditional_dBB_bands.csv", conditional_rows, list(conditional_rows[0]))

    rank_map = np.load(_cache_paths(run_dir)["rank_map"])
    pilot_rows, pilot_metadata = _pilot_rows(catalogs_dir, rank_map)
    write_csv(analysis_dir / "pilot_sample.csv", pilot_rows, list(pilot_rows[0]))
    write_json(analysis_dir / "pilot_sample_metadata.json", pilot_metadata)
    _write_scientific_summary(analysis_dir, summary_rows, correlation_rows, cross_scale, pilot_rows, pilot_metadata)
    _write_analysis_completion(
        analysis_dir,
        artifact_graph_sha256_value=current_graph_sha256,
        build_marker=build_marker,
        verify_marker=verify_marker,
    )
    _require_analysis_completion(
        analysis_dir,
        artifact_graph_sha256_value=current_graph_sha256,
        build_marker=build_marker,
        verify_marker=verify_marker,
    )
    log("analysis and pilot proposal complete")


def _lookup_summary(rows: Sequence[Mapping[str, object]], l_sub: int, prop: str, key: str) -> float:
    return float(next(row[key] for row in rows if row["L_sub"] == l_sub and row["property"] == prop))


def _lookup_correlation(rows: Sequence[Mapping[str, object]], l_sub: int, prop: str) -> float:
    return float(next(
        row["spearman_r"]
        for row in rows
        if row["L_sub"] == l_sub and row["property_1"] == "dBB" and row["property_2"] == prop
    ))


def _write_scientific_summary(
    output_dir: Path,
    summary_rows: Sequence[Mapping[str, object]],
    correlation_rows: Sequence[Mapping[str, object]],
    cross_scale: Sequence[Mapping[str, object]],
    pilot_rows: Sequence[Mapping[str, object]],
    pilot_metadata: Mapping[str, object],
) -> None:
    lines = [
        "# Prompt 1 Domain Census Summary",
        "",
        "## dBB Scale Dependence",
        "",
        "| L_sub / dx | subvolumes | q16 | median | q84 | q95 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for l_sub in L_SUB_VALUES:
        count = int(_lookup_summary(summary_rows, l_sub, "dBB", "count"))
        q16 = _lookup_summary(summary_rows, l_sub, "dBB", "q0.16")
        median = _lookup_summary(summary_rows, l_sub, "dBB", "q0.5")
        q84 = _lookup_summary(summary_rows, l_sub, "dBB", "q0.84")
        q95 = _lookup_summary(summary_rows, l_sub, "dBB", "q0.95")
        lines.append(f"| {l_sub} | {count} | {q16:.6g} | {median:.6g} | {q84:.6g} | {q95:.6g} |")
    lines.extend([
        "",
        "The bounded complements `B_mean_sq_over_B2_mean` and `deltaB_sq_over_B2_mean` "
        "are included in every catalog so large `dBB` can be separated into weak-mean-field "
        "cancellation and genuinely large fluctuation amplitude.",
        "",
        "## Correlates Of dBB",
        "",
        "| L_sub / dx | B_mean | deltaB | rho_sigma / rho_mean |",
        "|---:|---:|---:|---:|",
    ])
    for l_sub in L_SUB_VALUES:
        lines.append(
            f"| {l_sub} | {_lookup_correlation(correlation_rows, l_sub, 'B_mean'):.5f} "
            f"| {_lookup_correlation(correlation_rows, l_sub, 'deltaB'):.5f} "
            f"| {_lookup_correlation(correlation_rows, l_sub, 'rho_sigma_over_mean'):.5f} |"
        )
    lines.extend([
        "",
        "`dBB` is exactly the ratio `deltaB / B_mean`. The correlation table separates the two "
        "available magnetic contributions: weak mean-field cancellation and genuinely elevated "
        "fluctuation amplitude. Density and conserved-momentum statistics are retained as "
        "environmental correlates. These are domain-wide rank correlations, not causal claims.",
        "",
        "## Cross-Scale Comparison",
        "",
        "| L_sub / dx | coarse dBB vs child mean Spearman | coarse dBB vs child median Spearman |",
        "|---:|---:|---:|",
    ])
    for row in cross_scale:
        lines.append(
            f"| {row['L_sub']} | {float(row['coarse_dBB_vs_child_mean_spearman']):.5f} "
            f"| {float(row['coarse_dBB_vs_child_median_spearman']):.5f} |"
        )
    lines.extend([
        "",
        "## Reconstruction Boundary",
        "",
        "The primary `mhd_u_bcc` files store conserved momenta, not primitive velocities. "
        "SGS products are intentionally excluded because they are not trustworthy. Velocity "
        "moments, Mach numbers, kinetic partitions, pressure, mixed-field statistics, and "
        "off-diagonal covariance tensors are therefore unavailable. `vA_*_proxy` columns are "
        "explicit coarse summaries formed only from reconstructable primary moments.",
        "",
        "## Pilot Proposal",
        "",
        f"The proposed pilot contains {len(pilot_rows)} `L_sub={pilot_metadata['primary_pilot_L_sub']}` "
        "regions selected after inspecting the census. It includes representative dBB regimes, "
        f"{pilot_metadata['matched_pair_count']} accepted low/high-dBB matched pairs below the "
        "documented robust-score threshold, and targeted outliers. "
        "No full-resolution pilot extraction or structure-function calculation has been launched.",
        "",
        f"{pilot_metadata['scale_choice_rationale']}",
        "",
        "See `pilot_sample.csv` for coordinates, environmental properties, rank-file lists, and "
        "memory estimates.",
    ])
    (output_dir / "scientific_summary.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="inspect representative rank-zero cbin headers")
    probe.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    probe.add_argument("--time", type=float, default=6.0)
    probe.add_argument("--output", required=True)
    probe.set_defaults(func=command_probe)

    build = subparsers.add_parser("build", help="strictly assemble cbin products and build catalogs")
    build.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    build.add_argument("--run-dir", required=True)
    build.add_argument("--time", type=float, default=6.0)
    build.add_argument("--input-file", default=INPUT_FILE_DEFAULT)
    build.set_defaults(func=command_build)

    analyze = subparsers.add_parser("analyze", help="plot catalogs, summarize distributions, propose pilot")
    analyze.add_argument("--run-dir", required=True)
    analyze.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
