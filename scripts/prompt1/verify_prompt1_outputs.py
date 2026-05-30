#!/usr/bin/env python3
"""Independently verify completed Prompt 1 catalog artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbin_tools import (  # noqa: E402
    DOMAIN_CELLS,
    EXPECTED_RANKS,
    MESH_BLOCK_CELLS,
    PRIMARY_BASES,
    PRIMARY_LABELS,
    RANK_LATTICE,
    aggregate_blocks,
    aggregate_raw_moment_error_bounds,
    file_sha256,
    load_array_cache,
    raw_moment_statistics,
    robust_nonnegative,
    source_raw_moment_error_bounds,
    source_shard_inventory_sha256,
)
from run_prompt1_catalog import (  # noqa: E402
    L_SUB_VALUES,
    SOURCE_SCALE,
    _cache_paths,
    artifact_graph,
    artifact_graph_sha256,
)

FORBIDDEN_SGS_DERIVED_COLUMNS = {
    "delta_u_favre",
    "kinetic_energy_mean",
    "mean_flow_kinetic_energy_favre",
    "turbulent_kinetic_energy_favre",
    "pressure_mean",
    "sound_speed_coarse",
    "Ms_favre",
    "MA_mean_proxy",
    "MA_energy",
    "magnetic_to_kinetic_energy",
    "turbulent_to_mean_kinetic_energy_favre",
    "u_dot_B_mean_volume",
}
FORBIDDEN_SGS_DERIVED_PATTERNS = (
    "_favre",
    "reynolds_",
    "magnetic_covariance_",
    "internal_energy",
    "pressure",
    "sound_speed",
    "kinetic_energy",
    "magnetic_to_kinetic",
    "u_dot_b",
    "_mean_volume",
)
FORBIDDEN_SGS_PROVENANCE_TERMS = (
    "mhd_sgs",
    "sgs_cache",
    "sgs_manifest",
    "raw_sgs",
    "sgs_snapshot",
)
GRID_COLUMNS = frozenset({
    "subvolume_id", "L_sub", "grid_i", "grid_j", "grid_k",
    "cell_i0", "cell_i1", "cell_j0", "cell_j1", "cell_k0", "cell_k1",
    "center_i", "center_j", "center_k", "center_x1", "center_x2", "center_x3",
    "x1_min", "x1_max", "x2_min", "x2_max", "x3_min", "x3_max",
    "source_cbin_scale",
    "source_cbin_i0", "source_cbin_i1", "source_cbin_j0", "source_cbin_j1",
    "source_cbin_k0", "source_cbin_k1",
    "rank_x1_0", "rank_x1_1", "rank_x2_0", "rank_x2_1", "rank_x3_0", "rank_x3_1",
    "wraps_periodic_boundary", "parent_L_sub", "parent_subvolume_id", "required_rank_count",
})
PRIMARY_STATS = (
    "mean", "variance", "variance_error_bound", "sigma", "central_moment_3", "central_moment_4",
    "central_moment_3_error_bound", "central_moment_4_error_bound",
    "skewness", "skewness_error_bound", "kurtosis", "kurtosis_error_bound",
    "excess_kurtosis", "moment_flags",
)
FLAG_COLUMNS = (
    *(f"{base}_moment_flags" for base in PRIMARY_BASES),
    "deltaB_flags", "dBB_flags", "B_mean_fraction_flags", "deltaB_fraction_flags",
    "rho_sigma_over_mean_flags",
)
EXPECTED_PRIMARY_ONLY_CATALOG_COLUMNS = frozenset({
    *GRID_COLUMNS,
    *(f"{base}_{stat}" for base in PRIMARY_BASES for stat in PRIMARY_STATS),
    "B_mean", "B_mean_error_bound", "B_rms", "B2_mean", "deltaB", "deltaB_sq",
    "deltaB_sq_error_bound", "dBB", "B_mean_sq_over_B2_mean", "deltaB_sq_over_B2_mean",
    "deltaB_flags", "dBB_flags", "B_mean_fraction_flags", "deltaB_fraction_flags",
    "rho_sigma_over_mean", "rho_sigma_over_mean_flags", "magnetic_energy_mean",
    "vA_mean_proxy", "vA_rms_like_proxy", "u_mass_weighted_mean_x",
    "u_mass_weighted_mean_y", "u_mass_weighted_mean_z", "catalog_validity_flags",
})


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _require_no_sgs_provenance(payload: object, label: str) -> None:
    encoded = json.dumps(payload, sort_keys=True).lower()
    forbidden = sorted(term for term in FORBIDDEN_SGS_PROVENANCE_TERMS if term in encoded)
    require(not forbidden, f"{label}: forbidden SGS provenance metadata {forbidden}")


def _require_primary_only_catalog(
    columns: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    label: str,
) -> None:
    forbidden = sorted(
        name for name in columns
        if name in FORBIDDEN_SGS_DERIVED_COLUMNS
        or name.lower().startswith(("ms_", "ma_"))
        or any(pattern in name.lower() for pattern in FORBIDDEN_SGS_DERIVED_PATTERNS)
    )
    require(not forbidden, f"{label}: forbidden SGS-derived columns {forbidden}")
    _require_no_sgs_provenance(metadata, label)
    actual = set(columns)
    require(
        actual == EXPECTED_PRIMARY_ONLY_CATALOG_COLUMNS,
        f"{label}: catalog schema mismatch; "
        f"missing={sorted(EXPECTED_PRIMARY_ONLY_CATALOG_COLUMNS - actual)} "
        f"unexpected={sorted(actual - EXPECTED_PRIMARY_ONLY_CATALOG_COLUMNS)}",
    )


def _require_primary_raw_cache(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    label: str,
) -> None:
    _require_no_sgs_provenance(metadata, label)
    actual = set(arrays)
    expected = set(PRIMARY_LABELS)
    require(
        actual == expected,
        f"{label}: raw payload fields do not match primary schema; "
        f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}",
    )


def _require_array_match(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    require(actual.shape == expected.shape, f"{label}: shape mismatch")
    if expected.dtype.kind in "uib":
        require(np.array_equal(actual, expected), f"{label}: exact-value mismatch")
        return
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), f"{label}: NaN-availability mismatch")
    require(np.array_equal(np.isposinf(actual), np.isposinf(expected)), f"{label}: positive-inf mismatch")
    require(np.array_equal(np.isneginf(actual), np.isneginf(expected)), f"{label}: negative-inf mismatch")
    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    require(np.array_equal(actual_finite, expected_finite), f"{label}: finite-availability mismatch")
    require(
        np.allclose(actual[expected_finite], expected[expected_finite], rtol=3.0e-5, atol=3.0e-5),
        f"{label}: formula mismatch",
    )


def _compact_expected(values: np.ndarray) -> np.ndarray:
    if np.asarray(values).dtype.kind == "f":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(values, dtype=np.float32)
    return np.asarray(values)


def _expected_grid_columns(l_sub: int) -> dict[str, np.ndarray]:
    n = DOMAIN_CELLS // l_sub
    require(DOMAIN_CELLS % l_sub == 0, f"L_sub={l_sub} does not tile domain={DOMAIN_CELLS}")
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
    return {name: np.asarray(values).reshape(-1) for name, values in columns.items()}


def _verify_expected_grid(columns: Mapping[str, np.ndarray], l_sub: int, label: str) -> None:
    expected = _expected_grid_columns(l_sub)
    require(set(expected) == GRID_COLUMNS, f"{label}: verifier grid schema mismatch")
    for name in sorted(GRID_COLUMNS):
        require(np.array_equal(columns[name], expected[name]), f"{label}: exact grid mismatch for {name}")


def _expected_safe_ratio(
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
        unresolved = np.isfinite(denominator) & (np.abs(denominator) <= denominator_error_bound)
        flags[unresolved] |= 2
        valid &= ~unresolved
    result[valid] = numerator[valid] / denominator[valid]
    flags[~valid] |= 1
    return result, flags


def _verify_foundational_catalog_against_raw(
    columns: Mapping[str, np.ndarray],
    raw_primary: Mapping[str, np.ndarray],
    l_sub: int,
    label: str,
) -> None:
    factor = l_sub // SOURCE_SCALE
    raw_bounds_by_base: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for base in PRIMARY_BASES:
        raw_moments = tuple(raw_primary[f"{base}_{suffix}"] for suffix in ("1st", "2nd", "3rd", "4th"))
        source_bounds = source_raw_moment_error_bounds(*raw_moments, source_cells=SOURCE_SCALE**3)
        raw_bounds = aggregate_raw_moment_error_bounds(raw_moments, source_bounds, factor)
        raw_bounds_by_base[base] = raw_bounds
        aggregated = tuple(aggregate_blocks(moment, factor) for moment in raw_moments)
        stats = raw_moment_statistics(*aggregated, abs_error_bounds=raw_bounds)
        for stat in PRIMARY_STATS:
            _require_array_match(
                columns[f"{base}_{stat}"],
                _compact_expected(stats[stat]).reshape(-1),
                f"{label}: {base}_{stat}",
            )
    b_mean_error_bound = np.sqrt(sum(raw_bounds_by_base[f"bcc{axis}"][0] ** 2 for axis in (1, 2, 3)))
    _require_array_match(
        columns["B_mean_error_bound"],
        _compact_expected(b_mean_error_bound).reshape(-1),
        f"{label}: B_mean_error_bound",
    )


def _expected_primary_derived(
    columns: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    b_mean = np.sqrt(sum(columns[f"bcc{axis}_mean"] ** 2 for axis in (1, 2, 3)))
    b2_mean = sum(
        columns[f"bcc{axis}_mean"] ** 2 + columns[f"bcc{axis}_variance"]
        for axis in (1, 2, 3)
    )
    delta_b_sq_error_bound = sum(columns[f"bcc{axis}_variance_error_bound"] for axis in (1, 2, 3))
    delta_b_sq, delta_b_flags = robust_nonnegative(
        b2_mean - b_mean**2,
        b2_mean,
        abs_error_bound=delta_b_sq_error_bound,
    )
    delta_b = np.sqrt(delta_b_sq)
    dBB, dBB_flags = _expected_safe_ratio(
        delta_b,
        b_mean,
        denominator_error_bound=columns["B_mean_error_bound"],
    )
    bmean_fraction, bmean_fraction_flags = _expected_safe_ratio(b_mean**2, b2_mean)
    delta_b_fraction, delta_b_fraction_flags = _expected_safe_ratio(delta_b_sq, b2_mean)
    rho = columns["dens_mean"]
    rho_cv, rho_cv_flags = _expected_safe_ratio(columns["dens_sigma"], rho)
    derived = {name: _compact_expected(values) for name, values in {
        "B_mean": b_mean,
        "B_rms": np.sqrt(b2_mean),
        "B2_mean": b2_mean,
        "deltaB": delta_b,
        "deltaB_sq": delta_b_sq,
        "deltaB_sq_error_bound": delta_b_sq_error_bound,
        "dBB": dBB,
        "B_mean_sq_over_B2_mean": bmean_fraction,
        "deltaB_sq_over_B2_mean": delta_b_fraction,
        "rho_sigma_over_mean": rho_cv,
        "magnetic_energy_mean": 0.5 * b2_mean,
        "vA_mean_proxy": b_mean / np.sqrt(rho),
        "vA_rms_like_proxy": np.sqrt(b2_mean / rho),
    }.items()}
    flags = {
        "deltaB_flags": delta_b_flags,
        "dBB_flags": dBB_flags,
        "B_mean_fraction_flags": bmean_fraction_flags,
        "deltaB_fraction_flags": delta_b_fraction_flags,
        "rho_sigma_over_mean_flags": rho_cv_flags,
    }
    return derived, flags


def _verify_primary_formulas(columns: Mapping[str, np.ndarray], label: str) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        expected, _ = _expected_primary_derived(columns)
        for name, values in expected.items():
            _require_array_match(columns[name], values, f"{label}: {name}")
        rho = columns["dens_mean"]
        require(np.all(np.isfinite(rho) & (rho > 0.0)), f"{label}: nonpositive density mean")
        _require_array_match(columns["dens_sigma"], np.sqrt(columns["dens_variance"]), f"{label}: dens_sigma")
        for axis in (1, 2, 3):
            _require_array_match(
                columns[f"mom{axis}_sigma"],
                np.sqrt(columns[f"mom{axis}_variance"]),
                f"{label}: mom{axis}_sigma",
            )
            _require_array_match(
                columns[f"u_mass_weighted_mean_{'xyz'[axis - 1]}"],
                columns[f"mom{axis}_mean"] / rho,
                f"{label}: mass-weighted mean velocity axis {axis}",
            )


def _verify_primary_flags(
    columns: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    label: str,
) -> None:
    _, expected_flags = _expected_primary_derived(columns)
    for name, values in expected_flags.items():
        _require_array_match(columns[name], values, f"{label}: {name}")
    expected_flag_bits = {name: bit for bit, name in enumerate(FLAG_COLUMNS)}
    require(metadata.get("catalog_validity_flag_bits") == expected_flag_bits, f"{label}: validity-bit metadata mismatch")
    aggregate = np.zeros_like(columns["catalog_validity_flags"], dtype=np.uint32)
    for bit, name in enumerate(FLAG_COLUMNS):
        aggregate |= (np.asarray(columns[name]) != 0).astype(np.uint32) << bit
    _require_array_match(columns["catalog_validity_flags"], aggregate, f"{label}: catalog_validity_flags")


def _verify_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    require(int(manifest["expected_ranks"]) == EXPECTED_RANKS, f"{path}: rank count mismatch")
    require(int(manifest["holes"]) == 0, f"{path}: coverage holes")
    require(int(manifest["overlaps"]) == 0, f"{path}: coverage overlaps")
    require(int(manifest["missing_rank_locations"]) == 0, f"{path}: missing rank locations")
    return manifest


def _verify_catalog(
    path: Path,
    l_sub: int,
    *,
    primary_cache_sha256: str,
    validation_token_sha256: str,
    raw_primary: Mapping[str, np.ndarray],
) -> dict[str, object]:
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    require(manifest_path.is_file(), f"{path}: missing external catalog manifest")
    manifest = json.loads(manifest_path.read_text())
    _require_no_sgs_provenance(manifest, str(manifest_path))
    require(manifest.get("catalog_sha256") == file_sha256(path), f"{path}: catalog hash mismatch")
    require(int(manifest["L_sub"]) == l_sub, f"{path}: external manifest L_sub mismatch")
    require(manifest.get("primary_cache_sha256") == primary_cache_sha256, f"{path}: primary-cache provenance mismatch")
    require(manifest.get("validation_token_sha256") == validation_token_sha256, f"{path}: token provenance mismatch")
    columns, metadata = load_array_cache(path)
    _require_primary_only_catalog(columns, metadata, str(path))
    for key in ("L_sub", "primary_cache_sha256", "source_hashes", "validation_token_sha256"):
        require(metadata.get(key) == manifest.get(key), f"{path}: embedded/external manifest mismatch for {key}")
    expected_count = (DOMAIN_CELLS // l_sub) ** 3
    require(int(metadata["L_sub"]) == l_sub, f"{path}: metadata L_sub mismatch")
    require(int(metadata["subvolume_count"]) == expected_count, f"{path}: metadata row-count mismatch")
    require(float(metadata["tiled_domain_fraction"]) == 1.0, f"{path}: incomplete tiling")
    require("catalog_validity_flag_bits" in metadata, f"{path}: missing validity-bit metadata")
    for name, values in columns.items():
        require(values.ndim == 1, f"{path}: {name} is not flattened")
        require(values.size == expected_count, f"{path}: {name} row-count mismatch")

    require(np.array_equal(columns["subvolume_id"], np.arange(expected_count)), f"{path}: non-contiguous IDs")
    require(np.all(columns["L_sub"] == l_sub), f"{path}: row L_sub mismatch")
    require(np.all(columns["wraps_periodic_boundary"] == 0), f"{path}: unexpected wrapped row")
    _verify_expected_grid(columns, l_sub, str(path))
    for axis, cell0, cell1 in (
        ("x1", "cell_i0", "cell_i1"),
        ("x2", "cell_j0", "cell_j1"),
        ("x3", "cell_k0", "cell_k1"),
    ):
        require(np.all(columns[cell1] - columns[cell0] == l_sub), f"{path}: {axis} cell width mismatch")
        require(np.all(columns[cell0] >= 0) and np.all(columns[cell1] <= DOMAIN_CELLS), f"{path}: {axis} bounds")
        require(
            _max_abs(columns[f"{axis}_min"] - (-0.5 + columns[cell0] / DOMAIN_CELLS)) < 1.0e-6,
            f"{path}: {axis} physical lower bounds",
        )
        require(
            _max_abs(columns[f"{axis}_max"] - (-0.5 + columns[cell1] / DOMAIN_CELLS)) < 1.0e-6,
            f"{path}: {axis} physical upper bounds",
        )

    source_factor = l_sub // SOURCE_SCALE
    for axis in ("i", "j", "k"):
        require(
            np.all(columns[f"source_cbin_{axis}1"] - columns[f"source_cbin_{axis}0"] == source_factor),
            f"{path}: source-cbin {axis} width mismatch",
        )
    rank_counts = (
        (columns["rank_x1_1"] - columns["rank_x1_0"])
        * (columns["rank_x2_1"] - columns["rank_x2_0"])
        * (columns["rank_x3_1"] - columns["rank_x3_0"])
    )
    require(np.array_equal(columns["required_rank_count"], rank_counts), f"{path}: rank-count mapping mismatch")
    if l_sub < max(L_SUB_VALUES):
        parent_count = (DOMAIN_CELLS // (2 * l_sub)) ** 3
        require(np.all((columns["parent_subvolume_id"] >= 0) & (columns["parent_subvolume_id"] < parent_count)),
                f"{path}: parent IDs out of range")
    else:
        require(np.all(columns["parent_subvolume_id"] == -1), f"{path}: top-level parent IDs must be absent")

    _verify_foundational_catalog_against_raw(columns, raw_primary, l_sub, str(path))
    _verify_primary_formulas(columns, str(path))
    _verify_primary_flags(columns, metadata, str(path))
    complement = columns["B_mean_sq_over_B2_mean"] + columns["deltaB_sq_over_B2_mean"]
    valid_complement = np.isfinite(complement)
    require(_max_abs(complement[valid_complement] - 1.0) < 3.0e-5, f"{path}: magnetic complements")
    return {
        "L_sub": l_sub,
        "rows": expected_count,
        "validity_flagged_rows": int(np.count_nonzero(columns["catalog_validity_flags"])),
        "required_rank_count_min": int(np.min(columns["required_rank_count"])),
        "required_rank_count_max": int(np.max(columns["required_rank_count"])),
        "magnetic_complement_max_abs_error": _max_abs(complement[valid_complement] - 1.0),
    }


def verify(run_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "VERIFY_COMPLETE.json").unlink(missing_ok=True)
    build_marker = run_dir / "BUILD_COMPLETE.json"
    require(build_marker.is_file(), f"{run_dir}: missing BUILD_COMPLETE.json")
    build_metadata = json.loads(build_marker.read_text())
    _require_no_sgs_provenance(build_metadata, str(build_marker))
    cache_paths = _cache_paths(run_dir)
    primary_manifest = _verify_manifest(cache_paths["primary_manifest"])
    _require_no_sgs_provenance(primary_manifest, str(cache_paths["primary_manifest"]))
    require(primary_manifest.get("product") == "mhd_u_bcc", "primary manifest product mismatch")
    require(primary_manifest.get("fields") == list(PRIMARY_LABELS), "primary manifest fields mismatch")
    require(primary_manifest.get("var_names") == list(PRIMARY_LABELS), "primary manifest schema mismatch")
    validation_token_path = run_dir / "validation" / "VALIDATION_COMPLETE.json"
    validation_token = json.loads(validation_token_path.read_text())
    _require_no_sgs_provenance(validation_token, str(validation_token_path))
    validation_token_sha256 = file_sha256(validation_token_path)
    primary_cache_sha256 = file_sha256(cache_paths["primary"])
    require(primary_manifest.get("cache_sha256") == primary_cache_sha256, "primary cache hash mismatch")
    require(primary_manifest.get("validation_token_sha256") == validation_token_sha256, "primary token hash mismatch")
    primary_arrays, embedded_primary_metadata = load_array_cache(cache_paths["primary"])
    _require_primary_raw_cache(primary_arrays, embedded_primary_metadata, str(cache_paths["primary"]))
    for key in (
        "data_root", "scale", "time", "cycle", "source_hashes", "validation_token_sha256",
        "product", "basename", "fields", "var_names", "expected_ranks", "rank_map_sha256",
        "source_shard_inventory_sha256",
    ):
        require(
            embedded_primary_metadata.get(key) == primary_manifest.get(key),
            f"primary cache embedded/external manifest mismatch for {key}",
        )
    for manifest in (primary_manifest,):
        inventory_sha256 = source_shard_inventory_sha256(
            manifest["data_root"],
            manifest["product"],
            int(manifest["scale"]),
            manifest["basename"],
            expected_ranks=int(manifest["expected_ranks"]),
        )
        require(
            manifest.get("source_shard_inventory_sha256") == inventory_sha256,
            f"{manifest['product']}: source-shard inventory changed",
        )
    rank_map = np.load(cache_paths["rank_map"])
    rank_map_hash = file_sha256(cache_paths["rank_map"])
    require(primary_manifest.get("rank_map_sha256") == rank_map_hash, "primary rank-map hash mismatch")
    require(rank_map.shape == (RANK_LATTICE[2], RANK_LATTICE[1], RANK_LATTICE[0]), "rank-map shape mismatch")
    require(np.array_equal(np.sort(rank_map.reshape(-1)), np.arange(EXPECTED_RANKS)), "rank-map IDs mismatch")

    catalogs_dir = run_dir / "catalogs"
    catalog_rows = []
    for l_sub in L_SUB_VALUES:
        log(f"verifying catalog L_sub={l_sub}")
        path = catalogs_dir / f"catalog_L{l_sub}.npz"
        require(path.is_file(), f"missing catalog {path}")
        require((catalogs_dir / f"catalog_L{l_sub}.complete").is_file(), f"missing completion marker for {path}")
        catalog_rows.append(_verify_catalog(
            path,
            l_sub,
            primary_cache_sha256=primary_cache_sha256,
            validation_token_sha256=validation_token_sha256,
            raw_primary=primary_arrays,
        ))
    del primary_arrays
    graph = artifact_graph(run_dir)
    _require_no_sgs_provenance(graph, "recomputed artifact graph")
    graph_sha256 = artifact_graph_sha256(graph)
    require(build_metadata.get("artifact_graph_sha256") == graph_sha256, "build artifact-graph hash mismatch")
    payload: dict[str, object] = {
        "passed": True,
        "run_dir": str(run_dir),
        "build_complete_sha256": hashlib.sha256(build_marker.read_bytes()).hexdigest(),
        "build_metadata": build_metadata,
        "artifact_graph": graph,
        "artifact_graph_sha256": graph_sha256,
        "primary_manifest": primary_manifest,
        "catalogs": catalog_rows,
    }
    write_json(output_dir / "verification_results.json", payload)
    lines = [
        "# Prompt 1 Catalog Verification",
        "",
        "Overall result: **PASS**",
        "",
        "| L_sub / dx | Rows | Flagged rows | Required ranks min | Required ranks max | Complement max abs error |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in catalog_rows:
        lines.append(
            f"| {row['L_sub']} | {row['rows']} | {row['validity_flagged_rows']} "
            f"| {row['required_rank_count_min']} | {row['required_rank_count_max']} "
            f"| {row['magnetic_complement_max_abs_error']:.6g} |"
        )
    (output_dir / "verification_report.md").write_text("\n".join(lines) + "\n")
    write_json(output_dir / "VERIFY_COMPLETE.json", {
        "passed": True,
        "verification_results_sha256": hashlib.sha256(
            (output_dir / "verification_results.json").read_bytes()
        ).hexdigest(),
        "build_complete_sha256": payload["build_complete_sha256"],
        "artifact_graph_sha256": graph_sha256,
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "verification"
    payload = verify(run_dir, output_dir)
    log(f"verified {len(payload['catalogs'])} catalogs")


if __name__ == "__main__":
    main()
