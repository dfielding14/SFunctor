#!/usr/bin/env python3
"""Run bounded Phase 3 finite-domain 3-D sampler validation and smoke tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1.cbin_tools import file_sha256
from sfunctor.analysis.finite_domain import (
    constant_sp_shapes,
    coverage_rows,
    fit_interval_sensitivity_rows,
    offset_support_rows,
    result_to_npz_payload,
    summary_rows,
)
from sfunctor.core.finite_domain import (
    FiniteDomainConfig,
    compute_finite_domain_structure_functions,
    generate_fibonacci_displacements,
)
from sfunctor.reference_3d import compute_finite_domain_structure_functions_reference

BENCHMARK_CUBE_IDS = (
    "L640_sub00370",
    "L640_sub03942",
    "L640_sub00579",
    "L640_sub00738",
)
DEFAULT_PHASE2_ROOT = Path(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "phase2_extract/benchmark_verified_release_primary_20260530"
)
FIELD_PATHS = {
    "rho": "dens.npy",
    "v_x": "velx.npy",
    "v_y": "vely.npy",
    "v_z": "velz.npy",
    "B_x": "bcc1.npy",
    "B_y": "bcc2.npy",
    "B_z": "bcc3.npy",
}
MANIFEST_FIELD_NAMES = {
    "rho": "dens",
    "v_x": "velx",
    "v_y": "vely",
    "v_z": "velz",
    "B_x": "bcc1",
    "B_y": "bcc2",
    "B_z": "bcc3",
}
SCHEMA_VERSION = 2
FIT_SENSITIVITY_INTERVALS = ((8.0, 64.0), (16.0, 96.0), (8.0, 96.0))
FIT_STABILITY_ABSOLUTE_TOLERANCE = 0.2
ROBUSTNESS_SEEDS = (20260530, 20260531, 20260532)


def _json_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_builtin(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_builtin(value.tolist())
    if isinstance(value, np.generic):
        return _json_builtin(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(_json_builtin(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(_json_builtin(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _source_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "sfunctor" / "core" / "directional.py",
        root / "sfunctor" / "core" / "finite_domain.py",
        root / "sfunctor" / "analysis" / "finite_domain.py",
        root / "sfunctor" / "reference_3d.py",
        root / "scripts" / "phase1" / "cbin_tools.py",
        root / "job_scripts" / "phase3" / "run_phase3_sampler_andes.sh",
    )
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in paths if path.exists()}
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {
        "commit": commit,
        "dirty": dirty,
        "implementation_source_hashes": hashes,
        "implementation_sha256": _mapping_sha256(hashes),
    }


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _phase2_source_identity(
    phase2_root: Path,
    cube_id: str,
    *,
    verify_arrays: bool = True,
) -> dict[str, Any]:
    if cube_id not in BENCHMARK_CUBE_IDS:
        raise ValueError(f"Phase 3 is restricted to approved benchmark IDs: {cube_id}")
    cube_root = phase2_root / cube_id
    completion_path = cube_root / "COMPLETE.json"
    manifest_path = cube_root / "manifest.json"
    completion = json.loads(completion_path.read_text())
    if completion.get("cube_id") != cube_id:
        raise RuntimeError(f"Phase 2 completion marker has the wrong cube ID: {completion_path}")
    manifest_sha256 = file_sha256(manifest_path)
    if completion.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError(f"Phase 2 completion marker is stale: {completion_path}")
    manifest = json.loads(manifest_path.read_text())
    analysis_field_sha256 = {}
    for logical_name, manifest_name in MANIFEST_FIELD_NAMES.items():
        metadata = manifest["output_fields"][manifest_name]
        path = cube_root / metadata["relative_path"]
        observed = file_sha256(path) if verify_arrays else metadata["sha256"]
        if observed != metadata["sha256"]:
            raise RuntimeError(f"Phase 2 analysis input checksum mismatch: {path}")
        analysis_field_sha256[metadata["relative_path"]] = observed
    return {
        "cube_id": cube_id,
        "phase2_root": str(phase2_root),
        "completion_relative_path": f"{cube_id}/COMPLETE.json",
        "completion_sha256": file_sha256(completion_path),
        "manifest_relative_path": f"{cube_id}/manifest.json",
        "manifest_sha256": manifest_sha256,
        "analysis_field_sha256": analysis_field_sha256,
    }


def _load_cube(phase2_root: Path, cube_id: str) -> dict[str, np.ndarray]:
    fields_root = phase2_root / cube_id / "fields"
    arrays = {
        name: np.load(fields_root / relative_path, mmap_mode="r")
        for name, relative_path in FIELD_PATHS.items()
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes)) != (640, 640, 640):
        raise RuntimeError(f"Phase 3 expected one 640^3 KJI cube for {cube_id}, got {sorted(shapes)}")
    return arrays


def _nominal_radii(ell_max: int) -> tuple[int, ...]:
    if ell_max < 8:
        raise ValueError("ell_max must be at least 8 cells")
    candidates = [4, 8, 16, 32, 64, 96, 128, 160]
    values = [value for value in candidates if value <= ell_max]
    if values[-1] != ell_max:
        values.append(ell_max)
    return tuple(sorted(set(values)))


def _ell_bin_edges(radii: Sequence[int]) -> np.ndarray:
    radii_array = np.asarray(tuple(radii), dtype=float)
    interior = np.sqrt(radii_array[:-1] * radii_array[1:])
    return np.concatenate(([max(0.5, radii_array[0] / np.sqrt(2.0))], interior, [radii_array[-1] * 1.08 + 1.0]))


def _displacement_payload(ell_max: int, directions_per_radius: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    radii = _nominal_radii(ell_max)
    displacements = generate_fibonacci_displacements(
        radii, directions_per_radius=directions_per_radius
    )
    edges = _ell_bin_edges(radii)
    physical_ell = np.linalg.norm(displacements.astype(float), axis=1)
    displacements = displacements[physical_ell <= ell_max]
    physical_ell = np.linalg.norm(displacements.astype(float), axis=1)
    payload = {
        "nominal_radii_cells": radii,
        "ell_max_cells": ell_max,
        "directions_per_radius": directions_per_radius,
        "signed_closure": True,
        "displacement_count": len(displacements),
        "minimum_actual_ell_cells": float(physical_ell.min()),
        "maximum_actual_ell_cells": float(physical_ell.max()),
        "sha256": hashlib.sha256(displacements.tobytes()).hexdigest(),
    }
    return displacements, edges, payload


def _config(
    *,
    pair_mode: str,
    ell_edges: np.ndarray,
    sample_count: int,
    pair_batch_size: int,
    seed: int,
    p_values: Sequence[float],
) -> FiniteDomainConfig:
    return FiniteDomainConfig(
        ell_bin_edges=ell_edges,
        p_values=tuple(p_values),
        pair_mode=pair_mode,
        sample_count=sample_count,
        pair_batch_size=pair_batch_size,
        seed=seed,
    )


def _config_payload(config: FiniteDomainConfig) -> dict[str, Any]:
    return {
        "ell_bin_edges": config.ell_bin_edges,
        "p_values": config.p_values,
        "cell_sizes": config.cell_sizes,
        "pair_mode": config.pair_mode,
        "sample_count": config.sample_count,
        "pair_batch_size": config.pair_batch_size,
        "seed": config.seed,
        "theta_parallel_max": config.theta_parallel_max,
        "theta_perpendicular_min": config.theta_perpendicular_min,
        "phi_xi_max": config.phi_xi_max,
        "phi_lambda_min": config.phi_lambda_min,
        "B_epsilon": config.B_epsilon,
        "q_perp_epsilon": config.q_perp_epsilon,
        "r_perp_epsilon": config.r_perp_epsilon,
        "include_subvolume_mean": config.include_subvolume_mean,
    }


def _mode_summary(
    result: Any,
    *,
    fit_interval: tuple[float, float],
    min_count: int,
) -> dict[str, Any]:
    core = result.nested_core_bounds_kji
    retained_fraction = None
    if core is not None:
        retained_fraction = float(
            np.prod([stop - start for start, stop in core]) / np.prod(result.cube_shape_kji)
        )
    sensitivity_rows = fit_interval_sensitivity_rows(
        result, FIT_SENSITIVITY_INTERVALS, min_count=min_count
    )
    return {
        "pair_mode": result.pair_mode,
        "elapsed_seconds": result.elapsed_seconds,
        "cube_shape_kji": result.cube_shape_kji,
        "nested_core_bounds_kji": core,
        "nested_core_retained_volume_fraction": retained_fraction,
        "rho0": result.rho0,
        "rho0_provenance": result.rho0_provenance,
        "sampled_pairs_by_ell": result.sampled_pairs,
        "eligible_pairs_by_ell": result.eligible_pairs,
        "cube_candidate_pairs_by_ell": result.cube_candidate_pairs,
        "excluded_boundary_pairs_by_ell": result.excluded_boundary_pairs,
        "displacements_per_bin": result.displacements_per_bin,
        "out_of_range_displacements": result.out_of_range_displacements,
        "summary_rows": summary_rows(result, fit_interval, min_count=min_count),
        "fit_interval_sensitivity_rows": sensitivity_rows,
        "fit_stability_rows": _fit_stability_rows(sensitivity_rows),
        "coverage_rows": coverage_rows(result, min_count=min_count),
        "offset_support_rows": offset_support_rows(result),
    }


def _fit_stability_rows(sensitivity_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize alternate-window slope spread without treating it as a science gate."""

    grouped: dict[tuple[str, str, str, float, str], list[dict[str, Any]]] = {}
    for row in sensitivity_rows:
        for direction, slope in row["slopes"].items():
            key = (
                str(row["q"]),
                str(row["geometry"]),
                str(row["measurement"]),
                float(row["p"]),
                str(direction),
            )
            grouped.setdefault(key, []).append(
                {
                    "fit_interval": tuple(row["fit_interval"]),
                    "fit_quality": row["fit_quality"][direction],
                    "slope": slope,
                }
            )
    output = []
    for (q_name, geometry, measurement, p_value, direction), fits in sorted(grouped.items()):
        finite_slopes = [
            float(item["slope"])
            for item in fits
            if item["fit_quality"] == "ok" and np.isfinite(item["slope"])
        ]
        spread = max(finite_slopes) - min(finite_slopes) if finite_slopes else None
        output.append(
            {
                "q": q_name,
                "geometry": geometry,
                "measurement": measurement,
                "p": p_value,
                "direction": direction,
                "fits": fits,
                "finite_fit_count": len(finite_slopes),
                "absolute_slope_spread": spread,
                "absolute_slope_spread_tolerance": FIT_STABILITY_ABSOLUTE_TOLERANCE,
                "stable_across_tested_intervals": (
                    len(finite_slopes) == len(FIT_SENSITIVITY_INTERVALS)
                    and spread is not None
                    and spread <= FIT_STABILITY_ABSOLUTE_TOLERANCE
                ),
                "interpretation": "bounded fit-window sensitivity diagnostic; not a physical uncertainty estimate",
            }
        )
    return output


def _relative_mode_difference(left: Any, right: Any) -> dict[str, Any]:
    left_values, right_values = left.moments, right.moments
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    denominator = np.maximum(np.maximum(np.abs(left_values), np.abs(right_values)), 1.0e-30)
    relative = np.abs(left_values - right_values) / denominator
    return {
        "compared_populated_bins": int(np.count_nonzero(valid)),
        "median_relative_difference": float(np.median(relative[valid])) if np.any(valid) else None,
        "maximum_relative_difference": float(np.max(relative[valid])) if np.any(valid) else None,
        "sampling_depth_matched": left.sample_count == right.sample_count,
        "sample_count": {"left": left.sample_count, "right": right.sample_count},
        "interpretation": "finite-support and spatial-inhomogeneity diagnostic; not an equality test",
    }


def _relative_array_difference(left_values: np.ndarray, right_values: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    denominator = np.maximum(np.maximum(np.abs(left_values), np.abs(right_values)), 1.0e-30)
    relative = np.abs(left_values - right_values) / denominator
    return {
        "compared_populated_bins": int(np.count_nonzero(valid)),
        "median_relative_difference": float(np.median(relative[valid])) if np.any(valid) else None,
        "maximum_relative_difference": float(np.max(relative[valid])) if np.any(valid) else None,
    }


def _save_mode(path: Path, result: Any, displacements: np.ndarray, displacement_payload: Mapping[str, Any]) -> None:
    payload = result_to_npz_payload(result)
    payload["displacements_ijk"] = displacements
    payload["displacement_metadata_json"] = np.asarray(
        json.dumps(_json_builtin(displacement_payload), sort_keys=True, separators=(",", ":"))
    )
    _atomic_write_npz(path, payload)


def _verify_cube_output(
    cube_root: Path,
    phase2_root: Path,
    cube_id: str,
    *,
    expected_configuration_sha256: str | None = None,
    phase2_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    marker_path = cube_root / "COMPLETE.json"
    summary_path = cube_root / "summary.json"
    marker = json.loads(marker_path.read_text())
    source = dict(phase2_source or _phase2_source_identity(phase2_root, cube_id))
    current_source_version = _source_version()
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "passed"
        or marker.get("cube_id") != cube_id
    ):
        raise RuntimeError(f"invalid Phase 3 completion marker: {marker_path}")
    if marker.get("phase2_source") != source:
        raise RuntimeError(f"Phase 3 marker is stale relative to Phase 2 input: {marker_path}")
    if marker.get("implementation_sha256") != current_source_version["implementation_sha256"]:
        raise RuntimeError(f"Phase 3 marker was produced by a different implementation: {marker_path}")
    if marker.get("summary_sha256") != file_sha256(summary_path):
        raise RuntimeError(f"Phase 3 summary checksum mismatch: {summary_path}")
    result_sha256 = marker.get("result_sha256", {})
    if set(result_sha256) != {"nested_core.npz", "all_valid_pairs.npz"}:
        raise RuntimeError(f"Phase 3 marker has an invalid result inventory: {marker_path}")
    for relative_path, expected in result_sha256.items():
        if file_sha256(cube_root / relative_path) != expected:
            raise RuntimeError(f"Phase 3 result checksum mismatch: {cube_root / relative_path}")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "passed"
        or summary.get("cube_id") != cube_id
        or summary.get("phase2_source") != source
    ):
        raise RuntimeError(f"invalid Phase 3 summary identity: {cube_id}")
    if summary.get("configuration_sha256") != marker.get("configuration_sha256"):
        raise RuntimeError(f"Phase 3 configuration binding mismatch: {cube_id}")
    if _mapping_sha256(summary.get("configuration")) != marker.get("configuration_sha256"):
        raise RuntimeError(f"Phase 3 configuration checksum mismatch: {cube_id}")
    if summary.get("source_version", {}).get("implementation_sha256") != marker.get("implementation_sha256"):
        raise RuntimeError(f"Phase 3 source binding mismatch: {cube_id}")
    if (
        expected_configuration_sha256 is not None
        and marker.get("configuration_sha256") != expected_configuration_sha256
    ):
        raise RuntimeError(f"Phase 3 output configuration does not match the requested run: {cube_id}")
    return {
        "status": "passed",
        "cube_id": cube_id,
        "summary_sha256": marker["summary_sha256"],
        "result_sha256": marker["result_sha256"],
        "configuration_sha256": marker["configuration_sha256"],
    }


def run_cube(
    *,
    phase2_root: Path,
    output_root: Path,
    cube_id: str,
    ell_max: int,
    directions_per_radius: int,
    sample_count: int,
    robustness_sample_count: int,
    pair_batch_size: int,
    seed: int,
    fit_interval: tuple[float, float],
    fit_min_count: int,
) -> dict[str, Any]:
    """Run nested-core primary and all-valid robustness modes for one cube."""

    if ell_max > 160:
        raise ValueError("Phase 3 ell_max must not exceed L_sub / 4 = 160")
    if directions_per_radius > 128:
        raise ValueError("Phase 3 directions_per_radius must not exceed 128")
    if sample_count > 131072 or robustness_sample_count > 32768:
        raise ValueError("Phase 3 sample caps exceed the bounded validation scope")
    if pair_batch_size > 32768:
        raise ValueError("Phase 3 pair_batch_size must not exceed 32768")
    source = _phase2_source_identity(phase2_root, cube_id)
    displacements, ell_edges, displacement_payload = _displacement_payload(
        ell_max, directions_per_radius
    )
    configs = {
        "nested_core": _config(
            pair_mode="nested_core",
            ell_edges=ell_edges,
            sample_count=sample_count,
            pair_batch_size=pair_batch_size,
            seed=seed,
            p_values=(2.0,),
        ),
        "all_valid_pairs": _config(
            pair_mode="all_valid_pairs",
            ell_edges=ell_edges,
            sample_count=robustness_sample_count,
            pair_batch_size=pair_batch_size,
            seed=seed,
            p_values=(2.0,),
        ),
    }
    configuration = {
        "cube_id": cube_id,
        "q_names": ("B", "u"),
        "fit_interval_cells": fit_interval,
        "fit_min_count": fit_min_count,
        "displacements": displacement_payload,
        "modes": {name: _config_payload(config) for name, config in configs.items()},
        "axis_order": "KJI: array axes are (k=x3, j=x2, i=x1); offsets are (di,dj,dk)",
        "units": "AthenaK code units with 4 pi absorbed where applicable",
    }
    configuration_sha256 = _mapping_sha256(configuration)
    final_root = output_root / cube_id
    if (final_root / "COMPLETE.json").exists():
        return {
            "reused": True,
            **_verify_cube_output(
                final_root,
                phase2_root,
                cube_id,
                expected_configuration_sha256=configuration_sha256,
                phase2_source=source,
            ),
        }
    if final_root.exists():
        raise RuntimeError(f"incomplete Phase 3 cube output already exists: {final_root}")
    partial_root = output_root / f".{cube_id}.partial"
    if partial_root.exists():
        raise RuntimeError(f"stale Phase 3 partial directory exists: {partial_root}")
    partial_root.mkdir(parents=True)

    source_version = _source_version()
    started = time.perf_counter()
    arrays = _load_cube(phase2_root, cube_id)
    results = {}
    summaries = {}
    for mode, config in configs.items():
        result = compute_finite_domain_structure_functions(
            arrays, displacements, config=config, q_names=("B", "u")
        )
        _save_mode(partial_root / f"{mode}.npz", result, displacements, displacement_payload)
        results[mode] = result
        summaries[mode] = _mode_summary(result, fit_interval=fit_interval, min_count=fit_min_count)
    elapsed = time.perf_counter() - started
    publication_source_version = _source_version()
    if publication_source_version != source_version:
        raise RuntimeError("Phase 3 implementation sources changed during computation")
    publication_phase2_source = _phase2_source_identity(phase2_root, cube_id)
    if publication_phase2_source != source:
        raise RuntimeError("Phase 2 analysis inputs changed during computation")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "cube_id": cube_id,
        "phase2_source": source,
        "source_version": source_version,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "mode_summaries": summaries,
        "nested_vs_all_valid": _relative_mode_difference(
            results["nested_core"], results["all_valid_pairs"]
        ),
        "performance": {
            "total_wall_seconds": elapsed,
            "peak_rss_kib": _peak_rss_kib(),
        },
    }
    _atomic_write_json(partial_root / "summary.json", summary)
    result_sha256 = {
        f"{mode}.npz": file_sha256(partial_root / f"{mode}.npz")
        for mode in configs
    }
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "cube_id": cube_id,
        "phase2_source": source,
        "implementation_sha256": source_version["implementation_sha256"],
        "configuration_sha256": configuration_sha256,
        "summary_sha256": file_sha256(partial_root / "summary.json"),
        "result_sha256": result_sha256,
        "published_unix_seconds": time.time(),
    }
    _atomic_write_json(partial_root / "COMPLETE.json", marker)
    partial_root.replace(final_root)
    return {
        "reused": False,
        **_verify_cube_output(
            final_root,
            phase2_root,
            cube_id,
            expected_configuration_sha256=configuration_sha256,
            phase2_source=source,
        ),
    }


def validate_synthetic(output: Path) -> dict[str, Any]:
    """Run a compact compute-node synthetic gate independent of pytest."""

    rng = np.random.default_rng(20260530)
    shape = (5, 6, 7)
    cube = {
        "rho": 1.0 + rng.random(shape),
        "B_x": 1.0 + 0.1 * rng.standard_normal(shape),
        "B_y": 0.2 * rng.standard_normal(shape),
        "B_z": 0.2 * rng.standard_normal(shape),
        "v_x": rng.standard_normal(shape),
        "v_y": rng.standard_normal(shape),
        "v_z": rng.standard_normal(shape),
    }
    displacements = np.asarray(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, -1), (1, -1, 1), (-1, 1, -1)]
    )
    oracle_checks = []
    for pair_mode in ("nested_core", "all_valid_pairs"):
        config = FiniteDomainConfig(
            np.asarray([0.5, 1.5, 2.5]),
            p_values=(1.0, 2.0, 3.0, 4.0),
            pair_mode=pair_mode,
            sample_count=None,
            pair_batch_size=11,
            seed=17,
        )
        fast = compute_finite_domain_structure_functions(
            cube, displacements, config=config, q_names=("B", "u", "vA", "vA_ref")
        )
        slow = compute_finite_domain_structure_functions_reference(
            cube, displacements, config=config, q_names=("B", "u", "vA", "vA_ref")
        )
        passed = (
            np.array_equal(fast.counts, slow.counts)
            and np.array_equal(fast.exclusions, slow.exclusions)
            and np.allclose(fast.sums, slow.sums, rtol=2.0e-14, atol=2.0e-14)
            and np.allclose(fast.sums_sq, slow.sums_sq, rtol=2.0e-14, atol=2.0e-14)
            and np.array_equal(fast.sampled_pairs, slow.sampled_pairs)
            and np.array_equal(fast.eligible_pairs, slow.eligible_pairs)
        )
        oracle_checks.append({"pair_mode": pair_mode, "passed": bool(passed)})

    isotropic_shape = (14, 14, 14)
    isotropic_rng = np.random.default_rng(137)
    isotropic_cube = {
        "rho": np.ones(isotropic_shape),
        "B_x": np.zeros(isotropic_shape),
        "B_y": np.zeros(isotropic_shape),
        "B_z": np.ones(isotropic_shape),
        "v_x": isotropic_rng.standard_normal(isotropic_shape),
        "v_y": isotropic_rng.standard_normal(isotropic_shape),
        "v_z": isotropic_rng.standard_normal(isotropic_shape),
    }
    isotropic_offsets = generate_fibonacci_displacements((1.0, 2.0, 3.0), directions_per_radius=48)
    isotropic = compute_finite_domain_structure_functions(
        isotropic_cube,
        isotropic_offsets,
        config=FiniteDomainConfig(
            np.asarray([0.5, 1.5, 2.5, 3.5]),
            pair_mode="all_valid_pairs",
            sample_count=None,
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    total = isotropic.measurement_names.index("total")
    parallel = isotropic.direction_names.index("parallel")
    perpendicular = isotropic.direction_names.index("perpendicular")
    iso_parallel = float(np.nansum(isotropic.sums[0, 0, total, parallel, 0]) / np.sum(isotropic.counts[0, 0, total, parallel, 0]))
    iso_perpendicular = float(
        np.nansum(isotropic.sums[0, 0, total, perpendicular, 0])
        / np.sum(isotropic.counts[0, 0, total, perpendicular, 0])
    )
    isotropic_ratio = iso_parallel / iso_perpendicular

    guide_shape = (8, 8, 8)
    k, j, i = np.indices(guide_shape, dtype=float)
    guide_cube = {
        "rho": np.ones(guide_shape),
        "B_x": np.zeros(guide_shape),
        "B_y": np.zeros(guide_shape),
        "B_z": np.ones(guide_shape),
        "v_x": np.zeros(guide_shape),
        "v_y": k,
        "v_z": np.zeros(guide_shape),
    }
    guide = compute_finite_domain_structure_functions(
        guide_cube,
        np.asarray([(0, 0, 1), (1, 0, 0)]),
        config=FiniteDomainConfig(
            np.asarray([0.5, 1.5]),
            pair_mode="all_valid_pairs",
            sample_count=None,
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    guide_parallel = float(guide.moments[0, 0, total, parallel, 0, 0])
    guide_perpendicular = float(guide.moments[0, 0, total, perpendicular, 0, 0])

    ribbon_shape = (10, 10, 10)
    k, j, i = np.indices(ribbon_shape, dtype=float)
    ribbon_cube = {
        "rho": np.ones(ribbon_shape),
        "B_x": np.zeros(ribbon_shape),
        "B_y": np.zeros(ribbon_shape),
        "B_z": np.ones(ribbon_shape),
        "v_x": 0.5 * k + 2.0 * j + i,
        "v_y": np.zeros(ribbon_shape),
        "v_z": np.zeros(ribbon_shape),
    }
    ribbon_offsets = np.asarray(
        [
            (0, 0, radius)
            for radius in (1, 2, 3, 4)
        ]
        + [
            (radius, 0, 0)
            for radius in (1, 2, 3, 4)
        ]
        + [
            (0, radius, 0)
            for radius in (1, 2, 3, 4)
        ]
    )
    ribbon = compute_finite_domain_structure_functions(
        ribbon_cube,
        ribbon_offsets,
        config=FiniteDomainConfig(
            np.asarray([0.5, 1.5, 2.5, 3.5, 4.5]),
            pair_mode="all_valid_pairs",
            sample_count=None,
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    ribbon_shapes = constant_sp_shapes(ribbon, "u", 2.0, (4.0,))

    degenerate_shape = (3, 3, 4)
    _, _, i = np.indices(degenerate_shape)
    degenerate_cube = {
        "rho": np.ones(degenerate_shape),
        "B_x": np.where(i % 2 == 0, 1.0, -1.0),
        "B_y": np.zeros(degenerate_shape),
        "B_z": np.zeros(degenerate_shape),
        "v_x": np.zeros(degenerate_shape),
        "v_y": np.ones(degenerate_shape),
        "v_z": np.zeros(degenerate_shape),
    }
    degenerate = compute_finite_domain_structure_functions(
        degenerate_cube,
        np.asarray([(1, 0, 0)]),
        config=FiniteDomainConfig(
            np.asarray([0.5, 1.5]),
            pair_mode="all_valid_pairs",
            sample_count=None,
            include_subvolume_mean=False,
        ),
        q_names=("u",),
    )
    weak_B = degenerate.exclusion_names.index("weak_B_direction")
    degenerate_weak_B_count = int(degenerate.exclusions[0, 0, weak_B, 0])

    sensitivity_shape = (12, 12, 12)
    sensitivity_cube = {name: values[:12, :12, :12] for name, values in isotropic_cube.items()}
    core_fractions = {}
    for ell_max in (2, 4):
        offsets = generate_fibonacci_displacements((1.0, float(ell_max)), directions_per_radius=16)
        result = compute_finite_domain_structure_functions(
            sensitivity_cube,
            offsets,
            config=FiniteDomainConfig(
                np.asarray([0.5, 1.5, 2.5, 4.5]),
                pair_mode="nested_core",
                sample_count=64,
                include_subvolume_mean=False,
            ),
            q_names=("u",),
        )
        core = result.nested_core_bounds_kji
        assert core is not None
        core_fractions[str(ell_max)] = float(
            np.prod([stop - start for start, stop in core]) / np.prod(result.cube_shape_kji)
        )

    science_checks = {
        "isotropic_total_parallel_over_perpendicular": {
            "value": isotropic_ratio,
            "accepted_interval": [0.8, 1.2],
            "passed": bool(0.8 <= isotropic_ratio <= 1.2),
        },
        "guide_field_parallel_exceeds_perpendicular": {
            "parallel_S2": guide_parallel,
            "perpendicular_S2": guide_perpendicular,
            "passed": bool(guide_parallel > 0.0 and guide_perpendicular == 0.0),
        },
        "ribbon_fixed_S2_scale_ordering": {
            "ell_parallel": float(ribbon_shapes["parallel"][0]),
            "xi": float(ribbon_shapes["xi"][0]),
            "lambda": float(ribbon_shapes["lambda"][0]),
            "passed": bool(
                ribbon_shapes["parallel_quality"][0] == "ok"
                and ribbon_shapes["xi_quality"][0] == "ok"
                and ribbon_shapes["lambda_quality"][0] == "ok"
                and ribbon_shapes["parallel"][0] > ribbon_shapes["xi"][0] > ribbon_shapes["lambda"][0]
            ),
        },
        "degenerate_local_B_is_excluded": {
            "weak_B_direction_count": degenerate_weak_B_count,
            "passed": bool(degenerate_weak_B_count > 0),
        },
        "nested_core_shrinks_with_ell_max": {
            "retained_volume_fraction": core_fractions,
            "passed": bool(core_fractions["4"] < core_fractions["2"]),
        },
    }
    passed = all(item["passed"] for item in oracle_checks) and all(
        item["passed"] for item in science_checks.values()
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "oracle_checks": oracle_checks,
        "science_checks": science_checks,
        "source_version": _source_version(),
        "peak_rss_kib": _peak_rss_kib(),
    }
    _atomic_write_json(output, payload)
    if payload["status"] != "passed":
        raise RuntimeError("synthetic fast-versus-oracle validation failed")
    return payload


def _verify_seed_robustness_output(
    robustness_root: Path,
    phase2_root: Path,
    cube_id: str,
) -> dict[str, Any]:
    marker_path = robustness_root / "ROBUSTNESS_COMPLETE.json"
    summary_path = robustness_root / "seed_robustness_summary.json"
    marker = json.loads(marker_path.read_text())
    source = _phase2_source_identity(phase2_root, cube_id)
    current_source_version = _source_version()
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("status") != "passed"
        or marker.get("cube_id") != cube_id
        or marker.get("phase2_source") != source
        or marker.get("implementation_sha256") != current_source_version["implementation_sha256"]
    ):
        raise RuntimeError(f"invalid Phase 3 seed-robustness marker: {marker_path}")
    if marker.get("summary_sha256") != file_sha256(summary_path):
        raise RuntimeError(f"Phase 3 seed-robustness summary checksum mismatch: {summary_path}")
    result_sha256 = marker.get("result_sha256", {})
    expected_paths = {f"all_valid_pairs_seed{seed}.npz" for seed in ROBUSTNESS_SEEDS[1:]}
    if set(result_sha256) != expected_paths:
        raise RuntimeError(f"Phase 3 seed-robustness marker has an invalid result inventory: {marker_path}")
    for relative_path, expected in result_sha256.items():
        if file_sha256(robustness_root / relative_path) != expected:
            raise RuntimeError(f"Phase 3 seed-robustness checksum mismatch: {robustness_root / relative_path}")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "passed"
        or summary.get("cube_id") != cube_id
        or summary.get("phase2_source") != source
        or summary.get("source_version", {}).get("implementation_sha256")
        != marker.get("implementation_sha256")
    ):
        raise RuntimeError(f"invalid Phase 3 seed-robustness summary identity: {summary_path}")
    if _mapping_sha256(summary.get("configuration")) != marker.get("configuration_sha256"):
        raise RuntimeError(f"Phase 3 seed-robustness configuration checksum mismatch: {summary_path}")
    base_path = robustness_root.parent / cube_id / "all_valid_pairs.npz"
    if summary.get("base_result_sha256") != file_sha256(base_path):
        raise RuntimeError(f"Phase 3 seed-robustness base result checksum mismatch: {base_path}")
    return {
        "status": "passed",
        "cube_id": cube_id,
        "summary_sha256": marker["summary_sha256"],
        "result_sha256": marker["result_sha256"],
        "configuration_sha256": marker["configuration_sha256"],
    }


def _verify_subvolume_parallel_coverage(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Require empirical fixed-frame parallel occupancy throughout the fitted range."""

    configuration = summary["configuration"]
    fit_interval = tuple(float(value) for value in configuration["fit_interval_cells"])
    min_count = int(configuration["fit_min_count"])
    q_names = tuple(configuration["q_names"])
    rows = summary["mode_summaries"]["nested_core"]["coverage_rows"]
    ell_centers = sorted(
        {
            float(row["ell_center"])
            for row in rows
            if fit_interval[0] <= float(row["ell_center"]) <= fit_interval[1]
        }
    )
    if not ell_centers:
        raise RuntimeError("Phase 3 nested-core coverage has no fitted separation bins")
    counts: dict[str, dict[str, int]] = {}
    for q_name in q_names:
        selected = {
            float(row["ell_center"]): int(row["accepted"])
            for row in rows
            if row["q"] == q_name
            and row["geometry"] == "subvolume_mean"
            and row["measurement"] == "perpendicular"
            and row["direction"] == "parallel"
            and float(row["p"]) == 2.0
            and fit_interval[0] <= float(row["ell_center"]) <= fit_interval[1]
        }
        if sorted(selected) != ell_centers or any(value < min_count for value in selected.values()):
            raise RuntimeError(
                f"Phase 3 subvolume-mean parallel coverage is underpowered for {summary['cube_id']} {q_name}"
            )
        counts[q_name] = {str(ell): selected[ell] for ell in ell_centers}
    return {
        "status": "passed",
        "cube_id": summary["cube_id"],
        "fit_interval_cells": fit_interval,
        "minimum_required_count": min_count,
        "accepted_counts": counts,
    }


def run_seed_robustness(
    *,
    phase2_root: Path,
    output_root: Path,
    cube_id: str,
    ell_max: int,
    directions_per_radius: int,
    sample_count: int,
    pair_batch_size: int,
    fit_interval: tuple[float, float],
    fit_min_count: int,
) -> dict[str, Any]:
    """Publish a matched-depth all-valid repeated-seed diagnostic for one cube."""

    if tuple(ROBUSTNESS_SEEDS) != tuple(sorted(set(ROBUSTNESS_SEEDS))):
        raise RuntimeError("ROBUSTNESS_SEEDS must be sorted and unique")
    _verify_cube_output(output_root / cube_id, phase2_root, cube_id)
    cube_summary = json.loads((output_root / cube_id / "summary.json").read_text())
    source = _phase2_source_identity(phase2_root, cube_id)
    displacements, ell_edges, displacement_payload = _displacement_payload(
        ell_max, directions_per_radius
    )
    configs = {
        seed: _config(
            pair_mode="all_valid_pairs",
            ell_edges=ell_edges,
            sample_count=sample_count,
            pair_batch_size=pair_batch_size,
            seed=seed,
            p_values=(2.0,),
        )
        for seed in ROBUSTNESS_SEEDS
    }
    expected_base = _config_payload(configs[ROBUSTNESS_SEEDS[0]])
    if (
        cube_summary["configuration"]["displacements"] != _json_builtin(displacement_payload)
        or cube_summary["configuration"]["modes"]["all_valid_pairs"] != _json_builtin(expected_base)
    ):
        raise RuntimeError("seed-robustness diagnostic does not match the published base cube configuration")

    final_root = output_root / "seed_robustness"
    if (final_root / "ROBUSTNESS_COMPLETE.json").exists():
        return {"reused": True, **_verify_seed_robustness_output(final_root, phase2_root, cube_id)}
    if final_root.exists():
        raise RuntimeError(f"incomplete Phase 3 seed-robustness output already exists: {final_root}")
    partial_root = output_root / ".seed_robustness.partial"
    if partial_root.exists():
        raise RuntimeError(f"stale Phase 3 seed-robustness partial directory exists: {partial_root}")
    partial_root.mkdir(parents=True)

    source_version = _source_version()
    arrays = _load_cube(phase2_root, cube_id)
    base_path = output_root / cube_id / "all_valid_pairs.npz"
    with np.load(base_path, allow_pickle=False) as base_payload:
        base_moments = base_payload["moments"].copy()
    summaries = {
        str(ROBUSTNESS_SEEDS[0]): cube_summary["mode_summaries"]["all_valid_pairs"],
    }
    comparisons = {}
    result_paths = {}
    for seed in ROBUSTNESS_SEEDS[1:]:
        result = compute_finite_domain_structure_functions(
            arrays, displacements, config=configs[seed], q_names=("B", "u")
        )
        relative_path = f"all_valid_pairs_seed{seed}.npz"
        _save_mode(partial_root / relative_path, result, displacements, displacement_payload)
        result_paths[relative_path] = partial_root / relative_path
        summaries[str(seed)] = _mode_summary(
            result, fit_interval=fit_interval, min_count=fit_min_count
        )
        comparisons[str(seed)] = _relative_array_difference(base_moments, result.moments)

    publication_source_version = _source_version()
    if publication_source_version != source_version:
        raise RuntimeError("Phase 3 implementation sources changed during seed-robustness computation")
    publication_phase2_source = _phase2_source_identity(phase2_root, cube_id)
    if publication_phase2_source != source:
        raise RuntimeError("Phase 2 analysis inputs changed during seed-robustness computation")
    configuration = {
        "cube_id": cube_id,
        "q_names": ("B", "u"),
        "fit_interval_cells": fit_interval,
        "fit_min_count": fit_min_count,
        "displacements": displacement_payload,
        "pair_mode": "all_valid_pairs",
        "sample_count": sample_count,
        "pair_batch_size": pair_batch_size,
        "seeds": ROBUSTNESS_SEEDS,
    }
    configuration_sha256 = _mapping_sha256(configuration)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "cube_id": cube_id,
        "phase2_source": source,
        "source_version": source_version,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "base_result_sha256": file_sha256(base_path),
        "seed_summaries": summaries,
        "repeat_seed_vs_base": comparisons,
    }
    _atomic_write_json(partial_root / "seed_robustness_summary.json", summary)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "cube_id": cube_id,
        "phase2_source": source,
        "implementation_sha256": source_version["implementation_sha256"],
        "configuration_sha256": configuration_sha256,
        "summary_sha256": file_sha256(partial_root / "seed_robustness_summary.json"),
        "result_sha256": {
            relative_path: file_sha256(path)
            for relative_path, path in result_paths.items()
        },
        "published_unix_seconds": time.time(),
    }
    _atomic_write_json(partial_root / "ROBUSTNESS_COMPLETE.json", marker)
    partial_root.replace(final_root)
    return {
        "reused": False,
        **_verify_seed_robustness_output(final_root, phase2_root, cube_id),
    }


def profile_controls(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    """Measure bounded single-factor sampler costs on one approved real cube."""

    cube_id = BENCHMARK_CUBE_IDS[0]
    output_path = output_root / "control_profile.json"
    marker_path = output_root / "CONTROL_PROFILE_COMPLETE.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        payload = json.loads(output_path.read_text())
        source = _phase2_source_identity(phase2_root, cube_id)
        if (
            marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("status") != "passed"
            or marker.get("cube_id") != cube_id
            or marker.get("implementation_sha256") != _source_version()["implementation_sha256"]
            or marker.get("profile_sha256") != file_sha256(output_path)
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("status") != "passed"
            or payload.get("cube_id") != cube_id
            or payload.get("phase2_source") != source
            or payload.get("source_version", {}).get("implementation_sha256")
            != marker.get("implementation_sha256")
        ):
            raise RuntimeError(f"invalid Phase 3 control-profile marker: {marker_path}")
        return {"reused": True, **payload}
    if output_path.exists():
        raise RuntimeError(f"incomplete Phase 3 control profile already exists: {output_path}")

    source = _phase2_source_identity(phase2_root, cube_id)
    source_version = _source_version()
    arrays = _load_cube(phase2_root, cube_id)
    scenarios = (
        ("baseline", 64, 16, 4096, 1024, ("B", "u"), (2.0,)),
        ("B_only", 64, 16, 4096, 1024, ("B",), (2.0,)),
        ("u_only", 64, 16, 4096, 1024, ("u",), (2.0,)),
        ("batch_512", 64, 16, 4096, 512, ("B", "u"), (2.0,)),
        ("batch_2048", 64, 16, 4096, 2048, ("B", "u"), (2.0,)),
        ("samples_2048", 64, 16, 2048, 1024, ("B", "u"), (2.0,)),
        ("samples_8192", 64, 16, 8192, 1024, ("B", "u"), (2.0,)),
        ("ell_max_128", 128, 16, 4096, 1024, ("B", "u"), (2.0,)),
        ("p_1_2_3_4", 64, 16, 4096, 1024, ("B", "u"), (1.0, 2.0, 3.0, 4.0)),
    )
    rows = []
    for name, ell_max, directions, samples, batch_size, q_names, p_values in scenarios:
        displacements, ell_edges, displacement_payload = _displacement_payload(ell_max, directions)
        config = _config(
            pair_mode="nested_core",
            ell_edges=ell_edges,
            sample_count=samples,
            pair_batch_size=batch_size,
            seed=20260530,
            p_values=p_values,
        )
        result = compute_finite_domain_structure_functions(
            arrays, displacements, config=config, q_names=q_names
        )
        rows.append(
            {
                "scenario": name,
                "ell_max_cells": ell_max,
                "directions_per_radius": directions,
                "displacement_count": displacement_payload["displacement_count"],
                "separation_bin_count": len(ell_edges) - 1,
                "direction_bin_names": result.direction_names,
                "q_names": q_names,
                "p_values": p_values,
                "sample_count_per_displacement": samples,
                "pair_batch_size": batch_size,
                "sampler_elapsed_seconds": result.elapsed_seconds,
            }
        )
    if _source_version() != source_version:
        raise RuntimeError("Phase 3 implementation sources changed during control profiling")
    if _phase2_source_identity(phase2_root, cube_id) != source:
        raise RuntimeError("Phase 2 analysis inputs changed during control profiling")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "cube_id": cube_id,
        "phase2_source": source,
        "source_version": source_version,
        "chunk_size_note": "The in-memory sampler has no separate I/O chunk control; pair_batch_size is its bounded work-chunk control.",
        "peak_rss_kib": _peak_rss_kib(),
        "scenarios": rows,
    }
    _atomic_write_json(output_path, payload)
    _atomic_write_json(
        marker_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "cube_id": cube_id,
            "implementation_sha256": source_version["implementation_sha256"],
            "profile_sha256": file_sha256(output_path),
            "published_unix_seconds": time.time(),
        },
    )
    return {"reused": False, **payload}


def summarize(phase2_root: Path, output_root: Path) -> dict[str, Any]:
    """Verify four smoke outputs and write the ledger-backed Phase 4 forecast."""

    synthetic_path = output_root / "synthetic_validation.json"
    synthetic = json.loads(synthetic_path.read_text())
    if (
        synthetic.get("schema_version") != SCHEMA_VERSION
        or synthetic.get("status") != "passed"
        or synthetic.get("source_version", {}).get("implementation_sha256")
        != _source_version()["implementation_sha256"]
    ):
        raise RuntimeError("Phase 3 synthetic validation artifact is missing or stale")
    robustness = _verify_seed_robustness_output(
        output_root / "seed_robustness", phase2_root, BENCHMARK_CUBE_IDS[0]
    )
    cubes = []
    for cube_id in BENCHMARK_CUBE_IDS:
        verification = _verify_cube_output(output_root / cube_id, phase2_root, cube_id)
        summary = json.loads((output_root / cube_id / "summary.json").read_text())
        cubes.append({"verification": verification, "summary": summary})
    campaign_configs = {
        json.dumps(
            {
                key: value
                for key, value in item["summary"]["configuration"].items()
                if key != "cube_id"
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in cubes
    }
    if len(campaign_configs) != 1:
        raise RuntimeError("Phase 3 smoke cubes do not share one campaign configuration")
    if not all(item["summary"]["nested_vs_all_valid"]["sampling_depth_matched"] for item in cubes):
        raise RuntimeError("Phase 3 smoke requires matched nested-core and all-valid sampling depth")
    subvolume_parallel_coverage = [
        _verify_subvolume_parallel_coverage(item["summary"]) for item in cubes
    ]
    measured_seconds = sum(item["summary"]["performance"]["total_wall_seconds"] for item in cubes)
    measured_bytes = sum(
        sum((output_root / item["summary"]["cube_id"] / name).stat().st_size for name in ("nested_core.npz", "all_valid_pairs.npz", "summary.json", "COMPLETE.json"))
        for item in cubes
    )
    peak_rss = max(item["summary"]["performance"]["peak_rss_kib"] for item in cubes)
    scale_factor = 21.0 / len(BENCHMARK_CUBE_IDS)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "cube_ids": BENCHMARK_CUBE_IDS,
        "measured_cube_count": len(BENCHMARK_CUBE_IDS),
        "campaign_cube_count": 21,
        "forecast_scale_factor": scale_factor,
        "measured_total_wall_seconds": measured_seconds,
        "forecast_21_cube_sampler_wall_seconds": measured_seconds * scale_factor,
        "forecast_21_cube_sampler_node_hours": measured_seconds * scale_factor / 3600.0,
        "measured_output_bytes": measured_bytes,
        "forecast_21_cube_sampler_output_bytes": measured_bytes * scale_factor,
        "peak_rss_kib": peak_rss,
        "source_version": _source_version(),
        "synthetic_validation_sha256": file_sha256(synthetic_path),
        "seed_robustness": robustness,
        "seed_robustness_marker_sha256": file_sha256(
            output_root / "seed_robustness" / "ROBUSTNESS_COMPLETE.json"
        ),
        "subvolume_mean_parallel_coverage": subvolume_parallel_coverage,
        "cube_summary_sha256": {
            f"{cube_id}/summary.json": file_sha256(output_root / cube_id / "summary.json")
            for cube_id in BENCHMARK_CUBE_IDS
        },
    }
    summary_json = output_root / "phase3_smoke_summary.json"
    _atomic_write_json(summary_json, payload)
    markdown = output_root / "phase3_smoke_summary.md"
    markdown.write_text(
        "# Phase 3 four-cube sampler smoke summary\n\n"
        f"- Measured four-cube sampler wall time: `{measured_seconds:.3f} s`\n"
        f"- Linear 21-cube sampler forecast: `{measured_seconds * scale_factor:.3f} s`\n"
        f"- Linear 21-cube sampler node-hour forecast: `{measured_seconds * scale_factor / 3600.0:.6f}`\n"
        f"- Peak RSS: `{peak_rss} KiB`\n"
        f"- Measured output bytes: `{measured_bytes}`\n"
    )
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "cube_ids": BENCHMARK_CUBE_IDS,
        "summary_sha256": file_sha256(summary_json),
        "summary_markdown_sha256": file_sha256(markdown),
        "synthetic_validation_sha256": file_sha256(synthetic_path),
        "seed_robustness_marker_sha256": file_sha256(
            output_root / "seed_robustness" / "ROBUSTNESS_COMPLETE.json"
        ),
        "published_unix_seconds": time.time(),
    }
    _atomic_write_json(output_root / "PHASE3_SMOKE_COMPLETE.json", marker)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("synthetic", "smoke", "verify", "robustness", "profile-controls", "summarize"))
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cube-id", action="append")
    parser.add_argument("--ell-max", type=int, default=128)
    parser.add_argument("--directions-per-radius", type=int, default=96)
    parser.add_argument("--sample-count", type=int, default=32768)
    parser.add_argument("--robustness-sample-count", type=int, default=32768)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--fit-interval", type=float, nargs=2, default=(8.0, 96.0))
    parser.add_argument("--fit-min-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cube_ids = tuple(args.cube_id or BENCHMARK_CUBE_IDS)
    if any(cube_id not in BENCHMARK_CUBE_IDS for cube_id in cube_ids):
        raise SystemExit("Phase 3 is restricted to the four approved Phase 2 cube IDs")
    if args.action == "synthetic":
        print(json.dumps(validate_synthetic(args.output_root / "synthetic_validation.json"), indent=2, sort_keys=True))
    elif args.action == "smoke":
        for cube_id in cube_ids:
            payload = run_cube(
                phase2_root=args.phase2_root,
                output_root=args.output_root,
                cube_id=cube_id,
                ell_max=args.ell_max,
                directions_per_radius=args.directions_per_radius,
                sample_count=args.sample_count,
                robustness_sample_count=args.robustness_sample_count,
                pair_batch_size=args.pair_batch_size,
                seed=args.seed,
                fit_interval=tuple(args.fit_interval),
                fit_min_count=args.fit_min_count,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.action == "verify":
        for cube_id in cube_ids:
            print(json.dumps(_verify_cube_output(args.output_root / cube_id, args.phase2_root, cube_id), indent=2, sort_keys=True))
    elif args.action == "robustness":
        if tuple(cube_ids) != (BENCHMARK_CUBE_IDS[0],):
            raise SystemExit("Phase 3 seed robustness is restricted to the first approved benchmark cube")
        print(
            json.dumps(
                run_seed_robustness(
                    phase2_root=args.phase2_root,
                    output_root=args.output_root,
                    cube_id=cube_ids[0],
                    ell_max=args.ell_max,
                    directions_per_radius=args.directions_per_radius,
                    sample_count=args.sample_count,
                    pair_batch_size=args.pair_batch_size,
                    fit_interval=tuple(args.fit_interval),
                    fit_min_count=args.fit_min_count,
                ),
                indent=2,
                sort_keys=True,
                )
            )
    elif args.action == "profile-controls":
        print(json.dumps(profile_controls(args.phase2_root, args.output_root), indent=2, sort_keys=True))
    else:
        print(json.dumps(summarize(args.phase2_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
