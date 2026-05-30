#!/usr/bin/env python3
"""Reproducible CPU microbenchmarks for validated SFunctor kernels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sfunctor.core.directional import DirectionalConfig, compute_directional_structure_functions
from sfunctor.core.histograms import Channel, N_CHANNELS, compute_histogram_for_disp_2D
from sfunctor.core.physics import compute_vA, compute_z_plus_minus
from sfunctor.io.slice_io import load_slice_npz
from sfunctor.reference import compute_directional_structure_functions_reference


def _fields(shape=(64, 64), seed=3):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.1 * rng.random(shape)
    vector = lambda offset=0.0: offset + 0.1 * rng.normal(size=shape)
    vx, vy, vz = vector(), vector(), vector()
    bx, by, bz = vector(1.0), vector(), vector()
    vax, vay, vaz = compute_vA(bx, by, bz, rho)
    zp, zm = compute_z_plus_minus(vx, vy, vz, vax, vay, vaz)
    zeros = np.zeros(shape)
    return {
        "v_x": vx, "v_y": vy, "v_z": vz,
        "B_x": bx, "B_y": by, "B_z": bz, "rho": rho,
        "vA_x": vax, "vA_y": vay, "vA_z": vaz,
        "zp_x": zp[0], "zp_y": zp[1], "zp_z": zp[2],
        "zm_x": zm[0], "zm_y": zm[1], "zm_z": zm[2],
        "omega_x": zeros, "omega_y": zeros, "omega_z": zeros,
        "j_x": zeros, "j_y": zeros, "j_z": zeros,
        "curv_x": zeros, "curv_y": zeros, "curv_z": zeros,
        "grad_rho_x": zeros, "grad_rho_y": zeros, "grad_rho_z": zeros,
    }


def _call_hist(fields, displacement, edges, compact):
    dx, dy = displacement
    return compute_histogram_for_disp_2D(
        fields["v_x"], fields["v_y"], fields["v_z"],
        fields["B_x"], fields["B_y"], fields["B_z"], fields["rho"],
        fields["vA_x"], fields["vA_y"], fields["vA_z"],
        fields["zp_x"], fields["zp_y"], fields["zp_z"],
        fields["zm_x"], fields["zm_y"], fields["zm_z"],
        fields["omega_x"], fields["omega_y"], fields["omega_z"],
        fields["j_x"], fields["j_y"], fields["j_z"],
        fields["curv_x"], fields["curv_y"], fields["curv_z"],
        fields["grad_rho_x"], fields["grad_rho_y"], fields["grad_rho_z"],
        int(dx), int(dy), 3, 128,
        edges["ell"], edges["theta"], edges["phi"], edges["delta"],
        2, (1.0, 1.0, 1.0), 123, compact,
    )


def _legacy_allocation_benchmark() -> dict[str, float]:
    fields = _fields()
    edges = {
        "ell": np.linspace(0.0, 64.0, 49),
        "theta": np.linspace(0.0, np.pi / 2.0, 9),
        "phi": np.linspace(0.0, np.pi / 2.0, 8),
        "delta": tuple(np.linspace(0.0, 10.0, 128) for _ in range(N_CHANNELS)),
    }
    displacements = np.array([[1, 0], [0, 1], [2, 1], [1, 2], [3, 1], [1, 3], [2, 3], [3, 2]])
    _call_hist(fields, displacements[0], edges, True)  # JIT warm-up
    output = {}
    for compact in (False, True):
        total = np.zeros((N_CHANNELS, 48, 8, 7, 127), dtype=np.int64)
        start = perf_counter()
        for displacement in displacements:
            part = _call_hist(fields, displacement, edges, compact)
            ell = np.hypot(*displacement)
            ell_index = np.searchsorted(edges["ell"], ell, side="right") - 1
            if compact:
                total[:, ell_index] += part[:, 0]
            else:
                total += part
        output["compact_seconds" if compact else "dense_seconds"] = perf_counter() - start
        output["total_counts"] = int(total.sum())
    dense_bytes = N_CHANNELS * 48 * 8 * 7 * 127 * np.dtype(np.int64).itemsize
    compact_bytes = N_CHANNELS * 1 * 8 * 7 * 127 * np.dtype(np.int64).itemsize
    output.update(dense_histogram_mib=dense_bytes / 2**20, compact_slab_mib=compact_bytes / 2**20)
    output["speedup"] = output["dense_seconds"] / output["compact_seconds"]
    return output


def _sampler_benchmark(population: int, sample_count: int) -> dict[str, float]:
    output = {"population": population, "sample_count": sample_count}
    rng = np.random.default_rng(2)
    start = perf_counter()
    rng.choice(population, size=sample_count, replace=False)
    output["without_replacement_seconds"] = perf_counter() - start
    start = perf_counter()
    rng.integers(0, population, size=sample_count)
    output["replacement_seconds"] = perf_counter() - start
    output["speedup"] = output["without_replacement_seconds"] / output["replacement_seconds"]
    return output


def _slice_to_directional_data(slice_data):
    return {
        "rho": slice_data["rho"],
        "B_x": slice_data["B_x"], "B_y": slice_data["B_y"], "B_z": slice_data["B_z"],
        "v_x": slice_data["v_x"], "v_y": slice_data["v_y"], "v_z": slice_data["v_z"],
    }


def _directional_benchmark(slice_path: str | None) -> dict[str, float | list[int] | str]:
    data = _slice_to_directional_data(load_slice_npz(slice_path, stride=8)) if slice_path else {
        "rho": _fields((32, 32))["rho"],
        "B_x": _fields((32, 32))["B_x"], "B_y": _fields((32, 32))["B_y"], "B_z": _fields((32, 32))["B_z"],
        "v_x": _fields((32, 32))["v_x"], "v_y": _fields((32, 32))["v_y"], "v_z": _fields((32, 32))["v_z"],
    }
    displacements = np.array([[1, 0], [0, 1], [1, 1], [2, 1]])
    config = DirectionalConfig(np.array([0.5, 1.5, 2.5, 4.0]), include_global=True)
    start = perf_counter()
    slow = compute_directional_structure_functions_reference(data, displacements, slice_axis=3, config=config)
    slow_seconds = perf_counter() - start
    fast = compute_directional_structure_functions(data, displacements, slice_axis=3, config=config)
    return {
        "source": slice_path or "synthetic",
        "shape": list(np.asarray(data["rho"]).shape),
        "reference_seconds": slow_seconds,
        "vectorized_seconds": fast.elapsed_seconds,
        "speedup": slow_seconds / fast.elapsed_seconds,
        "counts_equal": bool(np.array_equal(slow.counts, fast.counts)),
        "sums_close": bool(np.allclose(slow.sums, fast.sums, rtol=1e-13, atol=1e-13)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", default=None, help="Optional extracted slice NPZ; benchmark loads it with stride 8")
    parser.add_argument("--sampler_population", type=int, default=2560**2)
    parser.add_argument("--sampler_draws", type=int, default=2000)
    parser.add_argument("--output", default=None, help="Optional JSON output")
    args = parser.parse_args(argv)
    results = {
        "sampler": _sampler_benchmark(args.sampler_population, args.sampler_draws),
        "legacy_histogram_allocation": _legacy_allocation_benchmark(),
        "directional_reference": _directional_benchmark(args.slice),
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
