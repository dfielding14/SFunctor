#!/usr/bin/env python3
"""Selective readers and moment utilities for the Prompt 1 cbin census."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


DOMAIN_CELLS = 10240
MESH_BLOCK_CELLS = (160, 320, 320)  # x1, x2, x3
RANK_LATTICE = tuple(n // b for n, b in zip((DOMAIN_CELLS,) * 3, MESH_BLOCK_CELLS))
EXPECTED_RANKS = math.prod(RANK_LATTICE)
PRIMARY_BASES = ("dens", "mom1", "mom2", "mom3", "ener", "bcc1", "bcc2", "bcc3")
PRIMARY_LABELS = tuple(f"{base}_{suffix}" for base in PRIMARY_BASES
                       for suffix in ("1st", "2nd", "3rd", "4th"))
FLOAT32_EPS = np.finfo(np.float32).eps
FLOAT64_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class BinaryRecord:
    """Metadata for one mesh-block payload in a bin or cbin shard."""

    indices: tuple[int, int, int, int, int, int]
    logical: tuple[int, int, int, int]
    geometry: tuple[float, float, float, float, float, float]
    payload_offset: int
    shape_kji: tuple[int, int, int]


@dataclass(frozen=True)
class BinaryShard:
    """Parsed header and record offsets for one AthenaK binary shard."""

    path: str
    time: float
    cycle: int
    coarsen_factor: int
    number_of_moments: int
    var_names: tuple[str, ...]
    variable_dtype: np.dtype
    location_dtype: np.dtype
    full_grid_cells: tuple[int, int, int]
    mesh_block_cells: tuple[int, int, int]
    coarse_grid_cells: tuple[int, int, int]
    coarse_block_cells: tuple[int, int, int]
    nghost: int
    records: tuple[BinaryRecord, ...]
    file_size: int


def _parse_assignment(line: bytes) -> tuple[str, str]:
    text = line.decode("utf-8").strip()
    if "=" not in text:
        raise ValueError(f"expected assignment line, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), value.strip()


def _parse_parameter_dump(raw: bytes) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    section = ""
    for raw_line in raw.decode("utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            section = line[1:-1]
            sections.setdefault(section, {})
            continue
        if "=" in line and section:
            key, value = line.split("=", 1)
            sections[section][key.strip()] = value.strip()
    return sections


def parse_binary_shard(path: str | Path, *, expect_cbin: bool | None = None) -> BinaryShard:
    """Parse shard metadata and payload offsets without loading payload arrays."""

    path = str(path)
    with open(path, "rb") as stream:
        magic = stream.readline().split()
        if not magic or magic[0] != b"Athena":
            raise ValueError(f"{path}: not an Athena binary output")
        if magic[-1].split(b"=")[-1] != b"1.1":
            raise ValueError(f"{path}: unsupported Athena binary version")

        _, preheader_text = _parse_assignment(stream.readline())
        preheader_lines = int(preheader_text)
        preheader: dict[str, str] = {}
        for _ in range(preheader_lines - 1):
            key, value = _parse_assignment(stream.readline())
            preheader[key] = value

        _, nvars_text = _parse_assignment(stream.readline())
        nvars = int(nvars_text)
        var_tokens = stream.readline().decode("utf-8").split()
        var_names = tuple(var_tokens[1:])
        if len(var_names) != nvars:
            raise ValueError(f"{path}: header declares {nvars} variables, found {len(var_names)}")

        _, dump_size_text = _parse_assignment(stream.readline())
        dump = stream.read(int(dump_size_text))
        sections = _parse_parameter_dump(dump)
        mesh = sections["mesh"]
        meshblock = sections["meshblock"]

        factor = int(preheader.get("coarsening factor", "1"))
        moments = int(preheader.get("number of moments", "1"))
        is_cbin = factor != 1 or "coarsening factor" in preheader
        if expect_cbin is True and not is_cbin:
            raise ValueError(f"{path}: expected cbin header")
        if expect_cbin is False and is_cbin:
            raise ValueError(f"{path}: expected full-resolution bin header")

        loc_bytes = int(preheader["size of location"])
        var_bytes = int(preheader["size of variable"])
        if loc_bytes not in (4, 8) or var_bytes not in (4, 8):
            raise ValueError(f"{path}: unsupported float byte size")
        location_dtype = np.dtype("<f8" if loc_bytes == 8 else "<f4")
        variable_dtype = np.dtype("<f8" if var_bytes == 8 else "<f4")

        full_grid = tuple(int(mesh[f"nx{axis}"]) for axis in (1, 2, 3))
        block_cells = tuple(int(meshblock[f"nx{axis}"]) for axis in (1, 2, 3))
        coarse_grid = tuple(n // factor for n in full_grid)
        coarse_block = tuple(n // factor for n in block_cells)
        nghost = int(mesh["nghost"])
        file_size = Path(path).stat().st_size

        records: list[BinaryRecord] = []
        while stream.tell() < file_size:
            raw_indices = stream.read(24)
            if len(raw_indices) != 24:
                raise ValueError(f"{path}: truncated mesh-block indices")
            indices_array = np.frombuffer(raw_indices, dtype="<i4").astype(np.int64) - nghost
            indices = tuple(int(value) for value in indices_array)
            nx = indices[1] - indices[0] + 1
            ny = indices[3] - indices[2] + 1
            nz = indices[5] - indices[4] + 1
            if min(nx, ny, nz) <= 0:
                raise ValueError(f"{path}: invalid mesh-block output indices {indices}")

            raw_logical = stream.read(16)
            raw_geometry = stream.read(6 * loc_bytes)
            if len(raw_logical) != 16 or len(raw_geometry) != 6 * loc_bytes:
                raise ValueError(f"{path}: truncated mesh-block metadata")
            logical = tuple(int(value) for value in np.frombuffer(raw_logical, dtype="<i4"))
            geometry = tuple(float(value) for value in np.frombuffer(raw_geometry, dtype=location_dtype))
            payload_offset = stream.tell()
            payload_bytes = nvars * nx * ny * nz * variable_dtype.itemsize
            if payload_offset + payload_bytes > file_size:
                raise ValueError(f"{path}: truncated payload")
            stream.seek(payload_bytes, 1)
            records.append(BinaryRecord(indices, logical, geometry, payload_offset, (nz, ny, nx)))

        if stream.tell() != file_size:
            raise ValueError(f"{path}: payload accounting did not reach EOF")
        if not records:
            raise ValueError(f"{path}: shard has no mesh-block records")

    return BinaryShard(
        path=path,
        time=float(preheader["time"]),
        cycle=int(preheader["cycle"]),
        coarsen_factor=factor,
        number_of_moments=moments,
        var_names=var_names,
        variable_dtype=variable_dtype,
        location_dtype=location_dtype,
        full_grid_cells=full_grid,
        mesh_block_cells=block_cells,
        coarse_grid_cells=coarse_grid,
        coarse_block_cells=coarse_block,
        nghost=nghost,
        records=tuple(records),
        file_size=file_size,
    )


def read_record_fields(
    shard: BinaryShard,
    fields: Sequence[str],
    *,
    record_index: int = 0,
    copy: bool = True,
) -> dict[str, np.ndarray]:
    """Read selected payload fields from one parsed shard record."""

    record = shard.records[record_index]
    cube = np.memmap(
        shard.path,
        dtype=shard.variable_dtype,
        mode="r",
        offset=record.payload_offset,
        shape=(len(shard.var_names), *record.shape_kji),
    )
    positions = {name: index for index, name in enumerate(shard.var_names)}
    missing = [field for field in fields if field not in positions]
    if missing:
        raise ValueError(f"{shard.path}: missing requested fields {missing}")
    return {
        field: np.array(cube[positions[field]], copy=copy)
        for field in fields
    }


def file_sha256(path: str | Path) -> str:
    """Hash one artifact without loading it into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_identity(shard: BinaryShard) -> dict[str, object]:
    """Return stable rank-zero snapshot identity fields for provenance tokens."""

    return {
        "basename": Path(shard.path).name,
        "time": shard.time,
        "cycle": shard.cycle,
        "coarsen_factor": shard.coarsen_factor,
        "number_of_moments": shard.number_of_moments,
        "var_names": list(shard.var_names),
        "variable_dtype": shard.variable_dtype.str,
        "location_dtype": shard.location_dtype.str,
        "full_grid_cells": list(shard.full_grid_cells),
        "mesh_block_cells": list(shard.mesh_block_cells),
        "coarse_grid_cells": list(shard.coarse_grid_cells),
        "coarse_block_cells": list(shard.coarse_block_cells),
        "rank0_file_size": shard.file_size,
    }


def discover_snapshot(
    data_root: str | Path,
    product: str,
    scale: int,
    *,
    target_time: float,
) -> BinaryShard:
    """Find a rank-zero cbin snapshot by header time rather than suffix."""

    pattern = str(Path(data_root) / f"cbin_{product}_{scale}" / "rank_00000000" / "*.cbin")
    candidates = [parse_binary_shard(path, expect_cbin=True) for path in sorted(glob.glob(pattern))]
    matches = [item for item in candidates if math.isclose(item.time, target_time, abs_tol=5.0e-6)]
    if len(matches) != 1:
        found = [(Path(item.path).name, item.time, item.cycle) for item in candidates]
        raise ValueError(f"expected one {product}_{scale} snapshot at t={target_time}, found {found}")
    return matches[0]


def shard_path(data_root: str | Path, product: str, scale: int, rank: int, basename: str) -> Path:
    return Path(data_root) / f"cbin_{product}_{scale}" / f"rank_{rank:08d}" / basename


def _update_source_shard_inventory_digest(digest, rank: int, path: Path) -> None:
    """Fingerprint one source path using metadata that changes on content replacement."""

    stat = path.stat()
    digest.update(
        (
            f"{rank:08d}\t{path.name}\t{stat.st_size}\t"
            f"{stat.st_mtime_ns}\t{stat.st_ctime_ns}\n"
        ).encode()
    )


def source_shard_inventory_sha256(
    data_root: str | Path,
    product: str,
    scale: int,
    basename: str,
    *,
    expected_ranks: int = EXPECTED_RANKS,
) -> str:
    """Fingerprint every expected source shard without rereading payload bytes."""

    digest = hashlib.sha256()
    for rank in range(expected_ranks):
        path = shard_path(data_root, product, scale, rank, basename)
        if not path.is_file():
            raise FileNotFoundError(f"missing expected shard {path}")
        _update_source_shard_inventory_digest(digest, rank, path)
    return digest.hexdigest()


def _expected_geometry(logical: Sequence[int]) -> tuple[float, float, float, float, float, float]:
    faces: list[float] = []
    for coordinate, count in zip(logical[:3], RANK_LATTICE):
        faces.extend((-0.5 + coordinate / count, -0.5 + (coordinate + 1) / count))
    return tuple(faces)  # type: ignore[return-value]


def assemble_product(
    data_root: str | Path,
    product: str,
    scale: int,
    basename: str,
    fields: Sequence[str],
    *,
    expected_time: float,
    expected_cycle: int,
    expected_var_names: Sequence[str] | None = None,
    expected_ranks: int = EXPECTED_RANKS,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """Stream all rank shards into global arrays with strict tiling checks."""

    if expected_ranks != EXPECTED_RANKS:
        raise ValueError("partial product assembly is not a catalog census")
    coarse_n = DOMAIN_CELLS // scale
    output = {field: np.empty((coarse_n, coarse_n, coarse_n), dtype=np.float32) for field in fields}
    coverage = np.zeros((coarse_n, coarse_n, coarse_n), dtype=np.uint8)
    rank_map = np.full((RANK_LATTICE[2], RANK_LATTICE[1], RANK_LATTICE[0]), -1, dtype=np.int32)
    total_bytes = 0
    file_sizes: set[int] = set()
    source_inventory_digest = hashlib.sha256()

    for rank in range(expected_ranks):
        path = shard_path(data_root, product, scale, rank, basename)
        if not path.is_file():
            raise FileNotFoundError(f"missing expected shard {path}")
        _update_source_shard_inventory_digest(source_inventory_digest, rank, path)
        shard = parse_binary_shard(path, expect_cbin=True)
        total_bytes += shard.file_size
        file_sizes.add(shard.file_size)
        if not math.isclose(shard.time, expected_time, abs_tol=5.0e-6) or shard.cycle != expected_cycle:
            raise ValueError(f"{path}: time/cycle mismatch")
        if shard.coarsen_factor != scale:
            raise ValueError(f"{path}: coarsening factor mismatch")
        if expected_var_names is not None and tuple(expected_var_names) != shard.var_names:
            raise ValueError(f"{path}: variable schema mismatch")
        if len(shard.records) != 1:
            raise ValueError(f"{path}: expected one mesh-block record")

        record = shard.records[0]
        rx, ry, rz, level = record.logical
        if level != 0:
            raise ValueError(f"{path}: unexpected refinement level {level}")
        if not (0 <= rx < RANK_LATTICE[0] and 0 <= ry < RANK_LATTICE[1] and 0 <= rz < RANK_LATTICE[2]):
            raise ValueError(f"{path}: logical location out of bounds {record.logical}")
        if rank_map[rz, ry, rx] != -1:
            raise ValueError(f"{path}: duplicate logical owner {record.logical}")
        rank_map[rz, ry, rx] = rank
        expected_geometry = _expected_geometry(record.logical)
        if not np.allclose(record.geometry, expected_geometry, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{path}: geometry mismatch {record.geometry} != {expected_geometry}")

        nz, ny, nx = record.shape_kji
        if (nx, ny, nz) != shard.coarse_block_cells:
            raise ValueError(f"{path}: local cbin shape mismatch")
        i0, j0, k0 = rx * nx, ry * ny, rz * nz
        target = np.s_[k0:k0 + nz, j0:j0 + ny, i0:i0 + nx]
        coverage[target] += 1
        selected = read_record_fields(shard, fields)
        for field, values in selected.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: non-finite values in field {field}")
            output[field][target] = values

    holes = int(np.count_nonzero(coverage == 0))
    overlaps = int(np.count_nonzero(coverage > 1))
    missing_rank_locations = int(np.count_nonzero(rank_map < 0))
    if holes or overlaps or missing_rank_locations:
        raise ValueError(
            f"{product}_{scale}: invalid tiling holes={holes}, overlaps={overlaps}, "
            f"missing_rank_locations={missing_rank_locations}"
        )
    manifest = {
        "product": product,
        "scale": scale,
        "basename": basename,
        "time": expected_time,
        "cycle": expected_cycle,
        "expected_ranks": expected_ranks,
        "global_shape_kji": list(coverage.shape),
        "fields": list(fields),
        "total_bytes": total_bytes,
        "file_sizes": sorted(file_sizes),
        "source_shard_inventory_sha256": source_inventory_digest.hexdigest(),
        "holes": holes,
        "overlaps": overlaps,
        "missing_rank_locations": missing_rank_locations,
    }
    return output, rank_map, manifest


def save_array_cache(path: str | Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    """Write a compressed NPZ cache with an embedded JSON metadata record."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def load_array_cache(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path) as cache:
        metadata = json.loads(str(cache["__metadata_json__"]))
        arrays = {key: cache[key] for key in cache.files if key != "__metadata_json__"}
    return arrays, metadata


def aggregate_blocks(values: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping factor-cubed blocks in [k,j,i] order."""

    if factor == 1:
        return values.astype(np.float64, copy=False)
    nz, ny, nx = values.shape
    if nz % factor or ny % factor or nx % factor:
        raise ValueError(f"shape {values.shape} is not divisible by factor={factor}")
    return values.reshape(
        nz // factor, factor, ny // factor, factor, nx // factor, factor
    ).mean(axis=(1, 3, 5), dtype=np.float64)


def _rounding_gamma(operations: int) -> float:
    product = operations * FLOAT64_EPS
    if product >= 1.0:
        raise ValueError(f"too many operations for rounding bound: {operations}")
    return product / (1.0 - product)


def float32_quantization_bound(values: np.ndarray) -> np.ndarray:
    """Return a conservative half-ULP serialization bound for float32 values."""

    stored = np.asarray(values, dtype=np.float32)
    lower = np.nextafter(stored, np.float32(-np.inf), dtype=np.float32)
    upper = np.nextafter(stored, np.float32(np.inf), dtype=np.float32)
    return 0.5 * np.maximum(
        np.abs(stored.astype(np.float64) - lower.astype(np.float64)),
        np.abs(upper.astype(np.float64) - stored.astype(np.float64)),
    )


def source_raw_moment_error_bounds(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    *,
    source_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bound serialized source-voxel raw moments, including writer summation."""

    one, two, three, four = (
        np.asarray(values, dtype=np.float64)
        for values in (first, second, third, fourth)
    )
    q1, q2, q3, q4 = (
        float32_quantization_bound(values)
        for values in (first, second, third, fourth)
    )
    gamma1 = _rounding_gamma(source_cells + 1)
    gamma2 = _rounding_gamma(source_cells + 2)
    gamma3 = _rounding_gamma(source_cells + 3)
    gamma4 = _rounding_gamma(source_cells + 4)
    second_upper = (np.abs(two) + q2) / (1.0 - gamma2)
    fourth_upper = (np.abs(four) + q4) / (1.0 - gamma4)
    return (
        q1 + gamma1 * np.sqrt(second_upper),
        q2 + gamma2 * second_upper,
        q3 + gamma3 * np.sqrt(second_upper * fourth_upper),
        q4 + gamma4 * fourth_upper,
    )


def aggregate_raw_moment_error_bounds(
    raw_moments: Sequence[np.ndarray],
    source_bounds: Sequence[np.ndarray],
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average source bounds and add float64 reduction error for a target cube."""

    if len(raw_moments) != 4 or len(source_bounds) != 4:
        raise ValueError("four raw moments and four error bounds are required")
    reduction_bound = _rounding_gamma(max(1, factor**3))
    return tuple(
        aggregate_blocks(bound, factor) + reduction_bound * aggregate_blocks(np.abs(moment), factor)
        for moment, bound in zip(raw_moments, source_bounds)
    )  # type: ignore[return-value]


def raw_moment_statistics(
    m1: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    m4: np.ndarray,
    *,
    abs_error_bounds: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Convert raw scalar moments into robust central and standardized moments."""

    one = np.asarray(m1, dtype=np.float64)
    two = np.asarray(m2, dtype=np.float64)
    three = np.asarray(m3, dtype=np.float64)
    four = np.asarray(m4, dtype=np.float64)
    if abs_error_bounds is None:
        d1, d2, d3, d4 = (np.zeros_like(one),) * 4
    else:
        if len(abs_error_bounds) != 4:
            raise ValueError("four raw-moment absolute-error bounds are required")
        d1, d2, d3, d4 = (
            np.asarray(bound, dtype=np.float64)
            for bound in abs_error_bounds
        )
        if any(bound.shape != one.shape for bound in (d1, d2, d3, d4)):
            raise ValueError("raw-moment absolute-error bounds must match moment shapes")
    variance_raw = two - one * one
    evaluation2 = _rounding_gamma(32) * (np.abs(two) + np.abs(one * one))
    variance_error_bound = d2 + 2.0 * np.abs(one) * d1 + d1 * d1 + evaluation2
    flags = np.zeros(one.shape, dtype=np.uint8)
    nonfinite = ~np.isfinite(one + two + three + four + d1 + d2 + d3 + d4)
    flags[nonfinite] |= 128
    small_negative = (variance_raw < 0.0) & (variance_raw >= -variance_error_bound)
    invalid_variance = variance_raw < -variance_error_bound
    variance = variance_raw.copy()
    variance[small_negative] = 0.0
    variance[invalid_variance] = np.nan
    flags[small_negative] |= 1
    flags[invalid_variance] |= 2
    unresolved_variance = np.isfinite(variance) & (variance <= variance_error_bound)
    flags[unresolved_variance] |= 4

    central3 = three - 3.0 * one * two + 2.0 * one**3
    central4 = four - 4.0 * one * three + 6.0 * one * one * two - 3.0 * one**4
    evaluation3 = _rounding_gamma(32) * (
        np.abs(three) + np.abs(3.0 * one * two) + np.abs(2.0 * one**3)
    )
    central3_error_bound = (
        d3
        + 3.0 * (np.abs(one) * d2 + np.abs(two) * d1 + d1 * d2)
        + 2.0 * ((np.abs(one) + d1) ** 3 - np.abs(one) ** 3)
        + evaluation3
    )
    evaluation4 = _rounding_gamma(32) * (
        np.abs(four) + np.abs(4.0 * one * three) + np.abs(6.0 * one * one * two)
        + np.abs(3.0 * one**4)
    )
    central4_error_bound = (
        d4
        + 4.0 * (np.abs(one) * d3 + np.abs(three) * d1 + d1 * d3)
        + 6.0 * ((np.abs(one) + d1) ** 2 * (np.abs(two) + d2) - np.abs(one) ** 2 * np.abs(two))
        + 3.0 * ((np.abs(one) + d1) ** 4 - np.abs(one) ** 4)
        + evaluation4
    )
    invalid_central4 = central4 < -central4_error_bound
    flags[invalid_central4] |= 8
    small_negative_central4 = (central4 < 0.0) & ~invalid_central4
    flags[small_negative_central4] |= 32
    central4[small_negative_central4] = 0.0
    central4[invalid_central4] = np.nan

    sigma = np.sqrt(variance)
    skewness_error_bound = _normalized_moment_error_bound(
        central3, central3_error_bound, variance, variance_error_bound, denominator_power=1.5
    )
    kurtosis_error_bound = _normalized_moment_error_bound(
        central4, central4_error_bound, variance, variance_error_bound, denominator_power=2.0
    )
    unresolved_central3 = np.abs(central3) <= central3_error_bound
    unresolved_central4 = np.isfinite(central4) & (central4 <= central4_error_bound)
    flags[unresolved_central3] |= 16
    flags[unresolved_central4] |= 64
    valid_common = np.isfinite(sigma) & ~unresolved_variance & ~nonfinite
    valid_skewness = valid_common & ~unresolved_central3
    valid_kurtosis = valid_common & np.isfinite(central4) & ~unresolved_central4
    skewness = np.full(one.shape, np.nan, dtype=np.float64)
    kurtosis = np.full(one.shape, np.nan, dtype=np.float64)
    skewness[valid_skewness] = central3[valid_skewness] / sigma[valid_skewness] ** 3
    kurtosis[valid_kurtosis] = central4[valid_kurtosis] / variance[valid_kurtosis] ** 2
    return {
        "mean": one,
        "variance": variance,
        "variance_error_bound": variance_error_bound,
        "sigma": sigma,
        "central_moment_3": central3,
        "central_moment_4": central4,
        "central_moment_3_error_bound": central3_error_bound,
        "central_moment_4_error_bound": central4_error_bound,
        "skewness": skewness,
        "skewness_error_bound": skewness_error_bound,
        "kurtosis": kurtosis,
        "kurtosis_error_bound": kurtosis_error_bound,
        "excess_kurtosis": kurtosis - 3.0,
        "moment_flags": flags,
    }


def _normalized_moment_error_bound(
    numerator: np.ndarray,
    numerator_error_bound: np.ndarray,
    variance: np.ndarray,
    variance_error_bound: np.ndarray,
    *,
    denominator_power: float,
) -> np.ndarray:
    """Bound a normalized central moment over numerator and variance intervals."""

    numerator = np.asarray(numerator, dtype=np.float64)
    numerator_error_bound = np.asarray(numerator_error_bound, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    variance_error_bound = np.asarray(variance_error_bound, dtype=np.float64)
    lower_variance = variance - variance_error_bound
    upper_variance = variance + variance_error_bound
    valid = (
        np.isfinite(numerator)
        & np.isfinite(numerator_error_bound)
        & np.isfinite(variance)
        & np.isfinite(variance_error_bound)
        & (numerator_error_bound >= 0.0)
        & (variance_error_bound >= 0.0)
        & (lower_variance > 0.0)
    )
    result = np.full(numerator.shape, np.inf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nominal = numerator / variance**denominator_power
        candidates = (
            (numerator - numerator_error_bound) / lower_variance**denominator_power,
            (numerator - numerator_error_bound) / upper_variance**denominator_power,
            (numerator + numerator_error_bound) / lower_variance**denominator_power,
            (numerator + numerator_error_bound) / upper_variance**denominator_power,
        )
        result[valid] = np.maximum.reduce([
            np.abs(candidate[valid] - nominal[valid])
            for candidate in candidates
        ])
    return result


def robust_nonnegative(
    values: np.ndarray,
    scale: np.ndarray,
    *,
    abs_error_bound: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppress unresolved cancellation while preserving material-invalid flags."""

    raw = np.asarray(values, dtype=np.float64)
    if abs_error_bound is None:
        tolerance = 64.0 * FLOAT32_EPS * np.maximum(np.abs(scale), 1.0)
    else:
        tolerance = np.asarray(abs_error_bound, dtype=np.float64)
        if tolerance.shape != raw.shape:
            raise ValueError("absolute-error bound must match values")
    flags = np.zeros(raw.shape, dtype=np.uint8)
    small_negative = (raw < 0.0) & (raw >= -tolerance)
    invalid = raw < -tolerance
    result = raw.copy()
    result[small_negative] = 0.0
    result[invalid] = np.nan
    flags[small_negative] |= 1
    flags[invalid] |= 2
    unresolved = np.isfinite(result) & (result <= tolerance)
    result[unresolved] = np.nan
    flags[unresolved] |= 4
    return result, flags


def morton_rank(rx: int, ry: int, rz: int) -> int:
    """Encode the observed AthenaK rank ordering for validation probes."""

    rank = 0
    for bit in range(6):
        rank |= ((rx >> bit) & 1) << (3 * bit)
        rank |= ((ry >> bit) & 1) << (3 * bit + 1)
        rank |= ((rz >> bit) & 1) << (3 * bit + 2)
    return rank


def required_rank_ids(
    cell_bounds: Sequence[int],
    rank_map: np.ndarray | None = None,
) -> list[int]:
    """Return full-resolution rank IDs touched by half-open fine-cell bounds."""

    i0, i1, j0, j1, k0, k1 = (int(value) for value in cell_bounds)
    ranges = (
        range(i0 // MESH_BLOCK_CELLS[0], math.ceil(i1 / MESH_BLOCK_CELLS[0])),
        range(j0 // MESH_BLOCK_CELLS[1], math.ceil(j1 / MESH_BLOCK_CELLS[1])),
        range(k0 // MESH_BLOCK_CELLS[2], math.ceil(k1 / MESH_BLOCK_CELLS[2])),
    )
    ranks: list[int] = []
    for rz in ranges[2]:
        for ry in ranges[1]:
            for rx in ranges[0]:
                if rank_map is None:
                    ranks.append(morton_rank(rx, ry, rz))
                else:
                    rank = int(rank_map[rz, ry, rx])
                    if rank < 0:
                        raise ValueError(f"missing rank owner for logical coordinate {(rx, ry, rz)}")
                    ranks.append(rank)
    return sorted(ranks)


def flatten_columns(columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).reshape(-1) for key, value in columns.items()}


def finite_quantiles(values: np.ndarray, quantiles: Iterable[float]) -> list[float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return [float("nan") for _ in quantiles]
    return [float(value) for value in np.quantile(finite, tuple(quantiles))]
