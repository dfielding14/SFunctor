"""Validated rank-local Athena block manifests and grouped block reads.

The manifest is intentionally storage-facing.  Scientific kernels should
consume slices or chunk iterators built from these records rather than infer
rank ownership from a static decomposition formula.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from . import bin_convert_new as bc

__all__ = [
    "RankBlock",
    "RankManifest",
    "build_rank_manifest",
    "read_blocks_grouped",
]


@dataclass(frozen=True)
class RankBlock:
    """One mesh block and its actual rank-local ownership metadata."""

    rank: int
    filename: str
    local_index: int
    logical_location: tuple[int, int, int]
    level: int
    geometry: tuple[float, float, float, float, float, float]
    index_bounds: tuple[int, int, int, int, int, int]

    def contains(self, axis: int, coordinate: float) -> bool:
        """Return whether the half-open block interval contains *coordinate*."""

        if axis not in (1, 2, 3):
            raise ValueError("axis must be 1, 2, or 3")
        lower, upper = self.geometry[2 * (axis - 1):2 * axis]
        return lower <= coordinate < upper


@dataclass(frozen=True)
class RankManifest:
    """Validated metadata for one snapshot distributed across rank files."""

    snapshot_basename: str
    rank_files: tuple[str, ...]
    blocks: tuple[RankBlock, ...]
    time: float
    cycle: int
    var_names: tuple[str, ...]
    global_shape: tuple[int, int, int]
    meshblock_shape: tuple[int, int, int]
    domain_bounds: tuple[float, float, float, float, float, float]

    def blocks_intersecting(self, axis: int, coordinate: float) -> tuple[RankBlock, ...]:
        """Return blocks intersecting one half-open slice coordinate."""

        return tuple(block for block in self.blocks if block.contains(axis, coordinate))

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(sorted({block.level for block in self.blocks}))


def _rank_from_path(path: Path) -> int:
    name = path.parent.name
    if not name.startswith("rank_") or not name[5:].isdigit():
        raise ValueError(f"Cannot parse rank number from {path}")
    return int(name[5:])


def _validate_geometry(path: Path, local_index: int, geometry: np.ndarray) -> tuple[float, ...]:
    geometry = tuple(float(value) for value in geometry)
    if len(geometry) != 6 or not np.all(np.isfinite(geometry)):
        raise ValueError(f"{path}: block {local_index} has invalid geometry")
    if any(geometry[index] >= geometry[index + 1] for index in (0, 2, 4)):
        raise ValueError(f"{path}: block {local_index} has non-positive physical extent")
    return geometry


def build_rank_manifest(
    rank0_filename: str | Path,
    *,
    metadata_reader: Callable[[str], Mapping[str, object]] | None = None,
) -> RankManifest:
    """Scan and validate every rank file belonging to one snapshot."""

    metadata_reader = bc.read_binary_metadata if metadata_reader is None else metadata_reader
    rank0_filename = Path(rank0_filename)
    rank_files = sorted(rank0_filename.parent.parent.glob(f"rank_*/{rank0_filename.name}"))
    if not rank_files:
        raise FileNotFoundError(f"No rank files found for snapshot {rank0_filename.name}")
    rank_ids = sorted(_rank_from_path(path) for path in rank_files)
    if rank_ids != list(range(rank_ids[-1] + 1)):
        raise ValueError(f"Snapshot rank-file coverage is incomplete: found ranks {rank_ids}")

    reference = None
    blocks = []
    seen_owners: set[tuple[int, int, int, int]] = set()
    expected_keys = (
        "time", "cycle", "var_names", "Nx1", "Nx2", "Nx3",
        "nx1_mb", "nx2_mb", "nx3_mb",
        "x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
    )
    for path in rank_files:
        metadata = metadata_reader(str(path))
        if reference is None:
            reference = metadata
        else:
            for key in expected_keys:
                if metadata[key] != reference[key]:
                    raise ValueError(f"{path}: inconsistent snapshot metadata[{key!r}]")
        rank = _rank_from_path(path)
        logical = np.asarray(metadata["mb_logical"])
        geometry = np.asarray(metadata["mb_geometry"])
        indices = np.asarray(metadata["mb_index"])
        n_mbs = int(metadata["n_mbs"])
        if logical.shape != (n_mbs, 4) or geometry.shape != (n_mbs, 6) or indices.shape != (n_mbs, 6):
            raise ValueError(f"{path}: inconsistent mesh-block metadata shapes")
        for local_index in range(n_mbs):
            location = tuple(int(value) for value in logical[local_index, :3])
            level = int(logical[local_index, 3])
            if level < 0 or any(value < 0 for value in location):
                raise ValueError(f"{path}: block {local_index} has invalid logical location")
            owner = (level, *location)
            if owner in seen_owners:
                raise ValueError(f"{path}: duplicated logical mesh-block owner {owner}")
            seen_owners.add(owner)
            blocks.append(
                RankBlock(
                    rank=rank,
                    filename=str(path),
                    local_index=local_index,
                    logical_location=location,
                    level=level,
                    geometry=_validate_geometry(path, local_index, geometry[local_index]),
                    index_bounds=tuple(int(value) for value in indices[local_index]),
                )
            )
    assert reference is not None
    return RankManifest(
        snapshot_basename=rank0_filename.name,
        rank_files=tuple(str(path) for path in rank_files),
        blocks=tuple(blocks),
        time=float(reference["time"]),
        cycle=int(reference["cycle"]),
        var_names=tuple(str(name) for name in reference["var_names"]),
        global_shape=tuple(int(reference[name]) for name in ("Nx1", "Nx2", "Nx3")),
        meshblock_shape=tuple(int(reference[name]) for name in ("nx1_mb", "nx2_mb", "nx3_mb")),
        domain_bounds=tuple(float(reference[name]) for name in ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max")),
    )


def read_blocks_grouped(
    manifest: RankManifest,
    blocks: Iterable[RankBlock],
    *,
    quantities: Sequence[str] | None = None,
    reader: Callable[[str], Mapping[str, object]] | None = None,
) -> dict[RankBlock, dict[str, np.ndarray]]:
    """Load selected blocks while parsing each required rank file exactly once."""

    reader = bc.read_binary if reader is None else reader
    quantities = tuple(manifest.var_names if quantities is None else quantities)
    unknown = set(quantities) - set(manifest.var_names)
    if unknown:
        raise ValueError(f"Unknown quantities requested from rank files: {sorted(unknown)}")
    grouped: dict[str, list[RankBlock]] = defaultdict(list)
    for block in blocks:
        grouped[block.filename].append(block)

    output = {}
    for filename, selected in grouped.items():
        raw = reader(filename)
        if int(raw["cycle"]) != manifest.cycle or float(raw["time"]) != manifest.time:
            raise ValueError(f"{filename}: rank data do not match manifest snapshot")
        if tuple(str(name) for name in raw["var_names"]) != manifest.var_names:
            raise ValueError(f"{filename}: rank data variable schema does not match manifest")
        n_mbs = int(raw["n_mbs"])
        for block in selected:
            if not 0 <= block.local_index < n_mbs:
                raise ValueError(f"{filename}: missing local block index {block.local_index}")
            output[block] = {
                name: np.asarray(raw["mb_data"][name][block.local_index])
                for name in quantities
            }
    return output
