"""Tests for file-derived rank ownership and grouped Athena block reads."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sfunctor.io import bin_convert_new as bc
from sfunctor.io.rank_manifest import build_rank_manifest, read_blocks_grouped


def _metadata(logical_locations):
    logical_locations = np.asarray(logical_locations, dtype=np.int32)
    n_mbs = len(logical_locations)
    geometry = np.array(
        [
            [location[0], location[0] + 1, location[1], location[1] + 1, location[2], location[2] + 1]
            for location in logical_locations
        ],
        dtype=float,
    )
    return {
        "time": 2.0,
        "cycle": 7,
        "var_names": ["dens", "velx"],
        "Nx1": 4,
        "Nx2": 2,
        "Nx3": 2,
        "nx1_mb": 2,
        "nx2_mb": 2,
        "nx3_mb": 2,
        "x1min": 0.0,
        "x1max": 4.0,
        "x2min": 0.0,
        "x2max": 2.0,
        "x3min": 0.0,
        "x3max": 2.0,
        "n_mbs": n_mbs,
        "mb_index": np.tile(np.array([0, 1, 0, 1, 0, 1]), (n_mbs, 1)),
        "mb_logical": logical_locations,
        "mb_geometry": geometry,
    }


def _touch_rank_files(tmp_path):
    paths = []
    for rank in range(2):
        path = tmp_path / f"rank_{rank:08d}" / "Turb.full_mhd_w_bcc.00000.bin"
        path.parent.mkdir()
        path.touch()
        paths.append(path)
    return paths


def _write_binary(path: Path):
    header = "\n".join(
        [
            "<mesh>",
            "nx1 = 2",
            "nx2 = 2",
            "nx3 = 2",
            "nghost = 0",
            "x1min = 0.0",
            "x1max = 1.0",
            "x2min = 0.0",
            "x2max = 1.0",
            "x3min = 0.0",
            "x3max = 1.0",
            "<meshblock>",
            "nx1 = 2",
            "nx2 = 2",
            "nx3 = 2",
            "",
        ]
    ).encode()
    block = np.arange(16, dtype=np.float64).reshape(2, 2, 2, 2)
    with path.open("wb") as fp:
        fp.write(b"Athena binary output version=1.1\n")
        fp.write(b"pheader_count=5\n")
        fp.write(b"time=2.0\n")
        fp.write(b"cycle=7\n")
        fp.write(b"size of location=8\n")
        fp.write(b"size of variable=8\n")
        fp.write(b"nvars=2\n")
        fp.write(b"variables= dens velx\n")
        fp.write(f"header size={len(header)}\n".encode())
        fp.write(header)
        fp.write(np.array([0, 1, 0, 1, 0, 1], dtype=np.int32).tobytes())
        fp.write(np.array([0, 0, 0, 0], dtype=np.int32).tobytes())
        fp.write(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64).tobytes())
        fp.write(block.tobytes())


def test_binary_metadata_scanner_skips_payload_but_matches_full_reader(tmp_path):
    path = tmp_path / "rank.bin"
    _write_binary(path)
    metadata = bc.read_binary_metadata(path)
    full = bc.read_binary(path)
    assert "mb_data" not in metadata
    for key in ("time", "cycle", "var_names", "Nx1", "Nx2", "Nx3", "n_mbs"):
        assert metadata[key] == full[key]
    assert np.array_equal(metadata["mb_index"], full["mb_index"])
    assert np.array_equal(metadata["mb_logical"], full["mb_logical"])
    assert np.array_equal(metadata["mb_geometry"], full["mb_geometry"])


def test_manifest_uses_file_ownership_and_grouped_reader_opens_each_rank_once(tmp_path):
    paths = _touch_rank_files(tmp_path)
    metadata = {
        str(paths[0]): _metadata([(0, 0, 0, 0), (1, 0, 0, 0)]),
        str(paths[1]): _metadata([(0, 1, 0, 0)]),
    }
    manifest = build_rank_manifest(paths[0], metadata_reader=lambda path: metadata[path])
    assert len(manifest.blocks) == 3
    assert {block.rank for block in manifest.blocks} == {0, 1}

    reads = Counter()

    def reader(path):
        reads[path] += 1
        rank_metadata = dict(metadata[path])
        rank_metadata["mb_data"] = {
            name: [np.full((2, 2, 2), block + offset) for block in range(rank_metadata["n_mbs"])]
            for name, offset in (("dens", 10), ("velx", 20))
        }
        return rank_metadata

    loaded = read_blocks_grouped(manifest, manifest.blocks, quantities=("dens",), reader=reader)
    assert reads == Counter({str(paths[0]): 1, str(paths[1]): 1})
    assert len(loaded) == 3
    assert all(tuple(values) == ("dens",) for values in loaded.values())


def test_manifest_rejects_duplicate_logical_owners(tmp_path):
    paths = _touch_rank_files(tmp_path)
    metadata = {
        str(paths[0]): _metadata([(0, 0, 0, 0)]),
        str(paths[1]): _metadata([(0, 0, 0, 0)]),
    }
    with pytest.raises(ValueError, match="duplicated logical"):
        build_rank_manifest(paths[0], metadata_reader=lambda path: metadata[path])


def test_manifest_rejects_noncontiguous_rank_files(tmp_path):
    rank0 = tmp_path / "rank_00000000" / "Turb.full_mhd_w_bcc.00000.bin"
    rank2 = tmp_path / "rank_00000002" / rank0.name
    rank0.parent.mkdir()
    rank2.parent.mkdir()
    rank0.touch()
    rank2.touch()
    with pytest.raises(ValueError, match="coverage is incomplete"):
        build_rank_manifest(rank0, metadata_reader=lambda _: _metadata([(0, 0, 0, 0)]))
