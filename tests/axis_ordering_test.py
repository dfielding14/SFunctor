"""Synthetic regression tests for AthenaK KJI slice extraction."""

from __future__ import annotations

from pathlib import Path
from inspect import Parameter, signature

import numpy as np
import pytest

from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.io import extract


PRIMITIVES = ("dens", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3")
DERIVED = (
    "vortx", "vorty", "vortz",
    "currx", "curry", "currz",
    "curvx", "curvy", "curvz",
    "grad_rho_x", "grad_rho_y", "grad_rho_z",
)


def _central_difference(arr: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * spacing)


def _make_volume(mesh: dict[str, float | int]) -> dict[str, np.ndarray]:
    x1f = np.linspace(mesh["x1min"], mesh["x1max"], mesh["nx1"] + 1)
    x2f = np.linspace(mesh["x2min"], mesh["x2max"], mesh["nx2"] + 1)
    x3f = np.linspace(mesh["x3min"], mesh["x3max"], mesh["nx3"] + 1)
    x = 0.5 * (x1f[:-1] + x1f[1:])
    y = 0.5 * (x2f[:-1] + x2f[1:])
    z = 0.5 * (x3f[:-1] + x3f[1:])
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    lx = mesh["x1max"] - mesh["x1min"]
    ly = mesh["x2max"] - mesh["x2min"]
    lz = mesh["x3max"] - mesh["x3min"]
    sx = np.sin(2.0 * np.pi * (xx - mesh["x1min"]) / lx)
    sy = np.sin(2.0 * np.pi * (yy - mesh["x2min"]) / ly)
    sz = np.sin(2.0 * np.pi * (zz - mesh["x3min"]) / lz)
    cx = np.cos(2.0 * np.pi * (xx - mesh["x1min"]) / lx)
    cy = np.cos(2.0 * np.pi * (yy - mesh["x2min"]) / ly)
    cz = np.cos(2.0 * np.pi * (zz - mesh["x3min"]) / lz)

    return {
        "dens": 2.0 + 0.10 * sx + 0.20 * cy + 0.30 * sz,
        "velx": 0.40 * sy + 0.30 * cz,
        "vely": 0.50 * sz + 0.20 * cx,
        "velz": 0.60 * sx + 0.10 * cy,
        "bcc1": 1.20 + 0.10 * sy + 0.05 * cz,
        "bcc2": 0.80 + 0.07 * sz + 0.04 * cx,
        "bcc3": 0.60 + 0.06 * sx + 0.03 * cy,
    }


def _derived_from_volume(volume: dict[str, np.ndarray], mesh: dict[str, float | int]) -> dict[str, np.ndarray]:
    dx = (mesh["x1max"] - mesh["x1min"]) / mesh["nx1"]
    dy = (mesh["x2max"] - mesh["x2min"]) / mesh["nx2"]
    dz = (mesh["x3max"] - mesh["x3min"]) / mesh["nx3"]

    def d_dx(arr):
        return _central_difference(arr, 2, dx)

    def d_dy(arr):
        return _central_difference(arr, 1, dy)

    def d_dz(arr):
        return _central_difference(arr, 0, dz)

    vx, vy, vz = volume["velx"], volume["vely"], volume["velz"]
    bx, by, bz = volume["bcc1"], volume["bcc2"], volume["bcc3"]
    rho = volume["dens"]

    bmag = np.sqrt(bx * bx + by * by + bz * bz + 1e-10)
    bx_unit, by_unit, bz_unit = bx / bmag, by / bmag, bz / bmag

    return {
        "vortx": d_dy(vz) - d_dz(vy),
        "vorty": d_dz(vx) - d_dx(vz),
        "vortz": d_dx(vy) - d_dy(vx),
        "currx": d_dy(bz) - d_dz(by),
        "curry": d_dz(bx) - d_dx(bz),
        "currz": d_dx(by) - d_dy(bx),
        "curvx": bx_unit * d_dx(bx_unit) + by_unit * d_dy(bx_unit) + bz_unit * d_dz(bx_unit),
        "curvy": bx_unit * d_dx(by_unit) + by_unit * d_dy(by_unit) + bz_unit * d_dz(by_unit),
        "curvz": bx_unit * d_dx(bz_unit) + by_unit * d_dy(bz_unit) + bz_unit * d_dz(bz_unit),
        "grad_rho_x": d_dx(rho),
        "grad_rho_y": d_dy(rho),
        "grad_rho_z": d_dz(rho),
    }


def _plane(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 1:
        return arr[:, :, index]
    if axis == 2:
        return arr[:, index, :]
    return arr[index, :, :]


def test_extract_2d_slice_respects_kji_order_on_elongated_domain(tmp_path, monkeypatch):
    mesh = {
        "nx1": 4, "nx2": 6, "nx3": 8,
        "x1min": -0.5, "x1max": 0.5,
        "x2min": -1.0, "x2max": 1.0,
        "x3min": -1.5, "x3max": 1.5,
    }
    meshblock = {"nx1": 2, "nx2": 2, "nx3": 2}
    volume = _make_volume(mesh)
    expected = {**volume, **_derived_from_volume(volume, mesh)}

    locations = extract._build_meshblock_locations(2, 3, 4)
    n_meshblocks = len(locations)
    n_ranks = 2
    blocks_per_rank = n_meshblocks // n_ranks
    x1f = np.linspace(mesh["x1min"], mesh["x1max"], mesh["nx1"] + 1)
    x2f = np.linspace(mesh["x2min"], mesh["x2max"], mesh["nx2"] + 1)
    x3f = np.linspace(mesh["x3min"], mesh["x3max"], mesh["nx3"] + 1)

    for rank in range(n_ranks):
        rank_dir = tmp_path / "data" / "data_synthetic" / "bin" / f"rank_{rank:08d}"
        rank_dir.mkdir(parents=True)
        (rank_dir / "Turb.full_mhd_w_bcc.00000.bin").touch()

    def fake_athinput(_):
        return {"mesh": mesh, "meshblock": meshblock}

    def fake_read(filename, meshblock_index_in_file):
        rank = int(Path(filename).parent.name.split("_")[-1])
        gid = rank * blocks_per_rank + meshblock_index_in_file
        i3, i2, i1 = locations[gid]
        i = slice(i1 * 2, (i1 + 1) * 2)
        j = slice(i2 * 2, (i2 + 1) * 2)
        k = slice(i3 * 2, (i3 + 1) * 2)
        data = {name: arr[k, j, i] for name, arr in volume.items()}
        data["x1f"] = x1f[i1 * 2:(i1 + 1) * 2 + 1]
        data["x2f"] = x2f[i2 * 2:(i2 + 1) * 2 + 1]
        data["x3f"] = x3f[i3 * 2:(i3 + 1) * 2 + 1]
        return data

    def fake_read_binary(filename):
        rank = int(Path(filename).parent.name.split("_")[-1])
        rank_locations = locations[rank * blocks_per_rank:(rank + 1) * blocks_per_rank]
        data = {
            "time": 0.0,
            "cycle": 0,
            "var_names": list(volume),
            "Nx1": mesh["nx1"], "Nx2": mesh["nx2"], "Nx3": mesh["nx3"],
            "nx1_mb": 2, "nx2_mb": 2, "nx3_mb": 2,
            "x1min": mesh["x1min"], "x1max": mesh["x1max"],
            "x2min": mesh["x2min"], "x2max": mesh["x2max"],
            "x3min": mesh["x3min"], "x3max": mesh["x3max"],
            "n_mbs": len(rank_locations),
            "mb_index": np.tile(np.array([0, 1, 0, 1, 0, 1]), (len(rank_locations), 1)),
            "mb_logical": np.array([(i1, i2, i3, 0) for i3, i2, i1 in rank_locations]),
            "mb_geometry": np.array(
                [
                    (
                        x1f[i1 * 2], x1f[(i1 + 1) * 2],
                        x2f[i2 * 2], x2f[(i2 + 1) * 2],
                        x3f[i3 * 2], x3f[(i3 + 1) * 2],
                    )
                    for i3, i2, i1 in rank_locations
                ]
            ),
            "mb_data": {name: [] for name in volume},
        }
        for i3, i2, i1 in rank_locations:
            i = slice(i1 * 2, (i1 + 1) * 2)
            j = slice(i2 * 2, (i2 + 1) * 2)
            k = slice(i3 * 2, (i3 + 1) * 2)
            for name, arr in volume.items():
                data["mb_data"][name].append(arr[k, j, i])
        return data

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(extract.bc, "athinput", fake_athinput)
    monkeypatch.setattr(extract.bc, "read_single_rank_binary_as_athdf", fake_read)
    monkeypatch.setattr(extract.bc, "read_binary_metadata", fake_read_binary)
    monkeypatch.setattr(extract.bc, "read_binary", fake_read_binary)

    requests = {1: 0.2, 2: 0.3, 3: 0.75}
    edges = {
        1: np.linspace(mesh["x1min"], mesh["x1max"], mesh["nx1"] + 1),
        2: np.linspace(mesh["x2min"], mesh["x2max"], mesh["nx2"] + 1),
        3: np.linspace(mesh["x3min"], mesh["x3max"], mesh["nx3"] + 1),
    }

    for axis, value in requests.items():
        result = extract.extract_2d_slice("synthetic", axis, value, file_number=0, save=False)
        index = extract._cell_index(edges[axis], value)
        for field in PRIMITIVES + DERIVED:
            np.testing.assert_allclose(result[field], _plane(expected[field], axis, index), atol=1e-12)


@pytest.mark.parametrize("axis", [0, 4])
def test_extract_2d_slice_rejects_invalid_axis(axis):
    with pytest.raises(ValueError, match="axis must be 1, 2, or 3"):
        extract.extract_2d_slice("synthetic", axis, 0.0, save=False)


def test_analyze_slice_requires_explicit_axis():
    assert signature(analyze_slice).parameters["axis"].default is Parameter.empty
    with pytest.raises(ValueError, match="axis must be 1, 2, or 3"):
        analyze_slice({}, axis=0)


def test_invalid_all_nan_cache_is_rejected(tmp_path):
    path = tmp_path / "invalid_slice.npz"
    np.savez(path, **{name: np.full((2, 3), np.nan) for name in PRIMITIVES})
    assert extract._load_valid_cache(path, (2, 3), PRIMITIVES) is None


def test_partially_filled_cache_is_rejected(tmp_path):
    path = tmp_path / "partial_slice.npz"
    data = {name: np.ones((2, 3)) for name in PRIMITIVES}
    data["dens"][0, 0] = np.nan
    np.savez(path, **data)
    assert extract._load_valid_cache(path, (2, 3), PRIMITIVES) is None


def test_invalid_cache_is_quarantined_before_recomputation(tmp_path, monkeypatch):
    mesh = {
        "nx1": 2, "nx2": 2, "nx3": 2,
        "x1min": -0.5, "x1max": 0.5,
        "x2min": -0.5, "x2max": 0.5,
        "x3min": -0.5, "x3max": 0.5,
    }
    meshblock = {"nx1": 2, "nx2": 2, "nx3": 2}
    rank_dir = tmp_path / "data" / "data_synthetic" / "bin" / "rank_00000000"
    rank_dir.mkdir(parents=True)
    (rank_dir / "Turb.full_mhd_w_bcc.00000.bin").touch()
    cache_dir = tmp_path / "cache"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(extract.bc, "athinput", lambda _: {"mesh": mesh, "meshblock": meshblock})
    cache_path = Path(extract._build_cache_fname("synthetic", 3, 0.0, 0, str(cache_dir)))
    invalid = {name: np.ones((2, 2)) for name in PRIMITIVES}
    invalid["dens"][0, 0] = np.nan
    np.savez(cache_path, **invalid)

    def stop_after_quarantine(cache_fname, **_):
        assert not Path(cache_fname).exists()
        quarantined = list(cache_dir.glob(f"{cache_path.name}.invalid.*"))
        assert len(quarantined) == 1
        with np.load(quarantined[0]) as npz:
            assert np.isnan(npz["dens"][0, 0])
        raise RuntimeError("stop after quarantine")

    monkeypatch.setattr(extract, "_acquire_cache_lock", stop_after_quarantine)
    with pytest.raises(RuntimeError, match="stop after quarantine"):
        extract.extract_2d_slice("synthetic", 3, 0.0, file_number=0, cache_dir=str(cache_dir))
