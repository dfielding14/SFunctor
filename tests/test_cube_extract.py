from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from sfunctor.io.cube_extract import (
    AXIS_ORDER,
    CubeExtractionError,
    CubeSelection,
    extract_cube,
    file_sha256,
    plan_cube_blocks,
    preflight_selection,
    summarize_benchmark,
    validate_output_positions,
    verify_cube_output,
)

FIELDS = ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3")
BLOCK = (2, 3, 4)
GLOBAL = (4, 6, 8)


def _write_full_shard(
    path: Path,
    *,
    rank_id: int,
    logical: tuple[int, int, int],
    cycle: int = 42,
    corrupt_logical: tuple[int, int, int] | None = None,
    geometry_shift: float = 0.0,
) -> None:
    rx, ry, rz = corrupt_logical or logical
    nx, ny, nz = BLOCK
    x0, x1 = logical[0] * nx / GLOBAL[0], (logical[0] + 1) * nx / GLOBAL[0]
    y0, y1 = logical[1] * ny / GLOBAL[1], (logical[1] + 1) * ny / GLOBAL[1]
    z0, z1 = logical[2] * nz / GLOBAL[2], (logical[2] + 1) * nz / GLOBAL[2]
    kk, jj, ii = np.indices((nz, ny, nx))
    base = (
        100.0 * (kk + logical[2] * nz)
        + 10.0 * (jj + logical[1] * ny)
        + ii
        + logical[0] * nx
    ).astype(np.float32)
    values = {
        "dens": base + 1.0,
        "velx": np.full_like(base, 0.1),
        "vely": np.full_like(base, 0.2),
        "velz": np.full_like(base, 0.3),
        "eint": base + 50.0,
        "bcc1": np.full_like(base, 1.0),
        "bcc2": np.full_like(base, 2.0),
        "bcc3": np.full_like(base, 3.0),
    }
    header_dump = f"""<mesh>
nx1 = {GLOBAL[0]}
nx2 = {GLOBAL[1]}
nx3 = {GLOBAL[2]}
nghost = 0
x1min = 0.0
x1max = 1.0
x2min = 0.0
x2max = 1.0
x3min = 0.0
x3max = 1.0
<meshblock>
nx1 = {BLOCK[0]}
nx2 = {BLOCK[1]}
nx3 = {BLOCK[2]}
""".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"Athena binary output version=1.1\n")
        handle.write(b"size of preheader=5\n")
        handle.write(b"time=1.25\n")
        handle.write(f"cycle={cycle}\n".encode())
        handle.write(b"size of location=8\n")
        handle.write(b"size of variable=4\n")
        handle.write(f"number of variables={len(FIELDS)}\n".encode())
        handle.write(("variables: " + " ".join(FIELDS) + "\n").encode())
        handle.write(f"header offset={len(header_dump)}\n".encode())
        handle.write(header_dump)
        handle.write(struct.pack("@6i", 0, nx - 1, 0, ny - 1, 0, nz - 1))
        handle.write(struct.pack("@4i", rx, ry, rz, 0))
        handle.write(struct.pack("@6d", x0 + geometry_shift, x1, y0, y1, z0, z1))
        for field in FIELDS:
            handle.write(values[field].tobytes(order="C"))


@pytest.fixture()
def synthetic_data(tmp_path: Path) -> tuple[Path, np.ndarray, str]:
    data_root = tmp_path / "data"
    rank_map = np.empty((2, 2, 2), dtype=np.int64)
    basename = "full.bin"
    rank_ids = iter((7, 2, 5, 0, 6, 1, 4, 3))
    for rz in range(2):
        for ry in range(2):
            for rx in range(2):
                rank_id = next(rank_ids)
                rank_map[rz, ry, rx] = rank_id
                _write_full_shard(
                    data_root / "bin" / f"rank_{rank_id:08d}" / basename,
                    rank_id=rank_id,
                    logical=(rx, ry, rz),
                )
    return data_root, rank_map, basename


def test_contained_block_plan() -> None:
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    plans = plan_cube_blocks(selection, rank_map=np.arange(8).reshape(2, 2, 2), mesh_block_cells=BLOCK)
    assert len(plans) == 1
    assert plans[0].cell_count == 24


@pytest.mark.parametrize(
    ("bounds", "expected_blocks"),
    (
        ((0, 2, 0, 3, 0, 4), 1),
        ((0, 3, 0, 3, 0, 4), 2),
        ((0, 2, 0, 4, 0, 4), 2),
        ((0, 2, 0, 3, 0, 5), 2),
        ((1, 3, 2, 4, 3, 5), 8),
    ),
)
def test_half_open_rank_faces_select_exact_blocks(
    bounds: tuple[int, int, int, int, int, int], expected_blocks: int
) -> None:
    selection = CubeSelection("faces", bounds)
    plans = plan_cube_blocks(
        selection,
        rank_map=np.arange(8).reshape(2, 2, 2),
        mesh_block_cells=BLOCK,
    )
    assert len(plans) == expected_blocks
    assert sum(plan.cell_count for plan in plans) == selection.cell_count


def test_cross_rank_extraction_preserves_kji_and_half_open_bounds(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    selection = CubeSelection("cross", (1, 4, 2, 6, 3, 8))
    output_root = tmp_path / "out"
    result = extract_cube(
        selection,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        expected_time=1.25,
        expected_cycle=42,
        mesh_block_cells=BLOCK,
    )
    assert result["status"] == "extracted"
    density = np.load(output_root / "cross" / "fields" / "dens.npy")
    kk, jj, ii = np.indices((5, 4, 3))
    expected = 100.0 * (kk + 3) + 10.0 * (jj + 2) + ii + 1 + 1.0
    assert density.shape == (5, 4, 3)
    np.testing.assert_array_equal(density, expected)
    manifest = json.loads((output_root / "cross" / "manifest.json").read_text())
    assert manifest["cell_count"] == 60
    assert manifest["axis_order"] == AXIS_ORDER


def test_preflight_rejects_missing_rank(
    synthetic_data: tuple[Path, np.ndarray, str]
) -> None:
    data_root, rank_map, basename = synthetic_data
    (data_root / "bin" / "rank_00000007" / basename).unlink()
    with pytest.raises(CubeExtractionError, match="missing source shard"):
        preflight_selection(
            CubeSelection("cross", (1, 4, 2, 6, 3, 8)),
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )


def test_preflight_rejects_absolute_basename_and_escaping_symlink(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    with pytest.raises(CubeExtractionError, match="one simple basename"):
        preflight_selection(
            selection,
            data_root=data_root,
            basename=str((data_root / "bin" / "rank_00000007" / basename).resolve()),
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )
    source = data_root / "bin" / "rank_00000007" / basename
    external = tmp_path / "external.bin"
    source.replace(external)
    source.symlink_to(external)
    with pytest.raises(CubeExtractionError, match="escapes its declared root"):
        preflight_selection(
            selection,
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )


def test_preflight_rejects_wrong_rank_owner(
    synthetic_data: tuple[Path, np.ndarray, str]
) -> None:
    data_root, rank_map, basename = synthetic_data
    rank_map = rank_map.copy()
    rank_map[1, 1, 1] = rank_map[0, 0, 0]
    with pytest.raises(CubeExtractionError, match="logical block"):
        preflight_selection(
            CubeSelection("cross", (1, 4, 2, 6, 3, 8)),
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )


def test_preflight_rejects_missing_rank_owner(
    synthetic_data: tuple[Path, np.ndarray, str]
) -> None:
    data_root, rank_map, basename = synthetic_data
    rank_map = rank_map.copy()
    rank_map[1, 1, 1] = -1
    with pytest.raises(CubeExtractionError, match="has no rank owner"):
        preflight_selection(
            CubeSelection("cross", (1, 4, 2, 6, 3, 8)),
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )


def test_preflight_rejects_header_mismatch(
    synthetic_data: tuple[Path, np.ndarray, str]
) -> None:
    data_root, rank_map, basename = synthetic_data
    rank_id = int(rank_map[1, 1, 1])
    _write_full_shard(
        data_root / "bin" / f"rank_{rank_id:08d}" / basename,
        rank_id=rank_id,
        logical=(1, 1, 1),
        cycle=43,
    )
    with pytest.raises(CubeExtractionError, match="header differs|cycle"):
        preflight_selection(
            CubeSelection("cross", (1, 4, 2, 6, 3, 8)),
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            expected_cycle=42,
            mesh_block_cells=BLOCK,
        )


def test_preflight_rejects_expected_snapshot_and_geometry_mismatch(
    synthetic_data: tuple[Path, np.ndarray, str]
) -> None:
    data_root, rank_map, basename = synthetic_data
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    with pytest.raises(CubeExtractionError, match="snapshot identity mismatch"):
        preflight_selection(
            selection,
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
            expected_header_identity={"cycle": 99},
        )
    rank_id = int(rank_map[0, 0, 0])
    _write_full_shard(
        data_root / "bin" / f"rank_{rank_id:08d}" / basename,
        rank_id=rank_id,
        logical=(0, 0, 0),
        geometry_shift=0.01,
    )
    with pytest.raises(CubeExtractionError, match="physical geometry mismatch"):
        preflight_selection(
            selection,
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
            domain_bounds=((0.0, 1.0),) * 3,
        )


def test_out_of_domain_bounds_are_rejected(
    synthetic_data: tuple[Path, np.ndarray, str]
) -> None:
    data_root, rank_map, basename = synthetic_data
    with pytest.raises(CubeExtractionError, match="outside rank map"):
        preflight_selection(
            CubeSelection("outside", (0, 5, 0, 3, 0, 4)),
            data_root=data_root,
            basename=basename,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )


def test_checksum_corruption_and_stale_completion_are_rejected(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    extract_cube(
        selection,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    density_path = output_root / "contained" / "fields" / "dens.npy"
    with density_path.open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"xxxx")
    with pytest.raises(CubeExtractionError, match="checksum mismatch"):
        verify_cube_output(output_root / "contained")
    manifest_path = output_root / "contained" / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(CubeExtractionError, match="stale completion marker"):
        verify_cube_output(output_root / "contained")


def test_manifest_schema_is_rejected_even_with_recomputed_completion_hash(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    cube_dir = output_root / "contained"
    manifest_path = cube_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_fields"].pop("eint")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion_path = cube_dir / "COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["manifest_sha256"] = file_sha256(manifest_path)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CubeExtractionError, match="field inventory"):
        verify_cube_output(cube_dir)


@pytest.mark.parametrize("mutation", ("alias", "escape"))
def test_manifest_paths_reject_aliasing_and_escape(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path, mutation: str
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    cube_dir = output_root / "contained"
    manifest_path = cube_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_fields"]["velx"]["relative_path"] = (
        "fields/dens.npy" if mutation == "alias" else "../outside.npy"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion_path = cube_dir / "COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["manifest_sha256"] = file_sha256(manifest_path)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CubeExtractionError, match="non-canonical|escapes"):
        verify_cube_output(cube_dir)


def test_copied_cube_directory_name_mismatch_is_rejected(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    import shutil

    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    shutil.copytree(output_root / "contained", output_root / "wrong_name")
    with pytest.raises(CubeExtractionError, match="directory name"):
        verify_cube_output(output_root / "wrong_name")


def test_erased_position_attestation_is_rejected(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    cube_dir = output_root / "contained"
    manifest_path = cube_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["position_validation"]["rows"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion_path = cube_dir / "COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["manifest_sha256"] = file_sha256(manifest_path)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CubeExtractionError, match="positional validation"):
        verify_cube_output(cube_dir)


def test_asymmetric_interior_position_probe_detects_corruption(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    cube_dir = output_root / "contained"
    manifest = json.loads((cube_dir / "manifest.json").read_text())
    density_path = cube_dir / "fields" / "dens.npy"
    density = np.load(density_path, mmap_mode="r+")
    density[1, 2, 1] += 1.0
    density.flush()
    assert validate_output_positions(cube_dir / "fields", manifest["preflight"])["status"] == "failed"


def test_changed_bounds_reuse_is_rejected_and_explicit_cleanup_rebuilds(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("same_id", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    changed = CubeSelection("same_id", (0, 3, 0, 3, 0, 4))
    with pytest.raises(CubeExtractionError, match="request mismatch"):
        extract_cube(
            changed,
            data_root=data_root,
            basename=basename,
            output_root=output_root,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )
    result = extract_cube(
        changed,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
        clean_incomplete=True,
    )
    assert result["status"] == "extracted"
    assert np.load(output_root / "same_id" / "fields" / "dens.npy").shape == (4, 3, 3)


def test_explicit_cleanup_rebuilds_checksum_corrupt_complete_output(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    extract_cube(
        selection,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    density_path = output_root / "contained" / "fields" / "dens.npy"
    with density_path.open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"xxxx")
    result = extract_cube(
        selection,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
        clean_incomplete=True,
    )
    assert result["status"] == "extracted"
    assert verify_cube_output(output_root / "contained")["status"] == "passed"


def test_source_mutation_after_preflight_is_rejected(
    synthetic_data: tuple[Path, np.ndarray, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfunctor.io.cube_extract as cube_extract

    data_root, rank_map, basename = synthetic_data
    original = cube_extract.preflight_selection

    def mutate_after_preflight(*args: object, **kwargs: object) -> dict:
        payload = original(*args, **kwargs)
        path = Path(payload["source_blocks"][0]["path"])
        path.touch()
        return payload

    monkeypatch.setattr(cube_extract, "preflight_selection", mutate_after_preflight)
    with pytest.raises(CubeExtractionError, match="changed after preflight"):
        extract_cube(
            CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
            data_root=data_root,
            basename=basename,
            output_root=tmp_path / "out",
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )


def test_stale_lock_requires_explicit_cleanup(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    lock_dir = output_root / ".contained.extract.lock"
    lock_dir.mkdir(parents=True)
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    with pytest.raises(CubeExtractionError, match="lock already exists"):
        extract_cube(
            selection,
            data_root=data_root,
            basename=basename,
            output_root=output_root,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )
    result = extract_cube(
        selection,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
        clean_stale_lock=True,
    )
    assert result["status"] == "extracted"


@pytest.mark.parametrize("mutation", ("hole", "overlap"))
def test_write_time_occupancy_corruption_is_rejected(
    synthetic_data: tuple[Path, np.ndarray, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import sfunctor.io.cube_extract as cube_extract

    data_root, rank_map, basename = synthetic_data
    original = cube_extract.preflight_selection

    def corrupt(*args: object, **kwargs: object) -> dict:
        payload = original(*args, **kwargs)
        if mutation == "hole":
            payload["source_blocks"] = payload["source_blocks"][:-1]
        else:
            payload["source_blocks"].append(dict(payload["source_blocks"][0]))
        return payload

    monkeypatch.setattr(cube_extract, "preflight_selection", corrupt)
    with pytest.raises(CubeExtractionError, match=f"assembly {mutation}"):
        extract_cube(
            CubeSelection("cross", (1, 4, 2, 6, 3, 8)),
            data_root=data_root,
            basename=basename,
            output_root=tmp_path / "out",
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )


def test_summary_rejects_non_official_cube_set(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    extract_cube(
        selection,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    with pytest.raises(CubeExtractionError, match="exact four benchmark IDs"):
        summarize_benchmark(output_root, ("contained",), campaign_count=1)


def test_negative_preflight_target_slice_is_rejected(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    cube_dir = output_root / "contained"
    manifest_path = cube_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["preflight"]["source_blocks"][0]["target_slices_kji"][0][0] = -1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion_path = cube_dir / "COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["manifest_sha256"] = file_sha256(manifest_path)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CubeExtractionError, match="target slices escape"):
        verify_cube_output(cube_dir)


def test_repinned_unsampled_voxel_swap_is_rejected_by_exhaustive_validation(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    cube_dir = output_root / "contained"
    manifest_path = cube_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sampled = {
        tuple(row["target_kji"]) for row in manifest["position_validation"]["rows"]
    }
    coordinates = [
        (k, j, i)
        for k in range(4)
        for j in range(3)
        for i in range(2)
        if (k, j, i) not in sampled
    ]
    left, right = coordinates[:2]
    for field in FIELDS:
        path = cube_dir / "fields" / f"{field}.npy"
        values = np.load(path, mmap_mode="r+")
        values[left], values[right] = values[right], values[left]
        values.flush()
        manifest["output_fields"][field]["sha256"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion_path = cube_dir / "COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["manifest_sha256"] = file_sha256(manifest_path)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CubeExtractionError, match="exhaustive source validation"):
        verify_cube_output(cube_dir)


def test_repinned_external_primitive_source_path_is_rejected(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    import shutil
    import sfunctor.io.cube_extract as cube_extract

    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    extract_cube(
        CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        mesh_block_cells=BLOCK,
    )
    cube_dir = output_root / "contained"
    manifest_path = cube_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    block = manifest["preflight"]["source_blocks"][0]
    external = tmp_path / "external.bin"
    shutil.copyfile(block["path"], external)
    block["path"] = str(external)
    block["stat_fingerprint"] = cube_extract._stat_fingerprint(external)
    block["sha256"] = file_sha256(external)
    preflight = manifest["preflight"]
    preflight["source_fingerprint_sha256"] = cube_extract._source_fingerprint_sha256(
        preflight["source_blocks"]
    )
    preflight["source_plan_sha256"] = cube_extract._source_plan_sha256(preflight["source_blocks"])
    manifest["extraction_configuration"]["source_fingerprint_sha256"] = preflight[
        "source_fingerprint_sha256"
    ]
    manifest["extraction_configuration"]["source_plan_sha256"] = preflight["source_plan_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion_path = cube_dir / "COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["manifest_sha256"] = file_sha256(manifest_path)
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CubeExtractionError, match="path is not canonical"):
        verify_cube_output(cube_dir)


def test_comparison_csv_round_trip_preserves_canonical_bytes(tmp_path: Path) -> None:
    import sfunctor.io.cube_extract as cube_extract

    validation = {
        "rows": [
            {
                "quantity": "dens_1st",
                "extracted": 1.0,
                "catalog": 1.0,
                "passed": True,
            }
        ]
    }
    path = tmp_path / "comparison.csv"
    cube_extract._write_comparison_csv(path, validation)
    assert path.read_bytes() == cube_extract._comparison_csv_text(validation).encode()
    round_tripped = json.loads(json.dumps(validation, sort_keys=True))
    assert path.read_bytes() == cube_extract._comparison_csv_text(round_tripped).encode()


def test_forged_pass_counters_do_not_hide_failed_rows() -> None:
    import sfunctor.io.cube_extract as cube_extract

    validation = {
        "status": "passed",
        "comparison_count": 1,
        "failure_count": 0,
        "rows": [{"quantity": "dens_1st", "passed": False}],
    }
    with pytest.raises(CubeExtractionError, match="failed, or incoherent"):
        cube_extract._require_passed_comparisons(validation, label="forged")


def test_mid_extraction_failure_cleans_partial_output(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sfunctor.io.cube_extract as cube_extract

    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(cube_extract, "_accumulate_primary_raw_sums", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        extract_cube(
            CubeSelection("contained", (0, 2, 0, 3, 0, 4)),
            data_root=data_root,
            basename=basename,
            output_root=output_root,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )
    assert not (output_root / ".contained.partial").exists()
    failure = json.loads((output_root / "contained.failed.json").read_text())
    assert failure["partial_output_retained"] is False


def test_partial_output_requires_explicit_cleanup(
    synthetic_data: tuple[Path, np.ndarray, str], tmp_path: Path
) -> None:
    data_root, rank_map, basename = synthetic_data
    output_root = tmp_path / "out"
    partial = output_root / ".contained.partial"
    partial.mkdir(parents=True)
    selection = CubeSelection("contained", (0, 2, 0, 3, 0, 4))
    with pytest.raises(CubeExtractionError, match="partial output blocks restart"):
        extract_cube(
            selection,
            data_root=data_root,
            basename=basename,
            output_root=output_root,
            rank_map=rank_map,
            mesh_block_cells=BLOCK,
        )
    result = extract_cube(
        selection,
        data_root=data_root,
        basename=basename,
        output_root=output_root,
        rank_map=rank_map,
        clean_partial=True,
        mesh_block_cells=BLOCK,
    )
    assert result["status"] == "extracted"
