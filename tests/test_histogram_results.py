"""Tests for bounded-memory, schema-validating histogram combination."""
from __future__ import annotations

import numpy as np
import pytest

from sfunctor.io.histogram_results import combine_histogram_files


def _write_result(
    path,
    *,
    value=1,
    node_id=0,
    total_nodes=1,
    start=0,
    end=2,
    ell_edges=(0.0, 2.0),
    metadata=None,
    dtype=np.int64,
    include_node=True,
    censoring=None,
):
    payload = {
        "hist": np.full((1, 1, 1, 1, 2), value, dtype=dtype),
        "channels": np.array(["D_V"]),
        "ell_bin_edges": np.array(ell_edges),
        "theta_bin_edges": np.array([0.0, np.pi / 2]),
        "phi_bin_edges": np.array([0.0, np.pi / 2]),
        "delta_bin_edges": np.array([np.array([0.0, 1.0, 2.0])], dtype=object),
        "metadata": metadata or {"slice": "slice_axis3.npz", "axis": 3, "stride": 1, "stencil_width": 2},
    }
    if include_node:
        payload["node_info"] = {
            "node_id": node_id,
            "total_nodes": total_nodes,
            "displacement_start": start,
            "displacement_end": end,
            "n_displacements": end - start,
        }
    if censoring is not None:
        payload["hist_censoring"] = np.asarray(censoring, dtype=np.int64)
        payload["censor_names"] = np.array(["accepted", "underflow", "overflow", "invalid"])
    np.savez(path, **payload)


def test_node_combiner_streams_exact_integer_counts(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_result(left, value=2, node_id=0, total_nodes=2, start=0, end=2)
    _write_result(right, value=3, node_id=1, total_nodes=2, start=2, end=5)
    result = combine_histogram_files([left, right], mode="node")
    assert result["hist"].dtype == np.int64
    assert np.array_equal(result["hist"], np.full((1, 1, 1, 1, 2), 5))
    assert result["metadata"]["total_displacements"] == 5


def test_node_combiner_rejects_missing_node(tmp_path):
    path = tmp_path / "node1.npz"
    _write_result(path, node_id=1, total_nodes=2, start=2, end=4)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        combine_histogram_files([path], mode="node")


def test_combiner_rejects_incompatible_edges(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_result(left, node_id=0, total_nodes=2, start=0, end=2)
    _write_result(right, node_id=1, total_nodes=2, start=2, end=4, ell_edges=(0.0, 3.0))
    with pytest.raises(ValueError, match="incompatible ell_bin_edges"):
        combine_histogram_files([left, right], mode="node")


def test_slice_combiner_preserves_axis_specific_metadata_without_requiring_same_axis(tmp_path):
    left = tmp_path / "axis1.npz"
    right = tmp_path / "axis3.npz"
    _write_result(
        left,
        value=2,
        metadata={"slice": "axis1", "axis": 1, "stride": 1, "stencil_width": 2, "total_displacements": 2},
        include_node=False,
    )
    _write_result(
        right,
        value=4,
        metadata={"slice": "axis3", "axis": 3, "stride": 1, "stencil_width": 2, "total_displacements": 3},
        include_node=False,
    )
    result = combine_histogram_files([left, right], mode="slice")
    assert np.array_equal(result["hist"], np.full((1, 1, 1, 1, 2), 6))
    assert result["metadata"]["total_displacements"] == 5
    assert [item["axis"] for item in result["slice_metadata"]] == [1, 3]


def test_combiner_rejects_float_histograms(tmp_path):
    path = tmp_path / "float.npz"
    _write_result(path, dtype=float)
    with pytest.raises(ValueError, match="integer dtype"):
        combine_histogram_files([path], mode="node")


def test_combiner_preserves_exact_censoring_counts(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    shape = (1, 1, 1, 1, 4)
    _write_result(left, node_id=0, total_nodes=2, start=0, end=2, censoring=np.full(shape, 2))
    _write_result(right, node_id=1, total_nodes=2, start=2, end=4, censoring=np.full(shape, 3))
    result = combine_histogram_files([left, right], mode="node")
    assert np.array_equal(result["hist_censoring"], np.full(shape, 5))
    assert np.array_equal(result["censor_names"], np.array(["accepted", "underflow", "overflow", "invalid"]))


def test_combiner_rejects_mixed_censoring_schema(tmp_path):
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_result(left, node_id=0, total_nodes=2, start=0, end=2, censoring=np.ones((1, 1, 1, 1, 4)))
    _write_result(right, node_id=1, total_nodes=2, start=2, end=4)
    with pytest.raises(ValueError, match="incompatible presence of hist_censoring"):
        combine_histogram_files([left, right], mode="node")
