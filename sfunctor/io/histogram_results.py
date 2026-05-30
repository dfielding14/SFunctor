"""Streaming combination and validation for unified histogram result files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = [
    "combine_histogram_files",
    "histogram_censoring_summary",
    "load_histogram_file",
]


SCHEMA_KEYS = (
    "channels",
    "ell_bin_edges",
    "theta_bin_edges",
    "phi_bin_edges",
    "delta_bin_edges",
)
COMMON_METADATA_KEYS = (
    "stride",
    "stencil_width",
    "N_random_subsamples",
    "cell_sizes",
    "random_seed",
)
OPTIONAL_COUNT_ARRAYS = {
    "hist_censoring": "censor_names",
}


def _metadata(npz) -> dict:
    return dict(npz["metadata"].item()) if "metadata" in npz else {}


def load_histogram_file(path: str | Path) -> dict[str, object]:
    """Load one histogram product into independent arrays and dictionaries."""

    with np.load(path, allow_pickle=True) as npz:
        if "hist" not in npz:
            raise ValueError(f"{path}: missing hist")
        result: dict[str, object] = {
            "hist": np.asarray(npz["hist"]),
            "metadata": _metadata(npz),
            "path": str(path),
        }
        for key in SCHEMA_KEYS:
            if key not in npz:
                raise ValueError(f"{path}: missing {key}")
            result[key] = np.asarray(npz[key])
        for count_key, schema_key in OPTIONAL_COUNT_ARRAYS.items():
            if (count_key in npz) != (schema_key in npz):
                raise ValueError(f"{path}: {count_key} and {schema_key} must appear together")
            if count_key in npz:
                result[count_key] = np.asarray(npz[count_key])
                result[schema_key] = np.asarray(npz[schema_key])
        if "node_info" in npz:
            result["node_info"] = dict(npz["node_info"].item())
    return result


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype == object or right.dtype == object:
        if left.shape != right.shape:
            return False
        return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left.flat, right.flat))
    return np.array_equal(left, right)


def _validate_compatible(reference: dict[str, object], candidate: dict[str, object], *, mode: str) -> None:
    path = candidate["path"]
    if np.asarray(reference["hist"]).shape != np.asarray(candidate["hist"]).shape:
        raise ValueError(f"{path}: histogram shape is incompatible with the first input")
    for key in SCHEMA_KEYS:
        if not _same_array(np.asarray(reference[key]), np.asarray(candidate[key])):
            raise ValueError(f"{path}: incompatible {key}")
    for count_key, schema_key in OPTIONAL_COUNT_ARRAYS.items():
        if (count_key in reference) != (count_key in candidate):
            raise ValueError(f"{path}: incompatible presence of {count_key}")
        if count_key in reference:
            if np.asarray(reference[count_key]).shape != np.asarray(candidate[count_key]).shape:
                raise ValueError(f"{path}: incompatible {count_key} shape")
            if not _same_array(np.asarray(reference[schema_key]), np.asarray(candidate[schema_key])):
                raise ValueError(f"{path}: incompatible {schema_key}")

    reference_metadata = reference["metadata"]
    candidate_metadata = candidate["metadata"]
    for key in COMMON_METADATA_KEYS:
        if key in reference_metadata or key in candidate_metadata:
            if reference_metadata.get(key) != candidate_metadata.get(key):
                raise ValueError(f"{path}: incompatible metadata[{key!r}]")
    if mode == "node" and reference_metadata.get("slice") != candidate_metadata.get("slice"):
        raise ValueError(f"{path}: node inputs refer to different slices")
    if mode == "node" and reference_metadata.get("axis") != candidate_metadata.get("axis"):
        raise ValueError(f"{path}: node inputs refer to different slice axes")


def _add_counts(total: np.ndarray, increment: np.ndarray, path: str) -> None:
    if increment.dtype.kind not in "iu":
        raise ValueError(f"{path}: histogram counts must use an integer dtype")
    if np.any(increment < 0):
        raise ValueError(f"{path}: histogram counts must be non-negative")
    if total.size and increment.size and int(total.max()) > np.iinfo(np.int64).max - int(increment.max()):
        raise OverflowError(f"{path}: int64 histogram accumulation could overflow")
    np.add(total, increment, out=total, casting="unsafe")


def _validate_node_coverage(node_infos: list[dict]) -> None:
    if not node_infos:
        raise ValueError("node combination requires node_info in every input")
    total_nodes = {int(info["total_nodes"]) for info in node_infos}
    if len(total_nodes) != 1:
        raise ValueError("node inputs disagree on total_nodes")
    expected = next(iter(total_nodes))
    node_ids = [int(info["node_id"]) for info in node_infos]
    if sorted(node_ids) != list(range(expected)):
        raise ValueError(f"node coverage is incomplete or duplicated: found {sorted(node_ids)}, expected 0..{expected - 1}")
    intervals = sorted((int(info["displacement_start"]), int(info["displacement_end"])) for info in node_infos)
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] != current[0]:
            raise ValueError(f"displacement intervals are incomplete or overlapping: {previous}, {current}")


def histogram_censoring_summary(
    hist_censoring: np.ndarray,
    censor_names: np.ndarray,
    channels: np.ndarray,
) -> list[dict[str, object]]:
    """Summarize accepted, censored, and invalid legacy values by channel."""

    hist_censoring = np.asarray(hist_censoring)
    censor_names = tuple(str(name) for name in np.asarray(censor_names))
    channels = tuple(str(name) for name in np.asarray(channels))
    expected = {"accepted", "underflow", "overflow", "invalid"}
    if hist_censoring.ndim != 5 or hist_censoring.shape[0] != len(channels):
        raise ValueError("hist_censoring must have shape (channel, ell, theta, phi, censor_kind)")
    if hist_censoring.shape[-1] != len(censor_names) or set(censor_names) != expected:
        raise ValueError(f"censor_names must contain exactly {sorted(expected)}")
    if hist_censoring.dtype.kind not in "iu" or np.any(hist_censoring < 0):
        raise ValueError("hist_censoring must contain non-negative integer counts")

    totals = hist_censoring.sum(axis=(1, 2, 3), dtype=np.int64)
    indices = {name: censor_names.index(name) for name in censor_names}
    rows = []
    for channel, counts in zip(channels, totals, strict=True):
        attempted = int(counts.sum())
        censored = int(counts[indices["underflow"]] + counts[indices["overflow"]])
        rows.append(
            {
                "channel": channel,
                **{name: int(counts[index]) for name, index in indices.items()},
                "attempted": attempted,
                "censored_fraction": censored / attempted if attempted else 0.0,
                "invalid_fraction": int(counts[indices["invalid"]]) / attempted if attempted else 0.0,
            }
        )
    return rows


def combine_histogram_files(paths: Iterable[str | Path], *, mode: str) -> dict[str, object]:
    """Stream compatible count arrays into one exact ``int64`` histogram."""

    paths = [str(path) for path in paths]
    if not paths:
        raise ValueError("at least one histogram file is required")
    if mode not in {"node", "slice"}:
        raise ValueError("mode must be 'node' or 'slice'")

    reference = load_histogram_file(paths[0])
    first_hist = np.asarray(reference["hist"])
    if first_hist.dtype.kind not in "iu":
        raise ValueError(f"{paths[0]}: histogram counts must use an integer dtype")
    hist_total = first_hist.astype(np.int64, copy=True)
    count_totals = {
        key: np.asarray(reference[key]).astype(np.int64, copy=True)
        for key in OPTIONAL_COUNT_ARRAYS
        if key in reference
    }
    node_infos = []
    slice_metadata = []
    slice_names = []
    total_displacements = 0

    for index, path in enumerate(paths):
        current = reference if index == 0 else load_histogram_file(path)
        if index:
            _validate_compatible(reference, current, mode=mode)
            _add_counts(hist_total, np.asarray(current["hist"]), path)
            for key, total in count_totals.items():
                _add_counts(total, np.asarray(current[key]), path)
        metadata = dict(current["metadata"])
        if mode == "node":
            if "node_info" not in current:
                raise ValueError(f"{path}: node combination requires node_info")
            info = dict(current["node_info"])
            node_infos.append(info)
            total_displacements += int(info["n_displacements"])
        else:
            slice_names.append(Path(path).stem)
            slice_metadata.append(metadata)
            total_displacements += int(metadata.get("total_displacements", 0))

    if mode == "node":
        _validate_node_coverage(node_infos)
        metadata = dict(reference["metadata"])
        metadata.update(
            n_nodes=len(node_infos),
            total_displacements=total_displacements,
            combined_from=len(paths),
            combine_mode="node",
        )
    else:
        metadata = {
            "n_slices": len(paths),
            "slice_names": slice_names,
            "total_displacements": total_displacements,
            "merged_from": len(paths),
            "combine_mode": "slice",
        }
        for key in COMMON_METADATA_KEYS:
            if key in reference["metadata"]:
                metadata[key] = reference["metadata"][key]

    return {
        "hist": hist_total,
        **count_totals,
        **{key: reference[key] for key in SCHEMA_KEYS},
        **{
            schema_key: reference[schema_key]
            for count_key, schema_key in OPTIONAL_COUNT_ARRAYS.items()
            if count_key in reference
        },
        "metadata": metadata,
        "node_infos": node_infos,
        "slice_metadata": slice_metadata,
    }
