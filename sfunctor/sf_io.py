import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "parse_slice_metadata",
    "load_slice_npz",
]


#: Regex pre-compiled once for performance.  Matches e.g. "slice_x1_-0.375_Turb_5120_beta25_dedt025_plm_0022.npz"
_FILENAME_RE = re.compile(
    r"slice_x(?P<axis>[123])_.*?_beta(?P<beta>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def parse_slice_metadata(file_path: str | Path) -> Tuple[int, float]:
    """Extract *axis* and *beta* from a slice filename.

    Parameters
    ----------
    file_path
        Full path or filename of the slice ``*.npz`` produced by
        ``extract_2d_slice``.

    Returns
    -------
    (axis, beta)
        *axis* is 1, 2, or 3.  *beta* is returned as ``float``.

    Raises
    ------
    ValueError
        If the filename does not match the expected pattern.
    """
    fname = Path(file_path).name  # strip directories
    m = _FILENAME_RE.search(fname)
    if not m:
        raise ValueError(
            f"Cannot parse axis/beta from filename '{fname}'."
            " Expected pattern like 'slice_x1_<...>_beta25_<...>.npz'"
        )
    axis = int(m.group("axis"))
    beta = float(m.group("beta"))
    return axis, beta


# -----------------------------------------------------------------------------
# Slice loader ----------------------------------------------------------------
# -----------------------------------------------------------------------------

_EXPECTED_KEYS = {
    # density & primitive fields
    "dens": "rho",
    "velx": "v_x",
    "vely": "v_y",
    "velz": "v_z",
    "bcc1": "B_x",
    "bcc2": "B_y",
    "bcc3": "B_z",
    # vorticity
    "vortx": "omega_x",
    "vorty": "omega_y",
    "vortz": "omega_z",
    # current density
    "currx": "j_x",
    "curry": "j_y",
    "currz": "j_z",
}


def load_slice_npz(file_path: str | Path, *, stride: int = 1) -> Dict[str, np.ndarray]:
    """Load a slice ``*.npz`` file and apply *stride* uniformly to all 2-D arrays.

    The function guarantees that every expected physical quantity is present in
    the returned dictionary.  Missing keys are filled with arrays of
    ``np.nan`` so that downstream code can rely on their existence.

    Parameters
    ----------
    file_path
        Path to the ``*.npz`` file produced by :pyfunc:`extract_2d_slice`.
    stride
        Spatial down-sampling factor.  A value of *n* keeps every *n*-th cell
        along both in-plane directions.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of canonical field names (``rho``, ``v_x``, …) to 2-D arrays
        with shape *(Ny/stride, Nx/stride)*.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with np.load(file_path, allow_pickle=True) as data:
        # Determine reference shape from first array (no stride yet)
        try:
            ref_arr = next(iter(data.values()))
        except StopIteration:
            raise ValueError(f"Slice file '{file_path}' is empty.")
        ny_full, nx_full = ref_arr.shape
        ny, nx = ny_full // stride, nx_full // stride

        out: Dict[str, np.ndarray] = {}
        for src_key, canon_key in _EXPECTED_KEYS.items():
            if src_key in data:
                arr = data[src_key][::stride, ::stride]
            else:
                # fill missing key with NaNs; useful during transition period
                arr = np.full((ny, nx), np.nan, dtype=float)
            out[canon_key] = arr.astype(float, copy=False)

    return out
