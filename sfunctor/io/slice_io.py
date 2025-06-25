"""I/O utilities for SFunctor slice data.

This module provides functions for loading and parsing 2D slice data files
produced by the extract_2d_slice module. It handles file validation,
metadata extraction, and data loading with proper error handling.
"""

import re
from pathlib import Path
from typing import Dict, Tuple
import logging

import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)

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
    file_path : str | Path
        Full path or filename of the slice ``*.npz`` produced by
        ``extract_2d_slice``. Must be a valid path string or Path object.

    Returns
    -------
    tuple[int, float]
        - axis: 1, 2, or 3 corresponding to x, y, or z dimension
        - beta: plasma beta parameter as float

    Raises
    ------
    TypeError
        If file_path is not a string or Path object.
    ValueError
        If the filename does not match the expected pattern or if axis
        is not in range [1, 3].

    Examples
    --------
    >>> axis, beta = parse_slice_metadata("slice_x1_-0.375_beta25_test.npz")
    >>> print(axis, beta)  # 1, 25.0
    """
    # Validate input type
    if not isinstance(file_path, (str, Path)):
        raise TypeError(
            f"file_path must be a string or Path object, got {type(file_path).__name__}"
        )
    
    try:
        fname = Path(file_path).name  # strip directories
    except Exception as e:
        raise ValueError(f"Invalid file path: {file_path}") from e
    
    m = _FILENAME_RE.search(fname)
    if not m:
        raise ValueError(
            f"Cannot parse axis/beta from filename '{fname}'."
            " Expected pattern like 'slice_x1_<...>_beta25_<...>.npz'"
        )
    
    try:
        axis = int(m.group("axis"))
        if axis not in (1, 2, 3):
            raise ValueError(f"Invalid axis {axis}, must be 1, 2, or 3")
        
        beta = float(m.group("beta"))
        if beta < 0:
            raise ValueError(f"Invalid beta {beta}, must be non-negative")
        
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to parse axis/beta from matched groups: {e}"
        ) from e
    
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
    file_path : str | Path
        Path to the ``*.npz`` file produced by ``extract_2d_slice``.
        Must be a valid file path that exists.
    stride : int, optional
        Spatial down-sampling factor.  A value of *n* keeps every *n*-th cell
        along both in-plane directions. Must be a positive integer.
        Default is 1 (no downsampling).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of canonical field names (``rho``, ``v_x``, …) to 2-D arrays
        with shape *(Ny/stride, Nx/stride)*. All arrays are float type.

    Raises
    ------
    TypeError
        If file_path is not a string or Path object, or if stride is not an integer.
    ValueError
        If stride is not positive, if the file is empty, or if arrays have
        inconsistent shapes.
    FileNotFoundError
        If the specified file does not exist.
    OSError
        If there are issues reading the file (permissions, corruption, etc.).

    Examples
    --------
    >>> data = load_slice_npz("slice_x1_0.0_beta25_0001.npz")
    >>> print(data.keys())  # dict_keys(['rho', 'v_x', 'v_y', ...])
    
    >>> # Load with downsampling
    >>> data_small = load_slice_npz("slice.npz", stride=4)
    """
    # Input validation
    if not isinstance(file_path, (str, Path)):
        raise TypeError(
            f"file_path must be a string or Path object, got {type(file_path).__name__}"
        )
    
    if not isinstance(stride, int):
        raise TypeError(f"stride must be an integer, got {type(stride).__name__}")
    
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    
    try:
        file_path = Path(file_path)
    except Exception as e:
        raise ValueError(f"Invalid file path: {file_path}") from e
    
    if not file_path.exists():
        raise FileNotFoundError(f"Slice file not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path exists but is not a file: {file_path}")
    
    if not file_path.suffix == '.npz':
        raise ValueError(
            f"Expected .npz file extension, got '{file_path.suffix}' for {file_path}"
        )
    
    try:
        with np.load(file_path, allow_pickle=True) as data:
            # Determine reference shape from first array (no stride yet)
            try:
                ref_arr = next(iter(data.values()))
            except StopIteration:
                raise ValueError(f"Slice file '{file_path}' is empty.")
            
            # Validate array shape
            if ref_arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays in slice file, got {ref_arr.ndim}D array"
                )
            
            ny_full, nx_full = ref_arr.shape
            
            # Check if stride is compatible with array size
            if stride > min(ny_full, nx_full):
                raise ValueError(
                    f"Stride {stride} is larger than smallest array dimension "
                    f"({min(ny_full, nx_full)}). Arrays have shape {ref_arr.shape}"
                )
            
            ny, nx = ny_full // stride, nx_full // stride
            
            if ny == 0 or nx == 0:
                raise ValueError(
                    f"Stride {stride} results in empty array. "
                    f"Original shape: {ref_arr.shape}"
                )

            out: Dict[str, np.ndarray] = {}
            for src_key, canon_key in _EXPECTED_KEYS.items():
                if src_key in data:
                    arr = data[src_key]
                    
                    # Validate array consistency
                    if arr.shape != (ny_full, nx_full):
                        raise ValueError(
                            f"Inconsistent array shapes in file: {src_key} has shape "
                            f"{arr.shape}, expected {(ny_full, nx_full)}"
                        )
                    
                    # Apply stride
                    arr = arr[::stride, ::stride]
                else:
                    # fill missing key with NaNs; useful during transition period
                    arr = np.full((ny, nx), np.nan, dtype=float)
                
                # Ensure float type
                out[canon_key] = arr.astype(float, copy=False)
    
    except OSError as e:
        raise OSError(f"Failed to read slice file '{file_path}': {e}") from e
    except Exception as e:
        # Re-raise with more context if it's not already one of our exceptions
        if not isinstance(e, (TypeError, ValueError, FileNotFoundError, OSError)):
            raise RuntimeError(f"Unexpected error loading '{file_path}': {e}") from e
        raise

    return out 