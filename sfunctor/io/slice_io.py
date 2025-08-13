"""Simplified I/O utilities for SFunctor slice data.

Minimal validation for GPU-ready code.
"""
import re
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np

# Simple regex for filename parsing
_FILENAME_RE = re.compile(
    r"(?:.*beta(?P<beta>[0-9]+(?:\.[0-9]+)?).*axis(?P<axis2>[123])|slice_x(?P<axis1>[123]).*beta(?P<beta2>[0-9]+(?:\.[0-9]+)?))",
    re.IGNORECASE,
)

# Expected field mapping
_FIELD_MAP = {
    "dens": "rho",
    "velx": "v_x", "vely": "v_y", "velz": "v_z",
    "bcc1": "B_x", "bcc2": "B_y", "bcc3": "B_z",
    "vortx": "omega_x", "vorty": "omega_y", "vortz": "omega_z",
    "currx": "j_x", "curry": "j_y", "currz": "j_z",
    "curvx": "curv_x", "curvy": "curv_y", "curvz": "curv_z",
    "grad_rho_x": "grad_rho_x", "grad_rho_y": "grad_rho_y", "grad_rho_z": "grad_rho_z",
}


def parse_slice_metadata(file_path: Union[str, Path]) -> Tuple[int, float]:
    """Extract axis and beta from filename."""
    fname = Path(file_path).name
    m = _FILENAME_RE.search(fname)
    if not m:
        # Default values if can't parse
        return 3, 1.0
    
    axis = int(m.group("axis1") or m.group("axis2") or "3")
    beta = float(m.group("beta") or m.group("beta2") or "1.0")
    return axis, beta


def load_slice_npz(file_path: Union[str, Path], stride: int = 1) -> Dict[str, np.ndarray]:
    """Load slice NPZ file with minimal validation.
    
    GPU optimization note: Validation should happen once at pipeline entry,
    not in every function call.
    """
    with np.load(file_path) as npz:
        fields = {}
        
        # Load and rename fields
        for old_name, new_name in _FIELD_MAP.items():
            if old_name in npz:
                arr = npz[old_name]
                if stride > 1:
                    arr = arr[::stride, ::stride]
                fields[new_name] = arr
            else:
                # Create dummy field if missing
                shape = next(iter(npz.values())).shape
                if stride > 1:
                    shape = (shape[0] // stride, shape[1] // stride)
                fields[new_name] = np.zeros(shape, dtype=np.float32)
    
    return fields