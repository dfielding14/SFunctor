"""YAML configuration helpers for SFunctor.

This module provides utilities to load a YAML config file, optionally apply a
named profile overlay, and validate the resulting runtime configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml

from sfunctor.utils.cli import RunConfig

__all__ = [
    "load_config",
    "validate_config",
    "create_config_template",
    "_deep_merge",
    "get_default_config_path",
]


def get_default_config_path() -> Path | None:
    """Return the default config path if present in the CWD.

    Preference order is ``./sfunctor.yaml`` then ``./sfunctor.yml``.
    """
    for name in ("sfunctor.yaml", "sfunctor.yml"):
        path = Path(name)
        if path.exists():
            return path
    return None


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries, returning a new merged dict.

    - For keys present in both dicts:
      - If both values are dict-like, merge recursively.
      - Otherwise, the overlay value replaces the base value.
    - Keys only in overlay are added.
    """
    merged: Dict[str, Any] = dict(base)
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(overlay_value, Mapping):
            merged[key] = _deep_merge(base_value, overlay_value)
        else:
            merged[key] = overlay_value
    return merged


def load_config(file_path: Union[str, Path], *, profile: str | None = None) -> Dict[str, Any]:
    """Load a YAML config file and optionally apply a profile overlay.

    Parameters
    ----------
    file_path:
        Path to a YAML file.
    profile:
        Optional profile name under the top-level ``profiles`` key.

    Returns
    -------
    dict
        Parsed config with the profile overlay applied (if requested).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got {type(config).__name__}")

    if profile is None:
        return config

    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict) or profile not in profiles:
        raise KeyError(f"Profile '{profile}' not found")

    overlay = profiles[profile]
    if overlay is None:
        return config
    if not isinstance(overlay, dict):
        raise ValueError(
            f"Profile '{profile}' must be a mapping, got {type(overlay).__name__}"
        )

    return _deep_merge(config, overlay)


def validate_config(config: RunConfig) -> None:
    """Validate a :class:`~sfunctor.utils.cli.RunConfig` instance."""
    if config.stride <= 0:
        raise ValueError("stride must be positive")

    if config.stencil_width not in (2, 3, 5):
        raise ValueError("stencil_width must be one of {2, 3, 5}")

    if config.file_name is None and config.slice_list is None:
        raise ValueError("One of file_name or slice_list must be provided")

    if config.file_name is not None and config.slice_list is not None:
        raise ValueError("file_name and slice_list are mutually exclusive")

    if config.n_disp_total <= 0:
        raise ValueError("n_disp_total must be positive")

    if config.N_random_subsamples <= 0:
        raise ValueError("N_random_subsamples must be positive")

    if config.n_ell_bins <= 0:
        raise ValueError("n_ell_bins must be positive")


def create_config_template() -> str:
    """Return a default YAML configuration template as a string."""
    return """\
analysis:
  stride: 1
  n_disp_total: 1000
  N_random_subsamples: 1000
  n_ell_bins: 128
  n_processes: 0
  stencil_width: 2

binning:
  n_theta_bins: 18
  n_phi_bins: 16
  # Shared Δ binning applied to every channel (same bin count, per-channel ranges)
  N_delta_bin_edges: 201
  log_delta_bin_edges_min: []
  log_delta_bin_edges_max: []

output:
  out_dir: "."

physics:
  enabled: true

profiles:
  quick_test:
    analysis:
      n_disp_total: 100
      N_random_subsamples: 100
      n_ell_bins: 16
      n_processes: 1
"""

