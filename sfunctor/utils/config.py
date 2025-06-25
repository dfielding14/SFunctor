"""Configuration file support for SFunctor.

This module provides functionality to load and validate configuration
from YAML files, merge with command-line arguments, and support
configuration profiles for different analysis scenarios.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging
from dataclasses import dataclass, fields

from sfunctor.utils.cli import RunConfig

logger = logging.getLogger(__name__)

__all__ = [
    "load_config",
    "merge_config_with_args",
    "validate_config",
    "get_default_config_path",
    "create_config_template",
]


def get_default_config_path() -> Path:
    """Get the path to the default configuration file.
    
    Searches in order:
    1. sfunctor.yaml in current directory
    2. .sfunctor.yaml in current directory
    3. ~/.sfunctor/config.yaml
    4. $SFUNCTOR_CONFIG environment variable
    
    Returns
    -------
    Path or None
        Path to config file if found, None otherwise.
    """
    # Check current directory
    for filename in ["sfunctor.yaml", ".sfunctor.yaml"]:
        path = Path(filename)
        if path.exists():
            return path
    
    # Check home directory
    home_config = Path.home() / ".sfunctor" / "config.yaml"
    if home_config.exists():
        return home_config
    
    # Check environment variable
    env_config = os.environ.get("SFUNCTOR_CONFIG")
    if env_config:
        path = Path(env_config)
        if path.exists():
            return path
    
    return None


def load_config(config_path: Path | str, profile: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : Path or str
        Path to the configuration file.
    profile : str, optional
        Name of the profile to load from the config file.
        If None, uses base configuration only.
    
    Returns
    -------
    dict
        Configuration dictionary.
    
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    yaml.YAMLError
        If config file is invalid YAML.
    KeyError
        If specified profile doesn't exist.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")
    
    if config is None:
        config = {}
    
    # If a profile is specified, merge it with base config
    if profile:
        if 'profiles' not in config:
            raise KeyError(f"No profiles section in {config_path}")
        
        if profile not in config['profiles']:
            available = list(config.get('profiles', {}).keys())
            raise KeyError(
                f"Profile '{profile}' not found. Available profiles: {available}"
            )
        
        # Merge profile settings with base config
        profile_config = config['profiles'][profile]
        base_config = {k: v for k, v in config.items() if k != 'profiles'}
        config = _deep_merge(base_config, profile_config)
    
    return config


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.
    
    Parameters
    ----------
    base : dict
        Base dictionary.
    overlay : dict
        Dictionary to merge on top of base.
    
    Returns
    -------
    dict
        Merged dictionary.
    """
    result = base.copy()
    
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> RunConfig:
    """Merge configuration file values with command-line arguments.
    
    Command-line arguments take precedence over config file values.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from YAML file.
    args : argparse.Namespace
        Parsed command-line arguments.
    
    Returns
    -------
    RunConfig
        Merged configuration.
    """
    # Extract analysis parameters from config
    analysis_config = config.get('analysis', {})
    
    # Build RunConfig, preferring command-line args over config file
    return RunConfig(
        stride=getattr(args, 'stride', analysis_config.get('stride', 1)),
        file_name=getattr(args, 'file_name', None),
        slice_list=getattr(args, 'slice_list', None),
        n_disp_total=getattr(args, 'n_disp_total', 
                             analysis_config.get('n_disp_total', 1000)),
        N_random_subsamples=getattr(args, 'N_random_subsamples',
                                   analysis_config.get('N_random_subsamples', 1000)),
        n_ell_bins=getattr(args, 'n_ell_bins',
                          analysis_config.get('n_ell_bins', 128)),
        n_processes=getattr(args, 'n_processes',
                           analysis_config.get('n_processes', 0)),
        stencil_width=getattr(args, 'stencil_width',
                             analysis_config.get('stencil_width', 2)),
    )


def validate_config(config: RunConfig) -> None:
    """Validate configuration parameters.
    
    Parameters
    ----------
    config : RunConfig
        Configuration to validate.
    
    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    # Validate positive integers
    for field_name in ['stride', 'n_disp_total', 'N_random_subsamples', 'n_ell_bins']:
        value = getattr(config, field_name)
        if value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value}")
    
    # Validate stencil width
    if config.stencil_width not in [2, 3, 5]:
        raise ValueError(
            f"stencil_width must be 2, 3, or 5, got {config.stencil_width}"
        )
    
    # Validate file inputs
    if config.file_name is None and config.slice_list is None:
        raise ValueError("Either file_name or slice_list must be specified")
    
    if config.file_name is not None and config.slice_list is not None:
        raise ValueError("file_name and slice_list are mutually exclusive")
    
    # Validate n_processes
    if config.n_processes < 0:
        raise ValueError(f"n_processes must be non-negative, got {config.n_processes}")


def create_config_template() -> str:
    """Generate a template configuration file with all parameters.
    
    Returns
    -------
    str
        YAML configuration template as a string.
    """
    template = """# SFunctor Configuration File
# 
# This file defines default parameters for structure function analysis.
# Command-line arguments will override these values.

# Analysis parameters
analysis:
  # Spatial downsampling factor (must be positive integer)
  stride: 1
  
  # Total number of displacement vectors to generate
  n_disp_total: 1000
  
  # Number of random spatial positions per displacement
  N_random_subsamples: 1000
  
  # Number of logarithmic displacement magnitude bins
  n_ell_bins: 128
  
  # Finite difference stencil width (2, 3, or 5)
  stencil_width: 2
  
  # Number of worker processes (0 = auto-detect)
  n_processes: 0

# Histogram binning parameters
binning:
  # Angular bins for theta (latitude)
  n_theta_bins: 18
  
  # Angular bins for phi (azimuth)
  n_phi_bins: 18
  
  # Structure function value range
  sf_min: 1.0e-4
  sf_max: 10.0
  sf_bins: 128
  
  # Cross-product value range
  product_min: 1.0e-5
  product_max: 1.0e5
  product_bins: 128

# Output configuration
output:
  # Output format (currently only 'npz' supported)
  format: "npz"
  
  # Enable compression for output files
  compression: true
  
  # Directory for output files
  output_dir: "./results"
  
  # Save intermediate results
  save_intermediate: false

# Physics parameters
physics:
  # Minimum density threshold for Alfven velocity calculation
  min_density: 1.0e-10

# Predefined profiles for different use cases
profiles:
  # Quick test runs with reduced sampling
  quick_test:
    analysis:
      n_disp_total: 100
      N_random_subsamples: 100
      n_ell_bins: 32
  
  # High-resolution analysis
  high_resolution:
    analysis:
      stride: 1
      n_disp_total: 50000
      N_random_subsamples: 10000
      n_ell_bins: 256
  
  # Standard production runs
  production:
    analysis:
      n_disp_total: 20000
      N_random_subsamples: 5000
      n_ell_bins: 128
  
  # Memory-efficient settings for large datasets
  memory_efficient:
    analysis:
      stride: 4
      n_disp_total: 5000
      N_random_subsamples: 2000
      n_processes: 4  # Limit parallelism to reduce memory
"""
    return template


if __name__ == "__main__":
    # If run as a script, create a template config file
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "sfunctor_template.yaml"
    
    template = create_config_template()
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"Created configuration template: {output_path}")