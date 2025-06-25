"""Utility functions for SFunctor.

This subpackage contains helper functions and utilities used throughout
the package.

Modules
-------
cli
    Command-line interface parsing and configuration
config
    Configuration file loading and validation
displacements
    Displacement vector generation for structure functions
logging
    Logging configuration and utilities
"""

from sfunctor.utils.cli import parse_cli, RunConfig
from sfunctor.utils.config import (
    load_config,
    create_config_template,
    get_default_config_path,
)
from sfunctor.utils.displacements import (
    find_ell_bin_edges,
    build_displacement_list,
)

__all__ = [
    # CLI
    "parse_cli",
    "RunConfig",
    # Config
    "load_config",
    "create_config_template",
    "get_default_config_path",
    # Displacements
    "find_ell_bin_edges",
    "build_displacement_list",
]