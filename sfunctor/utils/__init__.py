"""Utility functions for SFunctor.

This subpackage contains helper functions and utilities used throughout
the package.

Modules
-------
cli
    Command-line interface parsing and configuration
displacements
    Displacement vector generation for structure functions
logging
    Logging configuration and utilities
"""

from sfunctor.utils.cli import parse_cli, RunConfig
from sfunctor.utils.displacements import (
    find_ell_bin_edges,
    build_displacement_list,
)

__all__ = [
    # CLI
    "parse_cli",
    "RunConfig",
    # Displacements
    "find_ell_bin_edges",
    "build_displacement_list",
]