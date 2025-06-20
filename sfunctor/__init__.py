"""
SFunctor - Frontier Turbulence Structure-Function Analysis

A high-performance Python pipeline for computing anisotropic, angle-resolved
structure functions from 2D slices of large 3D MHD simulations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Make key modules available at package level
from . import sf_cli, sf_displacements, sf_histograms, sf_io, sf_parallel, sf_physics

__all__ = [
    "sf_cli",
    "sf_io",
    "sf_displacements",
    "sf_physics",
    "sf_histograms",
    "sf_parallel",
]
