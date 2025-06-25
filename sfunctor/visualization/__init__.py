"""Visualization tools for structure function analysis.

This subpackage provides plotting functions for visualizing structure
function results and diagnostic plots.

Modules
-------
plots
    Main plotting functions for structure functions
diagnostics
    Diagnostic plots for data quality and convergence
"""

from sfunctor.visualization.plots import plot_structure_functions

__all__ = [
    "plot_structure_functions",
]