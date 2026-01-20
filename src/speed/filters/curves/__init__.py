"""Curve extraction and processing for mesh refinement.

This package provides tools for extracting refinement curves from DEM data
for use in Cubit mesh generation.
"""

from .dem_processing import load_dem, interpolate_dem, compute_gradient
from .curve_extraction import (
    extract_refinement_curves,
    close_curves_to_boundary,
    smooth_curves,
)
from .curve_simplification import simplify_curves
from .curve_enlargement import (
    enlarge_curves,
    enlarge_curve,
    shrink_curves,
    shrink_curve,
    estimate_curve_area,
    calculate_offset_distance,
)
from .curve_export import (
    export_curves_xyz,
    export_curves_vtk,
    export_curves_pvd,
    export_curves_sat,
)
from .curve_visualization import visualize_results, debug_visualization, visualize_enlarged_curves, visualize_shrunk_curves

__all__ = [
    # DEM processing
    "load_dem",
    "interpolate_dem",
    "compute_gradient",
    # Curve extraction
    "extract_refinement_curves",
    "close_curves_to_boundary",
    "smooth_curves",
    # Curve simplification
    "simplify_curves",
    # Curve enlargement/shrinking
    "enlarge_curves",
    "enlarge_curve",
    "shrink_curves",
    "shrink_curve",
    "estimate_curve_area",
    "calculate_offset_distance",
    # Export
    "export_curves_xyz",
    "export_curves_vtk",
    "export_curves_pvd",
    "export_curves_sat",
    # Visualization
    "visualize_results",
    "debug_visualization",
    "visualize_enlarged_curves",
    "visualize_shrunk_curves",
]
