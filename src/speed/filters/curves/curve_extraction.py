"""Curve extraction and processing for mesh refinement zones.

This module provides functions for extracting curves from gradient fields,
smoothing, closing, and processing curves for mesh refinement applications.
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from .curve_simplification import simplify_curves


def close_curves_to_boundary(
    curves: List[np.ndarray], x: np.ndarray, y: np.ndarray
) -> List[np.ndarray]:
    """
    Close curves by connecting endpoints to create complete loops.

    Ensures each curve is closed by connecting the last point back to the
    first point. This creates closed refinement zones suitable for Cubit mesh
    refinement.

    Args:
        curves: List of curve arrays, each curve is shape (n_points, 2)
                containing [x, y] coordinates
        x: 2D array of x-coordinates (for domain bounds extraction)
        y: 2D array of y-coordinates (for domain bounds extraction)

    Returns:
        closed_curves: List of closed curve arrays with last point duplicated if needed
    """
    closed_curves = []

    for curve in curves:
        # Close the curve by connecting end to start
        if len(curve) > 1:
            # Check if curve is already closed
            if not np.allclose(curve[0], curve[-1]):
                # Close the curve by adding first point at the end
                closed_curve = np.vstack([curve, curve[0:1]])
            else:
                closed_curve = curve.copy()
        else:
            closed_curve = curve.copy()

        closed_curves.append(closed_curve)

    return closed_curves


def smooth_curves(
    curves: List[np.ndarray], window_length: int = 5, polyorder: int = 3
) -> List[np.ndarray]:
    """
    Smooth curves using Savitzky-Golay filter.

    Applies a Savitzky-Golay filter to smooth curve coordinates while
    preserving sharp features. Window length is automatically adjusted if curve
    is too short or if the requested length is even. Each curve's x and y
    coordinates are smoothed independently.

    Args:
        curves: List of curve arrays, each curve is shape (n_points, 2)
                containing [x, y] coordinates
        window_length: Window length for Savitzky-Golay filter, must be odd
                      and >= polyorder+1 (default: 5)
        polyorder: Polynomial order for Savitzky-Golay filter (default: 3)

    Returns:
        smoothed_curves: List of smoothed curve arrays, same structure as input
    """
    smoothed_curves = []

    for curve in curves:
        if len(curve) < window_length:
            # Skip smoothing if curve is too short
            smoothed_curves.append(curve)
            continue

        # Adjust window length if needed
        actual_window = min(window_length, len(curve))
        if actual_window % 2 == 0:
            actual_window -= 1
        actual_window = max(actual_window, polyorder + 1)

        # Smooth x and y separately
        smooth_x = savgol_filter(curve[:, 0], actual_window, polyorder)
        smooth_y = savgol_filter(curve[:, 1], actual_window, polyorder)

        smoothed_curve = np.column_stack([smooth_x, smooth_y])
        smoothed_curves.append(smoothed_curve)

    return smoothed_curves


def extract_refinement_curves(
    x: np.ndarray,
    y: np.ndarray,
    grad_mag: np.ndarray,
    percentile: float = 90,
    min_points: int = 10,
    smooth: bool = True,
    simplify: bool = True,
    simplify_epsilon: float = 2.0,
) -> Tuple[List[np.ndarray], float]:
    """
    Extract refinement curves from high-gradient regions using contour lines.

    Creates contour lines at a threshold gradient level. The threshold is
    determined by the specified percentile of the gradient magnitude
    distribution. Contour lines are extracted using matplotlib's contour
    algorithm and filtered to remove very short segments that may be artifacts.
    Optionally simplifies curves to reduce point density.

    Args:
        x: 2D array of x-coordinates, shape (ny, nx)
        y: 2D array of y-coordinates, shape (ny, nx)
        grad_mag: 2D array of gradient magnitudes, shape (ny, nx)
        percentile: Threshold percentile for gradient magnitude, 0-100
                   (default: 90). Higher values extract fewer, sharper curves
        min_points: Minimum number of points required per curve
                   (default: 10)
        smooth: Whether to smooth extracted curves with Savitzky-Goyal
               filter (default: True)
        simplify: Whether to simplify curves using RDP algorithm
                 (default: True)
        simplify_epsilon: Maximum distance threshold for point removal in
                         RDP simplification (default: 2.0)

    Returns:
        Tuple containing:
        - curves: List of curve arrays, each shape (n_points, 2)
                 containing [x, y] coordinates
        - threshold: Gradient magnitude threshold value used for extraction
    """
    # Compute threshold
    threshold = np.percentile(grad_mag, percentile)

    # Create contour at threshold level using matplotlib
    # Minimal figure for contour extraction
    fig, ax = plt.subplots(figsize=(1, 1))
    contour_set = ax.contour(x, y, grad_mag, levels=[threshold])
    plt.close(fig)

    curves = []

    # Extract line segments from allsegs
    # Contains only actual contour lines, no connectors
    for level_segs in contour_set.allsegs:
        for segment in level_segs:
            # Filter out very short segments (likely artifacts)
            if len(segment) >= min_points:
                curves.append(segment)

    # Smooth curves if requested
    if smooth and len(curves) > 0:
        curves = smooth_curves(curves)

    # Simplify curves if requested
    if simplify and len(curves) > 0:
        curves = simplify_curves(curves, epsilon=simplify_epsilon)

    return curves, threshold
