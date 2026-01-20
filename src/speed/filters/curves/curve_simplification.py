"""Curve simplification utilities for reducing point density in polylines.

This module provides functions for simplifying curves using the Ramer-Douglas-Peucker
algorithm and downsampling techniques. These operations reduce the number of points
in a curve while preserving its essential shape characteristics.
"""

from typing import List
import numpy as np


def simplify_curves(curves: List[np.ndarray], epsilon: float = 1.0) -> List[np.ndarray]:
    """
    Simplify curves using Ramer-Douglas-Peucker (RDP) algorithm.

    The RDP algorithm reduces the number of points in a curve by recursively
    removing points that are within a specified distance (epsilon) from a
    simplified line segment. Curves are simplified adaptively based on their
    original size, with larger curves receiving more aggressive simplification.
    Very short curves (< 3 points after simplification) are filtered out.

    Args:
        curves: List of curve arrays, each curve is shape (n_points, 2)
                containing [x, y] coordinates
        epsilon: Maximum perpendicular distance from a point to the simplified
                line segment for it to be kept (default: 1.0). Larger values
                produce more aggressive simplification.

    Returns:
        simplified_curves: List of simplified curve arrays, filtered to include
                          only curves with >= 3 points
    """

    def rdp_simplify(curve: np.ndarray, epsilon: float) -> np.ndarray:
        """Ramer-Douglas-Peucker simplification (recursive implementation)."""
        if len(curve) < 3:
            return curve

        # Find the point with maximum perpendicular distance from line
        start = curve[0]
        end = curve[-1]

        # Vector from start to end
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            # Degenerate case: start and end are essentially the same
            return np.array([start, end])

        # Compute perpendicular distance for all intermediate points
        dists = np.abs(np.cross(line_vec, start - curve[1:-1])) / line_len

        max_idx = np.argmax(dists) + 1  # +1 because we excluded first point
        max_dist = dists[max_idx - 1]

        # If max distance exceeds threshold, keep the point and recurse
        if max_dist > epsilon:
            # Recursively simplify the two segments
            left = rdp_simplify(curve[: max_idx + 1], epsilon)
            right = rdp_simplify(curve[max_idx:], epsilon)
            # Combine, removing duplicate middle point
            return np.vstack([left[:-1], right])
        else:
            # Remove all intermediate points
            return np.array([start, end])

    def downsample_curve(curve: np.ndarray, max_points: int = 100) -> np.ndarray:
        """Downsample curve to max_points using uniform spacing."""
        if len(curve) <= max_points:
            return curve

        # Calculate cumulative distance along curve
        distances = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        cumsum = np.concatenate(([0], np.cumsum(distances)))
        total_dist = cumsum[-1]

        # Sample uniformly along the curve
        target_distances = np.linspace(0, total_dist, max_points)
        interpolated = np.interp(target_distances, cumsum, np.arange(len(curve)))
        indices = np.unique(np.round(interpolated).astype(int))
        return curve[indices]

    simplified_curves = []
    for curve in curves:
        original_len = len(curve)

        # Adaptive simplification based on curve size
        if original_len > 150:
            # Very large curves: use downsampling instead of aggressive RDP
            # to preserve shape while reducing points
            max_downsampled = 150
            simplified = downsample_curve(curve, max_points=max_downsampled)
        elif original_len > 50:
            # Medium curves: gentle RDP simplification
            adaptive_epsilon = epsilon * 0.3
            max_downsampled = 120
            simplified = rdp_simplify(curve, adaptive_epsilon)
            if len(simplified) > max_downsampled:
                simplified = downsample_curve(simplified, max_points=max_downsampled)
        else:
            # Small curves: very gentle simplification
            adaptive_epsilon = epsilon * 0.2
            max_downsampled = 50
            simplified = rdp_simplify(curve, adaptive_epsilon)
            if len(simplified) > max_downsampled:
                simplified = downsample_curve(simplified, max_points=max_downsampled)

        if len(simplified) >= 3:  # Keep only curves with >= 3 points
            simplified_curves.append(simplified)

    return simplified_curves
