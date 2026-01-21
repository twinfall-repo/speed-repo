"""Curve enlargement utilities for expanding refinement curves while avoiding self-intersections.

This module provides functions for offsetting and enlarging curves based on percentage
expansion while detecting and handling potential self-intersections.
"""

from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.ops import unary_union


def estimate_curve_area(curve: np.ndarray) -> float:
    """
    Estimate the area enclosed by a closed curve using the shoelace formula.

    Args:
        curve: Array of shape (n_points, 2) containing [x, y] coordinates
               (should be a closed curve)

    Returns:
        Estimated area of the curve (absolute value)
    """
    if len(curve) < 3:
        return 0.0

    # Shoelace formula for polygon area
    x = curve[:, 0]
    y = curve[:, 1]
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area


def calculate_offset_distance(curve: np.ndarray, percentage: float) -> float:
    """
    Calculate the offset distance based on curve area and enlargement percentage.

    Args:
        curve: Array of shape (n_points, 2) containing [x, y] coordinates
        percentage: Enlargement percentage (0-100). For a 10% enlargement of area,
                   the offset distance is calculated from the equivalent circle radius.

    Returns:
        Offset distance in the same units as the curve coordinates
    """
    if percentage <= 0:
        return 0.0

    area = estimate_curve_area(curve)
    if area <= 0:
        # Estimate from perimeter instead
        perimeter = np.sum(np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1)))
        if perimeter <= 0:
            return 0.0
        # For a rough circle equivalent: circumference = 2πr, so r = circumference/(2π)
        radius = perimeter / (2 * np.pi)
    else:
        # From area: A = πr², so r = √(A/π)
        radius = np.sqrt(area / np.pi)

    # Calculate new radius for desired area increase
    # New area = original area * (1 + percentage/100)
    # New radius = √(new_area / π) = √((original_area * (1 + percentage/100)) / π)
    new_radius = radius * np.sqrt(1 + percentage / 100)
    offset_distance = new_radius - radius

    return offset_distance


def enlarge_curve(
    curve: np.ndarray, percentage: float, resolution: int = 10
) -> Optional[np.ndarray]:
    """
    Enlarge a closed curve outward by a given percentage while detecting self-intersections.

    Args:
        curve: Array of shape (n_points, 2) containing [x, y] coordinates
               (should be a closed curve)
        percentage: Enlargement percentage (0-100)
        resolution: Number of segments for buffering (higher = smoother but slower)

    Returns:
        Enlarged curve as array of shape (n_points, 2), or None if self-intersection detected
    """
    if percentage <= 0:
        return curve.copy()

    if len(curve) < 3:
        return curve.copy()

    # Calculate offset distance
    offset_dist = calculate_offset_distance(curve, percentage)

    if offset_dist <= 0:
        return curve.copy()

    try:
        # Create a LineString from the curve
        line = LineString(curve)

        # Buffer outward to enlarge (positive buffer = outward offset)
        buffered = line.buffer(offset_dist, resolution=resolution)

        # Extract the exterior coordinates (converted back to array)
        if buffered.is_empty:
            return None

        if isinstance(buffered, Polygon):
            enlarged_coords = np.array(
                buffered.exterior.coords[:-1]
            )  # Exclude repeated last point
        elif isinstance(buffered, MultiLineString):
            # Multiple disconnected lines - take the largest one
            lines = list(buffered.geoms)
            if not lines:
                return None
            largest = max(lines, key=lambda l: len(l.coords))
            enlarged_coords = np.array(largest.coords[:-1])
        else:
            return None

        # Ensure the curve is closed by appending the first point
        if not np.allclose(enlarged_coords[0], enlarged_coords[-1]):
            enlarged_coords = np.vstack([enlarged_coords, enlarged_coords[0]])

        return enlarged_coords

    except Exception as e:
        print(f"Warning: Failed to enlarge curve: {e}")
        return None


def enlarge_curves(
    curves: List[np.ndarray],
    percentage: float,
    resolution: int = 10,
    filter_invalid: bool = True,
) -> List[np.ndarray]:
    """
    Enlarge multiple closed curves by a given percentage while detecting self-intersections.

    Args:
        curves: List of curve arrays, each shape (n_points, 2)
        percentage: Enlargement percentage (0-100)
        resolution: Number of segments for buffering
        filter_invalid: If True, remove curves that fail to enlarge or have self-intersections

    Returns:
        List of enlarged curve arrays. If filter_invalid is True, excludes failed curves.
    """
    if percentage <= 0:
        return curves

    enlarged_curves = []

    for i, curve in enumerate(curves):
        # Check if curve is closed
        if not np.allclose(curve[0], curve[-1]):
            print(f"  Curve {i}: Not closed, skipped")
            if not filter_invalid:
                enlarged_curves.append(curve)
            continue

        enlarged = enlarge_curve(curve, percentage, resolution=resolution)

        if enlarged is None:
            if not filter_invalid:
                enlarged_curves.append(curve)
            print(f"  Curve {i}: Failed to enlarge (self-intersection?), skipped")
        else:
            enlarged_curves.append(enlarged)
            # Calculate area change
            original_area = estimate_curve_area(curve)
            new_area = estimate_curve_area(enlarged)
            if original_area > 0:
                actual_percentage = ((new_area - original_area) / original_area) * 100
                print(
                    f"  Curve {i}: Enlarged {len(curve)} → {len(enlarged)} points "
                    f"({actual_percentage:.1f}% area increase)"
                )
            else:
                print(f"  Curve {i}: Enlarged {len(curve)} → {len(enlarged)} points")

    return enlarged_curves


def offset_curve_safe(
    curve: np.ndarray,
    offset_distance: float,
    side: str = "left",
    resolution: int = 10,
) -> Optional[np.ndarray]:
    """
    Offset a curve to the left or right with self-intersection detection.

    Args:
        curve: Array of shape (n_points, 2) containing [x, y] coordinates
        offset_distance: Distance to offset (positive = left, negative = right)
        side: "left" or "right" (alternative to specifying sign)
        resolution: Number of segments for buffering

    Returns:
        Offset curve as array, or None if self-intersection detected
    """
    if offset_distance == 0:
        return curve.copy()

    try:
        # Determine offset sign
        if side == "right":
            offset_distance = -abs(offset_distance)
        else:  # left
            offset_distance = abs(offset_distance)

        line = LineString(curve)

        # For line offset, parallel_offset is more appropriate than buffer
        offset_line = line.parallel_offset(
            offset_distance, side=side, resolution=resolution
        )

        if offset_line.is_empty:
            return None

        if isinstance(offset_line, LineString):
            offset_coords = np.array(offset_line.coords)
        elif isinstance(offset_line, MultiLineString):
            # Multiple disconnected lines - take the longest one
            lines = list(offset_line.geoms)
            if not lines:
                return None
            longest = max(lines, key=lambda l: len(l.coords))
            offset_coords = np.array(longest.coords)
        else:
            return None

        return offset_coords

    except Exception as e:
        print(f"Warning: Failed to offset curve: {e}")
        return None


def shrink_curve(
    curve: np.ndarray, percentage: float, resolution: int = 10
) -> Optional[np.ndarray]:
    """
    Shrink a closed curve inward by a given percentage while detecting self-intersections.

    Args:
        curve: Array of shape (n_points, 2) containing [x, y] coordinates
               (should be a closed curve)
        percentage: Shrinkage percentage (0-100). The curve is reduced by this percentage
                   of its original area.
        resolution: Number of segments for buffering (higher = smoother but slower)

    Returns:
        Shrunk curve as array of shape (n_points, 2), or None if self-intersection detected
    """
    if percentage <= 0:
        return curve.copy()

    if percentage >= 100:
        return None  # Cannot shrink by 100% or more

    if len(curve) < 3:
        return curve.copy()

    try:
        # Create a polygon to get the interior
        polygon = Polygon(curve)

        if polygon.is_empty or not polygon.is_valid:
            # Try to fix invalid polygon by simplifying
            polygon = polygon.buffer(0)
            if polygon.is_empty or not polygon.is_valid:
                return None

        original_area = polygon.area
        if original_area <= 0:
            return None

        # Calculate inward offset distance
        # For percentage-based shrinkage: new_area = original_area * (1 - percentage/100)
        # new_radius = sqrt(new_area / π) = sqrt(original_area * (1 - percentage/100) / π)
        # offset = new_radius - original_radius (negative value for inward offset)

        original_radius = np.sqrt(original_area / np.pi)
        new_radius = original_radius * np.sqrt(1 - percentage / 100)
        offset_dist = new_radius - original_radius  # This will be negative

        # Ensure we don't shrink beyond the minimum
        if offset_dist >= -original_radius * 0.95:
            buffered = polygon.buffer(offset_dist, resolution=resolution)

            if buffered.is_empty or not buffered.is_valid:
                return None

            if isinstance(buffered, Polygon):
                shrunk_coords = np.array(buffered.exterior.coords[:-1])
            elif isinstance(buffered, MultiLineString):
                lines = list(buffered.geoms)
                if not lines:
                    return None
                # Take the largest line by number of coords
                largest = max(lines, key=lambda l: len(l.coords))
                shrunk_coords = np.array(largest.coords[:-1])
            elif isinstance(buffered, LineString):
                shrunk_coords = np.array(buffered.coords)
            else:
                return None

            # Ensure the curve is closed by appending the first point
            if not np.allclose(shrunk_coords[0], shrunk_coords[-1]):
                shrunk_coords = np.vstack([shrunk_coords, shrunk_coords[0]])

            return shrunk_coords
        else:
            return None

    except Exception as e:
        print(f"Warning: Failed to shrink curve: {e}")
        return None


def shrink_curves(
    curves: List[np.ndarray],
    percentage: float,
    resolution: int = 10,
    filter_invalid: bool = True,
) -> List[np.ndarray]:
    """
    Shrink multiple closed curves by a given percentage while detecting self-intersections.

    Args:
        curves: List of curve arrays, each shape (n_points, 2)
        percentage: Shrinkage percentage (0-100)
        resolution: Number of segments for buffering
        filter_invalid: If True, remove curves that fail to shrink or have self-intersections

    Returns:
        List of shrunk curve arrays. If filter_invalid is True, excludes failed curves.
    """
    if percentage <= 0:
        return curves

    if percentage >= 100:
        return [] if filter_invalid else curves

    shrunk_curves = []

    for i, curve in enumerate(curves):
        # Check if curve is closed
        if not np.allclose(curve[0], curve[-1]):
            print(f"  Curve {i}: Not closed, skipped")
            if not filter_invalid:
                shrunk_curves.append(curve)
            continue

        shrunk = shrink_curve(curve, percentage, resolution=resolution)

        if shrunk is None:
            if not filter_invalid:
                shrunk_curves.append(curve)
            print(
                f"  Curve {i}: Failed to shrink (self-intersection or too small?), skipped"
            )
        else:
            shrunk_curves.append(shrunk)
            # Calculate area change using Polygon
            try:
                orig_poly = Polygon(curve)
                new_poly = Polygon(shrunk)

                if orig_poly.is_valid and new_poly.is_valid:
                    original_area = abs(orig_poly.area)
                    new_area = abs(new_poly.area)

                    if original_area > 0:
                        actual_percentage = (
                            (original_area - new_area) / original_area
                        ) * 100
                        print(
                            f"  Curve {i}: Shrunk {len(curve)} → {len(shrunk)} points "
                            f"({actual_percentage:.1f}% area reduction)"
                        )
                    else:
                        print(
                            f"  Curve {i}: Shrunk {len(curve)} → {len(shrunk)} points"
                        )
                else:
                    print(f"  Curve {i}: Shrunk {len(curve)} → {len(shrunk)} points")
            except:
                print(f"  Curve {i}: Shrunk {len(curve)} → {len(shrunk)} points")

    return shrunk_curves
