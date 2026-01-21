"""Profile and depth profile generation from intersection curves."""

import numpy as np


def create_depth_profiles(intersections, normal, plane_values, depth=10.0):
    """
    Create closed 3D shapes from piecewise linear intersection curves.

    Args:
        intersections: List of 2D piecewise linear curves (y, z points)
        normal: Normal direction of planes (must be [1, 0, 0])
        plane_values: X-plane position values
        depth: Depth to extrude downward from curves

    Returns:
        List of 3D profiles with piecewise linear top and bottom curves
    """
    profiles = []

    # Find global z minimum across all curves for consistent bottom
    z_min_global = float("inf")
    for curve_2d in intersections:
        z_min_global = min(z_min_global, curve_2d[:, 1].min())

    # Set the same bottom z for all curves
    z_bottom_global = z_min_global - depth

    for i, curve_2d in enumerate(intersections):
        # Curve is in Y-Z plane at X = plane_values[i]
        plane_x = plane_values[i]
        y_curve = curve_2d[:, 0]
        z_curve = curve_2d[:, 1]

        # Create 3D top curve (piecewise linear)
        x_top = np.full_like(y_curve, plane_x)
        top_curve = np.column_stack([x_top, y_curve, z_curve])

        # Create flat bottom curve at the SAME global z for all curves
        bottom_curve = np.column_stack(
            [x_top, y_curve, np.full_like(y_curve, z_bottom_global)]
        )

        profiles.append(
            {
                "top": top_curve,
                "bottom": bottom_curve,
                "plane_val": plane_x,
                "plane_coord": 0,  # X coordinate
            }
        )

    return profiles


def write_closed_curves_vtk(profiles, filename):
    """
    Export closed 3D curves (top + sides + bottom) to VTK format.

    Args:
        profiles: List of profile dictionaries with 'top' and 'bottom' curves
        filename: Output VTK file path
    """
    # Build closed polylines from profiles
    closed_curves_3d = []
    for profile in profiles:
        top = profile["top"]
        bottom = profile["bottom"]
        # Create closed curve: top -> bottom (reversed)
        closed_curve = np.vstack(
            [
                top,  # Top curve from surface intersection
                bottom[-1:],  # Connect to bottom at end
                bottom[::-1],  # Bottom curve (reversed)
                top[:1],  # Close back to start
            ]
        )
        closed_curves_3d.append(closed_curve)

    # Write to VTK
    all_points = []
    polylines = []

    for curve in closed_curves_3d:
        start_idx = len(all_points)
        all_points.extend(curve)
        num_points = len(curve)
        polylines.append((start_idx, num_points))

    all_points = np.array(all_points)

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Closed 3D Curves\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {len(all_points)} float\n")
        for pt in all_points:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        num_cells = len(polylines)
        total_cells_data = sum(n + 1 for _, n in polylines)
        f.write(f"LINES {num_cells} {total_cells_data}\n")
        for start_idx, num_points in polylines:
            f.write(f"{num_points}")
            for i in range(num_points):
                f.write(f" {start_idx + i}")
            f.write("\n")

        f.write(f"CELL_DATA {num_cells}\n")
        f.write("SCALARS curve_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for curve_id in range(num_cells):
            f.write(f"{curve_id}\n")
