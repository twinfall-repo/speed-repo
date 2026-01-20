"""Export utilities for refinement curves to various CAD and visualization formats.

This module provides functions to export refinement curves to multiple file formats:
- XYZ: Simple point cloud format
- VTK: ParaView visualization format (polylines)
- PVD: ParaView Data Collection (multi-file format)
- SAT: ACIS geometry format for Cubit mesh refinement
"""

import os
from typing import List, Union
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.interpolate import griddata


def export_curves_xyz(
    curves: List[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    filename: Union[str, Path],
) -> None:
    """
    Export refinement curves to XYZ format (point cloud).

    Exports each curve as a sequence of XYZ points. Z values are interpolated
    using nearest-neighbor method from the DEM grid. Each curve is labeled with
    a comment line in the output file.

    Args:
        curves: List of curve arrays, each shape (n_points, 2)
                containing [x, y] coordinates
        x: 2D array of x-coordinates for interpolation
        y: 2D array of y-coordinates for interpolation
        z: 2D array of elevation values for interpolation
        filename: Output file path for XYZ format

    Returns:
        None
    """
    # Flatten the grid for interpolation
    points = np.column_stack([x.ravel(), y.ravel()])
    values = z.ravel()

    with open(filename, "w") as f:
        f.write("# Refinement curves in XYZ format\n")
        for curve_idx, curve in enumerate(curves):
            f.write(f"# Curve {curve_idx}\n")
            for point in curve:
                # Interpolate z value
                z_val = griddata(points, values, [point], method="nearest")[0]
                f.write(f"{point[0]:.6f} {point[1]:.6f} {z_val:.6f}\n")


def export_curves_vtk(
    curves: List[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    filename: Union[str, Path],
    z_scale: float = 1.0,
) -> None:
    """
    Export refinement curves to VTK format for ParaView visualization.

    Creates a VTK PolyData file containing all curves as polylines. Each curve
    is stored as a separate line with interpolated z values. Z values can be
    scaled for vertical exaggeration. The file includes cell data with curve
    IDs for visualization purposes.

    Args:
        curves: List of curve arrays, each shape (n_points, 2)
                containing [x, y] coordinates
        x: 2D array of x-coordinates for interpolation
        y: 2D array of y-coordinates for interpolation
        z: 2D array of elevation values for interpolation
        filename: Output VTK file path
        z_scale: Scale factor to multiply z values (default: 1.0 for no scaling)

    Returns:
        None
    """
    # Flatten grid for interpolation
    points_grid = np.column_stack([x.ravel(), y.ravel()])
    values_z = z.ravel()

    # Ensure target directory for VTK exists
    vtk_dir = os.path.dirname(str(filename))
    if vtk_dir and not os.path.exists(vtk_dir):
        os.makedirs(vtk_dir)

    # Collect all points and connectivity info
    all_points = []
    polylines = []  # List of (start_idx, num_points)

    for curve in curves:
        start_idx = len(all_points)

        for point_2d in curve:
            # Interpolate z value
            z_val = griddata(points_grid, values_z, [point_2d], method="nearest")[0]
            all_points.append([point_2d[0], point_2d[1], z_val * z_scale])

        num_points = len(curve)
        polylines.append((start_idx, num_points))

    all_points = np.array(all_points)

    # Write VTK file
    with open(filename, "w") as f:
        # VTK Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Refinement Curves\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {len(all_points)} float\n")
        for pt in all_points:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        # Cells (polylines)
        num_cells = len(polylines)
        total_cells_data = sum(n + 1 for _, n in polylines)  # +1 for count per line

        f.write(f"LINES {num_cells} {total_cells_data}\n")
        for start_idx, num_points in polylines:
            f.write(f"{num_points}")
            for i in range(num_points):
                f.write(f" {start_idx + i}")
            f.write("\n")

        # Cell data: color each curve differently
        f.write(f"CELL_DATA {num_cells}\n")
        f.write("SCALARS curve_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for curve_id in range(num_cells):
            f.write(f"{curve_id}\n")


def export_curves_sat(
    curves: List[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    output_dir: Union[str, Path],
    z_scale: float = 1.0,
) -> None:
    """Export refinement curves to SAT files (ACIS format) for mesh refinement.

    Creates individual SAT (ACIS geometry) files for each curve. Each file
    contains a closed spline curve suitable for defining mesh refinement zones
    in Cubit. Z values are scaled according to the z_scale parameter. The
    output directory is created if it doesn't exist.

    Args:
        curves: List of closed curve arrays, each shape (n_points, 2)
                containing [x, y] coordinates
        x: 2D array of x-coordinates for interpolation
        y: 2D array of y-coordinates for interpolation
        z: 2D array of elevation values for interpolation
        output_dir: Output directory path to save SAT files
        z_scale: Scale factor to multiply z values (default: 1.0 for no scaling)

    Returns:
        None
    """
    # Flatten grid for interpolation
    points_grid = np.column_stack([x.ravel(), y.ravel()])
    values_z = z.ravel()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tol = 1e-6

    for curve_id, curve in enumerate(curves):
        sat_filename = os.path.join(output_dir, f"refinement_curve_{curve_id:03d}.sat")

        # 3D points with scaled z
        points_3d = []
        for point_2d in curve:
            z_val = griddata(points_grid, values_z, [point_2d], method="nearest")[0]
            points_3d.append([point_2d[0], point_2d[1], z_val * z_scale])
        points_3d = np.array(points_3d)
        n_points = len(points_3d)

        if n_points < 2:
            continue

        now = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

        with open(sat_filename, "w") as f:
            # Header
            f.write("3100 0 1 0\n")
            f.write(f"13 Cubit 2022.11 14 ACIS 31.0.1 NT 24 {now}\n")
            f.write(f"1 {tol} 1e-10\n")
            f.write(
                "T @77 ADDADQCN9HAJ8XXXMVD484HDJ4343BP2NAKZJXPX2B3MMJ5E3DTCNE74K8T3FJ64BBPBJA647A6CP\n\n"
            )

            # Edge entity (simplified: param 0 to n_points-1)
            f.write(
                f"edge $1 -1 -1 $-1 $2 0 $3 {n_points - 1} $-1 $4 forward "
                "@7 unknown T\n"
            )
            # Start and end points
            p_start = points_3d[0]
            p_end = points_3d[-1]
            f.write(f"{p_start[0]:.0f} {p_start[1]:.0f} {p_start[2]:.0f}\n")
            f.write(f"{p_end[0]:.0f} {p_end[1]:.0f} {p_end[2]:.0f}\n")
            f.write("#\n\n")

            # Integer attribute for edge
            f.write("integer_attrib-name_attrib-gen-attrib\n")
            f.write("$-1 -1 $-1 $-1 $0\n")
            f.write("2 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1\n")
            f.write("@8 CUBIT_ID 2\n")
            f.write("#\n\n")

            # Two vertices
            f.write("vertex $5 -1 -1 $-1 $0 $6 #\n")
            f.write("vertex $7 -1 -1 $-1 $0 $8 #\n\n")

            # intcurve-curve entity as degree-1 nubs polyline
            f.write(
                "intcurve-curve $-1 -1 -1 $-1 forward "
                f"{{ exactcur 0 full nubs 1 open {n_points}\n"
            )

            # Knot vector (multiplicity 1, uniform)
            knot_str = ""
            for i in range(n_points):
                knot_str += f"{i} 1 "
            f.write(knot_str + "\n")

            # Control points
            for pt in points_3d:
                f.write(f"{pt[0]:.0f} {pt[1]:.0f} {pt[2]:.0f}\n")

            # Required boilerplate
            f.write("0\n")
            f.write("null_surface\n")
            f.write("null_surface\n")
            f.write("nullbs\n")
            f.write("nullbs\n")
            f.write("-1\n")
            f.write("-1\n")
            f.write("I I\n")
            f.write("0\n")
            f.write("0\n")
            f.write("0\n")
            f.write("-1\n")
            f.write(f"\tnone F F 0 F {n_points - 1} }} F 0 F {n_points - 1} #\n")
            f.write("\n")

            # Integer attribute for curve
            f.write("integer_attrib-name_attrib-gen-attrib\n")
            f.write("$-1 -1 $-1 $-1 $2\n")
            f.write("2 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1\n")
            f.write("@8 CUBIT_ID 3\n")
            f.write("#\n\n")

            # Start point
            f.write(
                f"point $-1 -1 -1 $-1 {p_start[0]:.0f} {p_start[1]:.0f} "
                f"{p_start[2]:.0f} #\n"
            )
            f.write("integer_attrib-name_attrib-gen-attrib\n")
            f.write("$-1 -1 $-1 $-1 $3\n")
            f.write("2 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1\n")
            f.write("@8 CUBIT_ID 4\n")
            f.write("#\n\n")

            # End point
            f.write(
                f"point $-1 -1 -1 $-1 {p_end[0]:.0f} {p_end[1]:.0f} {p_end[2]:.0f} #\n"
            )

            f.write("End-of-ACIS-data\n")


def export_curves_pvd(
    curves: List[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    filename: Union[str, Path],
    z_scale: float = 1.0,
) -> None:
    """
    Export refinement curves to PVD (ParaView Data Collection) format.

    Creates individual VTK files for each curve and a PVD manifest file for
    easy visualization in ParaView. Each curve can be independently displayed
    or hidden for analysis. Z values are scaled according to the z_scale
    parameter. The output directory is created if needed.

    Args:
        curves: List of curve arrays, each shape (n_points, 2)
                containing [x, y] coordinates
        x: 2D array of x-coordinates for interpolation
        y: 2D array of y-coordinates for interpolation
        z: 2D array of elevation values for interpolation
        filename: Output base filename (extension .pvd will be added if needed)
        z_scale: Scale factor to multiply z values (default: 1.0 for no scaling)

    Returns:
        None
    """
    # Convert to Path for easier handling
    filename_path = Path(filename)

    # Create output directory if needed
    base_dir = filename_path.parent
    if base_dir and not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    base_name = filename_path.stem

    # Flatten grid for interpolation
    points_grid = np.column_stack([x.ravel(), y.ravel()])
    values_z = z.ravel()

    # Create PVD file
    pvd_content = '<?xml version="1.0"?>\n'
    pvd_content += '<VTKFile type="Collection" version="0.1">\n'
    pvd_content += "  <Collection>\n"

    # Write each curve as separate VTK and add to collection
    for curve_id, curve in enumerate(curves):
        vtk_name = f"{base_name}_curve_{curve_id:03d}.vtk"
        vtk_path = base_dir / vtk_name

        # Create VTK for this curve
        with open(vtk_path, "w") as f:
            # Points with z values
            all_points = []
            for point_2d in curve:
                z_val = griddata(points_grid, values_z, [point_2d], method="nearest")[0]
                all_points.append([point_2d[0], point_2d[1], z_val * z_scale])

            all_points = np.array(all_points)

            # VTK Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"Refinement Curve {curve_id}\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # Points
            f.write(f"POINTS {len(all_points)} float\n")
            for pt in all_points:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

            # Single polyline
            f.write(f"LINES 1 {len(all_points) + 1}\n")
            f.write(f"{len(all_points)}")
            for i in range(len(all_points)):
                f.write(f" {i}")
            f.write("\n")

        # Add to PVD collection
        pvd_content += f'    <DataSet timestep="{curve_id}" file="{vtk_name}"/>\n'

    pvd_content += "  </Collection>\n"
    pvd_content += "</VTKFile>\n"

    # Write PVD file
    pvd_filename = filename_path.with_suffix(".pvd")
    with open(pvd_filename, "w") as f:
        f.write(pvd_content)
