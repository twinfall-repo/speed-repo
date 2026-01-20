#!/usr/bin/env python3
"""Main pipeline for DEM refinement curve extraction.

This script orchestrates the complete workflow for extracting mesh refinement
curves from Digital Elevation Model (DEM) data. The extracted curves identify
high-gradient regions suitable for mesh refinement in Cubit.

Workflow:
1. Load DEM from file
2. Interpolate to higher resolution
3. Compute gradient magnitude
4. Extract refinement curves at high-gradient threshold
5. Close curves to form complete loops
6. Export to multiple formats (XYZ, VTK, PVD, SAT)
7. Display debug visualization

Usage:
    python extract_refinement_curves.py
"""

import sys
from pathlib import Path

# Add src to path to import speed modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))

from curves.dem_processing import load_dem, interpolate_dem, compute_gradient
from curves.curve_extraction import extract_refinement_curves, close_curves_to_boundary
from curves.curve_export import (
    export_curves_xyz,
    export_curves_vtk,
    export_curves_pvd,
    export_curves_sat,
)
from curves.curve_enlargement import enlarge_curves, shrink_curves
from curves.curve_visualization import debug_visualization, visualize_enlarged_curves


def export_curve_version(
    curves, x, y, z, folder, version_name, z_scale=1.0
) -> None:
    """
    Export curves to all formats in a versioned subdirectory.

    Args:
        curves: List of curve arrays to export
        x: 2D array of x-coordinates
        y: 2D array of y-coordinates
        z: 2D array of elevation values
        folder: Base output folder
        version_name: Name of the version (e.g., 'original', 'enlarged', 'shrunk')
        z_scale: Z scaling factor for VTK/PVD export
    """
    version_dir = folder / f"refinement_curves_{version_name}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Export to all formats
    export_curves_sat(curves, x, y, z, version_dir / "refinement_curves_sat")
    print(f"   {version_name.capitalize()}: SAT files → {version_dir / 'refinement_curves_sat'}")

    export_curves_xyz(curves, x, y, z, version_dir / "refinement_curves.xyz")
    print(f"   {version_name.capitalize()}: XYZ file → {version_dir / 'refinement_curves.xyz'}")

    export_curves_vtk(
        curves, x, y, z, version_dir / "refinement_curves.vtk", z_scale=z_scale
    )
    print(f"   {version_name.capitalize()}: VTK file → {version_dir / 'refinement_curves.vtk'}")

    export_curves_pvd(
        curves, x, y, z, version_dir / "refinement_curves", z_scale=z_scale
    )
    print(f"   {version_name.capitalize()}: PVD file → {version_dir / 'refinement_curves.pvd'}")



def main() -> None:
    """
    Main processing pipeline for DEM refinement curve extraction.

    Orchestrates the complete workflow:
    1. Load DEM from file
    2. Interpolate to higher resolution
    3. Compute gradient magnitude
    4. Extract refinement curves at high-gradient threshold
    5. Close curves to form complete loops
    6. Export to multiple formats (XYZ, VTK, PVD, SAT)
    7. Display debug visualization

    Configuration parameters can be modified in the function body for different DEM files
    or processing requirements.

    Returns:
        None
    """

    # Configuration
    # folder = Path("/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED/Files/test")
    # file_name = "DTM5x5points_out.txt"
    # nx, ny = 5, 5
    # gradient_percentile = 10
    # z_scale = 1.0
    # simplify = False
    # enlarge_percentage = 0.0  # No enlargement
    # shrink_percentage = 0.0   # No shrinking

    folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED/Files/rialba"
    )
    file_name = "DTMRialba5m_4Cubit_CUT_out.txt"
    nx, ny = 54, 54
    gradient_percentile = 90
    z_scale = 5.0
    simplify = True
    enlarge_percentage = (
        0.0  # Enlargement percentage (e.g., 10.0 for 10% area increase)
    )
    shrink_percentage = 0.0  # Shrinkage percentage (e.g., 10.0 for 10% area reduction)

    interp_factor = 2

    print("=" * 60)
    print("DEM Refinement Curve Extraction")
    print("=" * 60)

    # Load data
    print(f"\n1. Loading DEM from: {file_name}")
    x, y, z = load_dem(folder / file_name, nx=nx, ny=ny)
    print(f"   Original grid: {z.shape}")

    # Compute grid spacing
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    print(f"   Grid spacing: dx={dx:.2f}, dy={dy:.2f}")
    print(f"   Z range: {z.min():.2f} to {z.max():.2f}")

    # Interpolate
    print(f"\n2. Interpolating to {interp_factor}x resolution...")
    x_fine, y_fine, z_fine = interpolate_dem(x, y, z, factor=interp_factor)
    print(f"   Interpolated grid: {z_fine.shape}")
    dx_fine = dx / interp_factor
    dy_fine = dy / interp_factor

    # Compute gradients
    print(f"\n3. Computing gradient magnitude...")
    grad_mag = compute_gradient(z_fine, dx_fine, dy_fine)
    print(f"   Gradient range: {grad_mag.min():.6f} to {grad_mag.max():.6f}")

    # Extract curves
    print(f"\n4. Extracting refinement curves (percentile={gradient_percentile})...")
    curves, threshold = extract_refinement_curves(
        x_fine,
        y_fine,
        grad_mag,
        percentile=gradient_percentile,
        min_points=10,
        smooth=False,
        simplify=simplify,
        simplify_epsilon=8.0,
    )
    print(f"   Threshold: {threshold:.6f}")
    print(f"   Number of curves: {len(curves)}")
    for i, curve in enumerate(curves):
        print(f"     Curve {i}: {len(curve)} points")

    # Close curves to domain boundary
    print(f"\n4b. Closing curves to domain boundary...")
    curves = close_curves_to_boundary(curves, x_fine, y_fine)
    print(f"   All curves closed")

    # Enlarge or shrink curves if requested
    curves_export = curves
    if enlarge_percentage > 0:
        print(f"\n4c. Enlarging curves by {enlarge_percentage}%...")
        curves_export = enlarge_curves(curves, enlarge_percentage, resolution=10)
        print(
            f"   Enlargement complete: {len(curves)} original → {len(curves_export)} enlarged"
        )
    elif shrink_percentage > 0:
        print(f"\n4c. Shrinking curves by {shrink_percentage}%...")
        curves_export = shrink_curves(curves, shrink_percentage, resolution=10)
        print(
            f"   Shrinking complete: {len(curves)} original → {len(curves_export)} shrunk"
        )

    # Export curves
    print(f"\n5. Exporting curves...")
    
    # Always export original curves
    export_curve_version(curves, x_fine, y_fine, z_fine, folder, "original", z_scale)
    
    # Export modified version if enlargement or shrinking was applied
    if enlarge_percentage > 0:
        export_curve_version(curves_export, x_fine, y_fine, z_fine, folder, "enlarged", z_scale)
    elif shrink_percentage > 0:
        export_curve_version(curves_export, x_fine, y_fine, z_fine, folder, "shrunk", z_scale)

    # Visualize
    print(f"\n6. Generating visualization...")
    debug_visualization(x_fine, y_fine, z_fine, grad_mag, curves, threshold)

    # Visualize enlargement or shrinking comparison if applied
    if enlarge_percentage > 0 and len(curves_export) > 0:
        print(f"\n6b. Generating enlargement comparison...")
        visualize_enlarged_curves(
            x_fine,
            y_fine,
            z_fine,
            grad_mag,
            curves,
            curves_export,
            threshold,
            enlarge_percentage,
        )
    elif shrink_percentage > 0 and len(curves_export) > 0:
        print(f"\n6b. Generating shrinkage comparison...")
        visualize_enlarged_curves(
            x_fine,
            y_fine,
            z_fine,
            grad_mag,
            curves,
            curves_export,
            threshold,
            -shrink_percentage,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
