#!/usr/bin/env python3
"""Generate mesh profiles from point cloud data using Delaunay surface fitting.

Pipeline:
1. Load point cloud and fit Delaunay surface
2. Extract intersection curves from parallel planes
3. Create closed 3D profiles with flat bottoms
4. Generate 2D triangular meshes with variable sizing (fine at top, coarse at bottom)
5. Export to VTK for visualization in ParaView
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path for existing modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speed.filters.mesh.surface_fitting import (
    load_points,
    fit_surface,
    extract_intersection_curves,
    export_surface_vtk,
)
from speed.filters.curves.profile_generation import (
    create_depth_profiles,
    write_closed_curves_vtk,
)
from speed.filters.mesh.mesh_generation import generate_all_meshes


def main():
    """Main pipeline."""
    script_dir = Path(__file__).parent

    # Try both possible file locations
    file_options = [
        script_dir.parent
        / "cubit_python"
        / "CubitPython4SPEED"
        / "Files"
        / "rialba"
        / "DTMRialba5m_4Cubit_CUT_out.txt",
        Path(
            "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED/Files/rialba/DTMRialba5m_4Cubit_CUT_out.txt"
        ),
    ]

    filepath = None
    for fp in file_options:
        if fp.exists():
            filepath = fp
            break

    if filepath is None:
        print(
            f"Error: Could not find DTMRialba5m_4Cubit_CUT_out.txt in expected locations"
        )
        return

    print(f"Loading points from: {filepath}")
    z_factor = 5.0
    points = load_points(filepath, z_factor=z_factor)
    print(f"Loaded {len(points)} points (z scaled by {z_factor})")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # Fit surface using Delaunay triangulation
    print("\nFitting surface with Delaunay triangulation...")
    zi_grid, xi_grid, yi_grid, spl = fit_surface(points, grid_size=150)

    # Export surface to VTK
    print("\n" + "=" * 60)
    print("SURFACE EXPORT")
    print("=" * 60)

    surface_dir = script_dir / "surface"
    surface_dir.mkdir(exist_ok=True)
    surface_vtk = surface_dir / "fitted_surface.vtk"

    print(f"\nExporting surface to VTK...")
    export_surface_vtk(xi_grid, yi_grid, zi_grid, surface_vtk)
    print(f"Surface saved to: {surface_vtk}")

    # Define parallel planes
    print("\n" + "=" * 60)
    print("PARALLEL PLANES INTERSECTION")
    print("=" * 60)

    normal = [1, 0, 0]  # X-direction (vertical planes)
    x_min, x_max = xi_grid.min(), xi_grid.max()
    plane_step = (x_max - x_min) / 20  # 20 equally-spaced intervals

    plane_values = np.arange(x_min, x_max + plane_step / 2, plane_step)

    print(f"\nGenerating {len(plane_values)} parallel planes")
    print(f"  Normal direction: {normal}")
    print(f"  Plane range: [{plane_values.min():.2f}, {plane_values.max():.2f}]")
    print(f"  Step size: {plane_step:.2f}")

    # Extract intersections using piecewise linear interpolation
    print("Extracting intersection curves by sampling surface...")
    intersections = extract_intersection_curves(yi_grid, plane_values, spl)
    print(f"Found {len(intersections)} intersection curves")

    # Create depth profiles for meshing
    print("\n" + "=" * 60)
    print("DEPTH PROFILES")
    print("=" * 60)

    depth = (zi_grid.max() - zi_grid.min()) / 2  # Use half the Z-range as depth
    print(f"\nCreating depth profiles with depth = {depth:.2f}")

    profiles = create_depth_profiles(intersections, normal, plane_values, depth=depth)
    print(f"Created {len(profiles)} depth profiles")

    # Export intersection curves to VTK for visualization in ParaView
    print("\n" + "=" * 60)
    print("CURVE EXPORT")
    print("=" * 60)

    curves_dir = script_dir / "intersection_curves"
    curves_dir.mkdir(exist_ok=True)

    closed_vtk = curves_dir / "closed_curves.vtk"
    print(f"Exporting {len(profiles)} closed 3D curves to {closed_vtk.name}...")
    write_closed_curves_vtk(profiles, closed_vtk)
    print(f"Curves saved to: {curves_dir}")

    # Generate meshes for profiles
    print("\n" + "=" * 60)
    print("MESH GENERATION")
    print("=" * 60)

    mesh_dir = script_dir / "profile_meshes"
    # Choose a reasonable default mesh size based on overall surface extent
    x_extent = xi_grid.max() - xi_grid.min()
    y_extent = yi_grid.max() - yi_grid.min()
    mesh_size = (
        max(x_extent, y_extent) / 100.0 if (x_extent > 0 and y_extent > 0) else 1.0
    )

    print(f"\nGenerating 2D triangular meshes for {len(profiles)} profiles...")
    print(f"  Target mesh size: {mesh_size:.2f}")
    print(f"  Output directory: {mesh_dir}")

    mesh_results = generate_all_meshes(profiles, mesh_dir, mesh_size=mesh_size)

    print(f"\nMesh generation complete!")
    print(f"  Outputs saved to: {mesh_dir}")


if __name__ == "__main__":
    main()
