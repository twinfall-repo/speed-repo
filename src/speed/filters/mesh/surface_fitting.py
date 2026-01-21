"""Surface fitting functions using Delaunay triangulation."""

import numpy as np


def load_points(filepath, z_factor=5.0):
    """Load point cloud from text file and scale z coordinates.

    Args:
        filepath: Path to the point cloud file
        z_factor: Scaling factor for z coordinates (default: 5.0)

    Returns:
        points: (N, 3) array with scaled z values
    """
    points = np.loadtxt(filepath)
    points[:, 2] *= z_factor
    return points


def fit_surface(points, grid_size=150):
    """
    Create a piecewise linear surface from scattered points using Delaunay triangulation.

    Args:
        points: (N, 3) array of [x, y, z] coordinates
        grid_size: resolution of the output grid

    Returns:
        zi_grid, xi_grid, yi_grid, and interpolator function for evaluation
    """
    from scipy.interpolate import LinearNDInterpolator

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create regular grid for surface evaluation
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Create Delaunay-based linear interpolator
    interpolator = LinearNDInterpolator(list(zip(x, y)), z)

    # Evaluate on grid
    zi_grid = interpolator(xi_grid, yi_grid)

    return zi_grid, xi_grid, yi_grid, interpolator


def extract_intersection_curves(yi_grid, plane_values, spl):
    """
    Extract intersection curves between X-planes and surface as piecewise linear curves.

    Samples the fitted surface at each plane position to get (y, z) points.
    The resulting curves are piecewise linear (straight segments connecting sampled points).

    Args:
        yi_grid: Y coordinate grid
        plane_values: List of X-plane positions
        spl: Fitted surface spline for sampling

    Returns:
        List of piecewise linear curves (each curve is array of [y, z] points)
    """
    intersections = []
    y_vec = yi_grid[:, 0]  # Y coordinates from grid

    for plane_x in plane_values:
        # Sample z-values from the fitted surface at this X position
        try:
            z_vals = np.asarray(spl(np.array([plane_x]), y_vec)).reshape(-1)

            # Check for invalid values
            if np.any(~np.isfinite(z_vals)):
                print(f"Warning: Invalid values for plane at x={plane_x:.2f}")
                continue

            # Create piecewise linear curve from sampled points
            curve_points = np.column_stack([y_vec, z_vals])
            intersections.append(curve_points)
        except Exception as e:
            print(
                f"Warning: Could not sample surface for plane at x={plane_x:.2f}: {e}"
            )
            continue

    return intersections


def export_surface_vtk(xi_grid, yi_grid, zi_grid, filename):
    """
    Export the fitted surface as a VTK structured grid for visualization in ParaView.

    Args:
        xi_grid: X coordinates mesh grid (ny, nx)
        yi_grid: Y coordinates mesh grid (ny, nx)
        zi_grid: Z coordinates mesh grid (ny, nx)
        filename: Output VTK file path
    """
    ny, nx = xi_grid.shape

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Surface from Delaunay triangulation\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"POINTS {nx * ny} float\n")

        # Write points (flatten in correct order for structured grid)
        for j in range(ny):
            for i in range(nx):
                x = xi_grid[j, i]
                y = yi_grid[j, i]
                z = zi_grid[j, i] if np.isfinite(zi_grid[j, i]) else 0.0
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
