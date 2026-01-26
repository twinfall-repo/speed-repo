"""Surface fitting functions using Delaunay triangulation or smooth interpolants."""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def load_points(filepath, z_factor=1.0, skip_header=0):
    """Load point cloud from text file and scale z coordinates.

    Args:
        filepath: Path to the point cloud file
        z_factor: Scaling factor for z coordinates (default: 1.0)
        skip_header: Number of header lines to skip in the file (default: 0)
    Returns:
        points: (N, 3) array with scaled z values
    """
    points = np.loadtxt(filepath, skiprows=skip_header)
    points[:, 2] *= z_factor
    return points


def inner_rectangle_from_points(pts, trim_fraction=0.01):
    """
    Compute an axis-aligned rectangle contained inside the point cloud footprint.

    The rectangle is defined by trimming a fraction of extreme points along x and y.

    Args:
        pts: (N, 3) array of [x, y, z] coordinates
        trim_fraction: fraction to trim from each tail (e.g., 0.01 trims 1% low/high)

    Returns:
        (x_min, x_max, y_min, y_max): rectangle limits
    """
    if pts.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0

    trim_fraction = np.clip(trim_fraction, 0.0, 0.49)
    x = pts[:, 0]
    y = pts[:, 1]
    x_min = np.quantile(x, trim_fraction)
    x_max = np.quantile(x, 1.0 - trim_fraction)
    y_min = np.quantile(y, trim_fraction)
    y_max = np.quantile(y, 1.0 - trim_fraction)
    return x_min, x_max, y_min, y_max


def filter_points_in_rectangle(pts, rect):
    """
    Keep only points whose (x, y) lie inside the given rectangle.

    Args:
        pts: (N, 3) array of [x, y, z] coordinates
        rect: tuple (x_min, x_max, y_min, y_max)

    Returns:
        filtered_pts: points inside the rectangle
    """
    x_min, x_max, y_min, y_max = rect
    mask = (
        (pts[:, 0] >= x_min)
        & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min)
        & (pts[:, 1] <= y_max)
    )
    return pts[mask]


def save_as_xyz(pts, filepath, tag=1):
    """Save points as .xys file.
    Args:
        pts: (N, 3) array of [x, y, z] coordinates
        filepath: Output file path
    """
    with open(filepath, "w") as f:
        f.write(f"{pts.shape[0]}\n\n")
        for x, y, z in pts:
            f.write(f"{tag} {x} {y} {z}\n")


def center_xy(pts, z_scale=1.0):
    """Center the points in (0, 0) for x and y, and scale z.

    Args:
        pts: (N, 3) array of [x, y, z] coordinates
        z_scale: Scaling factor for z coordinates (default: 1.0)
    Returns:
        centered_pts: (N, 3) array of centered and scaled points
    """
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    centered_pts = pts.copy()
    centered_pts[:, 0] -= x_center
    centered_pts[:, 1] -= y_center
    centered_pts[:, 2] *= z_scale

    return centered_pts


def downsample_points(pts, target_count=None, grid_size=None):
    """
    Downsample point cloud using spatial binning (grid-based averaging).

    Either specify target_count or grid_size (grid_size takes precedence).

    Args:
        pts: (N, 3) array of [x, y, z] coordinates
        target_count: approximate target number of points after downsampling
        grid_size: number of bins along each dimension (creates grid_size^2 cells)

    Returns:
        downsampled_pts: reduced point cloud
    """
    if pts.shape[0] == 0:
        return pts

    if grid_size is None:
        if target_count is None:
            return pts
        # Estimate grid_size from target_count: grid_size^2 ~ target_count
        grid_size = max(2, int(np.sqrt(target_count)))

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Create bins
    x_bins = np.linspace(x.min(), x.max(), grid_size + 1)
    y_bins = np.linspace(y.min(), y.max(), grid_size + 1)

    # Digitize points into bins
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    x_idx = np.clip(x_idx, 0, grid_size - 1)
    y_idx = np.clip(y_idx, 0, grid_size - 1)

    # Average points in each bin
    downsampled = []
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x_idx == i) & (y_idx == j)
            if np.any(mask):
                downsampled.append([x[mask].mean(), y[mask].mean(), z[mask].mean()])

    return np.array(downsampled) if downsampled else pts


def fit_surface(pts):
    """
    Create a triangulated surface mesh directly from scattered points using Delaunay triangulation.

    This approach naturally handles non-rectangular footprints without boundary artifacts
    since the mesh only exists where there is data.

    Args:
        pts: (N, 3) array of [x, y, z] coordinates

    Returns:
        pts: (N, 3) array of points
        triangles: (M, 3) array of triangle vertex indices
        tri: Delaunay triangulation object for interpolation
    """
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Create Delaunay triangulation in XY plane
    tri = Delaunay(np.column_stack([x, y]))
    
    # Create interpolator using the triangulation
    interpolator = LinearNDInterpolator(tri, z)
    
    return pts, tri.simplices, interpolator


def extract_intersection_curves(pts, interpolator, plane_values, num_samples=200):
    """
    Extract intersection curves between X-planes and triangulated surface.

    Samples the interpolated surface at each plane position to get (y, z) points.

    Args:
        pts: (N, 3) array of surface points
        interpolator: Interpolation function from fit_surface
        plane_values: List of X-plane positions
        num_samples: Number of Y samples for each curve

    Returns:
        List of curves (each curve is array of [y, z] points)
    """
    intersections = []
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    y_vec = np.linspace(y_min, y_max, num_samples)

    for plane_x in plane_values:
        try:
            # Create points along this X-plane
            x_vec = np.full_like(y_vec, plane_x)
            
            # Interpolate z-values
            z_vals = interpolator(x_vec, y_vec)

            # Filter out NaN values (outside convex hull)
            valid = np.isfinite(z_vals)
            if not np.any(valid):
                print(f"Warning: No valid values for plane at x={plane_x:.2f}")
                continue

            # Create curve from valid points
            curve_points = np.column_stack([y_vec[valid], z_vals[valid]])
            intersections.append(curve_points)
        except Exception as e:
            print(f"Warning: Could not sample surface for plane at x={plane_x:.2f}: {e}")
            continue

    return intersections


def export_surface_mesh_vtk(pts, triangles, filename):
    """
    Export triangulated surface mesh as VTK unstructured grid (POLYDATA).

    Args:
        pts: (N, 3) array of [x, y, z] coordinates
        triangles: (M, 3) array of triangle vertex indices
        filename: Output VTK file path
    """
    n_points = len(pts)
    n_triangles = len(triangles)

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Triangulated surface from Delaunay\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_points} float\n")

        # Write points
        for pt in pts:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        # Write triangles
        f.write(f"\nPOLYGONS {n_triangles} {n_triangles * 4}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    print(f"Exported {n_points} points and {n_triangles} triangles to {filename}")


# Keep old function for backward compatibility but mark deprecated
def export_surface_vtk(xi_grid, yi_grid, zi_grid, filename):
    """
    Export the fitted surface as a VTK structured grid for visualization in ParaView.
    
    DEPRECATED: Use export_surface_mesh_vtk for triangulated meshes instead.

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
