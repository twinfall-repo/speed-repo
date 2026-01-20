"""DEM (Digital Elevation Model) processing utilities.

This module provides functions for loading, interpolating, and analyzing DEM data.
Includes support for gradient computation and resolution enhancement through
bivariate spline interpolation.
"""

from typing import Tuple
import numpy as np
from scipy.interpolate import RectBivariateSpline


def load_dem(
    filename: str, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DEM (Digital Elevation Model) file with x, y, z columns.

    Data is assumed to be ordered row-wise from top-left corner. The file
    should contain three columns (x, y, z) with one point per line. Data will
    be reshaped into 2D arrays of shape (ny, nx).

    Args:
        filename: Path to DEM text file
        nx: Number of columns in the DEM grid
        ny: Number of rows in the DEM grid

    Returns:
        Tuple of three 2D arrays:
        - x: 2D array of x-coordinates, shape (ny, nx)
        - y: 2D array of y-coordinates, shape (ny, nx)
        - z: 2D array of elevation values, shape (ny, nx)
    """
    data = np.loadtxt(filename)
    x = data[:, 0].reshape((ny, nx))
    y = data[:, 1].reshape((ny, nx))
    z = data[:, 2].reshape((ny, nx))
    return x, y, z


def interpolate_dem(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, factor: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate DEM to higher resolution using RectBivariateSpline.

    Uses scipy's RectBivariateSpline to smoothly interpolate elevation values
    to a finer grid. Automatically handles coordinate arrays that are not
    strictly increasing by flipping them. The resulting grid has dimensions
    (ny*factor, nx*factor).

    Args:
        x: 2D array of x-coordinates, shape (ny, nx)
        y: 2D array of y-coordinates, shape (ny, nx)
        z: 2D array of elevation values, shape (ny, nx)
        factor: Interpolation factor (default: 2 = 2x resolution increase)

    Returns:
        Tuple of three 2D arrays with interpolated values:
        - x_fine: Interpolated x-coordinates, shape (ny*factor, nx*factor)
        - y_fine: Interpolated y-coordinates, shape (ny*factor, nx*factor)
        - z_fine: Interpolated elevation values, shape (ny*factor, nx*factor)
    """
    ny, nx = z.shape
    xs = x[0, :]
    ys = y[:, 0]

    # Ensure xs and ys are strictly increasing for RectBivariateSpline
    if xs[0] > xs[-1]:
        xs = xs[::-1]
        z = z[:, ::-1]
    if ys[0] > ys[-1]:
        ys = ys[::-1]
        z = z[::-1, :]

    # Create interpolator
    interp = RectBivariateSpline(xs, ys, z)

    # Generate finer grid
    xs_fine = np.linspace(xs.min(), xs.max(), nx * factor)
    ys_fine = np.linspace(ys.min(), ys.max(), ny * factor)

    # Interpolate z values
    z_fine = interp(xs_fine, ys_fine, grid=True)
    x_fine, y_fine = np.meshgrid(xs_fine, ys_fine)

    return x_fine, y_fine, z_fine


def compute_gradient(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute gradient magnitude of the DEM surface.

    Calculates the magnitude of the gradient (slope) at each grid point using
    central differences. The gradient is computed as:
    grad_mag = sqrt((∂z/∂x)² + (∂z/∂y)²)

    Args:
        z: 2D array of elevation values, shape (ny, nx)
        dx: Grid spacing in x direction (meters)
        dy: Grid spacing in y direction (meters)

    Returns:
        grad_mag: 2D array of gradient magnitudes, same shape as z
    """
    dzdy, dzdx = np.gradient(z, dy, dx)
    grad_mag = np.sqrt(dzdx**2 + dzdy**2)
    return grad_mag
