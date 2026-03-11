from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import griddata


# Add src to path for existing modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speed.filters.mesh.surface_fitting import center_xy


def main():
    script_dir = Path(__file__).parent
    dem_folder = (
        script_dir.parent / "../../cubit_python/CubitPython4SPEED/Files/rialba_full"
    )

    dem_file = dem_folder / "DTMRialba5m_4Cubit_noCUT_Yflipped.txt"
    new_dtm_file = dem_folder / "DTMRialba5m_4Cubit_noCUT_Yflipped_resampled.txt"

    nx = 300
    ny = 300

    dem_pts = np.loadtxt(dem_file, comments="#")
    print("Shape of DEM points:", dem_pts.shape[0], dem_pts.shape[1] / 3)

    dem_pts = np.flipud(dem_pts).reshape(-1, 3)
    dem_pts = center_xy(dem_pts)

    x, y, z = dem_pts[:, 0], dem_pts[:, 1], dem_pts[:, 2]

    # Create a grid of control points for the BSpline surface
    x_grid = np.linspace(min(x), max(x), nx)
    y_grid = np.linspace(min(y), max(y), ny)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Interpolate scattered data onto the grid
    zz = griddata((x, y), z, (xx, yy), method="linear")
    new_pts = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

    np.savetxt(new_dtm_file, new_pts)

    # After call the file main.py in the rialba folder


if __name__ == "__main__":
    main()
