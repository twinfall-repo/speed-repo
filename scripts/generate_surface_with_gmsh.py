import sys
import gmsh
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))
from curves.dem_processing import load_dem


def main():
    commond_folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED/Files"
    )

    # Parameters for the rialba case
    folder = commond_folder / "rialba"
    file_name = "DTMRialba5m_4Cubit_CUT_out.txt"
    nx, ny = 54, 54
    depth = 100
    surf_name = "rialba_surface.step"

    # Parameter for the rialba full case
    # folder = commond_folder / "rialba_full"
    # file_name = "DTMRialba5m_4Cubit_noCUT_Yflipped_resampled.txt"
    # nx, ny = 300, 300  # nx=columns (x direction), ny=rows (y direction)
    # depth = 100
    # surf_name = "rialba_full_surface.step"

    x_vals, y_vals, z_vals = load_dem(folder / file_name, nx=nx, ny=ny)

    print(f"Loaded DEM shape: {x_vals.shape}")

    gmsh.initialize()
    gmsh.model.add("bspline_surface")

    # Get grid size from the arrays
    nx, ny = x_vals.shape
    print(f"Using grid size: nx={nx}, ny={ny}")

    # === TOP SURFACE ===
    point_tags_top = []

    for i in range(nx):
        row = []
        for j in range(ny):
            x = x_vals[i, j]
            y = y_vals[i, j]
            z = z_vals[i, j]
            tag = gmsh.model.occ.addPoint(x, y, z)
            row.append(tag)
        point_tags_top.append(row)

    flat_points_top = [tag for row in point_tags_top for tag in row]
    gmsh.model.occ.addBSplineSurface(flat_points_top, nx)

    # === BOTTOM SURFACE (FLAT) ===
    z_bottom = z_vals.min() - depth
    point_tags_bottom = []

    for i in range(nx):
        row = []
        for j in range(ny):
            x = x_vals[i, j]
            y = y_vals[i, j]
            z = z_bottom
            tag = gmsh.model.occ.addPoint(x, y, z)
            row.append(tag)
        point_tags_bottom.append(row)

    flat_points_bottom = [tag for row in point_tags_bottom for tag in row]
    gmsh.model.occ.addBSplineSurface(flat_points_bottom, nx)

    gmsh.model.occ.synchronize()

    gmsh.write(str(folder / surf_name))

    gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    main()
