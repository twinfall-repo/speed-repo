import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))

from mesh.surface_fitting import save_as_xyz
from curves.curve_export import (
    export_curves_xyz,
    export_curves_vtk,
    export_curves_pvd,
    export_curves_sat,
)


def export_curve_version(curves, x, y, z, folder, version_name, z_scale=1.0) -> None:
    """Export curves to all formats in a versioned subdirectory."""
    version_dir = Path(folder) / f"refinement_curves_{version_name}"
    version_dir.mkdir(parents=True, exist_ok=True)

    export_curves_sat(curves, x, y, z, version_dir / "refinement_curves_sat")
    print(
        f"   {version_name.capitalize()}: SAT files → {version_dir / 'refinement_curves_sat'}"
    )

    export_curves_xyz(curves, x, y, z, version_dir / "refinement_curves.xyz")
    print(
        f"   {version_name.capitalize()}: XYZ file → {version_dir / 'refinement_curves.xyz'}"
    )

    export_curves_vtk(
        curves, x, y, z, version_dir / "refinement_curves.vtk", z_scale=z_scale
    )
    print(
        f"   {version_name.capitalize()}: VTK file → {version_dir / 'refinement_curves.vtk'}"
    )

    export_curves_pvd(
        curves, x, y, z, version_dir / "refinement_curves", z_scale=z_scale
    )
    print(
        f"   {version_name.capitalize()}: PVD file → {version_dir / 'refinement_curves.pvd'}"
    )


def build_grid_from_points(points: np.ndarray, decimals: int = 6):
    """Build structured x,y,z grids from scattered points assumed on a grid."""
    x_vals = np.unique(np.round(points[:, 0], decimals))
    y_vals = np.unique(np.round(points[:, 1], decimals))

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_grid = np.full_like(x_grid, fill_value=np.nan, dtype=float)

    x_index = {val: i for i, val in enumerate(x_vals)}
    y_index = {val: i for i, val in enumerate(y_vals)}

    for x, y, z in points:
        xi = x_index.get(round(x, decimals))
        yi = y_index.get(round(y, decimals))
        if xi is None or yi is None:
            continue
        z_grid[yi, xi] = z

    if np.isnan(z_grid).any():
        raise ValueError("Grid reconstruction failed: missing z values.")

    return x_grid, y_grid, z_grid


def main():
    main_folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED/Files"
    )
    # folder = main_folder / "rialba"
    # dtm = "DTMRialba5m_4Cubit_CUT.txt"

    folder = main_folder / "rialba_full"
    dtm = "DTMRialba5m_4Cubit_noCUT_Yflipped.txt"

    # shift for rialba, use the full case
    # center = [1527760.8, 5082534.8]

    # shift for full rialba
    center = [1527863.3, 5082212.3]

    dem_pts = np.loadtxt(os.path.join(folder, dtm), comments="#")
    print("Shape of DEM points:", dem_pts.shape[0], dem_pts.shape[1] / 3)

    dem_pts = np.flipud(dem_pts).reshape(-1, 3)
    dem_pts[:, 0] -= center[0]
    dem_pts[:, 1] -= center[1]

    x_grid, y_grid, z_grid = build_grid_from_points(dem_pts)

    print(f"X: min={dem_pts[:, 0].min():.3f}, max={dem_pts[:, 0].max():.3f}")
    print(f"Y: min={dem_pts[:, 1].min():.3f}, max={dem_pts[:, 1].max():.3f}")

    dtm_xyz = dtm[:-4] + "_out.xyz"
    save_as_xyz(dem_pts, os.path.join(folder, dtm_xyz))

    dtm_txt = dtm[:-4] + "_out.txt"
    np.savetxt(
        os.path.join(folder, dtm_txt),
        dem_pts,
        fmt="%.3f",
        delimiter=" ",
    )

    z_poly = np.amax(dem_pts[:, 2])  # just for visualization

    poly = "Polygon4FinerZmesh.txt"
    poly_pts = np.loadtxt(os.path.join(folder, poly), comments="#")
    poly_pts = np.hstack((poly_pts, np.full((poly_pts.shape[0], 1), z_poly)))

    poly_pts[:, 0] -= center[0]
    poly_pts[:, 1] -= center[1]

    poly_xyz = poly[:-4] + "_centered.xyz"
    save_as_xyz(poly_pts, os.path.join(folder, poly_xyz))

    poly_out = poly[:-4] + "_centered.txt"
    np.savetxt(
        os.path.join(folder, poly_out),
        poly_pts,
        fmt="%.3f",
        delimiter=" ",
    )

    # Ensure polygon curve is closed
    poly_curve = poly_pts[:, :2]

    # Always close the curve by ensuring first and last points are identical
    if not np.allclose(poly_curve[0], poly_curve[-1], atol=1e-6):
        poly_curve = np.vstack([poly_curve, poly_curve[0]])

    # Double-check closure
    distance = np.linalg.norm(poly_curve[-1] - poly_curve[0])
    print(f"\nPolygon curve closure check: {distance:.2e} (should be ~0)")
    print(f"  First point: {poly_curve[0]}")
    print(f"  Last point:  {poly_curve[-1]}")
    print(f"  Number of points: {len(poly_curve)}")

    export_curve_version(
        [poly_curve],
        x_grid,
        y_grid,
        z_grid,
        folder,
        "poly",
        z_scale=1.0,
    )


if __name__ == "__main__":
    main()
