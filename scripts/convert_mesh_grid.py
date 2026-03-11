#!/usr/bin/env python3
"""Mesh grid conversion and scaling utility.

This script reads a mesh file, optionally scales the z-coordinates,
and exports to both VTU (for visualization) and native mesh format.

Usage:
    python convert_mesh_grid.py
"""

import sys
from pathlib import Path

# Add src to path to import speed modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))

from mesh.mesh_io import read_mesh, write_mesh_to_vtu


def main() -> None:
    """
    Convert and scale mesh files.

    Reads a mesh file, applies z-scaling, and exports to VTU and mesh formats.
    """
    # Folder of the test case relative to this script
    folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED"
    )
    file_name = Path("Meshfile_full.mesh")
    vtk_file_name = Path("output_full.vtu")

    # Output directory - current script location
    folder_out = Path(__file__).parent

    # Usage
    path = folder / file_name
    vtk_path = folder_out / vtk_file_name

    print(f"Reading mesh from: {path}")
    # Read mesh
    mesh = read_mesh(path)
    print(f"xmim x: {mesh.points[:, 0].min()}, xmax: {mesh.points[:, 0].max()}")
    print(f"ymin y: {mesh.points[:, 1].min()}, ymax: {mesh.points[:, 1].max()}")
    print(f"zmin z: {mesh.points[:, 2].min()}, zmax: {mesh.points[:, 2].max()}")

    # Write to VTU file
    print(f"Writing VTU file to: {vtk_path}")
    write_mesh_to_vtu(mesh, vtk_path)

    print("Done!")


if __name__ == "__main__":
    main()
