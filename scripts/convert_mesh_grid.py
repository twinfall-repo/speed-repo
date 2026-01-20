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

from mesh.mesh_io import read_mesh, scale_mesh_z, write_mesh_to_vtu, write_mesh


def main() -> None:
    """
    Convert and scale mesh files.

    Reads a mesh file, applies z-scaling, and exports to VTU and mesh formats.
    """
    # Folder of the test case relative to this script
    folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED"
    )
    file_name = Path("Meshfile.mesh")
    vtk_file_name = Path("output.vtu")
    mesh_file_name = Path("output_scaled.mesh")
    z_factor = 5.0  # Scale z coordinates by this factor

    # Output directory - current script location
    folder_out = Path(__file__).parent

    # Usage
    path = folder / file_name
    vtk_path = folder_out / vtk_file_name
    mesh_path = folder_out / mesh_file_name

    print(f"Reading mesh from: {path}")
    # Read mesh
    mesh = read_mesh(path)
    print(f"  Nodes: {len(mesh.points)}")

    # Apply z-scaling
    print(f"Scaling z-coordinates by factor: {z_factor}")
    mesh_scaled = scale_mesh_z(mesh, z_factor=z_factor)

    # Write to VTU file
    print(f"Writing VTU file to: {vtk_path}")
    write_mesh_to_vtu(mesh_scaled, vtk_path)

    # Write back to native mesh format
    print(f"Writing mesh file to: {mesh_path}")
    write_mesh(mesh_scaled, mesh_path)

    print("Done!")


if __name__ == "__main__":
    main()
