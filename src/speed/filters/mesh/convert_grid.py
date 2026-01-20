from pathlib import Path
import os
from .mesh_io import read_mesh, scale_mesh_z, write_mesh_to_vtu, write_mesh


if __name__ == "__main__":
    # Folder of the test case relative to this script
    folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/cubit_python/CubitPython4SPEED"
    )
    file_name = Path("Meshfile.mesh")
    vtk_file_name = Path("output.vtu")
    mesh_file_name = Path("output_scaled.mesh")
    z_factor = 5.0  # Scale z coordinates by this factor

    # Folder in home directory for tutorials
    folder_out = Path(os.path.dirname(os.path.abspath(__file__)))

    # Usage
    path = folder.joinpath(file_name)
    vtk_path = folder_out.joinpath(vtk_file_name)
    mesh_path = folder_out.joinpath(mesh_file_name)

    # Read mesh
    mesh = read_mesh(path)

    # Apply z-scaling
    mesh_scaled = scale_mesh_z(mesh, z_factor=z_factor)

    # Write to VTU file
    write_mesh_to_vtu(mesh_scaled, vtk_path)

    # Write back to native mesh format
    write_mesh(mesh_scaled, mesh_path)
