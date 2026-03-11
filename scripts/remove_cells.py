import sys
from pathlib import Path

# Add src to path to import speed modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))

from mesh.mesh_io import read_mesh, write_mesh_to_vtu, read_stl_mesh, write_mesh
from mesh.mesh_operations import (
    initialize_dimension_field,
    mark_intersecting_hexahedra,
    mark_intersecting_quads,
    remove_3d_cells,
    remove_2d_cells,
)


def main() -> None:
    """
    Mark intersecting elements and remove them from the mesh.

    Reads a mesh file, checks for intersections with an STL surface,
    removes intersecting elements, and exports the result to VTU format.
    """
    # Folder of the test case relative to this script
    folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/codes/speed-repo/scripts/fine_grids_rialba"
    )
    file_name = Path("output_scaled.mesh")

    # Usage
    path = folder / file_name

    # Read mesh
    mesh = read_mesh(path)

    # Initialize dimension field for all cells
    initialize_dimension_field(mesh)

    # Read an stl mesh
    folder = Path(__file__).parent
    stl_path = folder / Path("frattura_verticale_shifted.stl")
    stl_mesh = read_stl_mesh(stl_path)

    # Write to VTU file (ASCII for easier inspection)
    write_mesh_to_vtu(mesh, folder / Path("output_original.vtu"))

    # Mark intersecting 3D cells
    mark_intersecting_hexahedra(mesh, stl_mesh)

    # Remove intersecting 3D cells (and add their faces to quad list)
    remove_3d_cells(mesh, new_quad_value=101)

    # Mark intersecting 2D quads (including newly created faces)
    mark_intersecting_quads(mesh, stl_mesh)

    # Write to VTU file (ASCII for easier inspection)
    write_mesh_to_vtu(mesh, folder / Path("output_test.vtu"))

    # Remove intersecting 2D quads
    remove_2d_cells(mesh)

    # Write to VTU file (ASCII for easier inspection)
    write_mesh_to_vtu(mesh, folder / Path("output_quads.vtu"))

    # Export mesh to Cubit format
    write_mesh(mesh, folder / Path("output_quads_cubit.mesh"))


if __name__ == "__main__":
    main()
