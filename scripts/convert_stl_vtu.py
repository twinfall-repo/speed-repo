import sys
from pathlib import Path

# Add src to path to import speed modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))

from mesh.mesh_io import write_mesh_to_vtu, read_stl_mesh, write_stl_mesh


def main() -> None:
    """
    Convert an STL file to VTU format for visualization.

    Reads an STL file and exports it to VTU format.
    """
    # Folder of the test case relative to this script
    folder = Path(__file__).parent
    stl_file_name = Path("frattura_verticale.stl")

    # Usage
    stl_path = folder / stl_file_name

    # Read an stl mesh
    stl_mesh = read_stl_mesh(stl_path)

    ##### Specific to this case #####
    # Value for the shift for Rialba
    # x_shift = -1527760.8
    # y_shift = -5082534.8

    # shift for full rialbam, use this one
    x_shift = -1527863.3
    y_shift = -5082212.3

    # Apply shift to points
    stl_mesh.points[:, 0] += x_shift
    stl_mesh.points[:, 1] += y_shift

    # Write to VTU file (ASCII for easier inspection)
    write_mesh_to_vtu(stl_mesh, folder / Path("frattura_verticale.vtu"))

    # Since the coordinates were shifted, we may want to write back the STL as well
    write_stl_mesh(stl_mesh, folder / Path("frattura_verticale_shifted.stl"))


if __name__ == "__main__":
    main()
