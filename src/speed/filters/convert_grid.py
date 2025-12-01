from pathlib import Path
from functions import mesh_to_vtu


if __name__ == "__main__":
    # Folder of the test case relative to this script
    folder = Path("PLANE_WAVE/1_TEST_SEM")
    file_name = Path("1_TEST_SEM.mesh")
    vtk_file_name = Path("output.vtu")

    # Folder in home directory for tutorials
    folder_speed = Path("/home/user/speed-tutorials")

    # Usage
    path = folder_speed.joinpath(folder).joinpath(file_name)
    vtk_path = folder_speed.joinpath(folder).joinpath(vtk_file_name)
    mesh_to_vtu(path, vtk_path)
