import meshio
import numpy as np
from pathlib import Path

import sys

# Add src to path to import speed modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))

from mesh.mesh_io import write_mesh_to_vtu


def main():
    folder = Path(__file__).parent
    # ---- points on the vertical plane x = 0 ----
    points = np.array(
        [
            [0.0, -50.0, 75],  # 0
            [0.0, 50.0, 75],  # 1
            [0.0, 50.0, 150.0],  # 2
            [0.0, -50.0, 150.0],  # 3
        ]
    )
    z_scale = 5.0
    points[:, 2] = points[:, 2] * z_scale

    # ---- two triangles covering the square ----
    cells = [
        (
            "triangle",
            np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                ]
            ),
        )
    ]

    mesh = meshio.Mesh(
        points=points,
        cells=cells,
    )

    meshio.write(str(folder / "vertical_plane.stl"), mesh)

    # Export the STL surface for reference
    write_mesh_to_vtu(mesh, folder / Path("vertical_plane.vtu"))


if __name__ == "__main__":
    main()
