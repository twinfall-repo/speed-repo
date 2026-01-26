import meshio
import numpy as np
from pathlib import Path


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


if __name__ == "__main__":
    main()
