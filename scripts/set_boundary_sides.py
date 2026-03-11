import sys
from pathlib import Path
import numpy as np
import porepy as pp

# Add src to path to import speed modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "speed" / "filters"))

pygeon_meshio_path = Path(
    "/home/elle/Dropbox/Work/PresentazioniArticoli/2026/progetti/pygeon_meshio"
)
sys.path.insert(0, str(pygeon_meshio_path))

# from mesh.mesh_io import write_mesh_to_vtu, write_mesh
import meshio_reader as pg_meshio


def main() -> None:
    """
    Mark intersecting elements and remove them from the mesh.

    Reads a mesh file, checks for intersections with an STL surface,
    removes intersecting elements, and exports the result to VTU format.
    """
    # Folder of the test case relative to this script
    folder = Path(
        "/home/elle/Dropbox/Work/PresentazioniArticoli/progetti/cariplo/codes/speed-repo/scripts/rialba"
    )
    file_name = Path("rialba.mesh")

    # Usage
    path = folder / file_name

    # Read mesh
    sd = pg_meshio.import_speed_grid(path)[0]
    sd.compute_geometry()

    export = pp.Exporter(sd, "cubit_grid", folder)
    export.write_vtu()

    # mark the boundary faces not top
    bd_faces = sd.tags["domain_boundary_faces"]

    left = np.isclose(sd.face_centers[0], sd.nodes[0].min())
    right = np.isclose(sd.face_centers[0], sd.nodes[0].max())
    front = np.isclose(sd.face_centers[1], sd.nodes[1].min())
    back = np.isclose(sd.face_centers[1], sd.nodes[1].max())
    bottom = np.isclose(sd.face_centers[2], sd.nodes[2].min())

    other_faces = np.logical_or.reduce((left, right, front, back, bottom))
    other_faces = np.logical_and(other_faces, bd_faces)

    tag_name = "other_boundary_faces"
    sd.tags[tag_name] = other_faces

    pg_meshio.export_speed_grid(
        sd, folder / Path("rialba_with_boundaries.mesh"), tag_name
    )

    # i quad di bordono devono essere ordinati ccw ed in maniera tale che tla normale sia
    # rivolta verso l'esterno. DOVREI AVERLO FATTO

    # gli hexa invece devono avere prima i 4 punti in basso ordinati ccw, e poi i 4 punti
    # in alto ordinati sempre ccw

    # DA PROVARE SU UNA MESH COARSE


if __name__ == "__main__":
    main()
