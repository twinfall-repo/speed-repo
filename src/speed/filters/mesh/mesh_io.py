"""Mesh I/O utilities for reading, writing, and transforming mesh files.

This module provides functions for working with mesh files:
- Reading mesh files in native format
- Writing mesh files to native and VTU formats
- Scaling mesh coordinates
"""

from pathlib import Path
from typing import Union
import meshio
import numpy as np


def read_mesh(filename: Union[str, Path]) -> meshio.Mesh:
    """Read a mesh file.

    Args:
        filename: Path to mesh file.

    Returns:
        meshio.Mesh object.
    """
    points = []
    cells = {"hexahedron": [], "quad": []}
    cells_data = {"hexahedron": [], "quad": []}

    with open(filename, "r") as f:
        lines = f.readlines()

    # First line: header
    header = lines[0].split()
    num_nodes = int(header[0])

    # Read nodes
    for i in range(1, num_nodes + 1):
        parts = lines[i].split()
        # Ignore node ID (parts[0])
        x, y, z = map(float, parts[1:4])
        points.append([x, y, z])

    # Read elements
    for i in range(num_nodes + 1, len(lines)):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        elem_type = parts[2]
        node_ids = [int(n) - 1 for n in parts[3:]]  # zero-based indexing
        cell_tag = int(parts[1])
        if elem_type == "hex":
            cells["hexahedron"].append(node_ids)
            cells_data["hexahedron"].append(cell_tag)
        elif elem_type == "quad":
            cells["quad"].append(node_ids)
            cells_data["quad"].append(cell_tag)

    # Convert to Meshio format
    cell_blocks = []
    for key, value in cells.items():
        if value:
            cell_blocks.append((key, value))

    cell_data_blocks = {"tag": []}
    for key, value in cells_data.items():
        if value:
            cell_data_blocks["tag"].append(value)

    return meshio.Mesh(points=points, cells=cell_blocks, cell_data=cell_data_blocks)


def read_stl_mesh(filename: Union[str, Path]) -> meshio.Mesh:
    """Read a mesh from an STL file.

    Args:
        filename: Path to STL file.

    Returns:
        meshio.Mesh object.
    """
    print(f"Reading STL mesh from file: {filename}")
    return meshio.read(filename)


def write_stl_mesh(mesh: meshio.Mesh, filename: Union[str, Path]) -> None:
    """Write a meshio.Mesh object to an STL file.

    Args:
        mesh: meshio.Mesh object to write.
        filename: Path to output STL file.
    """
    print(f"Writing mesh to STL file: {filename}")
    meshio.write(str(filename), mesh)


def scale_mesh_z(mesh: meshio.Mesh, z_factor: float) -> meshio.Mesh:
    """Scale the z-coordinates of a mesh by a given factor.

    Args:
        mesh: Input meshio.Mesh object.
        z_factor: Factor to scale z coordinates.

    Returns:
        New meshio.Mesh object with scaled z coordinates.
    """
    scaled_points = mesh.points.copy()
    scaled_points[:, 2] *= z_factor
    return meshio.Mesh(points=scaled_points, cells=mesh.cells, cell_data=mesh.cell_data)


def _tet_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Compute volume of a tetrahedron defined by 4 points."""
    return abs(np.linalg.det(np.vstack([b - a, c - a, d - a]))) / 6.0


def _hex_volume(points: np.ndarray) -> float:
    """Compute volume of a hexahedron by decomposing into tetrahedra."""
    p = points
    tets = [
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (1, 3, 4, 6),
        (1, 4, 5, 6),
        (3, 4, 6, 7),
    ]
    return sum(_tet_volume(p[i], p[j], p[k], p[l]) for i, j, k, l in tets)


def _scaled_jacobian_at_corner(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """Compute scaled Jacobian at a corner node.

    Args:
        p0: Corner node coordinates
        p1, p2, p3: Adjacent node coordinates

    Returns:
        Scaled Jacobian value
    """
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0

    det_J = np.linalg.det(np.vstack([e1, e2, e3]))
    norm_product = np.linalg.norm(e1) * np.linalg.norm(e2) * np.linalg.norm(e3)

    if norm_product < 1e-12:
        return 0.0

    return det_J / norm_product


def _hex_scaled_jacobian(points: np.ndarray) -> float:
    """Compute minimum scaled Jacobian for a hexahedron (Cubit convention).

    Standard hex ordering:
        4----7
       /|   /|
      5----6 |
      | 0--|-3
      |/   |/
      1----2

    Args:
        points: 8x3 array of hex node coordinates

    Returns:
        Minimum scaled Jacobian over all 8 corners
    """
    # Define adjacent nodes for each corner (Cubit hex ordering)
    corner_adjacency = [
        (0, 1, 3, 4),  # corner 0: adjacent 1, 3, 4
        (1, 0, 2, 5),  # corner 1: adjacent 0, 2, 5
        (2, 1, 3, 6),  # corner 2: adjacent 1, 3, 6
        (3, 0, 2, 7),  # corner 3: adjacent 0, 2, 7
        (4, 0, 5, 7),  # corner 4: adjacent 0, 5, 7
        (5, 1, 4, 6),  # corner 5: adjacent 1, 4, 6
        (6, 2, 5, 7),  # corner 6: adjacent 2, 5, 7
        (7, 3, 4, 6),  # corner 7: adjacent 3, 4, 6
    ]

    sj_values = []
    for corner, adj1, adj2, adj3 in corner_adjacency:
        sj = _scaled_jacobian_at_corner(
            points[corner], points[adj1], points[adj2], points[adj3]
        )
        sj_values.append(sj)

    return min(sj_values)


def add_cell_volumes(mesh: meshio.Mesh) -> meshio.Mesh:
    """Add per-cell volume and quality data to a meshio.Mesh."""
    volumes_per_block = []
    dims_per_block = []
    quality_per_block = []

    for cell_block in mesh.cells:
        cell_type = cell_block.type
        cell_conn = cell_block.data

        if cell_type == "hexahedron":
            vols = []
            qualities = []
            for conn in cell_conn:
                pts = mesh.points[np.array(conn)]
                vols.append(_hex_volume(pts))
                qualities.append(_hex_scaled_jacobian(pts))
            volumes_per_block.append(vols)
            dims_per_block.append([3] * len(cell_conn))
            quality_per_block.append(qualities)
        else:
            volumes_per_block.append([0.0] * len(cell_conn))
            dims_per_block.append([2] * len(cell_conn))
            quality_per_block.append([0.0] * len(cell_conn))

    cell_data = mesh.cell_data.copy() if mesh.cell_data else {}
    cell_data["volume"] = volumes_per_block
    cell_data["dim"] = dims_per_block
    cell_data["scaled_jacobian"] = quality_per_block

    return meshio.Mesh(points=mesh.points, cells=mesh.cells, cell_data=cell_data)


def write_mesh_to_vtu(
    mesh: meshio.Mesh, vtk_filename: Union[str, Path], binary: bool = True
) -> None:
    """Write a meshio.Mesh object to a VTU file.

    Args:
        mesh: meshio.Mesh object to write.
        vtk_filename: Path to output VTU file.
        binary: Whether to write in binary format (default: True).
    """
    print(f"Writing mesh to VTU file: {vtk_filename}")
    # mesh_with_volumes = add_cell_volumes(mesh)
    meshio.write(str(vtk_filename), mesh, binary=binary)


def mesh_to_vtu(
    filename: Union[str, Path], vtk_filename: Union[str, Path], z_factor: float = 1.0
) -> None:
    """Read mesh and export to VTU file (convenience wrapper).

    Args:
        filename: Path to mesh file.
        vtk_filename: Path to output VTU file.
        z_factor: Factor to scale z coordinates.
    """
    mesh = read_mesh(filename)
    if z_factor != 1.0:
        mesh = scale_mesh_z(mesh, z_factor)
    write_mesh_to_vtu(mesh, vtk_filename)


def write_mesh(mesh: meshio.Mesh, filename: Union[str, Path]) -> None:
    """Write a meshio.Mesh object to a native mesh file format.

    Writes in the same format as Meshfile.mesh (Cubit format):
    - First line: number_of_points number_of_elements 0 0 0
    - Node lines: node_id x y z
    - Element lines: element_id tag elem_type node_id1 node_id2 ...

    Quads are written before hexahedra. Tags are read from mesh.cell_data["tag"].

    Args:
        mesh: meshio.Mesh object to write.
        filename: Path to output mesh file.
    """
    # Count total elements
    total_elements = sum(len(cell_block.data) for cell_block in mesh.cells)

    with open(filename, "w") as f:
        # Write header (number of points, number of elements, and three zeros)
        f.write(f"{len(mesh.points)} {total_elements} 0 0 0\n")

        # Write nodes (1-based indexing)
        for i, point in enumerate(mesh.points):
            node_id = i + 1
            f.write(f"{node_id} {point[0]:.16e} {point[1]:.16e} {point[2]:.16e}\n")

        # Write elements: quads first, then hexahedra
        quad_id = 1

        # Get tag data if available
        tag_data = mesh.cell_data.get("tag", []) if mesh.cell_data else []

        # Write quads first
        for block_idx, cell_block in enumerate(mesh.cells):
            if cell_block.type != "quad":
                continue

            cell_data = cell_block.data
            elem_type = "quad"

            # Get tags for this block
            block_tags = tag_data[block_idx] if block_idx < len(tag_data) else None

            # Write each element
            for elem_idx, connectivity in enumerate(cell_data):
                # Get tag value for this element
                tag_value = block_tags[elem_idx] if block_tags is not None else 0

                # Convert from 0-based to 1-based indexing
                node_ids = [str(n + 1) for n in connectivity]
                f.write(f"{quad_id} {tag_value} {elem_type} {' '.join(node_ids)}\n")
                quad_id += 1

        # Write hexahedra
        hex_id = 1
        for block_idx, cell_block in enumerate(mesh.cells):
            if cell_block.type != "hexahedron":
                continue

            cell_data = cell_block.data
            elem_type = "hex"

            # Get tags for this block
            block_tags = tag_data[block_idx] if block_idx < len(tag_data) else None

            # Write each element
            for elem_idx, connectivity in enumerate(cell_data):
                # Get tag value for this element
                tag_value = block_tags[elem_idx] if block_tags is not None else 0

                # Convert from 0-based to 1-based indexing
                node_ids = [str(n + 1) for n in connectivity]
                f.write(f"{hex_id} {tag_value} {elem_type} {' '.join(node_ids)}\n")
                hex_id += 1
