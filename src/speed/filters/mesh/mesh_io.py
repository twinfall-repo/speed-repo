"""Mesh I/O utilities for reading, writing, and transforming mesh files.

This module provides functions for working with mesh files:
- Reading mesh files in native format
- Writing mesh files to native and VTU formats
- Scaling mesh coordinates
"""

from pathlib import Path
from typing import Union
import meshio


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
    return meshio.read(filename)


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


def write_mesh_to_vtu(
    mesh: meshio.Mesh, vtk_filename: Union[str, Path], binary: bool = False
) -> None:
    """Write a meshio.Mesh object to a VTU file.

    Args:
        mesh: meshio.Mesh object to write.
        vtk_filename: Path to output VTU file.
        binary: Whether to write in binary format (default: False).
    """
    print(f"Writing mesh to VTU file: {vtk_filename}")
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
