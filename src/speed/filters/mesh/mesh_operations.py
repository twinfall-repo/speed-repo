"""Mesh operations and analysis functions.

Functions include:
- Setting dimensional tags
- Marking elements intersecting an STL surface
- Removing marked 3D/2D elements while keeping cell_data aligned
"""

import copy
import numpy as np
import trimesh
from meshio import CellBlock


def initialize_dimension_field(mesh):
    """Initialize a "dim" field in cell_data marking the topological dimension of each cell."""

    if mesh.cell_data is None:
        mesh.cell_data = {}

    dims = []
    for cell_block in mesh.cells:
        num_cells = len(cell_block.data)
        if cell_block.type == "hexahedron":
            dims.append(np.full(num_cells, 3, dtype=int))
        elif cell_block.type == "quad":
            dims.append(np.full(num_cells, 2, dtype=int))
        else:
            dims.append(np.full(num_cells, 2, dtype=int))

    mesh.cell_data["dim"] = dims
    print("Initialized 'dim' field in cell_data")


def mark_intersecting_hexahedra(mesh, stl_mesh, tag="intersection", surface_tol=1e-4):
    """Mark hexahedral elements that intersect with the STL mesh.

    Uses ray-casting to check if any of the 12 actual hex edges intersect the surface.
    """
    stl_triangles = stl_mesh.cells_dict.get("triangle", [])
    if len(stl_triangles) == 0:
        print("Warning: No triangles found in STL mesh")
        return

    stl_trimesh = trimesh.Trimesh(vertices=stl_mesh.points, faces=stl_triangles)

    if mesh.cell_data is None:
        mesh.cell_data = {}
    if tag not in mesh.cell_data:
        mesh.cell_data[tag] = []

    while len(mesh.cell_data[tag]) < len(mesh.cells):
        zeros = np.zeros(len(mesh.cells[len(mesh.cell_data[tag])].data), dtype=int)
        mesh.cell_data[tag].append(zeros)

    for cell_block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type != "hexahedron":
            continue

        cell_data = cell_block.data
        num_cells = len(cell_data)
        intersection_flags = np.zeros(num_cells, dtype=int)

        print(f"Checking {num_cells} hexahedron elements...")

        # Define the 12 edges of a hexahedron
        hex_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical edges
        ]

        for idx, element in enumerate(cell_data):
            element_points = mesh.points[element]

            # Check if any vertex is inside the STL volume
            inside_flags = stl_trimesh.contains(element_points)
            if inside_flags.any():
                intersection_flags[idx] = 1
            else:
                # Check if any of the 12 edges intersect the surface
                element_edges_intersect = False
                for e1, e2 in hex_edges:
                    p1 = element_points[e1]
                    p2 = element_points[e2]
                    locations, _, _ = stl_trimesh.ray.intersects_location(
                        ray_origins=[p1], ray_directions=[p2 - p1]
                    )
                    if len(locations) > 0:
                        for loc in locations:
                            t = np.linalg.norm(loc - p1) / np.linalg.norm(p2 - p1)
                            if 0 <= t <= 1:
                                element_edges_intersect = True
                                break
                    if element_edges_intersect:
                        break

                if element_edges_intersect:
                    intersection_flags[idx] = 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{num_cells} hexahedron elements...")

        mesh.cell_data[tag][cell_block_idx] = intersection_flags
        marked = int(intersection_flags.sum())
        print(f"Marked {marked}/{num_cells} hexahedron elements as '{tag}'")


def mark_intersecting_quads(mesh, stl_mesh, tag="intersection", surface_tol=1e-4):
    """Mark quadrilateral elements that intersect with the STL mesh.

    Uses ray-casting to check if quad edges or diagonals intersect the surface.
    """
    stl_triangles = stl_mesh.cells_dict.get("triangle", [])
    if len(stl_triangles) == 0:
        print("Warning: No triangles found in STL mesh")
        return

    stl_trimesh = trimesh.Trimesh(vertices=stl_mesh.points, faces=stl_triangles)

    if mesh.cell_data is None:
        mesh.cell_data = {}
    if tag not in mesh.cell_data:
        mesh.cell_data[tag] = []

    while len(mesh.cell_data[tag]) < len(mesh.cells):
        zeros = np.zeros(len(mesh.cells[len(mesh.cell_data[tag])].data), dtype=int)
        mesh.cell_data[tag].append(zeros)

    for cell_block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type != "quad":
            continue

        cell_data = cell_block.data
        num_cells = len(cell_data)
        intersection_flags = np.zeros(num_cells, dtype=int)

        print(f"Checking {num_cells} quad elements...")

        # Define the 4 edges + 2 diagonals of a quad for thorough intersection checking
        quad_segments = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # 4 edges
            (0, 2),
            (1, 3),  # 2 diagonals
        ]

        for idx, element in enumerate(cell_data):
            element_points = mesh.points[element]

            # Check if any vertex is inside the STL volume
            inside_flags = stl_trimesh.contains(element_points)
            if inside_flags.any():
                intersection_flags[idx] = 1
            else:
                # Check if any edge or diagonal intersects the surface
                element_intersects = False
                for e1, e2 in quad_segments:
                    p1 = element_points[e1]
                    p2 = element_points[e2]
                    locations, _, _ = stl_trimesh.ray.intersects_location(
                        ray_origins=[p1], ray_directions=[p2 - p1]
                    )
                    if len(locations) > 0:
                        for loc in locations:
                            t = np.linalg.norm(loc - p1) / np.linalg.norm(p2 - p1)
                            if 0 <= t <= 1:
                                element_intersects = True
                                break
                    if element_intersects:
                        break

                if element_intersects:
                    intersection_flags[idx] = 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{num_cells} quad elements...")

        mesh.cell_data[tag][cell_block_idx] = intersection_flags
        marked = int(intersection_flags.sum())
        print(f"Marked {marked}/{num_cells} quad elements as '{tag}'")


def remove_3d_cells(mesh, tag="intersection", new_quad_value=1):
    """Remove 3D hexahedral cells marked with tag=1 and add their faces as quads."""

    if mesh.cell_data is None or tag not in mesh.cell_data:
        print(f"Warning: No '{tag}' field found in mesh cell_data")
        return

    if "tag" not in mesh.cell_data:
        mesh.cell_data["tag"] = []

    hex_block_idx = None
    hex_flags = None
    for block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type == "hexahedron":
            hex_block_idx = block_idx
            hex_flags = mesh.cell_data[tag][block_idx]
            break

    if hex_block_idx is None:
        print("Warning: No hexahedral elements found in mesh")
        return

    if len(mesh.cell_data["tag"]) <= hex_block_idx:
        num_hex = len(mesh.cells[hex_block_idx].data)
        mesh.cell_data["tag"].append(np.zeros(num_hex, dtype=int))

    hex_cells = mesh.cells[hex_block_idx].data
    new_quads = []
    extracted_quads_set = set()

    # Hexahedron face connectivity with outward-pointing normals (VTK ordering)
    # Vertices 0-3 are bottom, 4-7 are top
    hex_face_connectivity = [
        [0, 3, 2, 1],  # Bottom face (CCW from below)
        [4, 5, 6, 7],  # Top face (CCW from above)
        [0, 1, 5, 4],  # Front face (CCW from front)
        [2, 3, 7, 6],  # Back face (CCW from back)
        [0, 4, 7, 3],  # Left face (CCW from left)
        [1, 2, 6, 5],  # Right face (CCW from right)
    ]

    for hex_idx, hex_element in enumerate(hex_cells):
        if hex_flags[hex_idx] == 1:
            for face_nodes in hex_face_connectivity:
                quad_element = hex_element[np.array(face_nodes)]
                # Use ordered tuple to preserve vertex sequence and normal direction
                key = tuple(quad_element)
                if key not in extracted_quads_set:
                    new_quads.append(quad_element)
                    extracted_quads_set.add(key)

    quad_block_idx = None
    for block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type == "quad":
            quad_block_idx = block_idx
            break

    if len(new_quads) > 0:
        new_quads_array = np.array(new_quads)
        if quad_block_idx is None:
            mesh.cells.append(CellBlock("quad", new_quads_array))
            for key, data_list in mesh.cell_data.items():
                if key == "tag":
                    data_list.append(
                        np.full(len(new_quads_array), new_quad_value, dtype=int)
                    )
                elif key == "dim":
                    data_list.append(np.full(len(new_quads_array), 2, dtype=int))
                else:
                    data_list.append(np.zeros(len(new_quads_array), dtype=int))
        else:
            mesh.cells[quad_block_idx].data = np.vstack(
                [mesh.cells[quad_block_idx].data, new_quads_array]
            )
            for key, data_list in mesh.cell_data.items():
                quad_flags = data_list[quad_block_idx]
                if key == "tag":
                    new_flags = np.full(len(new_quads_array), new_quad_value, dtype=int)
                elif key == "dim":
                    new_flags = np.full(len(new_quads_array), 2, dtype=int)
                else:
                    new_flags = np.zeros(len(new_quads_array), dtype=int)
                data_list[quad_block_idx] = np.hstack([quad_flags, new_flags])

    keep_indices = np.where(hex_flags == 0)[0]
    mesh.cells[hex_block_idx].data = hex_cells[keep_indices]
    for key in mesh.cell_data:
        mesh.cell_data[key][hex_block_idx] = mesh.cell_data[key][hex_block_idx][
            keep_indices
        ]

    print(f"Removed {len(hex_cells) - len(keep_indices)} hexahedral elements")
    print(f"Added {len(new_quads)} quad elements from removed hexahedra faces")


def remove_2d_cells(mesh, tag="intersection"):
    """Remove 2D quadrilateral cells marked with tag=1 from the mesh."""

    if mesh.cell_data is None or tag not in mesh.cell_data:
        print(f"Warning: No '{tag}' field found in mesh cell_data")
        return

    quad_block_idx = None
    quad_flags = None

    for block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type == "quad":
            quad_block_idx = block_idx
            quad_flags = mesh.cell_data[tag][block_idx]
            break

    if quad_block_idx is None:
        print("Warning: No quadrilateral elements found in mesh")
        return

    quad_cells = mesh.cells[quad_block_idx].data
    keep_indices = np.where(quad_flags == 0)[0]
    mesh.cells[quad_block_idx].data = quad_cells[keep_indices]
    mesh.cell_data[tag][quad_block_idx] = quad_flags[keep_indices]

    print(f"Removed {len(quad_cells) - len(keep_indices)} quadrilateral elements")


def remove_cell_type_from_mesh(mesh, cell_type_to_remove="quad"):
    """Create a deep copy of the mesh with all elements of a specific type removed."""

    mesh_copy = copy.deepcopy(mesh)

    cells_to_remove = []
    for block_idx, cell_block in enumerate(mesh_copy.cells):
        if cell_block.type == cell_type_to_remove:
            cells_to_remove.append(block_idx)

    for block_idx in reversed(cells_to_remove):
        del mesh_copy.cells[block_idx]
        if mesh_copy.cell_data:
            for key in mesh_copy.cell_data:
                if len(mesh_copy.cell_data[key]) > block_idx:
                    del mesh_copy.cell_data[key][block_idx]

    return mesh_copy
