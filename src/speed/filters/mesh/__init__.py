"""Mesh I/O operations for SPEED mesh files.

This package provides tools for reading, writing, and transforming
mesh files between different formats.
"""

from .mesh_io import (
    read_mesh,
    scale_mesh_z,
    write_mesh_to_vtu,
    mesh_to_vtu,
    write_mesh,
)
from .mesh_operations import (
    initialize_dimension_field,
    mark_intersecting_hexahedra,
    mark_intersecting_quads,
    remove_3d_cells,
    remove_2d_cells,
    remove_cell_type_from_mesh,
)

__all__ = [
    "read_mesh",
    "scale_mesh_z",
    "write_mesh_to_vtu",
    "mesh_to_vtu",
    "write_mesh",
    "mark_intersecting_hexahedra",
    "mark_intersecting_quads",
    "remove_3d_cells",
    "remove_2d_cells",
    "remove_cell_type_from_mesh",
]
