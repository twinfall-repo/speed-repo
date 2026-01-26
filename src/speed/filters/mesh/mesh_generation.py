"""Mesh generation functions using gmsh."""

import numpy as np
from pathlib import Path
import gmsh


def generate_mesh_from_profile(profile, profile_id, output_dir, mesh_size=1.0):
    """
    Generate a 2D triangular mesh from a profile curve using gmsh.

    Args:
        profile: Dictionary with 'top' and 'bottom' 3D curves
        profile_id: ID for naming output files
        output_dir: Directory to save mesh files
        mesh_size: Target mesh element size

    Returns:
        Tuple of (nodes, triangles) where nodes is (n, 2) and triangles is (m, 3)
    """
    top_curve = profile["top"]
    bottom_curve = profile["bottom"]

    # Create a new model for this profile (assumes gmsh is initialized outside)
    gmsh.model.add(f"profile_{profile_id}")

    # Helper: remove consecutive duplicate points with larger threshold
    def dedupe_consecutive(pts, eps):
        if len(pts) == 0:
            return pts
        cleaned = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - cleaned[-1]) > eps:
                cleaned.append(p)
        return np.array(cleaned)

    # Clean top curve - use mesh_size as threshold
    eps = mesh_size * 0.5
    top_clean = dedupe_consecutive(top_curve, eps)

    # Bottom is a straight line - only need first and last points for a single edge
    bottom_simple = np.array([bottom_curve[0], bottom_curve[-1]])

    if len(top_clean) < 3:
        raise ValueError(
            f"Insufficient points in top curve after cleaning: {len(top_clean)}"
        )

    # Create single closed contour: top + bottom (just 2 points) reversed
    # Skip last point of top to avoid duplicate with first point of bottom
    closed_contour = np.vstack([top_clean[:-1], bottom_simple[::-1]])

    # Final deduplication on closed contour
    closed_contour = dedupe_consecutive(closed_contour, eps)

    if len(closed_contour) < 3:
        raise ValueError(
            f"Insufficient points in closed contour: {len(closed_contour)}"
        )

    # Create gmsh points with variable mesh size (coarser at bottom)
    # Find z range for this profile
    z_coords = closed_contour[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_range = z_max - z_min

    point_tags = []
    for pt in closed_contour:
        # Calculate normalized distance from bottom (0 at bottom, 1 at top)
        if z_range > 0:
            t = (pt[2] - z_min) / z_range  # 0 at bottom, 1 at top
        else:
            t = 0

        # Mesh size varies from 3*mesh_size at bottom to mesh_size at top (matching background field)
        local_mesh_size = mesh_size * (3.0 - 2.0 * t)
        point_tags.append(gmsh.model.geo.addPoint(pt[0], pt[1], pt[2], local_mesh_size))

    # Create closed loop
    line_tags = []
    for i in range(len(point_tags)):
        j = (i + 1) % len(point_tags)
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[j]))

    # Create curve loop and plane surface
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface_tag = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()

    # Add a mesh size field that varies with z-coordinate (coarser at bottom)
    # This ensures interior mesh also gets coarser toward the bottom
    field_tag = gmsh.model.mesh.field.add("MathEval")
    # Formula: mesh_size * (3 - 2 * (z - z_min) / z_range)
    # At z=z_min (bottom): mesh_size * 3
    # At z=z_max (top): mesh_size * 1
    if z_range > 0:
        formula = f"{mesh_size} * (3.0 - 2.0 * (z - {z_min}) / {z_range})"
    else:
        formula = f"{mesh_size}"
    gmsh.model.mesh.field.setString(field_tag, "F", formula)

    # Set this as the background mesh field
    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Smooth and optimize
    try:
        gmsh.model.mesh.smooth()
        gmsh.model.mesh.optimize("Netgen")
    except Exception:
        pass

    # Get mesh data
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, element_connectivity = gmsh.model.mesh.getElements()

    # Extract triangles (element type 2 is triangle)
    triangles = []
    for elem_type, elem_tags, connectivity in zip(
        element_types, element_tags, element_connectivity
    ):
        if elem_type == 2:  # Triangle
            # Connectivity is flat, reshape to (n_triangles, 3)
            triangles = np.array(connectivity, dtype=int).reshape(-1, 3)

    # Create node coordinates array
    nodes = node_coords.reshape(-1, 3)

    # Create mapping from gmsh node tags to indices
    node_map = {tag: i for i, tag in enumerate(node_tags)}

    # Remap triangle connectivity to 0-based indices
    if len(triangles) > 0:
        triangles_remapped = np.zeros_like(triangles)
        for i in range(len(triangles)):
            for j in range(3):
                triangles_remapped[i, j] = node_map[triangles[i, j]]
        triangles = triangles_remapped

    # Export to files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export node coordinates
    nodes_file = output_dir / f"profile_{profile_id:03d}_nodes.txt"
    np.savetxt(nodes_file, nodes, fmt="%.6f", header="x y z", comments="")

    # Export triangles (node IDs, 1-based for compatibility)
    triangles_file = output_dir / f"profile_{profile_id:03d}_triangles.txt"
    np.savetxt(
        triangles_file, triangles + 1, fmt="%d", header="node1 node2 node3", comments=""
    )

    # Export VTK mesh for ParaView (direct gmsh writer; 3D plane coordinates)
    vtk_file = output_dir / f"profile_{profile_id:03d}.vtk"
    gmsh.write(str(vtk_file))
    print(f"    Saved VTK: {vtk_file.name}")

    # Remove current model to avoid accumulating state
    try:
        gmsh.model.remove()
    except Exception:
        pass

    return nodes, triangles, nodes_file, triangles_file


def generate_all_meshes(profiles, output_dir, mesh_size=1.0):
    """
    Generate meshes for all profiles.

    Args:
        profiles: List of profile dictionaries
        output_dir: Directory to save mesh files
        mesh_size: Target mesh element size

    Returns:
        List of tuples (nodes, triangles) for each profile
    """
    results = []

    # Initialize gmsh once
    if not gmsh.isInitialized():
        gmsh.initialize()

    # Global mesh options aimed at good triangle quality/aspect ratio
    gmsh.option.setNumber(
        "Mesh.Algorithm", 6
    )  # Frontal-Delaunay (robust for triangles)
    gmsh.option.setNumber(
        "Mesh.Smoothing", 0
    )  # Disable smoothing to preserve size field
    gmsh.option.setNumber(
        "Mesh.Optimize", 0
    )  # Disable optimization to preserve size field
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2.0)

    for i, profile in enumerate(profiles):
        print(f"  Generating mesh for profile {i + 1}/{len(profiles)}...")
        try:
            nodes, triangles, nodes_file, triangles_file = generate_mesh_from_profile(
                profile, i, output_dir, mesh_size
            )
            results.append((nodes, triangles))
            print(f"    Nodes: {len(nodes)}, Triangles: {len(triangles)}")
            print(f"    Saved to {nodes_file.parent.name}/")
        except Exception as e:
            print(f"    Error: {e}")
            results.append((None, None))

    # Finalize gmsh once
    gmsh.finalize()

    return results
