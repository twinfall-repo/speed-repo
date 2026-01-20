# SPEED Scripts

This directory contains executable scripts for common SPEED data processing workflows.

## Scripts

### extract_refinement_curves.py

Extracts mesh refinement curves from Digital Elevation Model (DEM) data for use in Cubit mesh generation.

**Features:**
- Load and interpolate DEM data
- Compute gradient magnitude to identify high-variation regions
- Extract contour curves at high-gradient threshold
- Export curves in multiple formats (XYZ, VTK, PVD, SAT)
- Visualization of results

**Usage:**
```bash
python scripts/extract_refinement_curves.py
```

**Outputs:**
- `.xyz` files: Point cloud format
- `.vtk` files: ParaView visualization format
- `.pvd` files: ParaView collection format
- `.sat` files: ACIS geometry format for Cubit

### convert_mesh_grid.py

Converts and scales mesh files between formats.

**Features:**
- Read mesh files in native format
- Scale z-coordinates by a configurable factor
- Export to VTU format for ParaView visualization
- Export to native mesh format

**Usage:**
```bash
python scripts/convert_mesh_grid.py
```

**Outputs:**
- `output.vtu`: VTK visualization file
- `output_scaled.mesh`: Scaled mesh in native format

## Configuration

Edit the script files directly to modify:
- Input/output file paths
- Processing parameters (gradient percentiles, scaling factors, etc.)
- Grid dimensions

## Requirements

These scripts require the SPEED Python library modules located in `src/speed/filters/`.
