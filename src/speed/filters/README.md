# SPEED Filters

This directory contains organized modules for SPEED data processing and mesh preparation.

## Directory Structure

### curves/
Tools for extracting and processing mesh refinement curves from DEM data.

**Modules:**
- `dem_processing.py` - Load and interpolate DEM data
- `curve_extraction.py` - Extract curves from gradient fields
- `curve_simplification.py` - Reduce point density in curves
- `curve_export.py` - Export to XYZ, VTK, PVD, and SAT formats
- `curve_visualization.py` - Visualization utilities
- `refinement_curves.py` - Main orchestration module

**Usage:**
```python
from speed.filters.curves import (
    load_dem,
    interpolate_dem,
    extract_refinement_curves,
    export_curves_sat
)
```

### monitors/
Tools for processing and visualizing SPEED monitor output files.

**Modules:**
- `monitor_processing.py` - Process and convert monitor files
- `monitor_plotting.py` - Plot monitor time series
- `plot_monitors.py` - Entry point script for plotting
- `rewrite_monitor_format.py` - Entry point script for conversion

**Usage:**
```python
from speed.filters.monitors import (
    run_rewrite,
    plot_monitors,
    padded_name
)
```

### mesh/
Tools for mesh I/O operations and format conversion.

**Modules:**
- `mesh_io.py` - Read, write, and transform mesh files
- `convert_grid.py` - Entry point script for mesh conversion

**Usage:**
```python
from speed.filters.mesh import (
    read_mesh,
    scale_mesh_z,
    write_mesh_to_vtu
)
```

## Legacy Files

- `functions.py` - Original combined module (deprecated, use organized packages instead)

## See Also

- `../../scripts/` - Executable scripts for common workflows
