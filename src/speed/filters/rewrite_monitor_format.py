"""Read MONITOR.INFO and run conversion of MONITOR files to per-monitor files.

This module locates the `MONITOR.INFO` file in a given directory, parses the
global settings and MPI layout, and calls `process_generic` for the supported
monitor output types (D/V/A/E/S/O). The implementation is defensive and
includes type hints for static checking.
"""

from __future__ import annotations

from pathlib import Path

from functions import run_rewrite


if __name__ == "__main__":
    # Folder of the test case relative to this script
    folder = Path("PLANE_WAVE/1_TEST_SEM/MONITOR")

    # Folder in home directory for tutorials
    folder_speed = Path("/home/user/speed-tutorials")

    # Run with default OUT_OPT matching legacy script
    path = folder_speed.joinpath(folder)
    run_rewrite(path, path, OUT_OPT=(1, 0, 0, 0, 0, 0))
