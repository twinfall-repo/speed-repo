"""Plot monitor displacement components from SEM monitor files.

This script locates monitor files (e.g. `monitor00001.d`) and plots the
three displacement components (u_x, u_y, u_z) for a configurable set of
monitors. The plotting logic is wrapped in a reusable function with type
hints so it can be imported or executed as a script.
"""

from __future__ import annotations
from pathlib import Path

from functions import plot_monitors


if __name__ == "__main__":
    # Folder of the test case relative to this script
    folder = Path("PLANE_WAVE/1_TEST_SEM/MONITOR")

    # Folder in home directory for tutorials
    folder_speed = Path("/home/user/speed-tutorials")

    # Default parameters (previous script behavior)
    path = folder_speed.joinpath(folder)
    plot_monitors(path, num_mon=4, t0=0.0, T=10.0)
