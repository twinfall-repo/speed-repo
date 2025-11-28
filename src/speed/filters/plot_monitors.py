"""Plot monitor displacement components from SEM monitor files.

This script locates monitor files (e.g. `monitor00001.d`) and plots the
three displacement components (u_x, u_y, u_z) for a configurable set of
monitors. The plotting logic is wrapped in a reusable function with type
hints so it can be imported or executed as a script.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from functions import padded_name


def plot_monitors(
    monitors_dir: Union[str, Path],
    num_mon: int = 4,
    t0: float = 0.0,
    T: float = 10.0,
) -> None:
    """Plot displacement time series for a range of monitors.

    Args:
        monitors_dir: Directory containing monitor files (path-like or string).
        num_mon: Number of monitors to plot (monitor indices 1..num_mon).
        t0: Left x-axis limit for plots.
        T: Right x-axis limit for plots.
    """
    monitors_path = Path(monitors_dir)

    for i in range(1, num_mon + 1):
        filename = padded_name(i)
        filepath = monitors_path / filename

        if not filepath.exists():
            print(f"⚠ File not found: {filename}")
            continue

        # Load monitor data defensively
        try:
            sol_1 = np.loadtxt(filepath)
        except Exception as exc:  # pragma: no cover - I/O defensive handling
            print(f"⚠ Failed to load {filepath}: {exc}")
            continue

        if sol_1.size == 0:
            print(f"⚠ Empty file: {filepath}")
            continue

        sol_1 = np.atleast_2d(sol_1)

        if sol_1.shape[1] < 4:
            print(f"⚠ Unexpected number of columns in {filename}: {sol_1.shape[1]}")
            continue

        # Time and displacements
        time = sol_1[:, 0]
        ux = sol_1[:, 1]
        uy = sol_1[:, 2]
        uz = sol_1[:, 3]

        # Create figure
        plt.figure(i, figsize=(8, 10))

        # u_x
        plt.subplot(3, 1, 1)
        plt.plot(time, ux, linewidth=2)
        plt.grid(True)
        plt.xlim([t0, T])
        plt.ylim([-5, 5])
        plt.xlabel("t (s)")
        plt.ylabel("u_x (m)")
        plt.title(filename)
        plt.legend(["SEM"])

        # u_y
        plt.subplot(3, 1, 2)
        plt.plot(time, uy, linewidth=2)
        plt.grid(True)
        plt.xlim([t0, T])
        plt.ylim([-5, 5])
        plt.xlabel("t (s)")
        plt.ylabel("u_y (m)")
        plt.legend(["SEM"])

        # u_z
        plt.subplot(3, 1, 3)
        plt.plot(time, uz, linewidth=2)
        plt.grid(True)
        plt.xlim([t0, T])
        plt.ylim([-5, 5])
        plt.xlabel("t (s)")
        plt.ylabel("u_z (m)")
        plt.legend(["SEM"])

    # Show all figures
    plt.show()


if __name__ == "__main__":
    # Resolve default monitors directory relative to this script
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    default_monitors = current_dir.joinpath("../speed_tests/MONITOR/")

    # Default parameters (previous script behavior)
    plot_monitors(default_monitors, num_mon=4, t0=0.0, T=10.0)
