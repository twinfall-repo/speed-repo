"""Small utilities for reading/writing MATLAB-style monitor files.

This module provides helpers to construct zero-padded monitor filenames,
read monitor info files, write per-monitor output files and a generic
processor to split and write monitor data. Functions include type hints
and runtime checks to be friendly to static checkers such as `mypy`.
"""

from pathlib import Path
from typing import Sequence, Tuple, Union
import os
import numpy as np
import meshio

import matplotlib.pyplot as plt


def run_rewrite(
    path_start: Union[str, Path],
    path_end: Union[str, Path],
    OUT_OPT: Sequence[int] = (1, 0, 0, 0, 0, 0),
) -> None:
    """Read `MONITOR.INFO` from `path_start` and run processing for each type.

    Args:
        path_start: Directory containing `MONITOR.INFO` and MONITOR files.
        path_end: Directory where rewritten monitor files should be written.
        OUT_OPT: Sequence of six integers enabling outputs (D,V,A,E,S,O).
    """
    path_start = Path(path_start)
    path_end = Path(path_end)

    info_path = path_start / "MONITOR.INFO"
    if not info_path.exists():
        raise FileNotFoundError(f"MONITOR.INFO not found in {path_start}")

    info = np.loadtxt(info_path, dtype=float)
    info = np.atleast_1d(info)
    if info.size < 4:
        raise ValueError("MONITOR.INFO appears to be malformed or incomplete")

    # Global timing and MPI layout
    T = float(info[0])
    dt_s = float(info[1])
    ndt_monit = int(info[2])
    dt = ndt_monit * dt_s

    MPI_num_proc = int(info[3])
    MPI_mnt_id = info[4 : 4 + MPI_num_proc].astype(int)

    print(f"Found {MPI_num_proc} MPI processes.")

    # Helper to call process_generic with current configuration
    def proc(ext_in: str, ext_out: str, outopt_index: int, block_size: int) -> None:
        process_generic(
            ext_in,
            ext_out,
            outopt_index,
            block_size,
            OUT_OPT,
            str(path_start),
            str(path_end),
            MPI_num_proc,
            MPI_mnt_id,
        )

    # Run conversions for supported types
    proc("D", ".d", 0, 3)
    proc("V", ".v", 1, 3)
    proc("A", ".a", 2, 3)
    proc("E", ".e", 3, 6)
    proc("S", ".s", 4, 6)
    proc("O", ".o", 5, 3)

    print("\nAll processing complete.")


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
        print(f"Plotting monitor file: {filename}")
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

        plt.savefig(monitors_path / f"{filename}_displacement.png", dpi=150)

    # Show all figures
    plt.show()


def mesh_to_vtu(filename, vtk_filename) -> None:
    points = []
    cells = {"hexahedron": [], "quad": []}

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
        if elem_type == "hex":
            cells["hexahedron"].append(node_ids)
        elif elem_type == "quad":
            cells["quad"].append(node_ids)

    # Convert to Meshio format
    cell_blocks = []
    for key, value in cells.items():
        if value:
            cell_blocks.append((key, value))

    mesh = meshio.Mesh(points=points, cells=cell_blocks)
    meshio.write(vtk_filename, mesh)


def padded_name(i: int, prefix: str = "monitor", ext: str = ".d") -> str:
    """Return a zero-padded filename like MATLAB's `sprintf` rules.

    Args:
        i: Numeric identifier to include in the name. Usually a small
            positive integer.
        prefix: Filename prefix (e.g. 'monitor' or 'MONITOR').
        ext: Extension including the dot (e.g. '.d').

    Returns:
        The constructed filename, e.g. 'monitor00012.d'.
    """
    # Use Python's integer formatting to produce zero-padded numbers up to
    # 5 digits (matching the original behavior). If `i` has more than five
    # digits, the full number is used (no truncation).
    return f"{prefix}{i:05d}{ext}"


def load_monitor_info(filepath: str) -> Tuple[int, np.ndarray]:
    """Load a MONITORXXX.INFO file and return the number of monitors and ids.

    The `.INFO` files are expected to contain integers; the first integer is
    the number of monitors followed by that many monitor ids.

    Args:
        filepath: Path to the `.INFO` file.

    Returns:
        A tuple (num, ids) where `num` is the number of monitors and `ids`
        is a 1-D numpy array of integers of length `num`.

    Raises:
        ValueError: if the file does not contain enough entries.
    """
    info = np.loadtxt(filepath, dtype=int)
    info = np.atleast_1d(info)
    if info.size == 0:
        raise ValueError(f"Empty info file: {filepath}")

    num = int(info[0])
    ids = info[1 : 1 + num].astype(int)
    if ids.size != num:
        raise ValueError(f"INFO file {filepath} expected {num} ids, found {ids.size}")

    return num, ids


def write_monitor_file(
    path_out: str, filename: str, time: np.ndarray, values: np.ndarray
) -> None:
    """Write a monitor output file.

    Each line contains the time followed by the associated values for that
    monitor row, formatted in scientific notation with 8 decimals.

    Args:
        path_out: Directory to write the file into.
        filename: Filename to write.
        time: 1-D array-like time values (length must match `values.shape[0]`).
        values: 2-D array-like of shape (len(time), ncols) with monitor values.
    """
    fullpath = os.path.join(path_out, filename)
    time_arr = np.asarray(time)
    values_arr = np.asarray(values)

    if values_arr.ndim == 1:
        # single column: make it two-dimensional for consistent column stack
        values_arr = values_arr.reshape(-1, 1)

    # Ensure shapes align
    if time_arr.ndim != 1:
        time_arr = time_arr.ravel()

    if time_arr.shape[0] != values_arr.shape[0]:
        raise ValueError("Length of `time` must match number of rows in `values`")

    # Ensure output directory exists
    os.makedirs(path_out, exist_ok=True)

    # Stack time as the first column and use numpy.savetxt for reliable
    # formatting and slightly better performance on large arrays.
    data = np.column_stack((time_arr, values_arr))
    np.savetxt(fullpath, data, fmt="%10.8e", delimiter="   ")


def process_generic(
    ext_in: str,
    ext_out: str,
    outopt_index: int,
    block_size: int,
    OUT_OPT: Sequence[int],
    path_start: str,
    path_end: str,
    MPI_num_proc: int,
    MPI_mnt_id: Sequence[int],
) -> None:
    """Process monitor files of a given type and write per-monitor outputs.

    This function reads aggregated monitor files (e.g. MONITOR0001.D) from
    `path_start`, splits the per-monitor columns according to `block_size`, and
    writes individual monitor files into `path_end` using `write_monitor_file`.

    Args:
        ext_in: Input extension letter used in the MONITOR files (e.g. 'D').
        ext_out: Output extension including the dot (e.g. '.d').
        outopt_index: Index into `OUT_OPT` to check whether to run this type.
        block_size: Number of columns per monitor (3 for D/V/A/O, 6 for E/S).
        OUT_OPT: Sequence of integers controlling which outputs are enabled.
        path_start: Directory where input files are located.
        path_end: Directory where output files should be written.
        MPI_num_proc: Number of MPI processes used to generate MONITOR files.
        MPI_mnt_id: Sequence mapping process index to monitor presence (0 -> skip).
    """
    # Safety: ensure index is valid and enabled
    if outopt_index < 0 or outopt_index >= len(OUT_OPT) or OUT_OPT[outopt_index] != 1:
        return

    print(f"\n=== Processing type {ext_in} → {ext_out} ===")

    if len(MPI_mnt_id) < MPI_num_proc:
        raise ValueError("Length of `MPI_mnt_id` must be at least `MPI_num_proc`")

    for i in range(1, MPI_num_proc + 1):
        print(f"Processing MONITOR {i}...")

        # Skip if this process produced no monitor
        if MPI_mnt_id[i - 1] == 0:
            continue

        base_idx = i - 1

        fname_data = padded_name(base_idx, "MONITOR", f".{ext_in}")
        fname_info = padded_name(base_idx, "MONITOR", ".INFO")

        f_info_path = os.path.join(path_start, fname_info)
        if not os.path.exists(f_info_path):
            # No info file -> nothing to do for this MONITOR
            continue

        num_mon, ids = load_monitor_info(f_info_path)

        f_data_path = os.path.join(path_start, fname_data)
        if not os.path.exists(f_data_path):
            # Missing data file; skip this monitor set with a message
            print(f"  data file missing: {f_data_path}")
            continue

        try:
            val = np.loadtxt(f_data_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  failed to read {f_data_path}: {exc}")
            continue

        if val.size == 0:
            # Empty data file; nothing to write
            continue

        # Ensure 2-D shape even if single row present
        val = np.atleast_2d(val)

        # First column is time, following columns are monitor data
        time = val[:, 0]
        k = 0

        os.makedirs(path_end, exist_ok=True)

        for j in range(num_mon):
            this_id = int(ids[j])
            values = val[:, 1 + k : 1 + k + block_size]

            outname = padded_name(this_id, "monitor", ext_out)
            write_monitor_file(path_end, outname, time, values)

            k += block_size
