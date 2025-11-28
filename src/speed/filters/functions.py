"""Small utilities for reading/writing MATLAB-style monitor files.

This module provides helpers to construct zero-padded monitor filenames,
read monitor info files, write per-monitor output files and a generic
processor to split and write monitor data. Functions include type hints
and runtime checks to be friendly to static checkers such as `mypy`.
"""

from typing import Sequence, Tuple
import os
import numpy as np


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
