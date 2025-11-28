"""Read MONITOR.INFO and run conversion of MONITOR files to per-monitor files.

This module locates the `MONITOR.INFO` file in a given directory, parses the
global settings and MPI layout, and calls `process_generic` for the supported
monitor output types (D/V/A/E/S/O). The implementation is defensive and
includes type hints for static checking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np

from functions import process_generic


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


if __name__ == "__main__":
    # Resolve defaults relative to this script location
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    default_path = current_dir.joinpath("../speed_tests/MONITOR")

    # Run with default OUT_OPT matching legacy script
    run_rewrite(default_path, default_path, OUT_OPT=(1, 0, 0, 0, 0, 0))
