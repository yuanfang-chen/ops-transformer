#!/usr/bin/env python3
"""
Run the compiled C++ example to produce device outputs.

Note: `test_aclnn_causal_conv1d.cpp` uses hard-coded shapes. By default it reads/writes
`./case` relative to the process working directory. You can override the case directory
via env `CAUSAL_CONV1D_CASE_DIR` (this script exposes it as `--case_dir`).

Example:
  py attention/causal_conv1d/examples/run_cpp_dump.py --exe ./test_aclnn_causal_conv1d --work_dir .
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=str, required=True, help="Path to test_aclnn_causal_conv1d executable")
    ap.add_argument(
        "--case_dir",
        type=Path,
        default=None,
        help="Case directory for inputs/outputs (sets env CAUSAL_CONV1D_CASE_DIR). Defaults to work_dir/case.",
    )
    ap.add_argument(
        "--work_dir",
        type=Path,
        default=Path("."),
        help="Working directory to run the binary from (default case_dir is work_dir/case)",
    )
    args = ap.parse_args()

    work_dir = args.work_dir
    case_dir = args.case_dir if args.case_dir is not None else (work_dir / "case")
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.exe]

    print("Running:", " ".join(cmd))
    env = None
    if args.case_dir is not None:
        env = os.environ.copy()
        env["CAUSAL_CONV1D_CASE_DIR"] = str(case_dir)
    p = subprocess.run(cmd, cwd=str(work_dir), env=env, check=False)
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
