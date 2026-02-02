#!/usr/bin/env python3
"""
Compare device outputs vs torch reference outputs for a saved case directory.

Expected files in case_dir:
  meta.txt
  y_device.bin, conv_states_device.bin
  y_ref_torch.bin, conv_states_ref_torch.bin

Optional export:
  --dump_npy writes the loaded arrays as .npy into out_dir (default: case_dir)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


F32 = np.dtype("<f4")


def _nanmax_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    if np.isnan(x).all():
        return float("nan")
    return float(np.nanmax(np.abs(x)))


def _nanmax(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    if np.isnan(x).all():
        return float("nan")
    return float(np.nanmax(x))


def _print_sample_mismatches(
    *,
    name: str,
    dev: np.ndarray,
    ref: np.ndarray,
    bad_mask: np.ndarray,
    atol: float,
    rtol: float,
    max_samples: int = 5,
) -> None:
    if max_samples <= 0:
        return
    if bad_mask.ndim != dev.ndim:
        return
    found = 0
    if bad_mask.ndim == 2:
        for i in range(bad_mask.shape[0]):
            cols = np.nonzero(bad_mask[i])[0]
            if cols.size == 0:
                continue
            take = min(int(cols.size), max_samples - found)
            for c in cols[:take]:
                d = dev[i, c]
                r = ref[i, c]
                diff = abs(float(d) - float(r))
                tol = float(atol) + float(rtol) * abs(float(r))
                print(f"  {name}[{i},{int(c)}] dev={float(d):.6g} ref={float(r):.6g} diff={diff:.6g} tol={tol:.6g}")
            found += take
            if found >= max_samples:
                return
        return

    flat_mask = bad_mask.reshape(-1)
    flat_dev = dev.reshape(-1)
    flat_ref = ref.reshape(-1)
    idxs = np.nonzero(flat_mask)[0]
    for idx in idxs[:max_samples]:
        d = flat_dev[idx]
        r = flat_ref[idx]
        diff = abs(float(d) - float(r))
        tol = float(atol) + float(rtol) * abs(float(r))
        print(f"  {name}[{int(idx)}] dev={float(d):.6g} ref={float(r):.6g} diff={diff:.6g} tol={tol:.6g}")


def _print_first_nan(*, name: str, x: np.ndarray) -> None:
    if x.size == 0:
        return
    if x.ndim == 2:
        for i in range(x.shape[0]):
            cols = np.nonzero(np.isnan(x[i]))[0]
            if cols.size:
                j = int(cols[0])
                print(f"first_nan {name}[{i},{j}]")
                return
        return
    flat = np.isnan(x.reshape(-1))
    idxs = np.nonzero(flat)[0]
    if idxs.size:
        print(f"first_nan {name}[{int(idxs[0])}]")


def read_meta_txt(path: Path) -> Dict[str, int]:
    meta: Dict[str, int] = {}
    if not path.exists():
        return meta
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = "".join(ch for ch in raw if not ch.isspace())
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            meta[k] = int(v)
        except ValueError:
            continue
    return meta


def load_bin(path: Path, shape) -> np.ndarray:
    return np.fromfile(str(path), dtype=F32).reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None, help="Output dir for optional exports (default: case_dir)")
    ap.add_argument("--dump_npy", action="store_true", help="Export loaded outputs to .npy files")
    ap.add_argument(
        "--dump_npy_only",
        action="store_true",
        help="Only export .npy (skip comparison; implies --dump_npy)",
    )
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    case_dir: Path = args.case_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else case_dir
    dump_npy = bool(args.dump_npy or args.dump_npy_only)
    meta = read_meta_txt(case_dir / "meta.txt")
    batch_size = int(meta.get("batch_size", 32))
    total_seqlen = int(meta.get("total_seqlen", 8192))
    dim = int(meta.get("dim", 1024))
    width = int(meta.get("width", 4))

    y_dev_path = case_dir / "y_device.bin"
    y_ref_path = case_dir / "y_ref_torch.bin"
    s_dev_path = case_dir / "conv_states_device.bin"
    s_ref_path = case_dir / "conv_states_ref_torch.bin"

    y_shape = (total_seqlen, dim)
    s_shape = (batch_size, width - 1, dim)

    if bool(args.dump_npy_only):
        out_dir.mkdir(parents=True, exist_ok=True)
        converted = 0
        if y_dev_path.exists():
            np.save(str(out_dir / "y_device.npy"), load_bin(y_dev_path, y_shape))
            converted += 1
        if y_ref_path.exists():
            np.save(str(out_dir / "y_ref_torch.npy"), load_bin(y_ref_path, y_shape))
            converted += 1
        if s_dev_path.exists():
            np.save(str(out_dir / "conv_states_device.npy"), load_bin(s_dev_path, s_shape))
            converted += 1
        if s_ref_path.exists():
            np.save(str(out_dir / "conv_states_ref_torch.npy"), load_bin(s_ref_path, s_shape))
            converted += 1

        if converted == 0:
            print(f"ERROR: no output .bin files found under: {case_dir}")
            return 2

        print(f"Wrote .npy outputs to: {out_dir}")
        return 0

    y_dev = load_bin(y_dev_path, y_shape)
    y_ref = load_bin(y_ref_path, y_shape)
    s_dev = load_bin(s_dev_path, s_shape)
    s_ref = load_bin(s_ref_path, s_shape)

    if dump_npy:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "y_device.npy"), y_dev)
        np.save(str(out_dir / "y_ref_torch.npy"), y_ref)
        np.save(str(out_dir / "conv_states_device.npy"), s_dev)
        np.save(str(out_dir / "conv_states_ref_torch.npy"), s_ref)
        print(f"Wrote .npy outputs to: {out_dir}")

    y_dev_nan = int(np.isnan(y_dev).sum())
    y_ref_nan = int(np.isnan(y_ref).sum())
    y_dev_inf = int(np.isinf(y_dev).sum())
    y_ref_inf = int(np.isinf(y_ref).sum())
    y_dev_max_abs = _nanmax_abs(y_dev)
    y_ref_max_abs = _nanmax_abs(y_ref)

    y_diff = np.abs(y_dev - y_ref)
    y_tol = float(args.atol) + float(args.rtol) * np.abs(y_ref)
    y_bad = (y_diff > y_tol) | np.isnan(y_diff)
    y_bad_count = int(y_bad.sum())
    y_max = _nanmax(y_diff)
    y_diff_nan = int(np.isnan(y_diff).sum())

    s_diff = np.abs(s_dev - s_ref)
    s_bad = (s_diff > 0.0) | np.isnan(s_diff)
    s_bad_count = int(s_bad.sum())
    s_max = _nanmax(s_diff)

    print(
        f"y_dev_nan={y_dev_nan} y_dev_inf={y_dev_inf} y_dev_max_abs={y_dev_max_abs:.6g} "
        f"y_ref_nan={y_ref_nan} y_ref_inf={y_ref_inf} y_ref_max_abs={y_ref_max_abs:.6g} "
        f"y_diff_nan={y_diff_nan}"
    )
    if y_dev_nan:
        _print_first_nan(name="y_device", x=y_dev)
    if y_ref_nan:
        _print_first_nan(name="y_ref", x=y_ref)

    print(
        f"y_bad={y_bad_count}/{y_diff.size} y_max_abs_diff={y_max:.6g} "
        f"conv_states_bad={s_bad_count}/{s_diff.size} conv_states_max_abs_diff={s_max:.6g}"
    )
    if y_bad_count != 0:
        print("Sample y mismatches:")
        _print_sample_mismatches(
            name="y",
            dev=y_dev,
            ref=y_ref,
            bad_mask=y_bad,
            atol=float(args.atol),
            rtol=float(args.rtol),
            max_samples=5,
        )
    return 0 if (y_bad_count == 0 and s_bad_count == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
