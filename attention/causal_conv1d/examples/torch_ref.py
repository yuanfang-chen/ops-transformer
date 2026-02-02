#!/usr/bin/env python3
"""
PyTorch CPU reference for causal_conv1d (prefill/extend varlen) using the
same `.bin` files dumped/generated in a case directory (e.g. `./case/`).

Reads from case_dir:
  x.bin, weight.bin, bias.bin, conv_states.bin, query_start_loc.bin, cache_indices.bin, has_initial_state.bin, meta.txt

Writes to case_dir (by default):
  y_ref_torch.bin, conv_states_ref_torch.bin

Optional comparison if device outputs exist in case_dir:
  y_device.bin, conv_states_device.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


F32 = np.dtype("<f4")
I32 = np.dtype("<i4")
U8 = np.dtype("u1")


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


def load_bin(path: Path, dtype: np.dtype, shape) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=dtype)
    return arr.reshape(shape)


def save_bin(path: Path, arr: np.ndarray) -> None:
    arr.astype(arr.dtype, copy=False).tofile(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    q = ap.add_mutually_exclusive_group()
    q.add_argument("--quantize_fp16", dest="quantize_fp16", action="store_true",
                   help="Quantize outputs to fp16 before saving (default).")
    q.add_argument("--no_quantize_fp16", dest="quantize_fp16", action="store_false",
                   help="Do not fp16-quantize outputs before saving.")
    ap.set_defaults(quantize_fp16=True)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    quantize_fp16 = bool(args.quantize_fp16)

    case_dir: Path = args.case_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_meta_txt(case_dir / "meta.txt")
    batch_size = int(meta.get("batch_size", 32))
    total_seqlen = int(meta.get("total_seqlen", 8192))
    dim = int(meta.get("dim", 1024))
    width = int(meta.get("width", 4))
    activation_mode = int(meta.get("activation_mode", 1))
    pad_slot_id = int(meta.get("pad_slot_id", -1))
    if width != 4:
        raise SystemExit(f"width must be 4 for torch_ref.py (got {width})")

    x = load_bin(case_dir / "x.bin", F32, (total_seqlen, dim))
    weight = load_bin(case_dir / "weight.bin", F32, (width, dim))
    bias = load_bin(case_dir / "bias.bin", F32, (dim,))
    conv_states_in = load_bin(case_dir / "conv_states.bin", F32, (batch_size, width - 1, dim))
    qsl = load_bin(case_dir / "query_start_loc.bin", I32, (batch_size + 1,))
    cache_indices = load_bin(case_dir / "cache_indices.bin", I32, (batch_size,))
    has_initial_state = load_bin(case_dir / "has_initial_state.bin", U8, (batch_size,))

    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise SystemExit(
            f"PyTorch is required for torch_ref.py (import error: {e}). "
            "Install torch in your Python environment."
        )

    device = torch.device("cpu")
    # Match kernel semantics: FP16 IO + FP32 compute.
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float16).to(torch.float32)
    w_t = torch.from_numpy(weight).to(device=device, dtype=torch.float16).to(torch.float32)
    b_t = torch.from_numpy(bias).to(device=device, dtype=torch.float16).to(torch.float32)
    s_in_t = torch.from_numpy(conv_states_in).to(device=device, dtype=torch.float16).to(torch.float32)

    y = torch.zeros((total_seqlen, dim), device=device, dtype=torch.float32)
    s_out = s_in_t.clone()

    for seq in range(batch_size):
        start = int(qsl[seq])
        end = int(qsl[seq + 1])
        length = end - start
        if length <= 0:
            continue

        cache_idx = int(cache_indices[seq])
        if cache_idx == pad_slot_id:
            continue

        has_init = bool(has_initial_state[seq] != 0)
        if has_init:
            hist = s_in_t[cache_idx]  # [width-1, dim] = [h-3, h-2, h-1]
        else:
            hist = torch.zeros((width - 1, dim), device=device, dtype=torch.float32)

        x_seg = x_t[start:end]  # [L, dim]
        x_ext = torch.cat([hist, x_seg], dim=0)  # [L+3, dim]

        x0 = x_ext[3 : 3 + length]  # X_t
        x1 = x_ext[2 : 2 + length]  # X_{t-1}
        x2 = x_ext[1 : 1 + length]  # X_{t-2}
        x3 = x_ext[0 : 0 + length]  # X_{t-3}

        # Align with vLLM/Triton (PyTorch conv1d correlation) convention:
        #   w[0]*X_{t-3} + w[1]*X_{t-2} + w[2]*X_{t-1} + w[3]*X_t
        acc = b_t + x3 * w_t[0] + x2 * w_t[1] + x1 * w_t[2] + x0 * w_t[3]
        if activation_mode != 0:
            acc = F.silu(acc)

        y[start:end] = acc
        s_out[cache_idx] = x_ext[-(width - 1) :]

    if quantize_fp16:
        y_save = y.to(torch.float16).to(torch.float32)
        s_save = s_out.to(torch.float16).to(torch.float32)
    else:
        y_save = y
        s_save = s_out

    y_np = y_save.cpu().numpy().astype(F32, copy=False)
    s_np = s_save.cpu().numpy().astype(F32, copy=False)
    save_bin(out_dir / "y_ref_torch.bin", y_np)
    save_bin(out_dir / "conv_states_ref_torch.bin", s_np)

    print(f"Wrote torch reference outputs to: {out_dir}")

    # Optional compare vs device outputs if present.
    y_dev_path = case_dir / "y_device.bin"
    s_dev_path = case_dir / "conv_states_device.bin"
    if y_dev_path.exists() and s_dev_path.exists():
        y_dev = load_bin(y_dev_path, F32, (total_seqlen, dim))
        s_dev = load_bin(s_dev_path, F32, (batch_size, width - 1, dim))

        y_diff = np.abs(y_dev - y_np)
        y_tol = float(args.atol) + float(args.rtol) * np.abs(y_np)
        y_bad = y_diff > y_tol
        y_bad_count = int(y_bad.sum())
        y_max = float(y_diff.max()) if y_diff.size else 0.0

        s_diff = np.abs(s_dev - s_np)
        s_bad = s_diff > 0.0
        s_bad_count = int(s_bad.sum())
        s_max = float(s_diff.max()) if s_diff.size else 0.0

        print(
            f"Compare vs device: y_bad={y_bad_count}/{y_diff.size} y_max_abs_diff={y_max:.6g} "
            f"conv_states_bad={s_bad_count}/{s_diff.size} conv_states_max_abs_diff={s_max:.6g}"
        )
        return 0 if (y_bad_count == 0 and s_bad_count == 0) else 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
