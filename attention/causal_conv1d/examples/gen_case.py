#!/usr/bin/env python3
"""
Generate a CausalConv1d (prefill/extend varlen) test case and save inputs in a
Python-friendly `.bin` format.

These files match what `test_aclnn_causal_conv1d.cpp` dumps into `./case/`.

File format:
  - Float tensors are stored as float32 .bin (little-endian), but values are fp16-quantized.
  - Int tensors are stored as int32 .bin (little-endian).
  - Bool tensor `has_initial_state` is stored as uint8 .bin.

Produced files in out_dir:
  x.bin, weight.bin, bias.bin, conv_states.bin, query_start_loc.bin, cache_indices.bin, has_initial_state.bin, meta.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


F32 = np.dtype("<f4")
I32 = np.dtype("<i4")
U8 = np.dtype("u1")


def quantize_fp16_to_f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(F32, copy=False)


def write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(str(path))


def write_meta_txt(path: Path, meta: Dict[str, int]) -> None:
    lines = [
        f"batch_size={meta['batch_size']}",
        f"total_seqlen={meta['total_seqlen']}",
        f"dim={meta['dim']}",
        f"width={meta['width']}",
        f"activation_mode={meta['activation_mode']}",
        f"pad_slot_id={meta['pad_slot_id']}",
        "file_float_dtype=float32",
        "note=All float tensors are stored as float32 .bin; values are fp16-quantized.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_varlen_qsl(total_seqlen: int, batch_size: int) -> np.ndarray:
    qsl = np.zeros((batch_size + 1,), dtype=I32)
    base = total_seqlen // batch_size
    rem = total_seqlen % batch_size
    for i in range(batch_size):
        extra = 1 if i < rem else 0
        qsl[i + 1] = qsl[i] + base + extra
    return qsl


def generate_default_patterns(
    batch_size: int,
    total_seqlen: int,
    dim: int,
    width: int,
    init_state_seqs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # x[t, :] = (t+1)/total_seqlen
    t = (np.arange(total_seqlen, dtype=np.float32) + 1.0) / float(total_seqlen)
    x = np.repeat(t[:, None], dim, axis=1).astype(np.float32, copy=False)
    x = quantize_fp16_to_f32(x)

    # weight[j, c] = c + j*200
    c = np.arange(dim, dtype=np.float32)
    weight = np.stack([c + float(j * 200) for j in range(width)], axis=0)
    weight = quantize_fp16_to_f32(weight)

    # bias[c] = 1000 + c
    bias = (1000.0 + np.arange(dim, dtype=np.float32)).astype(np.float32, copy=False)
    bias = quantize_fp16_to_f32(bias)

    # conv_states[seq, pos, c] = ((pos+1)*10000 + seq*100 + c) / 10000 for seq < init_state_seqs else 0
    conv_states = np.zeros((batch_size, width - 1, dim), dtype=np.float32)
    for seq in range(min(init_state_seqs, batch_size)):
        for pos in range(width - 1):
            base = float((pos + 1) * 10000 + seq * 100)
            conv_states[seq, pos, :] = (base + np.arange(dim, dtype=np.float32)) / 10000.0
    conv_states = quantize_fp16_to_f32(conv_states)

    query_start_loc = build_varlen_qsl(total_seqlen, batch_size)
    cache_indices = np.arange(batch_size, dtype=I32)
    has_initial_state = np.zeros((batch_size,), dtype=U8)
    has_initial_state[: min(init_state_seqs, batch_size)] = 1

    return x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--total_seqlen", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--init_state_seqs", type=int, default=1)
    ap.add_argument("--activation_mode", type=int, default=1)
    ap.add_argument("--pad_slot_id", type=int, default=-1)
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("batch_size must be > 0")
    if args.total_seqlen <= 0:
        raise SystemExit("total_seqlen must be > 0")
    if args.dim <= 0:
        raise SystemExit("dim must be > 0")
    if args.width <= 0:
        raise SystemExit("width must be > 0")
    if args.width != 4:
        raise SystemExit("width must be 4 for this generator (kernel design assumes width=4)")
    if args.init_state_seqs < 0 or args.init_state_seqs > args.batch_size:
        raise SystemExit("init_state_seqs must be in [0, batch_size]")

    x, weight, bias, conv_states, qsl, cache_indices, has_initial_state = generate_default_patterns(
        batch_size=args.batch_size,
        total_seqlen=args.total_seqlen,
        dim=args.dim,
        width=args.width,
        init_state_seqs=args.init_state_seqs,
    )

    out_dir: Path = args.out_dir
    write_meta_txt(
        out_dir / "meta.txt",
        {
            "batch_size": int(args.batch_size),
            "total_seqlen": int(args.total_seqlen),
            "dim": int(args.dim),
            "width": int(args.width),
            "activation_mode": int(args.activation_mode),
            "pad_slot_id": int(args.pad_slot_id),
        },
    )

    write_bin(out_dir / "x.bin", x.astype(F32, copy=False))
    write_bin(out_dir / "weight.bin", weight.astype(F32, copy=False))
    write_bin(out_dir / "bias.bin", bias.astype(F32, copy=False))
    write_bin(out_dir / "conv_states.bin", conv_states.astype(F32, copy=False))
    write_bin(out_dir / "query_start_loc.bin", qsl.astype(I32, copy=False))
    write_bin(out_dir / "cache_indices.bin", cache_indices.astype(I32, copy=False))
    write_bin(out_dir / "has_initial_state.bin", has_initial_state.astype(U8, copy=False))

    print(f"Wrote case to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
