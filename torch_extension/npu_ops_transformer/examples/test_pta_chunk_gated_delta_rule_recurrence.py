#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
精度验证测试：chunk_gated_delta_rule_recurrence PTA 接口

用法：
    python3 test_pta_chunk_gated_delta_rule_recurrence.py

依赖：
    torch, torch_npu, npu_ops_transformer（从 torch_extension/ 安装或 sys.path 指向）

测试场景：
    - 默认：b=2, hv=4, cs=16, dk=64, dv=64, nChunks=6（batch0 占 2 chunks，batch1 占 4 chunks）
    - 不等长 batch（可调 cuSeqlens）
    - 多 chunk 递推正确性
"""

import sys
import os
import torch
import torch_npu

# ── 确保可以 import npu_ops_transformer ────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TORCH_EXT_DIR = os.path.join(_SCRIPT_DIR, '..', 'torch_extension')
if os.path.isdir(_TORCH_EXT_DIR):
    sys.path.insert(0, os.path.abspath(_TORCH_EXT_DIR))

from npu_ops_transformer.ops.chunk_gated_delta_rule_recurrence import (
    npu_chunk_gated_delta_rule_recurrence,
)


# ── CPU Golden Reference ────────────────────────────────────────────────────────
def cpu_golden(initial_state, kgexp, value, k_cumdecay, qgexp, gexp, cu_seqlens):
    """
    CPU 参考实现（float64 精度）。

    Shapes:
        initial_state : [b, hv, dv, dk]
        kgexp         : [hv, nC, cs, dk]
        value         : [hv, nC, cs, dv]
        k_cumdecay    : [hv, nC, cs, dk]
        qgexp         : [hv, nC, cs, dk]
        gexp          : [hv, nC, cs,  1]
        cu_seqlens    : [b+1]  (token counts, divisible by cs)

    Returns:
        state_out      [b, hv, dv, dk]
        attn_inter_out [hv, nC, cs, dv]
        v_new_out      [hv, nC, cs, dv]
    """
    b, hv, dv, dk = initial_state.shape
    _, nC, cs, _ = value.shape

    state = initial_state.double().clone()
    attn_inter = torch.zeros_like(value, dtype=torch.float64)
    v_new = torch.zeros_like(value, dtype=torch.float64)
    kgexp_d = kgexp.double()
    value_d = value.double()
    k_cumdecay_d = k_cumdecay.double()
    qgexp_d = qgexp.double()
    gexp_d = gexp.double()

    for bi in range(b):
        chunk_begin = cu_seqlens[bi].item() // cs
        chunk_end   = cu_seqlens[bi + 1].item() // cs

        for h in range(hv):
            for ci in range(chunk_begin, chunk_end):
                s_bh = state[bi, h]              # [dv, dk]
                kcd  = k_cumdecay_d[h, ci]       # [cs, dk]
                qg   = qgexp_d[h, ci]            # [cs, dk]
                kg   = kgexp_d[h, ci]            # [cs, dk]
                val  = value_d[h, ci]            # [cs, dv]
                g_sc = gexp_d[h, ci, cs - 1, 0] # scalar

                # C1: v_prime [cs, dv] = kcd @ s_bh.T
                v_prime = kcd @ s_bh.T           # [cs, dk] @ [dk, dv] = [cs, dv]

                # C2: attn_inter [cs, dv] = qg @ s_bh.T
                attn_inter[h, ci] = (qg @ s_bh.T).float()

                # V1: v_new = value - v_prime
                vn = val - v_prime               # [cs, dv]
                v_new[h, ci] = vn.float()

                # V0: state *= g_scalar
                s_bh.mul_(g_sc)

                # C3: state += v_new.T @ kgexp  → [dv, cs] @ [cs, dk] = [dv, dk]
                s_bh.add_(vn.T @ kg)

    return state.float(), attn_inter.float(), v_new.float()


# ── allclose comparison ─────────────────────────────────────────────────────────
def allclose_report(npu_t, golden_t, atol, rtol, name):
    npu = npu_t.cpu().float()
    gold = golden_t.cpu().float()
    if npu.shape != gold.shape:
        print(f"[FAIL] {name}: shape mismatch {npu.shape} vs {gold.shape}")
        return False
    abs_err = (npu - gold).abs()
    rel_err = abs_err / (gold.abs() + 1e-6)
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    mask = abs_err > (atol + rtol * gold.abs())
    fail_cnt = mask.sum().item()
    if fail_cnt > 0:
        idxs = mask.nonzero(as_tuple=False)[:5]
        for idx in idxs:
            i = tuple(idx.tolist())
            print(f"  {i}: npu={npu[i]:.6f}  gold={gold[i]:.6f}  "
                  f"abs={abs_err[i]:.2e}  rel={rel_err[i]:.2e}")
    passed = fail_cnt == 0
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:<24}  maxAbsErr={max_abs:.2e}  maxRelErr={max_rel:.2e}  fails={int(fail_cnt)}")
    return passed


# ── single test case ────────────────────────────────────────────────────────────
def run_test(b, hv, dk, dv, cs, cu_seqlens_tokens, label, scale_value=1.0, seed=42):
    """
    单次测试：构造随机输入 → CPU golden → NPU op → allclose 比较。
    cu_seqlens_tokens: list of ints, 长度 b+1, 值为 token 数（必须整除 cs）
    """
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"  b={b}, hv={hv}, dk={dk}, dv={dv}, cs={cs}, scale={scale_value}")
    print(f"  cu_seqlens={cu_seqlens_tokens}")

    torch.manual_seed(seed)
    nC = cu_seqlens_tokens[-1] // cs  # total chunk count

    # Generate host tensors (float32)
    state_h   = torch.randn(b, hv, dv, dk,  dtype=torch.float32) * 0.1
    kgexp_h   = torch.randn(hv, nC, cs, dk, dtype=torch.float32) * 0.1
    value_h   = torch.randn(hv, nC, cs, dv, dtype=torch.float32) * 0.1
    kcd_h     = torch.randn(hv, nC, cs, dk, dtype=torch.float32) * 0.1
    qgexp_h   = torch.randn(hv, nC, cs, dk, dtype=torch.float32) * 0.1
    gexp_h    = torch.rand (hv, nC, cs,  1, dtype=torch.float32) * 0.5 + 0.5  # (0.5, 1.0]
    cuseq_h   = torch.tensor(cu_seqlens_tokens, dtype=torch.int32)

    # CPU golden (float64 precision)
    gold_state, gold_attn, gold_vnew = cpu_golden(
        state_h.clone(), kgexp_h, value_h, kcd_h, qgexp_h, gexp_h, cuseq_h)

    # Upload to NPU
    device = torch.device('npu:0')
    state_npu  = state_h.clone().to(device)
    kgexp_npu  = kgexp_h.to(device)
    value_npu  = value_h.to(device)
    kcd_npu    = kcd_h.to(device)
    qgexp_npu  = qgexp_h.to(device)
    gexp_npu   = gexp_h.to(device)
    cuseq_npu  = cuseq_h.to(device)

    # Run NPU op
    state_out, attn_out, vnew_out = npu_chunk_gated_delta_rule_recurrence(
        state_npu, kgexp_npu, value_npu, kcd_npu, qgexp_npu, gexp_npu, cuseq_npu,
        scale_value
    )

    # Compare (float32 tolerance)
    ATOL, RTOL = 1e-4, 1e-3
    p1 = allclose_report(attn_out,  gold_attn,  ATOL, RTOL, "attn_inter_out")
    p2 = allclose_report(vnew_out,  gold_vnew,  ATOL, RTOL, "v_new_out")
    p3 = allclose_report(state_out, gold_state, ATOL, RTOL, "initial_state")

    passed = p1 and p2 and p3
    print(f"=> 结论：{'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    torch_npu.npu.set_device(0)

    results = []

    # Case 1: 默认参数（batch0=2chunks, batch1=4chunks, 共6chunks）
    results.append(run_test(
        b=2, hv=4, dk=64, dv=64, cs=16,
        cu_seqlens_tokens=[0, 2 * 16, 6 * 16],
        label="Case1: default (b=2,hv=4,dk=64,dv=64,cs=16,nC=6)",
    ))

    # Case 2: 单 batch 单 head（边界）
    results.append(run_test(
        b=1, hv=1, dk=32, dv=32, cs=8,
        cu_seqlens_tokens=[0, 4 * 8],
        label="Case2: b=1, hv=1 (boundary: min tasks)",
    ))

    # Case 3: 不等长 batch（batch0=1chunk, batch1=5chunks）
    results.append(run_test(
        b=2, hv=2, dk=64, dv=64, cs=16,
        cu_seqlens_tokens=[0, 1 * 16, 6 * 16],
        label="Case3: unequal batch lengths (1+5 chunks)",
    ))

    # Case 4: 较大 dk/dv（验证 dvTile 多轮循环）
    results.append(run_test(
        b=2, hv=2, dk=128, dv=128, cs=16,
        cu_seqlens_tokens=[0, 3 * 16, 6 * 16],
        label="Case4: larger dk=128, dv=128 (dvTile loop)",
    ))

    # Case 5: 多 chunk（验证跨 chunk 状态递推）
    results.append(run_test(
        b=1, hv=2, dk=64, dv=64, cs=8,
        cu_seqlens_tokens=[0, 12 * 8],
        label="Case5: many chunks (nC=12, cross-chunk state update)",
    ))

    print(f"\n{'='*60}")
    total = len(results)
    passed_n = sum(results)
    print(f"总计：{passed_n}/{total} 通过")
    sys.exit(0 if passed_n == total else 1)


if __name__ == '__main__':
    main()
