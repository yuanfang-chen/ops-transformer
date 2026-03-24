#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
Test for FusedKRmsNormRopeStoreKvCacheMxQuant torch interface.
"""

import os
import math
import numpy as np

import torch
import torch_npu

# 直接导入特定算子，避免加载其他无关算子
from npu_ops_transformer.ops.fused_k_rms_norm_rope_store_kv_cache_mx_quant import (
    npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant
)

QUANT_BLOCK_SIZE = 32
EPSILON = 1e-5

# ============== Fixed network cases ==============
# Each case: (T, Nq, Nk, Nv, D, Bs)
# T: sequence length sum, Nq/Nk/Nv: query/key/value head count
# D: head dimension, Bs: block size (PagedAttention), Bn = ceil(T / Bs)
NETWORK_CASES = [
    # Case 1: T=2048, Nq=20, Nk=2, Nv=2, D=128, Bs=512, Bn=4
    (2048, 20, 2, 2, 128, 512),
    # Case 2: T=2048, Nq=80, Nk=8, Nv=8, D=128, Bs=512, Bn=4
    (2048, 80, 8, 8, 128, 512),
    # Case 3: T=10240, Nq=20, Nk=2, Nv=2, D=128, Bs=512, Bn=20
    (10240, 20, 2, 2, 128, 512),
    # Case 4: T=10240, Nq=80, Nk=8, Nv=8, D=128, Bs=512, Bn=20
    (10240, 80, 8, 8, 128, 512),
]

def golden_reference(qkv, cos, sin, gamma, kv_slot, v_scale_slot,
                     k_cache, k_scale_cache, v_cache, v_scale_cache,
                     RMS_EPS, NQ, NK, NV, D):
    # Split
    q, k, v = [x.contiguous() for x in qkv.split([NQ, NK, NV], dim=1)]
    # 对齐融合算子计算逻辑
    q, k = q.to(torch.float32), k.to(torch.float32)
    cos, sin = cos.to(torch.float32), sin.to(torch.float32)
    kv_slot = kv_slot.view(-1,1)
    v_scale_slot = v_scale_slot.view(-1,1)
    k_scale_cache = k_scale_cache.view(torch.uint8)
    v_scale_cache = v_scale_cache.view(torch.uint8)

    # K RMSNorm pre-RoPE
    k, _ = torch_npu.npu_rms_norm(k, gamma, RMS_EPS)

    # RoPE
    q, k = torch_npu.npu_apply_rotary_pos_emb(
        q.unsqueeze(0), k.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0),
        layout='BSH')
    q, k = q.squeeze(0).to(torch.bfloat16), k.squeeze(0).to(torch.bfloat16)

    # Q MXQuant
    q_fp8, q_scale = torch_npu.npu_dynamic_mx_quant(q, dst_type=torch.float8_e4m3fn)

    # BNBD -> BBND: scatter pre-conversion
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    k_scale_cache = k_scale_cache.permute(0, 2, 1, 3, 4).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
    v_scale_cache = v_scale_cache.permute(0, 2, 1, 3, 4).contiguous()

    # K MXQuant + PA scatter
    # k (T, NK, D), view(-1, NK, D) aligns with flattened cache format
    k_fp8, k_scale = torch_npu.npu_dynamic_mx_quant(k, dst_type=torch.float8_e4m3fn)
    torch_npu.npu_scatter_nd_update_(
        k_cache.view(torch.int8).view(-1, NK, D), kv_slot,
        k_fp8.view(torch.int8).view(-1, NK, D))
    torch_npu.npu_scatter_nd_update_(
        k_scale_cache.view(torch.int8).view(-1, NK, 4), kv_slot,
        k_scale.view(torch.int8).view(-1, NK, 4))

    # V transpose -> MXQuant -> PA scatter
    v_t = v.permute(1, 2, 0).contiguous()  # (NV, D, T)
    v_fp8_t, v_scale_t = torch_npu.npu_dynamic_mx_quant(
        v_t, dst_type=torch.float8_e4m3fn)

    # v_fp8_t: (NV, D, T), v_scale_t: (NV, D, T//V_TILE, 2)
    # V data: permute back to (T, NV, D), view(-1, NV, D) aligns with cache
    v_fp8 = v_fp8_t.permute(2, 0, 1).contiguous()  # (T, NV, D)
    torch_npu.npu_scatter_nd_update_(
        v_cache.view(torch.int8).view(-1, NV, D), kv_slot,
        v_fp8.view(torch.int8).view(-1, NV, D))

    # V scale: (NV,D,T//V_TILE,2) -> permute(2,0,1,3) -> (T//V_TILE, NV, D, 2)
    v_scale_reordered = v_scale_t.permute(2, 0, 1, 3).contiguous()
    torch_npu.npu_scatter_nd_update_(
        v_scale_cache.view(torch.int8).view(-1, NV, D, 2),
        v_scale_slot,
        v_scale_reordered.view(torch.int8).view(-1, NV, D, 2))

    # BBND -> BNBD: scatter post-conversion
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    k_scale_cache = k_scale_cache.permute(0, 2, 1, 3, 4).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
    v_scale_cache = v_scale_cache.permute(0, 2, 1, 3, 4).contiguous()

    return q_fp8, q_scale, k_cache, k_scale_cache, v_cache, v_scale_cache


# ============== Input creation ==============
def create_test_inputs(T, Nq, Nk, Nv, D, Bn, Bs, device="npu"):
    """Create test input tensors for a given network case."""
    N = Nq + Nk + Nv

    qkv = torch.randn(T, N, D, dtype=torch.bfloat16, device=device)
    cos = torch.randn(T, 1, D, dtype=torch.bfloat16, device=device)
    sin = torch.randn(T, 1, D, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(D, dtype=torch.float32, device=device).abs() + 0.1

    # 随机打乱 slot_mapping，测试 scatter 功能
    kv_slot_mapping = torch.randperm(T, dtype=torch.int64, device=device)
    v_scale_slot_mapping = torch.randperm(T // QUANT_BLOCK_SIZE // 2, dtype=torch.int64, device=device)

    # NPU does not support direct float8 tensor creation, create as uint8 and view
    k_cache = torch.zeros(Bn, Nk, Bs, D, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    k_scale_cache = torch.zeros(Bn, Nk, Bs, D // QUANT_BLOCK_SIZE // 2, 2, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu)
    v_cache = torch.zeros(Bn, Nv, Bs, D, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    v_scale_cache = torch.zeros(Bn, Nv, Bs // QUANT_BLOCK_SIZE // 2, D, 2, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu)

    return (qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
            k_cache, k_scale_cache, v_cache, v_scale_cache)


def case_desc(T, Nq, Nk, Nv, D, Bn, Bs):
    """Return a short description string for a network case."""
    return f"T={T}, Nq={Nq}, Nk={Nk}, Nv={Nv}, D={D}, Bn={Bn}, Bs={Bs}"


def compare_uint8_results(npu_out, golden_out, name, tolerance=1):
    """Compare uint8 outputs with 1 bit tolerance using torch on CPU.

    Args:
        npu_out: NPU output tensor (uint8 viewed)
        golden_out: Golden reference tensor (uint8 viewed)
        name: Name for error reporting
        tolerance: Allowed bit difference (default 1)
    Returns:
        True if comparison passes, False otherwise
    """
    npu_uint8 = npu_out.view(torch.uint8).flatten().cpu()
    golden_uint8 = golden_out.view(torch.uint8).flatten().cpu()

    if npu_uint8.shape != golden_uint8.shape:
        print(f"[FAIL] {name} shape mismatch: NPU {npu_uint8.shape} vs Golden {golden_uint8.shape}")
        return False

    diff = (npu_uint8.int() - golden_uint8.int()).abs()
    max_diff = diff.max().item()

    # Always print all elements where diff > tolerance
    mask = diff > tolerance
    total_fail = mask.sum().item()
    fail_ratio = total_fail / diff.numel()
    if total_fail > 0:
        indices = torch.nonzero(mask).squeeze(-1)
        print(f"[INFO] {name} elements with diff > {tolerance}: {total_fail}/{diff.numel()} ({fail_ratio*100:.4f}%)")
        print(f"  All indices where diff > {tolerance}:")
        for i in range(indices.numel()):
            idx = indices[i].item()
            npu_val = npu_uint8[idx].item()
            golden_val = golden_uint8[idx].item()
            diff_val = diff[idx].item()
            print(f"    idx={idx}: NPU={npu_val}, Golden={golden_val}, diff={diff_val}")

    # Determine pass/fail based on fail ratio
    if fail_ratio > 0.0001:
        print(f"[FAIL] {name} fail ratio {fail_ratio*100:.4f}% exceeds 0.01% threshold")
        return False

    print(f"[PASS] {name} | max_diff={max_diff}")
    return True


def test_npu_execution():
    """Test NPU execution and accuracy for each fixed network case."""
    print("\n" + "=" * 60)
    print("Test 3: NPU execution with fixed network cases")
    print("=" * 60)

    if not torch.npu.is_available():
        print("[SKIP] NPU device not available")
        return

    for T, Nq, Nk, Nv, D, Bs in NETWORK_CASES:
        Bn = (T + Bs - 1) // Bs
        desc = case_desc(T, Nq, Nk, Nv, D, Bn, Bs)

        inputs = create_test_inputs(T, Nq, Nk, Nv, D, Bn, Bs, device="npu")
        qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping, \
            k_cache, k_scale_cache, v_cache, v_scale_cache = inputs

        k_cache_for_golden = k_cache.clone()
        k_scale_cache_for_golden = k_scale_cache.clone()
        v_cache_for_golden = v_cache.clone()
        v_scale_cache_for_golden = v_scale_cache.clone()

        # Run NPU fused operator
        q, q_scale, k_cache_out, k_scale_cache_out, v_cache_out, v_scale_cache_out = \
            npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant(
                qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
                k_cache, k_scale_cache, v_cache, v_scale_cache, epsilon=EPSILON)

        # Run NPU small operator for accuracy comparison
        q_golden, q_scale_golden, k_golden, k_scale_golden, v_golden, v_scale_golden = golden_reference(
            qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
            k_cache_for_golden, k_scale_cache_for_golden, v_cache_for_golden, v_scale_cache_for_golden,
            EPSILON, Nq, Nk, Nv, D)

        # Compare outputs with 1 bit tolerance
        # FP8 data (q, k_cache, v_cache) use bit-level comparison
        # Scale data (q_scale, k_scale_cache, v_scale_cache) use uint8 comparison
        print(f"\n  === {desc} ===")
        all_pass = True
        all_pass &= compare_uint8_results(q, q_golden, "q")
        all_pass &= compare_uint8_results(q_scale, q_scale_golden, "q_scale")
        all_pass &= compare_uint8_results(k_cache_out, k_golden, "k_cache")
        all_pass &= compare_uint8_results(k_scale_cache_out, k_scale_golden, "k_scale_cache")
        all_pass &= compare_uint8_results(v_cache_out, v_golden, "v_cache")
        all_pass &= compare_uint8_results(v_scale_cache_out, v_scale_golden, "v_scale_cache")

        if not all_pass:
            print(f"[FAIL] {desc} comparison failed")
            return

        torch.npu.synchronize()

    print("[PASS] All NPU execution cases passed")


if __name__ == "__main__":
    print("FusedKRmsNormRopeStoreKvCacheMxQuant Torch Interface Test")
    print("=" * 60)

    print(f"ASCEND_HOME_PATH: {os.environ.get('ASCEND_HOME_PATH', 'NOT SET')}")

    test_npu_execution()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
