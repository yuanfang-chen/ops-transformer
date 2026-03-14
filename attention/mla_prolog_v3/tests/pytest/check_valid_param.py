#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

# 合法的 head_num 取值
VALID_HEAD_NUMS = {1, 2, 4, 8, 16, 32, 64, 128}
# 合法的 He 取值
VALID_HE_VALUES = {1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192}

# ---------------------------------------------------------------------------
# 精度标准：按原始张量数据类型确定 rtol / atol
# ---------------------------------------------------------------------------
# BF16/FP16：累积误差较小，适中容差
# INT8：量化结果为整数，允许 ±1 误差（atol=1, rtol=0）
# FP8 (E4M3FN)：量化精度较低，容差较大
# FP32/FP64：高精度，严格容差
# UINT8 / INT32：整数类型，同 INT8 处理
DTYPE_TOLERANCES = {
    torch.bfloat16:      dict(rtol=0.01,  atol=0.02),
    torch.float16:       dict(rtol=0.01,  atol=0.02),
    torch.float32:       dict(rtol=1e-4,  atol=1e-4),
    torch.float64:       dict(rtol=1e-5,  atol=1e-5),
    torch.int8:          dict(rtol=0.0,   atol=1.0),
    torch.int32:         dict(rtol=0.0,   atol=1.0),
    torch.uint8:         dict(rtol=0.0,   atol=1.0),
    torch.float8_e4m3fn: dict(rtol=0.05,  atol=0.1),
}
_DEFAULT_TOL = dict(rtol=0.05, atol=0.05)


def _get_tol(dtype):
    """根据张量原始数据类型返回 (rtol, atol)。"""
    tol = DTYPE_TOLERANCES.get(dtype, _DEFAULT_TOL)
    return tol['rtol'], tol['atol']


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def validate_config(params):
    """校验 mla_prolog_v3 算子的参数约束。"""
    batch_size = params['batch_size']
    head_num = params['head_num']
    He = params['He']
    dtype = params['dtype']
    cache_mode = params['cache_mode']
    block_size = params['block_size']
    weight_quant_mode = params['weight_quant_mode']
    kv_cache_quant_mode = params['kv_cache_quant_mode']
    query_quant_mode = params['query_quant_mode']
    ckvkr_repo_mode = params['ckvkr_repo_mode']
    quant_scale_repo_mode = params['quant_scale_repo_mode']

    if batch_size > 65536:
        raise ValueError("batch_size (B) must <= 65536")
    if head_num not in VALID_HEAD_NUMS:
        raise ValueError(f"head_num (N) must be in {VALID_HEAD_NUMS}, got {head_num}")
    if He not in VALID_HE_VALUES:
        raise ValueError(f"He must be in {VALID_HE_VALUES}, got {He}")
    if dtype not in [torch.bfloat16, torch.int8, torch.float8_e4m3fn]:
        raise ValueError("dtype should be: bfloat16/int8/float8_e4m3fn")
    if cache_mode not in ["PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"]:
        raise ValueError(f"cache_mode not supported: {cache_mode}")
    if weight_quant_mode not in [0, 1, 2, 3]:
        raise ValueError(f"weight_quant_mode must be 0/1/2/3, got {weight_quant_mode}")
    if kv_cache_quant_mode not in [0, 1, 2, 3]:
        raise ValueError(f"kv_cache_quant_mode must be 0/1/2/3, got {kv_cache_quant_mode}")
    if query_quant_mode not in [0, 1]:
        raise ValueError(f"query_quant_mode must be 0/1, got {query_quant_mode}")
    if query_quant_mode == 1 and weight_quant_mode not in [2, 3]:
        raise ValueError("query_quant_mode=1 requires weight_quant_mode in [2, 3]")
    if kv_cache_quant_mode == 3:
        if ckvkr_repo_mode != 1:
            raise ValueError("kv_cache_quant_mode=3 requires ckvkr_repo_mode=1")
        if quant_scale_repo_mode != 1:
            raise ValueError("kv_cache_quant_mode=3 requires quant_scale_repo_mode=1")
    if block_size < 16 or block_size > 1024 or block_size % 16 != 0:
        raise ValueError(f"block_size must be in [16, 1024] and multiple of 16, got {block_size}")


# ---------------------------------------------------------------------------
# 精度比较
# ---------------------------------------------------------------------------

def check_single_output(name, expect, result, rtol=None, atol=None, max_error_rate=0.0):
    """比较单个输出张量的精度。

    Args:
        name:           输出张量名称（用于日志）。
        expect:         CPU Golden 张量（任意设备/dtype）。
        result:         NPU 实际输出张量（任意设备/dtype）。
        rtol:           相对误差容限。为 None 时根据 expect.dtype 自动推断。
        atol:           绝对误差容限。为 None 时根据 expect.dtype 自动推断。
        max_error_rate: 最大允许的离散误差点比例（0.0 = 严格模式，全部元素必须通过；
                        >0.0 = 离散误差模式，允许该比例以内的元素超出容限）。
    """
    native_dtype = expect.dtype

    # 根据原始 dtype 自动推断容差
    auto_rtol, auto_atol = _get_tol(native_dtype)
    if rtol is None:
        rtol = auto_rtol
    if atol is None:
        atol = auto_atol

    # 转为 float32 进行比较
    result_f32 = result.cpu().to(torch.float32).reshape(-1)
    expect_f32 = expect.cpu().to(torch.float32).reshape(-1)

    # 按照 torch.allclose 公式计算每个元素的误差：|result - expect| <= atol + rtol*|expect|
    abs_diff = (result_f32 - expect_f32).abs()
    threshold = atol + rtol * expect_f32.abs()
    fail_mask = abs_diff > threshold

    fail_count = int(fail_mask.sum())
    total = abs_diff.numel()
    error_rate = fail_count / total if total > 0 else 0.0

    logger.info(
        f"[{name}] dtype={native_dtype} rtol={rtol} atol={atol} "
        f"fail={fail_count}/{total} error_rate={error_rate:.4%}"
    )

    # 记录最差的几个离散误差点（便于调试）
    if fail_count > 0:
        worst_indices = torch.topk(abs_diff, min(5, fail_count)).indices
        logger.info(
            f"[{name}] worst diffs: "
            f"abs_diff={abs_diff[worst_indices].tolist()}, "
            f"expect={expect_f32[worst_indices].tolist()}, "
            f"result={result_f32[worst_indices].tolist()}"
        )

    if max_error_rate == 0.0:
        # 严格模式：全部元素必须通过
        assert fail_count == 0, (
            f"[{name}] precision check failed: {fail_count}/{total} elements exceed "
            f"tolerance (rtol={rtol}, atol={atol})"
        )
    else:
        # 离散误差模式：允许 max_error_rate 比例以内的点超出容限
        assert error_rate <= max_error_rate, (
            f"[{name}] error rate {error_rate:.4%} exceeds max_error_rate {max_error_rate:.4%} "
            f"({fail_count}/{total} failing elements)"
        )
        if fail_count > 0:
            logger.info(
                f"[{name}] {fail_count} discontinuous error(s) tolerated "
                f"(max_error_rate={max_error_rate:.4%})"
            )


def check_result(expect_list, result_list, max_error_rate=0.0):
    """比较 CPU 侧 Golden 与 NPU 侧结果的精度。

    输出列表顺序固定为：
      [0] queryOut       — BF16 / INT8（query_quant_mode=1 时）
      [1] queryRopeOut   — BF16
      [2] kvCache        — BF16 / INT8 / FP8（依 kv_cache_quant_mode）
      [3] krCache        — BF16 / INT8 / FP8（依 ckvkr_repo_mode）
      [4] deqScaleQNope  — FP32 / FP64（可选，torch.empty(0) 表示未启用）
      [5] queryNorm      — INT8 / FP8（可选）
      [6] deqScaleQNorm  — FP32 / FP64（可选）

    精度容差由各张量的原始 dtype 自动推断，无需手动指定。

    Args:
        expect_list:     来自 cal_mlaprolog 的输出列表。
        result_list:     来自 test_mla_prolog_v3 的输出列表（kv_cache/kr_cache 已插入 [2]/[3]）。
        max_error_rate:  传递给每个 check_single_output 的离散误差比例阈值。
                         0.0 = 严格模式；>0.0 = 离散误差模式（如 0.01 表示允许 1% 的点超出容限）。
    """
    output_names = ["queryOut", "queryRopeOut", "kvCache", "krCache",
                    "deqScaleQNope", "queryNorm", "deqScaleQNorm"]

    assert len(expect_list) == len(result_list), (
        f"Output count mismatch: expect {len(expect_list)}, got {len(result_list)}"
    )

    for name, expect, result in zip(output_names, expect_list, result_list):
        # 跳过双方均为空的可选输出
        if expect.numel() == 0 and result.numel() == 0:
            logger.info(f"[{name}] skipped (both empty)")
            continue
        check_single_output(name, expect, result, max_error_rate=max_error_rate)
