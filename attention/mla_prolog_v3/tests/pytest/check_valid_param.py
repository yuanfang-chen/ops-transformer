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
# 精度标准（来自遗留框架）
# ---------------------------------------------------------------------------

# 每个输出的精度参数（共 7 项，对应 7 个输出）
# 来源：precision_param_list（第 5、6 项复用 params[5]）
#   diff_thd    : 相对误差阈值，超过该值的元素计为"离散误差点"
#   pct_thd     : 最大允许的离散误差点比例
#   max_diff_thd: 允许的最大相对误差（绝对上限）
#   rtol        : 相对容差（用于 max/avg/rmse 统计检验）
#   atol        : 绝对容差（整数/FP8 类型的判定阈值）
OUTPUT_PRECISION_PARAMS = [
    # [0] queryOut
    dict(diff_thd=0.005, pct_thd=0.005, max_diff_thd=10.0, rtol=0.0078125, atol=0.0001),
    # [1] queryRopeOut
    dict(diff_thd=0.005, pct_thd=0.005, max_diff_thd=10.0, rtol=0.0078125, atol=0.0001),
    # [2] kvCache
    dict(diff_thd=0.005, pct_thd=0.005, max_diff_thd=10.0, rtol=0.0078125, atol=0.0001),
    # [3] krCache
    dict(diff_thd=0.005, pct_thd=0.005, max_diff_thd=10.0, rtol=0.0078125, atol=0.0001),
    # [4] deqScaleQNope
    dict(diff_thd=0.005, pct_thd=0.005, max_diff_thd=10.0, rtol=0.005,     atol=2.5e-05),
    # [5] queryNorm
    dict(diff_thd=0.005, pct_thd=0.005, max_diff_thd=10.0, rtol=0.005,     atol=2.5e-05),
    # [6] deqScaleQNorm（复用 params[5]）
    dict(diff_thd=0.005, pct_thd=0.005, max_diff_thd=10.0, rtol=0.005,     atol=2.5e-05),
]

# 浮点类型的统计精度标准（来自 bm_cmp_std）
# 对相对误差 RE = |diff| / clamp(|expect|, small_value) 计算统计量：
#   max_re  <= max_re_rtol  * rtol
#   avg_re  <= avg_re_rtol  * rtol
#   rmse    <= rmse_rtol    * rtol
# 对接近零的元素（|expect| < small_value），允许使用绝对误差 small_value_atol 替代
BM_CMP_STD = {
    torch.float32: dict(
        max_re_rtol=10.0, avg_re_rtol=2.0, rmse_rtol=2.0,
        small_value=0.0009765625, small_value_atol=1.52587890625e-05,
    ),
    torch.float16: dict(
        max_re_rtol=10.0, avg_re_rtol=2.0, rmse_rtol=2.0,
        small_value=0.0009765625, small_value_atol=1.52587890625e-05,
    ),
    torch.bfloat16: dict(
        max_re_rtol=10.0, avg_re_rtol=2.0, rmse_rtol=2.0,
        small_value=0.0009765625, small_value_atol=1.52587890625e-05,
    ),
}


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

def check_single_output(name, expect, result, prec_params, pct_thd_override=None):
    """比较单个输出张量的精度。

    对浮点类型（fp32/fp16/bf16）使用基于相对误差（RE）的统计检验：
      RE_k = |result_k - expect_k| / max(|expect_k|, small_value)
      1. PCT 检验：fraction(RE > diff_thd) <= pct_thd
      2. Max 检验：max(RE) <= max_diff_thd  且  max(RE) <= max_re_rtol * rtol
      3. Avg 检验：mean(RE) <= avg_re_rtol * rtol
      4. RMSE 检验：sqrt(mean(RE²)) <= rmse_rtol * rtol
      对近零元素（|expect| < small_value），允许 |diff| <= small_value_atol 替代相对误差通过。

    对整数/FP8 类型使用绝对误差检验：
      fail_mask = |diff| > atol
      PCT 检验：fraction(fail) <= pct_thd

    Args:
        name:             输出张量名称（用于日志）。
        expect:           CPU Golden 张量（任意设备/dtype）。
        result:           NPU 实际输出张量（任意设备/dtype）。
        prec_params:      OUTPUT_PRECISION_PARAMS 中对应的精度参数字典。
        pct_thd_override: 若设置，覆盖 prec_params['pct_thd']（用于调试）。
    """
    native_dtype = expect.dtype
    diff_thd     = prec_params['diff_thd']
    pct_thd      = prec_params['pct_thd'] if pct_thd_override is None else pct_thd_override
    max_diff_thd = prec_params['max_diff_thd']
    rtol         = prec_params['rtol']
    atol         = prec_params['atol']

    result_f32 = result.cpu().to(torch.float32).reshape(-1)
    expect_f32 = expect.cpu().to(torch.float32).reshape(-1)
    abs_diff   = (result_f32 - expect_f32).abs()
    total      = abs_diff.numel()

    cmp_std = BM_CMP_STD.get(native_dtype)

    if cmp_std is not None:
        # ---- 浮点类型：相对误差统计检验 ----
        small_value     = cmp_std['small_value']
        small_value_atol = cmp_std['small_value_atol']
        max_re_rtol     = cmp_std['max_re_rtol']
        avg_re_rtol     = cmp_std['avg_re_rtol']
        rmse_rtol       = cmp_std['rmse_rtol']

        denom = expect_f32.abs().clamp(min=small_value)
        re    = abs_diff / denom

        # 近零元素：|expect| < small_value 且 |diff| <= small_value_atol → 视为通过
        near_zero_pass = (expect_f32.abs() < small_value) & (abs_diff <= small_value_atol)

        # PCT 检验：RE > diff_thd 且非近零通过 → 计为离散误差点
        fail_mask  = (re > diff_thd) & ~near_zero_pass
        fail_count = int(fail_mask.sum())
        error_rate = fail_count / total if total > 0 else 0.0

        max_re = float(re.max())
        avg_re = float(re.mean())
        rmse   = float(re.pow(2).mean().sqrt())

        logger.info(
            f"[{name}] dtype={native_dtype} "
            f"max_re={max_re:.6f}(thd={max_re_rtol * rtol:.6f}) "
            f"avg_re={avg_re:.6f}(thd={avg_re_rtol * rtol:.6f}) "
            f"rmse={rmse:.6f}(thd={rmse_rtol * rtol:.6f}) "
            f"pct={error_rate:.4%}(thd={pct_thd:.4%})"
        )

        if fail_count > 0:
            worst = torch.topk(abs_diff[fail_mask], min(5, fail_count)).indices
            fail_indices = fail_mask.nonzero(as_tuple=True)[0]
            sample = fail_indices[worst]
            logger.info(
                f"[{name}] worst discontinuous: "
                f"re={re[sample].tolist()}, "
                f"expect={expect_f32[sample].tolist()}, "
                f"result={result_f32[sample].tolist()}"
            )

        # 1. PCT 检验
        assert error_rate <= pct_thd, (
            f"[{name}] PCT failed: {error_rate:.4%} > pct_thd={pct_thd:.4%} "
            f"({fail_count}/{total} elements with RE > diff_thd={diff_thd})"
        )
        # 2. Max diff 检验
        assert max_re <= max_diff_thd, (
            f"[{name}] max RE {max_re:.6f} > max_diff_thd={max_diff_thd}"
        )
        # 3. Max RE 统计检验
        assert max_re <= max_re_rtol * rtol, (
            f"[{name}] max RE {max_re:.6f} > max_re_rtol*rtol={max_re_rtol * rtol:.6f}"
        )
        # 4. Avg RE 统计检验
        assert avg_re <= avg_re_rtol * rtol, (
            f"[{name}] avg RE {avg_re:.6f} > avg_re_rtol*rtol={avg_re_rtol * rtol:.6f}"
        )
        # 5. RMSE 统计检验
        assert rmse <= rmse_rtol * rtol, (
            f"[{name}] RMSE {rmse:.6f} > rmse_rtol*rtol={rmse_rtol * rtol:.6f}"
        )

    else:
        # ---- 整数 / FP8 类型：绝对误差 + PCT 检验 ----
        fail_mask  = abs_diff > atol
        fail_count = int(fail_mask.sum())
        error_rate = fail_count / total if total > 0 else 0.0

        logger.info(
            f"[{name}] dtype={native_dtype} atol={atol} "
            f"pct={error_rate:.4%}(thd={pct_thd:.4%}) "
            f"fail={fail_count}/{total}"
        )

        if fail_count > 0:
            worst = torch.topk(abs_diff, min(5, fail_count)).indices
            logger.info(
                f"[{name}] worst diffs: "
                f"abs_diff={abs_diff[worst].tolist()}, "
                f"expect={expect_f32[worst].tolist()}, "
                f"result={result_f32[worst].tolist()}"
            )

        assert error_rate <= pct_thd, (
            f"[{name}] PCT failed: {error_rate:.4%} > pct_thd={pct_thd:.4%} "
            f"({fail_count}/{total} elements with abs_diff > atol={atol})"
        )


def check_result(expect_list, result_list, pct_thd_override=None):
    """比较 CPU 侧 Golden 与 NPU 侧结果的精度。

    输出列表顺序固定为：
      [0] queryOut       — BF16 / INT8（query_quant_mode=1 时）
      [1] queryRopeOut   — BF16
      [2] kvCache        — BF16 / INT8 / FP8（依 kv_cache_quant_mode）
      [3] krCache        — BF16 / INT8 / FP8（依 ckvkr_repo_mode）
      [4] deqScaleQNope  — FP32（可选，torch.empty(0) 表示未启用）
      [5] queryNorm      — INT8 / FP8（可选）
      [6] deqScaleQNorm  — FP32（可选）

    精度标准由 OUTPUT_PRECISION_PARAMS（per-output）和 BM_CMP_STD（per-dtype）共同决定。

    Args:
        expect_list:      来自 cal_mlaprolog 的输出列表。
        result_list:      来自 test_mla_prolog_v3 的输出列表（kv_cache/kr_cache 已插入 [2]/[3]）。
        pct_thd_override: 若设置，覆盖每个输出的 pct_thd（用于调试放宽精度要求）。
    """
    output_names = ["queryOut", "queryRopeOut", "kvCache", "krCache",
                    "deqScaleQNope", "queryNorm", "deqScaleQNorm"]

    assert len(expect_list) == len(result_list), (
        f"Output count mismatch: expect {len(expect_list)}, got {len(result_list)}"
    )
    assert len(output_names) == len(OUTPUT_PRECISION_PARAMS)

    for i, (name, expect, result) in enumerate(zip(output_names, expect_list, result_list)):
        # 跳过双方均为空的可选输出
        if expect.numel() == 0 and result.numel() == 0:
            logger.info(f"[{name}] skipped (both empty)")
            continue
        check_single_output(
            name, expect, result,
            prec_params=OUTPUT_PRECISION_PARAMS[i],
            pct_thd_override=pct_thd_override,
        )
