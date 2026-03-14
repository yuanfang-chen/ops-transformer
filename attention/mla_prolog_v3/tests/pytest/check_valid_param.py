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


def check_single_output(name, expect, result, rtol=0.05, atol=0.05, thres=0.005):
    """比较单个输出张量的精度。"""
    result_cpu = result.cpu().to(torch.float32)
    expect_cpu = expect.cpu().to(torch.float32)

    diff = result_cpu.reshape(-1) - expect_cpu.reshape(-1)
    idx = torch.nonzero(abs(diff) > thres).squeeze()
    values = diff[abs(diff) > thres]

    if idx.numel() > 0:
        logger.info(f"[{name}] diff大于{thres}的元素数: {idx.numel()}")
        if idx.numel() <= 10:
            logger.info(f"[{name}] diff大于{thres}的元素索引: {idx}")
            logger.info(f"[{name}] diff大于{thres}的元素值: {values}")

    pass_num = torch.sum(abs(diff.squeeze()) <= thres)
    total_num = diff.numel()
    accuracy = pass_num / total_num
    logger.info(f"[{name}] pass num: {pass_num}, total num: {total_num}, accuracy: {accuracy:.6f}")

    assert torch.allclose(result_cpu.flatten(), expect_cpu.flatten(), rtol=rtol, atol=atol), \
        f"[{name}] allclose failed (rtol={rtol}, atol={atol})"


def check_result(expect_list, result_list):
    """比较 CPU 侧 Golden 与 NPU 侧结果的精度。

    Args:
        expect_list: cal_mlaprolog 返回的 tuple (out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm)
        result_list: torch_npu.npu_mla_prolog_v3 返回的 tuple
    """
    output_names = ["queryOut", "queryRopeOut", "kvCache", "krCache",
                    "deqScaleQNope", "queryNorm", "deqScaleQNorm"]

    # 从 expect_list 中过滤 None 值
    expected_outputs = []
    expected_names = []
    for i, (name, val) in enumerate(zip(output_names, expect_list)):
        if val is not None:
            expected_outputs.append(val)
            expected_names.append(name)

    # result_list 长度可能不一致（NPU 只返回非 None 输出）
    assert len(expected_outputs) == len(result_list), \
        f"Output count mismatch: expected {len(expected_outputs)}, got {len(result_list)}"

    for name, expect, result in zip(expected_names, expected_outputs, result_list):
        check_single_output(name, expect, result)
