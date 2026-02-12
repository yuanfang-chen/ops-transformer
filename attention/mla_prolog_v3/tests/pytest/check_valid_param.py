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

import math
import os
import random
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)
_DISCONTINUOUS_MODE_LOGGED = False


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_discontinuous_error_cfg():
    enabled = _env_bool("MLA_PROLOG_V3_ENABLE_DISCONTINUOUS_ERROR", False)
    max_ratio = _env_float("MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_RATIO", 0.001)
    max_ratio = min(max(max_ratio, 0.0), 1.0)
    max_count = _env_int("MLA_PROLOG_V3_DISCONTINUOUS_ERROR_MAX_COUNT", 0)
    max_count = max(0, max_count)
    return enabled, max_ratio, max_count


def log_discontinuous_error_mode_once():
    global _DISCONTINUOUS_MODE_LOGGED
    if _DISCONTINUOUS_MODE_LOGGED:
        return
    enabled, max_ratio, max_count = get_discontinuous_error_cfg()
    if enabled:
        count_desc = str(max_count) if max_count > 0 else "disabled"
        logger.info(
            "[INFO]discontinuous error mode: ON "
            f"(max_ratio={max_ratio}, max_count={count_desc})"
        )
    else:
        logger.info("[INFO]discontinuous error mode: OFF")
    _DISCONTINUOUS_MODE_LOGGED = True


def validate_config(params):
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
            q_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
            weight_quant_mode, kv_quant_mode, query_quant_mode, ckvkr_repo_mode, \
            quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale = params
    # 校验
    if batch_size > 65536 or batch_size < 1:
        raise ValueError("batch_size must <= 65536 and >= 1")
    if q_seq > 16 or q_seq < 1:
        raise ValueError("seq must <= 16 and >= 1")
    if He != 7168:
        raise ValueError("He must = 7168")
    if Hcq != 1536:
        raise ValueError("Hcq must = 1536")
    if q_head_num > 256:
        raise ValueError("q_head_num must <= 256")
    if kv_head_num > 256:
        raise ValueError("kv_head_num must <= 256")
    if q_head_num % kv_head_num:
        raise ValueError("q_head_num should be integral multiple of kv_head_num")
    if Hckv != 512:
        raise ValueError("Hckv must = 512")
    if head_dim != 128:
        raise ValueError("head_dim must = 128")
    if rope_head_dim != 64:
        raise ValueError("rope_head_dim must = 64")
    if kv_head_num != 1:
        raise ValueError("kv_head_num must = 1")
    if block_size != 16 and block_size != 128:
        raise ValueError("block_size must = 16 or = 128")
    if dtype not in [torch.bfloat16, torch.float16, torch.int8]:
        raise ValueError("dtype should be: float16/bfloat16/int8")
    if input_layout not in ["BSH", "BSND", "BNSD"]:
        raise ValueError("input_layout should be: BSH/BSND/BNSD")
    if weight_quant_mode not in [0, 1, 2, 3]:
        raise ValueError("weight_quant_mode should be: 0/1/2/3")
    if kv_quant_mode not in [0, 1, 2, 3]:
        raise ValueError("kv_quant_mode should be: 0/1/2/3")
    if query_quant_mode not in [0, 1]:
        raise ValueError("query_quant_mode should be: 0/1")
    if ckvkr_repo_mode not in [0, 1]:
        raise ValueError("ckvkr_repo_mode should be: 0/1")
    if quant_scale_repo_mode not in [0, 1]:
        raise ValueError("quant_scale_repo_mode should be: 0/1")
    if tile_size <= 0:
        raise ValueError("tile_size must > 0")


def create_random_block_table(batch_size, act_seq_len_kv, block_size):
    # 计算最大序列长度
    max_kv_seq_length = max(act_seq_len_kv)

    # 计算每个序列实际需要的block数量
    valid_block_num = [math.ceil(act_seq_len_kv[i] / block_size) for i in range(batch_size)]
    valid_block_num_sum = sum(valid_block_num)

    # 计算最大block数量
    max_valid_block_num = math.ceil(max_kv_seq_length / block_size)

    # 随机分配block table
    block_table = -torch.ones(batch_size, max_valid_block_num, dtype=torch.int32)
    block_idx = list(range(valid_block_num_sum))
    random.shuffle(block_idx)
    block_idx_start = 0

    for i in range(batch_size):
        block_table[i][: valid_block_num[i]] = torch.tensor(
            block_idx[block_idx_start: block_idx_start + valid_block_num[i]],
            dtype=torch.int32
        )
        block_idx_start += valid_block_num[i]

    return block_table, valid_block_num_sum


def check_result(expect, result):
    discontinuous_mode_enabled, discontinuous_max_ratio, discontinuous_max_count = get_discontinuous_error_cfg()

    def _to_output_list(outputs):
        if isinstance(outputs, torch.Tensor):
            return [outputs]
        if isinstance(outputs, (list, tuple)):
            return list(outputs)
        raise TypeError(f"unsupported output container type: {type(outputs)}")

    def _to_compare_groups(payload):
        if isinstance(payload, dict):
            outputs = _to_output_list(payload.get("outputs", []))
            inplace = _to_output_list(payload.get("inplace", []))
            return outputs, inplace
        return _to_output_list(payload), []

    def _is_float8_dtype(dtype):
        return "float8" in str(dtype)

    def _is_integer_dtype(dtype):
        return dtype in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
        )

    def _dtype_tolerance(expect_dtype, result_dtype):
        # Integer/bool outputs are quantized/discrete and should match exactly.
        if _is_integer_dtype(expect_dtype) and _is_integer_dtype(result_dtype):
            return 0.0, 0.0
        # FP8 has lower precision; use looser tolerance.
        if _is_float8_dtype(expect_dtype) or _is_float8_dtype(result_dtype):
            return 0.2, 0.2
        # BF16/FP16 mixed math.
        if expect_dtype in (torch.float16, torch.bfloat16) or result_dtype in (torch.float16, torch.bfloat16):
            return 0.05, 0.05
        # FP32/FP64 path.
        return 0.001, 0.001

    def _assert_mismatch_policy(group_name, index, mismatch_mask, total_num):
        mismatch_num = int(torch.sum(mismatch_mask).item())
        mismatch_ratio = float(mismatch_num / total_num) if total_num > 0 else 0.0
        if not discontinuous_mode_enabled:
            assert mismatch_num == 0, \
                f"{group_name}[{index}] compare failed, mismatch_num={mismatch_num}, total_num={total_num}"
            return

        within_ratio = mismatch_ratio <= discontinuous_max_ratio
        within_count = (discontinuous_max_count == 0) or (mismatch_num <= discontinuous_max_count)
        assert within_ratio and within_count, (
            f"{group_name}[{index}] compare failed under discontinuous mode, "
            f"mismatch_num={mismatch_num}, total_num={total_num}, mismatch_ratio={mismatch_ratio}, "
            f"max_ratio={discontinuous_max_ratio}, max_count={discontinuous_max_count}"
        )

    def _compare_tensor_group(expect_group, result_group, group_name):
        if len(expect_group) != len(result_group):
            raise AssertionError(
                f"{group_name} count mismatch, expect={len(expect_group)}, result={len(result_group)}"
            )

        for i in range(len(expect_group)):
            expect_cpu = expect_group[i].cpu()
            result_cpu = result_group[i].cpu()

            if result_cpu.shape != expect_cpu.shape:
                raise AssertionError(
                    f"{group_name}[{i}] shape mismatch, expect={tuple(expect_cpu.shape)}, "
                    f"result={tuple(result_cpu.shape)}"
                )

            if result_cpu.numel() == 0 and expect_cpu.numel() == 0:
                logger.info(f"{group_name}[{i}] is empty tensor, skip compare")
                continue

            # Int8 outputs allow per-element error within +/-1.
            if expect_cpu.dtype == torch.int8 and result_cpu.dtype == torch.int8:
                result_i16 = result_cpu.to(torch.int16).reshape(-1)
                expect_i16 = expect_cpu.to(torch.int16).reshape(-1)
                abs_diff = torch.abs(result_i16 - expect_i16)
                mismatch_mask = abs_diff > 1
                mismatch_idx = torch.nonzero(mismatch_mask).reshape(-1)
                if mismatch_idx.numel() > 0:
                    logger.info(f"{group_name}[{i}] mismatch index(sample): {mismatch_idx[:20]}")
                    logger.info(
                        f"{group_name}[{i}] int8 abs diff > 1(sample): {abs_diff[mismatch_mask][:20]}"
                    )
                pass_num = torch.sum(~mismatch_mask)
                total_num = abs_diff.numel()
                accuracy = pass_num / total_num
                logger.info(
                    f"{group_name}[{i}] pass num: {pass_num}, total num: {total_num}, "
                    f"accuracy: {accuracy}, int8_abs_tol=1"
                )
                _assert_mismatch_policy(group_name, i, mismatch_mask, total_num)
                continue

            rtol, atol = _dtype_tolerance(expect_cpu.dtype, result_cpu.dtype)

            # Discrete outputs: strict equality.
            if rtol == 0.0 and atol == 0.0:
                result_i64 = result_cpu.to(torch.int64).reshape(-1)
                expect_i64 = expect_cpu.to(torch.int64).reshape(-1)
                diff = result_i64 - expect_i64
                mismatch_mask = diff != 0
                mismatch_idx = torch.nonzero(mismatch_mask).reshape(-1)
                if mismatch_idx.numel() > 0:
                    logger.info(f"{group_name}[{i}] mismatch index(sample): {mismatch_idx[:20]}")
                    logger.info(f"{group_name}[{i}] mismatch diff(sample): {diff[mismatch_mask][:20]}")
                pass_num = torch.sum(~mismatch_mask)
                total_num = diff.numel()
                accuracy = pass_num / total_num
                logger.info(
                    f"{group_name}[{i}] pass num: {pass_num}, total num: {total_num}, accuracy: {accuracy}"
                )
                _assert_mismatch_policy(group_name, i, mismatch_mask, total_num)
                continue

            result_fp32 = result_cpu.to(torch.float32).reshape(-1)
            expect_fp32 = expect_cpu.to(torch.float32).reshape(-1)
            abs_diff = torch.abs(result_fp32 - expect_fp32)
            allowed = atol + rtol * torch.abs(expect_fp32)
            mismatch_mask = abs_diff > allowed
            mismatch_idx = torch.nonzero(mismatch_mask).reshape(-1)
            if mismatch_idx.numel() > 0:
                logger.info(f"{group_name}[{i}] mismatch index(sample): {mismatch_idx[:20]}")
                logger.info(f"{group_name}[{i}] mismatch abs diff(sample): {abs_diff[mismatch_mask][:20]}")
            pass_num = torch.sum(~mismatch_mask)
            total_num = abs_diff.numel()
            accuracy = pass_num / total_num
            logger.info(
                f"{group_name}[{i}] pass num: {pass_num}, total num: {total_num}, accuracy: {accuracy}, "
                f"rtol={rtol}, atol={atol}"
            )
            _assert_mismatch_policy(group_name, i, mismatch_mask, total_num)

    expect_outputs, expect_inplace = _to_compare_groups(expect)
    result_outputs, result_inplace = _to_compare_groups(result)
    _compare_tensor_group(expect_outputs, result_outputs, "output")
    _compare_tensor_group(expect_inplace, result_inplace, "inplace")
