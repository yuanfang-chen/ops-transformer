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

import os
import torch

# 固定算子维度（始终为常数）
# Hcq: 1536, Hckv: 512, D: 128, Dr: 64, kv_head_num (N2): 1

# 定义测试参数组合
TEST_PARAMS = {
    # 基础场景 - CI 冒烟测试（无量化）
    "base_default": {
        "batch_size": [2],
        "seq_len": [4],
        "head_num": [8],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # INT8 权重量化场景
    "int8_weight_quant": {
        "batch_size": [2],
        "seq_len": [4],
        "head_num": [8],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [1, 2],
        "kv_cache_quant_mode": [0, 1, 2],
        "query_quant_mode": [0, 1],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # MXFP8 量化场景
    "mxfp8_quant": {
        "batch_size": [2],
        "seq_len": [4],
        "head_num": [8],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [3],
        "kv_cache_quant_mode": [0, 1, 3],
        "query_quant_mode": [0, 1],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # 逐块 KV 量化场景
    "per_tile_kv_quant": {
        "batch_size": [2],
        "seq_len": [4],
        "head_num": [8],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [3],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [1],
        "quant_scale_repo_mode": [1],
    },
    # 性能分析场景 - 典型推理工作负载
    # DeepSeek-V3 decode: BS=1, S=1, N=128, He=7168 (单token推理)
    "perf_decode_bs1": {
        "batch_size": [1],
        "seq_len": [1],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # DeepSeek-V3 decode: BS=8, S=1 (多batch推理)
    "perf_decode_bs8": {
        "batch_size": [8],
        "seq_len": [1],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # Prefill: BS=1, S=128, N=128 (首次token)
    "perf_prefill_s128": {
        "batch_size": [1],
        "seq_len": [128],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # Prefill: BS=1, S=512, N=128 (长序列首次填充)
    "perf_prefill_s512": {
        "batch_size": [1],
        "seq_len": [512],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # INT8 全量化 decode: 量化对decode性能影响
    "perf_decode_int8": {
        "batch_size": [1],
        "seq_len": [1],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [2],
        "kv_cache_quant_mode": [1],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # MXFP8 全量化 decode
    "perf_decode_mxfp8": {
        "batch_size": [1],
        "seq_len": [1],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [3],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # INT8 全量化 prefill
    "perf_prefill_int8_s128": {
        "batch_size": [1],
        "seq_len": [128],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [2],
        "kv_cache_quant_mode": [1],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # 小模型场景: He=2048, N=16 (较小的MLA配置)
    "perf_small_model": {
        "batch_size": [1],
        "seq_len": [1],
        "head_num": [16],
        "He": [2048],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # 大batch高吞吐场景: BS=32, S=1, N=128
    "perf_decode_bs32": {
        "batch_size": [32],
        "seq_len": [1],
        "head_num": [128],
        "He": [7168],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
    # Fuzz 测试参数
    "fuzz_default": {
        "batch_size": [1, 2, 4, 8],
        "seq_len": [1, 2, 4, 8, 16],
        "head_num": [1, 2, 4, 8, 16, 32, 64, 128],
        "He": [1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192],
        "dtype": [torch.bfloat16],
        "cache_mode": ["PA_BSND"],
        "block_size": [128],
        "weight_quant_mode": [0],
        "kv_cache_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
    },
}

# 按需选择要启用的测试参数
ENABLED_PARAMS = []

# 性能分析用例名称（perf_analyzer --mode report 使用）
PERF_CASE_NAMES = [
    "perf_decode_bs1",
    "perf_decode_bs8",
    "perf_decode_bs32",
    "perf_prefill_s128",
    "perf_prefill_s512",
    "perf_decode_int8",
    "perf_decode_mxfp8",
    "perf_prefill_int8_s128",
    "perf_small_model",
]

# Fuzz 测试参数（通过环境变量控制）
FUZZ_PARAMS = TEST_PARAMS["fuzz_default"]

# Fuzz 测试配置（通过环境变量控制）
FUZZ_ENABLED = os.environ.get("MLA_PROLOG_V3_ENABLE_FUZZ", "0") == "1"
FUZZ_CASES = int(os.environ.get("MLA_PROLOG_V3_FUZZ_CASES", "20"))
FUZZ_SEED = int(os.environ.get("MLA_PROLOG_V3_FUZZ_SEED", "42"))

# 成对覆盖测试参数（由 gen_coverage_testcases 生成）
from gen_coverage_testcases import generate_coverage_params
COVERAGE_PARAMS = generate_coverage_params()
