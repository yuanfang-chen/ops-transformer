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
ENABLED_PARAMS = [TEST_PARAMS["base_default"]]

# Fuzz 测试参数（通过环境变量控制）
FUZZ_PARAMS = TEST_PARAMS["fuzz_default"]

# Fuzz 测试配置（通过环境变量控制）
FUZZ_ENABLED = os.environ.get("MLA_PROLOG_V3_ENABLE_FUZZ", "0") == "1"
FUZZ_CASES = int(os.environ.get("MLA_PROLOG_V3_FUZZ_CASES", "20"))
FUZZ_SEED = int(os.environ.get("MLA_PROLOG_V3_FUZZ_SEED", "42"))

# 成对覆盖测试参数（由 gen_coverage_testcases 生成）
from gen_coverage_testcases import generate_coverage_params
COVERAGE_PARAMS = generate_coverage_params()
