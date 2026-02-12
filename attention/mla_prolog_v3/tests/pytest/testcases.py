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

import torch

# 定义测试参数组合
TEST_PARAMS = {
    # 基础场景：非量化PA_BSND
    "base_default": {
        "batch_size": [8],
        "He": [7168],
        "Hcq": [1536],
        "Hckv": [512],
        "q_head_num": [32],
        "kv_head_num": [1],
        "head_dim": [128],
        "rope_head_dim": [64],
        "q_seq": [1],
        "kv_seq": [1024],
        "block_size": [128],
        "input_layout": ["BSH"],
        "cache_mode": ["PA_BSND"],
        "cq_epsilon": [0.0005],
        "ckv_epsilon": [0.0005],
        "dtype": [torch.bfloat16],
        "weight_quant_mode": [0],
        "kv_quant_mode": [0],
        "query_quant_mode": [0],
        "ckvkr_repo_mode": [0],
        "quant_scale_repo_mode": [0],
        "tile_size": [128],
        "qc_qr_scale": [1.0],
        "kc_scale": [1.0],
    }
}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = [TEST_PARAMS["base_default"]]  # 或指定子集，如[TEST_PARAMS["base"], TEST_PARAMS["mixed"]]

