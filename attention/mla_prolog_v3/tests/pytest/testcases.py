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

# Full generalized matrix entry. test.py filters invalid quant/cache combinations.
TEST_PARAMS = {
    "generalized_all_modes": {
        "batch_size": [1],
        "He": [7168],
        "Hcq": [1536],
        "Hckv": [512],
        "q_head_num": [32],
        "kv_head_num": [1],
        "head_dim": [128],
        "rope_head_dim": [64],
        "q_seq": [1],
        "kv_seq": [128],
        "block_size": [128],
        "input_layout": ["BSH"],
        "cache_mode": ["PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"],
        "cq_epsilon": [0.0005],
        "ckv_epsilon": [0.0005],
        "dtype": [torch.bfloat16],
        "weight_quant_mode": [0, 1, 2, 3],
        "kv_quant_mode": [0, 1, 2, 3],
        "query_quant_mode": [0, 1],
        "ckvkr_repo_mode": [0, 1],
        "quant_scale_repo_mode": [0, 1],
        "tile_size": [128],
        "qc_qr_scale": [1.0],
        "kc_scale": [1.0],
    }
}

ENABLED_PARAMS = [TEST_PARAMS["generalized_all_modes"]]
