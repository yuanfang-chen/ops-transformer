#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch

# 定义测试参数组合
TEST_PARAMS = {
    # 基础场景 BSND TND
    "sas_default_params":{
        "layout_q": ["TND"],
        "layout_kv": ["PA_ND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.float8_e4m3fn],
        "cmp_kv_type": [torch.float8_e4m3fn],
        "B": [1],
        "S1": [1],
        "T1": [1],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [512],
        "block_num1": [16],
        "block_num2": [16],
        "block_size1": [128],
        "block_size2": [128],
        "cu_seqlens_q": [[0, 1]],
        "seqused_kv": [[1024]],
        "softmax_scale": [0.04167],
        "cmp_ratio": [1],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0],
        "kv_quant_mode": [1],
        "tile_size": [64],
        "rope_head_dim": [64],
        "template_run_mode": ["SWA"] # SWA SCFA CFA

        # decode首case
        # "layout_q": ["TND"],
        # "layout_kv": ["PA_ND"],
        # "q_type": [torch.bfloat16],
        # "ori_kv_type": [torch.bfloat16],
        # "cmp_kv_type": [torch.bfloat16],
        # "B": [1],
        # "S1": [1],
        # "T1": [1],
        # "N1": [64],
        # "N2": [1],
        # "D": [512],
        # "K": [512],
        # "block_num1": [65],
        # "block_num2": [17],
        # "block_size": [128],
        # "cu_seqlens_q": [[0, 1]],
        # "seqused_kv": [[8193]],
        # "softmax_scale": [0.04419417],
        # "cmp_ratio": [4],
        # "ori_mask_mode": [4],
        # "cmp_mask_mode": [3],
        # "ori_win_left": [128],
        # "ori_win_right": [0]

        # prefill首case
        # "layout_q": ["TND"],
        # "layout_kv": ["PA_ND"],
        # "q_type": [torch.bfloat16],
        # "ori_kv_type": [torch.bfloat16],
        # "cmp_kv_type": [torch.bfloat16],
        # "B": [1],
        # "S1": [8192],
        # "T1": [8192],
        # "N1": [64],
        # "N2": [1],
        # "D": [512],
        # "K": [512],
        # "block_num1": [65],
        # "block_num2": [17],
        # "block_size": [128],
        # "cu_seqlens_q": [[0, 8192]],
        # "seqused_kv": [[8192]],
        # "softmax_scale": [0.04419417],
        # "cmp_ratio": [4],
        # "ori_mask_mode": [4],
        # "cmp_mask_mode": [3],
        # "ori_win_left": [128],
        # "ori_win_right": [0]
    }
}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = [TEST_PARAMS["sas_default_params"]]