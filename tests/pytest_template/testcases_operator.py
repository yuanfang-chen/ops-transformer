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
    
    
    # ******Todo 1 算子入参设置

    "operator_default1":{
        # "batch_size": [4],
        # "q_t_size": [1],
        # "k_t_size":[1],
        # "block_size":[1],
    },

    "operator_default2":{
        # "batch_size": [6],
        # "q_t_size": [1],
        # "k_t_size":[1],
        # "block_size":[1],
    }
}

# 按需选择要启用的测试参数
ENABLED_PARAMS_1 = [TEST_PARAMS["operator_default1"]] 
ENABLED_PARAMS_2 = [TEST_PARAMS["operator_default2"]]