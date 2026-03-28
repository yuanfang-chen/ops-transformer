# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
    "Testcase0":{
        "B": [1],
        "seqlen": [1600],
        "nk": [1],
        "nv": [1],
        "dk": [128],
        "dv": [128],
        "chunk_size": [64],
        "data_type":[torch.bfloat16],
        "query_datarange":[[-10, 10]],
        "key_datarange": [[-10, 10]],
        "value_datarange": [[-10, 10]],
        "g_datarange": [[0, 1]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]]
    },
    "Testcase1":{
        "B": [1],
        "seqlen": [16384],
        "nk": [4],
        "nv": [4],
        "dk": [128],
        "dv": [128],
        "chunk_size": [64],
        "data_type":[torch.bfloat16],
        "query_datarange":[[-10, 10]],
        "key_datarange": [[-10, 10]],
        "value_datarange": [[-10, 10]],
        "g_datarange": [[0, 1]],
        "beta_datarange": [[0, 1]],
        "state_datarange": [[-10, 10]]
    }
}
#注意单个用例组内的用例尽量不要超过32
FIRST_CASE = [TEST_PARAMS["Testcase0"], TEST_PARAMS["Testcase1"]]
# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = FIRST_CASE #按需增加需要的case即可
