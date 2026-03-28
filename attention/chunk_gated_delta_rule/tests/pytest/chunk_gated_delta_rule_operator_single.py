# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu

from chunk_gated_delta_rule_golden import run_chunk_gated_delta_rule_eager, chunk_gated_delta_rule_native

class Generalized_operator():
    def forward(self, query, key, value, g, beta, chunk_size=64, initial_state=None):
        return chunk_gated_delta_rule_native(
            query, key, value, g, beta, chunk_size=chunk_size,
            initial_state=initial_state, output_final_state=False
        )

def output_operator(params):
    # 构造输入
    B, seqlen, nk, nv, dk, dv, chunk_size, data_type, \
    query_datarange, key_datarange, value_datarange, g_datarange, \
    beta_datarange, state_datarange = params

    print(f"params = {params}")

    run_chunk_gated_delta_rule_eager(
        B, seqlen, nk, nv, dk, dv, chunk_size=chunk_size,
        data_type=data_type, query_datarange=query_datarange,
        key_datarange=key_datarange, value_datarange=value_datarange,
        g_datarange=g_datarange, beta_datarange=beta_datarange,
        state_datarange=state_datarange
    )
