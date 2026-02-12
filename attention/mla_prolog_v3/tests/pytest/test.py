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

import itertools
import torch
import torch_npu
from testcases import ENABLED_PARAMS
# from testcases import GRAPH_MODE_PARAMS
import check_valid_param
import prologv3_no_quant_pa_bsnd
import pytest

DEBUG_ON = 0

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "batch_size", "He", "Hcq", "Hckv", "q_head_num",
        "kv_head_num", "head_dim", "rope_head_dim", "q_seq", "kv_seq",
        "block_size", "input_layout", "cache_mode", "cq_epsilon", "ckv_epsilon", "dtype",
        "weight_quant_mode", "kv_quant_mode", "query_quant_mode",
        "ckvkr_repo_mode", "quant_scale_repo_mode", "tile_size",
        "qc_qr_scale", "kc_scale"
    ]

    param_values = [
        locals()["param_batch_size"],
        locals()["param_He"],
        locals()["param_Hcq"],
        locals()["param_Hckv"],
        locals()["param_q_head_num"],
        locals()["param_kv_head_num"],
        locals()["param_head_dim"],
        locals()["param_rope_head_dim"],
        locals()["param_q_seq"],
        locals()["param_kv_seq"],
        locals()["param_block_size"],
        locals()["param_input_layout"],
        locals()["param_cache_mode"],
        locals()["param_cq_epsilon"],
        locals()["param_ckv_epsilon"],
        locals()["param_dtype"],
        locals()["param_weight_quant_mode"],
        locals()["param_kv_quant_mode"],
        locals()["param_query_quant_mode"],
        locals()["param_ckvkr_repo_mode"],
        locals()["param_quant_scale_repo_mode"],
        locals()["param_tile_size"],
        locals()["param_qc_qr_scale"],
        locals()["param_kc_scale"]
    ]

    # 生成所有的组合，并转换为字典列表
    locals()["param_combinations"] = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)


    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_mla_prolog_v3(param_combinations):   # 初始化参数和tensor
        batch_size = param_combinations['batch_size']
        He = param_combinations['He']
        Hcq = param_combinations['Hcq']
        Hckv = param_combinations['Hckv']
        q_head_num = param_combinations['q_head_num']
        kv_head_num = param_combinations['kv_head_num']
        head_dim = param_combinations['head_dim']
        rope_head_dim = param_combinations['rope_head_dim']
        q_seq = param_combinations['q_seq']
        kv_seq = param_combinations['kv_seq']
        block_size = param_combinations['block_size']
        input_layout = param_combinations['input_layout']
        cache_mode = param_combinations['cache_mode']
        cq_epsilon = param_combinations['cq_epsilon']
        ckv_epsilon = param_combinations['ckv_epsilon']
        dtype = param_combinations['dtype']
        weight_quant_mode = param_combinations['weight_quant_mode']
        kv_quant_mode = param_combinations['kv_quant_mode']
        query_quant_mode = param_combinations['query_quant_mode']
        ckvkr_repo_mode = param_combinations['ckvkr_repo_mode']
        quant_scale_repo_mode = param_combinations['quant_scale_repo_mode']
        tile_size = param_combinations['tile_size']
        qc_qr_scale = param_combinations['qc_qr_scale']
        kc_scale = param_combinations['kc_scale']

        torch_npu.npu.set_device(0)

        try:
            test_data = (
                batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim,
                q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype,
                weight_quant_mode, kv_quant_mode, query_quant_mode, ckvkr_repo_mode,
                quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale
                )
            # testcase参数校验逻辑
            check_valid_param.validate_config(test_data)
        except ValueError as e:
            pytest.skip(f"参数校验失败: {e}")
        if input_layout == "BSH":
            test_data = batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
                q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
                weight_quant_mode, kv_quant_mode, query_quant_mode, ckvkr_repo_mode, \
                quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale
            expect, result = prologv3_no_quant_pa_bsnd.test_prologv3_no_quant(test_data)

        check_valid_param.check_result(expect, result)

