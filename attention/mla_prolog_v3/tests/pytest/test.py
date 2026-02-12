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
import pytest
import torch_npu

import check_valid_param
import prologv3_generalized
from testcases import ENABLED_PARAMS

PARAM_NAMES = [
    "batch_size", "He", "Hcq", "Hckv", "q_head_num",
    "kv_head_num", "head_dim", "rope_head_dim", "q_seq", "kv_seq",
    "block_size", "input_layout", "cache_mode", "cq_epsilon", "ckv_epsilon", "dtype",
    "weight_quant_mode", "kv_quant_mode", "query_quant_mode",
    "ckvkr_repo_mode", "quant_scale_repo_mode", "tile_size",
    "qc_qr_scale", "kc_scale"
]


def _build_param_combinations():
    combinations = []
    for params in ENABLED_PARAMS:
        values = [params[name] for name in PARAM_NAMES]
        for combo in itertools.product(*values):
            param_dict = dict(zip(PARAM_NAMES, combo))
            valid, _ = prologv3_generalized.validate_quant_cache_combo(
                param_dict["cache_mode"],
                param_dict["weight_quant_mode"],
                param_dict["kv_quant_mode"],
                param_dict["query_quant_mode"],
                param_dict["ckvkr_repo_mode"],
                param_dict["quant_scale_repo_mode"],
            )
            if valid:
                combinations.append(param_dict)
    return combinations


PARAM_COMBINATIONS = _build_param_combinations()


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", PARAM_COMBINATIONS)
def test_mla_prolog_v3(param_combinations):
    torch_npu.npu.set_device(0)

    test_data = (
        param_combinations["batch_size"],
        param_combinations["He"],
        param_combinations["Hcq"],
        param_combinations["Hckv"],
        param_combinations["q_head_num"],
        param_combinations["kv_head_num"],
        param_combinations["head_dim"],
        param_combinations["rope_head_dim"],
        param_combinations["q_seq"],
        param_combinations["kv_seq"],
        param_combinations["block_size"],
        param_combinations["input_layout"],
        param_combinations["cache_mode"],
        param_combinations["cq_epsilon"],
        param_combinations["ckv_epsilon"],
        param_combinations["dtype"],
        param_combinations["weight_quant_mode"],
        param_combinations["kv_quant_mode"],
        param_combinations["query_quant_mode"],
        param_combinations["ckvkr_repo_mode"],
        param_combinations["quant_scale_repo_mode"],
        param_combinations["tile_size"],
        param_combinations["qc_qr_scale"],
        param_combinations["kc_scale"]
    )

    try:
        check_valid_param.validate_config(test_data)
    except ValueError as err:
        pytest.skip(f"参数校验失败: {err}")

    expect, result = prologv3_generalized.test_prologv3_generalized(test_data)
    check_valid_param.check_result(expect, result)
