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

import itertools
import torch
import torch_npu
from testcases_sas import ENABLED_PARAMS
import check_result
# import check_valid_param
import sparse_attn_sharedkv_process
import pytest

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "layout_q", "layout_kv", "q_type", "ori_kv_type", "cmp_kv_type", "B", "S1", "T1", "N1", "N2", "D", "K",
        "block_num1", "block_num2", "block_size", "cu_seqlens_q", "seqused_kv", "softmax_scale", "cmp_ratio",
        "ori_mask_mode", "cmp_mask_mode", "ori_win_left", "ori_win_right"
    ]

    param_values = [
        locals()["param_layout_q"],
        locals()["param_layout_kv"],
        locals()["param_q_type"],
        locals()["param_ori_kv_type"],
        locals()["param_cmp_kv_type"],
        locals()["param_B"],
        locals()["param_S1"],
        locals()["param_T1"],
        locals()["param_N1"],
        locals()["param_N2"],
        locals()["param_D"],
        locals()["param_K"],
        locals()["param_block_num1"],
        locals()["param_block_num2"],
        locals()["param_block_size"],
        locals()["param_cu_seqlens_q"],
        locals()["param_seqused_kv"],
        locals()["param_softmax_scale"],
        locals()["param_cmp_ratio"],
        locals()["param_ori_mask_mode"],
        locals()["param_cmp_mask_mode"],
        locals()["param_ori_win_left"],
        locals()["param_ori_win_right"],
    ]

    # 生成所有的组合，并转换为字典列表
    locals()["param_combinations"] = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_sparse_attn_sharedkv(param_combinations):   # 初始化参数和tensor
        layout_q = param_combinations['layout_q']
        layout_kv = param_combinations['layout_kv']
        q_type = param_combinations['q_type']
        ori_kv_type = param_combinations['ori_kv_type']
        cmp_kv_type = param_combinations['cmp_kv_type']
        B = param_combinations['B']
        S1 = param_combinations['S1']
        T1 = param_combinations['T1']
        N1 = param_combinations['N1']
        N2 = param_combinations['N2']
        D = param_combinations['D']
        K = param_combinations['K']
        block_num1 = param_combinations['block_num1']
        block_num2 = param_combinations['block_num2']
        block_size = param_combinations['block_size']
        cu_seqlens_q = param_combinations['cu_seqlens_q']
        seqused_kv = param_combinations['seqused_kv']
        softmax_scale = param_combinations['softmax_scale']
        cmp_ratio = param_combinations['cmp_ratio']
        ori_mask_mode = param_combinations['ori_mask_mode']
        cmp_mask_mode = param_combinations['cmp_mask_mode']
        ori_win_left = param_combinations['ori_win_left']
        ori_win_right = param_combinations['ori_win_right']

        torch_npu.npu.set_device(0)

        test_data = layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, \
                    block_num2, block_size, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, \
                    cmp_mask_mode, ori_win_left, ori_win_right
        print("test_data:", test_data)

        # # 输入参数的合法性校验
        # try:
        #     check_valid_param.check_valid_param(test_data)
        # except ValueError as e:
        #     pytest.skip(f"输入参数校验失败:{e}")

        # 获得cpu结果(真值)和算子结果（测试值）
        npu_result, cpu_result = sparse_attn_sharedkv_process.test_sas_process(test_data)
        print("npu_result.size():", npu_result.size())

        # 结果精度对比
        check_result.check_result(cpu_result, npu_result)