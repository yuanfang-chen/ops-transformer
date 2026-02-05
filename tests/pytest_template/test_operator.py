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
import check_result
import check_valid_param
import pytest

# ******入参调用
from testcases_operator import ENABLED_PARAMS_1 
# ******CPU侧算子逻辑实现获取golden与npu算子直调结果
import operator_single


for _, params in enumerate(ENABLED_PARAMS_1):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    ###### ******Todo 2 算子入参组合的生成，如下注释为示例

    # 生成所有参数组合
    param_names = [ .......]
    # param_names = [
    #     "batch_size", "q_t_size", "k_t_size", "block_size", "q_seq", "kv_seq", "q_head_num", "kv_head_num",
    #     "head_dim", "rope_dim", "q_dtype", "idx_dtype", "sparse_block_size","sparse_block_count",
    #     "kv_seq_act", "layout_kv", "layout_query"
    # ]

    param_values = [ ......]
    # param_values = [
    #     locals()["param_batch_size"],
    #     locals()["param_q_t_size"],
    #     locals()["param_k_t_size"],
    #     locals()["param_block_size"], 
    #     locals()["param_q_seq"],
    #     locals()["param_kv_seq"],
    #     locals()["param_q_head_num"],
    #     locals()["param_kv_head_num"],
    #     locals()["param_head_dim"],
    #     locals()["param_rope_dim"],
    #     locals()["param_q_dtype"],
    #     locals()["param_idx_dtype"],
    #     locals()["param_sparse_block_size"],
    #     locals()["param_sparse_block_count"],
    #     locals()["param_kv_seq_act"],
    #     locals()["param_layout_kv"],
    #     locals()["param_layout_query"]
    # ]

    # 生成所有的组合，并转换为字典列表
    locals()["param_combinations"] = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    ###### ******Todo 3 单算子直调入参的初始化，如下注释为示例,test_data为算子传递入参

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_sparse_flash_attention(param_combinations):   # 初始化参数和tensor
        # batch_size = param_combinations['batch_size']
        # q_t_size = param_combinations['q_t_size']
        # k_t_size = param_combinations['k_t_size']
        # block_size = param_combinations['block_size']
        # q_seq = param_combinations['q_seq']
        # kv_seq = param_combinations['kv_seq']
        # q_head_num = param_combinations['q_head_num']
        # kv_head_num = param_combinations['kv_head_num']
        # head_dim = param_combinations['head_dim']
        # rope_dim = param_combinations['rope_dim']
        # q_dtype = param_combinations['q_dtype']
        # idx_dtype = param_combinations['idx_dtype']
        # sparse_block_size = param_combinations['sparse_block_size']
        # sparse_block_count = param_combinations['sparse_block_count']
        # kv_seq_act = param_combinations['kv_seq_act']
        # layout_kv =  param_combinations['layout_kv']
        # layout_query =  param_combinations['layout_query']
        
        # test_data = batch_size, q_t_size, k_t_size, block_size, q_seq, kv_seq, q_head_num, kv_head_num, head_dim, rope_dim, \
        #             q_dtype, idx_dtype, sparse_block_size, sparse_block_count, kv_seq_act, layout_kv, layout_query

        torch_npu.npu.set_device(0)


    ###### ******Todo 4 算子入参的合法性校验

        # 输入参数的合法性校验
        try:
            check_valid_param.check_valid_param(test_data)
        except ValueError as e:
            pytest.skip(f"输入参数校验失败:{e}")


    ###### ******Todo 5  获取cpu 真值结果和npu结果

        # 获得cpu结果(真值)和算子结果（测试值）
        cpu_result, npu_result = operator_single.output_operator(test_data)

        # 结果精度对比
        check_result.check_result(cpu_result, npu_result)
