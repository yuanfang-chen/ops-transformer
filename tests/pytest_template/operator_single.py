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

import test
import torch
import torch_npu
import check_valid_param
import pytest
import random
import numpy as np
import math


###### ******Todo 5.1 cpu侧算子逻辑实现，定义类实现算子逻辑
class Generalized_operator()


def output_operator(params):

    ####### ******Todo 5.2 算子入参具体配置，注释为示例

    # batch_size, q_t_size, k_t_size, block_size, q_seq, kv_seq, q_head_num, kv_head_num, head_dim, rope_dim, \
    # q_dtype, idx_dtype, sparse_block_size, sparse_block_count, kv_seq_act, layout_kv, layout_query = params
    
    # if layout_query == "BSND":
    #     query = torch.tensor(np.random.uniform(-10, 10, (batch_size, q_seq, q_head_num, head_dim))).to(q_dtype).npu()
    # elif layout_query == "TND":
    #     query = torch.tensor(np.random.uniform(-10, 10, (t, q_head_num, head_dim))).to(q_dtype).npu()
    
    # if layout_kv == "BSND":
    #     key = torch.tensor(np.random.uniform(-5, 10, (batch_size, kv_seq, kv_head_num, head_dim))).to(q_dtype).npu()
    # elif layout_kv == "TND":
    #     key = torch.tensor(np.random.uniform(-10, 10, (t_k, kv_head_num, head_dim))).to(q_dtype).npu()
    # elif layout_kv == "PA_BSND":
    #     key = torch.tensor(np.random.uniform(-5, 10, (batch_size * (kv_seq // block_size), block_size, kv_head_num, head_dim))).to(q_dtype).npu()
    # value = key.clone()
    # idxs = random.sample(range(kv_seq_act - q_seq + 1), sparse_block_count)
    # sparse_indices = torch.tensor([idxs for _ in range(batch_size * q_seq * kv_head_num)]).\
    #                 reshape(batch_size, q_seq, kv_head_num, sparse_block_count).to(idx_dtype).npu()
    # query_rope = torch.tensor(np.random.uniform(-10, 10, (batch_size, q_seq, q_head_num, rope_dim))).to(q_dtype).npu()
    # key_rope = torch.tensor(np.random.uniform(-10, 10, (batch_size, kv_seq, kv_head_num, rope_dim))).to(q_dtype).npu()
    # act_seq_q = [q_seq] * batch_size
    # act_seq_kv = [kv_seq_act] * batch_size   
    # act_seq_q = torch.tensor(act_seq_q).to(idx_dtype).npu()
    # act_seq_kv = torch.tensor(act_seq_kv).to(idx_dtype).npu()
    # block_table = torch.tensor([range(batch_size * kv_seq // block_size)], dtype=torch.int32).reshape(batch_size, -1)
    # scale_value = 1 / ((head_dim + rope_dim)**0.5)
    # block_table = None
    # attention_mode = 2
    # return_softmax_lse = False

    ###### ******Todo 5.3 cpu侧算子结果复现，如下注释为示例

    # test_operator = Generalized_operator(
    #     batch_size, q_t_size, k_t_size, block_size, q_seq, kv_seq, q_head_num, kv_head_num, head_dim, rope_dim, 
    #     q_dtype, idx_dtype, sparse_block_size, sparse_block_count, kv_seq_act, layout_kv, layout_query)

    # cpu_result = test_operator.forward(query, key, value, sparse_indices, query_rope, key_rope, act_seq_q, act_seq_kv, block_table)

    ###### ******Todo 5.4 npu算子直调，如下注释为示例

    #  npu_result, _ , _ = torch_npu.npu_sparse_flash_attention(query, key, value,
    #                                                         sparse_indices, scale_value, 
    #                                                         block_table=block_table, 
    #                                                         actual_seq_lengths_query=act_seq_q, 
    #                                                         actual_seq_lengths_kv=act_seq_kv,
    #                                                         query_rope=query_rope,  
    #                                                         key_rope=key_rope, 
    #                                                         sparse_block_size=sparse_block_size,
    #                                                         layout_query=layout_query,
    #                                                         layout_kv=layout_kv, 
    #                                                         sparse_mode=3, 
    #                                                         pre_tokens=(1<<63)-1, 
    #                                                         next_tokens=(1<<63)-1,
    #                                                         attention_mode = 2, 
    #                                                         return_softmax_lse = False)
    
    torch.npu.synchronize()
    return cpu_result, npu_result

