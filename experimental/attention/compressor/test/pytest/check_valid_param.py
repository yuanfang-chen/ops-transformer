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

import math
import random
import logging 
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


# ******Todo 4 算子入参的合法性校验，如下注释为示例

def check_valid_param(params):
    batch_size, hidden_size, Seq_len, head_dim, block_size, rope_head_dim, cmp_ratio, coff, norm_eps, \
    start_p, rotary_mode, layout_x, data_type, cu_seqlens, seqused, start_pos = params

    ## ======================== check input params start ========================
    # if S_max != None and start_pos != None:
    #     if layout_x == "TH":
    #         for i in range(batch_size):
    #             if start_pos[i] + (cu_seqlens[i + 1] - cu_seqlens[i]) > S_max:
    #                 print(f"Error: for batch {i} when shape of x is (T, hidden_size), start_pos[{i}] + (cu_seqlens[{i + 1}] - cu_seqlens[{i}]) > S_max, "
    #                     f"start_pos[{i}]={start_pos[i]}, cu_seqlens[{i + 1}]={cu_seqlens[i + 1]}, cu_seqlens[{i}]={cu_seqlens[i]}, S_max={S_max}")
    #                 return
    #             if seqused is not None:
    #                 if seqused[i] > (cu_seqlens[i + 1] - cu_seqlens[i]):
    #                     print(f"Error: for batch {i} when shape of x is (T, hidden_size), seqused[{i}] > (cu_seqlens[{i + 1}] - cu_seqlens[{i}]), "
    #                         f"seqused[{i}]={seqused[i]}, cu_seqlens[{i + 1}]={cu_seqlens[i + 1]}, cu_seqlens[{i}]={cu_seqlens[i]}")
    #                     return
    #     else:
    #         for i in range(batch_size):
    #             if start_pos[i] + Seq_len > S_max:
    #                 print(f"Error: for batch {i} when shape of x is (batch_size, Seq_len, hidden_size), start_pos[{i}] + Seq_len > S_max, start_pos[{i}]={start_pos[i]}, Seq_len={Seq_len}, S_max={S_max}")
    #                 return
    #             if seqused is not None:
    #                 if seqused[i] > Seq_len:
    #                     print(f"Error: for batch {i} when shape of x is (batch_size, Seq_len, hidden_size), seqused[{i}] > Seq_len, seqused[{i}]={seqused[i]}, Seq_len={Seq_len}")
    #                     return
    ## ======================== check input params finish ========================


