#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__input__ = {
        "kernel": {
            "grouped_matmul_swiglu_quant_v2": "grouped_matmul_swiglu_quant_v2_inputs"
        }
}

from typing import List
import numpy as np

def grouped_matmul_swiglu_quant_v2_inputs(x, x_scale, group_list_ori, weight, weight_scale, weight_assist_matrix, bias,
                                          smooth_scale, dequant_mode:int = 0, dequant_dtype:int = 0,
                                          quant_mode: int = 0, quant_dtype:int = 0, transpose_weight: bool = 0,
                                          group_list_type:int = 0, tuning_config:List[int] = [0], **kwargs):
    x1 = x
    pertoken_scale = x_scale
    group_list_shape = group_list_ori
    x2 = weight # tensor list
    input_deq_scale = weight_scale

    if input_deq_scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(127, 130, size=pertoken_scale.shape).astype(np.uint8) 
        x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")

        x2_mx_gm = np.random.randint(127, 130, size=input_deq_scale.shape).astype(np.uint8)
        x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
        input_deq_scale = x2_mx.astype("float8_e8m0")

    group_num = group_list_shape.shape[0]  # 注意ttk csv里group_num设置的准确性
    group_list = []
    if 'group_list_expect' in kwargs:
        group_list = kwargs['group_list_expect']
    else:
        group_list = group_list_ori

    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp[-1] > x1.shape[0]:
        raise Exception('sum of grouplist: ({}) can not be greater than x1[0]: ({})'.format(group_list_tmp[-1], x1.shape[0]))
    print('input_deq_scale ', input_deq_scale)
    print('pertoken_scale ', pertoken_scale)
    return (x1, pertoken_scale, np.array(group_list), x2, input_deq_scale, weight_assist_matrix,  bias, smooth_scale),\
           (x1, pertoken_scale, np.array(group_list), x2, input_deq_scale, weight_assist_matrix, bias, smooth_scale)