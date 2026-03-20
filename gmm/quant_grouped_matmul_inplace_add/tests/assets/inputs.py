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
            "quant_grouped_matmul_inplace_add": "quant_grouped_matmul_inplace_add_golden"
        }
}

import numpy as np

def quant_grouped_matmul_inplace_add_golden(x1, x2, scale2, group_list_ori, y, scale1, group_list_type:int = 0,
                                            group_size: int = 0, **kwargs):
    input_deq_scale = scale2
    group_list_shape = group_list_ori
    pertoken_scale = scale1
    out_dtype = kwargs['output_dtypes'][0]
    
    # generate fp8_e8m0 scale
    if input_deq_scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(127, 130, size=pertoken_scale.shape).astype(np.uint8) # 127, 130
        x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")

        x2_mx_gm = np.random.randint(127, 130, size=input_deq_scale.shape).astype(np.uint8)
        x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
        input_deq_scale = x2_mx.astype("float8_e8m0")

    group_num = group_list_shape.shape[0]  # 注意ttk csv里group_num设置的准确性
    if 'group_list_expect' in kwargs:
        group_list_expect = kwargs['group_list_expect']# 全量化组需必传group_list_expect
        group_list = group_list_expect
    else:
        group_list = group_list_ori
    print("GMM INPUT FUNC, group_list: ", group_list)
    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp[-1] > x1.shape[0]:
        raise Exception('sum of grouplist: ({}) can not be greater than x1[0]: ({})'.format(group_list_tmp[-1], x1.shape[0]))
    return x1, x2, input_deq_scale, np.array(group_list), y, pertoken_scale
