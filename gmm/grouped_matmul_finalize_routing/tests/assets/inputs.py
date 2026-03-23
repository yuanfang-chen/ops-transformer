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
            "grouped_matmul_finalize_routing": "grouped_matmul_finalize_routing_inputs"
        }
}

from typing import List
import numpy as np
from ml_dtypes import bfloat16
import torch
import struct

def grouped_matmul_finalize_routing_inputs(x, w, scale, bias, pertoken_scale, group_list_ori, shared_input, logit, row_index,
                                           offset, dtype: int = 0, shared_input_weight: float = 1.0,
                                           shared_input_offset: int = 0, transpose_x: bool = False,
                                           transpose_w: bool = False, output_bs: int = 0, group_list_type: int = 1,
                                           tuning_config: List[int] = [0], **kwargs):
    
 
    x1 = x
    x2 = w
    group_list_shape = group_list_ori
    row_index_test = []
    for i in range(len(row_index) // group_list_shape.shape[0]):
        row_index_test.extend([i] * group_list_shape.shape[0])
    remain = len(row_index) % group_list_shape.shape[0]
    if remain:
        row_index_test.extend([0] * remain)
    row_index_test = np.array(row_index_test, dtype=np.int64)
    row_index = row_index_test

    # save input 
    x1_np = x1.view(np.uint8)
    x1_torch = torch.tensor(x1_np, dtype=torch.uint8)
    x2_np = x2.view(np.uint8)
    x2_torch = torch.tensor(x2_np, dtype=torch.uint8)
    shared_input_new = shared_input.astype(np.float32)
    shared_input_torch = torch.tensor(shared_input_new, dtype=torch.float32)
    logit_torch = torch.tensor(logit, dtype=torch.float32)
    row_index_torch = torch.tensor(row_index, dtype=torch.int32)

    # generate fp8_e8m0 scale
 
    if x1.dtype == 'float4_e2m1':
        x1_ori = np.random.uniform(0, 1, x1.shape).astype(np.float64)
        # x1_ori[:] = 1
        x1 = trans_np_bfloat16_tensor_to_fp4_e2m1(x1_ori.astype(bfloat16))
    elif x1.dtype == 'float4_e1m2':
        x1_ori = np.random.uniform(0, 1, x1.shape).astype(np.float64)
        x1 = trans_np_bfloat16_tensor_to_fp4_e1m2(x1_ori.astype(bfloat16))
 
    if x2.dtype == 'float4_e2m1':
        # 先转换成[N, K]生成随机数，再转换回[K, N]
        x2_ori = np.random.uniform(0, 1, x2.shape).astype(np.float64)
        # x2_ori[:] = 1
        x2 = trans_np_bfloat16_tensor_to_fp4_e2m1(x2_ori.astype(bfloat16))
    elif x2.dtype == 'float4_e1m2':
        x2_ori = np.random.uniform(0, 1, x2.shape).astype(np.float64)
        x2 = trans_np_bfloat16_tensor_to_fp4_e1m2(x2_ori.astype(bfloat16))
 
    if scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(127, 130, size=pertoken_scale.shape).astype(np.uint8) # 127, 130
        x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")
        u8 = pertoken_scale.view(np.uint8)
        pertoken_scale_torch = torch.tensor(u8, dtype=torch.uint8)
 
        x2_mx_gm = np.random.randint(127, 130, size=scale.shape).astype(np.uint8)
        x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
        scale = x2_mx.astype("float8_e8m0")
        u9 = scale.view(np.uint8)
        scale_torch = torch.tensor(u9, dtype=torch.uint8)
    group_num = group_list_shape.shape[0]  # 注意ttk csv里group_num设置的准确性
    if 'group_list_expect' in kwargs:
        group_list = kwargs['group_list_expect']
    else:
        group_list = group_list_ori
    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp[-1] > x1.shape[0]:
        raise Exception('sum of grouplist: ({}) can not be greater than x1[0]: ({})'.format(group_list_tmp[-1], x1.shape[0]))
    bias_n = bias.shape[-1]
    bias = np.zeros((x2.shape[0], bias_n)).astype(bfloat16)   # shape=(e, n) 的全 0 ndarray
    return x1, x2, scale, bias, pertoken_scale, np.array(group_list), shared_input, logit, row_index, None

def trans_np_bfloat16_tensor_to_fp4_e2m1(in_tensor):
    import numpy as np
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.uint8)
    in_tensor = in_tensor.reshape(multi_shape)

    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e2m1(in_tensor[i])

    out_tensor = out_tensor.astype(np.uint8)

    # 每两个fp4拼成一个uint8保存
    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape//2).astype(np.uint8)
    for i in range(multi_shape//2):
        fp4_tensor[i] = (out_tensor[i*2+1] << 4) | out_tensor[i*2]      # 按两两交叉顺序保存b4，比如b4两个数：0100 0010 存为b8后为0010 0100

    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor

def cvt_bfloat16_to_fp4_e2m1(x):
    import math
    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    x = eval(float_to_hex(x_abs))
    x = x >> 16

    ef = (x >> 7) & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 1
    maxExp = 3 # max E encoding value of e2m1 is 3
    expBias = 1 # Exponent Bias value of e2m1/e1m2 is 1
    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0) :
        eNorm = ef - 127 + 1 # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127 # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 1)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta) # determine if need to carry
        mRet = (mf >> mLenDelta)
        if (needRound) :
            mRet+=1
    else:
        eRet = (eNorm + expBias)
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = (mf >> mLenDelta)
        if (needRound) :
            mRet+=1

        if (((mRet & 0b10) != 0) and (needRound)) :
            eRet+=1
            mRet = 0

    if (eRet >= 3) :
        eRet = 3
    elif (eRet == 0 and mRet == 0b10) :
        eRet+=1
        mRet = 0

    return (((sRet) << 3) | ((eRet) << 1) | ((mRet) & 1))

def IsRoundOne(sign, man, truncLen):
    roundingTruncLen = 64
    if truncLen >= roundingTruncLen:
        mask0 = 0
    else:
        mask0 = 0x1 << truncLen
    if (truncLen > roundingTruncLen):
        mask1 = 0
    else:
        mask1 = 0x1 << (truncLen - 1)

    mask2 = mask1 - 1

    #ROUND_TO_NEAREST
    lastBit = (man & mask0) > 0      # Last bit after conversion
    truncHighBit = (man & mask1) > 0 # Highest bit in the truncated part
    truncLeft = (man & mask2) > 0    # Truncated left part (except for the highest bit)
    return truncHighBit and (truncLeft or lastBit)

def float_to_hex(f):
    return hex(struct.unpack('<I',struct.pack('<f',f))[0])

def trans_np_bfloat16_tensor_to_fp4_e1m2(in_tensor):
    import numpy as np
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.uint8)
    in_tensor = in_tensor.reshape(multi_shape)

    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e1m2(in_tensor[i])

    out_tensor = out_tensor.astype(np.uint8)
    # 每两个fp4拼成一个uint8保存
    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape//2).astype(np.uint8)
    for i in range(multi_shape//2):
        fp4_tensor[i] = (out_tensor[i*2+1] << 4) | out_tensor[i*2] # 按两两交叉顺序保存b4，比如b4两个数：0100 0010 存为b8后为0010 0100
    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor

def cvt_bfloat16_to_fp4_e1m2(x):
    import math
    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    x = eval(float_to_hex(x_abs))
    x = x >> 16

    ef = x >> 7 & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 2
    maxExp = 1 # max E encoding value of e1m2 is 3
    expBias = 1 # Exponent Bias value of e2m1/e1m2 is 1

    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0) :
        eNorm = ef - 127 + 1 # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127 # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 0b11)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta) # determine if need to carry
        mRet = (mf >> mLenDelta)
        if (needRound) :
            mRet+=1
    else:
        eRet = (eNorm + expBias)
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = (mf >> mLenDelta)
        if (needRound) :
            mRet+=1
        if (((mRet & 0b100) != 0) and (needRound)) :
            eRet+=1
            mRet = 0

    if (eRet >= 1) :
        eRet = 1
    elif (eRet == 0 and mRet == 0b100) :
        eRet+=1
        mRet = 0

    return (((sRet) << 3) | ((eRet) << 2) | ((mRet) & 3))