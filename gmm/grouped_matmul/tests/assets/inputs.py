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
            "grouped_matmul": "grouped_matmul_inputs"
        }
}

from typing import List
from ml_dtypes import bfloat16
import numpy as np
import struct
import random

def grouped_matmul_inputs(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                          group_list_ori, per_token_scale, split_item: int = 0,
                          dtype:int = 0, transpose_weight:bool = False, transpose_x:bool = False,
                          group_type:int = -1, group_list_type:int = 0, act_type:int = 0,
                          tuning_config:List[int] = [0], **kwargs):
    print("#########x", x, len(x), type(x))

    if len(scale[0]) != 0:
        input_deq_scale = scale[0]
    else:
        input_deq_scale = None
    
    if len(per_token_scale) == 0:
        per_token_scale = None



    x1 = x
    x2 = weight
    input_deq_scale = scale
    group_list_shape = group_list_ori
    pertoken_scale = per_token_scale
    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]
    testcase_name = kwargs['testcase_name']

    # 重新生成float4_e2m1或float4_e2m1的x1和x2数据
    if x1.dtype == 'float4_e2m1':
        if out_dtype == "float32":
            x1_ori = np.random.uniform(0, 7, x1.shape).astype(np.float64)
        else:
            x1_ori = np.random.uniform(-2, 2
            , x1.shape).astype(np.float64)
        x1 = trans_np_bfloat16_tensor_to_fp4_e2m1(x1_ori.astype(bfloat16))
    elif x1.dtype == 'float4_e1m2' or x1.dtype == 'hifloat4' :
        if out_dtype == "float32":
            x1_ori = np.random.uniform(0, 2, x1.shape).astype(np.float64)
        else:
            x1_ori = np.random.uniform(-2, 2, x1.shape).astype(np.float64)
        x1 = trans_np_bfloat16_tensor_to_fp4_e1m2(x1_ori.astype(bfloat16))

    if x2.dtype == 'float4_e2m1':
        # 先转换成[N, K]生成随机数，再转换回[K, N]
        if out_dtype == "float32":
            x2_ori = np.random.uniform(0, 7, x2.shape).astype(np.float64)
        else:
            x2_ori = np.random.uniform(-7, 7, x2.shape).astype(np.float64)
        x2 = trans_np_bfloat16_tensor_to_fp4_e2m1(x2_ori.astype(bfloat16))
    elif x2.dtype == 'float4_e1m2' or x2.dtype == 'hifloat4' :
        if out_dtype == "float32":
            x2_ori = np.random.uniform(0, 2, x2.shape).astype(np.float64)
        else:
            x2_ori = np.random.uniform(-2, 2, x2.shape).astype(np.float64)
        x2 = trans_np_bfloat16_tensor_to_fp4_e1m2(x2_ori.astype(bfloat16))

    # convert scale to uint64
    if (input_deq_scale.dtype == "uint64" or input_deq_scale.dtype == "int64") and pertoken_scale is None:
        input_deq_scale = scale_generate(scale.shape, offset, out_dtype, testcase_name)
    # generate fp8_e8m0 scale
    elif input_deq_scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(127, 130, size=pertoken_scale.shape).astype(np.uint8) # 127, 130
        x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")

        x2_mx_gm = np.random.randint(127, 130, size=input_deq_scale.shape).astype(np.uint8)
        x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
        input_deq_scale = x2_mx.astype("float8_e8m0")
    elif antiquant_scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        # 伪量化MxA8W4
        x1_mx_gm = np.random.uniform(124, 130, size=pertoken_scale.shape).astype(np.uint8)
        x1_mx = 2 ** (x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")
 
    if antiquant_scale.dtype == "float8_e8m0":
        x2_mx_gm = np.random.uniform(124, 130, size=antiquant_scale.shape).astype(np.uint8)
        x2_mx = 2 ** (x2_mx_gm.astype(np.float64) - 127)
        antiquant_scale = x2_mx.astype("float8_e8m0")
    group_num = group_list_shape.shape[0]  # 注意ttk csv里group_num设置的准确性
    group_list_expect = None
    if 'group_list_expect' in kwargs:
        group_list_expect = kwargs['group_list_expect']
    group_list = []
    if group_list_expect:  # 全量化组需必传group_list_expect
        group_list = group_list_expect
    elif group_num == 1:
        group_list.append(x1.shape[0])
    else:
        avgGroup = (x1.shape[0] + group_num - 1) // group_num
        tmp_num = x1.shape[0]
        print(" Input func x1 shape: ", x1.shape)
        for i in range(group_num - 1):
            tmp = min(tmp_num, avgGroup)
            # tmp = random.randint(0, tmp_num)
            tmp_num -= tmp
            group_list.append(tmp)
        group_list.append(tmp_num)
        if group_list_type == 0:
            for i in range(1, group_num):
                group_list[i] += group_list[i - 1]
    print("GMM INPUT FUNC, group_list: ", group_list)
    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp[-1] > x1.shape[0]:
        raise Exception('sum of grouplist: ({}) can not be greater than x1[0]: ({})'.format(group_list_tmp[-1], x1.shape[0]))
    return x1, x2, bias, input_deq_scale, offset, antiquant_scale, antiquant_offset, np.array(group_list), pertoken_scale

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
        # fp4_tensor[i] = (out_tensor[i*2] << 4) | out_tensor[i*2+1]      # 按常规顺序保存b4
        fp4_tensor[i] = (out_tensor[i*2+1] << 4) | out_tensor[i*2]      # 按两两交叉顺序保存b4，比如b4两个数：0100 0010 存为b8后为0010 0100

    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor

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
    mLenDelta = 7 - 1 #
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
        # fp4_tensor[i] = (out_tensor[i*2] << 4) | out_tensor[i*2+1] # 按常规顺序保存b4
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
    mLenDelta = 7 - 2 #
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

def scale_generate(deq_scale_shape, offset, out_dtype, testcase_name):
    has_offset = offset is not None

    fp32_deq_scale = np.random.uniform(low=-5, high=5, size=deq_scale_shape).astype(np.float32)
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
    #与高19位运算，模拟硬件
    uint32_deq_scale &= 0XFFFFE000

    if has_offset:
        offset_shape = offset.shape
        fp32_offset = np.random.uniform(low=-5, high=5, size=offset_shape).astype(np.float32)

    #dequant
    if out_dtype != "int8":#
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(deq_scale_shape)
        np.save(testcase_name + "_deq_scale.npy", fp32_deq_scale)
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
    #requant
    elif out_dtype == "int8":
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(deq_scale_shape)
        np.save(testcase_name + "_deq_scale.npy", fp32_deq_scale)
        s9_offset = 0
        if has_offset:
            np.save(testcase_name + "_offset.npy", fp32_offset)
            s9_offset = f32_2_s9(fp32_offset).astype(int).reshape(offset_shape)
            s9_offset &= 0X1FF
            s9_offset = s9_offset[0] if deq_scale_shape[-1] < offset_shape[-1] else s9_offset
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
        uint64_deq_scale |= np.uint64(s9_offset << 37)
        uint64_deq_scale |= 1 << 46
    return uint64_deq_scale

def f32_2_s9(array):
    array_round = np.round(array)
    array_round_clip = np.clip(array_round, -256, 255)
    return array_round_clip