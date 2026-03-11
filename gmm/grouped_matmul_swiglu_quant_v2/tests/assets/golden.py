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
__golden__ = {
        "kernel": {
            "grouped_matmul_swiglu_quant_v2": "grouped_matmul_swiglu_quant_v2_golden"
        }
}

from typing import List
import numpy as np
import torch

def grouped_matmul_swiglu_quant_v2_golden(x, x_scale, group_list, weight, weight_scale, weight_assist_matrix, bias,
                                          smooth_scale, dequant_mode:int = 0, dequant_dtype:int = 0,
                                          quant_mode: int = 0, quant_dtype:int = 0, transpose_weight: bool = 0,
                                          group_list_type:int = 0, tuning_config:List[int] = [0], **kwargs):
    
    x1, x2 = x, weight
    x2_ori = x2
    x1_dtype, x_scale_dtype, _, x2_dtype, weight_scale_dtype, _, _,_ = kwargs['input_dtypes']
    # x_format, weight_format, weight_scale_format,_,x_scale_format, _ ,_ = context.stc_input_formats
    trans_b = transpose_weight
    shape_m = x1.shape[0]
    shape_n = x2.shape[-1] if not trans_b else x2.shape[-2]
    shape_k = x1.shape[1]
    mx_quant_dtype = quant_dtype
    # mxFP4/8
    grouped_quant_output = []
    grouped_quant_scale_output = []
    group_num = len(group_list) # 不管什么分组，分组数一定等于该group_list长度
    is_mx_quant = x1_dtype in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2") and weight_scale_dtype == "float8_e8m0"

    # group_list_type 0: cumsum, 1: count
    if group_list_type == 1:
        group_list = np.cumsum(group_list)
    if is_mx_quant:
        #暂时只支持M轴分组
        pertoken_scale_mx = x_scale
        m, k0, k1 = pertoken_scale_mx.shape
        pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
        # broadcast，每个数对应32个数
        pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
        x1_dims = len(x1.shape)
        x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]
        x1 = np.pad(x1, [(0, 0)] * (x1_dims -1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x1 = x1 * pertoken_scale_mx_broadcast
        x1 = convert_to_high_precision(x1, x1_dtype)
        for i in range(group_num):
            scale_g = weight_scale[i]
            #reshape scale to shape n,ceil(k,64)*2
            if trans_b == False:
                scale_g = transform_tensor(scale_g)
            else:
                n, k0, k1 = scale_g.shape
                scale_g = scale_g.reshape(n, k0 * k1)

            x2 = x2_ori[i]
            # mxFP4单独处理transpose，统一使用(M,K)和(K,N)格式处理mxFP4
            if trans_b:
                x2 = np.swapaxes(x2, -1, -2)
                scale_g = scale_g.transpose()
            # broadcast，每个数对应32个数
            deq_scale_mx_broadcast = np.repeat(scale_g, 32, axis=-2)
            x2_dims = len(x2.shape)
            x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2.shape[-2]
            x2 = np.pad(x2, [(0, 0)] * (x2_dims -2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
            x2 = x2 * deq_scale_mx_broadcast
            # 升精度 & 转torch
            if i == 0:
                x1_temp = x1[:group_list[i], :]
            else:
                x1_temp = x1[group_list[i-1]:group_list[i], :]
            x2 = convert_to_high_precision(x2, x2_dtype)
            gmm_out = single_group_mm_cal(x1_temp, x2)
            torch.set_printoptions(threshold=torch.inf)
            # 检查是否有inf
            has_inf = torch.isinf(gmm_out).any()
            # 检查是否有nan
            has_nan = torch.isnan(gmm_out).any()

            print("是否包含inf：", has_inf.item())  # 输出 True（1）或 False（0）
            print("是否包含nan：", has_nan.item())  # 输出 True（1）或 False（0）
            swiglu_out = swiglu(gmm_out)
            quant_output, quant_scale_output = quant(swiglu_out, mx_quant_dtype)
            grouped_quant_output.append(quant_output)
            grouped_quant_scale_output.append(quant_scale_output)

    data_output = grouped_quant_output if not grouped_quant_output else np.concatenate(grouped_quant_output, axis=0)
    data_output = data_output.reshape(group_list[-1], int(shape_n/2))

    quant_scale_output = grouped_quant_scale_output if not grouped_quant_output else np.concatenate(grouped_quant_scale_output, axis=0)

    quant_scale_output = quant_scale_output.reshape(group_list[-1], int(shape_n/2/64), 2)
    return data_output, quant_scale_output

def convert_to_high_precision(input_tensor, input_type):
    import torch
    if input_type in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))
    elif input_type in ("int4"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.int32)).to(torch.int32)
    else:
        input_tensor = torch.from_numpy(input_tensor).to(torch.int32)
    return input_tensor

def single_group_mm_cal(x1, x2):
    return torch.matmul(x1, x2)

def swiglu(gmm_out):
    # 将结果分成两部分以应用 SwiGLU 激活函数
    c_temp4, gate = gmm_out.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
    c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘
    return c_temp6

def quant(swiglu_out, mx_quant_dtype):
    axis = 1
    block_size = 32
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(swiglu_out, axis, block_size)
    round_mode = "rint"
    share_exp = _mx_calculate_share_exp(fp_array, axis, mx_quant_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_quant_dtype, round_mode)
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis+1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    target_quant_dtype = ""
    if mx_quant_dtype == 0:
        target_quant_dtype = "float8_e4m3fn"
    elif mx_quant_dtype== 1:
        target_quant_dtype = "float8_e5m2"
    ele_dtype_np = eval(f"numpy_{target_quant_dtype}()")
    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array = scale_array.astype(numpy_float8_e8m0(), copy=False)
    return ele_array, scale_array

#按指定维度axis分割成固定大小的block_size
def _mx_reshape_to_blocks(fp_array: np.ndarray, axis: int, block_size: int):
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = np.pad(fp_array, pad, 'constant')
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape

def _mx_calculate_share_exp(fp_array, scale_axis, mx_quant_dtype):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    ele_emax = 0
    # float8_e4m3fn
    if mx_quant_dtype == 0:
        ele_emax = 8
    # float8_e5m2
    elif mx_quant_dtype == 1:
        ele_emax = 15
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis+1, keepdims=True)
    res = np.floor(
        np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax

    res[fp_abs_max == 0] = -float("inf")
    return res

def _mx_quantize_to_element_format(fp_array, share_exp,
                                   mx_ele_dtype, round_mode):
    exp_bits = 0
    mantissa_bits = 0
    emax = 0
    max_norm = 0
    # float8_e4m3fn
    if mx_ele_dtype == 0:
        exp_bits = 4
        mantissa_bits = 3
        emax = 8
        max_norm = 448
    # float8_e5m2
    elif mx_ele_dtype == 1:
        exp_bits = 5
        mantissa_bits = 2
        emax = 15
        max_norm = 57344

    ret = fp_array / (2 ** share_exp + (share_exp == -float("inf"))) - (share_exp == -float("inf")) * fp_array
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))
                                ).astype(fp_array.dtype, copy=False)
    # The minimum representable exponent for 8 exp bits is -126
    min_exp = -(2 ** (exp_bits - 1)) + exp_bits
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion of the number
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    # Set values > max_norm to Inf if desired, else clamp them
    # numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret

def _mx_round_mantissa(fp_array: np.ndarray, round_mode: str):
    """
    For example:
    fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
    - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
    - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
    - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
    - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    """
    if round_mode in ("rint", "even"):  # tie to even(c language rint)
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
        fp_array = np.sign(fp_array) * np.floor(
            np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
    elif round_mode == "floor":  # round to minus infinity(c language floor)
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":  # round to positive infinity(c language ceil)
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":  # round to zero(c language truncation)
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array

def numpy_float8_e8m0():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import float8_e8m0
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype!!! "
                            "Please install with `pip3 install en-dtypes`")
    
def _mx_undo_reshape_to_blocks(fp_array: np.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple):
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    # Remove extra dimension
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array

def transform_tensor(input_tensor):
    # 转置最后两维
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    # 重塑形状，将第一个轴和第二个轴相乘
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size *  height, width)
    return result