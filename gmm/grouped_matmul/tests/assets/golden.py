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
            "grouped_matmul": "grouped_matmul_golden"
        }
}

from typing import List
import numpy as np
import torch
import math
import struct
import os

def grouped_matmul_golden(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                          group_list, per_token_scale, split_item: int = 0,
                          dtype:int = 0, transpose_weight:bool = False, transpose_x:bool = False,
                          group_type:int = -1, group_list_type:int = 0, act_type:int = 0,
                          tuning_config:List[int] = [0], **kwargs):
    is1971 = False
    if kwargs['short_soc_version'] in ("Ascend910B", "Ascend910_93"):
        is1971 = True

    testcase_name = kwargs['testcase_name']

    x1, x2, pertoken_scale = x, weight, per_token_scale
    x1_original, x2_original, bias_original, scale_original, antiquant_scale_original, antiquant_offset_original, per_token_scale_original = x, weight, bias, scale, antiquant_scale, antiquant_offset, per_token_scale

    x1_dtype, x2_dtype, bias_dtype, scale_dtype, _, antiquant_scale_dtype, _, _, pertoken_scale_dtype = kwargs['input_dtypes']
    x1_format, x2_format, bias_format, scale_format, _, _, _, _, pertoken_scale_format = kwargs['input_formats']
    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]
    if out_dtype == 'bfloat16':
        out_dtype = bfloat16_conversion(output_dtypes)[0]
    trans_a = transpose_x
    trans_b = transpose_weight

    shape_m = x1.shape[0] if not trans_a else x1.shape[1]
    shape_n = x2.shape[-1] if not trans_b else x2.shape[-2]
    shape_k = x1.shape[-1] if not trans_a else x1.shape[-2]
    # group_type 0: M, 1: N, 2: K
    split_m = group_type == 0
    split_k = group_type == 2

    # 请遵守入口判断进入后再处理各自业务
    is_quant = scale.any()  # 全量化入口判断
    # 伪量化入口判断
    weight_quant_s8s4_flag = x1_dtype == "int8" and x2_dtype == "int4"
    if weight_quant_s8s4_flag:
        is_quant = False
    weight_quant_mxfp4_flag = "float4" in x2_dtype and "float8" in antiquant_scale_dtype
    print(f"weight_quant_s8s4_flag: {weight_quant_s8s4_flag}, weight_quant_mxfp4_flag: {weight_quant_mxfp4_flag}")
    is_weight_quant = antiquant_scale.any()
    print(f"is_quant: {is_quant}, is_weight_quant: {is_weight_quant}")

    # mxFP4/8
    outs = []
    group_num = len(group_list) # 不管什么分组，分组数一定等于该group_list长度
    if is_quant:
        gs_n = 128
        gs_k = 128
        x_quant_type = None
        w_quant_type = None
        is_mx_quant = x1_dtype in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2") and scale_dtype == "float8_e8m0"
        is_hif4 = x1_dtype in ("hifloat4") and scale_dtype == "float32"
        is_compatible = x1_dtype == 'int8' and scale_dtype not in ("uint64", "int64")

        if not is_mx_quant:
            x_quant_type, w_quant_type = infer_quant_mode(scale, pertoken_scale, trans_b, shape_m, shape_k, shape_n, group_num, split_m, split_k, gs_k, gs_n)
        print(f"Qunat mode is :{x_quant_type} {w_quant_type}")

        is_two_scale = x_quant_type == "perTensor" and w_quant_type == "perTensor"
        is_pertile = x_quant_type == "perGroup" and w_quant_type == "perBlock"
        is_pertoken = x_quant_type == "perToken"

        if w_quant_type == "perTensor":
            scale = scale.reshape([group_num, 1])
        if x_quant_type == "perTensor":
            pertoken_scale = pertoken_scale.reshape([group_num, 1])

        # group_list_type 0: cumsum, 1: count
        if group_list_type == 1:
            group_list = np.cumsum(group_list)
        if is_mx_quant:
            if split_m:
                # promote x1
                pertoken_scale_mx = per_token_scale_original
                #reshape pertoken to shape m, ceil(k,64)*2
                m, k0, k1 = pertoken_scale_mx.shape
                pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
                x1 = x1_original
                if x1_dtype == 'float4_e2m1':
                    x1 = trans_np_fp4_e2m1_tensor_to_bfloat16(x1).astype(np.float32)
                elif x1_dtype == 'float4_e1m2':
                    x1 = trans_np_fp4_e1m2_tensor_to_bfloat16(x1).astype(np.float32)
                if trans_a:
                    x1 = np.swapaxes(x1, -1, -2)
                    pertoken_scale_mx = pertoken_scale_mx.transpose()
                # broadcast，每个数对应32个数
                pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
                x1_dims = len(x1.shape)
                x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]
                x1 = np.pad(x1, [(0, 0)] * (x1_dims -1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
                x1 = x1 * pertoken_scale_mx_broadcast
                x1 = convert_to_high_precision(x1, x1_dtype)
                for i in range(group_num):
                    scale_g = scale_original[i]
                    #reshape scale to shape n,ceil(k,64)*2
                    if trans_b == False:
                        scale_g = transform_tensor(scale_g)
                    else:
                        n, k0, k1 = scale_g.shape
                        scale_g = scale_g.reshape(n, k0 * k1)
                    x2 = x2_original[i]
                    if x2_dtype == 'float4_e2m1':
                        x2 = trans_np_fp4_e2m1_tensor_to_bfloat16(x2).astype(np.float32)
                    elif x2_dtype == 'float4_e1m2':
                        x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32)
                    # mxFP4单独处理transpose，统一使用(M,K)和(K,N)格式处理mxFP4
                    if trans_b:
                        x2 = np.swapaxes(x2, -1, -2)
                        scale_g = scale_g.transpose()
                    bias_g = bias_original[i] if len(bias.shape) == 2 else None
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
                    out = single_group_mm_cal(x1_temp, x2, bias_g, None, None, None, out_dtype, bias_dtype, is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
            elif split_k:
                pertoken_scale_mx = per_token_scale_original
                x1 = x1_original  # (k1/32+k2/32+..., M),需分组取出
                x2 = x2_original  # (k1/32+k2/32+..., N),需分组取出
                deq_scale_mx = scale_original
                M = x1.shape[-1] # 单
                N = x2.shape[-1] # 单
                scale_k_offset = 0
                pertoken_scale_mx = transform_tensor(pertoken_scale_mx)
                deq_scale_mx = transform_tensor(deq_scale_mx)
                for i in range(group_num):
                    pre = 0 if i == 0 else group_list[i-1]
                    cur = group_list[i]
                    cur_scale_k = (cur // 64 + i + 1) * 2
                    if cur - pre == 0: # k is 0
                        out = np.zeros((M, N), dtype=out_dtype)
                        outs.append(out)
                        if i != group_num - 1:
                            if group_list[i+1] != 0:
                               scale_k_offset = cur_scale_k
                        continue
                    x1_g = x1[pre : cur, :]
                    x2_g = x2[pre : cur, :]
                    pertoken_scale_mx_g = pertoken_scale_mx[scale_k_offset :   cur_scale_k, :]
                    deq_scale_mx_g = deq_scale_mx[scale_k_offset :  cur_scale_k, :]
                    scale_k_offset = cur_scale_k
                    if trans_a:
                        x1_g = np.swapaxes(x1_g, -1, -2)
                        pertoken_scale_mx_g = pertoken_scale_mx_g.transpose()
                    # broadcast，每个数对应32个数
                    pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx_g, 32, axis=-1)
                    x1_dims = len(x1_g.shape)
                    x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1_g.shape[-1]
                    x1_g = np.pad(x1_g, [(0, 0)] * (x1_dims -1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
                    x1_g = x1_g * pertoken_scale_mx_broadcast
                    x1_g = convert_to_high_precision(x1_g, x1_dtype)
                    # mxFP4单独处理transpose，统一使用(M,K)和(K,N)格式处理mxFP4
                    if trans_b:
                        x2_g = np.swapaxes(x2_g, -1, -2)
                        deq_scale_mx_g = deq_scale_mx_g.transpose()
                    # broadcast，每个数对应32个数
                    deq_scale_mx_broadcast = np.repeat(deq_scale_mx_g, 32, axis=-2)
                    x2_dims = len(x2_g.shape)
                    x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2_g.shape[-2]
                    x2_g = np.pad(x2_g, [(0, 0)] * (x2_dims -2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
                    x2_g = x2_g * deq_scale_mx_broadcast
                    # 升精度 & 转torch
                    x2_g = convert_to_high_precision(x2_g, x2_dtype)
                    out = single_group_mm_cal(x1_g, x2_g, None, None, None, None, out_dtype, None, is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
        elif is_hif4:
            # promote x1
            pertoken_scale_hif4 = per_token_scale_original
            deq_scale_hif4 = scale_original
            #reshape pertoken to shape m, ceil(k,64)
            x1 = x1_original
            x2 = x2_original
            x1 = trans_np_fp4_e1m2_tensor_to_bfloat16(x1).astype(np.float32)
            x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32)
            #print(x1, x2, pertoken_scale_hif4)
            # broadcast，每个数对应64个数
            pertoken_scale_hif4 = repack_hif4_scale(pertoken_scale_hif4)
            #x2 = np.swapaxes(x2, -1, -2)
            #deq_scale_hif4 = np.swapaxes(deq_scale_hif4, -1, -2)
            x1_dims = len(x1.shape)
            x2_dims = len(x2.shape)
            x1_pad_len = pertoken_scale_hif4.shape[-1] - x1.shape[-1]
            #x2_pad_len = deq_scale_hif4.shape[-2] - x2.shape[-2]
            x1 = np.pad(x1, [(0, 0)] * (x1_dims -1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
            #x2 = np.pad(x2, [(0, 0)] * (x2_dims -2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
            x1 = x1 * pertoken_scale_hif4
            x1 = torch.from_numpy(x1.astype(np.float32))
            #x2 = x2 * deq_scale_hif4
            for i in range(group_num):
                scale_g = scale_original[i]
                scale_g = repack_hif4_scale(scale_g)
                #print(scale_g)
                x2 = x2_original[i]
                x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32)
                # HIFP4单独处理transpose，统一使用(M,K)和(K,N)格式处理HIFP4
                # transpose_b
                x2 = np.swapaxes(x2, -1, -2)
                scale_g = scale_g.transpose()
                x2_dims = len(x2.shape)
                #print("scale_g.shape[-2]:",scale_g.shape[-2],"x2.shape[-2]:",x2.shape[-2])
                x2_pad_len = scale_g.shape[-2] - x2.shape[-2]
                x2 = np.pad(x2, [(0, 0)] * (x2_dims -2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
                x2 = x2 * scale_g
                # 升精度 & 转torch
                if i == 0:
                    x1_temp = x1[:group_list[i], :]
                else:
                    x1_temp = x1[group_list[i-1]:group_list[i], :]
                x2 = torch.from_numpy(x2.astype(np.float32))
                out = torch.matmul(x1_temp, x2)
                outs.append(out)
            #print(outs)
        elif is_pertile:
            if trans_a:
                x1 = np.swapaxes(x1, -1, -2)
                pertoken_scale = np.swapaxes(pertoken_scale, -1, -2)
            if trans_b:
                x2 = np.swapaxes(x2, -1, -2)
                scale = np.swapaxes(scale, -1, -2)
            x1 = convert_to_high_precision(x1, x1_dtype) # fp8, hifp8同一个分支
            x2 = convert_to_high_precision(x2, x2_dtype)
            pertoken_scale = torch.from_numpy(pertoken_scale).to(torch.float32)
            scale = torch.from_numpy(scale).to(torch.float32)
            if split_m:
                for i in range(group_num):
                    scale_g = scale[i]
                    x2_g = x2[i]
                    pre = 0 if i == 0 else group_list[i-1]
                    cur = group_list[i]
                    x1_g = x1[pre : cur, :]
                    pertoken_scale_g = pertoken_scale[pre : cur, :]
                    out = pertile_mm_cal(x1_g, x2_g, pertoken_scale_g, scale_g, out_dtype)
                    outs.append(out)
            elif split_k:
                for i in range(group_num):
                    pre = 0 if i == 0 else group_list[i-1]
                    cur = group_list[i]
                    k_g = cur - pre
                    if k_g == 0: # k is 0
                        out = np.zeros((shape_m, shape_n), dtype=out_dtype)
                        outs.append(out)
                        continue
                    x1_g = x1[:, pre : cur]
                    x2_g = x2[pre : cur, :]
                    cur_scale_k = math.ceil(k_g / gs_k)
                    scale_k_offset = 0 if i == 0 else group_list[i-1] // gs_k + i
                    pertoken_scale_g = pertoken_scale[:, scale_k_offset : (scale_k_offset + cur_scale_k)]
                    scale_g = scale[scale_k_offset : (scale_k_offset + cur_scale_k), :]
                    out = pertile_mm_cal(x1_g, x2_g, pertoken_scale_g, scale_g, out_dtype)
                    outs.append(out)
        else:
            x1 = convert_to_high_precision(x1, x1_dtype)
            x2 = convert_to_high_precision(x2, x2_dtype)
            if is_mx_quant: # mxFP不做scale处理
                pass
            elif scale_dtype == "uint64" or scale_dtype == "int64":
                deq_scale = np.load(testcase_name + "_deq_scale.npy")
                deq_scale_tensor = torch.from_numpy(deq_scale)
            else:
                deq_scale = scale.astype(np.float32)
                deq_scale_tensor = torch.from_numpy(deq_scale)

            if is_pertoken and group_type == 0:
                pertoken_scale_slice = torch.unsqueeze(torch.from_numpy(pertoken_scale), dim=1).to(torch.float32)
            elif is_two_scale:
                two_scale = scale_generate(pertoken_scale * deq_scale)
                two_scale_tensor = torch.unsqueeze(torch.from_numpy(two_scale), dim=1).to(torch.float32)
            elif x_quant_type == "perTensor": # a pertensor b perchannel
                deq_scale_tensor *= torch.from_numpy(pertoken_scale).to(torch.float32)
            pertoken_scale_temp = None
            deq_scale_temp = None
            two_scale_temp = None
            bias_temp = None
            if group_type == 0: # M
                for i in range(group_num):
                    if i == 0:
                        x1_temp = x1[:group_list[i], :]
                    else:
                        x1_temp = x1[group_list[i - 1]:group_list[i], :]
                    x2_temp = x2[i]
                    if trans_b:
                        x2_temp = np.swapaxes(x2_temp, -1, -2)
                    pre = 0 if i == 0 else group_list[i-1]
                    cur = group_list[i]
                    if bias.any():
                        bias_temp = bias[i]
                    if is_pertoken:
                        pertoken_scale_temp = pertoken_scale_slice[pre:cur]
                        deq_scale_temp = deq_scale_tensor[i]
                    elif is_two_scale:
                        two_scale_temp = two_scale_tensor[i]
                    else:
                        deq_scale_temp = deq_scale_tensor[i]
                    out = single_group_mm_cal(x1_temp, x2_temp, bias_temp, pertoken_scale_temp, deq_scale_temp, two_scale_temp,
                                              out_dtype, bias_dtype,
                                              is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
            elif group_type == 2:
                for i in range(group_num):
                    M = x1.shape[-1] # 单
                    N = x2.shape[-1] # 单
                    pre = 0 if i == 0 else group_list[i-1]
                    cur = group_list[i]
                    if cur - pre == 0: # k is 0
                        out = np.zeros((M, N), dtype=out_dtype)
                        outs.append(out)
                        continue
                    x1_g = x1[pre:cur, :]
                    if trans_a:
                        x1_g = np.swapaxes(x1_g, -1, -2)
                    x2_g = x2[pre:cur, :]

                    if bias != []:
                        bias_temp = bias[i]
                    if is_pertoken:
                        pertoken_scale_temp = pertoken_scale[i].flatten()
                        pertoken_scale_temp = torch.unsqueeze(torch.from_numpy(pertoken_scale_temp), dim=1).to(torch.float32)
                        deq_scale_temp = deq_scale_tensor[i]
                    elif is_two_scale:
                        two_scale_temp = two_scale_tensor[i]
                    else:
                        deq_scale_temp = deq_scale_tensor[i]
                    out = single_group_mm_cal(x1_g, x2_g, bias_temp, pertoken_scale_temp, deq_scale_temp, two_scale_temp,
                                              out_dtype, bias_dtype,
                                              is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
    elif is_weight_quant:
        if weight_quant_s8s4_flag:
            x2_ori = x2_original
            print("x1 shape: {}, x1_dtype: {}, x2 shape: {}, x2 ori shape: {}, x2_dtype: {}, x2_format: {}".
                  format(x1.shape, x1_dtype, x2.shape, x2_ori.shape, x2_dtype, x2_format))
            print("scale shape: {}, dtype: {}".format(scale.shape, scale.dtype))
            print("per-group scale shape: {}, dtype: {}".format(antiquant_scale.shape, scale.dtype))
            print("per-token scale shape: {}, dtype: {}".format(pertoken_scale.shape, pertoken_scale.dtype))

            assert np.array_equal(reshape_last_two_dims(x2_ori), x2)
            assert not trans_a and not trans_b

            outs = cal_weight_quant_s8s4(x1, x2_ori, bias, scale, antiquant_scale, pertoken_scale, group_list,
                                         group_list_type, output_dtypes[0])
        elif weight_quant_mxfp4_flag:
            # A16MxF4和MxA8W4
            assert x2_format == "FRACTAL_NZ"
            outs = cal_weight_quant_mx(x1, x2, bias, antiquant_scale, pertoken_scale, group_list, group_list_type,
                                       x1_dtype, x2_dtype, output_dtypes[0], trans_b)
        else:
            outs = cal_weight_quant_a16w8(x1, x2, bias, antiquant_scale, antiquant_offset, group_list, group_list_type,
                                          x1_dtype, x2_dtype, output_dtypes[0], trans_b)
    else:
        if group_list_type == 1:
            group_list = np.cumsum(group_list)
        if group_type == 0:
            _, w_outer, w_inner = x2.shape
            if x2_dtype == "float16" or x2_dtype == "bfloat16":
                w_outer_align = (w_outer + 16) // w_outer * w_outer
                w_inner_align = (w_inner + 16) // w_inner * w_inner

            if trans_a:
                x1 = np.swapaxes(x1, -1, -2)
            x1 = torch.from_numpy(x1.astype(np.float32))
            for i in range(group_num):
                x2 = x2_original[i]
                if x2_format == "FRACTAL_NZ":
                    x2 = x2.transpose((1, 2, 0, 3)).reshape(w_outer_align, w_inner_align)
                    # 去掉非对齐部分
                    x2 = x2[:w_outer, :w_inner]

                if trans_b:
                    x2 = np.swapaxes(x2, -1, -2)
                x2 = torch.from_numpy(x2.astype(np.float32))
                if i == 0:
                    x1_temp = x1[:group_list[i], :]
                else:
                    x1_temp = x1[group_list[i-1]:group_list[i], :]
                bias_temp = None
                if bias.any():
                    bias_temp = bias[i]

                out = torch.matmul(x1_temp, x2)
                if bias_temp is not None:
                    bias_temp = torch.from_numpy(bias_temp.astype(np.float32))
                    out += bias_temp
                outs.append(out.numpy().astype(out_dtype))
        elif group_type == 2:
            x1 = torch.from_numpy(x1.astype(np.float32))
            x2 = torch.from_numpy(x2.astype(np.float32))
            for i in range(group_num):
                M = x1.shape[-1] # 单
                N = x2.shape[-1] # 单
                pre = 0 if i == 0 else group_list[i-1]
                cur = group_list[i]
                if cur - pre == 0: # k is 0
                    out = np.zeros((M, N), dtype=out_dtype)
                    outs.append(out)
                    continue
                x1_g = x1[pre:cur, :]
                if trans_a:
                    x1_g = np.swapaxes(x1_g, -1, -2)
                x2_g = x2[pre:cur, :]
                out = torch.matmul(x1_g, x2_g).numpy().astype(out_dtype)
                outs.append(out)
    real_out = outs if not outs else np.concatenate(outs, axis=0)
    return real_out

def bfloat16_conversion(container):
    """
    Convert bfloat16/int4/fp8 string to numpy dtype
    :param container:
    :return:
    """
    ret = list(container)
    special_dtypes = ("bfloat16", "int4",
                      "float8_e5m2", "float8_e4m3fn", "float8_e8m0",
                      "float4_e2m1", "float4_e1m2",
                      "hifloat8", "hifloat4")
    for sd in special_dtypes:
        for idx, dtype in enumerate(ret):
            if not isinstance(dtype, str):
                continue
            if sd == dtype:
                ret[idx] = eval(f"numpy_{sd}()")
    return ret

def infer_quant_mode(scale, pertoken_scale, trans_b, shape_m, shape_k, shape_n, group_num, split_m, split_k, gs_k, gs_n):
    x_quant_type = set()
    w_quant_type = set()
    if split_m:
        expect_scale_k = math.ceil(shape_k / gs_k)
    elif split_k:
        expect_scale_k = get_total_scale_k_for_split_k(shape_k, gs_k, group_num)

    if pertoken_scale is not None:
        if split_m:
            if len(pertoken_scale.shape) == 1:
                if pertoken_scale.shape[-1] == group_num:
                    x_quant_type.add("perTensor")
                elif pertoken_scale.shape[-1] == shape_m:
                    x_quant_type.add("perToken")
            else:
                if pertoken_scale.shape[-1] == expect_scale_k and pertoken_scale.shape[-2] == shape_m:
                    x_quant_type.add("perGroup")
                if pertoken_scale.shape[-1] == 1 and pertoken_scale.shape[-2] == group_num:
                    x_quant_type.add("perTensor")
                if pertoken_scale.shape[-1] == 1 and pertoken_scale.shape[-2] == shape_m:
                    x_quant_type.add("perToken")
        elif split_k:
            if len(pertoken_scale.shape) == 1:
                if pertoken_scale.shape[-1] == group_num:
                    x_quant_type.add("perTensor")
            else:
                if pertoken_scale.shape[-1] == 1 and pertoken_scale.shape[-2] == group_num:
                    x_quant_type.add("perTensor")
                if pertoken_scale.shape[-1] == shape_m and pertoken_scale.shape[-2] == group_num:
                    x_quant_type.add("perToken")
                if pertoken_scale.shape[-1] == shape_m and pertoken_scale.shape[-2] == expect_scale_k:
                    x_quant_type.add("perGroup")

    if len(scale.shape) == 1:
        if scale.shape[-1] == group_num:
             w_quant_type.add("perTensor")
    else:
        scale_k = scale.shape[-1] if trans_b else scale.shape[-2]
        scale_n = scale.shape[-2] if trans_b else scale.shape[-1]

        if scale.shape[-1] == 1 and scale.shape[-2] == group_num:
            w_quant_type.add("perTensor")
        if scale.shape[-1] == shape_n and scale.shape[-2] == group_num:
            w_quant_type.add("perChannel")
        if scale_k == expect_scale_k and scale_n == math.ceil(shape_n / gs_n):
            w_quant_type.add("perBlock")

    print(f"Quant mode list: X:{x_quant_type} W:{w_quant_type}")

    if len(w_quant_type) == 1 and len(x_quant_type) == 1:
        return next(iter(x_quant_type)), next(iter(w_quant_type))
    if "perGroup" in x_quant_type and "perBlock" in w_quant_type:
        return "perGroup", "perBlock"
    if "perTensor" in x_quant_type and "perTensor" in w_quant_type:
        return "perTensor", "perTensor"

    if "perGroup" in x_quant_type:
        x_quant_type.remove("perGroup")
    if "perBlock" in w_quant_type:
        w_quant_type.remove("perBlock")

    res_x = next(iter(x_quant_type)) if x_quant_type else None
    res_w = next(iter(w_quant_type)) if w_quant_type else None
    return res_x, res_w

def trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    import numpy as np
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)

    # 1个uint8包含两个fp4, 先拆成两个uint8
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape*2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape*2).astype(np.float32)

    for i in range(multi_shape):
        bfloat16_tensor[i*2], bfloat16_tensor[i*2+1] = cvt_fp4_e2m1_to_bfloat16(in_tensor[i])
        fp32_tensor[i*2] = struct.unpack('!f',struct.pack('!I',bfloat16_tensor[i*2]<<16))[0]
        fp32_tensor[i*2+1] = struct.unpack('!f',struct.pack('!I',bfloat16_tensor[i*2+1]<<16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor

def trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    import numpy as np
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)

    # 1个uint8包含两个fp4, 先拆成两个uint8
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape*2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape*2).astype(np.float32)

    for i in range(multi_shape):
        bfloat16_tensor[i*2], bfloat16_tensor[i*2+1] = cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
        fp32_tensor[i*2] = struct.unpack('!f',struct.pack('!I',bfloat16_tensor[i*2]<<16))[0]
        fp32_tensor[i*2+1] = struct.unpack('!f',struct.pack('!I',bfloat16_tensor[i*2+1]<<16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor

def cvt_fp4_e2m1_to_bfloat16(x):
    Fp4e2m1ToBf16 = {'0': 0x0, '1': 0x3F00, '2': 0x3F80, '3':0x3FC0,
                     '4': 0x4000, '5': 0x4040, '6': 0x4080, '7':0x40C0,
                     '8': 0x8000, '9': 0xBF00, '10': 0xBF80, '11':0xBFC0,
                     '12': 0xC000, '13': 0xC040, '14': 0xC080, '15':0xC0C0}

    x = int(x)
    first_fp4val = x & 0x0f
    second_fp4val = (x >> 4 )& 0x0f
    first_fp4str = str(first_fp4val)
    second_fp4str = str(second_fp4val)

    return Fp4e2m1ToBf16[first_fp4str], Fp4e2m1ToBf16[second_fp4str]


def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {'0': 0x0, '1': 0x3E80, '2': 0x3F00, '3':0x3F40,
                     '4': 0x3F80, '5': 0x3FA0, '6': 0x3FC0, '7':0x3FE0,
                     '8': 0x8000, '9': 0xBE80, '10': 0xBF00, '11':0xBF40,
                     '12': 0xBF80, '13': 0xBFA0, '14': 0xBFC0, '15':0xBFE0}

    x = int(x)
    first_fp4val = x & 0x0f
    second_fp4val = (x >> 4 )& 0x0f
    first_fp4str = str(first_fp4val)
    second_fp4str = str(second_fp4val)

    return Fp4e1m2ToBf16[first_fp4str], Fp4e1m2ToBf16[second_fp4str]

def get_total_scale_k_for_split_k(k, group_size, group_num):
    return k // group_size + group_num

def convert_to_high_precision(input_tensor, input_type):
    import torch
    if input_type in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8", "hifloat4"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))
    elif input_type in ("int4"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.int32)).to(torch.int32)
    else:
        input_tensor = torch.from_numpy(input_tensor).to(torch.int32)
    return input_tensor

def transform_tensor(input_tensor):
    # 转置最后两维
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    # 重塑形状，将第一个轴和第二个轴相乘
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size *  height, width)
    return result

def single_group_mm_cal(x1, x2, bias, pertoken_scale_slice, deq_scale_tensor, two_scale_tensor,
                        out_dtype, bias_dtype,
                        is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible):
    import torch
    out = torch.matmul(x1, x2)
    if bias is not None and bias_dtype == "int32":
        bias = torch.from_numpy(bias).to(torch.int32)
        out = torch.add(out, bias)
    if out_dtype == 'float16' or out_dtype == 'float32':
        if is_mx_quant:
            pass
        elif is_pertoken:
            out = out * deq_scale_tensor * pertoken_scale_slice
        elif is_two_scale:
            out = out * two_scale_tensor
        else:
            out = out * deq_scale_tensor
        if bias is not None and (bias_dtype == "float16" or bias_dtype == "float32"):
            bias_fp32 = torch.from_numpy(bias.astype(np.float32))
            out = (out + bias_fp32).numpy().astype(out_dtype)
        else:
            out = out.numpy().astype(out_dtype)
    else: #bf16
        if is_mx_quant or is_hif4:
            pass
        elif is_pertoken:
            out = out * deq_scale_tensor * pertoken_scale_slice
        elif is_two_scale:
            out = out * two_scale_tensor
        elif is_compatible:
            if deq_scale_tensor.shape[-1] == 1 and (bias_dtype != "bfloat16" and bias_dtype != "float32"):
                scale = scale_generate(deq_scale_tensor.numpy())
                deq_scale_tensor = torch.unsqueeze(torch.from_numpy(scale), dim=1).to(torch.float32)
            out = out * deq_scale_tensor
        else:
            out = out * deq_scale_tensor
        if bias is not None and (bias_dtype == "bfloat16" or bias_dtype == "float16" or bias_dtype == "float32"):
            bias_fp32 = torch.from_numpy(bias.astype(np.float32))
            out = (out + bias_fp32).numpy().astype(out_dtype)
        else:
            out = out.numpy().astype(out_dtype)
    return out

def scale_generate(fp32_deq_scale):
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    #与高19位运算，模拟硬件
    uint32_deq_scale &= 0XFFFFE000
    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)

    return fp32_deq_scale

def repack_hif4_scale(scale):
    # 获取 scale 的形状
    original_shape = scale.shape

    # 计算新的形状：最后一维扩展为原来的 64 倍
    new_shape = list(original_shape[:-1]) + [original_shape[-1] * 64]

    # 创建一个空的数组，数据类型为 fp32
    result = np.zeros(new_shape, dtype=np.float32)

    # 计算每个元素的值
    for idx in np.ndindex(original_shape):
        scale_value = np.uint32(scale[idx])
        for k in range(64):
            # 计算新的索引
            new_idx = list(idx)
            new_idx[-1] = idx[-1] * 64 + k
            new_idx = tuple(new_idx)
            high_8bits = (scale_value ) & 0xFF
            bits_9_to_16 = (scale_value >> 8) & 0xFF
            secd_idx = k // 8
            secd_bit = (bits_9_to_16 >> secd_idx) & 0x1
            high_16 = (scale_value >> 16) & 0xFFFF
            third_idx = k // 4
            third_bit = (high_16 >> third_idx) & 1
            m2=high_8bits & 0x3
            e6=high_8bits >>2
            result[new_idx] = np.float32(np.power(np.float32(2),(e6-48) + secd_bit + third_bit)*np.float32(1+(m2 >> 1)*(0.5)+(m2 & 0x1)*(0.5)**(2)))
            tmp = high_8bits + secd_bit + third_bit
    return result

def pertile_mm_cal(x1, x2, pertoken_scale, scale, out_dtype):
    # x1, x2(with scale) 按照trans = False, False进入该函数
    import torch
    group_size_k = 128
    group_size_n = 128
    M = x1.shape[0]
    K = x2.shape[-2]
    N = x2.shape[-1]
    out = torch.zeros(M, N)
    scale_n = torch.repeat_interleave(scale, repeats=group_size_n, dim=-1)
    scale_n = scale_n[:, :N]
    for k_idx in range((K + group_size_k - 1) // group_size_k):
        k_start = k_idx * group_size_k
        k_end = min((k_idx + 1) * group_size_k, K)
        scale_mul = pertoken_scale[:, k_idx].unsqueeze(1) * scale_n[k_idx, :].unsqueeze(0)
        out += torch.matmul(x1[:, k_start:k_end], x2[k_start:k_end, :]) * scale_mul
    if out_dtype == 'bfloat16':
        out_dtype = bfloat16_conversion(out_dtype)
    out = (out).numpy().astype(out_dtype) # 暂时只支持fp16/bf16/fp32
    return out

def reshape_last_two_dims(x, c0=32):
    def ceil_align(d, align):
        return (d + align - 1) // align * align

    # 获取原始shape
    original_shape = np.array(x.shape)
    ndim = x.ndim

    if ndim < 2:
        raise ValueError("输入数组的维度必须 >= 2")

    # 最后两维的原始大小
    d1, d2 = original_shape[-2], original_shape[-1]

    # 计算对齐后的最后两维大小
    aligned_d1 = ceil_align(d1, 16)
    aligned_d2 = ceil_align(d2, c0)

    # 计算padding量
    pad_d1 = aligned_d1 - d1
    pad_d2 = aligned_d2 - d2

    # 构造padding参数 (只对最后两维padding，其他维不padding)
    pad_width = [(0, 0)] * (ndim - 2) + [(0, pad_d1), (0, pad_d2)]

    # 执行padding
    x_padded = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

    # 计算新的最后两维的shape
    new_d1 = aligned_d1 // 16
    new_d2 = aligned_d2 // c0

    # 构造新的shape
    new_shape = list(original_shape[:-2]) + [new_d1, 16, new_d2, c0]

    # reshape
    x_reshaped = x_padded.reshape(new_shape)

    # 构造transpose参数
    transpose_order = list(range(ndim - 2)) + [ndim, ndim - 2, ndim - 1, ndim + 1]

    # transpose最后四维 (其他维度顺序不变)
    x_transposed = np.transpose(x_reshaped, axes=transpose_order)

    return x_transposed

def cal_weight_quant_s8s4(x1, x2_ori, bias, scale, antiquant_scale, pertoken_scale, group_list, group_list_type,
                          y_dtype):
    import torch
    shape_k = x2_ori.shape[-2]
    group_size = get_per_group_size(antiquant_scale, shape_k, False)
    print("antiquant groupSize: {}, shape_k: {}".format(group_size, shape_k))

    x2_tmp = x2_ori.astype(np.float16)
    antiquant_scale = antiquant_scale.astype(np.float16)
    antiquant_scale_broadcast = np.repeat(antiquant_scale, group_size, axis=-2)
    x2_tmp = x2_tmp * antiquant_scale_broadcast

    x1 = torch.from_numpy(x1.astype(np.int32)).to(torch.int32)
    x2 = torch.from_numpy(np.around(x2_tmp).astype(np.int32)).to(torch.int32)

    outs = []
    for i in range(len(group_list)):
        start_m, end_m = get_group_m(group_list, group_list_type, i)
        x1_group = x1[start_m:end_m, :]
        weight_group = x2[i]

        group_out = torch.matmul(x1_group, weight_group).to(torch.int32)
        group_out = group_out.numpy().astype(np.float32) * scale[i].reshape(1, -1) * \
                    pertoken_scale.reshape(-1, 1)[start_m:end_m, :]

        if (bias is not None) and (len(bias) != 0):
            group_out += bias[i, :].reshape(1, -1)

        if y_dtype == "bfloat16":
            outs.append(group_out.astype(bfloat16_conversion([y_dtype])[0]))
        else:
            outs.append(group_out.astype(np.float16))

    return outs

def get_per_group_size(antiquant_scale, k_size, trans_b):
    gn = antiquant_scale.shape[-2] if not trans_b else antiquant_scale.shape[-1]
    assert gn > 0 and k_size % gn == 0
    group_size = k_size // gn
    print(f"antiquant groupSize: {group_size}, k_size: {k_size}", flush=True)
    return group_size

def get_group_m(group_list, group_list_type, i):
    if group_list_type == 0:
        start_m = group_list[i - 1] if i > 0 else 0
        end_m = group_list[i]
    else:
        start_m = sum(group_list[:i])
        end_m = start_m + group_list[i]
    print("GMM GOLDEN: ")
    print(start_m)
    print(end_m)
    return start_m, end_m

def cal_weight_quant_mx(x1, x2, bias, antiquant_scale, pertoken_scale, group_list, group_list_type,
                        x1_dtype, x2_dtype, y_dtype, trans_b):
    import torch
    x1 = torch.from_numpy(x1.astype(np.float32))
    x2 = trans_weight_fp4(x1_dtype, x2_dtype, x2)
    print("cal_weight_quant_mx x2 shape 00000000000 ", x2.shape, flush=True)
    x2 = trans_nz2nd(x2)
    print("cal_weight_quant_mx x2 shape 11111111111 ", x2.shape, flush=True)
    x2 = torch.from_numpy(x2).to(torch.float16) if x1_dtype == "float16" else torch.from_numpy(x2)
    print(f"cal_weight_quant_mx x2 dtype 11111111111 {x2.dtype}", flush=True)

    shape_k = x2.shape[-2] if not trans_b else x2.shape[-1]
    shape_n = x2.shape[-1] if not trans_b else x2.shape[-2]
    print(f"cal_weight_quant_mx k{shape_k} n{shape_n}")

    if (bias is not None) and (len(bias) != 0):
        bias = torch.from_numpy(bias.astype(np.float32))
    else:
        bias = torch.from_numpy(np.zeros([x2.shape[0], shape_n], dtype=np.float32))

    group_size = get_per_group_size(antiquant_scale, shape_k, trans_b)
    assert group_size == 32

    antiquant_scale_broadcast, pertoken_scale_broadcast = get_mx_scale(antiquant_scale, group_size,
                                                                       pertoken_scale, trans_b, x1_dtype)
    outs = []
    for i in range(len(group_list)):
        start_m, end_m = get_group_m(group_list, group_list_type, i)
        x1_group = x1[start_m:end_m, :]
        weight_group = x2[i]  # (k, n) or (n, k)
        antiquant_scale_broadcast_i = antiquant_scale_broadcast[i]  # (k, n) or (n, k)
        weight_group = weight_group * antiquant_scale_broadcast_i
        if trans_b:
            weight_group = weight_group.T

        if x1_dtype == "float8_e4m3fn":
            x1_group = x1_group * pertoken_scale_broadcast[start_m:end_m, :]

        if x1_dtype == "bfloat16":
            x1_group = x1_group.to(torch.bfloat16)
            weight_group = weight_group.to(torch.bfloat16)

        group_out = torch.matmul(x1_group.to(torch.float32), weight_group.to(torch.float32))
        group_out += bias[i, :].reshape(1, -1).to(torch.float32)
        if y_dtype == "bfloat16":
            outs.append(group_out.numpy().astype(bfloat16_conversion([y_dtype])[0]))
        else:
            outs.append(group_out.numpy().astype(np.float16))

    return outs

def trans_weight_fp4(x1_dtype, x2_dtype, x2):
    if x2_dtype == 'float4_e1m2':
        if x1_dtype == 'bfloat16':
            x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32)
        else:
            x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32).astype(np.float16)
    else:
        if x1_dtype == 'bfloat16':
            x2 = trans_np_fp4_e2m1_tensor_to_bfloat16(x2).astype(np.float32)
        else:
            x2 = trans_np_fp4_e2m1_tensor_to_bfloat16(x2).astype(np.float32).astype(np.float16)
        # 如果e2m1是使用ml_dtypes里的float4_e2m1fn生成的则可以直接转换
        # x2 = x2.astype(np.float32)
    return x2

# kn : g n1 k1 k0 n0
# nk : g k1 n1 n0 k0
def trans_nz2nd(input_data):
    nd_data = input_data.transpose(0, 2, 3, 1, 4).reshape(input_data.shape[0],
                                                          input_data.shape[2] * input_data.shape[3],
                                                          input_data.shape[1] * input_data.shape[4])
    return nd_data

def get_mx_scale(antiquant_scale, group_size, pertoken_scale, trans_b, x1_dtype):
    import torch
    print(f"kankan antiquant_scale 00000000000 {antiquant_scale.dtype}", flush=True)
    antiquant_scale = antiquant_scale.astype(np.float32)
    antiquant_scale_broadcast = np.repeat(antiquant_scale, group_size, axis=-2) if not trans_b else np.repeat(
        antiquant_scale, group_size, axis=-1)
    print(f"kankan antiquant_scale_broadcast shape 00000000000 {antiquant_scale_broadcast.shape}", flush=True)
    antiquant_scale_broadcast = torch.from_numpy(antiquant_scale_broadcast)
    if x1_dtype == "float16":
        antiquant_scale_broadcast = antiquant_scale_broadcast.to(torch.float16)

    if x1_dtype == "float8_e4m3fn":
        pertoken_scale_broadcast = torch.from_numpy(
            np.repeat(pertoken_scale.astype(np.float32), group_size, axis=-1))
        print(f"kankan pertoken_scale_broadcast shape 00000000000 {pertoken_scale_broadcast.shape}",
              flush=True)
    else:
        pertoken_scale_broadcast = None

    return antiquant_scale_broadcast, pertoken_scale_broadcast

def cal_weight_quant_a16w8(x1, x2, bias, antiquant_scale, antiquant_offset, group_list, group_list_type,
                           x1_dtype, x2_dtype, y_dtype, trans_b):
    import torch
    print("weight quant a16w8")
    assert x1_dtype == y_dtype
    x1 = torch.from_numpy(x1.astype(np.float32))
    is_f8_input = x2_dtype in ["hifloat8", "float8_e5m2", "float8_e4m3fn"]

    if is_f8_input:
        if x1_dtype == "bfloat16":
            x2 = torch.from_numpy(x2.astype(np.float32)).to(torch.bfloat16)
        else:
            x2 = torch.from_numpy(x2.astype(np.float16))
    else:
        x2 = torch.from_numpy(x2.astype(np.int32)).to(torch.int32)

    print(f"x1 = {x1}", flush=True)
    print(f"x2 = {x2}", flush=True)

    if (bias is not None) and (len(bias) != 0):
        bias = torch.from_numpy(bias.astype(np.float32))
    else:
        n_size = x2.shape[-1] if not trans_b else x2.shape[-2]
        bias = torch.from_numpy(np.zeros([x2.shape[0], n_size], dtype=np.float32))

    if (antiquant_offset is not None) and (len(antiquant_offset) != 0):
        pass
    else:
        antiquant_offset = np.zeros(antiquant_scale.shape, dtype=antiquant_scale.dtype)

    if x1_dtype == "bfloat16":
        x2 = x2.to(torch.bfloat16)
        antiquant_scale = torch.from_numpy(antiquant_scale.astype(np.float32)).to(torch.bfloat16)
        antiquant_offset = torch.from_numpy(antiquant_offset.astype(np.float32)).to(torch.bfloat16)
    elif x1_dtype == "float16":
        x2 = x2.to(torch.float16)
        antiquant_scale = torch.from_numpy(antiquant_scale)

    outs = []
    for i in range(len(group_list)):
        start_m, end_m = get_group_m(group_list, group_list_type, i)
        x1_group = x1[start_m:end_m, :]

        weight_group = x2[i]
        if trans_b:
            weight_group = weight_group.T

        antiquant_scale_i = antiquant_scale[i].reshape(1, antiquant_scale[i].shape[0])
        antiquant_offset_i = antiquant_offset[i].reshape(1, antiquant_offset[i].shape[0])

        dequant_weight = (weight_group + antiquant_offset_i) * antiquant_scale_i

        if x1_dtype == "bfloat16":
            x1_group = x1_group.to(torch.bfloat16)
            dequant_weight = dequant_weight.to(torch.bfloat16)

        group_out = torch.matmul(x1_group.to(torch.float32), dequant_weight.to(torch.float32))
        group_out += bias[i].to(torch.float32)

        if y_dtype == "bfloat16":
            outs.append(group_out.numpy().astype(bfloat16_conversion([y_dtype])[0]))
        else:
            outs.append(group_out.numpy().astype(np.float16))

    return outs