#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# CPU reference implementation for mla_prolog_v3 operator.
# Adapted from the legacy aclnnMlaPrologV3RefNew.py by removing old framework
# dependencies (libs.tools, libs.training, tensorflow) and adding standalone
# entry points (build_mla_param, test_mla_prolog_v3).
#
# Core computation logic (cal_mlaprolog + all helpers) is kept unchanged.

import os
import re
import bisect
import math
import copy
import random
import numpy as np
import torch
import torch_npu

try:
    from ml_dtypes import float8_e4m3fn as ml_float8_e4m3fn
except ImportError:
    ml_float8_e4m3fn = None

try:
    from ml_dtypes import float8_e8m0fnu as float8_e8m0
except ImportError:
    try:
        from ml_dtypes import float8_e8m0 as float8_e8m0
    except ImportError:
        float8_e8m0 = None

COLOR_YELLOW = "\033[33m"
YELLOW_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
GREEN_RESET = "\033[0m"

# ===========================================================================
# Helper functions (kept unchanged from legacy aclnnMlaPrologV3RefNew.py)
# ===========================================================================


def str_to_bool_list(s: str):
    bool_list = []
    for item in s:
        if item == 0:
            bool_list.append(False)
        else:
            bool_list.append(True)
    return bool_list


def trans_input_dtype(input_dtype):
    if input_dtype == 'fp16':
        return 'float16'
    elif input_dtype == 'int8':
        return 'int8'
    elif input_dtype == 'uint8':
        return 'uint8'
    elif input_dtype == 'bf16':
        return 'bfloat16'
    elif input_dtype == 'bool':
        return 'bool'
    elif input_dtype == 'int32':
        return 'int32'
    elif input_dtype == 'fp32':
        return 'float32'
    else:
        return input_dtype


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


def scatter_pa_nz(cache, inputs, index, data_size=16):
    # cache:(B,B,N,H) inputs:(B*S1,H) index:(B*S1)
    bn = cache.shape[0]
    bs = cache.shape[1]
    n = cache.shape[2]
    h = cache.shape[3]
    bxs1 = inputs.shape[0]
    if len(cache.shape) != 4 or len(inputs.shape) != 2 or len(index.shape) != 1:
        return cache
    elif inputs.shape[1] != h:
        return cache
    elif index.shape[0] != bxs1:
        return cache

    data_num = math.ceil(h / data_size)
    for input_index in range(bxs1):
        index_num = index[input_index]
        block_num_index = int(index_num / bs)
        for data_index in range(data_num):
            data_index_in_block = data_index * bs + (index_num % bs)
            block_size_index = int(data_index_in_block / data_num)
            h_start_index = int(data_index_in_block % data_num) * data_size
            h_end_index = h_start_index + data_size
            input_start_index = data_index * data_size
            input_end_index = input_start_index + data_size
            for n_index in range(n):
                cache[block_num_index, block_size_index, n_index, h_start_index: h_end_index] = inputs[input_index,
                                                                                                input_start_index: input_end_index]

    return cache


def scatter_pa_blk_bsnd(cache: np.ndarray,
                        input_: np.ndarray,
                        index: np.ndarray,
                        seq_len,
                        B: int):
    assert cache.ndim == 4, "cache 必须是 (num_pa_blocks, blk_size, N, H)"
    num_pa_blocks, blk_size, N, H = cache.shape

    if isinstance(seq_len, (int, np.integer)):
        S1 = int(seq_len)
        assert input_.shape[0] == B * S1, f"BSND: input_.shape[0] 必须等于 B*S1={B*S1}"
        pages_per_b = math.ceil(S1 / blk_size)
        assert index.ndim == 1 and index.shape[0] == B * pages_per_b, \
            f"BSND: index 长度应为 B*ceil(S1/blk)={B*pages_per_b}，而现在是 {index.shape[0]}"
        seq_list  = np.full((B,), S1, dtype=np.int64)
        start_list = np.arange(B, dtype=np.int64) * S1
    else:
        seq_arr = np.asarray(seq_len).reshape(-1)
        B = int(seq_arr.shape[0])
        seq_list = np.empty(B, dtype=np.int64)
        seq_list[0] = int(seq_arr[0])
        if B > 1:
            seq_list[1:] = np.diff(seq_arr).astype(np.int64)
        start_list = np.empty(B, dtype=np.int64)
        start_list[0] = 0
        if B > 1:
            start_list[1:] = seq_arr[:-1].astype(np.int64)

        expected_len = int(np.sum((seq_list + blk_size - 1) // blk_size))
        assert index.ndim == 1 and index.shape[0] == expected_len, \
            f"TND: index 长度应为 {expected_len}，而现在是 {index.shape[0]}"

    flat_page_ptr = 0
    for b in range(B):
        len_b   = int(seq_list[b])
        base_in = int(start_list[b])
        if len_b <= 0:
            continue

        pages_b = (len_b + blk_size - 1) // blk_size
        for p in range(pages_b):
            pa_blk_id = int(index[flat_page_ptr + p])
            assert 0 <= pa_blk_id < num_pa_blocks, \
                f"pa_blk_id={pa_blk_id} 越界[0,{num_pa_blocks}), (b={b}, p={p})"

            start = base_in + p * blk_size
            end   = min(start + blk_size, base_in + len_b)
            sz    = end - start
            if sz <= 0:
                continue

            cache[pa_blk_id, :sz, :, :H] = input_[start:end, :H][:, None, :]

        flat_page_ptr += pages_b

    assert flat_page_ptr == index.shape[0], \
        f"index 被消费 {flat_page_ptr} 项，但长度为 {index.shape[0]}（不一致）"
    return cache


def scatter_pa_blk_nz(cache: np.ndarray,
                      input_: np.ndarray,
                      index: np.ndarray,
                      seq_len,
                      B,
                      data_size: int = 16):
    assert cache.ndim == 4, f"cache shape must be 4D, got {cache.shape}"
    assert input_.ndim == 2, f"input shape must be 2D, got {input_.shape}"
    assert index.ndim == 2, f"index shape must be 2D, got {index.shape}"

    num_pa_blocks, blk_size, N, H_pad = cache.shape
    H = input_.shape[1]

    if isinstance(seq_len, (int, np.integer)):
        S1 = int(seq_len)
        assert input_.shape[0] == B * S1, \
            f"input first dim({input_.shape[0]}) must be B*S1({B*S1}) for BSND"
        seq_list = np.full((B,), S1, dtype=np.int64)
        start_list = np.arange(B, dtype=np.int64) * S1
        expected_pages = math.ceil(S1 / blk_size)
        assert index.shape[1] == expected_pages, \
            f"index second dim({index.shape[1]}) must be ceil(S1/blk_size)({expected_pages})"
    else:
        seq_arr = np.asarray(seq_len).reshape(-1)
        B = int(seq_len.shape[0])
        total_T = int(seq_arr[-1])
        seq_list = np.empty(B, dtype=np.int64)
        seq_list[0] = int(seq_arr[0])
        if B > 1:
            seq_list[1:] = np.diff(seq_arr).astype(np.int64)
        start_list = np.empty(B, dtype=np.int64)
        start_list[0] = 0
        if B > 1:
            start_list[1:] = seq_arr[:-1].astype(np.int64)
        assert input_.shape[0] == total_T, \
            f"input first dim({input_.shape[0]}) must equal total tokens({total_T}) for TND"

        max_s1 = int(np.max(seq_list))
        need_pages = math.ceil(max_s1 / blk_size)
        assert index.shape[1] >= need_pages, \
            f"index second dim({index.shape[1]}) must be >= ceil(max_seq/blk_size)({need_pages}) for TND"

    data_num = math.ceil(H / data_size)

    for b in range(B):
        len_b = int(seq_list[b])
        inp_start = int(start_list[b])
        for t in range(len_b):
            page_id = t // blk_size
            tok_off_in_page = t - page_id * blk_size
            pa_blk_id = int(index[b, page_id])

            inp_row = inp_start + t

            for data_idx in range(data_num):
                data_index_in_block = data_idx * blk_size + tok_off_in_page
                block_size_index = data_index_in_block // data_num
                h_start = (data_index_in_block % data_num) * data_size
                h_end = h_start + data_size

                in_h_start = data_idx * data_size
                in_h_end = min(in_h_start + data_size, H)
                out_h_end = h_start + (in_h_end - in_h_start)

                for n in range(N):
                    cache[pa_blk_id, block_size_index, n, h_start:out_h_end] = \
                        input_[inp_row, in_h_start:in_h_end]

    return cache


def s8_saturation(inputdata):
    inputdata = torch.where(inputdata > 127, 127, inputdata)
    inputdata = torch.where(inputdata < -128, -128, inputdata)
    return inputdata.to(torch.int8)


def s9_saturation(inputdata):
    inputdata = torch.where(inputdata > 255, 255, inputdata)
    inputdata = torch.where(inputdata < -256, -256, inputdata)
    return inputdata


def quant(x, qscale):
    scaled_values = (x * qscale).round().to(torch.float32)
    s9_res = s9_saturation(scaled_values)
    s8_res_cal = s8_saturation(s9_res)
    return s8_res_cal


def numpy_float8_e4m3fn():
    try:
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                           "Please install with `pip3 install ml-dtypes`")

def quant_ckv_per_tensor(input, quant_scale_ckv):
    scaled_value = input * quant_scale_ckv
    scaled_value = np.round(scaled_value, 8)
    scaled_value = scaled_value.astype(numpy_float8_e4m3fn(), copy=False)
    return scaled_value


def dynamic_quant(inputs, smooth_scale):
    T = inputs.size(0)
    H = inputs.size(1)
    y = torch.zeros(T, H).to(torch.int32)
    scale = torch.zeros(T).to(torch.float32)
    inputs = inputs.reshape(T, H).to(torch.float32)
    if smooth_scale!=None:
        if len(smooth_scale.shape) != 2 or smooth_scale.shape[1] != H:
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        for bs_index in range(T):
            abs_bs_tensor = torch.abs(inputs[bs_index, :] * smooth_scale[0, :])
            scale_bs = abs_bs_tensor.max() / 127
            scale[bs_index] = scale_bs
            y[bs_index:] = torch.round(inputs[bs_index:] * smooth_scale[0, :] / scale_bs)
    else:
        for bs_index in range(T):
            abs_bs_tensor = torch.abs(inputs[bs_index, :])
            scale_bs = abs_bs_tensor.max() / 127
            scale[bs_index] = scale_bs
            y[bs_index:] = torch.round(inputs[bs_index:]/ scale_bs)
    return y, scale


def get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max

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

def _mx_calculate_share_exp(fp_array: np.ndarray, scale_axis: int, mx_ele_dtype: str):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(np.log2(max_norm))
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
    res = np.floor(
        np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax
    res = res + FP32_EXPONENT_BIAS
    res[fp_abs_max == 0] = -float("inf")
    return res

def _mx_round_mantissa(fp_array: np.ndarray, round_mode: str):
    if round_mode in ("rint", "even"):
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array

def _mx_quantize_to_element_format(fp_array: np.ndarray, share_exp: np.ndarray,
                                   mx_ele_dtype: str, round_mode: str):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r'e(\d+)m(\d+)', mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = fp_array / (2 ** (share_exp - 127))
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))
                              ).astype(fp_array.dtype, copy=False)
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    max_norm = get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret

def pad_to_even(tensor: np.ndarray, axis: int) -> np.ndarray:
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")

    shape = tensor.shape
    length = shape[axis]

    if length % 2 == 0:
        return tensor

    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)

    padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
    return padded_tensor

def _mx_undo_reshape_to_blocks(fp_array: np.ndarray, axis: int,
                               orig_shape: tuple, padded_shape: tuple):
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array

def interleave(tensor: np.ndarray, axis: int, n_group: int = 2) -> np.ndarray:
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")

    group_length = length // n_group
    shape = list(tensor.shape)

    new_shape = (
        shape[:axis] +
        [group_length,2] +
        shape[axis+1:])
    reshaped = tensor.reshape(new_shape)

    transpose_order = (
        list(range(0, axis+1)) +
        list(range(axis + 2, len(new_shape))) +
        [axis+1,])

    transposed = reshaped.transpose(transpose_order)

    return transposed


def dynamic_mx_quant_cq(fp_array: np.ndarray, mx_ele_dtype: str = "float4_e2m1",
                axis: int = -1, block_size: int = 32, round_mode: str = "rint") -> tuple:
    if not isinstance(fp_array, np.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")
    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)

    share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1
    share_exp[(share_exp - 127) > scale_emax] = float("NaN")
    share_exp[(share_exp - 127) < -scale_emax] = -scale_emax
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    # Convert element dtype
    ele_dtype_np = numpy_float8_e4m3fn() if mx_ele_dtype == "float8_e4m3fn" else None
    if ele_dtype_np is None:
        # For other dtypes, attempt eval (float4_e2m1 etc. from ml_dtypes)
        try:
            from ml_dtypes import float8_e4m3fn as _f8
            ele_dtype_np = eval(f"ml_float8_e4m3fn") if mx_ele_dtype == "float8_e4m3fn" else eval(f"{mx_ele_dtype}")
        except Exception:
            ele_dtype_np = np.float32
    scale_array = share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array_pad = pad_to_even(scale_array, axis=axis)
    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)
    scale_array = scale_array_pad.astype("uint8", copy=False)
    return scale_array, ele_array


def dynamic_mx_quant_qn(x, mla_param):
    scale_max = np.float32(448.0)
    x = x.astype('float32')
    input_mul = x
    input_abs = np.abs(input_mul)
    input_max = np.max(input_abs, axis=-1, keepdims=True)
    scale = input_max * (np.float32(1.0) / scale_max)
    input_scaled = input_mul / scale
    if mla_param['action_type'] != 'bm_output_gold':
        output_data = input_scaled.astype(numpy_float8_e4m3fn(), copy=False)
    else:
        output_data = input_scaled
    return scale, output_data


def dynamic_quant_ckv_with_amax(inputs: torch.Tensor, amax: torch.Tensor, smooth_scale: torch.Tensor = None):
    T, N, H = inputs.shape
    inputs = inputs.to(torch.float32)
    amax = amax.to(torch.float32).clamp(min=1e-8)

    if smooth_scale is not None:
        if smooth_scale.ndim != 3 or smooth_scale.shape[1:] != (N, H):
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        scaled_inputs = inputs * smooth_scale
    else:
        scaled_inputs = inputs

    scale = amax / 127.0
    y = torch.round(scaled_inputs / scale)

    return y, scale


def dynamic_quant_ckv_with_amax_fp8e4m3(inputs: torch.Tensor, amax: torch.Tensor):
    T, N = inputs.shape
    scale = amax / 448.0
    y = torch.clamp(inputs / scale, min=-448, max=448)
    return y, scale


def dynamic_quant_without_smooth_scale(inputs, out_deqq_shape_shape):
    T = inputs.size(0)
    N = inputs.size(1)
    H = inputs.size(2)
    quant_loops = inputs.size(0)
    eles_with_one_scale = inputs.size(1) * inputs.size(2)
    if len(out_deqq_shape_shape) == 3:
        quant_loops = inputs.size(0) * inputs.size(1)
        eles_with_one_scale = inputs.size(2)

    y = torch.zeros(quant_loops, eles_with_one_scale).to(torch.int32)
    scale = torch.zeros(quant_loops).to(torch.float32)
    inputs = inputs.reshape(quant_loops, eles_with_one_scale).to(torch.float32)
    max_values, _ = torch.max(torch.abs(inputs), dim=-1, keepdim=True)
    scale = max_values / 127
    y = torch.round(inputs/scale)
    y = s8_saturation(y)
    if len(out_deqq_shape_shape) == 2:
        return y.reshape(T, N, H), scale.reshape(quant_loops, 1).to(torch.float64)
    else:
        return y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64)


def dequant(inputs, deq_scale_q_nope, quant_scale_ckv):
    org_shape = inputs.size()
    quant_loops = inputs.size(0)
    eles_with_one_scale = inputs.size(1) * inputs.size(2)
    if len(deq_scale_q_nope.size()) == 3:
        quant_loops = inputs.size(0) * inputs.size(1)
        eles_with_one_scale = inputs.size(2)
    inputs = inputs.reshape(quant_loops, eles_with_one_scale)
    deq_scale_q_nope = deq_scale_q_nope.reshape(quant_loops, 1)
    for quant_idx in range(quant_loops):
        inputs[quant_idx, :] = inputs[quant_idx, :] * quant_scale_ckv / (deq_scale_q_nope[quant_idx, 0])
    return inputs.reshape(org_shape)


def trans_torch_fp8_e8m0_to_bf16(array):
    new_array = torch.zeros_like(array).to(torch.bfloat16)
    for i in range(array.size(0)):
        for j in range(array.size(1)):
            new_value = ieee_754_conversion(0, int(array[i][j]), 0)
            new_array[i][j] = new_value
    return new_array


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))
    return sign_mult * (2 ** exponent) * mant_mult


def convert_dict_values_to_torch(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            if value.dtype == numpy_float8_e4m3fn():
                output_dict[key] = torch.tensor(value.astype(np.float32)).to(torch.float8_e4m3fn)
            elif float8_e8m0 is not None and value.dtype == float8_e8m0:
                output_dict[key] = torch.tensor(value.astype(np.int8))
            else:
                output_dict[key] = torch.from_numpy(value)
        else:
            output_dict[key] = value
    return output_dict

def get_current_seq_len(token_idx: int, actual_seq_length: np.ndarray) -> int:
    prefix_sum = actual_seq_length.flatten().tolist()
    batch_id = bisect.bisect_right(prefix_sum, token_idx)
    if batch_id == 0:
        seq_len = prefix_sum[0]
    else:
        seq_len = prefix_sum[batch_id] - prefix_sum[batch_id - 1]
    return seq_len


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


# ===========================================================================
# Core computation logic (kept unchanged from legacy aclnnMlaPrologV3RefNew.py)
# ===========================================================================

def cal_mlaprolog(mla_param):
    B = mla_param['B']
    S1 = mla_param['S1']
    S2 = mla_param['S2']
    D = mla_param['D']
    Dr = mla_param['Dr']
    N1 = mla_param['N1']
    N2 = mla_param['N2']
    He = mla_param['He']
    Hckv = mla_param['Hckv']
    Hcq = mla_param['Hcq']
    T = mla_param['T']
    BlockSize = mla_param['block_size']
    index_table = mla_param['cache_index_tensor']
    x_dtype = mla_param["x_dtype"]
    uqqr_dtype = mla_param["w_uq_qr_dtype"]
    uk_dtype = mla_param["w_uk_dtype"]
    kv_dtype = mla_param["kv_cache_dtype"]
    kr_dtype = mla_param["kr_cache_dtype"]

    token_x = mla_param["x_tensor"]
    cos = mla_param["cos_tensor"]
    sin = mla_param["sin_tensor"]

    qnorm_flag = mla_param["qnorm_flag"]

    deq_scale_q_nope = torch.empty(0)
    deq_scale_qcqr = None
    out_qnorm = torch.empty(0)
    out_deq_qnorm = torch.empty(0)
    enable_quant_output = True if mla_param["query_quant_mode"] == 1 and (mla_param["weight_quant_mode"] == 2 or mla_param["weight_quant_mode"] == 3) else False
    quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
    actual_seq_lengths = mla_param["actual_seq_len"]
    if enable_quant_output:
        out_deqq_shape_shape = mla_param['out_deqq_shape']

    # BS合轴
    if not mla_param["t_flag"]:
        T = B * S1
        token_x = token_x.reshape(T, He)
        cos = cos.reshape(T, Dr)
        sin = sin.reshape(T, Dr)
        seq_len = S1
    else:
        seq_len = actual_seq_lengths

    # -------------------------------------------------------------------
    # matmul1 : token_x(B*S1,He) * w_dq (He,Hcq) -> matmul1_res(B*S1,Hcq)
    # -------------------------------------------------------------------
    w_dq = mla_param["w_dq_tensor"]
    matmul1_dtype = torch.float32
    # matmul1预处理
    token_x_new = token_x
    if x_dtype == 'int8':
        token_x_new = token_x_new.to(torch.int32)
        token_x = token_x.to(torch.int32)
        w_dq = w_dq.to(torch.int32)
        matmul1_dtype = torch.int32
    elif x_dtype == 'float8_e4m3fn':
        token_x_new = token_x_new.to(torch.bfloat16)
        token_x = token_x.to(torch.bfloat16)
        deq_scale_x = mla_param["deq_scale_x_tensor"]
        if not mla_param["t_flag"]:
            deq_scale_x = deq_scale_x.reshape(B * S1, He // 32)
        else:
            deq_scale_x = deq_scale_x.reshape(T, He // 32)
        deq_scale_x = trans_torch_fp8_e8m0_to_bf16(deq_scale_x)
        xs0 = deq_scale_x.shape[0]
        xs1 = deq_scale_x.shape[1]
        grp_size = 32
        for xs0_idx in range(xs0):
            for xs1_idx in range(xs1):
                token_cur = token_x_new[xs0_idx:xs0_idx + 1, xs1_idx * grp_size:(xs1_idx + 1) * grp_size]
                scale_x = deq_scale_x[xs0_idx:xs0_idx + 1, xs1_idx:(xs1_idx + 1)]
                scale_x = torch.full((1, grp_size), scale_x.item())
                token_cur = token_cur * scale_x
                token_x_new[xs0_idx:xs0_idx + 1, xs1_idx * grp_size:(xs1_idx + 1) * grp_size] = token_cur
        w_dq = mla_param["w_dq_tensor"]
        w_dq = w_dq.to(torch.bfloat16)
        deq_scale_w_dq =  mla_param["deq_scale_w_dq_tensor"]
        deq_scale_w_dq = trans_torch_fp8_e8m0_to_bf16(deq_scale_w_dq)
        dqs0 = deq_scale_w_dq.shape[0]
        dqs1 = deq_scale_w_dq.shape[1]
        for dqs0_idx in range(dqs0):
            for dqs1_idx in range(dqs1):
                scale_w_dq = deq_scale_w_dq[dqs0_idx:dqs0_idx + 1, dqs1_idx:(dqs1_idx + 1)]
                w_dq[dqs1_idx * grp_size:(dqs1_idx + 1) * grp_size, dqs0_idx:dqs0_idx + 1] *= scale_w_dq
    # matmul1计算
    x_shape = "(T,He)" if mla_param["t_flag"] else "(B*S1,He)"
    token_x_new = token_x_new.to(torch.float32)
    w_dq = w_dq.to(torch.float32)
    matmul1_res = torch.matmul(token_x_new, w_dq).to(matmul1_dtype)
    # matmul1后处理
    if mla_param["weight_quant_mode"] == 2:
        deq_scale_x = mla_param["deq_scale_x_tensor"]
        deq_scale_w_dq = mla_param["deq_scale_w_dq_tensor"]
        matmul1_res = matmul1_res.to(torch.float32)
        for t_index in range(T):
            matmul1_res[t_index, :] = matmul1_res[t_index, :] * deq_scale_x[t_index, 0]
        for h_index in range(Hcq):
            matmul1_res[:, h_index] = matmul1_res[:, h_index] * deq_scale_w_dq[0, h_index]
    elif x_dtype == 'bfloat16':
        matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)
    matmul1_res_shape = "(T,Hcq)" if mla_param["t_flag"] else "(B*S1,Hcq)"

    # ----------------------------------------------------------------------
    # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
    # ----------------------------------------------------------------------
    ep1 = float(mla_param["epsilon_cq"])
    gamma1 = mla_param["gamma_cq_tensor"]
    norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
    norm1_res *= gamma1
    qc_qr_scale = float(mla_param["qc_qr_scale"])
    norm1_res *= qc_qr_scale

    # ----------------------------------------------------------------------------------
    # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
    # ----------------------------------------------------------------------------------
    w_uq_qr = mla_param["w_uq_qr_tensor"]
    # matmul2预处理
    matmul2_dtype = torch.float32
    if mla_param["weight_quant_mode"] == 1 or mla_param["weight_quant_mode"] == 2:
        w_uq_qr = w_uq_qr.to(torch.int32)
        matmul2_dtype = torch.int32
        smo_scale_cq = mla_param["smo_scale_cq_tensor"]
        norm1_res, deq_scale_qcqr = dynamic_quant(norm1_res, smo_scale_cq)
        if qnorm_flag:
            if mla_param['action_type'] == 'bm_output_gold':
                out_qnorm = norm1_res.to(torch.int8)
                out_deq_qnorm = deq_scale_qcqr.to(torch.float64)
            else:
                out_qnorm = norm1_res
                out_deq_qnorm = deq_scale_qcqr
    elif mla_param["weight_quant_mode"] == 3:
        w_uq_qr = w_uq_qr.to(torch.bfloat16)
        deq_scale_uqqr = mla_param["deq_scale_w_uqqr_tensor"]
        deq_scale_uqqr = trans_torch_fp8_e8m0_to_bf16(deq_scale_uqqr)
        uqqrs0 = deq_scale_uqqr.shape[0]
        uqqrs1 = deq_scale_uqqr.shape[1]
        grp_size = 32
        for uqqrs0_idx in range(uqqrs0):
            for uqqrs1_idx in range(uqqrs1):
                scale_uqqr = deq_scale_uqqr[uqqrs0_idx:uqqrs0_idx + 1, uqqrs1_idx:(uqqrs1_idx + 1)]
                w_uq_qr[uqqrs1_idx * grp_size:(uqqrs1_idx + 1) * grp_size, uqqrs0_idx:uqqrs0_idx + 1] *= scale_uqqr
        norm1_res = norm1_res.to(torch.bfloat16)
        deq_scale_qcqr_np, norm1_res_np = dynamic_mx_quant_cq(norm1_res.float().numpy(), "float8_e4m3fn")
        norm1_res = torch.tensor(norm1_res_np.astype(np.float32)).to(torch.float8_e4m3fn)
        deq_scale_qcqr = torch.from_numpy(deq_scale_qcqr_np)
        deq_scale_qcqr = deq_scale_qcqr.reshape(deq_scale_qcqr.shape[0], deq_scale_qcqr.shape[1] * deq_scale_qcqr.shape[2])
        if qnorm_flag:
            if mla_param['action_type'] == 'bm_output_gold':
                out_qnorm = norm1_res.to(torch.float32)
                out_deq_qnorm = deq_scale_qcqr.to(torch.int8)
            else:
                out_qnorm = norm1_res
                out_deq_qnorm = deq_scale_qcqr
        norm1_res = norm1_res.to(torch.bfloat16)
        deq_scale_qcqr = trans_torch_fp8_e8m0_to_bf16(deq_scale_qcqr)
        qcqrs0 = deq_scale_qcqr.shape[0]
        qcqrs1 = deq_scale_qcqr.shape[1]
        grp_size = 32
        for qcqrs0_idx in range(qcqrs0):
            for qcqrs1_idx in range(qcqrs1):
                normal_cur = norm1_res[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx * grp_size:(qcqrs1_idx + 1) * grp_size]
                scale_qcqr = deq_scale_qcqr[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx:(qcqrs1_idx + 1)]
                scale_qcqr = torch.full((1, grp_size), scale_qcqr.item())
                normal_cur = normal_cur * scale_qcqr
                norm1_res[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx * grp_size:(qcqrs1_idx + 1) * grp_size] = normal_cur
    elif mla_param["weight_quant_mode"] == 0:
        norm1_res = norm1_res.to(torch.bfloat16).to(torch.float32)
        if qnorm_flag:
            out_qnorm = norm1_res

    # matmul2计算
    norm1_res = norm1_res.to(torch.float32)
    w_uq_qr = w_uq_qr.to(torch.float32)
    matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(matmul2_dtype)
    # matmul2后处理
    if mla_param["weight_quant_mode"] == 1 or mla_param["weight_quant_mode"] == 2:
        deq_scale_uqqr = mla_param["deq_scale_w_uqqr_tensor"]
        matmul2_res = matmul2_res.to(torch.float32)
        for t_index in range(T):
            matmul2_res[t_index, :] = matmul2_res[t_index, :] * deq_scale_qcqr[t_index]
        for nddr_index in range(matmul2_res.shape[1]):
            matmul2_res[:, nddr_index] = matmul2_res[:, nddr_index] * deq_scale_uqqr[0, nddr_index]
    elif uqqr_dtype == 'bfloat16':
        matmul2_res = matmul2_res.to(torch.bfloat16).to(torch.float32)
    matmul2_res = matmul2_res.reshape(T, N1, D + Dr)

    # -------------------------------------------------------------------------------------
    # splitD1 : matmul2_res(B*S1,N,D+Dr) -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
    # -------------------------------------------------------------------------------------
    splitd1_res1 = matmul2_res[:, :, :D]
    splitd1_res2 = matmul2_res[:, :, D:]
    splitd1_res1_shape = "(T,N1,D)" if mla_param["t_flag"] else "(B*S1,N1,D)"
    splitd1_res2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B*S1,N1,Dr)"
    # -------------------------------------------------------------------------
    # matmul3 : -> splitd1_res1(B*S1,N,D) * w_uk(N,D,Hckv) -> out1(B,S1,N,Hckv)
    # -------------------------------------------------------------------------
    w_uk = mla_param["w_uk_tensor"]
    # matmul3预处理
    splitd1_res1 = splitd1_res1.transpose(0, 1)
    out1 = torch.zeros((N1, T, Hckv))
    matmul3_dtype = torch.float32
    if mla_param["weight_quant_mode"] == 3:
        matmul3_dtype = torch.bfloat16
        w_uk = w_uk.to(torch.bfloat16)
        splitd1_res1 = splitd1_res1.to(torch.bfloat16)
    else:
        w_uk = w_uk.to(torch.float32)
        if uk_dtype == 'bfloat16':
            splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
    # matmul3计算
    for n1_index in range(N1):
        out1[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :].to(torch.float32), w_uk[n1_index, :, :].to(torch.float32)).to(matmul3_dtype)
    # matmul3后处理
    # DynamicQuant
    out1 = out1.transpose(0, 1)
    if enable_quant_output:
        if mla_param["weight_quant_mode"] == 3:
            out1 = out1.to(torch.bfloat16).to(torch.float32)
            deq_scale_q_nope_np, out1_np = dynamic_mx_quant_qn(out1.numpy(), mla_param)
            if mla_param['action_type'] == 'bm_output_gold':
                out1 = torch.tensor(out1_np.astype(np.float32))
                deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np).to(torch.float64)
            else:
                out1 = torch.tensor(out1_np.astype(np.float32)).to(torch.float8_e4m3fn)
                deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np)
        else:
            out1 = out1.to(torch.bfloat16).to(torch.float32)
            out1, deq_scale_q_nope = dynamic_quant_without_smooth_scale(out1, out_deqq_shape_shape)
    out1 = out1 if mla_param["t_flag"] else out1.reshape(B, S1, N1, Hckv)
    out1_shape = "(T,N1,Hckv)" if mla_param["t_flag"] else "(B,S1,N,Hckv)"

    # -------------------------------------------------------------------------------------
    # rotary1 : -> splitd1_res2(B*S1,N,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> out2(B,S1,N,Dr)
    # -------------------------------------------------------------------------------------
    splitd1_res2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B*S1,N1,Dr)"
    cos_shape = "(T,Dr)" if mla_param["t_flag"] else "(B*S1,Dr)"
    expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
    expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
    q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
    out2 = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
    if enable_quant_output:
        out2 = out2.to(torch.bfloat16).to(torch.float32)
        out2 = dequant(out2, deq_scale_q_nope, quant_scale_ckv)
    out2 = out2 if mla_param["t_flag"] else out2.reshape(B, S1, N1, Dr)
    out2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B,S1,N1,Dr)"

    # -------------------------------------------------------------------------------
    # matmul4 : token_x(B*S1,He) * w_kv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
    # -------------------------------------------------------------------------------
    w_kv_kr = mla_param["w_dkv_kr_tensor"]
    # matmul4预处理
    matmul4_dtype = torch.float32
    if mla_param["weight_quant_mode"] == 2:
        w_kv_kr = w_kv_kr.to(torch.int32)
        matmul4_dtype = torch.int32
    elif mla_param["weight_quant_mode"] == 3:
        w_kv_kr = w_kv_kr.to(torch.bfloat16)
        deq_scale_dkvkr = mla_param["deq_scale_w_dkvkr_tensor"]
        deq_scale_dkvkr = trans_torch_fp8_e8m0_to_bf16(deq_scale_dkvkr)
        dkvkrs0 = deq_scale_dkvkr.shape[0]
        dkvkrs1 = deq_scale_dkvkr.shape[1]
        for dkvkrs0_idx in range(dkvkrs0):
            for dkvkrs1_idx in range(dkvkrs1):
                scale_dkvkr = deq_scale_dkvkr[dkvkrs0_idx:dkvkrs0_idx + 1, dkvkrs1_idx:(dkvkrs1_idx + 1)]
                w_kv_kr[dkvkrs1_idx * grp_size:(dkvkrs1_idx + 1) * grp_size, dkvkrs0_idx:dkvkrs0_idx + 1] *= scale_dkvkr
    # matmul4计算
    matmul4_res = torch.matmul(token_x_new.to(torch.float32), w_kv_kr.to(torch.float32)).to(matmul4_dtype)
    # matmul4后处理
    if mla_param["weight_quant_mode"] == 2:
        deq_scale_x = mla_param["deq_scale_x_tensor"]
        deq_scale_dkvkr = mla_param["deq_scale_w_dkvkr_tensor"]
        matmul4_res = matmul4_res.to(torch.float32)
        for t_index in range(T):
            matmul4_res[t_index, :] = matmul4_res[t_index, :] * deq_scale_x[t_index, 0]
        for h_index in range(Hckv + Dr):
            matmul4_res[:, h_index] = matmul4_res[:, h_index] * deq_scale_dkvkr[0, h_index]
    elif x_dtype == 'bfloat16':
        matmul4_res = matmul4_res.to(torch.bfloat16).to(torch.float32)
    matmul4_res_shape = "(T,Hckv+Dr)" if mla_param["t_flag"] else "(B*S1,Hckv+Dr)"

    # -------------------------------------------------------------------------------------
    # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
    # -------------------------------------------------------------------------------------
    splitd2_res1 = matmul4_res[:, :Hckv]
    splitd2_res2 = matmul4_res[:, Hckv:]
    splitd2_res1_shape = "(T,Hckv)" if mla_param["t_flag"] else "(B*S1,Hckv)"
    splitd2_res2_shape = "(T,Dr)" if mla_param["t_flag"] else "(B*S1,Dr)"

    # -------------------------------------------------------------------------------------
    # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
    # -------------------------------------------------------------------------------------
    k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
    rotary2_res = (k * cos) + (rotate_half(k) * sin)
    # rotary2后处理
    quant_scale_ckr = mla_param["quant_scale_ckr_tensor"]
    if mla_param["weight_quant_mode"] == 1 and mla_param["kv_quant_mode"] == 2:
        rotary2_res = quant(rotary2_res, quant_scale_ckr)

    # ----------------------------------------------------------------------------
    # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
    # ----------------------------------------------------------------------------
    ep2 = float(mla_param["epsilon_ckv"])
    gamma2 = mla_param["gamma_ckv_tensor"]
    norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
    norm2_res *= gamma2

    Dtile = Hckv
    # rmsnorm2后处理
    if mla_param["kv_quant_mode"] == 1 or mla_param["kv_quant_mode"] == 2:
        if mla_param["weight_quant_mode"] == 3:
            quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
            norm2_res_np = quant_ckv_per_tensor(norm2_res.numpy(), quant_scale_ckv.numpy())
            if mla_param['action_type'] == 'bm_output_gold':
                norm2_res = torch.tensor(norm2_res_np.astype(np.float32))
            else:
                norm2_res = torch.tensor(norm2_res_np.astype(np.float32)).to(torch.float8_e4m3fn)
        else:
            quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
            norm2_res = quant(norm2_res, quant_scale_ckv)
    elif mla_param["kv_quant_mode"] == 3:
        if mla_param["weight_quant_mode"] == 3:
            norm2_res = norm2_res.reshape(T * Hckv//mla_param["tile_size"], mla_param["tile_size"])
            eps = 1e-4
            amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
            amax = torch.clamp(amax, min=eps)
            norm2_res, deq_scale_ckv = dynamic_quant_ckv_with_amax_fp8e4m3(norm2_res, amax)
            deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
            norm2_res = norm2_res.reshape(T, Hckv).to(torch.float8_e4m3fn)
            rotary2_res_bf16 = rotary2_res.to(torch.bfloat16)
            if mla_param["ckvkr_repo_mode"] == 1:
                norm2_res = torch.cat((norm2_res, rotary2_res_bf16.view(torch.float8_e4m3fn)), axis = -1)
                Dtile = Dtile + Dr * 2
            if mla_param["quant_scale_repo_mode"] == 1:
                norm2_res = torch.cat((norm2_res, deq_scale_ckv.view(torch.float8_e4m3fn)), axis = -1)
                Dtile = Dtile + Hckv//mla_param["tile_size"] * 4
            if mla_param['action_type'] == 'bm_output_gold':
                norm2_res = norm2_res.to(torch.float32)
                deq_scale_ckv = deq_scale_ckv.to(torch.float64)
        else:
            norm2_res = norm2_res.reshape(T, Hckv//mla_param["tile_size"], mla_param["tile_size"])
            eps = 1e-8
            amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
            amax = torch.clamp(amax, min=eps)*mla_param["k_nope_clip_alpha_tensor"]
            clip_res = torch.clamp(norm2_res, min=-amax, max=amax)
            norm2_res, deq_scale_ckv = dynamic_quant_ckv_with_amax(clip_res, amax)
            deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
            norm2_res = norm2_res.reshape(T, Hckv).to(torch.int8)
            rotary2_res_bf16 = rotary2_res.to(torch.bfloat16)
            if mla_param["ckvkr_repo_mode"] == 1:
                norm2_res = torch.cat((norm2_res, rotary2_res_bf16.view(torch.int8)), axis = -1)
                Dtile = Dtile + Dr * 2
            if mla_param["quant_scale_repo_mode"] == 1:
                norm2_res = torch.cat((norm2_res, deq_scale_ckv.view(torch.int8)), axis = -1)
                Dtile = Dtile + Hckv//mla_param["tile_size"] * 4
    # -------------------------------------------------------------------------------------------------------
    # scatter1 : norm2_res(B*S1,Hckv) * kv_cache(B,N2,S2,Hckv/B,B,N2,Hckv) -> out3(B,N2,S2,Hckv/B,B,N2,Hckv)
    # -------------------------------------------------------------------------------------------------------
    kv_cache = copy.deepcopy(mla_param["kv_cache_tensor"])

    if kv_cache.numel() == 0:
        out3 = kv_cache
        out4 = copy.deepcopy(mla_param["kr_cache_tensor"])
        return out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm


    out3_shape = kv_cache.shape
    out3_info = "(B,B,N2,Hckv)"
    if not mla_param['pa_flag']:
        kv_cache = kv_cache.transpose(2, 1)
        out3_info = "(B,N2,S2,Hckv)"
    if kv_dtype in ["int8","float8_e4m3fn"]:
        scatter_size = 32
    elif kv_dtype == "bf16":
        scatter_size = 16
        kv_cache = kv_cache.to(torch.bfloat16)
    else:
        scatter_size = 16


    if mla_param['cache_mode'] == "PA_BLK_NZ":
        kv_cache = scatter_pa_blk_nz(kv_cache, norm2_res, index_table, seq_len, scatter_size)
    elif mla_param['cache_mode'] == "PA_BLK_BSND":
        kv_cache = scatter_pa_blk_bsnd(kv_cache, norm2_res, index_table, seq_len, B)
    elif mla_param['cache_mode'] == "PA_NZ":
        kv_cache = scatter_pa_nz(kv_cache, norm2_res, index_table.reshape(T), scatter_size)
    else:
        if mla_param['cache_mode'] == "PA_BSND":
            B_num = out3_shape[0]
            kv_cache = kv_cache.reshape(B_num * BlockSize, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    kv_cache[index_table.reshape(T)[i], j, :] = norm2_res[i, :]
        else:
            kv_cache = kv_cache.reshape(B * S2, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    kv_cache[i, j, :] = norm2_res[i, :]
    out3 = kv_cache.reshape(out3_shape)

    # -------------------------------------------------------------------------------------
    # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
    # -------------------------------------------------------------------------------------
    k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
    rotary2_res = (k * cos) + (rotate_half(k) * sin)
    # rotary2后处理
    quant_scale_ckr = mla_param["quant_scale_ckr_tensor"]
    if mla_param["flaglist"][19]:
        rotary2_res = quant(rotary2_res, quant_scale_ckr)

    # ----------------------------------------------------------------------------------------------
    # scatter2 : rotary2_res(B*S1,Dr) * kr_cache(B,N2,S2,Dr/B,B,N2,Dr) -> out4(B,N2,S2,Dr/B,B,N2,Dr)
    # ----------------------------------------------------------------------------------------------
    if mla_param['ckvkr_repo_mode'] == 1:
        out4 = copy.deepcopy(mla_param["kr_cache_tensor"])
    else:
        kr_cache = copy.deepcopy(mla_param["kr_cache_tensor"])
        out4_shape = kr_cache.shape
        out4_info = "(B,B,N2,Dr)"
        if not mla_param['pa_flag']:
            kr_cache = kr_cache.transpose(2, 1)
            out4_info = "(B,N2,S2,Dr)"
        if kr_dtype in ["int8","float8_e4m3fn"]:
            scatter_size = 32
        elif kr_dtype == "bf16":
            scatter_size = 16
            kr_cache = kr_cache.to(torch.bfloat16)
        else:
            scatter_size = 16
        if mla_param['cache_mode'] == "PA_BLK_NZ":
            kr_cache = scatter_pa_blk_nz(kr_cache, rotary2_res, index_table, seq_len, scatter_size)
        elif mla_param['cache_mode'] == "PA_BLK_BSND":
            kr_cache = scatter_pa_blk_bsnd(kr_cache, rotary2_res, index_table, seq_len, B)
        elif mla_param['cache_mode'] == "PA_NZ":
            kr_cache = scatter_pa_nz(kr_cache, rotary2_res, index_table.reshape(T), scatter_size)
        else:
            if mla_param['cache_mode'] == "PA_BSND":
                B_num = out4_shape[0]
                kr_cache = kr_cache.reshape(B_num * BlockSize, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[index_table.reshape(T)[i], j, :] = rotary2_res[i, :]
            else:
                kr_cache = kr_cache.reshape(B * S2, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[i, j, :] = rotary2_res[i, :]
        out4 = kr_cache.reshape(out4_shape)


    return out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm


# ===========================================================================
# New standalone entry points
# ===========================================================================

# Fixed operator dimensions
HCQ = 1536
HCKV = 512
D = 128
DR = 64
N2 = 1


def _create_tensor(shape, dtype, generator=None):
    """Create a random tensor with the given shape and dtype."""
    if dtype == torch.int8:
        return torch.randint(-10, 10, shape, dtype=torch.int8)
    elif dtype == torch.float8_e4m3fn:
        return torch.randn(shape, dtype=torch.bfloat16, generator=generator).to(torch.float8_e4m3fn)
    else:
        return torch.randn(shape, dtype=dtype, generator=generator)


def build_mla_param(params):
    """Construct mla_param dict from test parameters.

    Creates all input tensors with random data and builds the mla_param
    dictionary compatible with cal_mlaprolog().

    Args:
        params: dict with keys: batch_size, seq_len, head_num, He, dtype,
                cache_mode, block_size, weight_quant_mode, kv_cache_quant_mode,
                query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode.

    Returns:
        (mla_param, npu_inputs) tuple where:
            mla_param: dict for cal_mlaprolog()
            npu_inputs: dict of tensors for NPU operator call
    """
    B = params['batch_size']
    S1 = params['seq_len']
    N1 = params['head_num']
    He = params['He']
    T = B * S1
    S2 = S1  # Same as S1 for decode scenario
    block_size = params['block_size']
    cache_mode = params['cache_mode']
    weight_quant_mode = params['weight_quant_mode']
    kv_quant_mode = params['kv_cache_quant_mode']
    query_quant_mode = params['query_quant_mode']
    ckvkr_repo_mode = params['ckvkr_repo_mode']
    quant_scale_repo_mode = params['quant_scale_repo_mode']
    qnorm_flag = params.get('query_norm_flag', False)
    tile_size = params.get('tile_size', 128)
    qc_qr_scale = params.get('qc_qr_scale', 1.0)
    kc_scale = params.get('kc_scale', 1.0)
    epsilon_cq = params.get('epsilon_cq', 1e-5)
    epsilon_ckv = params.get('epsilon_ckv', 1e-5)

    seed = params.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # --- Determine tensor dtypes based on quant modes ---

    # token_x dtype
    if weight_quant_mode in [0, 1]:
        x_dtype = torch.bfloat16
        x_dtype_str = 'bfloat16'
    elif weight_quant_mode == 2:
        x_dtype = torch.int8
        x_dtype_str = 'int8'
    else:  # 3
        x_dtype = torch.float8_e4m3fn
        x_dtype_str = 'float8_e4m3fn'

    # w_dq dtype
    if weight_quant_mode in [0, 1]:
        w_dq_dtype = torch.bfloat16
    elif weight_quant_mode == 2:
        w_dq_dtype = torch.int8
    else:
        w_dq_dtype = torch.float8_e4m3fn

    # w_uq_qr dtype
    if weight_quant_mode == 0:
        w_uq_qr_dtype = torch.bfloat16
        w_uq_qr_dtype_str = 'bfloat16'
    elif weight_quant_mode in [1, 2]:
        w_uq_qr_dtype = torch.int8
        w_uq_qr_dtype_str = 'int8'
    else:
        w_uq_qr_dtype = torch.float8_e4m3fn
        w_uq_qr_dtype_str = 'float8_e4m3fn'

    # w_uk dtype - always bf16
    w_uk_dtype = torch.bfloat16
    w_uk_dtype_str = 'bfloat16'

    # w_dkv_kr dtype
    if weight_quant_mode in [0, 1]:
        w_dkv_kr_dtype = torch.bfloat16
    elif weight_quant_mode == 2:
        w_dkv_kr_dtype = torch.int8
    else:
        w_dkv_kr_dtype = torch.float8_e4m3fn

    # kv/kr cache dtype
    if kv_quant_mode == 0:
        kv_cache_dtype = torch.bfloat16
        kv_cache_dtype_str = 'bfloat16'
        kr_cache_dtype = torch.bfloat16
        kr_cache_dtype_str = 'bfloat16'
    elif kv_quant_mode in [1, 2]:
        if weight_quant_mode == 3:
            kv_cache_dtype = torch.float8_e4m3fn
            kv_cache_dtype_str = 'float8_e4m3fn'
            kr_cache_dtype = torch.float8_e4m3fn
            kr_cache_dtype_str = 'float8_e4m3fn'
        else:
            kv_cache_dtype = torch.int8
            kv_cache_dtype_str = 'int8'
            kr_cache_dtype = torch.int8
            kr_cache_dtype_str = 'int8'
    else:  # kv_quant_mode == 3
        if weight_quant_mode == 3:
            kv_cache_dtype = torch.float8_e4m3fn
            kv_cache_dtype_str = 'float8_e4m3fn'
        else:
            kv_cache_dtype = torch.int8
            kv_cache_dtype_str = 'int8'
        kr_cache_dtype = torch.bfloat16
        kr_cache_dtype_str = 'bfloat16'

    # --- Read t_flag from params ---
    t_flag = params.get('t_flag', True)

    # --- Create input tensors (on CPU) ---
    if t_flag:
        token_x = _create_tensor((T, He), x_dtype, generator)
        rope_sin = torch.randn(T, DR, dtype=torch.bfloat16, generator=generator)
        rope_cos = torch.randn(T, DR, dtype=torch.bfloat16, generator=generator)
    else:
        token_x = _create_tensor((B, S1, He), x_dtype, generator)
        rope_sin = torch.randn(B, S1, DR, dtype=torch.bfloat16, generator=generator)
        rope_cos = torch.randn(B, S1, DR, dtype=torch.bfloat16, generator=generator)
    w_dq = _create_tensor((He, HCQ), w_dq_dtype, generator)
    w_uq_qr = _create_tensor((HCQ, N1 * (D + DR)), w_uq_qr_dtype, generator)
    w_uk = _create_tensor((N1, D, HCKV), w_uk_dtype, generator)
    w_dkv_kr = _create_tensor((He, HCKV + DR), w_dkv_kr_dtype, generator)
    gamma_cq = torch.randn(HCQ, dtype=torch.bfloat16, generator=generator)
    gamma_ckv = torch.randn(HCKV, dtype=torch.bfloat16, generator=generator)

    # --- Cache setup ---
    block_num = math.ceil(T / block_size)
    if block_num == 0:
        block_num = 1

    # Determine kv_cache Dtile (for per-tile repo modes)
    Dtile = HCKV
    if kv_quant_mode == 3 and ckvkr_repo_mode == 1:
        Dtile += DR * 2
    if kv_quant_mode == 3 and quant_scale_repo_mode == 1:
        Dtile += HCKV // tile_size * 4

    if cache_mode in ("PA_BSND", "PA_NZ"):
        kv_cache = _create_tensor((block_num, block_size, N2, Dtile), kv_cache_dtype, generator)
        if ckvkr_repo_mode == 1:
            kr_cache = torch.empty(0)
        else:
            kr_cache = _create_tensor((block_num, block_size, N2, DR), kr_cache_dtype, generator)
        cache_index = torch.arange(T, dtype=torch.int64)
    elif cache_mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
        pages_per_batch = math.ceil(S2 / block_size)
        total_blocks = B * pages_per_batch
        kv_cache = _create_tensor((total_blocks, block_size, N2, Dtile), kv_cache_dtype, generator)
        if ckvkr_repo_mode == 1:
            kr_cache = torch.empty(0)
        else:
            kr_cache = _create_tensor((total_blocks, block_size, N2, DR), kr_cache_dtype, generator)
        cache_index = torch.arange(total_blocks, dtype=torch.int64).reshape(B, pages_per_batch)
    elif cache_mode == "BSND":
        kv_cache = _create_tensor((B, S2, N2, Dtile), kv_cache_dtype, generator)
        if ckvkr_repo_mode == 1:
            kr_cache = torch.empty(0)
        else:
            kr_cache = _create_tensor((B, S2, N2, DR), kr_cache_dtype, generator)
        cache_index = None
    elif cache_mode == "TND":
        kv_cache = _create_tensor((T, N2, Dtile), kv_cache_dtype, generator)
        if ckvkr_repo_mode == 1:
            kr_cache = torch.empty(0)
        else:
            kr_cache = _create_tensor((T, N2, DR), kr_cache_dtype, generator)
        cache_index = None
    else:
        raise ValueError(f"Unsupported cache_mode: {cache_mode}")

    # --- Dequant/quant scale tensors ---
    deq_scale_x = torch.ones(1, dtype=torch.float32)
    deq_scale_w_dq = torch.ones(1, dtype=torch.float32)
    deq_scale_w_uqqr = torch.ones(1, dtype=torch.float32)
    deq_scale_w_dkvkr = torch.ones(1, dtype=torch.float32)
    quant_scale_ckv = torch.ones(1, dtype=torch.float32)
    quant_scale_ckr = torch.ones(1, dtype=torch.float32)
    smo_scale_cq = None
    k_nope_clip_alpha = torch.ones(1, dtype=torch.float32)

    grp_size = 32

    if weight_quant_mode == 2:
        # INT8 full quantization: per-token scale for x, per-channel scale for weights
        deq_scale_x = torch.rand(T, 1, dtype=torch.float32) + 0.01
        deq_scale_w_dq = torch.rand(1, HCQ, dtype=torch.float32) + 0.01
        deq_scale_w_uqqr = torch.rand(1, N1 * (D + DR), dtype=torch.float32) + 0.01
        deq_scale_w_dkvkr = torch.rand(1, HCKV + DR, dtype=torch.float32) + 0.01
    elif weight_quant_mode == 3:
        # MXFP8: block-scaled e8m0 format
        deq_scale_x = torch.randint(100, 150, (T, He // grp_size), dtype=torch.uint8)
        deq_scale_w_dq = torch.randint(100, 150, (HCQ, He // grp_size), dtype=torch.uint8)
        deq_scale_w_uqqr = torch.randint(100, 150, (N1 * (D + DR), HCQ // grp_size), dtype=torch.uint8)
        deq_scale_w_dkvkr = torch.randint(100, 150, (HCKV + DR, He // grp_size), dtype=torch.uint8)

    if weight_quant_mode in [1, 2]:
        # smooth_scale_cq for dynamic_quant
        smo_scale_cq = torch.rand(1, HCQ, dtype=torch.float32) + 0.5

    if kv_quant_mode in [1, 2]:
        quant_scale_ckv = torch.tensor([1.0 / 127.0], dtype=torch.float32)
        quant_scale_ckr = torch.tensor([1.0 / 127.0], dtype=torch.float32)

    if kv_quant_mode == 3:
        k_nope_clip_alpha = torch.tensor([1.0], dtype=torch.float32)

    # --- Flaglist ---
    flaglist = [0] * 25
    # flag[19]: quantize kr_cache separately
    flaglist[19] = 1 if (kv_quant_mode in [1, 2] and ckvkr_repo_mode == 0) else 0
    # flag[20]: smooth_scale_cq present
    flaglist[20] = 1 if weight_quant_mode in [1, 2] else 0
    # flag[21]: deq_scale_q_nope output
    flaglist[21] = 1 if (query_quant_mode == 1 and weight_quant_mode in [2, 3]) else 0
    # flag[22]: query_norm output
    flaglist[22] = 1 if qnorm_flag else 0
    # flag[23]: deq_scale_q_norm output
    flaglist[23] = 1 if (qnorm_flag and weight_quant_mode in [1, 2, 3]) else 0
    # flag[24]: actual_seq_len (needed for PA_BLK modes)
    flaglist[24] = 1 if cache_mode in ("PA_BLK_BSND", "PA_BLK_NZ") else 0
    flaglist_bool = str_to_bool_list(flaglist)

    # --- PA flag ---
    pa_flag = "PA" in cache_mode

    # --- actual_seq_len for PA_BLK modes ---
    if cache_mode in ("PA_BLK_BSND", "PA_BLK_NZ"):
        actual_seq_len = torch.tensor([S1] * B, dtype=torch.int64)
    else:
        actual_seq_len = None

    # --- Output shapes (for enable_quant_output) ---
    enable_quant_output = (query_quant_mode == 1 and weight_quant_mode in [2, 3])
    out_deqq_shape = [T, N1, 1] if enable_quant_output else []

    # --- Build mla_param dict ---
    mla_param = {
        "B": B, "S1": S1, "S2": S2, "D": D, "Dr": DR,
        "N1": N1, "N2": N2, "He": He, "Hcq": HCQ, "Hckv": HCKV,
        "T": T, "block_size": block_size, "cache_mode": cache_mode,
        "qnorm_flag": qnorm_flag,
        "flaglist": flaglist_bool,
        "actual_seq_len": actual_seq_len,
        # Tensor fields
        "x_tensor": token_x, "x_dtype": x_dtype_str,
        "w_dq_tensor": w_dq, "w_dq_dtype": str(w_dq.dtype).replace("torch.", ""),
        "w_uq_qr_tensor": w_uq_qr, "w_uq_qr_dtype": w_uq_qr_dtype_str,
        "w_uk_tensor": w_uk, "w_uk_dtype": w_uk_dtype_str,
        "w_dkv_kr_tensor": w_dkv_kr,
        "gamma_cq_tensor": gamma_cq,
        "gamma_ckv_tensor": gamma_ckv,
        "sin_tensor": rope_sin, "cos_tensor": rope_cos,
        "cache_index_tensor": cache_index,
        "kv_cache_tensor": kv_cache, "kv_cache_dtype": kv_cache_dtype_str,
        "kr_cache_tensor": kr_cache, "kr_cache_dtype": kr_cache_dtype_str,
        # Scalar params
        "epsilon_cq": epsilon_cq, "epsilon_ckv": epsilon_ckv,
        "weight_quant_mode": weight_quant_mode,
        "kv_quant_mode": kv_quant_mode,
        "query_quant_mode": query_quant_mode,
        "ckvkr_repo_mode": ckvkr_repo_mode,
        "quant_scale_repo_mode": quant_scale_repo_mode,
        "tile_size": tile_size,
        "qc_qr_scale": qc_qr_scale,
        "kc_scale": kc_scale,
        # Scale tensors
        "deq_scale_x_tensor": deq_scale_x,
        "deq_scale_w_dq_tensor": deq_scale_w_dq,
        "deq_scale_w_uqqr_tensor": deq_scale_w_uqqr,
        "deq_scale_w_dkvkr_tensor": deq_scale_w_dkvkr,
        "quant_scale_ckv_tensor": quant_scale_ckv,
        "quant_scale_ckr_tensor": quant_scale_ckr,
        "smo_scale_cq_tensor": smo_scale_cq,
        "k_nope_clip_alpha_tensor": k_nope_clip_alpha,
        # Flags
        "pa_flag": pa_flag,
        "t_flag": t_flag,
        "device": "cpu",
        "action_type": "bm_output_gold",
        # Output shape hints
        "enable_quant_output": enable_quant_output,
        "out_deqq_shape": out_deqq_shape,
    }

    # --- Build NPU input dict ---
    npu_inputs = {
        "token_x": token_x,
        "weight_dq": w_dq,
        "weight_uq_qr": w_uq_qr,
        "weight_uk": w_uk,
        "weight_dkv_kr": w_dkv_kr,
        "rmsnorm_gamma_cq": gamma_cq,
        "rmsnorm_gamma_ckv": gamma_ckv,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "kv_cache": copy.deepcopy(kv_cache),
        "kr_cache": copy.deepcopy(kr_cache),
        "cache_index": cache_index,
        "deq_scale_x": deq_scale_x if weight_quant_mode in [2, 3] else None,
        "deq_scale_w_dq": deq_scale_w_dq if weight_quant_mode in [2, 3] else None,
        "deq_scale_w_uqqr": deq_scale_w_uqqr if weight_quant_mode in [1, 2, 3] else None,
        "deq_scale_w_dkvkr": deq_scale_w_dkvkr if weight_quant_mode in [2, 3] else None,
        "quant_scale_ckv": quant_scale_ckv if kv_quant_mode in [1, 2, 3] else None,
        "quant_scale_ckr": quant_scale_ckr if kv_quant_mode in [1, 2] else None,
        "smooth_scales_cq": smo_scale_cq,
        "k_nope_clip_alpha": k_nope_clip_alpha if kv_quant_mode == 3 else None,
        "actual_seq_len": actual_seq_len,
    }

    return mla_param, npu_inputs


def test_mla_prolog_v3(params):
    """Run MLA Prolog V3: CPU golden reference + NPU operator.

    Args:
        params: dict from testcases with test parameters.

    Returns:
        (expect_list, result_list): tuple of CPU golden outputs and NPU outputs.
    """
    mla_param, npu_inputs = build_mla_param(params)

    # --- CPU golden reference ---
    expect = cal_mlaprolog(mla_param)
    # expect = (out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm)

    # --- NPU operator call ---
    torch_npu.npu.set_device(0)

    # Move tensors to NPU
    token_x_npu = npu_inputs["token_x"].npu()
    w_dq_npu = npu_inputs["weight_dq"].npu()
    w_uq_qr_npu = npu_inputs["weight_uq_qr"].npu()
    w_uk_npu = npu_inputs["weight_uk"].npu()
    w_dkv_kr_npu = npu_inputs["weight_dkv_kr"].npu()
    gamma_cq_npu = npu_inputs["rmsnorm_gamma_cq"].npu()
    gamma_ckv_npu = npu_inputs["rmsnorm_gamma_ckv"].npu()
    rope_sin_npu = npu_inputs["rope_sin"].npu()
    rope_cos_npu = npu_inputs["rope_cos"].npu()
    kv_cache_npu = npu_inputs["kv_cache"].npu()
    kr_cache_npu = npu_inputs["kr_cache"].npu()

    # NZ format cast for 2D weight matrices
    w_dq_cast = torch_npu.npu_format_cast(w_dq_npu.contiguous(), 29)
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr_npu.contiguous(), 29)
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr_npu.contiguous(), 29)

    # Optional inputs
    cache_index_npu = npu_inputs["cache_index"].npu() if npu_inputs["cache_index"] is not None else None
    deq_scale_x_npu = npu_inputs["deq_scale_x"].npu() if npu_inputs["deq_scale_x"] is not None else None
    deq_scale_w_dq_npu = npu_inputs["deq_scale_w_dq"].npu() if npu_inputs["deq_scale_w_dq"] is not None else None
    deq_scale_w_uqqr_npu = npu_inputs["deq_scale_w_uqqr"].npu() if npu_inputs["deq_scale_w_uqqr"] is not None else None
    deq_scale_w_dkvkr_npu = npu_inputs["deq_scale_w_dkvkr"].npu() if npu_inputs["deq_scale_w_dkvkr"] is not None else None
    quant_scale_ckv_npu = npu_inputs["quant_scale_ckv"].npu() if npu_inputs["quant_scale_ckv"] is not None else None
    quant_scale_ckr_npu = npu_inputs["quant_scale_ckr"].npu() if npu_inputs["quant_scale_ckr"] is not None else None
    smooth_scales_cq_npu = npu_inputs["smooth_scales_cq"].npu() if npu_inputs["smooth_scales_cq"] is not None else None
    k_nope_clip_alpha_npu = npu_inputs["k_nope_clip_alpha"].npu() if npu_inputs["k_nope_clip_alpha"] is not None else None
    actual_seq_len_npu = npu_inputs["actual_seq_len"].npu() if npu_inputs["actual_seq_len"] is not None else None

    weight_quant_mode = params['weight_quant_mode']
    kv_quant_mode = params['kv_cache_quant_mode']
    query_quant_mode = params['query_quant_mode']
    ckvkr_repo_mode = params['ckvkr_repo_mode']
    quant_scale_repo_mode = params['quant_scale_repo_mode']
    cache_mode = params['cache_mode']
    block_size = params['block_size']
    qnorm_flag = params.get('query_norm_flag', False)
    tile_size = params.get('tile_size', 128)
    qc_qr_scale = params.get('qc_qr_scale', 1.0)
    kc_scale = params.get('kc_scale', 1.0)
    epsilon_cq = params.get('epsilon_cq', 1e-5)
    epsilon_ckv = params.get('epsilon_ckv', 1e-5)

    result = torch_npu.npu_mla_prolog_v3(
        token_x_npu, w_dq_cast, w_uq_qr_cast,
        w_uk_npu, w_dkv_kr_cast,
        gamma_cq_npu, gamma_ckv_npu,
        rope_sin_npu, rope_cos_npu,
        kv_cache_npu, kr_cache_npu,
        cache_index=cache_index_npu,
        dequant_scale_x=deq_scale_x_npu,
        dequant_scale_w_dq=deq_scale_w_dq_npu,
        dequant_scale_w_uq_qr=deq_scale_w_uqqr_npu,
        dequant_scale_w_dkv_kr=deq_scale_w_dkvkr_npu,
        quant_scale_ckv=quant_scale_ckv_npu,
        quant_scale_ckr=quant_scale_ckr_npu,
        smooth_scales_cq=smooth_scales_cq_npu,
        actual_seq_len=actual_seq_len_npu,
        k_nope_clip_alpha=k_nope_clip_alpha_npu,
        rmsnorm_epsilon_cq=epsilon_cq,
        rmsnorm_epsilon_ckv=epsilon_ckv,
        cache_mode=cache_mode,
        query_norm_flag=qnorm_flag,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_quant_mode,
        query_quant_mode=query_quant_mode,
        ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        qc_qr_scale=qc_qr_scale,
        kc_scale=kc_scale,
    )

    torch.npu.synchronize()

    # kv_cache and kr_cache are updated in-place by the NPU operator.
    # The returned tuple only contains: (queryOut, queryRopeOut, [deqScaleQNope], [queryNorm], [deqScaleQNorm]).
    # We insert the in-place-updated caches at positions [2] and [3] to match the CPU golden order:
    # (out1, out2, out3=kv_cache, out4=kr_cache, deq_scale_q_nope, out_qnorm, out_deq_qnorm)
    npu_outputs = list(result) if isinstance(result, (tuple, list)) else [result]
    result_list = npu_outputs[:2] + [kv_cache_npu, kr_cache_npu] + npu_outputs[2:]

    # CPU golden: (out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm)
    # Empty tensors are used for unused optional outputs.
    expect_list = list(expect)

    return expect_list, result_list
