
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import os
import re
import numpy as np
import torch
import time
import math
import random
import copy
import ctypes
import libs.tools as tools
import tensorflow as tf
import libs.training.run_aclnn as training

cfg_fk = tools.ConfigFmk()
COLOR_YELLOW = "\033[33m"  # 黄色
YELLOW_RESET = "\033[0m"  # 重置黄色
COLOR_GREEN = "\033[32m"
GREEN_RESET = "\033[0m"

WEIGHT_QUANT_MODE_FULL_INT8 = 2
WEIGHT_QUANT_MODE_MXFP8_FULL = 3
WEIGHT_QUANT_MODE_FULL_FP8_E4M3 = 4
WEIGHT_QUANT_MODE_FULL_HIF8 = 5

INT8_DTYPE_MAX = 127.0
FP8_E4M3_DTYPE_MAX = 448.0
HIF8_DTYPE_MAX = 32768.0


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
    # 异常检查
    if len(cache.shape) != 4 or len(inputs.shape) != 2 or len(index.shape) != 1:
        print(
            f"[ERROR]scatter_pa_nz got wrong inputs dims: cache-{cache.shape}, inputs-{inputs.shape}, index-{index.shape}")
        return cache
    elif inputs.shape[1] != h:
        print(
            f"[ERROR]scatter_pa_nz got wrong inputs h: cache-{cache.shape}, inputs-{inputs.shape}")
        return cache
    elif index.shape[0] != bxs1:
        print(
            f"[ERROR]scatter_pa_nz got wrong inputs b*s1: input-{inputs.shape}, index-{index.shape}")
        return cache

    # scatter计算
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
                # print(f"[DEBUG]index_num:{index_num} inputs[{input_index},{input_start_index}~{input_end_index}] -> cache[{block_num_index},{block_size_index},{n_index},{h_start_index}~{h_end_index}]")
                cache[block_num_index, block_size_index, n_index, h_start_index: h_end_index] = inputs[input_index,
                                                                                                input_start_index: input_end_index]

    return cache


import math
import numpy as np

def scatter_pa_blk_bsnd(cache: np.ndarray,
                        input_: np.ndarray,
                        index: np.ndarray,
                        seq_len,
                        B: int):
    """
    将 input_ 写入 PA_BLK_BSND 格式的 cache（CPU端参考实现，按页写入）。
    - cache: (num_pa_blocks, blk_size, N, H)
    - input_: (total_tokens, H)  # 按 batch 拼接后的 token 序列
    - index: (sum_pages,)  # 扁平一维，指向 cache 第0维（块 id）
             第 i 个元素给出第 i 个“页”的 pa_blk_id（无需 // blk_size）
    - seq_len: BSND: int；TND: 1D 数组（长度表或前缀和）
    - B: batch size（TND 中会被 seq_len 推导覆盖）
    """
    assert cache.ndim == 4, "cache 必须是 (num_pa_blocks, blk_size, N, H)"
    num_pa_blocks, blk_size, N, H = cache.shape

    # -------- 解析 BSND / TND --------
    if isinstance(seq_len, (int, np.integer)):
        # BSND：每个 batch 的长度相同
        S1 = int(seq_len)
        assert input_.shape[0] == B * S1, f"BSND: input_.shape[0] 必须等于 B*S1={B*S1}"
        pages_per_b = math.ceil(S1 / blk_size)
        # index 为扁平一维，长度应为 B * pages_per_b
        assert index.ndim == 1 and index.shape[0] == B * pages_per_b, \
            f"BSND: index 长度应为 B*ceil(S1/blk)={B*pages_per_b}，而现在是 {index.shape[0]}"
        seq_list  = np.full((B,), S1, dtype=np.int64)
        start_list = np.arange(B, dtype=np.int64) * S1
    else:
        # TND：长度表或前缀和
        seq_arr = np.asarray(seq_len).reshape(-1)
        B = int(seq_arr.shape[0])

        # 兼容两种输入：长度表 或 前缀和（常见形式：prefix = [len0, len0+len1, ...]）
        if B >= 2 and seq_arr[0] >= 0:
            # 按前缀和解析（更鲁棒）
            seq_list = np.empty(B, dtype=np.int64)
            seq_list[0] = int(seq_arr[0])
            if B > 1:
                seq_list[1:] = np.diff(seq_arr).astype(np.int64)
            start_list = np.empty(B, dtype=np.int64)
            start_list[0] = 0
            if B > 1:
                start_list[1:] = seq_arr[:-1].astype(np.int64)
        else:
            # 视为长度表
            seq_list  = seq_arr.astype(np.int64)
            start_list = np.zeros(B, dtype=np.int64)
            if B > 1:
                start_list[1:] = np.cumsum(seq_list[:-1])

        # TND: index 是扁平一维，长度应为 sum_b ceil(seq[b]/blk_size)
        expected_len = int(np.sum((seq_list + blk_size - 1) // blk_size))
        assert index.ndim == 1 and index.shape[0] == expected_len, \
            f"TND: index 长度应为 {expected_len}，而现在是 {index.shape[0]}"

    # -------- 主循环：按 batch -> page 写入 --------
    flat_page_ptr = 0  # index 的页指针（扁平）
    for b in range(B):
        len_b   = int(seq_list[b])
        base_in = int(start_list[b])
        if len_b <= 0:
            continue

        pages_b = (len_b + blk_size - 1) // blk_size  # 上取整
        for p in range(pages_b):
            # 本页的 cache 块 id（已是第0维索引，无需 // blk_size）
            pa_blk_id = int(index[flat_page_ptr + p])
            assert 0 <= pa_blk_id < num_pa_blocks, \
                f"pa_blk_id={pa_blk_id} 越界[0,{num_pa_blocks}), (b={b}, p={p})"

            # 计算 input_ 切片范围（最后一页可能不足 blk_size）
            start = base_in + p * blk_size
            end   = min(start + blk_size, base_in + len_b)
            sz    = end - start
            if sz <= 0:
                continue

            # 写入：(sz, H) -> (sz, 1, H) 广播到 N -> (sz, N, H)
            cache[pa_blk_id, :sz, :, :H] = input_[start:end, :H][:, None, :]

            # 若 sz < blk_size，其余位置保持原值（通常 cache 预先清零/无效）

        flat_page_ptr += pages_b

    # 检查 index 是否正好用完
    assert flat_page_ptr == index.shape[0], \
        f"index 被消费 {flat_page_ptr} 项，但长度为 {index.shape[0]}（不一致）"
    return cache


def scatter_pa_blk_nz(cache: np.ndarray,
                      input_: np.ndarray,
                      index: np.ndarray,
                      seq_len,                    # int 表示 BSND；np.ndarray 表示 TND（长度数组或前缀和）
                      B,
                      data_size: int = 16):
    """
    支持 BSND/TND 的 PA_BLK_NZ scatter 参考实现（CPU 端验证用）。

    Args:
        cache: (num_pa_blocks, blk_size, N, H_pad)
        input_:
            - BSND: (B*S1, H)
            - TND:  (sum(seq_b), H)，seq_b 可不同
        index: (B, ceil(S1/blk_size))，元素为 cachePaOffset (单位：行)
               注：TND 时，index 的第二维通常仍以 S1=max(seq_b) 对齐，较短 batch 多余页不会使用
        seq_len:
            - int:   S1，所有 batch 的序列长度一致（BSND）
            - array: shape [B] 或 [B,1]
                     * 若严格单调非减 → 当作“前缀和”
                     * 否则 → 当作“每个 batch 的长度列表”
        data_size: 列打包粒度（常见 16/32/64）

    Returns:
        cache: 原位写入并返回
    """
    # ------------ 基本检查 ------------
    assert cache.ndim == 4, f"cache shape must be 4D, got {cache.shape}"
    assert input_.ndim == 2, f"input shape must be 2D, got {input_.shape}"
    assert index.ndim == 2, f"index shape must be 2D, got {index.shape}"

    num_pa_blocks, blk_size, N, H_pad = cache.shape
    H = input_.shape[1]

    # ------------ 解析 BSND/TND 的序列长度与起始偏移 ------------
    if isinstance(seq_len, (int, np.integer)):
        # BSND
        S1 = int(seq_len)
        assert input_.shape[0] == B * S1, \
            f"input first dim({input_.shape[0]}) must be B*S1({B*S1}) for BSND"
        # 每个 batch 的长度与起始偏移
        seq_list = np.full((B,), S1, dtype=np.int64)
        start_list = np.arange(B, dtype=np.int64) * S1
        # index 页数检查
        expected_pages = math.ceil(S1 / blk_size)
        assert index.shape[1] == expected_pages, \
            f"index second dim({index.shape[1]}) must be ceil(S1/blk_size)({expected_pages})"
    else:
        # TND：seq_len 为 ndarray，是前缀和数组
        seq_arr = np.asarray(seq_len).reshape(-1)
        B = int(seq_len.shape[0])
        total_T = int(seq_arr[-1])
        # 转换为 per-batch 长度
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

        # 对 index 第二维做“足够大”检查（通常对齐到 max_seq）
        max_s1 = int(np.max(seq_list))
        need_pages = math.ceil(max_s1 / blk_size)
        assert index.shape[1] >= need_pages, \
            f"index second dim({index.shape[1]}) must be >= ceil(max_seq/blk_size)({need_pages}) for TND"

    # ------------ 按 batch / token 写入 ------------
    data_num = math.ceil(H / data_size)   # 有效列块数（最后一块可能不满 data_size）

    for b in range(B):
        len_b = int(seq_list[b])
        inp_start = int(start_list[b])
        # 对每个 token（该 batch 内的相对下标 t）
        for t in range(len_b):
            page_id = t // blk_size
            tok_off_in_page = t - page_id * blk_size  # 避免二次除法
            pa_blk_id = int(index[b, page_id])  # 单位：行

            inp_row = inp_start + t

            # 列块循环（NZ 映射规则）
            for data_idx in range(data_num):
                data_index_in_block = data_idx * blk_size + tok_off_in_page
                block_size_index = data_index_in_block // data_num
                h_start = (data_index_in_block % data_num) * data_size
                h_end = h_start + data_size

                # 处理最后一个分块可能越界 H 的情况（只写有效列）
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
    # 使用广播机制来避免显式的循环
    scaled_values = (x * qscale).round().to(torch.float32)
    s9_res = s9_saturation(scaled_values)
    s8_res_cal = s8_saturation(s9_res)
    return s8_res_cal


def numpy_float8_e4m3fn():
    try:
        # noinspection PyUnresolvedReferences
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


def _get_mode_dtype_max(weight_quant_mode):
    if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_FP8_E4M3:
        return FP8_E4M3_DTYPE_MAX
    if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_HIF8:
        return HIF8_DTYPE_MAX
    return INT8_DTYPE_MAX


def _dynamic_quant_clip_range(dtype_max, quant_dtype):
    if quant_dtype == torch.int8:
        return -128.0, 127.0
    return -float(dtype_max), float(dtype_max)


def _clamp_to_dtype_range_float(inputs, dtype_max):
    clip_min, clip_max = _dynamic_quant_clip_range(dtype_max, torch.float32)
    return torch.clamp(inputs.to(torch.float32), min=clip_min, max=clip_max)


def dynamic_quant(inputs, smooth_scale, dtype_max=INT8_DTYPE_MAX, quant_dtype=torch.int32):
    T = inputs.size(0)
    H = inputs.size(1)
    y = torch.zeros(T, H).to(torch.float32)
    scale = torch.zeros(T).to(torch.float32)
    inputs = inputs.reshape(T, H).to(torch.float32)
    if smooth_scale!=None:
        if len(smooth_scale.shape) != 2 or smooth_scale.shape[1] != H:
            print(f"[ERROR]dynamic_quant got wrong input smooth_scale:{tuple(smooth_scale.shape)}, expected:({H})")
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        for bs_index in range(T):
            abs_bs_tensor = torch.abs(inputs[bs_index, :] * smooth_scale[0, :])
            scale_bs = abs_bs_tensor.max() / float(dtype_max)
            scale[bs_index] = scale_bs
            y[bs_index:] = torch.round(inputs[bs_index:] * smooth_scale[0, :] / scale_bs)
    else:
        for bs_index in range(T):
            abs_bs_tensor = torch.abs(inputs[bs_index, :])
            scale_bs = abs_bs_tensor.max() / float(dtype_max)
            scale[bs_index] = scale_bs
            y[bs_index:] = torch.round(inputs[bs_index:]/ scale_bs)
    if quant_dtype == torch.int32:
        y = y.to(torch.int32)
    else:
        clip_min, clip_max = _dynamic_quant_clip_range(dtype_max, quant_dtype)
        y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
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
    print(f"_mx_reshape_to_blocks: {fp_array.shape}")
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
    #### 
    res = res + FP32_EXPONENT_BIAS
    res[fp_abs_max == 0] = -float("inf")
    return res

def _mx_round_mantissa(fp_array: np.ndarray, round_mode: str):
    """
    For example:
    fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, -0.0, 0.0, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
    - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
    - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
    - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0,   0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
    - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    """
    if round_mode in ("rint", "even"):  # tie to even(c language rint)
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":  # round to minus infinity(c language floor)
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":   # round to positive infinity(c language ceil)
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":  # round to zero(c language truncation)
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
    # The minimum representable exponent
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion of the number
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    # Undo scaling
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    # Set values > max_norm to Inf if desired, else clamp them
    max_norm = get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret

def pad_to_even(tensor: np.ndarray, axis: int) -> np.ndarray:
    """
    在指定的 axis 上将 tensor 对齐到偶数长度（即按 2 对齐），不足补 0。

    参数:
        tensor (np.ndarray): 输入数组
        axis (int): 要对齐的轴（从 0 开始）

    返回:
        np.ndarray: 对齐后的数组
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")

    shape = tensor.shape
    length = shape[axis]

    # 如果已经是偶数，直接返回原数组
    if length % 2 == 0:
        return tensor

    # 构造 pad_width：仅对目标 axis 补一个 0
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)  # 在 axis 维度末尾补一个 0

    padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
    return padded_tensor

def _mx_undo_reshape_to_blocks(fp_array: np.ndarray, axis: int,
                               orig_shape: tuple, padded_shape: tuple):
    # Undo tile reshaping
    fp_array = fp_array.reshape(padded_shape)
    # Undo padding
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    # Remove extra dimension
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array

def interleave(tensor: np.ndarray, axis: int, n_group: int = 2) -> np.ndarray:
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
    # 获取目标轴的长度
    length = tensor.shape[axis]
    # 检查是否可整除
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")

    group_length = length // n_group  # 每组长度
    shape = list(tensor.shape)

    # 重塑形状：在目标轴后插入组维度
    new_shape = (
        shape[:axis] +
        [group_length,2] +
        shape[axis+1:])
    reshaped = tensor.reshape(new_shape)

    # 构建转置顺序：交换组维度和组内维度
    transpose_order = (
        list(range(0, axis+1)) +  # 目标轴之前的维度
        list(range(axis + 2, len(new_shape))) +
        [axis+1,])  # 后续维度

    # 执行转置
    transposed = reshaped.transpose(transpose_order)

    return transposed

from ml_dtypes import float8_e4m3fn
from en_dtypes import float8_e8m0
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
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    #### new
    share_exp[(share_exp - 127) > scale_emax] = float("NaN")
    share_exp[(share_exp - 127) < -scale_emax] = -scale_emax
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = eval(f"{mx_ele_dtype}")
    # share_exp is always float32
    #### scale_array = 2 ** share_exp
    scale_array = share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    # Cube only supports even scales. need to pad zero.
    scale_array_pad = pad_to_even(scale_array, axis=axis)
    print(f"scale_array_pad: {scale_array_pad.shape}")
    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    # when axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    print(f"result_shape: {np.array(result_shape).shape}")
    print(f"scale_array_pad1: {scale_array_pad.shape}")
    scale_array_pad = scale_array_pad.reshape(result_shape)
    print(f"scale_array_pad2: {scale_array_pad.shape}, {scale_array_pad.dtype}")
    scale_array = scale_array_pad.astype("uint8", copy=False)
    print(f"{scale_array}")
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
        output_data = input_scaled.astype("float8_e4m3fn", copy=False)
    else:
        output_data = input_scaled
    print(f"scale shape: {scale.shape}")
    print(f"output shape: {output_data.shape}")
    print(f"scale: {scale}")
    print(f"output: {output_data}")
    return scale, output_data


def dynamic_quant_ckv_with_amax(inputs: torch.Tensor, amax: torch.Tensor, smooth_scale: torch.Tensor = None):
    """
        三维动态量化 FP32 -> INT8（每行 scale 由外部提供 amax）
        inputs: [T, N, H] 张量
        amax: [T, N, 1] 每行 H 的最大绝对值
        smooth_scale: None 或 [1, N, H] 张量
        返回:
            y: [T, N, H] 量化结果
            scale: [T, N] 每行 scale
    """
    T, N, H = inputs.shape
    inputs = inputs.to(torch.float32)
    amax = amax.to(torch.float32).clamp(min=1e-8)  # 避免除零

    if smooth_scale is not None:
        if smooth_scale.ndim != 3 or smooth_scale.shape[1:] != (N, H):
            print(f"[ERROR] dynamic_quant got wrong input smooth_scale:{tuple(smooth_scale.shape)}, expected:(1, {N}, {H})")
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        scaled_inputs = inputs * smooth_scale  # 广播 [T,N,H] * [1,N,H]
    else:
        scaled_inputs = inputs

    scale = amax / 127.0

    # 量化
    y = torch.round(scaled_inputs / scale)

    return y, scale


# quant_type: ['per_token', 'per_token_head'], input shape: [T,N1,Hckv]
def dynamic_quant_without_smooth_scale(inputs, out_deqq_shape_shape, dtype_max=INT8_DTYPE_MAX, quant_dtype=torch.int8):
    T = inputs.size(0)
    N = inputs.size(1)
    H = inputs.size(2)
    quant_loops = inputs.size(0)
    eles_with_one_scale = inputs.size(1) * inputs.size(2)
    if len(out_deqq_shape_shape) == 3:  # [BS, N, 1], per_token per_head
        quant_loops = inputs.size(0) * inputs.size(1)
        eles_with_one_scale = inputs.size(2)

    y = torch.zeros(quant_loops, eles_with_one_scale).to(torch.float32)
    scale = torch.zeros(quant_loops).to(torch.float32)
    inputs = inputs.reshape(quant_loops, eles_with_one_scale).to(torch.float32)
    max_values, _ = torch.max(torch.abs(inputs), dim=-1, keepdim=True)
    scale = max_values / float(dtype_max)
    y = torch.round(inputs/scale)
    clip_min, clip_max = _dynamic_quant_clip_range(dtype_max, quant_dtype)
    y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
    if len(out_deqq_shape_shape) == 2:  # [BS, 1], per_token
        print(f"[INFO]dynamic_quant_without_smooth_scale in per_token mode")
        return y.reshape(T, N, H), scale.reshape(quant_loops, 1).to(torch.float64)
    else:
        print(f"[INFO]dynamic_quant_without_smooth_scale in per_head mode")
        return y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64)


# len(deq_scale_q_nope) == 3 则是per_token_head, len(deq_scale_q_nope) == 2则是per_token
def dequant(inputs, deq_scale_q_nope, quant_scale_ckv):
    org_shape = inputs.size()
    quant_loops = inputs.size(0)
    eles_with_one_scale = inputs.size(1) * inputs.size(2)
    if len(deq_scale_q_nope.size()) == 3:  # per_token_head
        quant_loops = inputs.size(0) * inputs.size(1)
        eles_with_one_scale = inputs.size(2)
    inputs = inputs.reshape(quant_loops, eles_with_one_scale)
    deq_scale_q_nope = deq_scale_q_nope.reshape(quant_loops, 1)
    for quant_idx in range(quant_loops):
        inputs[quant_idx, :] = inputs[quant_idx, :] * quant_scale_ckv / (deq_scale_q_nope[quant_idx, 0])
    return inputs.reshape(org_shape)


def trans_torch_fp8_e8m0_to_bf16(array):
    from ml_dtypes import bfloat16
    new_array = torch.zeros_like(array).to(torch.bfloat16)
    for i in range(array.size(0)):  # 遍历行
        for j in range(array.size(1)):  # 遍历列
            new_value = ieee_754_conversion(0, int(array[i][j]), 0)
            new_array[i][j] = new_value
    return new_array


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    """ Convert binary data into the floating point value """
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))
    return sign_mult * (2 ** exponent) * mant_mult


def convert_dict_values_to_torch(input_dict):
    # 创建一个新的字典来存储转换后的结果
    output_dict = {}
    # 遍历输入字典中的每个键值对
    for key, value in input_dict.items():
        # 检查值是否是NumPy数组
        if isinstance(value, np.ndarray):
            # 如果是，则将其转换为torch.Tensor并添加到输出字典中
            if value.dtype == "float8_e4m3fn":
                print(f"e4m3: {key}")
                output_dict[key] = torch.tensor(value.astype(np.float32)).to(torch.float8_e4m3fn)
            elif value.dtype == "float8_e8m0":
                print(f"e8m0: {key}")
                output_dict[key] = torch.tensor(value.astype(np.int8))
            else:
                print(f"else: {key}")
                output_dict[key] = torch.from_numpy(value)
        else:
            # 如果不是，则直接复制该值
            output_dict[key] = value
    return output_dict

def get_current_seq_len(token_idx: int, actual_seq_length: np.ndarray) -> int:
    """
    给定 T 维下的 token 索引 (token_idx)，返回它所属 batch 的 sequence length。

    Args:
        token_idx (int): 当前 token 在 TND layout 中的 index (0-based)
        actual_seq_length (np.ndarray): shape [B,1]，每个 batch 的序列长度前缀和

    Returns:
        int: 当前 token 所属 batch 的序列长度
    """
    # 转成一维 list，方便用 bisect
    prefix_sum = actual_seq_length.flatten().tolist()

    # 找到 token 属于哪个 batch
    batch_id = bisect.bisect_right(prefix_sum, token_idx)

    # 当前 batch 的 seq_len = prefix_sum[batch_id] - prefix_sum[batch_id-1]
    if batch_id == 0:
        seq_len = prefix_sum[0]
    else:
        seq_len = prefix_sum[batch_id] - prefix_sum[batch_id - 1]

    return seq_len

def get_param(torch_tensor_list, params):
    # 将有效参数读取到字典中
    res = {}
    res['normal_flag'] = True
    res["action_type"] = params['action_type']
    # device
    cfg_fk = tools.ConfigFmk()
    res["device"] = cfg_fk.device_type
    # flag_list
    res["flaglist"] = str_to_bool_list(params['flaglist'])
    # core params
    res["B"] = params['b']
    res["N1"] = params['n1']
    res["N2"] = params['n2']
    res["D"] = params['d']
    res["Dr"] = params['dr']
    res["S1"] = params['s1']
    res["S2"] = params['s2']
    res["He"] = params['he']
    res["Hcq"] = params['hcq']
    res["Hckv"] = params['hckv']
    res["T"] = params['t']
    res["block_size"] = params['block_size']
    res["cache_mode"] = params['cache_mode']
    res["qnorm_flag"] = params['qnorm_flag']

    # actual_seq_len
    res["actual_seq_len"] = None
    if res['flaglist'][24]:
        res["actual_seq_len"] = params["actual_seq_value"]

    # token_x
    res["x_shape"] = params['shape_input'][0]
    res["x_dtype"] = trans_input_dtype(params['dtype_input'][0])
    res["x_tensor"] = torch_tensor_list[0]

    # w_dq
    res["w_dq_shape"] = params['shape_input'][1]
    res["w_dq_dtype"] = trans_input_dtype(params['dtype_input'][1])
    res["w_dq_tensor"] = torch_tensor_list[1]

    # w_uq_qr
    res["w_uq_qr_shape"] = params['shape_input'][2]
    res["w_uq_qr_dtype"] = trans_input_dtype(params['dtype_input'][2])
    res["w_uq_qr_tensor"] = torch_tensor_list[2]

    # w_uk
    res["w_uk_shape"] = params['shape_input'][3]
    res["w_uk_dtype"] = trans_input_dtype(params['dtype_input'][3])
    res["w_uk_tensor"] = torch_tensor_list[3]

    # w_uk
    res["w_dkv_kr_shape"] = params['shape_input'][4]
    res["w_dkv_kr_dtype"] = trans_input_dtype(params['dtype_input'][4])
    res["w_dkv_kr_tensor"] = torch_tensor_list[4]

    # gamma_cq
    res["gamma_cq_shape"] = params['shape_input'][5]
    res["gamma_cq_dtype"] = trans_input_dtype(params['dtype_input'][5])
    res["gamma_cq_tensor"] = torch_tensor_list[5]

    # gamma_ckv
    res["gamma_ckv_shape"] = params['shape_input'][6]
    res["gamma_ckv_dtype"] = trans_input_dtype(params['dtype_input'][6])
    res["gamma_ckv_tensor"] = torch_tensor_list[6]

    # sin
    res["sin_shape"] = params['shape_input'][7]
    res["sin_dtype"] = trans_input_dtype(params['dtype_input'][7])
    res["sin_tensor"] = torch_tensor_list[7]

    # cos
    res["cos_shape"] = params['shape_input'][8]
    res["cos_dtype"] = trans_input_dtype(params['dtype_input'][8])
    res["cos_tensor"] = torch_tensor_list[8]

    # cache_index
    res["cache_index_shape"] = params['shape_input'][9]
    print("cache_index_shape ori: ", res["cache_index_shape"])
    res["cache_index_dtype"] = trans_input_dtype(params['dtype_input'][9])
    res["cache_index_tensor"] = torch_tensor_list[9]

    # kv_cache
    res["kv_cache_shape"] = params['shape_input'][10]
    res["kv_cache_dtype"] = trans_input_dtype(params['dtype_input'][10])
    res["kv_cache_tensor"] = torch_tensor_list[10]

    # kr_cache
    res["kr_cache_shape"] = params['shape_input'][11]
    res["kr_cache_dtype"] = trans_input_dtype(params['dtype_input'][11])
    res["kr_cache_tensor"] = torch_tensor_list[11]

    # epsilon_cq
    res["epsilon_cq"] = params['epsilon_cq']

    # epsilon_ckv
    res["epsilon_ckv"] = params['epsilon_ckv']

    # block_size
    res["block_size"] = params['block_size']

    # weight_quant_mode
    res["weight_quant_mode"] = params['weight_quant_mode']
    # kv_quant_mode
    res["kv_quant_mode"] = params['kv_quant_mode']
    # query_quant_mode
    res["query_quant_mode"] = params['query_quant_mode']
    # ckvkr_repo_mode
    res["ckvkr_repo_mode"] = params['ckvkr_repo_mode']
    # quant_scale_repo_mode
    res["quant_scale_repo_mode"] = params['quant_scale_repo_mode']
    # tile_size
    res["tile_size"] = params['tile_size']
    # k_nope_clip_alpha
    # res["k_nope_clip_alpha"] = params['k_nope_clip_alpha']
    # qc_qr_scale
    res["qc_qr_scale"] = params['qc_qr_scale']
    # kc_scale
    res["kc_scale"] = params['kc_scale']

    # out_q
    res["out_q_shape"] = params['shape_output'][0]
    res["out_q_dtype"] = trans_input_dtype(params['dtype_output'][0])
    print("testest==========================")
    print(params['dtype_output'][0])
    print(res["out_q_dtype"])

    # out_qrope
    res["out_qrope_shape"] = params['shape_output'][1]
    res["out_qrope_dtype"] = trans_input_dtype(params['dtype_output'][1])

    # out_kvcache
    res["out_kvcache_shape"] = params['shape_output'][2]
    res["out_kvcache_dtype"] = trans_input_dtype(params['dtype_output'][2])

    # out_krcache
    res["out_krcache_shape"] = params['shape_output'][3]
    res["out_krcache_dtype"] = trans_input_dtype(params['dtype_output'][3])

    # out_deqq_shape
    output_opt_idx = 4
    res["enable_quant_output"] = False
    if res['flaglist'][21]:
        res["enable_quant_output"] = True
        res["out_deqq_shape"] = params['shape_output'][output_opt_idx]
        res["out_deqq_dtype"] = trans_input_dtype(params['dtype_output'][output_opt_idx])
        output_opt_idx += 1

    if res['flaglist'][22]:
        res["out_qnorm_shape"] = params['shape_output'][output_opt_idx]
        res["out_qnorm_dtype"] = trans_input_dtype(params['dtype_output'][output_opt_idx])
        output_opt_idx += 1

    if res['flaglist'][23]:
        res["out_deq_qnorm_shape"] = params['shape_output'][output_opt_idx]
        res["out_deq_qnorm_shape"] = trans_input_dtype(params['dtype_output'][output_opt_idx])
        output_opt_idx += 1


    # deq_scale_x
    res["deq_scale_x_shape"] = params['shape_input'][12]
    res["deq_scale_x_dtype"] = trans_input_dtype(params['dtype_input'][12])
    res["deq_scale_x_tensor"] = torch_tensor_list[12]

    # deq_scale_w_dq
    res["deq_scale_w_dq_shape"] = params['shape_input'][13]
    res["deq_scale_w_dq_dtype"] = trans_input_dtype(params['dtype_input'][13])
    res["deq_scale_w_dq_tensor"] = torch_tensor_list[13]

    # deq_scale_w_uqqr
    res["deq_scale_w_uqqr_shape"] = params['shape_input'][14]
    res["deq_scale_w_uqqr_dtype"] = trans_input_dtype(params['dtype_input'][14])
    res["deq_scale_w_uqqr_tensor"] = torch_tensor_list[14]

    # deq_scale_w_dkvkr
    res["deq_scale_w_dkvkr_shape"] = params['shape_input'][15]
    res["deq_scale_w_dkvkr_dtype"] = trans_input_dtype(params['dtype_input'][15])
    res["deq_scale_w_dkvkr_tensor"] = torch_tensor_list[15]

    # quant_scale_ckv
    res["quant_scale_ckv_shape"] = params['shape_input'][16]
    res["quant_scale_ckv_dtype"] = trans_input_dtype(params['dtype_input'][16])
    res["quant_scale_ckv_tensor"] = torch_tensor_list[16]
    print('quant_scale_ckv_tensor', torch_tensor_list[16].shape, res["quant_scale_ckv_shape"])

    # quant_scale_ckr
    res["quant_scale_ckr_shape"] = params['shape_input'][17]
    res["quant_scale_ckr_dtype"] = trans_input_dtype(params['dtype_input'][17])
    res["quant_scale_ckr_tensor"] = torch_tensor_list[17]

    # smooth_scale_cq
    res["smo_scale_cq_tensor"] = None
    if res['flaglist'][20]:
        res["smo_scale_cq_shape"] = params['shape_input'][18]
        res["smo_scale_cq_dtype"] = trans_input_dtype(params['dtype_input'][18])
        res["smo_scale_cq_tensor"] = torch_tensor_list[18]

    res["k_nope_clip_alpha_shape"] = params['shape_input'][20]
    res["k_nope_clip_alpha_dtype"] = trans_input_dtype(params['dtype_input'][20])
    res["k_nope_clip_alpha_tensor"] = torch_tensor_list[20]

    # flag 判断
    res["pa_flag"] = False
    position = res["cache_mode"].find("PA")
    if position != -1:
        res["pa_flag"] = True

    res["t_flag"] = False
    if res["x_tensor"].ndim == 2:
        res["t_flag"] = True

    return res

def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

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
    print("==========index_table: ", index_table.size())
    x_dtype = mla_param["x_dtype"]
    uqqr_dtype = mla_param["w_uq_qr_dtype"]
    uk_dtype = mla_param["w_uk_dtype"]
    kv_dtype = mla_param["kv_cache_dtype"]
    kr_dtype = mla_param["kr_cache_dtype"]

    token_x = mla_param["x_tensor"]
    cos = mla_param["cos_tensor"]
    sin = mla_param["sin_tensor"]

    qnorm_flag = mla_param["qnorm_flag"]

    deq_scale_q_nope = None
    deq_scale_qcqr = None
    out_qnorm = None
    out_deq_qnorm = None
    mode45_full_quant = mla_param["weight_quant_mode"] in (
        WEIGHT_QUANT_MODE_FULL_FP8_E4M3, WEIGHT_QUANT_MODE_FULL_HIF8)
    mode45_dtype_max = _get_mode_dtype_max(mla_param["weight_quant_mode"])
    enable_quant_output = True if mla_param["query_quant_mode"] == 1 and (
        mla_param["weight_quant_mode"] in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_MXFP8_FULL,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8,
        )) else False
    # enable_quant_output =  mla_param["enable_quant_output"]
    quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
    actual_seq_lengths = mla_param["actual_seq_len"]
    print("*********actual_seq_value: ", actual_seq_lengths)
    if enable_quant_output:
        out_deqq_shape_shape = mla_param['out_deqq_shape']

    # BS合轴
    if not mla_param["t_flag"]:
        print("[INFO]非TH场景，相关参数进行BS合轴")
        T = B * S1
        token_x = token_x.reshape(T, He)
        cos = cos.reshape(T, Dr)
        sin = sin.reshape(T, Dr)
        seq_len = S1
    else:
        seq_len = actual_seq_lengths

    print("[INFO]========================================")
    print("[INFO]>>>>>>>>  Start to calculate  >>>>>>>>>>")
    print("[INFO]========================================")
    # -------------------------------------------------------------------
    # matmul1 : token_x(B*S1,He) * w_dq (He,Hcq) -> matmul1_res(B*S1,Hcq)
    # -------------------------------------------------------------------
    w_dq = mla_param["w_dq_tensor"]
    matmul1_dtype = torch.float32
    # matmul1预处理
    print(f"[TEST]====================token: {token_x}")
    token_x_new = token_x
    if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_INT8 and x_dtype == 'int8':
        token_x_new = token_x_new.to(torch.int32)
        token_x = token_x.to(torch.int32)
        w_dq = w_dq.to(torch.int32)
        matmul1_dtype = torch.int32
        if mla_param["device"] == "gpu":
            token_x = token_x.to('cpu')
            w_dq = w_dq.to('cpu')
    elif mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL and x_dtype == 'float8_e4m3fn':
        # token_x类型转换
        token_x_new = token_x_new.to(torch.bfloat16)
        token_x = token_x.to(torch.bfloat16)
        # deq_scale_x类型转换
        deq_scale_x = mla_param["deq_scale_x_tensor"]
        print(f"====scale x shape is: {deq_scale_x.shape}, B is {B}, S1 is {S1} he is {He}")
        deq_scale_x = deq_scale_x.reshape(B*S1, He//32)
        deq_scale_x = trans_torch_fp8_e8m0_to_bf16(deq_scale_x)
        xs0 = deq_scale_x.shape[0]
        xs1 = deq_scale_x.shape[1]
        grp_size = 32
        for xs0_idx in range(xs0):
            for xs1_idx in range(xs1):
                # copy from
                token_cur = token_x_new[xs0_idx:xs0_idx + 1, xs1_idx * grp_size:(xs1_idx + 1) * grp_size]
                # broadcast
                scale_x = deq_scale_x[xs0_idx:xs0_idx + 1, xs1_idx:(xs1_idx + 1)]
                scale_x = torch.full((1, grp_size), scale_x.item())
                # mul
                token_cur = token_cur * scale_x
                # copy to
                token_x_new[xs0_idx:xs0_idx + 1, xs1_idx * grp_size:(xs1_idx + 1) * grp_size] = token_cur
        # w_dq反量化
        w_dq = mla_param["w_dq_tensor"]
        w_dq = w_dq.to(torch.bfloat16)
        # deq_scale_w_dq 类型转换
        deq_scale_w_dq =  mla_param["deq_scale_w_dq_tensor"]
        deq_scale_w_dq = trans_torch_fp8_e8m0_to_bf16(deq_scale_w_dq)
        dqs0 = deq_scale_w_dq.shape[0]
        dqs1 = deq_scale_w_dq.shape[1]
        for dqs0_idx in range(dqs0):
            for dqs1_idx in range(dqs1):
                scale_w_dq = deq_scale_w_dq[dqs0_idx:dqs0_idx + 1, dqs1_idx:(dqs1_idx + 1)]
                w_dq[dqs1_idx * grp_size:(dqs1_idx + 1) * grp_size, dqs0_idx:dqs0_idx + 1] *= scale_w_dq
        # 转cpu
        if mla_param["device"] == "gpu":
            token_x_new = token_x_new.to('cpu')
            token_x = token_x.to('cpu')
            w_dq = w_dq.to('cpu')
    # matmul1计算
    x_shape = "(T,He)" if mla_param["t_flag"] else "(B*S1,He)"
    print(
        f"[INFO]matmul1 start. token_x{x_shape}:{tuple(token_x.shape)}|{token_x.dtype} w_dq(He,Hcq):{tuple(w_dq.shape)}|{w_dq.dtype} matmul1_dtype:{matmul1_dtype}")
    token_x_new = token_x_new.to(torch.float32)
    w_dq = w_dq.to(torch.float32)
    if mode45_full_quant:
        token_x_new = _clamp_to_dtype_range_float(token_x_new, mode45_dtype_max)
        w_dq = _clamp_to_dtype_range_float(w_dq, mode45_dtype_max)
    matmul1_res = torch.matmul(token_x_new, w_dq).to(matmul1_dtype)
    # matmul1后处理
    if mla_param["weight_quant_mode"] in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8): # x_dtype == 'int8':
        if mla_param["device"] == "gpu":
            matmul1_res = matmul1_res.to('cuda')
        deq_scale_x = mla_param["deq_scale_x_tensor"]
        deq_scale_w_dq = mla_param["deq_scale_w_dq_tensor"]
        matmul1_res = matmul1_res.to(torch.float32)
        for t_index in range(T):
            matmul1_res[t_index, :] = matmul1_res[t_index, :] * deq_scale_x[t_index, 0]
        for h_index in range(Hcq):
            matmul1_res[:, h_index] = matmul1_res[:, h_index] * deq_scale_w_dq[0, h_index]
        print(f"[INFO]deq1 end. matmul1_res dtype trans to {matmul1_res.dtype}")
    elif x_dtype == 'bfloat16':
        print(f"[INFO]cast matmul1_res->bfloat16->float32")
        matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)
    matmul1_res_shape = "(T,Hcq)" if mla_param["t_flag"] else "(B*S1,Hcq)"
    print(f"[INFO]matmul1 end. matmul1_res{matmul1_res_shape}:{tuple(matmul1_res.shape)}|{matmul1_res.dtype}")

    # ----------------------------------------------------------------------
    # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
    # ----------------------------------------------------------------------
    ep1 = float(mla_param["epsilon_cq"])
    gamma1 = mla_param["gamma_cq_tensor"]
    print(
        f"[INFO]rmsnorm1 start. matmul1_res{matmul1_res_shape}:{tuple(matmul1_res.shape)}|{matmul1_res.dtype} gamma_cq(Hcq):{tuple(gamma1.shape)}|{gamma1.dtype}")
    norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
    norm1_res *= gamma1
    qc_qr_scale = float(mla_param["qc_qr_scale"])
    norm1_res *= qc_qr_scale
    print(f"[INFO]rmsnorm1 end. norm1_res{matmul1_res_shape}:{tuple(norm1_res.shape)}|{norm1_res.dtype}")

    # ----------------------------------------------------------------------------------
    # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
    # ----------------------------------------------------------------------------------
    w_uq_qr = mla_param["w_uq_qr_tensor"]
    # matmul2预处理
    matmul2_dtype = torch.float32
    if mla_param["weight_quant_mode"] == 1 or mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_INT8: # uqqr_dtype == 'int8':
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
        if mla_param["device"] == "gpu":
            w_uq_qr = w_uq_qr.to('cpu')
            deq_scale_qcqr = deq_scale_qcqr.to('cuda')
        print(f"[INFO]dynamic_quant end. norm1_res dtype trans to {norm1_res.dtype}")
    elif mode45_full_quant:
        smo_scale_cq = mla_param["smo_scale_cq_tensor"]
        norm1_res, deq_scale_qcqr = dynamic_quant(
            norm1_res,
            smo_scale_cq,
            dtype_max=mode45_dtype_max,
            quant_dtype=torch.float32,
        )
        if qnorm_flag:
            out_qnorm = norm1_res
            out_deq_qnorm = deq_scale_qcqr
        print(f"[INFO]dynamic_quant end. norm1_res dtype trans to {norm1_res.dtype}")
    elif mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL:
		# w_uq_qr类型转换
        w_uq_qr = w_uq_qr.to(torch.bfloat16)
        # deq_scale_uqqr类型转换
        deq_scale_uqqr = mla_param["deq_scale_w_uqqr_tensor"]
        deq_scale_uqqr = trans_torch_fp8_e8m0_to_bf16(deq_scale_uqqr)
        uqqrs0 = deq_scale_uqqr.shape[0]
        uqqrs1 = deq_scale_uqqr.shape[1]
        grp_size = 32
        for uqqrs0_idx in range(uqqrs0):
            for uqqrs1_idx in range(uqqrs1):
                scale_uqqr = deq_scale_uqqr[uqqrs0_idx:uqqrs0_idx + 1, uqqrs1_idx:(uqqrs1_idx + 1)]
                w_uq_qr[uqqrs1_idx * grp_size:(uqqrs1_idx + 1) * grp_size, uqqrs0_idx:uqqrs0_idx + 1] *= scale_uqqr
        # deq_scale_qcqr 为 int 类型，norm1_res 为 e4m3 类型
        norm1_res = norm1_res.to(torch.bfloat16)
        deq_scale_qcqr_np, norm1_res_np = dynamic_mx_quant_cq(norm1_res.float().numpy(), "float8_e4m3fn")
        norm1_res = torch.tensor(norm1_res_np.astype(np.float32)).to(torch.float8_e4m3fn)
        deq_scale_qcqr = torch.from_numpy(deq_scale_qcqr_np)
        deq_scale_qcqr = deq_scale_qcqr.reshape(deq_scale_qcqr.shape[0], deq_scale_qcqr.shape[1] * deq_scale_qcqr.shape[2])
        # norm1_res类型转换
        norm1_res = norm1_res.to(torch.bfloat16)
        # deq_scale_qcqr类型转换
        deq_scale_qcqr = trans_torch_fp8_e8m0_to_bf16(deq_scale_qcqr)
        qcqrs0 = deq_scale_qcqr.shape[0]
        qcqrs1 = deq_scale_qcqr.shape[1]
        grp_size = 32
        for qcqrs0_idx in range(qcqrs0):
            for qcqrs1_idx in range(qcqrs1):
                # copy from
                normal_cur = norm1_res[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx * grp_size:(qcqrs1_idx + 1) * grp_size]
                # broadcast
                scale_qcqr = deq_scale_qcqr[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx:(qcqrs1_idx + 1)]
                scale_qcqr = torch.full((1, grp_size), scale_qcqr.item())
                # mul
                normal_cur = normal_cur * scale_qcqr
                # copy to
                norm1_res[qcqrs0_idx:qcqrs0_idx + 1, qcqrs1_idx * grp_size:(qcqrs1_idx + 1) * grp_size] = normal_cur
    elif uqqr_dtype == 'bfloat16':
        print(f"[INFO]cast norm1_res->bfloat16->float32")
        norm1_res = norm1_res.to(torch.bfloat16).to(torch.float32)
        if qnorm_flag:
            out_qnorm = norm1_res

    # matmul2计算
    print(
        f"[INFO]matmul2 start. norm1_res{matmul1_res_shape}:{tuple(norm1_res.shape)}|{norm1_res.dtype} w_uq_qr(Hcq,N*(D+Dr)):{tuple(w_uq_qr.shape)}|{w_uq_qr.dtype} matmul2_dtype:{matmul2_dtype}")
    norm1_res = norm1_res.to(torch.float32)
    w_uq_qr = w_uq_qr.to(torch.float32)
    if mode45_full_quant:
        norm1_res = _clamp_to_dtype_range_float(norm1_res, mode45_dtype_max)
        w_uq_qr = _clamp_to_dtype_range_float(w_uq_qr, mode45_dtype_max)
    matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(matmul2_dtype)
    # matmul2后处理
    if mla_param["weight_quant_mode"] in (
            1,
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8): # uqqr_dtype == 'int8':
        if mla_param["device"] == "gpu":
            matmul2_res = matmul2_res.to('cuda')
        deq_scale_uqqr = mla_param["deq_scale_w_uqqr_tensor"]
        matmul2_res = matmul2_res.to(torch.float32)
        for t_index in range(T):
            matmul2_res[t_index, :] = matmul2_res[t_index, :] * deq_scale_qcqr[t_index]
        for nddr_index in range(matmul2_res.shape[1]):
            matmul2_res[:, nddr_index] = matmul2_res[:, nddr_index] * deq_scale_uqqr[0, nddr_index]
        print(f"[INFO]deq2 end. matmul2_res dtype trans to {matmul2_res.dtype}")
    elif uqqr_dtype == 'bfloat16':
        print(f"[INFO]cast matmul2_res->bfloat16->float32")
        matmul2_res = matmul2_res.to(torch.bfloat16).to(torch.float32)
    matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
    print(f"[INFO]matmul2 end. matmul2_res(B*S1,N,D+Dr):{tuple(matmul2_res.shape)}|{matmul2_res.dtype}")

    # -------------------------------------------------------------------------------------
    # splitD1 : matmul2_res(B*S1,N,D+Dr) -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
    # -------------------------------------------------------------------------------------
    splitd1_res1 = matmul2_res[:, :, :D]  # 取前 D 维度
    splitd1_res2 = matmul2_res[:, :, D:]  # 取剩余的 Dr 维度
    splitd1_res1_shape = "(T,N1,D)" if mla_param["t_flag"] else "(B*S1,N1,D)"
    splitd1_res2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B*S1,N1,Dr)"
    print(
        f"[INFO]splitD1 end. splitd1_res1{splitd1_res1_shape}:{tuple(splitd1_res1.shape)} splitd1_res2{splitd1_res2_shape}:{tuple(splitd1_res2.shape)}")
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
        print(f"[INFO]cast w_uk->float32")
        w_uk = w_uk.to(torch.float32)
        if uk_dtype == 'bfloat16':
            splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
    # matmul3计算
    print(
        f"[INFO]matmul3 start. splitd1_res1(B,S1,N,D):{tuple(splitd1_res1.shape)}|{splitd1_res1.dtype} w_uk(N,D,Hckv):{tuple(w_uk.shape)}|{w_uk.dtype} matmul3_dtype:{matmul3_dtype}")
    for n1_index in range(N1):
        out1[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :].to(torch.float32), w_uk[n1_index, :, :].to(torch.float32)).to(matmul3_dtype)
    # matmul3后处理
    # DynamicQuant
    out1 = out1.transpose(0, 1)
    # torch.save(out1, 'out1.pt')
    if enable_quant_output:
        if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL:
            out1 = out1.to(torch.bfloat16).to(torch.float32)
            deq_scale_q_nope_np, out1_np = dynamic_mx_quant_qn(out1.numpy(), mla_param)
            if mla_param['action_type'] == 'bm_output_gold':
                out1 = torch.tensor(out1_np.astype(np.float32))
                deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np).to(torch.float64)
            else:
                out1 = torch.tensor(out1_np.astype(np.float32)).to(torch.float8_e4m3fn)
                deq_scale_q_nope = torch.from_numpy(deq_scale_q_nope_np)
        elif mode45_full_quant:
            out1 = out1.to(torch.bfloat16).to(torch.float32)
            out1, deq_scale_q_nope = dynamic_quant_without_smooth_scale(
                out1,
                out_deqq_shape_shape,
                dtype_max=mode45_dtype_max,
                quant_dtype=torch.float32,
            )
        else:
            out1 = out1.to(torch.bfloat16).to(torch.float32)
            out1, deq_scale_q_nope = dynamic_quant_without_smooth_scale(out1, out_deqq_shape_shape)
    out1 = out1 if mla_param["t_flag"] else out1.reshape(B, S1, N1, Hckv)
    out1_shape = "(T,N1,Hckv)" if mla_param["t_flag"] else "(B,S1,N,Hckv)"
    print(f"[INFO]matmul3 end. {COLOR_YELLOW}out1{out1_shape}:{out1.shape}|{out1.dtype}{YELLOW_RESET}")

    # -------------------------------------------------------------------------------------
    # rotary1 : -> splitd1_res2(B*S1,N,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> out2(B,S1,N,Dr)
    # -------------------------------------------------------------------------------------
    splitd1_res2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B*S1,N1,Dr)"
    cos_shape = "(T,Dr)" if mla_param["t_flag"] else "(B*S1,Dr)"
    print(
        f"[INFO]rotary1 start. splitd1_res2{splitd1_res2_shape}:{tuple(splitd1_res2.shape)}|{splitd1_res2.dtype} cos{cos_shape}:{tuple(cos.shape)}|{cos.dtype}")
    expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
    expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
    q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
    out2 = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
    if enable_quant_output:
        out2 = out2.to(torch.bfloat16).to(torch.float32)
        out2 = dequant(out2, deq_scale_q_nope, quant_scale_ckv)
    out2 = out2 if mla_param["t_flag"] else out2.reshape(B, S1, N1, Dr)
    out2_shape = "(T,N1,Dr)" if mla_param["t_flag"] else "(B,S1,N1,Dr)"
    print(f"[INFO]rotary1 end. {COLOR_YELLOW}out2{out2_shape}:{tuple(out2.shape)}|{out2.dtype}{YELLOW_RESET}")

    # -------------------------------------------------------------------------------
    # matmul4 : token_x(B*S1,He) * w_kv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
    # -------------------------------------------------------------------------------
    w_kv_kr = mla_param["w_dkv_kr_tensor"]
    print(f"===========================w_kv_kr:{w_kv_kr}")
    # matmul4预处理
    matmul4_dtype = torch.float32
    if mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_FULL_INT8: # x_dtype == 'int8':
        w_kv_kr = w_kv_kr.to(torch.int32)
        matmul4_dtype = torch.int32
        if mla_param["device"] == "gpu":
            w_kv_kr = w_kv_kr.to('cpu')
    elif mla_param["weight_quant_mode"] == WEIGHT_QUANT_MODE_MXFP8_FULL:
        # w_kv_kr反量化
        w_kv_kr = w_kv_kr.to(torch.bfloat16)
        # deq_scale_dkvkr 类型转换
        deq_scale_dkvkr = mla_param["deq_scale_w_dkvkr_tensor"]
        deq_scale_dkvkr = trans_torch_fp8_e8m0_to_bf16(deq_scale_dkvkr)
        dkvkrs0 = deq_scale_dkvkr.shape[0]
        dkvkrs1 = deq_scale_dkvkr.shape[1]
        for dkvkrs0_idx in range(dkvkrs0):
            for dkvkrs1_idx in range(dkvkrs1):
                scale_dkvkr = deq_scale_dkvkr[dkvkrs0_idx:dkvkrs0_idx + 1, dkvkrs1_idx:(dkvkrs1_idx + 1)]
                w_kv_kr[dkvkrs1_idx * grp_size:(dkvkrs1_idx + 1) * grp_size, dkvkrs0_idx:dkvkrs0_idx + 1] *= scale_dkvkr
    # matmul4计算
    print(
        f"[INFO]matmul4 start. token_x{x_shape}:{tuple(token_x.shape)}|{token_x.dtype} w_kv_kr(He,Hckv+Dr):{tuple(w_kv_kr.shape)}|{w_kv_kr.dtype} matmul4_dtype:{matmul4_dtype}")
    print(f"======================token_new: {token_x_new}")
    print(f"======================w_kv_kr:{w_kv_kr}")
    token_x_matmul4 = token_x_new.to(torch.float32)
    w_kv_kr = w_kv_kr.to(torch.float32)
    if mode45_full_quant:
        token_x_matmul4 = _clamp_to_dtype_range_float(token_x_matmul4, mode45_dtype_max)
        w_kv_kr = _clamp_to_dtype_range_float(w_kv_kr, mode45_dtype_max)
    matmul4_res = torch.matmul(token_x_matmul4, w_kv_kr).to(matmul4_dtype)
    print(f"===============================================mm4:{matmul4_res}")
    # matmul4后处理
    if mla_param["weight_quant_mode"] in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8): # x_dtype == 'int8':
        if mla_param["device"] == "gpu":
            matmul4_res = matmul4_res.to('cuda')
        deq_scale_x = mla_param["deq_scale_x_tensor"]
        deq_scale_dkvkr = mla_param["deq_scale_w_dkvkr_tensor"]
        matmul4_res = matmul4_res.to(torch.float32)
        for t_index in range(T):
            matmul4_res[t_index, :] = matmul4_res[t_index, :] * deq_scale_x[t_index, 0]
        for h_index in range(Hckv + Dr):
            matmul4_res[:, h_index] = matmul4_res[:, h_index] * deq_scale_dkvkr[0, h_index]
        print(f"[INFO]deq3 end. matmul4_res dtype trans to {matmul4_res.dtype}")
    elif x_dtype == 'bfloat16':
        print(f"[INFO]cast matmul4_res->bfloat16->float32")
        matmul4_res = matmul4_res.to(torch.bfloat16).to(torch.float32)
    matmul4_res_shape = "(T,Hckv+Dr)" if mla_param["t_flag"] else "(B*S1,Hckv+Dr)"
    print(f"[INFO]matmul4 end. matmul4_res{matmul4_res_shape}:{tuple(matmul4_res.shape)}|{matmul4_res.dtype}")

    # -------------------------------------------------------------------------------------
    # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
    # -------------------------------------------------------------------------------------
    splitd2_res1 = matmul4_res[:, :Hckv]  # 取前 Hckv 维度
    splitd2_res2 = matmul4_res[:, Hckv:]  # 取剩余的 Dr 维度
    splitd2_res1_shape = "(T,Hckv)" if mla_param["t_flag"] else "(B*S1,Hckv)"
    splitd2_res2_shape = "(T,Dr)" if mla_param["t_flag"] else "(B*S1,Dr)"
    print(
        f"[INFO]splitD2 end. splitd2_res1{splitd2_res1_shape}:{tuple(splitd2_res1.shape)}, splitd2_res2{splitd2_res2_shape}:{tuple(splitd2_res2.shape)}")

    # -------------------------------------------------------------------------------------
    # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
    # -------------------------------------------------------------------------------------
    print(
        f"[INFO]rotary2 start. splitd2_res2{splitd2_res2_shape}:{tuple(splitd2_res2.shape)}|{splitd2_res2.dtype} cos{cos_shape}:{tuple(cos.shape)}|{cos.dtype}")
    k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
    rotary2_res = (k * cos) + (rotate_half(k) * sin)
    print(f"[INFO]rotary2 end. rotary2_res{splitd2_res2_shape}:{tuple(rotary2_res.shape)}|{rotary2_res.dtype}")
    # rotary2后处理
    quant_scale_ckr = mla_param["quant_scale_ckr_tensor"]
    if mla_param["weight_quant_mode"] == 1 and mla_param["kv_quant_mode"] == 2: # mla_param["flaglist"][19]:
        rotary2_res = quant(rotary2_res, quant_scale_ckr)
        print(f"[INFO]quant2 end. rotary2_res dtype trans to {rotary2_res.dtype}")

    # ----------------------------------------------------------------------------
    # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
    # ----------------------------------------------------------------------------
    ep2 = float(mla_param["epsilon_ckv"])
    gamma2 = mla_param["gamma_ckv_tensor"]
    print(
        f"[INFO]rmsnorm2 start. splitd2_res1{splitd2_res1_shape}:{tuple(splitd2_res1.shape)}|{splitd2_res1.dtype} gamma_ckv(Hckv):{tuple(gamma2.shape)}|{gamma2.dtype}")
    norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
    norm2_res *= gamma2

    # kc_scale = float(mla_param["kc_scale"])
    # norm2_res *= kc_scale
    # print(f"[INFO]rmsnorm2 end. norm2_res{splitd2_res1_shape}:{tuple(norm2_res.shape)}|{norm2_res.dtype}")

    # import pdb; pdb.set_trace()
    Dtile = Hckv
    # rmsnorm2后处理
    if mla_param["kv_quant_mode"] == 1 or mla_param["kv_quant_mode"] == 2: # kv_dtype == "int8":
        if mla_param["weight_quant_mode"] == 3:
            quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
            norm2_res_np = quant_ckv_per_tensor(norm2_res.numpy(), quant_scale_ckv.numpy())
            if mla_param['action_type'] == 'bm_output_gold':
                norm2_res = torch.tensor(norm2_res_np.astype(np.float32))
            else:
                norm2_res = torch.tensor(norm2_res_np.astype(np.float32)).to(torch.float8_e4m3fn)
            print(f"[INFO]quant1 end. norm2_res dtype trans to {norm2_res.dtype}")
        else:
            quant_scale_ckv = mla_param["quant_scale_ckv_tensor"]
            norm2_res = quant(norm2_res, quant_scale_ckv)
            print(f"[INFO]quant1 end. norm2_res dtype trans to {norm2_res.dtype}")
    elif mla_param["kv_quant_mode"] == 3:
        norm2_res = norm2_res.reshape(T, Hckv//mla_param["tile_size"], mla_param["tile_size"])
        # amax, _ = torch.max(norm2_res, dim=1, keepdim=True)
        eps = 1e-8  # 防止除零
        amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
        amax = torch.clamp(amax, min=eps)*mla_param["k_nope_clip_alpha_tensor"]
        clip_res = torch.clamp(norm2_res, min=-amax, max=amax)
        norm2_res, deq_scale_ckv = dynamic_quant_ckv_with_amax(clip_res, amax)
        deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
        norm2_res = norm2_res.reshape(T, Hckv).to(torch.int8)
        rotary2_res_bf16 = rotary2_res.to(torch.bfloat16)   # .astype(tf.bfloat16.as_numpy_dtype)
        # import pdb; pdb.set_trace()
        if mla_param["ckvkr_repo_mode"] == 1:
            # norm2_res = torch.cat((norm2_res, rotary2_res.astype(tf.bfloat16.as_numpy_dtype)), dim=-1)
            norm2_res = torch.cat((norm2_res, rotary2_res_bf16.view(torch.int8)), axis = -1)
            Dtile = Dtile + Dr * 2
        if mla_param["quant_scale_repo_mode"] == 1:
            # norm2_res = torch.cat((norm2_res, deq_scale_ckv), dim=-1)
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
    if kv_dtype == "int8":
        scatter_size = 32
    elif kv_dtype == "bf16":
        scatter_size = 16
        kv_cache = kv_cache.to(torch.bfloat16)
    else:
        scatter_size = 16
    print(
        f"[INFO]scatter1 start. norm2_res{splitd2_res1_shape}:{tuple(norm2_res.shape)}|{norm2_res.dtype} kv_cache{out3_info}:{tuple(kv_cache.shape)}|{kv_cache.dtype} scatter_size:{scatter_size}")


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
    print(f"[INFO]scatter1 end. {COLOR_YELLOW}out3{out3_info}:{tuple(out3.shape)}|{out3.dtype}{YELLOW_RESET}")

    # -------------------------------------------------------------------------------------
    # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
    # -------------------------------------------------------------------------------------
    print(
        f"[INFO]rotary2 start. splitd2_res2{splitd2_res2_shape}:{tuple(splitd2_res2.shape)}|{splitd2_res2.dtype} cos{cos_shape}:{tuple(cos.shape)}|{cos.dtype}")
    k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
    rotary2_res = (k * cos) + (rotate_half(k) * sin)
    print(f"[INFO]rotary2 end. rotary2_res{splitd2_res2_shape}:{tuple(rotary2_res.shape)}|{rotary2_res.dtype}")
    # rotary2后处理
    quant_scale_ckr = mla_param["quant_scale_ckr_tensor"]
    if mla_param["flaglist"][19]:
        rotary2_res = quant(rotary2_res, quant_scale_ckr)
        print(f"[INFO]quant2 end. rotary2_res dtype trans to {rotary2_res.dtype}")

    # ----------------------------------------------------------------------------------------------
    # scatter2 : rotary2_res(B*S1,Dr) * kr_cache(B,N2,S2,Dr/B,B,N2,Dr) -> out4(B,N2,S2,Dr/B,B,N2,Dr)
    # ----------------------------------------------------------------------------------------------
    if mla_param['ckvkr_repo_mode'] == 1:
        out4 = copy.deepcopy(mla_param["kr_cache_tensor"])
    else:
        kr_cache = copy.deepcopy(mla_param["kr_cache_tensor"])
        # kr_cache = np.zeros_like(kr_cache_input)
        # kr_cache = torch.from_numpy(kr_cache)
        out4_shape = kr_cache.shape
        out4_info = "(B,B,N2,Dr)"
        if not mla_param['pa_flag']:
            kr_cache = kr_cache.transpose(2, 1)
            out4_info = "(B,N2,S2,Dr)"
        if kr_dtype == "int8":
            scatter_size = 32
        elif kr_dtype == "bf16":
            scatter_size = 16
            kr_cache = kr_cache.to(torch.bfloat16)
        else:
            scatter_size = 16
        print(
            f"[INFO]scatter2 start. rotary2_res{splitd2_res2_shape}:{tuple(rotary2_res.shape)}|{rotary2_res.dtype} kr_cache{out4_info}:{tuple(kr_cache.shape)}|{kr_cache.dtype} scatter_size:{scatter_size}")
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
        print(f"[INFO]scatter2 end. {COLOR_YELLOW}out4{out4_info}:{tuple(out4.shape)}|{out4.dtype}{YELLOW_RESET}")

    print("[INFO]========================================")
    print("[INFO]>>>>>>>>   Calculate success  >>>>>>>>>>")
    print("[INFO]========================================")

    return out1, out2, out3, out4, deq_scale_q_nope, out_qnorm, out_deq_qnorm


def aclnn_op_func_cpu(torch_tensor_list, params):
    print(f"[INFO]Now run aclnn_op_func_cpu！")
    print(params)
    # 获取参数
    mla_param = get_param(torch_tensor_list, params)
    print(f"[INFO]cache_mode : {mla_param['cache_mode']}")
    out1 = torch.zeros(mla_param['out_q_shape'])
    out2 = torch.zeros(mla_param['out_qrope_shape'])
    # out3 = torch.from_numpy(mla_param["kv_cache_tensor"])
    out3 = torch.tensor(mla_param["kv_cache_tensor"].astype(np.float32)).to(torch.float8_e4m3fn)
    out4 = torch.from_numpy(mla_param["kr_cache_tensor"])
    if not mla_param['normal_flag']:
        return out1, out2, out3, out4
    B = mla_param['B']
    S1 = mla_param['S1']
    S2 = mla_param['S2']
    if B == 0 or S1 == 0 or (0 in mla_param['out_q_shape']):
        if mla_param["enable_quant_output"]:
            deq_scale_q_nope = torch.zeros(mla_param['out_deqq_shape'])
            return out1, out2, out3, out4, deq_scale_q_nope
        return out1, out2, out3, out4


    # sin/cos 处理: 1、生成合理的sin/cos tensor 2、覆写sin/cos tensor(暂不做)

    # index处理：1.bm_output/bm_output_gold:直接读取 2.bm:生成合理的indextable并覆写
    # if "output" not in params["action_type"]:
    #     print("[INFO]Begin to gen index table")
    #     index_table_shape = mla_param['cache_index_shape']
    #     if S2==0:
    #         index_table = np.arange(B * S1)
    #     else:
    #         # 计算总元素数量
    #         total_elements = B * S2
    #         # 创建一个包含从0到total_elements-1的所有数字的列表
    #         if "BLK" in mla_param['cache_mode']:
    #             all_numbers = np.arange(out3.shape[0])
    #         else:
    #             total_elements = B * S2
    #             all_numbers = np.arange(total_elements)
    #         # 对这个列表进行随机排序
    #         shuffled_numbers = np.random.permutation(all_numbers)
    #         # 将排序后的列表重塑为所需的形状 (B, S1)
    #         required_num_elements = math.prod(int(x) for x in index_table_shape)
    #         index_table = shuffled_numbers[:required_num_elements].reshape(index_table_shape)

    #     mla_param['cache_index_tensor'] = index_table
    #     print("===== index_table: ", index_table)

    #     # index_table 覆盖原有的input
    #     tools.modify_alcnn_input_file(ids=9, origin_index=[9], type='tensor', mode='rewrite',
    #                                   tensors=torch.tensor(index_table, dtype=torch.int64),
    #                                   params=params)

    #     print("*** para: ", mla_param['actual_seq_len'])
    #     if mla_param['actual_seq_len'] is not None:
    #         actual_seq_tensor = np.array(mla_param['actual_seq_len'])
    #         tools.modify_alcnn_input_file(ids=19, origin_index=[19], type='tensor', mode='rewrite',  # TODO 确定好编号
    #                                     tensors=torch.tensor(actual_seq_tensor, dtype=torch.int64),
    #                                     params=params)

    # mla_param_torch = convert_dict_values_to_torch(mla_param)
    # out1, out2, out3, out4, deq_scale_q_nope, query_norm, deq_scale_q_norm = cal_mlaprolog(mla_param_torch)


    # results = [
    #     v for v in (
    #         out1, out2, out3, out4,
    #         deq_scale_q_nope, query_norm, deq_scale_q_norm
    #     )
    #     if v is not None
    # ]
    # # 只保留非空输出
    # return results

    if "output" not in params["action_type"]:
        print("[INFO]Begin to gen index table")
        index_table_shape = mla_param['cache_index_shape']
        if S2==0:
            index_table = np.arange(B * S1)
            if mla_param['action_type'] != 'bm_output_gold':
                if not mla_param["t_flag"]:
                    index_table = index_table.reshape((B,S1))
        else:
            # 计算总元素数量
            if mla_param['action_type'] != 'bm_output_gold':
                total_elements = B * S2
            # 创建一个包含从0到total_elements-1的所有数字的列表
            if "BLK" in mla_param['cache_mode']:
                all_numbers = np.arange(out3.shape[0])
            else:
                total_elements = B * S2
                all_numbers = np.arange(total_elements)
            # 对这个列表进行随机排序
            shuffled_numbers = np.random.permutation(all_numbers)
            # 将排序后的列表重塑为所需的形状 (B, S1)
            required_num_elements = math.prod(int(x) for x in index_table_shape)
            index_table = shuffled_numbers[:required_num_elements].reshape(index_table_shape)

        mla_param['cache_index_tensor'] = index_table
        print("===== index_table: ", index_table)

        # index_table 覆盖原有的input
        tools.modify_alcnn_input_file(ids=9, origin_index=[9], type='tensor', mode='rewrite',
                                      tensors=torch.tensor(index_table, dtype=torch.int64),
                                      params=params)

        print("*** para: ", mla_param['actual_seq_len'])
        if mla_param['actual_seq_len'] is not None:
            actual_seq_tensor = np.array(mla_param['actual_seq_len'])
            tools.modify_alcnn_input_file(ids=19, origin_index=[19], type='tensor', mode='rewrite',  # TODO 确定好编号
                                        tensors=torch.tensor(actual_seq_tensor, dtype=torch.int64),
                                        params=params)

    mla_param_torch = convert_dict_values_to_torch(mla_param)
    out1, out2, out3, out4, deq_scale_q_nope, query_norm, deq_scale_q_norm = cal_mlaprolog(mla_param_torch)
    if mla_param['action_type'] != 'bm_output_gold' and mla_param_torch['qnorm_flag'] :
        query_norm = query_norm.reshape(mla_param_torch['out_qnorm_shape'])

    results = [
        v for v in (
            out1, out2, out3, out4,
            deq_scale_q_nope, query_norm, deq_scale_q_norm
        )
        if v is not None
    ]
    # 只保留非空输出
    return results


def aclnn_op_func_gpu(torch_tensor_list, params):
    print(f"[INFO]Now run aclnn_op_func_gpu！")
    print(params)
    # 获取参数
    mla_param = get_param(torch_tensor_list, params)
    # 计算
    out1, out2, out3, out4, deq_scale_q_nope, query_norm, deq_scale_q_norm = cal_mlaprolog(mla_param_torch)

    results = [
        v for v in (
            out1, out2, out3, out4,
            deq_scale_q_nope, query_norm, deq_scale_q_norm
        )
        if v is not None
    ]
    # 只保留非空输出
    return results


def trans_format(tensor, format, name):
    import torch_npu
    print(f"[INFO]Do trans_format: {name}:{format}")
    if format == 'FRACTAL_NZ':
        new_tensor = torch_npu.npu_format_cast(tensor.contiguous(), 29)
        print(f"[INFO]Trans {name} from {torch_npu.get_npu_format(tensor)} to {torch_npu.get_npu_format(new_tensor)}!")
    elif format == 'ND':
        new_tensor = torch_npu.npu_format_cast(tensor.contiguous(), 2)
        print(f"[INFO]Trans {name} from {torch_npu.get_npu_format(tensor)} to {torch_npu.get_npu_format(new_tensor)}!")
    else:
        new_tensor = tensor
    return new_tensor


def aclnn_op_func_npu(torch_tensor_list, params):
    print(f"[INFO]Now run aclnn_op_func_npu!")
    import torch_npu
    import torchair as tng
    import torchair._contrib.custom_torch_ops
    from torchair.configs.compiler_config import CompilerConfig
    # import logging
    # torch._logging.set_logs(dynamo=logging.DEBUG,aot=logging.DEBUG,output_code=True,graph_code=True,recompiles=True)

    # 获取参数
    print(params)
    mla_param = get_param(torch_tensor_list, params)

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
    x_dtype = mla_param["x_dtype"]

    # format
    w_dq_format = params['format_input'][1]
    w_uq_qr_format = params['format_input'][2]
    w_dkv_kr_format = params['format_input'][4]

    w_dq = mla_param['w_dq_tensor']
    w_dq = trans_format(w_dq, w_dq_format, "w_dq")
    w_uq_qr = mla_param['w_uq_qr_tensor']
    w_uq_qr = trans_format(w_uq_qr, w_uq_qr_format, "w_uq_qr")
    w_dkv_kr = mla_param['w_dkv_kr_tensor']
    w_dkv_kr = trans_format(w_dkv_kr, w_dkv_kr_format, "w_dkv_kr")

    torch._dynamo.reset()

    # 构图
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model_Rope(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, cos, sin):
            out = torch.ops.npu_inference.npu_interleave_rope(x, cos, sin)
            return out

    rope_model = Model_Rope().npu()
    rope_model = torch.compile(rope_model, backend=npu_backend, dynamic=False)

    class Model_RmsnormRope(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, kv, weight, cos, sin, kv_len, cache_rope, cache, ep, mode):
            out1, out2 = torch.ops.npu_inference.npu_kv_rmsnorm_rope_cache(
                kv, weight, cos, sin, kv_len, cache_rope, cache,
                epsilon=ep, cache_mode=mode)
            return out1, out2

    norm_rope_model = Model_RmsnormRope().npu()
    norm_rope_model = torch.compile(norm_rope_model, backend=npu_backend, dynamic=False)

    print("[INFO]========================================")
    print("[INFO]>>>>>>>  Start cal npu small  >>>>>>>>>>")
    print("[INFO]========================================")

    # matmul1
    token_x = mla_param['x_tensor']
    if x_dtype == 'int8':
        deq_scale_cq = mla_param['deq_scale_cq_tensor']
        print(f"[INFO]npu_quant_matmul1:token_x[{tuple(token_x.shape)}|{token_x.dtype}] "
              f"w_dq[{tuple(w_dq.shape)}|{w_dq.dtype}] "
              f"deq_scale_cq[{tuple(deq_scale_cq.shape)}|{deq_scale_cq.dtype}]")
        matmul1_res = torch_npu.npu_quant_matmul(token_x, w_dq, deq_scale_cq, bias=None, output_dtype=torch.int8)
    else:
        print(f"[INFO]matmul1:token_x[{tuple(token_x.shape)}|{token_x.dtype}] w_dq[{tuple(w_dq.shape)}|{w_dq.dtype}]")
        matmul1_res = torch.matmul(token_x, w_dq)
    print(f"[INFO]matmul1_res[{tuple(matmul1_res.shape)}|{matmul1_res.dtype}]")

    # rmsnorm1
    ep1 = mla_param['epsilon_cq']
    gamma_cq = mla_param['gamma_cq_tensor']
    print(
        f"[INFO]npu_add_rms_norm:matmul1_res[{tuple(matmul1_res.shape)}|{matmul1_res.dtype}] gamma_cq[{tuple(gamma_cq.shape)}|{gamma_cq.dtype}]")
    residual = torch.zeros(tuple(matmul1_res.shape)).to(matmul1_res.dtype).npu()
    rmsnorm1_res = torch_npu.npu_add_rms_norm(residual, matmul1_res, gamma_cq, ep1)[0]
    rmsnorm1_res = rmsnorm1_res.view(B, S1, Hcq)
    print(f"[INFO]rmsnorm1_res[{tuple(rmsnorm1_res.shape)}|{rmsnorm1_res.dtype}]")

    # matmul2
    if x_dtype == 'int8':
        deq_scale_qc_qr = mla_param['deq_scale_qc_qr_tensor']
        print(f"[INFO]npu_quant_matmul2:rmsnorm1_res[{tuple(rmsnorm1_res.shape)}|{rmsnorm1_res.dtype}] "
              f"w_uq_qr[{tuple(w_uq_qr.shape)}|{w_uq_qr.dtype}] "
              f"deq_scale_qc_qr[{tuple(deq_scale_qc_qr.shape)}|{deq_scale_qc_qr.dtype}]")
        matmul2_res = torch_npu.npu_quant_matmul(rmsnorm1_res, w_uq_qr, deq_scale_qc_qr, bias=None,
                                                 output_dtype=torch.float16)
    else:
        print(
            f"[INFO]matmul2:rmsnorm1_res[{tuple(rmsnorm1_res.shape)}|{rmsnorm1_res.dtype}] w_uq_qr[{tuple(w_uq_qr.shape)}|{w_uq_qr.dtype}]")
        matmul2_res = torch.matmul(rmsnorm1_res, w_uq_qr)
    matmul2_res = matmul2_res.reshape(B, S1, N1, D + Dr).transpose(1, 2)
    print(f"[INFO]matmul2_res[{tuple(matmul2_res.shape)}|{matmul2_res.dtype}]")

    # split1
    split1_res1, split1_res2 = torch.split(matmul2_res, [D, Dr], dim=-1)
    # if S1 == 1:
    #     split1_res2 = split1_res2.view(B, N1, S1, Dr)  # b,n,s,d
    # else:
    #     split1_res2 = split1_res2.transpose(1, 2)
    print(
        f"[INFO]split1_res1[{tuple(split1_res1.shape)}|{split1_res1.dtype}] split1_res2[{tuple(split1_res2.shape)}|{split1_res2.dtype}]")

    # matmul3
    w_uk = mla_param['w_uk_tensor']
    split1_res1 = split1_res1.reshape(B * S1, N1, D).transpose(0, 1)  # bs,n,d -> n,bs,d
    print(
        f"[INFO]matmul3: split1_res1[{tuple(split1_res1.shape)}|{split1_res1.dtype}] w_uk[{tuple(w_uk.shape)}|{w_uk.dtype}]")
    out1 = torch.matmul(split1_res1, w_uk).transpose(1, 0).view(B, S1, N1, Hckv)
    # if S1 == 1:
    #    out1 = out1.view(B, N1, S1, Hckv)
    # else:
    #    out1 = out1.transpose(1, 2)  # b,s,n,d -> b,n,s,d
    print(f"[INFO]out1[{tuple(out1.shape)}|{out1.dtype}]")

    # rope
    cos = mla_param["cos_tensor"]
    sin = mla_param["sin_tensor"]
    expanded_cos = cos.unsqueeze(2).transpose(1, 2)
    expanded_sin = sin.unsqueeze(2).transpose(1, 2)
    print(
        f"[INFO]npu_interleave_rope:split1_res2[{tuple(split1_res2.shape)}|{split1_res2.dtype}] expanded_cos[{tuple(expanded_cos.shape)}|{expanded_cos.dtype}]")
    out2 = rope_model(split1_res2, expanded_cos, expanded_sin)
    print(f"[INFO]out2[{tuple(out2.shape)}|{out2.dtype}]")

    # matmul4
    if x_dtype == 'int8':
        # 存疑，待确认
        deq_scale_ckv_kr = mla_param['deq_scale_ckv_kr_tensor']
        matmul4_res = torch_npu.npu_quant_matmul(token_x, w_dkv_kr, deq_scale_ckv_kr, bias=None,
                                                 output_dtype=torch.float16)
    else:
        print(
            f"[INFO]matmul4: token_x[{tuple(token_x.shape)}|{token_x.dtype}] w_dkv_kr[{tuple(w_dkv_kr.shape)}|{w_dkv_kr.dtype}]")
        matmul4_res = torch.matmul(token_x, w_dkv_kr)
    print(f"[INFO]matmul4_res[{tuple(matmul4_res.shape)}|{matmul4_res.dtype}]")

    # rmsnorm2
    ep2 = mla_param['epsilon_ckv']
    gamma_ckv = mla_param['gamma_ckv_tensor']
    kv_cache = mla_param["kv_cache_tensor"]
    kr_cache = mla_param["kr_cache_tensor"]
    cache_index = mla_param["cache_index_tensor"]
    matmul4_res = matmul4_res.view(B, N2, S1, -1)

    if mla_param["pa_flag"]:
        cache_index = cache_index.view(-1)
        # kr_cache = kr_cache.unsqueeze(2)
        # kv_cache = kv_cache.unsqueeze(2)
        print(f"[INFO]npu_kv_rmsnorm_rope_cache(PA): matmul4_res[{tuple(matmul4_res.shape)}|{matmul4_res.dtype}] "
              f"gamma_ckv[{tuple(gamma_ckv.shape)}|{gamma_ckv.dtype}] "
              f"kv_cache[{tuple(kv_cache.shape)}|{kv_cache.dtype}] "
              f"kr_cache[{tuple(kr_cache.shape)}|{kr_cache.dtype}] "
              f"cache_index[{tuple(cache_index.shape)}|{cache_index.dtype}]")
        out4, out3 = norm_rope_model(
            matmul4_res, gamma_ckv, expanded_cos, expanded_sin, cache_index, kr_cache, kv_cache, ep2, "PA")
        out3 = out3.squeeze(2)
        out4 = out4.squeeze(2)
    else:
        cache_index = cache_index.view(B, -1)
        print(f"[INFO]npu_kv_rmsnorm_rope_cache: matmul4_res[{tuple(matmul4_res.shape)}|{matmul4_res.dtype}] "
              f"gamma_ckv[{tuple(gamma_ckv.shape)}|{gamma_ckv.dtype}] "
              f"kv_cache[{tuple(kv_cache.shape)}|{kv_cache.dtype}] "
              f"kr_cache[{tuple(kr_cache.shape)}|{kr_cache.dtype}] "
              f"cache_index[{tuple(cache_index.shape)}|{cache_index.dtype}]")
        out4, out3 = norm_rope_model(
            matmul4_res, gamma_ckv, expanded_cos, expanded_sin, cache_index, kr_cache, kv_cache, ep2, "Norm")
    print(f"[INFO]out3[{tuple(out3.shape)}|{out3.dtype}]")
    print(f"[INFO]out4[{tuple(out4.shape)}|{out4.dtype}]")

    print("[INFO]========================================")
    print("[INFO]>>>>>>>>   Calculate success  >>>>>>>>>>")
    print("[INFO]========================================")
    return out1, out2, out3, out4


def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def get_output_list(is_list_or_tuple, input_list, output_value_list):
    while True:
        if not is_list_or_tuple:
            return output_value_list
        else:
            is_list_or_tuple, input_list = find_output_value(input_list, output_value_list)


def aclnn_op_func(torch_tensor_list, params):
    action_type = params["action_type"]
    cfg_fk = tools.ConfigFmk()

    if action_type in ("bm", "bm_output", "multi_bm", "bm_gold", "bm_output_gold", "one"):
        if cfg_fk.device_type in ("gpu",):
            device_type = "gpu"
        else:
            device_type = "cpu"
    else:
        device_type = "npu"
    print("device_type", device_type)

    if device_type == "cpu":
        print(f"cpu 执行中...")
        return aclnn_op_func_cpu(torch_tensor_list, params)

    elif device_type == "gpu":
        print(f"gpu 执行中...")
        device_id = os.environ['DEVICE_ID']
        print("device_id is ", device_id)
        torch.cuda.set_device(f"cuda:{device_id}")
        return aclnn_op_func_gpu(torch_tensor_list, params)

    else:
        print(f"npu 执行中...")
        return aclnn_op_func_npu(torch_tensor_list, params)


def get_operator(input_tensor_list, params, device, device_id):
    if device == "npu":
        torch.npu.set_device(f"npu:{device_id}")
    torch_tensor_list = list()
    for i in range(len(input_tensor_list)):
        input_tensor = input_tensor_list[i]

        if any("Tensor" in _key for _key in params.keys()):
            if device == "cpu" and input_tensor.dtype == torch.float16:
                input_tensor = input_tensor.float()
            torch_tensor_list.append(input_tensor.to(device))
        else:
            if device == "cpu":
                if str(input_tensor.dtype) == "bfloat16":
                    input_tensor = input_tensor.astype(np.float32)
                torch_tensor_list.append(input_tensor)
            elif device == "npu":
                if str(input_tensor.dtype) == "uint64":
                    print(type(input_tensor), "torch不支持uint64！")
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.int64)).to(torch.int64)
                    torch_tensor_list.append(input_tensor_i.to(device))
                elif str(input_tensor.dtype) == "bfloat16":
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)

                    torch_tensor_list.append(input_tensor_i.to(device))
                elif str(input_tensor.dtype) == "float8_e4m3fn":
                    print(f"=========float8_e4m3fn*********==========")
                    # print(i)
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.float8_e4m3fn)
                    # print("input_tensor_i type", input_tensor_i.dtype)
                    torch_tensor_list.append(input_tensor_i.to(device))
                else:
                    input_tensor_i = torch.from_numpy(input_tensor)
                    torch_tensor_list.append(input_tensor_i.to(device))
            elif device == "cuda":
                if str(input_tensor.dtype) == "uint64":
                    print(type(input_tensor), "torch不支持uint64！")
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.int64)).to(torch.int64)
                    torch_tensor_list.append(input_tensor_i.to(device))

                elif str(input_tensor.dtype) == "bfloat16":
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32))
                    torch_tensor_list.append(input_tensor_i.to(device))
                else:
                    input_tensor_i = torch.from_numpy(input_tensor)
                    torch_tensor_list.append(input_tensor_i.to(device))

    print("=============== params['dtype_output']: ", params['dtype_output'])
    if device == "npu":
        output_data = training.aclnn_execute_npu(torch_tensor_list, params)
    elif device == "npu_off":
        training.aclnn_execute_npu_off(torch_tensor_list, params)
        return
    else:
        output_data = training.aclnn_execute_cpu(torch_tensor_list, params)

    if isinstance(output_data, list) or isinstance(output_data, tuple):
        output_value = []
        get_output_list(True, output_data, output_value)
        output_list = []
        print("=============== params['dtype_output']: ", params['dtype_output'])
        for i in range(len(output_value)):
            tensor_tmp = output_value[i].to("cpu")
            if params["dtype_output"][i] == "bf16" or params["dtype_output"][i] == "float8_e4m3fn":
                tensor_tmp = tensor_tmp.float()
            tensor_tmp = tensor_tmp.detach().numpy()
            if params["dtype_output"][i] == "float8_e4m3fn":
                from ml_dtypes import float8_e4m3fn
                tensor_tmp = tensor_tmp.astype(float8_e4m3fn, copy=False)
            if device == "cpu" and tools.get_np_dtype(params["dtype_output"][0]) == np.float16:
                tensor_tmp = tensor_tmp.astype(np.float16)
            output_list.append(tensor_tmp)
        return output_list
    elif isinstance(output_data, bool):
        output_data = output_data
    else:
        output_data = output_data.to("cpu")
        if params["dtype_output"][0] == "bf16":
            output_data = output_data.float()
        output_data = output_data.detach().numpy()
        if device == "cpu" and params["dtype_output"][0] and tools.get_np_dtype(
                params["dtype_output"][0]) == np.float16:
            output_data = output_data.astype(np.float16)

    return output_data


# @set_timeout(120)
def aclnnMlaPrologV3WeightNz(action_type, params, case_path, op_name, soc_version, framework):
    case_name = params["case_name"]
    params["op_name"] = op_name
    params["case_path"] = case_path
    result, error_percent, max_error = "Pass", 100.0, 1.0
    if action_type in ("bm", "one"):
        tools.print_log("-----------------------Start to generate cpu pytorch atenIR golden--------")
        result, error_percent, max_error = training.run_pytorch_training_cpu(params)
    if action_type in ("bm_input",):
        tools.print_log("-----------------------Start to run bm_input-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_input(params)
    if action_type in ("bm_output",):
        tools.print_log("-----------------------Start to run bm_output-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_output(params)
    if action_type in ("bm_gold",):
        tools.print_log("-----------------------Start to run bm_gold-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_gold(params)
    if action_type in ("bm_output_gold",):
        tools.print_log("-----------------------Start to run bm_output_gold-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_output_gold(params)
    if action_type in ("npu", "one") and result == "Pass":
        tools.print_log("-----------------------Start to run aclnn online--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_online(params)
    if action_type in ("npu_off",) and result == "Pass":
        tools.print_log("-----------------------Start to run aclnn offline--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_offline_multi_input(params)
    if action_type in ("npu_off_all",) and result == "Pass":
        if params['index'] == 0:
            tools.print_log("-----------------------Start to run aclnn offline all--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_offline_all_multi_input(params)
    if action_type in ("compare",) and result == "Pass":
        tools.print_log("-----------------------Start to compare aclnn offline--------")
        result, error_percent, max_error = training.run_pytorch_training_compare_multi_input(params)
    return result, error_percent, max_error
