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

import os
import torch
try:
    import torch_npu
except Exception:
    torch_npu = None
import check_valid_param
import pytest
import math
import copy
import numpy as np
import random
from typing import Dict, Optional, Tuple

COLOR_YELLOW = "\033[33m"
YELLOW_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
GREEN_RESET = "\033[0m"

WEIGHT_QUANT_MODE_FULL_INT8 = 2
WEIGHT_QUANT_MODE_MXFP8_FULL = 3
WEIGHT_QUANT_MODE_FULL_FP8_E4M3 = 4
WEIGHT_QUANT_MODE_FULL_HIF8 = 5

INT8_DTYPE_MAX = 127.0
FP8_E4M3_DTYPE_MAX = 448.0
HIF8_DTYPE_MAX = 32768.0
FP8_BLOCK_SIZE = 32


def _info_log_enabled():
    return os.getenv("MLA_PROLOG_V3_CPU_INFO_LOG", "0").strip().lower() in ("1", "true", "yes", "on")


INFO_LOG_ENABLED = _info_log_enabled()


def info_log(message):
    if INFO_LOG_ENABLED:
        print(message)


def _get_hif8_dtype():
    if torch_npu is None:
        return None
    return getattr(torch_npu, "hifloat8", None)


def has_hif8_dtype():
    return _get_hif8_dtype() is not None


def _get_full_quant_mode_config(weight_quant_mode):
    if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_INT8:
        return {
            "input_dtype": torch.int8,
            "dtype_max": INT8_DTYPE_MAX,
            "query_quant_dtype": torch.int8,
        }
    if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_FP8_E4M3:
        if not hasattr(torch, "float8_e4m3fn"):
            return None
        return {
            "input_dtype": torch.float8_e4m3fn,
            "dtype_max": FP8_E4M3_DTYPE_MAX,
            "query_quant_dtype": torch.float8_e4m3fn,
        }
    if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_HIF8:
        hif8_dtype = _get_hif8_dtype()
        if hif8_dtype is None:
            return None
        return {
            "input_dtype": hif8_dtype,
            "dtype_max": HIF8_DTYPE_MAX,
            "query_quant_dtype": hif8_dtype,
        }
    return None


def _get_quant_dtype_max(quant_dtype):
    if quant_dtype == torch.int8:
        return INT8_DTYPE_MAX
    if hasattr(torch, "float8_e4m3fn") and quant_dtype == torch.float8_e4m3fn:
        return FP8_E4M3_DTYPE_MAX
    hif8_dtype = _get_hif8_dtype()
    if hif8_dtype is not None and quant_dtype == hif8_dtype:
        return HIF8_DTYPE_MAX
    raise ValueError(f"unsupported quant dtype: {quant_dtype}")


def _get_kv_tile_quant_dtype(kv_quant_mode, kv_cache_dtype=None, weight_quant_mode=None):
    if kv_quant_mode != 3:
        return None

    if kv_cache_dtype is not None:
        if kv_cache_dtype in (torch.int8,):
            return kv_cache_dtype
        if hasattr(torch, "float8_e4m3fn") and kv_cache_dtype == torch.float8_e4m3fn:
            return kv_cache_dtype
        hif8_dtype = _get_hif8_dtype()
        if hif8_dtype is not None and kv_cache_dtype == hif8_dtype:
            return kv_cache_dtype
        raise ValueError(f"unsupported kv tile quant dtype: {kv_cache_dtype}")

    if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
        if not hasattr(torch, "float8_e4m3fn"):
            raise ValueError("weight_quant_mode=3 requires torch.float8_e4m3fn support")
        return torch.float8_e4m3fn

    full_quant_mode_config = _get_full_quant_mode_config(weight_quant_mode)
    if full_quant_mode_config is not None:
        return full_quant_mode_config["input_dtype"]
    return torch.int8


def _dynamic_quant_clip_range(dtype_max, quant_dtype):
    if quant_dtype == torch.int8:
        return -128.0, 127.0
    return -float(dtype_max), float(dtype_max)


def _fp8_blockwise_quant(inputs, block_size=FP8_BLOCK_SIZE, dtype_max=FP8_E4M3_DTYPE_MAX,
                         quant_dtype=None):
    """FP8 per-block quantization (block_size along last dim). Returns (quantized, scale)."""
    if quant_dtype is None:
        quant_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float32
    if inputs.ndim != 2:
        raise ValueError(f"fp8 blockwise quant expects 2D input, got shape={tuple(inputs.shape)}")
    t_size, h_size = inputs.shape
    if h_size % block_size != 0:
        raise ValueError(f"fp8 blockwise quant requires H % {block_size} == 0, got H={h_size}")
    inputs_fp32 = inputs.to(torch.float32)
    blocks = h_size // block_size
    reshaped = inputs_fp32.reshape(t_size, blocks, block_size)
    amax = torch.max(torch.abs(reshaped), dim=-1)[0]
    scale = torch.clamp(amax / float(dtype_max), min=1e-8)
    scaled = torch.round(reshaped / scale.unsqueeze(-1))
    clipped = torch.clamp(scaled, min=-float(dtype_max), max=float(dtype_max))
    quantized = clipped.to(quant_dtype).reshape(t_size, h_size)
    return quantized, scale


def _dequant_fp8_blockwise(inputs, scale, block_size=FP8_BLOCK_SIZE):
    """Dequant FP8 tensor with per-block scales along last dim."""
    if inputs.ndim != 2:
        raise ValueError(f"fp8 blockwise dequant expects 2D input, got shape={tuple(inputs.shape)}")
    t_size, h_size = inputs.shape
    if h_size % block_size != 0:
        raise ValueError(f"fp8 blockwise dequant requires H % {block_size} == 0, got H={h_size}")
    blocks = h_size // block_size
    inputs_fp32 = inputs.to(torch.float32).reshape(t_size, blocks, block_size)
    scale_fp32 = scale.to(torch.float32).reshape(t_size, blocks, 1)
    return (inputs_fp32 * scale_fp32).reshape(t_size, h_size)


def _dequant_fp8_weight(weight, scale, block_size=FP8_BLOCK_SIZE):
    """Dequant FP8 weight with per-block scales along first dim."""
    if weight.ndim != 2:
        raise ValueError(f"fp8 weight dequant expects 2D input, got shape={tuple(weight.shape)}")
    k_size, n_size = weight.shape
    if k_size % block_size != 0:
        raise ValueError(f"fp8 weight dequant requires K % {block_size} == 0, got K={k_size}")
    weight_fp32 = weight.to(torch.float32)
    scale_fp32 = scale.to(torch.float32)
    blocks = k_size // block_size
    weight_reshaped = weight_fp32.reshape(blocks, block_size, n_size)
    scale_t = scale_fp32.transpose(0, 1).reshape(blocks, 1, n_size)
    return (weight_reshaped * scale_t).reshape(k_size, n_size)


# ===================== Helper Functions =====================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


def s8_saturation(inputdata):
    inputdata = torch.where(inputdata > 127, 127, inputdata)
    inputdata = torch.where(inputdata < -128, -128, inputdata)
    return inputdata.to(torch.int8)


def s9_saturation(inputdata):
    inputdata = torch.where(inputdata > 255, 255, inputdata)
    inputdata = torch.where(inputdata < -256, -256, inputdata)
    return inputdata


def quant(x, qscale):
    """INT8 quantization with saturation."""
    scaled_values = (x * qscale).round().to(torch.float32)
    s9_res = s9_saturation(scaled_values)
    s8_res_cal = s8_saturation(s9_res)
    return s8_res_cal


def quant_with_scale(x, qscale, quant_dtype=torch.int8):
    if quant_dtype == torch.int8:
        return quant(x, qscale)
    dtype_max = _get_quant_dtype_max(quant_dtype)
    scaled_values = torch.round(x.to(torch.float32) * qscale.to(torch.float32))
    clip_min, clip_max = _dynamic_quant_clip_range(dtype_max, quant_dtype)
    return torch.clamp(scaled_values, min=clip_min, max=clip_max).to(quant_dtype)


def numpy_float8_e4m3fn():
    try:
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype! "
                           "Please install with `pip3 install ml-dtypes`")


def quant_ckv_per_tensor(input_data, quant_scale_ckv):
    """FP8 per-tensor quantization for kv cache."""
    scaled_value = input_data * quant_scale_ckv
    scaled_value = np.round(scaled_value, 8)
    scaled_value = scaled_value.astype(numpy_float8_e4m3fn(), copy=False)
    return scaled_value


def dynamic_quant(inputs, smooth_scale, dtype_max=INT8_DTYPE_MAX, quant_dtype=torch.int32):
    """Dynamic quantization with optional smooth scale. Returns (quantized, scale)."""
    T = inputs.size(0)
    H = inputs.size(1)
    y = torch.zeros(T, H).to(torch.float32)
    scale = torch.zeros(T).to(torch.float32)
    inputs = inputs.reshape(T, H).to(torch.float32)
    if smooth_scale is not None:
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
            y[bs_index:] = torch.round(inputs[bs_index:] / scale_bs)
    if quant_dtype == torch.int32:
        y = y.to(torch.int32)
    else:
        clip_min, clip_max = _dynamic_quant_clip_range(dtype_max, quant_dtype)
        y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
    return y, scale


def dynamic_quant_without_smooth_scale(inputs, out_deqq_shape_shape, dtype_max=INT8_DTYPE_MAX, quant_dtype=torch.int8):
    """Dynamic quantization for query output. Returns (quantized, scale)."""
    T = inputs.size(0)
    N = inputs.size(1)
    H = inputs.size(2)
    quant_loops = inputs.size(0)
    eles_with_one_scale = inputs.size(1) * inputs.size(2)
    if len(out_deqq_shape_shape) == 3:  # [BS, N, 1], per_token per_head
        quant_loops = inputs.size(0) * inputs.size(1)
        eles_with_one_scale = inputs.size(2)
    y = torch.zeros(quant_loops, eles_with_one_scale).to(torch.int32)
    scale = torch.zeros(quant_loops).to(torch.float32)
    inputs = inputs.reshape(quant_loops, eles_with_one_scale).to(torch.float32)
    max_values, _ = torch.max(torch.abs(inputs), dim=-1, keepdim=True)
    scale = max_values / float(dtype_max)
    y = torch.round(inputs / scale)
    clip_min, clip_max = _dynamic_quant_clip_range(dtype_max, quant_dtype)
    y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
    if len(out_deqq_shape_shape) == 2:  # [BS, 1], per_token
        return y.reshape(T, N, H), scale.reshape(quant_loops, 1).to(torch.float64)
    else:
        return y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64)


def dynamic_quant_ckv_with_amax(inputs, amax, smooth_scale=None, quant_dtype=torch.int8):
    """3D dynamic quantization with external amax. Returns (quantized, scale)."""
    T, N, H = inputs.shape
    dtype_max = _get_quant_dtype_max(quant_dtype)
    inputs = inputs.to(torch.float32)
    amax = amax.to(torch.float32).clamp(min=1e-8)
    if smooth_scale is not None:
        if smooth_scale.ndim != 3 or smooth_scale.shape[1:] != (N, H):
            print(f"[ERROR]dynamic_quant got wrong smooth_scale:{tuple(smooth_scale.shape)}")
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        scaled_inputs = inputs * smooth_scale
    else:
        scaled_inputs = inputs
    scale = amax / float(dtype_max)
    y = torch.round(scaled_inputs / scale)
    clip_min, clip_max = _dynamic_quant_clip_range(dtype_max, quant_dtype)
    y = torch.clamp(y, min=clip_min, max=clip_max).to(quant_dtype)
    return y, scale


def dequant(inputs, deq_scale_q_nope, quant_scale_ckv):
    """Dequantization for rotary output."""
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


# ===================== Scatter Functions =====================

def scatter_pa_nz(cache, inputs, index, data_size=16):
    """Scatter for PA_NZ cache mode."""
    bn = cache.shape[0]
    bs = cache.shape[1]
    n = cache.shape[2]
    h = cache.shape[3]
    bxs1 = inputs.shape[0]
    if len(cache.shape) != 4 or len(inputs.shape) != 2 or len(index.shape) != 1:
        print(f"[ERROR]scatter_pa_nz wrong dims: cache-{cache.shape}, inputs-{inputs.shape}, index-{index.shape}")
        return cache
    elif inputs.shape[1] != h:
        print(f"[ERROR]scatter_pa_nz wrong h: cache-{cache.shape}, inputs-{inputs.shape}")
        return cache
    elif index.shape[0] != bxs1:
        print(f"[ERROR]scatter_pa_nz wrong b*s1: input-{inputs.shape}, index-{index.shape}")
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
                cache[block_num_index, block_size_index, n_index, h_start_index:h_end_index] = \
                    inputs[input_index, input_start_index:input_end_index]
    return cache


def scatter_pa_blk_bsnd(cache, input_, index, seq_len, B):
    """Scatter for PA_BLK_BSND cache mode."""
    assert cache.ndim == 4, "cache must be (num_pa_blocks, blk_size, N, H)"
    num_pa_blocks, blk_size, N, H = cache.shape

    if isinstance(seq_len, (int, np.integer)):
        S1 = int(seq_len)
        assert input_.shape[0] == B * S1
        pages_per_b = math.ceil(S1 / blk_size)
        if index.ndim == 2:
            assert index.shape[0] == B and index.shape[1] == pages_per_b
            index = index.reshape(-1)
        else:
            assert index.ndim == 1 and index.shape[0] == B * pages_per_b
        seq_list = np.full((B,), S1, dtype=np.int64)
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
        assert index.ndim == 1 and index.shape[0] == expected_len

    flat_page_ptr = 0
    for b in range(B):
        len_b = int(seq_list[b])
        base_in = int(start_list[b])
        if len_b <= 0:
            continue
        pages_b = (len_b + blk_size - 1) // blk_size
        for p in range(pages_b):
            pa_blk_id = int(index[flat_page_ptr + p])
            assert 0 <= pa_blk_id < num_pa_blocks
            start = base_in + p * blk_size
            end = min(start + blk_size, base_in + len_b)
            sz = end - start
            if sz <= 0:
                continue
            cache[pa_blk_id, :sz, :, :H] = input_[start:end, :H][:, None, :]
        flat_page_ptr += pages_b
    assert flat_page_ptr == index.shape[0]
    return cache


def scatter_pa_blk_nz(cache, input_, index, seq_len, B, data_size=16):
    """Scatter for PA_BLK_NZ cache mode."""
    assert cache.ndim == 4
    assert input_.ndim == 2
    assert index.ndim == 2
    num_pa_blocks, blk_size, N, H_pad = cache.shape
    H = input_.shape[1]

    if isinstance(seq_len, (int, np.integer)):
        S1 = int(seq_len)
        assert input_.shape[0] == B * S1
        seq_list = np.full((B,), S1, dtype=np.int64)
        start_list = np.arange(B, dtype=np.int64) * S1
        expected_pages = math.ceil(S1 / blk_size)
        assert index.shape[1] == expected_pages
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
        assert input_.shape[0] == total_T
        max_s1 = int(np.max(seq_list))
        need_pages = math.ceil(max_s1 / blk_size)
        assert index.shape[1] >= need_pages

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
                in_h_start = data_idx * data_size
                in_h_end = min(in_h_start + data_size, H)
                out_h_end = h_start + (in_h_end - in_h_start)
                for n in range(N):
                    cache[pa_blk_id, block_size_index, n, h_start:out_h_end] = \
                        input_[inp_row, in_h_start:in_h_end]
    return cache


# ===================== Seed =====================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===================== GeneralizedPrologV3 =====================

class GeneralizedPrologV3:
    def __init__(self, params):
        self.batch_size, self.He, self.Hcq, self.Hckv, self.q_head_num, self.kv_head_num, \
        self.head_dim, self.rope_head_dim, self.q_seq, self.block_size, \
        self.input_layout, self.cache_mode, self.bs_fused_flag, self.cq_epsilon, self.ckv_epsilon, self.dtype, \
        self.weight_quant_mode, self.kv_quant_mode, self.query_quant_mode, \
        self.ckvkr_repo_mode, self.quant_scale_repo_mode, self.smooth_scales_cq_flag, self.query_norm_flag, \
        self.tile_size, self.qc_qr_scale, self.kc_scale = params

    def forward(self, inputs):
        """
        Generalized forward supporting all quant and cache modes.
        inputs: dict with keys for base tensors and optional quant tensors.
        """
        B = self.batch_size
        S1 = self.q_seq
        # kv_seq is not effective for this kernel path; cache sequence follows query sequence.
        S2 = S1
        D = self.head_dim
        Dr = self.rope_head_dim
        N1 = self.q_head_num
        N2 = self.kv_head_num
        He = self.He
        Hckv = self.Hckv
        Hcq = self.Hcq
        BlockSize = self.block_size
        T = B * S1

        weight_quant_mode = self.weight_quant_mode
        kv_quant_mode = self.kv_quant_mode
        query_quant_mode = self.query_quant_mode
        cache_mode = self.cache_mode
        query_norm_flag = int(self.query_norm_flag)
        smooth_scales_cq_flag = int(self.smooth_scales_cq_flag)
        cq_epsilon = self.cq_epsilon
        ckv_epsilon = self.ckv_epsilon
        qc_qr_scale = self.qc_qr_scale
        kc_scale = self.kc_scale

        full_quant_mode_config = _get_full_quant_mode_config(weight_quant_mode)
        if weight_quant_mode in (WEIGHT_QUANT_MODE_FULL_FP8_E4M3, WEIGHT_QUANT_MODE_FULL_HIF8) and \
                full_quant_mode_config is None:
            raise ValueError(f"weight_quant_mode={weight_quant_mode} requires unavailable quant dtype support")

        mode2_family_full_quant = weight_quant_mode in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8,
        )
        enable_quant_output = (query_quant_mode == 1 and weight_quant_mode in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_MXFP8_FULL,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8,
        ))
        query_quant_dtype = torch.int8
        query_dtype_max = INT8_DTYPE_MAX
        if mode2_family_full_quant and full_quant_mode_config is not None:
            query_dtype_max = full_quant_mode_config["dtype_max"]
            query_quant_dtype = full_quant_mode_config["query_quant_dtype"]
        if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
            if hasattr(torch, "float8_e4m3fn"):
                query_quant_dtype = torch.float8_e4m3fn
            query_dtype_max = FP8_E4M3_DTYPE_MAX
        deq_scale_q_nope = None

        def _build_expected_kernel_outputs(query, query_rope, deq_scale_q_nope_tensor,
                                            query_norm_tensor, deq_scale_q_norm_tensor):
            query_ret = query.to(torch.bfloat16) if not enable_quant_output else query
            query_rope_ret = query_rope.to(torch.bfloat16) if not enable_quant_output else query_rope
            if deq_scale_q_nope_tensor is None:
                deq_scale_q_nope_ret = torch.empty((0,), dtype=torch.float32)
            else:
                deq_scale_q_nope_ret = deq_scale_q_nope_tensor.to(torch.float32)
            if query_norm_tensor is None:
                query_norm_ret = torch.empty((0,), dtype=torch.float32)
            else:
                query_norm_ret = query_norm_tensor
            if deq_scale_q_norm_tensor is None:
                deq_scale_q_norm_ret = torch.empty((0,), dtype=torch.float32)
            else:
                deq_scale_q_norm_ret = deq_scale_q_norm_tensor
            return [query_ret, query_rope_ret, deq_scale_q_nope_ret, query_norm_ret, deq_scale_q_norm_ret]

        # Unpack base tensors (all moved to CPU)
        token_x = inputs['token_x'].cpu()
        w_dq = inputs['w_dq'].cpu()
        w_uq_qr = inputs['w_uq_qr'].cpu()
        w_uk = inputs['w_uk'].cpu()
        w_dkv_kr = inputs['w_dkv_kr'].cpu()
        gamma_cq = inputs['gamma_cq'].cpu()
        gamma_ckv = inputs['gamma_ckv'].cpu()
        sin = inputs['sin'].cpu()
        cos = inputs['cos'].cpu()
        index_table = inputs['index_table'].cpu()
        kv_cache = inputs['kv_cache'].cpu()
        kr_cache = inputs['kr_cache'].cpu()
        kv_tile_quant_dtype = _get_kv_tile_quant_dtype(
            kv_quant_mode, kv_cache_dtype=kv_cache.dtype, weight_quant_mode=weight_quant_mode
        )

        # Optional quant tensors
        deq_scale_x = inputs.get('deq_scale_x')
        deq_scale_w_dq = inputs.get('deq_scale_w_dq')
        deq_scale_w_uqqr = inputs.get('deq_scale_w_uqqr')
        deq_scale_w_dkvkr = inputs.get('deq_scale_w_dkvkr')
        quant_scale_ckv = inputs.get('quant_scale_ckv')
        quant_scale_ckr = inputs.get('quant_scale_ckr')
        smooth_scale_cq = inputs.get('smooth_scale_cq')
        actual_seq_len = inputs.get('actual_seq_len')
        k_nope_clip_alpha = inputs.get('k_nope_clip_alpha')
        if not smooth_scales_cq_flag:
            smooth_scale_cq = None
        if deq_scale_x is not None:
            deq_scale_x = deq_scale_x.cpu()
        if deq_scale_w_dq is not None:
            deq_scale_w_dq = deq_scale_w_dq.cpu()
        if deq_scale_w_uqqr is not None:
            deq_scale_w_uqqr = deq_scale_w_uqqr.cpu()
        if deq_scale_w_dkvkr is not None:
            deq_scale_w_dkvkr = deq_scale_w_dkvkr.cpu()
        if quant_scale_ckv is not None:
            quant_scale_ckv = quant_scale_ckv.cpu()
        if quant_scale_ckr is not None:
            quant_scale_ckr = quant_scale_ckr.cpu()
        if smooth_scale_cq is not None:
            smooth_scale_cq = smooth_scale_cq.cpu()
        if actual_seq_len is not None:
            actual_seq_len = actual_seq_len.cpu()
        if k_nope_clip_alpha is not None:
            k_nope_clip_alpha = k_nope_clip_alpha.cpu()

        # token_x can be BS-fused (T, He) or non-fused (B, S1, He)
        t_flag = token_x.ndim == 2
        if t_flag:
            T = token_x.shape[0]
        token_x = token_x.reshape(T, He)
        sin = sin.reshape(T, Dr)
        cos = cos.reshape(T, Dr)
        if cache_mode in ("PA_BLK_NZ", "PA_BLK_BSND"):
            # Keep original index rank for block modes.
            index_table = index_table
        else:
            index_table = index_table.reshape(-1) if index_table.numel() > 0 else index_table

        info_log("[INFO]========================================")
        info_log("[INFO]>>>>>>>>  Start to calculate  >>>>>>>>>>")
        info_log("[INFO]========================================")

        # -------------------------------------------------------------------
        # matmul1 : token_x(B*S1,He) * w_dq(He,Hcq) -> matmul1_res(B*S1,Hcq)
        # -------------------------------------------------------------------
        matmul1_dtype = torch.float32
        token_x_new = token_x.clone()
        if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
            if deq_scale_x is None or deq_scale_w_dq is None:
                raise ValueError("mxfp8 quant requires deq_scale_x and deq_scale_w_dq")
            token_x_new = _dequant_fp8_blockwise(token_x_new, deq_scale_x)
            w_dq = _dequant_fp8_weight(w_dq, deq_scale_w_dq)
        elif mode2_family_full_quant:
            if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_INT8:
                token_x_new = token_x_new.to(torch.int32)
                w_dq = w_dq.to(torch.int32)
                matmul1_dtype = torch.int32
            else:
                token_x_new = token_x_new.to(full_quant_mode_config["input_dtype"])
                w_dq = w_dq.to(full_quant_mode_config["input_dtype"])

        info_log(f"[INFO]matmul1 start. token_x:{tuple(token_x.shape)}|{token_x.dtype}"
                 f" w_dq:{tuple(w_dq.shape)}|{w_dq.dtype}")
        token_x_new = token_x_new.to(torch.float32)
        w_dq = w_dq.to(torch.float32)
        matmul1_res = torch.matmul(token_x_new, w_dq).to(matmul1_dtype)

        # matmul1 post-processing
        if mode2_family_full_quant:
            matmul1_res = matmul1_res.to(torch.float32)
            for t_index in range(T):
                matmul1_res[t_index, :] = matmul1_res[t_index, :] * deq_scale_x[t_index, 0]
            for h_index in range(Hcq):
                matmul1_res[:, h_index] = matmul1_res[:, h_index] * deq_scale_w_dq[0, h_index]
            info_log(f"[INFO]deq1 end. matmul1_res dtype={matmul1_res.dtype}")
        elif weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
            matmul1_res = matmul1_res.to(torch.float32)
        else:
            matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)
        info_log(f"[INFO]matmul1 end. matmul1_res:{tuple(matmul1_res.shape)}|{matmul1_res.dtype}")

        # ----------------------------------------------------------------------
        # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
        # ----------------------------------------------------------------------
        ep1 = float(cq_epsilon)
        norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
        norm1_res *= gamma_cq
        norm1_res *= qc_qr_scale
        info_log(f"[INFO]rmsnorm1 end. norm1_res:{tuple(norm1_res.shape)}|{norm1_res.dtype}")

        query_norm_tensor = None
        deq_scale_q_norm_tensor = None

        # ----------------------------------------------------------------------------------
        # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
        # ----------------------------------------------------------------------------------
        matmul2_dtype = torch.float32
        deq_scale_qcqr = None
        norm1_fp8 = None
        if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
            if deq_scale_w_uqqr is None:
                raise ValueError("mxfp8 quant requires deq_scale_w_uq_qr")
            norm1_fp8, deq_scale_qcqr = _fp8_blockwise_quant(
                norm1_res, block_size=FP8_BLOCK_SIZE, dtype_max=FP8_E4M3_DTYPE_MAX
            )
            norm1_res = _dequant_fp8_blockwise(norm1_fp8, deq_scale_qcqr, block_size=FP8_BLOCK_SIZE)
            w_uq_qr = _dequant_fp8_weight(w_uq_qr, deq_scale_w_uqqr, block_size=FP8_BLOCK_SIZE)
        elif weight_quant_mode == 1:
            w_uq_qr = w_uq_qr.to(torch.int32)
            matmul2_dtype = torch.int32
            norm1_res, deq_scale_qcqr = dynamic_quant(
                norm1_res, smooth_scale_cq, dtype_max=INT8_DTYPE_MAX, quant_dtype=torch.int32
            )
            info_log(f"[INFO]dynamic_quant end. norm1_res dtype={norm1_res.dtype}")
        elif mode2_family_full_quant:
            if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_INT8:
                w_uq_qr = w_uq_qr.to(torch.int32)
                matmul2_dtype = torch.int32
                norm1_res, deq_scale_qcqr = dynamic_quant(
                    norm1_res, smooth_scale_cq, dtype_max=INT8_DTYPE_MAX, quant_dtype=torch.int32
                )
            else:
                w_uq_qr = w_uq_qr.to(full_quant_mode_config["input_dtype"])
                norm1_res, deq_scale_qcqr = dynamic_quant(
                    norm1_res,
                    smooth_scale_cq,
                    dtype_max=full_quant_mode_config["dtype_max"],
                    quant_dtype=full_quant_mode_config["input_dtype"],
                )
            info_log(f"[INFO]dynamic_quant end. norm1_res dtype={norm1_res.dtype}")
        else:
            norm1_res = norm1_res.to(torch.bfloat16).to(torch.float32)

        if query_norm_flag:
            if weight_quant_mode == 0:
                query_norm_tensor = norm1_res
                deq_scale_q_norm_tensor = torch.empty((0,), dtype=torch.float32)
            elif weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
                query_norm_tensor = norm1_fp8
                deq_scale_q_norm_tensor = deq_scale_qcqr
                fp8_e8m0_dtype = _get_mxfp8_e8m0_dtype()
                if fp8_e8m0_dtype is not None:
                    deq_scale_q_norm_tensor = deq_scale_q_norm_tensor.to(fp8_e8m0_dtype)
            else:
                query_norm_tensor = norm1_res
                if deq_scale_qcqr is None:
                    deq_scale_q_norm_tensor = torch.empty((0,), dtype=torch.float32)
                else:
                    deq_scale_q_norm_tensor = deq_scale_qcqr
        else:
            if weight_quant_mode == 0:
                query_norm_tensor = torch.empty((0,), dtype=torch.float32)
                deq_scale_q_norm_tensor = torch.empty((0,), dtype=torch.float32)
            elif weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
                q_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float32
                scale_dtype = _get_mxfp8_e8m0_dtype() or torch.float32
                query_norm_tensor = torch.empty((0,), dtype=q_dtype)
                deq_scale_q_norm_tensor = torch.empty((0,), dtype=scale_dtype)
            else:
                if full_quant_mode_config is not None:
                    qnorm_dtype = full_quant_mode_config["input_dtype"]
                else:
                    qnorm_dtype = torch.int32
                query_norm_tensor = torch.empty((0,), dtype=qnorm_dtype)
                deq_scale_q_norm_tensor = torch.empty((0,), dtype=torch.float32)

        norm1_res = norm1_res.to(torch.float32)
        w_uq_qr = w_uq_qr.to(torch.float32)
        matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(matmul2_dtype)

        # matmul2 post-processing
        if weight_quant_mode == 1 or mode2_family_full_quant:
            matmul2_res = matmul2_res.to(torch.float32)
            for t_index in range(T):
                matmul2_res[t_index, :] = matmul2_res[t_index, :] * deq_scale_qcqr[t_index]
            for nddr_index in range(matmul2_res.shape[1]):
                matmul2_res[:, nddr_index] = matmul2_res[:, nddr_index] * deq_scale_w_uqqr[0, nddr_index]
            info_log(f"[INFO]deq2 end. matmul2_res dtype={matmul2_res.dtype}")
        elif weight_quant_mode != WEIGHT_QUANT_MODE_MXFP8_FULL:
            matmul2_res = matmul2_res.to(torch.bfloat16).to(torch.float32)
        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
        info_log(f"[INFO]matmul2 end. matmul2_res:{tuple(matmul2_res.shape)}|{matmul2_res.dtype}")

        if query_norm_tensor is not None and query_norm_tensor.numel() != 0 and not t_flag:
            query_norm_tensor = query_norm_tensor.reshape(B, S1, Hcq)

        # -------------------------------------------------------------------------------------
        # splitD1 : matmul2_res -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
        # -------------------------------------------------------------------------------------
        splitd1_res1 = matmul2_res[:, :, :D]
        splitd1_res2 = matmul2_res[:, :, D:]
        info_log(f"[INFO]splitD1 end. res1:{tuple(splitd1_res1.shape)} res2:{tuple(splitd1_res2.shape)}")

        # -------------------------------------------------------------------------
        # matmul3 : splitd1_res1(B*S1,N,D) * w_uk(N,D,Hckv) -> out1(B,S1,N,Hckv)
        # -------------------------------------------------------------------------
        splitd1_res1 = splitd1_res1.transpose(0, 1)
        out1 = torch.zeros((N1, T, Hckv))
        matmul3_dtype = torch.float32
        w_uk = w_uk.to(torch.float32)
        splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
        for n1_index in range(N1):
            out1[n1_index, :, :] = torch.matmul(
                splitd1_res1[n1_index, :, :], w_uk[n1_index, :, :]).to(matmul3_dtype)
        out1 = out1.transpose(0, 1)

        # matmul3 post-processing: DynamicQuant for query output
        if enable_quant_output:
            out_deqq_shape_shape = [T, N1, 1]  # per_token_head
            out1 = out1.to(torch.bfloat16).to(torch.float32)
            if mode2_family_full_quant:
                out1, deq_scale_q_nope = dynamic_quant_without_smooth_scale(
                    out1,
                    out_deqq_shape_shape,
                    dtype_max=query_dtype_max,
                    quant_dtype=query_quant_dtype
                )
            else:
                out1, deq_scale_q_nope = dynamic_quant_without_smooth_scale(out1, out_deqq_shape_shape)
        out1 = out1 if t_flag else out1.reshape(B, S1, N1, Hckv)
        info_log(f"[INFO]matmul3 end. {COLOR_YELLOW}out1:{out1.shape}|{out1.dtype}{YELLOW_RESET}")

        # -------------------------------------------------------------------------------------
        # rotary1 : splitd1_res2(B*S1,N,Dr) * cos/sin -> out2(B,S1,N,Dr)
        # -------------------------------------------------------------------------------------
        expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
        expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
        q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
        out2 = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
        if enable_quant_output:
            out2 = out2.to(torch.bfloat16).to(torch.float32)
            out2 = dequant(out2, deq_scale_q_nope, quant_scale_ckv)
        out2 = out2 if t_flag else out2.reshape(B, S1, N1, Dr)
        info_log(f"[INFO]rotary1 end. {COLOR_YELLOW}out2:{tuple(out2.shape)}|{out2.dtype}{YELLOW_RESET}")

        # -------------------------------------------------------------------------------
        # matmul4 : token_x(B*S1,He) * w_dkv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
        # -------------------------------------------------------------------------------
        matmul4_dtype = torch.float32
        if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
            if deq_scale_w_dkvkr is None:
                raise ValueError("mxfp8 quant requires deq_scale_w_dkv_kr")
            w_dkv_kr = _dequant_fp8_weight(w_dkv_kr, deq_scale_w_dkvkr)
        elif mode2_family_full_quant:
            if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_INT8:
                w_dkv_kr = w_dkv_kr.to(torch.int32)
                matmul4_dtype = torch.int32
            else:
                w_dkv_kr = w_dkv_kr.to(full_quant_mode_config["input_dtype"])

        token_x_matmul4 = token_x_new.to(torch.float32)
        w_dkv_kr = w_dkv_kr.to(torch.float32)
        matmul4_res = torch.matmul(token_x_matmul4, w_dkv_kr).to(matmul4_dtype)

        # matmul4 post-processing
        if mode2_family_full_quant:
            matmul4_res = matmul4_res.to(torch.float32)
            for t_index in range(T):
                matmul4_res[t_index, :] = matmul4_res[t_index, :] * deq_scale_x[t_index, 0]
            for h_index in range(Hckv + Dr):
                matmul4_res[:, h_index] = matmul4_res[:, h_index] * deq_scale_w_dkvkr[0, h_index]
            info_log(f"[INFO]deq3 end. matmul4_res dtype={matmul4_res.dtype}")
        elif weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
            matmul4_res = matmul4_res.to(torch.float32)
        else:
            matmul4_res = matmul4_res.to(torch.bfloat16).to(torch.float32)
        info_log(f"[INFO]matmul4 end. matmul4_res:{tuple(matmul4_res.shape)}|{matmul4_res.dtype}")

        # -------------------------------------------------------------------------------------
        # splitD2 : matmul4_res -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
        # -------------------------------------------------------------------------------------
        splitd2_res1 = matmul4_res[:, :Hckv]
        splitd2_res2 = matmul4_res[:, Hckv:]
        info_log(f"[INFO]splitD2 end. res1:{tuple(splitd2_res1.shape)} res2:{tuple(splitd2_res2.shape)}")

        # ----------------------------------------------------------------------------
        # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
        # ----------------------------------------------------------------------------
        ep2 = float(ckv_epsilon)
        norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
        norm2_res *= gamma_ckv
        info_log(f"[INFO]rmsnorm2 end. norm2_res:{tuple(norm2_res.shape)}|{norm2_res.dtype}")

        Dtile = Hckv
        # rmsnorm2 post-processing: kv cache quantization
        if kv_quant_mode in (1, 2):
            if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
                norm2_res_np = quant_ckv_per_tensor(norm2_res.numpy(), quant_scale_ckv.numpy())
                norm2_res = torch.from_numpy(np.asarray(norm2_res_np, dtype=np.float32))
            elif weight_quant_mode in (
                    WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
                    WEIGHT_QUANT_MODE_FULL_HIF8):
                norm2_res = quant_with_scale(
                    norm2_res,
                    quant_scale_ckv,
                    quant_dtype=full_quant_mode_config["input_dtype"],
                )
            else:
                norm2_res = quant(norm2_res, quant_scale_ckv)
            info_log(f"[INFO]quant1 end. norm2_res dtype={norm2_res.dtype}")
        elif kv_quant_mode == 3:
            tile_size = self.tile_size
            norm2_res = norm2_res.reshape(T, Hckv // tile_size, tile_size)
            eps = 1e-8
            amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
            if k_nope_clip_alpha is None:
                k_nope_clip_alpha = torch.ones((1,), dtype=torch.float32)
            amax = torch.clamp(amax, min=eps) * k_nope_clip_alpha
            clip_res = torch.clamp(norm2_res, min=-amax, max=amax)
            norm2_res, deq_scale_ckv = dynamic_quant_ckv_with_amax(
                clip_res, amax, quant_dtype=kv_tile_quant_dtype
            )
            deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
            norm2_res = norm2_res.reshape(T, Hckv)
            if self.ckvkr_repo_mode == 1:
                # Compute rotary2 early for merged storage
                cos_flat = cos.reshape(T, Dr)
                sin_flat = sin.reshape(T, Dr)
                k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
                rotary2_res_early = (k * cos_flat) + (rotate_half(k) * sin_flat)
                rotary2_bf16 = rotary2_res_early.to(torch.bfloat16)
                rotary2_packed = rotary2_bf16.contiguous().view(kv_tile_quant_dtype)
                norm2_res = torch.cat((norm2_res, rotary2_packed), axis=-1)
                Dtile = Dtile + Dr * 2
            if self.quant_scale_repo_mode == 1:
                deq_scale_ckv_packed = deq_scale_ckv.contiguous().view(kv_tile_quant_dtype)
                norm2_res = torch.cat((norm2_res, deq_scale_ckv_packed), axis=-1)
                Dtile = Dtile + Hckv // tile_size * 4

        # -------------------------------------------------------------------------------------
        # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
        # -------------------------------------------------------------------------------------
        cos_flat = cos.reshape(T, Dr)
        sin_flat = sin.reshape(T, Dr)
        k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
        rotary2_res = (k * cos_flat) + (rotate_half(k) * sin_flat)
        info_log(f"[INFO]rotary2 end. rotary2_res:{tuple(rotary2_res.shape)}|{rotary2_res.dtype}")
        # rotary2 post-processing: quantize if weight_quant_mode==1 and kv_quant_mode==2
        if weight_quant_mode == 1 and kv_quant_mode == 2:
            rotary2_res = quant(rotary2_res, quant_scale_ckr)
            info_log(f"[INFO]quant2 end. rotary2_res dtype={rotary2_res.dtype}")

        # Determine scatter dtype info
        pa_flag = cache_mode.startswith("PA")
        if kv_quant_mode in (1, 2, 3):
            kv_scatter_size = 32
        else:
            kv_scatter_size = 16
        if weight_quant_mode == 1 and kv_quant_mode == 2:
            kr_scatter_size = 32
        else:
            kr_scatter_size = 16

        # -------------------------------------------------------------------------------------------------------
        # scatter1 : norm2_res(B*S1,Dtile) -> kv_cache
        # -------------------------------------------------------------------------------------------------------
        kv_cache = copy.deepcopy(kv_cache)
        if kv_cache.numel() == 0:
            out3 = kv_cache
            out4 = copy.deepcopy(kr_cache)
            return {
                "outputs": _build_expected_kernel_outputs(
                    out1, out2, deq_scale_q_nope, query_norm_tensor, deq_scale_q_norm_tensor
                ),
                "inplace": [out3, out4]
            }

        out3_shape = kv_cache.shape
        if cache_mode == "BSND":
            expected = (B, S2, N2, Dtile)
            if tuple(out3_shape) != expected:
                raise ValueError(f"BSND kv_cache shape should be {expected}, got {tuple(out3_shape)}")
        elif cache_mode == "TND":
            expected = (B * S2, N2, Dtile)
            if tuple(out3_shape) != expected:
                raise ValueError(f"TND kv_cache shape should be {expected}, got {tuple(out3_shape)}")
        if kv_quant_mode == 0:
            kv_cache = kv_cache.to(torch.bfloat16)
            norm2_res_scatter = norm2_res.to(torch.bfloat16)
        else:
            norm2_res_scatter = norm2_res

        info_log(f"[INFO]scatter1 start. norm2_res:{tuple(norm2_res_scatter.shape)}|{norm2_res_scatter.dtype}"
                 f" kv_cache:{tuple(kv_cache.shape)}|{kv_cache.dtype}")

        if cache_mode == "PA_BLK_NZ":
            kv_cache = scatter_pa_blk_nz(kv_cache, norm2_res_scatter, index_table, S1, B, kv_scatter_size)
        elif cache_mode == "PA_BLK_BSND":
            kv_cache = scatter_pa_blk_bsnd(kv_cache, norm2_res_scatter, index_table, S1, B)
        elif cache_mode == "PA_NZ":
            kv_cache = scatter_pa_nz(kv_cache, norm2_res_scatter, index_table.reshape(T), kv_scatter_size)
        elif cache_mode == "PA_BSND":
            B_num = out3_shape[0]
            kv_cache = kv_cache.reshape(B_num * BlockSize, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    kv_cache[index_table.reshape(T)[i], j, :] = norm2_res_scatter[i, :]
        elif cache_mode == "TND":
            kv_cache = kv_cache.reshape(-1, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    kv_cache[i, j, :] = norm2_res_scatter[i, :]
        else:
            # BSND
            kv_cache = kv_cache.reshape(B * S2, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    kv_cache[i, j, :] = norm2_res_scatter[i, :]
        out3 = kv_cache.reshape(out3_shape)
        info_log(f"[INFO]scatter1 end. {COLOR_YELLOW}out3:{tuple(out3.shape)}|{out3.dtype}{YELLOW_RESET}")

        # ----------------------------------------------------------------------------------------------
        # scatter2 : rotary2_res(B*S1,Dr) -> kr_cache
        # ----------------------------------------------------------------------------------------------
        if self.ckvkr_repo_mode == 1:
            out4 = copy.deepcopy(kr_cache)
        else:
            kr_cache = copy.deepcopy(kr_cache)
            out4_shape = kr_cache.shape
            if cache_mode == "BSND":
                expected = (B, S2, N2, Dr)
                if tuple(out4_shape) != expected:
                    raise ValueError(f"BSND kr_cache shape should be {expected}, got {tuple(out4_shape)}")
            elif cache_mode == "TND":
                expected = (B * S2, N2, Dr)
                if tuple(out4_shape) != expected:
                    raise ValueError(f"TND kr_cache shape should be {expected}, got {tuple(out4_shape)}")
            if weight_quant_mode == 1 and kv_quant_mode == 2:
                rotary2_scatter = rotary2_res
            else:
                kr_cache = kr_cache.to(torch.bfloat16)
                rotary2_scatter = rotary2_res.to(torch.bfloat16)

            info_log(f"[INFO]scatter2 start. rotary2:{tuple(rotary2_scatter.shape)}|{rotary2_scatter.dtype}"
                     f" kr_cache:{tuple(kr_cache.shape)}|{kr_cache.dtype}")

            if cache_mode == "PA_BLK_NZ":
                kr_cache = scatter_pa_blk_nz(kr_cache, rotary2_scatter, index_table, S1, B, kr_scatter_size)
            elif cache_mode == "PA_BLK_BSND":
                kr_cache = scatter_pa_blk_bsnd(kr_cache, rotary2_scatter, index_table, S1, B)
            elif cache_mode == "PA_NZ":
                kr_cache = scatter_pa_nz(kr_cache, rotary2_scatter, index_table.reshape(T), kr_scatter_size)
            elif cache_mode == "PA_BSND":
                B_num = out4_shape[0]
                kr_cache = kr_cache.reshape(B_num * BlockSize, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[index_table.reshape(T)[i], j, :] = rotary2_scatter[i, :]
            elif cache_mode == "TND":
                kr_cache = kr_cache.reshape(-1, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[i, j, :] = rotary2_scatter[i, :]
            else:
                # BSND
                kr_cache = kr_cache.reshape(B * S2, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[i, j, :] = rotary2_scatter[i, :]
            out4 = kr_cache.reshape(out4_shape)
            info_log(f"[INFO]scatter2 end. {COLOR_YELLOW}out4:{tuple(out4.shape)}|{out4.dtype}{YELLOW_RESET}")

        info_log("[INFO]========================================")
        info_log("[INFO]>>>>>>>>   Calculate success  >>>>>>>>>>")
        info_log("[INFO]========================================")

        return {
            "outputs": _build_expected_kernel_outputs(
                out1, out2, deq_scale_q_nope, query_norm_tensor, deq_scale_q_norm_tensor
            ),
            "inplace": [out3, out4]
        }


def _get_mxfp8_e8m0_dtype():
    for name in ("float8_e8m0fnu", "float8_e8m0fnuz", "float8_e8m0"):
        if hasattr(torch, name):
            return getattr(torch, name)
    return None


def _get_device_name():
    if torch_npu is None:
        return ""
    try:
        if hasattr(torch_npu.npu, "current_device"):
            dev_id = torch_npu.npu.current_device()
        else:
            dev_id = 0
        if hasattr(torch_npu.npu, "get_device_name"):
            return str(torch_npu.npu.get_device_name(dev_id))
    except Exception:
        return ""
    return ""


def is_mxfp8_runtime_supported():
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    if _get_mxfp8_e8m0_dtype() is None:
        return False
    try:
        import ml_dtypes  # noqa: F401
    except Exception:
        return False
    device_name = _get_device_name().lower()
    return "950" in device_name


def validate_quant_cache_combo(cache_mode,
                               weight_quant_mode,
                               kv_quant_mode,
                               query_quant_mode,
                               ckvkr_repo_mode,
                               quant_scale_repo_mode):
    if weight_quant_mode not in (0, 1, 2, 3, 4, 5):
        return False, "weight_quant_mode should be in {0,1,2,3,4,5}"
    if weight_quant_mode == 0 and kv_quant_mode != 0:
        return False, "weight_quant_mode=0 only supports kv_quant_mode=0"
    if weight_quant_mode == 1 and kv_quant_mode not in (0, 2, 3):
        return False, "weight_quant_mode=1 only supports kv_quant_mode in {0,2,3}"
    if weight_quant_mode in (2, 3, 4, 5) and kv_quant_mode not in (0, 1, 3):
        return False, "weight_quant_mode in {2,3,4,5} only supports kv_quant_mode in {0,1,3}"

    if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL and not is_mxfp8_runtime_supported():
        return False, "mxfp8 full quant needs float8 support on Ascend 950"
    if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_FP8_E4M3 and not hasattr(torch, "float8_e4m3fn"):
        return False, "fp8_e4m3 full quant needs torch.float8_e4m3fn support"
    if weight_quant_mode == WEIGHT_QUANT_MODE_FULL_HIF8 and not has_hif8_dtype():
        return False, "hif8 full quant needs torch_npu.hifloat8 support"

    if kv_quant_mode == 3:
        if cache_mode in ("PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ"):
            return False, f"cache_mode={cache_mode} does not support per-tile kv quant"
        if ckvkr_repo_mode != 1 or quant_scale_repo_mode != 1:
            return False, "per-tile kv quant requires ckvkr_repo_mode=1 and quant_scale_repo_mode=1"
    else:
        if ckvkr_repo_mode != 0 or quant_scale_repo_mode != 0:
            return False, "non per-tile kv quant requires ckvkr_repo_mode=0 and quant_scale_repo_mode=0"

    expect_query_quant = int(weight_quant_mode in (2, 3, 4, 5) and kv_quant_mode == 1)
    if query_quant_mode != expect_query_quant:
        return False, f"query_quant_mode should be {expect_query_quant} for this quant scenario"

    return True, ""


def _rand_tensor(shape, dtype, generator):
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=torch.int8, generator=generator)
    hif8_dtype = _get_hif8_dtype()
    if hif8_dtype is not None and dtype == hif8_dtype:
        return (torch.rand(shape, dtype=torch.float32, generator=generator) * 2 - 1).to(dtype)
    if dtype in tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
                      if hasattr(torch, n)):
        return (torch.rand(shape, dtype=torch.float32, generator=generator) * 2 - 1).to(dtype)
    return torch.rand(shape, dtype=torch.float32, generator=generator).to(dtype)


def _rand_scale(shape, generator, min_val=0.01, max_val=1.0):
    return torch.rand(shape, dtype=torch.float32, generator=generator) * (max_val - min_val) + min_val


def _build_cache_index(cache_mode, B, S1, S2, block_size, block_num, t_flag, generator):
    if cache_mode in ("BSND", "TND"):
        return torch.empty((0,), dtype=torch.int64), None

    T = B * S1
    if cache_mode in ("PA_BSND", "PA_NZ"):
        max_index = block_num * block_size
        index = torch.randperm(max_index, generator=generator).to(torch.int64)[:T]
        if t_flag:
            return index, None
        return index.reshape(B, S1), None

    pages_per_b = math.ceil(S1 / block_size)
    total_pages = B * pages_per_b
    page_ids = torch.randperm(block_num, generator=generator).to(torch.int64)
    if page_ids.numel() < total_pages:
        repeat_times = math.ceil(total_pages / page_ids.numel())
        page_ids = page_ids.repeat(repeat_times)
    page_ids = page_ids[:total_pages]

    if t_flag:
        actual_seq_len = torch.arange(1, B + 1, dtype=torch.int32) * S1
        return page_ids, actual_seq_len
    return page_ids.reshape(B, pages_per_b), None


def _cache_shape(cache_mode, B, S2, N2, last_dim, block_size):
    if cache_mode.startswith("PA"):
        block_num = math.ceil(B * S2 / block_size)
        return (block_num, block_size, N2, last_dim)
    if cache_mode == "BSND":
        return (B, S2, N2, last_dim)
    return (B * S2, N2, last_dim)  # TND


PROLOGV3_PARAM_NAMES = (
    "batch_size", "He", "Hcq", "Hckv", "q_head_num", "kv_head_num", "head_dim", "rope_head_dim",
    "q_seq", "block_size", "input_layout", "cache_mode", "bs_fused_flag", "cq_epsilon", "ckv_epsilon",
    "dtype", "weight_quant_mode", "kv_quant_mode", "query_quant_mode", "ckvkr_repo_mode",
    "quant_scale_repo_mode", "smooth_scales_cq_flag", "query_norm_flag", "tile_size", "qc_qr_scale",
    "kc_scale"
)


def _params_to_named_dict(params) -> Dict[str, object]:
    return dict(zip(PROLOGV3_PARAM_NAMES, params))


def _normalize_override_dtype(dtype):
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and dtype.startswith("torch."):
        dtype_name = dtype.split(".", 1)[1]
        if not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported torch dtype override: {dtype}")
        return getattr(torch, dtype_name)
    raise TypeError(f"unsupported dtype override: {dtype}")


def _rand_tensor_for_override(shape, dtype, generator):
    shape = tuple(int(dim) for dim in shape)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, generator=generator)
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.randint(-16, 16, shape, dtype=dtype, generator=generator)
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return torch.randn(shape, dtype=torch.float32, generator=generator).clamp(-4.0, 4.0).to(dtype)
    hif8_dtype = _get_hif8_dtype()
    if hif8_dtype is not None and dtype == hif8_dtype:
        return torch.randn(shape, dtype=torch.float32, generator=generator).clamp(-4.0, 4.0).to(dtype)
    return torch.randn(shape, dtype=torch.float32, generator=generator).to(dtype)


def _move_tensor_to_runtime_device(tensor, runtime_device):
    if runtime_device == "cpu":
        return tensor.cpu()
    if runtime_device == "npu":
        if torch_npu is None:
            raise RuntimeError("runtime_device='npu' requires torch_npu")
        return tensor.npu()
    raise ValueError(f"unsupported runtime_device={runtime_device}")


def _make_override_tensor(name, reference_tensor, override_spec, generator, runtime_device):
    if not override_spec.get("present", True):
        return None

    base_dtype = reference_tensor.dtype if reference_tensor is not None else None
    dtype = _normalize_override_dtype(override_spec.get("dtype", base_dtype))
    if dtype is None:
        raise ValueError(f"override for {name} requires explicit dtype when no reference tensor exists")

    if "shape" in override_spec:
        shape = tuple(int(dim) for dim in override_spec["shape"])
    elif reference_tensor is not None:
        shape = tuple(reference_tensor.shape)
    else:
        raise ValueError(f"override for {name} requires explicit shape when no reference tensor exists")

    if name == "cache_index":
        tensor = torch.zeros(shape, dtype=dtype)
    elif name == "actual_seq_len" and len(shape) == 1:
        tensor = torch.arange(1, shape[0] + 1, dtype=dtype)
    elif 0 in shape:
        tensor = torch.empty(shape, dtype=dtype)
    else:
        tensor = _rand_tensor_for_override(shape, dtype, generator)

    return _move_tensor_to_runtime_device(tensor, runtime_device)


def _build_op_attrs(named_params: Dict[str, object]) -> Dict[str, object]:
    return {
        "rmsnorm_epsilon_cq": named_params["cq_epsilon"],
        "rmsnorm_epsilon_ckv": named_params["ckv_epsilon"],
        "cache_mode": named_params["cache_mode"],
        "query_norm_flag": bool(named_params["query_norm_flag"]),
        "weight_quant_mode": named_params["weight_quant_mode"],
        "kv_cache_quant_mode": named_params["kv_quant_mode"],
        "query_quant_mode": named_params["query_quant_mode"],
        "ckvkr_repo_mode": named_params["ckvkr_repo_mode"],
        "quant_scale_repo_mode": named_params["quant_scale_repo_mode"],
        "tile_size": named_params["tile_size"],
        "qc_qr_scale": named_params["qc_qr_scale"],
        "kc_scale": named_params["kc_scale"],
    }


def _build_forward_inputs(runtime_inputs: Dict[str, object]) -> Dict[str, object]:
    return {
        "token_x": runtime_inputs["token_x"],
        "w_dq": runtime_inputs["w_dq"],
        "w_uq_qr": runtime_inputs["w_uq_qr"],
        "w_uk": runtime_inputs["w_uk"],
        "w_dkv_kr": runtime_inputs["w_dkv_kr"],
        "gamma_cq": runtime_inputs["rmsnorm_gamma_cq"],
        "gamma_ckv": runtime_inputs["rmsnorm_gamma_ckv"],
        "sin": runtime_inputs["rope_sin"],
        "cos": runtime_inputs["rope_cos"],
        "index_table": runtime_inputs["cache_index"],
        "kv_cache": runtime_inputs["kv_cache"],
        "kr_cache": runtime_inputs["kr_cache"],
        "deq_scale_x": runtime_inputs["dequant_scale_x"],
        "deq_scale_w_dq": runtime_inputs["dequant_scale_w_dq"],
        "deq_scale_w_uqqr": runtime_inputs["dequant_scale_w_uq_qr"],
        "deq_scale_w_dkvkr": runtime_inputs["dequant_scale_w_dkv_kr"],
        "quant_scale_ckv": runtime_inputs["quant_scale_ckv"],
        "quant_scale_ckr": runtime_inputs["quant_scale_ckr"],
        "smooth_scale_cq": runtime_inputs["smooth_scales_cq"],
        "actual_seq_len": runtime_inputs["actual_seq_len"],
        "k_nope_clip_alpha": runtime_inputs["k_nope_clip_alpha"],
    }


def _execute_npu_case(runtime_inputs: Dict[str, object], op_attrs: Dict[str, object]):
    if torch_npu is None:
        raise RuntimeError("NPU execution requires torch_npu")
    w_dq_cast = torch_npu.npu_format_cast(runtime_inputs["w_dq"].contiguous(), 29)
    w_uq_qr_cast = torch_npu.npu_format_cast(runtime_inputs["w_uq_qr"].contiguous(), 29)
    w_dkv_kr_cast = torch_npu.npu_format_cast(runtime_inputs["w_dkv_kr"].contiguous(), 29)

    op_kwargs = dict(op_attrs)
    if str(op_attrs["cache_mode"]).startswith("PA") and runtime_inputs["cache_index"] is not None:
        op_kwargs["cache_index"] = runtime_inputs["cache_index"]
    if runtime_inputs["dequant_scale_x"] is not None:
        op_kwargs["dequant_scale_x"] = runtime_inputs["dequant_scale_x"]
    if runtime_inputs["dequant_scale_w_dq"] is not None:
        op_kwargs["dequant_scale_w_dq"] = runtime_inputs["dequant_scale_w_dq"]
    if runtime_inputs["dequant_scale_w_uq_qr"] is not None:
        op_kwargs["dequant_scale_w_uq_qr"] = runtime_inputs["dequant_scale_w_uq_qr"]
    if runtime_inputs["dequant_scale_w_dkv_kr"] is not None:
        op_kwargs["dequant_scale_w_dkv_kr"] = runtime_inputs["dequant_scale_w_dkv_kr"]
    if runtime_inputs["quant_scale_ckv"] is not None:
        op_kwargs["quant_scale_ckv"] = runtime_inputs["quant_scale_ckv"]
    if runtime_inputs["quant_scale_ckr"] is not None:
        op_kwargs["quant_scale_ckr"] = runtime_inputs["quant_scale_ckr"]
    if runtime_inputs["smooth_scales_cq"] is not None:
        op_kwargs["smooth_scales_cq"] = runtime_inputs["smooth_scales_cq"]
    if runtime_inputs["actual_seq_len"] is not None:
        op_kwargs["actual_seq_len"] = runtime_inputs["actual_seq_len"]
    if runtime_inputs["k_nope_clip_alpha"] is not None:
        op_kwargs["k_nope_clip_alpha"] = runtime_inputs["k_nope_clip_alpha"]

    result = torch_npu.npu_mla_prolog_v3(
        runtime_inputs["token_x"], w_dq_cast, w_uq_qr_cast,
        runtime_inputs["w_uk"], w_dkv_kr_cast,
        runtime_inputs["rmsnorm_gamma_cq"], runtime_inputs["rmsnorm_gamma_ckv"],
        runtime_inputs["rope_sin"], runtime_inputs["rope_cos"],
        runtime_inputs["kv_cache"], runtime_inputs["kr_cache"],
        **op_kwargs
    )
    torch.npu.synchronize()

    if isinstance(result, torch.Tensor):
        kernel_outputs = [result]
    elif isinstance(result, (list, tuple)):
        kernel_outputs = list(result)
    else:
        raise RuntimeError(f"unsupported npu_mla_prolog_v3 output type: {type(result)}")

    if len(kernel_outputs) == 5:
        kernel_outputs_aligned = kernel_outputs
    elif len(kernel_outputs) >= 7:
        kernel_outputs_aligned = [
            kernel_outputs[0],
            kernel_outputs[1],
            kernel_outputs[4],
            kernel_outputs[5],
            kernel_outputs[6],
        ]
    else:
        raise RuntimeError(
            f"unexpected npu_mla_prolog_v3 output count: {len(kernel_outputs)} (expect 5 or >=7)"
        )

    return {
        "outputs": kernel_outputs_aligned,
        "inplace": [runtime_inputs["kv_cache"], runtime_inputs["kr_cache"]],
    }


def _apply_case_overrides(case_payload, attr_overrides=None, input_overrides=None):
    runtime_inputs = dict(case_payload["runtime_inputs"])
    op_attrs = dict(case_payload["op_attrs"])
    if attr_overrides:
        op_attrs.update(attr_overrides)

    if input_overrides:
        override_generator = torch.Generator().manual_seed(int(case_payload["seed"]) + 97)
        for name, spec in input_overrides.items():
            if name not in runtime_inputs:
                raise KeyError(f"unsupported input override: {name}")
            runtime_inputs[name] = _make_override_tensor(
                name,
                runtime_inputs.get(name),
                spec,
                override_generator,
                runtime_device=case_payload["runtime_device"],
            )

    updated_payload = dict(case_payload)
    updated_payload["runtime_inputs"] = runtime_inputs
    updated_payload["op_attrs"] = op_attrs
    return updated_payload


def _build_default_case_payload(params, validate_quant_combo=True, runtime_device="npu"):
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
    q_seq, block_size, input_layout, cache_mode, bs_fused_flag, cq_epsilon, ckv_epsilon, dtype, \
    weight_quant_mode, kv_quant_mode, query_quant_mode, ckvkr_repo_mode, \
    quant_scale_repo_mode, smooth_scales_cq_flag, query_norm_flag, tile_size, qc_qr_scale, kc_scale = params

    if validate_quant_combo:
        is_valid, reason = validate_quant_cache_combo(cache_mode,
                                                      weight_quant_mode,
                                                      kv_quant_mode,
                                                      query_quant_mode,
                                                      ckvkr_repo_mode,
                                                      quant_scale_repo_mode)
        if not is_valid:
            pytest.skip(f"skip invalid quant/cache combo: {reason}")

    seed = 3
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    B = batch_size
    S1 = q_seq
    S2 = S1
    N1 = q_head_num
    N2 = kv_head_num
    D = head_dim
    Dr = rope_head_dim
    T = B * S1
    t_flag = bool(bs_fused_flag) or cache_mode == "TND"
    fp8_e8m0_dtype = _get_mxfp8_e8m0_dtype()

    # Dtypes by scenario
    if weight_quant_mode == 0:
        token_dtype = torch.bfloat16
        w_dq_dtype = torch.bfloat16
        w_uq_qr_dtype = torch.bfloat16
        w_dkv_kr_dtype = torch.bfloat16
    elif weight_quant_mode == 1:
        token_dtype = torch.bfloat16
        w_dq_dtype = torch.bfloat16
        w_uq_qr_dtype = torch.int8
        w_dkv_kr_dtype = torch.bfloat16
    elif weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
        token_dtype = torch.float8_e4m3fn
        w_dq_dtype = torch.float8_e4m3fn
        w_uq_qr_dtype = torch.float8_e4m3fn
        w_dkv_kr_dtype = torch.float8_e4m3fn
    elif weight_quant_mode in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8):
        full_quant_mode_config = _get_full_quant_mode_config(weight_quant_mode)
        if full_quant_mode_config is None:
            pytest.skip(f"skip weight_quant_mode={weight_quant_mode} due to unavailable quant dtype")
        token_dtype = full_quant_mode_config["input_dtype"]
        w_dq_dtype = full_quant_mode_config["input_dtype"]
        w_uq_qr_dtype = full_quant_mode_config["input_dtype"]
        w_dkv_kr_dtype = full_quant_mode_config["input_dtype"]
    else:
        pytest.skip(f"skip unsupported weight_quant_mode={weight_quant_mode}")

    if kv_quant_mode == 0:
        kv_cache_dtype = torch.bfloat16
    elif kv_quant_mode == 1:
        if weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
            kv_cache_dtype = torch.float8_e4m3fn
        elif weight_quant_mode in (
                WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
                WEIGHT_QUANT_MODE_FULL_HIF8):
            kv_cache_dtype = full_quant_mode_config["input_dtype"]
        else:
            kv_cache_dtype = torch.int8
    elif kv_quant_mode == 2:
        kv_cache_dtype = torch.int8
    else:
        kv_cache_dtype = _get_kv_tile_quant_dtype(kv_quant_mode, weight_quant_mode=weight_quant_mode)

    if ckvkr_repo_mode == 1:
        kr_cache_dtype = None
    elif weight_quant_mode == 1 and kv_quant_mode == 2:
        kr_cache_dtype = torch.int8
    else:
        kr_cache_dtype = torch.bfloat16

    token_shape = (T, He) if t_flag else (B, S1, He)
    rope_shape = (T, Dr) if t_flag else (B, S1, Dr)

    Dtile = Hckv
    if kv_quant_mode == 3:
        if ckvkr_repo_mode == 1:
            Dtile += Dr * 2
        if quant_scale_repo_mode == 1:
            Dtile += Hckv // tile_size * 4

    block_num = math.ceil(B * S2 / block_size)
    cache_index, actual_seq_len = _build_cache_index(cache_mode, B, S1, S2, block_size, block_num, t_flag, generator)

    kv_cache_shape = _cache_shape(cache_mode, B, S2, N2, Dtile, block_size)
    if ckvkr_repo_mode == 1:
        kr_cache_shape = (0,)
    else:
        kr_cache_shape = _cache_shape(cache_mode, B, S2, N2, Dr, block_size)

    token_x = _move_tensor_to_runtime_device(_rand_tensor(token_shape, token_dtype, generator), runtime_device)
    w_dq = _move_tensor_to_runtime_device(_rand_tensor((He, Hcq), w_dq_dtype, generator), runtime_device)
    w_uq_qr = _move_tensor_to_runtime_device(
        _rand_tensor((Hcq, N1 * (D + Dr)), w_uq_qr_dtype, generator),
        runtime_device,
    )
    w_uk = _move_tensor_to_runtime_device(_rand_tensor((N1, D, Hckv), torch.bfloat16, generator), runtime_device)
    w_dkv_kr = _move_tensor_to_runtime_device(
        _rand_tensor((He, Hckv + Dr), w_dkv_kr_dtype, generator),
        runtime_device,
    )
    rmsnorm_gamma_cq = _move_tensor_to_runtime_device(_rand_tensor((Hcq,), torch.bfloat16, generator), runtime_device)
    rmsnorm_gamma_ckv = _move_tensor_to_runtime_device(_rand_tensor((Hckv,), torch.bfloat16, generator), runtime_device)
    rope_sin = _move_tensor_to_runtime_device(_rand_tensor(rope_shape, torch.bfloat16, generator), runtime_device)
    rope_cos = _move_tensor_to_runtime_device(_rand_tensor(rope_shape, torch.bfloat16, generator), runtime_device)

    kv_cache = _move_tensor_to_runtime_device(_rand_tensor(kv_cache_shape, kv_cache_dtype, generator), runtime_device)
    if kr_cache_dtype is None:
        kr_cache = _move_tensor_to_runtime_device(torch.empty(kr_cache_shape, dtype=torch.bfloat16), runtime_device)
    else:
        kr_cache = _move_tensor_to_runtime_device(_rand_tensor(kr_cache_shape, kr_cache_dtype, generator), runtime_device)
    cache_index = _move_tensor_to_runtime_device(cache_index, runtime_device)
    if actual_seq_len is not None:
        actual_seq_len = _move_tensor_to_runtime_device(actual_seq_len, runtime_device)

    deq_scale_x = None
    deq_scale_w_dq = None
    deq_scale_w_uq_qr = None
    deq_scale_w_dkv_kr = None
    quant_scale_ckv = None
    quant_scale_ckr = None
    smooth_scale_cq = None
    k_nope_clip_alpha = None

    if weight_quant_mode == 1:
        deq_scale_w_uq_qr = _move_tensor_to_runtime_device(
            _rand_scale((1, N1 * (D + Dr)), generator), runtime_device
        )
        if smooth_scales_cq_flag:
            smooth_scale_cq = _move_tensor_to_runtime_device(_rand_scale((1, Hcq), generator), runtime_device)
    elif weight_quant_mode in (
            WEIGHT_QUANT_MODE_FULL_INT8,
            WEIGHT_QUANT_MODE_FULL_FP8_E4M3,
            WEIGHT_QUANT_MODE_FULL_HIF8):
        deq_scale_x = _move_tensor_to_runtime_device(_rand_scale((T, 1), generator), runtime_device)
        deq_scale_w_dq = _move_tensor_to_runtime_device(_rand_scale((1, Hcq), generator), runtime_device)
        deq_scale_w_uq_qr = _move_tensor_to_runtime_device(
            _rand_scale((1, N1 * (D + Dr)), generator),
            runtime_device,
        )
        deq_scale_w_dkv_kr = _move_tensor_to_runtime_device(
            _rand_scale((1, Hckv + Dr), generator),
            runtime_device,
        )
        if smooth_scales_cq_flag:
            smooth_scale_cq = _move_tensor_to_runtime_device(_rand_scale((1, Hcq), generator), runtime_device)
    elif weight_quant_mode == WEIGHT_QUANT_MODE_MXFP8_FULL:
        if fp8_e8m0_dtype is None:
            pytest.skip("float8_e8m0 dtype is unavailable for mxfp8 scenario")
        deq_scale_x = _move_tensor_to_runtime_device(torch.ones((T, He // 32), dtype=fp8_e8m0_dtype), runtime_device)
        deq_scale_w_dq = _move_tensor_to_runtime_device(
            torch.ones((Hcq, He // 32), dtype=fp8_e8m0_dtype),
            runtime_device,
        )
        deq_scale_w_uq_qr = _move_tensor_to_runtime_device(
            torch.ones((N1 * (D + Dr), Hcq // 32), dtype=fp8_e8m0_dtype),
            runtime_device,
        )
        deq_scale_w_dkv_kr = _move_tensor_to_runtime_device(
            torch.ones((Hckv + Dr, He // 32), dtype=fp8_e8m0_dtype),
            runtime_device,
        )

    if kv_quant_mode == 1:
        quant_scale_ckv = _move_tensor_to_runtime_device(_rand_scale((1,), generator), runtime_device)
    elif kv_quant_mode == 2:
        quant_scale_ckv = _move_tensor_to_runtime_device(_rand_scale((1, Hckv), generator), runtime_device)
        quant_scale_ckr = _move_tensor_to_runtime_device(_rand_scale((1, Dr), generator), runtime_device)
    elif kv_quant_mode == 3:
        k_nope_clip_alpha = _move_tensor_to_runtime_device(
            _rand_scale((1,), generator, min_val=0.9, max_val=1.1),
            runtime_device,
        )

    runtime_inputs = {
        "token_x": token_x,
        "w_dq": w_dq,
        "w_uq_qr": w_uq_qr,
        "w_uk": w_uk,
        "w_dkv_kr": w_dkv_kr,
        "rmsnorm_gamma_cq": rmsnorm_gamma_cq,
        "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "cache_index": cache_index,
        "kv_cache": kv_cache,
        "kr_cache": kr_cache,
        "dequant_scale_x": deq_scale_x,
        "dequant_scale_w_dq": deq_scale_w_dq,
        "dequant_scale_w_uq_qr": deq_scale_w_uq_qr,
        "dequant_scale_w_dkv_kr": deq_scale_w_dkv_kr,
        "quant_scale_ckv": quant_scale_ckv,
        "quant_scale_ckr": quant_scale_ckr,
        "smooth_scales_cq": smooth_scale_cq,
        "actual_seq_len": actual_seq_len,
        "k_nope_clip_alpha": k_nope_clip_alpha,
    }

    return {
        "params": params,
        "named_params": _params_to_named_dict(params),
        "seed": seed,
        "runtime_device": runtime_device,
        "runtime_inputs": runtime_inputs,
        "op_attrs": _build_op_attrs(_params_to_named_dict(params)),
    }


def test_prologv3_generalized(params):
    case_payload = _build_default_case_payload(params, validate_quant_combo=True, runtime_device="npu")
    expect = GeneralizedPrologV3(params).forward(_build_forward_inputs(case_payload["runtime_inputs"]))
    result_aligned = _execute_npu_case(case_payload["runtime_inputs"], case_payload["op_attrs"])
    return expect, result_aligned


def run_prologv3_npu_only(params, attr_overrides=None, input_overrides=None):
    case_payload = _build_default_case_payload(params, validate_quant_combo=True, runtime_device="npu")
    case_payload = _apply_case_overrides(
        case_payload,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
    )
    return _execute_npu_case(case_payload["runtime_inputs"], case_payload["op_attrs"])


def run_prologv3_cpu_only(params, attr_overrides=None, input_overrides=None, validate_quant_combo=True):
    case_payload = _build_default_case_payload(
        params,
        validate_quant_combo=validate_quant_combo,
        runtime_device="cpu",
    )
    case_payload = _apply_case_overrides(
        case_payload,
        attr_overrides=attr_overrides,
        input_overrides=input_overrides,
    )
    result = GeneralizedPrologV3(params).forward(_build_forward_inputs(case_payload["runtime_inputs"]))
    return case_payload, result
