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

import torch
import torch_npu
import check_valid_param
import pytest
import math
import copy
import numpy as np
import random

COLOR_YELLOW = "\033[33m"
YELLOW_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
GREEN_RESET = "\033[0m"


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


def dynamic_quant(inputs, smooth_scale):
    """Dynamic quantization with optional smooth scale. Returns (quantized_int8, scale)."""
    T = inputs.size(0)
    H = inputs.size(1)
    y = torch.zeros(T, H).to(torch.int32)
    scale = torch.zeros(T).to(torch.float32)
    inputs = inputs.reshape(T, H).to(torch.float32)
    if smooth_scale is not None:
        if len(smooth_scale.shape) != 2 or smooth_scale.shape[1] != H:
            print(f"[ERROR]dynamic_quant got wrong input smooth_scale:{tuple(smooth_scale.shape)}, expected:({H})")
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
            y[bs_index:] = torch.round(inputs[bs_index:] / scale_bs)
    return y, scale


def dynamic_quant_without_smooth_scale(inputs, out_deqq_shape_shape):
    """Dynamic quantization for query output. Returns (quantized_int8, scale)."""
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
    scale = max_values / 127
    y = torch.round(inputs / scale)
    y = s8_saturation(y)
    if len(out_deqq_shape_shape) == 2:  # [BS, 1], per_token
        return y.reshape(T, N, H), scale.reshape(quant_loops, 1).to(torch.float64)
    else:
        return y.reshape(T, N, H), scale.reshape(T, N, 1).to(torch.float64)


def dynamic_quant_ckv_with_amax(inputs, amax, smooth_scale=None):
    """3D dynamic quantization with external amax. Returns (quantized, scale)."""
    T, N, H = inputs.shape
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
    scale = amax / 127.0
    y = torch.round(scaled_inputs / scale)
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
        self.head_dim, self.rope_head_dim, self.q_seq, self.kv_seq, self.block_size, \
        self.input_layout, self.cache_mode, self.cq_epsilon, self.ckv_epsilon, self.dtype, \
        self.weight_quant_mode, self.kv_quant_mode, self.query_quant_mode, \
        self.ckvkr_repo_mode, self.quant_scale_repo_mode, self.tile_size, \
        self.qc_qr_scale, self.kc_scale = params

    def forward(self, inputs):
        """
        Generalized forward supporting all quant and cache modes.
        inputs: dict with keys for base tensors and optional quant tensors.
        """
        B = self.batch_size
        S1 = self.q_seq
        S2 = self.kv_seq
        D = self.head_dim
        Dr = self.rope_head_dim
        N1 = self.q_head_num
        N2 = self.kv_head_num
        He = self.He
        Hckv = self.Hckv
        Hcq = self.Hcq
        BlockSize = self.block_size
        T = B * S1
        block_num = math.ceil(B * S2 / BlockSize)

        weight_quant_mode = self.weight_quant_mode
        kv_quant_mode = self.kv_quant_mode
        query_quant_mode = self.query_quant_mode
        cache_mode = self.cache_mode
        cq_epsilon = self.cq_epsilon
        ckv_epsilon = self.ckv_epsilon
        qc_qr_scale = self.qc_qr_scale
        kc_scale = self.kc_scale

        enable_quant_output = (query_quant_mode == 1 and weight_quant_mode in (2, 3))
        deq_scale_q_nope = None

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

        # Reshape to 2D: (B, S1, ...) -> (T, ...)
        token_x = token_x.reshape(T, He)
        sin = sin.reshape(T, Dr)
        cos = cos.reshape(T, Dr)
        index_table = index_table.reshape(-1) if index_table.numel() > 0 else index_table

        print("[INFO]========================================")
        print("[INFO]>>>>>>>>  Start to calculate  >>>>>>>>>>")
        print("[INFO]========================================")

        # -------------------------------------------------------------------
        # matmul1 : token_x(B*S1,He) * w_dq(He,Hcq) -> matmul1_res(B*S1,Hcq)
        # -------------------------------------------------------------------
        matmul1_dtype = torch.float32
        token_x_new = token_x.clone()
        if weight_quant_mode == 2:
            token_x_new = token_x_new.to(torch.int32)
            w_dq = w_dq.to(torch.int32)
            matmul1_dtype = torch.int32

        print(f"[INFO]matmul1 start. token_x:{tuple(token_x.shape)}|{token_x.dtype}"
              f" w_dq:{tuple(w_dq.shape)}|{w_dq.dtype}")
        token_x_new = token_x_new.to(torch.float32)
        w_dq = w_dq.to(torch.float32)
        matmul1_res = torch.matmul(token_x_new, w_dq).to(matmul1_dtype)

        # matmul1 post-processing
        if weight_quant_mode == 2:
            matmul1_res = matmul1_res.to(torch.float32)
            for t_index in range(T):
                matmul1_res[t_index, :] = matmul1_res[t_index, :] * deq_scale_x[t_index, 0]
            for h_index in range(Hcq):
                matmul1_res[:, h_index] = matmul1_res[:, h_index] * deq_scale_w_dq[0, h_index]
            print(f"[INFO]deq1 end. matmul1_res dtype={matmul1_res.dtype}")
        else:
            matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)
        print(f"[INFO]matmul1 end. matmul1_res:{tuple(matmul1_res.shape)}|{matmul1_res.dtype}")

        # ----------------------------------------------------------------------
        # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
        # ----------------------------------------------------------------------
        ep1 = float(cq_epsilon)
        norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
        norm1_res *= gamma_cq
        norm1_res *= qc_qr_scale
        print(f"[INFO]rmsnorm1 end. norm1_res:{tuple(norm1_res.shape)}|{norm1_res.dtype}")

        # ----------------------------------------------------------------------------------
        # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
        # ----------------------------------------------------------------------------------
        matmul2_dtype = torch.float32
        deq_scale_qcqr = None
        if weight_quant_mode in (1, 2):
            w_uq_qr = w_uq_qr.to(torch.int32)
            matmul2_dtype = torch.int32
            norm1_res, deq_scale_qcqr = dynamic_quant(norm1_res, smooth_scale_cq)
            print(f"[INFO]dynamic_quant end. norm1_res dtype={norm1_res.dtype}")
        else:
            norm1_res = norm1_res.to(torch.bfloat16).to(torch.float32)

        norm1_res = norm1_res.to(torch.float32)
        w_uq_qr = w_uq_qr.to(torch.float32)
        matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(matmul2_dtype)

        # matmul2 post-processing
        if weight_quant_mode in (1, 2):
            matmul2_res = matmul2_res.to(torch.float32)
            for t_index in range(T):
                matmul2_res[t_index, :] = matmul2_res[t_index, :] * deq_scale_qcqr[t_index]
            for nddr_index in range(matmul2_res.shape[1]):
                matmul2_res[:, nddr_index] = matmul2_res[:, nddr_index] * deq_scale_w_uqqr[0, nddr_index]
            print(f"[INFO]deq2 end. matmul2_res dtype={matmul2_res.dtype}")
        else:
            matmul2_res = matmul2_res.to(torch.bfloat16).to(torch.float32)
        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
        print(f"[INFO]matmul2 end. matmul2_res:{tuple(matmul2_res.shape)}|{matmul2_res.dtype}")

        # -------------------------------------------------------------------------------------
        # splitD1 : matmul2_res -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
        # -------------------------------------------------------------------------------------
        splitd1_res1 = matmul2_res[:, :, :D]
        splitd1_res2 = matmul2_res[:, :, D:]
        print(f"[INFO]splitD1 end. res1:{tuple(splitd1_res1.shape)} res2:{tuple(splitd1_res2.shape)}")

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
            out1, deq_scale_q_nope = dynamic_quant_without_smooth_scale(out1, out_deqq_shape_shape)
        out1 = out1.reshape(B, S1, N1, Hckv)
        print(f"[INFO]matmul3 end. {COLOR_YELLOW}out1:{out1.shape}|{out1.dtype}{YELLOW_RESET}")

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
        out2 = out2.reshape(B, S1, N1, Dr)
        print(f"[INFO]rotary1 end. {COLOR_YELLOW}out2:{tuple(out2.shape)}|{out2.dtype}{YELLOW_RESET}")

        # -------------------------------------------------------------------------------
        # matmul4 : token_x(B*S1,He) * w_dkv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
        # -------------------------------------------------------------------------------
        matmul4_dtype = torch.float32
        if weight_quant_mode == 2:
            w_dkv_kr = w_dkv_kr.to(torch.int32)
            matmul4_dtype = torch.int32

        matmul4_res = torch.matmul(token_x_new.to(torch.float32), w_dkv_kr.to(torch.float32)).to(matmul4_dtype)

        # matmul4 post-processing
        if weight_quant_mode == 2:
            matmul4_res = matmul4_res.to(torch.float32)
            for t_index in range(T):
                matmul4_res[t_index, :] = matmul4_res[t_index, :] * deq_scale_x[t_index, 0]
            for h_index in range(Hckv + Dr):
                matmul4_res[:, h_index] = matmul4_res[:, h_index] * deq_scale_w_dkvkr[0, h_index]
            print(f"[INFO]deq3 end. matmul4_res dtype={matmul4_res.dtype}")
        else:
            matmul4_res = matmul4_res.to(torch.bfloat16).to(torch.float32)
        print(f"[INFO]matmul4 end. matmul4_res:{tuple(matmul4_res.shape)}|{matmul4_res.dtype}")

        # -------------------------------------------------------------------------------------
        # splitD2 : matmul4_res -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
        # -------------------------------------------------------------------------------------
        splitd2_res1 = matmul4_res[:, :Hckv]
        splitd2_res2 = matmul4_res[:, Hckv:]
        print(f"[INFO]splitD2 end. res1:{tuple(splitd2_res1.shape)} res2:{tuple(splitd2_res2.shape)}")

        # ----------------------------------------------------------------------------
        # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
        # ----------------------------------------------------------------------------
        ep2 = float(ckv_epsilon)
        norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
        norm2_res *= gamma_ckv
        print(f"[INFO]rmsnorm2 end. norm2_res:{tuple(norm2_res.shape)}|{norm2_res.dtype}")

        Dtile = Hckv
        # rmsnorm2 post-processing: kv cache quantization
        if kv_quant_mode in (1, 2):
            norm2_res = quant(norm2_res, quant_scale_ckv)
            print(f"[INFO]quant1 end. norm2_res dtype={norm2_res.dtype}")
        elif kv_quant_mode == 3:
            tile_size = self.tile_size
            norm2_res = norm2_res.reshape(T, Hckv // tile_size, tile_size)
            eps = 1e-8
            amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
            amax = torch.clamp(amax, min=eps) * k_nope_clip_alpha
            clip_res = torch.clamp(norm2_res, min=-amax, max=amax)
            norm2_res, deq_scale_ckv = dynamic_quant_ckv_with_amax(clip_res, amax)
            deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
            norm2_res = norm2_res.reshape(T, Hckv).to(torch.int8)
            if self.ckvkr_repo_mode == 1:
                # Compute rotary2 early for merged storage
                cos_flat = cos.reshape(T, Dr)
                sin_flat = sin.reshape(T, Dr)
                k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
                rotary2_res_early = (k * cos_flat) + (rotate_half(k) * sin_flat)
                rotary2_bf16 = rotary2_res_early.to(torch.bfloat16)
                norm2_res = torch.cat((norm2_res, rotary2_bf16.view(torch.int8)), axis=-1)
                Dtile = Dtile + Dr * 2
            if self.quant_scale_repo_mode == 1:
                norm2_res = torch.cat((norm2_res, deq_scale_ckv.view(torch.int8)), axis=-1)
                Dtile = Dtile + Hckv // tile_size * 4

        # -------------------------------------------------------------------------------------
        # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
        # -------------------------------------------------------------------------------------
        cos_flat = cos.reshape(T, Dr)
        sin_flat = sin.reshape(T, Dr)
        k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
        rotary2_res = (k * cos_flat) + (rotate_half(k) * sin_flat)
        print(f"[INFO]rotary2 end. rotary2_res:{tuple(rotary2_res.shape)}|{rotary2_res.dtype}")
        # rotary2 post-processing: quantize if weight_quant_mode==1 and kv_quant_mode==2
        if weight_quant_mode == 1 and kv_quant_mode == 2:
            rotary2_res = quant(rotary2_res, quant_scale_ckr)
            print(f"[INFO]quant2 end. rotary2_res dtype={rotary2_res.dtype}")

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
            outputs = [out1.to(torch.bfloat16), out2.to(torch.bfloat16), out3, out4]
            if enable_quant_output:
                outputs.append(deq_scale_q_nope)
            return outputs

        out3_shape = kv_cache.shape
        if not pa_flag:
            kv_cache = kv_cache.transpose(2, 1)
        if kv_quant_mode == 0:
            kv_cache = kv_cache.to(torch.bfloat16)
            norm2_res_scatter = norm2_res.to(torch.bfloat16)
        else:
            norm2_res_scatter = norm2_res

        print(f"[INFO]scatter1 start. norm2_res:{tuple(norm2_res_scatter.shape)}|{norm2_res_scatter.dtype}"
              f" kv_cache:{tuple(kv_cache.shape)}|{kv_cache.dtype}")

        if cache_mode == "PA_BLK_NZ":
            kv_cache = scatter_pa_blk_nz(kv_cache, norm2_res_scatter, index_table, S1, kv_scatter_size)
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
        else:
            # BSND or TND
            kv_cache = kv_cache.reshape(B * S2, N2, Dtile)
            for i in range(T):
                for j in range(N2):
                    kv_cache[i, j, :] = norm2_res_scatter[i, :]
        out3 = kv_cache.reshape(out3_shape)
        print(f"[INFO]scatter1 end. {COLOR_YELLOW}out3:{tuple(out3.shape)}|{out3.dtype}{YELLOW_RESET}")

        # ----------------------------------------------------------------------------------------------
        # scatter2 : rotary2_res(B*S1,Dr) -> kr_cache
        # ----------------------------------------------------------------------------------------------
        if self.ckvkr_repo_mode == 1:
            out4 = copy.deepcopy(kr_cache)
        else:
            kr_cache = copy.deepcopy(kr_cache)
            out4_shape = kr_cache.shape
            if not pa_flag:
                kr_cache = kr_cache.transpose(2, 1)
            if weight_quant_mode == 1 and kv_quant_mode == 2:
                rotary2_scatter = rotary2_res
            else:
                kr_cache = kr_cache.to(torch.bfloat16)
                rotary2_scatter = rotary2_res.to(torch.bfloat16)

            print(f"[INFO]scatter2 start. rotary2:{tuple(rotary2_scatter.shape)}|{rotary2_scatter.dtype}"
                  f" kr_cache:{tuple(kr_cache.shape)}|{kr_cache.dtype}")

            if cache_mode == "PA_BLK_NZ":
                kr_cache = scatter_pa_blk_nz(kr_cache, rotary2_scatter, index_table, S1, kr_scatter_size)
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
            else:
                # BSND or TND
                kr_cache = kr_cache.reshape(B * S2, N2, Dr)
                for i in range(T):
                    for j in range(N2):
                        kr_cache[i, j, :] = rotary2_scatter[i, :]
            out4 = kr_cache.reshape(out4_shape)
            print(f"[INFO]scatter2 end. {COLOR_YELLOW}out4:{tuple(out4.shape)}|{out4.dtype}{YELLOW_RESET}")

        print("[INFO]========================================")
        print("[INFO]>>>>>>>>   Calculate success  >>>>>>>>>>")
        print("[INFO]========================================")

        # Build return list
        out1_ret = out1.to(torch.bfloat16) if not enable_quant_output else out1
        out2_ret = out2.to(torch.bfloat16) if not enable_quant_output else out2
        outputs = [out1_ret, out2_ret, out3, out4]
        if enable_quant_output:
            outputs.append(deq_scale_q_nope)
        return outputs

