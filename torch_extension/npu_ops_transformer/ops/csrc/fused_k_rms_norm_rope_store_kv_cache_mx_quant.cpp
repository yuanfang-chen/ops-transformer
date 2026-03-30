/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

constexpr int DIM_ONE = 1;
constexpr int DIM_TWO = 2;
constexpr int DIM_THREE = 3;
constexpr int DIM_FOUR = 4;
constexpr int DIM_FIVE = 5;
constexpr int64_t QUANT_BLOCK_SIZE = 32;
constexpr int64_t DIGIT_TWO = 2;

/**
 * @brief Wrapper for FusedKRmsNormRopeStoreKvCacheMxQuant
 *
 * Inputs:
 *   qkv:                  [T, N, D]        BF16
 *   cos:                  [T, 1, D]        BF16
 *   sin:                  [T, 1, D]        BF16
 *   gamma:                [D]              FP32
 *   kv_slot_mapping:      [T]              INT64
 *   v_scale_slot_mapping: [T]              INT64
 *   k_cache:              [Bn, Nk, Bs, D]  FP8_E4M3FN
 *   k_scale_cache:        [Bn, Nk, Bs, D//32] FP8_E8M0
 *   v_cache:              [Bn, Nv, Bs, D]  FP8_E4M3FN
 *   v_scale_cache:        [Bn, Nv, Bs//32, D] FP8_E8M0
 *
 * Outputs:
 *   q:              [T, Nq, D]             FP8_E4M3FN
 *   q_scale:        [T, Nq, D//32//2, 2]  FP8_E8M0
 *   k_cache:        [Bn, Nk, Bs, D]       FP8_E4M3FN  (inplace)
 *   k_scale_cache:  [Bn, Nk, Bs, D//32]   FP8_E8M0    (inplace)
 *   v_cache:        [Bn, Nv, Bs, D]       FP8_E4M3FN  (inplace)
 *   v_scale_cache:  [Bn, Nv, Bs//32, D]   FP8_E8M0    (inplace)
 *
 * Attr:
 *   epsilon: float, default 1e-5
 */
tensor_list npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant(
    const at::Tensor &qkv,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &gamma,
    const at::Tensor &kv_slot_mapping,
    const at::Tensor &v_scale_slot_mapping,
    at::Tensor &k_cache,
    at::Tensor &k_scale_cache,
    at::Tensor &v_cache,
    at::Tensor &v_scale_cache,
    float epsilon)
{
    // Input validation
    TORCH_CHECK(qkv.dim() == DIM_THREE, "qkv must be 3D tensor [T, N, D], got ", qkv.dim(), "D.");
    TORCH_CHECK(cos.dim() == DIM_THREE, "cos must be 3D tensor [T, 1, D], got ", cos.dim(), "D.");
    TORCH_CHECK(sin.dim() == DIM_THREE, "sin must be 3D tensor [T, 1, D], got ", sin.dim(), "D.");
    TORCH_CHECK(gamma.dim() == DIM_ONE, "gamma must be 1D tensor [D], got ", gamma.dim(), "D.");
    TORCH_CHECK(kv_slot_mapping.dim() == DIM_ONE, "kv_slot_mapping must be 1D tensor [T], got ",
                kv_slot_mapping.dim(), "D.");
    TORCH_CHECK(v_scale_slot_mapping.dim() == DIM_ONE, "v_scale_slot_mapping must be 1D tensor, got ",
                v_scale_slot_mapping.dim(), "D.");
    TORCH_CHECK(k_cache.dim() == DIM_FOUR, "k_cache must be 4D tensor [Bn, Nk, Bs, D], got ",
                k_cache.dim(), "D.");
    TORCH_CHECK(k_scale_cache.dim() == DIM_FIVE, "k_scale_cache must be 5D tensor [Bn, Nk, Bs, D/32/2, 2], got ",
                k_scale_cache.dim(), "D.");
    TORCH_CHECK(v_cache.dim() == DIM_FOUR, "v_cache must be 4D tensor [Bn, Nv, Bs, D], got ",
                v_cache.dim(), "D.");
    TORCH_CHECK(v_scale_cache.dim() == DIM_FIVE, "v_scale_cache must be 5D tensor [Bn, Nv, Bs/32/2, D, 2], got ",
                v_scale_cache.dim(), "D.");

    // Derive output shapes
    int64_t T = qkv.size(0);
    int64_t N = qkv.size(1);
    int64_t D = qkv.size(2);
    int64_t Nk = k_cache.size(1);
    int64_t Nv = v_cache.size(1);
    int64_t Nq = N - Nk - Nv;
    TORCH_CHECK(Nq > 0, "Nq must be positive, N=", N, ", Nk=", Nk, ", Nv=", Nv, ".");

    // Create output tensors q and q_scale
    at::Tensor q = at::empty({T, Nq, D},
        at::TensorOptions().dtype(k_cache.scalar_type())
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));

    at::Tensor q_scale = at::empty({T, Nq, D / QUANT_BLOCK_SIZE / DIGIT_TWO, DIGIT_TWO},
        at::TensorOptions().dtype(k_scale_cache.scalar_type())
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));

    // Call ACLNN kernel
    ACLNN_CMD(aclnnFusedKRmsNormRopeStoreKvCacheMxQuant,
              qkv, cos, sin, gamma, kv_slot_mapping, v_scale_slot_mapping,
              k_cache, k_scale_cache, v_cache, v_scale_cache,
              q, q_scale, k_cache, k_scale_cache, v_cache, v_scale_cache,
              epsilon);

    return std::tie(q, q_scale, k_cache, k_scale_cache, v_cache, v_scale_cache);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant",
          &npu_fused_k_rms_norm_rope_store_kv_cache_mx_quant,
          "fused_k_rms_norm_rope_store_kv_cache_mx_quant");
}
} // op_api
