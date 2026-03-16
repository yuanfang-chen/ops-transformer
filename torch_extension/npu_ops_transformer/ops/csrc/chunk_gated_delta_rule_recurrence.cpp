/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_recurrence.cpp
 * \brief PTA wrapper for aclnnChunkGatedDeltaRuleRecurrence.
 *
 * Interface:
 *   npu_chunk_gated_delta_rule_recurrence(
 *       initial_state [b, hv, dv, dk]  float32  (modified in-place)
 *       kgexp         [hv, nC, cs, dk] float32
 *       value         [hv, nC, cs, dv] float32
 *       k_cumdecay    [hv, nC, cs, dk] float32
 *       qgexp         [hv, nC, cs, dk] float32
 *       gexp          [hv, nC, cs,  1] float32
 *       cu_seqlens    [b+1]            int32
 *       scale_value   float (default 1.0)
 *   ) -> (Tensor initial_state, Tensor attn_inter_out, Tensor v_new_out)
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

using tensor_list3 = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

/**
 * @brief PTA wrapper for aclnnChunkGatedDeltaRuleRecurrence.
 *
 * initial_state is modified in-place (state update across chunks).
 * Returns (initial_state, attn_inter_out, v_new_out).
 */
tensor_list3 npu_chunk_gated_delta_rule_recurrence(
    at::Tensor &initial_state,
    const at::Tensor &kgexp,
    const at::Tensor &value,
    const at::Tensor &k_cumdecay,
    const at::Tensor &qgexp,
    const at::Tensor &gexp,
    const at::Tensor &cu_seqlens,
    double scale_value)
{
    TORCH_CHECK(initial_state.dim() == 4,
                "initial_state must be 4-D [b, hv, dv, dk], got ", initial_state.dim(), "-D");
    TORCH_CHECK(value.dim() == 4,
                "value must be 4-D [hv, nChunks, cs, dv], got ", value.dim(), "-D");
    TORCH_CHECK(kgexp.dim() == 4,
                "kgexp must be 4-D [hv, nChunks, cs, dk], got ", kgexp.dim(), "-D");
    TORCH_CHECK(cu_seqlens.dim() == 1,
                "cu_seqlens must be 1-D [b+1], got ", cu_seqlens.dim(), "-D");
    TORCH_CHECK(initial_state.scalar_type() == at::kFloat,
                "initial_state must be float32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt,
                "cu_seqlens must be int32");

    // Allocate output tensors — same shape and dtype as value
    at::Tensor attn_inter_out = at::empty_like(value);
    at::Tensor v_new_out      = at::empty_like(value);

    float scale_val = static_cast<float>(scale_value);

    ACLNN_CMD(aclnnChunkGatedDeltaRuleRecurrence,
              initial_state, kgexp, value, k_cumdecay, qgexp, gexp, cu_seqlens,
              scale_val, attn_inter_out, v_new_out);

    return std::tie(initial_state, attn_inter_out, v_new_out);
}

// Bind to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_chunk_gated_delta_rule_recurrence",
          &npu_chunk_gated_delta_rule_recurrence,
          "ChunkGatedDeltaRuleRecurrence NPU operator (PTA interface)");
}

} // namespace op_api
