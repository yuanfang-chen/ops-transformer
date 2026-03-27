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
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_
#define OPS_OP_PROTO_INC_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */

REG_OP(FusedKRmsNormRopeStoreKvCacheMxQuant)
    .INPUT(qkv, TensorType({DT_BF16}))                   // [T, N, D]
    .INPUT(cos, TensorType({DT_BF16}))                   // [T, 1, D]
    .INPUT(sin, TensorType({DT_BF16}))                   // [T, 1, D]
    .INPUT(gamma, TensorType({DT_FLOAT}))                // [D]
    .INPUT(kv_slot_mapping, TensorType({DT_INT64}))      // [T]
    .INPUT(v_scale_slot_mapping, TensorType({DT_INT64})) // [T/32/2]
    .INPUT(k_cache, TensorType({DT_FP8_E4M3FN}))         // [Bn, Nk, Bs, D]
    .INPUT(k_scale_cache, TensorType({DT_FP8_E8M0}))     // [Bn, Nk, Bs, D/32/2, 2]
    .INPUT(v_cache, TensorType({DT_FP8_E4M3FN}))         // [Bn, Nv, Bs, D]
    .INPUT(v_scale_cache, TensorType({DT_FP8_E8M0}))     // [Bn, Nv, Bs/32/2, D, 2]
    .OUTPUT(q, TensorType({DT_FP8_E4M3FN}))              // [T, Nq, D]
    .OUTPUT(q_scale, TensorType({DT_FP8_E8M0}))          // [T, Nq, D/32/2, 2]
    .OUTPUT(k_cache, TensorType({DT_FP8_E4M3FN}))        // [Bn, Nk, Bs, D]
    .OUTPUT(k_scale_cache, TensorType({DT_FP8_E8M0}))    // [Bn, Nk, Bs, D/32/2, 2]
    .OUTPUT(v_cache, TensorType({DT_FP8_E4M3FN}))        // [Bn, Nv, Bs, D]
    .OUTPUT(v_scale_cache, TensorType({DT_FP8_E8M0}))    // [Bn, Nv, Bs/32/2, D, 2]
    .ATTR(epsilon, Float, 1e-5)
    .OP_END_FACTORY_REG(FusedKRmsNormRopeStoreKvCacheMxQuant)
} // namespace ge

#endif // OPS_OP_PROTO_INC_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_