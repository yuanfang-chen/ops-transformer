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
    .INPUT(qkv, TensorType({DT_BF16}))
    .INPUT(cos, TensorType({DT_BF16}))
    .INPUT(sin, TensorType({DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(kv_slot_mapping, TensorType({DT_INT64}))
    .INPUT(v_scale_slot_mapping, TensorType({DT_INT64}))
    .INPUT(k_cache, TensorType({DT_FP8_E4M3FN}))
    .INPUT(k_scale_cache, TensorType({DT_FP8_E8M0}))
    .INPUT(v_cache, TensorType({DT_FP8_E4M3FN}))
    .INPUT(v_scale_cache, TensorType({DT_FP8_E8M0}))
    .OUTPUT(q, TensorType({DT_FP8_E4M3FN}))
    .OUTPUT(q_scale, TensorType({DT_FP8_E8M0}))
    .OUTPUT(k_cache, TensorType({DT_FP8_E4M3FN}))
    .OUTPUT(k_scale_cache, TensorType({DT_FP8_E8M0}))
    .OUTPUT(v_cache, TensorType({DT_FP8_E4M3FN}))
    .OUTPUT(v_scale_cache, TensorType({DT_FP8_E8M0}))
    .ATTR(epsilon, Float, 1e-5)
    .OP_END_FACTORY_REG(FusedKRmsNormRopeStoreKvCacheMxQuant)
} // namespace ge

#endif // OPS_OP_PROTO_INC_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_