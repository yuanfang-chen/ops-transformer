/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_k_rms_norm_rope_store_kv_cache_mx_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FusedKRmsNormRopeStoreKvCacheMxQuant);

aclnnStatus FusedKRmsNormRopeStoreKvCacheMxQuant(
    const aclTensor* qkv, const aclTensor* cos, const aclTensor* sin, const aclTensor* gamma,
    const aclTensor* kvSlotMapping, const aclTensor* vScaleSlotMapping,
    aclTensor* kCache, aclTensor* kScaleCache, aclTensor* vCache, aclTensor* vScaleCache,
    aclTensor* q, aclTensor* qScale,
    aclTensor* kCacheOut, aclTensor* kScaleCacheOut, aclTensor* vCacheOut, aclTensor* vScaleCacheOut,
    float epsilon, aclOpExecutor* executor)
{
    L0_DFX(FusedKRmsNormRopeStoreKvCacheMxQuant, qkv, cos, sin, gamma, kvSlotMapping, vScaleSlotMapping,
           kCache, kScaleCache, vCache, vScaleCache, epsilon);

    // infershape
    auto ret = INFER_SHAPE(
        FusedKRmsNormRopeStoreKvCacheMxQuant,
        OP_INPUT(qkv, cos, sin, gamma, kvSlotMapping, vScaleSlotMapping, kCache, kScaleCache, vCache, vScaleCache),
        OP_OUTPUT(q, qScale, kCacheOut, kScaleCacheOut, vCacheOut, vScaleCacheOut),
        OP_ATTR(epsilon));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedKRmsNormRopeStoreKvCacheMxQuant InferShape failed.");
        return ret;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FusedKRmsNormRopeStoreKvCacheMxQuant,
        OP_INPUT(qkv, cos, sin, gamma, kvSlotMapping, vScaleSlotMapping, kCache, kScaleCache, vCache, vScaleCache),
        OP_OUTPUT(q, qScale, kCacheOut, kScaleCacheOut, vCacheOut, vScaleCacheOut),
        OP_ATTR(epsilon));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedKRmsNormRopeStoreKvCacheMxQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return ret;
    }

    return ACLNN_SUCCESS;
}
} // namespace l0op
