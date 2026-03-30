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
 * \file aclnn_fused_k_rms_norm_rope_store_kv_cache_mx_quant.cpp
 * \brief
 */

#include "aclnn_fused_k_rms_norm_rope_store_kv_cache_mx_quant.h"
#include "fused_k_rms_norm_rope_store_kv_cache_mx_quant.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace fused_k_rms_norm_rope_store_kv_cache_mx_quant {
static inline bool CheckNotNull(
    const aclTensor* qkv, const aclTensor* cos, const aclTensor* sin, const aclTensor* gamma,
    const aclTensor* kvSlotMapping, const aclTensor* vScaleSlotMapping,
    aclTensor* kCache, aclTensor* kScaleCache, aclTensor* vCache, aclTensor* vScaleCache,
    aclTensor* q, aclTensor* qScale)
{
    OP_CHECK_NULL(qkv, return false);
    OP_CHECK_NULL(cos, return false);
    OP_CHECK_NULL(sin, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(kvSlotMapping, return false);
    OP_CHECK_NULL(vScaleSlotMapping, return false);
    OP_CHECK_NULL(kCache, return false);
    OP_CHECK_NULL(kScaleCache, return false);
    OP_CHECK_NULL(vCache, return false);
    OP_CHECK_NULL(vScaleCache, return false);
    OP_CHECK_NULL(q, return false);
    OP_CHECK_NULL(qScale, return false);
    return true;
}
} // namespace fused_k_rms_norm_rope_store_kv_cache_mx_quant

aclnnStatus aclnnFusedKRmsNormRopeStoreKvCacheMxQuantGetWorkspaceSize(
    const aclTensor* qkv, const aclTensor* cos, const aclTensor* sin, const aclTensor* gamma,
    const aclTensor* kvSlotMapping, const aclTensor* vScaleSlotMapping,
    aclTensor* kCache, aclTensor* kScaleCache, aclTensor* vCache, aclTensor* vScaleCache,
    aclTensor* q, aclTensor* qScale,
    aclTensor* kCacheOut, aclTensor* kScaleCacheOut, aclTensor* vCacheOut, aclTensor* vScaleCacheOut,
    float epsilon, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnFusedKRmsNormRopeStoreKvCacheMxQuant,
                   DFX_IN(qkv, cos, sin, gamma, kvSlotMapping, vScaleSlotMapping,
                          kCache, kScaleCache, vCache, vScaleCache, epsilon),
                   DFX_OUT(q, qScale, kCacheOut, kScaleCacheOut, vCacheOut, vScaleCacheOut));

    // 参数检查
    auto ret = fused_k_rms_norm_rope_store_kv_cache_mx_quant::CheckNotNull(
        qkv, cos, sin, gamma, kvSlotMapping, vScaleSlotMapping,
        kCache, kScaleCache, vCache, vScaleCache, q, qScale);
    CHECK_RET(ret, ACLNN_ERR_PARAM_NULLPTR);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 将输入转换成连续的tensor
    auto qkvContiguous = l0op::Contiguous(qkv, uniqueExecutor.get());
    CHECK_RET(qkvContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto cosContiguous = l0op::Contiguous(cos, uniqueExecutor.get());
    CHECK_RET(cosContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto sinContiguous = l0op::Contiguous(sin, uniqueExecutor.get());
    CHECK_RET(sinContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto gammaContiguous = l0op::Contiguous(gamma, uniqueExecutor.get());
    CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto kvSlotMappingContiguous = l0op::Contiguous(kvSlotMapping, uniqueExecutor.get());
    CHECK_RET(kvSlotMappingContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto vScaleSlotMappingContiguous = l0op::Contiguous(vScaleSlotMapping, uniqueExecutor.get());
    CHECK_RET(vScaleSlotMappingContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto kCacheContiguous = l0op::Contiguous(kCache, uniqueExecutor.get());
    kCache = const_cast<aclTensor*>(kCacheContiguous);
    auto kScaleCacheContiguous = l0op::Contiguous(kScaleCache, uniqueExecutor.get());
    kScaleCache = const_cast<aclTensor*>(kScaleCacheContiguous);
    auto vCacheContiguous = l0op::Contiguous(vCache, uniqueExecutor.get());
    vCache = const_cast<aclTensor*>(vCacheContiguous);
    auto vScaleCacheContiguous = l0op::Contiguous(vScaleCache, uniqueExecutor.get());
    vScaleCache = const_cast<aclTensor*>(vScaleCacheContiguous);

    // 调用l0接口进行计算
    auto l0Ret = l0op::FusedKRmsNormRopeStoreKvCacheMxQuant(
        qkvContiguous, cosContiguous, sinContiguous, gammaContiguous,
        kvSlotMappingContiguous, vScaleSlotMappingContiguous,
        kCache, kScaleCache, vCache, vScaleCache,
        q, qScale, kCacheOut, kScaleCacheOut, vCacheOut, vScaleCacheOut,
        epsilon, uniqueExecutor.get());
    CHECK_RET(l0Ret == ACLNN_SUCCESS, l0Ret);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedKRmsNormRopeStoreKvCacheMxQuant(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedKRmsNormRopeStoreKvCacheMxQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
