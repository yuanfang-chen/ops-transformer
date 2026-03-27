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
 * \file aclnn_fused_k_rms_norm_rope_store_kv_cache_mx_quant.h
 * \brief
 */

#ifndef OP_API_ACLNN_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_
#define OP_API_ACLNN_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedKRmsNormRopeStoreKvCacheMxQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] qkv: 计算输入张量，npu device侧的aclTensor，数据类型支持BF16，shape为[T, N, D]，数据格式支持ND。
 * @param [in] cos: 计算输入张量，npu device侧的aclTensor，数据类型支持BF16，shape为[T, 1, D]，数据格式支持ND。
 * @param [in] sin: 计算输入张量，npu device侧的aclTensor，数据类型支持BF16，shape为[T, 1, D]，数据格式支持ND。
 * @param [in] gamma: 计算输入张量，npu device侧的aclTensor，数据类型支持FLOAT32，shape为[D]，数据格式支持ND。
 * @param [in] kvSlotMapping: 计算输入张量，npu device侧的aclTensor，数据类型支持INT64，shape为[T]，数据格式支持ND。
 * @param [in] vScaleSlotMapping: 计算输入张量，npu device侧的aclTensor，数据类型支持INT64，shape为[T]，数据格式支持ND。
 * @param [in] kCache: 计算输入输出张量（原地更新），npu device侧的aclTensor，数据类型支持FP8_E4M3FN，shape为[Bn, Nk, Bs, D]，数据格式支持ND。
 * @param [in] kScaleCache: 计算输入输出张量（原地更新），npu device侧的aclTensor，数据类型支持FP8_E8M0，shape为[Bn, Nk, Bs, D//32//2, 2]，数据格式支持ND。
 * @param [in] vCache: 计算输入输出张量（原地更新），npu device侧的aclTensor，数据类型支持FP8_E4M3FN，shape为[Bn, Nv, Bs, D]，数据格式支持ND。
 * @param [in] vScaleCache: 计算输入输出张量（原地更新），npu device侧的aclTensor，数据类型支持FP8_E8M0，shape为[Bn, Nv, Bs//32, D]，数据格式支持ND。
 * @param [out] q: 计算输出张量，npu device侧的aclTensor，数据类型支持FP8_E4M3FN，shape为[T, Nq, D]，数据格式支持ND。
 * @param [out] qScale: 计算输出张量，npu device侧的aclTensor，数据类型支持FP8_E8M0，shape为[T, Nq, D//32//2, 2]，数据格式支持ND。
 * @param [out] kCacheOut: 计算输出张量（与kCache同地址，原地更新），npu device侧的aclTensor。
 * @param [out] kScaleCacheOut: 计算输出张量（与kScaleCache同地址，原地更新），npu device侧的aclTensor。
 * @param [out] vCacheOut: 计算输出张量（与vCache同地址，原地更新），npu device侧的aclTensor。
 * @param [out] vScaleCacheOut: 计算输出张量（与vScaleCache同地址，原地更新），npu device侧的aclTensor。
 * @param [in] epsilon: 计算属性，数据类型支持float，防止计算除0，默认值1e-5。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFusedKRmsNormRopeStoreKvCacheMxQuantGetWorkspaceSize(
    const aclTensor* qkv, const aclTensor* cos, const aclTensor* sin, const aclTensor* gamma,
    const aclTensor* kvSlotMapping, const aclTensor* vScaleSlotMapping,
    aclTensor* kCache, aclTensor* kScaleCache, aclTensor* vCache, aclTensor* vScaleCache,
    aclTensor* q, aclTensor* qScale,
    aclTensor* kCacheOut, aclTensor* kScaleCacheOut, aclTensor* vCacheOut, aclTensor* vScaleCacheOut,
    float epsilon, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFusedKRmsNormRopeStoreKvCacheMxQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFusedKRmsNormRopeStoreKvCacheMxQuant(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_H_
