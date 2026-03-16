/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_RECURRENCE_H
#define OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_RECURRENCE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ChunkGatedDeltaRuleRecurrence 第一段接口：计算 workspace 大小。
 *
 * @param [in/out] initialState  float32 [b, hv, dv, dk]，输入初始 state，同时作为输出返回更新后的 state。
 * @param [in]     kgexp         float32 [hv, n_chunks, cs, dk]
 * @param [in]     value         float32 [hv, n_chunks, cs, dv]
 * @param [in]     kCumdecay     float32 [hv, n_chunks, cs, dk]
 * @param [in]     qgexp         float32 [hv, n_chunks, cs, dk]
 * @param [in]     gexp          float32 [hv, n_chunks, cs, 1]
 * @param [in]     cuSeqlens     int32   [b+1]
 * @param [in]     scaleValue    float32 属性，默认 1.0
 * @param [out]    attnInterOut  float32 [hv, n_chunks, cs, dv]
 * @param [out]    vNewOut       float32 [hv, n_chunks, cs, dv]
 * @param [out]    workspaceSize workspace 大小（字节）。
 * @param [out]    executor      op 执行器。
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnChunkGatedDeltaRuleRecurrenceGetWorkspaceSize(
    aclTensor       *initialState,
    const aclTensor *kgexp,
    const aclTensor *value,
    const aclTensor *kCumdecay,
    const aclTensor *qgexp,
    const aclTensor *gexp,
    const aclTensor *cuSeqlens,
    float            scaleValue,
    aclTensor       *attnInterOut,
    aclTensor       *vNewOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);

/**
 * @brief ChunkGatedDeltaRuleRecurrence 第二段接口：执行算子。
 *
 * @param [in] workspace      Device 侧 workspace 地址。
 * @param [in] workspaceSize  workspace 大小，由第一段接口获取。
 * @param [in] executor       op 执行器。
 * @param [in] stream         acl stream。
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnChunkGatedDeltaRuleRecurrence(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_RECURRENCE_H
