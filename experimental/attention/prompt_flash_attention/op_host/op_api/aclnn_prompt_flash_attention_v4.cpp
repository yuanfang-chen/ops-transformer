/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_prompt_flash_attention_v4.h"
#include "aclnn_prompt_flash_attention_inner.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus aclnnPromptFlashAttentionV4GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclTensor *sabiTensor,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1, // quantScale1 of V4
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayout, // inputLayout of V4
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
        return InnerPromptFlashAttentionV4GetWorkspaceSize(query, key, value, pseShift, attenMask, sabiTensor,
                                                              actualSeqLengths, actualSeqLengthsKv,
                                                              deqScale1, quantScale1, deqScale2,
                                                              quantScale2, quantOffset2,
                                                              numHeads, scaleValue, preTokens, nextTokens,
                                                              inputLayout, numKeyValueHeads, sparseMode,
                                                              innerPrecise, attentionOut, workspaceSize, executor);
    }

aclnnStatus aclnnPromptFlashAttentionV4(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream) { // V4 call aclnn inner
        return InnerPromptFlashAttentionV4(workspace, workspaceSize, executor, stream);
    }

}

#ifdef __cplusplus
}
#endif