/**
?* This program is free software, you can redistribute it and/or modify.
?* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_causal_conv1d_update.h"

#include "causal_conv1d_update.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static aclnnStatus CheckParams(const aclTensor* x,
                               const aclTensor* weight,
                               const aclTensor* biasOptional,
                               aclTensor* convState,
                               const aclTensor* convStateIndicesOptional,
                               const aclTensor* y)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(convState, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(y, return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_DTYPE_NOT_SAME(weight, x, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(convState, x, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(y, x, return ACLNN_ERR_PARAM_INVALID);

    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(biasOptional, x, return ACLNN_ERR_PARAM_INVALID);
    }

    if (convStateIndicesOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(convStateIndicesOptional, DataType::DT_INT32, return ACLNN_ERR_PARAM_INVALID);
    }
    
    // output shape should match x (decode/update: y matches x)
    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnCausalConv1dUpdateGetWorkspaceSize(const aclTensor* x,
                                                   const aclTensor* weight,
                                                   aclTensor* convState,
                                                   const aclTensor* convStateIndicesOptional,
                                                   const aclTensor* biasOptional,
                                                   const aclTensor* numAcceptedTokensOptional,
                                                   const aclTensor* queryStartLocOptional,
                                                   int64_t activationMode,
                                                   int64_t padSlotId,
                                                   const aclTensor* y,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dUpdate, DFX_IN(x, weight, convState, convStateIndicesOptional, biasOptional, numAcceptedTokensOptional, queryStartLocOptional, activationMode, padSlotId),
                   DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, weight, biasOptional, convState, convStateIndicesOptional, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto yInfer =
        l0op::CausalConv1dUpdate(x, weight, convState, convStateIndicesOptional, biasOptional, numAcceptedTokensOptional, queryStartLocOptional, activationMode, padSlotId,
                                uniqueExecutor.get());
    CHECK_RET(yInfer != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopy = l0op::ViewCopy(yInfer, y, uniqueExecutor.get());
    CHECK_RET(viewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCausalConv1dUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1dUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

