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
 * \file aclnn_moe_init_routing_v3_mx_quant.cpp
 * \brief
 */
#include <algorithm>
#include <tuple>
#include <cstddef>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "moe_init_routing_v3_mx_quant.h"
#include "aclnn_moe_init_routing_v3_mx_quant.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static inline bool CheckNotNull(const aclTensor *x,
                                const aclTensor *expertIdx,
                                const aclTensor *yOut,
                                const aclTensor *mxscaleOut,
                                const aclTensor *expandedRowIdxOut) {
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIdx, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(mxscaleOut, return false);
    OP_CHECK_NULL(expandedRowIdxOut, return false);
    return true;
}

ACLNN_API aclnnStatus aclnnMoeInitRoutingV3MxQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIdx,
    const aclTensor *scaleOptional,
    const aclTensor *offsetOptional,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag,
    int64_t axis,
    char *roundModeOptional,
    int64_t dstType,
    int64_t blocksize,
    int64_t scaleAlg,
    const aclTensor *yOut,
    const aclTensor *mxscaleOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional,
    const aclTensor *expandedScaleOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMoeInitRoutingV3MxQuant,
                    DFX_IN(x, expertIdx, scaleOptional, offsetOptional,
                           activeNum, expertCapacity, expertNum, dropPadMode,
                           expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
                           axis, roundModeOptional, dstType, blocksize, scaleAlg),
                    DFX_OUT(yOut, mxscaleOut, expandedRowIdxOut,
                            expertTokensCountOrCumsumOutOptional, expandedScaleOutOptional));

    // Null check for required tensors
    auto ret = CheckNotNull(x, expertIdx, yOut, mxscaleOut, expandedRowIdxOut);
    CHECK_RET(ret, ACLNN_ERR_PARAM_NULLPTR);

    // Create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Make inputs contiguous
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto expertIdxContiguous = l0op::Contiguous(expertIdx, uniqueExecutor.get());
    CHECK_RET(expertIdxContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclTensor *scaleContiguous = nullptr;
    if (scaleOptional != nullptr) {
        scaleContiguous = l0op::Contiguous(scaleOptional, uniqueExecutor.get());
        CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    const aclTensor *offsetContiguous = nullptr;
    if (offsetOptional != nullptr) {
        offsetContiguous = l0op::Contiguous(offsetOptional, uniqueExecutor.get());
        CHECK_RET(offsetContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    // Call l0op
    auto routingResult = l0op::MoeInitRoutingV3MxQuant(
        xContiguous, expertIdxContiguous, scaleContiguous, offsetContiguous,
        activeNum, expertCapacity, expertNum, dropPadMode,
        expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
        axis, roundModeOptional, dstType, blocksize, scaleAlg,
        yOut, mxscaleOut, expandedRowIdxOut,
        expertTokensCountOrCumsumOutOptional, expandedScaleOutOptional,
        uniqueExecutor.get());

    auto [yOut_, mxscaleOut_, expandedRowIdxOut_, expertTokensCountOrCumsumOut_, expandedScaleOut_] = routingResult;

    // Required outputs must not be null
    bool hasNullptr = (yOut_ == nullptr) || (mxscaleOut_ == nullptr) || (expandedRowIdxOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // ViewCopy results to output tensors (handles non-contiguous output)
    auto viewCopyYResult = l0op::ViewCopy(yOut_, yOut, uniqueExecutor.get());
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyMxscaleResult = l0op::ViewCopy(mxscaleOut_, mxscaleOut, uniqueExecutor.get());
    CHECK_RET(viewCopyMxscaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyExpandedRowIdxResult = l0op::ViewCopy(expandedRowIdxOut_, expandedRowIdxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedRowIdxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (expertTokensCountOrCumsumOut_ != nullptr && expertTokensCountOrCumsumOutOptional != nullptr) {
        auto viewCopyResult = l0op::ViewCopy(expertTokensCountOrCumsumOut_, expertTokensCountOrCumsumOutOptional, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (expandedScaleOut_ != nullptr && expandedScaleOutOptional != nullptr) {
        auto viewCopyResult = l0op::ViewCopy(expandedScaleOut_, expandedScaleOutOptional, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeInitRoutingV3MxQuant(void *workspace, uint64_t workspaceSize,
                                                    aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeInitRoutingV3MxQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
