/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_rotary_position_embedding_v2.h"
#include "rotary_position_embedding.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<op::DataType> SUPPORT_DTYPE_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckDtype(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, const aclTensor *rotate,
                              const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, SUPPORT_DTYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(cos, SUPPORT_DTYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sin, SUPPORT_DTYPE_LIST, return false);
    if (rotate != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(rotate, SUPPORT_DTYPE_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, SUPPORT_DTYPE_LIST, return false);
    return true;
}

static inline bool CheckDtypeConsistency(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                                         const aclTensor *rotate, const aclTensor *out)
{
    auto xDtype = x->GetDataType();
    if (cos->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cos dtype should be same as x, actual x dtype is %s and cos dtype is %s.",
            op::ToString(xDtype).GetString(), op::ToString(cos->GetDataType()).GetString());
        return false;
    }
    if (sin->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sin dtype should be same as x, actual x dtype is %s and sin dtype is %s.",
            op::ToString(xDtype).GetString(), op::ToString(sin->GetDataType()).GetString());
        return false;
    }
    if (rotate != nullptr && rotate->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotate dtype should be same as x, actual x dtype is %s and rotate dtype is %s.",
            op::ToString(xDtype).GetString(), op::ToString(rotate->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dtype should be same as x, actual x dtype is %s and out dtype is %s.",
            op::ToString(xDtype).GetString(), op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckMode(int64_t mode)
{
    if (mode < 0 || mode > 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode should be in range [0, 3], but is %lld.", mode);
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensor *x, const aclTensor *out)
{
    auto xShape = x->GetViewShape();
    auto outShape = out->GetViewShape();
    if (xShape.GetDimNum() != outShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and out dim num should be same.");
        return false;
    }
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (xShape.GetDim(i) != outShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and out shape should be same.");
            return false;
        }
    }
    return true;
}

static inline bool TensorContiguousProcess(const aclTensor *&contiguousTensor, aclOpExecutor *executor)
{
    if (contiguousTensor == nullptr) {
        OP_LOGD("RotaryPositionEmbeddingV2 no need to do contiguous process.");
        return true;
    }

    contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, const aclTensor *rotate,
                               const aclTensor *out, int64_t mode)
{
    CHECK_RET(CheckDtype(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeConsistency(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckMode(mode), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(const aclTensor *&x, const aclTensor *&cos, const aclTensor *&sin,
                              const aclTensor *&rotate, aclOpExecutor *executor)
{
    CHECK_RET(TensorContiguousProcess(x, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(cos, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(sin, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(rotate, executor), ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

}

aclnnStatus aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                           int64_t mode, const aclTensor* rotate, aclTensor* out,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRotaryPositionEmbeddingV2,
        DFX_IN(x, cos, sin, mode, rotate),
        DFX_OUT(out));

    auto ret = CheckParams(x, cos, sin, rotate, out, mode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    const aclTensor *xProcessed = x;
    const aclTensor *cosProcessed = cos;
    const aclTensor *sinProcessed = sin;
    const aclTensor *rotateProcessed = rotate;

    ret = PreProcess(xProcessed, cosProcessed, sinProcessed, rotateProcessed, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto result = l0op::RotaryPositionEmbedding(xProcessed, cosProcessed, sinProcessed, rotateProcessed, mode,
                                                uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRotaryPositionEmbeddingV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRotaryPositionEmbeddingV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
