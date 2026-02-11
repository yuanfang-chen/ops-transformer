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
 * \file aclnn_mhc_post.cpp
 * \brief MhcPost ACLNN API implementation
 */

#include "aclnn_mhc_post.h"

#include <dlfcn.h>
#include <new>

#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t DIM_LIMIT_UPPER = 8;
static const int64_t DIM_LIMIT_LOWER = 1;

struct MhcPostParams {
    const aclTensor *x = nullptr;
    const aclTensor *h_res = nullptr;
    const aclTensor *h_out = nullptr;
    const aclTensor *h_post = nullptr;
    const aclTensor *y = nullptr;
};

static aclnnStatus CheckNotNull(const aclTensor *x, const aclTensor *h_res, const aclTensor *h_out,
                                const aclTensor *h_post, const aclTensor *y)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(h_res != nullptr, ACLNN_ERR_PARAM_NULLPTR, "h_res must not be nullptr.");
    CHECK_COND(h_out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "h_out must not be nullptr.");
    CHECK_COND(h_post != nullptr, ACLNN_ERR_PARAM_NULLPTR, "h_post must not be nullptr.");
    CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const MhcPostParams &params)
{
    // x: FP16 or BF16
    const std::initializer_list<op::DataType> xSupportList = {
        op::DataType::DT_FLOAT16, op::DataType::DT_BF16
    };
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x, xSupportList, return ACLNN_ERR_PARAM_INVALID);

    // h_res: FP32 only
    OP_CHECK_DTYPE_NOT_MATCH(params.h_res, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    // h_out: must be same as x
    OP_CHECK_DTYPE_NOT_SAME(params.x, params.h_out, return ACLNN_ERR_PARAM_INVALID);

    // h_post: FP32 only
    OP_CHECK_DTYPE_NOT_MATCH(params.h_post, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    // y: must be same as x
    OP_CHECK_DTYPE_NOT_SAME(params.x, params.y, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const MhcPostParams &params)
{
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    CHECK_COND(xDimNum >= DIM_LIMIT_LOWER && xDimNum <= DIM_LIMIT_UPPER, ACLNN_ERR_PARAM_INVALID,
               "x dim should within 1 ~ 8, but x dim is %zu", xDimNum);

    // h_out must have same shape as x
    auto hOutDimNum = params.h_out->GetViewShape().GetDimNum();
    CHECK_COND(hOutDimNum == xDimNum, ACLNN_ERR_PARAM_INVALID,
               "h_out dim should be same as x dim, but h_out dim is %zu, x dim is %zu", hOutDimNum, xDimNum);

    for (size_t i = 0; i < xDimNum; ++i) {
        auto xDimValue = params.x->GetViewShape().GetDim(i);
        auto hOutDimValue = params.h_out->GetViewShape().GetDim(i);
        CHECK_COND(xDimValue == hOutDimValue, ACLNN_ERR_PARAM_INVALID,
                   "x dim[%zu] %ld is not equal to h_out dim[%zu] %ld", i, xDimValue, i, hOutDimValue);
    }

    // h_post must be broadcastable with x (same shape or scalar)
    auto hPostDimNum = params.h_post->GetViewShape().GetDimNum();
    if (hPostDimNum > 1) {
        CHECK_COND(hPostDimNum == xDimNum, ACLNN_ERR_PARAM_INVALID,
                   "h_post dim should be 1 (scalar) or same as x dim, but h_post dim is %zu, x dim is %zu",
                   hPostDimNum, xDimNum);
        for (size_t i = 0; i < hPostDimNum; ++i) {
            auto xDimValue = params.x->GetViewShape().GetDim(i);
            auto hPostDimValue = params.h_post->GetViewShape().GetDim(i);
            CHECK_COND(xDimValue == hPostDimValue, ACLNN_ERR_PARAM_INVALID,
                       "x dim[%zu] %ld is not equal to h_post dim[%zu] %ld", i, xDimValue, i, hPostDimValue);
        }
    }

    // Check output shape
    auto yDimNum = params.y->GetViewShape().GetDimNum();
    CHECK_COND(yDimNum == xDimNum, ACLNN_ERR_PARAM_INVALID,
               "y dim should be same as x dim, but y dim is %zu, x dim is %zu", yDimNum, xDimNum);

    for (size_t i = 0; i < xDimNum; ++i) {
        auto xDimValue = params.x->GetViewShape().GetDim(i);
        auto yDimValue = params.y->GetViewShape().GetDim(i);
        CHECK_COND(xDimValue == yDimValue, ACLNN_ERR_PARAM_INVALID,
                   "x dim[%zu] %ld is not equal to y dim[%zu] %ld", i, xDimValue, i, yDimValue);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const MhcPostParams &params)
{
    op::Format xFormat = params.x->GetStorageFormat();
    op::Format yFormat = params.y->GetStorageFormat();

    bool isXFormatValid = xFormat < Format::FORMAT_END && !op::IsPrivateFormat(xFormat);
    CHECK_COND(isXFormatValid, ACLNN_ERR_PARAM_INVALID, "format of x %s is invalid.",
               op::ToString(xFormat).GetString());

    bool isYFormatValid = yFormat < Format::FORMAT_END && !op::IsPrivateFormat(yFormat);
    CHECK_COND(isYFormatValid, ACLNN_ERR_PARAM_INVALID, "format of y %s is invalid.",
               op::ToString(yFormat).GetString());

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParam(const MhcPostParams &params)
{
    CHECK_COND(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid format.");
    CHECK_COND(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dtype.");
    CHECK_COND(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape.");

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor *x, const aclTensor *h_res, const aclTensor *h_out,
                                          const aclTensor *h_post, const aclTensor *y, uint64_t *workspaceSize,
                                          aclOpExecutor **executor)
{
    CHECK_COND(CheckNotNull(x, h_res, h_out, h_post, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
               "one of required inputs for aclnnMhcPostGetWorkspaceSize is nullptr.");

    MhcPostParams params{x, h_res, h_out, h_post, y};

    aclnnStatus ret = CheckParam(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Check if input tensors are empty
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    L2_DFX_PHASE_1(aclnnMhcPost,
                   DFX_IN(params.x, params.h_res, params.h_out, params.h_post),
                   DFX_OUT(params.y));

    // Call l0 interface
    // Note: The actual L0 implementation will be added later
    // For now, create a placeholder implementation
    // TODO: Implement actual L0 kernel call

    // Get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPost);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in MhcPost launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif