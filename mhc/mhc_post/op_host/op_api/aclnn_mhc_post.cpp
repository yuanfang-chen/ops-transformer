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
 * \file aclnn_mhc_post.cpp
 * \brief MhcPost ACLNN API implementation
 */

#include "aclnn_mhc_post.h"
#include "mhc_post.h"

#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
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

static const int64_t DIM_NUM_3 = 3;
static const int64_t DIM_NUM_4 = 4;

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

static bool IsTNDFormat(size_t dimNum)
{
    return dimNum == DIM_NUM_3;
}

static bool IsBSNDFormat(size_t dimNum)
{
    return dimNum == DIM_NUM_4;
}

static aclnnStatus CheckShape3D(const MhcPostParams &params)
{
    // x shape: [T, n, D]
    auto xDim0 = params.x->GetViewShape().GetDim(0);
    auto xDim1 = params.x->GetViewShape().GetDim(1);
    auto xDim2 = params.x->GetViewShape().GetDim(2);

    // h_res shape: [T, n, n]
    auto hResDim0 = params.h_res->GetViewShape().GetDim(0);
    auto hResDim1 = params.h_res->GetViewShape().GetDim(1);
    auto hResDim2 = params.h_res->GetViewShape().GetDim(2);
    CHECK_COND(hResDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "h_res dim[0] %ld is not equal to x dim[0] %ld", hResDim0, xDim0);
    CHECK_COND(hResDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "h_res dim[1] %ld is not equal to x dim[1] %ld", hResDim1, xDim1);
    CHECK_COND(hResDim2 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "h_res dim[2] %ld must equal h_res dim[1] %ld (n x n matrix)", hResDim2, hResDim1);

    // h_out shape: [T, D]
    auto hOutDimNum = params.h_out->GetViewShape().GetDimNum();
    auto hOutDim0 = params.h_out->GetViewShape().GetDim(0);
    auto hOutDim1 = params.h_out->GetViewShape().GetDim(1);
    CHECK_COND(hOutDimNum == 2, ACLNN_ERR_PARAM_INVALID,
               "h_out dim num should be 2 for 3D format, but got %zu", hOutDimNum);
    CHECK_COND(hOutDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "h_out dim[0] %ld is not equal to x dim[0] %ld", hOutDim0, xDim0);
    CHECK_COND(hOutDim1 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "h_out dim[1] %ld is not equal to x dim[2] %ld", hOutDim1, xDim2);

    // h_post shape: [T, n]
    auto hPostDimNum = params.h_post->GetViewShape().GetDimNum();
    auto hPostDim0 = params.h_post->GetViewShape().GetDim(0);
    auto hPostDim1 = params.h_post->GetViewShape().GetDim(1);
    CHECK_COND(hPostDimNum == 2, ACLNN_ERR_PARAM_INVALID,
               "h_post dim num should be 2 for 3D format, but got %zu", hPostDimNum);
    CHECK_COND(hPostDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "h_post dim[0] %ld is not equal to x dim[0] %ld", hPostDim0, xDim0);
    CHECK_COND(hPostDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "h_post dim[1] %ld is not equal to x dim[1] %ld", hPostDim1, xDim1);

    // y shape: [T, n, D]
    auto yDim0 = params.y->GetViewShape().GetDim(0);
    auto yDim1 = params.y->GetViewShape().GetDim(1);
    auto yDim2 = params.y->GetViewShape().GetDim(2);
    CHECK_COND(yDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "y dim[0] %ld is not equal to x dim[0] %ld", yDim0, xDim0);
    CHECK_COND(yDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "y dim[1] %ld is not equal to x dim[1] %ld", yDim1, xDim1);
    CHECK_COND(yDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "y dim[2] %ld is not equal to x dim[2] %ld", yDim2, xDim2);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape4D(const MhcPostParams &params)
{
    // x shape: [B, S, n, D]
    auto xDim0 = params.x->GetViewShape().GetDim(0);
    auto xDim1 = params.x->GetViewShape().GetDim(1);
    auto xDim2 = params.x->GetViewShape().GetDim(2);
    auto xDim3 = params.x->GetViewShape().GetDim(3);

    // h_res shape: [B, S, n, n]
    auto hResDimNum = params.h_res->GetViewShape().GetDimNum();
    auto hResDim0 = params.h_res->GetViewShape().GetDim(0);
    auto hResDim1 = params.h_res->GetViewShape().GetDim(1);
    auto hResDim2 = params.h_res->GetViewShape().GetDim(2);
    auto hResDim3 = params.h_res->GetViewShape().GetDim(3);
    CHECK_COND(hResDimNum == DIM_NUM_4, ACLNN_ERR_PARAM_INVALID,
               "h_res dim num should be 4 for 4D format, but got %zu", hResDimNum);
    CHECK_COND(hResDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "h_res dim[0] %ld is not equal to x dim[0] %ld", hResDim0, xDim0);
    CHECK_COND(hResDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "h_res dim[1] %ld is not equal to x dim[1] %ld", hResDim1, xDim1);
    CHECK_COND(hResDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "h_res dim[2] %ld is not equal to x dim[2] %ld", hResDim2, xDim2);
    CHECK_COND(hResDim3 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "h_res dim[3] %ld must equal h_res dim[2] %ld (n x n matrix)", hResDim3, hResDim2);

    // h_out shape: [B, S, D]
    auto hOutDimNum = params.h_out->GetViewShape().GetDimNum();
    auto hOutDim0 = params.h_out->GetViewShape().GetDim(0);
    auto hOutDim1 = params.h_out->GetViewShape().GetDim(1);
    auto hOutDim2 = params.h_out->GetViewShape().GetDim(2);
    CHECK_COND(hOutDimNum == 3, ACLNN_ERR_PARAM_INVALID,
               "h_out dim num should be 3 for 4D format, but got %zu", hOutDimNum);
    CHECK_COND(hOutDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "h_out dim[0] %ld is not equal to x dim[0] %ld", hOutDim0, xDim0);
    CHECK_COND(hOutDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "h_out dim[1] %ld is not equal to x dim[1] %ld", hOutDim1, xDim1);
    CHECK_COND(hOutDim2 == xDim3, ACLNN_ERR_PARAM_INVALID,
               "h_out dim[2] %ld is not equal to x dim[3] %ld", hOutDim2, xDim3);

    // h_post shape: [B, S, n]
    auto hPostDimNum = params.h_post->GetViewShape().GetDimNum();
    auto hPostDim0 = params.h_post->GetViewShape().GetDim(0);
    auto hPostDim1 = params.h_post->GetViewShape().GetDim(1);
    auto hPostDim2 = params.h_post->GetViewShape().GetDim(2);
    CHECK_COND(hPostDimNum == 3, ACLNN_ERR_PARAM_INVALID,
               "h_post dim num should be 3 for 4D format, but got %zu", hPostDimNum);
    CHECK_COND(hPostDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "h_post dim[0] %ld is not equal to x dim[0] %ld", hPostDim0, xDim0);
    CHECK_COND(hPostDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "h_post dim[1] %ld is not equal to x dim[1] %ld", hPostDim1, xDim1);
    CHECK_COND(hPostDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "h_post dim[2] %ld is not equal to x dim[2] %ld", hPostDim2, xDim2);

    // y shape: [B, S, n, D]
    auto yDim0 = params.y->GetViewShape().GetDim(0);
    auto yDim1 = params.y->GetViewShape().GetDim(1);
    auto yDim2 = params.y->GetViewShape().GetDim(2);
    auto yDim3 = params.y->GetViewShape().GetDim(3);
    CHECK_COND(yDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "y dim[0] %ld is not equal to x dim[0] %ld", yDim0, xDim0);
    CHECK_COND(yDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "y dim[1] %ld is not equal to x dim[1] %ld", yDim1, xDim1);
    CHECK_COND(yDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "y dim[2] %ld is not equal to x dim[2] %ld", yDim2, xDim2);
    CHECK_COND(yDim3 == xDim3, ACLNN_ERR_PARAM_INVALID,
               "y dim[3] %ld is not equal to x dim[3] %ld", yDim3, xDim3);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const MhcPostParams &params)
{
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    CHECK_COND(IsTNDFormat(xDimNum) || IsBSNDFormat(xDimNum), ACLNN_ERR_PARAM_INVALID,
               "x dim should be 3 (TND format) or 4 (BSND format), but got %zu", xDimNum);

    if (IsTNDFormat(xDimNum)) {
        CHECK_COND(CheckShape3D(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape for 3D format");
    } else {
        CHECK_COND(CheckShape4D(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape for 4D format");
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

static aclnnStatus InputsContiguousAndTransFormat(const aclTensor *tensor, const aclTensor *&reformatedTensor,
                                                  const std::string &tensorName, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    reformatedTensor = l0op::Contiguous(tensor, executor);
    CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s Contiguous failed.", tensorName.c_str());
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

    // Convert input tensors to contiguous
    const aclTensor *reformatedX = nullptr;
    const aclTensor *reformatedHRes = nullptr;
    const aclTensor *reformatedHOut = nullptr;
    const aclTensor *reformatedHPost = nullptr;

    ret = InputsContiguousAndTransFormat(x, reformatedX, "x", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputsContiguousAndTransFormat(h_res, reformatedHRes, "h_res", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputsContiguousAndTransFormat(h_out, reformatedHOut, "h_out", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputsContiguousAndTransFormat(h_post, reformatedHPost, "h_post", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Call l0 interface: MhcPost kernel
    // Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
    const aclTensor *mhcPostResult =
        l0op::MhcPost(reformatedX, reformatedHRes, reformatedHOut, reformatedHPost, uniqueExecutor.get());
    CHECK_RET(mhcPostResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Convert output tensor to contiguous tensor and copy to output
    auto viewCopyResult = l0op::ViewCopy(mhcPostResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

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