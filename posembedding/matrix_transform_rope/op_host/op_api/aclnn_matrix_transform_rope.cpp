/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of software repository for the full text of the License.
 */

#include "aclnn_matrix_transform_rope.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matrix_transform_rope.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr int64_t X_DIM_NUM = 4;
static constexpr int64_t ROTATE_DIM_NUM = 2;

static const std::initializer_list<op::DataType> X_TYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                                const aclTensor *rotate, const aclTensor *out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(cos, return false);
    OP_CHECK_NULL(sin, return false);
    OP_CHECK_NULL(rotate, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                                   const aclTensor *rotate, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(cos, X_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sin, X_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(rotate, X_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, X_TYPE_SUPPORT_LIST, return false);

    // check dtype consistency
    if (x->GetDataType() != cos->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and cos dtype should be same, actual x dtype is %s and cos dtype is %s.",
            op::ToString(x->GetDataType()).GetString(), op::ToString(cos->GetDataType()).GetString());
        return false;
    }
    if (x->GetDataType() != sin->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and sin dtype should be same, actual x dtype is %s and sin dtype is %s.",
            op::ToString(x->GetDataType()).GetString(), op::ToString(sin->GetDataType()).GetString());
        return false;
    }
    if (x->GetDataType() != rotate->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and rotate dtype should be same, actual x dtype is %s and rotate dtype is %s.",
            op::ToString(x->GetDataType()).GetString(), op::ToString(rotate->GetDataType()).GetString());
        return false;
    }
    if (x->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and out dtype should be same, actual x dtype is %s and out dtype is %s.",
            op::ToString(x->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckFormat(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                               const aclTensor *rotate, const aclTensor *out)
{
    if (x->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's format should be ND, actual is [%s].",
            op::ToString(x->GetStorageFormat()).GetString());
        return false;
    }
    if (cos->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cos's format should be ND, actual is [%s].",
            op::ToString(cos->GetStorageFormat()).GetString());
        return false;
    }
    if (sin->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sin's format should be ND, actual is [%s].",
            op::ToString(sin->GetStorageFormat()).GetString());
        return false;
    }
    if (rotate->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotate's format should be ND, actual is [%s].",
            op::ToString(rotate->GetStorageFormat()).GetString());
        return false;
    }
    if (out->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out's format should be ND, actual is [%s].",
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDimNum(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                               const aclTensor *rotate, const aclTensor *out)
{
    OP_CHECK_WRONG_DIMENSION(x, X_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(cos, X_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(sin, X_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(rotate, ROTATE_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(out, X_DIM_NUM, return false);
    return true;
}

static inline bool CheckShape(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                              const aclTensor *rotate, const aclTensor *out)
{
    const op::Shape xShape = x->GetViewShape();
    const op::Shape cosShape = cos->GetViewShape();
    const op::Shape sinShape = sin->GetViewShape();
    const op::Shape rotateShape = rotate->GetViewShape();
    const op::Shape outShape = out->GetViewShape();

    // check rotate shape [D, D]
    int64_t rotateD = rotateShape.GetDim(0);
    if (rotateShape.GetDim(1) != rotateD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rotate shape should be [D, D], but is [%lld, %lld].",
            rotateShape.GetDim(0), rotateShape.GetDim(1));
        return false;
    }

    // check x last dim D matches rotate D
    int64_t xD = xShape.GetDim(3);
    if (xD != rotateD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x last dim %lld should match rotate dim %lld.", xD, rotateD);
        return false;
    }

    // check out shape matches x shape
    for (int64_t i = 0; i < X_DIM_NUM; i++) {
        if (outShape.GetDim(i) != xShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape should match x shape.");
            return false;
        }
    }

    OP_LOGD("MatrixTransformRope check shape success.");
    return true;
}

static inline bool CheckEmptyTensor(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                                    const aclTensor *rotate, const aclTensor *out)
{
    if (x->IsEmpty() || cos->IsEmpty() || sin->IsEmpty() || rotate->IsEmpty() || out->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MatrixTransformRope not support to process empty tensor currently.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                               const aclTensor *rotate, const aclTensor *out)
{
    // 1. check null
    CHECK_RET(CheckNotNull(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. check dtype
    CHECK_RET(CheckDtypeValid(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_INVALID);
    // 3. check dim num
    CHECK_RET(CheckDimNum(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_INVALID);
    // 4. check shape
    CHECK_RET(CheckShape(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_INVALID);
    // 5. check format
    CHECK_RET(CheckFormat(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_INVALID);
    // 6. check empty tensor
    CHECK_RET(CheckEmptyTensor(x, cos, sin, rotate, out), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("MatrixTransformRope check params success.");
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnMatrixTransformRopeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *cos, const aclTensor *sin, const aclTensor *rotate,
    const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMatrixTransformRope,
        DFX_IN(x, cos, sin, rotate),
        DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, cos, sin, rotate, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // call l0 op
    auto result = l0op::MatrixTransformRope(x, cos, sin, rotate, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // viewcopy to output
    auto viewCopyResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatrixTransformRope(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatrixTransformRope);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
