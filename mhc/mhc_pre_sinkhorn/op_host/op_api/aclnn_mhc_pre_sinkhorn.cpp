/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mhc_pre_sinkhorn.h"
#include "mhc_pre_sinkhorn.h"
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

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;
static constexpr int64_t MIN_NUMITERS = 1;
static constexpr int64_t MAX_NUMITERS = 100;
static constexpr int64_t SUPPORT_DIM_NUM_3 = 3;
static constexpr int64_t SUPPORT_DIM_NUM_4 = 4;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

static bool CheckNotNull(const aclTensor *h_res, int64_t outFlag, const aclTensor *h_res_sinkhorn,
                         const aclTensor *norm_out, const aclTensor *sum_out)
{
    OP_CHECK_NULL(h_res, return false);
    OP_CHECK_NULL(h_res_sinkhorn, return false);
    if (outFlag) {
        OP_CHECK_NULL(norm_out, return false);
        OP_CHECK_NULL(sum_out, return false);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *h_res, int64_t outFlag, const aclTensor *h_res_sinkhorn,
                            const aclTensor *norm_out, const aclTensor *sum_out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(h_res, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(h_res_sinkhorn, DTYPE_SUPPORT_LIST, return false);
    if (outFlag) {
        OP_CHECK_DTYPE_NOT_SUPPORT(norm_out, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(sum_out, DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor *h_res, int64_t outFlag, const aclTensor *h_res_sinkhorn,
                        const aclTensor *norm_out, const aclTensor *sum_out)
{
    if (h_res->GetStorageFormat() != h_res_sinkhorn->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. h_res [%s], h_res_sinkhorn [%s].",
                ToString(h_res->GetStorageFormat()).GetString(), ToString(h_res_sinkhorn->GetStorageFormat()).GetString());
        return false;
    }
    if (outFlag) {
        if (h_res->GetStorageFormat() != norm_out->GetStorageFormat()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. h_res [%s], norm_out [%s].",
                    ToString(h_res->GetStorageFormat()).GetString(), ToString(norm_out->GetStorageFormat()).GetString());
            return false;
        }
        if (h_res->GetStorageFormat() != sum_out->GetStorageFormat()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. h_res [%s], sum_out [%s].",
                    ToString(h_res->GetStorageFormat()).GetString(), ToString(sum_out->GetStorageFormat()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor *h_res, int64_t outFlag, const aclTensor *h_res_sinkhorn,
                       const aclTensor *norm_out, const aclTensor *sum_out, int64_t numIters)
{
    OP_CHECK_SHAPE_NOT_EQUAL(h_res, h_res_sinkhorn, return false);

    if (numIters < MIN_NUMITERS || numIters > MAX_NUMITERS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numIters value error, numIters must be in 1 to 100, but got numIters = %ld .",
                numIters);
        return false;
    }

    if (outFlag != 0 && outFlag != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outFlag value error, outFlag must be 0 or 1, but got outFlag = %ld .",
                outFlag);
        return false;
    }

    auto hResShape = h_res->GetViewShape();
    auto hResDim = hResShape.GetDimNum();
    if (hResDim != SUPPORT_DIM_NUM_3 && hResDim != SUPPORT_DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim must be 3 or 4, but got Dim = %ld .", hResDim);
        return false;
    }

    int64_t n0 = 0;
    int64_t n1 = 0;
    if (hResDim == SUPPORT_DIM_NUM_3) {
        n0 = hResShape.GetDim(DIM_ONE);
        n1 = hResShape.GetDim(DIM_TWO);
    } else if (hResDim == SUPPORT_DIM_NUM_4) {
        n0 = hResShape.GetDim(DIM_TWO);
        n1 = hResShape.GetDim(DIM_THREE);
    }
    if ((n0 != n1) || (n0 != N_VALID_4 && n0 != N_VALID_6 && n0 != N_VALID_8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "n0 must equal n1, and n must be %ld or %ld or %ld, but got n0 = %ld, n1 = %ld", N_VALID_4, N_VALID_6,
                N_VALID_8, n0, n1);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor *h_res, int64_t outFlag, float eps, int64_t numIters,
                                      const aclTensor *h_res_sinkhorn, const aclTensor *norm_out, const aclTensor *sum_out)
{
    CHECK_RET(CheckNotNull(h_res, outFlag, h_res_sinkhorn, norm_out, sum_out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(h_res, outFlag, h_res_sinkhorn, norm_out, sum_out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(h_res, outFlag, h_res_sinkhorn, norm_out, sum_out, numIters), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(h_res, outFlag, h_res_sinkhorn, norm_out, sum_out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornGetWorkspaceSize(const aclTensor *h_res, float eps, int64_t numIters,
                                             int outFlag, aclTensor *h_res_sinkhorn,
                                             aclTensor *norm_out, aclTensor *sum_out,
                                             uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreSinkhorn, DFX_IN(h_res, eps, numIters, outFlag),
                    DFX_OUT(h_res_sinkhorn, norm_out, sum_out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t actualOutFlag = 1;
    if (norm_out == nullptr || sum_out == nullptr) {
        Shape emptyShape({});
        norm_out = (uniqueExecutor.get())->AllocTensor(emptyShape, h_res->GetDataType(), Format::FORMAT_ND);
        sum_out = (uniqueExecutor.get())->AllocTensor(emptyShape, h_res->GetDataType(), Format::FORMAT_ND);
        actualOutFlag = 0;
    }

    auto ret = CheckParams(h_res, actualOutFlag, eps, numIters, h_res_sinkhorn, norm_out, sum_out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (h_res->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support empty input tensor.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const aclTensor *h_res_contiguous = l0op::Contiguous(h_res, uniqueExecutor.get());
    CHECK_RET(h_res_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor *h_res_sinkhorn_contiguous = const_cast<aclTensor*>(l0op::Contiguous(h_res_sinkhorn, uniqueExecutor.get()));
    CHECK_RET(h_res_sinkhorn_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *norm_out_contiguous = nullptr;
    const aclTensor *sum_out_contiguous = nullptr;
    if (actualOutFlag) {
        norm_out_contiguous = l0op::Contiguous(norm_out, uniqueExecutor.get());
        CHECK_RET(norm_out_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        sum_out_contiguous = l0op::Contiguous(sum_out, uniqueExecutor.get());
        CHECK_RET(sum_out_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        norm_out_contiguous = norm_out;
        sum_out_contiguous = sum_out;
    }

    auto opExecutor = uniqueExecutor.get();
    opExecutor->SetInputs(h_res_contiguous);
    opExecutor->SetAttrs(eps, numIters, actualOutFlag);
    opExecutor->SetOutputs(h_res_sinkhorn_contiguous, norm_out_contiguous, sum_out_contiguous);

    ret = opExecutor->Compile(workspaceSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *executor = uniqueExecutor.release();
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSinkhorn, DFX_IN(workspace, workspaceSize), DFX_OUT());

    CHECK_NULL_WITH_REPORT(executor, return ACLNN_ERR_PARAM_PARAM_INVALID);
    CHECK_NULL_WITH_REPORT(stream, return ACLNN_ERR_PARAM_PARAM_INVALID);

    auto ret = executor->Run(workspace, workspaceSize, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
