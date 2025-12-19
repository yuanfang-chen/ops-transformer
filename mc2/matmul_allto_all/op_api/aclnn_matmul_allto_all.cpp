/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul_allto_all.cpp
 * \brief
 */
#include "aclnn_matmul_allto_all.h"
#include <algorithm>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "common/op_host/op_api/matmul_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t ONE_DIM = 1;
static constexpr size_t TWO_DIMS = 2;
static constexpr size_t HCCL_GROUP_NAME_MAX = 128U;
static constexpr int64_t KVALUE_MIN = 1;
static constexpr int64_t KVALUE_MAX = 65535;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_X = {
    op::DataType::DT_FLOAT16,
    op::DataType::DT_BF16
};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {
    op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT,
};
extern aclnnStatus aclnnInnerMatmulAlltoAllGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                            const aclTensor* x1_scale, const aclTensor* x2_scale, const aclTensor* comm_scale, 
                                                            const aclTensor* x1_offset, const aclTensor* x2_offset,
                                                            const char* group, int64_t world_size, const aclIntArray* all2all_axes, 
                                                            int64_t y_dtype, int64_t x1_quant_mode, int64_t x2_quant_mode,
                                                            int64_t comm_quant_mode, int64_t comm_quant_dtype,
                                                            bool transpose_x1, bool transpose_x2, int64_t group_size, 
                                                            const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMatmulAlltoAll(void *workspace, uint64_t workspaceSize,
                                            aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor* x, const aclTensor* weight, const aclTensor* y) {
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weight, 
                            const aclTensor* bias, const aclTensor* y) 
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, DTYPE_SUPPORT_LIST_X, return false);

    if (bias != nullptr) {
        auto biasDtype = bias->GetDataType();
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_BIAS, return false);
        if (x->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(x, bias, return false);
        } else if (biasDtype != op::DataType::DT_FLOAT){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The dtype of bias must be [DT_FLOAT] when x's dtype is [DT_BFLOAT16], but get: [%s].",
            op::ToString(biasDtype).GetString());
            return false;
        }
    }

    OP_CHECK_DTYPE_NOT_SAME(x, weight, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, y, return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                       bool isTransWeight, const aclTensor* y)
{
    OP_CHECK_WRONG_DIMENSION(x, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(weight, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(y, TWO_DIMS, return false);

    auto kdimX = x->GetViewShape().GetDim(1);
    auto kdimWeight = isTransWeight ? weight->GetViewShape().GetDim(1) : weight->GetViewShape().GetDim(0);
    if (kdimX != kdimWeight) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "The k-axis of x and weight should be same, but x's k-axis is: %ld and weight's k-axis is: %ld.", kdimX, kdimWeight);
        return false;
    }
    if (kdimX < KVALUE_MIN || kdimX > KVALUE_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
        "The k-axis should be in range[1, 65535], but it is: %ld.", kdimX);
        return false;
    }

    auto nVal = isTransWeight ? weight->GetViewShape().GetDim(0) : weight->GetViewShape().GetDim(1);
    if (bias != nullptr){
        OP_CHECK_WRONG_DIMENSION(bias, ONE_DIM, return false);
        auto biasDim = bias->GetViewShape().GetDim(0);
        if (biasDim != nVal) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The n-axis of weight and bias should be same, but weight's n-axis is: %ld and bias's n-axis is: %ld.", nVal, biasDim);
            return false;
        }
    }

    return true;
}

// 入参教验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, const aclTensor *y,
                               bool transpose_x2, const char* group)
{
    OP_LOGD("aclnn_matmul_allto_all CheckParams start");
    CHECK_RET(CheckNotNull(x, weight, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, weight, bias, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, weight, bias, transpose_x2, y), ACLNN_ERR_PARAM_INVALID);
    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnn_matmul_allto_all CheckParams success");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                                const aclIntArray* alltoAllAxesOptional, const char* group,
                                                bool transposeX1, bool transposeX2, const aclTensor* output,
                                                uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMatmulAlltoAllGetWorkspaceSize start");
    auto ret_param = CheckParams(x1, x2, biasOptional, output, transposeX2, group);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclTensor* x1_scale = nullptr;
    aclTensor* x2_scale = nullptr;
    aclTensor* comm_scale = nullptr;
    aclTensor* x1_offset = nullptr;
    aclTensor* x2_offset = nullptr;
    int64_t world_size = -1;
    int64_t y_dtype = output->GetDataType();
    int64_t x1_quant_mode = 0;
    int64_t x2_quant_mode = 0;
    int64_t comm_quant_mode = 0;
    int64_t comm_quant_dtype = 28;
    int64_t group_size = 0;

    aclnnStatus ret = aclnnInnerMatmulAlltoAllGetWorkspaceSize(x1, x2, biasOptional,
                                                               x1_scale, x2_scale, comm_scale, x1_offset, x2_offset,
                                                               group, world_size, alltoAllAxesOptional, 
                                                               y_dtype, x1_quant_mode, x2_quant_mode, comm_quant_mode, comm_quant_dtype,
                                                               transposeX1, transposeX2, group_size,
                                                               output, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAll, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    return ret;
}

aclnnStatus aclnnMatmulAlltoAll(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerMatmulAlltoAll(workspace, workspaceSize, executor, stream);
    return ret;
}
#ifdef __cplusplus
}
#endif