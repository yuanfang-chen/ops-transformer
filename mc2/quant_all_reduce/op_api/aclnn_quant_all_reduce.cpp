/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_all_reduce.cpp
 * \brief
 */
#include "aclnn_quant_all_reduce.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"

using namespace op;

namespace {
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// T-G量化支持的Dtype
static const std::initializer_list<op::DataType> X_DTYPE_TG_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2
};
static const std::initializer_list<op::DataType> SCALES_DTYPE_TG_SUPPORT_LIST = {
    op::DataType::DT_FLOAT
};

// MX量化支持的Dtype
static const std::initializer_list<op::DataType> X_DTYPE_MX_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2
};
static const std::initializer_list<op::DataType> SCALES_DTYPE_MX_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E8M0
};

// output支持的Dtype
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT
};

// 检查入参是否为nullptr
static bool  QuantAllReduceCheckNotNull(const aclTensor* x, const aclTensor* scales, const aclTensor* output)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(scales, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

// 检查T-G量化方案中x、scales、output的数据类型是否在算子的支持列表内
static bool  QuantAllReduceCheckTGAllDtypesValid(const aclTensor* x, const aclTensor* scales,
                                                 const aclTensor* output)
{
    if (CheckType(x->GetDataType(), X_DTYPE_TG_SUPPORT_LIST) &&             \
        CheckType(scales->GetDataType(), SCALES_DTYPE_TG_SUPPORT_LIST) &&   \
        CheckType(output->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
        return true;
    } else {
        return false;
    }
}

// 检查MX量化方案中x、scales、output的数据类型是否在算子的支持列表内
static bool  QuantAllReduceCheckMXAllDtypesValid(const aclTensor* x, const aclTensor* scales,
                                                 const aclTensor* output)
{
    if (CheckType(x->GetDataType(), X_DTYPE_MX_SUPPORT_LIST) &&             \
        CheckType(scales->GetDataType(), SCALES_DTYPE_MX_SUPPORT_LIST) &&   \
        CheckType(output->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
        return true;
    } else {
        return false;
    }
}

// 统一数据类型检查
static bool  QuantAllReduceCheckAllDtypesValid(const aclTensor* x, const aclTensor* scales,
                                               const aclTensor* output)
{
    bool isAllDtypesValid = false;
    isAllDtypesValid = (QuantAllReduceCheckTGAllDtypesValid(x, scales, output) || \
                        QuantAllReduceCheckMXAllDtypesValid(x, scales, output));
    if (!isAllDtypesValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,                                                            \
                "Input tensors x:%s, scales:%s and output:%s are not simultaneously supported.",    \
                op::ToString(x->GetDataType()).GetString(),                                         \
                op::ToString(scales->GetDataType()).GetString(),                                    \
                op::ToString(output->GetDataType()).GetString());
    }
    return isAllDtypesValid;
}

// 数据格式检查
static bool QuantAllReduceIsAllFormatND(aclFormat xFormat, aclFormat scalesFormat, aclFormat outputFormat)
{
    if (xFormat == aclFormat::ACL_FORMAT_ND &&      \
        scalesFormat == aclFormat::ACL_FORMAT_ND && \
        outputFormat == aclFormat::ACL_FORMAT_ND) {
        return true;
    } else {
        return false;
    }
}

static bool QuantAllReduceCheckAllFormatValid(const aclTensor* x, const aclTensor* scales,
                                              const aclTensor* output)
{
    aclFormat xFormat, scalesFormat, outputFormat;
    if (aclGetFormat(x, &xFormat) != ACLNN_SUCCESS) {
        OP_LOGD("QuantAllReduce, aclGetFormat failed for x !");
        return false;
    }
    if (aclGetFormat(scales, &scalesFormat) != ACLNN_SUCCESS) {
        OP_LOGD("QuantAllReduce, aclGetFormat failed for scales !");
        return false;
    }
    if (aclGetFormat(output, &outputFormat) != ACLNN_SUCCESS) {
        OP_LOGD("QuantAllReduce, aclGetFormat failed for output !");
        return false;
    }
    if (!QuantAllReduceIsAllFormatND(xFormat, scalesFormat, outputFormat)) {
        OP_LOGD("QuantAllReduce, Recieved tensor format is not ND !");
        return false;
    }
    return true;
}

// 参数综合校验
static aclnnStatus QuantAllReduceCheckParams(const aclTensor* x, const aclTensor* scales,
                                             const aclTensor* output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(QuantAllReduceCheckNotNull(x, scales, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(QuantAllReduceCheckAllDtypesValid(x, scales, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查参数数据格式是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(QuantAllReduceCheckAllFormatValid(x, scales, output), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
}

extern "C" aclnnStatus aclnnInnerQuantAllReduceGetWorkspaceSize(const aclTensor* x, const aclTensor* scales,
                                                                const char* group, const char* reduceOp,
                                                                uint64_t yDtype, aclTensor* output,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor);

extern "C" aclnnStatus aclnnInnerQuantAllReduce(void* workspace, uint64_t workspaceSize,
                                                aclOpExecutor* executor, const aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnQuantAllReduceGetWorkspaceSize(const aclTensor* x, const aclTensor* scales,
                                                           const char* group, const char* reduceOp,
                                                           aclTensor* output, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor)
{
    aclnnStatus retParam = QuantAllReduceCheckParams(x, scales, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerQuantAllReduceGetWorkspaceSize(x, scales, group, reduceOp, yDtype,
                                                               output, workspaceSize, executor);
    OP_LOGD("QuantAllReduce, aclnnnGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnQuantAllReduce(void* workspace, uint64_t workspaceSize,
                                           aclOpExecutor* executor, const aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerQuantAllReduce(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "QuantAllReduce, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
