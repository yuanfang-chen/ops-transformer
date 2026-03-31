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
 * \file aclnn_all_gather_add.cpp
 * \brief 新算子 ACLNN 接口实现
 */

#include <cstring>

#include "aclnn_all_gather_add.h"
#include "securec.h"  // 若涉及字符串/内存安全，可按需保留
#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"

using namespace op;

namespace {
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

constexpr size_t K_EXPECTED_DIM_NUM = 2U;
constexpr int64_t K_A_ROWS = 1024;
constexpr int64_t K_A_COLS = 2048;
constexpr int64_t K_B_ROWS = 2048;
constexpr int64_t K_B_COLS = 2048;
constexpr int64_t K_OUTPUT_ROWS = 2048;
constexpr int64_t K_OUTPUT_COLS = 2048;
constexpr int64_t K_EXPECTED_RANK_SIZE = 2;
constexpr int64_t K_EXPECTED_COMM_TURN = 2;
constexpr size_t K_GROUP_NAME_LENGTH_MAX = 128U;

aclnnStatus CheckNotNull(const aclTensor* a, const aclTensor* b, const char* group, const aclTensor* aGathered,
                         const aclTensor* c, const uint64_t* workspaceSize, aclOpExecutor* const* executor)
{
    OP_CHECK_NULL(a, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(b, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aGathered, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(c, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AllGatherAdd, group is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

bool CheckGroup(const char* group)
{
    const size_t groupLen = strnlen(group, K_GROUP_NAME_LENGTH_MAX);
    if (groupLen == 0U || groupLen >= K_GROUP_NAME_LENGTH_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AllGatherAdd, group length must be in range (0, %zu), but got %zu.", K_GROUP_NAME_LENGTH_MAX,
                groupLen);
        return false;
    }
    return true;
}

bool CheckFormat(const aclTensor* tensor, const char* name)
{
    if (tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AllGatherAdd, %s format must be ND, but got %s.", name,
                op::ToString(tensor->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

bool CheckDType(const aclTensor* tensor, const char* name)
{
    if (tensor->GetDataType() != op::DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AllGatherAdd, %s dtype must be FLOAT16, but got %s.", name,
                op::ToString(tensor->GetDataType()).GetString());
        return false;
    }
    return true;
}

bool CheckShape(const aclTensor* tensor, const char* name, int64_t expectedRows, int64_t expectedCols)
{
    const auto &shape = tensor->GetViewShape();
    if (shape.GetDimNum() != K_EXPECTED_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AllGatherAdd, %s dim num must be %zu, but got %zu.", name,
                K_EXPECTED_DIM_NUM, shape.GetDimNum());
        return false;
    }
    if (shape.GetDim(0) != expectedRows || shape.GetDim(1) != expectedCols) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AllGatherAdd, %s shape must be [%ld, %ld], but got [%ld, %ld].", name,
                expectedRows, expectedCols, shape.GetDim(0), shape.GetDim(1));
        return false;
    }
    return true;
}

aclnnStatus CheckParams(const aclTensor* a, const aclTensor* b, const char* group, int64_t rankSize, int64_t commTurn,
                        const aclTensor* aGathered, const aclTensor* c, const uint64_t* workspaceSize,
                        aclOpExecutor* const* executor)
{
    const aclnnStatus notNullRet = CheckNotNull(a, b, group, aGathered, c, workspaceSize, executor);
    CHECK_RET(notNullRet == ACLNN_SUCCESS, notNullRet);
    CHECK_RET(CheckGroup(group), ACLNN_ERR_PARAM_INVALID);

    if (rankSize != K_EXPECTED_RANK_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AllGatherAdd, rankSize must be %ld, but got %ld.", K_EXPECTED_RANK_SIZE,
                rankSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (commTurn != K_EXPECTED_COMM_TURN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AllGatherAdd, current version only supports commTurn = %ld, but got %ld.",
                K_EXPECTED_COMM_TURN, commTurn);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckDType(a, "a") && CheckDType(b, "b") && CheckDType(aGathered, "aGathered") && CheckDType(c, "c"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(a, "a") && CheckFormat(b, "b") && CheckFormat(aGathered, "aGathered") && CheckFormat(c, "c"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(a, "a", K_A_ROWS, K_A_COLS), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(b, "b", K_B_ROWS, K_B_COLS), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(aGathered, "aGathered", K_OUTPUT_ROWS, K_OUTPUT_COLS), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(c, "c", K_OUTPUT_ROWS, K_OUTPUT_COLS), ACLNN_ERR_PARAM_INVALID);

    if (aGathered == c) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AllGatherAdd, aGathered and c must be different output tensors.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

// 可选: 匿名namespace定义静态Dtype支持表等辅助

extern aclnnStatus aclnnInnerAllGatherAddGetWorkspaceSize(const aclTensor* a, const aclTensor* b, const char* group,
                                                          int64_t rankSize, int64_t commTurn, aclTensor* aGathered,
                                                          aclTensor* c, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor);
extern aclnnStatus aclnnInnerAllGatherAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor* a, const aclTensor* b, const char* group,
                                              int64_t rankSize, int64_t commTurn, aclTensor* aGathered, aclTensor* c,
                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const aclnnStatus retParam = CheckParams(a, b, group, rankSize, commTurn, aGathered, c, workspaceSize, executor);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    return aclnnInnerAllGatherAddGetWorkspaceSize(a, b, group, rankSize, commTurn, aGathered, c,
                                                  workspaceSize, executor);
}

aclnnStatus aclnnAllGatherAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(stream, return ACLNN_ERR_PARAM_NULLPTR);
    if (workspace == nullptr && workspaceSize != 0UL) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AllGatherAdd, workspace is nullptr while workspaceSize is %lu.",
                workspaceSize);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    }

    const aclnnStatus ret = aclnnInnerAllGatherAdd(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "AllGatherAdd, launch task failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
