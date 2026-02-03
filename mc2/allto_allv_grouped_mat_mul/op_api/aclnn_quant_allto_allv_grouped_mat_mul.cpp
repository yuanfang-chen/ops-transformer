/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#include "aclnn_quant_allto_allv_grouped_mat_mul.h"
#include "allto_allv_grouped_mat_mul_checker.h"
#include <algorithm>
#include "securec.h"
#include "op_mc2.h"
#include "acl/acl.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "hccl_util.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
namespace {
using namespace op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
};

// 需要使用的常量定义
static constexpr int64_t ZERO = 0;

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor *gmmX,
    const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
    const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const aclTensor *gmmXScaleOptional,
    const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t gmmXQuantDtype, int64_t mmXQuantDtype,
    const aclTensor *gmmYOut, const aclTensor *mmYOutOptional, const aclTensor *permuteOutOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY)
{
    if (gmmX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmX should not be null.");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmWeight should not be null.");
        return false;
    }
    if (gmmY == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmY should not be null.");
        return false;
    }
    return true;
}

// 检查暂不支持的输入参数是否为空，必须为空
static bool CheckNotSupportNull(const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional)
{
    if (gmmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmXOffsetOptional should be null.");
        return false;
    }
    if (gmmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmWeightOffsetOptional should be null.");
        return false;
    }
    if (mmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input mmXOffsetOptional should be null.");
        return false;
    }
    if (mmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input mmWeightOffsetOptional should be null.");
        return false;
    }
    return true;
}

// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *gmmX, const aclTensor *gmmWeight)
{
    auto mVal = gmmX->GetViewShape().GetDim(0);
    auto kVal1 = gmmX->GetViewShape().GetDim(1);
    auto kVal2 = gmmWeight->GetViewShape().GetDim(1);
    auto nVal = gmmWeight->GetViewShape().GetDim(2);
    OP_API_CHECK((mVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX is empty tensor with zero dimM, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal1 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((nVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    return true;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志
static bool CheckFormat(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *gmmY, const aclTensor *mmYOptional)
{
    // 输入格式不支持私有格式
    if (IsPrivateFormat(gmmX->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnQuantAlltoAllVGroupMatmul, gmmX format %s does not support private format.",
            op::ToString(gmmX->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(gmmWeight->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnQuantAlltoAllVGroupMatmul, gmmWeight format %s does not support private format.",
            op::ToString(gmmWeight->GetStorageFormat()).GetString());
        return false;
    }
    if (mmXOptional != nullptr) {
        if (IsPrivateFormat(mmXOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantAlltoAllVGroupMatmul, mmXOptional format %s does not support private format.",
                op::ToString(mmXOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if (IsPrivateFormat(mmWeightOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantAlltoAllVGroupMatmul, mmWeightOptional format %s does not support private format.",
                op::ToString(mmWeightOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(gmmY->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnQuantAlltoAllVGroupMatmul, gmmY format %s does not support private format.",
            op::ToString(gmmY->GetStorageFormat()).GetString());
        return false;
    }
    if (mmYOptional != nullptr) {
        if (IsPrivateFormat(mmYOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantAlltoAllVGroupMatmul, mmYOptional format %s does not support private format.",
                op::ToString(mmYOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
    const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const char *group, bool permuteOutFlag,
    const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // // 检查必选入参出参为非空
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }

    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, left: %u, right: %u, "
            "mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is nullptr: %u",
            (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))),
            (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))),
            mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional output flag does not match optional output ptr!");
        return false;
    }
    return true;
}

static bool CheckQuantValid(int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, const aclTensor *gmmXScaleOptional,
    const aclTensor *gmmWeightScaleOptional, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional)
{
    if (static_cast<QuantModeType>(gmmXQuantMode) == QuantModeType::NO_QUANT) {
        if ((gmmXScaleOptional != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmXScaleOptional should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(gmmXQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((gmmXScaleOptional == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmXScaleOptional not be null.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(gmmWeightQuantMode) == QuantModeType::NO_QUANT) {
        if ((gmmWeightScaleOptional != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmWeightScaleOptional should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(gmmWeightQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((gmmWeightScaleOptional == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmWeightScaleOptional not be null.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmXQuantMode) == QuantModeType::NO_QUANT) {
        if ((mmXScaleOptional != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mmXScaleOptional should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmXQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((mmXScaleOptional == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mmXScaleOptional not be null.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmWeightQuantMode) == QuantModeType::NO_QUANT) {
        if ((mmWeightScaleOptional != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mmWeightScaleOptional should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmWeightQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((mmWeightScaleOptional == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mmWeightScaleOptional not be null.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
    const aclTensor *gmmWeightScaleOptional, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
    const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t gmmXQuantDType, int64_t mmXQuantDType, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, const char *group, int64_t epWorldSize,
    bool permuteOutFlag, const aclTensor *gmmY, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    (void)epWorldSize; // Unused
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group,
        permuteOutFlag, mmYOptional, permuteOutOptional),
        ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 检查参数是否为空指针√
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY), ACLNN_ERR_PARAM_NULLPTR);

    // 检查暂不支持的参数是否为空，不影响场景√
    CHECK_RET(
        CheckNotSupportNull(gmmXOffsetOptional, gmmWeightOffsetOptional, mmXOffsetOptional, mmWeightOffsetOptional),
        ACLNN_ERR_PARAM_INVALID);

    // 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(gmmX, gmmWeight), ACLNN_ERR_PARAM_INVALID);

    // 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(gmmX, gmmWeight, mmXOptional, mmWeightOptional, gmmY, mmYOptional), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckQuantValid(gmmXQuantMode, gmmWeightQuantMode, gmmXScaleOptional, gmmWeightScaleOptional,
        mmXQuantMode, mmWeightQuantMode, mmXScaleOptional, mmWeightScaleOptional),
        ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("aclnnQuantMatmulAlltoAll checkParams success");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnQuantAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor *gmmX,
    const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional, const aclTensor *gmmWeightScaleOptional,
    const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t gmmXQuantDType,
    int64_t mmXQuantDType, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, const aclTensor *gmmY,
    const aclTensor *mmYOptional, const aclTensor *permuteOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    aclnnStatus ret_param = CheckParams(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, gmmXOffsetOptional,
        gmmWeightOffsetOptional, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        mmXScaleOptional, mmWeightScaleOptional, mmXOffsetOptional, mmWeightOffsetOptional, gmmXQuantMode,
        gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, gmmXQuantDType, mmXQuantDType, sendCounts, recvCounts,
        transGmmWeight, transMmWeight, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = allto_allv_grouped_mat_mul_checker::CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(gmmX, gmmWeight, sendCountsTensorOptional,
        recvCountsTensorOptional, mmXOptional, mmWeightOptional, gmmXScaleOptional, gmmWeightScaleOptional,
        mmXScaleOptional, mmWeightScaleOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
        transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode,
        gmmXQuantDType, mmXQuantDType, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnQuantAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND950) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace
