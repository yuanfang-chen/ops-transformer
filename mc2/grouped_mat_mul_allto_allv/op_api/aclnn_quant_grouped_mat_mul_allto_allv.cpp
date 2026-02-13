/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "securec.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "acl/acl.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "hccl_util.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_quant_grouped_mat_mul_allto_allv.h"

namespace{
using namespace op;

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
    DYN_PERTOKEN_QUANT = 7
};


enum class NnopbaseHcclServerType : uint32_t { // HCCL Server
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

static constexpr int64_t ZERO = 0;

extern "C" aclnnStatus aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize( // Innner的参数要保持与def一致
    const aclTensor* gmmX, 
    const aclTensor* gmmWeight,
    const aclTensor* sendCountsTensorOptional, 
    const aclTensor* recvCountsTensorOptional,
    const aclTensor* mmXOptional, 
    const aclTensor* mmWeightOptional,
    const aclTensor* gmmXScaleOptional,
    const aclTensor* gmmWeightScaleOptional,
    const aclTensor* gmmXOffsetOptional,
    const aclTensor* gmmWeightOffsetOptional,
    const aclTensor* mmXScaleOptional,
    const aclTensor* mmWeightScaleOptional,
    const aclTensor* mmXOffsetOptional,
    const aclTensor* mmWeightOffsetOptional,
    const aclTensor* commQuantScaleOptional,
    const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    int64_t commQuantMode, int64_t groupSize, int64_t commQuantDtypeOptional,
    const aclTensor* yOut, const aclTensor* mmYOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor
);

extern "C" aclnnStatus aclnnInnerGroupedMatMulAlltoAllv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *y)
{
    if (gmmX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmX should not be null.");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmWeight should not be null.");
        return false;
    }
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "y should not be null.");
        return false;
    }
    return true;
}

// check nullptr
static bool CheckNullStatus(const aclTensor* gmmX, const aclTensor* gmmWeight,
                            const aclTensor* sendCountsTensorOptional,
                            const aclTensor* recvCountsTensorOptional,
                            const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, 
                            const aclTensor* mmXScaleOptional, const char* group,
                            bool transGmmWeight, bool transMmWeight, 
                            const aclTensor* y, const aclTensor* mmYOptional)
{
    (void)transGmmWeight;
    (void)transMmWeight;
    
    // 如果scale不是optional，也需要进行判断非空的操作？
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {  
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;  // 无需修改
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {   // HCCL_GROUP_NAME_MAX = 128U
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }     // 无需修改
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, "
                "left: %u, right: %u, mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is "
                "nullptr: %u",
                (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))),
                (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))),
                mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    return true;
}

// 检查暂不支持的输入参数是否为空，必须为空 
static bool CheckNotSupportNull(const aclTensor* gmmXOffsetOptional, const aclTensor* gmmWeightOffsetOptional,
                                const aclTensor* mmXOffsetOptional, const aclTensor* mmWeightOffsetOptional) 
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

// check send and recv
static aclnnStatus CheckSendAndRecv(const aclIntArray* sendCounts, const aclIntArray* recvCounts)
{
    if (sendCounts == nullptr) {  // 通信发送的数据量
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {  // 接收发送的数据量
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U;  // recvCounts的大小
    uint64_t sendSize = 0U;  // recvCounts的大小
    aclGetIntArraySize(recvCounts, &recvSize);
    aclGetIntArraySize(sendCounts, &sendSize);
    if (recvSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sendSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

//检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor* gmmX, const aclTensor* gmmWeight,
                                const aclTensor* y,
                                const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
                                const aclTensor* mmYOptional) {
    auto h1Val = gmmX->GetViewShape().GetDim(1);

    auto h1Val2 = gmmWeight->GetViewShape().GetDim(1);
    auto n1Val = gmmWeight->GetViewShape().GetDim(2);

    auto yMval = y->GetViewShape().GetDim(0);
    auto yNval = y->GetViewShape().GetDim(1);

    OP_API_CHECK((h1Val == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "gmmX is empty tensor with zero dimN, which is unsupported.");
      return false;
    });
    OP_API_CHECK((h1Val2 == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "gmmWeight is empty tensor with zero dimN, which is unsupported.");
      return false;
    });
    OP_API_CHECK((n1Val == ZERO), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "gmmWeight is empty tensor with zero dimK, which is unsupported.");
      return false;
    });
    OP_API_CHECK((yMval == ZERO), {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y is empty tensor with zero dimM, which is unsupported.");
            return false;
    });
    OP_API_CHECK((yNval == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y is empty tensor with zero dimN, which is unsupported.");
        return false;
    });

    if (mmXOptional == nullptr || mmWeightOptional == nullptr || mmYOptional == nullptr) {
        return true;
    }

    auto mmDim0 = mmXOptional->GetViewShape().GetDim(0);
    auto mmDim1 = mmXOptional->GetViewShape().GetDim(1);

    auto mmWdim0 = mmWeightOptional->GetViewShape().GetDim(0);
    auto mmWdim1 = mmWeightOptional->GetViewShape().GetDim(1);

    auto mmYdim0 = mmYOptional->GetViewShape().GetDim(0);
    auto mmYdim1 = mmYOptional->GetViewShape().GetDim(1);

    OP_API_CHECK(
        ((!((mmDim0 != ZERO) && (mmDim1 != ZERO) && (mmWdim0 != ZERO) && (mmWdim1 != ZERO) && (mmYdim0 != ZERO) &&
            (mmYdim1 != ZERO))) &&
        (!((mmDim0 == ZERO) && (mmDim1 == ZERO) && (mmWdim0 == ZERO) && (mmWdim1 == ZERO) && (mmYdim0 == ZERO) &&
            (mmYdim1 == ZERO)))),
        {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional and mmYOptional should all be empty tensor or all not be empty tensor.");
            return false;
        });

    return true;
}

// 检查所有要用到的format是否为ND，不支持私有格式，如果内部不为ND格式，打印warning日志，将format转换为ND格式
static bool CheckFormat(const aclTensor* gmmX, const aclTensor* gmmWeight, 
                        const aclTensor* gmmXScaleOptional, const aclTensor* gmmWeightScaleOptional,
                        const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
                        const aclTensor* y, const aclTensor* mmYOptional)
{
    // 输入格式不支持私有格式
    if (IsPrivateFormat(gmmX->GetStorageFormat())) { //
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, gmmX format %s does not support private format.",
                op::ToString(gmmX->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(gmmWeight->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, gmmWeight format %s does not support private format.",
                op::ToString(gmmWeight->GetStorageFormat()).GetString());
        return false;
    }
    if (gmmXScaleOptional != nullptr) {
        if (IsPrivateFormat(gmmXScaleOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, gmmXScaleOptional format %s does not support private format.",
                op::ToString(gmmXScaleOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (gmmWeightScaleOptional != nullptr) {
        if (IsPrivateFormat(gmmWeightScaleOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, gmmWeightScaleOptional format %s does not support private format.",
                op::ToString(gmmWeightScaleOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (mmXOptional != nullptr) {
        if (IsPrivateFormat(mmXOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, mmXOptional format %s does not support private format.",
                op::ToString(mmXOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if (IsPrivateFormat(mmWeightOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, mmWeightOptional format %s does not support private format.",
                op::ToString(mmWeightOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(y->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, y format %s does not support private format.",
                op::ToString(y->GetStorageFormat()).GetString());
        return false;
    }
    if (mmYOptional != nullptr) {
        if (IsPrivateFormat(mmYOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantGroupMatmulAlltoAll, mmYOptional format %s does not support private format.",
                op::ToString(mmYOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor* gmmX, const aclTensor* gmmWeight, 
                        const aclTensor* gmmXScaleOptional, const aclTensor* gmmWeightScaleOptional,
                        const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
                        const aclTensor* y, const aclTensor* mmYOptional)
{
    // 内部只处理ND格式，这里做reformat操作
    if (gmmX->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("gmmX origin format is %s.", op::ToString(gmmX->GetStorageFormat()).GetString());
        gmmX = l0op::ReFormat(gmmX, op::Format::FORMAT_ND);
        CHECK_RET(gmmX != nullptr, false);
    }
    if (gmmWeight->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("gmmWeight origin format is %s.", op::ToString(gmmWeight->GetStorageFormat()).GetString());
        gmmWeight = l0op::ReFormat(gmmWeight, op::Format::FORMAT_ND);
        CHECK_RET(gmmWeight != nullptr, false);
    }
    if (gmmXScaleOptional != nullptr) {
        if (gmmXScaleOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("gmmXScaleOptional origin format is %s.", op::ToString(gmmXScaleOptional->GetStorageFormat()).GetString());
            gmmXScaleOptional = l0op::ReFormat(gmmXScaleOptional, op::Format::FORMAT_ND);
            CHECK_RET(gmmXScaleOptional != nullptr, false);
        }
    }
    if (gmmWeightScaleOptional != nullptr) {
        if (gmmWeightScaleOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("gmmWeightScaleOptional origin format is %s.", op::ToString(gmmWeightScaleOptional->GetStorageFormat()).GetString());
            gmmWeightScaleOptional = l0op::ReFormat(gmmWeightScaleOptional, op::Format::FORMAT_ND);
            CHECK_RET(gmmWeightScaleOptional != nullptr, false);
        }
    }
    if (mmXOptional != nullptr) {
        if (mmXOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("mmXOptional origin format is %s.", op::ToString(mmXOptional->GetStorageFormat()).GetString());
            mmXOptional = l0op::ReFormat(mmXOptional, op::Format::FORMAT_ND);
            CHECK_RET(mmXOptional != nullptr, false);
        }
    }
    if (mmWeightOptional != nullptr) {
        if (mmWeightOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("mmWeightOptional origin format is %s.", op::ToString(mmWeightOptional->GetStorageFormat()).GetString());
            mmWeightOptional = l0op::ReFormat(mmWeightOptional, op::Format::FORMAT_ND);
            CHECK_RET(mmWeightOptional != nullptr, false);
        }
    }
    if (y->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("y origin format is %s.", op::ToString(y->GetStorageFormat()).GetString());
        y = l0op::ReFormat(y, op::Format::FORMAT_ND);
        CHECK_RET(y != nullptr, false);
    }
    if (mmYOptional != nullptr) {
        if (mmYOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("mmYOptional origin format is %s.", op::ToString(mmYOptional->GetStorageFormat()).GetString());
            mmYOptional = l0op::ReFormat(mmYOptional, op::Format::FORMAT_ND);
            CHECK_RET(mmYOptional != nullptr, false);
        }
    }
    return true;
}

// 检查量化参数是否合法
static bool CheckQuantMode(int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, 
                            int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                            const aclTensor *gmmXScaleOptional,
 	                        const aclTensor *gmmWeightScaleOptional,
 	                        const aclTensor *mmXScaleOptional, 
                            const aclTensor *mmWeightScaleOptional) 
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
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mmWeightScaleOptional should not be null.");
            return false;
        }
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* gmmXScaleOptional,
                              const aclTensor* gmmWeightScaleOptional, const aclTensor* gmmXOffsetOptional,
                              const aclTensor* gmmWeightOffsetOptional,
                              const aclTensor* sendCountsTensorOptional, const aclTensor* recvCountsTensorOptional,
                              const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
                              const aclTensor* mmXScaleOptional, const aclTensor* mmWeightScaleOptional,
                              const aclTensor* mmXOffsetOptional, const aclTensor* mmWeightOffsetOptional,
                              int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
                              int64_t mmWeightQuantMode, int64_t commQuantMode, const char* group,
                              int64_t epWorldSize,
                              const aclIntArray* sendCounts, const aclIntArray* recvCounts, bool transGmmWeight,
                              bool transMmWeight, const aclTensor* y, const aclTensor* mmYOptional,
                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    (void)epWorldSize;
    (void)sendCounts;
    (void)recvCounts;
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
                              mmWeightOptional, mmXScaleOptional,
                              group, transGmmWeight, transMmWeight, y, mmYOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckNotSupportNull(gmmXOffsetOptional, gmmWeightOffsetOptional, mmXOffsetOptional, mmWeightOffsetOptional),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotEmptyTensor(gmmX, gmmWeight, y, mmXOptional, mmWeightOptional,
                                    mmYOptional), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckQuantMode(gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode,
                            gmmXScaleOptional, gmmWeightScaleOptional, mmXScaleOptional, mmWeightScaleOptional),
                            ACLNN_ERR_PARAM_INVALID);
    if (commQuantMode != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "commQuantMode only supports 0, but got %ld.", commQuantMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnnQuantMatmulAlltoAll checkParams success");

    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* gmmXScaleOptional,
    const aclTensor* gmmWeightScaleOptional,
    const aclTensor* gmmXOffsetOptional,
    const aclTensor* gmmWeightOffsetOptional,
    const aclTensor* sendCountsTensorOptional,
    const aclTensor* recvCountsTensorOptional, 
    const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, const aclTensor* mmXScaleOptional,
    const aclTensor* mmWeightScaleOptional, const aclTensor* mmXOffsetOptional,
    const aclTensor* mmWeightOffsetOptional, const aclTensor* commQuantScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    int64_t commQuantMode, int64_t commQuantDtypeOptional, 
    // 规避cc文件编译问题
    int64_t groupSize,
    const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, 
    const aclTensor* y, const aclTensor* mmYOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret_param =
        CheckParams(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, gmmXOffsetOptional,
                    gmmWeightOffsetOptional,
                    sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                    mmXScaleOptional, mmWeightScaleOptional, mmXOffsetOptional, mmWeightOffsetOptional,
                    gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, commQuantMode,
                    group, epWorldSize,
                    sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional, workspaceSize,
                    executor);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    char* str_group = const_cast<char*>(group);
    // 规避cc文件编译问题
    // int64_t groupSize = 0;
    // int64_t gmmYDtype = 28;
    // int64_t mmYDtype = 28;

    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
            gmmX, gmmWeight,
            sendCountsTensorOptional,
            recvCountsTensorOptional,
            mmXOptional, mmWeightOptional,
            gmmXScaleOptional, gmmWeightScaleOptional,
            gmmXOffsetOptional, gmmWeightOffsetOptional,
            mmXScaleOptional, mmWeightScaleOptional,
            mmXOffsetOptional, mmWeightOffsetOptional, commQuantScaleOptional,
            str_group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
            gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, commQuantMode,
            groupSize, commQuantDtypeOptional,
            y, mmYOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnQuantGroupedMatMulAlltoAllv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND950) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
    return ret;
}
}