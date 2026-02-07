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
static constexpr size_t MAX_BSK_LEN = 52428800U;
static constexpr size_t MAX_H1_LEN = 65536U;
static constexpr size_t MAX_EXPERT_SIZE = 256U;
static constexpr size_t MAX_E_SIZE = 32U;
static constexpr size_t MAX_H2_LEN = 12288U;
static constexpr size_t MAX_N_LEN = 65536U;
static constexpr size_t MIN_K_LEN = 2U;
static constexpr size_t MAX_K_LEN = 8U;

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScaleOptional, const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXScaleOptional,
    const aclTensor *mmWeightScaleOptional, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t gmmXQuantDtype,
    int64_t mmXQuantDtype, const aclTensor *gmmYOut, const aclTensor *mmYOutOptional,
    const aclTensor *permuteOutOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                            const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                            bool permuteOutFlag, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // // 检查必选入参出参为非空
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))) &&
        (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr)))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
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

// 检查必要输入是否为空/quantMode=1，必须非空/1
static bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                         const aclTensor *gmmXScaleOptional, const aclTensor *gmmWeightScaleOptional,
                         int64_t gmmXQuantMode, int64_t gmmWeightQuantMode)
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
    if (gmmXScaleOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmXScaleOptional should not be null.");
        return false;
    }
    if (gmmWeightScaleOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmXScaleOptional should not be null.");
        return false;
    }
    if (gmmXQuantMode != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmXQuantMode should be 1.");
        return false;
    }
    if (gmmWeightQuantMode != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gmmWeightQuantMode should be 1.");
        return false;
    }
    return true;
}

static bool CheckDimValid(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                     const aclTensor *gmmXScaleOptional, const aclTensor *gmmWeightScaleOptional,
                     const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const aclTensor *mmYOptional,
                     const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional)
{
    if ((gmmX != nullptr) && (gmmWeight != nullptr) && (gmmY != nullptr)) {
        if ((gmmX->GetViewShape().GetDimNum() != 2) && (gmmWeight->GetViewShape().GetDimNum() != 3) &&
            (gmmY->GetViewShape().GetDimNum() != 2)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dimensions of gmmX, gmmWeight and gmmY do not match.");
            return false;
        }
    }
    if ((gmmXScaleOptional != nullptr) && (gmmWeightScaleOptional != nullptr)) {
        if ((gmmXScaleOptional->GetViewShape().GetDimNum() != 1) && (gmmWeightScaleOptional->GetViewShape().GetDimNum() != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dimensions of gmmXScaleOptional and gmmWeightScaleOptional are not both one.");
            return false;
        }
    }
    if ((mmXOptional != nullptr) && (mmXScaleOptional != nullptr)) {
        if (((mmXOptional->GetViewShape().GetDimNum()) != 2) && ((mmXScaleOptional->GetViewShape().GetDimNum()) != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmXOptional dim is not two or mmXScaleOptional dim is not one.");
            return false;
        }
    }
    if ((mmWeightOptional != nullptr) && (mmWeightScaleOptional != nullptr)) {
        if (((mmWeightOptional->GetViewShape().GetDimNum()) != 2) && ((mmWeightScaleOptional->GetViewShape().GetDimNum()) != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeightOptional dim is not two.");
            return false;
        }
    }
    if (mmYOptional != nullptr) {
        if ((mmYOptional->GetViewShape().GetDimNum()) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmYOptional dim is not two.");
            return false;
        }
    }
    return true;
}

// 根据API定义，列出输入的所能支持的所有dtype
static const std::initializer_list<op::DataType> IN_DTYPE_SUPPORT_LIST = {op::DataType::DT_HIFLOAT8};
// 根据API定义，列出输入Scale所能支持的所有dtype
static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
// 根据API定义，列出输出output所能支持的所有dtype
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                           op::DataType::DT_BF16};
// 根据API定义，列出mode所能支持的所有dtype
static const std::initializer_list<op::DataType> QUANT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64};
// 根据API定义，列出sendcounts/recvCounts所能支持的所有dtype
static const std::initializer_list<op::DataType> COUNT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64}; 
static const std::initializer_list<op::DataType> EP_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64};                                                 
// 校验所有输入的参数类型是否正确
static bool CheckDtypesValid(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                             const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXOptional,
                             const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                             const aclTensor *mmWeightScaleOptional, const aclTensor *gmmY,
                             const aclTensor *mmYOptional, const aclTensor *permuteOutOptional, int64_t gmmXQuantMode,
                             int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                             const aclIntArray *sendCounts, const aclIntArray *recvCounts, int64_t epWorldSize)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmX, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeight, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmY, OUT_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(gmmXScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeightScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(gmmXQuantMode, QUANT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeightQuantMode, QUANT_DTYPE_SUPPORT_LIST, return false);

    if ((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr) &&
        (mmXScaleOptional != nullptr) && (mmWeightScaleOptional != nullptr)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(mmXOptional, IN_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmWeightOptional, IN_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmYOptional, OUT_DTYPE_SUPPORT_LIST, return false);

        OP_CHECK_DTYPE_NOT_SUPPORT(mmXScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmWeightScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);

        OP_CHECK_DTYPE_NOT_SUPPORT(mmXQuantMode, QUANT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmWeightQuantMode, QUANT_DTYPE_SUPPORT_LIST, return false);
    }

    if (permuteOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteOutOptional, IN_DTYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(sendCounts, COUNT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(recvCounts, COUNT_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(recvCounts, EP_DTYPE_SUPPORT_LIST, return false);
    return true;
}

// 检查暂不支持的输入参数是否为空，必须为空
static bool CheckNotSupportNull(const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
                                const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional,
                                int64_t gmmXQuantDType, int64_t mmXQuantDType)
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
    if (gmmXQuantDType != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input gmmXQuantDType should be 0.");
        return false;
    }
    if (mmXQuantDType != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input mmXQuantDType should be 0.");
        return false;
    }
    return true;
}


// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const aclTensor *mmYOptional)
{
    auto mVal = gmmX->GetViewShape().GetDim(0);
    auto kVal1 = gmmX->GetViewShape().GetDim(1);
    auto kVal2 = gmmWeight->GetViewShape().GetDim(1);
    auto nVal = gmmWeight->GetViewShape().GetDim(2);

    auto outmVal = gmmY->GetViewShape().GetDim(0);
    auto outnVal = gmmY->GetViewShape().GetDim(1);
    
    auto mmVal1 = mmXOptional->GetViewShape().GetDim(0);
    auto mnVal1 = mmXOptional->GetViewShape().GetDim(1);

    auto mmVal2 = mmWeightOptional->GetViewShape().GetDim(0);
    auto mnVal2 = mmWeightOptional->GetViewShape().GetDim(1);

    auto outmmVal = mmYOptional->GetViewShape().GetDim(0);
    auto outmnVal = mmYOptional->GetViewShape().GetDim(1);
    
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

    OP_API_CHECK((outmVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmY is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((outnVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmY is empty tensor with zero dimN, which is unsupported.");
        return false;
    });

    OP_API_CHECK(((!((mmVal1 != ZERO) && (mnVal1 != ZERO) && (mmVal2 != ZERO) && (mnVal2 != ZERO) && (outmmVal != ZERO) && (outmnVal != ZERO))) && 
        (!((mmVal1 == ZERO) && (mnVal1 == ZERO) && (mmVal2 == ZERO) && (mnVal2 == ZERO) && (outmmVal == ZERO) && (outmnVal == ZERO)))), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmXOptional, mmWeightOptional and mmYOptional should all be empty tensor or all not be empty tensor.");
        return false;
    });
    return true;
}

static bool CheckQuantValid(int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, const aclTensor *gmmXScaleOptional,
const aclTensor *gmmWeightScaleOptional, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                            const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional) {
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

// 检查维度
static bool CheckShape(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                       const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXOptional,
                       const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                       const aclTensor *mmWeightScaleOptional, const aclTensor *gmmY, const aclTensor *mmYOptional,
                       int64_t epWorldSize)
{
    if ((gmmX->GetViewShape().GetDim(0) == ZERO) || (gmmX->GetViewShape().GetDim(0) > MAX_BSK_LEN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of the first dimension of gmmX does not match.");
        return false;
    }
    if ((gmmX->GetViewShape().GetDim(1) == ZERO) || (gmmX->GetViewShape().GetDim(1) > MAX_H1_LEN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of the second dimension of gmmX does not match.");
        return false;
    }
    if ((((gmmWeight->GetViewShape().GetDim(0)) * epWorldSize) > MAX_EXPERT_SIZE) || ((gmmWeight->GetViewShape().GetDim(0)) > MAX_E_SIZE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the size of e does not match.");
        return false;
    }
    if (gmmWeight->GetViewShape().GetDim(1) != gmmX->GetViewShape().GetDim(1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of the second dimension of gmmWeight does not match.");
        return false;
    }
    if ((gmmWeight->GetViewShape().GetDim(2) == ZERO) || (gmmWeight->GetViewShape().GetDim(2) > MAX_N_LEN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of the third dimension of gmmWeight does not match.");
        return false;
    }
    if((gmmXScaleOptional->GetViewShape().GetDim(0) != 1) ||( gmmWeightScaleOptional->GetViewShape().GetDim(0) != 1)){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScaleOptional or gmmWeightScaleOptional do not match.");
        return false;
    }
    if((gmmY->GetViewShape().GetDim(0) != gmmX->GetViewShape().GetDim(0)) || (gmmY->GetViewShape().GetDim(1) != gmmWeight->GetViewShape().GetDim(2))){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of gmmY does not match.");
        return false;
    }
    if (mmXOptional != nullptr) {
        if ((mmXOptional->GetViewShape().GetDim(0) == ZERO) || (mmXOptional->GetViewShape().GetDim(0) == ZERO) ||
            (mmXOptional->GetViewShape().GetDim(1) > MAX_H2_LEN)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of mmX does not match.");
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if (((mmWeightOptional->GetViewShape().GetDim(0) != mmXOptional->GetViewShape().GetDim(1)) || (mmWeightOptional->GetViewShape().GetDim(1) == ZERO) ||
            (mmWeightOptional->GetViewShape().GetDim(1) > MAX_N_LEN)) || (mmXScaleOptional->GetViewShape().GetDim(0) != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of mmWeight  or mmXScaleOptional do not match.");
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if (((mmWeightOptional->GetViewShape().GetDim(0) != mmXOptional->GetViewShape().GetDim(1)) || (mmWeightOptional->GetViewShape().GetDim(1) == ZERO) ||
            (mmWeightOptional->GetViewShape().GetDim(1) > MAX_N_LEN)) || (mmWeightScaleOptional->GetViewShape().GetDim(0) != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of mmWeight or mmWeightScaleOptional do not match.");
            return false;
        }
    }
    if (mmYOptional != nullptr) {
        if ((mmYOptional->GetViewShape().GetDim(0) != mmXOptional->GetViewShape().GetDim(0)) ||
            (mmYOptional->GetViewShape().GetDim(1) != mmWeightOptional->GetViewShape().GetDim(1))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of mmYOptional does not match.");
            return false;
        }
    }
    return true;
}

// 检查所有要用到的输入format是否为ND，如果内部不为ND格式，会打印warning日志
static bool CheckFormat(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *mmXOptional,
                        const aclTensor *mmWeightOptional, const aclTensor *gmmY, const aclTensor *mmYOptional)
{
    // 输入格式只支持ND格式
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

static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                               const aclTensor *gmmWeightScaleOptional, const aclTensor *gmmXOffsetOptional,
                               const aclTensor *gmmWeightOffsetOptional, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
                               const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               int64_t gmmXQuantDType, int64_t mmXQuantDType, const aclIntArray *sendCounts,
                               const aclIntArray *recvCounts, const char *group, int64_t epWorldSize,
                               bool permuteOutFlag, const aclTensor *gmmY, const aclTensor *mmYOptional,
                               const aclTensor *permuteOutOptional)
{
    (void)epWorldSize; // Unused
    // 1.检查空状态
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,mmXScaleOptional,mmWeightScaleOptional
                              permuteOutFlag, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    //  检查group长度是否小于等于128
    CHECK_RET(allto_allv_grouped_mat_mul_checker::CheckGroup(group), ACLNN_ERR_PARAM_INVALID);
    // 3.检查参数是否为空
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY, gmmXScaleOptional, gmmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode), ACLNN_ERR_PARAM_INVALID);
    // 4.检查暂不支持的参数是否为空，不影响场景
    CHECK_RET(
        CheckNotSupportNull(gmmXOffsetOptional, gmmWeightOffsetOptional, mmXOffsetOptional, mmWeightOffsetOptional, gmmXQuantDType, mmXQuantDType),
        ACLNN_ERR_PARAM_INVALID);
    // 5.检查维度
    CHECK_RET(CheckDimValid(gmmX, gmmWeight, gmmY, gmmXScaleOptional, gmmWeightScaleOptional, mmXOptional, mmWeightOptional,
                       mmYOptional, mmXScaleOptional, mmWeightScaleOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 5.检查空tensor
    CHECK_RET(CheckNotEmptyTensor(gmmX, gmmWeight, gmmY, mmXOptional, mmWeightOptional, mmYOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查所有输入/量化数据类型
    CHECK_RET(CheckDtypesValid(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, mmXOptional,
                               mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmY, mmYOptional,
                               permuteOutOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, sendCounts, recvCounts, epWorldSize),
              ACLNN_ERR_PARAM_INVALID);
    // 8.检查Quant
    CHECK_RET(CheckQuantValid(gmmXQuantMode, gmmWeightQuantMode, gmmXScaleOptional, gmmWeightScaleOptional,
                              mmXQuantMode, mmWeightQuantMode, mmXScaleOptional, mmWeightScaleOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查shape
    CHECK_RET(CheckShape(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, mmXOptional, mmWeightOptional,
                         mmXScaleOptional, mmWeightScaleOptional, gmmY, mmYOptional, epWorldSize),
              ACLNN_ERR_PARAM_INVALID);
    // 6.检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(gmmX, gmmWeight, mmXOptional, mmWeightOptional, gmmY, mmYOptional), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("aclnnQuantMatmulAlltoAll checkParams success");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnQuantAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
    const aclTensor *gmmWeightScaleOptional, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
    const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t gmmXQuantDType, int64_t mmXQuantDType, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    bool permuteOutFlag, const aclTensor *gmmY, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret_param = CheckParams(
        gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, gmmXOffsetOptional, gmmWeightOffsetOptional,
        sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional,
        mmWeightScaleOptional, mmXOffsetOptional, mmWeightOffsetOptional, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, gmmXQuantDType, mmXQuantDType, sendCounts, recvCounts, group, epWorldSize,
        permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = allto_allv_grouped_mat_mul_checker::CheckSendAndRecv(sendCounts, recvCounts, gmmX, gmmY);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        gmmXScaleOptional, gmmWeightScaleOptional, mmXScaleOptional, mmWeightScaleOptional, group, epWorldSize,
        sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, gmmXQuantDType, mmXQuantDType, gmmY, mmYOptional, permuteOutOptional,
        workspaceSize, executor);
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
