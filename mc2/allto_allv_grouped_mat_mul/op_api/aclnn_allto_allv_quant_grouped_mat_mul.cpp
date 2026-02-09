/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#include "aclnn_allto_allv_quant_grouped_mat_mul.h"
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

static bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, bool permuteOutFlag,
                            const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // // 检查必选入参出参为非空
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
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
                         const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, int64_t gmmXQuantMode,
                         int64_t gmmWeightQuantMode)
{
    if (gmmX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmX should not be null.");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmWeight should not be null.");
        return false;
    }
    if (gmmY == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmY should not be null.");
        return false;
    }
    if (gmmXScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScale should not be null.");
        return false;
    }
    if (gmmWeightScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightScale should not be null.");
        return false;
    }
    if (gmmXQuantMode != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXQuantMode should be 1.");
        return false;
    }
    if (gmmWeightQuantMode != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightQuantMode should be 1.");
        return false;
    }
    return true;
}

static bool CheckDimValid(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                          const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, const aclTensor *mmXOptional,
                          const aclTensor *mmWeightOptional, const aclTensor *mmYOptional,
                          const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional)
{
    if ((gmmX != nullptr) && (gmmWeight != nullptr) && (gmmY != nullptr)) {
        if ((gmmX->GetViewShape().GetDimNum() != 2) && (gmmWeight->GetViewShape().GetDimNum() != 3) &&
            (gmmY->GetViewShape().GetDimNum() != 2)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dimensions of gmmX, gmmWeight and gmmY do not match.");
            return false;
        }
    }
    if ((gmmXScale != nullptr) && (gmmWeightScale != nullptr)) {
        if ((gmmXScale->GetViewShape().GetDimNum() != 1) && (gmmWeightScale->GetViewShape().GetDimNum() != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dimensions of gmmXScale and gmmWeightScale are not both one.");
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
        if (((mmWeightOptional->GetViewShape().GetDimNum()) != 2) &&
            ((mmWeightScaleOptional->GetViewShape().GetDimNum()) != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeightOptional dim is not two.");
            return false;
        }
    }
    if (mmYOptional != nullptr) {
        if ((mmYOptional->GetViewShape().GetDimNum()) != 2) {
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
// 校验所有输入的参数类型是否正确
static bool CheckDtypesValid(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                             const aclTensor *gmmWeightScale, const aclTensor *mmXOptional,
                             const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                             const aclTensor *mmWeightScaleOptional, const aclTensor *gmmY,
                             const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmX, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeight, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmY, OUT_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(gmmXScale, SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeightScale, SCALE_DTYPE_SUPPORT_LIST, return false);

    if ((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr) &&
        (mmXScaleOptional != nullptr) && (mmWeightScaleOptional != nullptr)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(mmXOptional, IN_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmWeightOptional, IN_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmYOptional, OUT_DTYPE_SUPPORT_LIST, return false);

        OP_CHECK_DTYPE_NOT_SUPPORT(mmXScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmWeightScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);
    }

    if (permuteOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteOutOptional, IN_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

// 检查暂不支持的输入参数是否为空，必须为空
static bool CheckNotSupportNull(const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
                                const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional)
{
    if (gmmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmXOffsetOptional should be null.");
        return false;
    }
    if (gmmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmWeightOffsetOptional should be null.");
        return false;
    }
    if (mmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input mmXOffsetOptional should be null.");
        return false;
    }
    if (mmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input mmWeightOffsetOptional should be null.");
        return false;
    }
    return true;
}


// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY)
{
    auto mVal = gmmX->GetViewShape().GetDim(0);
    auto kVal1 = gmmX->GetViewShape().GetDim(1);
    auto kVal2 = gmmWeight->GetViewShape().GetDim(1);
    auto nVal = gmmWeight->GetViewShape().GetDim(2);

    auto outmVal = gmmY->GetViewShape().GetDim(0);
    auto outnVal = gmmY->GetViewShape().GetDim(1);

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
    return true;
}

static bool CheckQuantValid(int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, const aclTensor *gmmXScale,
                            const aclTensor *gmmWeightScale, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                            const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional)
{
    if (static_cast<QuantModeType>(gmmXQuantMode) == QuantModeType::NO_QUANT) {
        if ((gmmXScale != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScale should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(gmmXQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((gmmXScale == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScale not be null.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(gmmWeightQuantMode) == QuantModeType::NO_QUANT) {
        if ((gmmWeightScale != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightScale should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(gmmWeightQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((gmmWeightScale == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightScale not be null.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmXQuantMode) == QuantModeType::NO_QUANT) {
        if ((mmXScaleOptional != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmXScaleOptional should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmXQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((mmXScaleOptional == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmXScaleOptional not be null.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmWeightQuantMode) == QuantModeType::NO_QUANT) {
        if ((mmWeightScaleOptional != nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeightScaleOptional should be empty.");
            return false;
        }
    }
    if (static_cast<QuantModeType>(mmWeightQuantMode) == QuantModeType::PERTENSOR_QUANT) {
        if ((mmWeightScaleOptional == nullptr)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeightScaleOptional not be null.");
            return false;
        }
    }
    return true;
}

bool is_power_of_two(int64_t n)
{
    return n > 0 && (n & (n - 1)) == 0 && n >= 2 && n <= 128;
}

// 检查shape大小
static bool CheckGmmShape(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                          const aclTensor *gmmWeightScale, const aclTensor *gmmY, int64_t epWorldSize)
{
    if ((gmmX->GetViewShape().GetDim(0) == ZERO) || (gmmX->GetViewShape().GetDim(0) > MAX_BSK_LEN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of the first dimension of gmmX does not match.");
        return false;
    }
    if ((gmmX->GetViewShape().GetDim(1) == ZERO) || (gmmX->GetViewShape().GetDim(1) > MAX_H1_LEN)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of the second dimension of gmmX does not match.");
        return false;
    }
    if ((((gmmWeight->GetViewShape().GetDim(0)) * epWorldSize) > MAX_EXPERT_SIZE) ||
        ((gmmWeight->GetViewShape().GetDim(0)) > MAX_E_SIZE) || (!((gmmWeight->GetViewShape().GetDim(0)) > ZERO))) {
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
    if ((gmmXScale->GetViewShape().GetDim(0) != 1) || (gmmWeightScale->GetViewShape().GetDim(0) != 1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScale or gmmWeightScale do not match.");
        return false;
    }
    if ((gmmY->GetViewShape().GetDim(0) != gmmX->GetViewShape().GetDim(0)) ||
        (gmmY->GetViewShape().GetDim(1) != gmmWeight->GetViewShape().GetDim(2))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of gmmY does not match.");
        return false;
    }

    if (!(is_power_of_two(epWorldSize))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the size of epWorldSize does not match.");
        return false;
    }
    return true;
}

static bool CheckMmShape(const aclTensor *gmmX, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                       const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                       const aclTensor *mmYOptional)
{
    if (mmXOptional != nullptr) {
        auto k1 = (gmmX->GetViewShape().GetDim(0)) % (mmXOptional->GetViewShape().GetDim(0));
        auto k2 = (gmmX->GetViewShape().GetDim(0)) / (mmXOptional->GetViewShape().GetDim(0));
        if ((mmXOptional->GetViewShape().GetDim(0) == ZERO) || (mmXOptional->GetViewShape().GetDim(0) == ZERO) ||
            (mmXOptional->GetViewShape().GetDim(1) > MAX_H2_LEN)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of mmX does not match.");
            return false;
        }
        if ((k1 != ZERO) || (!(k2 >= MIN_K_LEN && k2 <= MAX_K_LEN))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the size of k does not match.");
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if (((mmWeightOptional->GetViewShape().GetDim(0) != mmXOptional->GetViewShape().GetDim(1)) ||
             (mmWeightOptional->GetViewShape().GetDim(1) == ZERO) ||
             (mmWeightOptional->GetViewShape().GetDim(1) > MAX_N_LEN)) ||
            (mmXScaleOptional->GetViewShape().GetDim(0) != 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of mmWeight  or mmXScaleOptional do not match.");
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if (((mmWeightOptional->GetViewShape().GetDim(0) != mmXOptional->GetViewShape().GetDim(1)) ||
             (mmWeightOptional->GetViewShape().GetDim(1) == ZERO) ||
             (mmWeightOptional->GetViewShape().GetDim(1) > MAX_N_LEN)) ||
            (mmWeightScaleOptional->GetViewShape().GetDim(0) != 1)) {
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
static bool CheckFormat(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                        const aclTensor *gmmWeightScale, const aclTensor *mmXOptional,
                        const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                        const aclTensor *mmWeightScaleOptional, const aclTensor *gmmY, const aclTensor *mmYOptional)
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
    if ((IsPrivateFormat(gmmXScale->GetStorageFormat())) || (IsPrivateFormat(gmmWeightScale->GetStorageFormat()))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "aclnnQuantAlltoAllVGroupMatmul, gmmXScale or gmmWeightScale format %s does not support private format.",
            op::ToString(gmmXScale->GetStorageFormat()).GetString());
        return false;
    }
    if (mmXOptional != nullptr) {
        if ((IsPrivateFormat(mmXOptional->GetStorageFormat())) ||
            (IsPrivateFormat(mmXScaleOptional->GetStorageFormat()))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantAlltoAllVGroupMatmul, mmXOptional or mmXScaleOptional format %s does not support "
                    "private format.",
                    op::ToString(mmXOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if ((IsPrivateFormat(mmWeightOptional->GetStorageFormat())) ||
            (IsPrivateFormat(mmWeightScaleOptional->GetStorageFormat()))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantAlltoAllVGroupMatmul, mmWeightOptional or mmWeightScaleOptional format %s does not "
                    "support private format.",
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

static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                               const aclTensor *gmmWeightScale, const aclTensor *gmmXOffsetOptional,
                               const aclTensor *gmmWeightOffsetOptional, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
                               const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               const char *group, int64_t epWorldSize, bool permuteOutFlag, const aclTensor *gmmY,
                               const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    (void)epWorldSize; // Unused
    // 1.检查空状态
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                              permuteOutFlag, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    //  检查group长度是否小于等于128
    CHECK_RET(allto_allv_grouped_mat_mul_checker::CheckGroup(group), ACLNN_ERR_PARAM_INVALID);
    // 3.检查参数是否为空
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY, gmmXScale, gmmWeightScale, gmmXQuantMode, gmmWeightQuantMode),
              ACLNN_ERR_PARAM_INVALID);
    // 4.检查暂不支持的参数是否为空，不影响场景
    CHECK_RET(
        CheckNotSupportNull(gmmXOffsetOptional, gmmWeightOffsetOptional, mmXOffsetOptional, mmWeightOffsetOptional),
        ACLNN_ERR_PARAM_INVALID);
    // 5.检查维度
    CHECK_RET(CheckDimValid(gmmX, gmmWeight, gmmY, gmmXScale, gmmWeightScale, mmXOptional, mmWeightOptional,
                            mmYOptional, mmXScaleOptional, mmWeightScaleOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 5.检查空tensor
    CHECK_RET(CheckNotEmptyTensor(gmmX, gmmWeight, gmmY), ACLNN_ERR_PARAM_INVALID);
    // 检查所有输入/量化数据类型
    CHECK_RET(CheckDtypesValid(gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmXOptional, mmWeightOptional,
                               mmXScaleOptional, mmWeightScaleOptional, gmmY, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 8.检查Quant
    CHECK_RET(CheckQuantValid(gmmXQuantMode, gmmWeightQuantMode, gmmXScale, gmmWeightScale, mmXQuantMode,
                              mmWeightQuantMode, mmXScaleOptional, mmWeightScaleOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查shape
    CHECK_RET(CheckGmmShape(gmmX, gmmWeight, gmmXScale, gmmWeightScale, gmmY, epWorldSize), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckMmShape(gmmX, mmXOptional, mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, mmYOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 6.检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmXOptional, mmWeightOptional, mmXScaleOptional,
                          mmWeightScaleOptional, gmmY, mmYOptional),
              ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("aclnnQuantMatmulAlltoAll checkParams success");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    bool permuteOutFlag, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype, int64_t mmDtype, const aclTensor *gmmY,
    const aclTensor *mmYOptional, const aclTensor *permuteOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    bool permuteOutFlag, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t groupSize, const aclTensor *gmmY, const aclTensor *mmYOptional,
    const aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    int64_t yDtype = gmmY->GetDataType(); // yDtype根据实际output的类型赋值，图模式需要该参数
    int64_t mmDtype = mmYOptional->GetDataType();

    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, gmmXScale,
        gmmWeightScale, gmmXOffsetOptional, gmmWeightOffsetOptional, mmXScaleOptional, mmWeightScaleOptional,
        mmXOffsetOptional, mmWeightOffsetOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
        transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, groupSize,
        yDtype, mmDtype, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    int64_t groupSize, bool permuteOutFlag, const aclTensor *gmmY, const aclTensor *mmYOptional,
    const aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret_param = CheckParams(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale, gmmXOffsetOptional, gmmWeightOffsetOptional,
        sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional,
        mmWeightScaleOptional, mmXOffsetOptional, mmWeightOffsetOptional, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = allto_allv_grouped_mat_mul_checker::CheckSendAndRecv(sendCounts, recvCounts, gmmX, gmmY);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    aclnnStatus ret = InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, gmmXScale,
        gmmWeightScale, gmmXOffsetOptional, gmmWeightOffsetOptional, mmXScaleOptional, mmWeightScaleOptional,
        mmXOffsetOptional, mmWeightOffsetOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
        transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, groupSize,
        gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
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
