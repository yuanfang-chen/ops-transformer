/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_grouped_mat_mul_allto_allv.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "common/utils/hccl_util.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "platform/soc_spec.h"
#include "opdev/platform.h"
#include "mc2_aclnn_util.h"
#include "securec.h"
#include <algorithm>

namespace {
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

static constexpr int64_t DIM_TWO = 2;
static constexpr int64_t DIM_THREE = 3;

extern "C" aclnnStatus aclnnInnerQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScaleOptional, const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXScaleOptional,
    const aclTensor *mmWeightScaleOptional, const aclTensor *commQuantScaleOptional, const char *group,
    int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight,
    bool transMmWeight, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t commQuantMode, int64_t groupSize, int64_t commQuantDtypeOptional, int64_t yDtype,
    int64_t mmDtype, const aclTensor *yOut, const aclTensor *mmYOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerQuantGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize,
                                                             aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

const std::initializer_list<op::DataType> MX_INPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E4M3FN,
                                                                         op::DataType::DT_FLOAT8_E5M2};
const std::initializer_list<op::DataType> MX_SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};
const std::initializer_list<op::DataType> MX_OUTPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                          op::DataType::DT_BF16};


static int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

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

// 检查 mm 系列 optional 参数一致性：全空或全非空
static bool CheckMmOptionalConsistency(const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                                       const aclTensor *mmYOptional, const aclTensor *mmXScaleOptional,
                                       const aclTensor *mmWeightScaleOptional)
{
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
    // 如果mmXOptional为空，则mmXScaleOptional, mmWeightScaleOptional也必须为空
    if (mmXOptional == nullptr) {
        if (mmXScaleOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmXScaleOptional should be null when mmXOptional is null.");
            return false;
        }
        if (mmWeightScaleOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeightScaleOptional should be null when mmWeightOptional is null.");
            return false;
        }
    }
    return true;
}

// check nullptr
static bool CheckNullStatus(const aclTensor *gmmX, const aclTensor *gmmWeight,
                            const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const char *group,
                            const aclTensor *y, const aclTensor *mmYOptional, const aclTensor *mmXScaleOptional,
                            const aclTensor *mmWeightScaleOptional)
{
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) { // HCCL_GROUP_NAME_MAX = 128U
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    return CheckMmOptionalConsistency(mmXOptional, mmWeightOptional, mmYOptional, mmXScaleOptional,
                                      mmWeightScaleOptional);
}

static aclnnStatus CheckIntArrayNotEmpty(const aclIntArray *arr, const char *name)
{
    if (arr == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s should not be null.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t size = 0U;
    aclGetIntArraySize(arr, &size);
    if (size == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s size should not be 0.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// check send and recv
static aclnnStatus CheckSendAndRecv(const aclIntArray *sendCounts, const aclIntArray *recvCounts)
{
    auto ret = CheckIntArrayNotEmpty(sendCounts, "sendCounts");
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    return CheckIntArrayNotEmpty(recvCounts, "recvCounts");
}

// 检查是否有空tensor
static bool CheckGmmNotEmpty(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *y)
{
    auto h1Val = gmmX->GetViewShape().GetDim(1);

    auto h1Val2 = gmmWeight->GetViewShape().GetDim(1);
    auto n1Val = gmmWeight->GetViewShape().GetDim(DIM_TWO);
    auto yMval = y->GetViewShape().GetDim(0);
    auto yNval = y->GetViewShape().GetDim(1);

    OP_API_CHECK((h1Val == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    OP_API_CHECK((h1Val2 == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    OP_API_CHECK((n1Val == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((yMval == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y is empty tensor with zero dimM, which is unsupported.");
        return false;
    });
    OP_API_CHECK((yNval == 0), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    return true;
}

static bool CheckMmNotEmptyOrAllEmpty(const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                                      const aclTensor *mmYOptional)
{
    if (mmXOptional == nullptr || mmWeightOptional == nullptr || mmYOptional == nullptr) {
        return true;
    }
    auto mmDim0 = mmXOptional->GetViewShape().GetDim(0);
    auto mmDim1 = mmXOptional->GetViewShape().GetDim(1);
    auto mmWdim0 = mmWeightOptional->GetViewShape().GetDim(0);
    auto mmWdim1 = mmWeightOptional->GetViewShape().GetDim(1);
    auto mmYdim0 = mmYOptional->GetViewShape().GetDim(0);
    auto mmYdim1 = mmYOptional->GetViewShape().GetDim(1);
    bool allNotEmpty =
        ((mmDim0 != 0) && (mmDim1 != 0) && (mmWdim0 != 0) && (mmWdim1 != 0) && (mmYdim0 != 0) && (mmYdim1 != 0));
    bool allEmpty =
        ((mmDim0 == 0) && (mmDim1 == 0) && (mmWdim0 == 0) && (mmWdim1 == 0) && (mmYdim0 == 0) && (mmYdim1 == 0));
    OP_API_CHECK(((!allNotEmpty) && (!allEmpty)), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional and mmYOptional should all be empty tensor or all not be empty tensor.");
        return false;
    });

    return true;
}

// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *y,
                                const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                                const aclTensor *mmYOptional)
{
    CHECK_RET(CheckGmmNotEmpty(gmmX, gmmWeight, y), false);
    return CheckMmNotEmptyOrAllEmpty(mmXOptional, mmWeightOptional, mmYOptional);
}


// 检查所有要用到的format是否为ND，不支持私有格式，如果内部不为ND格式，打印warning日志，将format转换为ND格式
static bool CheckFormat(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                        const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXOptional,
                        const aclTensor *mmWeightOptional, const aclTensor *y, const aclTensor *mmYOptional)
{
    // 定义内联检查函数
    auto checkNotPrivate = [](const aclTensor *tensor, const char *name) -> bool {
        if (tensor == nullptr) {
            return true;
        }
        if (IsPrivateFormat(tensor->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantGroupMatmulAlltoAll, %s format %s does not support private format.", name,
                    op::ToString(tensor->GetStorageFormat()).GetString());
            return false;
        }
        return true;
    };

    // 必传参数检查
    auto checkRequired = [](const aclTensor *tensor, const char *name) -> bool {
        if (IsPrivateFormat(tensor->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnQuantGroupMatmulAlltoAll, %s format %s does not support private format.", name,
                    op::ToString(tensor->GetStorageFormat()).GetString());
            return false;
        }
        return true;
    };

    // 执行检查
    if (!checkRequired(gmmX, "gmmX"))
        return false;
    if (!checkRequired(gmmWeight, "gmmWeight"))
        return false;
    if (!checkNotPrivate(gmmXScaleOptional, "gmmXScaleOptional"))
        return false;
    if (!checkNotPrivate(gmmWeightScaleOptional, "gmmWeightScaleOptional"))
        return false;
    if (!checkNotPrivate(mmXOptional, "mmXOptional"))
        return false;
    if (!checkNotPrivate(mmWeightOptional, "mmWeightOptional"))
        return false;
    if (!checkRequired(y, "y"))
        return false;
    if (!checkNotPrivate(mmYOptional, "mmYOptional"))
        return false;

    return true;
}

static bool ReFormatTensorToND(const aclTensor *tensor, const char *name)
{
    if (tensor != nullptr && tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("%s origin format is %s.", name, op::ToString(tensor->GetStorageFormat()).GetString());
        tensor = l0op::ReFormat(tensor, op::Format::FORMAT_ND);
        CHECK_RET(tensor != nullptr, false);
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                          const aclTensor *gmmWeightScaleOptional, const aclTensor *mmXOptional,
                          const aclTensor *mmWeightOptional, const aclTensor *y, const aclTensor *mmYOptional)
{
    CHECK_RET(ReFormatTensorToND(gmmX, "gmmX"), false);
    CHECK_RET(ReFormatTensorToND(gmmWeight, "gmmWeight"), false);
    CHECK_RET(ReFormatTensorToND(gmmXScaleOptional, "gmmXScaleOptional"), false);
    CHECK_RET(ReFormatTensorToND(gmmWeightScaleOptional, "gmmWeightScaleOptional"), false);
    CHECK_RET(ReFormatTensorToND(mmXOptional, "mmXOptional"), false);
    CHECK_RET(ReFormatTensorToND(mmWeightOptional, "mmWeightOptional"), false);
    CHECK_RET(ReFormatTensorToND(y, "y"), false);
    CHECK_RET(ReFormatTensorToND(mmYOptional, "mmYOptional"), false);
    return true;
}

static bool CheckMxDType(const aclTensor *x, const aclTensor *weight, const aclTensor *xScale,
                         const aclTensor *weightScale, const aclTensor *y, const char *xName, const char *weightName)
{
    if (!CheckType(x->GetDataType(), MX_INPUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In Mx QuantMode, %s support DTYPE_FLOAT8_E4M3FN and DTYPE_FLOAT8_E5M2, but got %s.", xName,
                op::ToString(x->GetDataType()).GetString());
        return false;
    }
    if (!CheckType(weight->GetDataType(), MX_INPUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In Mx QuantMode, %s support DTYPE_FLOAT8_E4M3FN and DTYPE_FLOAT8_E5M2, but got %s.", weightName,
                op::ToString(weight->GetDataType()).GetString());
        return false;
    }
    if (!CheckType(xScale->GetDataType(), MX_SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In Mx QuantMode, %s support DT_FLOAT_E8M0, but got %s.", xName,
                op::ToString(xScale->GetDataType()).GetString());
        return false;
    }
    if (!CheckType(weightScale->GetDataType(), MX_SCALE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In Mx QuantMode, %s support DT_FLOAT_E8M0, but got %s.", weightName,
                op::ToString(weightScale->GetDataType()).GetString());
        return false;
    }
    if (!CheckType(y->GetDataType(), MX_OUTPUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In Mx QuantMode, y support DT_FLOAT16 or DT_BF16, but got %s.",
                op::ToString(y->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckPerTensorQuantMode(const aclTensor *xScale, const aclTensor *weightScale, const char *xName,
                                    const char *weightName)
{
    if (xScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s should not be empty in PerTensor mode.", xName);
        return false;
    }
    if (weightScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s should not be empty in PerTensor mode.", weightName);
        return false;
    }
    return true;
}

static bool CheckMxQuantMode(const aclTensor *xScale, const aclTensor *weightScale, const aclTensor *x,
                             const aclTensor *weight, const aclTensor *y, const char *xName, const char *weightName)
{
    if (xScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s should not be empty in MX mode.", xName);
        return false;
    }
    if (weightScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s should not be empty in MX mode.", weightName);
        return false;
    }
    return CheckMxDType(x, weight, xScale, weightScale, y, xName, weightName);
}

static bool CheckUnsupportQuantMode(QuantModeType mode, const char *xName)
{
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s quantMode (%ld) is not support yet.", xName, static_cast<int64_t>(mode));
    return false;
}

// 检查量化参数是否合法
static bool CheckQuantMode(int64_t xQuantMode, int64_t weightQuantMode, const aclTensor *XScaleOptional,
                           const aclTensor *WeightScaleOptional, const aclTensor *x, const aclTensor *weight,
                           const aclTensor *y, const char *xName, const char *weightName)
{
    QuantModeType xMode = static_cast<QuantModeType>(xQuantMode);
    QuantModeType wMode = static_cast<QuantModeType>(weightQuantMode);
    if (xMode != wMode) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s QuantMode and %s QuanMode should be the same, but got %ld and %ld.", xName,
                weightName, static_cast<int64_t>(xMode), static_cast<int64_t>(wMode));
        return false;
    }
    // 按量化模式分支校验
    switch (xMode) {
        case QuantModeType::NO_QUANT:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Quant template unsupport NO_QUAN mode.");
            return false;
        case QuantModeType::PERTENSOR_QUANT:
            return CheckPerTensorQuantMode(XScaleOptional, WeightScaleOptional, xName, weightName);
        case QuantModeType::MX_QUANT:
            return CheckMxQuantMode(XScaleOptional, WeightScaleOptional, x, weight, y, xName, weightName);
        case QuantModeType::PERCHANNEL_QUANT:
        case QuantModeType::PERTOKEN_QUANT:
        case QuantModeType::PERGROUP_QUANT:
        case QuantModeType::PERBLOCK_QUANT:
        case QuantModeType::DYN_PERTOKEN_QUANT:
            return CheckUnsupportQuantMode(xMode, xName);
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unknown %s quanMode: %ld.", xName, static_cast<int64_t>(xMode));
            return false;
    }
}

static bool CheckMmConsistency(const aclTensor *gmmX, const aclTensor *gmmWeight,
                                     const aclTensor *mmXOptional, const aclTensor *mmWeightOptional)
{
    if (mmXOptional == nullptr || mmWeightOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmX and mmWeight should both be set or both be nullptr.");
        return false;
    }
    if (mmXOptional->GetDataType() != gmmX->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmX dtype should be the same as gmmX dtype.");
        return false;
    }
    if (mmWeightOptional->GetDataType() != gmmWeight->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmWeight dtype should be the same as gmmWeight dtype.");
        return false;
    }
    return true;
}

static bool CheckQuantParams(int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, const aclTensor *gmmX,
                             const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                             const aclTensor *gmmWeightScaleOptional, const aclTensor *y,
                             int64_t mmXQuantMode, int64_t mmWeightQuantMode, const aclTensor *mmXOptional,
                             const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                             const aclTensor *mmWeightScaleOptional, const aclTensor *mmYOptional)
{
    // 1) gmm 一定要有量化模式，且当前只支持 TT(1) / MX(6)
    if (!CheckQuantMode(gmmXQuantMode, gmmWeightQuantMode, gmmXScaleOptional, gmmWeightScaleOptional,
                        gmmX, gmmWeight, y, "gmmX", "gmmWeight")) {
        return false;
    }
    // 2) mm 不存在时，允许没有量化模式
    if (mmXOptional == nullptr && mmWeightOptional == nullptr) {
        return true;
    }
    // 3) 共享专家强校验：输入类型、转置配置必须与 gmm 一致
    if (!CheckMmConsistency(gmmX, gmmWeight, mmXOptional, mmWeightOptional)) {
        return false;
    }
    // 4) mm 存在时，mm 不能是非量化，且必须与 gmm 保持完全一致
    if (mmXQuantMode != gmmXQuantMode || mmWeightQuantMode != gmmWeightQuantMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When mm inputs are set, mm quant modes should be exactly the same as gmm quant modes. "
                "Expect (%ld, %ld), but got (%ld, %ld).",
                gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode);
        return false;
    }
    if (!CheckQuantMode(mmXQuantMode, mmWeightQuantMode, mmXScaleOptional, mmWeightScaleOptional,
                        mmXOptional, mmWeightOptional, mmYOptional, "mmX", "mmWeight")) {
        return false;
    }
    return true;
}

// 检查tensor最后两维是否转置（stride不连续）
static bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

// 处理支持转置的tensor物理排布不连续问题（gmmWeight, 3D）
static const aclTensor *TransGmmWeightTensor(const aclTensor *gmmWeight)
{
    uint64_t storageShapeDimNum = gmmWeight->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = gmmWeight->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = gmmWeight->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = gmmWeight->GetViewShape().GetDim(i);
    }
    viewDim[1] = gmmWeight->GetViewShape().GetDim(2);
    viewDim[2] = gmmWeight->GetViewShape().GetDim(1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(gmmWeight, &dataType);
    auto transStride = gmmWeight->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[1] = transStride[2];
    stride[2] = transStride[1];

    auto offset = gmmWeight->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, gmmWeight->GetTensor()->GetAddr());
}

// 处理支持转置的tensor物理排布不连续问题（mmWeightOptional, 2D）
static const aclTensor *TransMmWeightOptionalTensor(const aclTensor *mmWeightOptional)
{
    uint64_t storageShapeDimNum = mmWeightOptional->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = mmWeightOptional->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = mmWeightOptional->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = mmWeightOptional->GetViewShape().GetDim(i);
    }
    viewDim[0] = mmWeightOptional->GetViewShape().GetDim(1);
    viewDim[1] = mmWeightOptional->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(mmWeightOptional, &dataType);
    auto transStride = mmWeightOptional->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = mmWeightOptional->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, mmWeightOptional->GetTensor()->GetAddr());
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
                               const aclTensor *gmmWeightScaleOptional, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               int64_t commQuantMode, const char *group, int64_t epWorldSize,
                               const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight,
                               bool transMmWeight, const aclTensor *y, const aclTensor *mmYOptional,
                               uint64_t *workspaceSize, aclOpExecutor **executor)
{
    (void)epWorldSize;
    (void)sendCounts;
    (void)recvCounts;
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
                              mmWeightOptional, group, y, mmYOptional, mmXScaleOptional, mmWeightScaleOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckNotEmptyTensor(gmmX, gmmWeight, y, mmXOptional, mmWeightOptional, mmYOptional),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, mmXOptional, mmWeightOptional, y,
                          mmYOptional),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckQuantParams(gmmXQuantMode, gmmWeightQuantMode, gmmX, gmmWeight, gmmXScaleOptional,
                               gmmWeightScaleOptional, y, mmXQuantMode, mmWeightQuantMode, mmXOptional,
                               mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, mmYOptional),
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
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScaleOptional,
    const aclTensor *gmmWeightScaleOptional, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional, const aclTensor *commQuantScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    int64_t commQuantMode, int64_t commQuantDtypeOptional, int64_t groupSize, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    const aclTensor *y, const aclTensor *mmYOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // MX 量化场景通过 stride 检测 weight/scale 的转置状态
    bool isMxQuant = (gmmXQuantMode == static_cast<int64_t>(QuantModeType::MX_QUANT));

    if (isMxQuant) {
        // === gmmWeight 转置检测 ===
        bool notContiguousGmm = IsTransposeLastTwoDims(gmmWeight);
        if (notContiguousGmm && transGmmWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight not contiguous and transGmmWeight is set!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguousGmm && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            transGmmWeight = !transGmmWeight;
            gmmWeight = TransGmmWeightTensor(gmmWeight);
            CHECK_RET(gmmWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // === gmmWeightScale 转置检测 ===
        if (gmmWeightScaleOptional != nullptr && MC2Aclnn::IsNeedScaleTrans(gmmWeightScaleOptional)) {
            if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
                gmmWeightScaleOptional = TransGmmWeightTensor(gmmWeightScaleOptional);
                CHECK_RET(gmmWeightScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }

        // === mmWeight 转置检测 ===
        if (mmWeightOptional != nullptr) {
            bool notContiguousMm = IsTransposeLastTwoDims(mmWeightOptional);
            if (notContiguousMm && transMmWeight) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmWeight not contiguous and transMmWeight is set!");
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (notContiguousMm && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
                transMmWeight = !transMmWeight;
                mmWeightOptional = TransMmWeightOptionalTensor(mmWeightOptional);
                CHECK_RET(mmWeightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }

        // === mmWeightScale 转置检测 ===
        if (mmWeightScaleOptional != nullptr && IsTransposeLastTwoDims(mmWeightScaleOptional)) {
            if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
                mmWeightScaleOptional = TransMmWeightOptionalTensor(mmWeightScaleOptional);
                CHECK_RET(mmWeightScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }

        // === GMM 和 MM 转置一致性校验 ===
        if (mmWeightOptional != nullptr && transGmmWeight != transMmWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "transGmmWeight(%d) and transMmWeight(%d) must be the same.",
                    transGmmWeight, transMmWeight);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    auto retParam = CheckParams(gmmX, gmmWeight, gmmXScaleOptional, gmmWeightScaleOptional, sendCountsTensorOptional,
                                recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional,
                                mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                                mmWeightQuantMode, commQuantMode, group, epWorldSize, sendCounts, recvCounts,
                                transGmmWeight, transMmWeight, y, mmYOptional, workspaceSize, executor);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    auto retSendAndRecv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(retSendAndRecv == ACLNN_SUCCESS, retSendAndRecv);

    char *strGroup = const_cast<char *>(group);

    int64_t yDtype = y->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();

    aclnnStatus ret = aclnnInnerQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        gmmXScaleOptional, gmmWeightScaleOptional, mmXScaleOptional, mmWeightScaleOptional, commQuantScaleOptional,
        strGroup, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, commQuantMode, groupSize, commQuantDtypeOptional, yDtype, mmDtype, y,
        mmYOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnQuantGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerQuantGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace