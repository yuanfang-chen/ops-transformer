/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_reduce_scatter_check_tiling.cpp
 * \brief
 */
#include "quant_reduce_scatter_check_tiling.h"

namespace MC2Tiling {
constexpr size_t X_INDEX = 0;
constexpr size_t SCALE_INDEX = 1;
constexpr size_t OUTPUT_INDEX = 0;
constexpr size_t GROUP_INDEX = 0;
constexpr size_t RANK_SIZE_INDEX = 1;
constexpr size_t TRANSPOSE_X2_INDEX = 2;
constexpr size_t OUT_PUT_DTYPE_INDEX = 3;
constexpr size_t EPSILON_INDEX = 4;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t NUM_THREE = 3;

ge::graphStatus TilingCheckQbmmReduceScatterAddRmsNormCast::CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    // group空字符串校验
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(nodeName, "groupPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::string(groupPtr).empty(),
        OP_LOGE(nodeName, "group should not be empty."), return ge::GRAPH_FAILED);
    // transpose校验
    const bool *transposeX2Ptr = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(transposeX2Ptr == nullptr, OP_LOGE(nodeName, "transposeX2Ptr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*transposeX2Ptr,
        OP_LOGE(nodeName, "transposeX2 should be false."), return ge::GRAPH_FAILED);
    // 输出type校验（猜测属性的dtype和最终输出的y1的类型应该保持一致）
    const int64_t *outputTypePtr = attrs->GetAttrPointer<int64_t>(OUTPUT_DTYPE_INDEX);
    OP_TILING_CHECK(outputTypePtr == nullptr, OP_LOGE(nodeName, "outputTypePtr is nullptr."), return ge::GRAPH_FAILED);
    ge::DataType outputType = static_cast<ge::DataType>(*outputTypePtr);
    OP_TILING_CHECK(outputType != ge::DT_FLOAT,
                    OP_LOGE(nodeName, "outPutType should be float, but actual value is %s.",
                            Ops::Base::ToString(outputType).c_str()),
                    return ge::GRAPH_FAILED);
    // epsilon校验
    const float *epsilonPtr = attrs->GetAttrPointer<float>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(epsilonPtr == nullptr, OP_LOGE(nodeName, "epsilonPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epsilonPtr != 1e-6f,
        OP_LOGE(nodeName, "epsilon should be 1e-6f."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool TilingCheckQbmmReduceScatterAddRmsNormCast::CheckInputTensorDim(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *x1Shape = context->GetInputShape(X1_INDEX);
    OP_TILING_CHECK(x1Shape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    const gert::StorageShape *x2Shape = context->GetInputShape(X2_INDEX);
    OP_TILING_CHECK(x2Shape == nullptr, OP_LOGE(nodeName, "x2Shape is null."), return false);
    
    size_t x1Dim = x1Shape->GetStorageShape().GetDimNum();
    size_t x2Dim = x2Shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(x1Dim != TWO_DIMS,
        OP_LOGE(nodeName, "x1Dim should be 2, but current x1Dim is %lu.", xDim), return false);
    OP_TILING_CHECK(x2Dim != FOUR_DIMS,
        OP_LOGE(nodeName, "x1Dim should be 4, but current x2Dim is %lu.", xDim), return false);
    uint64_t x1ValueOne = xShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x1ValueTwo = xShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x2ValueOne = scaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x2ValueTwo = scaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t x2ValueThree = scaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t x2ValueFour = scaleShape->GetStorageShape().GetDim(DIM_ONE);

    // 校验x1
    OP_TILING_CHECK((x1ValueOne != 252) || (x1ValueTwo == 2560),
            OP_LOGE(nodeName, "x1 shape should be (252,2560)"), return false);
    // 校验x2
    OP_TILING_CHECK((x2ValueOne != 160) || (x2ValueTwo != 160) || (x2ValueThree != 16) || (x2ValueFour != 32), 
            OP_LOGE(nodeName, "x2 shape should be (160,160,16,32)"
        "x %lu which is invalid.", scaleValueOne, xValueOne), return false);
    // TODO:待完善
    return true;
}

bool TilingCheckQbmmReduceScatterAddRmsNormCast::CheckOutputTensorDim(const gert::TilingContext *context,
                                                         const QuantReduceScatterTilingParams &params)
{
    // TODO:待完善
}

bool TilingCheckQbmmReduceScatterAddRmsNormCast::CheckTensorDataType(const gert::TilingContext *context,
                                                        QuantReduceScatterTilingParams &params)
{
    // TODO:待完善
    return true;
}

bool TilingCheckQbmmReduceScatterAddRmsNormCast::CheckTensorDim(const gert::TilingContext *context)
{
    // TODO:待完善
    return true;
}

bool TilingCheckQbmmReduceScatterAddRmsNormCast::CheckTensorFormat(const gert::TilingContext *context)
{
    // TODO:待完善
    return true;
}

ge::graphStatus QbmmReduceScatterAddRmsNormCastCheckTiling::TilingCheckQbmmReduceScatterAddRmsNormCast(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(!CheckTensorDataType(context),
        OP_LOGE(nodeName, "params dtype is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDim(context),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace MC2Tiling
