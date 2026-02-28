/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_infershape.cpp
 * \brief
 */
#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "mc2_hcom_topo_info.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr size_t OUTPUT_DIM_SIZE = 2;
static constexpr int64_t NEG_ONE = -1;

static constexpr size_t INPUT_X1_INDEX = 0;
static constexpr size_t INPUT_X2_INDEX = 1;
static constexpr size_t INPUT_RESIDUAL_INDEX = 2;
static constexpr size_t INPUT_Y_INDEX = 3;
static constexpr size_t INPUT_GAMMA_INDEX = 4;
static constexpr size_t INPUT_SCALE_INDEX = 5;
static constexpr size_t INPUT_SMOOTH_SCALE_INDEX = 6;
static constexpr size_t INPUT_BIAS_INDEX = 7;
static constexpr size_t OUTPUT_OUTPUT_INDEX = 0;
static constexpr size_t OUTPUT_Z_INDEX = 1;
static constexpr size_t OUTPUT_ADD_RMS_NORM_OUT_INDEX = 2;
static constexpr size_t OUTPUT_DYNAMIC_QUANT_OUT_INDEX = 3;
static constexpr size_t OUTPUT_ALL_GATHER_DATA_OUT_INDEX = 4;
static constexpr size_t OUTPUT_ALL_GATHER_SCALES_OUT_INDEX = 5;
static constexpr size_t INPUT_ATTR_GROUP_INDEX = 1;
static constexpr size_t INPUT_ATTR_RANKSIZE_INDEX = 1;
static constexpr size_t INPUT_ATTR_TRANSPOSE_X2_INDEX = 2;
static constexpr size_t INPUT_ATTR_DTYPE_INDEX = 3;
static constexpr size_t INPUT_ATTR_RESIDUAL_NORM_MODE_INDEX = 4;

constexpr size_t GROUP = 0;
constexpr size_t RANK_SIZE = 1;
constexpr size_t TRANSPOSE_X2 = 2;

static ge::graphStatus InferShapeAddRmsNormDynamicQuantAllGatherQbmm(gert::InferShapeContext* context)
{
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAddRmsNormDynamicQuantAllGatherQbmm.");
    // 获取输入shape
    const gert::Shape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    const gert::Shape *residualShape = context->GetInputShape(INPUT_RESIDUAL_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, residualShape);
    const gert::Shape *yShape = context->GetInputShape(INPUT_Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Shape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    const gert::Shape *scaleShape = context->GetInputShape(INPUT_SCALE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    const gert::Shape *smoothScaleShape = context->GetOptionalInputShape(INPUT_SMOOTH_SCALE_INDEX);
    const gert::Shape *biasShape = context->GetOptionalInputShape(INPUT_BIAS_INDEX);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto group = attrs->GetAttrPointer<char>(INPUT_ATTR_GROUP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, group);
    const auto ranksize = attrs->GetAttrPointer<int64_t>(INPUT_ATTR_RANKSIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranksize);
    const auto transposeX2 = attrs->GetAttrPointer<int64_t>(INPUT_ATTR_TRANSPOSE_X2_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeX2);
    const auto dtype = attrs->GetAttrPointer<int64_t>(INPUT_ATTR_DTYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dtype);
    const auto residualNormMode = attrs->GetAttrPointer<int64_t>(INPUT_ATTR_RESIDUAL_NORM_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, residualNormMode);

    // TODO 强校验 待后续修改
    char groupStr = *group;
    int64_t rankSize = *ranksize;
    bool isTransX2 = *transposeX2;
    OP_CHECK_IF(groupStr == nullptr, OP_LOGE(context->GetNodeName(), "Get group failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rankSize != 4, OP_LOGE(context->GetNodeName(),
        "ranksize shoule be 4, but got", rankSize), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*dtype != -1, OP_LOGE(context->GetNodeName(),
        "dtype shoule be -1, but got", *dtype), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*residualNormMode != 0, OP_LOGE(context->GetNodeName(),
        "residualNormMode shoule be 0, but got", *residualNormMode), return ge::GRAPH_FAILED);

    auto x2Desc = context->GetInputDesc(INPUT_X2_INDEX);
    bool isX2NZ = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ;
    int64_t x2DimK = !(isX2NZ) ? x2Shape->GetDim(0) : x2Shape->GetDim(1) * x2Shape->GetDim(2);
    int64_t x2DimN = !(isX2NZ) ? x2Shape->GetDim(1) : x2Shape->GetDim(0) * x2Shape->GetDim(3);

    // int64_t dimM = ((x1Shape->GetDimNum() == 1U) ? NEG_ONE : x1Shape->GetDim(0));
    // int64_t dimKX1 = ((x1Shape->GetDimNum() == 1U) ? NEG_ONE : x1Shape->GetDim(1));
    // int64_t dimKX2 = ((x2Shape->GetDimNum() == 1U) ? NEG_ONE : x2DimK);
    // int64_t dimN = ((x2Shape->GetDimNum() == 1U) ? NEG_ONE : x2DimN);
    int64_t dimM = x1Shape->GetDim(0);
    int64_t dimKX1 = x1Shape->GetDim(1);
    int64_t dimKX2 = !(isX2NZ) ? x2DimK : x2DimN;
    int64_t dimN = !(isX2NZ) ? x2DimN : x2DimK;

    OP_LOGI(context->GetNodeName(), "group = %s isTransX2 %d x1.M = [%ld] x1.K = [%ld]"
        " x2.K = [%ld] x2.N = [%ld] rankSize = [%ld].", groupStr, isTransX2,
        dimM, dimKX1, dimKX2, dimN, rankSize);

    OP_CHECK_IF(dimKX1 != dimKX2, OP_LOGE(context->GetNodeName(),
        "Input x1/x2 dim k must be same, but given x1.k %ld, x2.k %ld.", dimKX1, dimKX2),
        return ge::GRAPH_FAILED);
    // 不支持k = 0
    OP_CHECK_IF(dimKX1 == 0, OP_LOGE(context->GetNodeName(),
        "X1/X2 are empty tensors with zero dimK."), return ge::GRAPH_FAILED);

    // 动态shape入图时 m轴-1时，不再进行(dimM * rankSize)的处理
    if (dimM == -1) {
        rankSize = 1;
    }

    gert::Shape *outputShape = context->GetOutputShape(OUTPUT_OUTPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShape->SetDimNum(OUTPUT_DIM_SIZE);
    outputShape->SetDim(0, dimM * rankSize);
    outputShape->SetDim(1, dimN);
    gert::Shape *zShape = context->GetOutputShape(OUTPUT_Z_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, zShape);
    zShape->SetDimNum(OUTPUT_DIM_SIZE);
    zShape->SetDim(0, dimM);
    zShape->SetDim(1, dimKX1);

    gert::Shape* addRmsNormOutShape = context->GetOutputShape(OUTPUT_ADD_RMS_NORM_OUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, addRmsNormOutShape);
    addRmsNormOutShape->SetDimNum(OUTPUT_DIM_SIZE);
    addRmsNormOutShape->SetDim(0, dimM);
    addRmsNormOutShape->SetDim(1, dimKX1);
    gert::Shape* dynamicQuantOutShape = context->GetOutputShape(OUTPUT_DYNAMIC_QUANT_OUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dynamicQuantOutShape);
    dynamicQuantOutShape->SetDimNum(OUTPUT_DIM_SIZE);
    dynamicQuantOutShape->SetDim(0, dimM);
    dynamicQuantOutShape->SetDim(1, dimKX1);
    gert::Shape* allGatherDataOutShape = context->GetOutputShape(OUTPUT_ALL_GATHER_DATA_OUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, allGatherDataOutShape);
    allGatherDataOutShape->SetDimNum(OUTPUT_DIM_SIZE);
    allGatherDataOutShape->SetDim(0, dimM * rankSize);
    allGatherDataOutShape->SetDim(1, dimKX1);
    gert::Shape* allGatherScalesOutShape = context->GetOutputShape(OUTPUT_ALL_GATHER_SCALES_OUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, allGatherScalesOutShape);
    allGatherScalesOutShape->SetDimNum(1);
    allGatherScalesOutShape->SetDim(0, dimM * rankSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAddRmsNormDynamicQuantAllGatherQbmm(gert::InferDataTypeContext* context)
{
    auto z_type = context->GetInputDataType(0);
    auto output_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, output_dtype);
    context->SetOutputDataType(1, z_type);

    auto addRmsNormOut_type = context->GetInputDataType(0);
    auto dynamicQuantOut_dtype = DT_INT8;
    auto allGatherDataOut_type = DT_INT8;
    auto allGatherScalesOut_dtype = DT_FLOAT;
    context->SetOutputDataType(2, addRmsNormOut_dtype);
    context->SetOutputDataType(3, dynamicQuantOut_type);
    context->SetOutputDataType(4, allGatherDataOut_dtype);
    context->SetOutputDataType(5, allGatherScalesOut_type);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormDynamicQuantAllGatherQbmm)
    .InferShape(InferShapeAddRmsNormDynamicQuantAllGatherQbmm)
    .InferDataType(InferDataTypeAddRmsNormDynamicQuantAllGatherQbmm);
} // namespace ops
