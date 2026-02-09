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
constexpr size_t GROUP = 0;
constexpr size_t RANK_SIZE = 1;
constexpr size_t TRANSPOSE_X2 = 2;
constexpr size_t OUTPUT_DTYPE = 3;
constexpr size_t SUPPORT_DIM_SIZE = 2;
static ge::graphStatus InferShapeAddRmsNormDynamicQuantAllGatherQbmm(gert::InferShapeContext* context)
{
    auto x1MatrixShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1MatrixShape);
    auto x2MatrixShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2MatrixShape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isTransB = attrs->GetAttrPointer<bool>(TRANSPOSE_X2);
    const int64_t* rankSizeAttr = attrs->GetAttrPointer<int64_t>(RANK_SIZE);

    const char* groupStr = attrs->GetAttrPointer<char>(GROUP);
    OP_LOGE_IF(groupStr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get group failed.");
    int64_t rankSize = *rankSizeAttr;

    auto dimM = x1MatrixShape->GetDim(0);
    auto dimKX1 = x1MatrixShape->GetDim(1);
    auto dimKX2 = !(*isTransB) ? x2MatrixShape->GetDim(0) : x2MatrixShape->GetDim(1);
    auto dimN = !(*isTransB) ? x2MatrixShape->GetDim(1) : x2MatrixShape->GetDim(0);

    OP_LOGI(
        context->GetNodeName(),
        "group = %s isTransB %d x1.M = [%ld] x1.K = [%ld]"
        " x2.K = [%ld] x2.N = [%ld] rankSize = [%ld].",
        groupStr, (*isTransB), x1MatrixShape->GetDim(0), x1MatrixShape->GetDim(1),
        x2MatrixShape->GetDim(0), x2MatrixShape->GetDim(1), rankSize);
    if (dimKX1 != dimKX2) {
        OP_LOGE(
            context->GetNodeName(), "Input x1/x2 dim k must be same, but given x1.k %ld, x2.k %ld.", dimKX1,
            dimKX2);
        return ge::GRAPH_FAILED;
    }

    // 动态shape入图时 m轴-1时，不再进行(dimM * rankSize)的处理
    if (dimM == -1) {
        rankSize = 1;
    }
    // 不支持k = 0
    if (dimKX1 == 0) {
        dimM = dimN = 0;
        OP_LOGE(context->GetNodeName(), "X1/X2 are empty tensors with zero dimK.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShape->SetDimNum(SUPPORT_DIM_SIZE);
    outputShape->SetDim(0, dimM * rankSize);
    outputShape->SetDim(1, dimN);
    gert::Shape* zShape = context->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, zShape);
    zShape->SetDimNum(SUPPORT_DIM_SIZE);
    zShape->SetDim(0, dimM);
    zShape->SetDim(1, dimKX1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAddRmsNormDynamicQuantAllGatherQbmm(gert::InferDataTypeContext* context)
{
    auto z_type = context->GetInputDataType(0);

    auto y_dtype = static_cast<ge::DataType>(*context->GetAttrs()->GetAttrPointer<int64_t>(OUTPUT_DTYPE));
    context->SetOutputDataType(0, y_dtype);
    context->SetOutputDataType(1, z_type);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormDynamicQuantAllGatherQbmm)
    .InferShape(InferShapeAddRmsNormDynamicQuantAllGatherQbmm)
    .InferDataType(InferDataTypeAddRmsNormDynamicQuantAllGatherQbmm);
} // namespace ops
