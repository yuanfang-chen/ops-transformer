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
 * \file moe_distribute_combine_add_rms_norm_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;
namespace ops {
static constexpr uint32_t IDX_ZERO = 0;
static constexpr uint32_t IDX_ONE = 1;
static constexpr uint32_t IDX_TWO = 2;
static constexpr uint32_t IDX_THREE = 3;
static constexpr uint32_t IDX_FOUR = 4;
static constexpr uint32_t IDX_FIVE = 5;
static constexpr uint32_t IDX_SIX = 6;
static constexpr uint32_t IDX_SEVEN = 7;
static constexpr uint32_t IDX_EIGHT = 8;

static constexpr uint32_t INPUT_X1_INDEX = 0;
static constexpr uint32_t INPUT_X2_INDEX = 1;
static constexpr uint32_t INPUT_Y_INDEX = 2;
static constexpr uint32_t INPUT_GAMMA_INDEX = 3;
static constexpr uint32_t INPUT_SCALE_INDEX = 4;
static constexpr uint32_t INPUT_BIAS_INDEX = 5;
static constexpr uint32_t INPUT_PERTOKEN_SCALE_INDEX = 6;

static constexpr uint32_t OUTPUT_Y1_INDEX = 0;
static constexpr uint32_t OUTPUT_Y2_INDEX = 1;
static constexpr uint32_t OUTPUT_X_INDEX = 2;
 
static constexpr uint32_t ATTR_RANK_SIZE_INDEX = 1;

 
static ge::graphStatus QbmmReduceScatterAddRmsNormCastInferShape(gert::InferShapeContext* context) {

    OP_LOGD(context->GetNodeName(), "Begin to do Infershape of QbmmReduceScatterAddRmsNormCastInferShape.");

    // 获取输入shape
    const gert::Shape* x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::Shape* x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::Shape* yShape = context->GetInputShape(INPUT_Y_INDEX);
    const gert::Shape* gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::Shape* scaleShape = context->GetInputShape(INPUT_SCALE_INDEX);
    const gert::Shape* biasShape = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    const gert::Shape* ptScaleShape = context->GetOptionalInputShape(INPUT_PERTOKEN_SCALE_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
 
     // 获取输出shape
    gert::Shape* y1OutShape = context->GetOutputShape(OUTPUT_Y1_INDEX);
    gert::Shape* y2OutShape = context->GetOutputShape(OUTPUT_Y2_INDEX);
    gert::Shape* xOutShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y1OutShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y2OutShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xOutShape);

    // 获取属性
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    // const auto rankSize = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    // OPS_CHECK_NULL_WITH_CONTEXT(context, rankSize);

    // 获取基础形状m, n, k
    // int64_t m = x1Shape->GetDim(0);
    // int64_t k = x1Shape->GetDim(1);
    // int64_t n = x2Shape->GetDim(2);
    // 设置y1输出shape [M, N/rank_size] (float)
    y1OutShape->SetDimNum(IDX_TWO);
    y1OutShape->SetDim(IDX_ZERO, 63);
    y1OutShape->SetDim(IDX_ONE, 5120);
    OP_LOGD(context->GetNodeName(), "y1 out shape set to [%ld, %ld]",63, 5120);

    // 设置y2输出shape [M, N/rank_size] (bf16)
    y2OutShape->SetDimNum(IDX_TWO);
    y2OutShape->SetDim(IDX_ZERO, 63);
    y2OutShape->SetDim(IDX_ONE, 5120);
    OP_LOGD(context->GetNodeName(), "y2 out shape set to [%ld, %ld]", 63, 5120);

    // 设置x输出shape [M, N/rank_size] (bf16)
    xOutShape->SetDimNum(IDX_TWO);
    xOutShape->SetDim(IDX_ZERO, 63);
    xOutShape->SetDim(IDX_ONE, 5120);
    OP_LOGD(context->GetNodeName(), "x out shape set to [%ld, %ld]", 63, 5120);

    OP_LOGD(context->GetNodeName(), "End to do QbmmReduceScatterAddRmsNormCastInferShape.");
    return GRAPH_SUCCESS;
}
 
static ge::graphStatus QbmmReduceScatterAddRmsNormCastInferDataType(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
 
    OP_LOGD(context->GetNodeName(), "QbmmReduceScatterAddRmsNormCastInferDataType begin");

    // 设置输出数据类型
    context->SetOutputDataType(OUTPUT_Y1_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_Y2_INDEX, ge::DT_BF16);
    context->SetOutputDataType(OUTPUT_X_INDEX, ge::DT_BF16);

    OP_LOGD(context->GetNodeName(), "QbmmReduceScatterAddRmsNormCastInferDataType end");
    
    return GRAPH_SUCCESS;
}
 
IMPL_OP_INFERSHAPE(QbmmReduceScatterAddRmsNormCast)
    .InferShape(QbmmReduceScatterAddRmsNormCastInferShape)
    .InferDataType(QbmmReduceScatterAddRmsNormCastInferDataType);
}  // namespace ops