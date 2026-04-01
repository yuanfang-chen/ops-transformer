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
 * \file moe_init_routing_v3_mx_quant_infershape.cpp
 * \brief MoeInitRoutingV3MxQuant InferShape and InferDataType
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t OTHER_SHAPE = -1;

// Input indices
static constexpr int64_t INDEX_INPUT_X = 0;
static constexpr int64_t INDEX_INPUT_EXPERT_IDX = 1;

// Output indices
static constexpr int64_t OUTPUT_Y = 0;
static constexpr int64_t OUTPUT_MXSCALE = 1;
static constexpr int64_t OUTPUT_EXPANDED_ROW_IDX = 2;
static constexpr int64_t OUTPUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 3;
static constexpr int64_t OUTPUT_EXPANDED_SCALE = 4;

// Attr indices (must match _def.cpp declaration order)
static constexpr int64_t ATTR_ACTIVE_NUM = 0;
static constexpr int64_t ATTR_EXPERT_CAPACITY = 1;
static constexpr int64_t ATTR_EXPERT_NUM = 2;
static constexpr int64_t ATTR_DROP_PAD_MODE = 3;
static constexpr int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
static constexpr int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
static constexpr int64_t ATTR_AXIS = 6;
static constexpr int64_t ATTR_ROUND_MODE = 7;
static constexpr int64_t ATTR_DST_TYPE = 8;
static constexpr int64_t ATTR_BLOCKSIZE = 9;
static constexpr int64_t ATTR_SCALE_ALG = 10;

static constexpr int64_t DEFAULT_BLOCKSIZE = 32;

static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

static bool IsSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 == OTHER_SHAPE || dim2 == OTHER_SHAPE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus CheckInputShape(
    const gert::InferShapeContext* context, const gert::Shape* xShape, const gert::Shape* expertIdxShape)
{
    int64_t xN = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
    if (xN < OTHER_SHAPE || cols < OTHER_SHAPE) {
        OP_LOGE(context->GetNodeName(), "Invalid x shape.");
        return ge::GRAPH_FAILED;
    }

    int64_t expertIdxN = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(0);
    int64_t expertIdxK = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
    if (expertIdxN < OTHER_SHAPE || expertIdxK < OTHER_SHAPE) {
        OP_LOGE(context->GetNodeName(), "Invalid expertIdx shape.");
        return ge::GRAPH_FAILED;
    }

    if (!IsSameDim(xN, expertIdxN)) {
        OP_LOGE(context->GetNodeName(),
            "The first dim of x(%ld) and expertIdx(%ld) should be equal.", xN, expertIdxN);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParams(
    const gert::InferShapeContext* context,
    const gert::Shape* xShape, const gert::Shape* expertIdxShape,
    int64_t activeNum, int64_t expertCapacity, int64_t expertNum,
    int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag)
{
    // x shape check
    if (xShape->GetDimNum() == 1U) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "The dynamic dim of x should be -2.");
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(context->GetNodeName(), "The dim of x should be 2 or dynamic.");
        return ge::GRAPH_FAILED;
    }

    // expert_idx shape check
    if (expertIdxShape->GetDimNum() == 1U) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "The dynamic dim of expertIdx should be -2.");
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(context->GetNodeName(), "The dim of expertIdx should be 2 or dynamic.");
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode < 0 || dropPadMode > 1) {
        OP_LOGE(context->GetNodeName(), "The dropPadMode should be 0 or 1.");
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1)) {
        OP_LOGE(context->GetNodeName(),
            "The expertCapacity and expertNum should be greater than 0 when dropPadMode is 1.");
        return ge::GRAPH_FAILED;
    }

    if (expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > 2) {
        OP_LOGE(context->GetNodeName(), "The expertTokensCountOrCumsumFlag should be 0, 1 or 2.");
        return ge::GRAPH_FAILED;
    }

    if (expertTokensCountOrCumsumFlag > 0 && expertNum <= 0) {
        OP_LOGE(context->GetNodeName(),
            "The expertNum should be greater than 0 when expertTokensCountOrCumsumFlag is greater than 0.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4MoeInitRoutingV3MxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRoutingV3MxQuantInferShape.");

    // Get attrs
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_NUM);
    const int64_t activeNum = (activeNumPtr == nullptr) ? -1 : *activeNumPtr;

    const int64_t* expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY);
    const int64_t expertCapacity = (expertCapacityPtr == nullptr) ? -1 : *expertCapacityPtr;

    const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
    const int64_t expertNum = (expertNumPtr == nullptr) ? -1 : *expertNumPtr;

    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
    const int64_t dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;

    const int64_t* expertTokensCountOrCumsumFlagPtr =
        attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG);
    const int64_t expertTokensCountOrCumsumFlag =
        (expertTokensCountOrCumsumFlagPtr == nullptr) ? 0 : *expertTokensCountOrCumsumFlagPtr;

    const bool* expertTokensBeforeCapacityFlagPtr =
        attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG);
    const bool expertTokensBeforeCapacityFlag =
        (expertTokensBeforeCapacityFlagPtr == nullptr) ? false : *expertTokensBeforeCapacityFlagPtr;

    const int64_t* blocksizePtr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCKSIZE);
    const int64_t blocksize = (blocksizePtr == nullptr) ? DEFAULT_BLOCKSIZE : *blocksizePtr;

    // Get input shapes
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* expertIdxShape = context->GetInputShape(INDEX_INPUT_EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);

    // Check input shapes
    if (CheckInputShape(context, xShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckParams(context, xShape, expertIdxShape,
                    activeNum, expertCapacity, expertNum,
                    dropPadMode, expertTokensCountOrCumsumFlag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Extract dims
    int64_t n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
    int64_t k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);

    // Compute num_expanded_tokens
    int64_t numExpandedTokens = OTHER_SHAPE;
    int64_t expandedRowIdxNum = OTHER_SHAPE;

    if (n > 0 && k > 0) {
        expandedRowIdxNum = n * k;
        // activeNum is a token count (in 'n' domain), not assignment count
        int64_t effectiveN = (activeNum > 0 && activeNum < n) ? activeNum : n;
        int64_t nk = effectiveN * k;
        if (dropPadMode > 0 && expertCapacity > 0 && expertNum > 0) {
            numExpandedTokens = expertNum * expertCapacity;
        } else {
            numExpandedTokens = nk;
        }
    }

    // Output y: [num_expanded_tokens, hidden_size]
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    if (dropPadMode > 0 && expertNum > 0 && expertCapacity > 0) {
        yShape->SetDimNum(3U);
        yShape->SetDim(0U, expertNum);
        yShape->SetDim(1U, expertCapacity);
        yShape->SetDim(2U, cols < 0 ? OTHER_SHAPE : cols);
    } else {
        yShape->SetDimNum(DIM_TWO);
        yShape->SetDim(0U, numExpandedTokens);
        yShape->SetDim(1U, cols < 0 ? OTHER_SHAPE : cols);
    }

    // Output mxscale: [num_expanded_tokens, ceil(hidden_size / blocksize)]
    gert::Shape* mxscaleShape = context->GetOutputShape(OUTPUT_MXSCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxscaleShape);
    int64_t scaleCols = (cols > 0 && blocksize > 0) ? CeilDiv(cols, blocksize) : OTHER_SHAPE;
    if (dropPadMode > 0 && expertNum > 0 && expertCapacity > 0) {
        mxscaleShape->SetDimNum(3U);
        mxscaleShape->SetDim(0U, expertNum);
        mxscaleShape->SetDim(1U, expertCapacity);
        mxscaleShape->SetDim(2U, scaleCols);
    } else {
        mxscaleShape->SetDimNum(DIM_TWO);
        mxscaleShape->SetDim(0U, numExpandedTokens);
        mxscaleShape->SetDim(1U, scaleCols);
    }

    // Output expanded_row_idx: [n * k]
    gert::Shape* expandedRowIdxShape = context->GetOutputShape(OUTPUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    expandedRowIdxShape->SetDimNum(DIM_ONE);
    expandedRowIdxShape->SetDim(0U, expandedRowIdxNum);

    // Output expert_tokens_count_or_cumsum: [expert_num] (optional, only when flag > 0 and dropPadMode == 0)
    gert::Shape* expertTokensCountOrCumsumShape =
        context->GetOutputShape(OUTPUT_EXPERT_TOKENS_COUNT_OR_CUMSUM);
    if (expertTokensCountOrCumsumFlag > 0 && expertTokensCountOrCumsumShape != nullptr) {
        expertTokensCountOrCumsumShape->SetDimNum(DIM_ONE);
        expertTokensCountOrCumsumShape->SetDim(0U, expertNum);
    }

    // Output expanded_scale: [num_expanded_tokens] (optional)
    gert::Shape* expandedScaleShape = context->GetOutputShape(OUTPUT_EXPANDED_SCALE);
    if (expandedScaleShape != nullptr) {
        if (dropPadMode > 0 && expertNum > 0 && expertCapacity > 0) {
            expandedScaleShape->SetDimNum(DIM_ONE);
            expandedScaleShape->SetDim(0U, expertNum * expertCapacity);
        } else {
            expandedScaleShape->SetDimNum(DIM_ONE);
            expandedScaleShape->SetDim(0U, numExpandedTokens);
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do MoeInitRoutingV3MxQuantInferShape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingV3MxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRoutingV3MxQuantInferDataType.");

    // y dtype depends on dst_type attr; framework dispatches based on _def.cpp dtype config
    // mxscale is always fp8_e8m0, expanded_row_idx and expert_tokens_count_or_cumsum are always int32
    // expanded_scale is always bf16
    // The output dtype is set by framework from _def.cpp DataType arrays, but we explicitly set
    // the fixed-dtype outputs here for safety.
    context->SetOutputDataType(OUTPUT_MXSCALE, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(OUTPUT_EXPANDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_EXPERT_TOKENS_COUNT_OR_CUMSUM, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_EXPANDED_SCALE, ge::DT_BF16);

    OP_LOGD(context->GetNodeName(), "End to do MoeInitRoutingV3MxQuantInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingV3MxQuant)
    .InferShape(InferShape4MoeInitRoutingV3MxQuant)
    .InferDataType(InferDataType4MoeInitRoutingV3MxQuant);
}  // namespace ops
