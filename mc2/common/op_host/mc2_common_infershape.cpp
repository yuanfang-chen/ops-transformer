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
 * \file mc2_common_infershape.cpp
 * \brief
 */

#include "op_host/mc2_common_infershape.h"
#include "mc2_log.h"

using namespace ge;
namespace ops {
// infershape 公共函数
ge::graphStatus CommonParamCheck(
    const gert::InferShapeContext* context, const size_t isTransAIndex, const size_t isTransBIndex, CommParas& commParas)
{
    commParas.x1MatrixShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commParas.x1MatrixShape);
    commParas.x2MatrixShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commParas.x2MatrixShape);
    if (commParas.x1MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE ||
        commParas.x2MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE) {
        OP_LOGE(context->GetNodeName(), "Input x1 and Input x2 must be the same with 2 dims.");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isTransA = attrs->GetAttrPointer<bool>(isTransAIndex);
    const bool* isTransB = attrs->GetAttrPointer<bool>(isTransBIndex);
    const int64_t* rankSizeAttr = attrs->GetAttrPointer<int64_t>(RANK_SIZE);

    const char* groupStr = attrs->GetAttrPointer<char>(GROUP);
    OP_LOGE_IF(groupStr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get group failed.");
    commParas.rankSize = -1;
    uint32_t rankNum = 0;
    if (*rankSizeAttr <= 0) {
        if ((Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(groupStr, &rankNum)) != HCCL_SUCCESS || rankNum == 0) {
            OP_LOGE(
                context->GetNodeName(), "Get rank size failed, group [%s], rankSize [%u]", groupStr, rankNum);
            return ge::GRAPH_FAILED;
        } else {
            commParas.rankSize = rankNum;
        }
        commParas.rankSize = static_cast<int64_t>(rankNum);
    } else {
        commParas.rankSize = *rankSizeAttr;
    }

    commParas.dimM = !(*isTransA) ? commParas.x1MatrixShape->GetDim(0) : commParas.x1MatrixShape->GetDim(1);
    commParas.dimKX1 = !(*isTransA) ? commParas.x1MatrixShape->GetDim(1) : commParas.x1MatrixShape->GetDim(0);
    commParas.dimKX2 = !(*isTransB) ? commParas.x2MatrixShape->GetDim(0) : commParas.x2MatrixShape->GetDim(1);
    commParas.dimN = !(*isTransB) ? commParas.x2MatrixShape->GetDim(1) : commParas.x2MatrixShape->GetDim(0);

    OP_LOGI(
        context->GetNodeName(),
        "group = %s isTransA %d isTransB %d x1.M = [%ld] x1.K = [%ld]"
        " x2.K = [%ld] x2.N = [%ld] rankSize = [%ld].",
        groupStr, (*isTransA), (*isTransB), commParas.x1MatrixShape->GetDim(0), commParas.x1MatrixShape->GetDim(1),
        commParas.x2MatrixShape->GetDim(0), commParas.x2MatrixShape->GetDim(1), commParas.rankSize);
    if (commParas.dimKX1 != commParas.dimKX2) {
        OP_LOGE(
            context->GetNodeName(), "Input x1/x2 dim k must be same, but given x1.k %ld, x2.k %ld.", commParas.dimKX1,
            commParas.dimKX2);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInferYShape(gert::InferShapeContext* context, CommParas& commParas)
{
    OP_LOGE_IF(
        CommonParamCheck(context, AG_IS_TRANS_A, AG_IS_TRANS_B, commParas) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "CommonParamCheck excute failed.");
    // 动态shape入图时 m轴-1时，不再进行(dimM * rankSize)的处理
    if (commParas.dimM == -1) {
        commParas.rankSize = 1;
    }
    // 不支持k = 0
    if (commParas.dimKX1 == 0) {
        commParas.dimM = commParas.dimN = 0;
        OP_LOGE(context->GetNodeName(), "X1/X2 are empty tensors with zero dimK.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(SUPPORT_DIM_SIZE);
    yShape->SetDim(0, commParas.dimM * commParas.rankSize);
    yShape->SetDim(1, commParas.dimN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInferGatherOutShape(gert::InferShapeContext* context, const CommParas& commParas,
                                                   const size_t gatherIndex)
{
    if (context->GetAttrs() == nullptr) {
        OP_LOGE(context->GetNodeName(), "get attrs failed.");
        return ge::GRAPH_FAILED;
    }
    const bool* isGatherOut = context->GetAttrs()->GetAttrPointer<bool>(gatherIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, isGatherOut);
    gert::Shape* gatherOutShape = context->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gatherOutShape);
    if (*isGatherOut) {
        gatherOutShape->SetDimNum(SUPPORT_DIM_SIZE);
        gatherOutShape->SetDim(0, commParas.dimM * commParas.rankSize);
        gatherOutShape->SetDim(1, commParas.dimKX1);
    } else {
        gatherOutShape->SetDimNum(1);
        gatherOutShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulCommonInferShape(gert::InferShapeContext* context, const size_t gatherIndex)
{
    CommParas commParas;
    OP_LOGE_IF(
        AllGatherMatmulInferYShape(context, commParas) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "InferShapeAllGatherMatmul inferYshape excute failed.");
    
    OP_LOGE_IF(
        AllGatherMatmulInferGatherOutShape(context, commParas, gatherIndex) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "InferShapeAllGatherMatmul inferYshape excute failed.");

    return GRAPH_SUCCESS;
}

/**
 * @brief x1和x2合法性校验
 *
 * @param context
 */
static ge::graphStatus CheckShapeForX(const gert::InferShapeContext* context)
{
    const auto x1Shape = context->GetInputShape(ALL_TO_ALL_MATMUL_INPUT_IDX.INDEX_IN_X1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK(x1Shape->GetDimNum() != DIM_TWO, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
              "x1 shape should be %ld, but the actual value is %ld.", DIM_TWO, x1Shape->GetDimNum()),
              return ge::GRAPH_FAILED);
    const auto x2Shape = context->GetInputShape(ALL_TO_ALL_MATMUL_INPUT_IDX.INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OPS_CHECK(x2Shape->GetDimNum() != DIM_TWO, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
              "x2 shape should be %ld, but the actual value is %ld.", DIM_TWO, x2Shape->GetDimNum()),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief x1Scale和x2Scale合法性校验
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckShapeForXScale(const gert::InferShapeContext* context, AlltoAllMatmulShapeInfo& shape)
{
    const auto attrs = context->GetAttrs();
    const int64_t* x1QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t* x2QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X2_QUANT_MODE);

    const auto x1ShapeScale = context->GetInputShape(INDEX_IN_X1_SCALE);
    if (x1ShapeScale != nullptr) {
        int64_t x1ShapeScaleDimNum = x1ShapeScale->GetDimNum();
        if (*x1QuantMode == X1_MXFP8_QUANT_NUM && *x2QuantMode == X2_MXFP8_QUANT_NUM) {
            // 只有mxfp8量化模式下，x1Scale才是3维
            OPS_CHECK(x1ShapeScaleDimNum != DIM_THREE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                      "x1Scale dim num must be %ld, but actual value is: %ld", DIM_THREE, x1ShapeScaleDimNum), return ge::GRAPH_FAILED);
            // x1Scale最后一维一定是2
            OPS_CHECK(x1ShapeScale->GetDim(x1ShapeScaleDimNum - 1) != X1_X2_SCALE_LAST_DIM, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                      "x1Scale last dim must be %ld, but actual value is: %ld",
                      X1_X2_SCALE_LAST_DIM, x1ShapeScale->GetDim(x1ShapeScaleDimNum - 1)), return ge::GRAPH_FAILED);
        } else {
            OPS_CHECK(x1ShapeScaleDimNum != DIM_ONE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                      "x1Scale shape must be %ld, but actual value is: %ld", DIM_ONE, x1ShapeScaleDimNum), return ge::GRAPH_FAILED);
        }
        // x1Scale第0维与m轴一致
        OPS_CHECK(x1ShapeScale->GetDim(0) != shape.m, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "x1Scale dim0 must be the same with matmul axis m, but actual x1Scale dim0 is: %ld, axis m is: %ld",
                  x1ShapeScale->GetDim(0), shape.m), return ge::GRAPH_FAILED);
    }

    const auto x2ShapeScale = context->GetInputShape(INDEX_IN_X2_SCALE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2ShapeScale);
    int64_t x2ShapeScaleDimNum = x2ShapeScale->GetDimNum();
    if (*x1QuantMode == X1_MXFP8_QUANT_NUM && *x2QuantMode == X2_MXFP8_QUANT_NUM) {
        // 只有mxfp8量化模式下，x2Scale才是3维
        OPS_CHECK(x2ShapeScaleDimNum != DIM_THREE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "x2Scale dim num must be %ld, but actual value is: %ld", DIM_THREE, x2ShapeScaleDimNum), return ge::GRAPH_FAILED);
        // x2Scale最后一维一定是2
        OPS_CHECK(x2ShapeScale->GetDim(x2ShapeScaleDimNum - 1) != X1_X2_SCALE_LAST_DIM, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "x2Scale last dim must be %ld, but actual value is: %ld",
                  X1_X2_SCALE_LAST_DIM, x2ShapeScale->GetDim(x2ShapeScaleDimNum - 1)), return ge::GRAPH_FAILED);
    } else {
        OPS_CHECK(x2ShapeScaleDimNum != DIM_ONE, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "x2Scale shape must be %ld, but actual value is: %ld", DIM_ONE, x2ShapeScaleDimNum), return ge::GRAPH_FAILED);
    }
    // x2Scale第0维与n轴一致
    OPS_CHECK(x2ShapeScale->GetDim(0) != shape.n, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
              "x2Scale dim0 must be the same with matmul axis n, but actual x2Scale dim0 is: %ld, axis n is: %ld",
              x2ShapeScale->GetDim(0), shape.n), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief k轴合法性校验
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckShapeForAxisK(const gert::InferShapeContext* context, AlltoAllMatmulShapeInfo& shape)
{
    OPS_CHECK(shape.k1 > AXIS_K_UPPER_LIMIT || shape.k2 > AXIS_K_UPPER_LIMIT, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                "axis k cannot exceed upper limit %ld, but actual k1 is: %ld, k2 is: %ld",
                AXIS_K_UPPER_LIMIT, shape.k1, shape.k2), return ge::GRAPH_FAILED);
    if (shape.k1 != shape.k2 / shape.rankNum) {
        OP_LOGE(context->GetNodeName(),
                "In allto_all_matmul x1.k must be the same to x2.k / rankSize, but actual get x1.k: %ld, x2.k: %ld, rankSize: %ld",
                shape.k1, shape.k2, shape.rankNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeForAllToAllMatmul(gert::InferShapeContext* context) {

}

ge::graphStatus AllToAllMatmulCommonInferShape(gert::InferShapeContext* context)
{
    OP_LOGE_IF(CheckShapeForAllToAllMatmul(context) != GRAPH_SUCCESS, GRAPH_FAILED,
               context->GetNodeName(), "Check shape for all_to_all_matmul excute failed.");
    return GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeForMatmulAllToAll(gert::InferShapeContext* context) {

}

ge::graphStatus MatmulAllToAllCommonInferShape(gert::InferShapeContext* context)
{
    OP_LOGE_IF(CheckShapeForMatmulAllToAll(context) != GRAPH_SUCCESS, GRAPH_FAILED,
               context->GetNodeName(), "Check shape for matmul_all_to_all excute failed.");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferMatmulReduceScatterCommon(gert::InferShapeContext* context)
{
    CommParas commParas;
    OP_LOGE_IF(
        CommonParamCheck(context, RS_IS_TRANS_A, RS_IS_TRANS_B, commParas) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "CommonParamCheck excute failed.");
    // 动态shape入图时 m轴-1时，不再进行(dimM / rankSize)的处理
    if (commParas.dimM == -1) {
        commParas.rankSize = 1;
    }
    if (commParas.dimKX1 == 0) {
        commParas.dimM = commParas.dimN = 0;
        OP_LOGE(context->GetNodeName(), "X1/X2 are empty tensors with zero dimK");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(SUPPORT_DIM_SIZE);
    yShape->SetDim(0, commParas.dimM / commParas.rankSize);
    yShape->SetDim(1, commParas.dimN);
    return GRAPH_SUCCESS;
}
} // namespace ops