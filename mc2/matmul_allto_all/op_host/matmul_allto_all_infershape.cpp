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
 * \file matmul_allto_all_infershape.cpp
 * \brief 图模式（动态图/静态图）走infershape
 */

#include "op_mc2.h"
#include <register/op_impl_registry.h>
#include "util/math_util.h"
#include "mc2_log.h"
#include "mc2_common_infershape.h"

using Ops::Base::CeilDiv;

namespace ops {
constexpr size_t INDEX_IN_X1 = 0;
constexpr size_t INDEX_IN_X2 = 1;
constexpr size_t INDEX_IN_X1_SCALE = 3;
constexpr size_t INDEX_IN_X2_SCALE = 4;
constexpr size_t INDEX_ATTR_GROUP = 0;
constexpr size_t INDEX_ATTR_Y_DTYPE = 3;
constexpr size_t INDEX_ATTR_X1_QUANT_MODE = 4;
constexpr size_t INDEX_ATTR_X2_QUANT_MODE = 5;
constexpr size_t INDEX_ATTR_TRANS_X1 = 8;
constexpr size_t INDEX_ATTR_TRANS_X2 = 9;
constexpr size_t INDEX_OUT = 0;
constexpr uint64_t NUM_ONE = 1;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t NUM_MINUS_ONE = -1;
constexpr uint64_t X1_QUANT_MODE_NUM = 3;
constexpr uint64_t X2_QUANT_MODE_NUM = 2;
static const char* INNER_DEBUG = "MC2: MatmulAlltoAll InferShape Debug";
const std::set<int> SUPPORT_RANK_NUM{2, 4, 8, 16};

struct MatmulAlltoAllShapeInfo {
    uint64_t output_dim;
    uint64_t rankNum;
    uint64_t m;
    uint64_t n;
    uint64_t k;
};

/**
 * @brief 校验x1scale的维度
 *
 * @param scale_shape
 * @param shape
 */
static ge::graphStatus CheckX1ScaleShape(
    const gert::Shape* scale_shape, MatmulAlltoAllShapeInfo& shape)
{
    if (scale_shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape expect_scale;
    expect_scale = {static_cast<long>(shape.m)};
    OPS_CHECK(
        expect_scale != *scale_shape,
        CUBE_INNER_ERR_REPORT(
            INNER_DEBUG, "Expect x1scale shape to be [%ld], but actually [%ld]", expect_scale[0],
            scale_shape->GetDim(0U)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验x2scale的维度
 *
 * @param scale_shape
 * @param shape
 */
static ge::graphStatus CheckX2ScaleShape(
    const gert::Shape* scale_shape, MatmulAlltoAllShapeInfo& shape)
{
    if (scale_shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape expect_scale;
    expect_scale = {static_cast<long>(shape.n)};
    OPS_CHECK(
        expect_scale != *scale_shape,
        CUBE_INNER_ERR_REPORT(
            INNER_DEBUG, "Expect x2scale shape to be [%ld], but actually [%ld]", expect_scale[0],
            scale_shape->GetDim(0U)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验MatmulAlltoAll输入shape，并记录输入m，n，k大小
 *
 * @param context
 * @param shape
 * @param is_arn
 */
static ge::graphStatus CheckShapeForMatmulAlltoAll(const gert::InferShapeContext* context, MatmulAlltoAllShapeInfo& shape, bool is_arn)
{
    const auto x1_shape = context->GetInputShape(INDEX_IN_X1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    const size_t x1_dim = x1_shape->GetDimNum();
    OPS_CHECK(x1_dim != DIM_TWO,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid dim number %zu of x1.", x1_dim), return ge::GRAPH_FAILED);
    const auto x2_shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    const size_t x2_dim = x2_shape->GetDimNum();
    OPS_CHECK(x2_dim != DIM_TWO,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid dim number %zu of x2.", x2_dim),
        return ge::GRAPH_FAILED);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* is_trans_x1 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X1);
    OPS_CHECK(
        is_trans_x1 != nullptr && *is_trans_x1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
        "x1 does not support transpose in matmul allto all."), return ge::GRAPH_FAILED);
    const bool* is_trans_x2 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X2);
    const bool trans_x2 = ((is_trans_x2 != nullptr) && (*is_trans_x2));
    shape.m = x1_shape->GetDim(0U);
    shape.k = x1_shape->GetDim(1U);
    shape.n = trans_x2 ? x2_shape->GetDim(0U) : x2_shape->GetDim(1U);
    const auto shapeX2KIndex = trans_x2 ? 1U : 0U;
    bool is_dynamic_shape = (shape.k == NUM_MINUS_ONE || x2_shape->GetDim(shapeX2KIndex) == NUM_MINUS_ONE);
    if (!is_dynamic_shape) {
        OPS_CHECK((shape.k != x2_shape->GetDim(shapeX2KIndex)),
            CUBE_INNER_ERR_REPORT(
                context->GetNodeName(), "Invalid shape for x1(k): %ld, x2(k): %ld", shape.k,
                x2_shape->GetDim(shapeX2KIndex)), return ge::GRAPH_FAILED);
        const int* x1_quant_mode = attrs->GetAttrPointer<int>(INDEX_ATTR_X1_QUANT_MODE);
        const int* x2_quant_mode = attrs->GetAttrPointer<int>(INDEX_ATTR_X2_QUANT_MODE);
        const size_t x1_scale_idx =
            is_arn ? static_cast<size_t>(MC2AddRmsNormInputIdx::K_SCALE) : INDEX_IN_X1_SCALE;
        const size_t x2_scale_idx =
            is_arn ? static_cast<size_t>(MC2AddRmsNormInputIdx::K_SCALE) : INDEX_IN_X2_SCALE;
        if (*x1_quant_mode == X1_QUANT_MODE_NUM && *x2_quant_mode == X2_QUANT_MODE_NUM) {
            OPS_CHECK(
                CheckX1ScaleShape(context->GetOptionalInputShape(x1_scale_idx), shape) != ge::GRAPH_SUCCESS,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check x1 scale shape."), return ge::GRAPH_FAILED);
            OPS_CHECK(
                CheckX2ScaleShape(context->GetOptionalInputShape(x2_scale_idx), shape) != ge::GRAPH_SUCCESS,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check x2 scale shape."), return ge::GRAPH_FAILED);
        }
    }
    shape.output_dim = x1_dim;
    OP_LOGD(INNER_DEBUG, "Matmul x1 dim %zu m %ld n %ld k %ld.", x1_dim, shape.m, shape.n, shape.k);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取，校验并记录卡数
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckRankDim(gert::InferShapeContext* context, MatmulAlltoAllShapeInfo& shape)
{
    const auto attrs = context->GetAttrs();
    const char* groupStr = attrs->GetAttrPointer<char>(INDEX_ATTR_GROUP);
    OP_LOGE_IF(groupStr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "Get matmul allto all group failed.");
    uint32_t rankDim = 0;
    // 通过通信域标识获取卡数
    if ((Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(groupStr, &rankDim)) != HCCL_SUCCESS) {
            OP_LOGE(
                context->GetNodeName(), "Get rank size failed, group [%s], rankDim [%u]", groupStr, rankDim);
            return ge::GRAPH_FAILED;
        }
    OPS_CHECK(rankDim == 0,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid rank number %zu in matmul allto all.", rankDim),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SUPPORT_RANK_NUM.find(rankDim) == SUPPORT_RANK_NUM.end(),
                    OP_LOGE(INNER_DEBUG, "Rank number should be 2 or 4 or 8 or 16, but the actual value is %ld.", rankDim),
                    return ge::GRAPH_FAILED);
    shape.rankNum = rankDim;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出shape
 *
 * @param context
 */
static ge::graphStatus InferShapeMatmulAlltoAll(gert::InferShapeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    OP_LOGD(INNER_DEBUG, "Start to infer shape of matmul allto all.");
    MatmulAlltoAllShapeInfo shape;
    OPS_CHECK(
        CheckShapeForMatmulAlltoAll(context, shape, false) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check shape for matmul allto all"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        CheckRankDim(context, shape) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check rank dim for matmul allto all."),
        return ge::GRAPH_FAILED);
    auto shape_out = context->GetOutputShape(INDEX_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_out);
    uint64_t out_first_dim = shape.m * shape.rankNum;
    uint64_t out_second_dim = CeilDiv(shape.n, shape.rankNum);
    shape_out->SetDimNum(shape.output_dim);
    if (shape.output_dim == DIM_TWO) {
        shape_out->SetDim(0U, out_first_dim);
        shape_out->SetDim(1U, out_second_dim);
    }
    OP_LOGI(
        INNER_DEBUG, "Matmul allto all output shape after infer shape, dim: %zu m: %ld n: %ld.", shape.output_dim, out_first_dim, out_second_dim);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出数据类型
 *
 * @param context
 */
static ge::graphStatus InferDataTypeMatmulAlltoAll(gert::InferDataTypeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    OP_LOGD(INNER_DEBUG, "Start to infer datatype of matmul allto all.");
    ge::DataType y_type = context->GetOutputDataType(0U);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* y_dtype_ptr = attrs->GetInt(INDEX_ATTR_Y_DTYPE);
    const uint64_t y_data_type = (y_dtype_ptr != nullptr ? *y_dtype_ptr : ge::DataType::DT_UNDEFINED);
    if (y_data_type != ge::DataType::DT_UNDEFINED) {
        y_type = static_cast<ge::DataType>(y_data_type);
    }
    return context->SetOutputDataType(0U, y_type);
}

IMPL_OP_INFERSHAPE(MatmulAlltoAll).InferShape(InferShapeMatmulAlltoAll).InferDataType(InferDataTypeMatmulAlltoAll);
} // namespace ops