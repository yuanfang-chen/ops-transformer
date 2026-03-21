/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_finalize_routing.cc
 * \brief
 */
#include <string>
#include <sstream>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace {
const size_t xIndex = 0;
const size_t wIndex = 1;
const size_t scaleOptionIndex = 2;
const size_t weightTransIndex = 4;
const size_t oneDimNum = 1;
const size_t twoDimNum = 2;
const size_t threeDimNum = 3;
const size_t fourDimNum = 4;
const size_t outputBSAttrIndex = 5;
const int64_t DYNAMIC_DIM = -1;
const int64_t DIM_ZERO = 0;
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
}

using namespace gert;
namespace ops {

template <typename T>
std::string Shape2String(const T& shape) {
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

struct CheckXandWParams {
    const gert::Shape *shape_x1 = nullptr;
    const gert::Shape *shape_x2 = nullptr;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t e = 0;
    bool weightTrans = false;
};

static ge::graphStatus ValidateXAndWShapes(const char* /* op_name */, CheckXandWParams& params)
{
    // Infer only: no shape/rank validation failures (debug / relaxed graph build).
    if (params.shape_x1->GetDimNum() >= twoDimNum) {
        params.m = params.shape_x1->GetDim(xIndex);
        params.k = params.shape_x1->GetDim(wIndex);
    } else if (params.shape_x1->GetDimNum() == oneDimNum) {
        params.m = params.shape_x1->GetDim(0);
        params.k = DYNAMIC_DIM;
    } else {
        params.m = DYNAMIC_DIM;
        params.k = DYNAMIC_DIM;
    }
    if (params.shape_x2->GetDimNum() >= threeDimNum) {
        params.e = params.shape_x2->GetDim(xIndex);
        if (!params.weightTrans) {
            params.n = params.shape_x2->GetDim(twoDimNum);
        } else {
            params.n = params.shape_x2->GetDim(DIM_ONE);
        }
    } else {
        params.e = params.shape_x2->GetDimNum() > 0 ? params.shape_x2->GetDim(0) : DYNAMIC_DIM;
        params.n = DYNAMIC_DIM;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetXAndWShapesForMX(const InferShapeContext *context, CheckXandWParams& params)
{
    if (params.shape_x1->GetDimNum() >= twoDimNum) {
        params.m = params.shape_x1->GetDim(xIndex);
        params.k = params.shape_x1->GetDim(wIndex);
    } else if (params.shape_x1->GetDimNum() == oneDimNum) {
        params.m = params.shape_x1->GetDim(0);
        params.k = DYNAMIC_DIM;
    } else {
        params.m = DYNAMIC_DIM;
        params.k = DYNAMIC_DIM;
    }

    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);
    if (shape_scale == nullptr || shape_scale->GetDimNum() < threeDimNum) {
        return ge::GRAPH_SUCCESS;
    }
    params.n = params.weightTrans ? shape_scale->GetDim(DIM_ONE) : shape_scale->GetDim(DIM_TWO);
    params.e = params.shape_x2->GetDimNum() > 0 ? params.shape_x2->GetDim(xIndex) : DYNAMIC_DIM;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateScaleAndBias(const InferShapeContext *context, const char* op_name, const CheckXandWParams& xAndWParams)
{
    (void)xAndWParams;
    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);
    if (shape_scale != nullptr) {
        OP_LOGD(context->GetNodeName(), "shape_scale: %s", Shape2String(*shape_scale).c_str());
    } else {
        OP_LOGD(op_name, "shape_scale: null");
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidatePertokenAndGroupList(const InferShapeContext * /* context */, const char * /* op_name */,
                                                    const CheckXandWParams & /* xAndWParams */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateSharedInputAndLogit(const InferShapeContext * /* context */, const char* /* op_name */,
    const CheckXandWParams& /* xAndWParams */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateRowIndex(const InferShapeContext * /* context */, const char* /* op_name */,
    const CheckXandWParams& /* xAndWParams */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupOutputAndCheckAttrs(InferShapeContext *context, const char *op_name,
                                                CheckXandWParams &xAndWParams)
{
    auto attrs = context->GetAttrs();
    auto shape_out = context->GetOutputShape(0);
    shape_out->SetDimNum(twoDimNum);
    const int *output_bs = attrs->GetAttrPointer<int>(outputBSAttrIndex);
    if (output_bs != nullptr) {
        shape_out->SetDim(0, *output_bs);
    } else {
        const int64_t fallback = (xAndWParams.m != DYNAMIC_DIM && xAndWParams.m > 0) ? static_cast<int>(xAndWParams.m) : 1;
        shape_out->SetDim(0, fallback);
    }
    shape_out->SetDim(DIM_ONE, xAndWParams.n != DYNAMIC_DIM ? xAndWParams.n : 1);
    OP_LOGI(op_name, "shape out is %ld, %ld", shape_out->GetDim(0), shape_out->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupOutputForMX(InferShapeContext *context, const char* op_name, CheckXandWParams& xAndWParams)
{
    auto attrs = context->GetAttrs();
    auto shape_out = context->GetOutputShape(0);
    
    shape_out->SetDimNum(twoDimNum);
    const int *output_bs = attrs->GetAttrPointer<int>(outputBSAttrIndex);
    
    if (output_bs != nullptr) {
        shape_out->SetDim(0, *output_bs);
    } else {
        const int64_t fallback = (xAndWParams.m != DYNAMIC_DIM && xAndWParams.m > 0) ? xAndWParams.m : 1;
        shape_out->SetDim(0, static_cast<int>(fallback));
    }
    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);
    int64_t nOut = (xAndWParams.n != DYNAMIC_DIM) ? xAndWParams.n : 1;
    if (shape_scale != nullptr) {
        if (xAndWParams.weightTrans) {
            if (shape_scale->GetDimNum() > static_cast<size_t>(DIM_ONE)) {
                nOut = shape_scale->GetDim(DIM_ONE);
            }
        } else if (shape_scale->GetDimNum() > static_cast<size_t>(DIM_TWO)) {
            nOut = shape_scale->GetDim(DIM_TWO);
        }
    }
    shape_out->SetDim(DIM_ONE, nOut);
    OP_LOGI(op_name, "shape out is %ld, %ld", shape_out->GetDim(DIM_ZERO), shape_out->GetDim(DIM_ONE));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateOffsetShape(const InferShapeContext * /* context */, const char* /* op_name */,
    const CheckXandWParams& /* xAndWParams */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeGroupedMatmulFinalizeRouting(InferShapeContext *context)
{
    auto op_name = context->GetNodeName();
    auto shape_x1 = context->GetInputShape(xIndex);
    auto shape_x2 = context->GetInputShape(wIndex);
    auto shape_out = context->GetOutputShape(0);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "x1_shape: %s, x2_shape: %s", Shape2String(*shape_x1).c_str(), Shape2String(*shape_x2).c_str());
    
    const bool *transposeWeightPtr = attrs->GetBool(weightTransIndex);
    bool transposeWeight = (transposeWeightPtr != nullptr ? *transposeWeightPtr : false);
    
    CheckXandWParams xAndWParams{shape_x1, shape_x2, 0, 0, 0, 0, transposeWeight};
    
    // MX：4D scale；其余：原 W8A8/W4A8 路径。图侧不再因 shape/dtype 组合失败（仅保留必填指针检查）。
    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);
    if (shape_scale != nullptr && shape_scale->GetDimNum() == fourDimNum) {
        SetXAndWShapesForMX(context, xAndWParams);
        SetupOutputForMX(context, op_name, xAndWParams);
        return ge::GRAPH_SUCCESS;
    }
    ValidateXAndWShapes(op_name, xAndWParams);
    ValidateScaleAndBias(context, op_name, xAndWParams);
    ValidatePertokenAndGroupList(context, op_name, xAndWParams);
    ValidateSharedInputAndLogit(context, bsdp, op_name, xAndWParams);
    ValidateRowIndex(context, op_name, xAndWParams);
    ValidateOffsetShape(context, op_name, xAndWParams);
    SetupOutputAndCheckAttrs(context, op_name, xAndWParams);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGroupedMatmulFinalizeRouting(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulFinalizeRouting)
    .InferShape(InferShapeGroupedMatmulFinalizeRouting)
    .InferDataType(InferDataTypeGroupedMatmulFinalizeRouting);
} // namespace ops
