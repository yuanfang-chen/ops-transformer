/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_sinkhorn_infershape.cpp
 * \brief mhc_sinkhorn_infershape
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;
using namespace std;

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;

constexpr size_t INDEX_EPS = 0;
constexpr size_t INDEX_NUM_ITERS = 1;
constexpr size_t INDEX_OUT_FLAG = 2;

constexpr size_t BSNN_DIMS = 4;
constexpr size_t TNN_DIMS = 3;
} // namespace

namespace ops {

void SetUnknownRank(gert::Shape &shape)
{
    shape.SetDimNum(0);
    shape.AppendDim(UNKNOWN_RANK_DIM_VALUE);
}

bool IsUnknownRank(const gert::Shape &shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

bool IsUnknownShape(const gert::Shape &shape)
{
    size_t dimNum = shape.GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        if (shape.GetDim(i) == UNKNOWN_DIM_VALUE) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus InferShape4MhcSinkhorn(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do MhcSinkhornInfershape.");
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    auto epsPtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(INDEX_EPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsPtr);
    auto numItersPtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(INDEX_NUM_ITERS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numItersPtr);
    auto outFlagPtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(INDEX_OUT_FLAG);
    OP_CHECK_NULL_WITH_CONTEXT(context, outFlagPtr);

    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*yShape);
        OP_LOGD(context->GetNodeName(), "MhcSinkhorn infershape handles unknown rank.");
        return ge::GRAPH_SUCCESS;
    }
    size_t xDims = xShape->GetDimNum();
    if (IsUnknownShape(*xShape)) {
        yShape->SetDimNum(xDims);
        for (size_t i = 0; i < xDims; ++i) {
            yShape->SetDim(i, UNKNOWN_DIM_VALUE);
        }
        OP_LOGD(context->GetNodeName(), "MhcSinkhorn infershape handles unknown shape.");
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF((xDims != DIMS_THREE) && (xDims != DIMS_FOUR),
                OP_LOGE(context->GetNodeName(), "The dim of x should be 3 or 4, but got %lu", xDims),
                return ge::GRAPH_FAILED);

    // Output shape is same as input x
    yShape->SetDimNum(xDims);
    for (size_t i = 0; i < xDims; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    ShowInputShapeInfo(context, xShape);
    ShowOutputShapeInfo(context, yShape);

    OP_LOGD(context, "End to do MhcSinkhornInfershape.");

    return GRAPH_SUCCESS; 
}

static graphStatus InferDtype4MhcSinkhorn(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "MhcSinkhornInferDtype enter");
    const ge::DataType x = context->GetInputDataType(0);
    context->SetOutputDataType(0, x);
    OP_LOGD(context->GetNodeName(), "MhcSinkhornInferDtype end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcSinkhorn)
    .InferShape(InferShape4MhcSinkhorn)
    .InferDataType(InferDtype4MhcSinkhorn);
} // namespace ops