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
 * \file mhc_post_infershape.cpp
 * \brief MhcPost infershape implementation
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime_util.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;

namespace ops {

static constexpr size_t INDEX_X = 0;
static constexpr size_t INDEX_HRES = 1;
static constexpr size_t INDEX_HOUT = 2;
static constexpr size_t INDEX_HPOST = 3;
static constexpr size_t INDEX_Y = 0;
static constexpr uint32_t SOC_VERSION_SIZE = 32;
static constexpr size_t DIMS_ONE = 1;
static constexpr size_t DIMS_TWO = 2;
static constexpr size_t DIMS_THREE = 3;
static constexpr size_t DIMS_FOUR = 4;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2LL;
static constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;


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

static void ShowInputShapeInfo(gert::InferShapeContext *context, const gert::Shape *xShape, const gert::Shape *hResShape, const gert::Shape *hOutShape, const gert::Shape *hPostShape)
{
    OP_LOGD(context, "xShape is: %s.", Ops::Base::ToString(*xShape).c_str());
    OP_LOGD(context, "hResShape is: %s.", Ops::Base::ToString(*hResShape).c_str());
    OP_LOGD(context, "hOutShape is: %s.", Ops::Base::ToString(*hOutShape).c_str());
    OP_LOGD(context, "hPostShape is: %s.", Ops::Base::ToString(*hPostShape).c_str());    
}

static void ShowOutputShapeInfo(gert::InferShapeContext *context, const gert::Shape *yShape)
{
    OP_LOGD(context, "yShape is: %s.", Ops::Base::ToString(*yShape).c_str());
}

bool IsPlatform950(const char *nodeName)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "Cannot get platform info!");
        return false;
    } else {
        return (optionalInfo.soc_version.find("950") != std::string::npos);
    }
}

static ge::graphStatus InferShapeForMhcPost(gert::InferShapeContext* context)
{
    if (!IsPlatform950(context->GetNodeName())) {
        OP_LOGD(context, "The current Platform is not support to do MhcPostInfershape.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Begin to do MhcPostInfershape.");
    const gert::Shape* xShape = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* hResShape = context->GetInputShape(INDEX_HRES);
    OP_CHECK_NULL_WITH_CONTEXT(context, hResShape);
    const gert::Shape* hOutShape = context->GetInputShape(INDEX_HOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, hOutShape);
    const gert::Shape* hPostShape = context->GetInputShape(INDEX_HPOST);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPostShape);
    gert::Shape* yShape = context->GetOutputShape(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*yShape);
        OP_LOGD(context->GetNodeName(), "MhcPost infershape handles unknown rank.");
        return ge::GRAPH_SUCCESS;
    }
    size_t xDims = xShape->GetDimNum();
    if (IsUnknownShape(*xShape)) {
        yShape->SetDimNum(xDims);
        for (size_t i = 0; i < xDims; ++i) {
            yShape->SetDim(i, UNKNOWN_DIM_VALUE);
        }
        OP_LOGD(context->GetNodeName(), "MhcPost infershape handles unknown shape.");
        return ge::GRAPH_SUCCESS;
    }
    
    size_t hResDims = hResShape->GetDimNum();
    size_t hOutDims = hOutShape->GetDimNum();
    size_t hPostDims = hPostShape->GetDimNum();
    OP_CHECK_IF((xDims != DIMS_THREE) && (xDims != DIMS_FOUR),
        OP_LOGE(context->GetNodeName(), "The dim of x should be 3 or 4, but got %d", xDims),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDims != hResDims,
        OP_LOGE(context->GetNodeName(), "The dims of x and hRes should be equal, but xDims is %d and hResDims is %d", xDims, hResDims),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((hOutDims != DIMS_TWO) && (hOutDims != DIMS_THREE),
        OP_LOGE(context->GetNodeName(), "The dim of hOut should be 2 or 3, but got %d", hOutDims),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(hOutDims != hPostDims,
        OP_LOGE(context->GetNodeName(), "The dims of hOut and hPost should be equal, but hOutDims is %d and hPostDims is %d", hOutDims, hPostDims),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(hResShape->GetDim(hResDims - DIMS_ONE) != hResShape->GetDim(hResDims - DIMS_TWO),
        OP_LOGE(context->GetNodeName(), "The last two dims of hRes should be same."),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDims - DIMS_ONE; ++i) {
        int32_t xDimI = xShape->GetDim(i);
        int32_t hResDimI = hResShape->GetDim(i);
        OP_CHECK_IF(xDimI != hResDimI,
            OP_LOGE(context->GetNodeName(), "xShape[%d] and hResShape[%d] should be same, but xShape[%d] is %d, hResShape[%d] is %d", i, i, i, xDimI, i, hResDimI),
            return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < hOutDims - DIMS_ONE; ++i) {
        int32_t hOutDimI = hOutShape->GetDim(i);
        int32_t hPostDimI = hPostShape->GetDim(i);
        OP_CHECK_IF(hOutDimI != hPostDimI,
            OP_LOGE(context->GetNodeName(), "hOutShape[%d] and hPostShape[%d] should be same, but hOutShape[%d] is %d, hPostShape[%d] is %d", i, i, i, hOutDimI, i, hPostDimI),
            return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < hResDims - DIMS_TWO; ++i) {
        int32_t hResDimI = hResShape->GetDim(i);
        int32_t hOutDimI = hOutShape->GetDim(i);
        OP_CHECK_IF(hResDimI != hOutDimI,
            OP_LOGE(context->GetNodeName(), "The shapes of hRes and hOut are invalid."),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(xShape->GetDim(xDims - DIMS_ONE) != hOutShape->GetDim(hOutDims - DIMS_ONE),
        OP_LOGE(context->GetNodeName(), "The last dims of x and hOut should be same."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(hPostShape->GetDim(hPostDims - DIMS_ONE) != hResShape->GetDim(hResDims - DIMS_ONE),
        OP_LOGE(context->GetNodeName(), "The last dims of hPost and hRes should be same."),
        return ge::GRAPH_FAILED);

    // Output shape is same as input x
    yShape->SetDimNum(xDims);
    for (size_t i = 0; i < xDims; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    ShowInputShapeInfo(context, xShape, hResShape, hOutShape, hPostShape); 
    ShowOutputShapeInfo(context, yShape);

    OP_LOGD(context, "End to do MhcPostInfershape.");

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMhcPost(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MhcPostInferDataType.");
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    // Output dtype is same as x
    const ge::DataType xDtype = context->GetInputDataType(INDEX_X);
    context->SetOutputDataType(INDEX_Y, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do MhcPostInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPost)
    .InferShape(InferShapeForMhcPost)
    .InferDataType(InferDataTypeForMhcPost);

} // namespace ops
