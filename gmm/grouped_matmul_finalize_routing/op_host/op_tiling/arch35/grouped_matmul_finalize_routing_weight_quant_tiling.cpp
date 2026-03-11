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
 * \file grouped_matmul_finalize_routing_weight_quant_tiling.cpp
 * \brief
 */
#include "grouped_matmul_finalize_routing_weight_quant_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/grouped_matmul_finalize_routing_weight_quant_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GroupedMatmulFinalizeRoutingArch35WeightQuantTiling;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace optiling {
using namespace GroupedMatmulFinalizeRoutingArch35TilingConstant;
using namespace GmmConstant;

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulFinalizeRouting, GMMFRWeightQuantTiling, GMMFR_WEIGHT_QUANT_TILING_VEC_ANTIQUANT);

enum DataSize GetSizeByDataType(ge::DataType dType) {
    if (dType == ge::DT_FLOAT4_E2M1 || dType == ge::DT_FLOAT4_E1M2 || dType == ge::DT_INT4) {
        return B4_DATA_SIZE;
    }
    if (ge::GetSizeByDataType(dType) == B8_DATA_SIZE) {
        return B8_DATA_SIZE;
    }
    return RESERVED;
}

bool GMMFRWeightQuantTiling::IsCapable()
{
    // 当前无多份模板，无需模板合法校验
    return true;
}


ge::graphStatus GMMFRWeightQuantTiling::GetPlatformInfo()
{
    auto compileInfoPtr = context->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
            OPS_REPORT_CUBE_INNER_ERR("GroupedMatmulFinalizeRouting", "CompileInfo is null"),
            return ge::GRAPH_FAILED);
    compileInfoPtr_ = compileInfoPtr;

    OP_LOGI(context_, "Compile info: aicNum(%lu) ubSize(%lu) l1Size(%lu) l0aSize(%lu) l0bSize(%lu) l0cSize(%lu).",
              compileInfoPtr_->aicNum, compileInfoPtr_->ubSize, compileInfoPtr_->l1Size, compileInfoPtr_->l0ASize,
              compileInfoPtr_->l0BSize, compileInfoPtr_->l0CSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OP_CHECK_IF(!InferScenario(), OP_LOGE(inputParams_.opName, "Failed to infer scenario."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!RunCheckFunc(), OP_LOGE(inputParams_.opName, "Failed to check input params."),
        return ge::GRAPH_FAILED);
    RunSetInputFunc();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantTiling::DoOpTiling()
{
    tilingData_.groupListType = inputParams_.groupListType;
    tilingData_.hasBias = inputParams_.hasBias;
    tilingData_.coreNum = compileInfoPtr_->aicNum;
    tilingData_.groupNum = inputParams_.groupNum;
    tilingData_.outputBs = inputParams_.outputBs;
    tilingData_.sharedInputOffset = inputParams_.shareInputOffset;
    tilingData_.sharedInputLen = inputParams_.sharedInputLen;
    tilingData_.sharedInputWeight = inputParams_.sharedInputWeight;

    tilingData_.kSize = inputParams_.kSize;
    tilingData_.nSize = inputParams_.nSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantTiling::DoLibApiTiling()
{
    // 当前为低阶api实现，无对应tiling
    return ge::GRAPH_SUCCESS;
}

uint64_t GMMFRWeightQuantTiling::GetTilingKey() constexpr
{
    return 0;
}
// 6、计算Workspace 大小
ge::graphStatus GMMFRWeightQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // get second variable
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is nullptr."),
                return ge::GRAPH_FAILED);  // check workspaces is not null
    workspaces[0] = 16777216U;  // 16 * 1024 * 1024: default workspace size
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus GMMFRWeightQuantTiling::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(context_->GetNodeName(), "RawTilingData is nullptr."),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

void GMMFRWeightQuantTiling::Reset(){
    tilingData_ = GMMFinalizeRoutingWeightQuantTilingData();
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                         context_->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
}

bool GMMFRWeightQuantTiling::InferScenario() {
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input xDesc is nullptr."), return false);
    auto wDesc = context_->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input wDesc is nullptr."), return false);

    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    if (wFormat == ge::FORMAT_FRACTAL_NZ_C0_16 || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32 ||
        wFormat == ge::FORMAT_FRACTAL_NZ_C0_4 || wFormat == ge::FORMAT_FRACTAL_NZ_C0_2) {
        wFormat = ge::FORMAT_FRACTAL_NZ;
    }

    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    auto scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : ge::DT_INT8;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    auto perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : ge::DT_INT8;
    auto xDtype = xDesc->GetDataType();
    auto wDtype = wDesc->GetDataType();
    OP_LOGD(context->GetNodeName(), "Current xDtype: %s, wDtype: %s, scaleDtype: %s, perTokenScaleDtype: %s, wFormat: %s",
        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(wDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(perTokenScaleDtype).c_str(),
        wFormat == ge::FORMAT_FRACTAL_NZ ? "FRACTAL_NZ" : "ND");
    if (xDtype == ge::DT_FLOAT8_E4M3FN && wDtype == ge::DT_FLOAT4_E2M1 && wFormat == ge::FORMAT_FRACTAL_NZ && 
        scaleDtype == ge::DT_FLOAT8_E8M0 && perTokenScaleDtype == ge::DT_FLOAT8_E8M0) {
        scenarioType_ = ScenarioType::MX_A8W4_WEIGHT_NZ;
        OP_LOGD(context->GetNodeName(), "Enable MX-A8W4-WEIGHT-NZ mode.");
        SetMxA8W4NzConditionFunc();
        SetMxA8W4NzInputFunc();
        return true;
    }

    OP_LOGE(context->GetNodeName(), "Only support MX-A8W4-WEIGHT-NZ mode. current xDtype: %s, wDtype: %s, scaleDtype");
    return false;
}

bool CheckMxA8W4NzInputPtr(gert::TilingContext *contex) {
    auto xDesc = contex->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input xDesc is nullptr."), return false);
    auto xStorageShape = contex->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input xStorageShape is nullptr."),
                return false);

    auto wDesc = contex->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input wDesc is nullptr."), return false);
    auto wStorageShape = contex->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input wStorageShape is nullptr."),
                return false);

    auto scaleDesc = contex->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input scaleDesc is nullptr."), return false);
    auto scaleStorageShape = contex->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input scaleStorageShape is nullptr."),
                return false);

    auto pertokenScaleDesc = contex->GetInputDesc(PERTOKEN_SCALE_INDEX);
    OP_CHECK_IF(pertokenScaleDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input pertokenScaleDesc is nullptr."), return false);
    auto pertokenScaleStorageShape = contex->GetInputShape(PERTOKEN_SCALE_INDEX);
    OP_CHECK_IF(pertokenScaleStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input pertokenScaleStorageShape is nullptr."),
                return false);
    return true;

    // todo 其他校验需要为空
}

bool CheckMxA8W4NzAttrPtr(gert::TilingContext *contex) {
    auto attrs = contex->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(contex->GetNodeName(), "Attrs is nullptr"), return false);

    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X);
    OP_CHECK_IF((transposeXPtr != nullptr && (*transposeXPtr)), OP_LOGE(contex->GetNodeName(), "transpose_x should be false or nullptr, but now is true"), return false);

    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_W);
    OP_CHECK_IF(transposeWeightPtr == nullptr, OP_LOGE(contex->GetNodeName(), "transpose_w should be true, but now is nullptr"), return false);
    OP_CHECK_IF((transposeWeightPtr != nullptr && !(*transposeWeightPtr)), OP_LOGE(contex->GetNodeName(), "transpose_w should be true, but now is false"), return false);

    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    OP_CHECK_IF(groupListTypePtr == nullptr, OP_LOGE(contex->GetNodeName(), "group_list_type should not be nullptr"), return false);
    OP_CHECK_IF(*groupListTypePtr != 0 && *groupListTypePtr != 1,
        OP_LOGE(contex->GetNodeName(), "Attr groupListType must be 0 or 1, actual is %d.",
            *groupListTypePtr),
        return false);

    const int64_t *outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    OP_CHECK_IF(outputDtypePtr == nullptr, OP_LOGE(contex->GetNodeName(), "dtype should not be nullptr"), return false);
    OP_CHECK_IF(*outputDtypePtr != 0,
        OP_LOGE(contex->GetNodeName(),
                "Attr dtype only support 0(float32), actual is %d.", *outputDtypePtr),
        return false);
    return true;
}

bool CheckMxA8W4InputShape(gert::TilingContext *contex) {
    auto xStorageShape = contex->GetInputShape(X_INDEX);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum != DIM_NUM_X,
                OP_LOGE(contex->GetNodeName(), "The dimension of x must be %u, actual is %zu", DIM_NUM_X, xDimNum),
                return false);

    auto wStorageShape = contex->GetInputShape(W_INDEX);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    auto wDimNum = wShape.GetDimNum();
    OP_CHECK_IF(
        wDimNum != DIM_NUM_WEIGHT,
        OP_LOGE(contex->GetNodeName(), "The dimension of w must be %u, actual is %zu", DIM_NUM_WEIGHT, wDimNum),
        return false);

    auto scaleStorageShape = contex->GetInputShape(SCALE_INDEX);
    const gert::Shape &scaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = scaleShape.GetDimNum();
    OP_CHECK_IF(scaleDimNum != DIM_NUM_MX_SCALE,
                OP_LOGE(contex->GetNodeName(), "The dimension of scale must be %u, actual is %zu",
                        DIM_NUM_MX_SCALE, scaleDimNum),
                return false);

    auto pertokenScaleStorageShape = contex->GetInputShape(PERTOKEN_SCALE_INDEX);
    const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
    auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
    OP_CHECK_IF(pertokenScaleDimNum != DIM_NUM_MX_PERTOKENSCALE,
                OP_LOGE(contex->GetNodeName(), "The dimension of pertokenScale must be %u, actual is %zu",
                        DIM_NUM_MX_PERTOKENSCALE, pertokenScaleDimNum),
                return false);
    // todo 校验 n, k 是否相等， group  size是否等于32， scale m  和x m的相等 desc是否存在
    return true;
}

bool CheckMxA8W4AttrWithInput(gert::TilingContext *contex) {
    auto attrs = contex->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(contex->GetNodeName(), "Attrs is nullptr"), return false);
    
    const int64_t *outputBSPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OUTPUT_BS);
    int64_t outputBS = outputBSPtr != nullptr ? *outputBSPtr : 0;
    OP_CHECK_IF(outputBS < 0,
        OP_LOGE(contex->GetNodeName(), "Attr outputBS should be >=0."),
        return false);

    auto sharedInputDesc = contex->GetInputDesc(SHARE_INPUT_INDEX);
    if (sharedInputDesc != nullptr) {
        auto sharedInputStorageShape = contex->GetInputShape(SHARE_INPUT_INDEX);
        OP_CHECK_IF(sharedInputStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input sharedInputStorageShape should not be nullptr."),
                    return false);

        const float *shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
        OP_CHECK_IF(shareInputWeightPtr == nullptr, OP_LOGE(contex->GetNodeName(), "Input shareInputWeightPtr should not be nullptr."),
            return false);
    
        const int64_t *shareInputOffsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
        OP_CHECK_IF(shareInputOffsetPtr == nullptr, OP_LOGE(contex->GetNodeName(), "Input shareInputOffsetPtr should not be nullptr."),
            return false);

        OP_CHECK_IF(
            (*shareInputOffsetPtr) < 0,
            OP_LOGE(contex->GetNodeName(), "Attr shareInputOffset should be >=0."),
            return false);
        OP_CHECK_IF(
            (*shareInputOffsetPtr) + sharedInputStorageShape->GetOriginShape().GetDim(0) > outputBS,
            OP_LOGE(contex->GetNodeName(), "Attr shareInputOffset[%lu] should less or equal to outputBS[%ld].", 
            (*shareInputOffsetPtr), outputBS),
            return false);
    }

    return true;
}

bool GMMFRWeightQuantTiling::SetMxA8W4NzConditionFunc() {
    checkConditionFuncs_.push_back(CheckMxA8W4NzInputPtr);
    checkConditionFuncs_.push_back(CheckMxA8W4NzAttrPtr);
    checkConditionFuncs_.push_back(CheckMxA8W4InputShape);
    checkConditionFuncs_.push_back(CheckMxA8W4AttrWithInput);
    return true;
}

bool GMMFRWeightQuantTiling::RunCheckFunc()
{
    for(int64_t conditionIdx = 0; conditionIdx < checkConditionFuncs_.size(); conditionIdx++) {
        OP_CHECK_IF(!checkConditionFuncs_[conditionIdx](context_), OP_LOGE(context_->GetNodeName(), "Failed to check input params in rule[%lu].", conditionIdx), return false);
    }
    return true;
}

bool SetMxA8W4NzAttrs(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    auto attrs = contex->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(contex->GetNodeName(), "Attrs is nullptr"), return false);
    
    inputParams.xTrans = false;
    inputParams.wTrans = true;

    const float *shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
    inputParams.sharedInputWeight = shareInputWeightPtr == nullptr ? 0.0f : *shareInputWeightPtr;

    const int64_t *shareInputOffsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
    inputParams.shareInputOffset = shareInputOffsetPtr == nullptr ? 0 : *shareInputOffsetPtr;

    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    inputParams.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : 0;

    const int64_t *outputBSPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OUTPUT_BS);
    inputParams.outputBS = outputBSPtr != nullptr ? *outputBSPtr : inputParams.mSize;

    const int64_t *outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    inputParams.outputDtype = outputDtypePtr != nullptr ? *outputDtypePtr : 0;
    return true;
}

bool SetMxA8W4NzInput(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    auto wDesc = contex->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input wDesc is nullptr."), return false);
    inputParams.wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));

    auto xStorageShape = contex->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    uint64_t xDimNum = static_cast<uint64_t>(xShape.GetDimNum());
    inputParams.mSize = xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX);
    inputParams.kSize = xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX);

    auto wStorageShape = contex->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();

    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    inputParams.nSize = wShape.GetDim(wDimNum - LAST_SECOND_DIM_INDEX);
    inputParams.groupNum = wShape.GetDim(0);

    auto biasDesc = contex->GetInputDesc(BIAS_INDEX);
    inputParams.hasBias = (biasDesc != nullptr);
    
    auto sharedInputDesc = contex->GetInputDesc(SHARE_INPUT_INDEX);
    if (sharedInputDesc != nullptr) {
        auto sharedInputStorageShape = contex->GetInputShape(SHARE_INPUT_INDEX);
        OP_CHECK_IF(sharedInputStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input sharedInputStorageShape is nullptr."), return false);
        inputParams.sharedInputLen = sharedInputStorageShape->GetOriginShape().GetDim(0);
    } else {
        inputParams.sharedInputLen = 0;
        inputParams.residualScale = 0.0f;
    }
    return true;
}

void GMMFRWeightQuantTiling::RunSetInputFunc()
{
    for(int64_t setFuncIdx = 0; setFuncIdx < SetInputFuncs_.size(); setFuncIdx++) {
        SetInputFuncs_[setFuncIdx](context_, inputParams_);
    }
}

bool GMMFRWeightQuantTiling::SetMxA8W4NzInputFunc() {
    SetInputFuncs_.push_back(SetMxA8W4NzAttrs);
    SetInputFuncs_.push_back(SetMxA8W4NzInput);
    return true;
}
} // namespace optiling