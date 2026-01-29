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
 * \file quant_grouped_mat_mul_allto_allv_adapter.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_adapter.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

// struct GQmmInputInfo {
//     uint64_t mSize = 0UL;
//     uint64_t kSize = 0UL;
//     uint64_t nSize = 0UL;
//     uint64_t groupNum = 0UL;
//     int64_t outDtype = 0L;
//     uint64_t kernelType = 0UL;
//     QuantMode aQuantMode = QuantMode::DEFAULT;
//     QuantMode bQuantMode = QuantMode::DEFAULT;
//     int8_t groupType = GroupedMatmul::NO_SPLIT;
//     int8_t groupListType = 0;
//     int8_t splitItem = 0;
//     int8_t actType = 0;
//     const char *opName = nullptr;
//     ge::DataType aDtype = ge::DT_INT8;
//     ge::DataType bDtype = ge::DT_INT8;
//     ge::DataType cDtype = ge::DT_FLOAT16;
//     ge::DataType biasDtype = ge::DT_INT32;
//     ge::DataType scaleDtype = ge::DT_UINT64;
//     ge::DataType perTokenScaleDtype = ge::DT_FLOAT;
//     ge::DataType outDataDtype = ge::DT_FLOAT16;
//     ge::DataType outScaleDtype = ge::DT_FLOAT;

//     ge::Format aFormat = ge::FORMAT_ND;
//     ge::Format bFormat = ge::FORMAT_ND;
//     ge::Format cFormat = ge::FORMAT_ND;
//     bool transA = false;
//     bool transB = false;
//     bool hasBias = false;
//     bool isSingleX = false;
//     bool isSingleW = false;
//     bool isSingleY = false;
// } inputParams_

using namespace optiling;
namespace MC2Tiling {

void QuantGroupedMatmulAllToAllvAdapter::Reset()
{
    tilingData_ = &tilingProcesser_->localTilingData_;
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                         context_->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
    return;
}

bool QuantGroupedMatmulAllToAllvAdapter::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs) {
        OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_ACT_TYPE + 1,
                  OP_LOGE(inputParams_.opName,
                                            "The num of attrs should be greater than %lu, actual is %zu",
                                            ATTR_INDEX_ACT_TYPE + 1, attrs->GetAttrNum()),
                  return false);
        const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
        const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
        const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
        const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);

        inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
        inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
        inputParams_.groupType = groupTypePtr != nullptr ? *groupTypePtr : inputParams_.groupType;
        inputParams_.splitItem = splitItemPtr != nullptr ? *splitItemPtr : inputParams_.splitItem;
        inputParams_.actType = actTypePtr != nullptr ? *actTypePtr : inputParams_.actType;
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    }
    OP_CHECK_IF(
        inputParams_.groupType != SPLIT_M && inputParams_.groupType != SPLIT_K,
        OP_LOGE(inputParams_.opName, "Only support group type is 0 or 2 when the dtype of x is %s, actual is %d",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(), inputParams_.groupType),
        return false);
    OP_CHECK_IF(
        (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) &&
            inputParams_.groupType != SPLIT_M,
        OP_LOGE(inputParams_.opName, "Only support group type to be 0 when the dtype of x is FLOAT4, actual is %d.",
                inputParams_.groupType),
        return false);
    if (inputParams_.groupType == SPLIT_M) {
        OP_CHECK_IF(inputParams_.transA,
                   OP_LOGE(inputParams_.opName, "When group type is 0, transA can only be false."),
                   return false);
    } else {
        OP_CHECK_IF(!inputParams_.transA,
                   OP_LOGE(inputParams_.opName, "When group type is 2, transA can only be true."),
                   return false);
        OP_CHECK_IF(inputParams_.transB,
                   OP_LOGE(inputParams_.opName, "When group type is 2, transB can only be false."),
                   return false);
    }

    inputParams_.isSingleX = (context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr);
    inputParams_.isSingleW = (context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr);
    // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
    inputParams_.isSingleY = (inputParams_.splitItem == 2 || inputParams_.splitItem == 3);
    return true;
}

bool QuantGroupedMatmulAllToAllvAdapter::AnalyzeDtype()
{
    static const std::vector<ge::DataType> legalInputDtypes = {
        // ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
        ge::DT_HIFLOAT8};
    auto xDesc = context_->GetDynamicInputDesc(GMM_X_INDEX, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.aDtype) == legalInputDtypes.end(),
        OP_LOGE(inputParams_.opName,
                "The dtype of x should be in {INT8, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2}, \
actual is %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
        return false);
    auto wDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    OP_CHECK_IF(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.bDtype) == legalInputDtypes.end(),
        OP_LOGE(inputParams_.opName,
                "The dtype of weight should be in {INT8, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2, FLOAT4_E2M1, \
FLOAT4_E1M2}, actual is %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
               OP_LOGE(inputParams_.opName,
                                         "Bias from tensor is not nullptr, but bias from desc is nullptr."),
               return false);
    inputParams_.biasDtype = inputParams_.hasBias ? biasDesc->GetDataType() : inputParams_.biasDtype;
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    isWeightNz_ = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
    if (isWeightNz_) {
        OP_CHECK_IF(!CheckDtypeForWeightNz(nullptr == pertokenScaleDesc),
                    OP_LOGE(inputParams_.opName, "CheckDtypeForWeightNz failed."), return false);
    }
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "yDesc is nullptr."), return false);
    inputParams_.cDtype = yDesc->GetDataType();
    if (inputParams_.hasBias) {
        OP_CHECK_IF(!CheckBiasDtype(), OP_LOGE(inputParams_.opName, "CheckBiasDtype failed."), return false);
    }
    return true;
}

bool QuantGroupedMatmulAllToAllvAdapter::AnalyzeInputs()
{
    auto xStorageShape = context_->GetDynamicInputShape(X_INDEX, 0);

    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    const gert::Shape &weightNzStorageShape = wStorageShape->GetStorageShape();

    // 全量化scale必须有值，目前无输出int32等不需要scale的场景
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."), return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(scaleDimNum < 1,
               OP_LOGE(inputParams_.opName,
                                         "The dimension of scale should be positive integer, actual is %zu.",
                                         scaleDimNum),
               return false);
    auto xScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."),
                return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!SetMKNList(), OP_LOGE(inputParams_.opName, "SetMKNList failed."), return false);
    OP_CHECK_IF(!SetQuantMode(wScaleShape, xScaleStorageShape, wShape),
               OP_LOGE(inputParams_.opName, "SetQuantMode failed."), return false);
    OP_CHECK_IF(!CheckQuantParams(xScaleStorageShape, wScaleShape),
               OP_LOGE(inputParams_.opName, "CheckQuantParams failed."), return false);

    if (isWeightNz_) {
        OP_CHECK_IF(!CheckShapeForWeightNz(weightNzStorageShape), OP_LOGE(context_->GetNodeName(), "CheckShapeForWeightNz failed."),
                    return false);
    }
    if (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF(!CheckFp4Shape(), OP_LOGE(inputParams_.opName, "CheckFp4Shape failed."), return false);
        if (inputParams_.hasBias) {
            auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
            OP_CHECK_IF(!CheckBiasShape(biasStorageShape),
                       OP_LOGE(inputParams_.opName, "CheckBiasShape failed."), return false);
        }
    }
    SetKernelType();
    return true;
}

// ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::GetShapeAttrsInfo()
// {
//     inputParams_.opName = context_->GetNodeName();
//     OP_CHECK_IF(!AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
//                OP_LOGE(inputParams_.opName, "Failed to analyze context_ info."),
//                return ge::GRAPH_FAILED);
//     return ge::GRAPH_SUCCESS;
// }

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetExpertInputParameters(uint32_t index)
{
    auto xShape = context_->GetInputDesc(MM_X_OPTIONAL_INDEX)->GetOriginShape();
    auto mSize = xShape.GetDim(DIM_0);
    sendCounts = 
    inputParams_.mSize = xShape.GetDim(DIM_0);


    // 是否做切分
    inputParams_.groupType = GroupedMatmul::NO_SPLIT;
    inputParams_.groupListType = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetSharedExpertInputParameters()
{
    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX);
    if (mmXDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto xShape = mmXDesc->GetOriginShape();
    inputParams_.mSize = xShape.GetDim(DIM_0);
    inputParams_.kSize = xShape.GetDim(DIM_1);
    
    auto weightShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    inputParams_.nSize = weightShape.GetDim(DIM_1);

    inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    inputParams_.bQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    // 是否做切分
    inputParams_.groupType = GroupedMatmul::NO_SPLIT;
    inputParams_.groupListType = 0;
    // 输出是否切分，0/1代表输出多tensor， 2/3代表输出单tensor
    inputParams_.splitItem = 2;
    inputParams_.actType = 0;
    inputParams_.aDtype = ge::DT_HIFLOAT8;
    inputParams_.bDtype = ge::DT_HIFLOAT8;
    // c outputDtype
    inputParams_.cDtype = ge::DT_FLOAT16;
    inputParams_.biasDtype = ge::DT_FLOAT;
    inputParams_.scaleDtype = ge::DT_FLOAT;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT;
    // inputParams_.outDataDtype = ge::DT_FLOAT16;
    // inputParams_.outScaleDtype = ge::DT_FLOAT;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.transB = false;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    SetKernelType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetCommonInputParams()
{
    inputParams_.opName = context_->GetNodeName();
    auto xShape = context_->GetDynamicInputShape(GMM_X_INDEX, 0)->GetOriginShape();
    inputParams_.mSize = xShape.GetDim(DIM_0);
    inputParams_.kSize = xShape.GetDim(DIM_1);
    
    auto weightShape = context_->GetDynamicInputShape(GMM_WEIGHT_INDEX, 0)->GetOriginShape();
    inputParams_.nSize = weightShape.GetDim(DIM_2);
    inputParams_.groupNum = 1;
    // inputParams_.outDtype = 0;
    // inputParams_.kernelType = context_->GetNodeName();
    inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    inputParams_.bQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    // 是否做切分
    inputParams_.groupType = GroupedMatmul::NO_SPLIT;
    inputParams_.groupListType = 0;
    // 输出是否切分，0/1代表输出多tensor， 2/3代表输出单tensor
    inputParams_.splitItem = 0;
    inputParams_.actType = 0;
    inputParams_.aDtype = ge::DT_HIFLOAT8;
    inputParams_.bDtype = ge::DT_HIFLOAT8;
    // c outputDtype
    inputParams_.cDtype = ge::DT_FLOAT16;
    inputParams_.biasDtype = ge::DT_FLOAT;
    inputParams_.scaleDtype = ge::DT_FLOAT;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT;
    // inputParams_.outDataDtype = ge::DT_FLOAT16;
    // inputParams_.outScaleDtype = ge::DT_FLOAT;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.transB = false;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    SetKernelType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetCommonContextParameters()
{
    GE_ASSERT_GRAPH_SUCCESS(SetCommonInputParams());
    GE_ASSERT_GRAPH_SUCCESS(GetPlatformInfo());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::Process()
{
    GE_ASSERT_GRAPH_SUCCESS(DoOpTiling());
    GE_ASSERT_GRAPH_SUCCESS(DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

}