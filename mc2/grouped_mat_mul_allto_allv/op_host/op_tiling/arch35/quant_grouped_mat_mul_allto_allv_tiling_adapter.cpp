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
 * \file quant_grouped_mat_mul_allto_allv_tiling_adapter.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace Mc2GroupedMatmul;
using namespace GmmConstant;
// namespace MC2Tiling {

// bool QuantGroupedMatmulAllToAllvAdapter::AnalyzeAttrs()
// {
//     auto attrs = context_->GetAttrs();
//     if (attrs) {
//         OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_ACT_TYPE + 1,
//                   OP_LOGE(inputParams_.opName,
//                                             "The num of attrs should be greater than %lu, actual is %zu",
//                                             ATTR_INDEX_ACT_TYPE + 1, attrs->GetAttrNum()),
//                   return false);
//         const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
//         const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
//         const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
//         const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
//         const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
//         const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);

//         inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
//         inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
//         inputParams_.groupType = groupTypePtr != nullptr ? *groupTypePtr : inputParams_.groupType;
//         inputParams_.splitItem = splitItemPtr != nullptr ? *splitItemPtr : inputParams_.splitItem;
//         inputParams_.actType = actTypePtr != nullptr ? *actTypePtr : inputParams_.actType;
//         inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
//     }
//     OP_CHECK_IF(
//         inputParams_.groupType != SPLIT_M && inputParams_.groupType != SPLIT_K,
//         OP_LOGE(inputParams_.opName, "Only support group type is 0 or 2 when the dtype of x is %s, actual is %d",
//                 ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(), inputParams_.groupType),
//         return false);
//     OP_CHECK_IF(
//         (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) &&
//             inputParams_.groupType != SPLIT_M,
//         OP_LOGE(inputParams_.opName, "Only support group type to be 0 when the dtype of x is FLOAT4, actual is %d.",
//                 inputParams_.groupType),
//         return false);
//     if (inputParams_.groupType == SPLIT_M) {
//         OP_CHECK_IF(inputParams_.transA,
//                    OP_LOGE(inputParams_.opName, "When group type is 0, transA can only be false."),
//                    return false);
//     } else {
//         OP_CHECK_IF(!inputParams_.transA,
//                    OP_LOGE(inputParams_.opName, "When group type is 2, transA can only be true."),
//                    return false);
//         OP_CHECK_IF(inputParams_.transB,
//                    OP_LOGE(inputParams_.opName, "When group type is 2, transB can only be false."),
//                    return false);
//     }

//     inputParams_.isSingleX = (context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr);
//     inputParams_.isSingleW = (context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr);
//     // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
//     inputParams_.isSingleY = (inputParams_.splitItem == 2 || inputParams_.splitItem == 3);
//     return true;
// }

// bool QuantGroupedMatmulAllToAllvAdapter::AnalyzeDtype()
// {
//     static const std::vector<ge::DataType> legalInputDtypes = {
//         // ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
//         ge::DT_HIFLOAT8};
//     auto xDesc = context_->GetDynamicInputDesc(GMM_X_INDEX, 0);
//     OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
//     inputParams_.aDtype = xDesc->GetDataType();
//     OP_CHECK_IF(
//         std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.aDtype) == legalInputDtypes.end(),
//         OP_LOGE(inputParams_.opName,
//                 "The dtype of x should be in {INT8, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2}, \
// actual is %s.",
//                 ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
//         return false);
//     auto wDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
//     OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
//     inputParams_.bDtype = wDesc->GetDataType();
//     OP_CHECK_IF(
//         std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.bDtype) == legalInputDtypes.end(),
//         OP_LOGE(inputParams_.opName,
//                 "The dtype of weight should be in {INT8, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2, FLOAT4_E2M1, \
// FLOAT4_E1M2}, actual is %s.",
//                 ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
//         return false);
//     inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
//     auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
//     inputParams_.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
//     auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
//     OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
//                OP_LOGE(inputParams_.opName,
//                                          "Bias from tensor is not nullptr, but bias from desc is nullptr."),
//                return false);
//     inputParams_.biasDtype = inputParams_.hasBias ? biasDesc->GetDataType() : inputParams_.biasDtype;
//     auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
//     inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
//     auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
//     inputParams_.perTokenScaleDtype =
//         pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
//     isWeightNz_ = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
//     if (isWeightNz_) {
//         OP_CHECK_IF(!CheckDtypeForWeightNz(nullptr == pertokenScaleDesc),
//                     OP_LOGE(inputParams_.opName, "CheckDtypeForWeightNz failed."), return false);
//     }
//     auto yDesc = context_->GetOutputDesc(Y_INDEX);
//     OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "yDesc is nullptr."), return false);
//     inputParams_.cDtype = yDesc->GetDataType();
//     if (inputParams_.hasBias) {
//         OP_CHECK_IF(!CheckBiasDtype(), OP_LOGE(inputParams_.opName, "CheckBiasDtype failed."), return false);
//     }
//     return true;
// }

// bool QuantGroupedMatmulAllToAllvAdapter::AnalyzeInputs()
// {
//     auto xStorageShape = context_->GetDynamicInputShape(X_INDEX, 0);

//     OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
//     const gert::Shape &xShape = xStorageShape->GetStorageShape();

//     auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
//     OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
//     const gert::Shape &wShape = wStorageShape->GetStorageShape();
//     const gert::Shape &weightNzStorageShape = wStorageShape->GetStorageShape();

//     // 全量化scale必须有值，目前无输出int32等不需要scale的场景
//     auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
//     OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."), return false);
//     const gert::Shape &wScaleShape = scaleStorageShape->GetStorageShape();
//     auto scaleDimNum = wScaleShape.GetDimNum();
//     OP_CHECK_IF(scaleDimNum < 1,
//                OP_LOGE(inputParams_.opName,
//                                          "The dimension of scale should be positive integer, actual is %zu.",
//                                          scaleDimNum),
//                return false);
//     auto xScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
//     OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."),
//                 return false);
//     OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
//     OP_CHECK_IF(!SetMKNList(), OP_LOGE(inputParams_.opName, "SetMKNList failed."), return false);
//     OP_CHECK_IF(!SetQuantMode(wScaleShape, xScaleStorageShape, wShape),
//                OP_LOGE(inputParams_.opName, "SetQuantMode failed."), return false);
//     OP_CHECK_IF(!CheckQuantParams(xScaleStorageShape, wScaleShape),
//                OP_LOGE(inputParams_.opName, "CheckQuantParams failed."), return false);

//     if (isWeightNz_) {
//         OP_CHECK_IF(!CheckShapeForWeightNz(weightNzStorageShape), OP_LOGE(context_->GetNodeName(), "CheckShapeForWeightNz failed."),
//                     return false);
//     }
//     if (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) {
//         OP_CHECK_IF(!CheckFp4Shape(), OP_LOGE(inputParams_.opName, "CheckFp4Shape failed."), return false);
//         if (inputParams_.hasBias) {
//             auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
//             OP_CHECK_IF(!CheckBiasShape(biasStorageShape),
//                        OP_LOGE(inputParams_.opName, "CheckBiasShape failed."), return false);
//         }
//     }
//     SetKernelType();
//     return true;
// }

// ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::GetShapeAttrsInfo()
// {
//     inputParams_.opName = context_->GetNodeName();
//     OP_CHECK_IF(!AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
//                OP_LOGE(inputParams_.opName, "Failed to analyze context_ info."),
//                return ge::GRAPH_FAILED);
//     return ge::GRAPH_SUCCESS;
// }

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetExpertInputParameters(const int32_t* sendCounts,
    uint64_t worldSize, uint64_t index, uint32_t epNums)
{
    // uint32_t worldSize = tilingProcesser_.localTilingData_.taskTilingInfo.epWorldSize;
    // auto sendCounts = &tilingProcesser_.localTilingData_.taskTilingInfo.sendCnt[0];
    uint64_t mSizePerLoop = 0;
    // index sendcounts起始   epNums 当前loop专家数 -- 每轮专家 与 尾轮专家
    if (epNums == 1) {
        for (uint32_t i = 0; i < worldSize; i++) {
            mSizePerLoop += sendCounts[index + i];
        }

        inputParams_.mSize = mSizePerLoop;
        inputParams_.isSingleX = true;
        inputParams_.isSingleW = true;
        inputParams_.isSingleY = true;
        mList_[0] = static_cast<int32_t>(mSizePerLoop);
        kList_[0] = static_cast<int32_t>(inputParams_.kSize);
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
        return ge::GRAPH_SUCCESS; 
    }

    // 每轮多专家
    for (uint32_t i = 0; i < epNums; i++) {
        mSizePerLoop = 0;
        for (uint32_t j = 0; j < worldSize; j++) {
            mSizePerLoop += sendCounts[index + i * worldSize + j];
        }
        mList_[i] = static_cast<int32_t>(mSizePerLoop);
        kList_[i] = static_cast<int32_t>(inputParams_.kSize);
        nList_[i] = static_cast<int32_t>(inputParams_.nSize);
    }

    inputParams_.isSingleX = false;
    inputParams_.isSingleW = false;
    inputParams_.isSingleY = false;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetSharedExpertInputParameters()
{
    auto mmXDesc = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    if (mmXDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto xShape = mmXDesc->GetStorageShape();
    inputParams_.mSize = xShape.GetDim(DIM_ZERO);
    inputParams_.kSize = xShape.GetDim(DIM_ONE);
    
    auto weightShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX)->GetStorageShape();
    inputParams_.nSize = weightShape.GetDim(DIM_ONE);

    inputParams_.aQuantMode = Mc2GroupedMatmul::QuantMode::PERTENSOR_MODE;
    inputParams_.bQuantMode = Mc2GroupedMatmul::QuantMode::PERTENSOR_MODE;
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::GmmConstant::NO_SPLIT;
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
    auto xShape = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    inputParams_.mSize = xShape.GetDim(DIM_ZERO);
    inputParams_.kSize = xShape.GetDim(DIM_ONE);
    
    auto weightShape = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    inputParams_.nSize = weightShape.GetDim(DIM_TWO);
    inputParams_.groupNum = 1;
    // inputParams_.outDtype = 0;
    // inputParams_.kernelType = context_->GetNodeName();
    inputParams_.aQuantMode = Mc2GroupedMatmul::QuantMode::PERTENSOR_MODE;
    inputParams_.bQuantMode = Mc2GroupedMatmul::QuantMode::PERTENSOR_MODE;
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::GmmConstant::NO_SPLIT;
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

void QuantGroupedMatmulAllToAllvAdapter::SetKernelType()
{
    // 以选择主模板设置kernelType, 0: dequant fixp随路（包含K轴分组）；1：dequant vector计算；2：perGroup-perBlock
    inputParams_.kernelType = 0UL;
    // mx K轴分组当前是独立的模板，后续归一
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        return;
    }
    // perGroup-perBlock(GB)有独立pertile模板
    if (inputParams_.bQuantMode == QuantMode::PERBLOCK_MODE) {
        inputParams_.kernelType = 2UL;
        return;
    }
    // pertensor-pertensor且没有后处理的bias，都可以走dequant fixp随路
    bool isPertensorCube = inputParams_.aQuantMode <= QuantMode::PERTENSOR_MODE &&
                           inputParams_.bQuantMode == QuantMode::PERTENSOR_MODE;
    bool isBiasEpilogue =
        inputParams_.aDtype == ge::DT_INT8 && inputParams_.hasBias && inputParams_.biasDtype != ge::DT_INT32;
    // 如果bias bf16/fp16/fp32，需mix模板进行后处理
    if (isPertensorCube && !isBiasEpilogue) {
        return;
    }
    bool isScaleEpilogue = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64);
    // 后处理的bias和（scale非64bits && ！isPertensorCube）需要走dequant vec模板
    if (isBiasEpilogue || isScaleEpilogue) {
        inputParams_.kernelType = 1UL;
    }
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.activeType = inputParams_.actType;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = static_cast<int8_t>(inputParams_.groupType);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    errno_t retM = memcpy_s(tilingData_.gmmArray.mList, sizeof(tilingData_.gmmArray.mList), mList_, sizeof(mList_));
    if (retM != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retM);
        return ge::GRAPH_FAILED;
    }
    errno_t retK = memcpy_s(tilingData_.gmmArray.kList, sizeof(tilingData_.gmmArray.kList), kList_, sizeof(kList_));
    if (retK!= EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retK);
        return ge::GRAPH_FAILED;
    }
    errno_t retN = memcpy_s(tilingData_.gmmArray.nList, sizeof(tilingData_.gmmArray.nList), nList_, sizeof(nList_));
    if (retN != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retN);
        return ge::GRAPH_FAILED;
    }
    // PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantGroupedMatmulAllToAllvAdapter::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

void QuantGroupedMatmulAllToAllvAdapter::CalBasicBlock()
{
    bool isGBQuantMode = inputParams_.aQuantMode == QuantMode::PERGROUP_MODE &&
                         inputParams_.bQuantMode == QuantMode::PERBLOCK_MODE;
    basicTiling_.baseM = std::min(inputParams_.mSize, static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256));
    basicTiling_.baseM = !inputParams_.transA ?
                             Ops::Base::CeilAlign(basicTiling_.baseM, CUBE_BLOCK) :
                             Ops::Base::CeilAlign(basicTiling_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype));
    if (isGBQuantMode) {
        // 不管M/K轴分组，单单单场景下，N不变，可以确定baseN
        if (inputParams_.nSize <= PER_BLOCK_GROUP_SIZE || basicTiling_.baseM > PER_BLOCK_GROUP_SIZE) {
            basicTiling_.baseN = PER_BLOCK_GROUP_SIZE;
        } else {
            basicTiling_.baseN = GmmConstant::BASIC_BLOCK_SIZE_256;
        }
        basicTiling_.baseK = PER_BLOCK_GROUP_SIZE;
        return;
    }
    basicTiling_.baseN = std::min(inputParams_.nSize, static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256));
    basicTiling_.baseN = inputParams_.transB ?
                             Ops::Base::CeilAlign(basicTiling_.baseN, CUBE_BLOCK) :
                             Ops::Base::CeilAlign(basicTiling_.baseN, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype));
    basicTiling_.baseK = Ops::Base::CeilAlign(
        std::min(GetShapeWithDataType(GmmConstant::BASIC_BLOCK_SIZE_128, inputParams_.aDtype), inputParams_.kSize),
        GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype));

    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        basicTiling_.baseK = Ops::Base::CeilAlign(basicTiling_.baseK, MXFP_BASEK_FACTOR); // mx_mmad requires basek align to 64
        bool isFp4Input = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
        if (isFp4Input && !inputParams_.transB) {
            // 64: mx_mmad requires the inner axis to align to 64
            basicTiling_.baseN = Ops::Base::CeilAlign(basicTiling_.baseN, static_cast<uint64_t>(64));
        }
    }
}

bool QuantGroupedMatmulAllToAllvAdapter::IsBiasInL1() const
{
    // 目前仅int8进bias int32需要进L1
    return inputParams_.hasBias && inputParams_.biasDtype == ge::DT_INT32;
}
uint64_t QuantGroupedMatmulAllToAllvAdapter::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    // shapeSize应该是偶数
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        // 2: 判断是否是偶数
        OP_CHECK_IF(shapeSize % 2 != 0,
                   OP_LOGE(
                       context_->GetNodeName(),
                       "To get size of matrix/array, the number of elements must be even when dtype is FLOAT4/INT4"),
                   return 0);
        // 1/2: 这几种数据类型的dsize=1/2
        return shapeSize / 2UL;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t QuantGroupedMatmulAllToAllvAdapter::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE_V120) {
        return depthInit;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        uint64_t baseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.aDtype);
        while ((depthScale * baseKSize) % GmmConstant::BASIC_BLOCK_SIZE_512 != 0 &&
               (depthScale * baseKSize) > GmmConstant::BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % GmmConstant::BASIC_BLOCK_SIZE_512 != 0 &&
            (depthScale * baseKSize) >= GmmConstant::BASIC_BLOCK_SIZE_256) {
            depthScale = GmmConstant::BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, static_cast<uint64_t>(1));
    } else {
        constexpr uint64_t index = 2; // 2: depth的值是2的幂
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}

void QuantGroupedMatmulAllToAllvAdapter::CalStepKs()
{
    // depthA,depthB 为1时，stepka, stepkb 只能是1.
    basicTiling_.stepKa = basicTiling_.depthA1 == 1UL ? 1UL : basicTiling_.depthA1 / DB_SIZE;
    basicTiling_.stepKb = basicTiling_.depthB1 == 1UL ? 1UL : basicTiling_.depthB1 / DB_SIZE;

    if (basicTiling_.stepKa * basicTiling_.baseK > inputParams_.kSize) {
        basicTiling_.stepKa = Ops::Base::CeilDiv(inputParams_.kSize, basicTiling_.baseK);
    }

    if (basicTiling_.stepKb * basicTiling_.baseK >= inputParams_.kSize) {
        basicTiling_.stepKb = Ops::Base::CeilDiv(inputParams_.kSize, basicTiling_.baseK);
    }
    // G-B量化场景下，限制stepK最大为4, 防止issue queue阻塞
    if (inputParams_.aQuantMode == QuantMode::PERGROUP_MODE &&
        inputParams_.bQuantMode == QuantMode::PERBLOCK_MODE) {
        basicTiling_.stepKa = std::min(basicTiling_.stepKa, static_cast<uint64_t>(4)); // 4: G-B最大stepk值
        basicTiling_.stepKb = std::min(basicTiling_.stepKb, static_cast<uint64_t>(4)); // 4: G-B最大stepk值
    }
    if (basicTiling_.stepKa >= basicTiling_.stepKb && basicTiling_.stepKa * basicTiling_.baseK < inputParams_.kSize) {
        basicTiling_.stepKa = basicTiling_.stepKa / basicTiling_.stepKb * basicTiling_.stepKb;
    }
    if (basicTiling_.stepKb > basicTiling_.stepKa && basicTiling_.stepKb * basicTiling_.baseK < inputParams_.kSize) {
        basicTiling_.stepKb = basicTiling_.stepKb / basicTiling_.stepKa * basicTiling_.stepKa;
    }

    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

void QuantGroupedMatmulAllToAllvAdapter::CalScaleFactors()
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    uint64_t baseScaleASize = GetSizeWithDataType(Ops::Base::CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseM,
                                                  inputParams_.perTokenScaleDtype);
    uint64_t baseScaleBSize =
        GetSizeWithDataType(Ops::Base::CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseN, inputParams_.scaleDtype);
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t leftL1Size =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(leftL1Size / (basicTiling_.depthA1 * baseScaleASize +
                                                            basicTiling_.depthB1 * baseScaleBSize));

    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleASize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(inputParams_.kSize / (basicTiling_.stepKa * basicTiling_.baseK));
    uint32_t scaleFactorB = static_cast<uint32_t>(inputParams_.kSize / (basicTiling_.stepKb * basicTiling_.baseK));
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    // 来自L1 size 的约束
    if (basicTiling_.scaleFactorA <= scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1Size -= (basicTiling_.scaleFactorA * basicTiling_.depthA1 * baseScaleASize);
        basicTiling_.scaleFactorB = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthB1 * baseScaleBSize)),
                                             basicTiling_.scaleFactorB);
    } else if (basicTiling_.scaleFactorB <= scaleInit && basicTiling_.scaleFactorA > scaleInit) {
        leftL1Size -= (basicTiling_.scaleFactorB * basicTiling_.depthB1 * baseScaleBSize);
        basicTiling_.scaleFactorA = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthA1 * baseScaleASize)),
                                             basicTiling_.scaleFactorA);
    } else if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1Size -=
            (scaleInit * basicTiling_.depthB1 * baseScaleBSize + scaleInit * basicTiling_.depthA1 * baseScaleASize);
        uint32_t scaleASec = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthA1 * baseScaleASize)),
                                      basicTiling_.scaleFactorA - scaleInit);
        uint32_t scaleBSec = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthB1 * baseScaleBSize)),
                                      basicTiling_.scaleFactorB - scaleInit);
        basicTiling_.scaleFactorA = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        basicTiling_.scaleFactorB = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::CalL1Depth(uint64_t leftL1Size)
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

    uint64_t baseScaleASize = 0;
    uint64_t baseScaleBSize = 0;
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        if (inputParams_.groupType == optiling::Mc2GroupedMatmul::GmmConstant::SPLIT_M) {
            baseScaleASize =
                GetSizeWithDataType(Ops::Base::CeilAlign(Ops::Base::CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE), 2UL) * basicTiling_.baseM,
                                    inputParams_.perTokenScaleDtype);
            baseScaleBSize =
                GetSizeWithDataType(Ops::Base::CeilAlign(Ops::Base::CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE), 2UL) * basicTiling_.baseN,
                                    inputParams_.scaleDtype);
        } else {
            baseScaleASize = GetSizeWithDataType(
                (basicTiling_.baseK / (MX_GROUP_SIZE * MXFP_MULTI_BASE_SIZE) + inputParams_.groupNum) *
                    MXFP_MULTI_BASE_SIZE * basicTiling_.baseM, // 2 is dim value of last scale dim
                inputParams_.perTokenScaleDtype);
            baseScaleBSize = GetSizeWithDataType(
                (basicTiling_.baseK / (MX_GROUP_SIZE * MXFP_MULTI_BASE_SIZE) + inputParams_.groupNum) *
                    MXFP_MULTI_BASE_SIZE * basicTiling_.baseN, // 2 is dim value of last pertokenScale dim
                inputParams_.scaleDtype);
        }
    }
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    OP_CHECK_IF(leftL1Size < baseL1Size,
               OP_LOGE(context_->GetNodeName(),
                                         "L1 space overflow. Free L1Size : %lu, used space: %lu", leftL1Size,
                                         baseL1Size),
               return ge::GRAPH_FAILED);
    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
    uint64_t leftL1SizeByDepthInit = leftL1Size - depthInit * (baseL1Size);
    uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
    uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
    basicTiling_.depthA1 = std::max(depthASec, depthBSec);
    basicTiling_.depthB1 = basicTiling_.depthA1;
    if (basicTiling_.depthA1 * baseL1Size > leftL1Size) {
        basicTiling_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
        basicTiling_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
    }
    CalStepKs();
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        CalScaleFactors();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::CalL1Tiling()
{
    basicTiling_.stepM = 1UL;
    basicTiling_.stepN = 1UL;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, basicTiling_.baseM);
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, basicTiling_.baseN);
    basicTiling_.singleCoreK = inputParams_.kSize;

    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t totalL1Size = aicoreParams_.l1Size;

    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        (basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) ? DB_SIZE : 1;
    uint64_t singleCoreBiasSize = IsBiasInL1() ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t singleCoreScaleSize =
        inputParams_.bQuantMode == QuantMode::PERCHANNEL_MODE && inputParams_.kernelType == 0 ?
            basicTiling_.baseN * scaleDtypeSize :
            0;
    uint64_t usedSize = singleCoreBiasSize + singleCoreScaleSize;
    OP_CHECK_IF(totalL1Size <= usedSize,
               OP_LOGE(context_->GetNodeName(), "L1 space overflow. L1Size: %lu, used space: %lu",
                                         totalL1Size, usedSize),
               return ge::GRAPH_FAILED);
    uint64_t leftL1Size = totalL1Size - usedSize;
    return CalL1Depth(leftL1Size);
}
bool QuantGroupedMatmulAllToAllvAdapter::IsCapable()
{
    return true;
}
ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS,
               OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.M = inputParams_.mSize;
    tilingData_.mmTilingData.N = inputParams_.nSize;
    tilingData_.mmTilingData.Ka = inputParams_.kSize;
    tilingData_.mmTilingData.Kb = inputParams_.kSize;
    tilingData_.mmTilingData.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.mmTilingData.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.mmTilingData.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.mmTilingData.depthA1 = basicTiling_.depthA1;
    tilingData_.mmTilingData.depthB1 = basicTiling_.depthB1;
    tilingData_.mmTilingData.stepM = basicTiling_.stepM;
    tilingData_.mmTilingData.stepN = basicTiling_.stepN;
    tilingData_.mmTilingData.stepKa = basicTiling_.stepKa;
    tilingData_.mmTilingData.stepKb = basicTiling_.stepKb;
    tilingData_.mmTilingData.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.mmTilingData.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.mmTilingData.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.mxTypePara = (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.mmTilingData.mxTypePara = (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::Process()
{
    GE_ASSERT_GRAPH_SUCCESS(DoOpTiling());
    GE_ASSERT_GRAPH_SUCCESS(DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

// }
