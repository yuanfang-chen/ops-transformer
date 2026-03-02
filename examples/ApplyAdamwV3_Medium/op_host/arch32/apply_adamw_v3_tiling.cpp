/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "apply_adamw_v3_tiling.h"

using namespace ge;

namespace optiling {
const size_t SYS_WORKSPACE = 16777216;
constexpr static int32_t OPTIONAL_INPUT_INDEX = 11;
const static std::map<size_t, string> TENSOR_INDEX_LIST = {{1, "m"}, {2, "v"}, {10, "grad"}};
const static std::map<size_t, string> SCALAR_INDEX_LIST = {{3, "beta1_power"},  {4, "beta2_power"}, {5, "lr"},
                                                            {6, "weight_decay"}, {7, "beta1"},       {8, "beta2"},
                                                            {9, "epsilon"}};

constexpr uint32_t FP16_TILING_KEY = 1;
constexpr uint32_t BF16_TILING_KEY = 2;
constexpr uint32_t FP32_TILING_KEY = 3;
constexpr uint32_t AMSGRAD_FP16_TILING_KEY = 11;
constexpr uint32_t AMSGRAD_BF16_TILING_KEY = 12;
constexpr uint32_t AMSGRAD_FP32_TILING_KEY = 13;

static float GetScalarValueFloat(gert::TilingContext* context, size_t inputIdx) {
    auto inputTensor = context->GetInputTensor(inputIdx);
    if (inputTensor == nullptr) {
        return 0.0f;
    }
    auto inputDesc = context->GetInputDesc(inputIdx);
    if (inputDesc == nullptr) {
        return 0.0f;
    }
    ge::DataType dtype = inputDesc->GetDataType();
    const float* dataFloat = inputTensor->GetData<float>();
    const uint16_t* dataHalf = inputTensor->GetData<uint16_t>();
    if (dtype == ge::DT_FLOAT) {
        if (dataFloat == nullptr) return 0.0f;
        return *dataFloat;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        if (dataHalf == nullptr) return 0.0f;
        return static_cast<float>(*dataHalf);
    }
    return 0.0f;
}

ge::graphStatus ApplyAdamwV3Tiling::CheckIsScalar(size_t inputIdx) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    if (storageShape.IsScalar() || storageShape.GetShapeSize() == 1) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus ApplyAdamwV3Tiling::CheckSameShape(size_t inputIdx, const gert::Shape& input0Shape) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto curStorageShape = inputShape->GetStorageShape();
    if (curStorageShape != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamwV3Tiling::CheckSameDtype(size_t inputIdx, const ge::DataType& input0Dtype) {
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto curDtype = inputDesc->GetDataType();
    if (curDtype != input0Dtype) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamwV3Tiling::CheckShapeAndType() {
    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto inputStorageShape = inputShape->GetStorageShape();

    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    ge::DataType input0Dtype = inputDesc->GetDataType();

    for (size_t idx = 1; idx < OPTIONAL_INPUT_INDEX; idx++) {
        if (idx == 3 || idx == 4 || idx == 5 || idx == 6 || idx == 7 || idx == 8 || idx == 9) {
            OP_CHECK_IF(CheckIsScalar(idx) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_, "input %zu is not scalar.", idx),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(CheckSameShape(idx, inputStorageShape) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_, "input %zu shape is not same as input0.", idx),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(CheckSameDtype(idx, input0Dtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_, "input %zu dtype is not same as input0.", idx),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamwV3Tiling::ReadScalarInputs() {
    beta1Power_ = GetScalarValueFloat(tilingContext_, 3);
    beta2Power_ = GetScalarValueFloat(tilingContext_, 4);
    lr_ = GetScalarValueFloat(tilingContext_, 5);
    weightDecay_ = GetScalarValueFloat(tilingContext_, 6);
    beta1_ = GetScalarValueFloat(tilingContext_, 7);
    beta2_ = GetScalarValueFloat(tilingContext_, 8);
    epsilon_ = GetScalarValueFloat(tilingContext_, 9);

    OP_LOGD(tilingContext_, "Scalar inputs: beta1Power=%f, beta2Power=%f, lr=%f, weightDecay=%f, beta1=%f, beta2=%f, epsilon=%f",
            beta1Power_, beta2Power_, lr_, weightDecay_, beta1_, beta2_, epsilon_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamwV3Tiling::CalcTilingParams() {
    auto compileInfo = reinterpret_cast<const ApplyAdamwV3CompileInfo*>(tilingContext_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);

    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    totalLength_ = storageShape.GetShapeSize();

    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    OP_LOGD(tilingContext_, "totalLength=%ld, coreNum=%lu, ubSize=%lu, dtype=%d", 
            totalLength_, compileInfo->coreNum, compileInfo->ubSize, inputDtype);

    // TilingKey 计算逻辑（符合设计文档 3.1 节）
    // TilingKey = dtype_base（1/2/3）+ amsgrad_offset（0/10）
    uint32_t dtypeBase = 0;
    if (inputDtype == ge::DT_FLOAT16) {
        dtypeBase = FP16_TILING_KEY;  // 1
    } else if (inputDtype == ge::DT_BF16) {
        dtypeBase = BF16_TILING_KEY;  // 2
    } else if (inputDtype == ge::DT_FLOAT) {
        dtypeBase = FP32_TILING_KEY;  // 3
    } else {
        OP_LOGE(tilingContext_, "Unsupported dtype: %d", inputDtype);
        return ge::GRAPH_FAILED;
    }
    uint32_t amsgradOffset = amsgradAttr_ ? 10 : 0;
    uint32_t tilingKey = dtypeBase + amsgradOffset;

    // 多核切分策略（符合设计文档 3.6 节）
    uint32_t coreNum = static_cast<uint32_t>(compileInfo->coreNum);
    uint32_t totalLength = static_cast<uint32_t>(totalLength_);
    uint32_t usedCoreNum = std::min(coreNum, totalLength);
    uint32_t avgElementsPerCore = totalLength / usedCoreNum;

    // UB 切分计算
    uint64_t ubSize = compileInfo->ubSize;
    constexpr uint64_t ubSizeAlign = 256;
    constexpr uint64_t elementSize = 4;  // float 中间计算精度
    uint64_t maxTileLength = (ubSize / 2 / elementSize / ubSizeAlign) * ubSizeAlign;
    if (maxTileLength == 0) {
        maxTileLength = ubSizeAlign;
    }

    // 计算每个核的 tile 数量和实际 tile 长度
    uint32_t tileLength = static_cast<uint32_t>(std::min(static_cast<uint64_t>(avgElementsPerCore), maxTileLength));
    if (tileLength == 0) {
        tileLength = 1;
    }
    uint32_t tileNumPerCore = (avgElementsPerCore + tileLength - 1) / tileLength;

    OP_LOGD(tilingContext_, "TilingKey=%u, usedCoreNum=%u, totalLength=%u, tileNumPerCore=%u, tileLength=%u",
            tilingKey, usedCoreNum, totalLength, tileNumPerCore, tileLength);

    // 设置 TilingData（直接使用 float 类型，不再使用 FloatToInt64Bits 转换）
    tilingData_.set_tilingKey(tilingKey);
    tilingData_.set_usedCoreNum(usedCoreNum);
    tilingData_.set_totalLength(totalLength);
    tilingData_.set_tileNumPerCore(tileNumPerCore);
    tilingData_.set_tileLength(tileLength);
    tilingData_.set_alignNum(static_cast<uint32_t>(ubSizeAlign));
    tilingData_.set_beta1Power(beta1Power_);
    tilingData_.set_beta2Power(beta2Power_);
    tilingData_.set_lr(lr_);
    tilingData_.set_weightDecay(weightDecay_);
    tilingData_.set_beta1(beta1_);
    tilingData_.set_beta2(beta2_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_maximizeFactor(maximizeAttr_ ? -1.0f : 1.0f);

    tilingContext_->SetBlockDim(usedCoreNum);
    tilingContext_->SetTilingKey(tilingKey);

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = SYS_WORKSPACE;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamwV3Tiling::SetTilingData() {
    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(), 
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamwV3Tiling::RunTiling() {
    OP_CHECK_IF(CheckShapeAndType() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_, "Check shape and type failed"),
                return ge::GRAPH_FAILED);

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const bool* amsgradAttr = attrs->GetAttrPointer<bool>(0);
    amsgradAttr_ = amsgradAttr != nullptr ? *amsgradAttr : false;

    const bool* maximizeAttr = attrs->GetAttrPointer<bool>(1);
    maximizeAttr_ = maximizeAttr != nullptr ? *maximizeAttr : false;
    
    OP_CHECK_IF(ReadScalarInputs() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_, "Read scalar inputs failed"),
                    return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(CalcTilingParams() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_, "Calc tiling params failed"),
                    return ge::GRAPH_FAILED);
    
    return SetTilingData();
}

static ge::graphStatus TilingForApplyAdamwV3(gert::TilingContext* context) {
    if (context == nullptr) {
        OP_LOGE("ApplyAdamwV3Tiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD("ApplyAdamwV3Tiling", "Enter TilingForApplyAdamwV3");

    ApplyAdamwV3Tiling tiling(context);
    return tiling.RunTiling();
}

static ge::graphStatus TilingPrepareForApplyAdamwV3(gert::TilingParseContext* context) {
    OP_LOGD(context, "TilingPrepareForApplyAdamwV3 enter.");
    
    auto compileInfoPtr = context->GetCompiledInfo<ApplyAdamwV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    
    compileInfoPtr->coreNum = 1;
    compileInfoPtr->ubSize = 1024 * 1024;
    
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        try {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
            if (coreNum > 0) {
                compileInfoPtr->coreNum = coreNum;
            }
            uint64_t ubSizeTemp = 0;
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTemp);
            if (ubSizeTemp > 0) {
                compileInfoPtr->ubSize = ubSizeTemp;
            }
        } catch (...) {
            OP_LOGW(context, "Failed to get platform info, using default values");
        }
    } else {
        OP_LOGW(context, "Platform info is null, using default values");
    }
    
    OP_LOGD(context, "TilingPrepareForApplyAdamwV3 exit. coreNum=%lu, ubSize=%lu", 
            compileInfoPtr->coreNum, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdamwV3).Tiling(TilingForApplyAdamwV3).TilingParse<ApplyAdamwV3CompileInfo>(TilingPrepareForApplyAdamwV3);
}  // namespace optiling
