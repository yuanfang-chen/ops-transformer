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
#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GroupedMatmulFinalizeRoutingArch35WeightQuantTiling;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace optiling {
using namespace GroupedMatmulFinalizeRoutingArch35TilingConstant;
using namespace GmmConstant;

// Additional constants for MX-A8W4 validation
constexpr uint32_t DIM_NUM_BIAS = 2;
constexpr uint32_t DIM_NUM_LOGIT = 1;
constexpr uint32_t DIM_NUM_ROW_INDEX = 1;
constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint32_t MX_SCALE_BLOCK_SIZE = 64;  // For scale/pertoken_scale K dimension
constexpr int64_t MX_INNER_DIM = 2;
constexpr uint32_t DIM_NUM_WEIGHT_NZ = 5;  // FRACTAL_NZ format: [E, K/32, N/16, 16, 32]

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
    auto compileInfoPtr = context_->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>();
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
    tilingData_.outputBs = inputParams_.outputBS;
    tilingData_.sharedInputOffset = inputParams_.shareInputOffset;
    tilingData_.sharedInputLen = inputParams_.sharedInputLen;
    tilingData_.sharedInputWeight = inputParams_.sharedInputWeight;
    tilingData_.groupSize = MX_BLOCK_SIZE;

    tilingData_.kSize = inputParams_.kSize;
    tilingData_.nSize = inputParams_.nSize;

    // Calculate initSize: if shared_input exists, initSize = shared_input_offset * n + (outputBs - shared_input.m) * n
    // if shared_input does not exist, initSize = outputBs * n
    if (inputParams_.sharedInputLen == 0) {
        tilingData_.initSize = inputParams_.outputBS * inputParams_.nSize;
    } else {
        tilingData_.initSize = inputParams_.shareInputOffset * inputParams_.nSize + 
                               (inputParams_.outputBS - inputParams_.sharedInputLen) * inputParams_.nSize;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantTiling::DoLibApiTiling()
{
    // 当前为低阶api实现，无对应tiling
    return ge::GRAPH_SUCCESS;
}

uint64_t GMMFRWeightQuantTiling::GetTilingKey() const
{
    std::cout<<"FFFFFFFFFFFFFFFFFFFFF "<<std::endl;
    return GET_TPL_TILING_KEY(0, 1);
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
    context_->SetBlockDim(tilingData_.coreNum);
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
    // GetPrimaryFormat returns the format value masked with 0xFF
    // FORMAT_FRACTAL_NZ = 29, FORMAT_FRACTAL_NZ_C0_32 = 51

    auto scaleDesc = context_->GetOptionalInputDesc(SCALE_INDEX);
    auto scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : ge::DT_INT8;
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    auto perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : ge::DT_INT8;
    auto xDtype = xDesc->GetDataType();
    auto wDtype = wDesc->GetDataType();
    OP_LOGD(context_->GetNodeName(), "Current xDtype: %s, wDtype: %s, scaleDtype: %s, perTokenScaleDtype: %s, wFormat: %s",
        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(wDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(perTokenScaleDtype).c_str(),
        (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) ? "FRACTAL_NZ" : "ND");
    if (xDtype == ge::DT_FLOAT8_E4M3FN && wDtype == ge::DT_FLOAT4_E2M1 && 
        (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) && 
        scaleDtype == ge::DT_FLOAT8_E8M0 && perTokenScaleDtype == ge::DT_FLOAT8_E8M0) {
        scenarioType_ = ScenarioType::MX_A8W4_WEIGHT_NZ;
        OP_LOGD(context_->GetNodeName(), "Enable MX-A8W4-WEIGHT-NZ mode.");
        SetMxA8W4NzConditionFunc();
        SetMxA8W4NzInputFunc();
        return true;
    }

    OP_LOGE(context_->GetNodeName(), "Only support MX-A8W4-WEIGHT-NZ mode. current xDtype: %s, wDtype: %s, scaleDtype");
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

    auto scaleDesc = contex->GetOptionalInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input scaleDesc is nullptr."), return false);
    auto scaleStorageShape = contex->GetOptionalInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input scaleStorageShape is nullptr."),
                return false);

    // pertoken_scale is optional - use GetOptionalInputDesc
    auto pertokenScaleDesc = contex->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    OP_CHECK_IF(pertokenScaleDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input pertokenScaleDesc is nullptr."), return false);
    auto pertokenScaleStorageShape = contex->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);
    OP_CHECK_IF(pertokenScaleStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input pertokenScaleStorageShape is nullptr."),
                return false);

    // Add: group_list must not be nullptr (optional input - use GetOptional API)
    auto groupListDesc = contex->GetOptionalInputDesc(GROUPLIST_INDEX);
    OP_CHECK_IF(groupListDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input groupListDesc is nullptr."),
                return false);
    auto groupListStorageShape = contex->GetOptionalInputShape(GROUPLIST_INDEX);
    OP_CHECK_IF(groupListStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input groupListStorageShape is nullptr."),
                return false);

    // Add: row_index must not be nullptr (required for 91095 - use GetInput API)
    auto rowIndexDesc = contex->GetInputDesc(ROW_INDEX_INDEX);
    OP_CHECK_IF(rowIndexDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input rowIndexDesc is nullptr."),
                return false);
    auto rowIndexStorageShape = contex->GetInputShape(ROW_INDEX_INDEX);
    OP_CHECK_IF(rowIndexStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input rowIndexStorageShape is nullptr."),
                return false);

    return true;
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

    // offset must be nullptr in MxA8W4 weight Nz scenario
    auto offsetDesc = contex->GetOptionalInputDesc(OFFSET_INDEX);
    OP_CHECK_IF(offsetDesc != nullptr,
                OP_LOGE(contex->GetNodeName(), "offset must be nullptr in MxA8W4 weight Nz scenario."),
                return false);

    return true;
}

// Helper: Validate x shape and extract M, K
static bool ValidateXShape(gert::TilingContext *contex, int64_t &mSize, int64_t &kSize) {
    auto xStorageShape = contex->GetInputShape(X_INDEX);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto xDimNum = xShape.GetDimNum();
    
    OP_CHECK_IF(xDimNum != DIM_NUM_X,
                OP_LOGE(contex->GetNodeName(), "The dimension of x must be %u, actual is %zu", DIM_NUM_X, xDimNum),
                return false);
    mSize = xShape.GetDim(0);
    kSize = xShape.GetDim(1);
    return true;
}

// Helper: Validate w shape and extract E, N, K
static bool ValidateWShape(gert::TilingContext *contex, ge::Format wFormat, 
                            int64_t &eFromW, int64_t &nSize, int64_t &kFromW) {
    auto wShape = contex->GetInputShape(W_INDEX);
    const gert::Shape &wStorageShape = wShape->GetStorageShape();
    auto wStorageDimNum = wStorageShape.GetDimNum();
    
    if (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        OP_CHECK_IF(wStorageDimNum != DIM_NUM_WEIGHT_NZ,
                    OP_LOGE(contex->GetNodeName(), "The dimension of w (FRACTAL_NZ) must be %u, actual is %zu", 
                            DIM_NUM_WEIGHT_NZ, wStorageDimNum),
                    return false);
        const gert::Shape &wOriginShape = wShape->GetOriginShape();
        nSize = wOriginShape.GetDim(1);
        kFromW = wOriginShape.GetDim(2);
    } else {
        OP_CHECK_IF(wStorageDimNum != DIM_NUM_WEIGHT,
                    OP_LOGE(contex->GetNodeName(), "The dimension of w must be %u, actual is %zu", DIM_NUM_WEIGHT, wStorageDimNum),
                    return false);
        nSize = wStorageShape.GetDim(1);
        kFromW = wStorageShape.GetDim(2);
    }
    eFromW = wStorageShape.GetDim(0);
    return true;
}

// Helper: Validate group_list shape matches E
static bool ValidateGroupListShape(gert::TilingContext *contex, int64_t eFromW) {
    auto groupListStorageShape = contex->GetOptionalInputShape(GROUPLIST_INDEX);
    const gert::Shape &groupListShape = groupListStorageShape->GetOriginShape();
    OP_CHECK_IF(eFromW != groupListShape.GetDim(0),
                OP_LOGE(contex->GetNodeName(), "E mismatch: w has E=%ld, group_list has %ld",
                        eFromW, groupListShape.GetDim(0)),
                return false);
    return true;
}

// Helper: Validate row_index shape is [M]
static bool ValidateRowIndexShape(gert::TilingContext *contex, int64_t mSize) {
    auto rowIndexStorageShape = contex->GetInputShape(ROW_INDEX_INDEX);
    const gert::Shape &rowIndexShape = rowIndexStorageShape->GetOriginShape();
    OP_CHECK_IF(rowIndexShape.GetDimNum() != DIM_NUM_ROW_INDEX,
                OP_LOGE(contex->GetNodeName(), "The dimension of row_index must be %u, actual is %zu",
                        DIM_NUM_ROW_INDEX, rowIndexShape.GetDimNum()),
                return false);
    OP_CHECK_IF(rowIndexShape.GetDim(0) != mSize,
                OP_LOGE(contex->GetNodeName(), "row_index shape[0]=%ld must equal M=%ld",
                        rowIndexShape.GetDim(0), mSize),
                return false);
    return true;
}

// Helper: Validate logit shape is [M] if present
static bool ValidateLogitShape(gert::TilingContext *contex, int64_t mSize) {
    auto logitDesc = contex->GetOptionalInputDesc(LOGIT_INDEX);
    if (logitDesc == nullptr) return true;
    
    auto logitStorageShape = contex->GetOptionalInputShape(LOGIT_INDEX);
    const gert::Shape &logitShape = logitStorageShape->GetOriginShape();
    OP_CHECK_IF(logitShape.GetDimNum() != DIM_NUM_LOGIT,
                OP_LOGE(contex->GetNodeName(), "The dimension of logit must be %u, actual is %zu",
                        DIM_NUM_LOGIT, logitShape.GetDimNum()),
                return false);
    OP_CHECK_IF(logitShape.GetDim(0) != mSize,
                OP_LOGE(contex->GetNodeName(), "logit shape[0]=%ld must equal M=%ld",
                        logitShape.GetDim(0), mSize),
                return false);
    return true;
}

// Helper: Validate bias shape is [E, N] if present and non-empty
static bool ValidateBiasShape(gert::TilingContext *contex, int64_t eFromW, int64_t nSize) {
    auto biasDesc = contex->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc == nullptr) return true;
    
    auto biasStorageShape = contex->GetOptionalInputShape(BIAS_INDEX);
    if (biasStorageShape == nullptr) return true;
    
    const gert::Shape &biasShape = biasStorageShape->GetOriginShape();
    if (biasShape.GetDimNum() == 0) return true;
    
    OP_CHECK_IF(biasShape.GetDimNum() != DIM_NUM_BIAS,
                OP_LOGE(contex->GetNodeName(), "The dimension of bias must be %u, actual is %zu",
                        DIM_NUM_BIAS, biasShape.GetDimNum()),
                return false);
    OP_CHECK_IF(biasShape.GetDim(0) != eFromW,
                OP_LOGE(contex->GetNodeName(), "bias shape[0]=%ld must equal E=%ld",
                        biasShape.GetDim(0), eFromW),
                return false);
    OP_CHECK_IF(biasShape.GetDim(1) != nSize,
                OP_LOGE(contex->GetNodeName(), "bias shape[1]=%ld must equal N=%ld",
                        biasShape.GetDim(1), nSize),
                return false);
    return true;
}

// Helper: Validate scale shape [E, N, CeilDiv(K, 64), 2]
static bool ValidateScaleShape(gert::TilingContext *contex, int64_t eFromW, int64_t nSize, int64_t kCeilDiv64) {
    auto scaleStorageShape = contex->GetOptionalInputShape(SCALE_INDEX);
    const gert::Shape &scaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = scaleShape.GetDimNum();
    
    OP_CHECK_IF(scaleDimNum != DIM_NUM_MX_SCALE,
                OP_LOGE(contex->GetNodeName(), "The dimension of scale must be %u, actual is %zu",
                        DIM_NUM_MX_SCALE, scaleDimNum),
                return false);
    OP_CHECK_IF(scaleShape.GetDim(0) != eFromW,
                OP_LOGE(contex->GetNodeName(), "scale shape[0]=%ld must equal E=%ld",
                        scaleShape.GetDim(0), eFromW),
                return false);
    OP_CHECK_IF(scaleShape.GetDim(1) != nSize,
                OP_LOGE(contex->GetNodeName(), "scale shape[1]=%ld must equal N=%ld",
                        scaleShape.GetDim(1), nSize),
                return false);
    OP_CHECK_IF(scaleShape.GetDim(2) != kCeilDiv64,
                OP_LOGE(contex->GetNodeName(), "scale shape[2]=%ld must equal CeilDiv(K, 64)=%ld",
                        scaleShape.GetDim(2), kCeilDiv64),
                return false);
    OP_CHECK_IF(scaleShape.GetDim(3) != MX_INNER_DIM,
                OP_LOGE(contex->GetNodeName(), "scale shape[3]=%ld must equal %d",
                        scaleShape.GetDim(3), MX_INNER_DIM),
                return false);
    return true;
}

// Helper: Validate pertoken_scale shape [M, CeilDiv(K, 64), 2]
static bool ValidatePertokenScaleShape(gert::TilingContext *contex, int64_t mSize, int64_t kCeilDiv64) {
    auto pertokenScaleStorageShape = contex->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);
    const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
    auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
    
    OP_CHECK_IF(pertokenScaleDimNum != DIM_NUM_MX_PERTOKENSCALE,
                OP_LOGE(contex->GetNodeName(), "The dimension of pertokenScale must be %u, actual is %zu",
                        DIM_NUM_MX_PERTOKENSCALE, pertokenScaleDimNum),
                return false);
    OP_CHECK_IF(pertokenScaleShape.GetDim(0) != mSize,
                OP_LOGE(contex->GetNodeName(), "pertokenScale shape[0]=%ld must equal M=%ld",
                        pertokenScaleShape.GetDim(0), mSize),
                return false);
    OP_CHECK_IF(pertokenScaleShape.GetDim(1) != kCeilDiv64,
                OP_LOGE(contex->GetNodeName(), "pertokenScale shape[1]=%ld must equal CeilDiv(K, 64)=%ld",
                        pertokenScaleShape.GetDim(1), kCeilDiv64),
                return false);
    OP_CHECK_IF(pertokenScaleShape.GetDim(2) != MX_INNER_DIM,
                OP_LOGE(contex->GetNodeName(), "pertokenScale shape[2]=%ld must equal %d",
                        pertokenScaleShape.GetDim(2), MX_INNER_DIM),
                return false);
    return true;
}

bool CheckMxA8W4InputShape(gert::TilingContext *contex) {
    // Get w format first
    auto wDesc = contex->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input wDesc is nullptr."), return false);
    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));

    // 1. Validate x shape and get M, K
    int64_t mSize, kSize;
    OP_CHECK_IF(!ValidateXShape(contex, mSize, kSize), 
                OP_LOGE(contex->GetNodeName(), "ValidateXShape failed."), return false);

    // 2. Validate w shape and get E, N, K
    int64_t eFromW, nSize, kFromW;
    OP_CHECK_IF(!ValidateWShape(contex, wFormat, eFromW, nSize, kFromW), 
                OP_LOGE(contex->GetNodeName(), "ValidateWShape failed."), return false);

    // Validate K consistency between x and w
    OP_CHECK_IF(kSize != kFromW,
                OP_LOGE(contex->GetNodeName(), "K dimension mismatch: x has K=%ld, w has K=%ld", kSize, kFromW),
                return false);

    // 3. Validate group_list shape
    OP_CHECK_IF(!ValidateGroupListShape(contex, eFromW), 
                OP_LOGE(contex->GetNodeName(), "ValidateGroupListShape failed."), return false);

    // 4. Validate row_index shape
    OP_CHECK_IF(!ValidateRowIndexShape(contex, mSize), 
                OP_LOGE(contex->GetNodeName(), "ValidateRowIndexShape failed."), return false);

    // 5. Validate logit shape
    OP_CHECK_IF(!ValidateLogitShape(contex, mSize), 
                OP_LOGE(contex->GetNodeName(), "ValidateLogitShape failed."), return false);

    // 6. Validate bias shape
    OP_CHECK_IF(!ValidateBiasShape(contex, eFromW, nSize), 
                OP_LOGE(contex->GetNodeName(), "ValidateBiasShape failed."), return false);

    // Calculate CeilDiv(K, 64) for scale validation
    int64_t kCeilDiv64 = (kSize + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE;

    // 7. Validate scale shape
    OP_CHECK_IF(!ValidateScaleShape(contex, eFromW, nSize, kCeilDiv64), 
                OP_LOGE(contex->GetNodeName(), "ValidateScaleShape failed."), return false);

    // 8. Validate pertoken_scale shape
    OP_CHECK_IF(!ValidatePertokenScaleShape(contex, mSize, kCeilDiv64), 
                OP_LOGE(contex->GetNodeName(), "ValidatePertokenScaleShape failed."), return false);

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

    auto sharedInputDesc = contex->GetOptionalInputDesc(SHARE_INPUT_INDEX);
    if (sharedInputDesc != nullptr) {
        auto sharedInputStorageShape = contex->GetOptionalInputShape(SHARE_INPUT_INDEX);
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
    if (outputBSPtr != nullptr) {
        inputParams.outputBS = *outputBSPtr;
    } else {
        // Use M / E as default value when outputBs is not provided
        OP_CHECK_IF(inputParams.groupNum == 0,
                    OP_LOGE(contex->GetNodeName(), "groupNum(E) is 0, cannot compute default outputBs as M/E."),
                    return false);
        inputParams.outputBS = inputParams.mSize / inputParams.groupNum;
        OP_LOGI(contex->GetNodeName(), "outputBs not provided, using default M/E = %ld/%ld = %ld",
                inputParams.mSize, inputParams.groupNum, inputParams.outputBS);
    }

    const int64_t *outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    inputParams.outputDtype = outputDtypePtr != nullptr ? *outputDtypePtr : 0;
    return true;
}

// Helper: Set w format in input params
static bool SetWFormat(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    auto wDesc = contex->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(contex->GetNodeName(), "Input wDesc is nullptr."), return false);
    inputParams.wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    return true;
}

// Helper: Extract M and K from x shape
static bool ExtractXDimensions(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    auto xStorageShape = contex->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    uint64_t xDimNum = static_cast<uint64_t>(xShape.GetDimNum());
    inputParams.mSize = xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX);
    inputParams.kSize = xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX);
    return true;
}

// Helper: Extract E, N from w shape
static bool ExtractWDimensions(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    auto wStorageShape = contex->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();

    if (inputParams.wFormat == ge::FORMAT_FRACTAL_NZ || inputParams.wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        const gert::Shape &wStorage = wStorageShape->GetStorageShape();
        inputParams.nSize = wStorage.GetDim(1);
    } else {
        uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
        inputParams.nSize = wShape.GetDim(wDimNum - LAST_SECOND_DIM_INDEX);
    }
    inputParams.groupNum = wShape.GetDim(0);
    return true;
}

// Helper: Set bias info
static void SetBiasInfo(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    auto biasDesc = contex->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc == nullptr) {
        inputParams.hasBias = false;
        return;
    }
    auto biasStorageShape = contex->GetOptionalInputShape(BIAS_INDEX);
    if (biasStorageShape == nullptr) {
        inputParams.hasBias = false;
        return;
    }
    const gert::Shape &biasShape = biasStorageShape->GetOriginShape();
    inputParams.hasBias = (biasShape.GetDimNum() > 0);
}

// Helper: Set shared input info
static bool SetSharedInputInfo(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    auto sharedInputDesc = contex->GetOptionalInputDesc(SHARE_INPUT_INDEX);
    if (sharedInputDesc == nullptr) {
        inputParams.sharedInputLen = 0;
        inputParams.residualScale = 0.0f;
        return true;
    }
    auto sharedInputStorageShape = contex->GetOptionalInputShape(SHARE_INPUT_INDEX);
    OP_CHECK_IF(sharedInputStorageShape == nullptr, OP_LOGE(contex->GetNodeName(), "Input sharedInputStorageShape is nullptr."), return false);
    inputParams.sharedInputLen = sharedInputStorageShape->GetOriginShape().GetDim(0);
    return true;
}

bool SetMxA8W4NzInput(gert::TilingContext *contex, GMMFRWeightQuantInputParams& inputParams) {
    OP_CHECK_IF(!SetWFormat(contex, inputParams), 
                OP_LOGE(contex->GetNodeName(), "SetWFormat failed."), return false);
    OP_CHECK_IF(!ExtractXDimensions(contex, inputParams), 
                OP_LOGE(contex->GetNodeName(), "ExtractXDimensions failed."), return false);
    OP_CHECK_IF(!ExtractWDimensions(contex, inputParams), 
                OP_LOGE(contex->GetNodeName(), "ExtractWDimensions failed."), return false);
    SetBiasInfo(contex, inputParams);
    OP_CHECK_IF(!SetSharedInputInfo(contex, inputParams), 
                OP_LOGE(contex->GetNodeName(), "SetSharedInputInfo failed."), return false);
    return true;
}

void GMMFRWeightQuantTiling::RunSetInputFunc()
{
    for(int64_t setFuncIdx = 0; setFuncIdx < SetInputFuncs_.size(); setFuncIdx++) {
        SetInputFuncs_[setFuncIdx](context_, inputParams_);
    }
}

bool GMMFRWeightQuantTiling::SetMxA8W4NzInputFunc() {
    SetInputFuncs_.push_back(SetMxA8W4NzInput);   // First: extract M and E from shapes
    SetInputFuncs_.push_back(SetMxA8W4NzAttrs);   // Second: use M/E for default outputBs
    return true;
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulFinalizeRouting, GMMFRWeightQuantTiling, GMMFR_WEIGHT_QUANT_TILING_VEC_ANTIQUANT);
} // namespace optiling