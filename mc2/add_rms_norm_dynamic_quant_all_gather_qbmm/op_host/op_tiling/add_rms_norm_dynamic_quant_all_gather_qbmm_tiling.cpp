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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_tiling.cpp
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
// #include "../../op_kernel/add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_key.h"
#include "../../op_kernel/add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_helper.h"
#include "add_rms_norm_dynamic_quant_v2_tiling.h"
#include "mc2_log.h"
#include "tiling/mc2_tiling_utils.h"

namespace MC2Tiling {

using namespace AscendC;
using namespace ge;

constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;

constexpr size_t X1_INDEX = 0;
constexpr size_t X2_INDEX = 1;
constexpr size_t RESIDUAL_INDEX = 2;
constexpr size_t Y_INDEX = 3;
constexpr size_t GAMMA_INDEX = 4;
constexpr size_t SCALE_INDEX = 5;
constexpr size_t SMOOTH_SCALE_INDEX = 6;
constexpr size_t BIAS_INDEX = 7;

constexpr size_t OUTPUT_INDEX = 0;
constexpr size_t Z_INDEX = 1;

constexpr size_t GROUP_INDEX = 0;
constexpr size_t RANK_SIZE_INDEX = 1;
constexpr size_t TRANSPOSE_X2_INDEX = 2;
constexpr size_t DTYPE_INDEX = 3;
constexpr size_t RESIDUAL_NORM_MODE_INDEX = 4;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t NUM_THREE = 3;
constexpr size_t TWO_DIMS = 2;
constexpr size_t FOUR_DIMS = 4;

constexpr uint64_t BASE_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

// matmul tiling 切分
constexpr int32_t SINGLE_CORE_M = 126;
constexpr int32_t SINGLE_CORE_N = 128;
constexpr int32_t SINGLE_CORE_K = 512;

// addRmsNorm 参数设置
constexpr float EPSILON = 1e-6;
constexpr float AVG_FACTOR = 1.0 / (float)5120.0;

/**
 * @brief 打印tilingData, addrms  and mamtul tcubetiling
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void PrintTilingDataInfo(gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "M is %u.", tilingData.addRmsNormDynamicQuantAllGatherTilingData.M);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "Ka is %u.", tilingData.addRmsNormDynamicQuantAllGatherTilingData.Ka);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "N is %u.", tilingData.addRmsNormDynamicQuantAllGatherTilingData.N);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "aivNum is %u.", tilingData.addRmsNormDynamicQuantAllGatherTilingData.aivNum);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "rankSize is %u.", tilingData.addRmsNormDynamicQuantAllGatherTilingData.rankSize);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "epsilon is %u.", tilingData.addRmsNormDynamicQuantAllGatherTilingData.epsilon);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "avgFactor is %u.", tilingData.addRmsNormDynamicQuantAllGatherTilingData.avgFactor);
    
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.M is %u.", tilingData.matmulTiling.M);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.Ka is %u.", tilingData.matmulTiling.Ka);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.Kb is %u.", tilingData.matmulTiling.Kb);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.N is %u.", tilingData.matmulTiling.N);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.singleCoreM is %u.", tilingData.matmulTiling.singleCoreM);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.singleCoreK is %u.", tilingData.matmulTiling.singleCoreK);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.singleCoreN is %u.", tilingData.matmulTiling.singleCoreN);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.baseM is %u.", tilingData.matmulTiling.baseM);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.baseK is %u.", tilingData.matmulTiling.baseK);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.baseN is %u.", tilingData.matmulTiling.baseN);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.stepM is %u.", tilingData.matmulTiling.stepM);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.stepKa is %u.", tilingData.matmulTiling.stepKa);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.stepKb is %u.", tilingData.matmulTiling.stepKb);
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "matmulTiling.stepN is %u.", tilingData.matmulTiling.stepN);

    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "Tiling end");
}

// addrmsNormdynamicquantallgatherqbmm tiling
static ge::graphStatus GetAddRmsNormDynamicQuantAllGatherQbmm(
    gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{   
    AddRmsNormDynamicQuantV2TilingHelper instanceNormV3TilingHelper(context);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OP_CHECK_IF(!status, OP_LOGE(context, "DoTiling Failed, return Failed."), return ge::GRAPH_FAILED);

    instanceNormV3TilingHelper.SetTilingData(&tilingData.addRmsNormDynamicQuantAllGatherTilingData);
    return ge::GRAPH_SUCCESS;
}

// matmul切分
static ge::graphStatus GetMatmultiling(
    gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{
    MmTilingHelper mmTilingHelper(context);
    OP_CHECK_IF(!mmTilingHelper.getMamtulArgs(),
        OP_LOGE(context, "MmTilingHelper get matmulArgs failed, return Failed."),
        return ge::GRAPH_FAILED);
    mmTilingHelper.InitCompileInfo();
    OP_CHECK_IF(!mmTilingHelper.InitTCubeTilingData(tilingData.matmulTiling),
        OP_LOGE(context, "MmTilingHelper InitTCubeTilingData failed, return Failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hcomm参数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static ge::graphStatus SetHcommCfg(const gert::TilingContext *context,
    AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    const char *nodeName = context->GetNodeName();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    OP_LOGD(nodeName, "group is %s in add_rms_norm_dynamic_quant_all_gather_qbmm.", groupPtr);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(std::string(groupPtr), OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    // MTE方式必要适配
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
//  需要修改核数设置
static void SetTilingData(gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData &tilingData)
{
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    numBlocks = ascendcPlatform.CalcTschBlockDim(aicNum, aicNum, aicNum);
    context->SetBlockDim(numBlocks);
    tilingData.addRmsNormDynamicQuantAllGatherTilingData.aivNum = aicNum;   // CV 1:1
}

/**
 * @brief 设置tilingKey, 先使用老的方式生成tilingkey
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 设置tilingKey模板参数
    // const uint64_t tilingKey = GET_TPL_TILING_KEY(MTE_COMM);
    context->SetTilingKey(1);
    // OP_LOGD(nodeName, "tilingKey is [%lu] in add_rms_norm_dynamic_quant_all_gather_qbmm.", tilingKey);
}

ge::graphStatus CheckAttrs(
    const gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    const char *nodeName = context->GetNodeName();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    // group空字符串校验
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(nodeName, "groupPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::string(groupPtr).empty(),
        OP_LOGE(nodeName, "group should not be empty."), return ge::GRAPH_FAILED);
    // rankSize校验
    const int64_t *rankSizePtr = attrs->GetAttrPointer<int64_t>(RANK_SIZE_INDEX);
    OP_TILING_CHECK(rankSizePtr == nullptr, OP_LOGE(nodeName, "rankSizePtr is nullptr."), return ge::GRAPH_FAILED);
    tilingData->addRmsNormDynamicQuantAllGatherTilingData.rankSize = *rankSizePtr;
    // transpose校验
    const bool *transposeX2Ptr = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    OP_TILING_CHECK(transposeX2Ptr == nullptr, OP_LOGE(nodeName, "transposeX2Ptr is nullptr."), return ge::GRAPH_FAILED);
    // 输出type校验
    const int64_t *outputTypePtr = attrs->GetAttrPointer<int64_t>(DTYPE_INDEX);
    OP_TILING_CHECK(outputTypePtr == nullptr, OP_LOGE(nodeName, "outputTypePtr is nullptr."), return ge::GRAPH_FAILED);
    // residual_norm_mode校验
    const float *residualNormModePtr = attrs->GetAttrPointer<float>(RESIDUAL_NORM_MODE_INDEX);
    OP_TILING_CHECK(residualNormModePtr == nullptr, OP_LOGE(nodeName, "residualNormModePtr is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputOutputTensorDim(
    const gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    // Check Shape Not NULL
    const gert::StorageShape* x1Shape = context->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context->GetInputShape(X2_INDEX);
    const gert::StorageShape* residualShape = context->GetInputShape(RESIDUAL_INDEX);
    const gert::StorageShape* yShape = context->GetInputShape(Y_INDEX);
    const gert::StorageShape* gammaShape = context->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* scaleShape = context->GetInputShape(SCALE_INDEX);
    const gert::StorageShape* smoothShape = context->GetOptionalInputShape(SMOOTH_SCALE_INDEX);
    const gert::StorageShape* biasShape = context->GetOptionalInputShape(BIAS_INDEX);

    const gert::StorageShape* outputShape = context->GetOutputShape(OUTPUT_INDEX);
    const gert::StorageShape* zShape = context->GetOutputShape(Z_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, residualShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, smoothShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);

    // Check Shape relations
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    size_t residualDimNum = residualShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t scaleDimNum = scaleShape->GetStorageShape().GetDimNum();
    size_t smoothDimNum = smoothShape->GetStorageShape().GetDimNum();
    size_t biasDimNum = biasShape->GetStorageShape().GetDimNum();
    uint64_t gammaValue = gammaShape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1Value = x1Shape->GetStorageShape().GetDim(1);

    OP_CHECK_IF((x1Shape->GetStorageShape().GetDim(1) != x2Shape->GetStorageShape().GetDim(1) * x2Shape->GetStorageShape().GetDim(2)),
        OP_LOGE(context->GetNodeName(), "x1dim1 is not same to x2dim1 * x2dim2."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((x1Shape->GetStorageShape() != residualShape->GetStorageShape()),
        OP_LOGE(context->GetNodeName(), "x1Shape is not same to residualShape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((x1Shape->GetStorageShape() != yShape->GetStorageShape()),
        OP_LOGE(context->GetNodeName(), "x1Shape is not same to yShape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(((x1Dim1Value != gammaValue)),
        OP_LOGE(context->GetNodeName(), "x1Dim1Value gammaValue not equal. x1Dim1Value=%lu, gammaValue=%lu ", x1Dim1Value, gammaValue), return ge::GRAPH_FAILED);
    OP_CHECK_IF((x1DimNum != TWO_DIMS) || (x2DimNum != FOUR_DIMS) || (yDimNum != TWO_DIMS) || (residualDimNum != TWO_DIMS),
        OP_LOGE(context->GetNodeName(),
        "The dim of x1, residual, y should be 2, and the dim of x2 should be 4, but current x1DimNum=%lu, x2DimNum=%lu, residualDimNum=%lu, yDimNum=%lu.",
        x1DimNum, x2DimNum, residualDimNum, yDimNum), return ge::GRAPH_FAILED);
    OP_CHECK_IF((smoothShape->GetStorageShape() != gammaShape->GetStorageShape()),
        OP_LOGE(context->GetNodeName(), "GammaShape is not same to smoothShape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(((x1DimNum != residualDimNum)),
        OP_LOGE(context->GetNodeName(), "Input x1/residual shape dims not equal. x1DimNum=%lu, residualDimNum=%lu ", x1DimNum, residualDimNum), return ge::GRAPH_FAILED);
    OP_CHECK_IF(((x1DimNum != yDimNum)),
        OP_LOGE(context->GetNodeName(), "Input x1/y shape dims not equal. x1DimNum=%lu, yDimNum=%lu ", x1DimNum, yDimNum), return ge::GRAPH_FAILED);
    OP_CHECK_IF(((scaleDimNum != 1)), OP_LOGE(context->GetNodeName(), "scale shape dims not equal to 1. scaleDimNum=%lu.", scaleDimNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((gammaDimNum != 1)), OP_LOGE(context->GetNodeName(), "gamma shape dims not equal to 1. gammaDimNum=%lu.", gammaDimNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((smoothDimNum != 1)), OP_LOGE(context->GetNodeName(), "smooth scale shape dims not equal to 1. smoothDimNum=%lu.", smoothDimNum),
        return ge::GRAPH_FAILED);
    
    tilingData->addRmsNormDynamicQuantAllGatherTilingData.M = x1Shape->GetStorageShape().GetDim(0);
    tilingData->addRmsNormDynamicQuantAllGatherTilingData.Ka = x1Dim1Value;
    tilingData->addRmsNormDynamicQuantAllGatherTilingData.N = \
        x2Shape->GetStorageShape().GetDim(0) * x2Shape->GetStorageShape().GetDim(3);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTensorDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto x1Desc = context->GetInputDesc(X1_INDEX);
    auto x2Desc = context->GetInputDesc(X2_INDEX);
    auto residualDesc = context->GetInputDesc(RESIDUAL_INDEX);
    auto yDesc = context->GetInputDesc(Y_INDEX);
    auto gammaDesc = context->GetInputDesc(GAMMA_INDEX);
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    auto smoothDesc = context->GetOptionalInputDesc(SMOOTH_SCALE_INDEX);
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    auto zDesc = context->GetOutputDesc(Z_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, residualDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, smoothDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, zDesc);

    OP_TILING_CHECK((x1Desc->GetDataType() != ge::DT_BF16) && (x1Desc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x1 dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(x1Desc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((x2Desc->GetDataType() != ge::DT_INT8),
        OP_LOGE(nodeName, "x2 dataType is invalid, dataType should be int8, but is %s.",
        Ops::Base::ToString(x1Desc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((residualDesc->GetDataType() != ge::DT_BF16) && (residualDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "residual dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(residualDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((yDesc->GetDataType() != ge::DT_BF16) && (yDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "y dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(yDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((gammaDesc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE(nodeName, "gamma dataType is invalid, dataType should be float32, but is %s.",
        Ops::Base::ToString(gammaDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((scaleDesc->GetDataType() != ge::DT_BF16) && (scaleDesc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE(nodeName, "scale dataType is invalid, dataType should be bf16 or float32, but is %s.",
        Ops::Base::ToString(scaleDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((smoothDesc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE(nodeName, "smooth dataType is invalid, dataType should be float32, but is %s.",
        Ops::Base::ToString(smoothDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((biasDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName, "bias dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(biasDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((outputDesc->GetDataType() != ge::DT_BF16) && (outputDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "output dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(outputDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((zDesc->GetDataType() != ge::DT_BF16) && (zDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "z dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(zDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
 
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTensorFormat(const gert::TilingContext *context)
{
    auto x1Desc = context->GetInputDesc(X1_INDEX);
    auto residualDesc = context->GetInputDesc(RESIDUAL_INDEX);
    auto yDesc = context->GetInputDesc(Y_INDEX);
    auto gammaDesc = context->GetInputDesc(GAMMA_INDEX);
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    auto smoothDesc = context->GetOptionalInputDesc(SMOOTH_SCALE_INDEX);
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    auto zDesc = context->GetOutputDesc(Z_INDEX);
    const char *nodeName = context->GetNodeName();
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "x1 format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(residualDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "residual format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "y format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(gammaDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "gamma format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(scaleDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "scale format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(smoothDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "smooth format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "bias format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(outputDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "output format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(zDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "z format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTCubeTiling(
    gert::TilingContext *context, AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MatmulApiTiling mmTiling(*ascendcPlatform);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    uint32_t M = tilingData->addRmsNormDynamicQuantAllGatherTilingData.M * \
        tilingData->addRmsNormDynamicQuantAllGatherTilingData.rankSize;
    uint32_t N = tilingData->addRmsNormDynamicQuantAllGatherTilingData.N;
    uint32_t K = tilingData->addRmsNormDynamicQuantAllGatherTilingData.Ka;

    uint32_t blockDim = context->GetBlockDim();
    const bool *transposeX2Ptr = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    bool isAtrans = true;
    bool isBtrans = *transposeX2Ptr;
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    bool hasBias = (biasDesc != nullptr);
    
    mmTiling.SetAType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, isAtrans);
    mmTiling.SetBType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, isBtrans);
    mmTiling.SetCType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mmTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    mmTiling.SetOrgShape(M, N, K);
    mmTiling.SetShape(SINGLE_CORE_M, SINGLE_CORE_N, SINGLE_CORE_K);
    mmTiling.SetFixSplit(SINGLE_CORE_M, SINGLE_CORE_N, SINGLE_CORE_K);
    mmTiling.EnableBias(hasBias);
    mmTiling.SetBufferSpace(-1, -1, -1);    // 默认使用该AI处理器所有空间
    // mmTiling.EnableMultiCoreSplitK(true);

    OP_TILING_CHECK(mmTiling.GetTiling(tilingData->matmulTiling) == -1,
                    OP_LOGE(nodeName, "failed to get tiling matmulTiling."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief add_rms_norm_dynamic_quant_all_gather_qbmm算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus AddRmsNormDynamicQuantAllGatherQbmmTilingFunc(gert::TilingContext *context)
{
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "Tiling start");
    OP_TILING_CHECK(context == nullptr,
                    OP_LOGE("add_rms_norm_dynamic_quant_all_gather_qbmm",
                        "failed to get tiling context in add_rms_norm_dynamic_quant_all_gather_qbmm."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
                    OP_LOGE("add_rms_norm_dynamic_quant_all_gather_qbmm",
                        "failed to get nodeName in add_rms_norm_dynamic_quant_all_gather_qbmm."),
                    return ge::GRAPH_FAILED);

    AddRmsNormDynamicQuantAllGatherQbmmTilingData *tilingData \
        = context->GetTilingData<AddRmsNormDynamicQuantAllGatherQbmmTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr in add_rms_norm_dynamic_quant_all_gather_qbmm."),
        return ge::GRAPH_FAILED);

    // 检查attrs
    OP_TILING_CHECK(CheckAttrs(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckAttrs failed."), return ge::GRAPH_FAILED);
    // 检查输入输出tensor的维度
    OP_TILING_CHECK(CheckInputOutputTensorDim(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckInputOutputTensorDim failed."), return ge::GRAPH_FAILED);
    // 检查输入输出tensor的数据类型
    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckTensorDataType failed."), return ge::GRAPH_FAILED);

    // 检查输入输出tensor的格式
    OP_TILING_CHECK(CheckTensorFormat(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckTensorFormat failed."), return ge::GRAPH_FAILED);

    // 设置 blockDim
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    numBlocks = ascendcPlatform.CalcTschBlockDim(aicNum, aicNum, aicNum);
    context->SetBlockDim(numBlocks);
    
    // 设置 AddRmsNorm 所需参数
    tilingData->addRmsNormDynamicQuantAllGatherTilingData.epsilon = EPSILON;
    tilingData->addRmsNormDynamicQuantAllGatherTilingData.avgFactor = AVG_FACTOR;

    // 调用matmul做tiling切分
    OP_TILING_CHECK(SetTCubeTiling(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetTCubeTiling failed."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetHcommCfg(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    SetTilingData(context, *tilingData);
    SetTilingKey(context);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = BASE_WORKSPACE_SIZE + \
        tilingData->addRmsNormDynamicQuantAllGatherTilingData.M * \
        tilingData->addRmsNormDynamicQuantAllGatherTilingData.N * \
        tilingData->addRmsNormDynamicQuantAllGatherTilingData.rankSize * \
        sizeof(int32_t);

    PrintTilingDataInfo(context, *tilingData);
    return ge::GRAPH_SUCCESS;
}

struct AddRmsNormDynamicQuantAllGatherQbmmCompileInfo {};

ge::graphStatus TilingParseForAddRmsNormDynamicQuantAllGatherQbmm(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
 
IMPL_OP_OPTILING(AddRmsNormDynamicQuantAllGatherQbmm)
    .Tiling(AddRmsNormDynamicQuantAllGatherQbmmTilingFunc)
    .TilingParse<AddRmsNormDynamicQuantAllGatherQbmmCompileInfo>(TilingParseForAddRmsNormDynamicQuantAllGatherQbmm);

} // namespace MC2Tiling
