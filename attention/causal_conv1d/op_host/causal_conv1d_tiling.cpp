/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "util/math_util.h"
#include "../op_kernel/causal_conv1d_tiling_data.h"
#include "../op_kernel/causal_conv1d_tiling_key.h"

#include <set>
#include <limits>

namespace optiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t CONV_STATES_INDEX = 3;
constexpr uint32_t QUERY_START_LOC_INDEX = 4;
constexpr uint32_t CACHE_INDICES_INDEX = 5;
constexpr uint32_t HAS_INITIAL_STATE_INDEX = 6;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;

struct CausalConv1dCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};

struct DimTileChoice {
    int64_t dimTileSize = 0;
    int64_t blocksPerSeq = 0;
    int64_t gridSize = 0;
};

static inline DimTileChoice ChooseDimTileSize(int64_t batch, int64_t dim, uint32_t coreNum)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512};
    DimTileChoice bestOver;
    int64_t bestOverGap = std::numeric_limits<int64_t>::max();
    DimTileChoice bestUnder;

    for (int64_t dimTileSize : candidates) {
        if (dim % dimTileSize != 0) {
            continue;
        }
        const int64_t blocksPerSeq = dim / dimTileSize;
        const int64_t gridSize = batch * blocksPerSeq;
        if (gridSize <= 0) {
            continue;
        }
        printf("[Tiling] candidate: dimTile=%ld, blocksPerSeq=%ld, grid=%ld, core=%u\n",
               dimTileSize, blocksPerSeq, gridSize, coreNum);
        if (gridSize >= static_cast<int64_t>(coreNum)) {
            const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
            if (gap < bestOverGap) {
                bestOver = {dimTileSize, blocksPerSeq, gridSize};
                bestOverGap = gap;
            }
        } else if (gridSize > bestUnder.gridSize ||
                   (gridSize == bestUnder.gridSize && dimTileSize < bestUnder.dimTileSize)) {
            bestUnder = {dimTileSize, blocksPerSeq, gridSize};
        }
    }
    DimTileChoice result = (bestOver.dimTileSize != 0) ? bestOver : bestUnder;
    printf("[Tiling] chosen: dimTile=%ld, blocksPerSeq=%ld, grid=%ld\n",
           result.dimTileSize, result.blocksPerSeq, result.gridSize);
    return result;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    auto compileInfoPtr = context->GetCompileInfo<CausalConv1dCompileInfo>();
    if (compileInfoPtr != nullptr && compileInfoPtr->coreNum != 0 && compileInfoPtr->ubSize != 0) {
        ubSize = compileInfoPtr->ubSize;
        coreNum = compileInfoPtr->coreNum;
        return ge::GRAPH_SUCCESS;
    }
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext* context, int64_t& activationMode, int64_t& padSlotId)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationModePtr);
    activationMode = *activationModePtr;
    OP_CHECK_IF(
        activationMode != 0 && activationMode != 1, OP_LOGE(context, "activationMode only supports 0/1"),
        return ge::GRAPH_FAILED);

    const int64_t* padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padSlotIdPtr);
    padSlotId = *padSlotIdPtr;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeDtypeInfo(gert::TilingContext* context, CausalConv1dTilingData& tiling)
{
    auto xShapePtr = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    int64_t dim = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t batch = 0;
    int64_t inputMode = 0;

    if (xShape.GetDimNum() == 2) {
        inputMode = 0;
        cuSeqlen = xShape.GetDim(0);
        dim = xShape.GetDim(1);
        seqLen = 0;
        OP_CHECK_IF(dim <= 0 || cuSeqlen < 0, OP_LOGE(context, "invalid x shape for 2D varlen mode"), return ge::GRAPH_FAILED);
    } else if (xShape.GetDimNum() == 3) {
        inputMode = 1;
        batch = xShape.GetDim(0);
        seqLen = xShape.GetDim(1);
        dim = xShape.GetDim(2);
        cuSeqlen = batch * seqLen;
        OP_CHECK_IF(batch <= 0 || dim <= 0 || seqLen <= 0, OP_LOGE(context, "invalid x shape for 3D batch mode"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "x must be 2D (cu_seqlen, dim) or 3D (batch, seqlen, dim)");
        return ge::GRAPH_FAILED;
    }

    auto wShapePtr = context->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    OP_CHECK_IF(wShape.GetDimNum() != 2, OP_LOGE(context, "weight must be 2D: (width, dim)"), return ge::GRAPH_FAILED);
    const int64_t width = wShape.GetDim(0);
    const int64_t wDim = wShape.GetDim(1);
    OP_CHECK_IF(wDim != dim, OP_LOGE(context, "weight.shape[1] must equal dim"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(width != 4, OP_LOGE(context, "MVP only supports width=4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dim != 1024 && dim != 2048 && dim != 4096 && dim != 8192,
                OP_LOGE(context, "MVP only supports dim in {1024, 2048, 4096, 8192}"),
                return ge::GRAPH_FAILED);

    auto sShapePtr = context->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sShapePtr);
    auto sShape = EnsureNotScalar(sShapePtr->GetStorageShape());
    OP_CHECK_IF(
        sShape.GetDimNum() != 3, OP_LOGE(context, "convStates must be 3D: (num_cache_lines, state_len, dim)"),
        return ge::GRAPH_FAILED);
    const int64_t numCacheLines = sShape.GetDim(0);
    const int64_t stateLen = sShape.GetDim(1);
    const int64_t sDim = sShape.GetDim(2);
    OP_CHECK_IF(numCacheLines <= 0, OP_LOGE(context, "convStates.shape[0] (num_cache_lines) must be > 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sDim != dim, OP_LOGE(context, "convStates.shape[2] must equal dim"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateLen < (width - 1), OP_LOGE(context, "convStates.shape[1] must be >= width-1"), return ge::GRAPH_FAILED);

    auto qslShapePtr = context->GetInputShape(QUERY_START_LOC_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qslShapePtr);
    auto qslShape = EnsureNotScalar(qslShapePtr->GetStorageShape());
    OP_CHECK_IF(qslShape.GetDimNum() != 1, OP_LOGE(context, "queryStartLoc must be 1D"), return ge::GRAPH_FAILED);
    const int64_t qslSize = qslShape.GetDim(0);
    OP_CHECK_IF(qslSize < 1, OP_LOGE(context, "queryStartLoc.size must be >= 1"), return ge::GRAPH_FAILED);

    if (inputMode == 0) {
        batch = qslSize - 1;
    }

    if (inputMode == 1) {
        OP_CHECK_IF(qslSize != batch + 1, OP_LOGE(context, "queryStartLoc.size must equal batch + 1"),
                    return ge::GRAPH_FAILED);
    }

    // cacheIndices: (batch)
    auto ciShapePtr = context->GetInputShape(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ciShapePtr);
    auto ciShape = EnsureNotScalar(ciShapePtr->GetStorageShape());
    OP_CHECK_IF(ciShape.GetDimNum() != 1, OP_LOGE(context, "cacheIndices must be 1D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ciShape.GetDim(0) != batch, OP_LOGE(context, "cacheIndices.size must equal batch"), return ge::GRAPH_FAILED);

    // hasInitialState: (batch)
    auto hisShapePtr = context->GetInputShape(HAS_INITIAL_STATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hisShapePtr);
    auto hisShape = EnsureNotScalar(hisShapePtr->GetStorageShape());
    OP_CHECK_IF(hisShape.GetDimNum() != 1, OP_LOGE(context, "hasInitialState must be 1D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(hisShape.GetDim(0) != batch, OP_LOGE(context, "hasInitialState.size must equal batch"), return ge::GRAPH_FAILED);

    // bias: (dim) optional
    tiling.hasBias = 0;
    auto biasShapePtr = context->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr != nullptr && biasShapePtr->GetStorageShape().GetDimNum() != 0) {
        auto biasShape = EnsureNotScalar(biasShapePtr->GetStorageShape());
        OP_CHECK_IF(biasShape.GetDimNum() != 1, OP_LOGE(context, "bias must be 1D: (dim,)"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(biasShape.GetDim(0) != dim, OP_LOGE(context, "bias.size must equal dim"), return ge::GRAPH_FAILED);
        tiling.hasBias = 1;
    }

    // dtype checks
    const std::set<ge::DataType> supportedXDtype = {ge::DT_BF16, ge::DT_FLOAT16};
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(supportedXDtype.count(xDtype) == 0, OP_LOGE(context, "x dtype only supports bf16/fp16"), return ge::GRAPH_FAILED);

    auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    OP_CHECK_IF(wDesc->GetDataType() != xDtype, OP_LOGE(context, "weight dtype must equal x dtype"), return ge::GRAPH_FAILED);

    if (tiling.hasBias == 1) {
        auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
        OP_CHECK_IF(biasDesc->GetDataType() != xDtype, OP_LOGE(context, "bias dtype must equal x dtype"), return ge::GRAPH_FAILED);
    }

    auto sDesc = context->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sDesc);
    OP_CHECK_IF(sDesc->GetDataType() != xDtype, OP_LOGE(context, "convStates dtype must equal x dtype"), return ge::GRAPH_FAILED);

    auto qslDesc = context->GetInputDesc(QUERY_START_LOC_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
    OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT32, OP_LOGE(context, "queryStartLoc dtype must be int32"), return ge::GRAPH_FAILED);

    auto ciDesc = context->GetInputDesc(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ciDesc);
    OP_CHECK_IF(ciDesc->GetDataType() != ge::DT_INT32, OP_LOGE(context, "cacheIndices dtype must be int32"), return ge::GRAPH_FAILED);

    auto hisDesc = context->GetInputDesc(HAS_INITIAL_STATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hisDesc);
    OP_CHECK_IF(hisDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(context, "hasInitialState dtype must be bool"), return ge::GRAPH_FAILED);

    tiling.dim = dim;
    tiling.cuSeqlen = cuSeqlen;
    tiling.seqLen = seqLen;
    tiling.inputMode = inputMode;
    tiling.width = width;
    tiling.stateLen = stateLen;
    tiling.numCacheLines = numCacheLines;
    tiling.batch = batch;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    uint32_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);

    // 2、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 3、设置tiling信息
    CausalConv1dTilingData* tiling = context->GetTilingData<CausalConv1dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(CausalConv1dTilingData), 0, sizeof(CausalConv1dTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetAttrsInfo(context, tiling->activationMode, tiling->padSlotId) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetShapeDtypeInfo(context, *tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeDtypeInfo error"),
        return ge::GRAPH_FAILED);

    const int64_t dim = tiling->dim;
    const int64_t batch = tiling->batch;
    OP_CHECK_IF(dim <= 0 || batch <= 0, OP_LOGE(context, "dim/batch must be positive"), return ge::GRAPH_FAILED);

    const DimTileChoice choice = ChooseDimTileSize(batch, dim, coreNum);
    OP_CHECK_IF(choice.dimTileSize <= 0 || choice.blocksPerSeq <= 0 || choice.gridSize <= 0,
                OP_LOGE(context, "invalid dim_tile_size selection"),
                return ge::GRAPH_FAILED);

    const uint32_t blockDim = (choice.gridSize < static_cast<int64_t>(coreNum))
                                  ? static_cast<uint32_t>(choice.gridSize)
                                  : coreNum;

    printf("[Tiling] FINAL: batch=%ld, dim=%ld, dimTileSize=%ld, blocksPerSeq=%ld, gridSize=%ld, blockDim=%u, coreNum=%u\n",
           batch, dim, choice.dimTileSize, choice.blocksPerSeq, choice.gridSize, blockDim, coreNum);

    context->SetBlockDim(blockDim);
    tiling->dimTileSize = choice.dimTileSize;
    tiling->blocksPerSeq = choice.blocksPerSeq;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(CAUSAL_CONV1D_TPL_SCH_MODE_DEFAULT);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<CausalConv1dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(compileInfoPtr->coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    OP_CHECK_IF(compileInfoPtr->ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(CausalConv1d)
    .Tiling(CausalConv1dTilingFunc)
    .TilingParse<CausalConv1dCompileInfo>(TilingParseForCausalConv1d);
} // namespace optiling
