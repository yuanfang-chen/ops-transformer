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
 * \file all_gather_add_tiling.cpp
 * \brief Host 侧 Tiling 实现
 */

#include <string>

#include "securec.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/mc2_tiling_utils.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/all_gather_add_tiling_data.h"
#include "../op_kernel/all_gather_add_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

constexpr size_t ATTR_GROUP_INDEX = 0U;
constexpr size_t ATTR_RANK_SIZE_INDEX = 1U;
constexpr size_t ATTR_COMM_TURN_INDEX = 2U;

constexpr uint32_t OP_TYPE_ALL_GATHER = 6;
constexpr int64_t EXPECT_RANK_SIZE = 2;
constexpr int64_t EXPECT_COMM_TURN = 2;
constexpr int64_t EXPECT_DIMS = 2;
constexpr int64_t TILE_ELEMENTS_PER_CORE_CALC = 8192;
constexpr const char *ALL_GATHER_ALG_CONFIG = "AllGather=level0:doublering";

const uint32_t WS_SYS_SIZE = 0;

struct AllGatherAddCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeDescAttrsInfo(const gert::TilingContext *context, int64_t &localInputElemCount,
                                             int64_t &totalOutputElemCount, int64_t &elemsPerRankPerRound,
                                             int64_t &rankSize, int64_t &commTurn, std::string &group)
{
    auto inputA = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputA);
    auto inputB = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputB);
    auto outputGathered = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputGathered);
    auto outputC = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputC);
    auto inputShapeA = EnsureNotScalar(inputA->GetStorageShape());
    auto inputShapeB = EnsureNotScalar(inputB->GetStorageShape());
    auto outputShapeGathered = EnsureNotScalar(outputGathered->GetStorageShape());
    auto outputShapeC = EnsureNotScalar(outputC->GetStorageShape());

    auto inputDescA = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescA);
    auto inputDescB = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescB);
    auto outputDescGathered = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDescGathered);
    auto outputDescC = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDescC);

    ge::DataType dtypeA = inputDescA->GetDataType();
    ge::DataType dtypeB = inputDescB->GetDataType();
    ge::DataType dtypeGathered = outputDescGathered->GetDataType();
    ge::DataType dtypeC = outputDescC->GetDataType();
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(inputDescA->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(inputDescB->GetStorageFormat()));
    ge::Format formatGathered = static_cast<ge::Format>(ge::GetPrimaryFormat(outputDescGathered->GetStorageFormat()));
    ge::Format formatC = static_cast<ge::Format>(ge::GetPrimaryFormat(outputDescC->GetStorageFormat()));

    OP_CHECK_IF(inputShapeA.GetDimNum() != EXPECT_DIMS || inputShapeB.GetDimNum() != EXPECT_DIMS ||
                    outputShapeGathered.GetDimNum() != EXPECT_DIMS || outputShapeC.GetDimNum() != EXPECT_DIMS,
        OP_LOGE(context, "all_gather_add only supports 2D tensors."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dtypeA != ge::DT_FLOAT16 || dtypeB != ge::DT_FLOAT16 ||
                    dtypeGathered != ge::DT_FLOAT16 || dtypeC != ge::DT_FLOAT16,
        OP_LOGE(context, "all_gather_add only supports FLOAT16 dtype."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(formatA != ge::FORMAT_ND || formatB != ge::FORMAT_ND ||
                    formatGathered != ge::FORMAT_ND || formatC != ge::FORMAT_ND,
        OP_LOGE(context, "all_gather_add only supports ND format."), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *groupPtr = attrs->GetStr(ATTR_GROUP_INDEX);
    const int64_t *rankSizePtr = attrs->GetInt(ATTR_RANK_SIZE_INDEX);
    const int64_t *commTurnPtr = attrs->GetInt(ATTR_COMM_TURN_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, groupPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, rankSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, commTurnPtr);
    OP_CHECK_IF(std::string(groupPtr).empty(), OP_LOGE(context, "group should not be empty."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*rankSizePtr != EXPECT_RANK_SIZE,
        OP_LOGE(context, "rank_size must be %ld, but got %ld.", EXPECT_RANK_SIZE, *rankSizePtr),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*commTurnPtr != EXPECT_COMM_TURN,
        OP_LOGE(context, "current version only supports comm_turn = %ld, but got %ld.",
                EXPECT_COMM_TURN, *commTurnPtr),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputShapeB.GetShapeSize() != outputShapeGathered.GetShapeSize() ||
                    outputShapeGathered.GetShapeSize() != outputShapeC.GetShapeSize(),
        OP_LOGE(context, "shape size relation is invalid for all_gather_add outputs."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputShapeA.GetShapeSize() * (*rankSizePtr) != outputShapeGathered.GetShapeSize(),
        OP_LOGE(context, "a_gathered shape size must equal a shape size multiplied by rank_size."),
        return ge::GRAPH_FAILED);

    group = groupPtr;
    rankSize = *rankSizePtr;
    commTurn = *commTurnPtr;

    localInputElemCount = inputShapeA.GetShapeSize();
    totalOutputElemCount = outputShapeGathered.GetShapeSize();
    elemsPerRankPerRound = Ops::Base::CeilDiv(localInputElemCount, commTurn);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, AllGatherAddTilingData *tilingData,
                                   const std::string &group)
{
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, OP_TYPE_ALL_GATHER, ALL_GATHER_ALG_CONFIG);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);
    OP_CHECK_IF(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(context, "mc2CcTilingConfig GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(context, "mc2CcTilingConfig GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(gert::TilingContext *context, AllGatherAddTilingData &tilingData,
                          int64_t rankSize, int64_t commTurn, int64_t localInputElemCount,
                          int64_t elemsPerRankPerTurn, int64_t totalOutputElemCount, int64_t coreNum)
{
    AllGatherAddTilingInfo &schedule = tilingData.allGatherAddTilingInfo;
    schedule.rankCount = rankSize;
    schedule.turnCount = commTurn;
    schedule.inputElementsPerRank = localInputElemCount;
    schedule.elementsPerTurnPerRank = elemsPerRankPerTurn;
    schedule.lastTurnElementsPerRank = localInputElemCount - elemsPerRankPerTurn * (commTurn - 1);
    schedule.outputElements = totalOutputElemCount;
    schedule.elementsPerCorePerTurn = Ops::Base::CeilDiv(elemsPerRankPerTurn, coreNum);
    schedule.tileElementsPerCoreCalc = TILE_ELEMENTS_PER_CORE_CALC;
    context->SetBlockDim(static_cast<uint32_t>(coreNum));
}

static void SetTilingKey(gert::TilingContext *context)
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(ALL_GATHER_ADD_SCH_MODE_BASIC);
    context->SetTilingKey(tilingKey);
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingDataInfo(gert::TilingContext *context, const AllGatherAddTilingData &tilingData, uint64_t workspaceSize,
                                int64_t coreNum)
{
    const char *nodeName = context->GetNodeName();
    const auto &schedule = tilingData.allGatherAddTilingInfo;
    OP_LOGD(nodeName, "turnCount is %ld in all_gather_add.", schedule.turnCount);
    OP_LOGD(nodeName, "inputElementsPerRank is %ld in all_gather_add.", schedule.inputElementsPerRank);
    OP_LOGD(nodeName, "elementsPerTurnPerRank is %ld in all_gather_add.", schedule.elementsPerTurnPerRank);
    OP_LOGD(nodeName, "lastTurnElementsPerRank is %ld in all_gather_add.", schedule.lastTurnElementsPerRank);
    OP_LOGD(nodeName, "elementsPerCorePerTurn is %ld in all_gather_add.", schedule.elementsPerCorePerTurn);
    OP_LOGD(nodeName, "tileElementsPerCoreCalc is %ld in all_gather_add.", schedule.tileElementsPerCoreCalc);
    OP_LOGD(nodeName, "outputElements is %ld in all_gather_add.", schedule.outputElements);
    OP_LOGD(nodeName, "blockDim is %ld in all_gather_add.", coreNum);
    OP_LOGD(nodeName, "workspaceSize is %lu in all_gather_add.", workspaceSize);
}

static ge::graphStatus AllGatherAddTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t localInputElemCount = 0;
    int64_t totalOutputElemCount = 0;
    int64_t elemsPerRankPerRound = 0;
    std::string group;
    int64_t rankSize = 0;
    int64_t commTurn = 0;
    OP_CHECK_IF(
        GetShapeDescAttrsInfo(context, localInputElemCount, totalOutputElemCount, elemsPerRankPerRound,
                              rankSize, commTurn, group) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeDescAttrsInfo error"),
        return ge::GRAPH_FAILED);

    AllGatherAddTilingData* tiling = context->GetTilingData<AllGatherAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AllGatherAddTilingData), 0, sizeof(AllGatherAddTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetHcommCfg(context, tiling, group) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetHcommCfg error"),
        return ge::GRAPH_FAILED);

    SetTilingData(context, *tiling, rankSize, commTurn, localInputElemCount,
                  elemsPerRankPerRound, totalOutputElemCount, coreNum);
    OP_CHECK_IF(
        SetWorkspace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetWorkspace error"),
        return ge::GRAPH_FAILED);
    SetTilingKey(context);
    PrintTilingDataInfo(context, *tiling, WS_SYS_SIZE, coreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAllGatherAdd([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherAdd)
    .Tiling(AllGatherAddTilingFunc)
    .TilingParse<AllGatherAddCompileInfo>(TilingParseForAllGatherAdd);

} // namespace optiling
