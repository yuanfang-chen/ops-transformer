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
 * \file moe_distribute_dispatch_v2_extend_tiling.cpp
 * \brief
 */

#include "moe_distribute_dispatch_v2_tiling_base.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
using namespace DataBase;

namespace optiling {
static uint64_t CalTilingKey(const gert::TilingContext *context, const bool isScales, const uint32_t quantMode,
    const uint32_t tpWorldSize, const bool isSetCommAlg)
{
    uint32_t fullMesh = TILINGKEY_NO_FULLMESH;
    bool tp = false;
    uint32_t tilingKeyQuantMode = quantMode;
    bool scaleMode = false;
    uint64_t tilingKey;
    uint32_t commMode = TILINGKEY_TPL_HOST_KFC;
    if (isScales) {
            scaleMode = true;
    }
    if (mc2tiling::GetSocVersion(context) == "Ascend950") {
        commMode = TILINGKEY_TPL_HOST_KFC;
        tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode,
                                                fullMesh, commMode, TILINGKEY_TPL_A5);
    }
    return tilingKey;
}

template<typename ConstChosen>
static ge::graphStatus MoeDistributeDispatchKfcAndDpuTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = static_cast<uint32_t>(QuantModeA5::NON_QUANT);
    bool isScales = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isPerformance = false;
    bool isSetCommAlg = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchKfcAndDpuTilingFuncImpl tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData<ConstChosen>(context, nodeName, *tilingData, groupEp, groupTp, isSetCommAlg) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(ConstChosen::SCALES_INDEX);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(ConstChosen::X_ACTIVE_MASK_INDEX);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ConstChosen::ELASTIC_INFO_INDEX);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.hasElasticInfo = hasElasticInfo;

    // 获取performanceInfo
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(ConstChosen::PERFORMANCE_INFO_INDEX);
    isPerformance = (performanceInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isPerformance = isPerformance;

    quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;

    // 检查quantMode和scales是否匹配
    if (mc2tiling::GetSocVersion(context) == "Ascend950") {
        OP_TILING_CHECK(CheckQuantModeAndScales<ConstChosen>(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d,quantMode is %u.",
            static_cast<int32_t>(isScales),quantMode), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT), OP_LOGE(nodeName, "cannot support static quant now."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT))) || ((!isScales) && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))),
            OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
            static_cast<int32_t>(isScales), quantMode), return ge::GRAPH_FAILED);
    }

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        TilingCheckMoeDistributeDispatch<ConstChosen>(context, nodeName, isActiveMask, isScales, hasElasticInfo, isPerformance, quantMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs<ConstChosen>(context, nodeName, *tilingData, localMoeExpertNum, isActiveMask, isSetCommAlg) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchV2Info.epRankId;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    bool isSharedExpert = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape<ConstChosen>(context, nodeName, *tilingData, quantMode, isScales,
        isSharedExpert, hasElasticInfo, isPerformance, static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    // OP_TILING_CHECK(CheckWinSize<ConstChosen>(context, *tilingData, nodeName, isSetCommAlg, localMoeExpertNum) != ge::GRAPH_SUCCESS,
    //     OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, groupEp, groupTp, tpWorldSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set hcomm cfg failed."), return ge::GRAPH_FAILED);
    uint64_t tilingKey = CalTilingKey(context, isScales, quantMode, tpWorldSize, isSetCommAlg);

    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    context->SetScheduleMode(1); // 设置为batch mode模式, 所有核同时启动
    tilingData->moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV2ExtendTilingFunc(gert::TilingContext* context)
{
    std::string socVersion = mc2tiling::GetSocVersion(context);
    ge::graphStatus ret;
    if (socVersion == "Ascend950") {
        ret = MoeDistributeDispatchKfcAndDpuTilingFuncImpl<TilingExtendConst>(context);
    } else {
        ret = ge::GRAPH_FAILED;
    }
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2Extend(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2ExtendTilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2Extend);
} // namespace optiling