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
 * \file moe_distribute_dispatch_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_tiling_key.h"
#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_dispatch_tiling_a2a3.h"
#include "tiling_base/tiling_templates_registry.h"
#include "mc2_hcom_topo_info.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
namespace {
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;

constexpr uint32_t DYN_SCALE_DIMS = 1;
constexpr uint32_t EXPAND_IDX_DIMS = 1;
constexpr uint64_t INIT_TILINGKEY = 1000;
constexpr uint32_t ARR_LENGTH = 128;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

constexpr uint32_t UNQUANT_MODE = 0;
constexpr uint32_t STATIC_QUANT_MODE = 1;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
constexpr uint32_t LAYERED_SUPPORT_K = 8;
constexpr uint32_t LAYERED_SUPPORT_K_MAX = 16;
constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
constexpr int32_t UNLAYERED_EXP_NUM_PER_RANK_A2 = 24;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
const char *K_INNER_DEBUG = "MoeDistributeDispatch Tiling Debug";
const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t MAX_TP_WORLD_SIZE = 2;
const int64_t BS_UPPER_BOUND = 512;

constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t STATIC_SCALES = 1;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr uint32_t NUM_10 = 10;
constexpr uint32_t NUM_100 = 100;
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 8;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t TP_WORLD_SIZE_TWO = 2U;

constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
}

namespace optiling {




static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE; // 16U * 1024U * 1024U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, MoeDistributeDispatchTilingData *tiling,
    const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize)
{
    const char *nodeName = context->GetNodeName();
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(3);   // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling1 failed"), return ge::GRAPH_FAILED);
}

static ge::graphStatus MoeDistributeDispatchTilingFunc(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchTilingData *tilingData = context->GetTilingData<MoeDistributeDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter MoeDistributeDispatch tiling check func.");

    // 获取入参属性
    groupEp = std::string(groupEpPtr);
    if (*tpWorldSizePtr > 1){
        groupTp = std::string(groupTpPtr);
    }
    // sharedExpertNumPtr 强制为1
    // globalBs/bs/k/h/a/aivNum/isQuant
    tilingData.moeDistributeDispatchInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeDispatchInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeDispatchInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData.moeDistributeDispatchInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchInfo.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    isScales = (scalesStorageShape != nullptr);
    tilingData->moeDistributeDispatchInfo.isQuant = isScales;
    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatch(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);
    // 检查属性的取值是否合法
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    tilingData.moeDistributeDispatchInfo.bs = static_cast<uint32_t>(xDim0);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchInfo.globalBs = bs * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isSharedExpert = false;
    }
    // 检查shape各维度并赋值h,k
    tilingData.moeDistributeDispatchInfo.h = static_cast<uint32_t>(xDim1);
    tilingData.moeDistributeDispatchInfo.k = static_cast<uint32_t>(expertIdsDim1);
    if (isSharedExpert) { // 本卡为共享专家
        A = globalBs / sharedExpertRankNum;
    } else {     // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, k);
    }
    tilingData.moeDistributeDispatchInfo.a = A;
    // 校验win区大小
    maxWindowSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() - MTE_STATE_ZONE_SIZE;
    tilingData->moeDistributeDispatchInfo.totalWinSizeEp = maxWindowSizeEp;
    // workspace
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, groupEp, groupTp, tpWorldSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHcommCfg failed."), return ge::GRAPH_FAILED);

    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);


    tilingData->moeDistributeDispatchInfo.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchInfo.aivNum = aivNum;

    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeDispatchCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

// A5 采用 MTE 方式通信，复用 A3 tiling
REGISTER_OPS_TILING_TEMPLATE(MoeDistributeDispatch, MoeDistributeDispatchTilingA2A3, 0);

ge::graphStatus MoeDistributeDispatchTilingA2A3::DoOpTiling()
{
    return MoeDistributeDispatchTilingFunc(context_);
}

ge::graphStatus MoeDistributeDispatchTiling(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}


bool MoeDistributeDispatchTilingA2A3::IsCapable()
{
    return true;
}

IMPL_OP_OPTILING(MoeDistributeDispatch)
    .Tiling(MoeDistributeDispatchTiling)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatch);
} // namespace optiling