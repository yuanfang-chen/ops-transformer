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
 * \file moe_distribute_combine_v2_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include "../../../moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_base.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_tiling.h"
// #include "arch35/moe_distribute_combine_tiling_arch35.h"
#include "../../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../op_kernel/moe_distribute_combine_v2_extend_tiling_key.h"
#include "mc2_hcom_topo_info.h"

using namespace Mc2Tiling;
using namespace common_const;
using namespace AscendC;
using namespace ge;


using Idx = common_const::IndexExtend;

namespace optiling {

static void PrintTilingDataInfo(const char *nodeName, MoeDistributeCombineV2ExtendTilingData& tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeCombineV2ExtendInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeCombineV2ExtendInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeCombineV2ExtendInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeCombineV2ExtendInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeCombineV2ExtendInfo.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeCombineV2ExtendInfo.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeDistributeCombineV2ExtendInfo.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeCombineV2ExtendInfo.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeCombineV2ExtendInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeCombineV2ExtendInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeCombineV2ExtendInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeCombineV2ExtendInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeCombineV2ExtendInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeCombineV2ExtendInfo.totalWinSizeEp);
    OP_LOGD(nodeName, "totalWinSizeTP is %lu.", tilingData.moeDistributeCombineV2ExtendInfo.totalWinSizeTp);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeCombineV2ExtendInfo.hasElasticInfo);
    OP_LOGD(nodeName, "isPerformance is %d.", tilingData.moeDistributeCombineV2ExtendInfo.isPerformance);
    OP_LOGD(nodeName, "hcclBufferSize is %d.", tilingData.moeDistributeCombineV2ExtendInfo.hcclBufferSize); // 新增属性打印
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context,
    MoeDistributeCombineV2ExtendTilingData &tilingData, const char *nodeName, std::string &groupEp, std::string &groupTp,
    uint32_t &commQuantMode)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(Idx::attr::ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(Idx::attr::ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(Idx::attr::ATTR_SHARED_EXPERT_NUM_INDEX));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_MOE_EXPERT_NUM_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(Idx::attr::ATTR_COMM_QUANT_MODE_INDEX));
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(Idx::attr::ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(Idx::attr::ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(Idx::attr::ATTR_CONST_EXPERT_NUM_INDEX));
    auto hcclBufferSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(Idx::attr::HCCL_BUFF_SIZE)); // 新增属性hcclBufferSize指针获取

    // 判空
    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH), OP_LOGE(nodeName, "groupEp is invalid."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardType is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE(nodeName, "commQuantMode is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(nodeName, "zeroExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(nodeName, "copyExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(nodeName, "constExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclBufferSizePtr == nullptr, OP_LOGE(nodeName, "hcclBufferSize is null."), return ge::GRAPH_FAILED); // // 新增属性hcclBufferSize指针判空


    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;

    OP_TILING_CHECK(
        (moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) > INT32_MAX,
        OP_LOGE(nodeName, "moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum exceeds MAX_INT32."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((zeroExpertNum < 0), OP_LOGE(nodeName,
        "zeroExpertNum less than 0, zeroExpertNum is %ld.", zeroExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0), OP_LOGE(nodeName,
        "copyExpertNum less than 0, copyExpertNum is %ld.", copyExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0), OP_LOGE(nodeName,
        "constExpertNum less than 0, constExpertNum is %ld.", constExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
        MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        MAX_TP_WORLD_SIZE, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((*commQuantModePtr != 0), OP_LOGE(nodeName,
            "commQuantMode only supports 0 when tpWorldSize > 1, but got commQuantMode=%ld, tpWorldSize=%ld.",
            *commQuantModePtr, *tpWorldSizePtr), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
        MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
        epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        MOE_EXPERT_MAX_NUM, moeExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*commQuantModePtr != 0) && (*commQuantModePtr != INT8_COMM_QUANT),
        OP_LOGE(nodeName, "commQuantMode only support 0(default) or 2(int8 comm quant), but got commQuantMode=%ld.",
        *commQuantModePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
            OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
            epWorldSize, *epRankIdPtr), return ge::GRAPH_FAILED);
        commQuantMode = static_cast<uint32_t>(*commQuantModePtr);
    groupEp = string(groupEpPtr);
    tilingData.moeDistributeCombineV2ExtendInfo.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDistributeCombineV2ExtendInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeCombineV2ExtendInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertRankNum = static_cast<uint32_t>(sharedExpertRankNum);
    if (tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertRankNum == 0U) {
        if (tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertNum == 1U) {
            tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertNum = 0U;
        }
    }
    tilingData.moeDistributeCombineV2ExtendInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDistributeCombineV2ExtendInfo.zeroExpertNum = static_cast<uint32_t>(zeroExpertNum);
    tilingData.moeDistributeCombineV2ExtendInfo.copyExpertNum = static_cast<uint32_t>(copyExpertNum);
    tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum = static_cast<uint32_t>(constExpertNum);
    tilingData.moeDistributeCombineV2ExtendInfo.hcclBufferSize = static_cast<uint64_t>(*hcclBufferSizePtr); // 新增属性赋值

    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2ExtendTilingData &tilingData,
    const char *nodeName, bool isShared, bool isActiveMask, uint32_t localMoeExpertNum, const bool hasElasticInfo, const bool isPerformance)
{
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(Idx::input::EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.moeExpertNum);
    int64_t zeroExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.zeroExpertNum);
    int64_t copyExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.copyExpertNum);
    int64_t constExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX || (expertIdsDim1 > moeExpertNum
        + zeroExpertNum + copyExpertNum + constExpertNum)),
        OP_LOGE(nodeName, "expertIds's dim1(K) should be in (0, min(%ld, moeExpertNum"
        " + zeroExpertNum + copyExpertNum + constExpertNum = %ld)], "
        "but got expertIds's dim1=%ld.", K_MAX, moeExpertNum
        + zeroExpertNum + copyExpertNum + constExpertNum, expertIdsDim1), return false);
    tilingData.moeDistributeCombineV2ExtendInfo.k = static_cast<uint32_t>(expertIdsDim1);

    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDistributeCombineV2ExtendInfo.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertRankNum;
    uint32_t rankNumPerSharedExpert = 0;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeCombineV2ExtendInfo.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    uint32_t maxSharedGroupNum = 0;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isShared) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localMoeExpertNum), expertIdsDim1);
    }

    const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.epWorldSize);
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(Idx::input::ELASTIC_INFO_INDEX);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);

        OP_TILING_CHECK(elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
            OP_LOGE(nodeName, "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
            "elasticInfo's dim0 is %ld, epWorldSize is %ld.",
            elasticInfoDim0, epWorldSize), return false);
        A = std::max( static_cast<int64_t>(maxBs * maxSharedGroupNum) , globalBs * std::min(static_cast<int64_t>(localMoeExpertNum), expertIdsDim1));
    }
    tilingData.moeDistributeCombineV2ExtendInfo.a = A;

    if (isPerformance) {
        const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(Idx::input::PERFORMANCE_INFO_INDEX);
        const int64_t performanceInfoDim0 = performanceInfoStorageShape->GetStorageShape().GetDim(0);

        OP_TILING_CHECK(performanceInfoDim0 != epWorldSize,
            OP_LOGE(nodeName, "performanceInfo's dim0 not equal to epWorldSize, "
            "performanceInfo's dim0 is %ld, epWorldSize is %ld.",
            performanceInfoDim0, epWorldSize), return false);
    }

    // 校验expandX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(Idx::input::EXPAND_X_INDEX);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
        "expandX's dim0 not greater than or equal to A * tpWorldSize, expandX's dim0 = %ld, A = %ld, tpWorldSize = %ld",
        expandXDim0, static_cast<int64_t>(A), tpWorldSize), return false);
    OP_TILING_CHECK((expandXDim1 < H_MIN) || (expandXDim1 > H_MAX),
        OP_LOGE(nodeName, "expandX's dim1(H) should be in [%ld, %ld], but got %ld.",
        H_MIN, H_MAX, expandXDim1), return false); // 32对齐
    tilingData.moeDistributeCombineV2ExtendInfo.h = static_cast<uint32_t>(expandXDim1);

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(Idx::input::ASSIST_INFO_INDEX);
    int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(assistInfoDim0 < static_cast<int64_t>(A * ASSIST_NUM_PER_A), OP_LOGE(nodeName,
        "assistInfoForCombine's dim0 < A * 128, assistInfoForCombine's dim0 is %ld, A * 128 is %ld.", assistInfoDim0, static_cast<int64_t>(A * ASSIST_NUM_PER_A)),
        return false);

    // 校验epSendCount和tpSendCount的维度
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.moeExpertPerRankNum);
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(Idx::input::EP_SEND_COUNTS_INDEX);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t localEpSendCountSize = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;

    if (hasElasticInfo) {
        localEpSendCountSize = std::max(epWorldSize,epWorldSize * moeExpertPerRankNum);
    }
    OP_TILING_CHECK(epSendCountDim0 < localEpSendCountSize * tpWorldSize, OP_LOGE(nodeName,
        "epSendCount's dim0 not greater than or equal to localEpSendCountSize * tpWorldSize, "
        "epSendCount's dim0 is %ld, localEpSendCountSize is %ld, tpWorldSize is %ld.",
        epSendCountDim0, localEpSendCountSize, tpWorldSize), return false);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        const gert::StorageShape *tpSendCountStorageShape = context->GetOptionalInputShape(Idx::input::TP_SEND_COUNTS_INDEX);
        const int64_t tpSendCountDim0 = tpSendCountStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(tpSendCountDim0 != tpWorldSize, OP_LOGE(nodeName,
            "tpSendCount's dim0 not equal to tpWorldSize, tpSendCount's dim0 is %ld, tpWorldSize is %ld.",
            tpSendCountDim0, tpWorldSize), return false);
    }
    // 校验expertScales的维度
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(Idx::input::EXPERT_SCALES_INDEX);
    int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertScalesDim0 != expertIdsDim0,
        OP_LOGE(nodeName, "expertScales's dim0 not equal to bs, expertScales's dim0 = %ld, bs = %ld",
        expertScalesDim0, expertIdsDim0), return false);
    OP_TILING_CHECK(expertScalesDim1 != expertIdsDim1, OP_LOGE(nodeName,
        "expertScales's dim1 not equal to k, expertScales's dim1 = %ld, k = %ld",
        expertScalesDim1, expertIdsDim1), return false);

    // 校验activeMask的维度
    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(Idx::input::X_ACTIVE_MASK_INDEX);
        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != expertIdsDim0,
            OP_LOGE(nodeName, "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
            "expertIds's dim0 is %ld", xActiveMaskDim0, expertIdsDim0), return false);
        OP_TILING_CHECK(((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdsDim1)),
            OP_LOGE(nodeName, "xActiveMask's dim1 not equal to expertIds's dim1, xActiveMask's dim1 is %ld, "
            "expertIds's dim1 is %ld", xActiveMaskStorageShape->GetStorageShape().GetDim(1), expertIdsDim1), return false);
    }

    // 校验sharedExpertX的维度
    const gert::StorageShape *sharedExpertXShape = context->GetOptionalInputShape(Idx::input::SHARED_EXPERT_X_INDEX);
    tilingData.moeDistributeCombineV2ExtendInfo.hasSharedExpertX = (sharedExpertXShape != nullptr);
    if (sharedExpertXShape != nullptr) {
        int64_t sharedExpertXDim0 = sharedExpertXShape->GetStorageShape().GetDim(0);
        int64_t sharedExpertXDim1 = sharedExpertXShape->GetStorageShape().GetDim(1);
        if (sharedExpertXShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
            OP_TILING_CHECK(sharedExpertXDim0 != expertIdsDim0,
                OP_LOGE(nodeName, "sharedExpertX's dim0 not equal to bs, sharedExpertX's dim0 = %ld, bs = %ld",
                sharedExpertXDim0, expertIdsDim0), return false);
            OP_TILING_CHECK(sharedExpertXDim1 != expandXDim1, OP_LOGE(nodeName,
                "sharedExpertX's dim1 not equal to h, sharedExpertX's dim1 = %ld, h = %ld",
                sharedExpertXDim1, expandXDim1), return false);
        } else {
            int64_t sharedExpertXDim2 = sharedExpertXShape->GetStorageShape().GetDim(TWO_DIMS);
            OP_TILING_CHECK(sharedExpertXDim0 * sharedExpertXDim1 != expertIdsDim0,
                OP_LOGE(nodeName, "sharedExpertX's dim0 * sharedExpertX's dim1 not equal to bs, sharedExpertX's dim0 * sharedExpertX's dim1 = %ld, bs = %ld",
                sharedExpertXDim0 * sharedExpertXDim1, expertIdsDim0), return false);
            OP_TILING_CHECK(sharedExpertXDim2 != expandXDim1, OP_LOGE(nodeName,
                "sharedExpertX's dim2 not equal to h, sharedExpertX's dim2 = %ld, h = %ld",
                sharedExpertXDim2, expandXDim1), return false);
        }
    }

    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(Idx::output::OUTPUT_X_INDEX);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName,
        "x's dim0 not equal to bs, bs = %ld, x's dim0 = %ld", expertIdsDim0, xDim0), return false);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName,
        "x's dim1 not equal to h, x's dim1 = %ld, h = %ld", xDim1, expandXDim1), return false);

    const gert::StorageShape* oriXShape = context->GetOptionalInputShape(Idx::input::ORI_X_INDEX);
    if (oriXShape != nullptr) {
        int64_t oriXDim0 = oriXShape->GetStorageShape().GetDim(0);
        int64_t oriXDim1 = oriXShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(oriXDim0 != expertIdsDim0,
            OP_LOGE(nodeName, "ori_x's dim0 not equal to bs, ori_x's dim0 = %ld, bs = %ld", oriXDim0, expertIdsDim0),
            return false);
        OP_TILING_CHECK(oriXDim1 != expandXDim1,
            OP_LOGE(nodeName, "ori_x's dim1 not equal to h, ori_x's dim1 = %ld, h = %ld", oriXDim1, expandXDim1),
            return false);
    }

    const gert::StorageShape* constExpertAlpha1Shape = context->GetOptionalInputShape(Idx::input::CONST_EXPERT_ALPHA_1_INDEX);
    if (constExpertAlpha1Shape != nullptr) {
        int64_t constExpertAlpha1Dim0 = constExpertAlpha1Shape->GetStorageShape().GetDim(0);
        int64_t constExpertAlpha1Dim1 = constExpertAlpha1Shape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertAlpha1Dim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum),
            OP_LOGE(nodeName,
                "const_expert_alpha_1's dim0 not equal to const_expert_num, const_expert_alpha_1's dim0 = %ld, "
                "const_expert_num = %u",
                constExpertAlpha1Dim0, tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum),
            return false);
        OP_TILING_CHECK(constExpertAlpha1Dim1 != expandXDim1,
            OP_LOGE(nodeName,
                "const_expert_alpha_1's dim1 not equal to h, const_expert_alpha_1's dim1 = %ld, h = %ld",
                constExpertAlpha1Dim1, expandXDim1),
            return false);
    }

    const gert::StorageShape* constExpertAlpha2Shape = context->GetOptionalInputShape(Idx::input::CONST_EXPERT_ALPHA_2_INDEX);
    if (constExpertAlpha2Shape != nullptr) {
        int64_t constExpertAlpha2Dim0 = constExpertAlpha2Shape->GetStorageShape().GetDim(0);
        int64_t constExpertAlpha2Dim1 = constExpertAlpha2Shape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertAlpha2Dim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum),
            OP_LOGE(nodeName,
                "const_expert_alpha_2's dim0 not equal to const_expert_num, const_expert_alpha_2's dim0 = %ld, "
                "const_expert_num = %u",
                constExpertAlpha2Dim0, tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum),
            return false);
        OP_TILING_CHECK(constExpertAlpha2Dim1 != expandXDim1,
            OP_LOGE(nodeName,
                "const_expert_alpha_2's dim1 not equal to h, const_expert_alpha_2's dim1 = %ld, h = %ld",
                constExpertAlpha2Dim1, expandXDim1),
            return false);
    }

    const gert::StorageShape* constExpertVShape = context->GetOptionalInputShape(Idx::input::CONST_EXPERT_V_INDEX);
    if (constExpertVShape != nullptr) {
        int64_t constExpertVDim0 = constExpertVShape->GetStorageShape().GetDim(0);
        int64_t constExpertVDim1 = constExpertVShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertVDim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum),
            OP_LOGE(nodeName,
                "const_expert_v's dim0 not equal to const_expert_num, const_expert_v's dim0 = %ld, const_expert_num = %u.",
                constExpertVDim0, tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum),
            return false);
        OP_TILING_CHECK(constExpertVDim1 != expandXDim1,
            OP_LOGE(nodeName, "const_expert_v's dim1 not equal to h, const_expert_v's dim1 = %ld, h = %ld.",
                constExpertVDim1, expandXDim1),
            return false);
    }

    return true;
}

static bool CheckSharedAttrs(const char *nodeName, const MoeDistributeCombineV2ExtendTilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support 0 when sharedExpertNum is 0, but got %u.",
        sharedExpertRankNum), return false);
    OP_TILING_CHECK((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support 0 when sharedExpertRankNum is 0, but got %u.",
        sharedExpertNum), return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName, "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
            "sharedExpertNum=%u.", sharedExpertRankNum, sharedExpertNum), return false);
    }

    return true;
}

static bool CheckAttrs(const gert::TilingContext *context, MoeDistributeCombineV2ExtendTilingData &tilingData,
    const char *nodeName, uint32_t &localMoeExpertNum, bool isActiveMask)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineV2ExtendInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeCombineV2ExtendInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeCombineV2ExtendInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2ExtendInfo.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
        OP_LOGE(nodeName, "Check shared expert related attributes failed."), return false);

    // 校验moe专家数量能否均分给多机
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but got moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.", moeExpertNum, epWorldSize,
        sharedExpertRankNum), return false);
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > LOCAL_EXPERT_MAX_SIZE),OP_LOGE(nodeName, "localMoeExpertNum is invalid, "
        "localMoeExpertNum * epWorldSize must be less than or equal to 2048, and localMoeExpertNum must be greater than 0, "
        "but got localMoeExpertNum * epWorldSize = %u, localMoeExpertNum = %u", localMoeExpertNum * epWorldSize, localMoeExpertNum), return false);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((localMoeExpertNum > 1) && (tpWorldSize > 1),
        OP_LOGE(nodeName, "Cannot support multi-moeExpert %u in a rank when tpWorldSize = %u > 1",
        localMoeExpertNum, tpWorldSize), return false);
    // 校验tp=2时是否没有动态缩容参数
    OP_TILING_CHECK((tpWorldSize > 1) && (tilingData.moeDistributeCombineV2ExtendInfo.hasElasticInfo), OP_LOGE(nodeName, "Cannot support elasticInfo "
        "when tpWorldSize = %u > 1", tpWorldSize), return false);
    tilingData.moeDistributeCombineV2ExtendInfo.moeExpertPerRankNum = localMoeExpertNum;

    // 校验输入expertIds的维度0并设bs
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(Idx::input::EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((expertIdsDim0 <= 0) || (expertIdsDim0 > BS_UPPER_BOUND),
        OP_LOGE(nodeName, "Invalid expertIds dims0(BS) %ld. Should be between [1, %ld].",
        expertIdsDim0, BS_UPPER_BOUND), return false);
    tilingData.moeDistributeCombineV2ExtendInfo.bs = static_cast<uint32_t>(expertIdsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(Idx::attr::ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBs is null."), return false);
    OP_LOGD(nodeName, "MoeDistributeCombineV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, expertIdsDim0, epWorldSize);

    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, expertIdsDim0, epWorldSize),  return false);
    OP_TILING_CHECK(((*globalBsPtr > (expertIdsDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
        OP_LOGE(nodeName, "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, expertIdsDim0, epWorldSize), return false);

    tilingData.moeDistributeCombineV2ExtendInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeCombineV2ExtendInfo.globalBs = static_cast<uint32_t>(expertIdsDim0) * epWorldSize;
    }

    uint32_t copyExpertNum = tilingData.moeDistributeCombineV2ExtendInfo.copyExpertNum;
    uint32_t constExpertNum = tilingData.moeDistributeCombineV2ExtendInfo.constExpertNum;

    const gert::StorageShape *oriXStorageShape = context->GetOptionalInputShape(Idx::input::ORI_X_INDEX);
    const gert::StorageShape *constExpertAlpha1StorageShape = context->GetOptionalInputShape(Idx::input::CONST_EXPERT_ALPHA_1_INDEX);
    const gert::StorageShape *constExpertAlpha2StorageShape = context->GetOptionalInputShape(Idx::input::CONST_EXPERT_ALPHA_2_INDEX);
    const gert::StorageShape *constExpertVStorageShape = context->GetOptionalInputShape(Idx::input::CONST_EXPERT_V_INDEX);

    OP_TILING_CHECK(copyExpertNum > 0 && oriXStorageShape == nullptr,
        OP_LOGE(nodeName, "oriX must exist when copyExpertNum > 0"), return false);
    OP_TILING_CHECK(constExpertNum > 0 && (oriXStorageShape == nullptr || constExpertAlpha1StorageShape == nullptr ||
                    constExpertAlpha2StorageShape == nullptr || constExpertVStorageShape == nullptr),
        OP_LOGE(nodeName, "oriX、alpha1、alpha2、V must exist when constExpertNum > 0"), return false);

    return true;
}

static void UbUsedCal(const uint64_t ubSize, const gert::TilingContext* context, MoeDistributeCombineV2ExtendTilingData *tilingData)
{
    uint32_t axisH = tilingData->moeDistributeCombineV2ExtendInfo.h;
    uint32_t axisBS = tilingData->moeDistributeCombineV2ExtendInfo.bs;
    uint32_t axisK = tilingData->moeDistributeCombineV2ExtendInfo.k;
    uint32_t zeroExpertNum = tilingData->moeDistributeCombineV2ExtendInfo.zeroExpertNum;
    uint32_t copyExpertNum = tilingData->moeDistributeCombineV2ExtendInfo.copyExpertNum;
    uint32_t constExpertNum = tilingData->moeDistributeCombineV2ExtendInfo.constExpertNum;
    bool isInputExpertMaskFlag = tilingData->moeDistributeCombineV2ExtendInfo.isExpertMask;
    bool isInputTokenMaskFlag = tilingData->moeDistributeCombineV2ExtendInfo.isTokenMask;
    bool enableSpecialExpert = (constExpertNum + zeroExpertNum + copyExpertNum > 0U);
    auto expandXDesc = context->GetInputDesc(Idx::input::EXPAND_X_INDEX);
    auto attrs = context->GetAttrs();
    auto commQuantModePtr = attrs->GetAttrPointer<int>(Idx::attr::ATTR_COMM_QUANT_MODE_INDEX);
    uint32_t maxSizeTokenBuf = (axisH * sizeof(expandXDesc->GetDataType()) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hExpandXTypeSize = axisH * sizeof(expandXDesc->GetDataType());
    uint32_t activeMaskAlignSize = axisBS * ((axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    uint32_t hExpandXAlign32Size = (hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hFloatSize = axisH * static_cast<uint32_t>(sizeof(float));
    uint32_t hFloatAlign32Size = (hFloatSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size;
    uint32_t flagRcvCount = axisK + tilingData->moeDistributeCombineV2ExtendInfo.sharedExpertNum;
    uint32_t hFloatAlign256Size = (hFloatSize + ALIGNED_LEN - 1) / ALIGNED_LEN * ALIGNED_LEN;
    uint32_t bsKNum = axisBS * axisK;
    uint32_t bsKFloatAlign = (bsKNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size > bsKFloatAlign ? hFloatAlign256Size : bsKFloatAlign;

    if (isInputExpertMaskFlag || enableSpecialExpert) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize * sizeof(DTYPE_SIZE_HALF);
        maxSizeTokenBuf = (activeMaskAlignSize > hExpandXAlign32Size ? activeMaskAlignSize : hExpandXAlign32Size);
        maxSizeRowTmpFloatBuf = (activeMaskAlignHalfSize > hFloatAlign32Size ? activeMaskAlignHalfSize : hFloatAlign32Size);
    }

    // LocalWindowCopy的ub使用总量
    uint32_t totalBufferSize = maxSizeTokenBuf + maxSizeRowTmpFloatBuf + mulBufSize + hFloatAlign32Size + hExpandXAlign32Size * BUFFER_NUM
        + flagRcvCount * STATE_OFFSET * BUFFER_NUM + UB_ALIGN;
    if (*commQuantModePtr == INT8_COMM_QUANT) {
        uint32_t scaleNum = (hExpandXAlign32Size / sizeof(expandXDesc->GetDataType())) / static_cast<uint32_t>(UB_ALIGN / sizeof(float));
        uint32_t scaleNumAlignSize = (scaleNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += scaleNumAlignSize;
    }
    if (isInputTokenMaskFlag) {
        uint32_t axisBsAlignSize = (axisBS * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += axisBsAlignSize + axisBsAlignSize * sizeof(DTYPE_SIZE_HALF) * BUFFER_NUM;
    }
    if (isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN + (axisBS * sizeof(int32_t) +
            UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN + (axisBS * axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    if (enableSpecialExpert && !isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    tilingData->moeDistributeCombineV2ExtendInfo.bufferNum = totalBufferSize > ubSize ? BUFFER_SINGLE : BUFFER_NUM;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeDistributeCombineV2ExtendTilingData* tilingData,
    const char *nodeName, uint32_t localMoeExpertNum)
{
    auto attrs = context->GetAttrs();
    uint64_t hcclBufferSizeEp = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.hcclBufferSize);
    uint64_t maxWindowSizeEp = 0;
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.epWorldSize);
    uint64_t k = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.k);
    uint64_t sharedExpertNum = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.sharedExpertNum);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.globalBs)/ epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE  + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // dispatch数据区 token首对齐512，有效token长度h_align_32b + scale(32b) + 三元组(3*4b)
    uint64_t tokenActualLen = ((h * MAX_OUT_DTYPE_SIZE  + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualSize = ((maxBs * tokenNeedSizeDispatch * epWorldSize * static_cast<uint64_t>(localMoeExpertNum))
        + (maxBs * tokenNeedSizeCombine * (k + static_cast<uint64_t>(sharedExpertNum)))) * DOUBLE_DATA_BUFFER;
    OP_TILING_CHECK((actualSize > maxWindowSizeEp),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE_EP is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
            " localMoeExpertNum = %u, sharedExpertNum = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * ep_worldsize * localMoeExpertNum) +"
            " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) = %luMB,"
            " HCCL_BUFFSIZE=%luMB.", maxBs, h, epWorldSize, localMoeExpertNum, sharedExpertNum,
            tokenNeedSizeDispatch, tokenNeedSizeCombine, k, actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / MB_SIZE),
            return ge::GRAPH_FAILED);
    tilingData->moeDistributeCombineV2ExtendInfo.totalWinSizeEp = maxWindowSizeEp;
    OP_LOGD(nodeName, "EpwindowSize = %lu", maxWindowSizeEp);

    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.tpWorldSize);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        uint64_t maxWindowSizeTp = 0;
        auto groupTpHccl = attrs->GetAttrPointer<char>(static_cast<int>(Idx::attr::ATTR_GROUP_TP_INDEX));
        OP_TILING_CHECK(mc2tiling::GetCclBufferSize(groupTpHccl, &maxWindowSizeTp, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Get TP HcclBufferSize failed, HcclBufferSizeTP is %lu", maxWindowSizeTp),
            return ge::GRAPH_FAILED);
        actualSize = static_cast<uint64_t>(tilingData->moeDistributeCombineV2ExtendInfo.a) * (tokenNeedSizeDispatch +
        tokenNeedSizeCombine) * DOUBLE_DATA_BUFFER;
        OP_TILING_CHECK((actualSize > maxWindowSizeTp),
        OP_LOGE(nodeName, "TP HCCL_BUFFSIZE is too SMALL, A = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            "NEEDED_HCCL_BUFFSIZE(A * (tokenNeedSizeDispatch + tokenNeedSizeCombine) * 2) = %luMB, TP HCCL_BUFFSIZE= %luMB.",
            tilingData->moeDistributeCombineV2ExtendInfo.a, tokenNeedSizeDispatch, tokenNeedSizeCombine, actualSize / MB_SIZE + 1UL,
            maxWindowSizeTp / MB_SIZE), return ge::GRAPH_FAILED);
        tilingData->moeDistributeCombineV2ExtendInfo.totalWinSizeTp = maxWindowSizeTp;
        OP_LOGD(nodeName, "TpwindowSize = %lu", maxWindowSizeTp);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA5ExtendTilingFuncImpl(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeCombineV2Extend Tiling func");
    MoeDistributeCombineV2ExtendTilingData *tilingData = context->GetTilingData<MoeDistributeCombineV2ExtendTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    bool isShared = true;
    uint32_t localMoeExpertNum = 1;
    bool isActiveMask = false;
    uint32_t commQuantMode = 0U;
    bool hasElasticInfo = false;
    bool isPerformance = false;

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, groupTp, commQuantMode) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting Idx::attr failed."), return ge::GRAPH_FAILED);

    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(Idx::input::X_ACTIVE_MASK_INDEX);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeCombineV2ExtendInfo.isTokenMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeCombineV2ExtendInfo.isExpertMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(Idx::input::ELASTIC_INFO_INDEX);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeCombineV2ExtendInfo.hasElasticInfo = hasElasticInfo;

    // 获取performanceInfo
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(Idx::input::PERFORMANCE_INFO_INDEX);
    isPerformance = (performanceInfoStorageShape != nullptr);
    tilingData->moeDistributeCombineV2ExtendInfo.isPerformance = isPerformance;

    // 检查输入输出的dim、format、dataType
    uint32_t tpWorldSize = tilingData->moeDistributeCombineV2ExtendInfo.tpWorldSize;
    OP_TILING_CHECK(TilingCheckMoeDistributeCombine<Idx>(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, tpWorldSize) !=
                    ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum, isActiveMask),
        OP_LOGE(nodeName, "Idx::attr check failed."), return ge::GRAPH_FAILED);

    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineV2ExtendInfo.sharedExpertRankNum;
    uint32_t epRankId = tilingData->moeDistributeCombineV2ExtendInfo.epRankId;

    isShared = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, isShared, isActiveMask, localMoeExpertNum, hasElasticInfo, isPerformance),
        OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    OP_TILING_CHECK(CheckWinSize(context, tilingData, nodeName, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    uint64_t tilingKey = CalTilingKey(tpWorldSize, commQuantMode);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->moeDistributeCombineV2ExtendInfo.aivNum = aivNum;
    tilingData->moeDistributeCombineV2ExtendInfo.totalUbSize = ubSize;
    UbUsedCal(ubSize, context, tilingData);
    context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %lu, ubsize = %lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineV2ExtendTilingFunc(gert::TilingContext* context)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(Idx::input::EXPAND_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %d",
                    static_cast<ge::DataType>(expandXDesc->GetDataType())), return ge::GRAPH_FAILED);

    // todo：hccl_topo_type的使用
    return MoeDistributeCombineA5ExtendTilingFuncImpl(context);
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineV2Extend(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineV2Extend)
    .Tiling(MoeDistributeCombineV2ExtendTilingFunc)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombineV2Extend);
} // namespace optiling