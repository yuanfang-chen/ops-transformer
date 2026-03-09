/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_grouped_mat_mul_quant_tiling_base.cc
 * \brief
 */

#include <string>
#include <numeric>
#include <climits>
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "context_util.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../allto_allv_grouped_mat_mul_tiling_base.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"
#include "allto_allv_grouped_mat_mul_quant_tiling_base.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

ge::graphStatus AlltoAllvGmmQuantTilingBase::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "start quant GetPlatformInfo.");
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end quant GetPlatformInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetShapeAttrsInfo.");
    if (GetCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTilingBase::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoOpTiling.");
    if (CheckGmmDType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMmDType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivCoreNum_, aicCoreNum_, aivCoreNum_));
    context_->SetTilingKey(GetTilingKey());
    OP_TILING_CHECK(SetHcclTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end DoOpTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTilingBase::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "start DoLibApiTiling.");
    uint64_t maxMSize = 0;
    uint64_t mSize = 0;
    for (uint64_t expertIdx = 0; expertIdx < e_; expertIdx++) {
        mSize = 0;
        for (uint64_t rankIdx = 0; rankIdx < epWorldSize_; rankIdx++) {
            mSize += recvCounts[rankIdx * e_ + expertIdx];
        }
        maxMSize = std::max(mSize, maxMSize);
    }
    if (maxMSize != 0) {
        auto &gmmQuantTilingData = tilingData->gmmQuantTilingData;
        SetGMMQuantParams(gmmQuantTilingData);
        SetTilingArray(gmmQuantTilingData, maxMSize, n1_, h1_);
        SetTilingParams(gmmQuantTilingData, maxMSize, n1_, h1_);
        PrintGMMQuantTilingData(gmmQuantTilingData);
    }
    if (bs_ != 0) {
        auto &mmQuantTilingData = tilingData->mmQuantTilingData;
        SetGMMQuantParams(mmQuantTilingData);
        SetTilingArray(mmQuantTilingData, bs_, n2_, h2_);
        SetTilingParams(mmQuantTilingData, bs_, n2_, h2_);
        PrintGMMQuantTilingData(mmQuantTilingData);
    }
    OP_LOGD(context_->GetNodeName(), "end DoLibApiTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTilingBase::GetWorkspaceSize()
{
    OP_LOGD(context_->GetNodeName(), "start GetWorkspaceSize.");
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "can not get workspace."),
        return ge::GRAPH_FAILED);
    
    uint64_t permuteOutSize = permuteOutFlag_ ? 0 : (a_ * h1_ * GetSizeByDataType(gmmXDataType_));
    // 将 permuteOutSize 对齐到 512 字节
    const uint64_t tensorListSize = 512;
    if (permuteOutSize % tensorListSize != 0) {
        permuteOutSize = (permuteOutSize + tensorListSize - 1) & ~(tensorListSize - 1);
    }
    // mx pergroup quant
    if (isMxfp8_) {
        uint64_t h1 = Ops::Base::CeilDiv(h1_, MX_BASIC_FACTOR);
        uint64_t permuteScaleOutSize = permuteOutFlag_ ? 0 : (a_ * h1 * 2 * GetSizeByDataType(gmmWeightDataType_));
        permuteOutSize += permuteScaleOutSize;
    }
    uint64_t groupListSize = sizeof(int64_t) * e_; // GMM计算所需的groupList GM空间大小
    // tensorListSize为kernel侧tensorlist开辟的空间
    workspaces[0] = libApiWorkSpaceSize_ + permuteOutSize + groupListSize + tensorListSize;
    OP_LOGD(context_->GetNodeName(), "end GetWorkspaceSize.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTilingBase::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "start PostTiling.");
    tilingData->taskTilingInfo.BSK = bsk_;
    tilingData->taskTilingInfo.BS = bs_;
    tilingData->taskTilingInfo.H1 = h1_;
    tilingData->taskTilingInfo.H2 = h2_;
    tilingData->taskTilingInfo.A = a_;
    tilingData->taskTilingInfo.N1 = n1_;
    tilingData->taskTilingInfo.N2 = n2_;
    tilingData->taskTilingInfo.epWorldSize = epWorldSize_;
    tilingData->taskTilingInfo.e = e_;
    tilingData->taskTilingInfo.mainLoopExpertNum = e_;
    tilingData->taskTilingInfo.tailLoopExpertNum = 0;
    tilingData->taskTilingInfo.totalLoopCount = e_;
    for (uint32_t i = 0; i < e_ * epWorldSize_; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = sendCounts[i];
        tilingData->taskTilingInfo.recvCnt[i] = recvCounts[i];
    }
    PrintTaskTilingInfo(tilingData->taskTilingInfo);
    OP_LOGD(context_->GetNodeName(), "end PostTiling.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvGmmQuantTilingBase::SetGMMQuantParams(
    Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData) const
{
    gmmQuantTilingData.gmmQuantParams.groupNum = SINGLE_GROUP_NUM;
    gmmQuantTilingData.gmmQuantParams.activeType = GMM_ACT_TYPE_NONE;
    gmmQuantTilingData.gmmQuantParams.aQuantMode = isMxfp8_ ? MX_PERGROUP_MODE : PERTENSOR_MODE;
    gmmQuantTilingData.gmmQuantParams.bQuantMode = isMxfp8_ ? MX_PERGROUP_MODE : PERTENSOR_MODE;
    gmmQuantTilingData.gmmQuantParams.singleX = 0;
    gmmQuantTilingData.gmmQuantParams.singleW = 0;
    gmmQuantTilingData.gmmQuantParams.singleY = 0;
    gmmQuantTilingData.gmmQuantParams.groupType = 0;
    gmmQuantTilingData.gmmQuantParams.groupListType = 1;
    gmmQuantTilingData.gmmQuantParams.hasBias = 0;
    gmmQuantTilingData.gmmQuantParams.reserved = 0;
}

void AlltoAllvGmmQuantTilingBase::SetTilingArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const
{
    gmmQuantTilingData.gmmArray.mList[0] = static_cast<int32_t>(M);
    gmmQuantTilingData.gmmArray.kList[0] = static_cast<int32_t>(K);
    gmmQuantTilingData.gmmArray.nList[0] = static_cast<int32_t>(N);
}

void AlltoAllvGmmQuantTilingBase::PrintGMMQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const
{
    const auto &mm = data.mmTilingData;
    const auto &quantParams = data.gmmQuantParams;
    const auto &gmmArray = data.gmmArray;

    std::stringstream ss;
    ss << "MM Tiling: M=" << mm.M << ", N=" << mm.N << ", K=" << mm.Ka << ", usedCoreNum=" << mm.usedCoreNum <<
        ", baseM=" << mm.baseM << ", baseN=" << mm.baseN << ", baseK=" << mm.baseK << ", singleCoreM=" <<
        mm.singleCoreM << ", singleCoreN=" << mm.singleCoreN << ", singleCoreK=" << mm.singleCoreK << ", dbL0C=" <<
        mm.dbL0C << ", depthA1=" << mm.depthA1 << ", depthB1=" << mm.depthB1 << ", stepKa=" << mm.stepKa <<
        ", stepKb=" << mm.stepKb << ", stepM=" << mm.stepM << ", stepN=" << mm.stepN << ", iterateOrder=" <<
        mm.iterateOrder;

    ss << "\nQuant Params: groupNum=" << quantParams.groupNum << ", activeType=" << quantParams.activeType <<
        ", aQuantMode=" << quantParams.aQuantMode << ", bQuantMode=" << quantParams.bQuantMode << ", singleX=" <<
        quantParams.singleX << ", singleW=" << quantParams.singleW << ", singleY=" << quantParams.singleY <<
        ", groupType=" << quantParams.groupType << ", groupListType=" << quantParams.groupListType << ", hasBias=" <<
        quantParams.hasBias << ", reserved=" << quantParams.reserved;

    ss << "\nArray: mList[0]=" << gmmArray.mList[0] << ", kList[0]=" << gmmArray.kList[0] << ", nList[0]=" <<
        gmmArray.nList[0];

    OP_LOGI(context_->GetNodeName(), "AlltoAllvGmmQuantTilingBase TilingParams:\n%s", ss.str().c_str());
}

void AlltoAllvGmmQuantTilingBase::PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo) const
{
    std::stringstream ss;
    ss << "TaskTilingInfo: ";
    ss << "BSK=" << taskTilingInfo.BSK << ", BS=" << taskTilingInfo.BS << ", H1=" << taskTilingInfo.H1 << ", H2=" <<
        taskTilingInfo.H2 << ", A=" << taskTilingInfo.A << ", N1=" << taskTilingInfo.N1 << ", N2=" << taskTilingInfo.N2;
    ss << ", epWorldSize=" << taskTilingInfo.epWorldSize << ", e=" << taskTilingInfo.e;
    ss << ", mainLoopExpertNum=" << taskTilingInfo.mainLoopExpertNum << ", tailLoopExpertNum=" <<
        taskTilingInfo.tailLoopExpertNum << ", totalLoopCount=" << taskTilingInfo.totalLoopCount;
    ss << "\nSendCounts: ";
    for (int i = 0; i < e_ * epWorldSize_; i++) {
        if (taskTilingInfo.sendCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.sendCnt[i];
        }
    }
    ss << "\nRecvCounts: ";
    for (int i = 0; i < e_ * epWorldSize_; i++) {
        if (taskTilingInfo.recvCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.recvCnt[i];
        }
    }
    OP_LOGI(context_->GetNodeName(), "%s", ss.str().c_str());
}

ge::graphStatus AlltoAllvGmmQuantTilingBase::SetHcclTiling() const
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(gmmXDataType_) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(context_->GetNodeName(), "alltoAllvDataType is not found in HCCL_DATA_TYPE."), return ge::GRAPH_FAILED);
    auto alltoAllvDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(gmmXDataType_)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType, alltoAllvDataType,
        alltoAllvDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.hcclInitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.a2avCcTiling failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
