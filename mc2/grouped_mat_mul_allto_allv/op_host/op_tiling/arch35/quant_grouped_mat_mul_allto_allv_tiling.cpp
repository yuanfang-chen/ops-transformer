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
 * \file quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_split_strategy.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {
ge::graphStatus GmmAlltoAllvTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Fail to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    contextInfo.args_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
};

// 待补充 量化类型判断
bool QuantGroupedMatmulAllToAllvTiling::IsCapable()
{
    return true;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CheckOpInputInfo()
{
    // check
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::CalTilingInferredInfo()
{
    constexpr uint64_t alignAddrLen = 512;
    auto yDesc = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX);
    auto yDType = yDesc->GetDataType();
    auto yDtypeSize = mc2tiling::GetDataTypeSize(opName_, yDType);
    inferredInfo.gmmResultLen = mc2tiling::AlignUp(
        gmmQTilingCommonInfoPtr->BSK * gmmQTilingCommonInfoPtr->N1 * yDtypeSize, alignAddrLen);
    
    inferredInfo.permuteLen = inferredInfo.gmmResultLen;
    auto mmyDesc = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX);
    if (mmyDesc != nullptr) {
        auto mmyDType = mmyDesc->GetDataType();
        auto mmyDtypeSize = mc2tiling::GetDataTypeSize(opName_, mmyDType);
        inferredInfo.mmResultLen = mc2tiling::AlignUp(
            gmmQTilingCommonInfoPtr->BSK * gmmQTilingCommonInfoPtr->N1 * mmyDtypeSize, alignAddrLen); 
    }
    // commLen
    inferredInfo.commLen = inferredInfo.gmmResultLen;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::SetTilingCommonInfo()
{
    auto gmmQTilingCommonInfoPtr = &localTilingData_.taskTilingInfo;
     
    auto xShape = context_->GetInputDesc(GMM_X_INDEX)->GetOriginShape();
    auto weightShape = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetOriginShape();
    epNum_ = weightShape.GetDim(DIM_0);
    gmmQTilingCommonInfoPtr->e = epNum_;

    auto gmmYShape = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetOriginShape();
    
    gmmQTilingCommonInfoPtr->BSK = xShape.GetDim(DIM_0);
    gmmQTilingCommonInfoPtr->H1 = xShape.GetDim(DIM_1);
    gmmQTilingCommonInfoPtr->N1 = gmmYShape.GetDim(DIM_1);

    auto attrs = context_->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    gmmQTilingCommonInfoPtr->epWorldSize = *epWorldSizePtr;

    gmmQTilingCommonInfoPtr->mainLoopExpertNum = 1;
    gmmQTilingCommonInfoPtr->tailLoopExpertNum = 1;
    gmmQTilingCommonInfoPtr->totalLoopCount = epNum_;

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    const int64_t* sendCounts = static_cast<const int64_t*>(sendCountsPtr->GetData());
    const int64_t* recvCounts = static_cast<const int64_t*>(recvCountsPtr->GetData());
    for (int i = 0; i < MAX_EXPERT_NUM; i++) {
        // memcpy_s
        gmmQTilingCommonInfoPtr->sendCnt[i] = sendCounts[i];
        gmmQTilingCommonInfoPtr->recvCnt[i] = recvCounts[i];
    }

    auto mmYDesc = context_->GetInputDesc(OUTPUT_MM_Y_INDEX);
    // if (mmYDesc != nullptr) {

    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::SetGmmA2avWorkspaceInfo()
{
    CalTilingInferredInfo();
    workspaceSize_ = libApiWorkSpaceSize_ + inferredInfo.gmmResultLen + inferredInfo.mmResultLen +
        inferredInfo.commLen + inferredInfo.permuteLen;
    localTilingData_.workspaceInfo.wsGmmSize = workspaceSize_;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::DoQuantGMMTiling()
{
    // 设置GMM切前信息
    QuantGroupedMatmulAllToAllvAdapter gmmTile(*this, context_);
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetCommonContextParameters());
    auto gmmTilingPtr = &localTilingData_.gmmTiling;
    // 当前为 epNums，每轮一专家
    gmmTilingPtr->count = taskTilingInfoPtr->totalLoopCount;
    auto taskTilingInfoPtr = &localTilingData_.taskTilingInfo;

    auto expertNumPerLoop = taskTilingInfoPtr->mainLoopExpertNum;
    // 每轮
    uint32_t loop;
    auto worldSize = gmmQTilingCommonInfoPtr->epWorldSize;
    for (loop = 0; loop < taskTilingInfoPtr->totalLoopCount - 1; loop++) {
        GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetExpertInputParameters(loop * expertNumPerLoop * worldSize, expertNumPerLoop));
        GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
        gmmTilingPtr->array[loop] = gmmTile.GetGmmQuantTilingData();
    }

    // 尾轮
    expertNumPerLoop = taskTilingInfoPtr->tailLoopExpertNum;
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetExpertInputParameters(loop * expertNumPerLoop * worldSize, expertNumPerLoop));
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    gmmTilingPtr->array[loop] = gmmTile.GetGmmQuantTilingData();

    // SharedMM切分
    auto status = gmmTile.SetSharedExpertInputParameters();
    if (status != ge::GRAPH_SUCCESS) {
        memset_s(&localTilingData_.sharedGmmTiling, sizeof(localTilingData_.sharedGmmTiling),
            0, sizeof(localTilingData_.sharedGmmTiling))
        return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    localTilingData_.sharedGmmTiling = gmmTile.GetGmmQuantTilingData();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QuantGroupedMatmulAllToAllvTiling::SetHcclTiling()
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    auto inputDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(C_INNER_DEBUG, "%s is Unsupported outputdata type!", Ops::Base::ToString(outputDataType).c_str()),
        return ge::GRAPH_FAILED);
    // OP_TILING_CHECK(
    //     // quantgmm， x1 dtype还是 alltoallv的inputdtpe吗??
    //     mc2tiling::HCCL_DATA_TYPE.find(inputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
    //     OP_LOGE(C_INNER_DEBUG, "%s is Unsupported inputdata type!", Ops::Base::ToString(inputDataType).c_str()),
    //     return ge::GRAPH_FAILED);   

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(groupEpPtr, alltoAllvCmd, alltoAllvConfig, 
                                         alltoAllvReduceType, alltoAllvDstDataType, alltoAllvSrcDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_->hcclA2avTiling.hcclInitTiling) != 0,
        OP_LOGE(C_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling hcclInitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_->hcclA2avTiling.a2avCcTiling) != 0,
        OP_LOGE(C_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling alltoAllvCcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    GE_ASSERT_GRAPH_SUCCESS(CheckOpInputInfo());
    // // 参数校验通过后赋值给全局上下文变量
    // GE_ASSERT_GRAPH_SUCCESS(InitTilingContextParameters());
    GE_ASSERT_GRAPH_SUCCESS(SetTilingCommonInfo());
    // 调用量化Matmul的tiling方法进行切分
    GE_ASSERT_GRAPH_SUCCESS(DoQuantGMMTiling());
    // hccl的tiling参数赋值处理
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    GE_ASSERT_GRAPH_SUCCESS(SetGmmA2avWorkspaceInfo());
    return ge::GRAPH_SUCCESS;
}

void QuantGroupedMatmulAllToAllvTiling::PrintQuantGmmA2avTilingData(QuantGmmA2avTilingData &outTilingData)
{
    return ;
    // PrintCommonTilingInfo(outTilingData.taskTilingInfo);
    // PrintSharedGmmTilingInfo(outTilingData.sharedGmmTiling);
    // PrintGmmQTilingDataInfo(outTilingData.gmmTiling);
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::PostTiling()
{
    QuantGmmA2avTilingData *outTilingData =
        context_->GetTilingData<QuantGmmA2avTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "MatmulAlltoAll postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(QuantGmmA2avTilingData),
            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(QuantGmmA2avTilingData));
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    PrintQuantGmmA2avTilingData(*outTilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t QuantGroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    // 先写死
    const uint64_t tilingKey = GET_TPL_TILING_KEY(0, 0, 1, 0);
    // OP_LOGD(opName_, "KCQUANTMODE,X2TRANSPOSE,DTYPEBIAS: [%d,%d,%d], TilingKey is [%lu].", KC_QUANT_MODE,
    //         x2TransposeFlag, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 构造函数，创建一个QuantGroupedMatmulAllToAllvTiling对象
 *
 * @param context
 */
QuantGroupedMatmulAllToAllvTiling::QuantGroupedMatmulAllToAllvTiling(gert::TilingContext *context) : GmmAlltoAllvTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(GroupedMatMulAlltoAllv, QuantGroupedMatmulAllToAllvTiling,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910_95), 1);
}
