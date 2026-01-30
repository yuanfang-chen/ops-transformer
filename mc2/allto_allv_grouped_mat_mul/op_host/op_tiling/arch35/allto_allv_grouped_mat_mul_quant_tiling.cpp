/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_grouped_mat_mul_tiling.cc
 * \brief
 */

#include <string>
#include <numeric>
#include <climits>
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "context_util.h"
#include "allto_allv_grouped_mat_mul_quant_tiling.h"
#include "../allto_allv_grouped_mat_mul_tiling_base.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
ge::graphStatus AlltoAllvGmmQuantTiling::GetContextAttr(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned nullptr!"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    auto permuteOutFlagPtr = attrs->GetAttrPointer<bool>(ATTR_PERMUTE_OUT_FLAG_INDEX);

    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "groupEpPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(A_INNER_DEBUG, "epWorldSizePtr is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "sendCountsPtr is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "recvCountsPtr is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "transGmmWeightPtr is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(transMmWeightPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "transMmWeightPtr is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(permuteOutFlagPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "permuteOutFlagPtr is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(A_INNER_DEBUG, "tilingData is null!"), return ge::GRAPH_FAILED);

    tilingData->taskTilingInfo.epWorldSize = *epWorldSizePtr;
    isGmmWeightTrans = *transGmmWeightPtr;
    isMmWeightTrans = *transMmWeightPtr;
    isPermuteOut = *permuteOutFlagPtr;

    const gert::StorageShape *mmXStorageShape = context->GetOptionalInputShape(MM_X_INDEX);
    const gert::StorageShape *mmWeightStorageShape = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
    const gert::StorageShape *outputMmYStorageShape = context->GetOutputShape(OUTPUT_MM_Y_INDEX);

    if (!((mmXStorageShape == nullptr) && (mmWeightStorageShape == nullptr) &&
        (outputMmYStorageShape == nullptr || outputMmYStorageShape->GetStorageShape().GetDimNum() == NUM_ZERO)) &&
        !((mmXStorageShape != nullptr) && (mmWeightStorageShape != nullptr) &&
        (outputMmYStorageShape != nullptr && outputMmYStorageShape->GetStorageShape().GetDimNum() != NUM_ZERO))) {
        OP_LOGE(A_INNER_DEBUG, "mmX, mmWeight and mmY should all be nullptr or all be not nullptr!");
        return ge::GRAPH_FAILED;
    }
    tilingData->taskTilingInfo.isNeedMM = (mmXStorageShape != nullptr);

    epGroup_ = groupEpPtr;
    epWorldSize_ = *epWorldSizePtr;

    OP_LOGI(A_INNER_DEBUG, "epGroup is %s, epWorldSize is %lu.", epGroup_, epWorldSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::GetShapeAndFormat(const gert::TilingContext *context)
{
    OP_TILING_CHECK((context->GetInputShape(GMM_X_INDEX) == nullptr) ||
        (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(A_INNER_DEBUG, "GetOutputShape gmmY returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context->GetInputDesc(GMM_X_INDEX) == nullptr,
        OP_LOGE(A_INNER_DEBUG, "GetInputDesc gmmX returned null."), return ge::GRAPH_FAILED);

    tilingData->taskTilingInfo.BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    tilingData->taskTilingInfo.H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    tilingData->taskTilingInfo.e = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(0);
    tilingData->taskTilingInfo.N1 = isGmmWeightTrans ?
        context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1) :
        context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO);

    tilingData->taskTilingInfo.A = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    mmDType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    mmDataTypeSize = GetSizeByDataType(mmDType_);

    maxM_ = tilingData->taskTilingInfo.A;
    maxK_ = tilingData->taskTilingInfo.H1;
    maxN_ = tilingData->taskTilingInfo.N1;
    if (tilingData->taskTilingInfo.isNeedMM) {
        tilingData->taskTilingInfo.BS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
        tilingData->taskTilingInfo.H2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
        tilingData->taskTilingInfo.N2 = isMmWeightTrans ?
            context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0) :
            context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
        maxMForMM_ = tilingData->taskTilingInfo.BS;
        maxKForMM_ = tilingData->taskTilingInfo.H2;
        maxNForMM_ = tilingData->taskTilingInfo.N2;
    } else {
        tilingData->taskTilingInfo.BS = 0U;
        tilingData->taskTilingInfo.H2 = 0U;
        tilingData->taskTilingInfo.N2 = 0U;
        maxMForMM_ = 0U;
        maxKForMM_ = 0U;
        maxNForMM_ = 0U;
    }
    checker.CheckerSetMNK(maxM_, maxN_, maxK_, maxMForMM_, maxKForMM_, maxNForMM_);
    OP_TILING_CHECK((context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX) != nullptr) ||
        (context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX) != nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsTensor and recvCountsTensor should all be null!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckMKN(const gert::TilingContext *context)
// {
//     return checker.CheckMKN(context);
// }

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckSendRecvDataVolumn(const gert::TilingContext *context) const
// {
//     // Data for communication between cards [2M,100M]
//     return checker.CheckSendRecvDataVolumn(context);
// }

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckShapeSize(const gert::TilingContext *context) const
// {
//     return checker.CheckShapeSize(context);
// }

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckAttrsShapeSize(const gert::TilingContext *context) const
// {
//     return checker.CheckAttrsShapeRelation(context);
// }

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckAttrsShapeRelation(const gert::TilingContext *context) const
// {
//     return checker.CheckAttrsShapeRelation(context);
// }

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckShapeRelation(const gert::TilingContext *context) const
// {
//     return checker.CheckShapeRelation(context);
// }

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckShapeDims(const gert::TilingContext *context)
// {
//     return checker.CheckShapeDims(context);
// }

ge::graphStatus AlltoAllvGmmQuantTiling::SetHcclTiling(const gert::TilingContext *context) const
{
    (void)context; // Unused
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(A_INNER_DEBUG, "Tiling Data is null!"), return ge::GRAPH_FAILED);

    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    auto DataType = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    // OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(DataType) == mc2tiling::HCCL_DATA_TYPE.end(),
    //     OP_LOGE(A_INNER_DEBUG, "%s is Unsupported outputdata type!", Ops::Base::ToString(DataType).c_str()),
    //     return ge::GRAPH_FAILED);

    auto alltoAllvDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(DataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(epGroup_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType,
        alltoAllvDataType, alltoAllvDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.hcclInitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.a2avCcTiling failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ge::graphStatus AlltoAllvGmmQuantTiling::CheckDType(const gert::TilingContext *context) const
// {
//     return checker.CheckDType(context);
// }

ge::graphStatus AlltoAllvGmmQuantTiling::Init(gert::TilingContext *context)
{
    tilingData = context->GetTilingData<QuantAlltoAllvGroupedMatmulTilingData>();
    OP_LOGI(A_INNER_DEBUG, "Init!!!!!!!!!!!");
    checker.tilingData = tilingData; // sharedptr
    OP_TILING_CHECK(GetContextAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Get context attr failed!"),
        return ge::GRAPH_FAILED);

    if (tilingData->taskTilingInfo.isNeedMM) {
        OP_TILING_CHECK(context->GetOptionalInputShape(MM_X_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOptionalInputShape of mm_x returns null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(context->GetOutputShape(OUTPUT_MM_Y_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOutputShape of mm_y returns null."), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(checker.CheckShapeDims(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check shape dim failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(checker.CheckQuantDType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check dtype failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(checker.CheckShapeRelation(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check shape relation failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetShapeAndFormat(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Get shape and format failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(checker.CheckShapeSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check shape size failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(checker.CheckAttrsShapeSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check Attrs shape size failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(checker.CheckAttrsShapeRelation(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check Attrs Shape Relation failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(checker.CheckSendRecvDataVolumn(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check Send Recv Data Volumn failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(checker.CheckMKN(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM CheckMKN failed."),
        return ge::GRAPH_FAILED);
    // TODO ADD checker check quant mode ...
    OP_LOGI(A_INNER_DEBUG,
        "AlltoAllvGmmQuantTiling: maxM_ is %d, maxK_ is %d, maxN_ is %d, maxMForMM_ is %d, maxKForMM_ is %d, "
        "maxNForMM_ "
        "is %d.",
        maxM_, maxK_, maxN_, maxMForMM_, maxKForMM_, maxNForMM_);
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmQuantTiling::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(A_INNER_DEBUG, "AlltoAllvGmmQuantTiling get tiling key %lu", tilingKey);
    return tilingKey;
}

uint64_t AlltoAllvGmmQuantTiling::GetTilingKey(const gert::TilingContext *context) const
{
    uint32_t templateMmDType = ADD_TPL_FP16;
    bool tilingkeyMm = false;
    bool tilingekyGmmTrans = false;
    bool tilingekyMmTrans = false;
    if (context->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_FLOAT16) {
        templateMmDType = ADD_TPL_FP16;
    } else if (context->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_BF16) {
        templateMmDType = ADD_TPL_BP16;
    } else {
        templateMmDType = ADD_TPL_HIF8;
    }
    if (tilingData->taskTilingInfo.isNeedMM) {
        tilingkeyMm = true;
    } else {
        tilingkeyMm = false;
    }
    if (isGmmWeightTrans) {
        tilingekyGmmTrans = true;
    } else {
        tilingekyGmmTrans = false;
    }
    if (isMmWeightTrans) {
        tilingekyMmTrans = true;
    } else {
        tilingekyMmTrans = false;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(templateMmDType, tilingkeyMm, tilingekyGmmTrans, tilingekyMmTrans);

    OP_LOGD(A_INNER_DEBUG, "end RunFusionKernelTiling, tilingKey is %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmQuantTiling::RunFusionKernelTiling(gert::TilingContext *context)
{
    OP_LOGD(A_INNER_DEBUG, "begin RunFusionKernelTiling.");

    OP_TILING_CHECK(SetHcclTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    static const uint32_t CORE_NUM = ascendcPlatform.GetCoreNumAiv();
    static const uint32_t AIC_NUM = ascendcPlatform.GetCoreNumAic();
    static const uint32_t AIV_NUM = ascendcPlatform.GetCoreNumAiv();
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    static const platform_ascendc::SocVersion SOC_VERSION = ascendcPlatform.GetSocVersion();

    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_TILING_CHECK((CORE_NUM == 0U || AIC_NUM == 0U || AIV_NUM == 0U),
        OP_LOGE(A_INNER_DEBUG, "platform[%d] info is invalid, coreNum=%u, aicNum=%u, aivNum=%u",
        static_cast<int>(SOC_VERSION), CORE_NUM, AIC_NUM, AIV_NUM),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((PLATFORM_SIZE.ubSize == 0U || PLATFORM_SIZE.l1Size == 0U || PLATFORM_SIZE.l0CSize == 0U ||
        PLATFORM_SIZE.l0ASize == 0U || PLATFORM_SIZE.l0BSize == 0U),
        OP_LOGE(A_INNER_DEBUG,
        "platform[%d] info is invalid, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
        static_cast<int>(SOC_VERSION), PLATFORM_SIZE.ubSize, PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize,
        PLATFORM_SIZE.l0ASize, PLATFORM_SIZE.l0BSize),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(DoAiCoreTiling(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "GMM_All_Reduce DoAiCoreTiling failed."), return ge::GRAPH_FAILED);

    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(CORE_NUM, AIC_NUM, AIV_NUM));

    size_t *workspaces = context->GetWorkspaceSizes(1); // 1: fixed value
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(A_INNER_DEBUG, "get workspace failed"), return ge::GRAPH_FAILED);

    uint64_t commOut = tilingData->taskTilingInfo.A * tilingData->taskTilingInfo.H1 * mmDataTypeSize;
    uint64_t permuteOut = isPermuteOut ?
        0 :
        (tilingData->taskTilingInfo.A * tilingData->taskTilingInfo.H1 * mmDataTypeSize);

    workspaces[0] = libApiWorkSpaceSize_ + commOut + permuteOut + 2 * 8;

    uint64_t tilingKey = GetTilingKey(context);
    context->SetTilingKey(tilingKey);

    OP_LOGD(A_INNER_DEBUG, "end RunFusionKernelTiling, tilingKey is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvGmmQuantTiling::PrintQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const
{
    const auto &mm = data.mmTilingData;
    const auto &quantParams = data.gmmQuantParams;
    const auto &gmmArray = data.gmmArray;
    std::stringstream ss;
    ss << "M=" << mm.M << " | N=" << mm.N << " | K=" << mm.Ka << " | usedCoreNum=" << mm.usedCoreNum << " | baseM=" <<
        mm.baseM << " | baseN=" << mm.baseN << " | baseK=" << mm.baseK << " | singleCoreM=" << mm.singleCoreM <<
        " | singleCoreN=" << mm.singleCoreN << " | singleCoreK=" << mm.singleCoreK << " | dbL0C=" << mm.dbL0C <<
        " | depthA1=" << mm.depthA1 << " | depthB1=" << mm.depthB1 << " | stepKa=" << mm.stepKa << " | stepKb=" <<
        mm.stepKb << " | stepM=" << mm.stepM << " | stepN=" << mm.stepN << " | iterateOrder=" << mm.iterateOrder;

    ss << " | Quant: groupNum=" << quantParams.groupNum << " | activeType=" << quantParams.activeType <<
        " | aQuantMode=" << quantParams.aQuantMode << " | bQuantMode=" << quantParams.bQuantMode << " | singleX=" <<
        quantParams.singleX << " | singleW=" << quantParams.singleW << " | singleY=" << quantParams.singleY <<
        " | groupType=" << quantParams.groupType << " | groupListType=" << quantParams.groupListType << " | hasBias=" <<
        quantParams.hasBias << " | reserved=" << quantParams.reserved;

    ss << " | Array: mList[0]=" << gmmArray.mList[0] << " | kList[0]=" << gmmArray.kList[0] << " | nList[0]=" <<
        gmmArray.nList[0];
    OP_LOGI("AlltoAllvGmmQuantTiling", "AlltoAllvGmmQuantTiling TilingParams = %s", ss.str().c_str());
}

ge::graphStatus AlltoAllvGmmQuantTiling::DoAiCoreTiling(const gert::TilingContext *context)
{
    OP_LOGD(A_INNER_DEBUG, "begin DoAiCoreTiling.");

    auto recvCountsPtr = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    const uint64_t *recvCounts = static_cast<const uint64_t *>(recvCountsPtr->GetData());
    for (uint64_t e = 0; e < 1; e++) {
        mSize_ = 0;
        for (uint64_t rank = 0; rank < tilingData->taskTilingInfo.epWorldSize; rank++) {
            mSize_ += recvCounts[rank * tilingData->taskTilingInfo.e + e];
        }

        auto &gmmQuantTilingData = tilingData->gmmQuantTilingData;

        SetGMMQuantParams(gmmQuantTilingData);
        SetGMMArray(gmmQuantTilingData);
        SetTilingParams(gmmQuantTilingData);
        PrintQuantTilingData(gmmQuantTilingData);
    }

    auto &mmQuantTilingData = tilingData->mmQuantTilingData;
    mSize_ = tilingData->taskTilingInfo.BS;
    SetGMMQuantParams(mmQuantTilingData);
    SetGMMArray(mmQuantTilingData);
    SetTilingParams(mmQuantTilingData);
    PrintQuantTilingData(mmQuantTilingData);

    OP_LOGD(A_INNER_DEBUG, "end DoAiCoreTiling.");
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvGmmQuantTiling::SetGMMQuantParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData)
{
    gmmQuantTilingData.gmmQuantParams.groupNum = SINGLE_GROUP_NUM;
    gmmQuantTilingData.gmmQuantParams.activeType = GMM_ACT_TYPE_NONE;
    gmmQuantTilingData.gmmQuantParams.aQuantMode = PERTENSOR_MODE;
    gmmQuantTilingData.gmmQuantParams.bQuantMode = PERTENSOR_MODE;
    gmmQuantTilingData.gmmQuantParams.singleX = 0;
    gmmQuantTilingData.gmmQuantParams.singleW = 0;
    gmmQuantTilingData.gmmQuantParams.singleY = 0;
    gmmQuantTilingData.gmmQuantParams.groupType = 0;
    gmmQuantTilingData.gmmQuantParams.groupListType = 1;
    gmmQuantTilingData.gmmQuantParams.hasBias = 0;
    gmmQuantTilingData.gmmQuantParams.reserved = 0;
}

void AlltoAllvGmmQuantTiling::SetGMMArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData)
{
    gmmQuantTilingData.gmmArray.mList[0] = static_cast<int32_t>(mSize_);
    gmmQuantTilingData.gmmArray.kList[0] = static_cast<int32_t>(tilingData->taskTilingInfo.H1);
    gmmQuantTilingData.gmmArray.nList[0] = static_cast<int32_t>(tilingData->taskTilingInfo.N1);
}

void AlltoAllvGmmQuantTiling::SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData)
{
    auto platformInfo = context_->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    auto &mm = gmmQuantTilingData.mmTilingData;

    mm.M = mSize_;
    mm.N = tilingData->taskTilingInfo.N1;
    mm.Ka = tilingData->taskTilingInfo.H1;
    mm.Kb = tilingData->taskTilingInfo.H1;
    mm.usedCoreNum = ascendcPlatform.GetCoreNumAic();
    mm.isBias = 0;
    mm.dbL0A = 2;
    mm.dbL0B = 2;

    mm.baseM = std::min(mSize_, static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseM = Ops::Base::CeilAlign(mm.baseM, static_cast<int32_t>(CUBE_BLOCK));

    mm.baseN =
        std::min(static_cast<int32_t>(tilingData->taskTilingInfo.N1), static_cast<int32_t>(BASIC_BLOCK_SIZE_256));
    mm.baseN = Ops::Base::CeilAlign(mm.baseN, static_cast<int32_t>(CUBE_BLOCK));

    mm.baseK =
        std::min(static_cast<int32_t>(tilingData->taskTilingInfo.H1), static_cast<int32_t>(BASIC_BLOCK_SIZE_128));
    mm.baseK = Ops::Base::CeilAlign(mm.baseK, static_cast<int32_t>(CUBE_REDUCE_BLOCK));

    mm.singleCoreM = std::min(mSize_, mm.baseM);
    mm.singleCoreN = std::min(static_cast<int32_t>(tilingData->taskTilingInfo.N1), mm.baseN);
    mm.singleCoreK = tilingData->taskTilingInfo.H1;

    uint64_t l0cRequired = mm.baseM * mm.baseN * DATA_SIZE_L0C * DB_SIZE;
    mm.dbL0C = (l0cRequired <= PLATFORM_SIZE.l0CSize) ? DB_SIZE : 1;

    mm.iterateOrder = 0U;

    uint64_t baseASize = mm.baseM * mm.baseK;
    uint64_t baseBSize = mm.baseN * mm.baseK;
    uint64_t baseL1Size = baseASize + baseBSize;

    uint64_t leftL1Size = PLATFORM_SIZE.l1Size;

    uint64_t depthInit = leftL1Size / baseL1Size;
    depthInit = std::max(depthInit, static_cast<uint64_t>(1));

    uint64_t depthScale = depthInit;
    while (depthScale * mm.baseK % BASIC_BLOCK_SIZE_512 != 0 && depthScale > 1) {
        depthScale--;
    }
    depthScale = std::max(depthScale, static_cast<uint64_t>(1));

    mm.depthA1 = depthScale;
    mm.depthB1 = depthScale;

    mm.stepKa = (mm.depthA1 > 1) ? (mm.depthA1 / DB_SIZE) : 1;
    mm.stepKb = (mm.depthB1 > 1) ? (mm.depthB1 / DB_SIZE) : 1;

    if (mm.stepKa * mm.baseK > mm.Ka) {
        mm.stepKa = Ops::Base::CeilDiv(mm.Ka, mm.baseK);
    }
    if (mm.stepKb * mm.baseK > mm.Kb) {
        mm.stepKb = Ops::Base::CeilDiv(mm.Kb, mm.baseK);
    }

    mm.depthA1 = mm.stepKa * DB_SIZE;
    mm.depthB1 = mm.stepKb * DB_SIZE;

    mm.stepM = 1;
    mm.stepN = 1;
}

static ge::graphStatus AlltoAllvGmmQuantTilingFunc(gert::TilingContext *context)
{
    AlltoAllvGmmQuantTiling *tiling = new AlltoAllvGmmQuantTiling(context);
    OP_TILING_CHECK(tiling->Init(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM tiling init failed."),
        return ge::GRAPH_FAILED);
    return tiling->RunFusionKernelTiling(context);
}

bool AlltoAllvGmmQuantTiling::IsCapable()
{
    if (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_HIFLOAT8) {
        return true;
    }
    return false;
}

ge::graphStatus AlltoAllvGmmQuantTiling::DoOpTiling()
{
    return AlltoAllvGmmQuantTilingFunc(context_);
}

ge::graphStatus AlltoAllvGmmQuantTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmQuantTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmQuantTiling, 1);
} // namespace optiling
