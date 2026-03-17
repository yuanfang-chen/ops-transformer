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
 * \file mx_quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "mx_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace optiling::Mc2GroupedMatmul;

// namespace Mc2GroupedMatmul {

const std::vector<uint32_t> MX_QUANT_GMM_X_DTYPE_LIST = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, };
const std::vector<uint32_t> MX_QUANT_GMM_WEIGHT_DTYPE_LIST = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, };
const std::vector<uint32_t> MX_QUANT_GMM_X_SCALE_DTYPE_LIST = {ge::DT_FLOAT8_E8M0, };
const std::vector<uint32_t> MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST = {ge::DT_FLOAT8_E8M0, };
const std::vector<uint32_t> MX_QUANT_GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16, };
const std::set<int64_t> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128, 256};
constexpr int64_t RANK_DEFAULT_NUM = -1;

static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

static ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, uint64_t dims, const char *shapeName,
    const char *opName_)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != dims),
        OP_LOGE(opName_, "The %s dimNum should be %lu, now is %lu.", shapeName, dims, dimNum), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool MxQuantGroupedMatmulAllToAllvTiling::IsCapable()
{
    QuantModePair mode = GetQuantMode(context_, opName_);
    OP_TILING_CHECK(mode == QUANT_PAIR_ERROR, OP_LOGE(opName_, "Fail to get attr quant mode."), return false);
    OP_LOGI(opName_, "mode is: %d", mode);
    if (mode == QUANT_PAIR_MX) {
        OP_LOGI(opName_, "MxQuantGroupedMatmulAllToAllvTiling MX mode capable.");
        return true;
    }
    OP_LOGI(opName_, "Skip MxQuantGroupedMatmulAllToAllvTiling MX.");
    return false;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsGmm()
{
    localParams_.gmmXDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    localParams_.gmmWeightDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
        OP_LOGE(opName_, "The Input gmmX Dtype should be in (DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, ), but gmmX is %s.",
        Ops::Base::ToString(localParams_.gmmXDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
        OP_LOGE(opName_, "The Input gmmWeight Dtype should be in (DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, ), but gmmWeight is %s.",
        Ops::Base::ToString(localParams_.gmmWeightDtype).c_str()), return ge::GRAPH_FAILED);
    localParams_.yDtype = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_Y_DTYPE_LIST, localParams_.yDtype),
        OP_LOGE(opName_, "The Output y Dtype should be in (DT_FLOAT16, DT_BF16, ), but y Dtype is %s.",
        Ops::Base::ToString(localParams_.yDtype).c_str()), return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;
    
    const gert::StorageShape* gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape* gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    const gert::StorageShape* yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE(opName_, "gmmXStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE(opName_, "gmmWeightStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(opName_, "yStorageShape is null!"),
        return ge::GRAPH_FAILED);
    auto status = CheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(yStorageShape, DIM_TWO, "yShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);

    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXDtype = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightDtype = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_DTYPE_LIST, localParams_.mmXDtype),
        OP_LOGE(opName_, "The Input mmX Dtype should be in (DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, ), but mmX is %s.",
        Ops::Base::ToString(localParams_.mmXDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
        OP_LOGE(opName_, "The Input mmWeight Dtype should be in (DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, ), but mmWeight is %s.",
        Ops::Base::ToString(localParams_.mmWeightDtype).c_str()), return ge::GRAPH_FAILED);
    localParams_.mmYDtype = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_Y_DTYPE_LIST, localParams_.mmYDtype),
        OP_LOGE(opName_, "The Output mmY Dtype should be in (DT_FLOAT16, DT_BF16, ), but mmY is %s.",
        Ops::Base::ToString(localParams_.mmYDtype).c_str()), return ge::GRAPH_FAILED);
    
    const gert::StorageShape* mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    const gert::StorageShape* mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    const gert::StorageShape* mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE(opName_, "mmXStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE(opName_, "mmWeightStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE(opName_, "mmYStorageShape is null!"),
        return ge::GRAPH_FAILED);
    auto status = CheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape", opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.Bs = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H2 = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.mmWeightDim0 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.mmWeightDim1 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);

    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(localParams_.Bs != mmYDim0,
        OP_LOGE(opName_, "mmX DIM0 %lu and mmY DIM0 %lu is not valid!", localParams_.Bs, mmYDim0),
        return ge::GRAPH_FAILED);

    localParams_.N2 = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsRelationGmm()
{
    localParams_.gmmXScaleDtype = context_->GetOptionalInputDesc(GMM_X_SCALE_OPTIONAL_INDEX)->GetDataType();
    localParams_.gmmWeightScaleDtype = context_->GetOptionalInputDesc(GMM_WEIGHT_SCALE_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.gmmXScaleDtype),
        OP_LOGE(opName_, "The Input gmmX Scale Dtype should be in (DT_FLOAT8_E8M0, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.gmmXScaleDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.gmmWeightScaleDtype),
        OP_LOGE(opName_, "The Input gmmWeight Scale Dtype should be in (DT_FLOAT8_E8M0, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.gmmWeightScaleDtype).c_str()), return ge::GRAPH_FAILED);

    const gert::StorageShape* gmmXScaleStorageShape = context_->GetOptionalInputShape(GMM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape* gmmWeightScaleStorageShape = context_->GetOptionalInputShape(GMM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(gmmXScaleStorageShape == nullptr, OP_LOGE(opName_, "gmmXScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightScaleStorageShape == nullptr, OP_LOGE(opName_, "gmmWeightScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(localParams_.gmmXQuantMode != QUANT_MX,
        OP_LOGE(opName_, "gmmXQuantMode just support tensor mode now, mode is %ld !", localParams_.gmmXQuantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmWeightQuantMode != QUANT_MX,
        OP_LOGE(opName_, "gmmWeightQuantMode just support tensor mode now, mode is %ld !", localParams_.gmmWeightQuantMode),
        return ge::GRAPH_FAILED);
    ge::graphStatus status = CheckShapeDimensions(gmmXScaleStorageShape, DIM_FOUR, "gmmXScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(gmmWeightScaleStorageShape, DIM_FOUR, "gmmWeightScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.gmmQuantSuit = QUANT_PAIR_TT;
    if (localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim2,
            OP_LOGE(opName_, "gmmX shape K %lu not match gmmWeight shape K %lu !", localParams_.H1, localParams_.gmmWeightDim2),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim1,
            OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", localParams_.N1, localParams_.gmmWeightDim1),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim1,
            OP_LOGE(opName_, "gmmX shape %lu not match gmmWeight shape %lu !", localParams_.H1, localParams_.gmmWeightDim1),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim2,
            OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", localParams_.N1, localParams_.gmmWeightDim2),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsRelationMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXScaleDtype = context_->GetOptionalInputDesc(MM_X_SCALE_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightScaleDtype = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_OPTIONAL_INDEX)->GetDataType();
    
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.mmXScaleDtype),
        OP_LOGE(opName_, "The Input mmX Scale Dtype should be in (DT_FLOAT8_E8M0, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.mmXScaleDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.mmWeightScaleDtype),
        OP_LOGE(opName_, "The Input mmWeight Scale Dtype should be in (DT_FLOAT8_E8M0, ), but Scale is %s.",
        Ops::Base::ToString(localParams_.mmWeightScaleDtype).c_str()), return ge::GRAPH_FAILED);

    const gert::StorageShape* mmXScaleStorageShape = context_->GetOptionalInputShape(MM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape* mmWeightScaleStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXScaleStorageShape == nullptr, OP_LOGE(opName_, "mmXScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightScaleStorageShape == nullptr, OP_LOGE(opName_, "mmWeightScaleStorageShape is null!"),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(localParams_.mmXQuantMode != QUANT_MX,
        OP_LOGE(opName_, "mmXQuantMode just support tensor mode now, mode is %ld !", localParams_.mmXQuantMode),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.mmWeightQuantMode != QUANT_MX,
        OP_LOGE(opName_, "mmWeightQuantMode just support tensor mode now, mode is %ld !", localParams_.mmWeightQuantMode),
        return ge::GRAPH_FAILED);
    ge::graphStatus status = CheckShapeDimensions(mmXScaleStorageShape, DIM_THREE, "mmXScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(mmWeightScaleStorageShape, DIM_THREE, "mmWeightScaleShape", opName_);
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.mmQuantSuit = QUANT_PAIR_MX;
    if (localParams_.isMmWeightTrans) {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim1,
            OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", localParams_.H2, localParams_.mmWeightDim1),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim0,
            OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", localParams_.N2, localParams_.mmWeightDim0),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim0,
            OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", localParams_.H2, localParams_.mmWeightDim0),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim1,
            OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", localParams_.N2, localParams_.mmWeightDim1),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE(opName_, "group is null."), return ge::GRAPH_FAILED);
    int64_t rankDim = 0;
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(opName_, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    if (*epWorldSizePtr == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankDim), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *epWorldSizePtr;
    }
    std::string supportRankSizeRange;
    for (const auto& v : SUPPORT_RANK_SIZE) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    OP_TILING_CHECK(SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
        OP_LOGE(opName_, "World_size should be %s, but the actual value is %ld.", supportRankSizeRange, rankDim),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (isEpNumMatch) {
        uint64_t expertNum = localParams_.ep * rankDim;
        OP_TILING_CHECK(expertNum > MAX_EXPERT_NUM,
            OP_LOGE(opName_, "expert Num lager than MAX_EXPERT_NUM, expertNum is %lu !", expertNum),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(opName_, "expert Per Rank is not match range, expertNum is %lu !", localParams_.ep);
        return ge::GRAPH_FAILED;
    }

    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_OPTIONAL_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE(opName_, "groupSizePtr is null !"), return ge::GRAPH_FAILED);
    localParams_.groupSize = *groupSizePtr;
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;

    OP_TILING_CHECK(((groupSizeM != MX_GROUP_SIZE_M) && (groupSizeM != 0)) ||
                    ((groupSizeN != MX_GROUP_SIZE_N) && (groupSizeN != 0)) ||
                    ((groupSizeK != MX_GROUP_SIZE_K) && (groupSizeK != 0)),
        CUBE_INNER_ERR_REPORT(
            opName_,
            "GroupSizeM, groupSizeN and groupSizeK should be 0 or 32 in mxfp scene,"
            " but actual is groupSizeM = %lu, groupSizeN = %lu, groupSizeK = %lu.",
            groupSizeM, groupSizeN, groupSizeK),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetInputOutputInfo()
{
    auto status = CheckOpInputSingleParamsTensor();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckParamsRelationAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // status = CheckMxQuantScaleShapes();
    // if (status != ge::GRAPH_SUCCESS) {
    //     return ge::GRAPH_FAILED;
    // }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::SetGmmA2avWorkspaceInfo()
{
    constexpr uint64_t alignAddrLen = 512;
    auto gmmYDtypeSize = mc2tiling::GetDataTypeSize(opName_, localParams_.gmmYDtype);
    inferredInfo_.gmmResultLen = mc2tiling::AlignUp(
        localParams_.A * localParams_.N1 * gmmYDtypeSize, alignAddrLen);
    localTilingData_.workspaceInfo.wsGmmOutputSize = inferredInfo_.gmmResultLen;
    localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize = 1 * 1024 * 1024;
    localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize = 1 * 1024 * 1024;
    workSpaceSize_ = libApiWorkSpaceSize_ + inferredInfo_.gmmResultLen +
        localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize +
        localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t MxQuantGroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
        localParams_.isMmWeightTrans, localParams_.gmmQuantSuit, localParams_.mmQuantSuit);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
        localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, localParams_.gmmQuantSuit,
        localParams_.mmQuantSuit, tilingKey);
    return tilingKey;
}

// 注册tiling类
REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatMulAlltoAllv, MxQuantGroupedMatmulAllToAllvTiling, 1);

// }

