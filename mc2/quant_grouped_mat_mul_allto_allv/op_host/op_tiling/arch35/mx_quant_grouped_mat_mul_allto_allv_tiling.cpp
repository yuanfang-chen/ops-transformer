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
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace optiling::Mc2GroupedMatmul;

// namespace Mc2GroupedMatmul {

const std::vector<uint32_t> QUANT_GMM_X_DTYPE_LIST = {ge::DT_HIFLOAT8,};
const std::vector<uint32_t> QUANT_GMM_WEIGHT_DTYPE_LIST = {ge::DT_HIFLOAT8,};
const std::vector<uint32_t> QUANT_GMM_X_SCALE_DTYPE_LIST = {ge::DT_FLOAT,};
const std::vector<uint32_t> QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST = {ge::DT_FLOAT,};
const std::vector<uint32_t> QUANT_GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
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
    if (mode == QUANT_PAIR_MX) {
        OP_LOGI(opName_, "MxQuantGroupedMatmulAllToAllvTiling MX mode capable.");
        return true;
    }
    OP_LOGI(opName_, "Skip MxQuantGroupedMatmulAllToAllvTiling MX.");
    return false;
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
