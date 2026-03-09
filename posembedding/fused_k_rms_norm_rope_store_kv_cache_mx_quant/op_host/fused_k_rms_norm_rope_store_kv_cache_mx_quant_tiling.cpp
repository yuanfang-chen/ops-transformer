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
 * \file fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling.cpp
 * \brief
 */

#include "fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {

bool FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::IsCapable()
{
    return !isRegbase_;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::GetShapeAttrsInfoInner()
{
    OP_CHECK_IF(GetShapeAttrsInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfo failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::CalUbTiling()
{
    int64_t tns = seqLength_ * numHead_;
    int64_t maxUbFactor = 32;
    constexpr static int64_t needUbSize = static_cast<int64_t>(170) * static_cast<int64_t>(1024);
    if (static_cast<int64_t>(ubSize_) >= static_cast<int64_t>(needUbSize)) {
        ubFactor_ = maxUbFactor;
    } else {
        ubFactor_ = 1;
    }
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::DoOpTiling()
{
    OP_CHECK_IF(GetShapeAttrsInfoInner() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfoInner failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckQkvValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckQkvValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckCosSinValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckCosSinValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGammaValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckGammaValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKvSlotMappingValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckKvSlotMappingValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVScaleSlotMappingValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckVScaleSlotMappingValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckKCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKScaleCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckKScaleCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckVCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVScaleCacheValid() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CheckVScaleCacheValid failed."), return ge::GRAPH_FAILED);

    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_seqLength(seqLength_);
    tilingData_.set_numHeadHeads(numHead_);
    tilingData_.set_qkvDim(qkvDim_);
    tilingData_.set_ropeRange(ropeRange_);
    tilingData_.set_numHeadQ(numHeadQ_);
    tilingData_.set_numHeadK(numHeadK_);
    tilingData_.set_numHeadV(numHeadV_);
    tilingData_.set_blockNum(blockNum_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_reciprocal(reciprocal_);

    int64_t tns = seqLength_ * numHead_;
    blockFactor_ = (tns + coreNum_ - 1) / coreNum_;
    int64_t numBlocks = (tns + blockFactor_ - 1) / blockFactor_;
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockDim(blockFactor_);

    blockFactorQ_ = blockFactor_;
    blockFactorK_ = blockFactor_;
    blockFactorV_ = blockFactor_;
    tilingData_.set_blockFactorQ(blockFactorQ_);
    tilingData_.set_blockFactorK(blockFactorK_);
    tilingData_.set_blockFactorV(blockFactorV_);

    blockDimQ_ = (seqLength_ * numHeadQ_ + blockFactorQ_ - 1) / blockFactorQ_;
    blockDimK_ = (seqLength_ * numHeadK_ + blockFactorK_ - 1) / blockFactorK;
    blockDimV_ = (seqLength_ * numHeadV_ + blockFactorV_ - 1) / blockFactorV_;
    tilingData_.set_blockDimQ(blockDimQ_);
    tilingData_.set_blockDimK(blockDimK_);
    tilingData_.set_blockDimV(blockDimV_);

    CalUbTiling();
    tilingData_.set_ubFactor(ubFactor_);

    ubFactorQ_ = ubFactor_;
    ubFactorK_ = ubFactor_;
    ubFactorV_ = ubFactor_;
    tilingData_.set_ubFactorQ(ubFactorQ_);
    tilingData_.set_ubFactorK(ubFactorK_);
    tilingData_.set_ubFactorV(ubFactorV_);

    tilingKey_ = TEMPLATE_DS_PRIORITY;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_blockDim());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = 0;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "Name: %s", context_->GetNodeName());
    OP_LOGI(context_->GetNodeName(), "TilingKey: %lu", tilingKey_);
    OP_LOGI(context_->GetNodeName(), "batchSize: %ld", tilingData_.get_batchSize());
    OP_LOGI(context_->GetNodeName(), "seqLength: %ld", tilingData_.get_seqLength());
    OP_LOGI(context_->GetNodeName(), "numHeadHeads: %ld", tilingData_.get_numHeadHeads());
    OP_LOGI(context_->GetNodeName(), "qkvDim: %ld", tilingData_.get_qkvDim());
    OP_LOGI(context_->GetNodeName(), "ropeRange: %ld", tilingData_.get_ropeRange());
    OP_LOGI(context_->GetNodeName(), "numHeadQ: %ld", tilingData_.get_numHeadQ());
    OP_LOGI(context_->GetNodeName(), "numHeadK: %ld", tilingData_.get_numHeadK());
    OP_LOGI(context_->GetNodeName(), "numHeadV: %ld", tilingData_.get_numHeadV());
    OP_LOGI(context_->GetNodeName(), "blockNum: %ld", tilingData_.get_blockNum());
    OP_LOGI(context_->GetNodeName(), "blockSize: %ld", tilingData_.get_blockSize());
    OP_LOGI(context_->GetNodeName(), "epsilon: %f", tilingData_.get_epsilon());
    OP_LOGI(context_->GetNodeName(), "blockFactor: %ld", tilingData_.get_blockFactor());
    OP_LOGI(context_->GetNodeName(), "blockFactorQ: %d", tilingData_.get_blockFactorQ());
    OP_LOGI(context_->GetNodeName(), "blockFactorK: %ld", tilingData_.get_blockFactorK());
    OP_LOGI(context_->GetNodeName(), "blockFactorV: %ld", tilingData_.get_blockFactorV());
    OP_LOGI(context_->GetNodeName(), "blockDim: %ld", tilingData_.get_blockDim());
    OP_LOGI(context_->GetNodeName(), "blockDimQ: %ld", tilingData_.get_blockDimQ());
    OP_LOGI(context_->GetNodeName(), "blockDimK: %ld", tilingData_.get_blockDimK());
    OP_LOGI(context_->GetNodeName(), "blockDimV: %ld", tilingData_.get_blockDimV());
    OP_LOGI(context_->GetNodeName(), "ubFactor: %ld", tilingData_.get_ubFactor());
    OP_LOGI(context_->GetNodeName(), "ubFactorQ: %ld", tilingData_.get_ubFactorQ());
    OP_LOGI(context_->GetNodeName(), "ubFactorK: %ld", tilingData_.get_ubFactorK());
    OP_LOGI(context_->GetNodeName(), "ubFactorV: %ld", tilingData_.get_ubFactorV());
    OP_LOGI(context_->GetNodeName(), "reciprocal: %f", tilingData_.get_reciprocal());
}

REGISTER_OPS_TILING_TEMPLATE(FusedKRmsNormRopeStoreKvCacheMxQuant, FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling, TEMPLATE_DS_PRIORITY);
} // namespace optiling
