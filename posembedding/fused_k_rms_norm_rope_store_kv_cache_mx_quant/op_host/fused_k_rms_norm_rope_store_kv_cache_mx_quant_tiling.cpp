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
    return true;
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


REGISTER_OPS_TILING_TEMPLATE(FusedKRmsNormRopeStoreKvCacheMxQuant, FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling, 1000);
} // namespace optiling
