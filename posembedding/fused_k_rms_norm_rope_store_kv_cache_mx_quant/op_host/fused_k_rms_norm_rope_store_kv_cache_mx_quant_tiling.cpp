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

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::CalUbTiling()
{
    // q,k ub切分在T轴，v的ub切分在T轴和N轴
    // 待具体按照比例进行分配，当前粗略分配
    int64_t inQueSize = 48 * 1024; // 开DB
    int64_t wsp0Size = 80 * 1024;
    int64_t wsp1Size = 40 * 1024;
    int64_t outQueSize = 30 * 1024;
    // 计算qUbFactor
    int64_t qUbFactor = inQueSize / (2 * tilingData_.get_headDim() * qkvDtypeSize_ +
                                     tilingData_.get_qNumHead() * tilingData_.get_headDim() * qkvDtypeSize_);
    qUbFactor = std::min(qUbFactor, wsp0Size / tilingData_.get_qNumHead() * tilingData_.get_headDim() * FLOAT16_BYTES);
    qUbFactor = std::min(qUbFactor, wsp1Size / tilingData_.get_qNumHead() * tilingData_.get_headDim() * FLOAT16_BYTES);
    qUbFactor = std::min(qUbFactor, outQueSize / (tilingData_.get_qNumHead() * tilingData_.get_headDim() * INT8_BYTES +
                                                  tilingData_.get_qNumHead() * INT8_BLOCK_ALIGN_NUM * INT8_BYTES));
    OP_CHECK_IF((qUbFactor <= 0), OP_LOGI(context_->GetNodeName(), "qUbFactor <= 0"), return ge::GRAPH_FAILED);
    tilingData_.set_qUbFactor(qUbFactor);
    // 计算kUbFactor
    int64_t kUbFactor = inQueSize / (2 * tilingData_.get_headDim() * qkvDtypeSize_ +
                                     tilingData_.get_kNumHead() * tilingData_.get_headDim() * qkvDtypeSize_);
    kUbFactor = std::min(kUbFactor, wsp0Size / tilingData_.get_kNumHead() * tilingData_.get_headDim() * FLOAT32_BYTES);
    kUbFactor = std::min(kUbFactor, wsp1Size / tilingData_.get_kNumHead() * tilingData_.get_headDim() * FLOAT16_BYTES);
    kUbFactor = std::min(kUbFactor, outQueSize / (tilingData_.get_kNumHead() * tilingData_.get_headDim() * INT8_BYTES +
                                                  tilingData_.get_kNumHead() * INT8_BLOCK_ALIGN_NUM * INT8_BYTES));
    OP_CHECK_IF((kUbFactor <= 0), OP_LOGI(context_->GetNodeName(), "kUbFactor <= 0"), return ge::GRAPH_FAILED);
    tilingData_.set_kUbFactor(kUbFactor);
    // 计算vUbFactor
    int64_t vNumHeadUbFactor = inQueSize / (QUANT_BLOCK_SIZE * DIGIT_TWO * tilingData_.get_headDim() * qkvDtypeSize_);
    vNumHeadUbFactor = std::min(
        vNumHeadUbFactor, wsp0Size / (2 * DIGIT_TWO * tilingData_.get_headDim() * FLOAT16_BYTES)); // 2表示两块空间
    vNumHeadUbFactor = std::min(vNumHeadUbFactor,
                                outQueSize / (tilingData_.get_headDim() * DIGIT_TWO * INT8_BYTES +
                                              QUANT_BLOCK_SIZE * DIGIT_TWO * tilingData_.get_headDim() * INT8_BYTES));
    OP_CHECK_IF((vNumHeadUbFactor <= 0), OP_LOGI(context_->GetNodeName(), "vNumHeadUbFactor <= 0"),
                return ge::GRAPH_FAILED);
    tilingData_.set_vTUbFactor(QUANT_BLOCK_SIZE * DIGIT_TWO);
    tilingData_.set_vNumHeadUbFactor(vNumHeadUbFactor);
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::DoOpTiling()
{
    tilingData_.set_seqLengthSum(seqLengthSum_);
    tilingData_.set_qkvNumHead(numHead_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_qNumHead(numHeadQ_);
    tilingData_.set_kNumHead(numHeadK_);
    tilingData_.set_vNumHead(numHeadV_);
    tilingData_.set_blockNum(blockNum_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_reciprocal(reciprocal_);

    int64_t qkblockFactor = (seqLengthSum_ + coreNum_ - 1) / coreNum_;
    int64_t qkUsedCoreNum = (seqLengthSum_ + qkblockFactor - 1) / qkblockFactor;
    // q，k的多核切分策略相同
    tilingData_.set_qUsedCoreNum(qkUsedCoreNum);
    tilingData_.set_qBlockFactor(qkblockFactor);
    tilingData_.set_kUsedCoreNum(qkUsedCoreNum);
    tilingData_.set_kBlockFactor(qkblockFactor);
    // seqLengthSum_保证是64的整数倍
    int64_t vQuantGroup = seqLengthSum_ / QUANT_BLOCK_SIZE / DIGIT_TWO;
    int64_t vGropuBlockFactor = (vQuantGroup + coreNum_ - 1) / coreNum_;
    int64_t vUsedCoreNum = (seqLengthSum_ + vGropuBlockFactor - 1) / vGropuBlockFactor;
    tilingData_.set_vUsedCoreNum(vUsedCoreNum);
    tilingData_.set_vBlockFactor(vGropuBlockFactor * QUANT_BLOCK_SIZE * DIGIT_TWO);

    auto status = CalUbTiling();

    tilingKey_ = 0;

    return status;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_blockDim());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = 0;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(FusedKRmsNormRopeStoreKvCacheMxQuant, FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling,
                             1000);
} // namespace optiling
