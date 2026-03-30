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
#include <iostream>
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
    // 三个Phase通过pipe Reset隔离，各自独立使用UB
    // 每个Phase的ubFactor = ubSize_ / 单个factor所需的总UB字节数
    constexpr int64_t COS_SIN_NUM = 2;
    int64_t Nq = tilingData_.get_qNumHead();
    int64_t Nk = tilingData_.get_kNumHead();
    int64_t D = tilingData_.get_headDim();
    int64_t qHiddenSize = Nq * D;

    // ======================== Phase1: Q rope + mxquant ========================
    // inQueue_  (DB=2): qUbFactor * (COS_SIN_NUM + Nq) * D * sizeof(bf16)
    // outQueue_ (1):    qUbFactor * (D + INT8_BLOCK_ALIGN_NUM) * Nq * sizeof(int8)
    // wsBuffer0_:       qUbFactor * Nq * D * sizeof(bf16)
    // wsBuffer1_:       qUbFactor * Nq * D * sizeof(bf16)
    int64_t qPerInQue = (COS_SIN_NUM + Nq) * D * qkvDtypeSize_;
    int64_t qPerOutQue = (D + INT8_BLOCK_ALIGN_NUM) * Nq * INT8_BYTES;
    int64_t qPerWsp0 = qHiddenSize * qkvDtypeSize_;
    int64_t qPerWsp1 = qHiddenSize * qkvDtypeSize_;
    int64_t qTotalPerFactor = DOUBLE_BUFFER * qPerInQue + qPerOutQue + qPerWsp0 + qPerWsp1;
    int64_t qUbFactor = ubSize_ / qTotalPerFactor;
    OP_CHECK_IF((qUbFactor <= 0),
        OP_LOGE(context_->GetNodeName(), "qUbFactor <= 0, ubSize=%ld, need per factor=%ld.", ubSize_, qTotalPerFactor),
        return ge::GRAPH_FAILED);
    tilingData_.set_qUbFactor(qUbFactor);

    // ======================== Phase2: K rms_norm + rope + mxquant + scatter ========================
    // inQueue_      (DB=2): kUbFactor * (COS_SIN_NUM + Nk) * D * sizeof(bf16)
    // inQueueGamma_ (1):    D * sizeof(float)                          -- 固定开销
    // outQueue_     (1):    kUbFactor * (D + INT8_BLOCK_ALIGN_NUM) * Nk * sizeof(int8)
    // wsBuffer0_:           kUbFactor * Nq * D * sizeof(float)         -- kernel中使用qHiddenSize
    // wsBuffer1_:           kUbFactor * Nq * D * sizeof(bf16)          -- kernel中使用qHiddenSize
    int64_t kFixedGamma = D * FLOAT32_BYTES;
    int64_t kPerInQue = (COS_SIN_NUM + Nk) * D * qkvDtypeSize_;
    int64_t kPerOutQue = (D + INT8_BLOCK_ALIGN_NUM) * Nk * INT8_BYTES;
    int64_t kPerWsp0 = qHiddenSize * FLOAT32_BYTES;
    int64_t kPerWsp1 = qHiddenSize * qkvDtypeSize_;
    int64_t kTotalPerFactor = DOUBLE_BUFFER * kPerInQue + kPerOutQue + kPerWsp0 + kPerWsp1;
    int64_t kUbFactor = (ubSize_ - kFixedGamma) / kTotalPerFactor;
    OP_CHECK_IF((kUbFactor <= 0),
        OP_LOGE(context_->GetNodeName(), "kUbFactor <= 0, ubSize=%ld, fixed=%ld, need per factor=%ld.",
                ubSize_, kFixedGamma, kTotalPerFactor),
        return ge::GRAPH_FAILED);
    tilingData_.set_kUbFactor(kUbFactor);

    // ======================== Phase3: V mxquant + scatter ========================
    // vTUbFactor固定为 QUANT_BLOCK_SIZE * DIGIT_TWO = 64，在N轴切分计算vNumHeadUbFactor
    // inQueue_  (DB=2): vTUbFactor * vNumHeadUbFactor * D * sizeof(bf16)
    // outQueue_ (1):    (vTUbFactor + vTUbFactor / QUANT_BLOCK_SIZE) * vNumHeadUbFactor * D * sizeof(int8)
    // wsBuffer0_:       2 * (vTUbFactor / QUANT_BLOCK_SIZE) * vNumHeadUbFactor * D * sizeof(bf16)
    int64_t vTUbFactor = QUANT_BLOCK_SIZE * DIGIT_TWO;
    int64_t vTBlocks = vTUbFactor / QUANT_BLOCK_SIZE;  // = 2
    int64_t vPerInQue = vTUbFactor * D * qkvDtypeSize_;
    int64_t vPerOutQue = (vTUbFactor + vTBlocks) * D * INT8_BYTES;
    int64_t vPerWsp0 = DIGIT_TWO * vTBlocks * D * qkvDtypeSize_;
    int64_t vTotalPerFactor = DOUBLE_BUFFER * vPerInQue + vPerOutQue + vPerWsp0;
    int64_t vNumHeadUbFactor = ubSize_ / vTotalPerFactor;
    OP_CHECK_IF((vNumHeadUbFactor <= 0),
        OP_LOGE(context_->GetNodeName(), "vNumHeadUbFactor <= 0, ubSize=%ld, need per factor=%ld.",
                ubSize_, vTotalPerFactor),
        return ge::GRAPH_FAILED);
    tilingData_.set_vTUbFactor(vTUbFactor);
    tilingData_.set_vNumHeadUbFactor(vNumHeadUbFactor);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::DoOpTiling()
{
    constexpr int64_t T_ALIGN = 64;
    OP_CHECK_IF((seqLengthSum_ % T_ALIGN != 0),
        OP_LOGE(context_->GetNodeName(), "T (seqLengthSum) must be a multiple of %ld, got %ld.",
                T_ALIGN, seqLengthSum_),
        return ge::GRAPH_FAILED);

    // Currently only support D=128
    constexpr int64_t SUPPORTED_HEAD_DIM = 128;
    OP_CHECK_IF((headDim_ != SUPPORTED_HEAD_DIM),
        OP_LOGE(context_->GetNodeName(), "headDim only supports %ld, got %ld.",
                SUPPORTED_HEAD_DIM, headDim_),
        return ge::GRAPH_FAILED);

    // Currently only support Bs=512
    constexpr int64_t SUPPORTED_BLOCK_SIZE = 512;
    OP_CHECK_IF((blockSize_ != SUPPORTED_BLOCK_SIZE),
        OP_LOGE(context_->GetNodeName(), "blockSize only supports %ld, got %ld.",
                SUPPORTED_BLOCK_SIZE, blockSize_),
        return ge::GRAPH_FAILED);

    // Currently only support (Nq, Nk, Nv) in {(20, 2, 2), (80, 8, 8)}
    bool validHeadConfig = (numHeadQ_ == 20 && numHeadK_ == 2 && numHeadV_ == 2) ||
                           (numHeadQ_ == 80 && numHeadK_ == 8 && numHeadV_ == 8);
    OP_CHECK_IF(!validHeadConfig,
        OP_LOGE(context_->GetNodeName(),
                "(Nq, Nk, Nv) only supports {(20,2,2), (80,8,8)}, got (%ld,%ld,%ld).",
                numHeadQ_, numHeadK_, numHeadV_),
        return ge::GRAPH_FAILED);

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
    int64_t vUsedCoreNum = (vQuantGroup + vGropuBlockFactor - 1) / vGropuBlockFactor;
    tilingData_.set_vUsedCoreNum(vUsedCoreNum);
    tilingData_.set_vBlockFactor(vGropuBlockFactor * QUANT_BLOCK_SIZE * DIGIT_TWO);

    auto status = CalUbTiling();

    tilingKey_ = 0;

    return status;
}

void FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::PrintTilingData()
{
    std::cout << "========== FusedKRmsNormRopeStoreKvCacheMxQuant TilingData ==========" << std::endl;
    std::cout << "seqLengthSum:     " << tilingData_.get_seqLengthSum() << std::endl;
    std::cout << "qkvNumHead:       " << tilingData_.get_qkvNumHead() << std::endl;
    std::cout << "qNumHead:         " << tilingData_.get_qNumHead() << std::endl;
    std::cout << "kNumHead:         " << tilingData_.get_kNumHead() << std::endl;
    std::cout << "vNumHead:         " << tilingData_.get_vNumHead() << std::endl;
    std::cout << "headDim:          " << tilingData_.get_headDim() << std::endl;
    std::cout << "blockNum:         " << tilingData_.get_blockNum() << std::endl;
    std::cout << "blockSize:        " << tilingData_.get_blockSize() << std::endl;
    std::cout << "qUsedCoreNum:     " << tilingData_.get_qUsedCoreNum() << std::endl;
    std::cout << "qBlockFactor:     " << tilingData_.get_qBlockFactor() << std::endl;
    std::cout << "qUbFactor:        " << tilingData_.get_qUbFactor() << std::endl;
    std::cout << "kUsedCoreNum:     " << tilingData_.get_kUsedCoreNum() << std::endl;
    std::cout << "kBlockFactor:     " << tilingData_.get_kBlockFactor() << std::endl;
    std::cout << "kUbFactor:        " << tilingData_.get_kUbFactor() << std::endl;
    std::cout << "vUsedCoreNum:     " << tilingData_.get_vUsedCoreNum() << std::endl;
    std::cout << "vBlockFactor:     " << tilingData_.get_vBlockFactor() << std::endl;
    std::cout << "vTUbFactor:       " << tilingData_.get_vTUbFactor() << std::endl;
    std::cout << "vNumHeadUbFactor: " << tilingData_.get_vNumHeadUbFactor() << std::endl;
    std::cout << "epsilon:          " << tilingData_.get_epsilon() << std::endl;
    std::cout << "reciprocal:       " << tilingData_.get_reciprocal() << std::endl;
    std::cout << "=====================================================================" << std::endl;
}

ge::graphStatus FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(
        std::max({tilingData_.get_qUsedCoreNum(), tilingData_.get_kUsedCoreNum(), tilingData_.get_vUsedCoreNum()}));
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = 0;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    // PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(FusedKRmsNormRopeStoreKvCacheMxQuant, FusedKRmsNormRopeStoreKvCacheMxQuantRegbaseTiling,
                             1000);
} // namespace optiling
