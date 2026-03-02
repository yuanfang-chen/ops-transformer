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
 * \file quant_bmm_reduce_scatter_int8_int8.h
 * \brief
 */

#ifndef QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H
#define QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../common/inc/kernel/mc2_common_def.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#include "../../common/new_mc2_mm/kernel/mc2_quant_batch_matmul.h"
#include "matmul_reduce_scatter_v2_c_tiling.h"

#define TEMPLATE_CLASS_PARAMS template <typename AType, typename BType, typename CType, typename ScaleType, typename ptScaleType, \
                                        class MMClass, bool IsPerToken, bool ATrans, bool BTrans>
#define TEMPLATE_FUNC_PARAMS AType, BType, CType, ScaleType, ptScaleType, MMClass, IsPerToken, ATrans, BTrans

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;

TEMPLATE_CLASS_PARAMS
class QuantBMMReduceScatterInt8 {
public:
    __aicore__ inline QuantBMMReduceScatterInt8(){ }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
                                Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData* tilingData, 
                                __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling, TPipe* tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void MatMulComputReduceScatter(GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM,
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling, uint32_t tileCnt,
                                                     GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPerChannel(GM_ADDR aGM, GM_ADDR cGM,
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling, uint32_t tileCnt,
                                                     GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPerToken(GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM, 
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling, uint32_t tileCnt,
                                                     GM_ADDR gmToFloat, bool isTail);
    __aicore__ inline void DoX2Dequant();
    __aicore__ inline void PostProcess();  // 计算后处理，等待通信结束，并终止hcclserver

private:
    Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData* tilingData_;
    TPipe* tPipe_{nullptr};
    GM_ADDR aGM_{nullptr};
    GM_ADDR bGM_{nullptr};
    GM_ADDR cGM_{nullptr};
    GM_ADDR biasGM_{nullptr};
    GM_ADDR x1ScaleGM_{nullptr};
    GM_ADDR x2ScaleGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR gmToFloat_{nullptr};
    __gm__ HcclCombinOpParam* context_{nullptr};
    uint32_t rankId_{0};
    AscendC::HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
    uint8_t debugMode_{0};
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;        // CCU模式
    AscendC::HcclHandle handles_[MAX_HANDLE];  // 最大支持64个handleId
    uint64_t preCoreNum_ = 0;
};

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM, GM_ADDR cGM, GM_ADDR contextGM,
    GM_ADDR workspaceGM, Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData* tilingData, __gm__ void* mc2InitTiling, 
    __gm__ void* mc2CcTiling, TPipe* tPipe)
{
    tilingData_ = tilingData;
    hccl_.Init(contextGM, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);
    context_ = (__gm__ HcclCombinOpParam *)(contextGM);
    tPipe_ = tPipe;
    debugMode_ = tilingData_->debugMode;
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    x1ScaleGM_ = x1ScaleGM;
    x2ScaleGM_ = x2ScaleGM;
    rankId_ = context_->rankId;
    auto&& cfg = tilingData_->param;
    // 划分workspace
    gmToFloat_ = workspaceGM;
    workspaceGM_ = gmToFloat_ + cfg.cToFloatLen;
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::PostProcess()
{
    auto&& cfg = tilingData_->param;
    // 等待reducescatter执行完成
    if ((GetBlockIdx() == 0) && (g_coreType == AIC)) {
        for (uint32_t i = 0; i < cfg.tileCnt + cfg.tailCnt; i++) {
            hccl_.Wait(handles_[i]);
        }
        // 终止hcclserver
        hccl_.Finalize();
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::Process()
{
    InnerProcess();
    PostProcess();
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::InnerProcess()
{
    auto&& tiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
    auto&& cfg = tilingData_->param;
    uint64_t aSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka);
    uint64_t cSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
    // fullmesh算法
    // 计算主块
    MatMulComputReduceScatter(aGM_, cGM_, x1ScaleGM_, tilingData_->quantBmmV3TileTiling, cfg.tileCnt, gmToFloat_,
        (cfg.tailM ? false : true), false);
    // 尾块计算逻辑还需修改
    if (cfg.tailM) {
        if constexpr (!IsPerToken) {
            cSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
            auto aGMTail = aGM_;
            uint64_t tileOffset = static_cast<uint64_t>(cfg.tileCnt) * cSize * sizeof(CType);
            auto cGMTail = cGM_ + tileOffset;
            auto gmToFloatTail = gmToFloat_ + tileOffset;
            MatMulComputReduceScatter(aGMTail, cGMTail, nullptr, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                gmToFloatTail, true, true);
            return;
        }

        auto aGMTail = aGM_ + aSize * sizeof(AType) * static_cast<uint64_t>(cfg.tileCnt);
        auto cGMTail = cGM_ + cSize * sizeof(CType) * static_cast<uint64_t>(cfg.tileCnt);
        auto gmToFloatTail = gmToFloat_ +
            static_cast<uint64_t>(cfg.tileCnt) * static_cast<uint64_t>(cfg.rankDim) * cSize * sizeof(CType);
        auto x1ScaleTail = x1ScaleGM_ + static_cast<uint64_t>(tiling.M) * sizeof(ptScaleType) * static_cast<uint64_t>(cfg.tileCnt);
        MatMulComputReduceScatter(aGMTail, cGMTail, x1ScaleTail, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                                  gmToFloatTail, true, true);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void
QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPerChannel(
    GM_ADDR aGM, GM_ADDR cGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling, uint32_t tileCnt,
    GM_ADDR gmToFloat, bool isLast, bool isTail)
{
    if ASCEND_IS_AIV {
        return;
    }
    auto&& cfg = tilingData_->param;
    auto&& tiling = qBMmtiling.matmulTiling;
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        for (uint32_t i = 0; i < tileCnt; i++) {
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
            AscendC::CrossCoreWaitFlag(3);
        }
        return;
    }
    // 如果尾块的当前核数大于主块使用核数，计算尾块当前核的preCoreNum_
    if (isTail && (GetBlockIdx() >= tilingData_->quantBmmV3TileTiling.matmulTiling.usedCoreNum)) {
        auto&& tileTiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
        uint64_t headSliceM = (cfg.rankM / cfg.rankDim - cfg.tailM * cfg.tailCnt) / cfg.tileCnt;
        uint64_t mCnt = DequantBmm::CeilDiv(headSliceM, tileTiling.baseM) * cfg.rankDim;
        uint64_t nCnt = DequantBmm::CeilDiv(tileTiling.N, tileTiling.baseN);
        preCoreNum_ = (mCnt * nCnt * cfg.tileCnt) % tileTiling.usedCoreNum;
    }
    auto recvCount = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
    auto cOffset = recvCount * sizeof(CType);
    // 归一化Matmul计算类，负责MC2的Matmul计算
    auto cWork = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat;
    auto recvBuffer = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? gmToFloat : cGM;
    auto shift = isTail ? cfg.tileCnt : 0;
    uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    uint8_t repeat = 1;
    tPipe_->Reset();
    MMClass mmv3;
    auto tempGM = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat_;
    mmv3.Init(aGM_, bGM_, biasGM_, x2ScaleGM_, x1ScaleGM_, tempGM, workspaceGM_, &qBMmtiling, GetTPipePtr(),
        cfg, isTail, false, preCoreNum_);
    for (uint32_t i = 0; i < tileCnt; i++) {
        mmv3.UpdateSlice(i, isTail);
        mmv3.Process(isLast && (i == (tileCnt - 1)));
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);
        recvCount = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? 1 : recvCount;
        handles_[i + shift] = hccl_.ReduceScatter<true>(
            cWork, recvBuffer, recvCount, dataType_, HcclReduceOp::HCCL_REDUCE_SUM, stride, repeat);
        cWork += cOffset;
        recvBuffer += cOffset;
    }
    preCoreNum_ = mmv3.GetPreCoreNum();
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void
QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPerToken(
    GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling, uint32_t tileCnt,
    GM_ADDR gmToFloat, bool isTail)
{
    auto&& cfg = tilingData_->param;
    auto&& tiling = qBMmtiling.matmulTiling;
    auto recvCount = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
    auto aOffset = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka) * sizeof(AType);
    auto cOffset = recvCount * sizeof(CType);
    auto raOffset = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankK) / cfg.rankDim * sizeof(AType);
    // 归一化Matmul计算类，负责MC2的Matmul计算

    auto aAddr = aGM;
    auto x1ScaleAddr = x1ScaleGM;
    auto cWork = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat;
    auto recvBuffer = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? gmToFloat : cGM;
    auto shift = isTail ? cfg.tileCnt : 0;
    uint64_t stride = 0;
    uint8_t repeat = 1;
    uint32_t strideCount = cfg.rankM / cfg.rankDim;

    uint64_t x1ScaleOffset = static_cast<uint64_t>(tiling.M) * sizeof(ptScaleType);
    uint64_t rx1ScaleOffset = static_cast<uint64_t>(cfg.rankM) / cfg.rankDim * sizeof(ptScaleType);

    for (uint32_t i = 0; i < tileCnt; i++) {
        for (uint32_t j = 0; j < cfg.rankDim; j++) {
            //计算A,C矩阵的首地址
            auto aWorkerAddr = aAddr + static_cast<uint64_t>(j) * raOffset;
            auto cWorkAddr = cWork + static_cast<uint64_t>(j) * cOffset;
            auto x1ScaleWorkAddr = x1ScaleAddr + static_cast<uint64_t>(j) * rx1ScaleOffset;
            this->tPipe_->Destroy();
            this->tPipe_->Init();
            MMClass op;
            op.Init(
                aWorkerAddr, bGM_, x2ScaleGM, nullptr, biasGM_, x1ScaleWorkAddr, cWorkAddr, 
                workspaceGM_, &qBMmtiling, GetTPipePtr());
            op.Process();
            PipeBarrier<PIPE_V>();
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            recvCount = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? 1 : recvCount;
            handles_[i + shift] = hccl_.ReduceScatter<true>(
                cWork, recvBuffer, recvCount, dataType_, HcclReduceOp::HCCL_REDUCE_SUM, stride, repeat);
        }
        aAddr += aOffset;
        cWork += cOffset * cfg.rankDim;
        x1ScaleAddr += x1ScaleOffset;
        recvBuffer += cOffset;
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void
QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::DoX2Dequant() {}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void
QuantBMMReduceScatterInt8<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatter(
    GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling,
    uint32_t tileCnt, GM_ADDR gmToFloat, bool isLast, bool isTail)
{
    if constexpr (IsPerToken) {
        MatMulComputReduceScatterPerToken(aGM, cGM, x1ScaleGM, x2ScaleGM_, qBMmtiling, tileCnt, gmToFloat, isTail);
    } else {
        MatMulComputReduceScatterPerChannel(aGM, cGM, qBMmtiling, tileCnt, gmToFloat, isLast, isTail);
    }
}
}  // namespace MatmulReduceScatterV2Impl

#endif  // QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H