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
 * \file quant_bmm_reduce_scatter_fp8_hif8.h
 * \brief
 */

#ifndef QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H
#define QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../common/inc/kernel/mc2_common_def.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_perblock.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#include "../../common/new_mc2_mm/kernel/mc2_quant_batch_matmul.h"
#include "../../common/inc/kernel/qbmm_mix_perblock_noncontiguous.h"
#include "matmul_reduce_scatter_v2_c_tiling.h"
#include "../../common/inc/kernel/reduce_sum.h"

#define TEMPLATE_CLASS_PARAMS template <typename AType, typename BType, typename CType, typename ScaleType, \
                                        class MMClass, bool IsPerBlock, bool ATrans, bool BTrans>
#define TEMPLATE_FUNC_PARAMS AType, BType, CType, ScaleType, MMClass, IsPerBlock, ATrans, BTrans

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;
using namespace AiVReduceSumImpl;

TEMPLATE_CLASS_PARAMS
class QuantBMMReduceScatter {
public:
    __aicore__ inline QuantBMMReduceScatter(){ }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
                                Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData* tilingData, 
                                __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling, TPipe* tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void MatMulReduceScatterSerial();
    __aicore__ inline void MatMulComputReduceScatter(GM_ADDR aGM, GM_ADDR recvGM, GM_ADDR x1ScaleGM,
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling, uint32_t tileCnt,
                                                     GM_ADDR sendGM, bool isLast, bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPertensor(GM_ADDR aGM, GM_ADDR recvGM, 
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBmmTiling, uint32_t tileCnt,
                                                     GM_ADDR sendGM, bool isLast, bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPerblock(GM_ADDR aGM, GM_ADDR recvGM, GM_ADDR x1ScaleGM, 
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBmmTiling, uint32_t tileCnt,
                                                     GM_ADDR sendGM, bool isTail);
    __aicore__ inline void PostProcess();  // 计算后处理，等待通信结束，并终止hcclserver

private:
    ReduceSumForAlltoAll<CType> reduceSum_; // AIV ReduceSum 相关实现

    Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData* tilingData_;
    TPipe* tPipe_{nullptr};
    GM_ADDR aGM_{nullptr};
    GM_ADDR bGM_{nullptr};
    GM_ADDR cGM_{nullptr};
    GM_ADDR biasGM_{nullptr};
    GM_ADDR x1ScaleGM_{nullptr};
    GM_ADDR x2ScaleGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR sendBuf_{nullptr};    // 存放 MatMul 输出（All2All send buffer）
    GM_ADDR recvBuf_{nullptr};    // 存放 All2All 接收的 slices（内容为 [slice_r_from_rank0][slice_r_from_rank1]...[slice_r_from_rankR-1]）
    __gm__ HcclCombinOpParam* context_{nullptr};
    uint32_t rankId_{0};
    AscendC::HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
    uint8_t debugMode_{0};
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;        // CCU模式
    AscendC::HcclHandle handles_[MAX_HANDLE];  // 最大支持64个handleId
    uint64_t preCoreNum_ = 0;
    uint32_t batchWeight_[MAX_HANDLE] = {0};
};

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::Init(
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
    for (uint32_t j = 0; j < cfg.rankDim; j++) {
        batchWeight_[j] = j;
    }
    // all2all 通信相关参数, 划分workspace
    uint64_t fullMN = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankN);  // M * N
    sendBuf_ = workspaceGM; // [0, fullMN)
    recvBuf_ = sendBuf_ + cfg.cToFloatLen; // [fullMN, 2*fullMN)
    workspaceGM_ = recvBuf_ + cfg.cToFloatLen; // [2*fullMN, 3*fullMN) 

    // === AIV ReudceSum 相关参数计算与初始化 ===
    auto&& tiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
    uint64_t aivNum = tiling.usedCoreNum * GetTaskRation(); // 启用的AIV 数量
    reduceSum_.Init(fullMN, cfg.rankDim, aivNum, recvBuf_, cGM_, tPipe_);
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::PostProcess()
{
    auto&& cfg = tilingData_->param;
    // 等待reducescatter执行完成
    if ((GetBlockIdx() == 0) && (g_coreType == AIC)) {
        for (uint32_t i = 0; i < cfg.tileCnt + cfg.tailCnt; i++) {
            hccl_.Wait(handles_[i]); // 等待所有通信完成
        }
        // 终止hcclserver
        hccl_.Finalize();
    }
    SyncAll<false>(); // 全核同步, 等待mm和hccl通信结束

    // Vector操作，reduce sum
    if ASCEND_IS_AIV {
        reduceSum_.ExecuteReduceSum(); // AIV执行 reduce_sum
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::Process()
{
    InnerProcess(); // 核心计算+通信
    PostProcess(); // 等待通信完成 + ReduceSum
}

// perblock量化场景当rankM不满足128*ranksize，走计算通信串行
TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulReduceScatterSerial()
{
    auto&& qBMmtiling = tilingData_->quantBmmV3TileTiling;
    auto&& tiling = qBMmtiling.matmulTiling;
    auto&& cfg = tilingData_->param;
    
    auto matmulOutSendBuf = sendBuf_; // 既作为 MatMul 的输出地址，也作为 AlltoAll 的发送缓冲地址
    auto recvBuffer = recvBuf_; // AlltoAll 的接收缓冲地址

    uint64_t recvCount = static_cast<uint64_t>(tiling.M / cfg.rankDim) * static_cast<uint64_t>(tiling.N);
    this->tPipe_->Destroy();
    this->tPipe_->Init();
    MMClass op;
    uint32_t strideCount = cfg.rankM / cfg.rankDim;
    op.Init(aGM_, bGM_, biasGM_, x2ScaleGM_, x1ScaleGM_, matmulOutSendBuf, workspaceGM_, &qBMmtiling, tPipe_,
            batchWeight_, strideCount, false);
    op.Process();
    SyncAll<false>();
    uint64_t stride = 0;
    uint8_t repeat = 1;
    if ASCEND_IS_AIC {
        recvCount = recvCount;
        handles_[0] = hccl_.AlltoAll<true>(
            matmulOutSendBuf, recvBuffer, recvCount, dataType_, stride, repeat);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::InnerProcess()
{
    auto&& tiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
    auto&& cfg = tilingData_->param;
    uint64_t aSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka);
    uint64_t cSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);

    // perblock场景下，未进行公式化切分，通算串行
    if constexpr (IsPerBlock) {
        if (tiling.M == cfg.rankM) {
            MatMulReduceScatterSerial();
            return;
        }
    }

    // fullmesh算法
    // 计算主块
    MatMulComputReduceScatter(aGM_, recvBuf_, x1ScaleGM_, tilingData_->quantBmmV3TileTiling, cfg.tileCnt, sendBuf_,
        (cfg.tailM ? false : true), false);
    // 计算尾块
    if (cfg.tailM) {
        if constexpr (!IsPerBlock) {
            cSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
            auto aGMTail = aGM_;
            uint64_t tileOffset = static_cast<uint64_t>(cfg.tileCnt) * cSize * sizeof(CType);
            auto recvGMTail = recvBuf_ + tileOffset;
            auto sendGMTail = sendBuf_ + tileOffset;
            MatMulComputReduceScatter(aGMTail, recvGMTail, nullptr, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                sendGMTail, true, true);
            return;
        }

        auto aGMTail = aGM_ + aSize * sizeof(AType) * static_cast<uint64_t>(cfg.tileCnt);
        auto recvGMTail = recvBuf_ + cSize * sizeof(CType) * static_cast<uint64_t>(cfg.tileCnt);
        auto sendGMTail = sendBuf_ +
            static_cast<uint64_t>(cfg.tileCnt) * static_cast<uint64_t>(cfg.rankDim) * cSize * sizeof(CType);
        auto x1ScaleTail = x1ScaleGM_ + static_cast<uint64_t>(tiling.M) / BLOCK_SIZE *
                                            CeilDiv(static_cast<uint64_t>(tiling.Ka), BLOCK_SIZE) * sizeof(ScaleType) *
                                            static_cast<uint64_t>(cfg.tileCnt);
        MatMulComputReduceScatter(aGMTail, recvGMTail, x1ScaleTail, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                                  sendGMTail, true, true);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void
QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPertensor(
    GM_ADDR aGM, 
    GM_ADDR recvGM, 
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBmmTiling, 
    uint32_t tileCnt,
    GM_ADDR sendGM, 
    bool isLast, 
    bool isTail)
{
    if ASCEND_IS_AIV {
        return;
    }

    // 获取配置与 Tiling 数据
    auto&& cfg = tilingData_->param;
    auto&& tiling = qBmmTiling.matmulTiling;

    // 空闲核处理：仅执行核间同步，不执行计算
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        for (uint32_t i = 0; i < tileCnt; i++) {
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
            AscendC::CrossCoreWaitFlag(3);
        }
        return;
    }

    // 尾块特殊逻辑：计算 preCoreNum_
    // 如果尾块的当前核索引大于主块使用的核数，需要重新计算 preCoreNum_
    if (isTail && (GetBlockIdx() >= tilingData_->quantBmmV3TileTiling.matmulTiling.usedCoreNum)) {
        auto&& tileTiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
        uint64_t headSliceM = (cfg.rankM / cfg.rankDim - cfg.tailM * cfg.tailCnt) / cfg.tileCnt;
        uint64_t mCnt = DequantBmm::CeilDiv(headSliceM, tileTiling.baseM) * cfg.rankDim;
        uint64_t nCnt = DequantBmm::CeilDiv(tileTiling.N, tileTiling.baseN);
        preCoreNum_ = (mCnt * nCnt * cfg.tileCnt) % tileTiling.usedCoreNum;
    }

    // 预计算通信常量
    // 单个 Rank 接收的元素数量 (M * N / rankDim)
    const uint64_t rankSliceElems = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
    // 单个 Rank 接收的字节偏移
    const uint64_t rankSliceBytes = rankSliceElems * sizeof(CType);
    // All2All 步长 (元素数): (rankM / rankDim) * rankN
    const uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    const uint8_t repeat = 1;
    const uint32_t handleShift = isTail ? cfg.tileCnt : 0;

    // 初始化指针
    GM_ADDR currSendPtr = sendGM;      // MatMul 输出缓冲区, All2All 发送缓冲区
    GM_ADDR currRecvPtr = recvGM;      // All2All 接收缓冲区

    // 初始化 MatMul 计算对象
    tPipe_->Reset();
    Mc2MatmulV3::Mc2QuantBatchMatmulASWKernel<AType, BType, ScaleType, float, CType, CubeFormat::ND, CubeFormat::ND,
        CubeFormat::ND, ATrans, BTrans> mmv3;
    mmv3.Init(aGM_, bGM_, biasGM_, x2ScaleGM_, x1ScaleGM_, sendBuf_, workspaceGM_, 
              &qBmmTiling, GetTPipePtr(), cfg, isTail, false, preCoreNum_);

    for (uint32_t i = 0; i < tileCnt; i++) {
        // 更新 Slice 并执行 MatMul
        mmv3.UpdateSlice(i, isTail);
        mmv3.Process(isLast && (i == (tileCnt - 1)));

        // 确保 MatMul 计算完成后再启动通信
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);

        // All2All 通信
        handles_[i + handleShift] = hccl_.AlltoAll<true>(
            currSendPtr,    // 发送缓冲区 
            currRecvPtr,    // 接收缓冲区
            rankSliceElems, // 元素数量
            dataType_,      // 数据类型
            stride,         // 步长
            repeat          // 重复次数
        );

        //  更新指针至下一个 Tile
        currSendPtr += rankSliceBytes;
        currRecvPtr += rankSliceBytes;
    }

    // 更新 preCoreNum_
    preCoreNum_ = mmv3.GetPreCoreNum();
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void
QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPerblock(
    GM_ADDR aGM, 
    GM_ADDR recvGM, 
    GM_ADDR x1ScaleGM, 
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBmmTiling, 
    uint32_t tileCnt,
    GM_ADDR sendGM, 
    bool isTail)
{
    // 获取配置与 Tiling 数据
    auto&& cfg = tilingData_->param;
    auto&& tiling = qBmmTiling.matmulTiling;

    // 单次通信的元素数量 (M * N)
    const uint64_t recvCount = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
    // A 矩阵单个 Tile 的字节偏移 (M * Ka * sizeof(AType))
    const uint64_t aTileBytes = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka) * sizeof(AType);
    // C 矩阵单个 Tile 的字节偏移 (M * N * sizeof(CType))
    const uint64_t cTileBytes = recvCount * sizeof(CType);
    // Rank 分片相关的 A 矩阵偏移 (保持原逻辑: rankM * rankK / rankDim * sizeof)
    const uint64_t rankABytes = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankK) / cfg.rankDim * sizeof(AType);
    
    // Scale 矩阵单个 Tile 的字节偏移
    const uint64_t scaleBlockM = tiling.M / BLOCK_SIZE;
    const uint64_t scaleBlockK = CeilDiv(static_cast<uint64_t>(tiling.Ka), BLOCK_SIZE);
    const uint64_t scaleTileBytes = scaleBlockM * scaleBlockK * sizeof(ScaleType);

    // 初始化指针 
    GM_ADDR currAPtr = aGM;
    GM_ADDR currScalePtr = x1ScaleGM;
    GM_ADDR currSendPtr = sendGM;   // MatMul 输出缓冲区, All2All 发送缓冲区
    GM_ADDR currRecvPtr = recvGM;   // All2All 接收缓冲区

    // 通信参数配置
    const uint32_t handleShift = isTail ? cfg.tileCnt : 0;
    const uint64_t stride = 0;
    const uint8_t repeat = 1;       // 通信重复次数
    const uint32_t strideCount = cfg.rankM / cfg.rankDim; // 传递给 MMClass 的步长参数

    for (uint32_t i = 0; i < tileCnt; i++) {
        this->tPipe_->Destroy();
        this->tPipe_->Init();

        // 执行 MatMul 计算
        MMClass op;
        op.Init(currAPtr, bGM_, biasGM_, x2ScaleGM_, currScalePtr, currSendPtr, workspaceGM_, &qBmmTiling, tPipe_, batchWeight_, strideCount, false);
        op.Process();

        // 同步保证 MatMul 计算完成
        PipeBarrier<PIPE_V>();
        SyncAll<false>();

        // All2All 通信
        if ASCEND_IS_AIC {
            handles_[i + handleShift] = hccl_.AlltoAll<true>(
                currSendPtr,    // 发送缓冲区
                currRecvPtr,    // 接收缓冲区
                recvCount,      // 元素数量
                dataType_,      // 数据类型
                stride,         // 步长
                repeat          // 重复次数
            );
        }

        // A 矩阵指针偏移
        currAPtr += aTileBytes;
        
        // Send 缓冲区指针偏移
        currSendPtr += cfg.rankDim * cTileBytes;
        
        // Scale 矩阵指针偏移
        currScalePtr += scaleTileBytes;
        
        // Recv 缓冲区指针偏移
        currRecvPtr += cTileBytes;
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void
QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatter(
    GM_ADDR aGM, GM_ADDR recvGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qBMmtiling,
    uint32_t tileCnt, GM_ADDR sendGM, bool isLast, bool isTail)
{
    if constexpr (IsPerBlock) {
        MatMulComputReduceScatterPerblock(aGM, recvGM, x1ScaleGM, qBMmtiling, tileCnt, sendGM, isTail);
    } else {
        MatMulComputReduceScatterPertensor(aGM, recvGM, qBMmtiling, tileCnt, sendGM, isLast, isTail);
    }
}
}  // namespace MatmulReduceScatterV2Impl

#endif  // QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H