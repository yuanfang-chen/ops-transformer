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
 * \file matmul_reduce_scatter_fp16_bf16.h
 * \brief
 */

#ifndef MATMUL_REDUCE_SCATTER_FP16_BF16_H
#define MATMUL_REDUCE_SCATTER_FP16_BF16_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../common/inc/kernel/mc2_common_def.h"
#include "../../common/new_mc2_mm/kernel/mc2_mat_mul_asw_kernel.h"
#include "../../common/new_mc2_mm/kernel/mc2_mat_mul_asw_block.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"
#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "matmul_reduce_scatter_v2_c_tiling.h"
#include "../../common/inc/kernel/reduce_sum.h"

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;
using namespace AiVReduceSumImpl;

template <typename AType, typename BType, typename BiasType, typename CType>
class MatmulReduceScatterFP16BF16 {
public:
    __aicore__ inline MatmulReduceScatterFP16BF16() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM,
                                GM_ADDR workspaceGM, Mc2Tiling::MatmulReduceScatterV2TilingData* tilingData, 
                                __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling, TPipe* tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void Compute(GM_ADDR cGM, Mc2MatMulV3TilingData& tiling, uint32_t count,
                                   GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void MatMulV3Compute(GM_ADDR cGM, Mc2MatMulV3TilingData& tiling, uint32_t count,
                                           GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void PostProcess();    // 计算后处理，等待通信结束，并终止hcclserver, 尾调用 ReduceSum

private:
    ReduceSumForAlltoAll<C_DTYPE> reduceSum_; // AIV ReduceSum 相关实现

    Mc2Tiling::MatmulReduceScatterV2TilingData* tilingData_;
    TPipe* tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    __gm__ HcclCombinOpParam* context_;
    uint32_t rankId_;
    AscendC::HcclDataType dataType_;
    uint8_t debugMode_;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;              // CCU模式
    AscendC::HcclHandle handles_[MAX_HANDLE];        // 最大支持64个handleId
    GM_ADDR sendBuf_;    // 存放 MatMul 输出（All2All send buffer）
    GM_ADDR recvBuf_;    // 存放 All2All 接收的 slices（内容为 [slice_r_from_rank0][slice_r_from_rank1]...[slice_r_from_rankR-1]）
};

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
    Mc2Tiling::MatmulReduceScatterV2TilingData* tilingData, __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling, TPipe* tPipe) {
    
    tilingData_ = tilingData;
    auto&& cfg = tilingData_->param;

    // 初始化 HCCL
    hccl_.Init(contextGM, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

    // 读取上下文和配置
    context_ = (__gm__ HcclCombinOpParam *)(contextGM);
    tPipe_ = tPipe;
    tPipe_->Reset();
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    debugMode_ = tilingData_->debugMode;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    rankId_ = context_->rankId;

    // all2all 通信相关参数, 划分workspace
    uint64_t fullMN = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankN);  // M * N
    sendBuf_ = workspaceGM;                                      // [0, fullMN)
    recvBuf_ = sendBuf_ + fullMN * sizeof(C_DTYPE);            // [fullMN, 2*fullMN)

    // === AIV ReudceSum 相关参数计算与初始化 ===
    auto&& tiling = tilingData_->mC2Mmv3TileTilingData.tCubeTiling;
    uint64_t aivNum = tiling.usedCoreNum * GetTaskRation(); // 启用的AIV 数量
    reduceSum_.Init(fullMN, cfg.rankDim, aivNum, recvBuf_, cGM_, tPipe_);
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::PostProcess()
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

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::Process()
{
    InnerProcess(); // 核心计算+通信
    PostProcess(); // 等待通信完成 + ReduceSum
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::InnerProcess()
{
    auto&& tiling = tilingData_->mC2Mmv3TileTilingData.tCubeTiling;
    auto&& cfg = tilingData_->param;

    // fullmesh算法
    // 计算主块（整除部分）
    Compute(recvBuf_, tilingData_->mC2Mmv3TileTilingData, cfg.tileCnt, sendBuf_, cfg.tailM ? false : true, false);
    // 计算尾块（非整除部分）
    if (cfg.tailM) {
        uint64_t tileSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
        uint64_t tileOffset = tileSize * static_cast<uint64_t>(cfg.tileCnt) * sizeof(C_DTYPE);
        auto recvGMTail = recvBuf_ + tileOffset;
        auto sendBufTail = sendBuf_ + tileOffset;
        Compute(recvGMTail, tilingData_->mC2Mmv3TailTilingData, cfg.tailCnt, sendBufTail, true, true);
    }
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::Compute(
    GM_ADDR recvGMAddr,
    Mc2MatMulV3TilingData& tiling,
    uint32_t count,
    GM_ADDR sendGMAddr,
    bool isLast,
    bool isTail)
{
    if ASCEND_IS_AIV {
        return;
    }

    if (block_idx >= tiling.tCubeTiling.usedCoreNum) {
        // 非活跃核：仅同步，不参与计算
        for (uint32_t i = 0; i < count; i++) {
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
            AscendC::CrossCoreWaitFlag(3);
        }
        return;
    }

    // Cube 核执行：MatMul + All2All
    MatMulV3Compute(recvGMAddr, tiling, count, sendGMAddr, isLast, isTail);
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::MatMulV3Compute(
    GM_ADDR recvGMAddr,
    Mc2MatMulV3TilingData& tiling, 
    uint32_t count,
    GM_ADDR sendGMAddr,
    bool isLast,
    bool isTail)
{
    auto&& cfg = tilingData_->param;
    cfg.rankID = rankId_;

    MC2MatmulV3::MC2MatmulAswKernelDerive<AType, BType, CType, BiasType, MC2MatmulV3::MC2MatmulAswBlockDerive> mmv3;
    
    // MatMul 结果先写入 send buffer（即 sendBuf_）
    auto tempGM = sendBuf_;
    mmv3.Init(aGM_, bGM_, tempGM, biasGM_, nullptr, nullptr, &tiling, GetTPipePtr(), cfg, isTail, false);

    // 每个 rank 应得的 M 维度大小
    uint64_t sliceM = static_cast<uint64_t>(tiling.tCubeTiling.M) / cfg.rankDim;
    // 每个 rank 分片的元素数量（M_per_rank * N）
    uint64_t rankSliceElems = sliceM * static_cast<uint64_t>(tiling.tCubeTiling.N);
    // 每个 rank 分片的字节数
    uint64_t rankSliceBytes = rankSliceElems * sizeof(C_DTYPE);

    // 当前发送缓冲区起始地址（从 sendGMAddr 开始逐 rank 偏移）
    GM_ADDR currSendPtr = sendGMAddr;
    // 当前接收缓冲区起始地址（All2All 写入目标）
    GM_ADDR currRecvPtr = recvGMAddr;

    // 若是尾块，通信 handle 起始偏移为 tileCnt
    uint32_t handleShift = isTail ? cfg.tileCnt : 0;

    // All2All 的 stride（单位：元素数），即每张卡数据在全局中的间隔
    uint64_t all2allStrideElems = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    uint8_t repeat = 1; // 通信重复次数（通常为1）

    for (uint32_t i = 0; i < count; i++) {
        mmv3.UpdateSlice(i, isTail); // 更新当前 MatMul slice 偏移
        mmv3.Process(isLast && (i == (count - 1))); // 执行 MatMul，结果写入 tempGM (sendBuf_)

        // 核间同步：确保 MatMul 完成后再启动通信
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);

        // 启动 All2All 通信：从 sendBuf_ 片段发往 recvGMAddr 对应位置
        handles_[i + handleShift] = hccl_.AlltoAll<true>(
            currSendPtr,           // send buffer
            currRecvPtr,           // recv buffer  
            rankSliceElems,        // 元素数量（注意：HCCL 接口通常传元素数，非字节数）
            dataType_,
            all2allStrideElems,    // stride in elements
            repeat
        );

        // 移动到下一个 rank 的分片位置
        currSendPtr += rankSliceBytes;
        currRecvPtr += rankSliceBytes;
    }

    mmv3.End();
}
} // namespace MatmulReduceScatterImpl

#endif  // MATMUL_REDUCE_SCATTER_FP16_BF16_H