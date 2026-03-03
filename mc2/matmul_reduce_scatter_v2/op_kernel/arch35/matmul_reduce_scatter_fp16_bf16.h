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

static constexpr uint16_t SYNC_AIC_ONLY_ALL_DET_FLAG = 4; // 用于 AIC 核间同步的 flagId
static constexpr uint16_t SYNC_AIC_AIV_DET_FLAG = 8; // 用于 AIC 与 AIV 核间同步的 flagId
static constexpr uint64_t SYNC_MODE0 = 0; // 核间同步模式 0
static constexpr uint64_t SYNC_MODE2 = 2; // 核间同步模式 2

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
    __aicore__ inline void PostProcess();
    __aicore__ inline void CubeNotifyVector();
    __aicore__ inline void VecWaitCube();

private:
    ReduceSumForAlltoAll<C_DTYPE> reduceSum_; // AIV ReduceSum 相关实现

    Mc2Tiling::MatmulReduceScatterV2TilingData* tilingData_;
    TPipe* tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    __gm__ HcclCombinOpParam* context_;
    AscendC::HcclDataType dataType_;
    uint8_t debugMode_;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;              // CCU模式
    AscendC::HcclHandle handles_[MAX_HANDLE];        // 最大支持64个handleId
    GM_ADDR sendBuf_;    // 存放 MatMul 输出（All2All send buffer）
    GM_ADDR recvBuf_;    // 存放 All2All 接收的 slices（内容为 [slice_r_from_rank0][slice_r_from_rank1]...[slice_r_from_rankR-1]）
    uint32_t rankId_{0};
    uint64_t aivNum_{0};
    uint64_t fullMN_{0};
    uint64_t tileOffset_{0};
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
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    debugMode_ = tilingData_->debugMode;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    rankId_ = context_->rankId;
    aivNum_ = cfg.aicCoreNum * GetTaskRation(); // 启用的AIV 数量, 此模板会全启用

    // all2all 通信相关参数, 划分workspace
    fullMN_ = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankN);  // M * N
    sendBuf_ = workspaceGM;                                      // [0, fullMN)
    recvBuf_ = sendBuf_ + fullMN_ * sizeof(C_DTYPE);            // [fullMN, 2*fullMN)
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::PostProcess()
{
    // 等待执行完成后，最后终止hcclserver
    if ((GetBlockIdx() == 0) && (g_coreType == AIV)) {
        hccl_.Finalize();
    }
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::Process()
{
    InnerProcess(); // 核心计算+通信
    PostProcess(); // 等待计算与通信完成, 终止hcclserver
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
        tileOffset_ = tileSize * static_cast<uint64_t>(cfg.tileCnt) * sizeof(C_DTYPE);
        auto recvGMTail = recvBuf_ + tileOffset_;
        auto sendBufTail = sendBuf_ + tileOffset_;
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
    // Cube 核执行 MatMul, Vector 核执行 all2all + reduceSum
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
    // 公共参数计算
    auto&& cfg = tilingData_->param;
    cfg.rankID = rankId_;

    // 每个 rank 应得的 M 维度大小
    uint64_t sliceM = static_cast<uint64_t>(tiling.tCubeTiling.M) / cfg.rankDim;
    // 每个 rank 分片的元素数量（M_per_rank * N）
    uint64_t rankSliceElems = sliceM * static_cast<uint64_t>(tiling.tCubeTiling.N);
    // 每个 rank 分片的字节数
    uint64_t rankSliceBytes = rankSliceElems * sizeof(C_DTYPE);

    // All2All 的 stride（单位：元素数），即每张卡数据在全局中的间隔
    uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    // 通信重复次数, 1次
    uint8_t repeat = 1;
    // 若是尾块，通信 handleId 存放的起始偏移为 tileCnt
    uint32_t handleShift = isTail ? cfg.tileCnt : 0;

    // AIC 执行 MatMul；初始化 -> 循环计算 -> 清理
    if ASCEND_IS_AIC {
        MC2MatmulV3::MC2MatmulAswKernelDerive<AType, BType, CType, BiasType, MC2MatmulV3::MC2MatmulAswBlockDerive> mmv3;
        mmv3.Init(aGM_, bGM_, sendBuf_, biasGM_, nullptr, nullptr, &tiling, GetTPipePtr(), cfg, isTail, false);
        for (uint32_t i = 0; i < count; i++) {
            mmv3.UpdateSlice(i, isTail);                  // 更新 slice 偏移
            mmv3.Process(isLast && (i == (count - 1)));   // 执行 MatMul
            // CV 同步，确保 MatMul 完成后再启动通信
            // AIC侧做完Matmul计算后通知AIV进行后处理
            CubeNotifyVector();
        }
        mmv3.End();
    }

    // AIV 执行 All2All 通信 + reduceSum；采用Cube双发流水线模式：通信 -> (等待+归约+通信) -> 等待+归约
    if ASCEND_IS_AIV {
        // 当前发送缓冲区起始地址
        GM_ADDR currSendPtr = sendGMAddr;
        // 当前接收缓冲区起始地址
        GM_ADDR currRecvPtr = recvGMAddr;
        // 当前 reduceSum 输出的起始地址
        GM_ADDR curOutPtr = isTail ? cGM_ + tileOffset_ : cGM_;

        // [循环外] 提前启动第 0 轮通信 (Prologue, 双发)
        VecWaitCube(); // 确保 MatMul 完成后再启动第一次通信
        handles_[0 + handleShift] = hccl_.AlltoAll<true>(
            currSendPtr,       
            currRecvPtr,       
            rankSliceElems,    
            dataType_,
            stride,            
            repeat
        );
        
        // 移动指针准备下一轮
        currSendPtr += rankSliceBytes;
        currRecvPtr += rankSliceBytes;
        curOutPtr += rankSliceBytes;

        // [循环内] 启动下一轮通信 ,并处理上一轮的数据
        // 循环次数为 count - 1，最后一轮通信在循环内启动，但计算在循环外
        for (uint32_t i = 0; i < count - 1; i++) {
            // 等待上一轮 (i - 1) 通信结束
            if (GetBlockIdx() == 0) {
                hccl_.Wait(handles_[i + handleShift]); 
            }

            // V同步，确保数据到达
            SyncAll<true>();

            // 启动本轮 (i) 通信
            // 此时 currSendPtr/currRecvPtr 已经指向了 i+1 的位置
            VecWaitCube(); 
            handles_[i + 1 + handleShift] = hccl_.AlltoAll<true>(
                currSendPtr,       
                currRecvPtr,       
                rankSliceElems,    
                dataType_,
                stride,            
                repeat
            );

            // Vector 操作：执行上一轮 (i - 1) 数据的 reduceSum
            GM_ADDR calcRecvPtr = currRecvPtr - rankSliceBytes;
            GM_ADDR calcOutPtr  = curOutPtr - rankSliceBytes;

            tPipe_->Reset();
            reduceSum_.Init(rankSliceElems, stride, cfg.rankDim, aivNum_, calcRecvPtr, calcOutPtr, tPipe_);
            reduceSum_.ExecuteReduceSum();

            // 移动指针准备再下一轮
            currSendPtr += rankSliceBytes;
            currRecvPtr += rankSliceBytes;
            curOutPtr += rankSliceBytes;
        }

        // [循环外] 处理最后一轮 (count-1) 的数据
        uint32_t lastIdx = count - 1;
        
        // 等待最后一轮通信结束
        if (GetBlockIdx() == 0) {
            hccl_.Wait(handles_[lastIdx + handleShift]); 
        }

        // V同步
        SyncAll<true>();

        // 执行最后一轮数据的 reduceSum
        // 此时的计算地址同样是 "当前指针 - 偏移量"
        GM_ADDR calcRecvPtr = currRecvPtr - rankSliceBytes;
        GM_ADDR calcOutPtr  = curOutPtr - rankSliceBytes;

        tPipe_->Reset();
        reduceSum_.Init(rankSliceElems, stride, cfg.rankDim, aivNum_, calcRecvPtr, calcOutPtr, tPipe_);
        reduceSum_.ExecuteReduceSum();
    }
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::CubeNotifyVector()
{
    // 先全 AIC 同步一次
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_DET_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_DET_FLAG);
    // 通知 AIV
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_DET_FLAG);
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::VecWaitCube()
{
    // 等待 AIC 完成
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIC_AIV_DET_FLAG);
}

} // namespace MatmulReduceScatterV2Impl

#endif  // MATMUL_REDUCE_SCATTER_FP16_BF16_H