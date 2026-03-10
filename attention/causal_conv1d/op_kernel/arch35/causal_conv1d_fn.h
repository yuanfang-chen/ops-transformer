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
 * \file causal_conv1d_fn.h
 * \brief CausalConv1dFn kernel class.
 *
 * 算子功能：对变长token序列执行因果一维卷积（逐特征通道）。
 *   y[i] = sum_{k=0}^{K-1}(w[k] * padded_x[i+k])   + x[i]  (残差连接)
 * 其中 padded_x = cat(cache_state, seq_x)，cache_state 来自历史缓存。
 *
 * 数据类型：FP16 / BF16
 * 目标硬件：Ascend 950 (A5)
 */

#ifndef CAUSAL_CONV1D_FN_H
#define CAUSAL_CONV1D_FN_H

#include "kernel_operator.h"
#include "./vf/compute.h"
#include "causal_conv1d_fn_struct.h"

namespace CausalConv1dFnNs {

using namespace AscendC;

// ============================================================================
// 常量
// ============================================================================
constexpr int32_t BUFFER_NUM   = 2;              // 双缓冲
constexpr uint32_t ALIGN_BYTES = 32;             // DataCopy 32 字节对齐单元
constexpr uint32_t MAX_K       = 6;              // 最大卷积核宽度
constexpr uint32_t MAX_BATCH   = 256;            // 最大 batch 数

// A5 向量寄存器宽度 256 字节；BF16/FP16 每个寄存器可容纳 128 个元素
constexpr uint32_t B16_REP_SIZE = 256 / sizeof(half); // = 128
// ============================================================================
// CausalConv1dFn Kernel 类
//
// 流程概述：
//   Init  → 解析 TilingData，绑定 GM，初始化 UB 队列/缓冲
//   Process →
//     1. LoadMetaData：一次性加载 seqStartIndex、cacheIndices、hasInitialState
//     2. 按 blockIndex 决定切 BS 还是切 Dim
//     3. ProcessMainCompute：双层循环（BS × Dim），每次调用 ProcessUBBlock
//     4. SyncAll（全核同步）
//     5. WriteCacheFromWorkspace：从 workspace 回写 cache state
// ============================================================================
template <typename T>
class CausalConv1dFn {
public:
    __aicore__ inline CausalConv1dFn() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convStates,
        GM_ADDR queryStartLoc,
        GM_ADDR cacheIndices,
        GM_ADDR initialStateMode,
        GM_ADDR y,
        GM_ADDR workspace,
        const CausalConv1dFnTilingData* tiling);

    __aicore__ inline void Process();

private:
    // 主计算：按 BS 切分场景的双层循环
    __aicore__ inline void ProcessMainComputeBS(
        uint64_t bsStart,
        uint32_t loopNumBS,   uint32_t ubFactorBS,   uint32_t ubTailFactorBS,
        uint32_t loopNumDim,  uint32_t ubFactorDim,  uint32_t ubTailFactorDim);

    // 主计算：按 Dim 切分场景
    __aicore__ inline void ProcessMainComputeDim(
        uint64_t dimStart,
        uint32_t loopNumBS,   uint32_t ubFactorBS,   uint32_t ubTailFactorBS,
        uint32_t loopNumDim,  uint32_t ubFactorDim,  uint32_t ubTailFactorDim);

    // 处理单个 UB 块：[bsStart, bsStart+bsSize) 行，[dimStart, dimStart+dimSize) 列
    // curBatchIdx 和 curSequenceIdx 由调用者传入，在函数内部会被更新
    // iStart: token 遍历的起始位置（考虑 UB 块间重叠）
    __aicore__ inline void ProcessUBBlock(
        uint32_t bsStart, uint32_t bsSize,
        uint32_t dimStart, uint32_t dimSize,
        uint32_t iStart,
        uint32_t& curBatchIdx, uint32_t& curSequenceIdx);

    // 处理需要 cache state 的 token（curSequenceIdx < K-1）
    // 返回处理的 token 数量
    __aicore__ inline uint16_t ProcessTokensNeedCache(
        LocalTensor<T>& xLocal, LocalTensor<T>& weightLocal,
        uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
        uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t ySkipBlocks,
        uint64_t batchStart, uint32_t curBatchLen, uint32_t curSequenceIdx,
        int32_t hasInitState, int64_t cIdx, uint32_t curBatchIdx,
        bool& reachBatchEnd);

    // 处理不需要 cache 的 token（curSequenceIdx >= K-1）
    // 返回处理的 token 数量
    __aicore__ inline uint16_t ProcessTokensNoCache(
        LocalTensor<T>& xLocal, LocalTensor<T>& weightLocal,
        uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
        uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t ySkipBlocks,
        uint64_t batchStart, uint32_t curBatchLen, uint32_t curSequenceIdx,
        int64_t cIdx, uint32_t curBatchIdx,
        bool& reachBatchEnd);

    // 回写 cache（短 batch，长度 < K）
    __aicore__ inline void WriteCacheShortBatch(
        LocalTensor<T>& cacheLocal, LocalTensor<T>& xLocal,
        uint32_t i, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
        uint32_t cacheSkipBlocks, uint32_t curBatchLen,
        int64_t cIdx, uint32_t curBatchIdx);

    // 回写 cache（长 batch，长度 >= K，到达 batch 末尾时）
    __aicore__ inline void WriteCacheLongBatch(
        LocalTensor<T>& xLocal,
        uint32_t i, uint16_t step, uint32_t dimSize, uint32_t dimBlocks,
        uint32_t dimStart, uint32_t cacheSkipBlocks,
        int64_t cIdx, uint32_t curBatchIdx);

    // 将 cache state 从 workspace 写回 cacheStates GM（全核同步后调用）
    __aicore__ inline void WriteCacheFromWorkspace();

    // 通过二分查找确定 globalSeqIdx 所属的 batch（0-indexed）
    __aicore__ inline uint32_t FindBatchIdx(uint64_t globalSeqIdx);

    // 一次性加载元数据到 UB
    __aicore__ inline void LoadMetaData();

    // 辅助：ceil(a / b)
    static __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
    // 辅助：向上对齐到 align 字节
    static __aicore__ inline uint32_t AlignUp(uint32_t n, uint32_t align) {
        return (n + align - 1) / align * align;
    }

private:
    // -------------------------------------------------------------------------
    // Tiling 参数
    // -------------------------------------------------------------------------
    uint32_t cuSeqLen_;
    uint32_t dim_;
    uint32_t kernelWidth_;       // K
    uint32_t batchSize_;
    uint64_t blockFactor_;
    uint64_t blockIndex_;
    uint32_t loopNumBS_;
    uint32_t loopNumDim_;
    uint32_t ubFactorBS_;
    uint32_t ubTailFactorBS_;
    uint32_t ubFactorDim_;
    uint32_t ubTailFactorDim_;
    uint32_t tailBlockloopNumBS_;
    uint32_t tailBlockloopNumDim_;
    uint32_t tailBlockubFactorBS_;
    uint32_t tailBlockubTailFactorBS_;
    uint32_t tailBlockubFactorDim_;
    uint32_t tailBlockubTailFactorDim_;
    uint32_t realCoreNum_;

    // stride（跨 sequence 的步长）
    uint32_t xStride_;           // x 的行 stride（>= dim_）
    uint32_t cacheStride_;       // cacheStates 的行 stride（>= dim_）

    // 有效 batch 范围（tiling 已过滤掉头尾的无效 batch）
    uint32_t validBatchStart_;   // 有效 batch 的起始索引
    uint32_t validBatchCount_;   // 有效 batch 的数量
    uint32_t validBatchEnd_;     // 有效 batch 的结束索引（validBatchStart_ + validBatchCount_）
    uint32_t validSeqStart_;     // 有效序列在 x 中的起始位置
    uint32_t validSeqLen_;       // 有效序列的总长度

    // 运行时辅助
    bool     isTailBlock_;
    uint32_t maxUbBS_;     // max(ubFactorBS, ubTailFactorBS, tail variants)
    uint32_t maxUbDim_;    // max(ubFactorDim, ubTailFactorDim, tail variants)

    // -------------------------------------------------------------------------
    // Pipeline & Global Tensors
    // -------------------------------------------------------------------------
    TPipe pipe_;

    GlobalTensor<T>       xGM_;
    GlobalTensor<T>       weightGM_;
    GlobalTensor<T>       cacheStatesGM_;
    GlobalTensor<int32_t> cacheIndicesGM_;
    GlobalTensor<int32_t> seqStartIndexGM_;
    GlobalTensor<int32_t> hasInitialStateGM_;   // 0: 用0填充cache计算, 1: 使用cache, 2: 前K-1个置0
    GlobalTensor<T>       yGM_;
    GlobalTensor<T>       workspaceGM_;         // 临时存放待更新的 cache rows


    // -------------------------------------------------------------------------
    // UB 队列 & 缓冲
    //
    // 按 requirement.md §6 分配：
    //   weightInQueue  : K × maxUbDim × sizeof(T)，BUF_NUM=1
    //   cacheQueue     : (K-1) × maxUbDim × sizeof(T)，BUF_NUM=1（TQueBind 可 VECIN/VECOUT）
    //   startLocInQueue: (MAX_BATCH+1) × sizeof(int64_t)，BUF_NUM=1
    //   indicesInQueue : MAX_BATCH × sizeof(int64_t)，BUF_NUM=1
    //   hasInitInQueue : MAX_BATCH × sizeof(int32_t)，BUF_NUM=1
    //   xQueue         : maxUbBS × maxUbDim × sizeof(T)，BUF_NUM=2（双缓冲，y 复用，也用于 workspace 回写）
    // -------------------------------------------------------------------------
    TQue<QuePosition::VECIN, 1>        weightInQueue_;
    TQue<QuePosition::VECIN, 1>        cacheQueue_;
    TQue<QuePosition::VECIN, 1>        startLocInQueue_;
    TQue<QuePosition::VECIN, 1>        indicesInQueue_;
    TQue<QuePosition::VECIN, 1>        hasInitInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xQueue_;  // x 搬入(VECIN)，y 搬出(VECOUT)

    // 从队列 DeQue 后持久持有的 meta 数据（在 Process 期间一直有效）
    LocalTensor<int32_t> seqStartLocal_;
    LocalTensor<int32_t> cacheIdxLocal_;
    LocalTensor<int32_t> hasInitLocal_;

    // -------------------------------------------------------------------------
    // 第一个 batch 跟踪（用于区分写 workspace 还是直接写 cache_states）
    // 只有当第一个 batch 不完整（首个 token 在前一个核）时才需要写 workspace
    // -------------------------------------------------------------------------
    uint32_t firstBatchIdx_;        // 当前核的第一个 batch 索引
    int64_t  firstBatchCIdx_;       // 第一个 batch 对应的 cacheIndices 值
    bool     firstBatchComplete_;   // 第一个 batch 是否完整（首个 token 在当前核）
    bool     firstBatchWrittenToWS_;// 是否已将第一个 batch 的 cache 写入 workspace
};

// ============================================================================
// Init
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::Init(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convStates,
    GM_ADDR queryStartLoc,
    GM_ADDR cacheIndices,
    GM_ADDR initialStateMode,
    GM_ADDR y,
    GM_ADDR workspace,
    const CausalConv1dFnTilingData* tiling)
{
    // --- 解析 tiling ---
    cuSeqLen_                  = tiling->cuSeqLen;
    dim_                       = tiling->dim;
    kernelWidth_               = tiling->kernelWidth;
    batchSize_                 = tiling->batch;
    blockFactor_               = tiling->blockFactor;
    blockIndex_                = tiling->blockIndex;
    loopNumBS_                 = tiling->loopNumBS;
    loopNumDim_                = tiling->loopNumDim;
    ubFactorBS_                = tiling->ubFactorBS;
    ubTailFactorBS_            = tiling->ubTailFactorBS;
    ubFactorDim_               = tiling->ubFactorDim;
    ubTailFactorDim_           = tiling->ubTailFactorDim;
    tailBlockloopNumBS_        = tiling->tailBlockloopNumBS;
    tailBlockloopNumDim_       = tiling->tailBlockloopNumDim;
    tailBlockubFactorBS_       = tiling->tailBlockubFactorBS;
    tailBlockubTailFactorBS_   = tiling->tailBlockubTailFactorBS;
    tailBlockubFactorDim_      = tiling->tailBlockubFactorDim;
    tailBlockubTailFactorDim_  = tiling->tailBlockubTailFactorDim;
    realCoreNum_               = tiling->realCoreNum;

    // stride（x 和 cacheStates 非连续存储）
    xStride_                   = tiling->xStride;
    cacheStride_               = tiling->cacheStride;

    // 有效 batch 范围
    validBatchStart_           = tiling->validBatchStart;
    validBatchCount_           = tiling->validBatchCount;
    validBatchEnd_             = validBatchStart_ + validBatchCount_;
    validSeqStart_             = tiling->validSeqStart;
    validSeqLen_               = tiling->validSeqLen;

    uint32_t blockIdx = GetBlockIdx();
    isTailBlock_ = (blockIdx == realCoreNum_ - 1);

    // 计算各方向上实际的最大 UB 因子（用于 buffer 分配）
    maxUbBS_ = ubFactorBS_;
    if (ubTailFactorBS_          > maxUbBS_) maxUbBS_ = ubTailFactorBS_;
    if (tailBlockubFactorBS_     > maxUbBS_) maxUbBS_ = tailBlockubFactorBS_;
    if (tailBlockubTailFactorBS_ > maxUbBS_) maxUbBS_ = tailBlockubTailFactorBS_;

    maxUbDim_ = ubFactorDim_;
    if (ubTailFactorDim_          > maxUbDim_) maxUbDim_ = ubTailFactorDim_;
    if (tailBlockubFactorDim_     > maxUbDim_) maxUbDim_ = tailBlockubFactorDim_;
    if (tailBlockubTailFactorDim_ > maxUbDim_) maxUbDim_ = tailBlockubTailFactorDim_;


    // --- 绑定 GM ---
    // xGM_ 从 validSeqStart_ 开始，这样 bsStart=0 对应有效序列的起始位置
    xGM_.SetGlobalBuffer((__gm__ T*)x + validSeqStart_ * xStride_, validSeqLen_ * dim_);
    weightGM_.SetGlobalBuffer((__gm__ T*)weight, kernelWidth_ * dim_);
    cacheStatesGM_.SetGlobalBuffer((__gm__ T*)convStates);
    cacheIndicesGM_.SetGlobalBuffer((__gm__ int32_t*)cacheIndices, batchSize_);
    seqStartIndexGM_.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc, batchSize_ + 1);
    hasInitialStateGM_.SetGlobalBuffer((__gm__ int32_t*)initialStateMode, batchSize_);
    yGM_.SetGlobalBuffer((__gm__ T*)y, cuSeqLen_ * dim_);
    if (workspace != nullptr) {
        GM_ADDR userWS = GetUserWorkspace(workspace);
        if (userWS != nullptr) {
            workspaceGM_.SetGlobalBuffer((__gm__ T*)userWS);
        }
    }

    // --- 计算 buffer 字节大小并对齐到 32 字节 ---
    uint32_t K = kernelWidth_;

    uint32_t weightBufBytes  = AlignUp(K * maxUbDim_ * sizeof(T),       ALIGN_BYTES);
    uint32_t cacheBufBytes   = AlignUp((K - 1) * maxUbDim_ * sizeof(T), ALIGN_BYTES);
    uint32_t startLocBytes   = AlignUp((batchSize_ + 1) * sizeof(int32_t), ALIGN_BYTES);
    uint32_t indicesBytes    = AlignUp(batchSize_ * sizeof(int32_t),       ALIGN_BYTES);
    uint32_t hasInitBytes    = AlignUp(batchSize_ * sizeof(int32_t),       ALIGN_BYTES);
    uint32_t xBufBytes       = AlignUp(maxUbBS_ * maxUbDim_ * sizeof(T),  ALIGN_BYTES);

    // --- 初始化 UB 队列 / 缓冲 ---
    pipe_.InitBuffer(weightInQueue_,   1,           weightBufBytes);
    pipe_.InitBuffer(cacheQueue_,      1,           cacheBufBytes);
    pipe_.InitBuffer(startLocInQueue_, 1,           startLocBytes);
    pipe_.InitBuffer(indicesInQueue_,  1,           indicesBytes);
    pipe_.InitBuffer(hasInitInQueue_,  1,           hasInitBytes);
    pipe_.InitBuffer(xQueue_,          BUFFER_NUM,  xBufBytes);

    // Debug: 打印所有初始化后的成员变量   
    //PRINTF("[Init] blockIdx=%u, cuSeqLen=%u, dim=%u, kernelWidth=%u, batchSize=%u, blockFactor=%lu, blockIndex=%lu, "
        //    "loopNumBS=%u, loopNumDim=%u, ubFactorBS=%u, ubTailFactorBS=%u, ubFactorDim=%u, ubTailFactorDim=%u, "
        //    "tailBlockloopNumBS=%u, tailBlockloopNumDim=%u, tailBlockubFactorBS=%u, tailBlockubTailFactorBS=%u, "
        //    "tailBlockubFactorDim=%u, tailBlockubTailFactorDim=%u, realCoreNum=%u, xStride=%u, cacheStride=%u, "
        //    "validBatchStart=%u, validBatchCount=%u, validBatchEnd=%u, validSeqStart=%u, validSeqLen=%u, "
        //    "isTailBlock=%d, maxUbBS=%u, maxUbDim=%u\n",
        //    GetBlockIdx(), cuSeqLen_, dim_, kernelWidth_, batchSize_, blockFactor_, blockIndex_,
        //    loopNumBS_, loopNumDim_, ubFactorBS_, ubTailFactorBS_, ubFactorDim_, ubTailFactorDim_,
        //    tailBlockloopNumBS_, tailBlockloopNumDim_, tailBlockubFactorBS_, tailBlockubTailFactorBS_,
        //    tailBlockubFactorDim_, tailBlockubTailFactorDim_, realCoreNum_, xStride_, cacheStride_,
        //    validBatchStart_, validBatchCount_, validBatchEnd_, validSeqStart_, validSeqLen_,
        //    (int)isTailBlock_, maxUbBS_, maxUbDim_);
}

// ============================================================================
// LoadMetaData：一次性加载 seqStartIndex、cacheIndices、hasInitialState
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::LoadMetaData()
{
    // seqStartIndex (queryStartLoc)
    {
        LocalTensor<int32_t> tmp = startLocInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams cpParams{1, static_cast<uint16_t>((batchSize_ + 1) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmp, seqStartIndexGM_[0], cpParams, padParams);
        startLocInQueue_.EnQue(tmp);
        seqStartLocal_ = startLocInQueue_.DeQue<int32_t>();
    }
    // cacheIndices
    {
        LocalTensor<int32_t> tmp = indicesInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams cpParams{1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmp, cacheIndicesGM_[0], cpParams, padParams);
        indicesInQueue_.EnQue(tmp);
        cacheIdxLocal_ = indicesInQueue_.DeQue<int32_t>();
    }
    // hasInitialState (initialStateMode)
    {
        LocalTensor<int32_t> tmp = hasInitInQueue_.AllocTensor<int32_t>();
        DataCopyExtParams cpParams{1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(tmp, hasInitialStateGM_[0], cpParams, padParams);
        hasInitInQueue_.EnQue(tmp);
        hasInitLocal_ = hasInitInQueue_.DeQue<int32_t>();
    }
    // 等待所有 MTE2 搬运完成，确保后续 GetValue (PIPE_S) 能正确读取数据
    PipeBarrier<PIPE_ALL>();
}

// ============================================================================
// FindBatchIdx：二分查找 globalSeqIdx 所在 batch
// 保证：seqStartLocal_[result] <= globalSeqIdx < seqStartLocal_[result+1]
// 搜索范围限定在 [validBatchStart_, validBatchStart_ + validBatchCount_)
// ============================================================================
template <typename T>
__aicore__ inline uint32_t CausalConv1dFn<T>::FindBatchIdx(uint64_t globalSeqIdx)
{
    uint32_t lo = validBatchStart_;
    uint32_t hi = validBatchStart_ + validBatchCount_ - 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1) / 2;
        if ((uint64_t)seqStartLocal_.GetValue(mid) <= globalSeqIdx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// ============================================================================
// ProcessUBBlock - 处理单个 UB 块
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::ProcessUBBlock(
    uint32_t bsStart, uint32_t bsSize,
    uint32_t dimStart, uint32_t dimSize,
    uint32_t iStart,
    uint32_t& curBatchIdx, uint32_t& curSequenceIdx)
{
    uint32_t K = kernelWidth_;
    uint32_t N = bsSize;

    // 计算各种 stride 参数
    uint32_t dimBytes   = dimSize * sizeof(T);
    uint32_t dimBlocks  = dimBytes / ALIGN_BYTES;
    uint32_t weightSkipBlocks = (dim_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t ySkipBlocks = weightSkipBlocks;
    uint32_t xSkipBlocks = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkipBlocks = (cacheStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    // 计算当前 batch 信息
    uint64_t batchStart  = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
    uint64_t batchEnd    = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
    uint32_t curBatchLen = (uint32_t)(batchEnd - batchStart);

    // Debug: 打印 batch 信息和 stride 参数
    //PRINTF("[ProcessUBBlock] K=%u, N=%u, dimBytes=%u, dimBlocks=%u, weightSkipBlocks=%u, ySkipBlocks=%u, xSkipBlocks=%u, cacheSkipBlocks=%u\n",K, N, dimBytes, dimBlocks, weightSkipBlocks, ySkipBlocks, xSkipBlocks, cacheSkipBlocks);
    //PRINTF("[ProcessUBBlock] batch info: curBatchIdx=%u, batchStart=%lu, batchEnd=%lu, curBatchLen=%u\n",curBatchIdx, batchStart, batchEnd, curBatchLen);

    // 加载 weight
    LocalTensor<T> weightLocal = weightInQueue_.AllocTensor<T>();
    {
        DataCopyExtParams wcp{static_cast<uint16_t>(K), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              static_cast<uint16_t>(weightSkipBlocks * ALIGN_BYTES), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // Debug: 打印 weight DataCopy 参数
        //PRINTF("[ProcessUBBlock] weight DataCopy: blockCount=%u, blockLen=%u, srcStride=%u, gmOffset=dimStart=%u\n",K, dimBlocks * ALIGN_BYTES, weightSkipBlocks * ALIGN_BYTES, dimStart);
        DataCopyPad(weightLocal, weightGM_[dimStart], wcp, padParams);
    }
    weightInQueue_.EnQue(weightLocal);
    weightLocal = weightInQueue_.DeQue<T>();

    // 加载 x
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    {
        DataCopyExtParams xcp{static_cast<uint16_t>(bsSize), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              static_cast<uint16_t>(xSkipBlocks * ALIGN_BYTES), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // Debug: 打印 x DataCopy 参数
        //PRINTF("[ProcessUBBlock] x DataCopy: blockCount=%u, blockLen=%u, srcStride=%u, gmOffset=%u (bsStart=%u, xStride=%u, dimStart=%u)\n",bsSize, dimBlocks * ALIGN_BYTES, xSkipBlocks * ALIGN_BYTES, bsStart * xStride_ + dimStart, bsStart, xStride_, dimStart);
        DataCopyPad(xLocal, xGM_[bsStart * xStride_ + dimStart], xcp, padParams);
    }
    xQueue_.EnQue(xLocal);
    xLocal = xQueue_.DeQue<T>();

    // 等待 MTE2 搬运完成，确保 weight 和 x 数据就绪
    PipeBarrier<PIPE_ALL>();

    // 主循环：遍历 UB 块中的 token
    uint32_t i = iStart;
    while (i < N) {
        // 确保上一轮迭代的所有操作完成
        PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_ALL>();

        int32_t hasInitState = hasInitLocal_.GetValue(curBatchIdx);
        int64_t cIdx = cacheIdxLocal_.GetValue(curBatchIdx);

        // Debug: 打印循环开始时的变量
        //PRINTF("[while] i=%u, N=%u, curBatchIdx=%u, curSequenceIdx=%u, hasInitState=%d, cIdx=%ld, K=%u\n",i, N, curBatchIdx, curSequenceIdx, hasInitState, (long)cIdx, K);

        uint16_t step = 0;
        bool reachBatchEnd = false;

        if (curSequenceIdx < K - 1) {
            // Debug: 进入 ProcessTokensNeedCache 分支
            //PRINTF("[while] curSequenceIdx(%u) < K-1(%u), calling ProcessTokensNeedCache\n", curSequenceIdx, K - 1);
            step = ProcessTokensNeedCache(
                xLocal, weightLocal, i, N, dimSize, dimBlocks,
                dimStart, cacheSkipBlocks, ySkipBlocks,
                batchStart, curBatchLen, curSequenceIdx,
                hasInitState, cIdx, curBatchIdx, reachBatchEnd);
            //PRINTF("[while] ProcessTokensNeedCache returned: step=%u, reachBatchEnd=%d\n", step, (int)reachBatchEnd);
        } else {
            // Debug: 进入 ProcessTokensNoCache 分支
            //PRINTF("[while] curSequenceIdx(%u) >= K-1(%u), calling ProcessTokensNoCache\n", curSequenceIdx, K - 1);
            step = ProcessTokensNoCache(
                xLocal, weightLocal, i, N, dimSize, dimBlocks,
                dimStart, cacheSkipBlocks, ySkipBlocks,
                batchStart, curBatchLen, curSequenceIdx,
                cIdx, curBatchIdx, reachBatchEnd);
            //PRINTF("[while] ProcessTokensNoCache returned: step=%u, reachBatchEnd=%d\n", step, (int)reachBatchEnd);
        }

        // 更新索引
        i += step;
        curSequenceIdx += step;
        //PRINTF("[while] after update: i=%u, curSequenceIdx=%u\n", i, curSequenceIdx);

        // 如果到达 batch 末尾，更新 batch 索引
        if (reachBatchEnd && curBatchIdx + 1 < validBatchEnd_) {
            curBatchIdx++;
            batchStart   = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
            batchEnd     = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
            curBatchLen  = (uint32_t)(batchEnd - batchStart);
            curSequenceIdx = 0;
            //PRINTF("[while] batch switched: new curBatchIdx=%u, batchStart=%lu, batchEnd=%lu, curBatchLen=%u\n",curBatchIdx, batchStart, batchEnd, curBatchLen);
        }
    }

    // 等待所有操作完成后再释放 buffer
    PipeBarrier<PIPE_ALL>();

    // 释放 buffer
    weightInQueue_.FreeTensor(weightLocal);
    xQueue_.FreeTensor(xLocal);
}

// ============================================================================
// ProcessTokensNeedCache - 处理需要 cache state 的 token（curSequenceIdx < K-1）
// ============================================================================
template <typename T>
__aicore__ inline uint16_t CausalConv1dFn<T>::ProcessTokensNeedCache(
    LocalTensor<T>& xLocal, LocalTensor<T>& weightLocal,
    uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
    uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t ySkipBlocks,
    uint64_t batchStart, uint32_t curBatchLen, uint32_t curSequenceIdx,
    int32_t hasInitState, int64_t cIdx, uint32_t curBatchIdx,
    bool& reachBatchEnd)
{
    uint32_t K = kernelWidth_;

    // Debug: 打印入口参数
    //PRINTF("[ProcessTokensNeedCache] ENTER: i=%u, N=%u, dimSize=%u, dimBlocks=%u, dimStart=%u\n",i, N, dimSize, dimBlocks, dimStart);
    //PRINTF("[ProcessTokensNeedCache] batchStart=%lu, curBatchLen=%u, curSequenceIdx=%u, hasInitState=%d, cIdx=%ld, curBatchIdx=%u, K=%u\n",batchStart, curBatchLen, curSequenceIdx, hasInitState, (long)cIdx, curBatchIdx, K);

    // 计算 step
    uint16_t step = K - 1 - curSequenceIdx;
    if (step > N - i) step = N - i;
    if (step > curBatchLen - curSequenceIdx) step = curBatchLen - curSequenceIdx;
    reachBatchEnd = (step == curBatchLen - curSequenceIdx);

    //PRINTF("[ProcessTokensNeedCache] step=%u, reachBatchEnd=%d\n", step, (int)reachBatchEnd);

    // 加载或初始化 cache state
    LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
    if (hasInitState == 1) {
        uint64_t cacheGmOffset = (uint64_t)cIdx * (K - 1) * cacheStride_ + dimStart;
        //PRINTF("[ProcessTokensNeedCache] Loading cache from GM: cacheGmOffset=%lu, cacheSkipBlocks=%u\n",cacheGmOffset, cacheSkipBlocks);
        DataCopyExtParams ccp{static_cast<uint16_t>(K - 1), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              static_cast<uint16_t>(cacheSkipBlocks * ALIGN_BYTES), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(cacheLocal, cacheStatesGM_[cacheGmOffset], ccp, padParams);
    } else {
        //PRINTF("[ProcessTokensNeedCache] Initializing cache with zeros (hasInitState=%d)\n", hasInitState);
        Duplicate(cacheLocal, (T)0, (K - 1) * dimSize);
    }
    cacheQueue_.EnQue(cacheLocal);
    cacheLocal = cacheQueue_.DeQue<T>();

    // 等待 cacheLocal 数据就绪
    // hasInitState==1: MTE2 搬运; 否则: Duplicate (PIPE_V)
    PipeBarrier<PIPE_ALL>();

    // 如果 batch 长度 < K 且到达 batch 末尾，回写 cache
    if (curBatchLen < K && reachBatchEnd) {
        //PRINTF("[ProcessTokensNeedCache] Short batch, writing cache\n");
        WriteCacheShortBatch(cacheLocal, xLocal, i, dimSize, dimBlocks, dimStart,
                             cacheSkipBlocks, curBatchLen, cIdx, curBatchIdx);
    }

    // 卷积计算（hasInitState == 2 时跳过）
    if (hasInitState != 2) {
        //PRINTF("[ProcessTokensNeedCache] Conv loop: step=%u tokens\n", step);
        for (uint32_t j = 0; j < step; j++) {
            uint32_t seqPos = curSequenceIdx + j;
            uint32_t stateSLen = K - 1 - seqPos;
            uint32_t xSLen = seqPos + 1;
            //PRINTF("[ProcessTokensNeedCache] Conv j=%u, seqPos=%u, stateSLen=%u, xSLen=%u, xSlice offset=%u, stateSlice offset=%u\n",j, seqPos, stateSLen, xSLen, i * dimSize, seqPos * dimSize);
            LocalTensor<T> xSlice = xLocal[i * dimSize];
            LocalTensor<T> stateSlice = cacheLocal[seqPos * dimSize];
            Conv1dNeedState(xSlice, weightLocal, stateSlice, stateSlice, stateSLen, xSLen, dimSize);
            // 每次 VF 计算后同步，确保写入完成
        }
        PipeBarrier<PIPE_ALL>();
    } else {
        //PRINTF("[ProcessTokensNeedCache] Skipping conv (hasInitState==2)\n");
    }

    // 写回 y
    uint64_t yGmOffset = (batchStart + curSequenceIdx) * dim_ + dimStart;
    //PRINTF("[ProcessTokensNeedCache] y DataCopy: blockCount=%u, blockLen=%u, dstStride=%u, yGmOffset=%lu, cacheLocalOffset=%u\n",step, dimBlocks * ALIGN_BYTES, ySkipBlocks * ALIGN_BYTES, yGmOffset, curSequenceIdx * dimSize);
    DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, ySkipBlocks * ALIGN_BYTES, 0};
    DataCopyPad(yGM_[yGmOffset],
                cacheLocal[curSequenceIdx * dimSize], ycp);

    // 等待 MTE3 写 GM 完成
    PipeBarrier<PIPE_ALL>();

    //PRINTF("[ProcessTokensNeedCache] EXIT: returning step=%u\n", step);
    cacheQueue_.FreeTensor(cacheLocal);
    return step;
}

// ============================================================================
// ProcessTokensNoCache - 处理不需要 cache 的 token（curSequenceIdx >= K-1）
// ============================================================================
template <typename T>
__aicore__ inline uint16_t CausalConv1dFn<T>::ProcessTokensNoCache(
    LocalTensor<T>& xLocal, LocalTensor<T>& weightLocal,
    uint32_t i, uint32_t N, uint32_t dimSize, uint32_t dimBlocks,
    uint32_t dimStart, uint32_t cacheSkipBlocks, uint32_t ySkipBlocks,
    uint64_t batchStart, uint32_t curBatchLen, uint32_t curSequenceIdx,
    int64_t cIdx, uint32_t curBatchIdx,
    bool& reachBatchEnd)
{
    uint32_t K = kernelWidth_;

    // Debug: 打印函数入口参数
    //PRINTF("[ProcessTokensNoCache] ENTER: i=%u, N=%u, dimSize=%u, dimBlocks=%u, dimStart=%u, cacheSkipBlocks=%u, ySkipBlocks=%u\n",i, N, dimSize, dimBlocks, dimStart, cacheSkipBlocks, ySkipBlocks);
    //PRINTF("[ProcessTokensNoCache] batchStart=%lu, curBatchLen=%u, curSequenceIdx=%u, cIdx=%ld, curBatchIdx=%u, K=%u\n",batchStart, curBatchLen, curSequenceIdx, (long)cIdx, curBatchIdx, K);

    // 计算 step
    uint32_t remainInBatch = curBatchLen - curSequenceIdx;
    uint16_t step = (N - i < remainInBatch) ? (N - i) : remainInBatch;
    reachBatchEnd = (step == remainInBatch);

    // Debug: 打印计算的 step
    //PRINTF("[ProcessTokensNoCache] remainInBatch=%u, N-i=%u, step=%u, reachBatchEnd=%d\n",remainInBatch, N - i, step, (int)reachBatchEnd);

    // 如果到达 batch 末尾，回写 cache
    if (reachBatchEnd) {
        //PRINTF("[ProcessTokensNoCache] reachBatchEnd=true, calling WriteCacheLongBatch\n");
        WriteCacheLongBatch(xLocal, i, step, dimSize, dimBlocks, dimStart,
                            cacheSkipBlocks, cIdx, curBatchIdx);
    }

    // 卷积计算
    //PRINTF("[ProcessTokensNoCache] Conv loop: step=%u tokens\n", step);
    for (uint32_t j = 0; j < step; j++) {
        uint32_t tokenIdx = i + j;
        uint32_t xStartIdx = tokenIdx - (K - 1);  // 卷积窗口起始位置
        //PRINTF("[ProcessTokensNoCache] Conv j=%u, tokenIdx=%u, xStartIdx=%u, xSlice offset=%u, ySlice offset=%u\n",j, tokenIdx, xStartIdx, xStartIdx * dimSize, tokenIdx * dimSize);
        LocalTensor<T> xSlice = xLocal[xStartIdx * dimSize];  // 输入：卷积窗口起始
        LocalTensor<T> ySlice = xLocal[xStartIdx * dimSize];   // 输出：当前 token 位置
        Conv1dNoNeedState(xSlice, weightLocal, ySlice, K, dimSize);
        // 每次 VF 计算后同步，确保写入完成
    }
    PipeBarrier<PIPE_ALL>();

    // 写回 y
    uint64_t yGmOffset = (batchStart + curSequenceIdx) * dim_ + dimStart;
    //PRINTF("[ProcessTokensNoCache] y DataCopy: blockCount=%u, blockLen=%u, dstStride=%u, yGmOffset=%lu, xLocalOffset=%u\n",step, dimBlocks * ALIGN_BYTES, ySkipBlocks * ALIGN_BYTES, yGmOffset, i * dimSize);
    DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                          0, ySkipBlocks * ALIGN_BYTES, 0};
    DataCopyPad(yGM_[yGmOffset], xLocal[(i - K + 1) * dimSize], ycp);

    // 等待 MTE3 写 GM 完成
    PipeBarrier<PIPE_ALL>();

    //PRINTF("[ProcessTokensNoCache] EXIT: returning step=%u\n", step);
    return step;
}

// ============================================================================
// WriteCacheShortBatch - 回写 cache（短 batch，长度 < K）
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::WriteCacheShortBatch(
    LocalTensor<T>& cacheLocal, LocalTensor<T>& xLocal,
    uint32_t i, uint32_t dimSize, uint32_t dimBlocks, uint32_t dimStart,
    uint32_t cacheSkipBlocks, uint32_t curBatchLen,
    int64_t cIdx, uint32_t curBatchIdx)
{
    uint32_t K = kernelWidth_;
    uint32_t xRowsInBatch = curBatchLen;
    uint32_t cacheRowsToKeep = (K - 1 > xRowsInBatch) ? (K - 1 - xRowsInBatch) : 0;

    bool needWriteWorkspace = (curBatchIdx == firstBatchIdx_ && !firstBatchComplete_);

    // 等待 cacheLocal 数据就绪（可能由 Duplicate 填充）
    PipeBarrier<PIPE_ALL>();

    if (needWriteWorkspace) {
        uint32_t blockIdx = GetBlockIdx();
        uint64_t wsOffset = (uint64_t)blockIdx * (K - 1) * dim_ + dimStart;
        // workspace 每行 dim_ 个元素，当前只写 dimSize 个，dstStride 跳过剩余部分
        uint32_t wsDstSkip = (dim_ - dimSize) * sizeof(T) / ALIGN_BYTES;

        if (cacheRowsToKeep > 0) {
            DataCopyExtParams wcp{static_cast<uint16_t>(cacheRowsToKeep),
                                  static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0, wsDstSkip, 0};
            DataCopyPad(workspaceGM_[wsOffset], cacheLocal[xRowsInBatch * dimSize], wcp);
        }
        if (xRowsInBatch > 0) {
            DataCopyExtParams wcp2{static_cast<uint16_t>(xRowsInBatch),
                                   static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0, wsDstSkip, 0};
            DataCopyPad(workspaceGM_[wsOffset + cacheRowsToKeep * dim_], xLocal[i * dimSize], wcp2);
        }
        firstBatchWrittenToWS_ = true;
    } else {
        uint64_t csOffset = (uint64_t)cIdx * (K - 1) * cacheStride_ + dimStart;

        if (cacheRowsToKeep > 0) {
            DataCopyExtParams wcp{static_cast<uint16_t>(cacheRowsToKeep),
                                  static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                                  0, static_cast<uint16_t>(cacheSkipBlocks * ALIGN_BYTES), 0};
            DataCopyPad(cacheStatesGM_[csOffset], cacheLocal[xRowsInBatch * dimSize], wcp);
        }
        if (xRowsInBatch > 0) {
            DataCopyExtParams wcp2{static_cast<uint16_t>(xRowsInBatch),
                                   static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                                   0, static_cast<uint16_t>(cacheSkipBlocks * ALIGN_BYTES), 0};
            DataCopyPad(cacheStatesGM_[csOffset + cacheRowsToKeep * cacheStride_], xLocal[i * dimSize], wcp2);
        }
    }
    // 等待 MTE3 写 GM 完成
    PipeBarrier<PIPE_ALL>();
}

// ============================================================================
// WriteCacheLongBatch - 回写 cache（长 batch，长度 >= K）
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::WriteCacheLongBatch(
    LocalTensor<T>& xLocal,
    uint32_t i, uint16_t step, uint32_t dimSize, uint32_t dimBlocks,
    uint32_t dimStart, uint32_t cacheSkipBlocks,
    int64_t cIdx, uint32_t curBatchIdx)
{
    uint32_t K = kernelWidth_;
    uint32_t batchEndInUB = i + step;
    uint32_t lastK1Start = batchEndInUB - (K - 1);
    uint32_t rowsToCopy = K - 1;

    bool needWriteWorkspace = (curBatchIdx == firstBatchIdx_ && !firstBatchComplete_);

    if (needWriteWorkspace) {
        uint32_t blockIdx = GetBlockIdx();
        uint64_t wsOffset = (uint64_t)blockIdx * (K - 1) * dim_ + dimStart;
        // workspace 每行 dim_ 个元素，当前只写 dimSize 个，dstStride 跳过剩余部分
        uint32_t wsDstSkip = (dim_ - dimSize) * sizeof(T) / ALIGN_BYTES;
        DataCopyExtParams wcp{static_cast<uint16_t>(rowsToCopy),
                              static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0, wsDstSkip, 0};
        DataCopyPad(workspaceGM_[wsOffset], xLocal[lastK1Start * dimSize], wcp);
        firstBatchWrittenToWS_ = true;
    } else {
        uint64_t csOffset = (uint64_t)cIdx * (K - 1) * cacheStride_ + dimStart;
        DataCopyExtParams wcp{static_cast<uint16_t>(rowsToCopy),
                              static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              0, static_cast<uint16_t>(cacheSkipBlocks * ALIGN_BYTES), 0};
        DataCopyPad(cacheStatesGM_[csOffset], xLocal[lastK1Start * dimSize], wcp);
    }
    // 等待 MTE3 写 GM 完成
    PipeBarrier<PIPE_ALL>();
}

// ============================================================================
// ProcessMainComputeBS：blockIndex == 0，按 BS 轴切分
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::ProcessMainComputeBS(
    uint64_t bsStart,
    uint32_t loopNumBS,  uint32_t ubFactorBS,  uint32_t ubTailFactorBS,
    uint32_t loopNumDim, uint32_t ubFactorDim, uint32_t ubTailFactorDim)
{
    // 每次 BS 方向前进的步长（由于有 K-1 重叠，每次实际前进 ubFactorBS - (K-1) 行）
    uint32_t K    = kernelWidth_;
    uint16_t step = (ubFactorBS > K - 1) ? (ubFactorBS - (K - 1)) : 1;

    // bsStart 是相对于有效序列的偏移，需要转换为全局位置用于 batch 边界判断
    uint64_t globalBsStart = bsStart + validSeqStart_;

    // 只在核开始时调用一次 FindBatchIdx（传入全局位置）
    uint32_t curBatchIdx = FindBatchIdx(globalBsStart);
    // -------------------------------------------------------------------------
    // 初始化第一个 batch 的跟踪信息
    // -------------------------------------------------------------------------
    firstBatchIdx_   = curBatchIdx;
    firstBatchCIdx_  = cacheIdxLocal_.GetValue(curBatchIdx);
    // 判断第一个 batch 是否完整：globalBsStart == batch 的起始位置
    uint64_t firstBatchStart = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
    firstBatchComplete_   = (globalBsStart == firstBatchStart);
    firstBatchWrittenToWS_ = false;

    // iStart：只有核 0 的第一个 BS 循环从 0 开始，其余从 K-1 开始
    // 提前计算，避免循环内重复判断
    bool isFirstCore = (bsStart == 0);
    uint32_t iStart = isFirstCore ? 0 : (K - 1);

    uint64_t curBsOff = 0;  // 相对于 bsStart 的偏移

    for (uint32_t bsLoop = 0; bsLoop < loopNumBS; bsLoop++) {
        uint32_t curBS = (bsLoop == loopNumBS - 1) ? ubTailFactorBS : ubFactorBS;
        uint64_t curBsStart = bsStart + curBsOff;

        // 计算实际开始处理的全局位置（非重叠部分的第一个 token）
        uint64_t actualStart = curBsStart + iStart + validSeqStart_;

        // 更新 curBatchIdx 到 actualStart 所在的 batch（大多数情况下不执行，因为上一轮已更新）
        while (curBatchIdx + 1 < validBatchEnd_ &&
               actualStart >= (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1)) {
            curBatchIdx++;
        }

        // 计算 curSequenceIdx（基于实际开始位置）
        uint64_t batchStart = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
        uint32_t curSequenceIdx = (uint32_t)(actualStart - batchStart);

        uint32_t dimOff = 0;
        uint32_t batchIdxForDim = curBatchIdx;
        uint32_t seqIdxForDim = curSequenceIdx;
        for (uint32_t dimLoop = 0; dimLoop < loopNumDim; dimLoop++) {
            uint32_t curDim = (dimLoop == loopNumDim - 1) ? ubTailFactorDim : ubFactorDim;
            // dim 循环处理同一块 BS 数据的不同 dim 切片，每次传入相同的初始 batch 状态
            batchIdxForDim = curBatchIdx;
            seqIdxForDim = curSequenceIdx;
            // Debug: 打印每次 ProcessUBBlock 调用参数
            //PRINTF("[ProcessMainComputeBS] bsLoop=%u, dimLoop=%u, ProcessUBBlock params: bsStart=%u, bsSize=%u, dimStart=%u, dimSize=%u, iStart=%u, curBatchIdx=%u, curSequenceIdx=%u\n",bsLoop, dimLoop, (uint32_t)curBsStart, curBS, dimOff, curDim, iStart, batchIdxForDim, seqIdxForDim);
            ProcessUBBlock((uint32_t)curBsStart, curBS, dimOff, curDim, iStart, batchIdxForDim, seqIdxForDim);
            dimOff += curDim;
        }
        // dim 循环结束后，从最后一次 ProcessUBBlock 获取更新后的 curBatchIdx
        curBatchIdx = batchIdxForDim;

        // BS 方向前进
        curBsOff += step;
        // 第一次循环后，iStart 始终为 K-1
        iStart = K - 1;
    }
}

// ============================================================================
// ProcessMainComputeDim：blockIndex == 1，按 Dim 轴切分
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::ProcessMainComputeDim(
    uint64_t dimStart,
    uint32_t loopNumBS,  uint32_t ubFactorBS,  uint32_t ubTailFactorBS,
    uint32_t loopNumDim, uint32_t ubFactorDim, uint32_t ubTailFactorDim)
{
    // 当切 Dim 时，每个核处理全部 BS（0..validSeqLen_-1），但只处理分配到的 Dim 切片
    // BS 从 0 开始（对应有效序列的起始），初始 batch 索引为 validBatchStart_
    uint32_t curBatchIdx = validBatchStart_;
    uint32_t K = kernelWidth_;

    // -------------------------------------------------------------------------
    // 初始化第一个 batch 的跟踪信息
    // 切 Dim 轴时，所有核都从 BS=0 开始，第一个 batch 始终是完整的
    // -------------------------------------------------------------------------
    firstBatchIdx_   = validBatchStart_;
    firstBatchCIdx_  = cacheIdxLocal_.GetValue(firstBatchIdx_);
    firstBatchComplete_   = true;  // BS=0 == seqStartLocal_[validBatchStart_]，第一个 batch 始终完整
    firstBatchWrittenToWS_ = false;

    // iStart：第一个 BS 循环从 0 开始，其余从 K-1 开始
    uint32_t iStart = 0;

    uint32_t bsOff = 0;
    for (uint32_t bsLoop = 0; bsLoop < loopNumBS; bsLoop++) {
        uint32_t curBS = (bsLoop == loopNumBS - 1) ? ubTailFactorBS : ubFactorBS;

        // 计算实际开始处理的全局位置（非重叠部分的第一个 token）
        uint64_t actualStart = (uint64_t)bsOff + iStart + validSeqStart_;

        // 更新 curBatchIdx 到 actualStart 所在的 batch（大多数情况下不执行，因为上一轮已更新）
        while (curBatchIdx + 1 < validBatchEnd_ &&
               actualStart >= (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1)) {
            curBatchIdx++;
        }

        // 计算 curSequenceIdx（基于实际开始位置，使用全局位置计算）
        uint64_t batchStart = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
        uint32_t curSequenceIdx = (uint32_t)(actualStart - batchStart);

        uint32_t dimOff = (uint32_t)dimStart;
        uint32_t batchIdxForDim = curBatchIdx;
        uint32_t seqIdxForDim = curSequenceIdx;
        for (uint32_t dimLoop = 0; dimLoop < loopNumDim; dimLoop++) {
            uint32_t curDim = (dimLoop == loopNumDim - 1) ? ubTailFactorDim : ubFactorDim;
            // dim 循环处理同一块 BS 数据的不同 dim 切片
            batchIdxForDim = curBatchIdx;
            seqIdxForDim = curSequenceIdx;
            ProcessUBBlock(bsOff, curBS, dimOff, curDim, iStart, batchIdxForDim, seqIdxForDim);
            dimOff += curDim;
        }
        // dim 循环结束后，从最后一次 ProcessUBBlock 获取更新后的 curBatchIdx
        curBatchIdx = batchIdxForDim;

        bsOff += curBS;
        // 第一次循环后，iStart 始终为 K-1
        iStart = K - 1;
    }
}

// ============================================================================
// WriteCacheFromWorkspace：SyncAll 后，从 workspace 回写 cacheStates
//
// 新设计：每个核最多只写一个 batch 的 cache 到 workspace（第一个不完整的 batch）
// 同步后，每个核只需要处理自己写的那份数据
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::WriteCacheFromWorkspace()
{
    // 如果没有写 workspace，直接返回
    if (!firstBatchWrittenToWS_) {
        return;
    }

    uint32_t blockIdx = GetBlockIdx();
    uint32_t K        = kernelWidth_;
    uint32_t rows     = K - 1;
    // 每行实际数据的字节数（dim 维度是连续的）
    uint32_t rowBytes = dim_ * sizeof(T);
    uint32_t rowBlocks = rowBytes / ALIGN_BYTES;
    // cacheStates 的行间跳过块数（workspace 连续，不需要 skip）
    uint32_t cacheSkip = (cacheStride_ - dim_) * sizeof(T) / ALIGN_BYTES;

    // workspace 按 blockIdx 索引
    uint64_t wsOff = (uint64_t)blockIdx * rows * dim_;
    // cacheStates 使用 firstBatchCIdx_
    uint64_t csOff = (uint64_t)firstBatchCIdx_ * rows * cacheStride_;

    // 直接 GM→GM 搬运（通过 UB 中转）
    LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();
    {
        DataCopyExtParams cp{static_cast<uint16_t>(rows),
                             static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                             0, 0, 0};  // workspace 连续，UB 连续
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(tmpBuf, workspaceGM_[wsOff], cp, padParams);
    }
    xQueue_.EnQue(tmpBuf);
    tmpBuf = xQueue_.DeQue<T>();

    // 等待 MTE2 搬运完成，确保 tmpBuf 数据就绪
    PipeBarrier<PIPE_ALL>();

    {
        DataCopyExtParams cp{static_cast<uint16_t>(rows),
                             static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                             0, cacheSkip, 0};  // UB 连续，cacheStates 非连续
        DataCopyPad(cacheStatesGM_[csOff], tmpBuf, cp);
    }

    // 等待 MTE3 写 GM 完成
    PipeBarrier<PIPE_ALL>();

    xQueue_.FreeTensor(tmpBuf);
}

// ============================================================================
// Process：主流程
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum_) {
        return;
    }

    // 1. 一次性加载元数据
    LoadMetaData();

    // 2. 按切分轴分派计算
    if (blockIndex_ == 0) {
        // 切 BS 轴：BS 方向有主尾核之分，Dim 方向不切分（每核处理完整 dim）
        // 核间有重叠：每核加载 blockFactor_ 个 token，有效输出 blockFactor_ - (K-1) 个
        uint32_t K = kernelWidth_;
        uint64_t effectiveStep = blockFactor_ - (K - 1);
        uint64_t bsStart = (uint64_t)blockIdx * effectiveStep;

        uint32_t loopBS  = isTailBlock_ ? tailBlockloopNumBS_        : loopNumBS_;
        uint32_t factBS  = isTailBlock_ ? tailBlockubFactorBS_       : ubFactorBS_;
        uint32_t tailBS  = isTailBlock_ ? tailBlockubTailFactorBS_   : ubTailFactorBS_;
        // Dim 方向不切分，所有核处理完整 dim
        uint32_t loopDim = loopNumDim_;
        uint32_t factDim = ubFactorDim_;
        uint32_t tailDim = ubTailFactorDim_;

        // Debug: 打印 ProcessMainComputeBS 调用参数
        //PRINTF("[Process] blockIdx=%u, blockIndex=0, calling ProcessMainComputeBS: bsStart=%lu, loopBS=%u, factBS=%u, tailBS=%u, loopDim=%u, factDim=%u, tailDim=%u\n",blockIdx, bsStart, loopBS, factBS, tailBS, loopDim, factDim, tailDim);

        ProcessMainComputeBS(bsStart, loopBS, factBS, tailBS, loopDim, factDim, tailDim);
    } else {
        // 切 Dim 轴：BS 方向不切分，Dim 方向有主尾核之分
        uint64_t dimStart = (uint64_t)blockIdx * blockFactor_;

        // BS 方向不切分，所有核处理全部 BS
        uint32_t loopBS  = loopNumBS_;
        uint32_t factBS  = ubFactorBS_;
        uint32_t tailBS  = ubTailFactorBS_;
        // Dim 方向有主尾核之分
        uint32_t loopDim = isTailBlock_ ? tailBlockloopNumDim_       : loopNumDim_;
        uint32_t factDim = isTailBlock_ ? tailBlockubFactorDim_      : ubFactorDim_;
        uint32_t tailDim = isTailBlock_ ? tailBlockubTailFactorDim_  : ubTailFactorDim_;

        ProcessMainComputeDim(dimStart, loopBS, factBS, tailBS, loopDim, factDim, tailDim);
    }

    // 3. 全核同步（保证所有核完成计算、workspace 写入完毕后再更新 cacheStates）
    SyncAll();

    // 4. 从 workspace 回写 cache state
    WriteCacheFromWorkspace();
    // if(blockIdx == realCoreNum_ - 1){
    //     for(uint32_t i = 0; i < 8; i++) {
    //         for(uint32_t j = 0; j < 256; j++){
    //             uint32_t index = i * 256 + j;
    //             //PRINTF("y[%u][%u] = %f", i, j, yGM_(index));
    //         }
    //     }
    // }
}

} // namespace CausalConv1dFnNs

#endif // CAUSAL_CONV1D_FN_H
