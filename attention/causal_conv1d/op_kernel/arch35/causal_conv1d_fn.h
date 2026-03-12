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
    // 主计算：二维切分场景（同时切 BS 和 Dim）
    __aicore__ inline void ProcessMainCompute(
        uint64_t bsStart, uint64_t dimStart,
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

    // 核内切分参数（主核）
    uint32_t loopNumBS_;
    uint32_t loopNumDim_;
    uint32_t ubFactorBS_;
    uint32_t ubTailFactorBS_;
    uint32_t ubFactorDim_;
    uint32_t ubTailFactorDim_;

    // 核内切分参数（尾核）
    uint32_t tailBlockloopNumBS_;
    uint32_t tailBlockloopNumDim_;
    uint32_t tailBlockubFactorBS_;
    uint32_t tailBlockubTailFactorBS_;
    uint32_t tailBlockubFactorDim_;
    uint32_t tailBlockubTailFactorDim_;

    // dim 方向核间切分
    uint32_t dimCoreNum_;            // dim 方向总核数
    uint32_t dimRemainderCores_;     // dim 方向前多少个核是主核
    uint32_t dimBlockFactor_;        // dim 方向主核处理的大小
    uint32_t dimBlockTailFactor_;    // dim 方向尾核处理的大小

    // BS 方向核间切分
    uint32_t bsCoreNum_;             // BS 方向总核数
    uint32_t bsRemainderCores_;      // BS 方向前多少个核是主核
    uint32_t bsBlockFactor_;         // BS 方向主核处理的长度（含 overlap）
    uint32_t bsBlockTailFactor_;     // BS 方向尾核处理的长度

    uint32_t realCoreNum_;

    // stride（跨 sequence 的步长）
    uint32_t xStride_;           // x 的行 stride（>= dim_）
    uint32_t cacheStride_;       // cacheStates 的行 stride（>= dim_）
    uint32_t residualConnection_; // 是否加残差：0-不需要，1-需要

    // 有效 batch 范围（tiling 已过滤掉头尾的无效 batch）
    uint32_t validBatchStart_;   // 有效 batch 的起始索引
    uint32_t validBatchCount_;   // 有效 batch 的数量
    uint32_t validBatchEnd_;     // 有效 batch 的结束索引（validBatchStart_ + validBatchCount_）
    uint32_t validSeqStart_;     // 有效序列在 x 中的起始位置
    uint32_t validSeqLen_;       // 有效序列的总长度

    // 运行时辅助：当前核的二维索引
    uint32_t bsIdx_;             // BS 方向的核索引
    uint32_t dimIdx_;            // Dim 方向的核索引
    bool     isBsTailCore_;      // BS 方向是否为尾核（idx >= bsRemainderCores）
    bool     isDimTailCore_;     // Dim 方向是否为尾核（idx >= dimRemainderCores）
    bool     initialStateModeNull_;  // initialStateMode 是否为空指针（None），为空时按 hasInitState=2 处理
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

    // 核内切分参数（主核）
    loopNumBS_                 = tiling->loopNumBS;
    loopNumDim_                = tiling->loopNumDim;
    ubFactorBS_                = tiling->ubFactorBS;
    ubTailFactorBS_            = tiling->ubTailFactorBS;
    ubFactorDim_               = tiling->ubFactorDim;
    ubTailFactorDim_           = tiling->ubTailFactorDim;

    // 核内切分参数（尾核）
    tailBlockloopNumBS_        = tiling->tailBlockloopNumBS;
    tailBlockloopNumDim_       = tiling->tailBlockloopNumDim;
    tailBlockubFactorBS_       = tiling->tailBlockubFactorBS;
    tailBlockubTailFactorBS_   = tiling->tailBlockubTailFactorBS;
    tailBlockubFactorDim_      = tiling->tailBlockubFactorDim;
    tailBlockubTailFactorDim_  = tiling->tailBlockubTailFactorDim;

    // dim 方向核间切分
    dimCoreNum_                = tiling->dimCoreNum;
    dimRemainderCores_         = tiling->dimRemainderCores;
    dimBlockFactor_            = tiling->dimBlockFactor;
    dimBlockTailFactor_        = tiling->dimBlockTailFactor;

    // BS 方向核间切分
    bsCoreNum_                 = tiling->bsCoreNum;
    bsRemainderCores_          = tiling->bsRemainderCores;
    bsBlockFactor_             = tiling->bsBlockFactor;
    bsBlockTailFactor_         = tiling->bsBlockTailFactor;

    realCoreNum_               = tiling->realCoreNum;

    // stride（x 和 cacheStates 非连续存储）
    xStride_                   = tiling->xStride;
    cacheStride_               = tiling->cacheStride;
    residualConnection_        = tiling->residualConnection;

    // 有效 batch 范围
    validBatchStart_           = tiling->validBatchStart;
    validBatchCount_           = tiling->validBatchCount;
    validBatchEnd_             = validBatchStart_ + validBatchCount_;
    validSeqStart_             = tiling->validSeqStart;
    validSeqLen_               = tiling->validSeqLen;

    // 计算当前核的二维索引
    uint32_t blockIdx = GetBlockIdx();
    bsIdx_  = blockIdx / dimCoreNum_;     // BS 方向的核索引
    dimIdx_ = blockIdx % dimCoreNum_;     // Dim 方向的核索引

    // 判断当前核在两个方向是主核还是尾核
    isBsTailCore_  = (bsIdx_  >= bsRemainderCores_);
    isDimTailCore_ = (dimIdx_ >= dimRemainderCores_);

    initialStateModeNull_ = (initialStateMode == nullptr);  // 检查是否为空指针

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
    if (!initialStateModeNull_) {
        hasInitialStateGM_.SetGlobalBuffer((__gm__ int32_t*)initialStateMode, batchSize_);
    }
    yGM_.SetGlobalBuffer((__gm__ T*)y, cuSeqLen_ * dim_);
    if (workspace != nullptr) {
        GM_ADDR userWS = GetUserWorkspace(workspace);
        if (userWS != nullptr) {
            // workspace 按 bsIdx 分配，每个 bsIdx 对应 (K-1) * dim_ 的空间
            workspaceGM_.SetGlobalBuffer((__gm__ T*)userWS, bsCoreNum_ * (kernelWidth_ - 1) * dim_);
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
    // 如果 initialStateMode 为空（None），则填充 2（按 hasInitState=2 的逻辑处理）
    {
        LocalTensor<int32_t> tmp = hasInitInQueue_.AllocTensor<int32_t>();
        if (initialStateModeNull_) {
            // initialStateMode 为 None，填充 2
            Duplicate(tmp, (int32_t)2, batchSize_);
        } else {
            // 从 GM 加载
            DataCopyExtParams cpParams{1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(tmp, hasInitialStateGM_[0], cpParams, padParams);
        }
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
        // PipeBarrier<PIPE_ALL>();
        // PipeBarrier<PIPE_ALL>();

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
            Conv1dNeedState(xSlice, weightLocal, stateSlice, stateSlice, stateSLen, xSLen, dimSize, residualConnection_);
            // 每次 VF 计算后同步，确保写入完成
        }
        PipeBarrier<PIPE_ALL>();
        // 写回 y
        uint64_t yGmOffset = (batchStart + curSequenceIdx) * dim_ + dimStart;
        //PRINTF("[ProcessTokensNeedCache] y DataCopy: blockCount=%u, blockLen=%u, dstStride=%u, yGmOffset=%lu, cacheLocalOffset=%u\n",step, dimBlocks * ALIGN_BYTES, ySkipBlocks * ALIGN_BYTES, yGmOffset, curSequenceIdx * dimSize);
        DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                            0, ySkipBlocks * ALIGN_BYTES, 0};
        DataCopyPad(yGM_[yGmOffset],
                    cacheLocal[curSequenceIdx * dimSize], ycp);
    } else {
        // hasInitState == 2，跳过卷积
        if (residualConnection_ == 1) {
            // 需要残差连接，直接把 x 搬到 yGM（y = x）
            uint64_t yGmOffset = (batchStart + curSequenceIdx) * dim_ + dimStart;
            DataCopyExtParams ycp{step, static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                                  0, ySkipBlocks * ALIGN_BYTES, 0};
            DataCopyPad(yGM_[yGmOffset], xLocal[i * dimSize], ycp);
        }
        // 如果 residualConnection_ == 0，cacheLocal 已填充 0，后面搬到 yGM 就是 0
    }
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
    // for (uint32_t j = 0; j < step; j++) {
    //     uint32_t tokenIdx = i + j;
    //     uint32_t xStartIdx = tokenIdx - (K - 1);  // 卷积窗口起始位置
    //     //PRINTF("[ProcessTokensNoCache] Conv j=%u, tokenIdx=%u, xStartIdx=%u, xSlice offset=%u, ySlice offset=%u\n",j, tokenIdx, xStartIdx, xStartIdx * dimSize, tokenIdx * dimSize);
    //     LocalTensor<T> xSlice = xLocal[xStartIdx * dimSize];  // 输入：卷积窗口起始
    //     LocalTensor<T> ySlice = xLocal[xStartIdx * dimSize];   // 输出：当前 token 位置
    //     Conv1dNoNeedState(xSlice, weightLocal, ySlice, K, dimSize);
    //     // 每次 VF 计算后同步，确保写入完成
    // }
    uint32_t xStartIdx = i - (K - 1);
    LocalTensor<T> xSlice = xLocal[xStartIdx * dimSize];  // 输入：卷积窗口起始
    LocalTensor<T> ySlice = xLocal[xStartIdx * dimSize];   // 输出：当前 token 位置
    Conv1dNoNeedState(xSlice, weightLocal, ySlice, step, dimSize, residualConnection_);
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
        // workspace 按 bsIdx_ 分配，加上 dimStart 偏移
        uint64_t wsOffset = (uint64_t)bsIdx_ * (K - 1) * dim_ + dimStart;
        // workspace 每行 dim_ 个元素，当前只写 dimSize 个，dstStride 跳过剩余部分
        uint32_t wsDstSkip = (dim_ - dimSize) * sizeof(T);

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
        // workspace 按 bsIdx_ 分配，加上 dimStart 偏移
        uint64_t wsOffset = (uint64_t)bsIdx_ * (K - 1) * dim_ + dimStart;
        // workspace 每行 dim_ 个元素，当前只写 dimSize 个，dstStride 跳过剩余部分
        uint32_t wsDstSkip = (dim_ - dimSize) * sizeof(T);
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
// ProcessMainCompute：统一的二维切分计算逻辑
// bsStart: 当前核在 BS 方向的起始位置（相对于有效序列）
// dimStart: 当前核在 Dim 方向的起始位置
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::ProcessMainCompute(
    uint64_t bsStart, uint64_t dimStart,
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
    // 判断第一个 batch 是否完整：bsIdx_ == 0 时肯定完整，否则检查 globalBsStart == batch 起始位置
    uint64_t firstBatchStart = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
    firstBatchComplete_   = (bsIdx_ == 0) || (globalBsStart == firstBatchStart);
    firstBatchWrittenToWS_ = false;

    // iStart：只有 bsIdx_ == 0 的核的第一个 BS 循环从 0 开始，其余从 K-1 开始
    bool isBsFirstCore = (bsIdx_ == 0);
    uint32_t iStart = isBsFirstCore ? 0 : (K - 1);

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

        uint32_t dimOff = (uint32_t)dimStart;
        uint32_t batchIdxForDim = curBatchIdx;
        uint32_t seqIdxForDim = curSequenceIdx;
        for (uint32_t dimLoop = 0; dimLoop < loopNumDim; dimLoop++) {
            uint32_t curDim = (dimLoop == loopNumDim - 1) ? ubTailFactorDim : ubFactorDim;
            // dim 循环处理同一块 BS 数据的不同 dim 切片，每次传入相同的初始 batch 状态
            batchIdxForDim = curBatchIdx;
            seqIdxForDim = curSequenceIdx;
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
// WriteCacheFromWorkspace：SyncAll 后，从 workspace 回写 cacheStates
//
// 新设计（二维切分）：
// - workspace 按 bsIdx_ 分配，每个 bsIdx_ 对应 (K-1) 行，每行 dim_ 个元素
// - 同一个 bsIdx_ 的不同 dimIdx_ 核写同一个 workspace 区域的不同 dim 切片
// - 每个核只回写自己负责的 dim 切片
// ============================================================================
template <typename T>
__aicore__ inline void CausalConv1dFn<T>::WriteCacheFromWorkspace()
{
    // 如果没有写 workspace，直接返回
    if (!firstBatchWrittenToWS_) {
        return;
    }

    uint32_t K        = kernelWidth_;
    uint32_t rows     = K - 1;

    // 计算当前核负责的 dim 范围
    uint32_t dimStart, dimSize;
    if (dimIdx_ < dimRemainderCores_) {
        dimStart = dimIdx_ * dimBlockFactor_;
        dimSize  = dimBlockFactor_;
    } else {
        dimStart = dimRemainderCores_ * dimBlockFactor_ + (dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
        dimSize  = dimBlockTailFactor_;
    }

    // 每行实际数据的字节数
    uint32_t rowBytes = dimSize * sizeof(T);
    uint32_t rowBlocks = rowBytes / ALIGN_BYTES;
    // cacheStates 的行间跳过块数
    uint32_t cacheSkip = (cacheStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    // workspace 的行间跳过块数（workspace 每行 dim_ 个元素，只取 dimSize 个）
    uint32_t wsSkip = (dim_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    // workspace 按 bsIdx_ 索引，加上 dimStart 偏移
    uint64_t wsOff = (uint64_t)bsIdx_ * rows * dim_ + dimStart;
    // cacheStates 使用 firstBatchCIdx_
    uint64_t csOff = (uint64_t)firstBatchCIdx_ * rows * cacheStride_ + dimStart;

    // 直接 GM→GM 搬运（通过 UB 中转）
    LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();
    {
        DataCopyExtParams cp{static_cast<uint16_t>(rows),
                             static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                             wsSkip * ALIGN_BYTES, 0, 0};  // workspace 非连续（srcStride），UB 连续
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
                             0, cacheSkip * ALIGN_BYTES, 0};  // UB 连续，cacheStates 非连续
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

    // 2. 计算当前核在 BS 方向的起始位置和处理长度
    uint32_t K = kernelWidth_;
    uint64_t bsStart;
    if (bsIdx_ < bsRemainderCores_) {
        // 主核：前 bsRemainderCores_ 个核
        // 有效步长 = bsBlockFactor_ - (K-1)
        uint64_t effectiveStep = bsBlockFactor_ - (K - 1);
        bsStart = (uint64_t)bsIdx_ * effectiveStep;
    } else {
        // 尾核：后面的核
        // 前面主核贡献的总有效长度
        uint64_t effectiveStepMain = bsBlockFactor_ - (K - 1);
        uint64_t effectiveStepTail = bsBlockTailFactor_ - (K - 1);
        bsStart = (uint64_t)bsRemainderCores_ * effectiveStepMain +
                  (uint64_t)(bsIdx_ - bsRemainderCores_) * effectiveStepTail;
    }

    // 3. 计算当前核在 Dim 方向的起始位置
    uint64_t dimStart;
    if (dimIdx_ < dimRemainderCores_) {
        // 主核
        dimStart = (uint64_t)dimIdx_ * dimBlockFactor_;
    } else {
        // 尾核
        dimStart = (uint64_t)dimRemainderCores_ * dimBlockFactor_ +
                   (uint64_t)(dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
    }

    // 4. 选择核内切分参数
    // BS 方向：主核（bsIdx_ < bsRemainderCores_）用主核参数，尾核用尾核参数
    uint32_t loopBS  = isBsTailCore_  ? tailBlockloopNumBS_        : loopNumBS_;
    uint32_t factBS  = isBsTailCore_  ? tailBlockubFactorBS_       : ubFactorBS_;
    uint32_t tailBS  = isBsTailCore_  ? tailBlockubTailFactorBS_   : ubTailFactorBS_;

    // Dim 方向：主核（dimIdx_ < dimRemainderCores_）用主核参数，尾核用尾核参数
    uint32_t loopDim = isDimTailCore_ ? tailBlockloopNumDim_       : loopNumDim_;
    uint32_t factDim = isDimTailCore_ ? tailBlockubFactorDim_      : ubFactorDim_;
    uint32_t tailDim = isDimTailCore_ ? tailBlockubTailFactorDim_  : ubTailFactorDim_;

    // 5. 调用统一的计算逻辑
    ProcessMainCompute(bsStart, dimStart, loopBS, factBS, tailBS, loopDim, factDim, tailDim);

    // 6. 全核同步（保证所有核完成计算、workspace 写入完毕后再更新 cacheStates）
    SyncAll();

    // 7. 从 workspace 回写 cache state
    WriteCacheFromWorkspace();
}

} // namespace CausalConv1dFnNs

#endif // CAUSAL_CONV1D_FN_H
