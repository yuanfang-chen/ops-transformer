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
 * \file causal_conv1d_update.h
 * \brief CausalConv1dUpdate kernel implementation
 *
 * 本文件实现了causal_conv1d_update算子的核函数，用于执行因果一维卷积并更新缓存状态。
 *
 * 算子功能：
 * 1. 对输入序列执行因果1D卷积（每个特征通道独立）
 * 2. 根据acceptTokenNum动态使用历史cache state和当前输入x
 * 3. 计算完成后自动更新cache state，确保后续推理能正确延续上下文
 * 4. 保证输出满足因果性约束（输出位置j只能看到输入位置0到j）
 *
 * 主要特点：
 * - 支持FP16/BF16数据类型
 * - 使用双Buffer机制提高数据搬运和计算的并行度
 * - 支持核间切分（只切batch维度）和核内切分（BS和Dim方向）
 * - 针对序列位置<K-1和>=K-1采用不同的计算策略
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "kernel_operator.h"
#include "../../op_host/causal_conv1d_update_tiling_arch35.h"
#include "./vf/compute.h"

using namespace AscendC;

// ========== 常量定义 ==========
constexpr int32_t BUFFER_NUM = 2;           // 双Buffer数量，用于重叠数据搬运和计算
constexpr int32_t ALIGN_BYTES = 32;      // 32字节对齐，用于DataCopyParams的stride和对齐计算

// TilingData结构定义在Host侧：op_host/causal_conv1d_update_tiling_arch35.h
// 核间切分策略：二维切分（Dim方向 × Batch方向）
// - Dim方向：256B对齐分割，dimChunkSize * (dimCoreCnt-1) + dimTailSize = dim
// - Batch方向：按照有效batch范围分配，支持过滤无效batch

// ========== 核函数主类 ==========
/**
 * @brief CausalConv1dUpdate核函数实现类
 * @tparam T 数据类型（half或bfloat16）
 *
 * 该类实现了因果1D卷积的完整计算流程：
 * 1. Init(): 初始化Global Memory指针、分配UB队列和TBuf
 * 2. Process(): 执行双重循环（BS方向×Dim方向），每次迭代处理一个UB块
 * 3. CopyIn/Compute/CopyOut: 三阶段流水线处理
 *
 * 内存布局：
 * - xQueue: 存储输入x数据 [batchNum, seqLen, currentDim]
 * - weightQueue: 存储卷积核 [K, currentDim]
 * - cacheQueue: 存储历史cache state [K-1+m, currentDim]
 * - indicesQueue: 存储cache索引 [batchNum]
 * - acceptTokenQueue: 存储接受的token数 [batchNum]
 * - yQueue: 存储输出y数据（复用xQueue的buffer）
 */
template <typename T>
class CausalConv1dUpdateKernel {
public:
    __aicore__ inline CausalConv1dUpdateKernel() {}
    __aicore__ inline ~CausalConv1dUpdateKernel() {}

    /**
     * @brief 初始化函数，设置所有Global Memory指针并分配UB资源
     * @param x 输入序列 [batch, m+1, dim]
     * @param weight 卷积核 [K, dim]
     * @param cacheState 输入的cache state [-1, K-1+m, dim]
     * @param cacheIndices cache索引 [batch]，指定每个batch对应的cache state位置
     * @param acceptTokenNum 接受的token数量 [batch]，可选输入
     * @param queryStartLoc query起始位置 [batch+1]
     * @param y 输出序列 [batch, m+1, dim]
     * @param outputCacheState 输出的cache state，原地更新
     * @param tilingData tiling参数结构体指针
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR cacheState,
                                GM_ADDR cacheIndices, GM_ADDR acceptTokenNum,
                                GM_ADDR queryStartLoc,
                                GM_ADDR y, CausalConv1dUpdateTilingData* tilingData);

    /**
     * @brief 主处理函数，执行双重循环处理所有数据
     *
     * 处理流程：
     * for loopBS in [0, loopNumBS):     # BS方向循环
     *     for j in [0, loopNumDim): # Dim方向循环
     *         CopyIn(loopBS, j)          # 从GM搬入数据到UB
     *         Compute(loopBS, j)         # 在UB上执行计算
     *         CopyOut(loopBS, j)         # 从UB搬出结果到GM
     */
    __aicore__ inline void Process();

private:
    // ========== 三阶段流水线函数 ==========

    /**
     * @brief 数据搬入阶段：从Global Memory拷贝数据到UB
     * @param batchLoop Batch方向的循环索引
     * @param dimLoop Dim方向的循环索引
     * @param indicesLocal cache索引的LocalTensor
     * @param acceptTokenLocal accept token数量的LocalTensor
     * @param queryStartLocLocal query起始位置的LocalTensor
     */
    __aicore__ inline void CopyIn(int32_t batchLoop, int32_t dimLoop);

    /**
     * @brief 计算阶段：在UB上执行因果1D卷积
     * @param batchLoop Batch方向的循环索引
     * @param dimLoop Dim方向的循环索引
     * @param indicesLocal cache索引的LocalTensor
     * @param acceptTokenLocal accept token数量的LocalTensor
     * @param queryStartLocLocal query起始位置的LocalTensor
     */
    __aicore__ inline void Compute(int32_t batchLoop, int32_t dimLoop,
                                    const LocalTensor<int32_t>& indicesLocal,
                                    const LocalTensor<int32_t>& acceptTokenLocal,
                                    const LocalTensor<int32_t>& queryStartLocLocal);

    /**
     * @brief 数据搬出阶段：从UB拷贝结果到Global Memory
     * @param batchLoop Batch方向的循环索引
     * @param dimLoop Dim方向的循环索引
     */
    __aicore__ inline void CopyOut(int32_t batchLoop, int32_t dimLoop);

    // ========== 核心计算函数 ==========

    /**
     * @brief 更新cache state到Global Memory
     * @param xLocal 输入x的LocalTensor
     * @param cacheLocal cache state的LocalTensor
     * @param acceptTokenLocal accept token数量的LocalTensor
     * @param batchInBlock 当前UB块内的batch索引
     * @param curBatchIdx 全局batch索引
     * @param stateOffset cache state的起始地址（绝对元素偏移量）
     * @param dimInnerOffset 当前核内的dim偏移
     * @param dimSize 当前处理的dim维度大小
     * @param curSeqLen 当前batch的实际序列长度
     */
    __aicore__ inline void UpdateCacheState(const LocalTensor<T>& xLocal, const LocalTensor<T>& cacheLocal,
                                             const LocalTensor<int32_t>& acceptTokenLocal,
                                             int32_t batchInBlock, int32_t curBatchIdx,
                                             int64_t stateOffset, int32_t dimInnerOffset,
                                             int32_t dimSize, int32_t curSeqLen);


    /**
     * @brief 在tensor中执行需要cache state的卷积计算（序列位置 < K-1）
     * @param xLocal 输入x的LocalTensor
     * @param weightLocal 卷积核的LocalTensor
     * @param cacheLocal cache state的LocalTensor
     * @param yLocal 输出y的LocalTensor
     * @param xOffset x中的偏移（元素数）
     * @param stateSLen 使用的state长度
     * @param xSLen 使用的x长度
     * @param dimSize 当前处理的dim维度大小
     */

    // ========== Global Memory指针 ==========
    GlobalTensor<T> xGm;                    // 输入序列 [batch, seqLen, dim]
    GlobalTensor<T> weightGm;               // 卷积核 [K, dim]
    GlobalTensor<T> cacheStateGm;           // 输入cache state [-1, K-1+m, dim]
    GlobalTensor<int32_t> cacheIndicesGm;   // cache索引 [batch]
    GlobalTensor<int32_t> acceptTokenNumGm; // 接受的token数 [batch]
    GlobalTensor<int32_t> queryStartLocGm;  // query起始位置 [batch+1]
    GlobalTensor<T>  ;                    // 输出序列 [batch, seqLen, dim]

    // ========== UB队列（Unified Buffer中的数据缓存） ==========
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;        // 输入x队列（双Buffer）
    TQue<QuePosition::VECIN, 1> weightQueue;            // 卷积核队列（单Buffer）
    TQue<QuePosition::VECIN, 1> cacheQueue;             // cache state队列（单Buffer）
    TQue<QuePosition::VECIN, 1> indicesQueue;           // cache索引队列（单Buffer）
    TQue<QuePosition::VECIN, 1> acceptTokenQueue;       // accept token队列（单Buffer）
    TQue<QuePosition::VECIN, 1> queryStartLocQueue;     // query起始位置队列（单Buffer）
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;       // 输出y队列（双Buffer，复用xQueue）

    TPipe pipe;                             // Pipeline管理对象

    // ========== Tiling参数（二维切分：Dim方向 × Batch方向） ==========
    // 核间切分参数
    int64_t usedCoreNum_;           // 总共使用的核数
    int64_t dimCoreCnt_;            // Dim方向的核数
    int64_t batchCoreCnt_;          // Batch方向的核数
    int64_t dimChunkSize_;          // 每个核处理的Dim大小（256B对齐）
    int64_t dimTailSize_;           // Dim尾核处理的大小
    int64_t batchPerCore_;          // 每个核处理的batch数
    int64_t batchTailPerCore_;      // Batch尾核处理的batch数
    int64_t validBatchStart_;       // 有效batch起始索引
    int64_t validBatchEnd_;         // 有效batch结束索引（包含）

    // 核内UB切分参数
    int64_t ubBatchSize_;           // UB循环每次处理的batch数
    int64_t ubDimSize_;             // UB循环每次处理的Dim大小（元素数）
    int64_t batchLoopCnt_;          // Batch方向的循环次数
    int64_t dimLoopCnt_;            // Dim方向的循环次数

    // ========== Shape参数 ==========
    int64_t batchSize_;      // Batch大小
    int64_t seqLen_;         // 序列长度（m+1）
    int64_t cuSeqLen_;       // 累积序列长度（2D输入用）
    int64_t dim_;            // 特征维度
    int64_t kernelSize_;     // 卷积核大小（K=3）
    int64_t cacheLen_;       // cache_state第二维
    int64_t xInputMode_;     // 输入模式：0=3D, 1=2D
    int32_t xStride_;
    int32_t cacheStride_;

    // ========== 运行时计算参数 ==========
    int32_t blockIdx_;           // 当前核的索引
    int32_t batchIdx_;           // 当前核在Batch维度的索引
    int32_t dimIdx_;             // 当前核在Dim维度的索引
    int32_t firstBatchIdx_;      // 当前核处理的起始batch索引
    int32_t currentBatchNum_;    // 当前核处理的batch数量
    int32_t dimOffset_;          // 当前核处理的Dim起始偏移（元素）
    int32_t currentDimSize_;     // 当前核处理的Dim大小
    bool hasAcceptTokenNum_;     // 是否提供了acceptTokenNum输入
    int32_t dimSum_;
    int32_t cacheLenSum_;
    
};

// ==================== 函数实现 ====================

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR cacheState,
    GM_ADDR cacheIndices, GM_ADDR acceptTokenNum,
    GM_ADDR queryStartLoc, GM_ADDR y, CausalConv1dUpdateTilingData* tilingData)
{
    // === 1. 获取核间切分参数（二维：Dim方向 × Batch方向） ===
    usedCoreNum_ = tilingData->usedCoreNum;
    dimCoreCnt_ = tilingData->dimCoreCnt;
    batchCoreCnt_ = tilingData->batchCoreCnt;
    dimChunkSize_ = tilingData->dimChunkSize;
    dimTailSize_ = tilingData->dimTailSize;
    batchPerCore_ = tilingData->batchPerCore;
    batchTailPerCore_ = tilingData->batchTailPerCore;
    validBatchStart_ = tilingData->validBatchStart;
    validBatchEnd_ = tilingData->validBatchEnd;

    // === 2. 获取核内UB切分参数 ===
    ubBatchSize_ = tilingData->ubBatchSize;
    ubDimSize_ = tilingData->ubDimSize;
    batchLoopCnt_ = tilingData->batchLoopCnt;
    dimLoopCnt_ = tilingData->dimLoopCnt;

    // === 3. 获取shape参数 ===
    batchSize_ = tilingData->batchSize;
    seqLen_ = tilingData->seqLen;
    cuSeqLen_ = tilingData->cuSeqLen;
    dim_ = tilingData->dim;
    kernelSize_ = tilingData->kernelSize;
    cacheLen_ = tilingData->cacheLen;
    hasAcceptTokenNum_ = tilingData->hasAcceptTokenNum;
    xInputMode_ = tilingData->xInputMode;
    xStride_ = tilingData->xStride;
    cacheStride_ = tilingData->cacheStride;

    // === 4. 计算当前核在二维grid中的索引 ===
    blockIdx_ = GetBlockIdx();
    batchIdx_ = blockIdx_ / dimCoreCnt_;  // Batch方向索引
    dimIdx_ = blockIdx_ % dimCoreCnt_;    // Dim方向索引

    // === 5. 计算当前核处理的Dim范围 ===
    dimSum_ = dim_ + xStride_;
    cacheLenSum_ = dim_ + cacheStride_;
    dimOffset_ = dimIdx_ * dimChunkSize_;  // Dim起始偏移
    // 判断是否为Dim方向尾核
    if (dimIdx_ == dimCoreCnt_ - 1) {
        currentDimSize_ = dimTailSize_;   // 尾核使用tail大小
    } else {
        currentDimSize_ = dimChunkSize_;  // 常规核使用chunk大小
    }

    // === 6. 计算当前核处理的Batch范围 ===
    // 从有效batch起始位置开始分配
    int64_t validBatchCount = validBatchEnd_ - validBatchStart_ + 1;
    int64_t batchStartOffset = validBatchStart_;

    firstBatchIdx_ = batchStartOffset + batchIdx_ * static_cast<int32_t>(batchPerCore_);
    // 判断是否为Batch方向尾核
    if (batchIdx_ == static_cast<int32_t>(batchCoreCnt_) - 1) {
        currentBatchNum_ = static_cast<int32_t>(batchTailPerCore_);
    } else {
        currentBatchNum_ = static_cast<int32_t>(batchPerCore_);
    }

    // 确保不超出有效batch范围
    int32_t maxBatchIdx = batchStartOffset + validBatchCount;
    if (firstBatchIdx_ + currentBatchNum_ > maxBatchIdx) {
        currentBatchNum_ = maxBatchIdx - firstBatchIdx_;
    }

    // === 7. 设置Global Memory buffers ===
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    weightGm.SetGlobalBuffer((__gm__ T*)weight);
    cacheStateGm.SetGlobalBuffer((__gm__ T*)cacheState);
    cacheIndicesGm.SetGlobalBuffer((__gm__ int32_t*)cacheIndices);
    if (hasAcceptTokenNum_) {
        acceptTokenNumGm.SetGlobalBuffer((__gm__ int32_t*)acceptTokenNum);
    }
    if(xInputMode_ == 1) {
        queryStartLocGm.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc);
    }

    // === 8. 初始化UB队列 ===

    // xQueue: 存储输入x数据（y复用此buffer）
    // 大小: ubBatchSize * seqLen * ubDimSize
    int32_t xQueueSize = ubBatchSize_ * seqLen_ * ubDimSize_ * sizeof(T);
    pipe.InitBuffer(xQueue, BUFFER_NUM, xQueueSize);

    // cacheQueue: 存储cache state
    // 大小: (K-1+seqLen-1) * ubDimSize = (K + seqLen - 2) * ubDimSize
    int32_t cacheQueueSize = cacheLen_ * ubDimSize_ * sizeof(T);
    pipe.InitBuffer(cacheQueue, 1, cacheQueueSize);

    // weightQueue: 存储卷积核 [K, ubDimSize]
    int32_t weightQueueSize = kernelSize_ * ubDimSize_ * sizeof(T);
    pipe.InitBuffer(weightQueue, 1, weightQueueSize);

    // indicesQueue: 存储cache索引
    int32_t indicesQueueSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    pipe.InitBuffer(indicesQueue, 1, indicesQueueSize);

    // acceptTokenQueue: 存储accept token数量
    int32_t acceptTokenQueueSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    pipe.InitBuffer(acceptTokenQueue, 1, acceptTokenQueueSize);
    // queryStartLocQueue: 只有二维TH的时候，存储query起始位置
    if(xInputMode_) {
        int32_t queryStartLocQueueSize = ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe.InitBuffer(queryStartLocQueue, 1, queryStartLocQueueSize);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::Process()
{
    // 预先搬运cache indices、accept token numbers和query start loc到UB
    // 这些是常驻数据，整个Process过程中都需要访问
    LocalTensor<int32_t> indicesLocal = indicesQueue.AllocTensor<int32_t>();
    LocalTensor<int32_t> acceptTokenLocal = acceptTokenQueue.AllocTensor<int32_t>();
    LocalTensor<int32_t> queryStartLocLocal = queryStartLocQueue.AllocTensor<int32_t>();

    DataCopyPadParams padParams{{false, 0, 0, 0}};

    // 拷贝cache indices：需要所有batch的信息
    uint16_t indicesBlockLen = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    DataCopyParams indicesCopyParams;
    indicesCopyParams.blockCount = 1;
    indicesCopyParams.blockLen = indicesBlockLen;
    indicesCopyParams.srcStride = 0;
    indicesCopyParams.dstStride = 0;
    DataCopyPad(indicesLocal, cacheIndicesGm, indicesCopyParams, padParams);

    // 拷贝accept token numbers
    Duplicate(acceptTokenLocal, static_cast<int32_t>(1), batchSize_);
    if (hasAcceptTokenNum_) {
        DataCopyPad(acceptTokenLocal, acceptTokenNumGm, indicesCopyParams, padParams);
    }

    // 拷贝query start loc：[batch+1]
    if(xInputMode_ == 1) {
        uint16_t queryStartLocBlockLen = ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        DataCopyParams queryStartLocCopyParams;
        queryStartLocCopyParams.blockCount = 1;
        queryStartLocCopyParams.blockLen = queryStartLocBlockLen;
        queryStartLocCopyParams.srcStride = 0;
        queryStartLocCopyParams.dstStride = 0;
        DataCopyPad(queryStartLocLocal, queryStartLocGm, queryStartLocCopyParams, padParams);
    }

    // === 双重循环处理所有数据 ===
    // 新tiling策略：外层Batch方向循环，内层Dim方向循环
    // 当前核只处理固定的Dim范围（dimOffset_ ~ dimOffset_+currentDimSize_）
    // 和固定的Batch范围（firstBatchIdx_ ~ firstBatchIdx_+currentBatchNum_）
    for (int32_t batchLoop = 0; batchLoop < batchLoopCnt_; batchLoop++) {
        for (int32_t dimLoop = 0; dimLoop < dimLoopCnt_; dimLoop++) {
            // 三阶段流水线：CopyIn -> Compute -> CopyOut
            CopyIn(batchLoop, dimLoop);
            Compute(batchLoop, dimLoop, indicesLocal, acceptTokenLocal, queryStartLocLocal);
        }
    }

    // 释放indices、acceptToken和queryStartLoc tensors
    indicesQueue.FreeTensor(indicesLocal);
    acceptTokenQueue.FreeTensor(acceptTokenLocal);
    queryStartLocQueue.FreeTensor(queryStartLocLocal);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::CopyIn(
    int32_t batchLoop, int32_t dimLoop)
{
    // === 1. 计算当前循环处理的batch数和dim大小 ===
    int32_t batchNumInLoop = (batchLoop == batchLoopCnt_ - 1) ? (currentBatchNum_ - batchLoop * ubBatchSize_) : ubBatchSize_;
    int32_t dimSizeInLoop = (dimLoop == dimLoopCnt_ - 1) ? (currentDimSize_ - dimLoop * ubDimSize_) : ubDimSize_;
    
    // === 2. 从队列中获取tensors ===
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    LocalTensor<T> weightLocal = weightQueue.AllocTensor<T>();

    DataCopyPadParams padParams{{false, 0, 0, 0}};

    // === 3. 计算当前循环的dim偏移（当前核内的相对偏移） ===
    int32_t dimInnerOffset = dimLoop * static_cast<int32_t>(ubDimSize_);

    // === 4. 拷贝x数据：[batchNumInLoop, seqLen, dimSizeInLoop] ===
    // 全局batch索引
    int32_t startBatchIdx = firstBatchIdx_ + batchLoop * ubBatchSize_;

    // 计算GM中的全局偏移
    // 对于3D输入: [batch, seq_len, dim]
    // GM偏移 = startBatchIdx * seqLen_ * dim_ + dimOffset_ + dimInnerOffset
    int32_t xOffset = startBatchIdx * seqLen_ * dimSum_ + dimOffset_ + dimInnerOffset;

    uint16_t blockLen = dimSizeInLoop * sizeof(T);
    uint16_t srcStrideBytes = (dimSum_ - dimSizeInLoop) * sizeof(T);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = batchNumInLoop * seqLen_;  // 一次性搬运batchNumInLoop * seqLen行
    dataCopyParams.blockLen = blockLen;
    dataCopyParams.srcStride = srcStrideBytes;
    dataCopyParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[xOffset], dataCopyParams, padParams);

    // === 5. 拷贝weight数据：[K, dimSizeInLoop] ===
    // weight的全局偏移 = dimOffset_ + dimInnerOffset
    int32_t weightOffset = dimOffset_ + dimInnerOffset;
    DataCopyParams weightCopyParams;
    weightCopyParams.blockCount = kernelSize_;
    weightCopyParams.blockLen = blockLen;
    weightCopyParams.srcStride = srcStrideBytes;
    weightCopyParams.dstStride = 0;
    DataCopyPad(weightLocal, weightGm[weightOffset], weightCopyParams, padParams);

    // === 6. 将tensors入队 ===
    xQueue.EnQue(xLocal);
    weightQueue.EnQue(weightLocal);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::Compute(
    int32_t batchLoop, int32_t dimLoop,
    const LocalTensor<int32_t>& indicesLocal,
    const LocalTensor<int32_t>& acceptTokenLocal,
    const LocalTensor<int32_t>& queryStartLocLocal)
{
    // === 1. 计算当前循环处理的batch数和dim大小 ===
    int32_t batchNumInLoop = (batchLoop == batchLoopCnt_ - 1) ? (currentBatchNum_ - batchLoop * ubBatchSize_) : ubBatchSize_;
    int32_t dimSizeInLoop = (dimLoop == dimLoopCnt_ - 1) ? (currentDimSize_ - dimLoop * ubDimSize_) : ubDimSize_;


    // === 2. 计算当前循环的dim偏移（当前核内的相对偏移） ===
    int32_t dimInnerOffset = dimLoop * ubDimSize_;

    // === 3. 从队列中取出输入tensors ===
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> weightLocal = weightQueue.DeQue<T>();

    // === 4. 分配输出tensor和cache tensor ===
    // LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
    LocalTensor<T> cacheLocal = cacheQueue.AllocTensor<T>();

    // === 5. 处理每个batch ===
    uint16_t blockLen = dimSizeInLoop * sizeof(T);
    uint16_t srcStrideBytes = (dimSum_ - dimSizeInLoop) * sizeof(T);
    DataCopyPadParams padParams{{false, 0, 0, 0}};

    // 当前循环处理的起始batch索引（当前核内的相对偏移）
    int32_t batchInnerOffset = batchLoop * ubBatchSize_;

    for (int32_t b = 0; b < batchNumInLoop; b++) {
        int32_t curBatchIdx = firstBatchIdx_ + batchInnerOffset + b;

        // 计算当前batch的实际序列长度（不等长场景）
        // int32_t curSeqLen = queryStartLocLocal[curBatchIdx + 1] - queryStartLocLocal[curBatchIdx];
        int32_t curSeqLen = seq_len;

        // 步骤1：拷贝当前batch对应的cache state从GM到UB
        int32_t cacheLen = kernelSize_ + curSeqLen - 2;
        int64_t stateOffset = indicesLocal[curBatchIdx];
        // cache在GM中的全局偏移 = stateOffset + dimOffset_ + dimInnerOffset
        int32_t cacheGmOffset = stateOffset + dimOffset_ + dimInnerOffset;

        DataCopyParams cacheCopyParams;
        cacheCopyParams.blockCount = cacheLen;
        cacheCopyParams.blockLen = blockLen;
        cacheCopyParams.srcStride = srcStrideBytes;
        cacheCopyParams.dstStride = 0;
        DataCopyPad(cacheLocal, cacheStateGm[cacheGmOffset], cacheCopyParams, padParams);

        // 步骤2：更新cache state到GM
        UpdateCacheState(xLocal, cacheLocal, acceptTokenLocal, batchInnerOffset + b, curBatchIdx,
                        stateOffset, dimInnerOffset, dimSizeInLoop, curSeqLen);

        // 步骤3：对序列的每个位置执行卷积
        // 注意：每个batch的x数据在xLocal中是连续存储的 [seqLen, dimSizeInLoop]
        int32_t xInnerOffset = (batchInnerOffset + b) * seqLen_ * dimSizeInLoop;
        int32_t acceptTokenNum = acceptTokenLocal[curBatchIdx];

        // === 4. 拷贝输出y数据到GM ===
        // 当前循环处理的起始batch索引（当前核内的相对偏移）
        int32_t batchInnerOffset = batchLoop * static_cast<int32_t>(ubBatchSize_);
        int32_t startBatchIdx = firstBatchIdx_ + batchInnerOffset;

        // GM中的全局偏移
        int32_t yOffset = startBatchIdx * seqLen_ * dim_ + dimOffset_ + dimInnerOffset;
        uint16_t blockLen = dimSizeInLoop * sizeof(T);
        uint16_t dstStrideBytes = (dim_ - dimSizeInLoop) * sizeof(T);


        // 情况A：序列位置 j ∈ [0, K-2]，需要使用cache state
        // 对于位置j，需要使用cache state的前(K-1-j)个元素和x的前(j+1)个元素
        for (int32_t j = 0; j < kernelSize_ - 1 && j < curSeqLen; j++) {
            uint8_t stateSLen = static_cast<uint8_t>(kernelSize_ - 1 - j);
            uint8_t xSLen = static_cast<uint8_t>(j + 1);
            // 注意：Conv1dNeedState会从stateAddr读取stateSLen行，从xAddr读取xSLen行
            // 这里xSlice指向batch数据起始，需要确保包含足够的x数据
            Conv1dNeedState(xLocal[xInnerOffset], weightLocal, cacheLocal[acceptTokenNum-1+j], cacheLocal[acceptTokenNum-1+j], stateSLen, xSLen);
        }
        DataCopyParams yGMParams;
        cacheCopyParams.blockCount = kernelSize_ - 1;
        cacheCopyParams.blockLen = blockLen;
        cacheCopyParams.srcStride = 0;
        cacheCopyParams.dstStride = dstStrideBytes;
        DataCopyPad(yGm[yOffset], cacheLocal[acceptTokenNum-1], cacheCopyParams);

        // 情况B：序列位置 j ∈ [K-1, curSeqLen-1]，只使用x数据
        // 对于位置j，需要使用x的[j-K+1]到[j]位置，共K个元素
        for (int32_t j = 0; j < curSeqLen - kernelSize_; j++) {
            // 获取包含足够x数据的slice（从x[j-K+1]开始）
            uint8_t xSLen = static_cast<uint8_t>(kernelSize_);

            Conv1dNoNeedState(xLocal[xInnerOffset + (j)], weightLocal, xLocal[xInnerOffset], xSLen);
        }
        DataCopyParams cacheCopyParams;
        cacheCopyParams.blockCount = cacheLen_ - (kernelSize_ - 1);
        cacheCopyParams.blockLen = blockLen;
        cacheCopyParams.srcStride = 0;
        cacheCopyParams.dstStride = dstStrideBytes;
        DataCopyPad(yGm[yOffset + (kernelSize_ - 1)* dim_], xLocal[xInnerOffset], cacheCopyParams, padParams);

    }

    // === 6. 释放输入tensors ===
    xQueue.FreeTensor(xLocal);
    weightQueue.FreeTensor(weightLocal);
    cacheQueue.FreeTensor(cacheLocal);

    // === 7. 将输出tensor入队 ===
    yQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::UpdateCacheState(
    const LocalTensor<T>& xLocal, const LocalTensor<T>& cacheLocal,
    const LocalTensor<int32_t>& acceptTokenLocal,
    int32_t batchInBlock, int32_t curBatchIdx, int64_t stateOffset,
    int32_t dimInnerOffset, int32_t dimSize, int32_t curSeqLen)
{
    // === 获取accept token数量 ===
    int32_t acceptTokenNum = acceptTokenLocal[curBatchIdx];
    int32_t cacheLen = kernelSize_ + seqLen_ - 2;

    // === Cache更新策略 ===
    // 新的cache state = [旧cache的后K-3行] + [x的所有curSeqLen行]

    // x在UB中的偏移
    int32_t xOffset = batchInBlock * seqLen_ * dimSize;
    // cache在GM中的全局偏移
    int64_t cacheGmOffset = stateOffset + dimOffset_ + dimInnerOffset;

    uint16_t blockLen = dimSize * sizeof(T);
    uint16_t dstStrideBytes = (cacheLenSum_ - dimSize) * sizeof(T);

    // === 步骤1：拷贝旧cache state的后cacheLen - seqLen_行（如果需要） ===
    if ((cacheLen - seqLen_) > 0 && ) {
        int32_t srcCacheOffset = (acceptTokenNum - 1 + cacheLen - seqLen_) * dimSize;

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = cacheLen - seqLen_;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = dstStrideBytes;
        DataCopyPad(cacheStateGm[cacheGmOffset], cacheLocal[srcCacheOffset], dataCopyParams);
    }

    // === 步骤2：拷贝x的所有行到cache state ===
    int64_t xToCacheOffset = cacheGmOffset + (cacheLen - seqLen_) * cacheLenSum_;

    DataCopyParams xToCacheCopyParams;
    xToCacheCopyParams.blockCount = seqLen_;
    xToCacheCopyParams.blockLen = blockLen;
    xToCacheCopyParams.srcStride = 0;
    xToCacheCopyParams.dstStride = dstStrideBytes;
    DataCopyPad(cacheStateGm[xToCacheOffset], xLocal[xOffset], xToCacheCopyParams);
}



#endif // CAUSAL_CONV1D_UPDATE_H
