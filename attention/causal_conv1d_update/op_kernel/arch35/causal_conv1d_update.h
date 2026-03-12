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
#include "causal_conv1d_update_struct.h"
#include "vf/compute.h"

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
    __aicore__ inline CausalConv1dUpdateKernel(TPipe* pipe) : pipe_(pipe) {};
    /**
     * @brief 初始化函数，设置所有Global Memory指针并分配UB资源
     * @param x 输入序列 [batch, m+1, dim]
     * @param weight 卷积核 [K, dim]
     * @param convStates 输入的cache state [-1, K-1+m, dim]
     * @param cacheIndices cache索引 [batch]，指定每个batch对应的cache state位置
     * @param numAcceptedToken 接受的token数量 [batch]，可选输入
     * @param queryStartLoc query起始位置 [batch+1]
     * @param y 输出序列 [batch, m+1, dim]
     * @param outputconvStates 输出的cache state，原地更新
     * @param tilingData tiling参数结构体指针
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR numAcceptedToken, GM_ADDR y,const CausalConv1dUpdateTilingData* tilingData);

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

    __aicore__ inline void CopyIn(int32_t batchLoop, int32_t dimLoop);

    __aicore__ inline void Compute(int32_t batchLoop, int32_t dimLoop, const LocalTensor<int32_t>& indicesLocal,
                const LocalTensor<int32_t>& acceptTokenLocal, const LocalTensor<int32_t>& queryStartLocLocal);

    __aicore__ inline void UpdateconvStates(const LocalTensor<T>& xLocal, const LocalTensor<T>& convStatesLocal, int32_t acceptToken,
                int32_t batchInBlock, int32_t curBatchIdx, int64_t convStatesOffset, int32_t dimInnerOffset, int32_t dimSize, int32_t curSeqLen);

     __aicore__ inline void InsertSync(const HardEvent& event);
    // ========== Global Memory指针 ==========
    GlobalTensor<T> xGm;                    // 输入序列 [batch, seqLen, dim]
    GlobalTensor<T> weightGm;               // 卷积核 [K, dim]
    GlobalTensor<T> convStatesGm;           // 输入cache state [-1, K-1+m, dim]
    GlobalTensor<int32_t> cacheIndicesGm;   // cache索引 [batch]
    GlobalTensor<int32_t> acceptTokenNumGm; // 接受的token数 [batch]
    GlobalTensor<int32_t> queryStartLocGm;  // query起始位置 [batch+1]
    GlobalTensor<T> yGm;                    // 输出序列 [batch, seqLen, dim]

    // ========== UB队列（Unified Buffer中的数据缓存） ==========
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xQueue;       // 输入x队列（双Buffer）
    TQue<QuePosition::VECIN, 1> weightQueue;            // 卷积核队列（单Buffer）
    TQue<QuePosition::VECIN, 1> cacheQueue;             // cache state队列（单Buffer）
    TQue<QuePosition::VECIN, 1> indicesQueue;           // cache索引队列（单Buffer）
    TQue<QuePosition::VECIN, 1> acceptTokenQueue;       // accept token队列（单Buffer）
    TQue<QuePosition::VECIN, 1> queryStartLocQueue;     // query起始位置队列（单Buffer）
    TQue<QuePosition::VECOUT, BUFFER_NUM> fQueue; 

    TPipe* pipe_;                             // Pipeline管理对象

    // ========== Tiling参数（二维切分：Dim方向 × Batch方向） ==========
    // 核间切分参数
    int64_t dimChunkSize_;          // 每个核处理的Dim大小（256B对齐）
    int64_t batchPerCore_;          // 每个核处理的batch数
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
    int32_t isResidualConnection_;

    // ========== 运行时计算参数 ==========
    int32_t blockIdx_;           // 当前核的索引
    int32_t batchIdx_;           // 当前核在Batch维度的索引
    int32_t dimIdx_;             // 当前核在Dim维度的索引
    int32_t firstBatchIdx_;      // 当前核处理的起始batch索引
    int32_t currentBatchNum_;    // 当前核处理的batch数量
    int32_t dimOffset_;          // 当前核处理的Dim起始偏移（元素）
    int32_t currentDimSize_;     // 当前核处理的Dim大小
    int64_t hasAcceptTokenNum_;     // 是否提供了acceptTokenNum输入
    int32_t dimSum_;
    int32_t cacheLenSum_;
    int32_t batchNumInLoop_;
    int32_t dimSizeInLoop_;
    int32_t dimInnerOffset_;
    
};

// ==================== 函数实现 ====================

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
    GM_ADDR numAcceptedToken, GM_ADDR y,const CausalConv1dUpdateTilingData* tilingData)
{
    // === 1. 获取核间切分参数（二维：Dim方向 × Batch方向） ===
    dimChunkSize_ = tilingData->dimChunkSize;
    batchPerCore_ = tilingData->batchPerCore;
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
    cacheLen_ = tilingData->stateLen;
    hasAcceptTokenNum_ = tilingData->hasAcceptTokenNum;
    xInputMode_ = tilingData->xInputMode;
    xStride_ = tilingData->xStride;
    cacheStride_ = tilingData->cacheStride;
    isResidualConnection_ = tilingData->isResidualConnection;

    // === 4. 计算当前核在二维grid中的索引 ===
    blockIdx_ = GetBlockIdx();
    batchIdx_ = blockIdx_ / tilingData->dimCoreCnt;  // Batch方向索引
    dimIdx_ = blockIdx_ % tilingData->dimCoreCnt;    // Dim方向索引

    // === 5. 计算当前核处理的Dim范围 ===
    dimSum_ = dim_ + xStride_;
    cacheLenSum_ = dim_ + cacheStride_;
    dimOffset_ = dimIdx_ * dimChunkSize_;  // Dim起始偏移
    // 判断是否为Dim方向尾核
    if (dimIdx_ == tilingData->dimCoreCnt - 1) {
        currentDimSize_ = tilingData->dimTailSize;   // 尾核使用tail大小
    } else {
        currentDimSize_ = dimChunkSize_;  // 常规核使用chunk大小
    }

    // === 6. 计算当前核处理的Batch范围 ===
    // 从有效batch起始位置开始分配
    int64_t validBatchCount = validBatchEnd_ - validBatchStart_ + 1;
    int64_t batchStartOffset = validBatchStart_;

    firstBatchIdx_ = batchStartOffset + batchIdx_ * static_cast<int32_t>(batchPerCore_);
    // 判断是否为Batch方向尾核
    if (batchIdx_ == static_cast<int32_t>(tilingData->batchCoreCnt) - 1) {
        currentBatchNum_ = static_cast<int32_t>(tilingData->batchTailPerCore);
    } else {
        currentBatchNum_ = static_cast<int32_t>(batchPerCore_);
    }

    // 确保不超出有效batch范围
    int32_t maxBatchIdx = batchStartOffset + validBatchCount;
    if (firstBatchIdx_ + currentBatchNum_ > maxBatchIdx) {
        currentBatchNum_ = maxBatchIdx - firstBatchIdx_;
    }

    // === 7. 设置Global Memory buffers ===
    if(xInputMode_ == 1) {
        xGm.SetGlobalBuffer((__gm__ T*)x, batchSize_ * seqLen_ * dimSum_);
        yGm.SetGlobalBuffer((__gm__ T*)y, batchSize_ * seqLen_ * dim_);
    } else {
        xGm.SetGlobalBuffer((__gm__ T*)x, cuSeqLen_ * dimSum_);
        yGm.SetGlobalBuffer((__gm__ T*)y, cuSeqLen_ * dim_);
    }
    weightGm.SetGlobalBuffer((__gm__ T*)weight, kernelSize_ * dim_);
    convStatesGm.SetGlobalBuffer((__gm__ T*)convStates);
    cacheIndicesGm.SetGlobalBuffer((__gm__ int32_t*)cacheIndices, batchSize_);
    if (hasAcceptTokenNum_ == 1) {
        acceptTokenNumGm.SetGlobalBuffer((__gm__ int32_t*)numAcceptedToken, batchSize_);
    }
    if(xInputMode_ == 1) {
        queryStartLocGm.SetGlobalBuffer((__gm__ int32_t*)queryStartLoc, batchSize_+1);
    }

    // === 8. 初始化UB队列 ===

    // xQueue: 存储输入x数据（y复用此buffer）
    // 大小: ubBatchSize * seqLen * ubDimSize
    int32_t xQueueSize = ubBatchSize_ * seqLen_ * ubDimSize_ * sizeof(T);
    pipe_->InitBuffer(xQueue, BUFFER_NUM, xQueueSize);

    // cacheQueue: 存储cache state
    // 大小: (K-1+seqLen-1) * ubDimSize = (K + seqLen - 2) * ubDimSize
    int32_t cacheQueueSize = cacheLen_ * ubDimSize_ * sizeof(T);
    pipe_->InitBuffer(cacheQueue, 1, cacheQueueSize);

    // weightQueue: 存储卷积核 [K, ubDimSize]
    int32_t weightQueueSize = kernelSize_ * ubDimSize_ * sizeof(T);
    pipe_->InitBuffer(weightQueue, 1, weightQueueSize);

    // indicesQueue: 存储cache索引
    int32_t indicesQueueSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    // printf("indicesQueueSize %d",indicesQueueSize);
    pipe_->InitBuffer(indicesQueue, 1, indicesQueueSize);

    // acceptTokenQueue: 存储accept token数量
    int32_t acceptTokenQueueSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    pipe_->InitBuffer(acceptTokenQueue, 1, acceptTokenQueueSize);

    // queryStartLocQueue: 只有二维TH的时候，存储query起始位置
    if(xInputMode_) {
        int32_t queryStartLocQueueSize = ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe_->InitBuffer(queryStartLocQueue, 1, queryStartLocQueueSize);
    }
    // pipe_->InitBuffer(fQueue, 1, ubDimSize_ * sizeof(T));
    int64_t all = 2*xQueueSize + cacheQueueSize + weightQueueSize + indicesQueueSize + acceptTokenQueueSize;
    // printf("all %d", all);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::Process()
{
    // 预先搬运cache indices、accept token numbers和query start loc到UB
    // 这些是常驻数据，整个Process过程中都需要访问
    LocalTensor<int32_t> indicesLocal = indicesQueue.AllocTensor<int32_t>();
    LocalTensor<int32_t> acceptTokenLocal = acceptTokenQueue.AllocTensor<int32_t>();
    LocalTensor<int32_t> queryStartLocLocal;

    DataCopyPadParams padParams{false, 0, 0, 0};
    uint32_t indicesBlockLen = batchSize_ * sizeof(int32_t);
    DataCopyParams indicesCopyParams;
    indicesCopyParams.blockCount = 1;
    indicesCopyParams.blockLen = indicesBlockLen;
    indicesCopyParams.srcStride = 0;
    indicesCopyParams.dstStride = 0;
    DataCopyPad(indicesLocal, cacheIndicesGm, indicesCopyParams, padParams);
    Duplicate(acceptTokenLocal, static_cast<int32_t>(1), batchSize_);
    InsertSync(HardEvent::V_MTE2);
    if (hasAcceptTokenNum_ == 1) {
        DataCopyPad(acceptTokenLocal, acceptTokenNumGm, indicesCopyParams, padParams);
    }
    InsertSync(HardEvent::MTE2_S);

    // 拷贝query start loc：[batch+1]
    if(xInputMode_ == 1) {
        queryStartLocLocal = queryStartLocQueue.AllocTensor<int32_t>();
        uint32_t queryStartLocBlockLen = (batchSize_ + 1) * sizeof(int32_t);
        DataCopyParams queryStartLocCopyParams;
        queryStartLocCopyParams.blockCount = 1;
        queryStartLocCopyParams.blockLen = queryStartLocBlockLen;
        queryStartLocCopyParams.srcStride = 0;
        queryStartLocCopyParams.dstStride = 0;
        DataCopyPad(queryStartLocLocal, queryStartLocGm, queryStartLocCopyParams, padParams);
    }

    for (int32_t batchLoop = 0; batchLoop < batchLoopCnt_; batchLoop++) {
        for (int32_t dimLoop = 0; dimLoop < dimLoopCnt_; dimLoop++) {
            CopyIn(batchLoop, dimLoop);
            Compute(batchLoop, dimLoop, indicesLocal, acceptTokenLocal, queryStartLocLocal);

        }
    }

    // 释放indices、acceptToken和queryStartLoc tensors
    indicesQueue.FreeTensor(indicesLocal);
    acceptTokenQueue.FreeTensor(acceptTokenLocal);
    if(xInputMode_ == 1) {
        queryStartLocQueue.FreeTensor(queryStartLocLocal);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::CopyIn(int32_t batchLoop, int32_t dimLoop)
{
    // === 1. 计算当前循环处理的batch数和dim大小 ===
    batchNumInLoop_ = (batchLoop == batchLoopCnt_ - 1) ? (currentBatchNum_ - batchLoop * ubBatchSize_) : ubBatchSize_;
    dimSizeInLoop_ = (dimLoop == dimLoopCnt_ - 1) ? (currentDimSize_ - dimLoop * ubDimSize_) : ubDimSize_;
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    LocalTensor<T> weightLocal = weightQueue.AllocTensor<T>();

    DataCopyPadParams padParams{false, 0, 0, 0};
    int32_t dimInnerOffset = dimLoop * static_cast<int32_t>(ubDimSize_);
    int32_t startBatchIdx = firstBatchIdx_ + batchLoop * ubBatchSize_;
    int32_t xOffset = startBatchIdx * seqLen_ * dimSum_ + dimOffset_ + dimInnerOffset;
    uint32_t blockLen = dimSizeInLoop_ * sizeof(T);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = batchNumInLoop_ * seqLen_;  // 一次性搬运batchNumInLoop * seqLen行
    dataCopyParams.blockLen = blockLen;
    dataCopyParams.srcStride = (dimSum_ - dimSizeInLoop_) * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[xOffset], dataCopyParams, padParams);

    // === 5. 拷贝weight数据：[K, dimSizeInLoop] ===
    // weight的全局偏移 = dimOffset_ + dimInnerOffset
    int32_t weightOffset = dimOffset_ + dimInnerOffset;
    DataCopyParams weightCopyParams;
    weightCopyParams.blockCount = kernelSize_;
    weightCopyParams.blockLen = blockLen;
    weightCopyParams.srcStride = (dim_ - dimSizeInLoop_) * sizeof(T);
    weightCopyParams.dstStride = 0;
    DataCopyPad(weightLocal, weightGm[weightOffset], weightCopyParams, padParams);

    // === 6. 将tensors入队 ===
    xQueue.EnQue(xLocal);
    weightQueue.EnQue(weightLocal);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::Compute(int32_t batchLoop, int32_t dimLoop, const LocalTensor<int32_t>& indicesLocal,
    const LocalTensor<int32_t>& acceptTokenLocal, const LocalTensor<int32_t>& queryStartLocLocal)
{
    // === 1. 计算当前循环处理的batch数和dim大小 ===
    int32_t batchInnerOffset = batchLoop * ubBatchSize_;
    int32_t dimInnerOffset = dimLoop * ubDimSize_;

    // === 3. 从队列中取出输入tensors ===
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> weightLocal = weightQueue.DeQue<T>();
    LocalTensor<T> convStatesLocal = cacheQueue.AllocTensor<T>();
    // === 5. 处理每个batch ===
    uint16_t blockLen = dimSizeInLoop_ * sizeof(T);
    uint16_t strideBytes = (dimSum_ - dimSizeInLoop_) * sizeof(T);
    DataCopyPadParams padParams{false, 0, 0, 0};

    for (int32_t b = 0; b < batchNumInLoop_; b++) {
        // printf("batch %d",b);
        int32_t curBatchIdx = firstBatchIdx_ + batchInnerOffset + b;
        int32_t curSeqLen = seqLen_;
        int64_t convStatesOffset = static_cast<int64_t>(indicesLocal.GetValue(curBatchIdx));
        int32_t acceptToken = acceptTokenLocal.GetValue(curBatchIdx);
        int32_t convStatesGmOffset = convStatesOffset * cacheLen_ * cacheLenSum_ + dimOffset_ + dimInnerOffset;

        DataCopyParams cacheCopyParams;
        cacheCopyParams.blockCount = cacheLen_;
        cacheCopyParams.blockLen = blockLen;
        cacheCopyParams.srcStride = strideBytes;
        cacheCopyParams.dstStride = 0;
        DataCopyPad(convStatesLocal, convStatesGm[convStatesGmOffset], cacheCopyParams, padParams); //convStates GM->UB
        InsertSync(HardEvent::MTE2_MTE3);
        // InsertSync(HardEvent::S_MTE3);
        // PipeBarrier<PIPE_ALL>();
        UpdateconvStates(xLocal, convStatesLocal, acceptToken, batchInnerOffset + b, curBatchIdx, convStatesOffset, 
                        dimInnerOffset, dimSizeInLoop_, curSeqLen);
        InsertSync(HardEvent::MTE2_V);
        int32_t xInnerOffset = (batchInnerOffset + b) * seqLen_ * dimSizeInLoop_;
        int32_t yOffset = curBatchIdx * seqLen_ * dim_ + dimOffset_ + dimInnerOffset;

        // 情况A：序列位置 j ∈ [0, K-2]，需要使用cache state
        for (int32_t j = 0; j < kernelSize_ - 1 && j < curSeqLen; j++) {
            uint8_t stateSLen = static_cast<uint8_t>(kernelSize_ - 1 - j);
            uint8_t xSLen = static_cast<uint8_t>(j + 1);
            LocalTensor<T> xSlice = xLocal[xInnerOffset];
            LocalTensor<T> stateSlice = convStatesLocal[(acceptToken-1+j)*dimSizeInLoop_];
            Conv1dNeedState(xSlice, weightLocal, stateSlice, stateSlice, stateSLen, xSLen, dimSizeInLoop_, isResidualConnection_);
        }
        InsertSync(HardEvent::V_MTE3);
        cacheQueue.EnQue<T>(convStatesLocal);
        convStatesLocal = cacheQueue.DeQue<T>();
        DataCopyParams yGMParams;
        yGMParams.blockCount = ((kernelSize_ - 1) < curSeqLen) ? kernelSize_ - 1 : curSeqLen;
        yGMParams.blockLen = blockLen;
        yGMParams.srcStride = 0;
        yGMParams.dstStride = strideBytes;
        DataCopyPad(yGm[yOffset], convStatesLocal[(acceptToken-1) *dimSizeInLoop_], yGMParams);

        // 情况B：序列位置 j ∈ [K-1, curSeqLen-1]，只使用x数据
        uint16_t blockCount = curSeqLen - kernelSize_+1;
        if((curSeqLen - kernelSize_+1) > 0) {
            for (int32_t j = 0; j < blockCount; j++) {
                uint8_t xSLen = static_cast<uint8_t>(kernelSize_);
                LocalTensor<T> InLocal = xLocal[xInnerOffset + j*dimSizeInLoop_];
                Conv1dNoNeedState(InLocal, weightLocal, InLocal, xSLen, static_cast<uint32_t>(dimSizeInLoop_), isResidualConnection_);
            }
            InsertSync(HardEvent::V_MTE3);  
            DataCopyParams xToCacheCopyParams2;
            xToCacheCopyParams2.blockCount = blockCount;
            xToCacheCopyParams2.blockLen = blockLen;
            xToCacheCopyParams2.srcStride = 0;
            xToCacheCopyParams2.dstStride = strideBytes;
            DataCopyPad(yGm[yOffset + (kernelSize_ - 1)* dim_], xLocal[xInnerOffset], xToCacheCopyParams2);
        }
        InsertSync(HardEvent::MTE3_MTE2);

    }

    // === 6. 释放输入tensors ===
    xQueue.FreeTensor(xLocal);
    weightQueue.FreeTensor(weightLocal);
    cacheQueue.FreeTensor(convStatesLocal);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::UpdateconvStates(
    const LocalTensor<T>& xLocal, const LocalTensor<T>& convStatesLocal,
    int32_t acceptToken, int32_t batchInBlock, int32_t curBatchIdx, int64_t convStatesOffset,
    int32_t dimInnerOffset, int32_t dimSize, int32_t curSeqLen)
{
    int32_t xOffset = batchInBlock * seqLen_ * dimSize;
    int64_t convStatesGmOffset = convStatesOffset * cacheLen_ * cacheLenSum_ + dimOffset_ + dimInnerOffset;
    uint32_t blockLen = dimSize * sizeof(T);
    uint32_t dstStrideBytes = (cacheLenSum_ - dimSize) * sizeof(T);
    // === 步骤1：拷贝旧cache state的后cacheLen - seqLen_行（如果需要） ===
    int32_t convStatesNeedRow = cacheLen_ - seqLen_;
    if (convStatesNeedRow > 0) {
        int32_t srcCacheOffset = (acceptToken - 1 + convStatesNeedRow) * dimSize;
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = cacheLen_ - seqLen_;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = dstStrideBytes;
        DataCopyPad(convStatesGm[convStatesGmOffset], convStatesLocal[srcCacheOffset], dataCopyParams);
    }

    // === 步骤2：拷贝x的所有行到cache state ===
    int64_t xToCacheOffset = convStatesGmOffset + convStatesNeedRow * cacheLenSum_;
    DataCopyParams xToCacheCopyParams;
    xToCacheCopyParams.blockCount = seqLen_;
    xToCacheCopyParams.blockLen = blockLen;
    xToCacheCopyParams.srcStride = 0;
    xToCacheCopyParams.dstStride = dstStrideBytes;
    DataCopyPad(convStatesGm[xToCacheOffset], xLocal[xOffset], xToCacheCopyParams);
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateKernel<T>::InsertSync(const HardEvent& event)
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    switch (event) {
        case HardEvent::V_MTE3:
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            break;
        case HardEvent::V_MTE2:
            SetFlag<HardEvent::V_MTE2>(eventID);
            WaitFlag<HardEvent::V_MTE2>(eventID);
            break;
        case HardEvent::MTE2_V:
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
            break;
        case HardEvent::MTE2_MTE3:
            SetFlag<HardEvent::MTE2_MTE3>(eventID);
            WaitFlag<HardEvent::MTE2_MTE3>(eventID);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlag<HardEvent::MTE3_MTE2>(eventID);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID);
            break;
        case HardEvent::S_MTE3:
            SetFlag<HardEvent::S_MTE3>(eventID);
            WaitFlag<HardEvent::S_MTE3>(eventID);
            break;
        case HardEvent::MTE2_S:
            SetFlag<HardEvent::MTE2_S>(eventID);
            WaitFlag<HardEvent::MTE2_S>(eventID);
            break;
        default:
            break;
    }
}

#endif // CAUSAL_CONV1D_UPDATE_H
