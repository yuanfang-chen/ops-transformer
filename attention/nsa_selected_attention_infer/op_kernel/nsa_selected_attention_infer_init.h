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
 * \file nsa_selected_attention_infer_init.h
 * \brief
 */

#pragma once

#include "nsa_selected_attention_infer.h"

 template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->nsaSelectAttentionInferSingleCoreParams.singleProcessSInnerSize;
    sInnerLoopTimes = tilingData->nsaSelectAttentionInferSingleCoreParams.sInnerLoopTimes;
    singleProcessSInnerSizeTail = tilingData->nsaSelectAttentionInferSingleCoreParams.singleProcessSInnerSizeTail;
    usedCoreNum = tilingData->nsaSelectAttentionInferSingleCoreParams.usedCoreNum;
    formerCoreNum = tilingData->nsaSelectAttentionInferSingleCoreParams.formerCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    mmResUbSize = tilingData->nsaSelectAttentionInferSingleCoreTensorSize.mmResUbSize;
    bmm2ResUbSize = tilingData->nsaSelectAttentionInferSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    gSize = tilingData->baseParams.nNumOfQInOneGroup;
    gSizeCube = gSize;
    gSizeVector = (gSize + 1) / numDouble; // vector0 : half gsize, align up
    gSizeStart = 0;
    if (tmpBlockIdx % numDouble == 1) { // vector1
    gSizeStart = gSizeVector;
    gSizeVector = gSize - gSizeVector;
    }
    kvSeqSize = tilingData->baseParams.seqSize;
    headDim = tilingData->baseParams.headSize;
    headDimV = tilingData->baseParams.headSizeV;
    selectedBlockSize = tilingData->baseParams.selectedBlockSize;
    selectedBlockCount = tilingData->baseParams.selectedBlockCount;
    msdIterNum = tilingData->baseParams.msdIterNum;
    headDimAlign = Align(headDim, BYTE_BLOCK);
    headDimVAlign = Align(headDimV, BYTE_BLOCK);
    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;
    isMtpFlag = tilingData->baseParams.isMtpFlag;
    qSeqSize = tilingData->baseParams.qSeqSize;
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        // queue
        pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);

        // tmpBuff
        pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);

        // 常驻buffer
        // 预留空间2K = 64 * 32，支持 gSize = 64
        // brcb 操作每次操作8*32字节输出，startRow接近64时，
        // 输出最多可能超出2k空间7*32字节， 这里预留256B防止越界
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            pipe->InitBuffer(softmaxMaxBuff[i], BUFFER_SIZE_BYTE_256B * numEight);
            pipe->InitBuffer(softmaxExpBuff[i], BUFFER_SIZE_BYTE_256B * numEight);
            pipe->InitBuffer(softmaxSumBuff[i], BUFFER_SIZE_BYTE_256B * numEight);
        }
        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_256B * numEight);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_256B * numEight);
    } else {  
    // L1
    pipe->InitBuffer(queryBufL1, L1_Q_SIZE);
    qL1Tensor = queryBufL1.Get<Q_T>();

    pipe->InitBuffer(kpBufL1, L1_KP_SIZE * 2);
    kpL1Tensor = kpBufL1.Get<KV_T>();

    pipe->InitBuffer(valueBufL1, L1_V_SIZE * 2);
    vL1Tensor = valueBufL1.Get<KV_T>();

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();

    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();

    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2);
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitActualQSeqLen(__gm__ uint8_t *actualQSeqLengths)
{
    actualLenQDims = tilingData->baseParams.actualLenQDims;
    if (actualLenQDims != 0) {
        actualQSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualQSeqLengths, actualLenQDims);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t copySize = gSize * headDimV;
    matmul::InitOutput<OUT_T>(attentionOutGm[(bIdx * kvHeadNum + n2Idx) * copySize], copySize, 0);
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::GetActualSeqLen()
{
    curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    uint32_t curActualSelNum = Ceil(curActualSeqLen, selectedBlockSize);
    uint32_t validTopkNum = 0;
    bool lastIndexFlag = false;
    bool lastIndexFlag2 = false;
    curBatchQseqlen = actualQSeqLengthsGm.GetValue(bIdx);
    uint32_t curTotalQSeqLenOffset = 0;
    if constexpr (LAYOUT_T == LAYOUT::TND) {
        for (uint64_t i = 0; i < bIdx; ++i) {
            curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
        }
    }
    for (uint64_t topkIdx = 0; topkIdx < selectedBlockCount; topkIdx++) {
        uint64_t baseOffset = 0;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            baseOffset = curTotalQSeqLenOffset * kvHeadNum * selectedBlockCount + s1Idx * kvHeadNum * selectedBlockCount;
        } else {
            baseOffset = bIdx * qSeqSize * kvHeadNum * selectedBlockCount + s1Idx * kvHeadNum * selectedBlockCount;
        }
        int32_t topKIndices = topKGm.GetValue(baseOffset + topkIdx);
        if (topKIndices > -1) {
            validTopkNum++;
        }
        if (topKIndices == (curActualSelNum - 1)) {
            lastIndexFlag = true;
        }
        if (topKIndices == (curActualSelNum - 2)) {
            lastIndexFlag2 = true;
        }
    }
    uint32_t maskedLen = curBatchQseqlen - 1 - s1Idx;
    uint32_t tailLen = curActualSeqLen - (curActualSelNum - 1) * selectedBlockSize;
    uint32_t tailMasked = maskedLen > tailLen ? tailLen : maskedLen;
    uint32_t penultimateMasked = tailMasked < maskedLen ? maskedLen - tailLen : 0;
    curActualSeqLenOri = curActualSeqLen - maskedLen;
    if (lastIndexFlag && !lastIndexFlag2) {
        curActualSeqLen = (curActualSeqLen - (curActualSelNum - validTopkNum) * selectedBlockSize) - tailMasked;
    } else if (lastIndexFlag && lastIndexFlag2) {
        curActualSeqLen = (curActualSeqLen - (curActualSelNum - validTopkNum) * selectedBlockSize) - maskedLen;
    } else if (!lastIndexFlag && lastIndexFlag2) {
        curActualSeqLen = validTopkNum * selectedBlockSize - penultimateMasked;
    } else if(!lastIndexFlag && !lastIndexFlag2) {
        curActualSeqLen = validTopkNum * selectedBlockSize;
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::GetBN2id(const uint32_t bn2Idx)
{
    uint32_t globalTaskIdx = beforeBlockSplitBn2Nums + bn2Idx;
    uint32_t batchId = 0;
    uint32_t prevBatchTasks = 0;
    // 1. 计算 batchId
    while (batchId < batchSize && globalTaskIdx >= prevBatchTasks + actualQSeqLengthsGm.GetValue(batchId) * kvHeadNum) {
        prevBatchTasks += actualQSeqLengthsGm.GetValue(batchId) * kvHeadNum;
        batchId++;
    }
    bIdx = batchId;
    // 2. 计算 kvheadId 和 S1Id
    uint32_t offset = globalTaskIdx - prevBatchTasks;
    n2Idx = offset / actualQSeqLengthsGm.GetValue(batchId);
    s1Idx = offset % actualQSeqLengthsGm.GetValue(batchId);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, n2Idx);
    }
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::UpdateInnerLoopCond()
{
    if (curActualSeqLen == 0) {
        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    int32_t remainSinnerSize = (int32_t)curActualSeqLen;
    int32_t computeSinnerSize = (int32_t)curActualSeqLen;
    if (remainSinnerSize > 0) {
        if (computeSinnerSize <= singleProcessSInnerSize) {
            singleProcessSInnerSizeTail = computeSinnerSize;
            sInnerLoopTimes = 1;
        } else {
            sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
        }
    } else {
        sInnerLoopTimes = 0;
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *topkIndices,  __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualQSeqLengths,  __gm__ uint8_t *actualSeqLengths,  __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const NsaSelectAttentionInferTilingData *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / numDouble;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }
    tilingData = tiling;
    InitTilingData();
    InitCalcParamsEach();
    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;
    blocktablePtr = blockTable;

    curSingleProcessSInnerSizeAlign = 0ULL; // prefix场景计算user prompt前必须重新初始化
    actualSingleProcessSInnerSize = 0ULL;
    actualSingleProcessSInnerSizeAlign = 0ULL;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)value);

    if (pipe != nullptr) {
        InitBuffers();
    }
    InitActualSeqLen(actualSeqLengths);
    InitActualQSeqLen(actualQSeqLengths);
    blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    topKGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices);

    if ASCEND_IS_AIV {
        for (int i = 0; i < PRE_LOAD_NUM; i++) {
            softmaxMaxUb[i] = softmaxMaxBuff[i].Get<T>();
            softmaxSumUb[i] = softmaxSumBuff[i].Get<T>();
            softmaxExpUb[i] = softmaxExpBuff[i].Get<T>();
        }
        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
    }

    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T);


    mm2ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T);

    vec2ResGm.SetGlobalBuffer(
            (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio* bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(T);
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个nsaSelectAttentionInferCoreParams
    // 内容拷贝到栈，减少拷贝时间。
#ifdef ASCENDC_CPU_DEBUG
    const uint32_t *coreSidxEnd = tilingData->nsaSelectAttentionInferCoreParams.coreSidxEnd;
#else
    uint32_t coreSidxEnd[50];
    copy_data_align64((uint8_t *)coreSidxEnd, (uint8_t *)(tilingData->nsaSelectAttentionInferCoreParams.coreSidxEnd),
                    sizeof(coreSidxEnd));
#endif
    bn2LoopTimes = coreSidxEnd[aiCoreIdx + 1] - coreSidxEnd[aiCoreIdx];
    beforeBlockSplitBn2Nums = coreSidxEnd[aiCoreIdx];
}