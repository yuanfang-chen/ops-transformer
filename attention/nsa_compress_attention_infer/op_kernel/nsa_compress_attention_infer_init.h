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
 * \file nsa_compress_attention_infer_init.h
 * \brief
 */

#include "nsa_compress_attention_infer.h"

#pragma once

namespace NSA_COMPRESS_ATTENTION_INFER {

#ifdef __DAV_C220_CUBE__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, 
    GM_ADDR blockTable, GM_ADDR actualQSeqLen, GM_ADDR actualKvSeqLen, GM_ADDR output, 
    GM_ADDR topkIndicesOut, GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    this->aicNum = GetBlockNum();
    this->splitBNTilingData = tilingData->splitBNParams;

    InitTilingData(tilingData);
    this->vecBlockIdx = GetBlockIdx();
    this->cubeBlockIdx = GetBlockIdx();

    // gm数据
    this->queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
    this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    this->actualQSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualQSeqLen);
    this->actualKvSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualKvSeqLen);
    this->outGm.SetGlobalBuffer((__gm__ OUT_T *)output);
    this->mm1ResGm.SetGlobalBuffer((__gm__ float *)workspace);
    this->scoreInGm.SetGlobalBuffer((__gm__ float *)(workspace + mm1ResWorkSpaceSize));
    this->mm2InGm.SetGlobalBuffer((__gm__ Q_T *)(workspace + mm1ResWorkSpaceSize + scoreInWorkSpaceSize));

    l1qBufAddrOffset = 0;
    l1kBufAddrOffset = BASE_L1_BUF_ADDR_OFFSET;
    l1vBufAddrOffset = l1kBufAddrOffset + BASE_L1_BUF_ADDR_OFFSET;
    l1pBufAddrOffset = l1vBufAddrOffset + BASE_L1_BUF_ADDR_OFFSET;

    l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1qBufAddrOffset);
    l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, KV_T>(l1kBufAddrOffset);
    l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, KV_T>(l1vBufAddrOffset);
    l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1pBufAddrOffset);

    l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, Q_T>(0);
    l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, KV_T>(0);
    l0cBufTensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
}

template <typename NCAIType> 
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::InitTilingData(const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    batchSize = tilingData->baseParams.batchSize;
    qSeqSize = tilingData->baseParams.qSeqSize;
    seqSize = tilingData->baseParams.seqSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    headSizeQk = tilingData->baseParams.headSizeQk;
    headSizeVo = tilingData->baseParams.headSizeVo;
    blockSize = tilingData->baseParams.blockSize;
    groupSize = tilingData->baseParams.groupSize;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    workSpaceElemNum = tilingData->baseParams.workSpaceElemNum;
    mm1ResWorkSpaceSize = tilingData->baseParams.mm1ResWorkSpaceSize;
    mm2InWorkSpaceSize = tilingData->baseParams.mm2InWorkSpaceSize;
    scoreInWorkSpaceSize = tilingData->baseParams.scoreInWorkSpaceSize;

    coreNumUsed = tilingData->splitBNParams.coreNumUsed;
    processNum = tilingData->splitBNParams.processNum;
    processPerBatch = tilingData->splitBNParams.processPerBatch;
    kvHeadSplitSize = tilingData->splitBNParams.kvHeadSplitSize;
    kvHeadSplitNum = tilingData->splitBNParams.kvHeadSplitNum;
    qSeqLenSplitSize = tilingData->splitBNParams.qSeqLenSplitSize;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::PreProcess(const uint32_t processIdx)
{
    uint32_t processCumSum = 0;
    int64_t curQSeqlen = 0;
    uint32_t qSeqLenSplitNum = 0;
    qSeqLenCumSum = 0;
    for (uint32_t i = 0; i < batchSize; i++) {
        curQSeqlen = (qSeqSize == 1) ? 1: actualQSeqLenGm.GetValue(i);
        qSeqLenSplitNum = (static_cast<uint32_t>(curQSeqlen) + qSeqLenSplitSize - 1) / qSeqLenSplitSize;
        if ((processCumSum + qSeqLenSplitNum * kvHeadSplitNum - 1) >= processIdx) {
            bIdx = i;
            break;
        }
        qSeqLenCumSum +=  (LAYOUT_T == LAYOUT::TND) ? curQSeqlen : qSeqSize;
        processCumSum += qSeqLenSplitNum * kvHeadSplitNum;
    }

    uint32_t processOffsetInCurBatch = processIdx - processCumSum;
    uint32_t qSeqLenSplitIdx = processOffsetInCurBatch / kvHeadSplitNum;
    qSeqLenOffset = qSeqLenSplitIdx * qSeqLenSplitSize;
    qSeqLenCurProcess = (qSeqLenSplitIdx == qSeqLenSplitNum - 1) ? (curQSeqlen - qSeqLenOffset) : qSeqLenSplitSize;

    uint32_t kvHeadSplitIdx = processOffsetInCurBatch % kvHeadSplitNum;
    kvHeadOffset = kvHeadSplitIdx * kvHeadSplitSize;
    curKvHeadNum = (kvHeadSplitIdx == kvHeadSplitNum - 1) ? (kvHeadNum - kvHeadOffset) : kvHeadSplitSize;
}

#elif __DAV_C220_VEC__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR blockTable, GM_ADDR actualQSeqLen, GM_ADDR actualKvSeqLen, GM_ADDR actualSelKvSeqLen, GM_ADDR output, GM_ADDR topkIndicesOut,
    GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    this->aicNum = GetBlockNum();

    this->splitBNTilingData = tilingData->splitBNParams;

    InitTilingData(tilingData);
    vecBlockIdx = GetBlockIdx();
    this->cubeBlockIdx = vecBlockIdx / GetSubBlockNum();

    this->actualQSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualQSeqLen);
    this->actualSelKvSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualSelKvSeqLen);
    this->actualKvSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualKvSeqLen);
    this->topkOutGm.SetGlobalBuffer((__gm__ int32_t *)topkIndicesOut);

    mm1ResGm.SetGlobalBuffer((__gm__ float *)workspace);
    scoreInGm.SetGlobalBuffer((__gm__ float *)(workspace + mm1ResWorkSpaceSize));
    mm2InGm.SetGlobalBuffer((__gm__ Q_T *)(workspace + mm1ResWorkSpaceSize + scoreInWorkSpaceSize));
    topKInGm.SetGlobalBuffer((__gm__ float *)(workspace + mm1ResWorkSpaceSize + scoreInWorkSpaceSize + mm2InWorkSpaceSize));
}

template <typename NCAIType> 
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::InitTilingData(const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    batchSize = tilingData->baseParams.batchSize;
    qSeqSize = tilingData->baseParams.qSeqSize;
    seqSize = tilingData->baseParams.seqSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    headSizeQk = tilingData->baseParams.headSizeQk;
    headSizeVo = tilingData->baseParams.headSizeVo;
    blockSize = tilingData->baseParams.blockSize;
    scaleValue = tilingData->baseParams.scaleValue;
    selectNum = tilingData->baseParams.selectNum;
    groupSize = tilingData->baseParams.groupSize;
    attenMaskFlag = tilingData->baseParams.attenMaskFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    workSpaceElemNum = tilingData->baseParams.workSpaceElemNum;
    mm1ResWorkSpaceSize = tilingData->baseParams.mm1ResWorkSpaceSize;
    mm2InWorkSpaceSize = tilingData->baseParams.mm2InWorkSpaceSize;
    scoreInWorkSpaceSize = tilingData->baseParams.scoreInWorkSpaceSize;
    topKInWorkSpaceSize = tilingData->baseParams.topKInWorkSpaceSize;

    coreNumUsed = tilingData->splitBNParams.coreNumUsed;

    softmaxTilingData = tilingData->softmaxTilingData;
    topkTilingData = tilingData->topkTilingData;

    processNum = tilingData->splitBNParams.processNum;
    processPerBatch = tilingData->splitBNParams.processPerBatch;
    kvHeadSplitSize = tilingData->splitBNParams.kvHeadSplitSize;
    kvHeadSplitNum = tilingData->splitBNParams.kvHeadSplitNum;
    qSeqLenSplitSize = tilingData->splitBNParams.qSeqLenSplitSize;

    // importance score
    ubSize = tilingData->baseParams.ubSize;
    selectSize = tilingData->baseParams.selectSize;
    compSizeL = tilingData->baseParams.compSizeL;
    compStrideD = tilingData->baseParams.compStrideD;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::PreProcess(const uint32_t processIdx)
{
    curSeqlen = actualKvSeqLenGm.GetValue(bIdx);
    // 对齐列数
    alignedColLen = AlignUp(curSeqlen, 16);

    uint32_t groupNumVec0 = CeilDiv(curKvHeadNum * qSeqLenCurProcess, 2);
    uint32_t groupNumVec1 = (curKvHeadNum * qSeqLenCurProcess) / 2;

    rowNumVec0 = groupNumVec0 * groupSize;
    rowNumVec1 = groupNumVec1 * groupSize;

    softmaxRowLenPerCore = (vecBlockIdx % 2 == 0) ? rowNumVec0 : rowNumVec1;
    uint32_t maxRowNumSM = 12 * 1024 / alignedColLen;
    softmaxBasicRowLen = (maxRowNumSM < softmaxRowLenPerCore) ? maxRowNumSM : softmaxRowLenPerCore;
    softmaxRowLoop = softmaxBasicRowLen == 0 ? 0 : (softmaxRowLenPerCore + softmaxBasicRowLen - 1) / softmaxBasicRowLen;

    // 每次处理的元素数量， 行数*对齐后的列数
    softmaxTileLength = softmaxBasicRowLen * alignedColLen;
    // 列需要填充几个数？
    softmaxRightPadding = alignedColLen - curSeqlen;

    // 设置softmax buffer
    const uint32_t softmaxInputbufOffset = 0;
    const uint32_t softmaxOut32bufOffset = softmaxTileLength * sizeof(float);
    const uint32_t softmaxOut16bufOffset = 2 * softmaxTileLength * sizeof(float);
    const uint32_t softmaxbufOffset = 3 * softmaxTileLength * sizeof(float);
    softmaxInputbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(softmaxInputbufOffset);
    softmaxOut32bufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(softmaxOut32bufOffset);
    softmaxOut16bufTensor = buf.GetBuffer<BufferType::ASCEND_UB, Q_T>(softmaxOut16bufOffset);
    sharedTmpBuffer = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(softmaxbufOffset);

    InitImportanceScoreParams();

    // 设置topk buffer
    const uint32_t topkInputbufOffset = 0;
    const uint32_t topkOutvaluebufOffset = BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    const uint32_t topkOutindexbufOffset = 2 * BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    const uint32_t arithbufferoffset = 3 * BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    const uint32_t topktempbufOffset = 4 * BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    topkInputbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(topkInputbufOffset);
    topkoutvalueLocal = buf.GetBuffer<BufferType::ASCEND_UB, float>(topkOutvaluebufOffset);
    topkoutindexLocal = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(topkOutindexbufOffset);
    arithbuffer = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(arithbufferoffset);
    topksharedTmpBuffer = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(topktempbufOffset);
}

template <typename NCAIType> 
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::InitImportanceScoreParams()
{   
    baseBlockSize = (ubSize - 18*1024 - 32*1024) / (2 + 0) / sizeof(float);
    uint32_t t = (curSeqlen-1) * compStrideD + compSizeL;
    gHeadNums = qHeadNum / kvHeadNum;
    gHeadNumsAlign = AlignUp(gHeadNums, 16);

    outS2 = t % selectSize == 0 ? t / selectSize : t / selectSize + 1;
    strideOut = selectSize / compStrideD;
    maxM = selectSize /compStrideD - 1;
    maxN = compSizeL /compStrideD - 1;

    impSocreCoreRowCount = (vecBlockIdx % 2 == 0) ? rowNumVec0 : rowNumVec1;
    uint32_t impSocreCoreRowCountAlign = AlignUp(impSocreCoreRowCount, 16);
    maxRowCountPerLoop = impSocreCoreRowCountAlign > 128 ? 128 : impSocreCoreRowCountAlign;
    maxBaseS2 = baseBlockSize / maxRowCountPerLoop;
    maxBaseJ = (maxBaseS2 - maxM - maxN) / strideOut + 1;
    maxBaseS2 = (maxBaseJ - 1) * strideOut + maxM + maxN + 1;
    baseBlockSize = AlignUp(maxBaseS2, 16) * maxRowCountPerLoop;
    const uint32_t impScoreInputbufOffset = 0;
    const uint32_t impScoreCalcbufOffset = baseBlockSize * sizeof(float);
    const uint32_t impScoreTmpbufOffset = 2 * baseBlockSize * sizeof(float);
    const uint32_t impScoreOffset = 2 * baseBlockSize * sizeof(float) + 1024;

    pslcTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(impScoreInputbufOffset);
    pslcCalcTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(impScoreCalcbufOffset);
    pslcTmpTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(impScoreTmpbufOffset);
    pslcSharedBuffer = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(impScoreOffset);

    uint32_t maxT = (seqSize-1) * compStrideD + compSizeL;
    maxOutS2 = maxT % selectSize == 0 ? maxT / selectSize : maxT / selectSize + 1;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::PreProcessOffset(const uint32_t processIdx)
{   
    qSeqLenCumSum = 0;
    uint32_t processCumSum = 0;
    int64_t curQSeqlen = 0;
    uint32_t qSeqLenSplitNum = 0;
    for (uint32_t i = 0; i < batchSize; i++) {
        curQSeqlen = (qSeqSize == 1) ? 1: actualQSeqLenGm.GetValue(i);
        qSeqLenSplitNum = (static_cast<uint32_t>(curQSeqlen) + qSeqLenSplitSize - 1) / qSeqLenSplitSize;
        if ((processCumSum + qSeqLenSplitNum * kvHeadSplitNum - 1) >= processIdx) {
            bIdx = i;
            break;
        }
        qSeqLenCumSum += (LAYOUT_T == LAYOUT::TND) ? curQSeqlen : qSeqSize;
        processCumSum += qSeqLenSplitNum * kvHeadSplitNum;
    }

    uint32_t processOffsetInCurBatch = processIdx - processCumSum;

    uint32_t kvHeadSplitIdx = processOffsetInCurBatch % kvHeadSplitNum;
    kvHeadOffset = kvHeadSplitIdx * kvHeadSplitSize;
    curKvHeadNum = (kvHeadSplitIdx == kvHeadSplitNum - 1) ? (kvHeadNum - kvHeadOffset) : kvHeadSplitSize;

    uint32_t qSeqLenSplitIdx = processOffsetInCurBatch / kvHeadSplitNum;
    qSeqLenOffset = qSeqLenSplitIdx * qSeqLenSplitSize;
    qSeqLenCurProcess = (qSeqLenSplitIdx == qSeqLenSplitNum - 1) ? (curQSeqlen - qSeqLenOffset) : qSeqLenSplitSize;
}

#endif
}