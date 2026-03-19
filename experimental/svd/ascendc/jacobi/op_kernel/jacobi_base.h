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
 * \file jacobi_base.h
 * \brief
 */

#ifndef JACOBI_BASE_H
#define JACOBI_BASE_H

#ifdef DEBUG_MODE
#define PRINT_VAL(X)   printf("Variable %s has value %d\n", #X, (int)X)
#else
#define PRINT_VAL(X)
#endif



#define MIN(A,B) (((A)<(B))?(A):(B))

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SVD {

template<typename T>
constexpr __aicore__ T GET_MIN_BLOCK_SIZE()
{
    constexpr uint32_t MIN_BLOCK_SIZE = 32;
    return static_cast<T>(MIN_BLOCK_SIZE);
}

template<typename SourceType, typename ResultType = uint8_t>
constexpr __aicore__ ResultType SMALL_MASK_NUM_ELEMENTS()
{
    return static_cast<ResultType>(GET_MIN_BLOCK_SIZE<size_t>() / sizeof(SourceType));
}


template<typename T>
constexpr __aicore__ T GET_MAX_BLOCK_SIZE()
{
    constexpr uint32_t MIN_BLOCK_SIZE = 256;
    return static_cast<T>(MIN_BLOCK_SIZE);
}

template<typename SourceType, typename ResultType = uint8_t>
constexpr __aicore__ uint8_t BIG_MASK_NUM_ELEMENTS()
{
    return static_cast<ResultType>(GET_MAX_BLOCK_SIZE<size_t>() / sizeof(SourceType));
}


using namespace AscendC;
template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
class JacobiBase {
private:
    struct RotateMaskInfo;
public:

    __aicore__ inline JacobiBase(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* s,
        __gm__ uint8_t* u, __gm__ uint8_t* v, __gm__ uint8_t* workspace,
        const JMTilingData* tilingData)
    {
        coreIdx = GetBlockIdxImpl();
        needSort = true;
        numIterations = tilingData->nIterations;
        nSize = tilingData->nSize;
        nSizeAligned = tilingData->nSizeAligned;
        mSize = tilingData->mSize;
        batchSize = (int32_t)tilingData->batchSize;
        numStages = tilingData->nStages;
        numGlobalSets = tilingData->numGlobalSets;
        globSetMaxSize = tilingData->globalSetSizeMax;
        globSetMinSize = tilingData->globalSetSizeMin;
        numGlobalSetsWithMaxSize = tilingData->numGlobalSetsWithMaxSize;
        numGlobalSetsWithMinSize = tilingData->numGlobalSetsWithMinSize;
        globSetMinOffset = numGlobalSetsWithMaxSize * globSetMaxSize;

        ubSize = tilingData->ubSize;
        rowSize = tilingData->aRowSize;
        rowSizeAligned = tilingData->aRowSizeAligned;
        rowMask = tilingData->nMaskSize;
        rowRepeatsNum = tilingData->nRepeatsNum;
        rowTail = tilingData->nTailSize;
        uMSize = tilingData->uMSize;
        uNSize = tilingData->uNSize;
        uNSizeAligned = tilingData->uNSizeAligned;

        vMSize = tilingData->vMSize;
        vNSize = tilingData->vNSize;
        vNSizeAligned = tilingData->vNSizeAligned;
        sMNSize = tilingData->sMNSize;
        sMNSizeAligned = ((sMNSize * sizeof(float) + 31) / 32) * 32;
        numAvailableVectorCores = tilingData->numUsedVectorCores;
        maxBucketSize = tilingData->maxBucketSize;
        tmpRowSize = tilingData->tmpMemRowSize;
        tmpRowSizeAligned = tilingData->tmpMemRowSizeAligned;

        uRepeatsNum = tilingData->uNumRepeats;
        uTailSize = tilingData->uTailSize;
        recordSize = tilingData->recordSize;
        recordSizeAligned = tilingData->recordSizeAligned;
        aGm.SetGlobalBuffer((__gm__ AType*)a);
        aTmpGm.SetGlobalBuffer((__gm__ float*)workspace, mSize * recordSizeAligned);
        sGm.SetGlobalBuffer((__gm__ SType*)s);
        uGm.SetGlobalBuffer((__gm__ UType*)u);
        vGm.SetGlobalBuffer((__gm__ VType*)v);
        pipe_->InitBuffer(ubBuf, ubSize);
        ubMemory = ubBuf.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        if (g_coreType == AIC) {
            return;
        }
        ProcessImpl();
    }
private:
    __aicore__ inline void AllocMem()
    {
        uint32_t offsetPtr = 0;

        //Small blocks for cos/sin vectorisation
        constexpr uint8_t MIN_BLOCK_ELEMENTS_NUM = SMALL_MASK_NUM_ELEMENTS<float>();
        constexpr uint32_t MIN_BLOCK_SIZE = GET_MIN_BLOCK_SIZE<uint32_t>();
        tmpBlockSpace = ubBuf.GetWithOffset<float>(32 * MIN_BLOCK_ELEMENTS_NUM, offsetPtr);

        {
            tmpBlock1 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock2 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock3 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock4 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock5 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock6 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock7 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock8 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock9 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock10 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock11 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock12 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock13 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock14 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock15 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock16 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock17 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock18 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock19 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock20 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock21 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock22 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock23 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock24 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock25 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock26 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock27 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock28 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock29 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock30 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock31 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;

            tmpBlock32 = ubBuf.GetWithOffset<float>(MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
            offsetPtr += MIN_BLOCK_SIZE;
        }

        uint32_t bucketElementsNum = maxBucketSize * recordSizeAligned;
        bucketSpace1 = ubBuf.GetWithOffset<float>(bucketElementsNum, offsetPtr);
        offsetPtr += bucketElementsNum * sizeof(float);
        bucketSpace2 = ubBuf.GetWithOffset<float>(bucketElementsNum, offsetPtr);
        offsetPtr += bucketElementsNum * sizeof(float);
        bucketSpace3 = ubBuf.GetWithOffset<float>(bucketElementsNum, offsetPtr);
        offsetPtr += bucketElementsNum * sizeof(float);

        //tmpMaskSpace, bucketSpace1, bucketSpace2, bucketSpace3
        uint32_t tmpMemElementsNum = 2 * maxBucketSize * tmpRowSizeAligned;
        tmpMemSpace = ubBuf.GetWithOffset<float>(tmpMemElementsNum, offsetPtr);
        offsetPtr += tmpMemElementsNum * sizeof(float);

        uint32_t tmpMaskElementsNum = 2 * maxBucketSize * rowMask;
        if (maxBucketSize == 1) {
            tmpMaskElementsNum = 2 * 2 * rowMask;
        }

        tmpMaskSpace = ubBuf.GetWithOffset<float>(tmpMaskElementsNum, offsetPtr);
        offsetPtr += tmpMaskElementsNum * sizeof(float);
    }

    __aicore__ inline void ProcessImpl()
    {
        AllocMem();
        Preprocess();
        JacobiProcess();
    }

    __aicore__ inline void JacobiProcess()
    {
        for (uint16_t iter = 0; iter < numIterations; ++iter) {
            bool isFinalIteration=(iter==numIterations-1);
            for (uint32_t stageID = 0; stageID < numStages; ++stageID) {
                bool isFinalStage = (stageID==numStages-1);
                numPhases = 1 << stageID;
                for (uint32_t phaseID = 0; phaseID < numPhases; ++phaseID) {
                    bool isFinalPhaseID = (phaseID==numPhases-1);
                    uint16_t leftGlobalSetId = 0;
                    uint16_t rightGlobalSetId = 0;
                    GetLeftRightGlobalSetIds(leftGlobalSetId, rightGlobalSetId, phaseID, numPhases, coreIdx, numAvailableVectorCores);
                    if (leftGlobalSetId < numGlobalSets && rightGlobalSetId < numGlobalSets) {
                        bool leftSetHasMaxSize = (leftGlobalSetId < numGlobalSetsWithMaxSize);
                        bool rightSetHasMaxSize = (rightGlobalSetId < numGlobalSetsWithMaxSize);
                        uint32_t leftBegin = (leftSetHasMaxSize) ? leftGlobalSetId * globSetMaxSize :
                            globSetMinOffset + (leftGlobalSetId - numGlobalSetsWithMaxSize) * globSetMinSize;
                        uint32_t leftEnd = (leftSetHasMaxSize) ? leftBegin + globSetMaxSize : leftBegin + globSetMinSize;
                        uint32_t rightBegin = (rightSetHasMaxSize) ? rightGlobalSetId * globSetMaxSize :
                            globSetMinOffset + (rightGlobalSetId - numGlobalSetsWithMaxSize) * globSetMinSize;
                        uint32_t rightEnd = (rightSetHasMaxSize) ? rightBegin + globSetMaxSize : rightBegin + globSetMinSize;
                        bool isSaveUV=isFinalIteration&&isFinalStage&&isFinalPhaseID;
                        for(uint32_t batchIdx=0;batchIdx<batchSize; ++batchIdx){
                            batchOffsetI = batchIdx*mSize*nSize;
                            batchOffsetU = batchIdx*uMSize*uNSize;
                            batchOffsetV = batchIdx*vMSize*vNSize;
                            batchOffsetS = batchIdx*sMNSize;
                            batchOffsetTmp = batchIdx*sMNSize*recordSizeAligned;
                            if (stageID == 0) {
                                ApplyGlobalSet(leftBegin, leftEnd, iter == 0);
                                ApplyGlobalSet(rightBegin, rightEnd, iter == 0);
                                ApplyGlobalSet(leftBegin, leftEnd, rightBegin, rightEnd, isSaveUV);
                            } else {
                                ApplyGlobalSet(leftBegin, leftEnd, rightBegin, rightEnd, isSaveUV);
                            }
                        }
                    }
                    CrossCoreSetFlag<0x0, PIPE_MTE3>(INTERNAL_SYNC_ALL);
                    CrossCoreWaitFlag(INTERNAL_SYNC_ALL);
                }
            }
        }
    }

    __aicore__ inline void GetLeftRightGlobalSetIds(uint16_t& dstLeftGlobalSetId, uint16_t& dstRightGlobalSetId,
        const uint32_t phaseID, const uint32_t numPhases, const uint32_t coreID, const uint32_t numCores)
    {

        uint32_t groupSize = numPhases * 2;
        uint32_t coresInGroup = groupSize / 2;
        if (coresInGroup > numCores) {
            //usually on last stage when number of coresInGroup is not equal real number of cores,
            // we try to reassing tasks from imaginary cores to real cores
            uint32_t realCoreIdx = 0;
            for (uint32_t virtualCoreIdx = 0; virtualCoreIdx < coresInGroup; virtualCoreIdx++)
            {
                uint32_t groupID = virtualCoreIdx / coresInGroup;
                uint32_t offsetInGroup = virtualCoreIdx % coresInGroup;
                uint16_t leftGlobalSetIdForVirtualCore = groupID * groupSize + offsetInGroup;
                uint32_t rightGlobalSetIdForVirtualCore = groupID * groupSize + coresInGroup + (offsetInGroup + phaseID) % numPhases;
                if (leftGlobalSetIdForVirtualCore >= numGlobalSets ||
                    rightGlobalSetIdForVirtualCore >= numGlobalSets) {
                    continue;
                }
                if (realCoreIdx == coreID) {
                    dstLeftGlobalSetId = leftGlobalSetIdForVirtualCore;
                    dstRightGlobalSetId = rightGlobalSetIdForVirtualCore;
                    return;
                }
                realCoreIdx++;
            }

            dstLeftGlobalSetId = numGlobalSets;
            dstRightGlobalSetId = numGlobalSets;
        } else {
            uint32_t groupID = coreID / coresInGroup;
            uint32_t offsetInGroup = coreID % coresInGroup;
            dstLeftGlobalSetId = groupID * groupSize + offsetInGroup;
            dstRightGlobalSetId = groupID * groupSize + coresInGroup + (offsetInGroup + phaseID) % numPhases;
        }

    }

    __aicore__ inline void JacobiProcessMultiCoreSimulator()
    {
        uint32_t nCoresGG = GetBlockNum();
        if (coreIdx > 0) {
            return;
        }
        for (uint16_t iter = 0; iter < numIterations; ++iter) {
            for (uint32_t stageID = 0; stageID < numStages; ++stageID) {
                numPhases = 1 << stageID;
                for (uint32_t phaseID = 0; phaseID < numPhases; ++phaseID) {
                    for (uint32_t coreIdx_ = 0; coreIdx_ < nCoresGG; ++coreIdx_) {
                        uint16_t leftGlobalSetId = 0;
                        uint16_t rightGlobalSetId = 0;

                        GetLeftRightGlobalSetIds(leftGlobalSetId, rightGlobalSetId, phaseID, numPhases, coreIdx_, nCoresGG);
                        if (leftGlobalSetId < numGlobalSets && rightGlobalSetId < numGlobalSets) {
                            bool leftSetHasMaxSize = (leftGlobalSetId < numGlobalSetsWithMaxSize);
                            bool rightSetHasMaxSize = (rightGlobalSetId < numGlobalSetsWithMaxSize);
                            uint32_t leftBegin = (leftSetHasMaxSize) ? leftGlobalSetId * globSetMaxSize :
                                globSetMinOffset + (leftGlobalSetId - numGlobalSetsWithMaxSize) * globSetMinSize;
                            uint32_t leftEnd = (leftSetHasMaxSize) ? leftBegin + globSetMaxSize : leftBegin + globSetMinSize;
                            uint32_t rightBegin = (rightSetHasMaxSize) ? rightGlobalSetId * globSetMaxSize :
                                globSetMinOffset + (rightGlobalSetId - numGlobalSetsWithMaxSize) * globSetMinSize;
                            uint32_t rightEnd = (rightSetHasMaxSize) ? rightBegin + globSetMaxSize : rightBegin + globSetMinSize;
                            for(uint32_t batchIdx=0;batchIdx<batchSize; ++batchIdx){
                                batchOffsetI = batchIdx*mSize*nSize;
                                batchOffsetU = batchIdx*uMSize*uNSize;
                                batchOffsetV = batchIdx*vMSize*vNSize;
                                batchOffsetS = batchIdx*sMNSize;
                                batchOffsetTmp = batchIdx*sMNSize*recordSizeAligned;
                                if (stageID == 0) {
                                    ApplyGlobalSet(leftBegin, leftEnd, iter == 0);
                                    ApplyGlobalSet(rightBegin, rightEnd, iter == 0);
                                    ApplyGlobalSet(leftBegin, leftEnd, rightBegin, rightEnd);
                                } else {
                                    ApplyGlobalSet(leftBegin, leftEnd, rightBegin, rightEnd);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    __aicore__ inline void Preprocess()
    {
        batchOffsetV = 0;
        batchOffsetU = 0;
        batchOffsetS = 0;
        batchOffsetI = 0;
        batchOffsetTmp =0;
    }

    __aicore__ inline void ApplyGlobalSet(const uint16_t lBegin, const uint16_t lEnd, const uint16_t rBegin,
        const uint16_t rEnd, const bool isFinalIteration=false)
    {
        if (coreIdx >= numAvailableVectorCores) {
            return;
        }
        uint32_t leftSetSize = lEnd - lBegin;
        uint32_t rightSetSize = rEnd - rBegin;
        uint32_t numBucketsInLeftSet = (leftSetSize + maxBucketSize - 1) / maxBucketSize;
        uint32_t numBucketsInRightSet = (rightSetSize + maxBucketSize - 1) / maxBucketSize;
        LocalTensor<float> leftBucketSpace, rightBucketSpace, leftNorms, rightNorms;
        {
            //Roole of choosing leftBucketSpace
            leftBucketSpace = bucketSpace1;
            rightBucketSpace = bucketSpace2;
            leftNorms = tmpBlock31;
            rightNorms = tmpBlock32;
        }
        for (uint32_t leftBucketId = 0; leftBucketId < numBucketsInLeftSet; ++leftBucketId) {
            uint32_t leftBucketBegin = lBegin + leftBucketId * maxBucketSize;
            uint32_t leftBucketEnd = MIN(leftBucketBegin + maxBucketSize, lEnd);
            LoadBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd, false);
            for (uint32_t rightBucketId = 0; rightBucketId < numBucketsInRightSet; ++rightBucketId) {
                uint32_t rightBucketBegin = rBegin + rightBucketId * maxBucketSize;
                uint32_t rightBucketEnd = MIN(rightBucketBegin + maxBucketSize, rEnd);
                LoadBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd, false);
                event_t evtMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(evtMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(evtMte2ToV);
                ApplyBucket(leftBucketSpace, leftNorms, rightBucketSpace, rightNorms, leftBucketEnd - leftBucketBegin, rightBucketEnd - rightBucketBegin, false);
                event_t evtVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(evtVToMte3);
                WaitFlag<HardEvent::V_MTE3>(evtVToMte3);
                if(!isFinalIteration || (isFinalIteration&&leftBucketId!=numBucketsInLeftSet-1))
                {
                    SaveBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd);
                }
                else
                {
                    PipeBarrier<PIPE_V>();
                    SaveOutputs<false>(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd);
                }
                if (rightBucketId < numBucketsInRightSet - 1) {
                    event_t evtMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                    SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
                }
            }
            if(!isFinalIteration)
            {
                SaveBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd);
            }
            else
            {
                PipeBarrier<PIPE_V>();
                SaveOutputs<true>(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd);
            }
            event_t evtMte3ToMte2D = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2D);
            WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2D);
        }

    }

    __aicore__ inline void ApplyGlobalSet(const uint16_t sBegin, const uint16_t sEnd, const bool isFirstLoadData = false)
    {
        if (coreIdx >= numAvailableVectorCores) {
            return;
        }
        uint32_t setSize = sEnd - sBegin;
        uint32_t numBucketsInSet = (setSize + maxBucketSize - 1) / maxBucketSize;
        bool isFirstLoadDataLocal = isFirstLoadData;
        LocalTensor<float> leftBucketSpace, rightBucketSpace, leftNorms, rightNorms;
        {
            //Roole of choosing leftBucketSpace
            leftBucketSpace = bucketSpace1;
            rightBucketSpace = bucketSpace2;
            leftNorms = tmpBlock31;
            rightNorms = tmpBlock32;
        }
        for (uint32_t leftBucketId = 0; leftBucketId < numBucketsInSet; ++leftBucketId) {
            uint32_t leftBucketBegin = sBegin + leftBucketId * maxBucketSize;
            uint32_t leftBucketEnd = MIN(leftBucketBegin + maxBucketSize, sEnd);

            LoadBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd, isFirstLoadDataLocal);
            event_t evtMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evtMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(evtMte2ToV);
            if(isFirstLoadDataLocal)
            {
                InitV(leftBucketSpace, leftBucketBegin, leftBucketEnd);
            }
            ApplyBucket(leftBucketSpace, leftNorms, leftBucketEnd - leftBucketBegin, isFirstLoadDataLocal);
            if (leftBucketId + 1 >= numBucketsInSet) {
                event_t evtVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(evtVToMte3);
                WaitFlag<HardEvent::V_MTE3>(evtVToMte3);
            }

            for (uint32_t rightBucketId = leftBucketId + 1; rightBucketId < numBucketsInSet; ++rightBucketId) {
                uint32_t rightBucketBegin = sBegin + rightBucketId * maxBucketSize;
                uint32_t rightBucketEnd = MIN(rightBucketBegin + maxBucketSize, sEnd);
                LoadBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd, isFirstLoadDataLocal);
                event_t evtMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(evtMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(evtMte2ToV);
                if(isFirstLoadDataLocal)
                {
                    InitV(rightBucketSpace, rightBucketBegin, rightBucketEnd);
                }
                ApplyBucket(leftBucketSpace, leftNorms, rightBucketSpace, rightNorms, leftBucketEnd - leftBucketBegin, rightBucketEnd - rightBucketBegin, isFirstLoadDataLocal);
                event_t evtVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(evtVToMte3);
                WaitFlag<HardEvent::V_MTE3>(evtVToMte3);
                SaveBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd);
                if (rightBucketId != numBucketsInSet - 1)
                {
                    event_t evtMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                    SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
                    WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
                }
            }
            isFirstLoadDataLocal = false;
            SaveBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd);
            event_t evtMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2);
        }
    }

    __aicore__ inline void ApplyBucket(const LocalTensor<float>& leftBucketSpace, const LocalTensor<float>& leftNorms, const LocalTensor<float>& rightBucketSpace,
        const LocalTensor<float>& rightNorms, const uint32_t leftBucketSize, const uint32_t rightBucketSize, const bool calculateRightNorms)
    {
        bool leftIsLess = (leftBucketSize <= rightBucketSize);
        uint32_t aNum = (leftIsLess) ? rightBucketSize : leftBucketSize;
        RotateMaskInfo maskInfo;
        bool calculateRightNorm = calculateRightNorms;
        for (uint32_t aOffset = 0; aOffset < aNum; ++aOffset) {
            maskInfo.normUpdateMask = 0;
            maskInfo.leftBucketSize = leftBucketSize;
            maskInfo.rightBucketSize = rightBucketSize;
            maskInfo.rightMaskU64Val = 0;
            maskInfo.leftMaskU64Val = 0;
            if (leftIsLess) {
                maskInfo.numPairs = leftBucketSize;
                for (uint32_t idx = 0; idx < leftBucketSize; ++idx) {
                    uint8_t rightIdx = (idx + aOffset) % rightBucketSize;
                    maskInfo.rightRowsIds[idx] = rightIdx;
                    maskInfo.rightMaskU8Val[idx] = 1 << rightIdx;
                    maskInfo.normUpdateMask |= 1 << rightIdx;
                    maskInfo.leftRowsIds[idx] = idx;
                    maskInfo.leftMaskU8Val[idx] = 1 << idx;
                }
                if (!calculateRightNorm) {
                    RotateBucket<true, false>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize, rightBucketSize, maskInfo);
                } else {
                    RotateBucket<true, true>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize, rightBucketSize, maskInfo);
                }
            } else {
                maskInfo.numPairs = rightBucketSize;
                for (uint32_t idx = 0; idx < rightBucketSize; ++idx) {
                    uint8_t leftIdx = (idx + aOffset) % leftBucketSize;
                    maskInfo.leftRowsIds[idx] = leftIdx;
                    maskInfo.leftMaskU8Val[idx] = 1 << leftIdx;
                    maskInfo.normUpdateMask |= 1 << leftIdx;
                    maskInfo.rightRowsIds[idx] = idx;
                    maskInfo.rightMaskU8Val[idx] = 1 << idx;
                }
                if (!calculateRightNorm) {
                    RotateBucket<false, false>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize, rightBucketSize, maskInfo);
                } else {
                    RotateBucket<false, true>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize, rightBucketSize, maskInfo);
                }
            }
            calculateRightNorm = false;
        }

    }

    template<typename T = uint32_t>
    __aicore__ inline T  ScalarLog2(const T val)
    {
        T tmp = val;
        T res = 0;
        while (tmp > 0) {
            res++;
            tmp = tmp >> 1;
        }

        return res - 1;
    }

    __aicore__ inline void ApplyBucket(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& norms, const uint32_t bucketSize, const bool calculateNorms)
    {
        if (bucketSize < 2) {
            if (calculateNorms) {
                RotateMaskInfo maskInfo;
                maskInfo.numPairs = bucketSize;
                for (int idx = 0; idx < bucketSize; ++idx) {
                    maskInfo.leftRowsIds[idx] = idx;
                }
                CalculateNormOfBucket(norms, bucketSpace, tmpMemSpace, tmpMaskSpace, maskInfo.leftRowsIds, bucketSize);
                PipeBarrier<PIPE_V>();
            }
            return;
        }
        uint32_t extendedBucketSize = 0;
        uint32_t aOffsetNumStages = 0;
        {
            uint32_t lg2 = ScalarLog2(bucketSize);
            uint32_t possibleEtendedBucketSize = 1 << lg2;
            if (possibleEtendedBucketSize == bucketSize) {
                aOffsetNumStages = lg2;
                extendedBucketSize = bucketSize;
            } else {
                aOffsetNumStages = lg2 + 1;
                extendedBucketSize = 1 << (aOffsetNumStages);
            }
        }
        RotateMaskInfo maskInfo;
        bool isCalculateNorm = calculateNorms;

        for (uint32_t aOffsetStage = 0; aOffsetStage < aOffsetNumStages; ++aOffsetStage) {
            uint32_t aOffsetNumPhases = 1 << aOffsetStage;

            for (uint32_t aOffsetPhase = 0; aOffsetPhase < aOffsetNumPhases; ++aOffsetPhase) {
                maskInfo.numPairs = 0;

                uint32_t groupSize = aOffsetNumPhases * 2;
                uint32_t pairsInGroup = groupSize / 2;
                maskInfo.leftMaskU64Val = 0;
                maskInfo.rightMaskU64Val = 0;
                maskInfo.normUpdateMask = 0;
                for (uint16_t pairID = 0; pairID < extendedBucketSize / 2; ++pairID) {
                    uint32_t groupID = pairID / pairsInGroup;
                    uint32_t offsetInGroup = pairID % pairsInGroup;

                    uint32_t leftRowIdx = groupID * groupSize + offsetInGroup;
                    uint32_t rightRowIdx = groupID * groupSize + pairsInGroup + (offsetInGroup + aOffsetPhase) % aOffsetNumPhases;
                    if ((leftRowIdx < bucketSize) && (rightRowIdx < bucketSize)) {
                        maskInfo.numPairs++;
                        maskInfo.leftRowsIds[maskInfo.numPairs - 1] = static_cast<uint8_t>(leftRowIdx);
                        maskInfo.leftMaskU8Val[maskInfo.numPairs - 1] = 1 << leftRowIdx;
                        maskInfo.rightRowsIds[maskInfo.numPairs - 1] = static_cast<uint8_t>(rightRowIdx);
                        maskInfo.rightMaskU8Val[maskInfo.numPairs - 1] = 1 << rightRowIdx;
                        maskInfo.normUpdateMask |= 1 << leftRowIdx;
                        maskInfo.normUpdateMask |= 1 << rightRowIdx;
                    }

                }

                maskInfo.leftBucketSize = maskInfo.rightBucketSize = bucketSize;
                if (!isCalculateNorm) {
                    RotateSelfBucket<false>(bucketSpace, norms, bucketSize, maskInfo);
                } else {
                    RotateSelfBucket<true>(bucketSpace, norms, bucketSize, maskInfo);
                }
                isCalculateNorm = false;
            }
        }

    }

    template<bool BUCKET_1_TARGET = true, bool NEED_CALCULATE_RIGHT_NORM = false>
    __aicore__ inline void CalculateDotProducts(const LocalTensor<float>& sqNormsBucket1, const LocalTensor<float>& sqNormsBucket2,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& bucket1, const LocalTensor<float>& norms1,
        const LocalTensor<float>& bucket2, const LocalTensor<float>& norms2, const RotateMaskInfo& maskInfo)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        LocalTensor<float>& tmpBucket1 = tmpMemSpace;
        LocalTensor<float> tmpBucket2 = tmpMemSpace[maxBucketSize * nSizeAligned];
        LocalTensor<float>& dotProductTmp1 = tmpMaskSpace;
        LocalTensor<float> dotProductTmp2 = tmpMaskSpace[rowMask * maxBucketSize];
        const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
        LocalTensor<float>& tmpNorm1 = tmpBlock18;
        LocalTensor<float>& tmpNorm2 = tmpBlock17;
#ifdef RUNTIME_NORM_GENERATION
        {
            Duplicate(dotProductTmp1, 0.f, 2 * rowMask * maxBucketSize);
            for (uint16_t rowIdx = 0; rowIdx < maskInfo.leftBucketSize; ++rowIdx) {
                Mul(tmpBucket1[rowIdx * nSizeAligned], bucket1[rowIdx * nSizeAligned],
                    bucket1[rowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1,1, repeatStride, repeatStride, repeatStride });
            }
            for (uint16_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
                Mul(tmpBucket2[rowIdx * nSizeAligned], bucket2[rowIdx * nSizeAligned],
                    bucket2[rowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1,1, repeatStride, repeatStride, repeatStride });
            }
            PipeBarrier<PIPE_V>();
            for (uint16_t rowIdx = 0; rowIdx < maskInfo.leftBucketSize; ++rowIdx) {
                Add(dotProductTmp1[rowIdx * rowMask], tmpBucket1[rowIdx * nSizeAligned], dotProductTmp1[rowIdx * rowMask], rowMask, rowRepeatsNum, { 1,1,1, 0, repeatStride, 0 });
            }
            for (uint16_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
                Add(dotProductTmp2[rowIdx * rowMask], tmpBucket2[rowIdx * nSizeAligned], dotProductTmp2[rowIdx * rowMask], rowMask, rowRepeatsNum, { 1,1,1, 0, repeatStride, 0 });
            }
            PipeBarrier<PIPE_V>();

            WholeReduceSum(tmpNorm1, dotProductTmp1, rowMask, static_cast<uint8_t>(maskInfo.leftBucketSize), 1, 1, repeatStride);
            WholeReduceSum(tmpNorm2, dotProductTmp2, rowMask, static_cast<uint8_t>(maskInfo.rightBucketSize), 1, 1, repeatStride);
            PipeBarrier<PIPE_V>();
            Sqrt(norms1, tmpNorm1, SMNE);
            Sqrt(norms2, tmpNorm2, SMNE);
            PipeBarrier<PIPE_V>();
        }
#endif
        if constexpr (NEED_CALCULATE_RIGHT_NORM) {
            CalculateRightNormAndDotProductForDoubleBucketsCase(dotProducts, norms2, bucket1, bucket2, maskInfo, tmpBucket1, tmpBucket2, dotProductTmp1, dotProductTmp2);
        } else {
            CalculateDotProductForDoubleBucketCase(dotProducts, bucket1, bucket2, maskInfo, tmpBucket1, dotProductTmp1);
        }

        PipeBarrier<PIPE_V>();
        LocalTensor<float>& broadcastedNorms = dotProductTmp1;
        LocalTensor<float> broadCastedNormsWithZero = dotProductTmp1[rowMask];
        Mul(sqNormsBucket1, norms1, norms1, SMNE);
        Mul(sqNormsBucket2, norms2, norms2, SMNE);

        PipeBarrier<PIPE_V>();
        if constexpr (BUCKET_1_TARGET) {
            GatherNormVector(sqNormsBucket2, sqNormsBucket2, maskInfo.rightMaskU64Val, broadCastedNormsWithZero, broadcastedNorms);
        } else {
            GatherNormVector(sqNormsBucket1, sqNormsBucket1, maskInfo.leftMaskU64Val, broadCastedNormsWithZero, broadcastedNorms);
        }

    }

    __aicore__ inline void CalculateRightNormAndDotProductForDoubleBucketsCase(const LocalTensor<float>& dstDotProduct, const LocalTensor<float>& dstRightNorms, const LocalTensor<float>& bucket1,
        const LocalTensor<float>& bucket2, const RotateMaskInfo& maskInfo, const LocalTensor<float>& tmpBucket1, const LocalTensor<float>& tmpBucket2,
        const LocalTensor<float>& dotProductTmp1, const LocalTensor<float>& dotProductTmp2)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
        Duplicate(dotProductTmp1, 0.f, rowMask * maskInfo.numPairs);
        Duplicate(dotProductTmp2, 0.f, rowMask * maskInfo.rightBucketSize);
        for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
            uint16_t leftRowIdx = maskInfo.leftRowsIds[rowIdx];
            uint16_t rightRowIdx = maskInfo.rightRowsIds[rowIdx];
            Mul(tmpBucket1[rowIdx * nSizeAligned], bucket1[leftRowIdx * nSizeAligned],
                bucket2[rightRowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1,1, repeatStride, repeatStride, repeatStride });
        }
        for (uint8_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
            Mul(tmpBucket2[rowIdx * nSizeAligned], bucket2[rowIdx * nSizeAligned],
                bucket2[rowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1,1, repeatStride, repeatStride, repeatStride });
        }
        PipeBarrier<PIPE_V>();
        for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
            Add(dotProductTmp1[rowIdx * rowMask], tmpBucket1[rowIdx * nSizeAligned], dotProductTmp1[rowIdx * rowMask], rowMask, rowRepeatsNum, { 1,1,1, 0, repeatStride, 0 });
        }
        for (uint16_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
            Add(dotProductTmp2[rowIdx * rowMask], tmpBucket2[rowIdx * nSizeAligned], dotProductTmp2[rowIdx * rowMask], rowMask, rowRepeatsNum, { 1,1,1, 0, repeatStride, 0 });
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstDotProduct, dotProductTmp1, rowMask, static_cast<uint8_t>(maskInfo.numPairs), 1, 1, repeatStride);
        WholeReduceSum(dstRightNorms, dotProductTmp2, rowMask, static_cast<uint8_t>(maskInfo.rightBucketSize), 1, 1, repeatStride);
        PipeBarrier<PIPE_V>();
        Sqrt(dstRightNorms, dstRightNorms, SMNE);

    }


    __aicore__ inline void CalculateDotProductForDoubleBucketCase(const LocalTensor<float>& dstDotProduct, const LocalTensor<float>& bucket1,
        const LocalTensor<float>& bucket2, const RotateMaskInfo& maskInfo, const LocalTensor<float>& tmpBucket1, const LocalTensor<float>& dotProductTmp1)
    {
        const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
        Duplicate(dotProductTmp1, 0.f, rowMask * maskInfo.numPairs);
        for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
            uint16_t leftRowIdx = maskInfo.leftRowsIds[rowIdx];
            uint16_t rightRowIdx = maskInfo.rightRowsIds[rowIdx];
            Mul(tmpBucket1[rowIdx * nSizeAligned], bucket1[leftRowIdx * nSizeAligned],
                bucket2[rightRowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1,1, repeatStride, repeatStride, repeatStride });
        }
        PipeBarrier<PIPE_V>();
        for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
            Add(dotProductTmp1[rowIdx * rowMask], tmpBucket1[rowIdx * nSizeAligned], dotProductTmp1[rowIdx * rowMask], rowMask, rowRepeatsNum, { 1,1,1, 0, repeatStride, 0 });
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstDotProduct, dotProductTmp1, rowMask, static_cast<uint8_t>(maskInfo.numPairs), 1, 1, repeatStride);
    }

    __aicore__ inline void CalculateNormOfBucket(const LocalTensor<float>& dstNormVector, const LocalTensor<float>& bucket,
        const LocalTensor<float>& tmpBucket, const LocalTensor<float>& dotProductTmp, const uint8_t(&rowIds)[sizeof(uint64_t)], const uint32_t bucketSize)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;

        Duplicate(dotProductTmp, 0.f, rowMask * bucketSize);
        for (uint16_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
            uint16_t rowId = rowIds[rowIdx];
            Mul(tmpBucket[rowIdx * nSizeAligned], bucket[rowId * nSizeAligned],
                bucket[rowId * nSizeAligned], rowMask, rowRepeatsNum, { 1,1,1, repeatStride, repeatStride, repeatStride });
        }
        PipeBarrier<PIPE_V>();
        for (uint16_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
            Add(dotProductTmp[rowIdx * rowMask], tmpBucket[rowIdx * nSizeAligned], dotProductTmp[rowIdx * rowMask], rowMask, rowRepeatsNum, { 1,1,1, 0, repeatStride, 0 });
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstNormVector, dotProductTmp, rowMask, static_cast<uint8_t>(bucketSize), 1, 1, repeatStride);
        PipeBarrier<PIPE_V>();
        Sqrt(dstNormVector, dstNormVector, SMNE);
    }

    __aicore__ inline void GatherNormVector(const LocalTensor<float>& dstNormVector, const LocalTensor<float>& srcNormVector, const uint64_t mask, const LocalTensor<float>& workLocal1,
        const LocalTensor<float>& workLocal2)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        Duplicate(workLocal1, 0.f, rowMask);

        PipeBarrier<PIPE_V>();

        Copy(workLocal2, srcNormVector, SMNE, SMNE, { 1,1,1,0 });
        PipeBarrier<PIPE_V>();
        uint64_t mask_[2] = { mask, 0 };
        Copy(workLocal1, workLocal2, mask_, 1, { 1, 1, 1, 1 });
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstNormVector, workLocal1, SMNE, static_cast<uint8_t>(SMNE), 1, 1, 1);
        PipeBarrier<PIPE_V>();

    }

    template<bool NEED_CALCULATE_NORM>
    __aicore__ inline void CalculateDotProductsForSelfBucket(const LocalTensor<float>& sqNorms1, const LocalTensor<float>& sqNorms2,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& normVector, const LocalTensor<float>& bucket, const uint32_t bucketSize, const RotateMaskInfo& maskInfo)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();


        LocalTensor<float>& tmpBucket1 = tmpMemSpace;
        LocalTensor<float> tmpBucket2 = tmpMemSpace[maxBucketSize * nSizeAligned];
        LocalTensor<float>& dotProductTmp1 = tmpMaskSpace;
        LocalTensor<float> dotProductTmp2 = tmpMaskSpace[maxBucketSize * rowMask];

        const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
#ifdef RUNTIME_NORM_GENERATION
        {
            Duplicate(dotProductTmp1, 0.f, rowMask * bucketSize);
            const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
            for (uint16_t rowID = 0; rowID < bucketSize; ++rowID) {
                Mul(tmpBucket1[rowID * nSizeAligned], bucket[rowID * nSizeAligned], bucket[rowID * nSizeAligned], rowMask, rowRepeatsNum, { 1,1,1, repeatStride, repeatStride, repeatStride });
            }

            PipeBarrier<PIPE_V>();
            for (uint16_t rowID = 0; rowID < bucketSize; ++rowID) {
                Add(dotProductTmp1[rowID * rowMask], tmpBucket1[rowID * nSizeAligned], dotProductTmp1[rowID * rowMask], rowMask, rowRepeatsNum, { 1,1,1, 0, repeatStride, 0 });
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(normVector, dotProductTmp1, rowMask, static_cast<uint8_t>(bucketSize), 1, 1, repeatStride);
            PipeBarrier<PIPE_V>();
            Sqrt(normVector, normVector, SMNE);
            PipeBarrier<PIPE_V>();
        }
#endif

        if constexpr (NEED_CALCULATE_NORM) {
            CalculateRightNormAndDotProductForDoubleBucketsCase(dotProducts, normVector, bucket, bucket, maskInfo, tmpBucket1, tmpBucket2, dotProductTmp1, dotProductTmp2);
        } else {
            CalculateDotProductForDoubleBucketCase(dotProducts, bucket, bucket, maskInfo, tmpBucket1, dotProductTmp1);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<float>& broadcastedNorms = dotProductTmp1;
        LocalTensor<float> sqNorms1BroadCasted = dotProductTmp1[rowMask];

        GatherNormsVectorToLeftRightNorms(sqNorms1, sqNorms2, normVector, maskInfo, broadcastedNorms, sqNorms1BroadCasted);
        PipeBarrier<PIPE_V>();
        Mul(sqNorms1, sqNorms1, sqNorms1, SMNE);
        Mul(sqNorms2, sqNorms2, sqNorms2, SMNE);

    }

    __aicore__ inline void GatherNormsVectorToLeftRightNorms(const LocalTensor<float>& dstSqNorms1, const LocalTensor<float>& dstSqNorms2, const LocalTensor<float>& srcNorms,
        const RotateMaskInfo& maskInfo, const LocalTensor<float>& workLocal1, const LocalTensor<float>& workLocal2)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        Copy(workLocal1, srcNorms, SMNE, SMNE, { 1,1,1,0 });
        Duplicate(workLocal2, 0.f, 2 * rowMask);
        PipeBarrier<PIPE_V>();
        uint64_t mask1[2] = { maskInfo.leftMaskU64Val, 0 };
        uint64_t mask2[2] = { maskInfo.rightMaskU64Val, 0 };
        Copy(workLocal2, workLocal1, mask1, 1, { 1, 1, 1, 1});
        Copy(workLocal2[rowMask], workLocal1, mask2, 1, { 1, 1, 1, 1});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstSqNorms1, workLocal2, SMNE, static_cast<uint8_t>(maskInfo.numPairs), 1, 1, 1);
        WholeReduceSum(dstSqNorms2, workLocal2[rowMask], SMNE, static_cast<uint8_t>(maskInfo.numPairs), 1, 1, 1);

    }

    template<bool LEFT_TARGET = true, bool NEED_CALCULATE_RIGHT_NORM = false>
    __aicore__ inline void RotateBucket(const LocalTensor<float>& leftBucketSpace, const LocalTensor<float>& rightBucketSpace, const LocalTensor<float>& leftNorms,
        const LocalTensor<float>& rightNorms, const uint32_t leftBucketSize, const uint32_t rightBucketSize, const RotateMaskInfo& maskInfo)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        constexpr uint8_t BMNE = BIG_MASK_NUM_ELEMENTS<float>();

        LocalTensor<float>& leftSquareNorms = tmpBlock9;
        LocalTensor<float>& rightSquareNorms = tmpBlock10;
        LocalTensor<float>& dotProducts = tmpBlock11;
        CalculateDotProducts<LEFT_TARGET, NEED_CALCULATE_RIGHT_NORM>(leftSquareNorms, rightSquareNorms, dotProducts,
            leftBucketSpace, leftNorms, rightBucketSpace, rightNorms, maskInfo);
        PipeBarrier<PIPE_V>();
        LocalTensor<float>& sinTensor = tmpBlock12;
        LocalTensor<float>& cosTensor = tmpBlock13;
        CalculateSinCos(leftSquareNorms, rightSquareNorms, dotProducts, sinTensor, cosTensor);
        PipeBarrier<PIPE_V>();
        {
            LocalTensor<float>& tmpVec1 = tmpBlock20;
            LocalTensor<float>& tmpVec2 = tmpBlock21;
            LocalTensor<float>& tmpVec3 = tmpBlock22;
            LocalTensor<float>& tmpVec4 = tmpBlock23;
            LocalTensor<float>& sinSq = tmpBlock24;
            LocalTensor<float>& cosSq = tmpBlock25;
            LocalTensor<float>& sin2XDoub = tmpBlock26;
            LocalTensor<float>& broadcastedNorms = tmpMaskSpace;
            LocalTensor<float> broadcastedNormsWithZero = tmpMaskSpace[rowMask];
            LocalTensor<float>& targetSquareNorms = (LEFT_TARGET) ? leftSquareNorms : rightSquareNorms;
            LocalTensor<float>& nonTargetSquareNorms = (LEFT_TARGET) ? rightSquareNorms : leftSquareNorms;
            const LocalTensor<float>& targetNorms = (LEFT_TARGET) ? leftNorms : rightNorms;
            const LocalTensor<float>& nonTargetNorms = (LEFT_TARGET) ? rightNorms : leftNorms;
            uint64_t targetMaskUpdate = (1 << maskInfo.numPairs) - 1;
            uint64_t targetMaskUpdater[2] = { targetMaskUpdate, 0 };
            uint64_t nonTargetScatteringMask[2] = { 0, 0 };
            uint64_t nonTargetMaskUpdater[2] = { maskInfo.normUpdateMask, 0 };
            if constexpr (LEFT_TARGET) {
                nonTargetScatteringMask[0] = maskInfo.rightMaskU64Val;
            } else {
                nonTargetScatteringMask[0] = maskInfo.leftMaskU64Val;
            }
            Duplicate(broadcastedNormsWithZero, 0.f, rowMask);
            Mul(sinSq, sinTensor, sinTensor, SMNE);
            Mul(cosSq, cosTensor, cosTensor, SMNE);
            Mul(sin2XDoub, sinTensor, cosTensor, SMNE);
            PipeBarrier<PIPE_V>();
            Muls(sin2XDoub, sin2XDoub, 2.f, SMNE);
            Mul(tmpVec1, leftSquareNorms, cosSq, SMNE);
            Mul(tmpVec2, leftSquareNorms, sinSq, SMNE);
            Mul(tmpVec3, rightSquareNorms, sinSq, SMNE);
            Mul(tmpVec4, rightSquareNorms, cosSq, SMNE);
            PipeBarrier<PIPE_V>();
            Mul(sin2XDoub, sin2XDoub, dotProducts, SMNE);
            Add(tmpVec1, tmpVec1, tmpVec3, SMNE);
            Add(tmpVec2, tmpVec2, tmpVec4, SMNE);
            PipeBarrier<PIPE_V>();
            Sub(leftSquareNorms, tmpVec1, sin2XDoub, SMNE);
            Add(rightSquareNorms, tmpVec2, sin2XDoub, SMNE);
            PipeBarrier<PIPE_V>();
            Sqrt(targetNorms, targetSquareNorms, targetMaskUpdater, 1, { 1,1,1,1 });
            Brcb(broadcastedNorms, nonTargetSquareNorms, 1, { 1,8 });
            PipeBarrier<PIPE_V>();
            Duplicate(nonTargetSquareNorms, 0.f, SMNE);
            Copy(broadcastedNormsWithZero, broadcastedNorms, nonTargetScatteringMask, 1, { 1,1,1,1 });
            PipeBarrier<PIPE_V>();
            Add(nonTargetSquareNorms, broadcastedNormsWithZero, nonTargetSquareNorms, SMNE, SMNE, { 1,1,1, 0,1,0 });
            PipeBarrier<PIPE_V>();
            Sqrt(nonTargetNorms, nonTargetSquareNorms, nonTargetMaskUpdater, 1, { 1,1,1,1 });
        }
        PipeBarrier<PIPE_V>();
        ApplySinCos<LEFT_TARGET>(leftBucketSpace, leftBucketSize, rightBucketSpace, rightBucketSize, sinTensor, cosTensor, maskInfo);
        PipeBarrier<PIPE_V>();

    }


    //Self bucket rotate
    template<bool NEED_CALCULATE_NORM>
    __aicore__ inline void RotateSelfBucket(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketSize, const RotateMaskInfo& maskInfo)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        constexpr uint8_t BMNE = BIG_MASK_NUM_ELEMENTS<float>();

        LocalTensor<float>& leftSquareNorms = tmpBlock9;
        LocalTensor<float>& rightSquareNorms = tmpBlock10;
        LocalTensor<float>& dotProducts = tmpBlock11;
        LocalTensor<float>& leftNorms = tmpBlock14;
        LocalTensor<float>& rightNorms = tmpBlock15;
        CalculateDotProductsForSelfBucket<NEED_CALCULATE_NORM>(leftSquareNorms, rightSquareNorms, dotProducts, normVector, bucketSpace, bucketSize, maskInfo);
        PipeBarrier<PIPE_V>();
        LocalTensor<float>& sinTensor = tmpBlock12;
        LocalTensor<float>& cosTensor = tmpBlock13;
        Duplicate(sinTensor, 0.f, SMNE);
        Duplicate(cosTensor, 0.f, SMNE);
        CalculateSinCos(leftSquareNorms, rightSquareNorms, dotProducts, sinTensor, cosTensor);
        PipeBarrier<PIPE_V>();
        //(Sqrt(leftSquareNorms), Sqrt(rightSquareNorms))-> normVector
        {
            LocalTensor<float>& tmpVec1 = tmpBlock20;
            LocalTensor<float>& tmpVec2 = tmpBlock21;
            LocalTensor<float>& tmpVec3 = tmpBlock22;
            LocalTensor<float>& tmpVec4 = tmpBlock23;
            LocalTensor<float>& newScatteredNorm = tmpBlock24;
            LocalTensor<float>& sinSq = tmpBlock25;
            LocalTensor<float>& cosSq = tmpBlock26;
            LocalTensor<float>& sin2XDoub = tmpBlock27;
            LocalTensor<float>& broadcastedLeftNorms = tmpMaskSpace;
            LocalTensor<float> broadcastedRightNorms = tmpMaskSpace[rowMask];
            LocalTensor<float> broadcastedScatteredNorms = tmpMaskSpace[2 * rowMask];
            Mul(sinSq, sinTensor, sinTensor, SMNE);
            Mul(cosSq, cosTensor, cosTensor, SMNE);
            Mul(sin2XDoub, sinTensor, cosTensor, SMNE);
            Duplicate(broadcastedScatteredNorms, 0.f, 2 * rowMask);
            PipeBarrier<PIPE_V>();
            Muls(sin2XDoub, sin2XDoub, 2.f, SMNE);
            Mul(tmpVec1, leftSquareNorms, cosSq, SMNE);
            Mul(tmpVec2, leftSquareNorms, sinSq, SMNE);
            Mul(tmpVec3, rightSquareNorms, sinSq, SMNE);
            Mul(tmpVec4, rightSquareNorms, cosSq, SMNE);
            PipeBarrier<PIPE_V>();
            Mul(sin2XDoub, sin2XDoub, dotProducts, SMNE);
            Add(tmpVec1, tmpVec1, tmpVec3, SMNE);
            Add(tmpVec2, tmpVec2, tmpVec4, SMNE);
            PipeBarrier<PIPE_V>();
            Sub(leftSquareNorms, tmpVec1, sin2XDoub, SMNE);
            Add(rightSquareNorms, tmpVec2, sin2XDoub, SMNE);
            PipeBarrier<PIPE_V>();
            Brcb(broadcastedLeftNorms, leftSquareNorms, 1, { 1,8 });
            Brcb(broadcastedRightNorms, rightSquareNorms, 1, { 1,8 });

            PipeBarrier<PIPE_V>();
            uint64_t mask1[2] = { maskInfo.leftMaskU64Val, 0 };
            uint64_t mask2[2] = { maskInfo.rightMaskU64Val, 0 };
            Copy(broadcastedScatteredNorms, broadcastedLeftNorms, mask1, 1, { 1,1,8,8 });
            Copy(broadcastedScatteredNorms[rowMask], broadcastedRightNorms, mask2, 1, { 1,1,8,8 });
            Duplicate(newScatteredNorm, 0.f, SMNE);
            PipeBarrier<PIPE_V>();
            Add(newScatteredNorm, broadcastedScatteredNorms, newScatteredNorm, SMNE, 2 * SMNE, { 1,1,1, 0,1,0 });
            PipeBarrier<PIPE_V>();
            uint64_t updateMask[2] = { maskInfo.normUpdateMask, 0 };
            Sqrt(normVector, newScatteredNorm, updateMask, 1, { 1,1,0,0 });
        }
        PipeBarrier<PIPE_V>();
        ApplySinCosForSelfBucket(bucketSpace, bucketSize, sinTensor, cosTensor, maskInfo);
        PipeBarrier<PIPE_V>();

    }

    template<bool BUCKET_1_TARGET = true>
    __aicore__ inline void  ApplySinCos(const LocalTensor<float>& bucket1, const uint32_t bucket1Size, const LocalTensor<float>& bucket2,
        const uint32_t bucket2Size, const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor, const RotateMaskInfo& maskInfo)
    {
        const uint32_t bucketSize = (BUCKET_1_TARGET) ? bucket1Size : bucket2Size;
        LocalTensor<float> sinBroadcastedValues = tmpMaskSpace;
        LocalTensor<float> cosBroadcastedValues = tmpMaskSpace[rowMask * maxBucketSize];
        LocalTensor<float>& calcTmpTensor1 = tmpMemSpace;
        LocalTensor<float>  calcTmpTensor2 = tmpMemSpace[maxBucketSize * nSizeAligned];
        if (rowMask * sizeof(float) < 256) {
            Brcb(cosBroadcastedValues, cosTensor, 1, { 1,1 });
            PipeBarrier<PIPE_V>();
            Brcb(sinBroadcastedValues, sinTensor, 1, { 1,1 });
            PipeBarrier<PIPE_V>();
        } else {
            Brcb(calcTmpTensor1, cosTensor, 1, { 1,8 });
            Brcb(calcTmpTensor2, sinTensor, 1, { 1,8 });
            PipeBarrier<PIPE_V>();
            Brcb(cosBroadcastedValues, calcTmpTensor1, static_cast<uint8_t>(bucketSize), { 1,8 });
            Brcb(sinBroadcastedValues, calcTmpTensor2, static_cast<uint8_t>(bucketSize), { 1,8 });

        }
        PipeBarrier<PIPE_V>();
        uint32_t uOffsetForBacket1 = bucket1Size*nSizeAligned;
        uint32_t uOffsetForBacket2 = bucket2Size*nSizeAligned;
        if constexpr (BUCKET_1_TARGET) {
            uint8_t maskRepeatStride = rowMask * sizeof(float) / 32;
            {
                uint16_t rowIdxWithOffset = maskInfo.rightRowsIds[0];
                Mul(calcTmpTensor1, bucket2[rowIdxWithOffset * nSizeAligned], sinBroadcastedValues, rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
                Mul(calcTmpTensor2, bucket2[rowIdxWithOffset * nSizeAligned], cosBroadcastedValues, rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
            }
            for (uint32_t rowIdx = 1; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.rightRowsIds[rowIdx];
                Mul(calcTmpTensor1[rowIdx * nSizeAligned], bucket2[rowIdxWithOffset * nSizeAligned], sinBroadcastedValues[rowIdx * rowMask], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
                Mul(calcTmpTensor2[rowIdx * nSizeAligned], bucket2[rowIdxWithOffset * nSizeAligned], cosBroadcastedValues[rowIdx * rowMask], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
            }
            PipeBarrier<PIPE_V>();
            Muls(calcTmpTensor1, calcTmpTensor1, -1.f, bucketSize * nSizeAligned);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                MulAddDst(calcTmpTensor1[rowIdx * nSizeAligned], cosBroadcastedValues[rowIdx * rowMask], bucket1[rowIdx * nSizeAligned], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
                MulAddDst(calcTmpTensor2[rowIdx * nSizeAligned], sinBroadcastedValues[rowIdx * rowMask], bucket1[rowIdx * nSizeAligned], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.rightRowsIds[rowIdx];
                Copy(bucket1[rowIdx * nSizeAligned], calcTmpTensor1[rowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
                Copy(bucket2[rowIdxWithOffset * nSizeAligned], calcTmpTensor2[rowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
            }
            PipeBarrier<PIPE_V>();
            
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.rightRowsIds[rowIdx];
                Mul(calcTmpTensor1[rowIdx * uNSizeAligned], sinBroadcastedValues[rowIdx * rowMask], bucket2[rowIdxWithOffset * uNSizeAligned + uOffsetForBacket2], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
                Mul(calcTmpTensor2[rowIdx * uNSizeAligned], cosBroadcastedValues[rowIdx * rowMask], bucket2[rowIdxWithOffset * uNSizeAligned + uOffsetForBacket2], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
            }
            PipeBarrier<PIPE_V>();
            Muls(calcTmpTensor1, calcTmpTensor1, -1.f, bucketSize * uNSizeAligned);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                MulAddDst(calcTmpTensor1[rowIdx * uNSizeAligned], cosBroadcastedValues[rowIdx * rowMask], bucket1[rowIdx * uNSizeAligned + uOffsetForBacket1], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
                MulAddDst(calcTmpTensor2[rowIdx * uNSizeAligned], sinBroadcastedValues[rowIdx * rowMask], bucket1[rowIdx * uNSizeAligned + uOffsetForBacket1], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.rightRowsIds[rowIdx];
                Copy(bucket1[rowIdx * uNSizeAligned + uOffsetForBacket1], calcTmpTensor1[rowIdx * uNSizeAligned], rowMask, uRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
                Copy(bucket2[rowIdxWithOffset * uNSizeAligned + uOffsetForBacket2], calcTmpTensor2[rowIdx * uNSizeAligned], rowMask, uRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
            }

            // TODO: Add for APPLYING SIN AND COS FOR U TENSOR

        } else {
            uint8_t maskRepeatStride = rowMask * sizeof(float) / 32;
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                Mul(calcTmpTensor1[rowIdx * nSizeAligned], bucket2[rowIdx * nSizeAligned], sinBroadcastedValues[rowIdx * rowMask], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
                Mul(calcTmpTensor2[rowIdx * nSizeAligned], bucket2[rowIdx * nSizeAligned], cosBroadcastedValues[rowIdx * rowMask], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
            }
            PipeBarrier<PIPE_V>();
            Muls(calcTmpTensor1, calcTmpTensor1, -1.f, bucketSize * nSizeAligned);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.leftRowsIds[rowIdx];
                MulAddDst(calcTmpTensor1[rowIdx * nSizeAligned], cosBroadcastedValues[rowIdx * rowMask], bucket1[rowIdxWithOffset * nSizeAligned], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
                MulAddDst(calcTmpTensor2[rowIdx * nSizeAligned], sinBroadcastedValues[rowIdx * rowMask], bucket1[rowIdxWithOffset * nSizeAligned], rowMask, rowRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.leftRowsIds[rowIdx];
                Copy(bucket1[rowIdxWithOffset * nSizeAligned], calcTmpTensor1[rowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
                Copy(bucket2[rowIdx * nSizeAligned], calcTmpTensor2[rowIdx * nSizeAligned], rowMask, rowRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                Mul(calcTmpTensor1[rowIdx * uNSizeAligned], bucket2[rowIdx * uNSizeAligned + uOffsetForBacket2], sinBroadcastedValues[rowIdx * rowMask], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
                Mul(calcTmpTensor2[rowIdx * uNSizeAligned], bucket2[rowIdx * uNSizeAligned + uOffsetForBacket2], cosBroadcastedValues[rowIdx * rowMask], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
            }
            PipeBarrier<PIPE_V>();
            Muls(calcTmpTensor1, calcTmpTensor1, -1.f, maxBucketSize * nSizeAligned);
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.leftRowsIds[rowIdx];
                MulAddDst(calcTmpTensor1[rowIdx * uNSizeAligned], cosBroadcastedValues[rowIdx * rowMask], bucket1[rowIdxWithOffset * uNSizeAligned + uOffsetForBacket1], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
                MulAddDst(calcTmpTensor2[rowIdx * uNSizeAligned], sinBroadcastedValues[rowIdx * rowMask], bucket1[rowIdxWithOffset * uNSizeAligned + uOffsetForBacket1], rowMask, uRepeatsNum,
                    { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
                uint16_t rowIdxWithOffset = maskInfo.leftRowsIds[rowIdx];
                Copy(bucket1[rowIdxWithOffset * uNSizeAligned + uOffsetForBacket1], calcTmpTensor1[rowIdx * uNSizeAligned], rowMask, uRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
                Copy(bucket2[rowIdx * uNSizeAligned + uOffsetForBacket2], calcTmpTensor2[rowIdx * uNSizeAligned], rowMask, uRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
            }
        }
        PipeBarrier<PIPE_V>();

    }

    template<bool BUCKET_1_TARGET = true>
    __aicore__ inline void  ApplySinCosForSelfBucket(const LocalTensor<float>& bucket, const uint32_t bucketSize, const LocalTensor<float>& sinTensor,
        const LocalTensor<float>& cosTensor, const RotateMaskInfo& maskInfo)
    {
        LocalTensor<float> sinBroadcastedValues = tmpMaskSpace;
        LocalTensor<float> cosBroadcastedValues = tmpMaskSpace[rowMask * maxBucketSize];
        if (rowMask * sizeof(float) < 256) {
            Brcb(cosBroadcastedValues, cosTensor, 1, { 1,1 });
            PipeBarrier<PIPE_V>();
            Brcb(sinBroadcastedValues, sinTensor, 1, { 1,1 });
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<float>& tmpMaskTensor1 = tmpMemSpace;
            LocalTensor<float>  tmpMaskTensor2 = tmpMemSpace[rowMask * bucketSize];
            Brcb(tmpMaskTensor1, cosTensor, 1, { 1, 8});
            Brcb(tmpMaskTensor2, sinTensor, 1, { 1, 8});
            PipeBarrier<PIPE_V>();
            Brcb(cosBroadcastedValues, tmpMaskTensor1, static_cast<uint8_t>(maskInfo.numPairs), { 1,8 });
            Brcb(sinBroadcastedValues, tmpMaskTensor2, static_cast<uint8_t>(maskInfo.numPairs), { 1,8 });
        }

        PipeBarrier<PIPE_V>();

        LocalTensor<float>& calcTmpTensor1 = tmpMemSpace;
        LocalTensor<float>  calcTmpTensor2 = tmpMemSpace[maxBucketSize * nSizeAligned];
        uint8_t maskRepeatStride = rowMask * sizeof(float) / 32;
        for (uint32_t pairID = 0; pairID < maskInfo.numPairs; ++pairID) {
            uint8_t secondRowIdx = maskInfo.rightRowsIds[pairID];
            Mul(calcTmpTensor1[pairID * nSizeAligned], sinBroadcastedValues[pairID * rowMask], bucket[secondRowIdx * nSizeAligned], rowMask, rowRepeatsNum,
                { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
            Mul(calcTmpTensor2[pairID * nSizeAligned], cosBroadcastedValues[pairID * rowMask], bucket[secondRowIdx * nSizeAligned], rowMask, rowRepeatsNum,
                { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
        }
        PipeBarrier<PIPE_V>();
        Muls(calcTmpTensor1, calcTmpTensor1, -1.f, bucketSize * nSizeAligned);
        PipeBarrier<PIPE_V>();
        for (uint32_t pairID = 0; pairID < maskInfo.numPairs; ++pairID) {
            uint8_t firstRowIdx = maskInfo.leftRowsIds[pairID];
            MulAddDst(calcTmpTensor1[pairID * nSizeAligned], bucket[firstRowIdx * nSizeAligned], cosBroadcastedValues[pairID * rowMask], rowMask, rowRepeatsNum,
                { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
            MulAddDst(calcTmpTensor2[pairID * nSizeAligned], bucket[firstRowIdx * nSizeAligned], sinBroadcastedValues[pairID * rowMask], rowMask, rowRepeatsNum,
                { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t pairID = 0; pairID < maskInfo.numPairs; ++pairID) {
            uint8_t firstRowIdx = maskInfo.leftRowsIds[pairID];
            uint8_t secondRowIdx = maskInfo.rightRowsIds[pairID];
            Copy(bucket[firstRowIdx * nSizeAligned], calcTmpTensor1[pairID * nSizeAligned], rowMask, rowRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
            Copy(bucket[secondRowIdx * nSizeAligned], calcTmpTensor2[pairID * nSizeAligned], rowMask, rowRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
        }

        // TODO: Add for APPLYING SIN AND COS FOR U TENSOR
        //Applaing U matrice
        PipeBarrier<PIPE_V>();
        uint32_t uOffset = bucketSize*nSizeAligned;
        for (uint32_t pairID = 0; pairID < maskInfo.numPairs; ++pairID) {
            uint8_t secondRowIdx = maskInfo.rightRowsIds[pairID];
            Mul(calcTmpTensor1[pairID * uNSizeAligned], sinBroadcastedValues[pairID * rowMask], bucket[secondRowIdx * uNSizeAligned + uOffset], rowMask, uRepeatsNum,
                { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
            Mul(calcTmpTensor2[pairID * uNSizeAligned], cosBroadcastedValues[pairID * rowMask], bucket[secondRowIdx * uNSizeAligned + uOffset], rowMask, uRepeatsNum,
                { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
        }
        PipeBarrier<PIPE_V>();
        Muls(calcTmpTensor1, calcTmpTensor1, -1.f, bucketSize * uNSizeAligned);
        PipeBarrier<PIPE_V>();
        for (uint32_t pairID = 0; pairID < maskInfo.numPairs; ++pairID) {
            uint8_t firstRowIdx = maskInfo.leftRowsIds[pairID];
            MulAddDst(calcTmpTensor1[pairID * uNSizeAligned], bucket[firstRowIdx * uNSizeAligned + uOffset], cosBroadcastedValues[pairID * rowMask], rowMask, uRepeatsNum,
                { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
            MulAddDst(calcTmpTensor2[pairID * uNSizeAligned], bucket[firstRowIdx * uNSizeAligned + uOffset], sinBroadcastedValues[pairID * rowMask], rowMask, uRepeatsNum,
                { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t pairID = 0; pairID < maskInfo.numPairs; ++pairID) {
            uint8_t firstRowIdx = maskInfo.leftRowsIds[pairID];
            uint8_t secondRowIdx = maskInfo.rightRowsIds[pairID];
            Copy(bucket[firstRowIdx * uNSizeAligned + uOffset], calcTmpTensor1[pairID * uNSizeAligned], rowMask, uRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
            Copy(bucket[secondRowIdx * uNSizeAligned + uOffset], calcTmpTensor2[pairID * uNSizeAligned], rowMask, uRepeatsNum, { 1,1, maskRepeatStride, maskRepeatStride });
        }

    }
    __aicore__ inline void InitV(const LocalTensor<float>& bucketSpace, const uint32_t bucketBegin, const uint32_t bucketEnd)
    {
        uint16_t numRows = bucketEnd - bucketBegin;
        Duplicate(bucketSpace[nSizeAligned*numRows], 0.f, uNSizeAligned*numRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t rowId = bucketBegin; rowId < bucketEnd; ++rowId) {
            uint32_t localBlockOffset = (rowId / rowMask) * rowMask;
            uint32_t onePosition = rowId % rowMask;
            uint64_t localMask = 1;
            localMask <<= onePosition;
            uint64_t mask[2] = { localMask, 0 };
            uint32_t rowOffset = nSizeAligned*numRows+ (rowId - bucketBegin) * uNSizeAligned;
            Duplicate(bucketSpace[rowOffset + localBlockOffset], 1.f, mask, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadBucket(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketBegin, const uint32_t bucketEnd, const bool isFirstLoadDataLocal)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        uint16_t numRows = bucketEnd - bucketBegin;

        if (!isFirstLoadDataLocal) {
            DataCopy(bucketSpace, aTmpGm[batchOffsetTmp + bucketBegin * recordSizeAligned], { numRows, static_cast<uint16_t>(recordSizeAligned * sizeof(AType) / 32), 0, 0 });
            DataCopyPad(normVector, sGm[batchOffsetS + bucketBegin], { 1, static_cast<uint16_t>(numRows * sizeof(AType)), 0, 0 }, { true, 0, static_cast<uint8_t>(SMNE - numRows),0 });
        } else {
            uint32_t nSizeAligned32 = ((((nSize * sizeof(AType)) + 31) / 32) * 32) / sizeof(AType);
            DataCopyPad(bucketSpace, aGm[batchOffsetI + bucketBegin * nSize], { numRows, static_cast<uint16_t>(nSize * sizeof(AType)), 0, static_cast<uint16_t>((nSizeAligned32 - nSizeAligned32) * sizeof(AType) / 32) },
                { true, 0, static_cast<uint8_t>(nSizeAligned32 - nSize), 0 });
        }
    }

    __aicore__ inline void SaveBucket(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketBegin, const uint32_t bucketEnd)
    {
        uint16_t numRows = bucketEnd - bucketBegin;
        DataCopy(aTmpGm[batchOffsetTmp + bucketBegin * recordSizeAligned], bucketSpace, { numRows, static_cast<uint16_t>(recordSizeAligned * sizeof(AType) / 32), 0, 0 });
        DataCopyPad(sGm[batchOffsetS + bucketBegin], normVector, { 1, static_cast<uint16_t>(numRows * sizeof(AType)), 0, 0 });
    }

    template<bool IS_LEFT_BUCKET_SAVING>
    __aicore__ inline void SaveOutputs(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketBegin, const uint32_t bucketEnd)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        uint16_t numRows = bucketEnd - bucketBegin;
        uint32_t uNSizeAligned32 = (((uNSize * sizeof(float) + 31) / 32) * 32) / sizeof(float);
        uint32_t vNSizeAligned32 = (((vNSize * sizeof(float) + 31) / 32) * 32) / sizeof(float);
        const uint8_t maskRepeatStride = rowMask * sizeof(float) / 32;
        LocalTensor<float> singularValues;
        LocalTensor<float> tmpCurrMemSpace, tmpCurrMaskSpace;
        if constexpr(IS_LEFT_BUCKET_SAVING)
        {
            singularValues = tmpBlock4;
            tmpCurrMemSpace = tmpMemSpace;
            tmpCurrMaskSpace = tmpMaskSpace;
        }
        else
        {
            singularValues = tmpBlock5;
            tmpCurrMemSpace = tmpMemSpace[maxBucketSize*nSizeAligned];
            tmpCurrMaskSpace = tmpMaskSpace[maxBucketSize*rowMask];
        }
        Duplicate(tmpCurrMaskSpace, 0.f, numRows * rowMask);
        PipeBarrier<PIPE_V>();
        
        for (uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
            MulAddDst(tmpCurrMaskSpace[rowIdx*rowMask], bucketSpace[rowIdx*nSizeAligned], bucketSpace[rowIdx*nSizeAligned], rowMask, rowRepeatsNum, {1, 1, 1, 0, maskRepeatStride, maskRepeatStride});
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(singularValues, tmpCurrMaskSpace, rowMask, static_cast<uint8_t>(numRows), 1, 1, maskRepeatStride);
        PipeBarrier<PIPE_V>();
        Sqrt(singularValues, singularValues, SMNE);
        PipeBarrier<PIPE_V>();
        Brcb(tmpCurrMemSpace, singularValues, 1, { 1, 8 });
        PipeBarrier<PIPE_V>();
        Brcb(tmpCurrMaskSpace, tmpCurrMemSpace, static_cast<uint8_t>(numRows), { 1, 8 });
        PipeBarrier<PIPE_V>();

        for (uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
            Div(bucketSpace[rowIdx * nSizeAligned], bucketSpace[rowIdx * nSizeAligned], tmpCurrMaskSpace[rowIdx * rowMask],
                rowMask, rowRepeatsNum, { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
        }

        PipeBarrier<PIPE_V>();
        event_t evtVToMte3_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evtVToMte3_1);
        WaitFlag<HardEvent::V_MTE3>(evtVToMte3_1);
        DataCopyPad(uGm[batchOffsetU + bucketBegin * uNSize], bucketSpace[numRows*nSizeAligned],
                {static_cast<uint16_t>(numRows), static_cast<uint16_t>(uNSize * sizeof(AType)), 
                 static_cast<uint16_t>((uNSizeAligned - uNSizeAligned32) * sizeof(AType) / 32), 0 });
        DataCopyPad(sGm[batchOffsetS + bucketBegin], singularValues, { 1, static_cast<uint16_t>(numRows * sizeof(AType)), 0, 0 });
        DataCopyPad(vGm[batchOffsetV + bucketBegin * vNSize], bucketSpace,
                    { static_cast<uint16_t>(numRows), static_cast<uint16_t>(vNSize * sizeof(AType)), static_cast<uint16_t>((nSizeAligned - vNSizeAligned32) * sizeof(AType) / 32),0 });
    }

    __aicore__ inline void CalculateSinCos(const LocalTensor<float>& leftSquareNorms, const LocalTensor<float>& rightSquareNorms,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& sinTensor,
        const LocalTensor<float>& cosTensor)
    {
        constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
        //Generating coeffs for rotating
        // tmpBlock1-8 used for tmp calculatins

        //tauTensor used for calculation tTensor
        LocalTensor<float>& tauTensor = tmpBlock1;
        //normSquareDiffIsPositive ensures  that sin and cos are selected so that the singular
        // values are sorted in descending order. this trick need to avoid  to sort outputs.
        LocalTensor<uint8_t> normSquareDiffIsPositive = tmpBlock2.ReinterpretCast<uint8_t>();
        // The vaules of  isFakeRotateCase indicate whether the vactors are orthogonal or not.
        // If vectors are orthogonal a fake rotation is performed
        LocalTensor<uint8_t> isFakeRotateCase = tmpBlock3.ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> isNotFakeRotateCase =  tmpBlock3.ReinterpretCast<uint8_t>();
        //Free tmpBlocks are {4,5,6,7,8}
        // {
            LocalTensor<float>& normSquareDiff = tmpBlock4;
            LocalTensor<float>& constValues = tmpBlock5;
            LocalTensor<float>& absDotProducts = tmpBlock6;
           
            Duplicate(constValues, 0.f, SMNE);

            Sub(normSquareDiff, rightSquareNorms, leftSquareNorms, SMNE);
            PipeBarrier<PIPE_V>();
            // Compare(normSquareDiffIsPositive, normSquareDiff, constValues, CMPMODE::GE, SMNE);
            VcmpvImpl((__ubuf__ uint8_t*)normSquareDiffIsPositive.GetPhyAddr(), (__ubuf__ float*)normSquareDiff.GetPhyAddr(), (__ubuf__ float*)constValues.GetPhyAddr(),
                CMPMODE::GE, SMNE, 1, { 1,1,1, 1,1,1 });
            Div(tauTensor, normSquareDiff, dotProducts, SMNE);
            PipeBarrier<PIPE_V>();
            Muls(tauTensor, tauTensor, 0.5f, SMNE);
            Abs(normSquareDiff, normSquareDiff, SMNE);
            Abs(absDotProducts, dotProducts, SMNE);
            Duplicate(constValues, NEAR_ZERO_VALUE, SMNE);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint8_t>& normSquareDiffIsZero = isFakeRotateCase;
            LocalTensor<uint8_t> dotProductIsZero = tmpBlock7.ReinterpretCast<uint8_t>();
            VcmpvImpl((__ubuf__ uint8_t*)normSquareDiffIsZero.GetPhyAddr(), (__ubuf__ float*)normSquareDiff.GetPhyAddr(), (__ubuf__ float*)constValues.GetPhyAddr(),
                CMPMODE::LT, SMNE, 1, { 1,1,1, 1,1,1 });
            // PipeBarrier<PIPE_V>();
            VcmpvImpl((__ubuf__ uint8_t*)dotProductIsZero.GetPhyAddr(), (__ubuf__ float*)absDotProducts.GetPhyAddr(), (__ubuf__ float*)constValues.GetPhyAddr(),
                CMPMODE::LT, SMNE, 1, { 1,1,1, 1,1,1 });
            PipeBarrier<PIPE_V>();
            Or(isFakeRotateCase, normSquareDiffIsZero, dotProductIsZero, SMNE);
            PipeBarrier<PIPE_V>();
        // }
        LocalTensor<float>& tTensor = tmpBlock4;
        //Free tmpBlocks are {5,6,7,8}
        //Calculating t = sign(tau)/(absTau+sqrt(1.f+tau*tau));
        // {
            LocalTensor<float>& absTaoTensor = tmpBlock5;
            LocalTensor<float>& tmp1 = tmpBlock6;
            LocalTensor<float>& tmp2 = tmpBlock7;
            //absTau = abs(tau)
            NotImpl((__ubuf__ uint16_t*)isNotFakeRotateCase.GetPhyAddr(), (__ubuf__ uint16_t*)isFakeRotateCase.GetPhyAddr(), SMNE, 1, {1,1,1,1});
            Abs(absTaoTensor, tauTensor, SMNE);
            // tmp1=tau*tau
            Mul(tmp1, tauTensor, tauTensor, SMNE);
            PipeBarrier<PIPE_V>();
            //tmp1 = 1.f+tau*tau
            Adds(tmp1, tmp1, 1.f, SMNE);
            //tmp2 = sign(tau)=tau/absTau
            Div(tmp2, tauTensor, absTaoTensor, SMNE);
            PipeBarrier<PIPE_V>();
            //tmp1 = sqrt(1.f+tau*tau)
            Sqrt(tmp1, tmp1, SMNE);
            PipeBarrier<PIPE_V>();
            //tmp1 = absTau+sqrt(1.f+tau*tau)
            Add(tmp1, tmp1, absTaoTensor, SMNE);
            PipeBarrier<PIPE_V>();
            // t = sign(tau)/(absTau+sqrt(1.f+tau*tau))
            Div(tTensor, tmp2, tmp1, SMNE);
            PipeBarrier<PIPE_V>();
        // }
        //taoTensor is free and can be reused
        LocalTensor<float>& s1Tensor = tmpBlock1;
        LocalTensor<float>& s2Tensor = tmpBlock5;
        //Free tmpBlocks are {6,7,8}
        // {

            LocalTensor<float>& negativeOneTensor = tmpBlock6;
            Duplicate(negativeOneTensor, -1.f, SMNE);
            //s1 = 1 / sqrt(1.f + t*t)
            //s1 = t*t
            Mul(s1Tensor, tTensor, tTensor, SMNE);
            Duplicate(s2Tensor, 1.f, SMNE);
            PipeBarrier<PIPE_V>();
            //s1 = 1.f + t*t
            Add(s1Tensor, s2Tensor, s1Tensor, SMNE);
            PipeBarrier<PIPE_V>();
            //s1 = sqrt(1.f + t*t)
            Sqrt(s1Tensor, s1Tensor, SMNE);
            PipeBarrier<PIPE_V>();
            //s1 = 1 / sqrt(1.f + t*t)
            Div(s1Tensor, s2Tensor, s1Tensor, SMNE);
            PipeBarrier<PIPE_V>();
            //s2 = t*s1
            Mul(s2Tensor, s1Tensor, tTensor, SMNE);
            //tTensor is free
            //  if(normSquareDiff>=0.f) s2=-s2;
            LocalTensor<float>& coeff = tmpBlock4;
            Select(coeff, normSquareDiffIsPositive, negativeOneTensor, 1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SMNE);
            PipeBarrier<PIPE_V>();
            Mul(s2Tensor, s2Tensor, coeff, SMNE);
            PipeBarrier<PIPE_V>();

        // }
        // Calculate sin, cos
        // Free tmpBlocks are {8}
        // {
            Select(sinTensor, normSquareDiffIsPositive, s1Tensor, s2Tensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
            // PipeBarrier<PIPE_V>();
            Select(cosTensor, normSquareDiffIsPositive, s2Tensor, s1Tensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
            PipeBarrier<PIPE_V>();
            // Free s1Tensor(tmpBlock1), s2Tensor(tmpBlock5), normSquareDiffIsPositive(tmpBlock2),
            Select(cosTensor, isNotFakeRotateCase, cosTensor, 1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SMNE);
            // PipeBarrier<PIPE_V>();
            Select(sinTensor, isNotFakeRotateCase, sinTensor, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SMNE);
            PipeBarrier<PIPE_V>();
        // }

    }
    
private:
    //constexpr functions
    template<typename DST_TYPE>
    constexpr __aicore__ DST_TYPE VEC_INSTRUCTION_NUM()
    {
        return static_cast<DST_TYPE>(VEC_INSTRUCTION_SET_SIZE / sizeof(float));
    }
private:
    static constexpr float NEAR_ZERO_VALUE = 0.0000000001f;
private:
    struct RotateMaskInfo
    {
        uint8_t leftRowsIds[sizeof(uint64_t)];
        uint8_t rightRowsIds[sizeof(uint64_t)];
        uint64_t normUpdateMask = 0;
        union
        {
            uint64_t leftMaskU64Val;
            uint8_t leftMaskU8Val[sizeof(uint64_t)];
        };
        union
        {
            uint64_t rightMaskU64Val;
            uint8_t rightMaskU8Val[sizeof(uint64_t)];
        };
        uint8_t numPairs = 0;
        uint8_t leftBucketSize = 0;
        uint8_t rightBucketSize = 0;
    };
    static constexpr uint64_t INTERNAL_SYNC_ALL = 0x1;
    TPipe* pipe_;
    GlobalTensor<AType> aGm;
    GlobalTensor<AType> aTmpGm;
    GlobalTensor<AType> sGm;
    GlobalTensor<AType> uGm;
    GlobalTensor<AType> vGm;
    uint32_t coreIdx, numAvailableVectorCores;
    uint32_t globSetMaxSize, globSetMinSize, numGlobalSetsWithMaxSize, numGlobalSetsWithMinSize, globSetMinOffset;
    uint32_t numIterations, numStages, numPhases, numGlobalSets, leftGlobalSetBegin, leftGlobalSetEnd, rightGlobalSetBegin, rightGlobalSetEnd;
    uint32_t mSize, nSize, nSizeAligned, uMSize, uNSize, uNSizeAligned, vMSize, vNSize, vNSizeAligned, sMNSize, sMNSizeAligned, recordSize, recordSizeAligned;
    int32_t batchSize;
    uint64_t batchOffsetV,batchOffsetU, batchOffsetS, batchOffsetI, batchOffsetTmp;
    uint32_t rowSize, rowSizeAligned, rowMask, rowRepeatsNum, rowTail, maxBucketSize, ubSize, uRepeatsNum, uTailSize, tmpRowSize, tmpRowSizeAligned;
    TBuf<TPosition::VECCALC> ubBuf;
    LocalTensor<uint8_t> ubMemory;
    LocalTensor<int32_t> arithmProgressive;
    LocalTensor<float>  tmpMemSpace, tmpMaskSpace, bucketSpace1, bucketSpace2, bucketSpace3;
    //Constant tensors
    LocalTensor<float> tmpBlock1, tmpBlock2, tmpBlock3, tmpBlock4, tmpBlock5,
        tmpBlock6, tmpBlock7, tmpBlock8, tmpBlock9, tmpBlock10, tmpBlock11, tmpBlock12, tmpBlock13,
        tmpBlock14, tmpBlock15, tmpBlock16, tmpBlock17, tmpBlock18, tmpBlock19, tmpBlock20, tmpBlock21,
        tmpBlock22, tmpBlock23, tmpBlock24, tmpBlock25, tmpBlock26, tmpBlock27, tmpBlock28, tmpBlock29,
        tmpBlock30, tmpBlock31, tmpBlock32, tmpBlockSpace;
    bool needSort;
};
}
#endif //JACOBI_BASE_H
