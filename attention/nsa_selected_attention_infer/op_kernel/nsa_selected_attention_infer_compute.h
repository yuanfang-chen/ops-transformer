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
 * \file nsa_selected_attention_infer_impl.h
 * \brief
 */

#pragma once

#include "nsa_selected_attention_infer_compute.h"

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::SoftmaxFlashV2Compute(uint32_t loop,
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t outIdx = loop % (PRE_LOAD_NUM);
    if (extraInfo[loop % PRE_LOAD_NUM].isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (loop -1) % (PRE_LOAD_NUM);
        inMaxTensor = softmaxMaxUb[inIdx][baseOffset];
        inSumTensor = softmaxSumUb[inIdx][baseOffset];
    }

    AscendC::SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb,  // dstTensor
        softmaxSumUb[outIdx][baseOffset],  // expSumTensor 输出tensor
        softmaxMaxUb[outIdx][baseOffset],   // maxTensor rowmax 
        mmResUb,  // srcTensor
        softmaxExpUb[outIdx][baseOffset],  // expMaxTensor maxi
        inSumTensor,  // inExpSumTensor
        inMaxTensor, 
        softmaxTmpUb,  // 临时空间
        newTiling, 
        srcShape);
    PipeBarrier<PIPE_V>();
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::ElewiseCompute(uint32_t loop, LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                              uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::CalcParams(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    info.loop = loop;
    if (loop != 0) {
        {
            if (curS2Idx + 1 >= curSInnerLoopTimes) {
                curS2Idx = 0;
                curS1Idx = (curS1Idx  + 1) % actualQSeqLengthsGm.GetValue(curBIdx);
                curN2Idx = (curN2Idx  + (curS1Idx == 0 ? 1 : 0)) % kvHeadNum;
                curBIdx = curBIdx + (curN2Idx == 0 && curS1Idx == 0 ? 1 : 0);
                curActSeqLenIsZero = true;
                while (curActSeqLenIsZero) {
                    bIdx = curBIdx;
                    n2Idx = curN2Idx;
                    s1Idx = curS1Idx;
                    GetActualSeqLen();
                    // 根据当前块实际长度, 重配flashattention循环条件
                    UpdateInnerLoopCond();
                    if (curActSeqLenIsZero) {
                        curBIdx++;
                        curN2Idx = 0;
                        continue;
                    }
                }
                curS2Idx = s2Idx;
                curSInnerLoopTimes = sInnerLoopTimes;
            } else {
                curS2Idx++;
            }
        }
    } else {
        {
            // b * S
            curActSeqLenIsZero = true;
            while (curActSeqLenIsZero) {
                bIdx = curBIdx;
                n2Idx = curN2Idx;
                s1Idx = curS1Idx;
                GetActualSeqLen();
                // 根据当前块实际长度, 重配flashattention循环条件
                UpdateInnerLoopCond();
                if (curActSeqLenIsZero) {
                    curBIdx++;
                    curN2Idx = 0;
                    continue;
                }
            }
            curS2Idx = s2Idx;
            curSInnerLoopTimes = sInnerLoopTimes;
        }
    }
    info.bIdx = curBIdx;
    info.n2Idx = curN2Idx;
    info.s2Idx = curS2Idx;
    info.s1Idx = curS1Idx;
    info.curSInnerLoopTimes = curSInnerLoopTimes;

    info.isFirstSInnerLoop = (info.s2Idx == 0);
    info.bn2IdxInPreCore = info.bn2IdxInCurCore;
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint32_t curTotalQSeqLenOffset = 0;
            for (uint64_t i = 0; i < info.bIdx; ++i) {
                curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
            }
            tensorACoreOffset = curTotalQSeqLenOffset * qHeadNum * headDim + info.s1Idx * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            tensorAttenOutCoreOffset = curTotalQSeqLenOffset * qHeadNum * headDimV + info.s1Idx * qHeadNum * headDimV + info.n2Idx * gSize * headDimV; //bsnd
        } else {
            // B,S1,N2,G,D
            tensorACoreOffset = info.bIdx * qSeqSize * qHeadNum * headDim + info.s1Idx * qHeadNum * headDim + info.n2Idx * gSize * headDim; //bsnd
            tensorAttenOutCoreOffset = info.bIdx * qSeqSize * qHeadNum * headDimV + info.s1Idx * qHeadNum * headDimV + info.n2Idx * gSize * headDimV; //bsnd
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    info.attenOutOffset = tensorAttenOutCoreOffset;
    info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        info.actualSingleProcessSInnerSize = singleProcessSInnerSizeTail;
    }
    info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSingleProcessSInnerSize, HALF_BLOCK_NUM);

    info.needSetOrgShape = (curSingleProcessSInnerSizeAlign != info.actualSingleProcessSInnerSizeAlign);
    if (info.needSetOrgShape) {
        curSingleProcessSInnerSizeAlign = info.actualSingleProcessSInnerSizeAlign;
    }

    uint64_t sInnerOffsetDataSize = info.s2Idx * singleProcessSInnerSize;

    info.s2BatchOffset = sInnerOffsetDataSize;
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ComputeMm1(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    gSize = gSizeCube;
    LocalTensor<KV_T> aL1Tensor = qL1Tensor;
    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
    CopyInMm1AToL1(aL1Tensor, info);
    SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT1);
    WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT1);
    if (info.bn2IdxInCurCore != info.bn2IdxInPreCore && info.isFirstSInnerLoop) {
        x = 0;
        currentTopDeal = -1;
        topKIdOffset = 0;
        idIntopKRecord = -1;
    }
    constexpr int64_t nCopyRowCount = 128; // n方向切分
    int64_t nCopyTimes = (info.actualSingleProcessSInnerSize + nCopyRowCount - 1) / nCopyRowCount;
    int64_t nTailCopyRowCount = info.actualSingleProcessSInnerSize - (nCopyTimes - 1) * nCopyRowCount;
    int64_t nTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (nCopyTimes - 1) * nCopyRowCount;
    for (int64_t nCopyIdx = 0, nActCopyRowCount = nCopyRowCount, nActCopyRowCountAlign = nCopyRowCount; nCopyIdx < nCopyTimes; nCopyIdx++) {
        if (nCopyIdx + 1 == nCopyTimes) {
            nActCopyRowCount = nTailCopyRowCount;
            nActCopyRowCountAlign = nTailCopyRowCountAlign;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + (kpL1BufIter % 2));
        LocalTensor<KV_T> bL1Tensor = kpL1Tensor[(kpL1BufIter % 2) * (L1_KP_SIZE / sizeof(KV_T))];
        int64_t topKBaseOffsetmm1 = 0;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint32_t curTotalQSeqLenOffset = 0;
            for (uint64_t i = 0; i < info.bIdx; ++i) {
                curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
            }
            topKBaseOffsetmm1 = curTotalQSeqLenOffset * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        } else {
            topKBaseOffsetmm1 = info.bIdx * qSeqSize * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        }
        int64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
        int64_t curSeqIdx = info.s2BatchOffset + nCopyIdx * nCopyRowCount;
        int64_t copyStartRowCnt = 0;
        while (copyStartRowCnt < nActCopyRowCount) {
            if (x == currentTopDeal) {
                topKIdOffset += 1;
            }
            int64_t idIntopK = topKGm.GetValue(topKBaseOffsetmm1 + topKIdOffset);
            if (idIntopK != -1) {
                if(idIntopKRecord != idIntopK) {
                    x = 0;
                    idIntopKRecord = idIntopK;
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenOri ) ?
                                        curActualSeqLenOri : (globalStart + selectedBlockSize);
                    currentTopDeal = globalEnd - globalStart;
                }
                if (currentTopDeal > 0) {
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenOri ) ?
                                        curActualSeqLenOri : (globalStart + selectedBlockSize);
                    globalStart += x;
                    int64_t start_offset = globalStart % kvCacheBlockSize;
                    int64_t start_block_idx = globalStart / kvCacheBlockSize;
                    int64_t end_block_idx = (globalEnd - 1) / kvCacheBlockSize;
                    int64_t reaminRowCnt = start_offset;
                    int64_t copyRowCnt = start_block_idx == end_block_idx ?
                                    globalEnd - globalStart : kvCacheBlockSize - reaminRowCnt;
                    if (copyStartRowCnt + copyRowCnt > nActCopyRowCount) {
                        copyRowCnt = nActCopyRowCount - copyStartRowCnt;
                    }
                    int64_t idInBlockTable =
                        blockTableGm.GetValue(blockTableBaseOffset + start_block_idx);
                    int64_t keyOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                    keyOffset += (int64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                    CopyInMm1BToL1ForPA(bL1Tensor, keyOffset, nActCopyRowCountAlign, copyStartRowCnt, copyRowCnt);
                    // 更新循环变量
                    copyStartRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                    x += copyRowCnt;
                } else {
                    x = currentTopDeal;
                }
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + (kpL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + (kpL1BufIter % 2));

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
        uint32_t kSplitSize = 128;
        uint32_t i = 0;
        uint32_t kSize = headDim;
        uint32_t kLoops = (kSize + kSplitSize - 1) / kSplitSize;
        uint32_t subKSize = kSplitSize;
        for (uint32_t k = 0; k < kLoops; k++) {
            if (k + 1 == kLoops) {
                subKSize = kSize - (kLoops - 1) * kSplitSize;
            }
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
            LoadDataMm1A(aL0Tensor, aL1Tensor, k, subKSize);
            SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
            LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
            LoadDataMm1B(bL0Tensor, bL1Tensor, k, subKSize, i, nActCopyRowCountAlign);
            SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
            MmadParams mmadParams;
            mmadParams.m = msdIterNum * gSize;
            if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                mmadParams.m = 16;
            }
            mmadParams.n = nActCopyRowCountAlign;
            mmadParams.k = subKSize;
            mmadParams.cmatrixInitVal = (k == 0); // true：C矩阵初始值为0；
            mmadParams.cmatrixSource = false;
            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
            aL0BufIter++;
            bL0BufIter++;
        }
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + (kpL1BufIter % 2));
        kpL1BufIter++;
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
        FixpipeParamsV220 fixParams;
        fixParams.nSize = nActCopyRowCountAlign;
        fixParams.mSize = msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
        fixParams.srcStride = ((fixParams.mSize + 15) / 16) * 16;
        fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
        fixParams.ndNum = 1;
        Fixpipe(mm1ResGm[(loop % (PRE_LOAD_NUM)) * mmResUbSize + nCopyIdx * nCopyRowCount], cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        cL0BufIter++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ComputeMm2(uint32_t loop) {
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    gSize = gSizeCube;
    constexpr uint32_t kCopyRowCount = 256;
    uint32_t kCopyTimes = (info.actualSingleProcessSInnerSize + kCopyRowCount - 1) / kCopyRowCount;
    uint32_t kTailCopyRowCount = info.actualSingleProcessSInnerSize - (kCopyTimes - 1) * kCopyRowCount;
    uint32_t kTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (kCopyTimes - 1) * kCopyRowCount;
    LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
    if (info.bn2IdxInCurCore != info.bn2IdxInPreCore && info.isFirstSInnerLoop) {
        xMM2 = 0;
        currentTopDealMM2 = -1;
        topKIdOffsetMM2 = 0;
        idIntopKRecordMM2 = -1;
    }
    for (uint32_t kCopyIdx = 0, kActCopyRowCount = kCopyRowCount, kActCopyRowCountAlign = kCopyRowCount; kCopyIdx < kCopyTimes; kCopyIdx++) {
        if (kCopyIdx + 1 == kCopyTimes) {
            kActCopyRowCount = kTailCopyRowCount;
            kActCopyRowCountAlign = kTailCopyRowCountAlign;
        }
        LocalTensor<KV_T> aL1Tensor = kpL1Tensor[(kpL1BufIter % 2) * L1_KP_SIZE / sizeof(KV_T)];
        WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + (kpL1BufIter % 2));
        CopyInMm2AToL1(aL1Tensor, info, kCopyIdx, kCopyRowCount, kActCopyRowCountAlign, kActCopyRowCount);
        SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + kpL1BufIter % 2);
        LocalTensor<KV_T> bL1Tensor = vL1Tensor[(vL1BufIter % 2) * (L1_V_SIZE / sizeof(KV_T))];
        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + (vL1BufIter % 2));
        int64_t topKBaseOffsetmm2 = 0;
        if constexpr (LAYOUT_T == LAYOUT::TND) {
            uint32_t curTotalQSeqLenOffset = 0;
            for (uint64_t i = 0; i < info.bIdx; ++i) {
                curTotalQSeqLenOffset += actualQSeqLengthsGm.GetValue(i);
            }
            topKBaseOffsetmm2 = curTotalQSeqLenOffset * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        } else {
            topKBaseOffsetmm2 = info.bIdx * qSeqSize * kvHeadNum * selectedBlockCount + info.s1Idx * kvHeadNum * selectedBlockCount;
        }
        int64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
        int64_t curSeqIdx = info.s2BatchOffset + kCopyIdx * kCopyRowCount;
        int64_t copyStartRowCnt = 0;
        uint64_t curActualSeqLenCurBatch = actualSeqLengthsGm.GetValue(info.bIdx);
        uint32_t curBatchQseqlen = actualQSeqLengthsGm.GetValue(info.bIdx);
        if (info.s1Idx +1 != curBatchQseqlen) {
            curActualSeqLenCurBatch -= curBatchQseqlen - 1 - info.s1Idx;
        }
        while (copyStartRowCnt < kActCopyRowCount) {
            if (xMM2 == currentTopDealMM2) {
                topKIdOffsetMM2 += 1;
            }
            int64_t idIntopK = topKGm.GetValue(topKBaseOffsetmm2 + topKIdOffsetMM2);
            if (idIntopK != -1) {
                if(idIntopKRecordMM2 != idIntopK) {
                     xMM2 = 0;
                    idIntopKRecordMM2 = idIntopK;
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenCurBatch ) ?
                                         curActualSeqLenCurBatch : (globalStart + selectedBlockSize);
                    currentTopDealMM2 = globalEnd - globalStart;
                }
                if (currentTopDealMM2 > 0) {
                    int64_t globalStart = idIntopK * selectedBlockSize;
                    int64_t globalEnd = (globalStart + selectedBlockSize > curActualSeqLenCurBatch ) ?
                                            curActualSeqLenCurBatch : (globalStart + selectedBlockSize);
                    globalStart += xMM2;
                    int64_t start_offset = globalStart % kvCacheBlockSize;
                    int64_t start_block_idx = globalStart / kvCacheBlockSize;
                    int64_t end_block_idx = (globalEnd - 1) / kvCacheBlockSize;
                    int64_t idInBlockTable =
                        blockTableGm.GetValue(blockTableBaseOffset + start_block_idx);

                    int64_t reaminRowCnt = start_offset;
                    int64_t copyRowCnt = start_block_idx == end_block_idx ?
                                        globalEnd - globalStart : kvCacheBlockSize - reaminRowCnt;
                    if (copyStartRowCnt + copyRowCnt > kActCopyRowCount) {
                        copyRowCnt = kActCopyRowCount - copyStartRowCnt;
                    }
                    int64_t valueOffset = idInBlockTable * kvCacheBlockSize * headDimV * kvHeadNum;
                    valueOffset += (int64_t)(info.n2Idx * headDimV) + reaminRowCnt * headDimV * kvHeadNum;
                    CopyInMm2BToL1ForPA(bL1Tensor, valueOffset, kActCopyRowCount, copyStartRowCnt, copyRowCnt);
                    copyStartRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                    xMM2 += copyRowCnt;
                } else {
                    xMM2 = currentTopDealMM2;
                }
            }
        }

        SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + (vL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + (vL1BufIter % 2));

        WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + kpL1BufIter % 2);
        {
            constexpr uint32_t baseK = 256 / sizeof(KV_T);
            uint32_t kLoopTimes = (kActCopyRowCountAlign + baseK - 1) / baseK;
            uint32_t kTailAlign = kActCopyRowCountAlign - (kLoopTimes - 1) * baseK;
            uint32_t kTail = kActCopyRowCount - (kLoopTimes - 1) * baseK;
            for (uint32_t i = 0, actualBaseKAlign = baseK, actualBaseK = baseK; i < kLoopTimes; i++) {
                if (i + 1 == kLoopTimes) {
                    actualBaseKAlign = kTailAlign;
                    actualBaseK = kTail;
                }
                WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                LocalTensor<KV_T> curAL1Tensor = aL1Tensor[16 * baseK * i];
                uint32_t mmRowCount = msdIterNum * gSize;
                uint32_t copyStrideL0 = 16 * actualBaseKAlign;
                uint32_t copyStrideL1 = 16 * kActCopyRowCountAlign;
                uint32_t copyIterNum = (mmRowCount + 15) / 16;
                for(int k = 0; k < copyIterNum; k++){
                    LoadDataMm2A(aL0Tensor[k * copyStrideL0], curAL1Tensor[k * copyStrideL1], actualBaseKAlign);
                }
                SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[bL0BufIter % 2 * L0B_PP_SIZE / sizeof(KV_T)];
                uint32_t blockElementCnt = 32 / sizeof(KV_T);
                LoadData2dTransposeParams loadData2DTransposeParamsForB;
                loadData2DTransposeParamsForB.startIndex = 0;
                loadData2DTransposeParamsForB.srcStride = 1;
                loadData2DTransposeParamsForB.dstFracGap = 0;
                loadData2DTransposeParamsForB.repeatTimes = (actualBaseKAlign / blockElementCnt) * (headDimVAlign / blockElementCnt);
                loadData2DTransposeParamsForB.dstGap = blockElementCnt / 16 - 1;
                uint32_t l1BaseOffset = baseK * headDimVAlign * i;
                LoadDataWithTranspose(bL0Tensor, bL1Tensor[l1BaseOffset], loadData2DTransposeParamsForB);
                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
                MmadParams mmadParams;
                mmadParams.m = msdIterNum * gSize;
                if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = 128;
                mmadParams.k = actualBaseK; // 无效数据不参与计算
                mmadParams.cmatrixInitVal = (kCopyIdx == 0) && (i == 0);
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                aL0BufIter++;
                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
                bL0BufIter++;
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + kpL1BufIter % 2);
        kpL1BufIter++;
        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + (vL1BufIter % 2));
        vL1BufIter++;
    }
    SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
    WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
    FixpipeParamsV220 fixParams;
    fixParams.nSize = 128;
    fixParams.mSize = msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
    fixParams.srcStride = ((fixParams.mSize + 15) / 16) * 16;
    fixParams.dstStride = 128;
    fixParams.ndNum = 1;
    Fixpipe(mm2ResGm[(loop % (PRE_LOAD_NUM)) * bmm2ResUbSize], cL0Tensor, fixParams);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
    cL0BufIter++;
}