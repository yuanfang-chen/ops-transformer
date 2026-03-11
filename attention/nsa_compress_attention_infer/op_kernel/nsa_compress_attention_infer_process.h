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
 * \file nsa_compress_attention_infer_process.h
 * \brief
 */

#include "nsa_compress_attention_infer.h"

#pragma once
using namespace NSA_COMPRESS_ATTENTION_INFER;

#ifdef __DAV_C220_CUBE__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::ProcessMm1(const uint32_t processIdx)
{   
    uint64_t strideQ = qHeadNum * headSizeQk;
    uint64_t strideK = kvHeadNum * headSizeQk;
    uint32_t headSizeQkRound = AlignUp(headSizeQk, 16U);
    uint32_t mm1ResPingPongFlag = (processIdx / aicNum) % 2;
    for (uint32_t curKvHeadIdx = 0; curKvHeadIdx < curKvHeadNum; curKvHeadIdx++) {
        uint32_t actualKvHeadIdx = kvHeadOffset + curKvHeadIdx;        
        uint32_t l1qPingPongFlag = curKvHeadIdx % 2;
        uint32_t qRowNum = groupSize * qSeqLenCurProcess;
        uint32_t qRowNumRound = (qRowNum + 16U - 1) / 16U * 16U;

        CopyQToL1(qRowNum, qRowNumRound, headSizeQkRound, actualKvHeadIdx, l1qPingPongFlag);
        int64_t curSeqlen = actualKvSeqLenGm.GetValue(bIdx);
        int64_t curSeqlenRound = AlignUp(curSeqlen, 16L);
        uint32_t sLoop = (static_cast<uint32_t>(curSeqlen) + blockSize - 1) / blockSize;
        uint32_t kvSeqTile = blockSize;

        for (uint32_t sIdx = 0; sIdx < sLoop; sIdx++) {
            uint32_t l1kPingPongFlag = (curKvHeadIdx * sLoop + sIdx) % 2;
            uint32_t l0cPingPongFlag = (curKvHeadIdx * sLoop + sIdx) % 2;
            if (sIdx == sLoop - 1) {
                kvSeqTile = curSeqlen - sIdx * blockSize;
            }
            uint32_t kvSeqTileRound = AlignUp(kvSeqTile, 16U);
            uint32_t blockTableOffset = maxBlockNumPerBatch * bIdx + sIdx;
            uint32_t blockIdx = blockTableGm.GetValue(blockTableOffset);
            uint32_t kBlockOffset = blockIdx * blockSize * strideK;
            uint32_t kHiddenOffset = actualKvHeadIdx * headSizeQk;
            uint32_t kCoreOffset = kBlockOffset + kHiddenOffset;

            CopyKToL1(kvSeqTile, kvSeqTileRound, strideK, kCoreOffset, l1kPingPongFlag);

            LoadQKToL0(qRowNum, qRowNumRound, headSizeQkRound, kvSeqTileRound,
                       l1qPingPongFlag, l1kPingPongFlag, sIdx, sLoop);

            PerformQKMmad(qRowNum, kvSeqTile, l0cPingPongFlag);

            CopySToWorkSpace(qRowNum, qRowNumRound, kvSeqTileRound, curSeqlenRound,
                             curKvHeadIdx, sIdx, l0cPingPongFlag, mm1ResPingPongFlag);
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::ProcessMm2(const uint32_t processIdx)
{   
    uint64_t strideO = qHeadNum * headSizeVo;
    uint64_t strideV = kvHeadNum * headSizeVo;
    uint32_t mm2InPingPongFlag = (processIdx / aicNum) % 2;
    for (uint32_t curKvHeadIdx = 0; curKvHeadIdx < curKvHeadNum; curKvHeadIdx++) {
        uint32_t actualKvHeadIdx = kvHeadOffset + curKvHeadIdx;

        for (uint32_t qSeqLenCurProcessIndex = 0; qSeqLenCurProcessIndex < qSeqLenCurProcess; qSeqLenCurProcessIndex++) {
            uint64_t oCoreOffset = (qSeqLenCumSum + qSeqLenOffset + qSeqLenCurProcessIndex) * strideO + actualKvHeadIdx * groupSize * headSizeVo;

            uint32_t pRowNum = groupSize;
            uint32_t pRowNumRound = AlignUp(pRowNum, 16U);
            uint32_t headSizeVoRound = AlignUp(headSizeVo, 16U);
            int64_t curSeqlen = actualKvSeqLenGm.GetValue(bIdx);
            uint32_t sLoop = (static_cast<uint32_t>(curSeqlen) + blockSize - 1) / blockSize;
            int64_t curSeqlenRound = AlignUp(curSeqlen, 16L);
            uint32_t kvSeqTile = blockSize;
            uint32_t l0cPingPongFlag = (curKvHeadIdx * qSeqLenCurProcess + qSeqLenCurProcessIndex) % 2;
            for (uint32_t sIdx = 0; sIdx < sLoop; sIdx++) {
                uint32_t l1pvPingPongFlag = (curKvHeadIdx * qSeqLenCurProcess * sLoop + qSeqLenCurProcessIndex * sLoop + sIdx) % 2;
                uint32_t l0abPingPongFlag = (curKvHeadIdx * qSeqLenCurProcess * sLoop + qSeqLenCurProcessIndex * sLoop + sIdx) % 2;
                if (sIdx == sLoop - 1) {
                    kvSeqTile = curSeqlen - sIdx * blockSize;
                }
                uint32_t kvSeqTileRound = AlignUp(kvSeqTile, 16U);
                uint64_t pCoreOffset = mm2InPingPongFlag * (mm2InWorkSpaceSize / 2 / 2) +
                                    cubeBlockIdx * workSpaceElemNum +
                                    (curKvHeadIdx * qSeqLenCurProcess + qSeqLenCurProcessIndex) * groupSize * curSeqlen +
                                    sIdx * blockSize;
                uint32_t blockTableOffset = maxBlockNumPerBatch * bIdx + sIdx;
                uint32_t blockIdx = blockTableGm.GetValue(blockTableOffset);
                uint32_t vBlockOffset = blockIdx * blockSize * strideV;
                uint32_t vHiddenOffset = actualKvHeadIdx * headSizeVo;
                uint32_t vCoreOffset = vBlockOffset + vHiddenOffset;

                CopyPVToL1(pRowNum, pRowNumRound, curSeqlen, kvSeqTile, kvSeqTileRound, strideV,
                        pCoreOffset, vCoreOffset, l0abPingPongFlag, l1pvPingPongFlag);

                LoadPVToL0(pRowNum, pRowNumRound, headSizeVoRound, kvSeqTileRound, l1pvPingPongFlag, l0abPingPongFlag);

                PerformPVMmad(pRowNum, kvSeqTile, l0abPingPongFlag, l0cPingPongFlag, sIdx);
            }
            CopyOToGm(pRowNum, pRowNumRound, oCoreOffset, l0cPingPongFlag);
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::Process()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(5);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
    for (uint32_t processIdx = cubeBlockIdx; processIdx < processNum; processIdx+=aicNum) {
        PreProcess(processIdx);
        ProcessMm1(processIdx);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(MM1_READY);
        CrossCoreWaitFlag(SOFTMAX_READY);
        ProcessMm2(processIdx);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
}

#elif __DAV_C220_VEC__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::Process()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
    for (uint32_t processIdx = cubeBlockIdx; processIdx < processNum; processIdx+=aicNum) {
        PreProcessOffset(processIdx);
        PreProcess(processIdx);
        CrossCoreWaitFlag(MM1_READY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
        // Softmax
        ProcessSoftmax(processIdx);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SOFTMAX_READY);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0); // score搬入等softmax搬出结束
        // ImportanceScore
        ProcessImportanceScore(processIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2); // topk搬入等score搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2); // topk搬入等score搬出结束
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0); // topk Duplicate等score搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0); // topk Duplicate等score搬出结束
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0); // topk SetValue等score搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0); // topk SetValue等score搬出结束
        // TopK
        ProcessTopK();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessSoftmax(uint32_t processIdx)
{   
    for (uint32_t ridx = 0; ridx < softmaxRowLoop; ridx++) {
        basicRowLenCal =
                static_cast<uint32_t>((ridx == softmaxRowLoop - 1) ? (softmaxRowLenPerCore - (softmaxRowLoop - 1) * softmaxBasicRowLen)
                                                            : softmaxBasicRowLen);  // 每核处理的最后一个行循环单独处理
        DataCopyParams splitCopyinParams;
        DataCopyParams splitCopyoutParams;
        DataCopyParams splitCopyout32Params;

        splitCopyinParams = {basicRowLenCal, (uint16_t)(alignedColLen * sizeof(float)), 0, 0};
        splitCopyoutParams = {basicRowLenCal, (uint16_t)(curSeqlen * sizeof(half)), 0, 0};

        uint16_t copyoutSrcStride = alignedColLen - curSeqlen >= 8 ? 1 : 0;
        splitCopyout32Params = {basicRowLenCal, (uint16_t)(curSeqlen * sizeof(float)), copyoutSrcStride, 0};
        // 计算从哪个数据开始
        SoftmaxComputeVecInGmOffset(processIdx, ridx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
        SoftmaxCopyIn(splitCopyinParams,ridx);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); // softmax计算等待softmax搬入结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0); // softmax计算等待softmax搬入结束
        SoftmaxCompute(ridx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); // softmax搬出等待softmax计算结束
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0); // softmax搬出等待softmax计算结束

        SoftmaxCopyOut(splitCopyoutParams, splitCopyout32Params, ridx);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessImportanceScore(uint32_t processIdx)
{
    uint32_t perBaseGroup = maxRowCountPerLoop / gHeadNums;
    uint32_t perCoreGroup = impSocreCoreRowCount / gHeadNums;
    uint32_t loopCnt = perBaseGroup > 0 ? CeilDiv(perCoreGroup, perBaseGroup) : 0;
    uint32_t tailGroupCnt = perCoreGroup % perBaseGroup;

    // S1方向切分
    for (uint32_t taskId = 0; taskId < loopCnt; taskId++) {
        uint32_t startRowIdx = taskId * perBaseGroup * gHeadNums;
        uint32_t curGroupCnt = (taskId == loopCnt - 1) ? (perCoreGroup - taskId * perBaseGroup) : perBaseGroup;
        uint32_t endRowIdx = startRowIdx + curGroupCnt * gHeadNums;
        ProcessImpScoreS2Loop(loopCnt, processIdx, taskId, startRowIdx, endRowIdx);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessImpScoreS2Loop(uint32_t loopCnt, uint32_t processIdx, uint32_t taskId, uint32_t startRowIdx, uint32_t endRowIdx)
{
    // S2方向上的切分
    uint32_t loopOutS2 =  CeilDiv(outS2, maxBaseJ);
    uint32_t vecOutSplitOffset = (vecBlockIdx % 2 == 0) ? 0 : workSpaceElemNum / 2;
    uint32_t impScoreOutputSubCoreOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 / gHeadNums * maxOutS2;
    impScoreInputGmOffset = vecBlockIdx / 2 * workSpaceElemNum + vecOutSplitOffset + startRowIdx * curSeqlen;
    impScoreOutputGmOffset = vecBlockIdx / 2 * workSpaceElemNum / gHeadNums + impScoreOutputSubCoreOffset + startRowIdx * outS2 / gHeadNums;

    for (uint32_t baseS2Id = 0; baseS2Id < loopOutS2; baseS2Id++) {
        uint32_t startJ = baseS2Id * maxBaseJ;
        ComputeImpScoreValue(loopCnt, loopOutS2, taskId, baseS2Id, startRowIdx, endRowIdx, startJ);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessTopK()
{
    DataCopyParams splitCopyintopkParams;
    DataCopyParams splitCopyouttopkParams;

    alignedoutS2 = AlignUp(outS2, 8);
    alignedTok32 = AlignUp(outS2, 32);
    alignedTokloop = AlignUp(outS2, 32) - alignedoutS2;
    topkRightPadding = alignedoutS2 - outS2;
    uint32_t impScoreOutputSubCoreOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 / gHeadNums * maxOutS2;
    uint32_t coreRowOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 / gHeadNums;
    uint32_t perCoreGroup = impSocreCoreRowCount / gHeadNums;
    splitCopyintopkParams = {1, (uint16_t)(outS2 * sizeof(float)), 0, 0};

    splitCopyouttopkParams = {1, (uint16_t)(selectNum * sizeof(int32_t)), 0, 0};
    // S1方向切分
    for (uint32_t taskId = 0; taskId < perCoreGroup; taskId++) {
        topkInputGmOffset = vecBlockIdx / 2 * workSpaceElemNum / gHeadNums + impScoreOutputSubCoreOffset + taskId * outS2;
        // 推算, taskId，qSeqLenCurProcess中第几个qSeqLen，第几个kvHeadSplit, 第几个kvHead
        uint32_t actualRowIdx = taskId + coreRowOffset;
        uint32_t kvHeadIdxInSplit = actualRowIdx / qSeqLenCurProcess;
        uint32_t qIdxInSplit = actualRowIdx % qSeqLenCurProcess;

        topkOutputGmOffset = qSeqLenCumSum * kvHeadNum * selectNum +
                                qSeqLenOffset * kvHeadNum * selectNum + qIdxInSplit * kvHeadNum * selectNum +
                                kvHeadOffset * selectNum + kvHeadIdxInSplit * selectNum;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
        TopkCopyIn(splitCopyintopkParams);
        if (taskId == perCoreGroup - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2); // topk计算等待topk搬入结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2); // topk计算等待topk搬入结束
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        TopkCompute();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
        TopkCopyOut(splitCopyouttopkParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
    }
}

#endif