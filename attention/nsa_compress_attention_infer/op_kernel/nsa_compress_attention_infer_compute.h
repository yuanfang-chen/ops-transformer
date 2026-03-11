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
 * \file nsa_compress_attention_infer_compute.h
 * \brief
 */

#include "nsa_compress_attention_infer.h"

#pragma once

#ifdef __DAV_C220_CUBE__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::PerformQKMmad(const uint32_t qRowNum,
                                                                             const uint32_t kvSeqTile,
                                                                             const uint32_t l0cPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag);
    AscendC::Mmad(
        l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM],
        l0aBufTensor,
        l0bBufTensor,
        AscendC::MmadParams(qRowNum, kvSeqTile, headSizeQk, 0, false, 1));
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::PerformPVMmad(const uint32_t pRowNum,
                                                                             const uint32_t kvSeqTile,
                                                                             const uint32_t l0abPingPongFlag,
                                                                             const uint32_t l0cPingPongFlag,
                                                                             const uint32_t sIdx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 4);
    if (sIdx == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag + 2);
    }
    AscendC::Mmad(
        l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM],
        l0aBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM],
        l0bBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM],
        AscendC::MmadParams(pRowNum, headSizeVo, kvSeqTile, 0, false, sIdx == 0));
    AscendC::PipeBarrier<PIPE_M>();
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag + 2);
}

#elif __DAV_C220_VEC__

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxComputeVecInGmOffset(uint32_t processIdx, uint32_t ridx)
{
    uint32_t mm1ResPingPongFlag = (processIdx / aicNum) % 2;
    uint32_t mm2InPingPongFlag = (processIdx / aicNum) % 2;
    uint32_t vecInSplitOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 * alignedColLen;  // 输入对齐
    uint32_t vecOutSplitOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 * curSeqlen;  // 输出不对齐
    uint32_t vecOutSplitOffsetOut = (vecBlockIdx % 2 == 0) ? 0 : workSpaceElemNum / 2;
    // 计算从哪个数据开始处理
    softmaxTmpVecGmOffset = vecBlockIdx / 2 * workSpaceElemNum + mm1ResPingPongFlag * (mm1ResWorkSpaceSize / DOUBLE_BUFFER / sizeof(float)) + 
                            vecInSplitOffset + ridx * softmaxBasicRowLen * alignedColLen;
    mm2InWorkSpaceOffset = vecBlockIdx / 2 * workSpaceElemNum + mm2InPingPongFlag * (mm2InWorkSpaceSize / DOUBLE_BUFFER / sizeof(half)) + 
                           vecOutSplitOffset + ridx * softmaxBasicRowLen * curSeqlen;
    scoreInWorkSpaceOffset = vecBlockIdx / 2 * workSpaceElemNum + vecOutSplitOffsetOut + ridx * softmaxBasicRowLen * curSeqlen;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ComputeCurrentTokenOffset(uint32_t startIdx)
{
    // 已知起点属于哪一个token（qSeqLenOffset），属于哪一个kvhead（kvHeadOffset），当前块的起始地址startIdx
    // 注意：groupIdx是在当前process的索引
    uint32_t groupIdx = startIdx / groupSize;
    // 计算的当前的qseqlen索引
    currentQSeqLenOffset = (groupIdx % qSeqLenCurProcess + qSeqLenOffset);
}


template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::AddMask(uint32_t ridx, uint32_t startIdx, uint32_t endIdx)
{
    const int32_t targetCol = curSeqlen - 1; // 目标列索引
    const float minValue = FLOAT_MIN_MASK; // 需设置的常量值

    for (int32_t row = startIdx; row <= endIdx; ++row) {
        // 计算内存偏移：行索引*列总数 + 列索引。这里算出来的是整个vector的偏移，需要计算每个softmaxloop之内的偏移
        uint32_t offset = row * alignedColLen + targetCol - ((vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 * alignedColLen);
        offset -= ridx * softmaxBasicRowLen * alignedColLen;
        softmaxInputbufTensor.SetValue(offset, minValue);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::MaskOperation(uint32_t ridx, uint32_t startIdx, uint32_t endIdx)
{
    // 根据当前qseqidx、kvseq的长度（alignedColLen）、压缩的l(压缩窗口的大小)判断当前seq是否需要掩蔽
    // 计算压缩的时候，原始序列多余的token
    // 计算最后一个有效窗户的起始位置, curSeqlen - compSizeL表示了窗口起点可能的最大坐标
    uint32_t curSelKvSeq = actualSelKvSeqLenGm.GetValue(bIdx);
    uint32_t curSelKvSeqAlignStrideD = (actualKvSeqLenGm.GetValue(bIdx) - 1) * compStrideD + compSizeL;
    uint32_t tailKvTokens = curSelKvSeq - curSelKvSeqAlignStrideD;

    ComputeCurrentTokenOffset(startIdx);
    // 判断是否需要mask
    int64_t currentQSeqlen = actualQSeqLenGm.GetValue(bIdx);
    if(currentQSeqlen - currentQSeqLenOffset - 1 > tailKvTokens){
        AddMask(ridx, startIdx, endIdx);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::MaskApply(uint32_t ridx)
{   
    AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);
    uint32_t startRowIdx = (vecBlockIdx % 2 == 0) ? ridx * softmaxBasicRowLen : ridx * softmaxBasicRowLen + rowNumVec0;
    uint32_t endRowIdx = startRowIdx + basicRowLenCal - 1;
    uint32_t curRowIdx = startRowIdx;
    uint32_t nextGroupsizeIdx = startRowIdx;

    // 等号可取，相当于当前只需要计算一行数据
    while (curRowIdx <= endRowIdx) {
        // 下一个groupsize整数倍的点
        nextGroupsizeIdx = (curRowIdx / groupSize + 1) * groupSize;

        if (nextGroupsizeIdx > endRowIdx) {
            // 只计算curRowIdx到endRowIdx之间的
            MaskOperation(ridx, curRowIdx, endRowIdx);
            break; // 处理完剩余部分后退出循环
        } else {
            // 处理从curRowIdx到nextGroupsizeIdx - 1之间的部分
            MaskOperation(ridx, curRowIdx, nextGroupsizeIdx - 1);
        }

        curRowIdx = nextGroupsizeIdx;
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(1);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxCompute(uint32_t ridx)
{
    // 设置softmax的shape
    SoftMaxShapeInfo srcShape = { basicRowLenCal, alignedColLen, basicRowLenCal, static_cast<uint32_t>(curSeqlen)};
    // 调用softmax接口
    AscendC::Muls<float>(
        softmaxInputbufTensor,
        softmaxInputbufTensor,
        scaleValue,
        softmaxTileLength
    );
    PipeBarrier<PIPE_V>();

    // Mask处理
    if (attenMaskFlag == 1){
        MaskApply(ridx);
    }

    SoftMax<float>(softmaxOut32bufTensor, softmaxInputbufTensor, sharedTmpBuffer, softmaxTilingData, srcShape);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束
    PipeBarrier<PIPE_V>();
    if (std::is_same<OUT_T, __bf16>::value) {
        AscendC::Cast(softmaxOut16bufTensor, softmaxOut32bufTensor, AscendC::RoundMode::CAST_RINT, softmaxOut32bufTensor.GetSize());
    } else {
        AscendC::Cast(softmaxOut16bufTensor, softmaxOut32bufTensor, AscendC::RoundMode::CAST_NONE, softmaxOut32bufTensor.GetSize());
    }
    PipeBarrier<PIPE_V>();
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ComputeImpScoreValue(uint32_t loopCnt, uint32_t loopOutS2, uint32_t taskId, uint32_t baseS2Id, uint32_t startRowIdx, uint32_t endRowIdx, uint32_t startJ) {
    int32_t offsetS2Split = (startJ + 1) * strideOut - maxM - maxN;
    offsetS2Split = offsetS2Split < 0 ? 0 : offsetS2Split;
    int64_t inputOffsetGm = impScoreInputGmOffset + offsetS2Split;
    uint16_t rowCount = endRowIdx - startRowIdx;
    uint32_t groupNum = rowCount / gHeadNums;
    uint32_t blockLength = maxBaseS2 * 4;

    if (startJ == outS2 - 1) {
        if (taskId == loopCnt - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        LocalTensor<int32_t> tmpInfLocal = pslcTensor.template ReinterpretCast<int32_t>();
        Duplicate(tmpInfLocal[0], FLOAT_TYPE_MASK, 128);
        PipeBarrier<PIPE_V>();
        int64_t outputOffsetTail = impScoreOutputGmOffset + startJ;
        blockLength = 1 * 4;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
        ImpScoreDataCopyOut(taskId, baseS2Id, outputOffsetTail, groupNum, blockLength, 0);
        return;
    }
    if ((offsetS2Split + maxBaseS2) > curSeqlen) {
        blockLength = (curSeqlen - offsetS2Split) * 4;
    }

    ImpScoreDataCopyIn(inputOffsetGm, rowCount, blockLength);

    if ((taskId == loopCnt - 1) && (baseS2Id == loopOutS2 - 1)) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
    }
    uint32_t baseS2Align = ((blockLength + 63) / 64 * 64) / 4;
    uint32_t rowAlign = (rowCount + 15) / 16 * 16;
    ImpScoreTransposeTensor(rowAlign, baseS2Align);

    uint32_t actualStartJ = startJ + 1;
    uint32_t actualEndJ = (actualStartJ + maxBaseJ) > outS2 ? outS2 + 1 : actualStartJ + maxBaseJ;
    uint32_t outNewS2 = actualEndJ - actualStartJ;
    uint32_t outNewS2Align = AlignUp(outNewS2, 16);
    ImpScoreAccumulation(startJ, rowCount, offsetS2Split);
    ImpScoreReduceSum(startJ, rowCount, outNewS2);

    int64_t outputOffsetGm = impScoreOutputGmOffset + startJ;
    blockLength = outNewS2 * 4;
    uint32_t srcStride = outNewS2Align - outNewS2 >= 8 ? 1 : 0;

    ImpScoreDataCopyOut(taskId, baseS2Id, outputOffsetGm, groupNum, blockLength, srcStride);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreTransposeTensor(uint32_t height, uint32_t width) {
    ConfusionTransposeTiling tiling;
    uint32_t blockSizeT = 8;
    uint32_t highBlock = height / 16;
    uint32_t stride = height * blockSizeT * 4 / 32;
    uint32_t repeat = width / blockSizeT;

    tiling.param0 = blockSizeT;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;

    ConfusionTranspose(pslcCalcTensor, pslcTensor, pslcSharedBuffer, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
    PipeBarrier<PIPE_V>();
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreAccumulation(uint32_t startJ, uint32_t rowCount, uint32_t offsetS2Split) {
    uint32_t actualStartJ = startJ + 1;
    uint32_t addLoop = maxM + maxN + 1;
    uint32_t times = 1;
    uint32_t rowAlign = (rowCount + 15) / 16 * 16;
    uint32_t actualEndJ = (actualStartJ + maxBaseJ) > outS2 ? outS2 + 1 : actualStartJ + maxBaseJ;
    uint32_t outNewS2 = actualEndJ - actualStartJ;

    Duplicate(pslcTensor, (float)0, baseBlockSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < addLoop; i++) {
        if (i < maxN) {
            times = i + 1;
        }
        if (i >= maxN && i <= maxM) {
            times = maxN + 1;
        }
        if (i > maxM) {
            times = addLoop - i;
        }
        int64_t pIdxOffset = i + offsetS2Split;
        for (uint32_t j=actualStartJ; j<actualEndJ; j++) {
            uint32_t pslcOffset = (j - actualStartJ) * rowAlign;

            int64_t pIdx = strideOut * j - pIdxOffset;
            
            uint32_t pcmpOffsetLocal = pIdx * rowAlign;
            if (pIdx < 0) {
                continue;
            }
            if ((pIdx + offsetS2Split) >= curSeqlen) {
                break;
            }
            if (times > 1) {
                float mulTimes = times * (float)1.0;
                Muls(pslcTmpTensor, pslcCalcTensor[pcmpOffsetLocal], mulTimes, rowCount);
                PipeBarrier<PIPE_V>();
                Add(pslcTensor[pslcOffset], pslcTensor[pslcOffset], pslcTmpTensor, rowCount);
                PipeBarrier<PIPE_V>();
            } else {
                Add(pslcTensor[pslcOffset], pslcTensor[pslcOffset], pslcCalcTensor[pcmpOffsetLocal], rowCount);
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreReduceSum(uint32_t startJ, uint32_t rowCount, uint32_t outNewS2) {
    uint32_t rowAlign = (rowCount + 15) / 16 * 16;
    uint32_t groupNum = rowCount / gHeadNums;
    LocalTensor<int32_t> tmpLocal = pslcTensor.template ReinterpretCast<int32_t>();
    if (startJ == 0 || outNewS2 == 1 ) {
        Duplicate(tmpLocal[0], FLOAT_TYPE_MASK, rowCount);
    }
    if ((outNewS2+startJ) == (outS2 - 1)) {
        Duplicate(tmpLocal[(outNewS2-1)*rowAlign], FLOAT_TYPE_MASK, rowAlign);
    }
    if (outNewS2 > 1 && (outNewS2+startJ) >= outS2 ) {
        Duplicate(tmpLocal[ (outNewS2-2)*rowAlign], FLOAT_TYPE_MASK, rowAlign*2);
    }
    PipeBarrier<PIPE_V>();

    uint32_t outNewS2Align = AlignUp(outNewS2, 16);
    ImpScoreTransposeTensor(outNewS2Align, rowAlign);

    Duplicate(pslcTensor, (float)0.0, baseBlockSize);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < groupNum; i++) {
        for (uint32_t j = 0; j < gHeadNums; j++) {
            Add(pslcTensor[i*outNewS2Align], pslcTensor[i*outNewS2Align], pslcCalcTensor[(i * gHeadNums + j) *outNewS2Align], outNewS2Align);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::TopkCompute()
{   
    TopKInfo topkinfo;
    topkinfo = {(int32_t)(1), (int32_t)(alignedTok32), (int32_t)(outS2)};
    for(int i=0; i<8; i++){
        arithbuffer.SetValue(i, i);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    uint32_t topindexloop = alignedTok32 / 8;
    for(int i=1; i<topindexloop; i++){
        Adds(arithbuffer[i * 8], arithbuffer, (int32_t)i * 8, 8);
        PipeBarrier<PIPE_V>();
    }
    AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL>(topkoutvalueLocal,
        topkoutindexLocal, topkInputbufTensor, arithbuffer, topksrcLocalFinish, topksharedTmpBuffer, selectNum, this->topkTilingData, topkinfo, true);
    PipeBarrier<PIPE_V>();
}

#endif