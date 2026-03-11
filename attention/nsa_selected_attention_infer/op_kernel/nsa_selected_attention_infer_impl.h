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

#include "nsa_selected_attention_infer.h"

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::DealBmm1ResBaseBlock(const uint32_t loop, uint32_t startRow,
                                                    uint32_t dealRowCount, uint32_t columnCount,
                                                    uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();

    uint64_t inOutGmOffset = (loop % PRE_LOAD_NUM) * mmResUbSize + (gSizeStart + startRow) * columnCount;
    LocalTensor<MM_OUT_T> tmpMmResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpMmResUb, mm1ResGm[inOutGmOffset], computeSize);
    inputQue1.EnQue(tmpMmResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(mmResUb, tmpMmResUb, computeSize);
    inputQue1.FreeTensor(tmpMmResUb);
    PipeBarrier<PIPE_V>();
    ElewiseCompute(loop, mmResUb, dealRowCount, columnCount);
    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(loop, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<KV_T> tmpMMResCastTensor = outputQue1.AllocTensor<KV_T>();
    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);

    outputQue1.EnQue(tmpMMResCastTensor);
    outputQue1.DeQue<KV_T>();
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    
    outputQue1.FreeTensor(tmpMMResCastTensor);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec1Inner(uint32_t loop)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    if (gSplitSize > gSizeVector) {
        gSplitSize = gSizeVector;
    }
    uint32_t loopCount = (gSizeVector + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSizeVector - (loopCount - 1) * gSplitSize;
    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(loop, i * gSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                                info.actualSingleProcessSInnerSize);
    }
}

template <typename NSAT>
__aicore__ inline void
NsaSelectAttentionInfer<NSAT>::DealBmm2ResBaseBlock(const uint32_t loop, uint32_t startRow,
                                                    uint32_t dealRowCount, uint32_t columnCount,
                                                    uint32_t actualColumnCount)
{
    ExtraInfo& info = extraInfo[loop % (PRE_LOAD_NUM)];
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    uint64_t inOutBaseOffset = (gSizeStart + startRow) * columnCount;
    uint64_t srcGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<MM_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
    inputQue1.EnQue(tmpBmm2ResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    inputQue1.FreeTensor(tmpBmm2ResUb);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
        uint64_t vec2ResGmOffset = ((loop - 1) % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);

        inputQue2.DeQue<T>();
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[loop % (PRE_LOAD_NUM)][baseOffset], dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();
        uint64_t vec2ResGmOffset = (loop % PRE_LOAD_NUM) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec2Inner(uint32_t loop)
{
    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimVAlign;
    if (gSplitSize > gSizeVector) {
        gSplitSize = gSizeVector;
    }
    uint32_t loopCount = (gSizeVector + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = gSizeVector - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(loop, i * gSplitSize, dealSize, headDimVAlign, headDimV);
    }
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec2L(uint32_t loop) {
    if (gSizeVector == 0) {
        return;
    }
    ProcessVec2Inner(loop);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::ProcessVec1L(uint32_t loop)
{
    if (gSizeVector == 0) {
        return;
    }
    ProcessVec1Inner(loop);
}

template <typename NSAT> __aicore__ inline void NsaSelectAttentionInfer<NSAT>::SetLoopTimes() {
    // 获取总的循环次数，此处假设所有head的S=Smax
    bn2s2LoopTimes = 0;
    bool hasInit = false;
    for (uint32_t bn2Idx = 0; bn2Idx < bn2LoopTimes; bn2Idx++) {
        GetBN2id(bn2Idx);
        GetActualSeqLen();
        // 根据当前块实际长度, 重配flashattention循环条件
        UpdateInnerLoopCond();
        if (curActSeqLenIsZero) {
            DealActSeqLenIsZero(bIdx, n2Idx);
            continue;
        }
        if (!hasInit) {
            curBIdx = bIdx;
            curN2Idx = n2Idx;
            curS2Idx = s2Idx;
            curS1Idx = s1Idx;
            curSInnerLoopTimes = sInnerLoopTimes;
            hasInit = true;
        }
        bn2s2LoopTimes += sInnerLoopTimes;
    }
}

template <typename NSAT>
__aicore__ inline bool NsaSelectAttentionInfer<NSAT>::IsFinish(uint32_t loop) {
  return (loop >= bn2s2LoopTimes);
}

template <typename NSAT>
__aicore__ inline void NsaSelectAttentionInfer<NSAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        SetLoopTimes();
        if ASCEND_IS_AIV {
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, gSize * 32 / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, gSize * 32 / sizeof(T));
        } else {
            // AllocEventID
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT1);
            SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);

            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
            SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

            SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
            SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }

        for (uint32_t i = 0; i < bn2s2LoopTimes; i = i + PRE_LOAD_NUM) {
            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;
                if (i != 0) {
                    if (!IsFinish(loop - PRE_LOAD_NUM)) {
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                            ProcessVec2L(loop - PRE_LOAD_NUM);
                        }
                    }
                }

                if (!IsFinish(loop)) {
                    CalcParams(loop);
                    if ASCEND_IS_AIC {
                        ComputeMm1(loop);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
                    }
                }
            }

            for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                uint32_t loop = i + j;
                if (!IsFinish(loop)) {
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
                        ProcessVec1L(loop);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag(SYNC_V1_C2_FLAG);
                        ComputeMm2(loop);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG);
                    }
                }
            }

            if (i + PRE_LOAD_NUM >= bn2s2LoopTimes) {
                for (uint32_t j = 0; j < PRE_LOAD_NUM; j++) {
                    uint32_t loop = i + j;
                    if (!IsFinish(loop)) {
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                            ProcessVec2L(loop);
                        }
                    }
                }
            }
        }

        if ASCEND_IS_AIC {
            // FreeEventID
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT1);
            WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);

            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
        }
    }
}