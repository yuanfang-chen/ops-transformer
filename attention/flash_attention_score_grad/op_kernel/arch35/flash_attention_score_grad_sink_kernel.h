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
 * \file flash_attention_score_grad_sink_kernel.h
 * \brief Sink Kernel for FAG: computes gradients for sink tokens (key_sink, value_sink)
 *        using BN2S2 template. Results are written to dkSinkWorkSpace/dvSinkWorkSpace.
 */
#ifndef _FLASH_ATTENTION_SCORE_GRAD_SINK_KERNEL_H_
#define _FLASH_ATTENTION_SCORE_GRAD_SINK_KERNEL_H_

#include "flash_attention_score_grad_s1s2_bn2s2_regbase.h"

namespace FagBaseApi {

FAG_CLASS_TEMPLATE
class FlashAttentionScoreGradSinkKernel
    : public FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE> {
public:
    using BASE_TYPE = FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>;
    using S1S2_TEMPLATE = typename BASE_TYPE::S1S2_TEMPLATE;

    __aicore__ inline FlashAttentionScoreGradSinkKernel(){};

    __aicore__ inline void Init(__gm__ uint8_t *keySink, __gm__ uint8_t *valueSink,
                                __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *y, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
                                __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                                __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                                __gm__ uint8_t *dpse, __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope,
                                __gm__ uint8_t *workspace,
                                FagOldTilingType ordTilingData, TPipe *pipeIn,
                                TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn,
                                TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSinkLoop();
    __aicore__ inline void ProcessReComputeLse(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetKeySinkOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetValueSinkOffset(FagRunInfo &runInfo);
    __aicore__ inline void SetSinkRunInfo(FagRunInfo &runInfo, int64_t taskId, int64_t blockInnerIdx);

    // Sink-specific GM tensors
    GlobalTensor<T1> keySinkGm;
    GlobalTensor<T1> valueSinkGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<float> dkSinkWorkSpaceGm;
    GlobalTensor<float> dvSinkWorkSpaceGm;
};

// ======================== Implementation ========================

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradSinkKernel<FAG_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *keySink, __gm__ uint8_t *valueSink,
    __gm__ uint8_t *dx, __gm__ uint8_t *query,
    __gm__ uint8_t *pseShift, __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *y, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
    __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
    __gm__ uint8_t *dpse, __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope,
    __gm__ uint8_t *workspace,
    FagOldTilingType ordTilingData, TPipe *pipeIn,
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn,
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn)
{
    // Call parent Init with keySink as key and valueSink as value
    // This sets up keyGm/valueGm to point to sink data, so all MM operations
    // will automatically use sink key/value tensors
    BASE_TYPE::Init(keySink, valueSink, dx, query, pseShift, dropMask, attenMask, y,
                    (__gm__ uint8_t *)nullptr, (__gm__ uint8_t *)nullptr, // softmaxMax/softmaxSum not used
                    prefixN, actualSeqQlen, actualSeqKvlen,
                    deqScaleQ, deqScaleK, deqScaleV, deqScaleDy,
                    queryRope, keyRope, dq, dk, dv, dpse, dqRope, dkRope,
                    workspace, ordTilingData, pipeIn, dsScmIn, pScmIn);

    // Set up sink-specific GM tensors
    keySinkGm.SetGlobalBuffer((__gm__ T1 *)keySink);
    valueSinkGm.SetGlobalBuffer((__gm__ T1 *)valueSink);
    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);

    // Set up sink workspace GM: dk_sink and dv_sink accumulated in FP32 workspace
    dkSinkWorkSpaceGm.SetGlobalBuffer(
        (__gm__ float *)workspace + this->tilingData->postTilingData.dkSinkWorkSpaceOffset / sizeof(float));
    dvSinkWorkSpaceGm.SetGlobalBuffer(
        (__gm__ float *)workspace + this->tilingData->postTilingData.dvSinkWorkSpaceOffset / sizeof(float));

    // Override dkWorkSpaceGm/dvWorkSpaceGm to point to sink workspace
    // so that parent's IterateMm4/IterateMm5 write to sink workspace
    this->dkWorkSpaceGm = dkSinkWorkSpaceGm;
    this->dvWorkSpaceGm = dvSinkWorkSpaceGm;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradSinkKernel<FAG_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    ProcessSinkLoop();
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradSinkKernel<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessSinkLoop()
{
    // Use sinkBlockNumList for block navigation instead of s1s2BNGS1S2BlockNumList
    if (this->tilingData->sinkBlockNumList.blockEnds[this->cBlockIdx] == 0) {
        return;
    }

    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for ping pong

    int64_t nextValidBlockInnerIdx = 0;
    int64_t blockInnerIdx = 0;
    int64_t dqkvBlockInnerIdx = 0;

    // Use sink block list for block starts
    nextValidBlockInnerIdx =
        this->GetNextValidIdx(runInfos[0], this->tilingData->sinkBlockNumList.blockStarts[this->cBlockIdx]);
    blockInnerIdx = nextValidBlockInnerIdx;

    while (blockInnerIdx < this->tilingData->sinkBlockNumList.blockEnds[this->cBlockIdx] + 1) {
        this->isLastLoop = (blockInnerIdx == -1);
        dqkvBlockInnerIdx = blockInnerIdx;

        if (taskId > 0) {
            // Softmax recompute using LSE: P = exp(S - lse)
            ProcessReComputeLse(runInfos[(taskId + 1) & 1]);
            this->WaitMm1Mm2Result();
        }

        if (!this->isLastLoop) {
            this->SetRunInfo(runInfos[taskId & 1], taskId, blockInnerIdx);
            blockInnerIdx++;
            nextValidBlockInnerIdx = this->GetNextValidIdx(runInfos[(taskId + 1) & 1], blockInnerIdx);

            // MM1: dy * value_sink^T and MM2: query * key_sink^T
            this->IterateMm1Mm2(runInfos[taskId & 1], nextValidBlockInnerIdx);

            // Use CopyInLse instead of CopyInMaxSum for LSE-based softmax
            CopyInLse<T2, S1S2_TEMPLATE::VECTOR_BASEM>(this->constInfo, runInfos[taskId & 1],
                                                        this->maxSumQue[taskId & 1], this->softmaxLseGm);
        }

        if (taskId > 0) {
            // Softmax recompute using LSE: P = exp(S - lse)
            ProcessReComputeLse(runInfos[(taskId + 1) & 1]);
            // MM3 (dq=ds*key_sink), MM4 (dk_sink=ds^T*query), MM5 (dv_sink=p^T*dy)
            this->IterateMm3Mm4Mm5(runInfos[(taskId + 1) & 1], dqkvBlockInnerIdx,
                                   this->GetKeyOffset(runInfos[taskId & 1]));
        }

        if (blockInnerIdx == -1) {
            break;
        }
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradSinkKernel<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessReComputeLse(FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax (LSE version)
    // P = exp(S * scale + pse + attenMask - lse)
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> mm2ResQueInTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<T2>();
    LocalTensor<T2> mm1ResQueInTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<T2>();

    CopyInAttenMask<IS_ATTEN_MASK, S1S2_TEMPLATE::VECTOR_BASEM, S1S2_TEMPLATE::VECTOR_BASEN>(
        this->constInfo, runInfo, this->attenMaskInfo, this->attenMaskOrYInQue,
        this->pseOrDyInQue, this->attenMaskU8Gm);
    CopyInPse<OUTDTYPE, T2, IS_PSE>(this->constInfo, runInfo, this->pseInfo, this->pseOrDyInQue, this->pseGm);

    // Use LSE softmax variant: exp(S - lse) instead of exp(S - max) / sum
    CalculatePseMulsSelSimpleSoftMaxLse<OUTDTYPE, T2, S1S2_TEMPLATE::IS_FP8_INPUT, IS_ATTEN_MASK, IS_PSE,
        IS_DETER_OLD(DETER_SPARSE_TYPE), S1S2_TEMPLATE::VECTOR_BASEM, S1S2_TEMPLATE::VECTOR_BASEN>(
        this->constInfo, runInfo, this->pseInfo, this->attenMaskInfo,
        this->maxSumQue[runInfo.commonRunInfo.taskIdMod2], this->attenMaskOrYInQue,
        this->pseOrDyInQue, mm2ResQueInTensor, mm2ResQueInTensor, this->pseSlope);

    if (this->dropInfo.dropMaskOuter) {
        if (this->dropInfo.boolMode) {
            CopyInDropOuter<IS_DROP>(this->dropMaskBuf, this->attenMaskOrYInQue,
                this->dropMaskWorkspaceGm, runInfo.commonRunInfo, this->constInfo.commonConstInfo, this->dropInfo);
        } else {
            CopyInDropOuter<IS_DROP>(this->dropMaskBuf, this->attenMaskOrYInQue,
                this->dropMaskGm, runInfo.commonRunInfo, this->constInfo.commonConstInfo, this->dropInfo);
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradSinkKernel<FAG_FUNCTION_PARAMS_TEMPLATE>::GetKeySinkOffset(FagRunInfo &runInfo)
{
    // key_sink shape: [sinkNum, N2, D]
    // offset = n2Idx * sinkS2Size * D + s2StartIdx * D
    int64_t d = this->constInfo.commonConstInfo.dSize;
    int64_t sinkS2Size = this->tilingData->s1s2BNGS1S2SplitCoreParams.sinkS2Size;
    return runInfo.commonRunInfo.n2DimIdx * sinkS2Size * d +
           runInfo.commonRunInfo.s2oDimIdx * this->tilingData->s1s2BNGS1S2SplitCoreParams.s2Inner * d;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradSinkKernel<FAG_FUNCTION_PARAMS_TEMPLATE>::GetValueSinkOffset(FagRunInfo &runInfo)
{
    // value_sink shape: [sinkNum, N2, D_V]
    int64_t dv = this->constInfo.commonConstInfo.dSizeV;
    int64_t sinkS2Size = this->tilingData->s1s2BNGS1S2SplitCoreParams.sinkS2Size;
    return runInfo.commonRunInfo.n2DimIdx * sinkS2Size * dv +
           runInfo.commonRunInfo.s2oDimIdx * this->tilingData->s1s2BNGS1S2SplitCoreParams.s2Inner * dv;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradSinkKernel<FAG_FUNCTION_PARAMS_TEMPLATE>::SetSinkRunInfo(
    FagRunInfo &runInfo, int64_t taskId, int64_t blockInnerIdx)
{
    // Decode block index for sink: (bIdx, n2Idx, gIdx, s2oIdx, s1oIdx)
    // fusedOuter = B * N2 * G * sinkS2Outer, block = fusedOuter * s1Outer
    int64_t sinkS2Outer = this->tilingData->s1s2BNGS1S2SplitCoreParams.s2SinkOuter;
    int64_t s1Outer = this->tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    int64_t g = this->tilingData->s1s2BNGS1S2BaseParams.g;
    int64_t n2 = this->tilingData->s1s2BNGS1S2BaseParams.n2;

    int64_t s1oIdx = blockInnerIdx % s1Outer;
    int64_t fusedIdx = blockInnerIdx / s1Outer;
    int64_t s2oIdx = fusedIdx % sinkS2Outer;
    int64_t remainder = fusedIdx / sinkS2Outer;
    int64_t gIdx = remainder % g;
    int64_t n2Idx = (remainder / g) % n2;
    int64_t bIdx = remainder / (g * n2);

    runInfo.commonRunInfo.boIdx = bIdx;
    runInfo.commonRunInfo.n2DimIdx = n2Idx;
    runInfo.commonRunInfo.gDimIdx = gIdx;
    runInfo.commonRunInfo.s2oDimIdx = s2oIdx;
    runInfo.commonRunInfo.s1oDimIdx = s1oIdx;
    runInfo.commonRunInfo.taskIdMod2 = taskId & 1;

    // Set s2RealSize for the current sink block
    int64_t sinkS2Size = this->tilingData->s1s2BNGS1S2SplitCoreParams.sinkS2Size;
    int64_t sinkS2Tail = this->tilingData->s1s2BNGS1S2SplitCoreParams.sinkS2Tail;
    int64_t s2Inner = this->tilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;
    runInfo.commonRunInfo.s2RealSize = (s2oIdx == sinkS2Outer - 1) ? sinkS2Tail : s2Inner;
}

} // namespace FagBaseApi

#endif // _FLASH_ATTENTION_SCORE_GRAD_SINK_KERNEL_H_
