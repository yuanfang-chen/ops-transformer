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
 * \file flash_attention_score_grad_s1s2_bn2s2_regbase.h
 * \brief
 */
#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2S2_REGBASE_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2S2_REGBASE_H_

#include "flash_attention_score_grad_s1s2_bn2gs1s2_regbase.h"
using namespace matmul;

__aicore__ constexpr TPosition GetC2Position(bool IS_L0C_REUSE, bool IS_DKV_RES_EXCEED_UB)
{
    if (IS_L0C_REUSE && !IS_DKV_RES_EXCEED_UB) {
        return TPosition::VECCALC;
    } else {
        return TPosition::GM;
    }
}

FAG_CLASS_TEMPLATE
class FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase
    : public FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE> {
public:
    __aicore__ inline FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *y, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
                                __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy, 
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                                __gm__ uint8_t *dpRope, __gm__ uint8_t *dkRope, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER_PREFIX(DETER_SPARSE_TYPE, IS_TND), IS_TND> *__restrict ordTilingData,
                                TPipe *pipeIn, TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn,
                                TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessNormal();
    __aicore__ inline void ProcessDeterV2();
    __aicore__ inline void DeterSync(int64_t loopIdx);
    __aicore__ inline int64_t GetKeyOffsetDeter(int64_t nextIndex);
    template <const bool IS_DK>
    __aicore__ inline void ProcessPostDeter(GlobalTensor<float> dkvWorkSpaceTensor, TQue<QuePosition::VECIN, 1> &inQue,
                                            GlobalTensor<T1> &dkvGmTensor, TQue<QuePosition::VECOUT, 1> &outQue);
    __aicore__ inline int8_t SpecialS2Index(int64_t dkvGmOffset);
    __aicore__ inline void IterateMm3Mm4Mm5(FagRunInfo &runInfo, int64_t nextIndex = -1, int64_t nextS2CvBegin = 0); // dq dk dv
    __aicore__ inline void IterateMm4(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset, int64_t nextIndex, bool isNextS2IdxNoChange);
    __aicore__ inline void IterateMm5(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset, int64_t nextIndex, bool isNextS2IdxNoChange);
    template <const bool IS_DK = false>
    __aicore__ inline void DkvMulsAndCast(FagRunInfo &runInfo, GlobalTensor<float> dkvWorkSpaceTensor,
                                          LocalTensor<T2> &dkvTensor, uint64_t dkvGmOffset,
                                          GlobalTensor<T1> &dkvGmTensor, TQue<QuePosition::VECOUT, 1> &outQue);
    template <const bool IS_DK = false>
    __aicore__ inline void DkvMulsAndCast(FagRunInfo &runInfo, GlobalTensor<float> dkvWorkSpaceTensor,
                                          TQue<QuePosition::VECIN, 1> &inQue, uint64_t dkvGmOffset,
                                          GlobalTensor<T1> &dkvGmTensor, TQue<QuePosition::VECOUT, 1> &outQue);
    using S1S2_TEMPLATE = FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>;
    constexpr static bool IS_DKV_RES_EXCEED_UB = S1S2_TEMPLATE::VECTOR_BASEN / 2 * S1S2_TEMPLATE::HEAD_DIM_ALIGN >
                                                 S1S2_TEMPLATE::VECTOR_BASEM *S1S2_TEMPLATE::VECTOR_BASEN;
    using aType = MatmulType<TPosition::TSCM, CubeFormat::NZ, T1, true, LayoutMode::NONE, true, TPosition::VECOUT>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using cType = MatmulType<GetC2Position(S1S2_TEMPLATE::IS_L0C_REUSE, IS_DKV_RES_EXCEED_UB), CubeFormat::ND, T2>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    constexpr static auto MM3_TILING_CFG_SAMEB =
        GetMatmulApiTiling<aType, bType, cType, biasType>(S1S2_TEMPLATE::MM3_CFG_SAMEB);
    Matmul<aType, bType, cType, biasType, MM3_TILING_CFG_SAMEB, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           Mm3ConstPolicySelector<S1S2_TEMPLATE::IS_TSCM_REUSE, S1S2_TEMPLATE::IS_TSCM_PRELOAD,
                                  S1S2_TEMPLATE::IS_L0C_REUSE>::template Result>
        mm3;

protected:
    uint64_t dkvWorkSpaceOffet{0};
    uint64_t dAlign16 = 0;
    uint64_t dvAlign16 = 0;
    int64_t deterGmOffset = 0;
    int8_t specialS2Index = -1;

    int64_t specialDkGmOffset = 0;
    int64_t specialDvGmOffset = 0;
    int64_t specialHalfS2RealSize = 0;
    int64_t specialFirstHalfS2RealSize = 0;
    bool deterNeedWait = false;
    bool isFirstBlock = true;
};

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *y, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
    __gm__ uint8_t *dpse, __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER_PREFIX(DETER_SPARSE_TYPE, IS_TND), IS_TND> *__restrict ordTilingData, TPipe *pipeIn,
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn, TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn)
{
    S1S2_TEMPLATE::Init(key, value, dx, query, pseShift, dropMask, attenMask, y, softmaxMax, softmaxSum, prefixN,
                        actualSeqQlen, actualSeqKvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, queryRope, keyRope,
                        dq, dk, dv, dpse, dqRope, dkRope, workspace, ordTilingData, pipeIn, dsScmIn, pScmIn);

    dkvWorkSpaceOffet = this->cBlockIdx * S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN;
    dAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSize);
    dvAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSizeV);

    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        this->deterGm.SetGlobalBuffer((__gm__ float *)workspace + this->tilingData->postTilingData.deterGmOffset / sizeof(T2));
        deterGmOffset = this->cBlockIdx * S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN * NUM_TWO; 
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        ProcessDeterV2();
    } else {
        ProcessNormal();
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
    FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessDeterV2()
{
    InitOutput<float>(this->deterGm[deterGmOffset + this->vSubBlockIdx * S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN], S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN, 0);
    int64_t loopMax = this->CalDeterMaxLoopNum();
    int64_t blockIdx = 0;
    int64_t taskId = 0;
    int64_t nextValidLoopIdx;
    int64_t nextblockIdx;

    dkvWorkSpaceOffet = this->cBlockIdx * S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN;
    dAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSize);
    dvAlign16 = AlignTo16(this->constInfo.commonConstInfo.dSizeV);

    FagRunInfo runInfos[2];  // for ping pongs
    this->CalDeterIndex(0, loopMax, nextValidLoopIdx, nextblockIdx, this->coordinateInfos[taskId & 1], runInfos[taskId & 1]);
 
    for (int64_t loopIdx = 0; loopIdx < loopMax + 1; loopIdx++) {
        if (loopIdx >= nextValidLoopIdx) {
            blockIdx = nextblockIdx;
            this->CalDeterIndex(loopIdx + 1, loopMax, nextValidLoopIdx, nextblockIdx, this->coordinateInfos[(taskId + 1) & 1], runInfos[(taskId + 1) & 1]);
        } else {
            blockIdx = -1;
        }
 
        bool isValidBlock = (blockIdx >= 0);
        isValidBlock = isValidBlock && this->IsValidDeterForTnd(runInfos[taskId & 1], blockIdx, this->coordinateInfos[taskId & 1]);
        nextblockIdx = nextblockIdx < 0 ? nextblockIdx : ((taskId + 1) & 1);
        
        if (!(runInfos[(taskId + 1) & 1].completed)) {
            this->ProcessSoftmaxGrad(runInfos[(taskId + 1) & 1]);  // softmaxGrad
            this->WaitMm1Mm2Result();
        }
        if (isValidBlock) {
            this->SetRunInfoDeterForTND(runInfos[taskId & 1], taskId, blockIdx, this->coordinateInfos[taskId & 1]);
            this->IterateMm1Mm2(runInfos[taskId & 1], nextblockIdx);
            CopyInMaxSum<T2, S1S2_TEMPLATE::VECTOR_BASEM>(
                this->constInfo, runInfos[taskId & 1], this->maxSumQue[taskId & 1], this->softmaxMaxGm, this->softmaxSumGm);
            runInfos[taskId & 1].completed = false;
        }
 
        if (!(runInfos[(taskId + 1) & 1].completed)) {
            this->isLastLoop = blockIdx < 0 && nextblockIdx < 0;
            this->ProcessReCompute(runInfos[(taskId + 1) & 1]);
            IterateMm3Mm4Mm5(runInfos[(taskId + 1) & 1], blockIdx < 0 ? nextblockIdx : blockIdx, GetKeyOffsetDeter((taskId & 1)));
            runInfos[(taskId + 1) & 1].completed = true;
        }
 
        DeterSync(loopIdx);
 
        if (isValidBlock) {
            taskId++;
        }
    }

    if (deterNeedWait) {
        mm3.WaitIterateAll();
        mm3.WaitIterateAll();
    }
    SyncAll<>();

    ProcessPostDeter<true>(this->deterGm[deterGmOffset], this->attenMaskOrYInQue, this->dkGm, this->dSOutQue);
    ProcessPostDeter<false>(this->deterGm[deterGmOffset + S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN], this->attenMaskOrYInQue, this->dvGm, this->pOutQue);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::DeterSync(int64_t loopIdx) {
    if (loopIdx == 0) {
        return;
    }

    // 此处复用dqIsNeedDeter和dkDvIsNeedDeter装载需要同步的轮次范围
    for (int8_t i = 0; i < MAX_CUBE_CORE_NUM; i++) {
        if (this->tilingData->s1s2BNGS1S2SplitCoreParams.dkDvIsNeedDeter[i] == 0) {
            return;
        }
        if (static_cast<uint64_t>(loopIdx) >= this->tilingData->s1s2BNGS1S2SplitCoreParams.dqIsNeedDeter[i] && static_cast<uint64_t>(loopIdx) <= this->tilingData->s1s2BNGS1S2SplitCoreParams.dkDvIsNeedDeter[i]) {
            SyncAll<true, syncAllConfigMte3ToMte3>();
            return;
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetKeyOffsetDeter(int64_t nextIndex) {
        if constexpr (!IS_TND || !IS_DETER_NEW(DETER_SPARSE_TYPE)) {
            return -1;
        }

        if (nextIndex == -1) {
            return -1;
        }

        int64_t nextN2oIdx = 0;
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t s2Offset = 0;

        int64_t bIdx = this->coordinateInfos[nextIndex].batchId;
        int64_t seqKvLenPrefix = bIdx == 0 ? 0 : ((__gm__ int64_t *)this->actualSeqKvlenAddr)[bIdx - 1];
        bOffset = seqKvLenPrefix * this->constInfo.commonConstInfo.n2D;

        nextN2oIdx = this->coordinateInfos[nextIndex].n1Idx / this->constInfo.commonConstInfo.gSize;
        n2Offset = nextN2oIdx * this->constInfo.commonConstInfo.dSize;
        s2Offset = this->coordinateInfos[nextIndex].s2Idx * S1S2_TEMPLATE::CUBE_BASEN * this->constInfo.commonConstInfo.n2D;
        return bOffset + n2Offset + s2Offset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessNormal()
{
    if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
        return;
    }

    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for ping pong

    int64_t nextValidBlockInnerIdx = 0;
    int64_t blockInnerIdx = 0;
    int64_t dqkvBlockInnerIdx = 0;
    nextValidBlockInnerIdx =
        this->GetNextValidIdx(runInfos[0], this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx]);
    blockInnerIdx = nextValidBlockInnerIdx;
    while (blockInnerIdx < this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] + 1) {
        this->isLastLoop = (blockInnerIdx == -1);
        dqkvBlockInnerIdx = blockInnerIdx; // save for dq dk dv next valid block index
        if (taskId > 0) {
            this->ProcessSoftmaxGrad(runInfos[(taskId + 1) & 1]); // softmaxGrad
            this->WaitMm1Mm2Result();
        }
        if (!this->isLastLoop) {
            this->SetRunInfo(runInfos[taskId & 1], taskId, blockInnerIdx);
            blockInnerIdx++;
            // get mm1 mm2 next valid block index and next s2 begin end
            nextValidBlockInnerIdx = this->GetNextValidIdx(runInfos[(taskId + 1) & 1], blockInnerIdx);
            this->IterateMm1Mm2(runInfos[taskId & 1], nextValidBlockInnerIdx);
            CopyInMaxSum<T2, S1S2_TEMPLATE::VECTOR_BASEM>(this->constInfo, runInfos[taskId & 1],
                                                          this->maxSumQue[taskId & 1], this->softmaxMaxGm,
                                                          this->softmaxSumGm);
        }
        if (taskId > 0) {
            this->ProcessReCompute(runInfos[(taskId + 1) & 1]);
            this->IterateMm3Mm4Mm5(runInfos[(taskId + 1) & 1], dqkvBlockInnerIdx, this->GetKeyOffset(runInfos[taskId & 1]));
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
FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm3Mm4Mm5(FagRunInfo &runInfo,
                                                                                            int64_t nextIndex, int64_t nextS2CvBegin)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> mm1ResQueInTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<T2>();
    LocalTensor<T2> mm2ResQueInTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<T2>();
    LocalTensor<T2> softmaxGradResTensor = this->softmaxGradResBuf.template Get<T2>();
    LocalTensor<T1> vecOutBuffer = this->dSOutQue.template AllocTensor<T1>();
    LocalTensor<uint8_t> selrIndexesTensor;
    if (runInfo.commonRunInfo.s2RealSize > static_cast<int32_t>(S2TemplateType::Aligned64)) {
        BroadcastSubMul<T2, static_cast<int32_t>(S2TemplateType::Aligned128), 0>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor, mm2ResQueInTensor,
                                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
    } else {
        if (this->constInfo.deterConstInfo.noNeedDeter) {
            BroadcastSubMul<T2, static_cast<int32_t>(S2TemplateType::Aligned64), 0>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor, mm2ResQueInTensor,
                                       runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        } else { // 64~128的脏数据需要清零，避免后面的mm有脏数据参与计算
            BroadcastSubMul<T2, static_cast<int32_t>(S2TemplateType::Aligned64), IS_DETER_OLD(DETER_SPARSE_TYPE)>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor,
                                              mm2ResQueInTensor, runInfo.commonRunInfo.halfS1RealSize,
                                              runInfo.commonRunInfo.s2RealSize);
        }
    }

    // input type fp32, no post, mov muls here
    if constexpr (IsSameType<T1, float>::value) {
        Muls(mm1ResQueInTensor, mm1ResQueInTensor, this->constInfo.scaleValue,
             S1S2_TEMPLATE::VECTOR_BASEM * S1S2_TEMPLATE::VECTOR_BASEN);
    }
    CastTransdataDeconflict<T1, T2, S1S2_TEMPLATE::VECTOR_BASEN>(vecOutBuffer, mm1ResQueInTensor, selrIndexesTensor, 
                                                                 S1S2_TEMPLATE::VECTOR_BASEM);
    this->dSOutQue.template EnQue(vecOutBuffer);
    this->dSOutQue.template DeQue<T1>();
    int64_t queryGmOffset = this->GetQueryOffset(runInfo);
    int64_t keyGmOffset = this->GetKeyOffset(runInfo);
    int64_t dxGmOffset = queryGmOffset;
    int64_t valueGmOffset = keyGmOffset;
    if constexpr (IS_D_NO_EQUAL) {
        dxGmOffset = this->GetDxOffset(runInfo);
        valueGmOffset = this->GetValueOffset(runInfo);
    }
    specialS2Index = SpecialS2Index(keyGmOffset);
    bool isNextS2IdxNoChange = (nextIndex != -1) && (nextS2CvBegin == keyGmOffset);

    ///////////////////////////////////////////////////////////////
    // Matmal3 dq
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    ///////////////////////////////////////////////////////////////
    this->IterateMm3(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset);
    this->dSOutQue.FreeTensor(vecOutBuffer);
    ///////////////////////////////////////////////////////////////
    // Matmal4 dk
    // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
    ///////////////////////////////////////////////////////////////
    IterateMm4(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset, nextIndex, isNextS2IdxNoChange);
    ///////////////////////////////////////////////////////////////
    // VF5: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOutBuffer1 = this->pOutQue.template AllocTensor<T1>();
    CalculateDropout<T2, IS_DROP, S1S2_TEMPLATE::VECTOR_BASEN>(this->constInfo, runInfo, this->dropInfo,
                                                               mm2ResQueInTensor, mm2ResQueInTensor, this->dropMaskBuf);
    CastTransdataDeconflict<T1, T2, S1S2_TEMPLATE::VECTOR_BASEN>(vecOutBuffer1, mm2ResQueInTensor, selrIndexesTensor, 
                                                                 S1S2_TEMPLATE::VECTOR_BASEM);
    this->pOutQue.EnQue(vecOutBuffer1);
    this->pOutQue.template DeQue<T1>();

    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    // left [B, N2, G, S1, S2] right [B, N2, G, S1, D] output [B, N2, 1, S2, D]
    ///////////////////////////////////////////////////////////////
    IterateMm5(runInfo, vecOutBuffer1, dxGmOffset, valueGmOffset, nextIndex, isNextS2IdxNoChange);
    this->pOutQue.FreeTensor(vecOutBuffer1);
}

FAG_FUNCTION_TEMPLATE
template <const bool IS_DK>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::DkvMulsAndCast(
    FagRunInfo &runInfo, GlobalTensor<float> dkvWorkSpaceTensor, TQue<QuePosition::VECIN, 1> &inQue,
    uint64_t dkvGmOffset, GlobalTensor<T1> &dkvGmTensor, TQue<QuePosition::VECOUT, 1> &outQue)
{
    uint32_t dSize = this->constInfo.commonConstInfo.dSize;
    uint32_t curDAlign = dAlign16;
    if constexpr (!IS_DK && IS_D_NO_EQUAL) {
        dSize = this->constInfo.commonConstInfo.dSizeV;
        curDAlign = dvAlign16;
    }

    uint32_t maxLoopSize = S1S2_TEMPLATE::VECTOR_BASEM * S1S2_TEMPLATE::VECTOR_BASEN / curDAlign; 
    uint32_t loopNum = Ceil<uint32_t>(runInfo.halfS2RealSize, maxLoopSize);
    if (loopNum == 0) {
        return;
    }

    uint32_t loopSize = Ceil<uint32_t>(runInfo.halfS2RealSize, loopNum);
    uint32_t tailLoopSize = runInfo.halfS2RealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.srcStride = 0;
    if constexpr (IS_TND) {
        intriParamsOut.dstStride = static_cast<uint32_t>((this->constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(T1));
        dkvGmOffset += this->vSubBlockIdx * runInfo.firstHalfS2RealSize * dSize * this->constInfo.commonConstInfo.n2G;
    } else {
        if (this->constInfo.commonConstInfo.layoutType == BNGSD) {
            intriParamsOut.dstStride = 0;
            dkvGmOffset += this->vSubBlockIdx * runInfo.firstHalfS2RealSize * dSize;
        } else if (this->constInfo.commonConstInfo.layoutType == SBNGD) {
            intriParamsOut.dstStride = static_cast<uint32_t>((this->constInfo.bSize * this->constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(T1));
            dkvGmOffset += this->vSubBlockIdx * runInfo.firstHalfS2RealSize * this->constInfo.commonConstInfo.n2G * this->constInfo.bSize * dSize;
        } else if (this->constInfo.commonConstInfo.layoutType == BSNGD) {
            intriParamsOut.dstStride = static_cast<uint32_t>((this->constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(T1));
            dkvGmOffset += this->vSubBlockIdx * runInfo.firstHalfS2RealSize * this->constInfo.commonConstInfo.n2G * dSize;
        }
    }

    uint32_t data_size = curLoopSize * curDAlign;
    for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
            data_size = curLoopSize * curDAlign;
        }

        LocalTensor<T2> dkvTensor = inQue.AllocTensor<T2>();
        DataCopy(dkvTensor,
                 dkvWorkSpaceTensor[this->vSubBlockIdx * runInfo.firstHalfS2RealSize * curDAlign + loopIdx * loopSize * curDAlign],
                 data_size);
        
        inQue.EnQue(dkvTensor);
        inQue.DeQue();
        if constexpr (IS_DK) {
            Muls(dkvTensor, dkvTensor, this->constInfo.scaleValue, data_size);
        }
        LocalTensor<T1> dkvCastTensor = outQue.template AllocTensor<T1>();
        Cast(dkvCastTensor, dkvTensor, RoundMode::CAST_ROUND, data_size);
        inQue.FreeTensor(dkvTensor);
        outQue.EnQue(dkvCastTensor);
        outQue.template DeQue<T1>();

        intriParamsOut.blockCount = curLoopSize;
        intriParamsOut.blockLen = dSize * sizeof(T1);

        DataCopyPad(dkvGmTensor[dkvGmOffset], dkvCastTensor, intriParamsOut);
        outQue.FreeTensor(dkvCastTensor);

        if constexpr (IS_TND) {
            dkvGmOffset += loopSize * dSize * this->constInfo.commonConstInfo.n2G;
        } else {
            if (this->constInfo.commonConstInfo.layoutType == BNGSD) {
                dkvGmOffset += loopSize * dSize;
            } else if (this->constInfo.commonConstInfo.layoutType == SBNGD) {
                dkvGmOffset += loopSize * this->constInfo.commonConstInfo.n2G * this->constInfo.bSize * dSize;
            } else if (this->constInfo.commonConstInfo.layoutType == BSNGD) {
                dkvGmOffset += loopSize * this->constInfo.commonConstInfo.n2G * dSize;
            }
        }
    }
}

FAG_FUNCTION_TEMPLATE
template <const bool IS_DK>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::DkvMulsAndCast(
    FagRunInfo &runInfo, GlobalTensor<float> dkvWorkSpaceTensor, LocalTensor<T2> &dkvTensor, uint64_t dkvGmOffset,
    GlobalTensor<T1> &dkvGmTensor, TQue<QuePosition::VECOUT, 1> &outQue)
{
    uint32_t dSize = this->constInfo.commonConstInfo.dSize;
    uint32_t curDAlign = dAlign16;
    if constexpr (!IS_DK && IS_D_NO_EQUAL) {
        dSize = this->constInfo.commonConstInfo.dSizeV;
        curDAlign = dvAlign16;
    }
    DataCopyExtParams intriParamsOut;
    intriParamsOut.blockCount = runInfo.halfS2RealSize;
    intriParamsOut.blockLen = dSize * sizeof(T1);
    intriParamsOut.srcStride = 0;
    uint32_t data_size = runInfo.halfS2RealSize * curDAlign;
    if constexpr (IS_TND) {
        intriParamsOut.dstStride = static_cast<uint32_t>((this->constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(T1));
        dkvGmOffset += this->vSubBlockIdx * dSize * runInfo.firstHalfS2RealSize;
    } else {
        if (this->constInfo.commonConstInfo.layoutType == BNGSD) {
            dkvGmOffset += this->vSubBlockIdx * runInfo.firstHalfS2RealSize * dSize;
        } else if (this->constInfo.commonConstInfo.layoutType == SBNGD) {
            dkvGmOffset += this->vSubBlockIdx * runInfo.firstHalfS2RealSize * this->constInfo.commonConstInfo.n2G * this->constInfo.bSize * dSize;
        } else if (this->constInfo.commonConstInfo.layoutType == BSNGD) {
            dkvGmOffset += this->vSubBlockIdx * runInfo.firstHalfS2RealSize * this->constInfo.commonConstInfo.n2G * dSize;
        }
    }
    if constexpr (IS_DK) {
        Muls(dkvTensor, dkvTensor, this->constInfo.scaleValue, data_size);
    }

    LocalTensor<T1> dkvCastTensor = outQue.template AllocTensor<T1>();
    Cast(dkvCastTensor, dkvTensor, RoundMode::CAST_ROUND, data_size);
    outQue.EnQue(dkvCastTensor);
    outQue.template DeQue<T1>();

    DataCopyPad(dkvGmTensor[dkvGmOffset], dkvCastTensor, intriParamsOut);
    outQue.FreeTensor(dkvCastTensor);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm4(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset,
    int64_t nextIndex, bool isNextS2IdxNoChange)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(3, runInfo.qDxPingPongIdx); // 3 means the 3th mm
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        this->GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.n2GD, actualS1Len,
                        actualS1Len, dAlign16);
    } else {
        if (this->constInfo.commonConstInfo.layoutType == BNGSD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.dSize,
                            this->constInfo.commonConstInfo.s1Size, this->constInfo.commonConstInfo.s1Size, dAlign16);
        } else if (this->constInfo.commonConstInfo.layoutType == SBNGD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.bN2GD,
                            this->constInfo.commonConstInfo.s1Size, this->constInfo.commonConstInfo.s1Size,
                            dAlign16);
        } else if (this->constInfo.commonConstInfo.layoutType == BSNGD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.n2GD,
                            this->constInfo.commonConstInfo.s1Size, this->constInfo.commonConstInfo.s1Size,
                            dAlign16);
        }
    }
    LocalTensor<T1> dsScmTensordq = this->dsScm.template AllocTensor<T1>();
    this->dsScm.EnQue(dsScmTensordq);
    this->dsScm.template DeQue<T1>();
    if constexpr (HAS_TAIL) {
        mm3.SetTail(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.dSize,
                    runInfo.commonRunInfo.s1RealSize);
    }
    mm3.SetTensorA(dsScmTensordq, true);
    mm3.SetTensorB(this->queryGm[dxOrQueryGmOffset]); // sameB
    if constexpr (!S1S2_TEMPLATE::IS_L0C_REUSE || IS_DKV_RES_EXCEED_UB) {
        mm3.SetSelfDefineData(flag);
        // nextS2Idx change indicates the basic block corresponding to the final result of dkv has been fully completed.
        if (specialS2Index != -1) {
            bool needAtoimc = !isFirstBlock;
            mm3.template IterateAll<false>(this->deterGm[specialS2Index * S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN * NUM_TWO], needAtoimc, false, nextIndex == -1);
        } else if (isNextS2IdxNoChange) {
            mm3.template IterateAll<false>(this->dkWorkSpaceGm[dkvWorkSpaceOffet], runInfo.isS2IdxNoChange, false,
                                           false);
        } else {
            mm3.template IterateAll<false>(this->dkWorkSpaceGm[dkvWorkSpaceOffet], runInfo.isS2IdxNoChange, false, true);
            mm3.WaitIterateAll();
            DkvMulsAndCast<true>(runInfo, this->dkWorkSpaceGm[dkvWorkSpaceOffet],
                                 this->attenMaskOrYInQue, keyOrValueGmOffset, this->dkGm,
                                 this->dSOutQue);
        }
    } else {
        flag.nextMorN = S1S2_TEMPLATE::L0C_BUF_NUM - DK_DV_L0C_BUF_NUM; // for dk l0c buffer idx
        mm3.SetSelfDefineData(flag);
        LocalTensor<T2> dkResTensor = this->attenMaskOrYInQue.template AllocTensor<T2>();
        if (nextIndex != -1) {
            mm3.template Iterate<true>(runInfo.isS2IdxNoChange);
            // 当前s2列的最后一个有效基本块
            if (!isNextS2IdxNoChange) {
                // BN2分核时, 每个s2*d都在l0c累加, 不需要额外开workspace, 且无需atomic add
                mm3.template GetTensorC<true>(dkResTensor, false, false);
                DkvMulsAndCast<true>(runInfo, this->dkWorkSpaceGm, dkResTensor, keyOrValueGmOffset, this->dkGm,
                                     this->dSOutQue);
            }
        } else {
            mm3.template Iterate<true>(runInfo.isS2IdxNoChange);
            mm3.template GetTensorC<true>(dkResTensor, false, false);
            DkvMulsAndCast<true>(runInfo, this->dkWorkSpaceGm, dkResTensor, keyOrValueGmOffset, this->dkGm,
                                 this->dSOutQue);
        }
        this->attenMaskOrYInQue.FreeTensor(dkResTensor);
    }
    mm3.End();
    this->dsScm.FreeTensor(dsScmTensordq);

    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        if (specialS2Index != -1) {
            deterNeedWait = nextIndex == -1;
        }
        if (this->cBlockIdx == specialS2Index) {
            specialDkGmOffset = keyOrValueGmOffset;
            specialHalfS2RealSize = runInfo.halfS2RealSize;
            specialFirstHalfS2RealSize = runInfo.firstHalfS2RealSize;
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int8_t FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::SpecialS2Index(int64_t dkvGmOffset) {
    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        for (int8_t i = 0; i < MAX_CUBE_CORE_NUM; i++) {
            if (this->tilingData->deterParam.deterPrefix2[i] == dkvGmOffset) {
                return i;
            }
        }
    }
    return -1;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm5(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer1, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset,
    int64_t nextIndex, bool isNextS2IdxNoChange)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(4, runInfo.qDxPingPongIdx); // 4 means the 4th mm
    if constexpr (IS_D_NO_EQUAL) {
        if constexpr (IS_TND) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            this->GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.n2GDv, actualS1Len,
                            actualS1Len, dvAlign16);
        } else {
            if (this->constInfo.commonConstInfo.layoutType == BNGSD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.dSizeV,
                                this->constInfo.commonConstInfo.s1Size, this->constInfo.commonConstInfo.s1Size, dvAlign16);
            } else if (this->constInfo.commonConstInfo.layoutType == SBNGD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.bN2GDv,
                                this->constInfo.commonConstInfo.s1Size, this->constInfo.commonConstInfo.s1Size,
                                dvAlign16);
            } else if (this->constInfo.commonConstInfo.layoutType == BSNGD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.n2GDv,
                                this->constInfo.commonConstInfo.s1Size, this->constInfo.commonConstInfo.s1Size,
                                dvAlign16);
            }
        }
    }
    LocalTensor<T1> pScmTensor = this->pScm.template AllocTensor<T1>();
    this->CopyUB2L1(runInfo, pScmTensor, vecOutBuffer1);
    this->pOutQue.FreeTensor(vecOutBuffer1);
    this->pScm.EnQue(pScmTensor);
    this->pScm.template DeQue<T1>();
    if constexpr (HAS_TAIL && IS_D_NO_EQUAL) {
        mm3.SetTail(runInfo.commonRunInfo.s2RealSize, this->constInfo.commonConstInfo.dSizeV,
                    runInfo.commonRunInfo.s1RealSize);
    }
    mm3.SetTensorA(pScmTensor, true);
    mm3.SetTensorB(this->dxGm[dxOrQueryGmOffset]); // sameB
    if constexpr (!S1S2_TEMPLATE::IS_L0C_REUSE || IS_DKV_RES_EXCEED_UB) {
        mm3.SetSelfDefineData(flag);
        // nextS2Idx change indicates the basic block corresponding to the final result of dkv has been fully completed.
        if (specialS2Index != -1) {
            bool needAtoimc = !isFirstBlock;
            mm3.template IterateAll<false>(this->deterGm[specialS2Index * S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN * NUM_TWO + S1S2_TEMPLATE::CUBE_BASEN * S1S2_TEMPLATE::HEAD_DIM_ALIGN], needAtoimc, false, nextIndex == -1);
        } else if (isNextS2IdxNoChange) {
            mm3.template IterateAll<false>(this->dvWorkSpaceGm[dkvWorkSpaceOffet], runInfo.isS2IdxNoChange, false, false);
        } else {
            mm3.template IterateAll<false>(this->dvWorkSpaceGm[dkvWorkSpaceOffet], runInfo.isS2IdxNoChange, false, true);
            mm3.WaitIterateAll();
            DkvMulsAndCast(runInfo, this->dvWorkSpaceGm[dkvWorkSpaceOffet],
                           this->attenMaskOrYInQue, keyOrValueGmOffset, this->dvGm,
                           this->pOutQue);
        }
    } else {
        flag.nextMorN = S1S2_TEMPLATE::L0C_BUF_NUM - 1; // for dv l0c buffer idx
        mm3.SetSelfDefineData(flag);
        LocalTensor<T2> dvResTensor = this->attenMaskOrYInQue.template AllocTensor<T2>();
        if (nextIndex != -1) {
            mm3.template Iterate<true>(runInfo.isS2IdxNoChange);
            // 当前s2列的最后一个有效基本块
            if (!isNextS2IdxNoChange) {
                mm3.template GetTensorC<true>(dvResTensor, false, false);
                DkvMulsAndCast(runInfo, this->dvWorkSpaceGm, dvResTensor, keyOrValueGmOffset, this->dvGm,
                               this->pOutQue);
            }
        } else {
            mm3.template Iterate<true>(runInfo.isS2IdxNoChange);
            mm3.template GetTensorC<true>(dvResTensor, false, false);
            DkvMulsAndCast(runInfo, this->dvWorkSpaceGm, dvResTensor, keyOrValueGmOffset, this->dvGm, this->pOutQue);
        }
        this->attenMaskOrYInQue.FreeTensor(dvResTensor);
    }
    mm3.End();
    this->pScm.FreeTensor(pScmTensor);

    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        isFirstBlock = false;
        if (this->cBlockIdx == specialS2Index) {
            specialDvGmOffset = keyOrValueGmOffset;
        }
    }
}

FAG_FUNCTION_TEMPLATE
template <const bool IS_DK>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessPostDeter(GlobalTensor<float> dkvWorkSpaceTensor, TQue<QuePosition::VECIN, 1> &inQue,
    GlobalTensor<T1> &dkvGmTensor, TQue<QuePosition::VECOUT, 1> &outQue)
{
    if (specialHalfS2RealSize == 0) {
        return;
    }
    uint32_t dSize = this->constInfo.commonConstInfo.dSize;
    uint32_t curDAlign = dAlign16;
    int64_t dkvGmOffset = IS_DK ? specialDkGmOffset : specialDvGmOffset;
    
    if constexpr (!IS_DK && IS_D_NO_EQUAL) {
        dSize = this->constInfo.commonConstInfo.dSizeV;
        curDAlign = dvAlign16;
    }

    uint32_t maxLoopSize = S1S2_TEMPLATE::VECTOR_BASEM * S1S2_TEMPLATE::VECTOR_BASEN / curDAlign; 
    uint32_t loopNum = Ceil<uint32_t>(specialHalfS2RealSize, maxLoopSize);
    if (loopNum == 0) {
        return;
    }

    uint32_t loopSize = Ceil<uint32_t>(specialHalfS2RealSize, loopNum);
    uint32_t tailLoopSize = specialHalfS2RealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.srcStride = 0;
    intriParamsOut.dstStride = static_cast<uint32_t>((this->constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(T1));
    dkvGmOffset += this->vSubBlockIdx * specialFirstHalfS2RealSize * dSize * this->constInfo.commonConstInfo.n2G;

    uint32_t data_size = curLoopSize * curDAlign;
    for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
            data_size = curLoopSize * curDAlign;
        }

        LocalTensor<T2> dkvTensor = inQue.AllocTensor<T2>();
        DataCopy(dkvTensor, dkvWorkSpaceTensor[this->vSubBlockIdx * specialFirstHalfS2RealSize * curDAlign + loopIdx * loopSize * curDAlign],
                 data_size);
        
        inQue.EnQue(dkvTensor);
        inQue.DeQue();
        if constexpr (IS_DK) {
            Muls(dkvTensor, dkvTensor, this->constInfo.scaleValue, data_size);
        }
        LocalTensor<T1> dkvCastTensor = outQue.template AllocTensor<T1>();
        Cast(dkvCastTensor, dkvTensor, RoundMode::CAST_ROUND, data_size);
        inQue.FreeTensor(dkvTensor);
        outQue.EnQue(dkvCastTensor);
        outQue.template DeQue<T1>();

        intriParamsOut.blockCount = curLoopSize;
        intriParamsOut.blockLen = dSize * sizeof(T1);

        DataCopyPad(dkvGmTensor[dkvGmOffset], dkvCastTensor, intriParamsOut);
        outQue.FreeTensor(dkvCastTensor);
        dkvGmOffset += loopSize * dSize * this->constInfo.commonConstInfo.n2G;
    }
}


#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2S2_H_