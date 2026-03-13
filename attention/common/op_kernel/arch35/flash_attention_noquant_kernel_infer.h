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
 * \file flash_attention_noquant_kernel_infer.h
 * \brief
 */

#ifndef FLASH_ATTENTION_NOQUANT_KERNEL_INFER_H_
#define FLASH_ATTENTION_NOQUANT_KERNEL_INFER_H_
#include "./flash_attention_noquant_kernel_base.h"
#include "./vf/vf_flash_decode.h"
#include "./infer_flash_attention_comm.h"
#include "./infer_flash_attention_kvcache.h"
#include "./infer_flash_attention_sparse.h"
#include "./flash_attention_noquant_block_vec_flashdecode_VF.h"

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType, typename FdBlockType>
class FlashAttentionNoQuantKernelInfer : public FlashAttentionNoQuantKernelBase<FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    static constexpr bool enableSplitCoreBalance = (pseMode == PseTypeEnum::PSE_NONE_TYPE && !enableKVPrefix && !POST_QUANT);    // TODO，输出转置、lse、左padding工作量待评估

    using BaseClass = FlashAttentionNoQuantKernelBase<FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>, CubeBlockType, VecBlockType>;
    /* =====================UB变量==================== */
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, 
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMainLoopBalance();
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void FlashDecode();
    /* =====================GM变量========================== */
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

private:
    FdBlockType fdService;
    static constexpr int64_t fdPrefetchLen = 2;


    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index,
                                                    RunParamStr<isInfer> &runParam);
};

template <typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>::InitUniqueConstInfo()
{
    if constexpr (isFd) {
        this->constInfo.splitKVNum = this->sharedParams.splitKVNum;
        this->constInfo.sInnerLoopSize = CeilDiv(this->constInfo.s2Size, this->constInfo.splitKVNum);
    }
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = this->sharedParams.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = this->sharedParams.isPostQuantBF16;
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.actualSeqLenSize = this->sharedParams.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(this->sharedParams.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(this->sharedParams.isKVHasLeftPadding);
    // pageAttention
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.blockSize = this->sharedParams.blockSize;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
    }

    this->constInfo.transposeLayout = this->sharedParams.transposeLayout;
    if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::NTD_TND)) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    } else if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_BNSD) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_BNSD)) {
        this->constInfo.attentionOutStride = 0;
    }

    // prefix
    if constexpr (enableKVPrefix) {
        this->constInfo.prefixLoopCount = (this->constInfo.actualKVPrefixSize + this->constInfo.s2BaseSize - 1) / this->constInfo.s2BaseSize;
    }
}

template <typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void
FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInfer<CubeBlockType, VecBlockType>::ProcessMainLoopBalance()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if (this->aicIdx >= actualCoreNums) {
        return;
    }

    int64_t taskId = 0;
    RunInfo<isInfer> runInfo[4];
    RunParamStr<isInfer> runParam;

    int32_t bN2Start = this->sharedParams.bnStartIdx;
    int32_t bN2End = this->sharedParams.bnEndIdx;
    int32_t gS1Start = this->sharedParams.gS1StartIdx;
    int32_t gS1End = this->sharedParams.gS1EndIdx;
    int32_t s2Start = this->sharedParams.s2StartIdx;
    int32_t s2End = this->sharedParams.s2EndIdx;

    // 注意这里不等于0是因为，推理的在SetRunInfo中第一次也需要赋值runInfo.s1oIdx，boIdx，n2oIdx，goIdx
    // 训练这些值在multiCoreInnerIdx = 0的时候都是0，两边不统一
    int64_t multiCoreInnerIdx = 1;
    for (int32_t bnIdx = bN2Start; bnIdx <= bN2End; ++bnIdx) {
        bool lastBN = (bnIdx == bN2End);
        runParam.boIdx = bnIdx / this->constInfo.n2Size;
        runParam.n2oIdx = bnIdx / % this->constInfo.n2Size;
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        // 计算s1LoopTimes，未考虑acutalSeq，pre/nextToken
        constexpr int32_t gS1BaseSize = static_cast<int32_t>(s1TemplateType);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            int64_t s1Size = (sIdx == 0) ? actualSeqQlenAddr[0] : actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1];
            runParam.s1LoopTimes = CeilDiv(s1Size * constInfo.gSize, gS1BaseSize);
        } else {
            runParam.s1LoopTimes = CeilDiv(constInfo.gS1, gS1BaseSize);
        }
        int32_t tempGS1End = lastBN ? gS1End : Max(runParam.s1LoopTimes - 1, 0);

        for (int32_t gS1Index = gS1Start; gS1Index <= tempGS1End; ++gS1Index) {
            bool lastGS1 = (gS1Index == tempGS1End);
            // if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            // } else {
            //     s2LoopLimit = 0;
            // }

            int32_t tempS2End, extraLoopTimes;
            if unlikely(lastBN && lastGS1) {
                tempS2End = s2End
                extraLoopTimes = 3;
            } else {
                tempS2End = runParam.s2LoopEndIdx;
                extraLoopTimes = 0;
            }

            for (int32_t s2Idx = s2Start; s2Idx < tempS2End + extraLoopTimes; ++s2Idx) {
                bool notLastThreeLoop = s2Idx < tempS2End;
                bool notLastTwoLoop = s2Idx < tempS2End + 1;
                bool notLast = s2Idx < tempS2End + 2;

                if (notLastThreeLoop) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2Idx, tempS2End - 1, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                    }
                }
                if (taskId > 0 && notLastTwoLoop) {
                    if ASCEND_IS_AIV {
                        auto &runInfo3 = runInfo[(taskId + 3) & 3];
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                            this->constInfo);
                    }
                }
                if (taskId > 1 && notLast) {
                    if ASCEND_IS_AIC {
                        RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        } else {
                            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        }
                    }
                }
                if (taskId > 2) {
                    if ASCEND_IS_AIV {
                        RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                        } else {
                            this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                        }
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
            s2Start = 0;
        }
        gS1Start = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if constexpr (isFd) {
        actualCoreNums = this->sharedParams.bSize * this->constInfo.n2Size *
                         this->constInfo.splitKVNum; // b * n2 * splitkv
    }
    if (this->aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内切分起点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = this->sharedParams.multiCoreInnerLimit;
    if constexpr (!isFd) {
        bnStartIdx = this->sharedParams.bnStartIdx;
        gS1StartIdx = this->sharedParams.multiCoreInnerOffset;
        if (likely((this->sharedParams.coreNum - 1) > this->aicIdx)) {
            bnEndIdx = this->sharedParams.bnEndIdx;
            if (nextGs1Idx != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = this->sharedParams.bSize * this->constInfo.n2Size *
                this->constInfo.headNumRatio;
        }
    } else {
        gS1StartIdx = 0;
        bnStartIdx = 0;
        bnEndIdx = 1;
        s2LoopLimit = 0;
    }
    int64_t taskId = 0;
    bool notLast = true;
    bool isLastBmm1 = false;
    LocalTensor<INPUT_T> scmTensor[2];
    RunInfo<isInfer> runInfo[4];
    RunParamStr<isInfer> runParam;

    if constexpr (isFd) {
        runParam.boIdx = this->aicIdx / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        runParam.n2oIdx = (this->aicIdx / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        bnStartIdx = runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }
    // 注意这里不等于0是因为，推理的在SetRunInfo中第一次也需要赋值runInfo.s1oIdx，boIdx，n2oIdx，goIdx
    // 训练这些值在multiCoreInnerIdx = 0的时候都是0，两边不统一
    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
            runParam.n2oIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, lastBN,
                                                                      nextGs1Idx);
        if constexpr (isFd) {
            if (this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum) >
                runParam.actualS2Size) {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParam.actualS2Size -
                    this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum);
                runParam.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize :
                                        tailSInnerLoopSize;
            }
            runParam.s1LoopTimes = 1;
        }
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }

            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastThreeLoop) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                    }
                }
                if (taskId > 0 && notLastTwoLoop) {
                    if ASCEND_IS_AIV {
                        auto &runInfo3 = runInfo[(taskId + 3) & 3];
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                            this->constInfo);
                    }
                }
                if (taskId > 1 && notLast) {
                    if ASCEND_IS_AIC {
                        RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        } else {
                            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        }
                    }
                }
                if (taskId > 2) {
                    if ASCEND_IS_AIV {
                        RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                        } else {
                            this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                        }
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    if constexpr (enableSplitCoreBalance) {
        ProcessMainLoopBalance();
    } else {
        ProcessMainLoop();
    }
    if constexpr (isFd) {
        if ASCEND_IS_AIV {
            // SyncAll();
            // this->vecBlock.InitFDBuffers(this->constInfo);
            // this->vecBlock.FlashDecodeCompute(this->constInfo, this->keyGm, this->actualSeqKvlenAddr);
            FlashDecode();
        }
    }
}

template <typename CubeBlockType, typename VecBlockType,typename FdBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>::FlashDecode()
{
    SyncAll();
    // this->vecBlock.InitFDBuffers(this->constInfo);
    // this->vecBlock.FlashDecodeCompute(this->constInfo, this->keyGm, this->actualSeqKvlenAddr);
    this->constInfo.bSize = this->sharedParams.bSize;
    fdService.InitParams(this->constInfo);
    if (this->constInfo.actualSeqLenSize != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)this->actualSeqQlenAddr, this->constInfo.actualSeqLenSize);
    }
    if (this->constInfo.actualSeqLenKVSize != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)this->actualSeqKvlenAddr, this->constInfo.actualSeqLenKVSize);
    }
    fdService.InitGlobalTensor(this->vecBlock.softmaxFDMaxGm, this->vecBlock.softmaxFDSumGm, this->vecBlock.accumOutGm, this->vecBlock.attentionOutGm, 
                                this->actualSeqLengthsGmQ, this->actualSeqLengthsGm);
    // fdService.InitSoftmaxLseGm(softmaxLseGm);
    fdService.InitBuffers(this->pipe);
    AscendC::ICachePreLoad(fdPrefetchLen);

    #ifdef ASCENDC_CPU_DEBUG
        const uint32_t *fdBN2Idx = this->tilingData->outerSplitParams.fdRes.fdBN2Idx;
        const uint32_t *fdMIdx = this->tilingData->outerSplitParams.fdRes.fdMIdx;
        const uint32_t *fdS2SplitNum = this->tilingData->outerSplitParams.fdRes.fdS2SplitNum;
        const uint32_t *fdBalanceMSplitNum = this->tilingData->outerSplitParams.fdRes.fdBalanceMSplitNum;
        const uint32_t *fdBalanceMTailSize = this->tilingData->outerSplitParams.fdRes.fdBalanceMTailSize;
        const uint32_t *fdBalanceEndIdx1 = this->tilingData->outerSplitParams.fdRes.fdBalanceEndIdx1;
        const uint32_t *fdBalanceEndIdx2 = this->tilingData->outerSplitParams.fdRes.fdBalanceEndIdx2;
    #else
        uint32_t fdBN2Idx[ARRAY_SIZE(this->tilingData->outerSplitParams.fdRes.fdBN2Idx)];
        uint32_t fdMIdx[ARRAY_SIZE(this->tilingData->outerSplitParams.fdRes.fdMIdx)];
        uint32_t fdS2SplitNum[ARRAY_SIZE(this->tilingData->outerSplitParams.fdRes.fdS2SplitNum)];
        uint32_t fdBalanceMSplitNum[ARRAY_SIZE(this->tilingData->outerSplitParams.fdRes.fdBalanceMSplitNum)];
        uint32_t fdBalanceMTailSize[ARRAY_SIZE(this->tilingData->outerSplitParams.fdRes.fdBalanceMTailSize)];
        uint32_t fdBalanceEndIdx1[ARRAY_SIZE(this->tilingData->outerSplitParams.fdRes.fdBalanceEndIdx1)];
        uint32_t fdBalanceEndIdx2[ARRAY_SIZE(this->tilingData->outerSplitParams.fdRes.fdBalanceEndIdx2)];
        copy_data_align64((uint8_t *)fdBN2Idx, (uint8_t *)(this->tilingData->outerSplitParams.fdRes.fdBN2Idx),
                    sizeof(fdBN2Idx));
        copy_data_align64((uint8_t *)fdMIdx, (uint8_t *)(this->tilingData->outerSplitParams.fdRes.fdMIdx),
                    sizeof(fdMIdx));
        copy_data_align64((uint8_t *)fdS2SplitNum, (uint8_t *)(this->tilingData->outerSplitParams.fdRes.fdS2SplitNum),
                    sizeof(fdS2SplitNum));
        copy_data_align64((uint8_t *)fdBalanceMSplitNum, (uint8_t *)(this->tilingData->outerSplitParams.fdRes.fdBalanceMSplitNum),
                    sizeof(fdBalanceMSplitNum));
        copy_data_align64((uint8_t *)fdBalanceMTailSize,
                    (uint8_t *)(this->tilingData->outerSplitParams.fdRes.fdBalanceMTailSize),
                    sizeof(fdBalanceMTailSize));
        copy_data_align64((uint8_t *)fdBalanceEndIdx1, (uint8_t *)(this->tilingData->outerSplitParams.fdRes.fdBalanceEndIdx1),
                    sizeof(fdBalanceEndIdx1));
        copy_data_align64((uint8_t *)fdBalanceEndIdx2,
                    (uint8_t *)(this->tilingData->outerSplitParams.fdRes.fdBalanceEndIdx2),
                    sizeof(fdBalanceEndIdx2));
    #endif

    FDparams fdParams = {fdBN2Idx, fdMIdx, fdS2SplitNum, fdBalanceMSplitNum, fdBalanceMTailSize, fdBalanceEndIdx1, fdBalanceEndIdx2,
            this->tilingData->outerSplitParams.fdRes.fdUsedVecNum,this->tilingData->outerSplitParams.fdRes.fdBalanceMBaseSize};
    fdService.AllocEventID();
    fdService.InitDecodeParams();
    fdService.FlashDecode(fdParams);
    fdService.FreeEventID();
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType, typename FdBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType, FdBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    constexpr uint64_t fp8QBlockSize = 128U; // 128 is SOuterSize
    constexpr uint64_t fp8KvBlockSize = 256U; // 256 is SInnerSize
    if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        if (runParam.boIdx == 0 || runParam.boIdx == 1) {
            this->s1ScaleNumAcc = runParam.boIdx == 0 ? 0 : CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = runParam.boIdx == 0 ? 0 : CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            this->s1SizeAcc = runParam.boIdx == 0 ? 0 : this->actualSeqQlenAddr[0];
            this->s2SizeAcc = runParam.boIdx == 0 ? 0 : this->actualSeqKvlenAddr[0];
        } else {
            this->s1ScaleNumAcc = CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            for (uint32_t boIdx = 1; boIdx < runParam.boIdx; boIdx++) {
                this->s1ScaleNumAcc += CeilDiv(this->actualSeqQlenAddr[boIdx] - this->actualSeqQlenAddr[boIdx - 1], fp8QBlockSize);
                this->s2ScaleNumAcc += CeilDiv(this->actualSeqKvlenAddr[boIdx] - this->actualSeqKvlenAddr[boIdx - 1], fp8KvBlockSize);
            }
            this->s1SizeAcc = this->actualSeqQlenAddr[runParam.boIdx - 1];
            this->s2SizeAcc = this->actualSeqKvlenAddr[runParam.boIdx - 1];
        }
    }
    // GS1合轴时，g轴信息包含在gS1中；GS1不合轴时，g轴信息包含在bn2g中；
    if (this->constInfo.isGqa) {
        runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}
}
#endif