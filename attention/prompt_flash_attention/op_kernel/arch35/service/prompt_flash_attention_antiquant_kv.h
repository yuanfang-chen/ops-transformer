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
 * \file prompt_flash_attention_antiquant_kv.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_ANTIQUANT_KV_H
#define PROMPT_FLASH_ATTENTION_ANTIQUANT_KV_H

#include "../comm/prompt_flash_attention_comm.h"
#include "../vf/vf_antiquant.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::Nd2NzParams;

template <typename PFAT>
class PromptFlashAttentionAntiQuantKV {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T, PFAT::calcMode>::softmaxType;
    __aicore__ inline PromptFlashAttentionAntiQuantKV() {};
    __aicore__ inline void Process(TSCM<QuePosition::VECIN, 1, 0x4> &keyValueScmQueue,
                                   GlobalTensor<KV_T> &keyValueGm,
                                   GlobalTensor<T> &antiquantScaleGm,
                                   GlobalTensor<T> &antiquantOffsetGm,
                                   TQue<QuePosition::VECIN, 1> &antiquantInputQueue,
                                   TQue<QuePosition::VECOUT, 1> &antiquantOutputQueue,
                                   TQue<QuePosition::VECIN, 1> &antiquantParamQueue,
                                   const TaskParam &params, const ConstParam &constParams,
                                   const PromptFlashAttentionTilingData* __restrict tilingData);

protected:
    __aicore__ inline void ProcessInner(GlobalTensor<KV_T> &keyValueGm,
                                        GlobalTensor<T> &antiquantScaleGm,
                                        GlobalTensor<T> &antiquantOffsetGm,
                                        TQue<QuePosition::VECIN, 1> &antiquantInputQueue,
                                        TQue<QuePosition::VECOUT, 1> &antiquantOutputQueue,
                                        TQue<QuePosition::VECIN, 1> &antiquantParamQueue,
                                        const TaskParam &params, const ConstParam &constParams,
                                        LocalTensor<T> &keyValueScm,
                                        const PromptFlashAttentionTilingData* __restrict tilingData);
    __aicore__ inline void CopyKey(GlobalTensor<KV_T> &keyValueGm, LocalTensor<KV_T> &keyValueSrcUb,
                                   const TaskParam &params, const ConstParam &constParams, int64_t antiLoopIdx, uint32_t step,
                                   int32_t headSize, DataCopyParams kvCopyParam);

    __aicore__ inline void AntiquantProcess(GlobalTensor<KV_T> &keyValueGm,
                                            TQue<QuePosition::VECIN, 1> &antiquantInputQueue,
                                            TQue<QuePosition::VECOUT, 1> &antiquantOutputQueue,
                                            TQue<QuePosition::VECIN, 1> &antiquantParamQueue,
                                            const TaskParam &params, const ConstParam &constParams,
                                            LocalTensor<T> &keyValueScm,
                                            const PromptFlashAttentionTilingData* __restrict tilingData,
                                            int64_t kvComputeSInnerSize, int64_t &keyL1Offset, int64_t antiLoopIdx, uint32_t step,
                                            int32_t headSize, DataCopyParams &kvCopyParam);

protected:
    static constexpr int32_t BLOCK_BYTE_NUM = 32; // 32:size of block
    static constexpr int32_t PFA_BUFFER_SIZE_BYTE_16K = 16384; // 16384:ub size of antiquant
private:
    LocalTensor<T> keyValueScm;
    LocalTensor<T> antiQuantScaleUb;
    LocalTensor<T> antiQuantOffsetUb;
};

template <typename T>
__aicore__ inline T AntiquantAlign(T num, T rnd) {
    return ((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd));
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKV<PFAT>::Process(TSCM<QuePosition::VECIN, 1, 0x4> &keyValueScmQueue,
                                   GlobalTensor<KV_T> &keyValueGm,
                                   GlobalTensor<T> &antiquantScaleGm,
                                   GlobalTensor<T> &antiquantOffsetGm,
                                   TQue<QuePosition::VECIN, 1> &antiquantInputQueue,
                                   TQue<QuePosition::VECOUT, 1> &antiquantOutputQueue,
                                   TQue<QuePosition::VECIN, 1> &antiquantParamQueue,
                                   const TaskParam &params, const ConstParam &constParams,
                                   const PromptFlashAttentionTilingData* __restrict tilingData)
{
    keyValueScm = keyValueScmQueue.template AllocTensor<T>();
    ProcessInner(keyValueGm,
                 antiquantScaleGm,
                 antiquantOffsetGm,
                 antiquantInputQueue,
                 antiquantOutputQueue,
                 antiquantParamQueue,
                 params, constParams,
                 keyValueScm,
                 tilingData);

    keyValueScmQueue.template EnQue(keyValueScm);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKV<PFAT>::CopyKey(GlobalTensor<KV_T> &keyValueGm, LocalTensor<KV_T> &keyValueSrcUb,
                                   const TaskParam &params, const ConstParam &constParams, int64_t antiLoopIdx, uint32_t step,
                                   int32_t headSize, DataCopyParams kvCopyParam)
{
    int64_t antiKeyOffset = 0;
    if constexpr (PFAT::layout == PFALayout::BNSD) {
        antiKeyOffset = params.tensorBOffset + static_cast<int64_t>(antiLoopIdx) * step * headSize; //BNSD
    } else {
        antiKeyOffset = params.tensorBOffset + static_cast<int64_t>(antiLoopIdx) * step * constParams.multiHeadK; // 暂未适配key与value的d不等长场景
    }
    DataCopy(keyValueSrcUb[0], keyValueGm[antiKeyOffset], kvCopyParam);
}

// kv伪量化函数
template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKV<PFAT>::AntiquantProcess(GlobalTensor<KV_T> &keyValueGm,
                                            TQue<QuePosition::VECIN, 1> &antiquantInputQueue,
                                            TQue<QuePosition::VECOUT, 1> &antiquantOutputQueue,
                                            TQue<QuePosition::VECIN, 1> &antiquantParamQueue,
                                            const TaskParam &params, const ConstParam &constParams,
                                            LocalTensor<T> &keyValueScm,
                                            const PromptFlashAttentionTilingData* __restrict tilingData,
                                            int64_t kvComputeSInnerSize, int64_t &keyL1Offset, int64_t antiLoopIdx, uint32_t step,
                                            int32_t headSize, DataCopyParams &kvCopyParam)
{
    LocalTensor<KV_T> keyValueSrcUb = antiquantInputQueue.template AllocTensor<KV_T>();
    kvCopyParam.blockCount = step;
    CopyKey(keyValueGm, keyValueSrcUb, params, constParams, antiLoopIdx, step, headSize, kvCopyParam);
    antiquantInputQueue.template EnQue(keyValueSrcUb);
    antiquantInputQueue.DeQue<KV_T>();

    LocalTensor<T> keyValueDstUb = antiquantOutputQueue.template AllocTensor<T>();

    uint32_t dealRowCount = kvComputeSInnerSize;
    if (!constParams.isAntiquantSymmetric) {
        if (tilingData->promptAttentionBaseParams.alignedHeadSize == DSIZE_CONST_64) {
            PfaAntiquantVF<T, KV_T, DSIZE_CONST_64, true>(keyValueSrcUb, keyValueDstUb, antiQuantOffsetUb, antiQuantScaleUb, dealRowCount);
        } else {
            PfaAntiquantVF<T, KV_T, DSIZE_CONST_128, true>(keyValueSrcUb, keyValueDstUb, antiQuantOffsetUb, antiQuantScaleUb, dealRowCount);
        }
    } else {
        if (tilingData->promptAttentionBaseParams.alignedHeadSize == DSIZE_CONST_64) {
            PfaAntiquantVF<T, KV_T, DSIZE_CONST_64, false>(keyValueSrcUb, keyValueDstUb, antiQuantOffsetUb, antiQuantScaleUb, dealRowCount);
        } else {
            PfaAntiquantVF<T, KV_T, DSIZE_CONST_128, false>(keyValueSrcUb, keyValueDstUb, antiQuantOffsetUb, antiQuantScaleUb, dealRowCount);
        }
    }

    antiquantInputQueue.FreeTensor(keyValueSrcUb);
    antiquantParamQueue.FreeTensor(antiQuantScaleUb);
    antiquantOutputQueue.template EnQue(keyValueDstUb);
    antiquantOutputQueue.DeQue<T>();

    uint16_t elementTypeSize = BLOCK_BYTE_NUM / sizeof(T);
    uint16_t lenBurst = dealRowCount;
    uint16_t nBurst = headSize / elementTypeSize;

    uint16_t srcStride = 1;
    uint16_t dstStep = AntiquantAlign((uint16_t)params.singleProcessSInnerSizeNow, (uint16_t)16); // 16:对齐单位
    uint16_t dstStride = dstStep - dealRowCount;
    
    DataCopyParams intriParamsTemp;
    intriParamsTemp.blockCount = nBurst;
    intriParamsTemp.blockLen = lenBurst;
    intriParamsTemp.dstStride = dstStride; 
    intriParamsTemp.srcStride = srcStride;
    DataCopy(keyValueScm, keyValueDstUb, intriParamsTemp);

    antiquantOutputQueue.FreeTensor(keyValueDstUb);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantKV<PFAT>::ProcessInner(GlobalTensor<KV_T> &keyValueGm,
                                                                           GlobalTensor<T> &antiquantScaleGm,
                                                                           GlobalTensor<T> &antiquantOffsetGm,
                                                                           TQue<QuePosition::VECIN, 1> &antiquantInputQueue,
                                                                           TQue<QuePosition::VECOUT, 1> &antiquantOutputQueue,
                                                                           TQue<QuePosition::VECIN, 1> &antiquantParamQueue,
                                                                           const TaskParam &params, const ConstParam &constParams,
                                                                           LocalTensor<T> &keyValueScm,
                                                                           const PromptFlashAttentionTilingData* __restrict tilingData)
{
    int64_t headSize = tilingData->promptAttentionBaseParams.headSize;
    uint32_t step = PFA_BUFFER_SIZE_BYTE_16K / sizeof(KV_T) / headSize;
    step = (step == 0) ? 1 : step;

    uint32_t kvAntiquantLoopTimes = (params.singleProcessSInnerSizeNow + step - 1) / step;
    uint32_t tailCnt = params.singleProcessSInnerSizeNow - (kvAntiquantLoopTimes - 1) * step;

    int64_t qElementCntPerReg = 256 / sizeof(T); // 256 :单个寄存器大小

    uint32_t scaleLoopCount = qElementCntPerReg / headSize; // d = 64时需要复制2份伪量化参数
    scaleLoopCount = (scaleLoopCount == 0) ? 1 : scaleLoopCount; // d = 512、256时至少拷贝一次

    LocalTensor<T> scaleUb = antiquantParamQueue.template AllocTensor<T>();
    LocalTensor<T> offsetUb = scaleUb[qElementCntPerReg];

    for (uint32_t i = 0; i < scaleLoopCount; i++) {
        DataCopy(scaleUb[i * headSize], antiquantScaleGm[params.batchNOffset / tilingData->promptAttentionBaseParams.headNumRatio * headSize], headSize);

        if (!constParams.isAntiquantSymmetric) {

            DataCopy(offsetUb[i * headSize], antiquantOffsetGm[params.batchNOffset / tilingData->promptAttentionBaseParams.headNumRatio * headSize], headSize);
        }
    }

    antiquantParamQueue.template EnQue(scaleUb);
    antiQuantScaleUb = antiquantParamQueue.DeQue<T>();
    antiQuantOffsetUb = antiQuantScaleUb[qElementCntPerReg];

    DataCopyParams kvCopyParam;
    kvCopyParam.blockLen = headSize / (BLOCK_BYTE_NUM / sizeof(KV_T));
    if constexpr (PFAT::layout == PFALayout::BNSD) {
        kvCopyParam.srcStride = 0;
    } else {
        kvCopyParam.srcStride = (constParams.multiHeadK - headSize) / BLOCK_BYTE_NUM;  // 暂未适配key与value的d不等长场景
    }
    kvCopyParam.dstStride = 0;
    kvCopyParam.blockCount = step;

    int64_t kvComputeSInnerSize = 0;
    int64_t keyL1Offset = 0;

    for (int64_t antiLoopIdx = 0; antiLoopIdx < kvAntiquantLoopTimes; antiLoopIdx++) {
        kvComputeSInnerSize = (antiLoopIdx == kvAntiquantLoopTimes - 1) ? tailCnt : step;
        kvComputeSInnerSize = kvComputeSInnerSize > step ? step : kvComputeSInnerSize;

        AntiquantProcess(keyValueGm, antiquantInputQueue, antiquantOutputQueue, 
                         antiquantParamQueue,
                         params, constParams,
                         keyValueScm,
                         tilingData, 
                         kvComputeSInnerSize, keyL1Offset, antiLoopIdx, step,
                         headSize, kvCopyParam);
        keyL1Offset += step * 16; // 16:offset of nd2nz
    }
}

#endif  // PROMPT_FLASH_ATTENTION_ANTIQUANT_KV_H