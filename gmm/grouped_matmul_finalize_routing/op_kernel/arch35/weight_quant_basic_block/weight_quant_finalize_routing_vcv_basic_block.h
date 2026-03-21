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
 * \file weight_quant_vcv_basic_block.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
#define GROUPED_MATMUL_WEIGHT_QUANT_VCV_BASIC_BLOCK_H

#include "../common/basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "../common/tool.h"
#include "../common/weight_quant_cube_compute.h"
#include "../common/basic_api/weight_quant_basic_api_v1.h"
#include "../common/weight_quant_vcv_basic_block_base.h"
#include "gmm_fr_weight_quant_vec_compute.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

namespace WeightQuantBatchMatmulV2::Arch35 {
#define GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM                                                              \
    template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType,             \
              typename perTokenScaleType, typename biasType, typename yType, typename sharedInputDType, const WqmmConfig &wqmmConfig, \
              const VecAntiQuantConfig &vecConfig>

#define GMM_WQ_VCV_BASIC_BLOCK_CLASS                                                                                \
    WQFRVcvMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, sharedInputDType,\
                                   wqmmConfig, vecConfig>                
GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
class WQFRVcvMatmulBasicBlock {
public:
    __aicore__ inline WQFRVcvMatmulBasicBlock(){};
    __aicore__ inline void Init(bool hasBias, uint64_t antiQuantGroupSize);
    __aicore__ inline void InitAtomicGm(uint64_t initSize, uint64_t sharedInputStartSize, uint64_t sharedInputSize, __gm__ sharedInputDType *shareInputAddr);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
                                            __gm__ scaleType *scale, __gm__ perTokenScaleType *perTokenScale,
                                            __gm__ biasType *bias, __gm__ yType *y, const bool hasBias,
                                            const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                             const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void End(const BasicBlockOffsetParam &curOffsetParam);

protected:
    __aicore__ inline void InitGmZeroWithIterate(uint64_t & yGmStartOffset, uint64_t yGmEndOffset, uint64_t sharedInputStartSize, uint64_t sharedInputSize);
    __aicore__ inline void ComputeBasicBlockAivNzNk(const BasicBlockOffsetParam &curOffsetParam,
                                                    const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void IterateNzNkWithKAiv(uint64_t &kMte2Offset, uint64_t kMte2Limit, uint64_t mte2RealN,
                                               uint64_t nL1Offset, const BasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void IterateNzKnWithKAic(const BasicBlockOffsetParam &curOffsetParam);

    template <const pipe_t pipe>
    __aicore__ inline void SetAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void SetAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    GmmFrVecCompute<xType, wType, antiQuantScaleType, biasType, yType, sharedInputDType, wqmmConfig, vecConfig>
        vecCompute_;
    WeightQuantBatchMatmulV2CubeCompute<xType, int32_t, antiQuantScaleType, perTokenScaleType, int32_t, wqmmConfig,
                                        WqbmmBasicApiV1<xType, biasType, yType, wqmmConfig.aTrans, wqmmConfig.bTrans>>
        cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightL1DbOffset_ = 0;
    uint64_t biasL1DbOffset_ = 0;
    bool hasBias_ = false;

    LocalTensor<xType> weightL1_;
    LocalTensor<biasType> biasL1_;
    LocalTensor<int32_t> ubOutputS32Buffer_;
};

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::Init(bool hasBias, uint64_t antiQuantGroupSize)
{
    hasBias_ = hasBias;
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256; // weight单块大小的标准shape是256*256
    static constexpr uint64_t MX_BIAS_L1_SIZE = BIAS_L1_SIZE * GetKBUnit<biasType>() * sizeof(biasType);
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;
    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;
    biasL1_ = LocalTensor<biasType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(biasType));
    if (hasBias_) {
        biasL1DbOffset_ = (l1RemainSize - MX_BIAS_L1_SIZE) / sizeof(biasType);
        l1RemainSize -= DOUBLE_BUFFER_NUM * MX_BIAS_L1_SIZE;
        l1StartSize += MX_BIAS_L1_SIZE;
    }

    if ASCEND_IS_AIC {
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize, biasL1DbOffset_, biasL1_);
    } else {
        vecCompute_.Init(GetTPipePtr(), hasBias_);
    }
    cvLoopIdx_ = 0;
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::InitAtomicGm(uint64_t initSize, uint64_t sharedInputStartSize, uint64_t sharedInputSize, __gm__ sharedInputDType *shareInputAddr)
{
    // 首4轮的mte3写出
    constexpr uint64_t initZeroBufferSize = GetGmmFRMxA8W4BufferInfo<vecConfig>().highBitDataUbSingleBufferSize / sizeof(float);
    uint64_t yGmOffset = GetBlockIdx() * initZeroBufferSize;
    InitGmZeroWithIterate(yGmOffset, Min(QUADRUPLE_BUFFER_NUM * AscendC::GetBlockNum() * initZeroBufferSize, initSize), sharedInputStartSize, sharedInputSize);

    // shared input mte3写出
    constexpr uint64_t mte2BufferSize =
        WeightQuantBatchMatmulV2::Arch35::GetMxA8W4NzBufferInfo<vecConfig>().weightInputLowBitUbSingleBufferSize / sizeof(float);
    for (uint64_t sharedInputGmOffset = AscendC::GetBlockIdx() * mte2BufferSize; sharedInputGmOffset <= sharedInputSize;
         sharedInputGmOffset += AscendC::GetBlockNum() * mte2BufferSize) {
        uint64_t initSharedInputRealSize = sharedInputGmOffset + mte2BufferSize > sharedInputSize ?
                                               sharedInputSize - sharedInputGmOffset :
                                               mte2BufferSize;
        vecCompute_.WaitVToMTE2();
        vecCompute_.CopyShareInputGmToUb(sharedInputGmOffset, initSharedInputRealSize, shareInputAddr);
        vecCompute_.CopyShareInputUbToGm(sharedInputGmOffset, initSharedInputRealSize);
        vecCompute_.SetVToMTE2();
    }
    // 剩余部分的mte3写出
    InitGmZeroWithIterate(yGmOffset, initSize, sharedInputStartSize, sharedInputSize);
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::InitGmZeroWithIterate(uint64_t & yGmStartOffset, uint64_t yGmEndOffset, uint64_t sharedInputStartSize, uint64_t sharedInputSize){
    constexpr uint64_t initZeroBufferSize = GetGmmFRMxA8W4BufferInfo<vecConfig>().highBitDataUbSingleBufferSize / sizeof(float);
    uint64_t bufferSize = initZeroBufferSize;
    for (uint64_t yGmOffset = yGmStartOffset; yGmOffset <= yGmEndOffset; yGmOffset += AscendC::GetBlockNum() * bufferSize){
        uint64_t initZeroRealSize = yGmOffset + bufferSize > yGmEndOffset ? yGmEndOffset - yGmOffset : bufferSize;
        if (yGmOffset >= sharedInputStartSize) {
            vecCompute_.InitGmToZero(yGmOffset + sharedInputSize, initZeroRealSize);
        } else if (yGmOffset + bufferSize <= sharedInputStartSize) {
            vecCompute_.InitGmToZero(yGmOffset, initZeroRealSize);
        }else {
            uint64_t initZeroFirstTail = sharedInputStartSize - yGmOffset;
            vecCompute_.InitGmToZero(yGmOffset, initZeroFirstTail);
            uint64_t initZeroSecondTail = initZeroRealSize - initZeroFirstTail;
            vecCompute_.InitGmToZero(yGmOffset + sharedInputSize, initZeroSecondTail);
        }
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::UpdateGlobalAddr(
    __gm__ xType *x, __gm__ wType *weight, __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
    __gm__ scaleType *scale, __gm__ perTokenScaleType *perTokenScale, __gm__ biasType *bias, __gm__ yType *y,
    const bool hasBias, const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, y, bias, scale, nullptr, perTokenScale, hasBias);
    } else {
        // For MX A8W4: scale is perChannelScale (float*), perTokenScale is also float*
        vecCompute_.UpdateGlobalAddr(weight, antiquantScale, antiquantOffset, 
                                     nullptr,
                                     nullptr, 
                                     bias, y, weightL2Cacheable);
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                                                       const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        ComputeBasicBlockAivNzNk(curOffsetParam, lastOffsetParam);
    } else {
        IterateNzKnWithKAic(curOffsetParam);
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlockAivNzNk(
    const BasicBlockOffsetParam &curOffsetParam, const BasicBlockOffsetParam &lastOffsetParam)
{
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;
    uint64_t mte2RealN = curOffsetParam.nL1Size;
    uint64_t nL1Offset = curOffsetParam.nOffset;
    
    // Split M dimension between two AIV cores (0 and 1)
    uint64_t antiquantYMSize = curOffsetParam.mL1Size - (curOffsetParam.mL1Size >> 1);
    uint64_t antiquantYMOffset = GetSubBlockIdx() == 0 ? 0 : antiquantYMSize;
    antiquantYMSize = GetSubBlockIdx() == 0 ? antiquantYMSize : (curOffsetParam.mL1Size >> 1);
    
    IterateNzNkWithKAiv(kMte2Offset, Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size), 
                        mte2RealN, nL1Offset, curOffsetParam);
    if (curCvLoopIdx > 0) {
        // y反量化在vec上两core切m
        uint64_t lastBasicBlockMSize = lastOffsetParam.mL1Size - (lastOffsetParam.mL1Size >> 1);
        uint64_t lastBasicBlockMOffset = GetSubBlockIdx() == 0 ? 0 : lastBasicBlockMSize;
        lastBasicBlockMSize = GetSubBlockIdx() == 0 ? lastBasicBlockMSize : (lastOffsetParam.mL1Size >> 1);

        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);
        // MulLogits not needed for weight quant - skipping
        vecCompute_.CopyYUbToGm(lastOffsetParam.nL1Size, lastBasicBlockMSize,
                                reinterpret_cast<__gm__ half *>(lastOffsetParam.yGmAddr), lastOffsetParam,
                                lastBasicBlockMOffset);
    }
    // todo 先不搞preload mte2, 收益有限
    IterateNzNkWithKAiv(kMte2Offset, curOffsetParam.kSize, mte2RealN, nL1Offset, curOffsetParam);
    // CopyRowIndexLogitGmToUb is not needed for finalize routing, skip or leave as placeholder
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzNkWithKAiv(
    uint64_t &kMte2Offset, uint64_t kMte2Limit, uint64_t mte2RealN, uint64_t nL1Offset,
    const BasicBlockOffsetParam &curOffsetParam)
{
    uint64_t kMte2BaseSize = curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM;
    
    for (; kMte2Offset < kMte2Limit; kMte2Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        vecCompute_.WaitVToMTE2();
        
        uint64_t mte2RealK = Min(curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM, 
                                 curOffsetParam.kSize - kMte2Offset - GetSubBlockIdx() * kMte2BaseSize);
        
        vecCompute_.CopyGmToUb(mte2RealN, mte2RealK, nL1Offset,
                               kMte2Offset + GetSubBlockIdx() * kMte2BaseSize, curOffsetParam);
        
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        
        // Prepare consume configs
        UbConsumeConfig ubConsumeConfig;
        ubConsumeConfig.l1RequireVfComputeRealN = curOffsetParam.nL1Size;
        ubConsumeConfig.l1RequireVfComputeRealK = curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM;
        ubConsumeConfig.calcMxBias = false;
        
        L1ConsumeConfig l1ConsumeConfig;
        l1ConsumeConfig.l1SplitTwoVecExternalOffset = 0;
        l1ConsumeConfig.l1RealExternalLen = curOffsetParam.nL1Size;
        l1ConsumeConfig.l1MxBiasSplitNOffset = 0;
        
        vecCompute_.WeightAntiQuantComputeNzNk(ubConsumeConfig, weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_],
                                    l1ConsumeConfig, biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_]);
        
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        vecCompute_.SetVToMTE2();
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzKnWithKAic(const BasicBlockOffsetParam &curOffsetParam)
{
    if (cvLoopIdx_ > 0) {
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        cubeCompute_.GetTensorC(ubOutputS32Buffer_);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }

    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize; 
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize
                                    ? curOffsetParam.kSize - kbL1Offset
                                    : curOffsetParam.kbL1Size;
        cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);
        cubeCompute_.CopyMxScaleGmToL1(curOffsetParam, kbL1Offset, cvLoopIdx_);
        cubeCompute_.WaitMTE1ToMTE2(cvLoopIdx_);
        cubeCompute_.CopyAAndBiasGmToL1(curOffsetParam, kbL1Offset, kbL1RealSize, curOffsetParam.nL1Size, cvLoopIdx_);
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
        cubeCompute_.LaunchMatmul(weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_], kbL1Offset, kbL1RealSize,
                                  curOffsetParam, cvLoopIdx_);  // mte1 mmad fixp流水
        cubeCompute_.SetMTE1ToMTE2(cvLoopIdx_);
        cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curOffsetParam);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}



GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::End(const BasicBlockOffsetParam &curOffsetParam)
{
    if ASCEND_IS_AIC {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(ubOutputS32Buffer_);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync(cvLoopIdx_);
    } else {
        if (cvLoopIdx_ > 0) {
            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            uint64_t lastBasicBlockMSize = curOffsetParam.mL1Size - (curOffsetParam.mL1Size >> 1);
            uint64_t lastBasicBlockMOffset = GetSubBlockIdx() == 0 ? 0 : lastBasicBlockMSize;
            lastBasicBlockMSize = GetSubBlockIdx() == 0 ? lastBasicBlockMSize : (curOffsetParam.mL1Size >> 1);
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);
            // MulLogits not needed for weight quant - skipping
            vecCompute_.CopyYUbToGm(curOffsetParam.nL1Size, lastBasicBlockMSize,
                                    reinterpret_cast<__gm__ half *>(curOffsetParam.yGmAddr), curOffsetParam,
                                    lastBasicBlockMOffset);
            
        }
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        vecCompute_.End();
    }
}
}  // namespace WeightQuantBatchMatmulV2::Arch35

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
