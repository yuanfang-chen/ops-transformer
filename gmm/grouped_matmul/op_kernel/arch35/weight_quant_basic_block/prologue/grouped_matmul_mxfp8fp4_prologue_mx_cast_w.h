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
 * \file grouped_matmul_mxfp8fp4_prologue_mx_cast_w.h
 * \brief
 */
#ifndef GROUPED_MATMUL_MXFP8FP4_PROLOGUE_MX_CAST_W_H
#define GROUPED_MATMUL_MXFP8FP4_PROLOGUE_MX_CAST_W_H

#include "../block/basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "tool.h"
#include "weight_quant_vec_compute.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TPosition;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class WeightQuantMatmulBasicBlockAiv {
public:
    __aicore__ inline WeightQuantMatmulBasicBlockAiv() = delete;
    __aicore__ inline WeightQuantMatmulBasicBlockAiv(bool hasBias, uint64_t aPrefetchSize,
                                                     const TCubeTiling *__restrict matmulTiling);
    __aicore__ inline void operator()(__gm__ wType *weight, __gm__ biasType *bias, const bool weightL2Cacheable,
                                      uint64_t kSize, uint64_t nL1Size, uint64_t nOffset, uint64_t kbL1Size,
                                      uint64_t nAlign);
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);
    __aicore__ inline void End();

protected:
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAicToAiv();
    __aicore__ inline void ComputeBasicBlockAivNdKnNzNk(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void mxBiasSetParamAndGmtoUb(const BasicBlockOffsetParam &offsetParam,
                                                   L1ConsumeConfig &l1ConsumeConfig, UbConsumeConfig &ubConsumeConfig,
                                                   const uint64_t kMte2Offset, const uint64_t mte2RealK);

    BasicBlockLibVectorAntiQuantCompute<xType, wType, antiQuantScaleType, biasType, yType, wqmmConfig, vecConfig>
        vectorCompute_;

    uint64_t cvLoopIdx_ = 0;

    LocalTensor<xType> weightL1_;
    LocalTensor<biasType> biasL1_;
    uint64_t weightL1DbOffset_;
    uint64_t biasL1DbOffset_;
    bool hasBias_;
};

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                 biasType, yType, wqmmConfig, vecConfig>::WeightQuantMatmulBasicBlockAiv(
    bool hasBias, uint64_t aPrefetchSize, const TCubeTiling *__restrict matmulTiling)
{
    (void)aPrefetchSize;
    (void)matmulTiling;
    hasBias_ = hasBias;
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;
    static constexpr uint64_t MX_BIAS_L1_SIZE = BIAS_L1_SIZE * GetKBUnit<biasType>() * sizeof(biasType);
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;
    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;
    biasL1_ = LocalTensor<biasType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(biasType));
    if (hasBias_) {
        biasL1DbOffset_ = (l1RemainSize - MX_BIAS_L1_SIZE) / sizeof(biasType);
    }

    if ASCEND_IS_AIV {
        vectorCompute_ = BasicBlockLibVectorAntiQuantCompute<xType, wType, antiQuantScaleType, biasType, yType,
                                                             wqmmConfig, vecConfig>(hasBias_);
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::mxBiasSetParamAndGmtoUb(
    const BasicBlockOffsetParam &offsetParam, L1ConsumeConfig &l1ConsumeConfig, UbConsumeConfig &ubConsumeConfig,
    const uint64_t kMte2Offset, const uint64_t mte2RealK)
{
    uint64_t ubMte2MxBiasNSize = 0;
    uint64_t ubMte2MxBiasNOffset = 0;
    if (hasBias_ && kMte2Offset == 0) {
        ubConsumeConfig.isBiasSingleVector = (mte2RealK == l1ConsumeConfig.l1RealExternalLen);
        if (ubConsumeConfig.isBiasSingleVector) {
            ubMte2MxBiasNSize = offsetParam.nL1Size;
        } else {
            uint64_t mxBiasVec0Nsize =
                offsetParam.nL1Size < MX_BIAS_SINGLE_VECTOR_SIZE ? offsetParam.nL1Size : MX_BIAS_SINGLE_VECTOR_SIZE;
            ubMte2MxBiasNSize = (GetSubBlockIdx() == 0) ? mxBiasVec0Nsize : (offsetParam.nL1Size - mxBiasVec0Nsize);
        }
        ubMte2MxBiasNOffset =
            offsetParam.nOffset + GetSubBlockIdx() * ((ubMte2MxBiasNSize != 0) ? MX_BIAS_SINGLE_VECTOR_SIZE : 0);
    }
    ubConsumeConfig.calcMxBias = (hasBias_ && (kMte2Offset == 0) && (ubMte2MxBiasNSize != 0));
    if (ubConsumeConfig.calcMxBias) {
        ubConsumeConfig.ubMxBiasNsize = ubMte2MxBiasNSize;
        l1ConsumeConfig.l1MxBiasSplitNOffset = GetSubBlockIdx() * MX_BIAS_SINGLE_VECTOR_SIZE;
        vectorCompute_.CopyMxBiasGmToUb(ubMte2MxBiasNSize, ubMte2MxBiasNOffset);
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::ComputeBasicBlockAivNdKnNzNk(
    const BasicBlockOffsetParam &offsetParam)
{
    uint64_t kMte2BaseSize = offsetParam.kbL1Size >> 1;

    UbConsumeConfig ubConsumeConfig;
    L1ConsumeConfig l1ConsumeConfig;
    ubConsumeConfig.l1RequireVfComputeRealN = offsetParam.nL1Size;
    ubConsumeConfig.kWeightLowBitUbOffset = 0;
    ubConsumeConfig.nWeightLowBitUbOffset = 0;
    l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * kMte2BaseSize;

    for (uint64_t kMte2Offset = 0; kMte2Offset < offsetParam.kSize; kMte2Offset += offsetParam.kbL1Size, cvLoopIdx_++) {
        l1ConsumeConfig.l1RealExternalLen = (kMte2Offset + offsetParam.kbL1Size) > offsetParam.kSize ?
                                                offsetParam.kSize - kMte2Offset :
                                                offsetParam.kbL1Size;
        uint64_t mte2RealK = GetSubBlockIdx() == 0 ? min(kMte2BaseSize, l1ConsumeConfig.l1RealExternalLen) :
                             l1ConsumeConfig.l1RealExternalLen > kMte2BaseSize ?
                                 l1ConsumeConfig.l1RealExternalLen - kMte2BaseSize :
                                 0;

        vectorCompute_.WaitVToMTE2();
        mxBiasSetParamAndGmtoUb(offsetParam, l1ConsumeConfig, ubConsumeConfig, kMte2Offset, mte2RealK);
        vectorCompute_.CopyGmToUb(offsetParam.nL1Size, mte2RealK, offsetParam.nOffset,
                                  kMte2Offset + GetSubBlockIdx() * kMte2BaseSize, offsetParam);

        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }
        ubConsumeConfig.l1RequireVfComputeRealK = mte2RealK;
        vectorCompute_.WeightAntiQuantComputeNzNk(ubConsumeConfig, weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_],
                                                  l1ConsumeConfig, biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_]);

        SetAivToAic();
        vectorCompute_.SetVToMTE2();
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::operator()(
    __gm__ wType *weight, __gm__ biasType *bias, const bool weightL2Cacheable, uint64_t kSize, uint64_t nL1Size,
    uint64_t nOffset, uint64_t kbL1Size, uint64_t nAlign)
{
    vectorCompute_.SetGlobalBuffer(weight, bias, weightL2Cacheable);
    BasicBlockOffsetParam offsetParam = {};
    offsetParam.kSize = kSize;
    offsetParam.nL1Size = nL1Size;
    offsetParam.nOffset = nOffset;
    offsetParam.kbL1Size = kbL1Size;
    offsetParam.nAlign = nAlign;
    ComputeBasicBlockAivNdKnNzNk(offsetParam);
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::PrefetchA(
    uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    (void)aPrefetchSize;
    (void)xSizeLimit;
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::End()
{
    if (cvLoopIdx_ > 0) {
        WaitAicToAiv();
    }
    if (cvLoopIdx_ > 1) {
        WaitAicToAiv();
    }
    vectorCompute_.End();
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::SetAivToAic()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAiv<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::WaitAicToAiv()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
}
}  // namespace WeightQuantBatchMatmulV2::Arch35

#endif  // GROUPED_MATMUL_MXFP8FP4_PROLOGUE_MX_CAST_W_H
