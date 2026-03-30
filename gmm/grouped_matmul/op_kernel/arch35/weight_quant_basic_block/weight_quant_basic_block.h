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
 * \file weight_quant_basic_block.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "tool.h"
#include "weight_quant_cube_compute.h"
#include "weight_quant_vec_compute.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TPipe;
using AscendC::TPosition;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class WeightQuantMatmulBasicBlock {
public:
    __aicore__ inline WeightQuantMatmulBasicBlock(){};
    __aicore__ inline void Init(bool hasBias, uint64_t aPrefetchSize,
                                const TCubeTiling *__restrict matmulTiling, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize, __gm__ xType *x, __gm__ wType *weight,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
                                            __gm__ scaleType *scale, __gm__ perTokenScaleType *perTokenScale,
                                            __gm__ biasType *bias, __gm__ yType *y, const bool hasBias,
                                            const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);
    __aicore__ inline void End();

protected:
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();
    __aicore__ inline void WaitAicToAiv();
    __aicore__ inline void ComputeBasicBlockAivNdKnNzNk(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void mxBiasSetParamAndGmtoUb(const BasicBlockOffsetParam &offsetParam,
                                                   L1ConsumeConfig &l1ConsumeConfig, UbConsumeConfig &ubConsumeConfig,
                                                   const uint64_t kMte2Offset, const uint64_t mte2RealK);
    __aicore__ inline void ComputeBasicBlockAic(const BasicBlockOffsetParam &offsetParam);

    BasicBlockLibVectorAntiQuantCompute<xType, wType, antiQuantScaleType, biasType, yType, wqmmConfig, vecConfig>
        vectorCompute_;

    WeightQuantBatchMatmulV2CubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType, yType, wqmmConfig,
                                        void> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;

    LocalTensor<xType> weightL1_;
    LocalTensor<biasType> biasL1_;
    uint64_t weightL1DbOffset_;
    uint64_t biasL1DbOffset_;
    bool hasBias_;
};

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::Init(
    bool hasBias, uint64_t aPrefetchSize, const TCubeTiling *__restrict matmulTiling, TPipe *tPipe)
{
    hasBias_ = hasBias;
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256; // weight单块大小的标准shape是256*256
    static constexpr uint64_t MX_BIAS_L1_SIZE = BIAS_L1_SIZE * GetKBUnit<biasType>() * sizeof(biasType);
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;
    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;
    uint64_t biasL1Offset = l1StartSize;
    biasL1_ = LocalTensor<biasType>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(biasType));
    if (hasBias_) {
        biasL1DbOffset_ = (l1RemainSize - MX_BIAS_L1_SIZE) / sizeof(biasType);
        l1RemainSize -= DOUBLE_BUFFER_NUM * MX_BIAS_L1_SIZE;
        l1StartSize += MX_BIAS_L1_SIZE;
    }

    if ASCEND_IS_AIC {
        cubeCompute_.MxA8W4Init(aPrefetchSize, l1RemainSize, l1StartSize, biasL1DbOffset_, matmulTiling, biasL1Offset);
    } else {
        vectorCompute_.Init(tPipe, hasBias_);
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::UpdateGlobalAddr(
    uint64_t mSize, uint64_t kSize, uint64_t nSize,
    __gm__ xType *x, __gm__ wType *weight, __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
    __gm__ scaleType *scale, __gm__ perTokenScaleType *perTokenScale, __gm__ biasType *bias, __gm__ yType *y,
    const bool hasBias, const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(mSize, kSize, nSize, x, y, bias, antiquantScale, scale, perTokenScale, hasBias);
    } else {
        vectorCompute_.UpdateGlobalAddr(mSize, kSize, nSize, weight, antiquantScale, antiquantOffset, nullptr, nullptr, bias,
                                        weightL2Cacheable);
    }
}

/*
 * 该函数作用为更新Mx的Bias的各种参数设置，并完成bias从GM到UB的搬运
 * isBiasSingleVector = True 表示只需要一个Vector核进行计算
 * calcMxBias = True 表示需要对Bias进行计算核搬运
 */
template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
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

/*
 * 该函数作用为根据转置属性，L1上shape大小以及vecconfig确定vec核的实际搬运量
 * ND TransB = False 两个vec核的mte2搬运量为 (curVecCoreMte2RealK, curVecCoreMte2RealN)
 * NZ TransB = True 两个vec核的mte2搬运量为 (CeilDiv(curVecCoreMte2RealK, C0), CeilDiv(curVecCoreMte2RealN, 16), 16, C0)
 */
template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::ComputeBasicBlockAivNdKnNzNk(
    const BasicBlockOffsetParam &offsetParam)
{
    // c:v为1：2, cube所需数据由两个v提供
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
        
        /*
         * 场景1：当前core为v0，mte2实际搬运的k值为mte2方向搬运标准值(kMte2BaseSize)和l1上k实际值的最小值
         * 场景2: 当前core为v1, 且l1实际的k值比core v1搬运值大，则mte2实际搬运的k值为两者之差
         * 场景3：当前core为v1, 且l1实际的k值比core v1搬运值小，则mte2无需搬运,设置为0即可
         */
        uint64_t mte2RealK = GetSubBlockIdx() == 0 ? min(kMte2BaseSize, l1ConsumeConfig.l1RealExternalLen)
                             : l1ConsumeConfig.l1RealExternalLen > kMte2BaseSize
                                 ? l1ConsumeConfig.l1RealExternalLen - kMte2BaseSize
                                 : 0;
        
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
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::ComputeBasicBlockAic(
    const BasicBlockOffsetParam &offsetParam)
{
    // 当前方案下，不会出现N方向计算量小于载入量的情况，所以没有N的循环
    for (uint64_t kbGmOffset = 0; kbGmOffset < offsetParam.kSize; kbGmOffset += offsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbGmOffset + offsetParam.kbL1Size) >= offsetParam.kSize
                                    ? offsetParam.kSize - kbGmOffset
                                    : offsetParam.kbL1Size;
        cubeCompute_.WaitScaleMTE1ToMTE2(kbGmOffset);
        cubeCompute_.CopyMxScaleGmToL1(offsetParam, kbGmOffset);
        cubeCompute_.WaitMTE1ToMTE2(kbGmOffset, offsetParam);
        cubeCompute_.CopyAAndBiasGmToL1(offsetParam, kbGmOffset, cvLoopIdx_);
        WaitAivToAic();
        cubeCompute_.LaunchMatmul((cvLoopIdx_ & 1) * weightL1DbOffset_, kbGmOffset, kbL1RealSize, cvLoopIdx_,
                                  offsetParam); // mte1 mmad fixp流水
        cubeCompute_.SetMTE1ToMTE2(kbGmOffset, offsetParam);
        cubeCompute_.SetScaleMTE1ToMTE2(kbGmOffset, offsetParam);
        SetAicToAiv();
    }
    cubeCompute_.GetTensorC(offsetParam);
    cubeCompute_.ClearAFullLoadFlag();  // 清除A全载时之前循环的set同步标记
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::ComputeBasicBlock(
    const BasicBlockOffsetParam &offsetParam)
{
    if ASCEND_IS_AIV {
        ComputeBasicBlockAivNdKnNzNk(offsetParam);
    } else {
        ComputeBasicBlockAic(offsetParam);
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::PrefetchA(
    uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    cubeCompute_.PrefetchA(aPrefetchSize, xSizeLimit);
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::End()
{
    if ASCEND_IS_AIC {
        cubeCompute_.EndSync();
    } else {
        if (cvLoopIdx_ > 0) {
            WaitAicToAiv();
        }
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }
        vectorCompute_.End();
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::SetAivToAic()
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
#endif
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::WaitAivToAic()
{
#ifndef __CCE_KT_TEST__
#ifndef __DAV_310R6__
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
#endif
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
#endif
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::SetAicToAiv()
{
#ifndef __CCE_KT_TEST__
#if !defined(__DAV_310R6__)
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
#endif
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                   biasType, yType, wqmmConfig, vecConfig>::WaitAicToAiv()
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
#endif
}
}  // namespace WeightQuantBatchMatmulV2::Arch35

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_H
