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
 * \file grouped_matmul_mxfp8fp4_block_mmad_resplit.h
 * \brief
 */
#ifndef GROUPED_MATMUL_MXFP8FP4_BLOCK_MMAD_RESPLIT_H
#define GROUPED_MATMUL_MXFP8FP4_BLOCK_MMAD_RESPLIT_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "../prologue/tool.h"
#include "weight_quant_cube_compute.h"

using AscendC::LocalTensor;
using AscendC::TPosition;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class WeightQuantMatmulBasicBlockAic {
public:
    __aicore__ inline WeightQuantMatmulBasicBlockAic() = delete;
    __aicore__ inline WeightQuantMatmulBasicBlockAic(bool hasBias, uint64_t aPrefetchSize,
                                                     const TCubeTiling *__restrict matmulTiling);
    template <typename TensorA, typename TensorC, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void operator()(const TensorA &tensorA, const TensorC &tensorC,
                                      const TensorScaleA &tensorScaleA, const TensorScaleB &tensorScaleB,
                                      uint64_t kaL1Size, uint64_t kbL1Size);
    __aicore__ inline void operator()(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);
    __aicore__ inline void End();

protected:
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();

    WeightQuantBatchMatmulV2CubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType, yType, wqmmConfig,
                                        void> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;

    LocalTensor<xType> weightL1_;
    LocalTensor<biasType> biasL1_;
    uint64_t weightL1DbOffset_;
    uint64_t biasL1DbOffset_;
    bool hasBias_;

    constexpr static uint64_t MX_SCALE_K_L1_SIZE = 4096;
};

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline WeightQuantMatmulBasicBlockAic<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                 biasType, yType, wqmmConfig, vecConfig>::WeightQuantMatmulBasicBlockAic(
    bool hasBias, uint64_t aPrefetchSize, const TCubeTiling *__restrict matmulTiling)
{
    hasBias_ = hasBias;
    biasL1DbOffset_ = 0;
    weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));

    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;
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

    cubeCompute_.MxA8W4Init(aPrefetchSize, l1RemainSize, l1StartSize, biasL1DbOffset_, matmulTiling, biasL1Offset);
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
template <typename TensorA, typename TensorC, typename TensorScaleA, typename TensorScaleB>
__aicore__ inline void WeightQuantMatmulBasicBlockAic<xType, wType, antiQuantScaleType, scaleType,
                                                      perTokenScaleType, biasType, yType, wqmmConfig,
                                                      vecConfig>::operator()(const TensorA &tensorA,
                                                                             const TensorC &tensorC,
                                                                             const TensorScaleA &tensorScaleA,
                                                                             const TensorScaleB &tensorScaleB,
                                                                             uint64_t kaL1Size,
                                                                             uint64_t kbL1Size)
{
    BasicBlockOffsetParam blockParam = {};
    blockParam.mL1Size = GetEleFromLayout<decltype(tensorC.Layout()), AttrInfo::SHAPE, AttrInfo::ROW, 1>(tensorC.Layout());
    blockParam.kSize = GetEleFromLayout<decltype(tensorA.Layout()), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(tensorA.Layout());
    blockParam.nL1Size =
        GetEleFromLayout<decltype(tensorC.Layout()), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(tensorC.Layout());
    blockParam.mSize = blockParam.mL1Size;
    blockParam.nSize = blockParam.nL1Size;
    blockParam.kaL1Size = kaL1Size;
    blockParam.kbL1Size = kbL1Size;
    cubeCompute_.UpdateGlobalTensor(tensorA, tensorC, tensorScaleA, tensorScaleB, hasBias_);
    for (uint64_t kbGmOffset = 0; kbGmOffset < blockParam.kSize; kbGmOffset += blockParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbGmOffset + blockParam.kbL1Size) >= blockParam.kSize ?
                                    blockParam.kSize - kbGmOffset :
                                    blockParam.kbL1Size;
        cubeCompute_.WaitScaleMTE1ToMTE2(kbGmOffset);
        cubeCompute_.CopyMxScaleGmToL1(blockParam, kbGmOffset);
        cubeCompute_.WaitMTE1ToMTE2(kbGmOffset, blockParam);
        cubeCompute_.CopyAAndBiasGmToL1(blockParam, kbGmOffset, cvLoopIdx_);
        WaitAivToAic();
        cubeCompute_.LaunchMatmul((cvLoopIdx_ & 1) * weightL1DbOffset_, kbGmOffset, kbL1RealSize, cvLoopIdx_,
                                  blockParam);
        cubeCompute_.SetMTE1ToMTE2(kbGmOffset, blockParam);
        cubeCompute_.SetScaleMTE1ToMTE2(kbGmOffset, blockParam);
        SetAicToAiv();
    }
    cubeCompute_.GetTensorC(blockParam);
    cubeCompute_.ClearAFullLoadFlag();
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAic<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::PrefetchA(
    uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    cubeCompute_.PrefetchA(aPrefetchSize, xSizeLimit);
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAic<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::End()
{
    cubeCompute_.EndSync();
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAic<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::WaitAivToAic()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename perTokenScaleType,
          typename biasType, typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantMatmulBasicBlockAic<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType,
                                                      biasType, yType, wqmmConfig, vecConfig>::SetAicToAiv()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}
}  // namespace WeightQuantBatchMatmulV2::Arch35

#endif  // GROUPED_MATMUL_MXFP8FP4_BLOCK_MMAD_RESPLIT_H
