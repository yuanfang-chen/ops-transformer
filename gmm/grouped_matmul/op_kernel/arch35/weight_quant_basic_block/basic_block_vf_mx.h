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
 * \file basic_block_vf_mx.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {
template <typename xType, typename wType, typename biasType>
struct MxA8W4NzParams {
    uint64_t loopKNum;
    uint64_t innerLoopNum;
    uint64_t loopKDstStride;
    uint64_t innerDstStride;
    uint64_t biasLoopNum;
    uint64_t nRealSizeAlign;
    __ubuf__ wType *weightLowBitPhyAddr;
    __ubuf__ xType *weightHighBitPhyAddr;
    __ubuf__ biasType *biasInUbAddr;
    __ubuf__ biasType *biasOutUbAddr;
};
static constexpr uint32_t E2M1_SHIFT_RIGHT_SIZE = 0x2;
static constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;
static constexpr uint32_t E2M1_AND_MASK = 0x9C;

template <typename xType, typename wType, typename biasType, bool calcMxBias, bool isBiasSingleVector>
__simd_callee__ inline void MxA8W4BiasCompute(MxA8W4NzParams<xType, wType, biasType> &mxA8W4NzParams)
{
    if constexpr (calcMxBias) {
        static constexpr biasType MX_BIAS_FACTOR = static_cast<biasType>(0.015625f);
        MicroAPI::RegTensor<biasType> biasVreg, biasFactorVreg;
        MicroAPI::MaskReg maskBiasAll = MicroAPI::CreateMask<biasType, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasFactorVreg, MX_BIAS_FACTOR,
                                                                                 maskBiasAll);
        if constexpr (isBiasSingleVector) {
            for (uint16_t loopBiasIdx = 0; loopBiasIdx < mxA8W4NzParams.biasLoopNum; ++loopBiasIdx) {
                MicroAPI::AddrReg biasAreg = MicroAPI::CreateAddrReg<biasType>(loopBiasIdx, VEC_MAX_ELEM_B16);
                MicroAPI::LoadAlign<biasType, MicroAPI::LoadDist::DIST_NORM>(biasVreg, mxA8W4NzParams.biasInUbAddr,
                                                                             biasAreg);
                MicroAPI::Mul<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasVreg, biasVreg, biasFactorVreg,
                                                                                   maskBiasAll);
                MicroAPI::StoreAlign<biasType, MicroAPI::StoreDist::DIST_NORM_B16>(mxA8W4NzParams.biasOutUbAddr,
                                                                                   biasVreg, biasAreg, maskBiasAll);
            }
        } else {
            MicroAPI::LoadAlign<biasType, MicroAPI::LoadDist::DIST_NORM>(biasVreg, mxA8W4NzParams.biasInUbAddr);
            MicroAPI::Mul<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasVreg, biasVreg, biasFactorVreg,
                                                                               maskBiasAll);
            MicroAPI::StoreAlign<biasType, MicroAPI::StoreDist::DIST_NORM_B16>(mxA8W4NzParams.biasOutUbAddr, biasVreg,
                                                                               maskBiasAll);
        }
    }
}

template <typename xType, typename wType, typename biasType, bool calcMxBias, bool isBiasSingleVector>
__simd_vf__ inline void AntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams)
{
    MxA8W4BiasCompute<xType, wType, biasType, calcMxBias, isBiasSingleVector>(mxA8W4NzParams);
    MicroAPI::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            // DIST_US_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7
            // Vd 0 1 0 1 2 3 2 3 4 5 4 5 6 7 6 7
            // 4bit物理地址位移 = 逻辑索引 >> 1
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                (MicroAPI::RegTensor<uint8_t> &)wLoad, (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                aregWeightB8In);

            MicroAPI::ShiftRight(wShr0, wLoad, wShrReg, preg);
            MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);
            MicroAPI::ShiftRight(wShr1, wShl, wShrReg, preg);
            MicroAPI::Select(wSel, wShr1, wShr0, pregVsel);
            MicroAPI::And(wAnd, wSel, wAndReg, preg);

            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, mxA8W4NzParams.loopKDstStride, innerLoopIdx, mxA8W4NzParams.innerDstStride);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr, (MicroAPI::RegTensor<uint8_t> &)wAnd,
                aregWeightB8Out, preg);
        }
    }
}
}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H