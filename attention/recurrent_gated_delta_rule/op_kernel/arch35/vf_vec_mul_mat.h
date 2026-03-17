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
 * \file vf_vec_mul_mat.h
 * \brief
 */

#ifndef VEC_MUL_MAT_INTERFACE_H
#define VEC_MUL_MAT_INTERFACE_H

#include "kernel_tensor.h"
namespace AscendC{
template <typename T>
__aicore__ inline void MatVecMulVF(const LocalTensor<T>& stateInUb, const LocalTensor<T>& gamaKInUb,
                                   const LocalTensor<T>& stateOutUb, const uint16_t row, const uint32_t col,
                                   const uint32_t stride)
{
    __ubuf__ float * stateUb = (__ubuf__ float*)stateInUb.GetPhyAddr();
    __ubuf__ float * gamaKUb = (__ubuf__ float*)gamaKInUb.GetPhyAddr();
    __ubuf__ float * outputUb = (__ubuf__ float*)stateOutUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint16_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vregState;
        AscendC::MicroAPI::RegTensor<T> vregGamaK;
        AscendC::MicroAPI::RegTensor<T> vregOutput;
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg tailMask;
        tailMask = AscendC::MicroAPI::UpdateMask<float>(dTail);
        constexpr static AscendC::MicroAPI::CastTrait castTraitPack2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToHalf = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ODD};

        uint32_t colOffset = 0;
        uint32_t rowOffset = 0;
        for (uint16_t j = 0; j < dLoops; j++) {
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregGamaK, gamaKUb + colOffset);
            rowOffset = 0;
            for (uint16_t i = 0; i < row; i++) {
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregState, stateUb + colOffset + rowOffset);
                AscendC::MicroAPI::Mul(vregOutput, vregState, vregGamaK, fullMask);
                AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(outputUb + colOffset + rowOffset, vregOutput, fullMask);
                rowOffset += stride;
            }
            colOffset += floatRepSize; 
        }

        if (dTailLoop > 0) {
            rowOffset = 0;
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregGamaK, gamaKUb + dLoops * floatRepSize);
            for (uint16_t i = 0; i < row; i++) {
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregState, stateUb + dLoops * floatRepSize + rowOffset);
                AscendC::MicroAPI::Mul(vregOutput, vregState, vregGamaK, tailMask);
                AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(outputUb + dLoops * floatRepSize + rowOffset, vregOutput, tailMask);
                rowOffset += stride;
            }
        }
    }
}

} // namespace

#endif // VEC_MUL_MAT_INTERFACE_H