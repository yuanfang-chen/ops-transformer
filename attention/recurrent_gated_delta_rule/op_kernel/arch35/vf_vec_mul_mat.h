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
__aicore__ inline void MatVecMulVF(const LocalTensor<T>& matrixInUb, const LocalTensor<T>& rowVecInUb,
                                   const LocalTensor<T>& matrixOutUb, const uint16_t row, const uint32_t col)
{
    __ubuf__ float * matrixUb = (__ubuf__ float*)matrixInUb.GetPhyAddr();
    __ubuf__ float * rowVecUb = (__ubuf__ float*)rowVecInUb.GetPhyAddr();
    __ubuf__ float * outputUb = (__ubuf__ float*)matrixOutUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint16_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;
    uint16_t dTailLoop = dTail > 0 ? 1 : 0;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregMatrix;
        MicroAPI::RegTensor<T> vregRowVec;
        MicroAPI::RegTensor<T> vregOutput;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg tailMask;
        tailMask = MicroAPI::UpdateMask<float>(dTail);
        constexpr static MicroAPI::CastTrait castTraitPack2 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        constexpr static MicroAPI::CastTrait castTraitF32ToHalf = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ODD};

        uint32_t colOffset = 0;
        uint32_t rowOffset = 0;
        for (uint16_t j = 0; j < dLoops; j++) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregRowVec, rowVecUb + colOffset);
            rowOffset = 0;
            for (uint16_t i = 0; i < row; i++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMatrix, matrixUb + colOffset + rowOffset);
                MicroAPI::Mul(vregOutput, vregMatrix, vregRowVec, fullMask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outputUb + colOffset + rowOffset, vregOutput, fullMask);
                rowOffset += col;
            }
            colOffset += floatRepSize; 
        }

        if (dTailLoop > 0) {
            rowOffset = 0;
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregRowVec, rowVecUb + dLoops * floatRepSize);
            for (uint16_t i = 0; i < row; i++) {
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMatrix, matrixUb + dLoops * floatRepSize + rowOffset);
                MicroAPI::Mul(vregOutput, vregMatrix, vregRowVec, tailMask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outputUb + dLoops * floatRepSize + rowOffset, vregOutput, tailMask);
                rowOffset += col;
            }
        }
    }
}

template <typename T>
__aicore__ inline void OuterAddVF(const LocalTensor<T>& outputUb, const LocalTensor<T>& deltaInUb,
                                  const LocalTensor<T>& kInUb, const LocalTensor<T>& stateInUb, const uint16_t row,
                                  const uint32_t col)
{
    __ubuf__ T * deltaBuf = (__ubuf__ T*)deltaInUb.GetPhyAddr();
    __ubuf__ T * kInputBuf = (__ubuf__ T*)kInUb.GetPhyAddr();
    __ubuf__ T * stateInBuf = (__ubuf__ T*)stateInUb.GetPhyAddr();
    __ubuf__ T * outputBuf = (__ubuf__ T*)outputUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint16_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregKInput;
        MicroAPI::RegTensor<T> vregDelta;
        MicroAPI::RegTensor<T> vregStateIn;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t j = 0; j < dLoops; j++) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregKInput, kInputBuf + j * floatRepSize);
            for (uint16_t i = 0; i < row; i++) {
                uint16_t loopOffset = i * col + j * floatRepSize;
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregDelta, deltaBuf + i); // 使用DIST_BRC_B32模式搬运将首元素broadcast到整个寄存器
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregStateIn, stateInBuf + loopOffset);
                MicroAPI::MulAddDst<T, MicroAPI::MaskMergeMode::ZEROING>(vregStateIn, vregKInput, vregDelta, pregAll);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregStateIn, pregAll);
            }
        }

        if (dTail > 0) {
            MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<T>(dTail);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregKInput, kInputBuf + dLoops * floatRepSize);
            for (uint16_t i = 0; i < row; i++) {
                uint16_t loopOffset = i * col + dLoops * floatRepSize;
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregDelta, deltaBuf + i); // 使用DIST_BRC_B32模式搬运将首元素broadcast到整个寄存器
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregStateIn, stateInBuf + loopOffset);
                MicroAPI::MulAddDst<T, MicroAPI::MaskMergeMode::ZEROING>(vregStateIn, vregKInput, vregDelta, tailMask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregStateIn, tailMask);
            }
        }
    }
}

} // namespace

#endif // VEC_MUL_MAT_INTERFACE_H