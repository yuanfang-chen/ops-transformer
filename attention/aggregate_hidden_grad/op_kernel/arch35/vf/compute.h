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
 * \file compute.h
 * \brief MicroAPI VF helpers for aggregate_hidden_grad (W=3)
 */

#ifndef AGGREGATE_HIDDEN_GRAD_VF_COMPUTE_H
#define AGGREGATE_HIDDEN_GRAD_VF_COMPUTE_H

#include "kernel_operator.h"

using namespace AscendC;

namespace AggHiddenGradVF {

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t REGSIZE = 256;
constexpr uint32_t B32_REP_SIZE = REGSIZE / sizeof(float); // 64 floats per vector reg

// Assumption: dimLen (H-tile) is always a multiple of 64 floats, so dimTail == 0.
// This holds because H is a multiple of 192 and hUB is 64 per tiling.

// Compute grad_input for a tile:
// gi[i] = go[i]*w2 + go[i+1]*w1 + go[i+2]*w0 (0-pad beyond sLen)
// Layout assumptions:
// - go rows are laid out as rows = bLen*sLen, row = b*sLen + s, row length dimLen along H
// - w layout is [w0, w1, w2], each vector length dimLen (contiguous along H)
// - gi has same layout as go

template <typename T>
__simd_vf__ void GradInputW3VF(__ubuf__ T *goAddr, __ubuf__ T *wAddr, __ubuf__ T *giAddr, uint32_t bLen, uint32_t sEff,
                               uint32_t sLen, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        // Load weights chunk (w0,w1,w2) for current H slice
        MicroAPI::RegTensor<T> w0B16, w1B16, w2B16;
        MicroAPI::RegTensor<float> w0B32, w1B32, w2B32;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, wAddr + 0 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, wAddr + 1 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, wAddr + 2 * dimLen + dimOff);
        MicroAPI::Cast<float, T, castTraitB162B32>(w0B32, w0B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w1B32, w1B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w2B32, w2B16, fullMask);

        for (uint32_t b = 0; b < bLen; ++b) {
            for (uint32_t s = 0; s < sEff; ++s) {
                uint32_t row = b * sLen + s;
                __ubuf__ T *goRow = goAddr + row * dimLen + dimOff;
                __ubuf__ T *giRow = giAddr + row * dimLen + dimOff;

                MicroAPI::RegTensor<T> goB16, goN1B16, goN2B16, giB16;
                MicroAPI::RegTensor<float> goB32, goN1B32, goN2B32, accB32, mulB32;

                // acc = go * w2
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goB16, goRow);
                MicroAPI::Cast<float, T, castTraitB162B32>(goB32, goB16, fullMask);
                MicroAPI::Mul(accB32, goB32, w2B32, fullMask);
                // + go[i+1] * w1
                if (s + 1 < sLen) {
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN1B16, goRow + dimLen);
                    MicroAPI::Cast<float, T, castTraitB162B32>(goN1B32, goN1B16, fullMask);
                    MicroAPI::Mul(mulB32, goN1B32, w1B32, fullMask);
                    MicroAPI::Add(accB32, accB32, mulB32, fullMask);
                }
                // + go[i+2] * w0
                if (s + 2 < sLen) {
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN2B16, goRow + 2 * dimLen);
                    MicroAPI::Cast<float, T, castTraitB162B32>(goN2B32, goN2B16, fullMask);
                    MicroAPI::Mul(mulB32, goN2B32, w0B32, fullMask);
                    MicroAPI::Add(accB32, accB32, mulB32, fullMask);
                }
                MicroAPI::Cast<T, float, castTraitB322B16>(giB16, accB32, fullMask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giRow, giB16, fullMask);
            }
        }
        dimOff += B32_REP_SIZE;
    }
}

// Compute grad_weight and ACCUMULATE into fp32 accumulators in UB:
// acc2 += sum(go[i]*in[i]), acc1 += sum(go[i+1]*in[i]), acc0 += sum(go[i+2]*in[i])
// Assumption: dimLen % 64 == 0, so no tail handling.

template <typename T>
__simd_vf__ void GradWeightW3VFAcc(__ubuf__ T *goAddr, __ubuf__ T *inAddr, __ubuf__ float *acc0Addr,
                                   __ubuf__ float *acc1Addr, __ubuf__ float *acc2Addr, uint32_t bLen, uint32_t sEff,
                                   uint32_t sLen, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        MicroAPI::RegTensor<float> acc0, acc1, acc2, mulB32, goB32, goN1B32, goN2B32, inB32;
        // Load existing accumulators
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_UNPACK_B32>(acc0, acc0Addr + dimOff);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_UNPACK_B32>(acc1, acc1Addr + dimOff);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_UNPACK_B32>(acc2, acc2Addr + dimOff);

        for (uint32_t b = 0; b < bLen; ++b) {
            for (uint32_t s = 0; s < sEff; ++s) {
                uint32_t row = b * sLen + s;
                __ubuf__ T *goRow = goAddr + row * dimLen + dimOff;
                __ubuf__ T *inRow = inAddr + row * dimLen + dimOff;
                MicroAPI::RegTensor<T> goB16, inB16, goN1B16, goN2B16;
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goB16, goRow);
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(inB16, inRow);
                MicroAPI::Cast<float, T, castTraitB162B32>(goB32, goB16, fullMask);
                MicroAPI::Cast<float, T, castTraitB162B32>(inB32, inB16, fullMask);
                // acc2 += go * in
                MicroAPI::Mul(mulB32, goB32, inB32, fullMask);
                MicroAPI::Add(acc2, acc2, mulB32, fullMask);
                // acc1 += go[i+1] * in (guard using sLen so last S block safely handles boundary)
                if (s + 1 < sLen) {
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN1B16, goRow + dimLen);
                    MicroAPI::Cast<float, T, castTraitB162B32>(goN1B32, goN1B16, fullMask);
                    MicroAPI::Mul(mulB32, goN1B32, inB32, fullMask);
                    MicroAPI::Add(acc1, acc1, mulB32, fullMask);
                }
                // acc0 += go[i+2] * in
                if (s + 2 < sLen) {
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN2B16, goRow + 2 * dimLen);
                    MicroAPI::Cast<float, T, castTraitB162B32>(goN2B32, goN2B16, fullMask);
                    MicroAPI::Mul(mulB32, goN2B32, inB32, fullMask);
                    MicroAPI::Add(acc0, acc0, mulB32, fullMask);
                }
            }
        }
        // Store back updated accumulators
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_PACK_B32>(acc0Addr + dimOff, acc0, fullMask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_PACK_B32>(acc1Addr + dimOff, acc1, fullMask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_PACK_B32>(acc2Addr + dimOff, acc2, fullMask);
        dimOff += B32_REP_SIZE;
    }
}

// Wrapper helpers from LocalTensor to VF

template <typename T>
__aicore__ inline void DoGradInput(LocalTensor<T> &goUb, LocalTensor<T> &wUb, LocalTensor<T> &giUb, uint32_t bLen,
                                   uint32_t sEff, uint32_t sLen, uint32_t dimLen)
{
    __ubuf__ T *goAddr = (__ubuf__ T *)goUb.GetPhyAddr();
    __ubuf__ T *wAddr = (__ubuf__ T *)wUb.GetPhyAddr();
    __ubuf__ T *giAddr = (__ubuf__ T *)giUb.GetPhyAddr();
    GradInputW3VF<T>(goAddr, wAddr, giAddr, bLen, sEff, sLen, dimLen);
}

// Accumulate into fp32 accumulators in UB

template <typename T>
__aicore__ inline void DoGradWeightAcc(LocalTensor<T> &goUb, LocalTensor<T> &inUb, LocalTensor<float> &acc0Ub,
                                       LocalTensor<float> &acc1Ub, LocalTensor<float> &acc2Ub, uint32_t bLen,
                                       uint32_t sEff, uint32_t sLen, uint32_t dimLen)
{
    __ubuf__ T *goAddr = (__ubuf__ T *)goUb.GetPhyAddr();
    __ubuf__ T *inAddr = (__ubuf__ T *)inUb.GetPhyAddr();
    __ubuf__ float *a0Addr = (__ubuf__ float *)acc0Ub.GetPhyAddr();
    __ubuf__ float *a1Addr = (__ubuf__ float *)acc1Ub.GetPhyAddr();
    __ubuf__ float *a2Addr = (__ubuf__ float *)acc2Ub.GetPhyAddr();
    GradWeightW3VFAcc<T>(goAddr, inAddr, a0Addr, a1Addr, a2Addr, bLen, sEff, sLen, dimLen);
}

} // namespace AggHiddenGradVF

#endif // AGGREGATE_HIDDEN_GRAD_VF_COMPUTE_H
