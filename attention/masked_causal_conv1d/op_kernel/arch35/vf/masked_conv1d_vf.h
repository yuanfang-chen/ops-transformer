/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_conv1d_vf.h
 * \brief VF function for MaskedCausalConv1d kernel-width-3 causal convolution (no mask).
 *
 * Implements "shift-2 in-place" strategy:
 *   y[s] is written back to ioBase[s-2] (x[s-2] already consumed).
 *   y[0], y[1] are written to prefixBase[0], prefixBase[1].
 *   After VF: ioBase[sCur-2..sCur-1] still holds x[sCur-2..sCur-1] for next tile prefix.
 *
 * Mask zeroing is handled at __aicore__ level (in Compute) to avoid b8 SLD.
 */

#ifndef MASKED_CONV1D_VF_H
#define MASKED_CONV1D_VF_H

#include "kernel_operator.h"

using namespace AscendC;

namespace MaskedCausalConv1dNs {

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

// H_REG: VF FP32 register width in elements (256B / 4B = 64)
constexpr uint32_t H_REG = 64;

// ============================================================================
// MaskedConv1dVF: 3-tap causal convolution (NO mask, mask handled by caller).
//
// ioBase   : [bUbCur][sCur][H_REG] T — input x; output y[2..sCur-1] at [0..sCur-3]
// prefixBase: [bUbCur][2][H_REG] T  — input x[-2..x[-1]; receives y[0..1]
// wBase    : [3][H_REG] T            — weight[0..2]
// sCur     : number of S tokens in this tile
// bUbCur   : number of B elements in this tile
// ============================================================================
template <typename T>
__simd_vf__ void MaskedConv1dVF(
    __ubuf__ T* ioBase,
    __ubuf__ T* prefixBase,
    __ubuf__ T* wBase,
    uint32_t    sCur,
    uint32_t    bUbCur)
{
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    // Load weight once, shared across all (b, s)
    MicroAPI::RegTensor<T>     w0B16, w1B16, w2B16;
    MicroAPI::RegTensor<float> w0F32, w1F32, w2F32;
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, wBase + 0 * H_REG);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, wBase + 1 * H_REG);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, wBase + 2 * H_REG);
    MicroAPI::Cast<float, T, castTraitB162B32>(w0F32, w0B16, maskFull);
    MicroAPI::Cast<float, T, castTraitB162B32>(w1F32, w1B16, maskFull);
    MicroAPI::Cast<float, T, castTraitB162B32>(w2F32, w2B16, maskFull);

    for (uint32_t b = 0; b < bUbCur; ++b) {
        uint32_t pfOff = b * 2 * H_REG;      // prefixBase element offset for this b
        uint32_t ioOff = b * sCur * H_REG;   // ioBase element offset for this b

        for (uint32_t s = 0; s < sCur; ++s) {
            // Source addresses for the three taps
            __ubuf__ T* addrPrev2 = (s == 0) ? (prefixBase + pfOff + 0 * H_REG)
                                  : (s == 1) ? (prefixBase + pfOff + 1 * H_REG)
                                             : (ioBase + ioOff + (s - 2) * H_REG);
            __ubuf__ T* addrPrev1 = (s == 0) ? (prefixBase + pfOff + 1 * H_REG)
                                             : (ioBase + ioOff + (s - 1) * H_REG);
            __ubuf__ T* addrCurr  = ioBase + ioOff + s * H_REG;

            // Load and cast to FP32
            MicroAPI::RegTensor<T>     xp2B16, xp1B16, xcB16;
            MicroAPI::RegTensor<float> xp2F32, xp1F32, xcF32;
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xp2B16, addrPrev2);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xp1B16, addrPrev1);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xcB16,  addrCurr);
            MicroAPI::Cast<float, T, castTraitB162B32>(xp2F32, xp2B16, maskFull);
            MicroAPI::Cast<float, T, castTraitB162B32>(xp1F32, xp1B16, maskFull);
            MicroAPI::Cast<float, T, castTraitB162B32>(xcF32,  xcB16,  maskFull);

            // y = x[-2]*w0 + x[-1]*w1 + x[0]*w2  (FP32 accumulation)
            MicroAPI::RegTensor<float> yF32, tmpF32;
            MicroAPI::Mul(yF32,   xp2F32, w0F32, maskFull);
            MicroAPI::Mul(tmpF32, xp1F32, w1F32, maskFull);
            MicroAPI::Add(yF32,   yF32,   tmpF32, maskFull);
            MicroAPI::Mul(tmpF32, xcF32,  w2F32, maskFull);
            MicroAPI::Add(yF32,   yF32,   tmpF32, maskFull);

            // Cast back to BF16/FP16
            MicroAPI::RegTensor<T> yB16;
            MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yF32, maskFull);

            // Shift-2 write-back: s=0,1 -> prefixBase; s>=2 -> ioBase[s-2]
            __ubuf__ T* dst = (s < 2) ? (prefixBase + pfOff + s * H_REG)
                                       : (ioBase + ioOff + (s - 2) * H_REG);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst, yB16, maskFull);
        }
        // After VF: ioBase[ioOff + (sCur-2)*H_REG] and ioBase[ioOff + (sCur-1)*H_REG]
        // still hold original x[sCur-2] and x[sCur-1] (never overwritten), ready for next tile.
    }
}

// ============================================================================
// CallMaskedConv1dVF: __aicore__ wrapper that converts LocalTensor to raw ptr
// ============================================================================
template <typename T>
__aicore__ inline void CallMaskedConv1dVF(
    LocalTensor<T> ioBuf,
    LocalTensor<T> pfBuf,
    LocalTensor<T> wBuf,
    uint32_t       sCur,
    uint32_t       bUbCur)
{
    __ubuf__ T* ioBase     = (__ubuf__ T*)ioBuf.GetPhyAddr();
    __ubuf__ T* prefixBase = (__ubuf__ T*)pfBuf.GetPhyAddr();
    __ubuf__ T* wBase      = (__ubuf__ T*)wBuf.GetPhyAddr();
    MaskedConv1dVF<T>(ioBase, prefixBase, wBase, sCur, bUbCur);
}

} // namespace MaskedCausalConv1dNs

#endif // MASKED_CONV1D_VF_H
