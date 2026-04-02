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
 * \file masked_causal_conv1d.h
 * \brief MaskedCausalConv1d AIV kernel implementation
 *
 * Shape: input [S,B,H], weight [3,H], mask [B,S] -> output [S,B,H]
 * Strategy: H×B×S 3D inter-core split, H outer loop, B middle, S inner;
 *           shift-2 in-place VF with prefixBuf for S boundaries.
 */

#ifndef MASKED_CAUSAL_CONV1D_H
#define MASKED_CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "./vf/masked_conv1d_vf.h"
#include "masked_causal_conv1d_struct.h"

namespace MaskedCausalConv1dNs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;    // double buffer for ioQueue
constexpr uint32_t ALIGN_BYTES = 32; // DataCopy 32-byte alignment unit

// ============================================================================
// MaskedCausalConv1d Kernel
// ============================================================================
template <typename T>
class MaskedCausalConv1d {
public:
    __aicore__ inline MaskedCausalConv1d() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR mask, GM_ADDR y,
                                const MaskedCausalConv1dTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MaskedCausalConv1dTilingData* tiling);
    __aicore__ inline void InitPrefix(uint32_t sStart, uint32_t b0, uint32_t bUbCur, uint32_t h0);
    __aicore__ inline void LoadMask(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur);
    __aicore__ inline void CopyIn(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0);
    __aicore__ inline void Compute(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur);
    __aicore__ inline void CopyOut(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0);

    // Helper: align up to align bytes
    static __aicore__ inline uint32_t AlignUp(uint32_t n, uint32_t align) {
        return (n + align - 1) / align * align;
    }

private:
    // -------------------------------------------------------------------------
    // Tiling parameters
    // -------------------------------------------------------------------------
    uint32_t S_, B_, H_;

    // H-dim inter-core
    uint32_t hCoreCnt_, hMainCnt_, hBlockFactor_, hBlockTailFactor_;
    // B-dim inter-core
    uint32_t bCoreCnt_, bMainCnt_, bBlockFactor_, bBlockTailFactor_;
    // S-dim inter-core
    uint32_t sCoreCnt_, sMainCnt_, sBlockFactor_, sBlockTailFactor_;

    // UB tile
    uint32_t hUb_;        // = H_REG = 64
    uint32_t ubFactorB_;
    uint32_t ubFactorS_;

    // Main-core loop params
    uint32_t loopNumH_,  ubTailFactorH_;
    uint32_t loopNumB_,  ubTailFactorB_;
    uint32_t loopNumS_,  ubTailFactorS_;

    // Tail-core loop params
    uint32_t tailBlockLoopNumH_,  tailBlockUbTailFactorH_;
    uint32_t tailBlockLoopNumB_,  tailBlockUbTailFactorB_;
    uint32_t tailBlockLoopNumS_,  tailBlockUbTailFactorS_;

    // Stride for non-contiguous x
    uint32_t xSStride_;  // S-dim stride (elements)
    uint32_t xBStride_;  // B-dim stride (elements)

    uint32_t realCoreNum_;

    // Core 3D index (decoded from blockIdx)
    uint32_t hCoreIdx_;
    uint32_t bCoreIdx_;
    uint32_t sCoreIdx_;

    // Current core's assigned ranges
    uint32_t hStart_, hLen_;
    uint32_t bStart_, bLen_;
    uint32_t sStart_, sLen_;

    // Loop params selected for this core (main vs tail)
    uint32_t myLoopNumH_,  myUbTailFactorH_;
    uint32_t myLoopNumB_,  myUbTailFactorB_;
    uint32_t myLoopNumS_,  myUbTailFactorS_;

    // -------------------------------------------------------------------------
    // Pipeline & Global Tensors
    // -------------------------------------------------------------------------
    TPipe pipe_;

    GlobalTensor<T>       xGM_;
    GlobalTensor<T>       weightGM_;
    GlobalTensor<uint8_t> maskGM_;   // [B][S] bool (uint8_t)
    GlobalTensor<T>       yGM_;

    // -------------------------------------------------------------------------
    // UB buffers
    // ioQueue   : [bUb][sUb][hUb] T, double buffer (ping-pong)
    // prefixQueue: [bUb][2][hUb] T, single, holds x[-2..x[-1] for S boundary
    // weightQueue: [3][hUb] T, single, loaded once per H-tile
    // maskQueue  : [bUb][sUb] uint8_t, single, loaded per S-tile
    // -------------------------------------------------------------------------
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> ioQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> prefixQueue_;   // [bUb][2][hUb]
    TQue<QuePosition::VECIN, 1> weightQueue_;   // [3][hUb]
    TQue<QuePosition::VECIN, 1> maskQueue_;     // [bUb][sUb] bytes
};

// ============================================================================
// ParseTilingData
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::ParseTilingData(
    const MaskedCausalConv1dTilingData* tiling)
{
    S_ = tiling->S;  B_ = tiling->B;  H_ = tiling->H;

    hCoreCnt_ = tiling->hCoreCnt;  hMainCnt_ = tiling->hMainCnt;
    hBlockFactor_ = tiling->hBlockFactor;  hBlockTailFactor_ = tiling->hBlockTailFactor;

    bCoreCnt_ = tiling->bCoreCnt;  bMainCnt_ = tiling->bMainCnt;
    bBlockFactor_ = tiling->bBlockFactor;  bBlockTailFactor_ = tiling->bBlockTailFactor;

    sCoreCnt_ = tiling->sCoreCnt;  sMainCnt_ = tiling->sMainCnt;
    sBlockFactor_ = tiling->sBlockFactor;  sBlockTailFactor_ = tiling->sBlockTailFactor;

    hUb_ = tiling->hUb;
    ubFactorB_ = tiling->ubFactorB;
    ubFactorS_ = tiling->ubFactorS;

    loopNumH_ = tiling->loopNumH;  ubTailFactorH_ = tiling->ubTailFactorH;
    loopNumB_ = tiling->loopNumB;  ubTailFactorB_ = tiling->ubTailFactorB;
    loopNumS_ = tiling->loopNumS;  ubTailFactorS_ = tiling->ubTailFactorS;

    tailBlockLoopNumH_ = tiling->tailBlockLoopNumH;
    tailBlockUbTailFactorH_ = tiling->tailBlockUbTailFactorH;
    tailBlockLoopNumB_ = tiling->tailBlockLoopNumB;
    tailBlockUbTailFactorB_ = tiling->tailBlockUbTailFactorB;
    tailBlockLoopNumS_ = tiling->tailBlockLoopNumS;
    tailBlockUbTailFactorS_ = tiling->tailBlockUbTailFactorS;

    xSStride_ = tiling->xSStride;
    xBStride_ = tiling->xBStride;
    realCoreNum_ = tiling->realCoreNum;
}

// ============================================================================
// Init
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR mask, GM_ADDR y,
    const MaskedCausalConv1dTilingData* tiling)
{
    ParseTilingData(tiling);

    // Decode 3D core index: blockIdx = hCoreIdx + bCoreIdx*hCoreCnt + sCoreIdx*(hCoreCnt*bCoreCnt)
    uint32_t blockIdx = GetBlockIdx();

    hCoreIdx_ = blockIdx % hCoreCnt_;
    bCoreIdx_ = (blockIdx / hCoreCnt_) % bCoreCnt_;
    sCoreIdx_ = blockIdx / (hCoreCnt_ * bCoreCnt_);

    // Compute this core's H range
    if (hCoreIdx_ < hMainCnt_) {
        hStart_ = hCoreIdx_ * hBlockFactor_;
        hLen_   = hBlockFactor_;
    } else {
        hStart_ = hMainCnt_ * hBlockFactor_ + (hCoreIdx_ - hMainCnt_) * hBlockTailFactor_;
        hLen_   = hBlockTailFactor_;
    }

    // Compute this core's B range
    if (bCoreIdx_ < bMainCnt_) {
        bStart_ = bCoreIdx_ * bBlockFactor_;
        bLen_   = bBlockFactor_;
    } else {
        bStart_ = bMainCnt_ * bBlockFactor_ + (bCoreIdx_ - bMainCnt_) * bBlockTailFactor_;
        bLen_   = bBlockTailFactor_;
    }

    // Compute this core's S range
    if (sCoreIdx_ < sMainCnt_) {
        sStart_ = sCoreIdx_ * sBlockFactor_;
        sLen_   = sBlockFactor_;
    } else {
        sStart_ = sMainCnt_ * sBlockFactor_ + (sCoreIdx_ - sMainCnt_) * sBlockTailFactor_;
        sLen_   = sBlockTailFactor_;
    }

    // Select loop params for this core
    bool isHTail = (hCoreIdx_ >= hMainCnt_);
    bool isBTail = (bCoreIdx_ >= bMainCnt_);
    bool isSTail = (sCoreIdx_ >= sMainCnt_);

    myLoopNumH_       = isHTail ? tailBlockLoopNumH_       : loopNumH_;
    myUbTailFactorH_  = isHTail ? tailBlockUbTailFactorH_  : ubTailFactorH_;
    myLoopNumB_       = isBTail ? tailBlockLoopNumB_       : loopNumB_;
    myUbTailFactorB_  = isBTail ? tailBlockUbTailFactorB_  : ubTailFactorB_;
    myLoopNumS_       = isSTail ? tailBlockLoopNumS_       : loopNumS_;
    myUbTailFactorS_  = isSTail ? tailBlockUbTailFactorS_  : ubTailFactorS_;

    // Bind GM
    xGM_.SetGlobalBuffer((__gm__ T*)x, (uint64_t)S_ * xSStride_);
    weightGM_.SetGlobalBuffer((__gm__ T*)weight, (uint64_t)3 * H_);
    maskGM_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    yGM_.SetGlobalBuffer((__gm__ T*)y, (uint64_t)S_ * B_ * H_);

    // Compute buffer sizes (aligned to 32 bytes)
    uint32_t ioBufBytes     = AlignUp(ubFactorB_ * ubFactorS_ * hUb_ * sizeof(T),  ALIGN_BYTES);
    uint32_t prefixBufBytes = AlignUp(ubFactorB_ * 2 * hUb_ * sizeof(T),           ALIGN_BYTES);
    uint32_t weightBufBytes = AlignUp(3 * hUb_ * sizeof(T),                        ALIGN_BYTES);
    uint32_t maskBufBytes   = AlignUp(ubFactorB_ * ubFactorS_ * sizeof(uint8_t),   ALIGN_BYTES);

    pipe_.InitBuffer(ioQueue_,    BUFFER_NUM, ioBufBytes);
    pipe_.InitBuffer(prefixQueue_, 1,          prefixBufBytes);
    pipe_.InitBuffer(weightQueue_, 1,          weightBufBytes);
    pipe_.InitBuffer(maskQueue_,   1,          maskBufBytes);
}

// ============================================================================
// InitPrefix: load x[sStart-2..sStart-1] to prefixQueue for each b in [b0, b0+bUbCur)
// Uses stride-based GM access (xSStride_, xBStride_).
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::InitPrefix(
    uint32_t sStart, uint32_t b0, uint32_t bUbCur, uint32_t h0)
{
    LocalTensor<T> pfBuf = prefixQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};


    for (uint32_t b = 0; b < bUbCur; ++b) {
        uint32_t pfOff = b * 2 * hUb_;  // element offset in prefixQueue for this b
        uint32_t pfBytes = pfOff * sizeof(T);
        if (sStart == 0) {
            // First core in S direction: both prefix rows are zero
            Duplicate(pfBuf[pfOff], (T)0, 2 * hUb_);
        } else if (sStart == 1) {
            // x[-1] is zero; x[0] loaded from GM
            Duplicate(pfBuf[pfOff], (T)0, hUb_);
            uint64_t gmOff = (uint64_t)0 * xSStride_ + (uint64_t)(b0 + b) * xBStride_ + h0;
            DataCopyExtParams cp1{1, static_cast<uint16_t>(hUb_ * sizeof(T)), 0, 0, 0};
            DataCopyPad(pfBuf[pfOff + hUb_], xGM_[gmOff], cp1, padParams);
        } else {
            // Load x[sStart-2] and x[sStart-1] from GM
            uint64_t gmOff0 = (uint64_t)(sStart - 2) * xSStride_ + (uint64_t)(b0 + b) * xBStride_ + h0;
            uint64_t gmOff1 = (uint64_t)(sStart - 1) * xSStride_ + (uint64_t)(b0 + b) * xBStride_ + h0;
            DataCopyExtParams cp{1, static_cast<uint16_t>(hUb_ * sizeof(T)), 0, 0, 0};
            DataCopyPad(pfBuf[pfOff],         xGM_[gmOff0], cp, padParams);
            DataCopyPad(pfBuf[pfOff + hUb_],  xGM_[gmOff1], cp, padParams);
        }
    }
    prefixQueue_.EnQue(pfBuf);
}

// ============================================================================
// LoadMask: load bUbCur × sCur mask values into maskQueue (per S-tile)
// mask GM layout: [B][S] (contiguous, stride = S)
// 使用连续加载避免对齐问题：一次性加载整个 S-tile 的所有 b 行
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::LoadMask(
    uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur)
{
    LocalTensor<uint8_t> mBuf = maskQueue_.AllocTensor<uint8_t>();
    // DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};

    // // 一次性加载所有 b 行：从 mask[b0, s0] 开始，行 stride = S_
    // // 每行 sCur 字节，共 bUbCur 行
    // uint64_t gmOff = (uint64_t)b0 * S_ + s0;
    // int64_t srcSkipBytes = static_cast<int64_t>((S_ - sCur) * sizeof(uint8_t));  // GM 中跳过剩余 S 列
    // DataCopyExtParams cp{
    //     static_cast<uint16_t>(bUbCur),           // rows = bUbCur
    //     static_cast<uint32_t>(sCur),             // 每行 sCur 字节
    //     srcSkipBytes,                            // GM 行间跳过的字节
    //     0, 0
    // };
    maskQueue_.EnQue(mBuf);
}

// ============================================================================
// CopyIn: load x[s0:s0+sCur, b0:b0+bUbCur, h0:h0+hUb] into ioQueue
// Non-contiguous: use xSStride_ for S-jump, iterate b with xBStride_
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::CopyIn(
    uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0)
{
    LocalTensor<T> ioBuf = ioQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (uint32_t b = 0; b < bUbCur; ++b) {
        uint64_t gmOffset   = (uint64_t)s0 * xSStride_ + (uint64_t)(b0 + b) * xBStride_ + h0;
        uint32_t dstOffset  = b * sCur * hUb_;  // 元素偏移
        uint32_t dstBytes   = dstOffset * sizeof(T);  // 字节偏移
        // srcJumpStride: bytes to skip in GM between consecutive S rows
        uint32_t srcSkipBytes = (xSStride_ - hUb_) * static_cast<uint32_t>(sizeof(T));
        DataCopyExtParams xcp{
            static_cast<uint16_t>(sCur),
            static_cast<uint16_t>(hUb_ * sizeof(T)),
            srcSkipBytes,
            0, 0
        };
        DataCopyPad(ioBuf[dstOffset], xGM_[gmOffset], xcp, padParams);
    }

    ioQueue_.EnQue(ioBuf);
}

// ============================================================================
// Compute: VF shift-2 in-place convolution, then apply mask at __aicore__ level
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Compute(uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur)
{
    LocalTensor<T>       ioBuf = ioQueue_.DeQue<T>();
    LocalTensor<T>       pfBuf = prefixQueue_.DeQue<T>();
    LocalTensor<T>       wBuf  = weightQueue_.DeQue<T>();
    LocalTensor<uint8_t> mBuf  = maskQueue_.DeQue<uint8_t>();

    // Call VF conv (no mask inside VF to avoid b8 SLD)
    CallMaskedConv1dVF<T>(ioBuf, pfBuf, wBuf, sCur, bUbCur);
    // Apply mask at __aicore__ level: zero out y where mask=0
    // After VF: y[s=0,1] are in pfBuf, y[s>=2] are in ioBuf[s-2]
    // mask 数据连续加载，布局：[b0 行所有 sCur][b1 行所有 sCur]...
    for (uint32_t b = 0; b < bUbCur; ++b) {
        for (uint32_t s = 0; s < sCur; ++s) {
            // 连续加载后，偏移 = b * sCur + s
            uint8_t maskVal = maskGM_.GetValue((b0 + b) * S_ + s0  + s);
            uint64_t off = (b0 + b) * S_ + s0  + s;
            if (maskVal == 0) {
                if (s < 2) {
                    Duplicate(pfBuf[b * 2 * hUb_ + s * hUb_], static_cast<T>(0), hUb_);
                } else {
                    Duplicate(ioBuf[b * sCur * hUb_ + (s - 2) * hUb_], static_cast<T>(0), hUb_);
                }
            }
        }
    }

    // 释放 mask 和 weight buffer（不再需要）
    maskQueue_.FreeTensor(mBuf);
    weightQueue_.FreeTensor(wBuf);
    // pfBuf 和 ioBuf 保留给 CopyOut 使用
    prefixQueue_.EnQue(pfBuf);
    ioQueue_.EnQue(ioBuf);
}

// ============================================================================
// CopyOut: write y back to GM in two segments
//   Segment 1: prefixQueue[b][0..1] -> yGM[s0, s0+1, b0+b, h0] (y[0], y[1])
//   Segment 2: ioQueue[b][0..sCur-3] -> yGM[s0+2..s0+sCur-1, b0+b, h0] (y[2..sCur-1])
//   Note: prefix for next S-tile is reloaded from GM by InitPrefix at top of S-loop.
// Output y GM layout: [S][B][H] contiguous (S-stride = B*H, B-stride = H).
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::CopyOut(
    uint32_t s0, uint32_t sCur, uint32_t b0, uint32_t bUbCur, uint32_t h0)
{
    LocalTensor<T> ioBuf    = ioQueue_.DeQue<T>();
    LocalTensor<T> pfTensor = prefixQueue_.DeQue<T>();

    // Output is contiguous [S][B][H]: stride between S rows = B_*H_
    uint32_t dstSkipBytes = (B_ * H_ - hUb_) * static_cast<uint32_t>(sizeof(T));

    for (uint32_t b = 0; b < bUbCur; ++b) {
        uint32_t pfOff = b * 2 * hUb_;
        uint32_t ioOff = b * sCur * hUb_;

        // Segment 1: y[0], y[1] from prefixQueue -> yGM[s0, s0+1]
        uint64_t dstOff0 = (uint64_t)s0 * B_ * H_ + (uint64_t)(b0 + b) * H_ + h0;
        DataCopyExtParams ycp1{2,
            static_cast<uint16_t>(hUb_ * sizeof(T)), 0, dstSkipBytes, 0};
        DataCopyPad(yGM_[dstOff0], pfTensor[pfOff], ycp1);

        // Segment 2: y[2..sCur-1] from ioBase[0..sCur-3] -> yGM[s0+2..s0+sCur-1]
        if (sCur > 2) {
            uint64_t dstOff2 = (uint64_t)(s0 + 2) * B_ * H_ + (uint64_t)(b0 + b) * H_ + h0;
            DataCopyExtParams ycp2{static_cast<uint16_t>(sCur - 2),
                static_cast<uint16_t>(hUb_ * sizeof(T)), 0, dstSkipBytes, 0};
            DataCopyPad(yGM_[dstOff2], ioBuf[ioOff], ycp2);
        }
    }

    // 释放所有 buffer
    ioQueue_.FreeTensor(ioBuf);
    prefixQueue_.FreeTensor(pfTensor);
}

// ============================================================================
// Process: main compute loop  H(outer) -> B(middle) -> S(inner)
// ============================================================================
template <typename T>
__aicore__ inline void MaskedCausalConv1d<T>::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum_) {
        return;
    }

    // ---- H outer loop ----
    for (uint32_t hIter = 0; hIter < myLoopNumH_; ++hIter) {
        uint32_t hUbCur = (hIter < myLoopNumH_ - 1) ? hUb_ : myUbTailFactorH_;
        uint32_t h0     = hStart_ + hIter * hUb_;
        // Load weight for this H-tile: weight[0..2, h0:h0+hUbCur]
        {
            LocalTensor<T> wBuf = weightQueue_.AllocTensor<T>();
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            uint32_t wSkipBytes = (H_ - hUbCur) * static_cast<uint32_t>(sizeof(T));
            DataCopyExtParams wcp{3,
                static_cast<uint16_t>(hUbCur * sizeof(T)),
                wSkipBytes, 0, 0};
            DataCopyPad(wBuf[0], weightGM_[h0], wcp, padParams);
            weightQueue_.EnQue(wBuf);
        }

        // ---- B middle loop ----
        for (uint32_t bIter = 0; bIter < myLoopNumB_; ++bIter) {
            uint32_t bUbCur = (bIter < myLoopNumB_ - 1) ? ubFactorB_ : myUbTailFactorB_;
            uint32_t b0     = bStart_ + bIter * ubFactorB_;

            // ---- S inner loop ----
            for (uint32_t sIter = 0; sIter < myLoopNumS_; ++sIter) {
                uint32_t sCur = (sIter < myLoopNumS_ - 1) ? ubFactorS_ : myUbTailFactorS_;
                uint32_t s0   = sStart_ + sIter * ubFactorS_;

                // Reload prefix each S-tile: x[s0-2..s0-1] (boundary for causal conv)
                InitPrefix(s0, b0, bUbCur, h0);
                LoadMask(s0, sCur, b0, bUbCur);
                CopyIn(s0, sCur, b0, bUbCur, h0);
                Compute(s0, sCur, b0, bUbCur);
                CopyOut(s0, sCur, b0, bUbCur, h0);
            }
        }
    }
}

} // namespace MaskedCausalConv1dNs

#endif // MASKED_CAUSAL_CONV1D_H
