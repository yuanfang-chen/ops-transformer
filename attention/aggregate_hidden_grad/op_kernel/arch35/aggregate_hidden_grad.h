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
 * \file aggregate_hidden_grad.h
 * \brief Arch35 AICore kernel for aggregate_hidden_grad (W=3)
 */

#ifndef AGGREGATE_HIDDEN_GRAD_H
#define AGGREGATE_HIDDEN_GRAD_H

#include "kernel_operator.h"
#include "vf/compute.h"
#include "aggregate_hidden_grad_struct.h"

namespace AggregateHiddenGradKernelNS {
using namespace AscendC;
using AggregateHiddenGradArch35Tiling::AggregateHiddenGradTilingDataV35;

template <typename DT>
class AggregateHiddenGradKernel {
public:
    static constexpr int kW = 3;
    static constexpr int kBufferNum = 2;
    static constexpr int kAlignBytes = 32;
    static constexpr int kMinHTile = 64;

    // Declarations
    __aicore__ inline AggregateHiddenGradKernel();

    __aicore__ inline void Init(GM_ADDR grad_output,
                                GM_ADDR input,
                                GM_ADDR weight,
                                GM_ADDR mask,
                                GM_ADDR grad_input,
                                GM_ADDR grad_weight,
                                const AggregateHiddenGradTilingDataV35 *td,
                                TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInGradOutput(int64_t bTile, int64_t sTile, int64_t hTile,
                                            int64_t bLen, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyInInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                       int64_t bLen, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyInWeight(int64_t hTile, int64_t hLenThis);
    __aicore__ inline void CopyInMask(int64_t bTile, int64_t sTile, int64_t bLen, int64_t sLen);
    __aicore__ inline void ComputeGradOutputMask(int64_t bLen, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void Compute(int64_t bTile, int64_t sTile, int64_t hTile,
                                   int64_t bLen, int64_t sEff, int64_t sLen, int64_t hLenThis,
                                   LocalTensor<DT> &wLocal);
    __aicore__ inline void CopyOutGradInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                            int64_t bLen, int64_t sEff, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyOutGradWeight(int64_t hTile, int64_t hLenThis);

private:
    // resources
    TPipe *pipe_;
    const AggregateHiddenGradTilingDataV35 *td_;
    TQue<QuePosition::VECIN, kBufferNum> gradOutQ_;
    TQue<QuePosition::VECIN, kBufferNum> inputQ_;
    TQue<QuePosition::VECIN, kBufferNum> weightQ_;
    TQue<QuePosition::VECIN, kBufferNum> maskQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradInQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradWeightQ_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    GlobalTensor<DT> gradOutGm_;
    GlobalTensor<DT> inputGm_;
    GlobalTensor<DT> weightGm_;
    GlobalTensor<DT> gradInGm_;
    GlobalTensor<DT> gradWeightGm_;
    GlobalTensor<bool> maskGm_;

    // cached tiling values
    int64_t H_{0}, S_{0}, B_{0}, W_{0}, dtypeSize_{0};
    int64_t hUB_{0}, bUB_{0}, sUB_{0};
    int64_t hMainCoreCnt_{0}, hTailCoreCnt_{0}, hMainSize_{0}, hTailSize_{0};
    int64_t hStart_{0}, hLen_{0};

    // fp32 accumulators for grad_weight
    LocalTensor<float> gwAccF32_[kW];
};


template <typename DT>
__aicore__ inline AggregateHiddenGradKernel<DT>::AggregateHiddenGradKernel() {}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::Init(GM_ADDR grad_output,
                                                           GM_ADDR input,
                                                           GM_ADDR weight,
                                                           GM_ADDR mask,
                                                           GM_ADDR grad_input,
                                                           GM_ADDR grad_weight,
                                                           const AggregateHiddenGradTilingDataV35 *td,
                                                           TPipe *pipe)
{
    pipe_ = pipe;
    td_ = td;
    // cache tiling fields
    H_ = td_->H;
    S_ = td_->S;
    B_ = td_->B;
    W_ = td_->W;
    dtypeSize_ = td_->dtypeSize;
    hUB_ = td_->hUB;
    bUB_ = td_->bUB;
    sUB_ = td_->sUB;
    hMainCoreCnt_ = td_->hMainCoreCnt;
    hTailCoreCnt_ = td_->hTailCoreCnt;
    hMainSize_ = td_->hMainSize;
    hTailSize_ = td_->hTailSize;

    // compute this core's H start/len
    uint64_t blkIdx = GetBlockIdx();
    if (blkIdx < static_cast<uint64_t>(hMainCoreCnt_)) {
        hStart_ = blkIdx * hMainSize_;
        hLen_ = hMainSize_;
    } else {
        uint64_t tailIdx = blkIdx - static_cast<uint64_t>(hMainCoreCnt_);
        hStart_ = static_cast<uint64_t>(hMainCoreCnt_) * hMainSize_ + tailIdx * hTailSize_;
        hLen_ = hTailSize_;
    }

    // bind GM tensors (assume contiguous ND layout)
    gradOutGm_.SetGlobalBuffer((__gm__ DT *)grad_output, S_ * B_ * H_);
    inputGm_.SetGlobalBuffer((__gm__ DT *)input, S_ * B_ * H_);
    weightGm_.SetGlobalBuffer((__gm__ DT *)weight, W_ * H_);
    gradInGm_.SetGlobalBuffer((__gm__ DT *)grad_input, S_ * B_ * H_);
    gradWeightGm_.SetGlobalBuffer((__gm__ DT *)grad_weight, W_ * H_);
    if (td_->hasMask) {
        maskGm_.SetGlobalBuffer((__gm__ bool *)mask, B_ * S_);
    }

    // init queues and tmp buffers
    pipe_->InitBuffer(gradOutQ_, kBufferNum, static_cast<uint32_t>(hUB_ * bUB_ * sUB_ * sizeof(DT)));
    pipe_->InitBuffer(inputQ_, kBufferNum, static_cast<uint32_t>(hUB_ * bUB_ * sUB_ * sizeof(DT)));
    pipe_->InitBuffer(weightQ_, kBufferNum, static_cast<uint32_t>(hUB_ * kW * sizeof(DT)));
    if (td_->hasMask) {
        uint32_t maskBytes = static_cast<uint32_t>(bUB_ * sUB_);
        uint32_t maskBufSize = (maskBytes + (kAlignBytes - 1)) / kAlignBytes * kAlignBytes;
        pipe_->InitBuffer(maskQ_, kBufferNum, maskBufSize);
    }
    pipe_->InitBuffer(gradInQ_, kBufferNum, static_cast<uint32_t>(hUB_ * bUB_ * sUB_ * sizeof(DT)));
    pipe_->InitBuffer(gradWeightQ_, kBufferNum, static_cast<uint32_t>(hUB_ * kW * sizeof(DT)));

    // tmp buffer holds fp32 accumulators for grad_weight: 3 * hUB_
    pipe_->InitBuffer(tmpBuf_, static_cast<uint32_t>(kW * hUB_ * sizeof(float)));
    uint32_t off = 0;
    gwAccF32_[0] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hUB_), off);
    off += static_cast<uint32_t>(hUB_ * sizeof(float));
    gwAccF32_[1] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hUB_), off);
    off += static_cast<uint32_t>(hUB_ * sizeof(float));
    gwAccF32_[2] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hUB_), off);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::Process()
{
    // Accumulate grad_weight in fp32 across all (b,s) tiles per h-tile, then cast+store once per h-tile
    for (int64_t hTile = 0; hTile < td_->hLoopCnt; ++hTile) {
        int64_t hLenThis = (hTile == td_->hLoopCnt - 1 && td_->hUBTail > 0) ? td_->hUBTail : hUB_;
        // zero fp32 accumulators for this h-tile
        Duplicate(gwAccF32_[0], 0.0f, static_cast<uint32_t>(hLenThis));
        Duplicate(gwAccF32_[1], 0.0f, static_cast<uint32_t>(hLenThis));
        Duplicate(gwAccF32_[2], 0.0f, static_cast<uint32_t>(hLenThis));

        // Load weight once per h-tile
        CopyInWeight(hTile, hLenThis);
        LocalTensor<DT> wLocal = weightQ_.DeQue<DT>();

        for (int64_t bTile = 0; bTile < td_->bLoopCnt; ++bTile) {
            int64_t bLen = (bTile == td_->bLoopCnt - 1 && td_->bUBTail > 0) ? td_->bUBTail : bUB_;
            for (int64_t sTile = 0; sTile < td_->sLoopCnt; ++sTile) {
                int64_t sLen = (sTile == td_->sLoopCnt - 1 && td_->sUBTail > 0) ? td_->sUBTail : sUB_;
                int64_t sEff = (sTile == td_->sLoopCnt - 1) ? sLen : (sLen - 2);
                CopyInGradOutput(bTile, sTile, hTile, bLen, sLen, hLenThis);
                CopyInInput(bTile, sTile, hTile, bLen, sLen, hLenThis);
                if (td_->hasMask) {
                    CopyInMask(bTile, sTile, bLen, sLen);
                    ComputeGradOutputMask(bLen, sLen, hLenThis);
                }
                Compute(bTile, sTile, hTile, bLen, sEff, sLen, hLenThis, wLocal);
                CopyOutGradInput(bTile, sTile, hTile, bLen, sEff, sLen, hLenThis);
            }
        }

        weightQ_.FreeTensor(wLocal);
        CopyOutGradWeight(hTile, hLenThis);
    }
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::ComputeGradOutputMask(int64_t bLen, int64_t sLen, int64_t hLenThis)
{
    LocalTensor<bool> mLocal = maskQ_.DeQue<bool>();
    LocalTensor<DT> goLocal = gradOutQ_.DeQue<DT>();
    for (int64_t b = 0; b < bLen; ++b) {
        for (int64_t s = 0; s < sLen; ++s) {
            bool flag = mLocal.GetValue(static_cast<uint32_t>(b * sLen + s));
            if (!flag) {
                uint32_t offset = static_cast<uint32_t>((b * sLen + s) * hLenThis);
                LocalTensor<DT> row = goLocal[offset];
                Duplicate(row, static_cast<DT>(0), static_cast<uint32_t>(hLenThis));
            }
        }
    }
    maskQ_.FreeTensor(mLocal);
    gradOutQ_.EnQue<DT>(goLocal);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::CopyInGradOutput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                       int64_t bLen, int64_t sLen, int64_t hLenThis)
{
    LocalTensor<DT> goLocal = gradOutQ_.AllocTensor<DT>();
    // Copy a [sLen, bLen, hLenThis] block along contiguous H
    DataCopyExtParams inParams{static_cast<uint16_t>(sLen * bLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0, 0};
    int64_t sStart = sTile * sUB_;
    int64_t bStart = bTile * bUB_;
    int64_t hOff = hTile * hUB_;
    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
    DataCopyPad(goLocal, gradOutGm_[base], inParams, padParams);
    gradOutQ_.EnQue<DT>(goLocal);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::CopyInInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                  int64_t bLen, int64_t sLen, int64_t hLenThis)
{
    LocalTensor<DT> inLocal = inputQ_.AllocTensor<DT>();
    DataCopyExtParams inParams{static_cast<uint16_t>(sLen * bLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0, 0};
    int64_t sStart = sTile * sUB_;
    int64_t bStart = bTile * bUB_;
    int64_t hOff = hTile * hUB_;
    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
    DataCopyPad(inLocal, inputGm_[base], inParams, padParams);
    inputQ_.EnQue<DT>(inLocal);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::CopyInWeight(int64_t hTile, int64_t hLenThis)
{
    LocalTensor<DT> wLocal = weightQ_.AllocTensor<DT>();
    DataCopyExtParams inParams{static_cast<uint16_t>(kW), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0, 0};
    int64_t hOff = hTile * hUB_;
    int64_t base = (hStart_ + hOff);
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
    DataCopyPad(wLocal, weightGm_[base], inParams, padParams);
    weightQ_.EnQue<DT>(wLocal);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::CopyInMask(int64_t bTile, int64_t sTile,
                                                                 int64_t bLen, int64_t sLen)
{
    LocalTensor<bool> mLocal = maskQ_.AllocTensor<bool>();
    // bool is 1B on Ascend: mask has shape [B,S] with 1 byte per element
    DataCopyExtParams inParams{static_cast<uint16_t>(bLen), static_cast<uint32_t>(sLen),
                               static_cast<uint32_t>(S_ - sLen), 0, 0};
    int64_t sStart = sTile * sUB_;
    int64_t bStart = bTile * bUB_;
    int64_t base = bStart * S_ + sStart;
    DataCopyPadExtParams<bool> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, maskGm_[base], inParams, padParams);
    maskQ_.EnQue<bool>(mLocal);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::Compute(int64_t bTile, int64_t sTile, int64_t hTile,
                                                              int64_t bLen, int64_t sEff, int64_t sLen,
                                                              int64_t hLenThis, LocalTensor<DT> &wLocal)
{
    LocalTensor<DT> goLocal = gradOutQ_.DeQue<DT>();
    LocalTensor<DT> inLocal = inputQ_.DeQue<DT>();
    LocalTensor<DT> giLocal = gradInQ_.AllocTensor<DT>();

    // masked in Process

    // Compute grad_input using VF (reuse weight already loaded for this h-tile)
    AggHiddenGradVF::DoGradInput<DT>(goLocal, wLocal, giLocal, static_cast<uint32_t>(bLen),
                                     static_cast<uint32_t>(sEff), static_cast<uint32_t>(sLen),
                                     static_cast<uint32_t>(hLenThis));

    // Accumulate grad_weight in fp32 using VF Acc variant
    AggHiddenGradVF::DoGradWeightAcc<DT>(goLocal, inLocal, gwAccF32_[0], gwAccF32_[1], gwAccF32_[2],
                                         static_cast<uint32_t>(bLen), static_cast<uint32_t>(sEff),
                                         static_cast<uint32_t>(sLen), static_cast<uint32_t>(hLenThis));

    gradOutQ_.FreeTensor(goLocal);
    inputQ_.FreeTensor(inLocal);
    gradInQ_.EnQue<DT>(giLocal);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::CopyOutGradInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                       int64_t bLen, int64_t sEff, int64_t sLen,
                                                                       int64_t hLenThis)
{
    LocalTensor<DT> giLocal = gradInQ_.DeQue<DT>();
    DataCopyExtParams outParams{static_cast<uint16_t>(sEff * bLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                                0, static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0};
    int64_t sStart = sTile * sUB_;
    int64_t bStart = bTile * bUB_;
    int64_t hOff = hTile * hUB_;
    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    DataCopyPad(gradInGm_[base], giLocal, outParams);
    gradInQ_.FreeTensor(giLocal);
}

template <typename DT>
__aicore__ inline void AggregateHiddenGradKernel<DT>::CopyOutGradWeight(int64_t hTile, int64_t hLenThis)
{
    // Cast fp32 accumulators to DT and write to GM: layout [W,H]
    int64_t hOff = hTile * hUB_;
    int64_t baseH = hStart_ + hOff;
    LocalTensor<DT> gwCast = gradWeightQ_.AllocTensor<DT>();
    for (int32_t k = 0; k < kW; ++k) {
        // Cast first hLenThis elements
        LocalTensor<DT> gwSlice = gwCast[static_cast<uint32_t>(k * hUB_)];
        Cast<DT, float>(gwSlice, gwAccF32_[k], RoundMode::CAST_RINT, static_cast<uint32_t>(hLenThis));
        int64_t base = static_cast<int64_t>(k) * H_ + baseH;
        DataCopyExtParams outParams{static_cast<uint16_t>(1), static_cast<uint32_t>(hLenThis * sizeof(DT)), 0, 0, 0};
        DataCopyPad(gradWeightGm_[base], gwSlice, outParams);
    }
    gradWeightQ_.FreeTensor(gwCast);
}

} // namespace AggregateHiddenGradKernelNS

#endif // AGGREGATE_HIDDEN_GRAD_H
