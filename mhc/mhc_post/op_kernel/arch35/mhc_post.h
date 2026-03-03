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
 * \file mhc_post.h
 * \brief MhcPost kernel implementation
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 *          where: (H_{l}^{res})^{T} * x_l represents matrix multiplication with transposed h_res
 *                h_{l}^{out} * H_{t}^{post} represents element-wise multiplication and broadcasting
 */

#ifndef ASCENDC_MHC_POST_H
#define ASCENDC_MHC_POST_H

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_post_tiling_data.h"
#include "mhc_post_tiling_key.h"

namespace MhcPost {
using namespace AscendC;

// Double Buffer configuration - Double Buffer提升Memory Bound算子性能
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2;       // Double Buffer depth for data tiles
constexpr uint32_t SINGLE_BUFFER_DEPTH = 1;       // Single Buffer depth for weights

#define TEMPLATE_DECLARE template<typename T, uint16_t USE_PERMANENT_X>
#define TEMPLATE_ARGS T, USE_PERMANENT_X

TEMPLATE_DECLARE
class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR output, GM_ADDR workspace,
                                const MhcPostTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void ComputeCopyOut(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void ComputeCopyOutAllX(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void CopyOutTile(int64_t bsIdx, int64_t dIdx, int64_t nI, int64_t dNum);
    __aicore__ inline void CopyInX(int64_t bsIdx, int64_t dIdx, int64_t nJ, int64_t dNum);

private:
    TPipe *pipe_;

    // Input queues - Double Buffer enabled (queue depth = DOUBLE_BUFFER_DEPTH)
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> hOutTileQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> xTileQueue_;

    // Output queues - Double Buffer enabled (queue depth = DOUBLE_BUFFER_DEPTH)
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_DEPTH> outputTileQueue_;

    // Intermediate buffers for float computation
    TBuf<QuePosition::VECCALC> hOutF32Buf_;
    TBuf<QuePosition::VECCALC> xF32Buf_;
    TBuf<QuePosition::VECCALC> outF32Buf_;

    // Global memory tensors - inputs (bf16/fp16)
    GlobalTensor<T> xGm_;
    GlobalTensor<T> hOutGm_;
    // Global memory tensors - inputs (float32)
    GlobalTensor<float> hResGm_;
    GlobalTensor<float> hPostGm_;

    // Global memory tensors - outputs (bf16/fp16)
    GlobalTensor<T> outputGm_;

    // Tiling parameters
    int64_t n_;
    int64_t D_;
    int64_t usedCoreNum_;
    int64_t normalCoreProcessNum_;
    int64_t tailCoreProcessNum_;
    int64_t bsInner_;
    int64_t bsOuter_;
    int64_t bsTail_;
    int64_t dInner_;
    int64_t dOuter_;
    int64_t dTail_;

    int64_t xFactor_;
    int64_t myItemCount_;
    int64_t itemStart_;
    uint32_t blockIdx_;
};

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                                                          GM_ADDR output, GM_ADDR workspace,
                                                          const MhcPostTilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;

    // Get tiling data using direct member access
    n_ = tilingData->n;
    D_ = tilingData->D;

    usedCoreNum_ = tilingData->usedCoreNum;
    normalCoreProcessNum_ = tilingData->normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData->tailCoreProcessNum;
    bsInner_ = tilingData->bsInner;
    bsOuter_ = tilingData->bsOuter;
    bsTail_ = tilingData->bsTail;
    dInner_ = tilingData->dInner;
    dOuter_ = tilingData->dOuter;
    dTail_ = tilingData->dTail;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // Calculate work distribution with remainder handling
    if (blockIdx_ < usedCoreNum_ - 1) {
        myItemCount_ = normalCoreProcessNum_;
    } else {
        myItemCount_ = tailCoreProcessNum_;
    }
    itemStart_ = blockIdx_ * normalCoreProcessNum_;

    xFactor_ = 1;
    if constexpr (USE_PERMANENT_X == 1) {
        xFactor_ = n_;
    }

    // Set global memory buffers
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    hOutGm_.SetGlobalBuffer((__gm__ T *)hOut);
    hResGm_.SetGlobalBuffer((__gm__ float *)hRes);
    hPostGm_.SetGlobalBuffer((__gm__ float *)hPost);
    outputGm_.SetGlobalBuffer((__gm__ T *)output);

    // Initialize input queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH for data tiles
    pipe_->InitBuffer(hOutTileQueue_, DOUBLE_BUFFER_DEPTH, dInner_ * sizeof(T));
    pipe_->InitBuffer(xTileQueue_, DOUBLE_BUFFER_DEPTH, xFactor_ * dInner_ * sizeof(T));

    // Initialize output queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH
    pipe_->InitBuffer(outputTileQueue_, DOUBLE_BUFFER_DEPTH, dInner_ * sizeof(T));

    // Initialize intermediate buffers
    pipe_->InitBuffer(hOutF32Buf_, dInner_ * sizeof(float));
    pipe_->InitBuffer(xF32Buf_, xFactor_ * dInner_ * sizeof(float));
    pipe_->InitBuffer(outF32Buf_, dInner_ * sizeof(float));
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t itemIdx = 0; itemIdx < myItemCount_; itemIdx++) {
        int64_t globalItemIdx = itemStart_ + itemIdx;
        int64_t bsIdx = globalItemIdx / dOuter_;
        int64_t dIdx = globalItemIdx - bsIdx * dOuter_;
        int64_t dNum = (dIdx < dOuter_ - 1) ? dInner_ : dTail_;

        CopyInHOut(bsIdx, dIdx, dNum);
        if constexpr (USE_PERMANENT_X == 1) {
            CopyInX(bsIdx, dIdx, 0, dNum);
            ComputeCopyOutAllX(bsIdx, dIdx, dNum);
        } else {
            ComputeCopyOut(bsIdx, dIdx, dNum);
        }
    }
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hOutOffset = bsIdx * D_ + dIdx * dInner_;
    LocalTensor<T> hOutTileLocal = hOutTileQueue_.AllocTensor<T>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(hOutTileLocal, hOutGm_[hOutOffset], copyParams, {false, 0, 0, 0});

    hOutTileQueue_.EnQue(hOutTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::ComputeCopyOut(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hPostBase = bsIdx * n_;
    int64_t hResBase = bsIdx * n_ * n_;

    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();
    LocalTensor<T> outputTile = outputTileQueue_.AllocTensor<T>();

    LocalTensor<float> hOutF32 = hOutF32Buf_.Get<float>();
    LocalTensor<float> outF32 = outF32Buf_.Get<float>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();

    Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, dNum);

    for (int64_t i = 0; i < n_; i++) {
        Muls(outF32, hOutF32, hPostGm_.GetValue(hPostBase + i), dNum);
        for (int64_t j = 0; j < n_; j++) {
            CopyInX(bsIdx, dIdx, j, dNum);
            LocalTensor<T> xTile = xTileQueue_.DeQue<T>();
            Cast(xF32, xTile, RoundMode::CAST_NONE, dNum);
            Axpy(outF32, xF32, hResGm_.GetValue(hResBase + j * n_ + i), dNum);
            xTileQueue_.FreeTensor(xTile);
        }

        Cast(outputTile, outF32, RoundMode::CAST_RINT, dNum);
        outputTileQueue_.EnQue(outputTile);
        CopyOutTile(bsIdx, dIdx, i, dNum);
    }

    hOutTileQueue_.FreeTensor(hOutTile);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::ComputeCopyOutAllX(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hPostBase = bsIdx * n_;
    int64_t hResBase = bsIdx * n_ * n_;
    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();
    LocalTensor<T> xTile = xTileQueue_.DeQue<T>();
    LocalTensor<T> outputTile = outputTileQueue_.AllocTensor<T>();

    LocalTensor<float> hOutF32 = hOutF32Buf_.Get<float>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();
    LocalTensor<float> outF32 = outF32Buf_.Get<float>();

    Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, dNum);
    Cast(xF32, xTile, RoundMode::CAST_NONE, n_ * dNum);

    for (int64_t i = 0; i < n_; i++) {
        Muls(outF32, hOutF32, hPostGm_.GetValue(hPostBase + i), dNum);
        for (int64_t j = 0; j < n_; j++) {
            Axpy(outF32, xF32[j * dNum], hResGm_.GetValue(hResBase + j * n_ + i), dNum);
        }

        Cast(outputTile, outF32, RoundMode::CAST_RINT, dNum);
        outputTileQueue_.EnQue(outputTile);
        CopyOutTile(bsIdx, dIdx, i, dNum);
    }

    hOutTileQueue_.FreeTensor(hOutTile);
    xTileQueue_.FreeTensor(xTile);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInX(int64_t bsIdx, int64_t dIdx, int64_t nJ, int64_t dNum)
{
    int64_t dStart = dIdx * dInner_;
    int64_t xBase = bsIdx * n_ * D_ + nJ * D_;
    int64_t xOffset = xBase + dStart;
    LocalTensor<T> xTileLocal = xTileQueue_.AllocTensor<T>();

    if constexpr (USE_PERMANENT_X == 1) {
        DataCopyExtParams copyParams = {static_cast<uint16_t>(n_), static_cast<uint32_t>(dNum * sizeof(T)),
                                        static_cast<uint32_t>((D_ - dNum) * sizeof(T)), 0, 0};
        DataCopyPad(xTileLocal, xGm_[xOffset], copyParams, {false, 0, 0, 0});
    } else {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(xTileLocal, xGm_[xOffset], copyParams, {false, 0, 0, 0});
    }

    xTileQueue_.EnQue(xTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyOutTile(int64_t bsIdx, int64_t dIdx, int64_t nI, int64_t dNum)
{
    int64_t dStart = dIdx * dInner_;
    int64_t outputBase = bsIdx * n_ * D_ + nI * D_;
    int64_t outputOffset = outputBase + dStart;
    LocalTensor<T> outputTile = outputTileQueue_.DeQue<T>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dTail_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm_[outputOffset], outputTile, copyParams);

    outputTileQueue_.FreeTensor(outputTile);
}

}  // namespace MhcPost

#endif  // ASCENDC_MHC_POST_H