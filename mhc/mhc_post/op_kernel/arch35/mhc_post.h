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

// Constants for memory alignment and buffer configuration
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16;     // 16 elements = 32 bytes for bf16/fp16
constexpr uint32_t FLOAT32_ALIGN_SIZE = 8;         // 8 elements = 32 bytes for float32

// Double Buffer configuration - Double Buffer提升Memory Bound算子性能
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2;       // Double Buffer depth for data tiles
constexpr uint32_t SINGLE_BUFFER_DEPTH = 1;       // Single Buffer depth for weights

// Template macro definitions - 用于编译时优化,类似DequantSwiGLU的quantIsOne设计
// IS_D_ALIGNED: D % 16 == 0 && nTilesD == 1
#define TEMPLATE_DECLARE template<typename T, uint16_t IS_D_ALIGNED>
#define TEMPLATE_ARGS T, IS_D_ALIGNED

TEMPLATE_DECLARE
class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                                GM_ADDR output, GM_ADDR workspace,
                                const MhcPostTilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInTile(uint32_t globalItemIdx, uint32_t tileId);
    __aicore__ inline void ComputeTile(uint32_t globalItemIdx, uint32_t tileId);
    __aicore__ inline void CopyOutTile(uint32_t globalItemIdx, uint32_t tileId, uint32_t nI);
    __aicore__ inline void CopyXTile(uint32_t globalItemIdx, uint32_t tileId, uint32_t nI, uint32_t nJ, LocalTensor<float> outF32);

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
    uint32_t totalItems_;
    uint32_t itemsPerCore_;
    uint32_t remainderItems_;
    uint32_t usedCores_;
    uint32_t S_;
    uint32_t n_;
    uint32_t D_;
    uint32_t tileD_;
    uint32_t nTilesD_;
    uint32_t alignedD_;
    uint32_t lastTileD_;

    uint32_t blockIdx_;
    uint32_t myItemCount_;
    uint32_t itemStart_;
};

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::Init(
    GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR output, GM_ADDR workspace,
    const MhcPostTilingData *tilingData,
    TPipe *tPipe)
{
    pipe_ = tPipe;

    // Get tiling data using direct member access
    totalItems_ = tilingData->totalItems;
    itemsPerCore_ = tilingData->itemsPerCore;
    remainderItems_ = tilingData->remainderItems;
    usedCores_ = tilingData->usedCores;
    S_ = tilingData->S;
    n_ = tilingData->n;
    D_ = tilingData->D;
    tileD_ = tilingData->tileD;
    nTilesD_ = tilingData->nTilesD;
    alignedD_ = tilingData->alignedD;
    lastTileD_ = tilingData->lastTileD;

    blockIdx_ = GetBlockIdx();

    // Calculate work distribution with remainder handling
    if (blockIdx_ < remainderItems_) {
        myItemCount_ = itemsPerCore_ + 1;
        itemStart_ = blockIdx_ * (itemsPerCore_ + 1);
    } else {
        myItemCount_ = itemsPerCore_;
        itemStart_ = remainderItems_ * (itemsPerCore_ + 1) + (blockIdx_ - remainderItems_) * itemsPerCore_;
    }

    // Set global memory buffers
    xGm_.SetGlobalBuffer((__gm__ T *)x, totalItems_ * n_ * D_);
    hOutGm_.SetGlobalBuffer((__gm__ T *)hOut, totalItems_ * D_);
    hResGm_.SetGlobalBuffer((__gm__ float *)hRes, totalItems_ * n_ * n_);
    hPostGm_.SetGlobalBuffer((__gm__ float *)hPost, totalItems_ * n_);
    outputGm_.SetGlobalBuffer((__gm__ T *)output, totalItems_ * n_ * D_);

    // Initialize input queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH for data tiles
    pipe_->InitBuffer(hOutTileQueue_, DOUBLE_BUFFER_DEPTH, tileD_ * sizeof(T));
    pipe_->InitBuffer(xTileQueue_, DOUBLE_BUFFER_DEPTH, tileD_ * sizeof(T));

    // Initialize output queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH
    pipe_->InitBuffer(outputTileQueue_, DOUBLE_BUFFER_DEPTH, tileD_ * sizeof(T));

    // Initialize intermediate buffers
    pipe_->InitBuffer(hOutF32Buf_, tileD_ * sizeof(float));
    pipe_->InitBuffer(xF32Buf_, tileD_ * sizeof(float));
    pipe_->InitBuffer(outF32Buf_, tileD_ * sizeof(float));
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::Process()
{
    for (uint32_t itemIdx = 0; itemIdx < myItemCount_; itemIdx++) {
        uint32_t globalItemIdx = itemStart_ + itemIdx;

        for (uint32_t tileId = 0; tileId < nTilesD_; tileId++) {
            CopyInTile(globalItemIdx, tileId);
            ComputeTile(globalItemIdx, tileId);
        }
    }
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInTile(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint32_t hOutBase = globalItemIdx * D_;

    LocalTensor<T> hOutTileLocal = hOutTileQueue_.AllocTensor<T>();

    if constexpr (IS_D_ALIGNED == 1) {
        // Fast path: D is aligned and single tile, use direct DataCopy
        DataCopy(hOutTileLocal, hOutGm_[hOutBase], tileD_);
    } else {
        // Slow path: multi-tile or non-aligned
        // Calculate how much data to actually copy from GM for this tile
        uint32_t remainingD = D_ - dStart;
        uint32_t gmCopyD = (remainingD >= tileD_) ? tileD_ : remainingD;

        // Load hOut tile
        if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
            DataCopy(hOutTileLocal, hOutGm_[hOutBase + dStart], gmCopyD);
        } else {
            // Non-aligned: use DataCopyPad with right padding
            uint32_t padSize = tileD_ - gmCopyD;
            uint8_t rightPad = (padSize > 255) ? 255 : static_cast<uint8_t>(padSize);
            DataCopyPadParams padParams = {true, 0, rightPad, 0};
            DataCopyPad(hOutTileLocal, hOutGm_[hOutBase + dStart],
                       {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0}, padParams);
        }
    }

    hOutTileQueue_.EnQue(hOutTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyXTile(uint32_t globalItemIdx, uint32_t tileId, uint32_t nI, uint32_t nJ, LocalTensor<float> outF32)
{
    uint32_t dStart = tileId * tileD_;
    uint32_t hResBase = globalItemIdx * n_ * n_;
    uint32_t xBase = globalItemIdx * n_ * D_;
    LocalTensor<T> xTileLocal = xTileQueue_.AllocTensor<T>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();

    if constexpr (IS_D_ALIGNED == 1) {
        // Fast path: D is aligned and single tile, use direct DataCopy
        DataCopy(xTileLocal, xGm_[xBase + nJ * tileD_], tileD_);
    } else {
        // Slow path: multi-tile or non-aligned
        // Calculate how much data to actually copy from GM for this tile
        uint32_t remainingD = D_ - dStart;
        uint32_t gmCopyD = (remainingD >= tileD_) ? tileD_ : remainingD;

        // Load x row by row
        if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
            // Aligned, use DataCopy row by row
            DataCopy(xTileLocal, xGm_[xBase + dStart + nJ * tileD_], gmCopyD);
        } else {
            // Non-aligned, use DataCopyPad row by row
            uint32_t padSize = tileD_ - gmCopyD;
            uint8_t rightPad = (padSize > 255) ? 255 : static_cast<uint8_t>(padSize);
            DataCopyPadParams padParams = {true, 0, rightPad, 0};
            DataCopyPad(xTileLocal, xGm_[xBase + dStart + nJ * tileD_],
                       {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0}, padParams);
        }
    }
    xTileQueue_.EnQue(xTileLocal);

    LocalTensor<T> xTile = xTileQueue_.DeQue<T>();
    float hResJI = hResGm_.GetValue(hResBase + nJ * n_ + nI);  // hRes[j,i] instead of hRes[i,j]
    Cast(xF32, xTile, RoundMode::CAST_NONE, tileD_);
    Axpy(outF32, xF32, hResJI, tileD_);

    xTileQueue_.FreeTensor(xTile);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::ComputeTile(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t hPostBase = globalItemIdx * n_;

    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();
    LocalTensor<T> outputTile = outputTileQueue_.AllocTensor<T>();

    // Get float32 work buffers
    LocalTensor<float> hOutF32 = hOutF32Buf_.Get<float>();
    LocalTensor<float> outF32 = outF32Buf_.Get<float>();

    // Convert inputs to float32
    Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, tileD_);

    // Compute output for each head: output[i] = hPost[i] * hOut + sum_j(hRes[j,i] * x[j])
    // This implements: x_{l+1}[i] = h_{l}^{out} * H_{t}^{post}[i] + sum_j((H_{l}^{res})^{T}[j,i] * x_l[j])
    // Note: hRes indexing is [j,i] to access transposed matrix element (H_res)^T[j,i]
    for (uint32_t i = 0; i < n_; i++) {
        // outF32 = hPost[i] * hOut
        Muls(outF32, hOutF32, hPostGm_.GetValue(hPostBase + i), tileD_);

        // outF32 += sum_j(hRes[j,i] * x[j])
        for (uint32_t j = 0; j < n_; j++) {
            CopyXTile(globalItemIdx, tileId, i, j, outF32);
        }

        // Convert to bf16/fp16 and store
        Cast(outputTile, outF32, RoundMode::CAST_RINT, tileD_);
        outputTileQueue_.EnQue(outputTile);
        CopyOutTile(globalItemIdx, tileId, i);
    }

    hOutTileQueue_.FreeTensor(hOutTile);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyOutTile(uint32_t globalItemIdx, uint32_t tileId, uint32_t nI)
{
    uint32_t dStart = tileId * tileD_;
    uint32_t outputBase = globalItemIdx * n_ * D_;

    LocalTensor<T> outputTile = outputTileQueue_.DeQue<T>();

    if constexpr (IS_D_ALIGNED == 1) {
        // Fast path: D is aligned and single tile
        DataCopy(outputGm_[outputBase + nI * D_], outputTile, tileD_);
    } else {
        // Slow path: multi-tile or non-aligned
        uint32_t remainingD = D_ - dStart;
        uint32_t gmCopyD = (remainingD >= tileD_) ? tileD_ : remainingD;

        // Copy output row by row
        if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
            DataCopy(outputGm_[outputBase + dStart + nI * D_], outputTile, gmCopyD);
        } else {
            DataCopyPad(outputGm_[outputBase + dStart + nI * D_], outputTile,
                       {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0});
        }
    }

    outputTileQueue_.FreeTensor(outputTile);
}

} // namespace MhcPost

#endif // ASCENDC_MHC_POST_H