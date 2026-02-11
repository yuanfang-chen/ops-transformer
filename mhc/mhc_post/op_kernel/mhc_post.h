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
 * Formula: y = (H_res)^T * x + h_out * h_post
 */

#ifndef ASCENDC_MHC_POST_H
#define ASCENDC_MHC_POST_H

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace MhcPost {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_NUM_FP16 = 16;    // 16 elements per block for FP16 (32 bytes)
constexpr uint32_t BLOCK_NUM_BF16 = 16;    // 16 elements per block for BF16 (32 bytes)
constexpr uint32_t BLOCK_NUM_FP32 = 8;     // 8 elements per block for FP32 (32 bytes)

template <typename T, typename T_FP32>
class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel() {}
    __aicore__ inline ~MhcPostKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                GM_ADDR h_post, GM_ADDR y, const MhcPostTilingData *__restrict tilingData)
    {
        this->totalLength = tilingData->total_length;
        this->coreNum = tilingData->core_num;
        this->singleCoreLength = tilingData->single_core_length;

        // Calculate this core's range
        uint32_t coreId = GetBlockIdx();
        this->startOffset = coreId * this->singleCoreLength;
        if (coreId == this->coreNum - 1) {
            this->endOffset = this->totalLength;
        } else {
            this->endOffset = (coreId + 1) * this->singleCoreLength;
        }
        if (this->endOffset > this->totalLength) {
            this->endOffset = this->totalLength;
        }

        // Initialize global buffers
        xGm.SetGlobalBuffer((__gm__ T *)x, this->totalLength);
        hResGm.SetGlobalBuffer((__gm__ T_FP32 *)h_res, this->totalLength);
        hOutGm.SetGlobalBuffer((__gm__ T *)h_out, this->totalLength);
        hPostGm.SetGlobalBuffer((__gm__ T_FP32 *)h_post, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ T *)y, this->totalLength);

        // Calculate tile length
        // UB size: 192KB on Ascend950
        constexpr uint32_t UB_SIZE = 192 * 1024;
        // We have 5 tensors: x, h_out (T), y (T) and h_res, h_post (FP32)
        // Total: 3 * sizeof(T) + 2 * sizeof(float)
        uint32_t usableUbSize = UB_SIZE;
        uint32_t bytesPerElement = 3 * sizeof(T) + 2 * sizeof(T_FP32);
        this->tileLength = usableUbSize / bytesPerElement / BLOCK_NUM_FP16 * BLOCK_NUM_FP16;

        if (this->tileLength == 0) {
            this->tileLength = BLOCK_NUM_FP16;
        }
        if (this->tileLength > 8192) {
            this->tileLength = 8192;
        }

        // Initialize queues
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(hResQueue, BUFFER_NUM, tileLength * sizeof(T_FP32));
        pipe.InitBuffer(hOutQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(hPostQueue, BUFFER_NUM, tileLength * sizeof(T_FP32));
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLength * sizeof(T));

        // Initialize TBuf for temp storage
        pipe.InitBuffer(tmpBuf, tileLength * sizeof(T_FP32));
    }

    __aicore__ inline void Process()
    {
        if (startOffset >= endOffset) {
            return;
        }

        uint32_t offset = startOffset;
        while (offset < endOffset) {
            uint32_t currentProcessLen = (tileLength < (endOffset - offset)) ? tileLength : (endOffset - offset);

            // CopyIn
            CopyIn(offset, currentProcessLen);

            // Compute: y = (H_res)^T * x + h_out * h_post
            Compute(currentProcessLen);

            // CopyOut
            CopyOut(offset, currentProcessLen);

            offset += currentProcessLen;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t currentProcessLen)
    {
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        LocalTensor<T_FP32> hResLocal = hResQueue.AllocTensor<T_FP32>();
        LocalTensor<T> hOutLocal = hOutQueue.AllocTensor<T>();
        LocalTensor<T_FP32> hPostLocal = hPostQueue.AllocTensor<T_FP32>();

        DataCopy(xLocal, xGm[offset], currentProcessLen);
        DataCopy(hResLocal, hResGm[offset], currentProcessLen);
        DataCopy(hOutLocal, hOutGm[offset], currentProcessLen);
        DataCopy(hPostLocal, hPostGm[offset], currentProcessLen);

        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        xQueue.EnQue(xLocal);
        hResQueue.EnQue(hResLocal);
        hOutQueue.EnQue(hOutLocal);
        hPostQueue.EnQue(hPostLocal);
    }

    __aicore__ inline void Compute(uint32_t currentProcessLen)
    {
        LocalTensor<T> xLocal = xQueue.DeQue<T>();
        LocalTensor<T_FP32> hResLocal = hResQueue.DeQue<T_FP32>();
        LocalTensor<T> hOutLocal = hOutQueue.AllocTensor<T>();
        LocalTensor<T_FP32> hPostLocal = hPostQueue.DeQue<T_FP32>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
        LocalTensor<T_FP32> tmpLocal = tmpBuf.Get<T_FP32>();

        // Step 1: Compute h_res * x (element-wise)
        Muls(hResLocal, hResLocal, xLocal, currentProcessLen);
        PipeBarrier<PIPE_V>();

        // Step 2: Cast result to FP16/BF16 for y
        Cast(yLocal, hResLocal, RoundMode::ROUND_HALF_TO_EVEN, currentProcessLen);
        PipeBarrier<PIPE_V>();

        // Step 3: Compute h_out * h_post
        // Cast h_out to FP32
        Cast(tmpLocal, hOutLocal, RoundMode::ROUND_HALF_TO_EVEN, currentProcessLen);
        PipeBarrier<PIPE_V>();

        // Element-wise multiply with h_post
        Mul(tmpLocal, tmpLocal, hPostLocal, currentProcessLen);
        PipeBarrier<PIPE_V>();

        // Step 4: Add to y
        // Cast tmp result back to T
        LocalTensor<T> tmpTLocal = tmpLocal.template ReinterpretCast<T>();
        Add(yLocal, yLocal, tmpTLocal, currentProcessLen);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        xQueue.FreeTensor(xLocal);
        hResQueue.FreeTensor(hResLocal);
        hPostQueue.FreeTensor(hPostLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t currentProcessLen)
    {
        LocalTensor<T> yLocal = yQueue.DeQue<T>();

        DataCopy(yGm[offset], yLocal, currentProcessLen);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        yQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECCALC, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> hResQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> hOutQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> hPostQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    TBuf<QuePosition::VECCALC> tmpBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T_FP32> hResGm;
    GlobalTensor<T> hOutGm;
    GlobalTensor<T_FP32> hPostGm;
    GlobalTensor<T> yGm;

    uint32_t totalLength;
    uint32_t coreNum;
    uint32_t singleCoreLength;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t endOffset;
};

} // namespace MhcPost

#endif // ASCENDC_MHC_POST_H