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
 * \file mhc_post.cpp
 * \brief MhcPost kernel entry point
 * Formula: y = (H_res)^T * x + h_out * h_post
 */

#include "mhc_post.h"
#include "mhc_post_tiling.h"

using namespace AscendC;

template <typename T, typename T_FP32>
class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel() {}
    __aicore__ inline ~MhcPostKernel() {}

    /**
     * @brief Initialize kernel
     * @param x Input x [batch, ..., M] - FP16/BF16
     * @param h_res Residual matrix [M, K] or [M] - FP32
     * @param h_out Output from mhc_pre [same as x] - FP16/BF16
     * @param h_post Post value [same as x or scalar] - FP32
     * @param y Output [same as x] - FP16/BF16
     * @param totalLength Total elements in x
     * @param coreNum Number of cores
     * @param singleCoreLength Elements per core
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                GM_ADDR h_post, GM_ADDR y, uint32_t totalLength, uint32_t coreNum,
                                uint32_t singleCoreLength)
    {
        this->totalLength = totalLength;

        // Calculate this core's range
        uint32_t coreId = GetBlockIdx();
        this->startOffset = coreId * singleCoreLength;
        if (coreId == coreNum - 1) {
            this->endOffset = totalLength;
        } else {
            this->endOffset = (coreId + 1) * singleCoreLength;
        }
        if (this->endOffset > totalLength) {
            this->endOffset = totalLength;
        }

        // Initialize global buffers
        xGm.SetGlobalBuffer((__gm__ T *)x, totalLength);
        hResGm.SetGlobalBuffer((__gm__ T_FP32 *)h_res, totalLength); // Assume h_res at least same size as x
        hOutGm.SetGlobalBuffer((__gm__ T *)h_out, totalLength);
        hPostGm.SetGlobalBuffer((__gm__ T_FP32 *)h_post, totalLength);
        yGm.SetGlobalBuffer((__gm__ T *)y, totalLength);

        // Calculate tile length based on UB size (192KB = 192 * 1024 bytes on Ascend950)
        constexpr uint32_t UB_SIZE = 192 * 1024;
        // We have 5 tensors: x, h_out (T), y (T) and h_res, h_post (FP32)
        // Total T tensors: x, h_out, y = 3 * sizeof(T)
        // Total FP32 tensors: h_res, h_post = 2 * sizeof(float)
        uint32_t usableUbSize = UB_SIZE;
        this->tileLength = (usableUbSize / (3 * sizeof(T) + 2 * sizeof(T_FP32))) / BLOCK_NUM_FP16 * BLOCK_NUM_FP16;
        if (this->tileLength == 0) {
            this->tileLength = BLOCK_NUM_FP16;
        }
        if (this->tileLength > 8192) {
            this->tileLength = 8192; // Max tile for better performance
        }

        // Initialize queues and buffers
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(hResQueue, BUFFER_NUM, tileLength * sizeof(T_FP32));
        pipe.InitBuffer(hOutQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(hPostQueue, BUFFER_NUM, tileLength * sizeof(T_FP32));
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLength * sizeof(T));

        // Initialize TBuf for temp storage (FP32 for h_out * h_post calculation)
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
        LocalTensor<T> hOutLocal = hOutQueue.EnQue<T>();
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
        Muls(tmpTLocal, tmpTLocal, static_cast<float>(1.0f), currentProcessLen); // Convert FP32*2 to T
        PipeBarrier<PIPE_V>();

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
    GlobalTensor<T> xGm;
    GlobalTensor<T_FP32> hResGm;
    GlobalTensor<T> hOutGm;
    GlobalTensor<T_FP32> hPostGm;
    GlobalTensor<T> yGm;

    TQue<QuePosition::VECCALC, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> hResQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> hOutQueue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> hPostQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    TBuf<QuePosition::VECCALC> tmpBuf;

    TPipe pipe;

    uint32_t totalLength;
    uint32_t startOffset;
    uint32_t endOffset;
    uint32_t tileLength;

    static constexpr uint32_t BLOCK_NUM_FP16 = 16;
};

// FP16 version
extern "C" __global__ __aicore__ void MhcPostFP16(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                                    GM_ADDR h_post, GM_ADDR y, GM_ADDR tiling,
                                                    GM_ADDR workspace)
{
    (void)workspace;

    optiling::MhcPostTiling tilingData;
    tilingData.set_totalLength(*(reinterpret_cast<uint32_t *>(tiling)));
    tilingData.set_coreNum(*(reinterpret_cast<uint32_t *>(tiling) + 1));
    tilingData.set_singleCoreLength(*(reinterpret_cast<uint32_t *>(tiling) + 2));

    MhcPostKernel<half, float> op;
    op.Init(x, h_res, h_out, h_post, y, tilingData.get_totalLength(),
            tilingData.get_coreNum(), tilingData.get_singleCoreLength());
    op.Process();
}

// BF16 version
extern "C" __global__ __aicore__ void MhcPostBF16(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                                    GM_ADDR h_post, GM_ADDR y, GM_ADDR tiling,
                                                    GM_ADDR workspace)
{
    (void)workspace;

    optiling::MhcPostTiling tilingData;
    tilingData.set_totalLength(*(reinterpret_cast<uint32_t *>(tiling)));
    tilingData.set_coreNum(*(reinterpret_cast<uint32_t *>(tiling) + 1));
    tilingData.set_singleCoreLength(*(reinterpret_cast<uint32_t *>(tiling) + 2));

    MhcPostKernel<bfloat16_t, float> op;
    op.Init(x, h_res, h_out, h_post, y, tilingData.get_totalLength(),
            tilingData.get_coreNum(), tilingData.get_singleCoreLength());
    op.Process();
}