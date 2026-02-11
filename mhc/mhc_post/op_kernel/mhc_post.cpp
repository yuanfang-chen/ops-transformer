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
 */

#include "mhc_post.h"
#include "mhc_post_tiling.h"

using namespace AscendC;

template <typename T>
class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel() {}
    __aicore__ inline ~MhcPostKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out, GM_ADDR h_post,
                                GM_ADDR y, uint32_t totalLength, uint32_t coreNum,
                                uint32_t singleCoreLength, float alpha, float beta, float gamma)
    {
        this->totalLength = totalLength;
        this->alpha = alpha;
        this->beta = beta;
        this->gamma = gamma;

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
        hResGm.SetGlobalBuffer((__gm__ T *)h_res, totalLength);
        hOutGm.SetGlobalBuffer((__gm__ T *)h_out, totalLength);
        hPostGm.SetGlobalBuffer((__gm__ T *)h_post, totalLength);
        yGm.SetGlobalBuffer((__gm__ T *)y, totalLength);

        // Calculate tile length based on UB size
        // UB size is 192KB = 192 * 1024 bytes for Ascend910B3
        constexpr uint32_t UB_SIZE = 192 * 1024;
        this->tileLength = MhcPost::GetTileLength<T>(endOffset - startOffset, UB_SIZE);
        if (this->tileLength == 0) {
            this->tileLength = BLOCK_SIZE_FP16;
        }

        // Initialize queues and buffers
        pipe.InitBuffer(xQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(hResQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(hOutQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(hPostQueue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM, tileLength * sizeof(T));

        // Initialize TBuf for temporary storage (for float32 precision)
        pipe.InitBuffer(tmpBuf, tileLength * sizeof(T) * 2); // 2x for float32 precision
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

            // Compute
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
        LocalTensor<T> hResLocal = hResQueue.AllocTensor<T>();
        LocalTensor<T> hOutLocal = hOutQueue.AllocTensor<T>();
        LocalTensor<T> hPostLocal = hPostQueue.AllocTensor<T>();

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
        LocalTensor<T> hResLocal = hResQueue.DeQue<T>();
        LocalTensor<T> hOutLocal = hOutQueue.DeQue<T>();
        LocalTensor<T> hPostLocal = hPostQueue.DeQue<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();

        // Compute y = x + alpha * h_res + beta * h_out + gamma * h_post
        MhcPost::ProcessOneTile<T>(yLocal, xLocal, hResLocal, hOutLocal, hPostLocal,
                                    alpha, beta, gamma, currentProcessLen);

        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        xQueue.FreeTensor(xLocal);
        hResQueue.FreeTensor(hResLocal);
        hOutQueue.FreeTensor(hOutLocal);
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
    GlobalTensor<T> hResGm;
    GlobalTensor<T> hOutGm;
    GlobalTensor<T> hPostGm;
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

    float alpha;
    float beta;
    float gamma;
};

extern "C" __global__ __aicore__ void MhcPostFP16(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                                    GM_ADDR h_post, GM_ADDR y, GM_ADDR tiling,
                                                    GM_ADDR workspace)
{
    (void)workspace;

    optiling::MhcPostTiling tilingData;
    tilingData.set_totalLength(*(reinterpret_cast<uint32_t *>(tiling)));
    tilingData.set_coreNum(*(reinterpret_cast<uint32_t *>(tiling) + 1));
    tilingData.set_singleCoreLength(*(reinterpret_cast<uint32_t *>(tiling) + 2));

    float alpha = 1.0f;
    float beta = 1.0f;
    float gamma = 1.0f;

    MhcPostKernel<half> op;
    op.Init(x, h_res, h_out, h_post, y, tilingData.get_totalLength(),
            tilingData.get_coreNum(), tilingData.get_singleCoreLength(),
            alpha, beta, gamma);
    op.Process();
}

extern "C" __global__ __aicore__ void MhcPostBF16(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                                    GM_ADDR h_post, GM_ADDR y, GM_ADDR tiling,
                                                    GM_ADDR workspace)
{
    (void)workspace;

    optiling::MhcPostTiling tilingData;
    tilingData.set_totalLength(*(reinterpret_cast<uint32_t *>(tiling)));
    tilingData.set_coreNum(*(reinterpret_cast<uint32_t *>(tiling) + 1));
    tilingData.set_singleCoreLength(*(reinterpret_cast<uint32_t *>(tiling) + 2));

    float alpha = 1.0f;
    float beta = 1.0f;
    float gamma = 1.0f;

    MhcPostKernel<bfloat16_t> op;
    op.Init(x, h_res, h_out, h_post, y, tilingData.get_totalLength(),
            tilingData.get_coreNum(), tilingData.get_singleCoreLength(),
            alpha, beta, gamma);
    op.Process();
}

extern "C" __global__ __aicore__ void MhcPostFloat(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                                    GM_ADDR h_post, GM_ADDR y, GM_ADDR tiling,
                                                    GM_ADDR workspace)
{
    (void)workspace;

    optiling::MhcPostTiling tilingData;
    tilingData.set_totalLength(*(reinterpret_cast<uint32_t *>(tiling)));
    tilingData.set_coreNum(*(reinterpret_cast<uint32_t *>(tiling) + 1));
    tilingData.set_singleCoreLength(*(reinterpret_cast<uint32_t *>(tiling) + 2));

    float alpha = 1.0f;
    float beta = 1.0f;
    float gamma = 1.0f;

    MhcPostKernel<float> op;
    op.Init(x, h_res, h_out, h_post, y, tilingData.get_totalLength(),
            tilingData.get_coreNum(), tilingData.get_singleCoreLength(),
            alpha, beta, gamma);
    op.Process();
}