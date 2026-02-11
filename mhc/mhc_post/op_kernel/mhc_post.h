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
 */

#ifndef ASCENDC_MHC_POST_H
#define ASCENDC_MHC_POST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace MhcPost {
using namespace AscendC;

constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32; // 32 bytes alignment
constexpr uint32_t BLOCK_SIZE_FP16 = 32;     // 32 elements per block for FP16 (32 * 2 = 64 bytes)

template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t dataLen)
{
    return (dataLen + BLOCK_SIZE_FP16 - 1) / BLOCK_SIZE_FP16;
}

template <typename T>
__aicore__ inline uint32_t GetAlignLength(uint32_t dataLen)
{
    return GetBlockNum<T>(dataLen) * BLOCK_SIZE_FP16;
}

/**
 * @brief Get the tile length for each iteration
 * @param totalLength Total elements to process
 * @param ubSize Unified Buffer size in bytes
 * @return Tile length in elements
 */
template <typename T>
__aicore__ inline uint32_t GetTileLength(uint32_t totalLength, uint32_t ubSize)
{
    // Calculate available UB size: we have 5 tensors (x, h_res, h_out, h_post, y)
    uint32_t totalTensors = 5;
    uint32_t usableUbSize = ubSize / totalTensors;
    uint32_t maxElements = usableUbSize / sizeof(T);

    // Align to block size
    uint32_t alignedMaxElements = (maxElements / BLOCK_SIZE_FP16) * BLOCK_SIZE_FP16;

    // Return min of totalLength and alignedMaxElements
    return totalLength < alignedMaxElements ? totalLength : alignedMaxElements;
}

template <typename T>
__aicore__ inline void ProcessOneTile(LocalTensor<T> &yLocal, LocalTensor<T> &xLocal,
                                       LocalTensor<T> &hResLocal, LocalTensor<T> &hOutLocal,
                                       LocalTensor<T> &hPostLocal, float alpha, float beta,
                                       float gamma, uint32_t processCount)
{
    // Compute y = x + alpha * h_res + beta * h_out + gamma * h_post
    // Step 1: Copy x to y
    DataCopy(yLocal, xLocal, processCount);
    PipeBarrier<PIPE_V>();

    // Step 2: y = y + alpha * h_res
    Muls(hResLocal, hResLocal, alpha, processCount);
    Add(yLocal, yLocal, hResLocal, processCount);
    PipeBarrier<PIPE_V>();

    // Step 3: y = y + beta * h_out
    Muls(hOutLocal, hOutLocal, beta, processCount);
    Add(yLocal, yLocal, hOutLocal, processCount);
    PipeBarrier<PIPE_V>();

    // Step 4: y = y + gamma * h_post
    Muls(hPostLocal, hPostLocal, gamma, processCount);
    Add(yLocal, yLocal, hPostLocal, processCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ProcessHalfPreciseTile(LocalTensor<T> &yLocal, LocalTensor<T> &xLocal,
                                                LocalTensor<T> &hResLocal, LocalTensor<T> &hOutLocal,
                                                LocalTensor<T> &hPostLocal, float alpha, float beta,
                                                float gamma, uint32_t processCount, LocalTensor<float> &tmp)
{
    // Cast to float for high precision computation
    LocalTensor<float> yLocalF32 = yLocal.template ReinterpretCast<float>();
    LocalTensor<float> xLocalF32 = xLocal.template ReinterpretCast<float>();
    LocalTensor<float> hResLocalF32 = hResLocal.template ReinterpretCast<float>();
    LocalTensor<float> hOutLocalF32 = hOutLocal.template ReinterpretCast<float>();
    LocalTensor<float> hPostLocalF32 = hPostLocal.template ReinterpretCast<float>();

    uint32_t f32Count = processCount / 2; // FP16 to FP32 reduces element count

    // Compute y = x + alpha * h_res + beta * h_out + gamma * h_post in float32
    // Step 1: Copy x to y
    DataCopy(yLocalF32, xLocalF32, f32Count);
    PipeBarrier<PIPE_V>();

    // Step 2: tmp = alpha * h_res
    Muls(tmp, hResLocalF32, alpha, f32Count);
    Add(yLocalF32, yLocalF32, tmp, f32Count);
    PipeBarrier<PIPE_V>();

    // Step 3: tmp = beta * h_out
    Muls(tmp, hOutLocalF32, beta, f32Count);
    Add(yLocalF32, yLocalF32, tmp, f32Count);
    PipeBarrier<PIPE_V>();

    // Step 4: tmp = gamma * h_post
    Muls(tmp, hPostLocalF32, gamma, f32Count);
    Add(yLocalF32, yLocalF32, tmp, f32Count);
    PipeBarrier<PIPE_V>();
}

} // namespace MhcPost

#endif // ASCENDC_MHC_POST_H