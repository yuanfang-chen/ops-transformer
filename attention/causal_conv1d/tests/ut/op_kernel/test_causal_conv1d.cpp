/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_causal_conv1d.cpp
 * \brief
 */

#include "../../../op_kernel/causal_conv1d.cpp"
#include "causal_conv1d_tiling.h"
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

class causal_conv1d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "causal_conv1d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "causal_conv1d_test TearDown\n" << endl;
    }
};

template <typename T>
static inline T FromFloat(float v)
{
    return static_cast<T>(v);
}

template <typename T>
static inline float ToFloat(T v)
{
    return static_cast<float>(v);
}

static float SiluRef(float x)
{
    return x / (1.0f + std::exp(-x));
}

// Reference implementation (weight layout: (width, dim), dim contiguous)
// Access: weight[j, c] = weight[j * dim + c]
static void ReferenceCausalConv1dFwdBatch(const float* x, const float* weight, const float* bias, float* y,
                                         float* convStates, int32_t dim, int32_t width, int32_t stateLen,
                                         const int32_t* queryStartLoc, int32_t batch, const int32_t* cacheIndices,
                                         const bool* hasInitialState, int32_t activationMode, int32_t padSlotId)
{
    for (int32_t seq = 0; seq < batch; ++seq) {
        const int32_t start = queryStartLoc[seq];
        const int32_t end = queryStartLoc[seq + 1];
        const int32_t len = end - start;
        if (len <= 0) {
            continue;
        }
        const int32_t cacheIdx = cacheIndices[seq];
        if (cacheIdx == padSlotId) {
            continue;
        }

        std::vector<float> hist(width - 1, 0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            if (hasInitialState[seq]) {
                for (int32_t h = 0; h < width - 1; ++h) {
                    hist[h] = convStates[(cacheIdx * stateLen + h) * dim + c];
                }
            } else {
                std::fill(hist.begin(), hist.end(), 0.0f);
            }

            for (int32_t t = 0; t < len; ++t) {
                float acc = bias != nullptr ? bias[c] : 0.0f;
                for (int32_t j = 0; j < width; ++j) {
                    const int32_t srcT = t - j;
                    float xval = 0.0f;
                    if (srcT >= 0) {
                        xval = x[(start + srcT) * dim + c];
                    } else {
                        const int32_t histIndex = (width - 1) + srcT; // srcT is negative
                        xval = hist[histIndex];
                    }
                    acc += weight[j * dim + c] * xval;
                }
                if (activationMode != 0) {
                    acc = SiluRef(acc);
                }
                y[(start + t) * dim + c] = acc;
            }

            // state writeback: new_state = tail_{width-1}(concat(hist, X))
            for (int32_t s = 0; s < width - 1; ++s) {
                const int32_t idxInConcat = len + s;
                float v = 0.0f;
                if (idxInConcat < (width - 1)) {
                    v = hist[idxInConcat];
                } else {
                    const int32_t tok = idxInConcat - (width - 1);
                    v = x[(start + tok) * dim + c];
                }
                convStates[(cacheIdx * stateLen + s) * dim + c] = v;
            }
        }
    }
}

TEST_F(causal_conv1d_test, fwd_bfloat16_basic)
{
    if (!std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with DTYPE_X != bfloat16_t";
        return;
    }

    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 4;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 0;

    const std::vector<int32_t> queryStartLoc = {0, 2, 4};
    const std::vector<int32_t> cacheIndices = {0, 1};
    const std::vector<bool> hasInitialState = {true, true};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(bfloat16_t);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(bfloat16_t);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(bfloat16_t);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(bfloat16_t);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int32_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(bool);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(bfloat16_t);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<bfloat16_t*>(xGm);
    auto* w = reinterpret_cast<bfloat16_t*>(wGm);
    auto* b = reinterpret_cast<bfloat16_t*>(bGm);
    auto* s = reinterpret_cast<bfloat16_t*>(sGm);
    auto* qsl = reinterpret_cast<int32_t*>(qslGm);
    auto* idx = reinterpret_cast<int32_t*>(idxGm);
    auto* his = reinterpret_cast<bool*>(hisGm);
    auto* y = reinterpret_cast<bfloat16_t*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<bfloat16_t>(0.01f * static_cast<float>(c) + 0.05f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<bfloat16_t>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<bfloat16_t>(0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<bfloat16_t>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<bfloat16_t>(0.2f + 0.01f * static_cast<float>(c) + 0.001f * static_cast<float>(h));
            s[(1 * stateLen + h) * dim + c] = FromFloat<bfloat16_t>(-0.2f + 0.02f * static_cast<float>(c) - 0.002f * static_cast<float>(h));
        }
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<bfloat16_t>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                 qsl, batch, idx, his, activationMode, padSlotId);

    // quantize expected output/state to BF16 (kernel writes BF16)
    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<bfloat16_t>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<bfloat16_t>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->activationMode = activationMode;
    tilingData->padSlotId = padSlotId;
    tilingData->hasBias = 1;
    tilingData->dimTileSize = 1024;
    tilingData->blocksPerSeq = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(causal_conv1d<0>, batch, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, yGm, workspace,
                reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], 5e-2) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], 5e-2) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(causal_conv1d_test, fwd_float16_basic)
{
    if (!std::is_same<DTYPE_X, half>::value) {
        GTEST_SKIP() << "Skip: compiled with DTYPE_X != half";
        return;
    }

    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 4;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int32_t> queryStartLoc = {0, 2, 4};
    const std::vector<int32_t> cacheIndices = {0, 1};
    const std::vector<bool> hasInitialState = {true, true};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(half);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(half);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(half);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(half);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int32_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(bool);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(half);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<half*>(xGm);
    auto* w = reinterpret_cast<half*>(wGm);
    auto* b = reinterpret_cast<half*>(bGm);
    auto* s = reinterpret_cast<half*>(sGm);
    auto* qsl = reinterpret_cast<int32_t*>(qslGm);
    auto* idx = reinterpret_cast<int32_t*>(idxGm);
    auto* his = reinterpret_cast<bool*>(hisGm);
    auto* y = reinterpret_cast<half*>(yGm);

    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<half>(0.01f * static_cast<float>(c) + 0.05f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<half>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<half>(0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<half>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<half>(0.2f + 0.01f * static_cast<float>(c) + 0.001f * static_cast<float>(h));
            s[(1 * stateLen + h) * dim + c] = FromFloat<half>(-0.2f + 0.02f * static_cast<float>(c) - 0.002f * static_cast<float>(h));
        }
    }
    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<half>(12345.0f);
    }

    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                 qsl, batch, idx, his, activationMode, padSlotId);

    // quantize expected output/state to FP16 (kernel writes FP16)
    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<half>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<half>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->activationMode = activationMode;
    tilingData->padSlotId = padSlotId;
    tilingData->hasBias = 1;
    tilingData->dimTileSize = 1024;
    tilingData->blocksPerSeq = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(causal_conv1d<0>, batch, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, yGm, workspace,
                reinterpret_cast<uint8_t*>(tilingData));

    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], 2e-2) << "y mismatch at i=" << i;
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], 2e-2) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Reference implementation for 3D batch mode (fixed seqLen for all sequences)
static void ReferenceCausalConv1dFwd3DBatch(const float* x, const float* weight, const float* bias, float* y,
                                            float* convStates, int32_t dim, int32_t width, int32_t stateLen,
                                            int32_t seqLen, int32_t batch, const int32_t* cacheIndices,
                                            const bool* hasInitialState, int32_t activationMode, int32_t padSlotId)
{
    for (int32_t seq = 0; seq < batch; ++seq) {
        const int32_t start = seq * seqLen;  // Fixed offset based on seqLen
        const int32_t len = seqLen;          // Fixed length for all sequences
        const int32_t cacheIdx = cacheIndices[seq];

        if (cacheIdx == padSlotId) {
            continue;
        }

        std::vector<float> hist(width - 1, 0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            if (hasInitialState[seq]) {
                for (int32_t h = 0; h < width - 1; ++h) {
                    hist[h] = convStates[(cacheIdx * stateLen + h) * dim + c];
                }
            } else {
                std::fill(hist.begin(), hist.end(), 0.0f);
            }

            for (int32_t t = 0; t < len; ++t) {
                float acc = bias != nullptr ? bias[c] : 0.0f;
                for (int32_t j = 0; j < width; ++j) {
                    const int32_t srcT = t - j;
                    float xval = 0.0f;
                    if (srcT >= 0) {
                        xval = x[(start + srcT) * dim + c];
                    } else {
                        const int32_t histIndex = (width - 1) + srcT;
                        xval = hist[histIndex];
                    }
                    acc += weight[j * dim + c] * xval;
                }
                if (activationMode != 0) {
                    acc = SiluRef(acc);
                }
                y[(start + t) * dim + c] = acc;
            }

            // State writeback
            for (int32_t s = 0; s < width - 1; ++s) {
                const int32_t idxInConcat = len + s;
                float v = 0.0f;
                if (idxInConcat < (width - 1)) {
                    v = hist[idxInConcat];
                } else {
                    const int32_t tok = idxInConcat - (width - 1);
                    v = x[(start + tok) * dim + c];
                }
                convStates[(cacheIdx * stateLen + s) * dim + c] = v;
            }
        }
    }
}

TEST_F(causal_conv1d_test, fwd_3d_batch_mode)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;
    // Test 3D batch mode: x.shape = (batch, seqlen, dim) with inputMode=1
    // This uses fixed seqLen for all sequences instead of queryStartLoc
    constexpr int32_t batch = 3;
    constexpr int32_t dim = 1024;
    constexpr int32_t seqLen = 4;  // Fixed sequence length for all batches
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t numCacheLines = 8;
    constexpr int32_t cuSeqlen = batch * seqLen;  // 12 total tokens
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;  // SiLU

    const std::vector<int32_t> cacheIndices = {1, 3, 5};
    const std::vector<bool> hasInitialState = {true, false, true};

    const size_t xBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);  // Still allocated but not used
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int32_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(bool);
    const size_t yBytes = static_cast<size_t>(cuSeqlen) * dim * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int32_t*>(qslGm);
    auto* idx = reinterpret_cast<int32_t*>(idxGm);
    auto* his = reinterpret_cast<bool*>(hisGm);
    auto* y = reinterpret_cast<T*>(yGm);

    // Initialize inputs - physical layout is (cuSeqlen, dim) with dim contiguous
    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.01f * static_cast<float>(c) + 0.1f * static_cast<float>(t));
        }
    }
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.01f * static_cast<float>(c));
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    // Seed cache lines with distinct history
    for (int32_t line : {1, 5}) {
        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t h = 0; h < stateLen; ++h) {
                s[(line * stateLen + h) * dim + c] = FromFloat<T>(0.5f + 0.01f * static_cast<float>(c) +
                                                                 0.001f * static_cast<float>(h) +
                                                                 0.1f * static_cast<float>(line));
            }
        }
    }
    // queryStartLoc is not used in 3D mode but still needs to be allocated
    for (int32_t i = 0; i <= batch; ++i) {
        qsl[i] = i * seqLen;  // Dummy values
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
    }
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    // Compute reference using 3D batch mode function
    std::vector<float> xRef(cuSeqlen * dim);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(cuSeqlen * dim, 0.0f);
    ReferenceCausalConv1dFwd3DBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width,
                                    stateLen, seqLen, batch, idx, his, activationMode, padSlotId);

    // Quantize expected output/state to DTYPE_X
    std::vector<float> yRefQ(cuSeqlen * dim);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    // Set up tiling data for 3D batch mode
    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->seqLen = seqLen;      // Non-zero for 3D batch mode
    tilingData->inputMode = 1;        // 3D batch mode
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->activationMode = activationMode;
    tilingData->padSlotId = padSlotId;
    tilingData->hasBias = 1;
    tilingData->dimTileSize = 1024;
    tilingData->blocksPerSeq = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(causal_conv1d<0>, batch, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, yGm, workspace,
                reinterpret_cast<uint8_t*>(tilingData));

    // Check output
    for (int32_t i = 0; i < cuSeqlen * dim; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }

    // Check state writeback
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test weight shape: (width, dim), dim contiguous
TEST_F(causal_conv1d_test, fwd_weight_shape_width_dim)
{
    if (!std::is_same<DTYPE_X, half>::value && !std::is_same<DTYPE_X, bfloat16_t>::value) {
        GTEST_SKIP() << "Skip: compiled with unsupported DTYPE_X";
        return;
    }

    using T = DTYPE_X;
    const float tol = std::is_same<T, bfloat16_t>::value ? 5e-2f : 2e-2f;
    constexpr int32_t dim = 1024;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 2;
    constexpr int32_t numCacheLines = 4;
    constexpr int32_t cuSeqlen = 6;
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int32_t> queryStartLoc = {0, 3, 6};
    const std::vector<int32_t> cacheIndices = {0, 1};
    const std::vector<bool> hasInitialState = {true, false};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);
    const size_t wBytes = static_cast<size_t>(width) * dim * sizeof(T);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(T);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(T);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int32_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(bool);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(T);

    uint8_t* xGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* wGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wBytes));
    uint8_t* bGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bBytes));
    uint8_t* sGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sBytes));
    uint8_t* qslGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(qslBytes));
    uint8_t* idxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* hisGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(hisBytes));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(CausalConv1dTilingData)));

    auto* x = reinterpret_cast<T*>(xGm);
    auto* w = reinterpret_cast<T*>(wGm);
    auto* b = reinterpret_cast<T*>(bGm);
    auto* s = reinterpret_cast<T*>(sGm);
    auto* qsl = reinterpret_cast<int32_t*>(qslGm);
    auto* idx = reinterpret_cast<int32_t*>(idxGm);
    auto* his = reinterpret_cast<bool*>(hisGm);
    auto* y = reinterpret_cast<T*>(yGm);

    // Initialize x
    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = FromFloat<T>(0.02f * static_cast<float>(c) + 0.15f * static_cast<float>(t));
        }
    }

    // Initialize weight in (width, dim) layout: w[j, c] = w[j * dim + c]
    for (int32_t j = 0; j < width; ++j) {
        for (int32_t c = 0; c < dim; ++c) {
            w[j * dim + c] = FromFloat<T>(0.002f * static_cast<float>(c + 1) * static_cast<float>(j + 1));
        }
    }

    // Initialize bias
    for (int32_t c = 0; c < dim; ++c) {
        b[c] = FromFloat<T>(0.02f * static_cast<float>(c));
    }

    // Initialize conv states
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = FromFloat<T>(-1000.0f);
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(0 * stateLen + h) * dim + c] = FromFloat<T>(0.3f + 0.01f * static_cast<float>(c) +
                                                           0.001f * static_cast<float>(h));
        }
    }

    for (int32_t i = 0; i < batch + 1; ++i) {
        qsl[i] = queryStartLoc[i];
    }
    for (int32_t i = 0; i < batch; ++i) {
        idx[i] = cacheIndices[i];
        his[i] = hasInitialState[i];
    }
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        y[i] = FromFloat<T>(12345.0f);
    }

    // Compute reference
    std::vector<float> xRef(dim * cuSeqlen);
    std::vector<float> wRef(width * dim);
    std::vector<float> bRef(dim);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) xRef[i] = ToFloat(x[i]);
    for (int32_t i = 0; i < width * dim; ++i) wRef[i] = ToFloat(w[i]);
    for (int32_t i = 0; i < dim; ++i) bRef[i] = ToFloat(b[i]);
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) sRef[i] = ToFloat(s[i]);

    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    ReferenceCausalConv1dFwdBatch(xRef.data(), wRef.data(), bRef.data(), yRef.data(), sRef.data(), dim, width, stateLen,
                                  qsl, batch, idx, his, activationMode, padSlotId);

    // Quantize expected output/state to DTYPE_X
    std::vector<float> yRefQ(dim * cuSeqlen);
    std::vector<float> sRefQ(numCacheLines * dim * stateLen);
    for (size_t i = 0; i < yRefQ.size(); ++i) yRefQ[i] = ToFloat(FromFloat<T>(yRef[i]));
    for (size_t i = 0; i < sRefQ.size(); ++i) sRefQ[i] = ToFloat(FromFloat<T>(sRef[i]));

    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(CausalConv1dTilingData));
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->numCacheLines = numCacheLines;
    tilingData->batch = batch;
    tilingData->activationMode = activationMode;
    tilingData->padSlotId = padSlotId;
    tilingData->hasBias = 1;
    tilingData->dimTileSize = 1024;
    tilingData->blocksPerSeq = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(causal_conv1d<0>, batch, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, yGm, workspace,
                reinterpret_cast<uint8_t*>(tilingData));

    // Verify output
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(ToFloat(y[i]), yRefQ[i], tol) << "y mismatch at i=" << i;
    }
    // Verify state writeback
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(ToFloat(s[i]), sRefQ[i], tol) << "state mismatch at i=" << i;
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(wGm);
    AscendC::GmFree(bGm);
    AscendC::GmFree(sGm);
    AscendC::GmFree(qslGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(hisGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

