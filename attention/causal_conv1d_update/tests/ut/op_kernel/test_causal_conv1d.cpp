/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

static float SiluRef(float x)
{
    return x / (1.0f + std::exp(-x));
}

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
            // 输出可置 0（与 kernel 行为对齐，便于断言）
            for (int32_t t = 0; t < len; ++t) {
                for (int32_t c = 0; c < dim; ++c) {
                    y[(start + t) * dim + c] = 0.0f;
                }
            }
            continue;
        }

        std::vector<float> hist(width - 1, 0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            if (hasInitialState[seq]) {
                for (int32_t h = 0; h < width - 1; ++h) {
                    hist[h] = convStates[(cacheIdx * dim + c) * stateLen + h];
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
                    acc += weight[c * width + j] * xval;
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
                convStates[(cacheIdx * dim + c) * stateLen + s] = v;
            }
        }
    }
}

TEST_F(causal_conv1d_test, fwd_float32_bias_activation_and_state)
{
    constexpr int32_t dim = 16;
    constexpr int32_t width = 4;
    constexpr int32_t stateLen = 3;
    constexpr int32_t batch = 3;
    constexpr int32_t numCacheLines = 8;
    constexpr int32_t cuSeqlen = 5; // lens: [2,0,3]
    constexpr int32_t padSlotId = -1;
    constexpr int32_t activationMode = 1;

    const std::vector<int32_t> queryStartLoc = {0, 2, 2, 5};
    const std::vector<int32_t> cacheIndices = {1, padSlotId, 2};
    const std::vector<bool> hasInitialState = {true, true, false};

    const size_t xBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(float);
    const size_t wBytes = static_cast<size_t>(dim) * width * sizeof(float);
    const size_t bBytes = static_cast<size_t>(dim) * sizeof(float);
    const size_t sBytes = static_cast<size_t>(numCacheLines) * dim * stateLen * sizeof(float);
    const size_t qslBytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);
    const size_t idxBytes = static_cast<size_t>(batch) * sizeof(int32_t);
    const size_t hisBytes = static_cast<size_t>(batch) * sizeof(bool);
    const size_t yBytes = static_cast<size_t>(dim) * cuSeqlen * sizeof(float);

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

    auto* x = reinterpret_cast<float*>(xGm);
    auto* w = reinterpret_cast<float*>(wGm);
    auto* b = reinterpret_cast<float*>(bGm);
    auto* s = reinterpret_cast<float*>(sGm);
    auto* qsl = reinterpret_cast<int32_t*>(qslGm);
    auto* idx = reinterpret_cast<int32_t*>(idxGm);
    auto* his = reinterpret_cast<bool*>(hisGm);
    auto* y = reinterpret_cast<float*>(yGm);

    // init inputs
    for (int32_t t = 0; t < cuSeqlen; ++t) {
        for (int32_t c = 0; c < dim; ++c) {
            x[t * dim + c] = 0.01f * static_cast<float>(c) + 0.1f * static_cast<float>(t);
        }
    }
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t j = 0; j < width; ++j) {
            w[c * width + j] = 0.001f * static_cast<float>(c + 1) * static_cast<float>(j + 1);
        }
        b[c] = 0.01f * static_cast<float>(c);
    }
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        s[i] = -1000.0f; // sentinel for "unchanged" check
    }
    // seed cache lines 1 and 2 with distinct history
    for (int32_t c = 0; c < dim; ++c) {
        for (int32_t h = 0; h < stateLen; ++h) {
            s[(1 * dim + c) * stateLen + h] = 0.5f + 0.01f * static_cast<float>(c) + 0.001f * static_cast<float>(h);
            s[(2 * dim + c) * stateLen + h] = -0.5f + 0.02f * static_cast<float>(c) - 0.002f * static_cast<float>(h);
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
        y[i] = 12345.0f;
    }

    // reference (make copies of x/conv_states and compute y/state)
    std::vector<float> yRef(dim * cuSeqlen, 0.0f);
    std::vector<float> sRef(numCacheLines * dim * stateLen);
    std::copy(s, s + sRef.size(), sRef.begin());
    ReferenceCausalConv1dFwdBatch(x, w, b, yRef.data(), sRef.data(), dim, width, stateLen, qsl, batch, idx, his,
                                 activationMode, padSlotId);

    // tiling data
    auto* tilingData = reinterpret_cast<CausalConv1dTilingData*>(tiling);
    tilingData->dim = dim;
    tilingData->cuSeqlen = cuSeqlen;
    tilingData->width = width;
    tilingData->stateLen = stateLen;
    tilingData->batch = batch;
    tilingData->activationMode = activationMode;
    tilingData->padSlotId = padSlotId;
    tilingData->hasBias = 1;

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(causal_conv1d<2>, 1, xGm, wGm, bGm, sGm, qslGm, idxGm, hisGm, yGm, workspace,
                reinterpret_cast<uint8_t*>(tilingData));

    // check output
    for (int32_t i = 0; i < dim * cuSeqlen; ++i) {
        ASSERT_NEAR(y[i], yRef[i], 1e-4) << "y mismatch at i=" << i;
    }

    // check state writeback: cache line 1 and 2 updated; others unchanged
    for (int32_t i = 0; i < numCacheLines * dim * stateLen; ++i) {
        ASSERT_NEAR(s[i], sRef[i], 1e-4) << "state mismatch at i=" << i;
    }

    // 释放资源
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
