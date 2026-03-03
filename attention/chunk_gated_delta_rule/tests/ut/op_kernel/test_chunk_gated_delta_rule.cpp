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
 * \file test_chunk_gated_delta_rule.cpp
 *
 */
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <cstring>
#include <iostream>
#include <string>
#include <cstdint>
#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

struct alignas(8) ChunkGatedDeltaRuleTilingData {
    int64_t aiCoreNum;
    int64_t t;
    int64_t nk;
    int64_t dk;
    int64_t nv;
    int64_t dv;
    int64_t b;
    int64_t hasGamma;
    int64_t chunkSize;
    int64_t maxGroupLength;
    int64_t interWorkspaceSz;
    int64_t stageWorkspaceSz;
    float scale;
};

extern "C" __global__ __aicore__ void
chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR initialState,
                       GM_ADDR seqlens, GM_ADDR gOptional, GM_ADDR out, GM_ADDR finalState,
                       GM_ADDR workspaceGM, GM_ADDR tilingGM);

template <typename T>
T* GmAllocWrapper(size_t size) {
    T *ptr = reinterpret_cast<T *>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

void InitTilingData(ChunkGatedDeltaRuleTilingData* tilingData, int64_t b, int64_t t, int64_t nk, int64_t nv,
                    int64_t dk, int64_t dv, int64_t hasGamma, int64_t chunkSize, int64_t maxGroupLength) {
    tilingData->aiCoreNum = 8;
    tilingData->t = t;
    tilingData->nk = nk;
    tilingData->dk = dk;
    tilingData->nv = nv;
    tilingData->dv = dv;
    tilingData->b = b;
    tilingData->hasGamma = hasGamma;
    tilingData->chunkSize = chunkSize;
    tilingData->maxGroupLength = maxGroupLength;

    // 计算中间workspace大小
    int64_t sizeLow = sizeof(bfloat16_t);  // bf16
    int64_t sizeHigh = sizeof(float);       // float
    int64_t s = maxGroupLength;

    tilingData->interWorkspaceSz = 0;
    tilingData->interWorkspaceSz += sizeHigh * nv * s;             // gCumExp
    tilingData->interWorkspaceSz += sizeLow * nv * s * dk;         // kCumDecay
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dk;        // vInner
    tilingData->interWorkspaceSz += sizeLow * nv * s * dk;         // qPrime
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dv;        // attnInter
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dv;        // vNew
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dk;        // kg
    tilingData->interWorkspaceSz += sizeHigh * nv * s * chunkSize; // qkt

    // stage workspace大小
    tilingData->stageWorkspaceSz = sizeHigh * chunkSize * (3 * chunkSize + dk + dv);
    tilingData->stageWorkspaceSz *= tilingData->aiCoreNum;

    tilingData->scale = 1.0f;
}

void InitInputData(uint8_t* queryGm, size_t shapeQ,
                   uint8_t* keyGm, size_t shapeK,
                   uint8_t* valueGm, size_t shapeV,
                   uint8_t* betaGm, size_t shapeBeta,
                   uint8_t* stateGm, size_t shapeState,
                   uint8_t* gammaGm, size_t shapeGamma,
                   uint8_t* seqlensGm, size_t b,
                   uint8_t* outGm, size_t shapeOut) {
    memset(queryGm, 0, shapeQ);
    memset(keyGm, 0, shapeK);
    memset(valueGm, 0, shapeV);
    memset(betaGm, 0, shapeBeta);
    memset(stateGm, 0, shapeState);
    memset(gammaGm, 0, shapeGamma);
    memset(outGm, 0, shapeOut);

    int32_t* seqlens = reinterpret_cast<int32_t*>(seqlensGm);
    for (size_t i = 0; i < b; ++i) {
        seqlens[i] = 64;  // 每个batch的序列长度设为64
    }
}

struct CGDRTestParams {
    int64_t hasGamma;
};

class ChunkGatedDeltaRuleTest : public testing::TestWithParam<CGDRTestParams> {
protected:
    int64_t b = 2;              // batch size
    int64_t nk = 4;             // key head num
    int64_t nv = 4;             // value head num
    int64_t dk = 64;            // key dimension
    int64_t dv = 64;            // value dimension
    int64_t t = 128;            // total tokens
    int64_t chunkSize = 64;     // chunk size
    int64_t maxGroupLength = 128;  // max group length

    size_t shapeQuery = t * nk * dk * sizeof(bfloat16_t);
    size_t shapeKey = t * nk * dk * sizeof(bfloat16_t);
    size_t shapeValue = t * nv * dv * sizeof(bfloat16_t);
    size_t shapeBeta = t * nv * sizeof(bfloat16_t);
    size_t shapeState = b * nv * dv * dk * sizeof(bfloat16_t);
    size_t shapeGamma = t * nv * sizeof(float);
    size_t shapeSeqlens = b * sizeof(int32_t);
    size_t shapeOut = t * nv * dv * sizeof(bfloat16_t);
    size_t tilingSize = sizeof(ChunkGatedDeltaRuleTilingData);

    uint8_t* queryGm = nullptr;
    uint8_t* keyGm = nullptr;
    uint8_t* valueGm = nullptr;
    uint8_t* betaGm = nullptr;
    uint8_t* stateGm = nullptr;
    uint8_t* gammaGm = nullptr;
    uint8_t* seqlensGm = nullptr;
    uint8_t* outGm = nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = nullptr;

    void SetUp() override {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        queryGm = GmAllocWrapper<uint8_t>(shapeQuery);
        keyGm = GmAllocWrapper<uint8_t>(shapeKey);
        valueGm = GmAllocWrapper<uint8_t>(shapeValue);
        betaGm = GmAllocWrapper<uint8_t>(shapeBeta);
        stateGm = GmAllocWrapper<uint8_t>(shapeState);
        gammaGm = GmAllocWrapper<uint8_t>(shapeGamma);
        seqlensGm = GmAllocWrapper<uint8_t>(shapeSeqlens);
        outGm = GmAllocWrapper<uint8_t>(shapeOut);

        // 计算workspace大小
        ChunkGatedDeltaRuleTilingData tempTiling;
        auto params = GetParam();
        InitTilingData(&tempTiling, b, t, nk, nv, dk, dv, params.hasGamma, chunkSize, maxGroupLength);
        size_t workspaceSize = 16 * 1024 * 1024;  // 16MB system workspace
        workspaceSize += tempTiling.interWorkspaceSz;
        workspaceSize += tempTiling.stageWorkspaceSz;
        workspace = GmAllocWrapper<uint8_t>(workspaceSize);
        tiling = GmAllocWrapper<uint8_t>(tilingSize);

        InitInputData(queryGm, shapeQuery,
                      keyGm, shapeKey,
                      valueGm, shapeValue,
                      betaGm, shapeBeta,
                      stateGm, shapeState,
                      gammaGm, shapeGamma,
                      seqlensGm, b,
                      outGm, shapeOut);

        ChunkGatedDeltaRuleTilingData* tilingData = reinterpret_cast<ChunkGatedDeltaRuleTilingData*>(tiling);
        InitTilingData(tilingData, b, t, nk, nv, dk, dv, params.hasGamma, chunkSize, maxGroupLength);
    }

    void TearDown() override {
        AscendC::GmFree(queryGm);
        AscendC::GmFree(keyGm);
        AscendC::GmFree(valueGm);
        AscendC::GmFree(betaGm);
        AscendC::GmFree(stateGm);
        AscendC::GmFree(gammaGm);
        AscendC::GmFree(seqlensGm);
        AscendC::GmFree(outGm);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

INSTANTIATE_TEST_SUITE_P(
    GeneralTests,
    ChunkGatedDeltaRuleTest,
    testing::Values(
        CGDRTestParams{1},    // general_test_01: 有Gamma
        CGDRTestParams{0}     // general_test_02: 无Gamma
    )
);

TEST_P(ChunkGatedDeltaRuleTest, RunTest) {
    auto params = GetParam();
    std::cout << "test config: hasGamma=" << params.hasGamma << std::endl;

    uint32_t blockDim = 8;
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(chunk_gated_delta_rule, blockDim,
                queryGm, keyGm, valueGm, betaGm, stateGm, seqlensGm,
                (params.hasGamma ? gammaGm : nullptr),
                outGm, stateGm, workspace, tiling);
}
