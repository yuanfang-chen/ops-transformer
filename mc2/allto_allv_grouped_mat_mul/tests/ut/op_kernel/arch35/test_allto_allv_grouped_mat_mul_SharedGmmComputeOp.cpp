/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../../op_kernel/mc2_templates/compute/shared_gmm_compute_op.h"

class SharedGmmComputeOpArch35Test : public testing::Test
{
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

struct SharedGmmTestParams {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t tempAddrSize;
};

// 完整填充 SharedGmm tiling（不做 RefreshTilingForM，必须手动填 M/singleCoreM/baseM/dbL0C）
static void FillSharedGmmTiling(MC2KernelTemplate::GMMQuantTilingData *td,
                                uint32_t M, uint32_t K, uint32_t N)
{
    memset(td, 0, sizeof(*td));
    // GMMQuantParams
    td->gmmQuantParams.groupNum      = 1;
    td->gmmQuantParams.activeType    = 0;
    td->gmmQuantParams.aQuantMode    = 1;   // PERTENSOR
    td->gmmQuantParams.bQuantMode    = 1;   // PERTENSOR
    td->gmmQuantParams.singleX       = 1;
    td->gmmQuantParams.singleW       = 1;
    td->gmmQuantParams.singleY       = 1;
    td->gmmQuantParams.groupType     = 0;   // SPLIT_M
    td->gmmQuantParams.groupListType = 0;
    td->gmmQuantParams.hasBias       = 0;
    td->gmmQuantParams.reserved      = 0;
    // GMMArray
    td->gmmArray.mList[0] = -1;  // SPLIT_M: M from groupList
    td->gmmArray.kList[0] = static_cast<int32_t>(K);
    td->gmmArray.nList[0] = static_cast<int32_t>(N);
    // TCubeTiling — must be fully populated
    td->mmTilingData.usedCoreNum     = 36;
    td->mmTilingData.M               = M;
    td->mmTilingData.N               = N;
    td->mmTilingData.Ka              = K;
    td->mmTilingData.Kb              = K;
    td->mmTilingData.singleCoreM     = M;
    td->mmTilingData.singleCoreN     = N;
    td->mmTilingData.singleCoreK     = K;
    td->mmTilingData.baseM           = 16;
    td->mmTilingData.baseN           = 32;
    td->mmTilingData.baseK           = 32;
    td->mmTilingData.depthA1         = 2;
    td->mmTilingData.depthB1         = 2;
    td->mmTilingData.stepM           = 1;
    td->mmTilingData.stepN           = 1;
    td->mmTilingData.isBias          = 0;
    td->mmTilingData.transLength     = 0;
    td->mmTilingData.iterateOrder    = 0;
    td->mmTilingData.shareMode       = 0;
    td->mmTilingData.shareL1Size     = 0;
    td->mmTilingData.shareL0CSize    = 0;
    td->mmTilingData.shareUbSize     = 0;
    td->mmTilingData.batchM          = 0;
    td->mmTilingData.batchN          = 0;
    td->mmTilingData.singleBatchM    = 0;
    td->mmTilingData.singleBatchN    = 0;
    td->mmTilingData.stepKa          = 1;
    td->mmTilingData.stepKb          = 1;
    td->mmTilingData.depthAL1CacheUB = 0;
    td->mmTilingData.depthBL1CacheUB = 0;
    td->mmTilingData.dbL0A           = 2;
    td->mmTilingData.dbL0B           = 2;
    td->mmTilingData.dbL0C           = 2;
}

// SharedGmmComputeOp wrapper (bTrans=true)
static void sharedGmmComputeOpWrapper(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                      GM_ADDR scaleA, GM_ADDR scaleB,
                                      GM_ADDR y, GM_ADDR tempAddr,
                                      GM_ADDR tilingBuf, GM_ADDR paramsBuf)
{
    auto *tiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(tilingBuf);
    auto *params = reinterpret_cast<SharedGmmTestParams *>(paramsBuf);
    AscendC::TPipe tPipe;
    MC2KernelTemplate::SharedGmmComputeOp<hifloat8_t, hifloat8_t, float, float, bfloat16_t,
                                           CubeFormat::ND, false, true> op;
    op.Init(x, weight, bias, scaleA, scaleB, y, tempAddr, params->tempAddrSize,
            params->m, params->n, params->k, tiling, &tPipe);
    op.Process();
}

// SharedGmmComputeOp wrapper (bTrans=false)
static void sharedGmmComputeOpWrapperNoTrans(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                             GM_ADDR scaleA, GM_ADDR scaleB,
                                             GM_ADDR y, GM_ADDR tempAddr,
                                             GM_ADDR tilingBuf, GM_ADDR paramsBuf)
{
    auto *tiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(tilingBuf);
    auto *params = reinterpret_cast<SharedGmmTestParams *>(paramsBuf);
    AscendC::TPipe tPipe;
    MC2KernelTemplate::SharedGmmComputeOp<hifloat8_t, hifloat8_t, float, float, bfloat16_t,
                                           CubeFormat::ND, false, false> op;
    op.Init(x, weight, bias, scaleA, scaleB, y, tempAddr, params->tempAddrSize,
            params->m, params->n, params->k, tiling, &tPipe);
    op.Process();
}

/**
 * Case1: bTrans=true, M=2, K=3, N=2, scaleA=2.0, scaleB=2.0
 * Data from GmmASWKernel Case1 / GmmComputeOp Case1
 * Expected: [[64, -64], [32, -32]]
 */
TEST_F(SharedGmmComputeOpArch35Test, Case1_SmallMatrix_bTransTrue)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 2, K = 3, N = 2;
    uint32_t NumBlocks = 36;

    // 1. Tiling
    auto *tiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillSharedGmmTiling(tiling, M, K, N);

    // 2. Params
    auto *params = reinterpret_cast<SharedGmmTestParams *>(
        AscendC::GmAlloc(sizeof(SharedGmmTestParams)));
    params->m = M;
    params->n = N;
    params->k = K;
    params->tempAddrSize = 4096;

    // 3. Data buffers
    int8_t *xData = (int8_t *)AscendC::GmAlloc(M * K * sizeof(int8_t));
    int8_t xVals[] = {8, -96, -94, 20, -108, -112};
    memcpy(xData, xVals, sizeof(xVals));

    // bTrans=true: weight in (N, K) layout
    int8_t *wData = (int8_t *)AscendC::GmAlloc(K * N * sizeof(int8_t));
    int8_t wVals[] = {-120, -108, -120, 8, 20, 8};
    memcpy(wData, wVals, sizeof(wVals));

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 2.0f;  // perTokenScale

    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 2.0f;  // weight scale

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    // 4. Temp workspace
    uint8_t *tempAddr = (uint8_t *)AscendC::GmAlloc(4096);
    memset(tempAddr, 0, 4096);

    // 5. ICPU_RUN_KF
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = sharedGmmComputeOpWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR)nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)tempAddr,
                (GM_ADDR)tiling, (GM_ADDR)params);

    // 6. Verify
    float expected[] = {64.0f, -64.0f, 32.0f, -32.0f};
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1e-3f) << "Case1 index " << i;
    }

    // 7. Free
    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(yData);
    AscendC::GmFree(tempAddr);
    AscendC::GmFree(tiling);
    AscendC::GmFree(params);
}

/**
 * Case2: bTrans=true, M=4, K=3, N=2, scaleA=2.0, scaleB=2.0
 * Data from GmmASWKernel Case3
 * Expected: [[48, -12], [8, 20], [-80, 48], [-44, 12]]
 */
TEST_F(SharedGmmComputeOpArch35Test, Case2_LargerM_bTransTrue)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 4, K = 3, N = 2;
    uint32_t NumBlocks = 36;

    // 1. Tiling
    auto *tiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillSharedGmmTiling(tiling, M, K, N);

    // 2. Params
    auto *params = reinterpret_cast<SharedGmmTestParams *>(
        AscendC::GmAlloc(sizeof(SharedGmmTestParams)));
    params->m = M;
    params->n = N;
    params->k = K;
    params->tempAddrSize = 4096;

    // 3. Data buffers
    int8_t *xData = (int8_t *)AscendC::GmAlloc(M * K * sizeof(int8_t));
    int8_t xVals[] = {-108, 20, 0, 8, 32, 20, 16, -108, 20, 16, -108, 0};
    memcpy(xData, xVals, sizeof(xVals));

    // bTrans=true: weight in (N, K) layout
    int8_t *wData = (int8_t *)AscendC::GmAlloc(K * N * sizeof(int8_t));
    int8_t wVals[] = {-120, 20, -108, 0, -120, 20};
    memcpy(wData, wVals, sizeof(wVals));

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 2.0f;

    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 2.0f;

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    // 4. Temp workspace
    uint8_t *tempAddr = (uint8_t *)AscendC::GmAlloc(4096);
    memset(tempAddr, 0, 4096);

    // 5. ICPU_RUN_KF
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = sharedGmmComputeOpWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR)nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)tempAddr,
                (GM_ADDR)tiling, (GM_ADDR)params);

    // 6. Verify
    float expected[] = {48.0f, -12.0f, 8.0f, 20.0f, -80.0f, 48.0f, -44.0f, 12.0f};
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1e-3f) << "Case2 index " << i;
    }

    // 7. Free
    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(yData);
    AscendC::GmFree(tempAddr);
    AscendC::GmFree(tiling);
    AscendC::GmFree(params);
}

/**
 * Case3: bTrans=false, M=10, K=10, N=13, scaleA=1.0, scaleB=1.0
 * Data from GmmComputeOp Case6 expert 0 (first 10 rows of x, first expert weight)
 * Expected: first 10x13 values of Case6 expected output
 */
TEST_F(SharedGmmComputeOpArch35Test, Case3_bTransFalse_Expert0FromCase6)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 10, K = 10, N = 13;
    uint32_t NumBlocks = 36;

    // 1. Tiling
    auto *tiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillSharedGmmTiling(tiling, M, K, N);

    // 2. Params
    auto *params = reinterpret_cast<SharedGmmTestParams *>(
        AscendC::GmAlloc(sizeof(SharedGmmTestParams)));
    params->m = M;
    params->n = N;
    params->k = K;
    params->tempAddrSize = 4096;

    // 3. x data: first 10 rows from Case6 (10x10 = 100 bytes)
    static const int8_t xVals[100] = {
        -112, -112, 8, 32, -108, 20, 8, -120, 16, -112,
        8, 8, -108, 32, 32, -120, -112, 0, 20, 20,
        -120, -112, -94, 8, 32, 0, -94, 16, -96, 20,
        -112, -112, -96, 20, -112, 8, -120, -112, 8, -120,
        8, -112, 32, -112, -94, 16, 32, 8, -96, 8,
        32, -120, -96, 16, -112, -120, 20, 20, -94, 8,
        -96, -96, 16, 32, 0, -112, 20, 8, 16, -96,
        -112, 0, -108, 0, -94, -112, -112, -94, -96, 16,
        -94, 20, 20, -120, 20, -94, 16, 0, -120, 32,
        -120, 0, 32, 16, 8, -120, 8, -94, -94, -96
    };
    int8_t *xData = (int8_t *)AscendC::GmAlloc(M * K * sizeof(int8_t));
    memcpy(xData, xVals, sizeof(xVals));

    // weight: expert 0 from Case6 (10x13 = 130 bytes, K x N layout, bTrans=false)
    static const int8_t wVals[130] = {
        -120, 16, 0, -96, -108, 32, -108, 20, 20, -112, -120, -112, 16,
        -108, -120, -112, 16, 8, 32, 20, 20, 20, 0, -96, -108, 16,
        32, -120, 8, 20, 32, -120, -120, -112, 32, -94, 32, 32, 16,
        16, 16, -94, -96, -94, -120, -96, -112, 8, 16, 20, 16, -96,
        -120, -108, 0, -112, 8, 16, -94, 20, -120, -96, -108, 20, 16,
        -94, 0, 8, 20, 20, 32, 20, -108, 20, -94, 20, -120, 32,
        8, 16, -108, 8, -120, 32, -94, -94, 0, -120, 32, -96, -112,
        -108, 8, -108, -96, -112, -120, -108, 0, -94, 32, 32, 20, -108,
        8, 0, -112, 8, 8, 20, -108, -112, 16, -96, 32, -96, 8,
        20, 20, -108, -96, 8, 16, -94, -96, -120, 16, -94, 20, -108
    };
    int8_t *wData = (int8_t *)AscendC::GmAlloc(K * N * sizeof(int8_t));
    memcpy(wData, wVals, sizeof(wVals));

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 1.0f;

    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 1.0f;

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    // 4. Temp workspace
    uint8_t *tempAddr = (uint8_t *)AscendC::GmAlloc(4096);
    memset(tempAddr, 0, 4096);

    // 5. ICPU_RUN_KF
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = sharedGmmComputeOpWrapperNoTrans;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR)nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)tempAddr,
                (GM_ADDR)tiling, (GM_ADDR)params);

    // 6. Verify — first 10x13 of Case6 expected output
    static const float expected[130] = {
        8.f, 9.f, -10.f, 21.f, -5.f, -8.f, 9.f, -41.f, 19.f, -13.f, 62.f, -11.f, -7.f,
        3.f, 5.f, -35.f, -49.f, -25.f, 18.f, -50.f, 11.f, -6.f, 6.f, -31.f, 9.f, -16.f,
        -21.f, -4.f, 2.f, -56.f, -20.f, -28.f, -6.f, 32.f, -53.f, 48.f, -63.f, 53.f, -21.f,
        -2.f, 7.f, -4.f, -1.f, -21.f, -16.f, 18.f, -12.f, -7.f, 20.f, 9.f, -16.f, -16.f,
        12.f, 23.f, 10.f, 20.f, 13.f, -3.f, 10.f, -44.f, 8.f, 2.f, 37.f, 5.f, -6.f,
        -14.f, 35.f, -24.f, -55.f, -57.f, 2.f, -26.f, 1.f, -32.f, 60.f, -8.f, -2.f, -41.f,
        32.f, -3.f, -16.f, 9.f, -17.f, -37.f, -28.f, -33.f, -15.f, 1.f, 78.f, 9.f, -31.f,
        20.f, 11.f, 18.f, 9.f, -7.f, -34.f, 55.f, 1.f, -4.f, 51.f, -55.f, -6.f, -19.f,
        41.f, -11.f, -19.f, 3.f, 24.f, -11.f, -32.f, -10.f, -19.f, 16.f, -38.f, 33.f, -25.f,
        24.f, -20.f, 27.f, 35.f, 7.f, -26.f, 28.f, 12.f, 30.f, -22.f, 1.f, 15.f, 16.f
    };
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1e-3f) << "Case3 index " << i;
    }

    // 7. Free
    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(yData);
    AscendC::GmFree(tempAddr);
    AscendC::GmFree(tiling);
    AscendC::GmFree(params);
}
