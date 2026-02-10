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
#include "../../../../op_kernel/3rd/gqmm_cube_on_the_fly.h"

class GmmAswKernelArch35Test : public testing::Test
{
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 为 GetTensorAddr 创建最小指针表: [byteOffset=8, dataAddr]
static uint8_t* WrapTensorPtr(void* dataPtr)
{
    uint64_t* buf = (uint64_t*)AscendC::GmAlloc(2 * sizeof(uint64_t));
    buf[0] = sizeof(uint64_t);
    buf[1] = reinterpret_cast<uint64_t>(dataPtr);
    return reinterpret_cast<uint8_t*>(buf);
}

// 公共 tiling 初始化函数，完全按照 gmm2.log 填充所有字段
static void FillCommonTiling(Mc2GroupedMatmulTilingData::GMMQuantTilingData* td,
                             uint32_t M, uint32_t K, uint32_t N)
{
    memset(td, 0, sizeof(*td));
    // GMMQuantParams（与 gmm2.log 完全一致）
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
    td->gmmArray.mList[0] = -1;  // SPLIT_M: 从 groupList 获取
    td->gmmArray.kList[0] = static_cast<int32_t>(K);
    td->gmmArray.nList[0] = static_cast<int32_t>(N);
    // TCubeTiling — 完全按照 gmm2.log（仅 M/singleCoreM 由参数传入）
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
    // 所有 LayoutInfo 和 Batch/mx 均为 0（memset 已处理）
}

// GmmASWKernel wrapper 函数，通过 ICPU_RUN_KF 调用
static void gmmASWKernelWrapper(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    auto* td = reinterpret_cast<Mc2GroupedMatmulTilingData::GMMQuantTilingData*>(tiling);
    AscendC::TPipe tPipe;
    AscendC::GmmASWKernel<hifloat8_t, hifloat8_t, float, float, bfloat16_t,
                          CubeFormat::ND, false, true> op;
    op.Init(x, weight, bias, scale, groupList, perTokenScale, y, workspace,
            &td->gmmQuantParams, &td->mmTilingData,
            reinterpret_cast<TILING_TYPE*>(&td->gmmArray), &tPipe);
    op.Process();
}

// GmmASWKernel UT: x=(2,3), w=(3,2), group_list=[2], x1_scale=2.0, x2_scale=2.0
// 预期输出 (bf16): [[64, -64], [32, -32]]
TEST_F(GmmAswKernelArch35Test, Case1)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 2, K = 3, N = 2;
    uint32_t NumBlocks = 36;

    // 1. Tiling
    size_t tilingSize = sizeof(Mc2GroupedMatmulTilingData::GMMQuantTilingData);
    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(tilingSize);
    auto* td = reinterpret_cast<Mc2GroupedMatmulTilingData::GMMQuantTilingData*>(tilingBuf);
    FillCommonTiling(td, M, K, N);

    // 2. 数据 buffer (实际数据)
    int8_t* xData = (int8_t*)AscendC::GmAlloc(M * K * sizeof(int8_t));
    int8_t xVals[] = {8, -96, -94, 20, -108, -112};
    memcpy(xData, xVals, sizeof(xVals));

    int8_t* wData = (int8_t*)AscendC::GmAlloc(K * N * sizeof(int8_t));
    // bTrans=true: CPU simulator 需要 (N, K) 布局，即原始 (K, N) 数据的转置
    // 原始 (K=3, N=2): [-120, 8, -108, 20, -120, 8]
    // 转置 (N=2, K=3): [-120, -108, -120, 8, 20, 8]
    int8_t wVals[] = {-120, -108, -120, 8, 20, 8};
    memcpy(wData, wVals, sizeof(wVals));

    float* scaleData = (float*)AscendC::GmAlloc(sizeof(float));
    scaleData[0] = 2.0f;  // x2_scale

    bfloat16_t* yData = (bfloat16_t*)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    // 3. 指针表包装 (GetTensorAddr 需要)
    uint8_t* xPtr = WrapTensorPtr(xData);
    uint8_t* wPtr = WrapTensorPtr(wData);
    uint8_t* scalePtr = WrapTensorPtr(scaleData);
    uint8_t* yPtr = WrapTensorPtr(yData);

    // 4. 平铺 buffer (不经过 GetTensorAddr)
    float* perTokenScale = (float*)AscendC::GmAlloc(sizeof(float));
    perTokenScale[0] = 2.0f;  // x1_scale

    int64_t* groupList = (int64_t*)AscendC::GmAlloc(sizeof(int64_t));
    groupList[0] = 2;  // M=2

    // 5. 通过 ICPU_RUN_KF 拉起 kernel
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmASWKernelWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xPtr, (GM_ADDR)wPtr, (GM_ADDR)nullptr, (GM_ADDR)scalePtr,
                (GM_ADDR)groupList, (GM_ADDR)perTokenScale, (GM_ADDR)yPtr,
                (GM_ADDR)nullptr, (GM_ADDR)tilingBuf);

    // 6. 校验
    float expected[] = {64.0f, -64.0f, 32.0f, -32.0f};
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1.0f) << "T1 index " << i;
    }

    // 7. 释放
    AscendC::GmFree(xData);
    AscendC::GmFree(xPtr);
    AscendC::GmFree(wData);
    AscendC::GmFree(wPtr);
    AscendC::GmFree(scaleData);
    AscendC::GmFree(scalePtr);
    AscendC::GmFree(yData);
    AscendC::GmFree(yPtr);
    AscendC::GmFree(perTokenScale);
    AscendC::GmFree(groupList);
    AscendC::GmFree(tilingBuf);
}

// GmmASWKernel UT: x=(2,3), w=(3,2), group_list=[2], x1_scale=2.0, x2_scale=1.0
// 预期输出 (bf16): [[4, 14], [-12, -26]]
TEST_F(GmmAswKernelArch35Test, Case2)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 2, K = 3, N = 2;
    uint32_t NumBlocks = 36;

    // 1. Tiling
    size_t tilingSize = sizeof(Mc2GroupedMatmulTilingData::GMMQuantTilingData);
    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(tilingSize);
    auto* td = reinterpret_cast<Mc2GroupedMatmulTilingData::GMMQuantTilingData*>(tilingBuf);
    FillCommonTiling(td, M, K, N);

    // 2. 数据 buffer
    int8_t* xData = (int8_t*)AscendC::GmAlloc(M * K * sizeof(int8_t));
    int8_t xVals[] = {-120, -120, 16, 20, -94, 16};
    memcpy(xData, xVals, sizeof(xVals));

    int8_t* wData = (int8_t*)AscendC::GmAlloc(K * N * sizeof(int8_t));
    // bTrans=true: CPU simulator 需要 (N, K) 布局
    // 原始 (K=3, N=2): [-108, -96, -120, 8, -120, 16]
    // 转置 (N=2, K=3): [-108, -120, -120, -96, 8, 16]
    int8_t wVals[] = {-108, -120, -120, -96, 8, 16};
    memcpy(wData, wVals, sizeof(wVals));

    float* scaleData = (float*)AscendC::GmAlloc(sizeof(float));
    scaleData[0] = 1.0f;  // x2_scale

    bfloat16_t* yData = (bfloat16_t*)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    // 3. 指针表包装
    uint8_t* xPtr = WrapTensorPtr(xData);
    uint8_t* wPtr = WrapTensorPtr(wData);
    uint8_t* scalePtr = WrapTensorPtr(scaleData);
    uint8_t* yPtr = WrapTensorPtr(yData);

    // 4. 平铺 buffer
    float* perTokenScale = (float*)AscendC::GmAlloc(sizeof(float));
    perTokenScale[0] = 2.0f;  // x1_scale

    int64_t* groupList = (int64_t*)AscendC::GmAlloc(sizeof(int64_t));
    groupList[0] = 2;  // M=2

    // 5. ICPU_RUN_KF 拉起
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmASWKernelWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xPtr, (GM_ADDR)wPtr, (GM_ADDR)nullptr, (GM_ADDR)scalePtr,
                (GM_ADDR)groupList, (GM_ADDR)perTokenScale, (GM_ADDR)yPtr,
                (GM_ADDR)nullptr, (GM_ADDR)tilingBuf);

    // 6. 校验
    float expected[] = {4.0f, 14.0f, -12.0f, -26.0f};
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1.0f) << "T2 index " << i;
    }

    // 7. 释放
    AscendC::GmFree(xData);
    AscendC::GmFree(xPtr);
    AscendC::GmFree(wData);
    AscendC::GmFree(wPtr);
    AscendC::GmFree(scaleData);
    AscendC::GmFree(scalePtr);
    AscendC::GmFree(yData);
    AscendC::GmFree(yPtr);
    AscendC::GmFree(perTokenScale);
    AscendC::GmFree(groupList);
    AscendC::GmFree(tilingBuf);
}

// GmmASWKernel UT: x=(4,3), w=(3,2), group_list=[4], x1_scale=2.0, x2_scale=2.0
// 预期输出 (bf16): [[48, -12], [8, 20], [-80, 48], [-44, 12]]
TEST_F(GmmAswKernelArch35Test, Case3)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 4, K = 3, N = 2;
    uint32_t NumBlocks = 36;

    // 1. Tiling
    size_t tilingSize = sizeof(Mc2GroupedMatmulTilingData::GMMQuantTilingData);
    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(tilingSize);
    auto* td = reinterpret_cast<Mc2GroupedMatmulTilingData::GMMQuantTilingData*>(tilingBuf);
    FillCommonTiling(td, M, K, N);

    // 2. 数据 buffer
    int8_t* xData = (int8_t*)AscendC::GmAlloc(M * K * sizeof(int8_t));
    int8_t xVals[] = {-108, 20, 0, 8, 32, 20, 16, -108, 20, 16, -108, 0};
    memcpy(xData, xVals, sizeof(xVals));

    int8_t* wData = (int8_t*)AscendC::GmAlloc(K * N * sizeof(int8_t));
    // bTrans=true: CPU simulator 需要 (N, K) 布局
    // 原始 (K=3, N=2): [-120, 0, 20, -120, -108, 20]
    // 转置 (N=2, K=3): [-120, 20, -108, 0, -120, 20]
    int8_t wVals[] = {-120, 20, -108, 0, -120, 20};
    memcpy(wData, wVals, sizeof(wVals));

    float* scaleData = (float*)AscendC::GmAlloc(sizeof(float));
    scaleData[0] = 2.0f;  // x2_scale

    bfloat16_t* yData = (bfloat16_t*)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    // 3. 指针表包装
    uint8_t* xPtr = WrapTensorPtr(xData);
    uint8_t* wPtr = WrapTensorPtr(wData);
    uint8_t* scalePtr = WrapTensorPtr(scaleData);
    uint8_t* yPtr = WrapTensorPtr(yData);

    // 4. 平铺 buffer
    float* perTokenScale = (float*)AscendC::GmAlloc(sizeof(float));
    perTokenScale[0] = 2.0f;  // x1_scale

    int64_t* groupList = (int64_t*)AscendC::GmAlloc(sizeof(int64_t));
    groupList[0] = 4;  // M=4

    // 5. ICPU_RUN_KF 拉起
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmASWKernelWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xPtr, (GM_ADDR)wPtr, (GM_ADDR)nullptr, (GM_ADDR)scalePtr,
                (GM_ADDR)groupList, (GM_ADDR)perTokenScale, (GM_ADDR)yPtr,
                (GM_ADDR)nullptr, (GM_ADDR)tilingBuf);

    // 6. 校验
    float expected[] = {48.0f, -12.0f, 8.0f, 20.0f, -80.0f, 48.0f, -44.0f, 12.0f};
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1.0f) << "T3 index " << i;
    }

    // 7. 释放
    AscendC::GmFree(xData);
    AscendC::GmFree(xPtr);
    AscendC::GmFree(wData);
    AscendC::GmFree(wPtr);
    AscendC::GmFree(scaleData);
    AscendC::GmFree(scalePtr);
    AscendC::GmFree(yData);
    AscendC::GmFree(yPtr);
    AscendC::GmFree(perTokenScale);
    AscendC::GmFree(groupList);
    AscendC::GmFree(tilingBuf);
}
