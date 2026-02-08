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
#include "../../../../op_kernel/mc2_templates/compute/gmm_compute_op.h"

class GmmComputeOpArch35Test : public testing::Test
{
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 填充 gmmBaseTiling（K/N 相关字段已知，M 相关字段由 GmmComputeOp 动态刷新）
static void FillBaseTiling(MC2KernelTemplate::GMMQuantTilingData *td, uint32_t K, uint32_t N)
{
    memset(td, 0, sizeof(*td));
    // GMMQuantParams
    td->gmmQuantParams.groupNum      = 1;  // 会被 GmmComputeOp 覆盖
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
    // GMMArray — K/N 已知，M 由 kernel 动态填写
    td->gmmArray.mList[0] = -1;  // SPLIT_M: 从 groupList 获取
    td->gmmArray.kList[0] = static_cast<int32_t>(K);
    td->gmmArray.nList[0] = static_cast<int32_t>(N);
    // TCubeTiling — K/N 相关字段已知, M 相关字段由 RefreshTilingForM 动态刷新
    td->mmTilingData.usedCoreNum     = 36;
    td->mmTilingData.M               = 0;   // 动态刷新
    td->mmTilingData.N               = N;
    td->mmTilingData.Ka              = K;
    td->mmTilingData.Kb              = K;
    td->mmTilingData.singleCoreM     = 0;   // 动态刷新
    td->mmTilingData.singleCoreN     = N;
    td->mmTilingData.singleCoreK     = K;
    td->mmTilingData.baseM           = 16;  // 动态刷新
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
    td->mmTilingData.dbL0C           = 2;  // 动态刷新
}

// GmmComputeOp wrapper: ProcessExperts(0, e)
static void gmmComputeOpWrapper(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR scaleA, GM_ADDR scaleB,
                                GM_ADDR y, GM_ADDR workspace,
                                GM_ADDR taskTilingBuf, GM_ADDR gmmBaseTilingBuf)
{
    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(taskTilingBuf);
    auto *baseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(gmmBaseTilingBuf);

    AscendC::TPipe tPipe;
    MC2KernelTemplate::GmmComputeOp<hifloat8_t, hifloat8_t, float, float, bfloat16_t,
                                     CubeFormat::ND, false, true, true> op;
    op.Init(x, weight, bias, scaleA, scaleB, y, workspace, taskTiling, baseTiling,
            reinterpret_cast<TILING_TYPE *>(&baseTiling->gmmArray), &tPipe);
    op.ProcessExperts(0, static_cast<uint32_t>(taskTiling->e));
}

/**
 * Case1: 单专家，小矩阵 (M=2, K=3, N=2)
 * 复用 GmmASWKernel Case1 的数据验证 GmmComputeOp 的基本正确性
 * x1_scale=2.0, x2_scale=2.0
 * 预期输出 (bf16): [[64, -64], [32, -32]]
 */
TEST_F(GmmComputeOpArch35Test, Case1_SingleExpert_SmallMatrix)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 2, K = 3, N = 2;
    constexpr uint32_t E = 1;
    uint32_t NumBlocks = 36;

    // 1. TaskTilingInfo
    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::TaskTilingInfo)));
    memset(taskTiling, 0, sizeof(*taskTiling));
    taskTiling->H1 = K;
    taskTiling->N1 = N;
    taskTiling->epWorldSize = 1;
    taskTiling->e = E;
    taskTiling->sendCnt[0] = M;  // 专家0 有 M 个 token

    // 2. gmmBaseTiling
    auto *gmmBaseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillBaseTiling(gmmBaseTiling, K, N);

    // 3. 数据 buffer
    int8_t *xData = (int8_t *)AscendC::GmAlloc(M * K * sizeof(int8_t));
    int8_t xVals[] = {8, -96, -94, 20, -108, -112};
    memcpy(xData, xVals, sizeof(xVals));

    int8_t *wData = (int8_t *)AscendC::GmAlloc(E * K * N * sizeof(int8_t));
    // bTrans=true: (N, K) 布局
    int8_t wVals[] = {-120, -108, -120, 8, 20, 8};
    memcpy(wData, wVals, sizeof(wVals));

    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 2.0f;  // x2_scale (weight scale)

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 2.0f;  // x1_scale (perTokenScale)

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    // 4. workspace（指针表 64B + groupList 8B + 余量）
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
    memset(workspace, 0, 4096);

    // 5. ICPU_RUN_KF 拉起
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmComputeOpWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR) nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)workspace,
                (GM_ADDR)taskTiling, (GM_ADDR)gmmBaseTiling);

    // 6. 校验
    float expected[] = {64.0f, -64.0f, 32.0f, -32.0f};
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1.0f) << "Case1 index " << i;
    }

    // 7. 释放
    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(yData);
    AscendC::GmFree(workspace);
    AscendC::GmFree(taskTiling);
    AscendC::GmFree(gmmBaseTiling);
}

/**
 * Case2: 多专家 (e=2)，验证 token 偏移和 weight 偏移
 * 专家0: M=2, 专家1: M=4
 * 均使用相同 K=3, N=2, x1_scale=2.0, x2_scale=2.0
 * 专家0 数据复用 Case1，专家1 数据复用 GmmASWKernel Case3
 */
TEST_F(GmmComputeOpArch35Test, Case2_MultiExpert_TokenOffset)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M0 = 2, M1 = 4, K = 3, N = 2;
    constexpr uint32_t E = 2;
    constexpr uint32_t totalM = M0 + M1;
    uint32_t NumBlocks = 36;

    // 1. TaskTilingInfo
    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::TaskTilingInfo)));
    memset(taskTiling, 0, sizeof(*taskTiling));
    taskTiling->H1 = K;
    taskTiling->N1 = N;
    taskTiling->epWorldSize = 1;
    taskTiling->e = E;
    taskTiling->sendCnt[0] = M0;
    taskTiling->sendCnt[1] = M1;

    // 2. gmmBaseTiling
    auto *gmmBaseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillBaseTiling(gmmBaseTiling, K, N);

    // 3. 数据 buffer — x: 专家0 (M0=2) + 专家1 (M1=4) 连续排列
    int8_t *xData = (int8_t *)AscendC::GmAlloc(totalM * K * sizeof(int8_t));
    // 专家0 的 x (复用 Case1)
    int8_t x0Vals[] = {8, -96, -94, 20, -108, -112};
    memcpy(xData, x0Vals, sizeof(x0Vals));
    // 专家1 的 x (复用 GmmASWKernel Case3)
    int8_t x1Vals[] = {-108, 20, 0, 8, 32, 20, 16, -108, 20, 16, -108, 0};
    memcpy(xData + M0 * K, x1Vals, sizeof(x1Vals));

    // weight: (E, N, K) 布局 (bTrans=true)，两个专家的 weight 连续
    int8_t *wData = (int8_t *)AscendC::GmAlloc(E * K * N * sizeof(int8_t));
    // 专家0 weight (复用 Case1)
    int8_t w0Vals[] = {-120, -108, -120, 8, 20, 8};
    memcpy(wData, w0Vals, sizeof(w0Vals));
    // 专家1 weight (复用 GmmASWKernel Case3)
    int8_t w1Vals[] = {-120, 20, -108, 0, -120, 20};
    memcpy(wData + K * N, w1Vals, sizeof(w1Vals));

    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 2.0f;

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 2.0f;

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(totalM * N * sizeof(bfloat16_t));
    memset(yData, 0, totalM * N * sizeof(bfloat16_t));

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
    memset(workspace, 0, 4096);

    // 5. ICPU_RUN_KF 拉起
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmComputeOpWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR) nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)workspace,
                (GM_ADDR)taskTiling, (GM_ADDR)gmmBaseTiling);

    // 6. 校验 — 专家0 输出
    float expected0[] = {64.0f, -64.0f, 32.0f, -32.0f};
    for (uint32_t i = 0; i < M0 * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected0[i], 1.0f) << "Case2 expert0 index " << i;
    }
    // 校验 — 专家1 输出
    float expected1[] = {48.0f, -12.0f, 8.0f, 20.0f, -80.0f, 48.0f, -44.0f, 12.0f};
    for (uint32_t i = 0; i < M1 * N; i++) {
        float actual = static_cast<float>(yData[M0 * N + i]);
        EXPECT_NEAR(actual, expected1[i], 1.0f) << "Case2 expert1 index " << i;
    }

    // 7. 释放
    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(yData);
    AscendC::GmFree(workspace);
    AscendC::GmFree(taskTiling);
    AscendC::GmFree(gmmBaseTiling);
}

/**
 * Case3: 某专家 token=0，验证跳过逻辑
 * 3 个专家: 专家0 M=2, 专家1 M=0, 专家2 M=4
 * 专家1 被跳过，输出只有专家0 和 专家2
 */
TEST_F(GmmComputeOpArch35Test, Case3_ZeroTokenExpert_Skip)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M0 = 2, M1 = 0, M2 = 4, K = 3, N = 2;
    constexpr uint32_t E = 3;
    constexpr uint32_t totalM = M0 + M1 + M2;
    uint32_t NumBlocks = 36;

    // 1. TaskTilingInfo
    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::TaskTilingInfo)));
    memset(taskTiling, 0, sizeof(*taskTiling));
    taskTiling->H1 = K;
    taskTiling->N1 = N;
    taskTiling->epWorldSize = 1;
    taskTiling->e = E;
    taskTiling->sendCnt[0] = M0;
    taskTiling->sendCnt[1] = M1;  // 0 tokens
    taskTiling->sendCnt[2] = M2;

    // 2. gmmBaseTiling
    auto *gmmBaseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillBaseTiling(gmmBaseTiling, K, N);

    // 3. 数据 buffer — x: 专家0 + 专家2 连续（专家1 无 token）
    int8_t *xData = (int8_t *)AscendC::GmAlloc(totalM * K * sizeof(int8_t));
    int8_t x0Vals[] = {8, -96, -94, 20, -108, -112};
    memcpy(xData, x0Vals, sizeof(x0Vals));
    int8_t x2Vals[] = {-108, 20, 0, 8, 32, 20, 16, -108, 20, 16, -108, 0};
    memcpy(xData + M0 * K, x2Vals, sizeof(x2Vals));

    // weight: 3 个专家的 weight（专家1 的 weight 存在但不会被访问）
    int8_t *wData = (int8_t *)AscendC::GmAlloc(E * K * N * sizeof(int8_t));
    memset(wData, 0, E * K * N);
    int8_t w0Vals[] = {-120, -108, -120, 8, 20, 8};
    memcpy(wData, w0Vals, sizeof(w0Vals));
    // 专家1 的 weight 填 0（不会被使用）
    // 专家2 的 weight
    int8_t w2Vals[] = {-120, 20, -108, 0, -120, 20};
    memcpy(wData + 2 * K * N, w2Vals, sizeof(w2Vals));

    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 2.0f;

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 2.0f;

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(totalM * N * sizeof(bfloat16_t));
    memset(yData, 0, totalM * N * sizeof(bfloat16_t));

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
    memset(workspace, 0, 4096);

    // 5. ICPU_RUN_KF 拉起
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmComputeOpWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR) nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)workspace,
                (GM_ADDR)taskTiling, (GM_ADDR)gmmBaseTiling);

    // 校验 — 专家0 输出 (前 M0*N 个)
    float expected0[] = {64.0f, -64.0f, 32.0f, -32.0f};
    for (uint32_t i = 0; i < M0 * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected0[i], 1.0f) << "Case3 expert0 index " << i;
    }
    // 校验 — 专家2 输出 (紧接专家0 输出之后，因为专家1 被跳过)
    float expected2[] = {48.0f, -12.0f, 8.0f, 20.0f, -80.0f, 48.0f, -44.0f, 12.0f};
    for (uint32_t i = 0; i < M2 * N; i++) {
        float actual = static_cast<float>(yData[M0 * N + i]);
        EXPECT_NEAR(actual, expected2[i], 1.0f) << "Case3 expert2 index " << i;
    }

    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(yData);
    AscendC::GmFree(workspace);
    AscendC::GmFree(taskTiling);
    AscendC::GmFree(gmmBaseTiling);
}

/**
 * Case4: 较大 M (M=300)，验证 baseM/singleCoreM/dbL0C 的动态刷新
 * 使用与 Case1 相同的 K=3, N=2 但 M=300
 * 重点验证 RefreshTilingForM 对大 M 值的处理：
 *   baseM = CeilAlign(min(300, 256), 16) = CeilAlign(256, 16) = 256
 *   singleCoreM = min(300, 256) = 256
 *   dbL0C = (256 * 32 * 4 * 2 = 65536 <= 262144) ? 2 : 1 = 2
 * 填入全零 x 数据，预期输出全零
 */
TEST_F(GmmComputeOpArch35Test, Case4_LargeM_DynamicTilingRefresh)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t M = 300, K = 3, N = 2;
    constexpr uint32_t E = 1;
    uint32_t NumBlocks = 36;

    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::TaskTilingInfo)));
    memset(taskTiling, 0, sizeof(*taskTiling));
    taskTiling->H1 = K;
    taskTiling->N1 = N;
    taskTiling->epWorldSize = 1;
    taskTiling->e = E;
    taskTiling->sendCnt[0] = M;

    auto *gmmBaseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillBaseTiling(gmmBaseTiling, K, N);

    // 全零 x 数据 → 输出全零
    int8_t *xData = (int8_t *)AscendC::GmAlloc(M * K * sizeof(int8_t));
    memset(xData, 0, M * K);

    int8_t *wData = (int8_t *)AscendC::GmAlloc(E * K * N * sizeof(int8_t));
    int8_t wVals[] = {-120, -108, -120, 8, 20, 8};
    memcpy(wData, wVals, sizeof(wVals));

    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 2.0f;

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 2.0f;

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(M * N * sizeof(bfloat16_t));
    memset(yData, 0, M * N * sizeof(bfloat16_t));

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
    memset(workspace, 0, 4096);

    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmComputeOpWrapper;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR) nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)workspace,
                (GM_ADDR)taskTiling, (GM_ADDR)gmmBaseTiling);

    // 全零输入 → 输出应全为 0
    for (uint32_t i = 0; i < M * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, 0.0f, 1.0f) << "Case4 index " << i;
    }

    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(yData);
    AscendC::GmFree(workspace);
    AscendC::GmFree(taskTiling);
    AscendC::GmFree(gmmBaseTiling);
}
