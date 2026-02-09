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

// 将 weight 从 (K, N) 布局转置为 (N, K) 布局，每个专家独立转置
static void TransposeWeightPerExpert(const int8_t *src, int8_t *dst,
                                     uint32_t E, uint32_t K, uint32_t N)
{
    for (uint32_t e = 0; e < E; e++) {
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t k = 0; k < K; k++) {
                dst[e * K * N + n * K + k] = src[e * K * N + k * N + n];
            }
        }
    }
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
    op.Init(x, weight, bias, scaleA, scaleB, y, workspace, 4096, taskTiling, baseTiling, &tPipe);
    op.ProcessExperts(0, static_cast<uint32_t>(taskTiling->e));
}

// GmmComputeOp wrapper (bTrans=false): weight 按 (K, N) 布局，不转置
static void gmmComputeOpWrapperNoTrans(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                       GM_ADDR scaleA, GM_ADDR scaleB,
                                       GM_ADDR y, GM_ADDR workspace,
                                       GM_ADDR taskTilingBuf, GM_ADDR gmmBaseTilingBuf)
{
    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(taskTilingBuf);
    auto *baseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(gmmBaseTilingBuf);

    AscendC::TPipe tPipe;
    MC2KernelTemplate::GmmComputeOp<hifloat8_t, hifloat8_t, float, float, bfloat16_t,
                                     CubeFormat::ND, false, false, true> op;
    op.Init(x, weight, bias, scaleA, scaleB, y, workspace, 4096, taskTiling, baseTiling, &tPipe);
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

/**
 * Case5: 5 专家，每专家 M=10，K=30, N=10
 * 来自 NPU 实测 gmmt2.log Run 1，scaleA=2.0, scaleB=1.0
 * NPU 验证误差 = 0.0
 */
TEST_F(GmmComputeOpArch35Test, Case5_FiveExperts_K30N10)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t K = 30, N = 10, E = 5;
    constexpr uint32_t totalM = 50;
    // CPU 模拟器限制：ProcessExperts 循环中 workspace（指针表+groupList）位于 GmAlloc 共享内存，
    // 多进程 fork 会导致 workspace 竞争覆盖。使用 NumBlocks=1 避免该问题。
    uint32_t NumBlocks = 1;

    // 1. TaskTilingInfo
    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::TaskTilingInfo)));
    memset(taskTiling, 0, sizeof(*taskTiling));
    taskTiling->H1 = K;
    taskTiling->N1 = N;
    taskTiling->epWorldSize = 1;
    taskTiling->e = E;
    taskTiling->sendCnt[0] = 10;
    taskTiling->sendCnt[1] = 10;
    taskTiling->sendCnt[2] = 10;
    taskTiling->sendCnt[3] = 10;
    taskTiling->sendCnt[4] = 10;

    // 2. gmmBaseTiling
    auto *gmmBaseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillBaseTiling(gmmBaseTiling, K, N);

    // 3. x 数据 (50x30 = 1500, hif8 int8 存储，来自 gmmt2.log x_hif8)
    static const int8_t xVals[1500] = {
        32, 16, 20, 8, 16, -94, -120, -120, -108, 32, -94, -96, 32, -112, -112, -120, 20, 0, 8, 32, 0, -112, -108, 32, -94, 32, 32, 0, 0, -120,
        0, 20, -120, 20, -108, -108, -96, 0, 8, 16, -94, -108, -96, 16, -94, -108, 32, -112, 32, 20, -108, 8, -108, 8, -94, 8, 0, 20, -108, -120,
        32, 20, 8, -120, 0, 32, -108, -120, -94, 20, 8, 8, -94, 32, -112, 8, -108, -94, 20, -112, -120, -94, -108, 32, -108, -112, -120, -120, -120, -120,
        -120, -112, -94, -108, 8, -96, 16, 8, -108, 0, 32, 20, -120, 32, 0, 8, 8, -94, -120, -108, -120, 32, -108, -94, -96, 8, -108, 16, 8, 16,
        32, -108, -112, 32, -94, 0, -108, 32, -94, -108, -94, 20, 8, 8, 8, 0, -112, -112, -112, 32, 16, -112, -94, 20, -120, -108, 16, -96, 16, -94,
        0, -96, -96, -112, 8, -112, -94, 32, 32, 16, 20, 16, 16, -112, -120, -108, -96, 20, -94, 32, 16, -94, 8, 20, -112, -112, 16, -112, 20, 0,
        -108, -120, 32, 8, 20, 8, -96, -108, 0, 16, -94, -112, -108, -94, 20, 20, -94, -112, -94, 0, 8, 20, 32, 8, 8, 8, -120, 8, 0, 32,
        32, 20, -112, 16, -94, 20, 20, -120, 8, -108, 8, -120, -120, -120, 8, 20, -120, 20, -120, -94, 16, 20, 8, -96, -94, 16, -112, -108, 20, 16,
        0, -120, -108, -108, -94, 0, 20, 0, -96, -96, 32, -96, 20, -96, -112, -108, 32, 32, -96, -94, -94, 20, 32, -120, -96, -94, 8, 32, 20, -94,
        32, -94, -108, -120, 16, 8, -108, -94, -108, -120, 0, 8, -108, 32, 16, 32, -120, -96, 16, -108, -120, 8, -96, 16, 0, 20, 20, 32, 20, 16,
        20, 20, -112, 20, -96, -94, 16, 0, 8, -120, 8, 32, 20, -94, 8, 32, -108, -120, 16, 20, -94, -120, -120, 20, 16, -108, 0, -108, 32, 16,
        -96, -112, -108, 16, 8, 32, 16, -94, 32, 8, -96, 32, -96, -108, 32, 32, -112, -120, -96, 32, -120, -96, -108, -96, 16, -120, 16, -112, 8, -120,
        -112, -108, 8, 0, -94, 0, 0, 20, 32, 0, -94, -120, 0, 16, -120, -120, 16, 16, 16, -94, 32, 0, -112, -112, -96, 0, 0, -94, 0, -94,
        -112, -120, -108, -112, -112, -120, -108, 32, -96, 8, 32, 16, -96, -108, -120, 8, -94, -120, -96, -120, -96, 16, -112, 0, -96, 0, 8, -94, 0, -96,
        -96, -96, -94, 0, 0, -94, -120, -120, -108, 0, 16, -94, 16, 8, -108, -120, -120, -94, 0, -120, -94, -120, -108, -120, 32, -96, -96, -94, -96, -94,
        20, 8, 8, -96, -120, 8, 8, 20, 32, -96, 8, 16, -112, 32, 0, -94, -112, -96, -120, -94, -96, 16, 20, 20, -96, 20, 0, -120, -108, -108,
        -94, 8, 0, -120, -120, 20, 20, -112, 8, 8, -112, -96, 8, -120, 16, 20, 0, 20, -96, 16, 20, 20, 0, -94, -120, -112, 20, -94, 0, -112,
        32, -120, -108, 16, -120, 0, 32, 0, -96, -120, 0, -120, 8, 8, 32, 32, 16, 8, 16, -112, 32, -94, -108, -120, 8, 20, -94, -120, -94, 32,
        0, 32, 16, 20, -96, 8, -120, -108, -96, -96, -120, 20, 8, -120, -96, 0, -112, 8, 20, 20, 0, -94, -120, -108, 16, 32, 16, 0, 16, -112,
        -120, -120, 16, 32, 32, -96, -112, 8, 8, 20, -94, 0, -96, -112, -120, -120, -120, -108, -94, -94, 32, 8, -108, 32, 32, -120, 8, 16, 20, -120,
        8, 8, -108, -120, 16, 20, -112, 20, -108, -108, 20, -112, -108, 32, -94, 32, -108, -96, 0, 0, 8, 16, -108, -112, 16, -108, -112, 0, -96, -120,
        8, 0, 32, -96, -94, -96, 16, 20, -120, -108, 0, -96, -94, 32, -96, -112, 16, 0, 0, 32, -108, 8, 20, 0, 32, -96, -108, 8, -112, -94,
        -108, -96, 20, -112, 8, -112, -94, 16, -96, -94, 20, 32, 20, 32, -112, 8, -112, -120, 16, -96, -112, 0, -108, -96, -112, -94, 8, 32, -120, 32,
        32, -112, -120, 0, 8, 20, -112, -108, 16, -108, 20, 32, -94, -112, -108, 32, 8, -112, 8, 20, 32, -94, 32, -120, 8, -94, 0, 32, -108, 0,
        -94, 0, 0, 8, -96, -108, 8, 8, -112, 16, 0, -112, 32, 0, -120, -112, 32, 8, 8, -96, -108, -120, 8, -120, 0, 8, 32, 0, -120, 32,
        -120, -96, -108, -96, 16, -108, -112, -96, -120, -120, 32, 20, -94, 20, -120, 0, -112, 0, 20, 16, -94, -94, -108, 20, -120, -112, -108, -108, 8, 8,
        -94, 8, 0, 16, -94, -94, -94, -108, -108, -96, -120, -108, 20, -120, 0, -94, -120, 32, 16, 16, 0, 32, 8, -120, -120, 16, -96, -108, -120, 8,
        -108, 20, 32, 0, 32, -96, 16, -120, -94, -108, -96, -94, 20, 0, 0, -108, -94, 32, -112, 8, 16, -108, 0, -120, 0, -108, -94, -120, -96, 20,
        -120, -94, -112, 8, 0, 16, -108, -96, 8, -108, 20, 16, 16, 20, 32, -120, 20, -112, -108, 0, 0, 16, -112, 32, 0, 16, -120, 16, -96, -112,
        8, 16, 0, 0, -112, -94, -94, 16, 16, -112, 16, 8, -120, 32, -94, 8, -120, 16, -120, -96, -112, 32, -94, 8, -94, 32, -120, 0, 32, 0,
        8, -108, 8, -120, -94, 16, 0, 20, -120, -120, 20, 8, 32, -94, -120, 8, -108, -94, -96, 20, 32, -120, 16, -112, 0, 0, -94, 8, 8, 0,
        -96, -120, 32, -120, -94, -112, -108, 20, -108, 8, 20, 0, 32, -120, -108, -108, -108, -94, 20, 16, 8, 8, -112, -94, 32, 8, -108, -96, 0, 32,
        0, 20, 32, -108, -112, -96, 16, 32, 8, 20, 0, -112, -120, -108, -96, 16, -96, 16, -108, 16, 8, 8, 8, 16, -108, 32, 16, -94, 20, -112,
        -108, -112, 32, 16, -108, 16, -96, 8, -108, 16, 16, -108, 32, -94, -120, -112, -112, -112, -108, -108, -108, -120, 0, 0, 20, -112, 20, 8, 0, 20,
        -96, 20, 8, 0, 8, 16, 20, -108, -112, -96, -120, 0, 20, -96, 32, -120, -120, -94, -112, -112, -108, 20, 16, 16, 16, -108, 20, 32, -120, 8,
        -108, -108, 32, -94, -112, 32, -96, -96, -94, -108, -96, -108, 32, 32, 20, -96, -112, -108, -108, 20, 16, -112, -94, -96, -120, 32, 8, 20, 8, -112,
        -120, 32, -108, 20, 20, 0, -94, -112, 20, 8, -120, -94, -96, -108, -108, 8, 32, 0, 20, 16, 8, 16, 20, 32, -120, 0, -108, 0, -94, 20,
        16, 32, 8, 0, 20, -108, 20, -94, -120, -108, -108, -108, 0, 16, 16, -96, 32, -120, -108, -108, 32, -108, 32, -120, 8, 32, -112, 8, -96, -94,
        20, -94, -94, -96, 32, -112, 32, -94, -108, 16, -96, -120, 8, 32, -96, -94, -96, -96, 0, 16, 0, -112, -112, -120, -120, -108, 0, -94, -108, 16,
        -108, -96, -108, 32, 0, 8, -120, 0, 32, -94, -112, 16, -120, 16, 8, -96, -120, -120, -94, -112, -112, -120, 8, -108, -120, -94, -94, 20, 0, 20,
        -120, 32, 16, -120, -120, -120, -120, 8, 8, -112, -112, 20, 16, -96, 20, 32, -120, -120, -108, -94, -108, -108, 20, -112, 20, -94, 16, -94, -112, -108,
        -94, 20, -120, -96, -108, 16, 0, -120, -108, 16, -94, 32, -120, 32, 8, -96, -96, -108, 8, -108, -112, 16, 32, 0, -94, -94, -112, -94, 16, 16,
        32, -108, 20, -120, -94, -108, -94, 16, 8, 20, -108, 0, 0, -96, -94, -108, 20, 16, 32, 8, 20, 32, -94, 32, -96, 20, -94, -120, 20, -108,
        -94, -108, 20, -96, 20, -94, -94, -112, -108, 8, 16, -96, -96, 0, 32, 8, -94, 32, -120, 8, -108, -108, 8, -112, -94, -108, 16, 20, 16, -112,
        32, -94, 32, -108, -96, -94, -96, 32, -120, -112, -108, -94, 20, 8, 16, -112, -112, 32, 16, -96, -94, -108, -94, -94, -108, -108, -108, -96, 20, 32,
        -120, -112, 32, 8, 8, 0, 32, 0, 20, 20, -112, -94, -94, 32, -120, 8, 16, 8, -108, 0, 8, -112, -96, -96, -108, 32, 32, -94, -108, 20,
        0, 8, 20, 0, -112, 0, -96, 20, 32, -120, 8, 0, 20, 32, 8, 32, 8, 16, -96, -96, 32, -120, 0, 16, -94, -120, -96, -108, 0, -112,
        20, -94, -120, 32, -120, 16, 20, -96, 8, 16, -112, -120, -120, 0, -120, 20, -108, -108, -96, -94, -94, -94, -108, -112, 0, -108, 0, 0, 16, 0,
        -96, -112, -112, 8, -108, -112, -112, 16, 16, 0, 0, 16, -108, 0, 32, 8, -112, -108, 32, 8, -94, -94, -96, -120, -94, -94, -96, -94, 32, 32,
        -94, -112, 32, -94, 8, 0, -94, -108, -108, 32, 0, -96, -96, 8, 20, 32, -108, 20, 16, -94, -96, 20, -120, -96, -120, -94, -108, -112, -120, 20
    };
    int8_t *xData = (int8_t *)AscendC::GmAlloc(totalM * K * sizeof(int8_t));
    memcpy(xData, xVals, sizeof(xVals));

    // 4. weight 数据 (5x30x10 = 1500, hif8 int8 (K,N) 布局，需转置为 (N,K))
    static const int8_t wValsKN[1500] = {
        // expert 0 (30x10)
        -120, -112, -94, -96, 16, -120, 8, 16, 16, 20,
        -112, -94, 0, 8, -120, -112, 32, 8, 8, 16,
        -108, 0, -108, -96, -112, 32, -120, -112, 16, 20,
        -108, -112, -120, -96, 0, -120, -112, -108, 20, -94,
        16, 8, -94, 20, 8, 0, 20, -120, 8, 20,
        -96, 8, -108, 32, -120, -94, 0, -112, 8, 32,
        8, 20, -94, -120, 8, 16, 8, -112, -112, 8,
        8, -96, -96, -108, 0, -120, -112, -120, 8, 0,
        -112, 8, 16, 20, -94, 0, 32, 20, -112, 20,
        -112, -108, 8, 20, 16, 0, -108, -108, 16, -120,
        8, -94, -112, -120, 20, 8, -120, 16, 16, -108,
        16, -96, -120, 8, -94, -94, -112, -112, 32, -120,
        8, 8, 32, 32, -112, 16, 32, 8, 20, 20,
        0, 16, -94, 16, 0, 32, -108, -120, -96, 32,
        -108, -112, -96, -108, 8, 0, 32, 32, 8, 20,
        20, -108, 20, -94, -112, 0, -120, 0, 32, -112,
        20, 16, 20, 8, -112, -96, 0, 32, -94, -96,
        -108, 32, -94, -96, -112, -108, -120, 16, 0, -94,
        8, -108, 16, 20, 8, -94, -94, 32, -120, 8,
        -120, -120, -108, 0, 20, 8, -108, 32, -108, 20,
        -120, -96, -112, -94, -112, 20, -96, -112, 8, -112,
        32, -112, -94, -96, -112, -94, -108, 32, 20, -112,
        0, 0, -108, -94, 32, -96, 20, 8, -96, -120,
        -108, 0, 20, -108, 16, -96, 16, 20, -120, 20,
        -108, 0, 0, 0, 32, -108, 8, 16, -120, -108,
        -120, 20, 8, 32, 16, 8, 20, 20, 20, -120,
        8, -108, 8, 8, -94, -108, -120, -108, -96, -96,
        20, -112, -108, 32, -108, -120, -94, -112, -94, -96,
        -96, 8, -120, 0, -108, 20, 8, -112, 16, -112,
        8, -94, 8, 16, -96, 20, 16, 0, 20, -94,
        // expert 1 (30x10)
        16, -96, 0, -94, -94, 16, -112, -96, -112, -108,
        -108, -112, 8, -112, -120, -96, 20, 8, 20, 16,
        0, 0, -108, 8, 8, 20, -120, -94, 20, -108,
        0, 0, 16, -108, 20, -108, -112, 20, 20, -108,
        -94, 32, -96, 32, -94, 0, -96, -96, 20, 16,
        20, -112, 20, 8, 0, 8, 0, -112, -94, -112,
        -120, -94, 32, -94, -120, -96, -108, 0, -108, 32,
        20, -96, 0, -94, 32, -120, 32, 20, 32, -120,
        -96, 20, 16, 32, -96, 32, 16, 0, 16, 8,
        0, 20, 20, 16, 20, 8, -94, -120, 8, 8,
        -108, 8, 8, -96, -96, 20, 32, 16, 32, 16,
        -108, -112, -108, 16, 8, 32, 16, -94, 16, 0,
        20, -96, -96, 32, -112, -96, 16, 32, 20, -96,
        -112, 8, 20, -96, -94, 32, 32, 32, -120, 16,
        32, -96, -96, 16, -108, -112, -112, -96, -112, 20,
        0, -96, -120, 0, -112, -96, -94, -120, 0, -96,
        20, -96, -108, -94, 20, -120, 8, -108, -112, 20,
        32, -112, 8, -96, -112, -96, -120, -94, 32, 0,
        -112, -94, 32, -96, 20, 8, -112, 20, -112, 16,
        16, -120, 20, 20, -94, -120, -94, 0, 20, 16,
        0, -96, -96, -120, -120, -94, 20, -112, -94, 32,
        -94, -94, 20, -120, 16, 16, -108, -108, 16, 16,
        -120, 16, -96, -94, -96, -120, 16, -120, -94, 16,
        8, -108, -96, -94, -120, 32, 32, 8, 8, -108,
        8, 16, 32, 20, -112, -112, 20, -108, 32, 8,
        0, 32, 32, -120, 16, -108, 32, 0, 32, 20,
        -94, 0, 32, -120, -94, 8, 20, -96, -120, 32,
        -108, -112, -96, -112, -94, 8, -96, 16, 20, 8,
        0, -120, 8, -108, -108, -120, 32, 16, 20, -108,
        32, 16, 8, 32, -120, 16, -94, 8, -108, -108,
        // expert 2 (30x10)
        16, -108, -94, -120, 16, -108, 32, 16, 16, -120,
        32, 0, -108, 20, 16, -108, 32, -112, -120, -94,
        20, -96, 32, 0, 32, 20, -112, 20, -112, 8,
        -112, 32, -120, -112, 16, 0, -112, 16, 0, 0,
        -120, 20, 8, -120, -94, 8, -94, -96, -96, -108,
        -94, 32, 8, 32, 16, -96, 0, -96, -108, -120,
        -112, -94, 8, -112, -120, 0, -112, 8, -94, -96,
        -108, -120, -112, -96, -108, -94, 32, 8, 16, -112,
        -120, 20, 0, -96, -120, 16, 32, -120, -120, 20,
        16, -96, 32, 0, -108, -96, 0, 0, -112, 0,
        0, -108, 16, -108, -108, -112, 32, -96, 0, 16,
        -94, 0, 8, 32, 20, 32, -94, 8, -108, 32,
        -94, -94, 8, -120, -108, -94, 16, -108, 16, 16,
        20, 32, -96, 32, 8, -94, 20, -120, -96, 8,
        20, 0, -112, 20, 20, -94, -108, 32, 0, 32,
        -96, -120, -112, 20, 8, -96, 32, -112, -96, 20,
        32, 20, 20, -96, -120, 16, 0, -94, 32, -94,
        -94, 16, -94, 16, 20, -120, -120, 32, 32, 0,
        20, -94, 0, -94, -108, 0, -94, -94, 32, 16,
        16, 20, -112, -96, 0, 20, 16, -108, 8, 32,
        -94, 20, 16, 8, 32, 0, -94, 20, -94, -112,
        32, -108, 0, 16, -96, 0, 8, 20, 8, 32,
        8, 0, -120, 0, 0, -96, 20, 0, -120, -94,
        -120, 16, -96, 20, -94, 8, -120, -94, 0, -94,
        -120, 0, 32, -96, 0, 32, 32, -112, 8, -120,
        16, 0, -108, 0, 20, -108, 8, 20, -96, -120,
        -108, 0, 16, 16, 32, -120, -108, -120, -96, -108,
        16, -96, -108, -94, -120, 20, -96, 32, 16, 8,
        -108, -96, -108, 0, -94, 8, 20, -112, 32, -96,
        -94, -96, -94, 32, -108, 20, 20, 0, -108, -120,
        // expert 3 (30x10)
        8, -96, -96, -120, 16, -96, -96, 32, -108, -120,
        -94, 8, -96, -96, 8, -94, 32, -94, 0, 8,
        20, -94, -120, -96, -94, -94, -112, 8, -112, 20,
        -108, -112, -120, 0, -120, 0, 8, -120, 32, 16,
        0, -108, -108, 16, -94, 32, 8, -120, -108, -96,
        -112, -108, -112, -96, 8, -96, 32, 32, 0, 20,
        -94, -108, 8, 8, 32, 8, -112, -112, -120, -96,
        -108, 0, 0, 8, 32, -96, -94, 8, -112, 0,
        32, 32, -112, 0, -94, -94, -108, 0, -94, -96,
        -108, 0, 8, -94, -96, 16, -96, -96, -112, -120,
        -120, 0, -120, 32, 8, -120, 16, -94, 8, 8,
        -108, -112, 32, 0, 8, -96, -96, -108, 8, 0,
        -96, 20, 20, 32, -94, -108, -94, -120, 0, -94,
        16, 8, 0, -112, -112, -120, 32, -120, -120, 20,
        0, -96, 8, 32, 32, -96, -120, -108, 20, 8,
        -96, 20, 8, -94, -108, -108, -108, -96, -112, -108,
        20, 32, -94, -94, 16, -108, -120, -108, 16, 8,
        8, 0, 0, -112, 32, -108, -120, 20, 0, 16,
        32, -94, -94, -108, 16, -94, -108, -94, 8, 0,
        -112, -108, 16, -108, 0, 0, -120, -112, 32, 8,
        16, 8, -96, -112, -108, -108, 32, 8, 32, 0,
        32, -108, -108, -108, 8, -96, 20, 8, -112, -94,
        8, 16, -108, -112, 8, -112, 16, -112, 32, -94,
        20, 16, 32, -120, 16, -94, -120, -112, -120, 32,
        16, -94, 20, 16, 20, -94, 0, -112, 16, -112,
        -120, -96, -108, 8, 8, -96, -112, 32, 32, -120,
        20, -120, -120, 0, -120, -112, -94, -96, -108, 8,
        20, 20, 0, -120, 20, 0, 32, 16, -108, 20,
        0, 16, 16, -108, 16, 32, -108, -96, 0, -96,
        -112, -112, 20, -108, -96, 8, 20, -108, 8, 16,
        // expert 4 (30x10)
        8, 20, 0, -94, 0, -108, -94, 0, -120, 0,
        0, -112, 32, -96, 20, -96, 0, 20, -96, -108,
        32, 20, 0, 16, -96, 16, -112, 32, -108, -108,
        16, 8, -94, 32, 8, 8, -96, 0, -108, 8,
        -120, -120, -112, 8, -108, -94, 8, -108, 8, -112,
        -112, 0, -112, -120, -112, -108, 8, 20, 32, 0,
        -108, -112, 16, 0, -108, -94, 8, 20, 32, 8,
        -94, -120, 20, 16, 20, 8, 8, 16, 16, 0,
        8, 16, -112, 20, -112, -120, 0, -112, 32, -94,
        -112, 32, -120, 0, -108, 0, -112, 32, -96, 8,
        -94, -120, -96, -120, 20, 20, -94, -120, 16, 16,
        16, 0, -112, -112, -108, -94, 0, -120, 8, 32,
        -108, -108, 16, 32, -112, 0, 0, 20, -120, 8,
        0, -96, -94, -112, 0, -96, -96, -108, -112, 0,
        -94, 16, -94, 8, -120, 16, 20, 0, 8, 8,
        0, 32, -96, -120, -108, 0, 16, 32, 32, -96,
        8, -108, -112, 20, -94, -120, 20, 16, -94, -108,
        20, 20, -112, 8, 32, 16, 20, -96, 32, -94,
        -120, 8, 20, 0, 16, -96, -112, -108, 20, -120,
        0, -120, 32, 20, 8, 32, 0, 8, -94, -108,
        8, -120, -112, -120, 8, 16, -108, 8, 20, -94,
        -96, 8, 32, -94, 8, 0, -94, -112, -120, 0,
        -96, -94, 32, 20, 32, -94, -96, 8, 16, -112,
        0, 20, 32, 16, 0, 8, -94, -96, -108, -120,
        32, 0, -120, -120, -96, 20, -120, -120, 0, -108,
        -94, 8, 8, 16, -108, -112, -112, 16, -108, -120,
        0, -120, -94, -120, 0, -112, 16, 20, -94, 16,
        20, 32, 32, -96, 16, -108, -108, -96, -120, -112,
        0, 32, 0, 0, -94, -94, -96, 0, 8, -120,
        32, 0, -112, -112, 20, 16, -120, -94, -112, -96
    };
    int8_t *wData = (int8_t *)AscendC::GmAlloc(E * K * N * sizeof(int8_t));
    memcpy(wData, wValsKN, E * K * N * sizeof(int8_t));

    // 5. scale
    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 1.0f;

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 2.0f;

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(totalM * N * sizeof(bfloat16_t));
    memset(yData, 0, totalM * N * sizeof(bfloat16_t));

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
    memset(workspace, 0, 4096);

    // 6. ICPU_RUN_KF 拉起 (bTrans=false，weight 保持 K×N 布局)
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmComputeOpWrapperNoTrans;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR) nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)workspace,
                (GM_ADDR)taskTiling, (GM_ADDR)gmmBaseTiling);

    // 7. 校验 (50x10, 来自 gmmt2.log expected_out bfloat16)
    static const float expected[500] = {
        -12.f, 38.f, 128.f, 44.f, 10.f, 78.f, 14.f, 20.f, -18.f, 62.f,
        64.f, 12.f, 148.f, 136.f, -16.f, -56.f, -166.f, 46.f, -166.f, -2.f,
        -58.f, -114.f, 44.f, 62.f, 102.f, 10.f, -98.f, -124.f, 46.f, 194.f,
        264.f, -62.f, -36.f, 110.f, -34.f, 130.f, -90.f, -38.f, 64.f, -70.f,
        -80.f, -14.f, 34.f, -172.f, 10.f, 14.f, -116.f, -78.f, 60.f, 98.f,
        -86.f, -24.f, 40.f, -4.f, -12.f, 78.f, 54.f, -38.f, 6.f, 8.f,
        -70.f, -34.f, 0.f, -114.f, -4.f, 40.f, 98.f, -62.f, 122.f, -28.f,
        -32.f, -12.f, -76.f, -162.f, -100.f, 22.f, 76.f, 0.f, 172.f, -74.f,
        118.f, 168.f, -16.f, -86.f, -4.f, -96.f, 34.f, -2.f, -160.f, -146.f,
        76.f, -32.f, 28.f, 106.f, -44.f, 4.f, -80.f, -20.f, 32.f, -38.f,
        42.f, -130.f, 50.f, 8.f, -26.f, -36.f, -38.f, 76.f, 138.f, -178.f,
        2.f, 134.f, 100.f, 336.f, -54.f, -70.f, -166.f, -148.f, -60.f, 28.f,
        48.f, -76.f, 38.f, -82.f, 252.f, -34.f, 162.f, 32.f, -124.f, 14.f,
        -44.f, 92.f, 58.f, 8.f, 176.f, 78.f, 86.f, 32.f, 82.f, -50.f,
        2.f, 176.f, 68.f, 132.f, 182.f, 12.f, 22.f, 244.f, 32.f, -32.f,
        -146.f, 48.f, -16.f, -144.f, 2.f, 256.f, 252.f, 36.f, -58.f, 52.f,
        28.f, -32.f, 98.f, 130.f, -2.f, -208.f, -70.f, -108.f, -100.f, 94.f,
        244.f, -138.f, -32.f, -64.f, 68.f, -202.f, -118.f, 30.f, -162.f, -14.f,
        14.f, 10.f, 118.f, -20.f, 30.f, -130.f, 108.f, 44.f, 80.f, -42.f,
        -102.f, 122.f, -112.f, 82.f, 80.f, 24.f, 34.f, -80.f, 102.f, -50.f,
        10.f, 94.f, 14.f, 8.f, -38.f, -48.f, 118.f, -98.f, -72.f, 66.f,
        300.f, 6.f, 32.f, -202.f, 2.f, 106.f, 176.f, 12.f, 178.f, -30.f,
        -98.f, -132.f, 24.f, 50.f, -42.f, 88.f, -96.f, 18.f, 44.f, 202.f,
        -92.f, 132.f, 40.f, -84.f, 114.f, 158.f, -58.f, -90.f, -64.f, 30.f,
        -10.f, -132.f, 64.f, -22.f, -40.f, -30.f, 14.f, -10.f, 58.f, -94.f,
        42.f, 56.f, -52.f, 34.f, -122.f, 156.f, -22.f, -176.f, 52.f, 110.f,
        128.f, 22.f, -56.f, 12.f, 20.f, 68.f, 10.f, 108.f, 224.f, 114.f,
        -14.f, 8.f, -40.f, 52.f, 38.f, 56.f, -88.f, 128.f, -16.f, -48.f,
        48.f, 178.f, 22.f, 46.f, 24.f, 20.f, -82.f, -24.f, -22.f, 154.f,
        54.f, -32.f, -186.f, 80.f, -64.f, -10.f, 144.f, 74.f, 86.f, 60.f,
        -152.f, 46.f, 108.f, 86.f, -18.f, 24.f, -4.f, 104.f, 106.f, -74.f,
        -74.f, -122.f, 100.f, 66.f, -116.f, 30.f, -12.f, -44.f, 134.f, -12.f,
        -90.f, 28.f, 10.f, -78.f, -50.f, -18.f, -188.f, 6.f, -64.f, -102.f,
        -26.f, 6.f, 146.f, 86.f, -84.f, 72.f, -14.f, 14.f, -8.f, 74.f,
        -18.f, -22.f, 82.f, 108.f, 44.f, -8.f, 116.f, -86.f, -14.f, -38.f,
        70.f, -52.f, 74.f, 88.f, -80.f, 130.f, 124.f, 240.f, 74.f, 156.f,
        102.f, 82.f, -198.f, -178.f, -82.f, -48.f, 202.f, -28.f, 54.f, 46.f,
        70.f, 22.f, -202.f, 94.f, 26.f, -10.f, 166.f, 126.f, 110.f, -26.f,
        -28.f, -68.f, 106.f, 134.f, -146.f, 370.f, 24.f, 78.f, -42.f, -66.f,
        10.f, 134.f, 126.f, 152.f, -14.f, 214.f, 202.f, 144.f, 8.f, 32.f,
        24.f, -60.f, -76.f, 40.f, -52.f, 12.f, 218.f, 180.f, 96.f, 66.f,
        -62.f, -136.f, 110.f, -76.f, 82.f, -150.f, 28.f, -6.f, 34.f, 176.f,
        62.f, 208.f, 230.f, 34.f, -32.f, 120.f, -138.f, -28.f, -90.f, -44.f,
        18.f, 108.f, -60.f, 26.f, 128.f, 122.f, 114.f, -86.f, 24.f, 30.f,
        102.f, 94.f, -46.f, 36.f, 64.f, 186.f, 136.f, -72.f, 62.f, 50.f,
        38.f, -14.f, -246.f, 68.f, -98.f, 68.f, 140.f, 126.f, -80.f, -72.f,
        -30.f, 10.f, -118.f, 78.f, 4.f, 66.f, -14.f, 56.f, 102.f, -80.f,
        124.f, 118.f, -208.f, -18.f, -158.f, -38.f, 60.f, 52.f, 78.f, 158.f,
        20.f, 58.f, -102.f, 90.f, 30.f, 90.f, 138.f, -78.f, 94.f, 124.f,
        54.f, 118.f, -110.f, -102.f, 50.f, 144.f, 128.f, -72.f, 116.f, -8.f
    };
    for (uint32_t i = 0; i < totalM * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1.0f) << "Case5 index " << i;
    }

    // 8. 释放
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
 * Case6: 3 专家，不等长 token 分布 [10,5,5]，K=10, N=13
 * 来自 NPU 实测 gmmt2.log Run 2，scaleA=1.0, scaleB=1.0
 * NPU 验证误差 = 0.0
 */
TEST_F(GmmComputeOpArch35Test, Case6_ThreeExperts_UnequalTokens_K10N13)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    constexpr uint32_t K = 10, N = 13, E = 3;
    constexpr uint32_t totalM = 20;
    // CPU 模拟器限制：ProcessExperts 循环中 workspace（指针表+groupList）位于 GmAlloc 共享内存，
    // 多进程 fork 会导致 workspace 竞争覆盖。使用 NumBlocks=1 避免该问题。
    uint32_t NumBlocks = 1;

    // 1. TaskTilingInfo
    auto *taskTiling = reinterpret_cast<MC2KernelTemplate::TaskTilingInfo *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::TaskTilingInfo)));
    memset(taskTiling, 0, sizeof(*taskTiling));
    taskTiling->H1 = K;
    taskTiling->N1 = N;
    taskTiling->epWorldSize = 1;
    taskTiling->e = E;
    taskTiling->sendCnt[0] = 10;
    taskTiling->sendCnt[1] = 5;
    taskTiling->sendCnt[2] = 5;

    // 2. gmmBaseTiling
    auto *gmmBaseTiling = reinterpret_cast<MC2KernelTemplate::GMMQuantTilingData *>(
        AscendC::GmAlloc(sizeof(MC2KernelTemplate::GMMQuantTilingData)));
    FillBaseTiling(gmmBaseTiling, K, N);

    // 3. x 数据 (20x10 = 200, hif8 int8 存储)
    static const int8_t xVals[200] = {
        -112, -112, 8, 32, -108, 20, 8, -120, 16, -112,
        8, 8, -108, 32, 32, -120, -112, 0, 20, 20,
        -120, -112, -94, 8, 32, 0, -94, 16, -96, 20,
        -112, -112, -96, 20, -112, 8, -120, -112, 8, -120,
        8, -112, 32, -112, -94, 16, 32, 8, -96, 8,
        32, -120, -96, 16, -112, -120, 20, 20, -94, 8,
        -96, -96, 16, 32, 0, -112, 20, 8, 16, -96,
        -112, 0, -108, 0, -94, -112, -112, -94, -96, 16,
        -94, 20, 20, -120, 20, -94, 16, 0, -120, 32,
        -120, 0, 32, 16, 8, -120, 8, -94, -94, -96,
        -120, 8, 0, -112, -94, 8, 32, 20, 0, -96,
        20, -108, 8, 0, -108, -112, 16, 8, -96, -120,
        -108, -112, 8, 16, -108, 0, -120, 0, 0, -94,
        8, -120, 8, -96, -96, 32, 32, -96, -94, -96,
        8, -108, 8, -112, -112, -120, 16, -108, 16, -112,
        -120, 20, -120, -94, 0, 0, -96, 0, 32, 8,
        -120, -96, 8, 16, 20, 0, 32, -120, 0, -112,
        32, -120, -120, -108, -108, 8, 20, 8, -96, 16,
        0, -120, -94, 0, 8, 20, 8, -94, -112, 0,
        16, -112, 20, 20, 20, -96, 16, -94, -120, 20
    };
    int8_t *xData = (int8_t *)AscendC::GmAlloc(totalM * K * sizeof(int8_t));
    memcpy(xData, xVals, sizeof(xVals));

    // 4. weight 数据 (3x10x13 = 390, hif8 int8 (K,N) 布局，需转置为 (N,K))
    static const int8_t wValsKN[390] = {
        // expert 0 (10x13)
        -120, 16, 0, -96, -108, 32, -108, 20, 20, -112, -120, -112, 16,
        -108, -120, -112, 16, 8, 32, 20, 20, 20, 0, -96, -108, 16,
        32, -120, 8, 20, 32, -120, -120, -112, 32, -94, 32, 32, 16,
        16, 16, -94, -96, -94, -120, -96, -112, 8, 16, 20, 16, -96,
        -120, -108, 0, -112, 8, 16, -94, 20, -120, -96, -108, 20, 16,
        -94, 0, 8, 20, 20, 32, 20, -108, 20, -94, 20, -120, 32,
        8, 16, -108, 8, -120, 32, -94, -94, 0, -120, 32, -96, -112,
        -108, 8, -108, -96, -112, -120, -108, 0, -94, 32, 32, 20, -108,
        8, 0, -112, 8, 8, 20, -108, -112, 16, -96, 32, -96, 8,
        20, 20, -108, -96, 8, 16, -94, -96, -120, 16, -94, 20, -108,
        // expert 1 (10x13)
        20, 16, -96, 32, 8, 32, -112, 8, 0, 0, 8, -108, 0,
        20, 8, -96, 16, 32, -112, 8, 32, -120, -96, 32, 16, -94,
        20, 16, -94, -108, 8, -94, 16, 8, 16, 0, -94, -96, 20,
        0, -120, -96, -112, -108, -108, -96, 16, -94, -94, -94, -94, 20,
        20, -96, 32, 8, 20, 8, -94, -112, -108, 20, -94, 0, -120,
        -120, 0, -96, -120, -94, -108, 8, 8, 16, -94, 0, -94, -94,
        20, -120, -120, -108, -96, -96, 32, 0, 0, -108, -108, -96, 16,
        -96, -108, -108, 16, -96, -112, -112, -94, -120, -112, -94, -108, 32,
        32, 20, -96, 32, 8, 20, 0, 20, 0, 20, -108, 8, -108,
        -120, -96, 32, -120, -120, 32, 8, 0, 16, -112, 0, 32, 32,
        // expert 2 (10x13)
        8, -120, -96, -96, 32, 32, -112, 20, 0, 0, 20, 32, 8,
        -120, 16, 0, 0, -108, 0, 0, -94, 32, 0, -112, 20, -112,
        16, 8, 0, 8, -108, 20, -112, -120, 20, 20, 32, -96, -112,
        -96, -96, -120, 0, 16, -112, 20, -108, 8, -120, -112, -96, 32,
        -96, 20, 32, 0, 8, -96, 16, -120, 0, 8, -108, -96, 8,
        16, 32, 8, 8, -96, -120, -112, 20, -120, 20, -96, 16, 16,
        32, -96, 16, 8, 16, 8, 16, -120, -94, 16, 20, -108, 32,
        8, 20, -94, 16, -120, 0, -108, 32, -120, -112, -96, -94, -94,
        32, -96, 0, -108, 16, 16, -120, -96, -120, 0, 8, -94, -112,
        32, -120, 16, -108, -94, -94, 8, -112, -96, -112, -108, 0, 20
    };
    int8_t *wData = (int8_t *)AscendC::GmAlloc(E * K * N * sizeof(int8_t));
    memcpy(wData, wValsKN, E * K * N * sizeof(int8_t));

    // 5. scale
    float *scaleBData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleBData[0] = 1.0f;

    float *scaleAData = (float *)AscendC::GmAlloc(sizeof(float));
    scaleAData[0] = 1.0f;

    bfloat16_t *yData = (bfloat16_t *)AscendC::GmAlloc(totalM * N * sizeof(bfloat16_t));
    memset(yData, 0, totalM * N * sizeof(bfloat16_t));

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
    memset(workspace, 0, 4096);

    // 6. ICPU_RUN_KF 拉起 (bTrans=false，weight 保持 K×N 布局)
    std::function<void(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                       GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)> func = gmmComputeOpWrapperNoTrans;
    ICPU_RUN_KF(func, NumBlocks,
                (GM_ADDR)xData, (GM_ADDR)wData, (GM_ADDR) nullptr,
                (GM_ADDR)scaleAData, (GM_ADDR)scaleBData,
                (GM_ADDR)yData, (GM_ADDR)workspace,
                (GM_ADDR)taskTiling, (GM_ADDR)gmmBaseTiling);

    // 7. 校验 (20x13, 来自 gmmt2.log expected_out bfloat16)
    static const float expected[260] = {
        8.f, 9.f, -10.f, 21.f, -5.f, -8.f, 9.f, -41.f, 19.f, -13.f, 62.f, -11.f, -7.f,
        3.f, 5.f, -35.f, -49.f, -25.f, 18.f, -50.f, 11.f, -6.f, 6.f, -31.f, 9.f, -16.f,
        -21.f, -4.f, 2.f, -56.f, -20.f, -28.f, -6.f, 32.f, -53.f, 48.f, -63.f, 53.f, -21.f,
        -2.f, 7.f, -4.f, -1.f, -21.f, -16.f, 18.f, -12.f, -7.f, 20.f, 9.f, -16.f, -16.f,
        12.f, 23.f, 10.f, 20.f, 13.f, -3.f, 10.f, -44.f, 8.f, 2.f, 37.f, 5.f, -6.f,
        -14.f, 35.f, -24.f, -55.f, -57.f, 2.f, -26.f, 1.f, -32.f, 60.f, -8.f, -2.f, -41.f,
        32.f, -3.f, -16.f, 9.f, -17.f, -37.f, -28.f, -33.f, -15.f, 1.f, 78.f, 9.f, -31.f,
        20.f, 11.f, 18.f, 9.f, -7.f, -34.f, 55.f, 1.f, -4.f, 51.f, -55.f, -6.f, -19.f,
        41.f, -11.f, -19.f, 3.f, 24.f, -11.f, -32.f, -10.f, -19.f, 16.f, -38.f, 33.f, -25.f,
        24.f, -20.f, 27.f, 35.f, 7.f, -26.f, 28.f, 12.f, 30.f, -22.f, 1.f, 15.f, 16.f,
        -12.f, 24.f, -45.f, -6.f, -35.f, -46.f, 43.f, -5.f, 15.f, -24.f, 11.f, -31.f, -7.f,
        -17.f, 4.f, -2.f, -17.f, -22.f, -10.f, 11.f, -21.f, 7.f, -5.f, 2.f, -28.f, 47.f,
        -19.f, 25.f, -24.f, -18.f, -16.f, -38.f, 4.f, 0.f, -7.f, 2.f, -8.f, -25.f, 0.f,
        -1.f, 32.f, -9.f, -37.f, -23.f, -42.f, 59.f, 7.f, 39.f, -19.f, 55.f, -34.f, -29.f,
        20.f, 32.f, -2.f, -4.f, 3.f, 8.f, 26.f, 10.f, 18.f, 31.f, 7.f, -3.f, -3.f,
        18.f, 25.f, 3.f, -16.f, -25.f, 2.f, -22.f, -16.f, 16.f, -8.f, -14.f, 21.f, -46.f,
        -8.f, -22.f, 23.f, 13.f, 31.f, -3.f, 21.f, 3.f, -22.f, 18.f, 18.f, -47.f, 31.f,
        34.f, 5.f, -19.f, -5.f, -4.f, 15.f, -14.f, 46.f, -31.f, 0.f, 16.f, 49.f, 16.f,
        -16.f, -3.f, 34.f, -5.f, 10.f, -25.f, 25.f, 5.f, -20.f, 7.f, -12.f, 51.f, 52.f,
        -11.f, -44.f, 32.f, -23.f, 22.f, -12.f, 36.f, -35.f, -8.f, 5.f, 39.f, -18.f, 51.f
    };
    for (uint32_t i = 0; i < totalM * N; i++) {
        float actual = static_cast<float>(yData[i]);
        EXPECT_NEAR(actual, expected[i], 1.0f) << "Case6 index " << i;
    }

    // 8. 释放
    AscendC::GmFree(xData);
    AscendC::GmFree(wData);
    AscendC::GmFree(scaleBData);
    AscendC::GmFree(scaleAData);
    AscendC::GmFree(yData);
    AscendC::GmFree(workspace);
    AscendC::GmFree(taskTiling);
    AscendC::GmFree(gmmBaseTiling);
}
