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
 * \file test_aclnn_fused_infer_attention_score_perf.cpp
 * \brief Vanilla kernel-execution perf benchmark for FIAS V5 on ascend910b.
 *        Times only the device-side Execute call via aclrtEvent; host-side
 *        tiling (GetWorkspaceSize) is excluded from the measurement window.
 *        V5 is the current API per CLAUDE.md (V1 deprecated, removal Dec 2026).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

namespace {

#define CHECK_ACL(x)                                                                                                   \
    do {                                                                                                               \
        aclError __err = (x);                                                                                          \
        if (__err != ACL_SUCCESS) {                                                                                    \
            printf("ACL error %d at %s:%d\n", __err, __FILE__, __LINE__);                                              \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_RET(x)                                                                                                   \
    do {                                                                                                               \
        auto __r = (x);                                                                                                \
        if (__r != ACL_SUCCESS) {                                                                                      \
            printf("aclnn error %d at %s:%d\n", __r, __FILE__, __LINE__);                                              \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

int64_t Numel(const std::vector<int64_t> &s)
{
    int64_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

aclTensor *MakeFp16Tensor(const std::vector<int64_t> &shape, void **devAddr)
{
    int64_t bytes = Numel(shape) * sizeof(uint16_t);
    CHECK_ACL(aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    // Zero-fill: kernel doesn't care about values for timing (no early-exit
    // paths on data), and we avoid host->device copy overhead.
    CHECK_ACL(aclrtMemset(*devAddr, bytes, 0, bytes));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) strides[i] = shape[i + 1] * strides[i + 1];
    return aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *devAddr);
}

struct BenchResult {
    double median_us;
    double min_us;
    double p10_us;
    double avg_us;
};

// Run FIAS V5 with the given shapes; time only the second-stage kernel call.
// layout is "BNSD". numKvHeads can be < numHeads for GQA.
BenchResult BenchFias(int64_t B, int64_t N, int64_t Nkv, int64_t Sq, int64_t Skv, int64_t D, int64_t sparseMode,
                      int warmup, int iters, aclrtStream stream)
{
    std::vector<int64_t> qShape = {B, N, Sq, D};
    std::vector<int64_t> kShape = {B, Nkv, Skv, D};
    std::vector<int64_t> vShape = {B, Nkv, Skv, D};
    std::vector<int64_t> oShape = {B, N, Sq, D};

    void *qDev = nullptr, *kDev = nullptr, *vDev = nullptr, *oDev = nullptr;
    aclTensor *qT = MakeFp16Tensor(qShape, &qDev);
    aclTensor *kT = MakeFp16Tensor(kShape, &kDev);
    aclTensor *vT = MakeFp16Tensor(vShape, &vDev);
    aclTensor *oT = MakeFp16Tensor(oShape, &oDev);

    aclTensor *kArr[1] = {kT};
    aclTensor *vArr[1] = {vT};
    aclTensorList *kList = aclCreateTensorList(kArr, 1);
    aclTensorList *vList = aclCreateTensorList(vArr, 1);

    // For sparseMode 2/3/4 the kernel needs a 2048x2048 compressed bool
    // attenMask. Contents don't matter for timing, but the tensor must
    // exist with that exact shape — see mask_checker.cpp.
    void *maskDev = nullptr;
    aclTensor *maskT = nullptr;
    if (sparseMode == 2 || sparseMode == 3 || sparseMode == 4) {
        std::vector<int64_t> maskShape = {2048, 2048};
        int64_t mBytes = 2048 * 2048;
        CHECK_ACL(aclrtMalloc(&maskDev, mBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemset(maskDev, mBytes, 0, mBytes));
        std::vector<int64_t> mStrides = {2048, 1};
        maskT = aclCreateTensor(maskShape.data(), maskShape.size(), aclDataType::ACL_BOOL, mStrides.data(), 0,
                                aclFormat::ACL_FORMAT_ND, maskShape.data(), maskShape.size(), maskDev);
    }

    char layout[8];
    std::strcpy(layout, "BNSD");
    double scaleValue = 1.0 / std::sqrt((double)D);
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    int64_t innerPrecise = 1; // high perf mode
    int64_t pseType = 0;       // no pse

    auto buildOnce = [&](aclOpExecutor **executor, uint64_t *wsSize) {
        CHECK_RET(aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
            qT, kList, vList,
            /* pseShift              */ nullptr,
            /* attenMask             */ maskT,
            /* actualSeqLengths      */ nullptr,
            /* actualSeqLengthsKv    */ nullptr,
            /* deqScale1             */ nullptr,
            /* quantScale1           */ nullptr,
            /* deqScale2             */ nullptr,
            /* quantScale2           */ nullptr,
            /* quantOffset2          */ nullptr,
            /* antiquantScale        */ nullptr,
            /* antiquantOffset       */ nullptr,
            /* blockTable            */ nullptr,
            /* queryPaddingSize      */ nullptr,
            /* kvPaddingSize         */ nullptr,
            /* keyAntiquantScale     */ nullptr,
            /* keyAntiquantOffset    */ nullptr,
            /* valueAntiquantScale   */ nullptr,
            /* valueAntiquantOffset  */ nullptr,
            /* keySharedPrefix       */ nullptr,
            /* valueSharedPrefix     */ nullptr,
            /* actualSharedPrefixLen */ nullptr,
            /* queryRope             */ nullptr,
            /* keyRope               */ nullptr,
            /* keyRopeAntiquantScale */ nullptr,
            /* dequantScaleQuery     */ nullptr,
            /* learnableSink         */ nullptr,
            /* qStartIdx             */ nullptr,  // V5-new
            /* kvStartIdx            */ nullptr,  // V5-new
            /* numHeads              */ N,
            /* scaleValue            */ scaleValue,
            /* preTokens             */ preTokens,
            /* nextTokens            */ nextTokens,
            /* inputLayout           */ layout,
            /* numKeyValueHeads      */ Nkv,
            /* sparseMode            */ sparseMode,
            /* innerPrecise          */ innerPrecise,
            /* blockSize             */ 0,
            /* antiquantMode         */ 0,
            /* softmaxLseFlag        */ false,
            /* keyAntiquantMode      */ 0,
            /* valueAntiquantMode    */ 0,
            /* queryQuantMode        */ 0,
            /* pseType               */ pseType,  // V5-new
            /* attentionOut          */ oT,
            /* softmaxLse            */ nullptr,
            wsSize, executor));
    };

    aclOpExecutor *exec0 = nullptr;
    uint64_t wsSize = 0;
    buildOnce(&exec0, &wsSize);
    void *wsAddr = nullptr;
    if (wsSize > 0) {
        CHECK_ACL(aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Warmup: each Execute consumes its executor, so re-tile per iter.
    for (int i = 0; i < warmup; ++i) {
        aclOpExecutor *exec = (i == 0) ? exec0 : nullptr;
        uint64_t ws = wsSize;
        if (i != 0) buildOnce(&exec, &ws);
        CHECK_RET(aclnnFusedInferAttentionScoreV5(wsAddr, ws, exec, stream));
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<double> us(iters);
    for (int i = 0; i < iters; ++i) {
        aclOpExecutor *exec = nullptr;
        uint64_t ws = wsSize;
        buildOnce(&exec, &ws);

        aclrtEvent start = nullptr, stop = nullptr;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&stop));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        CHECK_RET(aclnnFusedInferAttentionScoreV5(wsAddr, ws, exec, stream));
        CHECK_ACL(aclrtRecordEvent(stop, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        float ms = 0;
        CHECK_ACL(aclrtEventElapsedTime(&ms, start, stop));
        us[i] = ms * 1000.0;
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(stop));
    }

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensorList(kList);
    aclDestroyTensorList(vList);
    aclDestroyTensor(qT);
    aclDestroyTensor(oT);
    if (maskT) aclDestroyTensor(maskT);
    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(vDev);
    aclrtFree(oDev);
    if (maskDev) aclrtFree(maskDev);

    std::sort(us.begin(), us.end());
    double sum = 0;
    for (double t : us) sum += t;
    return {us[iters / 2], us.front(), us[iters / 10], sum / iters};
}

void RunCase(const char *tag, int64_t B, int64_t N, int64_t Nkv, int64_t Sq, int64_t Skv, int64_t D,
             int64_t sparseMode, aclrtStream stream)
{
    auto r = BenchFias(B, N, Nkv, Sq, Skv, D, sparseMode, /*warmup=*/10, /*iters=*/100, stream);
    // QK^T + attn*V flops: 4 * B * N * Sq * Skv * D (causal halves it but
    // we report dense for an upper-bound TFLOPs view).
    double flops = 4.0 * B * N * Sq * Skv * D;
    double tflops = flops / (r.median_us * 1e-6) / 1e12;
    // KV bytes touched per pass: 2 (K,V) * B * Nkv * Skv * D * 2 (fp16)
    double kvBytes = 4.0 * B * Nkv * Skv * D;
    double kvGBs = kvBytes / (r.median_us * 1e-6) / 1e9;
    printf("%-18s B=%ld N=%ld/%ld Sq=%-5ld Skv=%-5ld D=%ld sparse=%ld  "
           "median=%7.1f us  min=%7.1f  p10=%7.1f  avg=%7.1f  | %.2f TFLOPs  KV=%.1f GB/s\n",
           tag, B, N, Nkv, Sq, Skv, D, sparseMode, r.median_us, r.min_us, r.p10_us, r.avg_us, tflops, kvGBs);
}

} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    printf("=== FIAS V5 vanilla kernel perf (fp16, BNSD, ascend910b) ===\n");
    printf("warmup=10 iters=100, timing only the 2nd-stage Execute via aclrtEvent\n\n");

    // --- Decode (Sq=1) — bandwidth-bound on KV ---
    RunCase("decode/small",  /*B*/1, /*N*/8,  /*Nkv*/8,  /*Sq*/1, /*Skv*/1024, /*D*/128, /*sparse*/0, stream);
    RunCase("decode/medium", 1, 32, 32, 1, 4096, 128, 0, stream);
    RunCase("decode/large",  1, 32, 32, 1, 8192, 128, 0, stream);
    RunCase("decode/gqa",    1, 32, 8,  1, 4096, 128, 0, stream); // GQA 4:1
    RunCase("decode/batch4", 4, 32, 32, 1, 4096, 128, 0, stream);

    // --- Prefill (Sq=Skv) — compute-bound, causal mask via sparseMode=2 ---
    RunCase("prefill/1k",    1, 32, 32, 1024, 1024, 128, 2, stream);
    RunCase("prefill/2k",    1, 32, 32, 2048, 2048, 128, 2, stream);
    RunCase("prefill/4k",    1, 32, 32, 4096, 4096, 128, 2, stream);

    printf("\n=== done ===\n");

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
