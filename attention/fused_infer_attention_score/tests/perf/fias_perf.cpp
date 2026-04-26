/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

// FIAS V5 NPU kernel-time micro-benchmark.
//
// Times the second aclnn entry point (`aclnnFusedInferAttentionScoreV5`) on a
// single decode-shaped input with aclrtEvent-based per-iteration timing. The
// first entry point (`...GetWorkspaceSize`) is called once outside the loop —
// its cost is host-side tiling, not what we're measuring.
//
// Intended use: build before a refactor, capture median, build after, diff.
// Decode shape choices below mirror the case the IsPa / HasAttenMask runtime
// guards most affect. Adjust shapes in `BuildConfig()` as needed.
//
// Modeled on:
//   - experimental/mhc/mhc_pre/test/perf_test.cpp (timing loop pattern)
//   - attention/fused_infer_attention_score/examples/arch35/
//     test_aclnn_fused_infer_attention_score_v5.cpp (FIAS V5 call structure)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

namespace {

#define CHECK_ACL(expr)                                                      \
    do {                                                                     \
        auto _err = (expr);                                                  \
        if (_err != ACL_SUCCESS) {                                           \
            std::fprintf(stderr, "ACL error %d at %s:%d (%s)\n", _err,       \
                         __FILE__, __LINE__, #expr);                         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

struct Config {
    // Decode shape: one Q token, 4K KV context, GQA 32→8 heads, head_dim=128, fp16.
    // Layout BNSD because the FIAS V5 example in-tree uses BNSD; this is the
    // path the existing example known-good case exercises.
    int64_t batch = 1;
    int64_t seqlen_q = 1;
    int64_t seqlen_kv = 4096;
    int64_t num_q_heads = 32;
    int64_t num_kv_heads = 8;
    int64_t head_dim = 128;

    int warmup = 5;
    int iters = 100;
    const char *layout = "BNSD";
};

int64_t ShapeProd(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// Allocate a device tensor with uninitialized contents. Correctness is not the
// goal here — kernel runtime depends on shape + dtype + layout, not on values.
aclTensor *AllocTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                       void **deviceAddrOut) {
    int64_t numel = ShapeProd(shape);
    size_t elemSize = 0;
    switch (dtype) {
        case ACL_FLOAT16:  elemSize = 2; break;
        case ACL_BF16:     elemSize = 2; break;
        case ACL_FLOAT:    elemSize = 4; break;
        case ACL_BOOL:     elemSize = 1; break;
        default:
            std::fprintf(stderr, "unsupported dtype %d\n", dtype);
            std::exit(1);
    }
    size_t bytes = static_cast<size_t>(numel) * elemSize;
    void *devAddr = nullptr;
    CHECK_ACL(aclrtMalloc(&devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    aclTensor *t = aclCreateTensor(
        shape.data(), shape.size(), dtype,
        strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), devAddr);
    *deviceAddrOut = devAddr;
    return t;
}

}  // namespace

int main(int argc, char **argv) {
    Config cfg{};

    // 1. ACL init.
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // 2. Allocate Q / K / V / output. Layout: BNSD = [B, N, S, D].
    std::vector<int64_t> qShape = {cfg.batch, cfg.num_q_heads,  cfg.seqlen_q,  cfg.head_dim};
    std::vector<int64_t> kShape = {cfg.batch, cfg.num_kv_heads, cfg.seqlen_kv, cfg.head_dim};
    std::vector<int64_t> vShape = {cfg.batch, cfg.num_kv_heads, cfg.seqlen_kv, cfg.head_dim};
    std::vector<int64_t> oShape = {cfg.batch, cfg.num_q_heads,  cfg.seqlen_q,  cfg.head_dim};

    void *qDev = nullptr, *kDev = nullptr, *vDev = nullptr, *oDev = nullptr;
    aclTensor *qT = AllocTensor(qShape, ACL_FLOAT16, &qDev);
    aclTensor *kT = AllocTensor(kShape, ACL_FLOAT16, &kDev);
    aclTensor *vT = AllocTensor(vShape, ACL_FLOAT16, &vDev);
    aclTensor *oT = AllocTensor(oShape, ACL_FLOAT16, &oDev);

    // FIAS V5 takes K/V as TensorList for paged-KV layouts. For non-paged,
    // a single-element list is the canonical wrapping (matches the in-tree
    // example at examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp:147).
    aclTensor *kArr[1] = {kT};
    aclTensor *vArr[1] = {vT};
    auto kList = aclCreateTensorList(kArr, 1);
    auto vList = aclCreateTensorList(vArr, 1);

    // PSE: a per-head fp32 scalar tensor with pseType=2, mirroring the in-tree
    // example exactly. Removing PSE entirely (pseShift=nullptr, pseType=0)
    // would change the kernel's dispatch path, which we don't want as a
    // first-cut benchmark — that's a separate sweep.
    std::vector<int64_t> pseShape = {cfg.num_q_heads};
    void *pseDev = nullptr;
    aclTensor *pseT = AllocTensor(pseShape, ACL_FLOAT, &pseDev);

    // 3. Scalar params. Values mirror examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp:177-192.
    int64_t numHeads = cfg.num_q_heads;
    int64_t numKvHeads = cfg.num_kv_heads;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(cfg.head_dim));
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    std::string layoutStr(cfg.layout);
    std::vector<char> layoutBuf(layoutStr.begin(), layoutStr.end());
    layoutBuf.push_back('\0');
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int64_t blockSize = 0;
    int64_t antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    int64_t queryQuantMode = 0;
    int64_t pseType = 2;

    // 4. First aclnn call: tiling + workspace size. Run once, reuse executor.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
        qT, kList, vList, pseT,
        /*attenMask*/ nullptr, /*actSeqLen*/ nullptr, /*actSeqLenKv*/ nullptr,
        /*deqScale1*/ nullptr, /*quantScale1*/ nullptr,
        /*deqScale2*/ nullptr, /*quantScale2*/ nullptr, /*quantOffset2*/ nullptr,
        /*antiquantScale*/ nullptr, /*antiquantOffset*/ nullptr,
        /*blockTable*/ nullptr,
        /*queryPaddingSize*/ nullptr, /*kvPaddingSize*/ nullptr,
        /*keyAntiquantScale*/ nullptr, /*keyAntiquantOffset*/ nullptr,
        /*valueAntiquantScale*/ nullptr, /*valueAntiquantOffset*/ nullptr,
        /*keySharedPrefix*/ nullptr, /*valueSharedPrefix*/ nullptr,
        /*actSharedPrefixLen*/ nullptr,
        /*queryRope*/ nullptr, /*keyRope*/ nullptr,
        /*keyRopeAntiquantScale*/ nullptr,
        /*dequantScaleQuery*/ nullptr, /*learnableSink*/ nullptr,
        /*qStartIdx*/ nullptr, /*kvStartIdx*/ nullptr,
        numHeads, scaleValue, preTokens, nextTokens, layoutBuf.data(),
        numKvHeads, sparseMode, innerPrecise, blockSize, antiquantMode,
        softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryQuantMode,
        pseType, oT, /*softmaxLse*/ nullptr,
        &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed: %d\n", ret);
        return ret;
    }
    void *wsAddr = nullptr;
    if (workspaceSize > 0U) {
        CHECK_ACL(aclrtMalloc(&wsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 5. Warmup. Don't time these — let frequency / cache stabilize.
    for (int i = 0; i < cfg.warmup; ++i) {
        ret = aclnnFusedInferAttentionScoreV5(wsAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "aclnnFusedInferAttentionScoreV5 (warmup) failed: %d\n", ret);
            return ret;
        }
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 6. Timed iterations. aclrtEvent is stream-aware; aclrtEventElapsedTime
    //    returns ms with sub-microsecond resolution on Ascend.
    std::vector<double> times_us(cfg.iters);
    for (int i = 0; i < cfg.iters; ++i) {
        aclrtEvent ev_start = nullptr, ev_end = nullptr;
        CHECK_ACL(aclrtCreateEvent(&ev_start));
        CHECK_ACL(aclrtCreateEvent(&ev_end));
        CHECK_ACL(aclrtRecordEvent(ev_start, stream));
        ret = aclnnFusedInferAttentionScoreV5(wsAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "aclnnFusedInferAttentionScoreV5 (iter %d) failed: %d\n", i, ret);
            return ret;
        }
        CHECK_ACL(aclrtRecordEvent(ev_end, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        float ms = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&ms, ev_start, ev_end));
        times_us[i] = static_cast<double>(ms) * 1000.0;
        CHECK_ACL(aclrtDestroyEvent(ev_start));
        CHECK_ACL(aclrtDestroyEvent(ev_end));
    }

    // 7. Report median / min / p95 / max in microseconds.
    std::sort(times_us.begin(), times_us.end());
    double median = times_us[cfg.iters / 2];
    double minv = times_us.front();
    double maxv = times_us.back();
    double p95 = times_us[static_cast<size_t>(cfg.iters * 0.95)];
    std::printf("fias_perf decode  B=%ld Sq=%ld Skv=%ld Nq=%ld Nkv=%ld D=%ld %s fp16  "
                "median=%.2f us  min=%.2f  p95=%.2f  max=%.2f  iters=%d  warmup=%d\n",
                cfg.batch, cfg.seqlen_q, cfg.seqlen_kv,
                cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
                cfg.layout, median, minv, p95, maxv, cfg.iters, cfg.warmup);

    // 8. Cleanup. Mirror the in-tree example
    // (examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp:243-257):
    // destroy each constituent tensor explicitly and don't call
    // aclDestroyTensorList — the example doesn't, and a process-end leak
    // here is a non-issue (kernel is reaped on exit anyway).
    aclDestroyTensor(qT);
    aclDestroyTensor(kT);
    aclDestroyTensor(vT);
    aclDestroyTensor(oT);
    aclDestroyTensor(pseT);
    (void)kList;
    (void)vList;
    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(vDev);
    aclrtFree(oDev);
    aclrtFree(pseDev);
    if (workspaceSize > 0U) aclrtFree(wsAddr);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());

    (void)argc; (void)argv;
    return 0;
}
