/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_aclnn_chunk_gated_delta_rule_recurrence.cpp
 *
 * Precision verification for ChunkGatedDeltaRuleRecurrence.
 * Runs a CPU golden reference and compares with NPU output via aclnn.
 *
 * Compile example (adjust paths as needed):
 *   g++ -std=c++17 -I${ASCEND_HOME_PATH}/include \
 *       -L${ASCEND_HOME_PATH}/lib64 \
 *       -lascendcl -laclnn_chunk_gated_delta_rule_recurrence \
 *       test_aclnn_chunk_gated_delta_rule_recurrence.cpp -o run_test
 *   ./run_test
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>

#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule_recurrence.h"

// ─── Helper macros ──────────────────────────────────────────────────────────
#define CHECK_ACL(ret, msg)                                                    \
    do {                                                                       \
        if ((ret) != ACL_SUCCESS) {                                            \
            printf("[ERROR] %s  ret=%d  line=%d\n", (msg), (int)(ret), __LINE__); \
            return (int)(ret);                                                 \
        }                                                                      \
    } while (0)

// ─── Test configuration ──────────────────────────────────────────────────────
struct TestConfig {
    int b         = 2;   // batch size
    int hv        = 4;   // head count
    int dk        = 64;  // key dimension
    int dv        = 64;  // value dimension
    int chunkSize = 16;  // cs
    // cu_seqlens = [0, 2*cs, (2+4)*cs]  → batch0: 2 chunks, batch1: 4 chunks
    std::vector<int32_t> cuSeqlens;
    int nChunks = 6;     // total across all batches
};

// ─── Device memory helpers ────────────────────────────────────────────────────
template <typename T>
static int UploadTensor(const std::vector<T> &host,
                         const std::vector<int64_t> &shape,
                         aclDataType dtype,
                         void **devPtr, aclTensor **tensor)
{
    size_t bytes = host.size() * sizeof(T);
    auto ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL(ret, "aclrtMalloc");
    ret = aclrtMemcpy(*devPtr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_ACL(ret, "aclrtMemcpy H2D");
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype,
                              nullptr, 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *devPtr);
    if (*tensor == nullptr) { printf("[ERROR] aclCreateTensor failed\n"); return -1; }
    return ACL_SUCCESS;
}

template <typename T>
static int DownloadTensor(void *devPtr, std::vector<T> &host)
{
    size_t bytes = host.size() * sizeof(T);
    return aclrtMemcpy(host.data(), bytes, devPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

// ─── CPU golden reference ─────────────────────────────────────────────────────
// Shapes:
//   state [b, hv, dv, dk]  (in-out)
//   kgexp/k_cumdecay/qgexp [hv, nChunks, cs, dk]
//   value [hv, nChunks, cs, dv]
//   gexp  [hv, nChunks, cs, 1]
//   cu_seqlens [b+1]  (in tokens; cs tokens per chunk)
//   attn_inter_out [hv, nChunks, cs, dv]
//   v_new_out      [hv, nChunks, cs, dv]
static void CpuGolden(const TestConfig &cfg,
                       const std::vector<float> &kgexp,
                       const std::vector<float> &value,
                       const std::vector<float> &kCumdecay,
                       const std::vector<float> &qgexp,
                       const std::vector<float> &gexp,
                       std::vector<float>       &state,     // in-out
                       std::vector<float>       &attnInter, // out
                       std::vector<float>       &vNew)      // out
{
    int b = cfg.b, hv = cfg.hv, dk = cfg.dk, dv = cfg.dv;
    int cs = cfg.chunkSize, nC = cfg.nChunks;

    // Accessor lambdas
    auto idxChunk = [&](int h, int ci, int s, int d, int D) {
        return ((h * nC + ci) * cs + s) * D + d;
    };
    auto idxState = [&](int bi, int h, int r, int c) {
        return ((bi * hv + h) * dv + r) * dk + c;
    };
    auto idxGexp = [&](int h, int ci, int s) {
        return ((h * nC + ci) * cs + s) * 1;
    };

    for (int bi = 0; bi < b; bi++) {
        int chunkBegin = cfg.cuSeqlens[bi]     / cs;
        int chunkEnd   = cfg.cuSeqlens[bi + 1] / cs;

        for (int h = 0; h < hv; h++) {
            for (int ci = chunkBegin; ci < chunkEnd; ci++) {

                // C1: v_prime[cs, dv] = k_cumdecay[cs, dk] @ state[dv, dk].T
                std::vector<float> vPrime(cs * dv, 0.0f);
                for (int s = 0; s < cs; s++) {
                    for (int d = 0; d < dv; d++) {
                        float acc = 0.0f;
                        for (int k = 0; k < dk; k++) {
                            acc += kCumdecay[idxChunk(h, ci, s, k, dk)]
                                 * state[idxState(bi, h, d, k)];
                        }
                        vPrime[s * dv + d] = acc;
                    }
                }

                // C2: attn_inter[cs, dv] = qgexp[cs, dk] @ state[dv, dk].T
                for (int s = 0; s < cs; s++) {
                    for (int d = 0; d < dv; d++) {
                        float acc = 0.0f;
                        for (int k = 0; k < dk; k++) {
                            acc += qgexp[idxChunk(h, ci, s, k, dk)]
                                 * state[idxState(bi, h, d, k)];
                        }
                        attnInter[idxChunk(h, ci, s, d, dv)] = acc;
                    }
                }

                // V1: v_new = value - v_prime
                for (int s = 0; s < cs; s++) {
                    for (int d = 0; d < dv; d++) {
                        float vn = value[idxChunk(h, ci, s, d, dv)] - vPrime[s * dv + d];
                        vNew[idxChunk(h, ci, s, d, dv)] = vn;
                    }
                }

                // V0: state *= gexp[h, ci, cs-1, 0]
                float gScalar = gexp[idxGexp(h, ci, cs - 1)];
                for (int r = 0; r < dv; r++) {
                    for (int c = 0; c < dk; c++) {
                        state[idxState(bi, h, r, c)] *= gScalar;
                    }
                }

                // C3: state += v_new.T @ kgexp   (state[dv,dk] += vNew[cs,dv].T @ kgexp[cs,dk])
                for (int r = 0; r < dv; r++) {
                    for (int c = 0; c < dk; c++) {
                        float acc = 0.0f;
                        for (int s = 0; s < cs; s++) {
                            acc += vNew[idxChunk(h, ci, s, r, dv)]
                                 * kgexp[idxChunk(h, ci, s, c, dk)];
                        }
                        state[idxState(bi, h, r, c)] += acc;
                    }
                }

            } // chunkIdx
        }     // head
    }         // batch
}

// ─── allclose comparison ─────────────────────────────────────────────────────
static bool AllClose(const std::vector<float> &npu,
                     const std::vector<float> &golden,
                     float atol, float rtol, const char *name)
{
    if (npu.size() != golden.size()) {
        printf("[FAIL] %s: size mismatch %zu vs %zu\n", name, npu.size(), golden.size());
        return false;
    }
    float maxAbsErr = 0.f, maxRelErr = 0.f;
    int   failCnt   = 0;
    for (size_t i = 0; i < npu.size(); i++) {
        float absErr = std::fabs(npu[i] - golden[i]);
        float relErr = absErr / (std::fabs(golden[i]) + 1e-6f);
        if (absErr > maxAbsErr) maxAbsErr = absErr;
        if (relErr > maxRelErr) maxRelErr = relErr;
        if (absErr > atol + rtol * std::fabs(golden[i])) {
            if (failCnt < 5) {
                printf("  [%zu] npu=%.6f golden=%.6f absErr=%.2e relErr=%.2e\n",
                       i, npu[i], golden[i], absErr, relErr);
            }
            failCnt++;
        }
    }
    bool pass = (failCnt == 0);
    printf("[%s] %-22s  maxAbsErr=%.2e  maxRelErr=%.2e  failCnt=%d\n",
           pass ? "PASS" : "FAIL", name, maxAbsErr, maxRelErr, failCnt);
    return pass;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main()
{
    TestConfig cfg;
    cfg.cuSeqlens = {0, 2 * cfg.chunkSize, cfg.nChunks * cfg.chunkSize};

    // ── ACL init ──────────────────────────────────────────────────────────
    CHECK_ACL(aclInit(nullptr),                          "aclInit");
    CHECK_ACL(aclrtSetDevice(0),                         "aclrtSetDevice");
    aclrtContext ctx;
    CHECK_ACL(aclrtCreateContext(&ctx, 0),               "aclrtCreateContext");
    CHECK_ACL(aclrtSetCurrentContext(ctx),               "aclrtSetCurrentContext");
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream),                "aclrtCreateStream");

    // ── Generate random data ──────────────────────────────────────────────
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gDist(0.5f, 1.0f);

    int64_t stateN    = (int64_t)cfg.b   * cfg.hv * cfg.dv * cfg.dk;
    int64_t chunk4dDk = (int64_t)cfg.hv  * cfg.nChunks * cfg.chunkSize * cfg.dk;
    int64_t chunk4dDv = (int64_t)cfg.hv  * cfg.nChunks * cfg.chunkSize * cfg.dv;
    int64_t gexpN     = (int64_t)cfg.hv  * cfg.nChunks * cfg.chunkSize * 1;

    std::vector<float> stateHost(stateN),    kgexpHost(chunk4dDk), valueHost(chunk4dDv),
                       kCumdecayHost(chunk4dDk), qgexpHost(chunk4dDk), gexpHost(gexpN);

    for (auto &v : stateHost)      { v = dist(rng); }
    for (auto &v : kgexpHost)      { v = dist(rng); }
    for (auto &v : valueHost)      { v = dist(rng); }
    for (auto &v : kCumdecayHost)  { v = dist(rng); }
    for (auto &v : qgexpHost)      { v = dist(rng); }
    for (auto &v : gexpHost)       { v = gDist(rng); }

    // ── CPU golden ────────────────────────────────────────────────────────
    std::vector<float> goldenState    = stateHost;
    std::vector<float> goldenAttnInter(chunk4dDv, 0.f);
    std::vector<float> goldenVNew(chunk4dDv, 0.f);

    CpuGolden(cfg, kgexpHost, valueHost, kCumdecayHost, qgexpHost, gexpHost,
              goldenState, goldenAttnInter, goldenVNew);

    // ── Upload tensors ────────────────────────────────────────────────────
    std::vector<int64_t> shapeState    = {cfg.b,  cfg.hv, cfg.dv, cfg.dk};
    std::vector<int64_t> shapeChunkDk  = {cfg.hv, cfg.nChunks, cfg.chunkSize, cfg.dk};
    std::vector<int64_t> shapeChunkDv  = {cfg.hv, cfg.nChunks, cfg.chunkSize, cfg.dv};
    std::vector<int64_t> shapeGexp     = {cfg.hv, cfg.nChunks, cfg.chunkSize, 1};
    std::vector<int64_t> shapeCuSeq    = {cfg.b + 1};

    void *devState=nullptr, *devKgexp=nullptr, *devValue=nullptr,
         *devKcd=nullptr,   *devQgexp=nullptr, *devGexp=nullptr,
         *devCuSeq=nullptr, *devAttn=nullptr,  *devVNew=nullptr;

    aclTensor *tState=nullptr, *tKgexp=nullptr, *tValue=nullptr,
              *tKcd=nullptr,   *tQgexp=nullptr, *tGexp=nullptr,
              *tCuSeq=nullptr, *tAttn=nullptr,  *tVNew=nullptr;

    std::vector<float>   attnInit(chunk4dDv, 0.f), vNewInit(chunk4dDv, 0.f);

    CHECK_ACL(UploadTensor(stateHost,      shapeState,   ACL_FLOAT, &devState, &tState),   "upload state");
    CHECK_ACL(UploadTensor(kgexpHost,      shapeChunkDk, ACL_FLOAT, &devKgexp, &tKgexp),   "upload kgexp");
    CHECK_ACL(UploadTensor(valueHost,      shapeChunkDv, ACL_FLOAT, &devValue, &tValue),   "upload value");
    CHECK_ACL(UploadTensor(kCumdecayHost,  shapeChunkDk, ACL_FLOAT, &devKcd,   &tKcd),     "upload kCumdecay");
    CHECK_ACL(UploadTensor(qgexpHost,      shapeChunkDk, ACL_FLOAT, &devQgexp, &tQgexp),   "upload qgexp");
    CHECK_ACL(UploadTensor(gexpHost,       shapeGexp,    ACL_FLOAT, &devGexp,  &tGexp),    "upload gexp");
    CHECK_ACL(UploadTensor(cfg.cuSeqlens,  shapeCuSeq,   ACL_INT32, &devCuSeq, &tCuSeq),   "upload cuSeqlens");
    CHECK_ACL(UploadTensor(attnInit,       shapeChunkDv, ACL_FLOAT, &devAttn,  &tAttn),    "upload attnOut");
    CHECK_ACL(UploadTensor(vNewInit,       shapeChunkDv, ACL_FLOAT, &devVNew,  &tVNew),    "upload vNewOut");

    // ── Run operator ──────────────────────────────────────────────────────
    uint64_t    wsSize   = 0;
    aclOpExecutor *exec  = nullptr;
    float scaleValue = 1.0f;

    auto ret = aclnnChunkGatedDeltaRuleRecurrenceGetWorkspaceSize(
        tState, tKgexp, tValue, tKcd, tQgexp, tGexp, tCuSeq,
        scaleValue, tAttn, tVNew, &wsSize, &exec);
    CHECK_ACL(ret, "GetWorkspaceSize");

    void *wsPtr = nullptr;
    if (wsSize > 0) {
        CHECK_ACL(aclrtMalloc(&wsPtr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
    }

    CHECK_ACL(aclnnChunkGatedDeltaRuleRecurrence(wsPtr, wsSize, exec, stream), "Execute");
    CHECK_ACL(aclrtSynchronizeStream(stream), "SynchronizeStream");

    // ── Download results ──────────────────────────────────────────────────
    std::vector<float> npuState(stateN), npuAttn(chunk4dDv), npuVNew(chunk4dDv);
    CHECK_ACL(DownloadTensor(devState, npuState), "download state");
    CHECK_ACL(DownloadTensor(devAttn,  npuAttn),  "download attn_inter");
    CHECK_ACL(DownloadTensor(devVNew,  npuVNew),  "download v_new");

    // ── Precision comparison ──────────────────────────────────────────────
    constexpr float ATOL = 1e-4f;
    constexpr float RTOL = 1e-3f;
    bool p1 = AllClose(npuAttn,  goldenAttnInter, ATOL, RTOL, "attn_inter_out");
    bool p2 = AllClose(npuVNew,  goldenVNew,      ATOL, RTOL, "v_new_out");
    bool p3 = AllClose(npuState, goldenState,     ATOL, RTOL, "initial_state");
    bool pass = p1 && p2 && p3;
    printf("\n=== 验证结论：%s ===\n", pass ? "PASS ✓" : "FAIL ✗");

    // ── Cleanup ───────────────────────────────────────────────────────────
    aclDestroyTensor(tState);  aclDestroyTensor(tKgexp); aclDestroyTensor(tValue);
    aclDestroyTensor(tKcd);    aclDestroyTensor(tQgexp); aclDestroyTensor(tGexp);
    aclDestroyTensor(tCuSeq);  aclDestroyTensor(tAttn);  aclDestroyTensor(tVNew);

    aclrtFree(devState);  aclrtFree(devKgexp); aclrtFree(devValue);
    aclrtFree(devKcd);    aclrtFree(devQgexp); aclrtFree(devGexp);
    aclrtFree(devCuSeq);  aclrtFree(devAttn);  aclrtFree(devVNew);
    if (wsSize > 0) aclrtFree(wsPtr);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();

    return pass ? 0 : 1;
}
