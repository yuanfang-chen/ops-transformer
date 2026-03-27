/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void fused_k_rms_norm_rope_store_kv_cache_mx_quant(
    GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR gamma, GM_ADDR kv_slot_mapping, GM_ADDR v_scale_slot_mapping,
    GM_ADDR k_cache, GM_ADDR k_scale_cache, GM_ADDR v_cache, GM_ADDR v_scale_cache, GM_ADDR q, GM_ADDR q_scale,
    GM_ADDR k_cache_out, GM_ADDR k_scale_cache_out, GM_ADDR v_cache_out, GM_ADDR v_scale_cache_out, GM_ADDR workspace,
    GM_ADDR tiling);

namespace {
constexpr int64_t QUANT_BLOCK = 32;
constexpr int64_t U8_ALIGN = 32;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

struct TestConfig {
    int64_t seqLengthSum;
    int64_t nq;
    int64_t nk;
    int64_t nv;
    int64_t headDim;
    int64_t blockSize;
    int64_t blockNum;
    uint32_t blockDim;
    int64_t qUsedCoreNum;
    int64_t qBlockFactor;
    int64_t qUbFactor;
    int64_t kUsedCoreNum;
    int64_t kBlockFactor;
    int64_t kUbFactor;
    int64_t vUsedCoreNum;
    int64_t vBlockFactor;
    int64_t vTUbFactor;
    int64_t vNumHeadUbFactor;
    bool useNegativeIndices;
};

static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (y != 0) ? (x + y - 1) / y : 0;
}

static inline int64_t CeilAlign(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

void RunTestWithConfig(const TestConfig &cfg)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    int64_t totalHead = cfg.nq + cfg.nk + cfg.nv;
    int64_t qScaleLastDim = cfg.headDim / QUANT_BLOCK / 2;
    int64_t vScaleSeq = cfg.blockSize / QUANT_BLOCK / 2;
    int64_t vScaleSlotNum = cfg.seqLengthSum / QUANT_BLOCK / 2;

    size_t qkvSize = cfg.seqLengthSum * totalHead * cfg.headDim * sizeof(bfloat16_t);
    size_t cosSize = cfg.seqLengthSum * cfg.headDim * sizeof(bfloat16_t);
    size_t sinSize = cfg.seqLengthSum * cfg.headDim * sizeof(bfloat16_t);
    size_t gammaSize = cfg.headDim * sizeof(float);
    size_t kvSlotMappingSize = cfg.seqLengthSum * sizeof(int64_t);
    size_t vScaleSlotMappingSize = (vScaleSlotNum > 0 ? vScaleSlotNum : 1) * sizeof(int64_t);
    size_t kCacheSize = cfg.blockNum * cfg.nk * cfg.blockSize * cfg.headDim * sizeof(uint8_t);
    size_t kScaleCacheSize = cfg.blockNum * cfg.nk * cfg.blockSize * qScaleLastDim * 2 * sizeof(uint8_t);
    size_t vCacheSize = cfg.blockNum * cfg.nv * cfg.blockSize * cfg.headDim * sizeof(uint8_t);
    size_t vScaleCacheSize = cfg.blockNum * cfg.nv * vScaleSeq * cfg.headDim * 2 * sizeof(uint8_t);
    size_t qSize = cfg.seqLengthSum * cfg.nq * cfg.headDim * sizeof(uint8_t);
    size_t qScaleSize = cfg.seqLengthSum * cfg.nq * qScaleLastDim * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(FusedKRmsNormRopeStoreKvCacheMxQuantTilingData);

    uint8_t *qkv = (uint8_t *)AscendC::GmAlloc(qkvSize);
    uint8_t *cosB = (uint8_t *)AscendC::GmAlloc(cosSize);
    uint8_t *sinB = (uint8_t *)AscendC::GmAlloc(sinSize);
    uint8_t *gamma = (uint8_t *)AscendC::GmAlloc(gammaSize);
    uint8_t *kvSlotMapping = (uint8_t *)AscendC::GmAlloc(kvSlotMappingSize);
    uint8_t *vScaleSlotMapping = (uint8_t *)AscendC::GmAlloc(vScaleSlotMappingSize);
    uint8_t *kCache = (uint8_t *)AscendC::GmAlloc(kCacheSize);
    uint8_t *kScaleCache = (uint8_t *)AscendC::GmAlloc(kScaleCacheSize);
    uint8_t *vCache = (uint8_t *)AscendC::GmAlloc(vCacheSize);
    uint8_t *vScaleCache = (uint8_t *)AscendC::GmAlloc(vScaleCacheSize);
    uint8_t *q = (uint8_t *)AscendC::GmAlloc(qSize);
    uint8_t *qScale = (uint8_t *)AscendC::GmAlloc(qScaleSize);
    uint8_t *kCacheOut = (uint8_t *)AscendC::GmAlloc(kCacheSize);
    uint8_t *kScaleCacheOut = (uint8_t *)AscendC::GmAlloc(kScaleCacheSize);
    uint8_t *vCacheOut = (uint8_t *)AscendC::GmAlloc(vCacheSize);
    uint8_t *vScaleCacheOut = (uint8_t *)AscendC::GmAlloc(vScaleCacheSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // Initialize kvSlotMapping with valid slot indices (or -1 for negative index tests)
    auto *slotMap = reinterpret_cast<int64_t *>(kvSlotMapping);
    for (int64_t i = 0; i < cfg.seqLengthSum; ++i) {
        if (cfg.useNegativeIndices && (i % 3 == 0)) {
            slotMap[i] = -1; // negative index: skip scatter write
        } else {
            slotMap[i] = i % (cfg.blockNum * cfg.blockSize);
        }
    }

    // Initialize vScaleSlotMapping
    auto *vSlotMap = reinterpret_cast<int64_t *>(vScaleSlotMapping);
    for (int64_t i = 0; i < vScaleSlotNum; ++i) {
        if (cfg.useNegativeIndices && (i % 2 == 0)) {
            vSlotMap[i] = -1;
        } else {
            vSlotMap[i] = i % (cfg.blockNum * vScaleSeq);
        }
    }

    auto *tilingData = reinterpret_cast<FusedKRmsNormRopeStoreKvCacheMxQuantTilingData *>(tiling);
    tilingData->seqLengthSum = cfg.seqLengthSum;
    tilingData->qkvNumHead = totalHead;
    tilingData->qNumHead = cfg.nq;
    tilingData->kNumHead = cfg.nk;
    tilingData->vNumHead = cfg.nv;
    tilingData->headDim = cfg.headDim;
    tilingData->blockNum = cfg.blockNum;
    tilingData->blockSize = cfg.blockSize;
    tilingData->qUsedCoreNum = cfg.qUsedCoreNum;
    tilingData->qBlockFactor = cfg.qBlockFactor;
    tilingData->qUbFactor = cfg.qUbFactor;
    tilingData->kUsedCoreNum = cfg.kUsedCoreNum;
    tilingData->kBlockFactor = cfg.kBlockFactor;
    tilingData->kUbFactor = cfg.kUbFactor;
    tilingData->vUsedCoreNum = cfg.vUsedCoreNum;
    tilingData->vBlockFactor = cfg.vBlockFactor;
    tilingData->vTUbFactor = cfg.vTUbFactor;
    tilingData->vNumHeadUbFactor = cfg.vNumHeadUbFactor;
    tilingData->epsilon = 1e-5f;
    tilingData->reciprocal = 1.0f / static_cast<float>(cfg.headDim);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fused_k_rms_norm_rope_store_kv_cache_mx_quant,
                cfg.blockDim,
                qkv,
                cosB,
                sinB,
                gamma,
                kvSlotMapping,
                vScaleSlotMapping,
                kCache,
                kScaleCache,
                vCache,
                vScaleCache,
                q,
                qScale,
                kCacheOut,
                kScaleCacheOut,
                vCacheOut,
                vScaleCacheOut,
                workspace,
                tiling);

    AscendC::GmFree(qkv);
    AscendC::GmFree(cosB);
    AscendC::GmFree(sinB);
    AscendC::GmFree(gamma);
    AscendC::GmFree(kvSlotMapping);
    AscendC::GmFree(vScaleSlotMapping);
    AscendC::GmFree(kCache);
    AscendC::GmFree(kScaleCache);
    AscendC::GmFree(vCache);
    AscendC::GmFree(vScaleCache);
    AscendC::GmFree(q);
    AscendC::GmFree(qScale);
    AscendC::GmFree(kCacheOut);
    AscendC::GmFree(kScaleCacheOut);
    AscendC::GmFree(vCacheOut);
    AscendC::GmFree(vScaleCacheOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

class fused_k_rms_norm_rope_store_kv_cache_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "fused_k_rms_norm_rope_store_kv_cache_mx_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "fused_k_rms_norm_rope_store_kv_cache_mx_quant_test TearDown" << endl;
    }
};
} // namespace

// Original smoke test: blockDim=64, only Phase1 is meaningfully executed
TEST_F(fused_k_rms_norm_rope_store_kv_cache_mx_quant_test, smoke_bf16_20_2_2)
{
    TestConfig cfg;
    cfg.seqLengthSum = 2048;
    cfg.nq = 20;
    cfg.nk = 2;
    cfg.nv = 2;
    cfg.headDim = 128;
    cfg.blockSize = 512;
    cfg.blockNum = 4;
    cfg.blockDim = 64;
    cfg.qUsedCoreNum = 64;
    cfg.qBlockFactor = 32;
    cfg.qUbFactor = 38;
    cfg.kUsedCoreNum = 64;
    cfg.kBlockFactor = 32;
    cfg.kUbFactor = 123;
    cfg.vUsedCoreNum = 16;
    cfg.vBlockFactor = 128;
    cfg.vTUbFactor = 64;
    cfg.vNumHeadUbFactor = 379;
    cfg.useNegativeIndices = false;
    RunTestWithConfig(cfg);
}

// Single core test: blockDim=1, all three phases (Phase1+Phase2+Phase3) execute on core 0.
// T=128, NQ=2, NK=2, NV=2. No tail blocks.
TEST_F(fused_k_rms_norm_rope_store_kv_cache_mx_quant_test, single_core_all_phases)
{
    TestConfig cfg;
    cfg.seqLengthSum = 128;
    cfg.nq = 2;
    cfg.nk = 2;
    cfg.nv = 2;
    cfg.headDim = 128;
    cfg.blockSize = 512;
    cfg.blockNum = 1;
    cfg.blockDim = 1;
    cfg.qUsedCoreNum = 1;
    cfg.qBlockFactor = 128;
    cfg.qUbFactor = 64;
    cfg.kUsedCoreNum = 1;
    cfg.kBlockFactor = 128;
    cfg.kUbFactor = 64;
    cfg.vUsedCoreNum = 1;
    cfg.vBlockFactor = 128;
    cfg.vTUbFactor = 64;
    cfg.vNumHeadUbFactor = 2;
    cfg.useNegativeIndices = false;
    RunTestWithConfig(cfg);
}

// Tail block + negative index test: blockDim=1, all three phases execute.
// qBlockFactor=192, qUbFactor=38 -> Phase1 tail block (192 mod 38 = 2)
// kBlockFactor=192, kUbFactor=38 -> Phase2 tail block (192 mod 38 = 2)
// vBlockFactor=192, vTUbFactor=128 -> Phase3 tail block = 64
// vNumHeadUbFactor=1 with NV=2 -> Phase3 N-dimension tail (2 = 1 + 1)
// Some kvSlotMapping entries set to -1 to trigger skip logic in ScatterUpdateK/V
TEST_F(fused_k_rms_norm_rope_store_kv_cache_mx_quant_test, tail_block_and_neg_index)
{
    TestConfig cfg;
    cfg.seqLengthSum = 192;
    cfg.nq = 2;
    cfg.nk = 2;
    cfg.nv = 2;
    cfg.headDim = 128;
    cfg.blockSize = 512;
    cfg.blockNum = 1;
    cfg.blockDim = 1;
    cfg.qUsedCoreNum = 1;
    cfg.qBlockFactor = 192;
    cfg.qUbFactor = 38;
    cfg.kUsedCoreNum = 1;
    cfg.kBlockFactor = 192;
    cfg.kUbFactor = 38;
    cfg.vUsedCoreNum = 1;
    cfg.vBlockFactor = 192;
    cfg.vTUbFactor = 128;
    cfg.vNumHeadUbFactor = 1;
    cfg.useNegativeIndices = true;
    RunTestWithConfig(cfg);
}

// Multi-head test: NQ=4, NK=2, NV=4, blockDim=1.
// Covers more V heads, and vNumHeadUbFactor=3 with NV=4 triggers N tail (4 = 3 + 1).
TEST_F(fused_k_rms_norm_rope_store_kv_cache_mx_quant_test, multi_head_4_2_4)
{
    TestConfig cfg;
    cfg.seqLengthSum = 128;
    cfg.nq = 4;
    cfg.nk = 2;
    cfg.nv = 4;
    cfg.headDim = 128;
    cfg.blockSize = 512;
    cfg.blockNum = 1;
    cfg.blockDim = 1;
    cfg.qUsedCoreNum = 1;
    cfg.qBlockFactor = 128;
    cfg.qUbFactor = 32;
    cfg.kUsedCoreNum = 1;
    cfg.kBlockFactor = 128;
    cfg.kUbFactor = 32;
    cfg.vUsedCoreNum = 1;
    cfg.vBlockFactor = 128;
    cfg.vTUbFactor = 64;
    cfg.vNumHeadUbFactor = 3;
    cfg.useNegativeIndices = false;
    RunTestWithConfig(cfg);
}
