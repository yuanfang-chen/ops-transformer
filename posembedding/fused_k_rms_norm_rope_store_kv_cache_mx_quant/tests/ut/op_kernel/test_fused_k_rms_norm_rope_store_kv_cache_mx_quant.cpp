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
constexpr int64_t T = 2048;
constexpr int64_t NQ = 20;
constexpr int64_t NK = 2;
constexpr int64_t NV = 2;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 512;
constexpr int64_t BLOCK_NUM = 4;
constexpr int64_t TOTAL_HEAD = NQ + NK + NV;
constexpr int64_t Q_SCALE_LAST_DIM = HEAD_DIM / 32 / 2;
constexpr int64_t V_SCALE_SEQ = BLOCK_SIZE / 32 / 2;
constexpr int64_t V_SCALE_SLOT_NUM = T / 32 / 2;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

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

TEST_F(fused_k_rms_norm_rope_store_kv_cache_mx_quant_test, smoke_bf16_20_2_2)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t qkvSize = T * TOTAL_HEAD * HEAD_DIM * sizeof(bfloat16_t);
    size_t cosSize = T * HEAD_DIM * sizeof(bfloat16_t);
    size_t sinSize = T * HEAD_DIM * sizeof(bfloat16_t);
    size_t gammaSize = HEAD_DIM * sizeof(float);
    size_t kvSlotMappingSize = T * sizeof(int64_t);
    size_t vScaleSlotMappingSize = V_SCALE_SLOT_NUM * sizeof(int64_t);
    size_t kCacheSize = BLOCK_NUM * NK * BLOCK_SIZE * HEAD_DIM * sizeof(uint8_t);
    size_t kScaleCacheSize = BLOCK_NUM * NK * BLOCK_SIZE * Q_SCALE_LAST_DIM * 2 * sizeof(uint8_t);
    size_t vCacheSize = BLOCK_NUM * NV * BLOCK_SIZE * HEAD_DIM * sizeof(uint8_t);
    size_t vScaleCacheSize = BLOCK_NUM * NV * V_SCALE_SEQ * HEAD_DIM * 2 * sizeof(uint8_t);
    size_t qSize = T * NQ * HEAD_DIM * sizeof(uint8_t);
    size_t qScaleSize = T * NQ * Q_SCALE_LAST_DIM * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(FusedKRmsNormRopeStoreKvCacheMxQuantTilingData);
    uint32_t blockDim = 64;

    uint8_t *qkv = (uint8_t *)AscendC::GmAlloc(qkvSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(cosSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(sinSize);
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

    auto *tilingData = reinterpret_cast<FusedKRmsNormRopeStoreKvCacheMxQuantTilingData *>(tiling);
    tilingData->seqLengthSum = T;
    tilingData->qkvNumHead = TOTAL_HEAD;
    tilingData->qNumHead = NQ;
    tilingData->kNumHead = NK;
    tilingData->vNumHead = NV;
    tilingData->headDim = HEAD_DIM;
    tilingData->blockNum = BLOCK_NUM;
    tilingData->blockSize = BLOCK_SIZE;
    tilingData->qUsedCoreNum = 64;
    tilingData->qBlockFactor = 32;
    tilingData->qUbFactor = 38;
    tilingData->kUsedCoreNum = 64;
    tilingData->kBlockFactor = 32;
    tilingData->kUbFactor = 123;
    tilingData->vUsedCoreNum = 16;
    tilingData->vBlockFactor = 128;
    tilingData->vTUbFactor = 64;
    tilingData->vNumHeadUbFactor = 379;
    tilingData->epsilon = 1e-5f;
    tilingData->reciprocal = 1.0f / static_cast<float>(HEAD_DIM);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(fused_k_rms_norm_rope_store_kv_cache_mx_quant,
                blockDim,
                qkv,
                cos,
                sin,
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
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
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
