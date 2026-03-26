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
#include <gtest/gtest.h>
#include "../../../op_host/fused_k_rms_norm_rope_store_kv_cache_mx_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t QKV_DTYPE_SCALE_GRANULARITY = 32;
constexpr int64_t MX_SCALE_PACK = 2;
constexpr float EPSILON = 1e-5f;

class FusedKRmsNormRopeStoreKvCacheMxQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedKRmsNormRopeStoreKvCacheMxQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedKRmsNormRopeStoreKvCacheMxQuantTiling TearDown" << std::endl;
    }
};

gert::TilingContextPara BuildTilingContext(int64_t seqLengthSum,
                                           int64_t numHeadQ,
                                           int64_t numHeadK,
                                           int64_t numHeadV,
                                           int64_t blockSize)
{
    const int64_t totalNumHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (seqLengthSum + blockSize - 1) / blockSize;
    const int64_t qScaleDim = HEAD_DIM / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK;
    const int64_t vScaleSeq = std::max<int64_t>(1, blockSize / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK);
    const int64_t vScaleSlotNum = std::max<int64_t>(1, seqLengthSum / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK);

    static optiling::FusedKRmsNormRopeStoreKvCacheMxQuantCompileInfo compileInfo = {};

    return gert::TilingContextPara(
        "FusedKRmsNormRopeStoreKvCacheMxQuant",
        {
            {{{seqLengthSum, totalNumHead, HEAD_DIM}, {seqLengthSum, totalNumHead, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{seqLengthSum, 1, HEAD_DIM}, {seqLengthSum, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{seqLengthSum, 1, HEAD_DIM}, {seqLengthSum, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{HEAD_DIM}, {HEAD_DIM}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{seqLengthSum}, {seqLengthSum}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{vScaleSlotNum}, {vScaleSlotNum}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, HEAD_DIM}, {blockNum, numHeadK, blockSize, HEAD_DIM}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}, {blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{blockNum, numHeadV, blockSize, HEAD_DIM}, {blockNum, numHeadV, blockSize, HEAD_DIM}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadV, vScaleSeq, HEAD_DIM, MX_SCALE_PACK}, {blockNum, numHeadV, vScaleSeq, HEAD_DIM, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {{{seqLengthSum, numHeadQ, HEAD_DIM}, {seqLengthSum, numHeadQ, HEAD_DIM}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{seqLengthSum, numHeadQ, qScaleDim, MX_SCALE_PACK}, {seqLengthSum, numHeadQ, qScaleDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, HEAD_DIM}, {blockNum, numHeadK, blockSize, HEAD_DIM}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}, {blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{blockNum, numHeadV, blockSize, HEAD_DIM}, {blockNum, numHeadV, blockSize, HEAD_DIM}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadV, vScaleSeq, HEAD_DIM, MX_SCALE_PACK}, {blockNum, numHeadV, vScaleSeq, HEAD_DIM, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(EPSILON)},
        },
        &compileInfo,
        "Ascend950");
}
// Flexible builder that allows customizing headDim and qkv dtype
gert::TilingContextPara BuildTilingContextCustom(int64_t seqLengthSum,
                                                  int64_t numHeadQ,
                                                  int64_t numHeadK,
                                                  int64_t numHeadV,
                                                  int64_t blockSize,
                                                  int64_t headDim,
                                                  ge::DataType qkvDtype)
{
    const int64_t totalNumHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (seqLengthSum + blockSize - 1) / blockSize;
    const int64_t qScaleDim = headDim / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK;
    const int64_t vScaleSeq = std::max<int64_t>(1, blockSize / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK);
    const int64_t vScaleSlotNum = std::max<int64_t>(1, seqLengthSum / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK);

    static optiling::FusedKRmsNormRopeStoreKvCacheMxQuantCompileInfo compileInfoCustom = {};

    return gert::TilingContextPara(
        "FusedKRmsNormRopeStoreKvCacheMxQuant",
        {
            {{{seqLengthSum, totalNumHead, headDim}, {seqLengthSum, totalNumHead, headDim}}, qkvDtype, ge::FORMAT_ND},
            {{{seqLengthSum, 1, headDim}, {seqLengthSum, 1, headDim}}, qkvDtype, ge::FORMAT_ND},
            {{{seqLengthSum, 1, headDim}, {seqLengthSum, 1, headDim}}, qkvDtype, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{seqLengthSum}, {seqLengthSum}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{vScaleSlotNum}, {vScaleSlotNum}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, headDim}, {blockNum, numHeadK, blockSize, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}, {blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{blockNum, numHeadV, blockSize, headDim}, {blockNum, numHeadV, blockSize, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadV, vScaleSeq, headDim, MX_SCALE_PACK}, {blockNum, numHeadV, vScaleSeq, headDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {{{seqLengthSum, numHeadQ, headDim}, {seqLengthSum, numHeadQ, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{seqLengthSum, numHeadQ, qScaleDim, MX_SCALE_PACK}, {seqLengthSum, numHeadQ, qScaleDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, headDim}, {blockNum, numHeadK, blockSize, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}, {blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{blockNum, numHeadV, blockSize, headDim}, {blockNum, numHeadV, blockSize, headDim}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadV, vScaleSeq, headDim, MX_SCALE_PACK}, {blockNum, numHeadV, vScaleSeq, headDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(EPSILON)},
        },
        &compileInfoCustom,
        "Ascend950");
}

// Builder that starts from valid context and overrides a specific input's shape
gert::TilingContextPara BuildTilingContextWithBadInput(
    int inputIndex,
    const gert::TilingContextPara::TensorDescription& badDesc)
{
    auto ctx = BuildTilingContext(2048, 20, 2, 2, 512);
    ctx.inputTensorDesc_[inputIndex] = badDesc;
    return ctx;
}

} // namespace

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_success_bf16_20_2_2)
{
    auto tilingContextPara = BuildTilingContext(2048, 20, 2, 2, 512);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
    EXPECT_GT(tilingInfo.tilingDataSize, 0U);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_t_not_aligned)
{
    auto tilingContextPara = BuildTilingContext(2000, 20, 2, 2, 512);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_head_config_unsupported)
{
    auto tilingContextPara = BuildTilingContext(2048, 16, 2, 2, 512);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_block_size_unsupported)
{
    auto tilingContextPara = BuildTilingContext(2048, 20, 2, 2, 256);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== New success path test ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_success_bf16_80_8_8)
{
    auto tilingContextPara = BuildTilingContext(2048, 80, 8, 8, 512);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
    EXPECT_GT(tilingInfo.tilingDataSize, 0U);
}

// ==================== headDim and dtype error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_headDim_unsupported)
{
    // headDim=64 is not supported (only 128)
    auto tilingContextPara = BuildTilingContextCustom(2048, 20, 2, 2, 512, 64, ge::DT_BF16);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_qkv_dtype_unsupported)
{
    // DT_FLOAT is not FP16 or BF16
    auto tilingContextPara = BuildTilingContextCustom(2048, 20, 2, 2, 512, 128, ge::DT_FLOAT);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckQkvValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_qkv_not_3d)
{
    // qkv as 2D instead of 3D
    auto tilingContextPara = BuildTilingContextWithBadInput(0,
        {{{2048, 3072}, {2048, 3072}}, ge::DT_BF16, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckCosSinValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_cos_not_3d)
{
    // cos as 2D instead of 3D
    auto tilingContextPara = BuildTilingContextWithBadInput(1,
        {{{2048, 128}, {2048, 128}}, ge::DT_BF16, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_cos_second_dim_not_1)
{
    // cos second dim = 2 instead of 1
    auto tilingContextPara = BuildTilingContextWithBadInput(1,
        {{{2048, 2, 128}, {2048, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_sin_not_3d)
{
    // sin as 2D instead of 3D
    auto tilingContextPara = BuildTilingContextWithBadInput(2,
        {{{2048, 128}, {2048, 128}}, ge::DT_BF16, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_sin_second_dim_not_1)
{
    // sin second dim = 2 instead of 1
    auto tilingContextPara = BuildTilingContextWithBadInput(2,
        {{{2048, 2, 128}, {2048, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckGammaValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_gamma_not_1d)
{
    // gamma as 2D instead of 1D
    auto tilingContextPara = BuildTilingContextWithBadInput(3,
        {{{1, 128}, {1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_gamma_dim_le_0)
{
    // gamma dimension = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(3,
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckKvSlotMappingValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_kv_slot_mapping_not_1d)
{
    // kv_slot_mapping as 2D
    auto tilingContextPara = BuildTilingContextWithBadInput(4,
        {{{1, 2048}, {1, 2048}}, ge::DT_INT64, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_kv_slot_mapping_dtype_wrong)
{
    // kv_slot_mapping dtype is INT32 instead of INT64
    auto tilingContextPara = BuildTilingContextWithBadInput(4,
        {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckVScaleSlotMappingValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_scale_slot_mapping_not_1d)
{
    // v_scale_slot_mapping as 2D
    auto tilingContextPara = BuildTilingContextWithBadInput(5,
        {{{1, 32}, {1, 32}}, ge::DT_INT64, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_scale_slot_mapping_dtype_wrong)
{
    // v_scale_slot_mapping dtype is INT32 instead of INT64
    auto tilingContextPara = BuildTilingContextWithBadInput(5,
        {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckKCacheValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_cache_not_4d)
{
    // k_cache as 3D instead of 4D
    auto tilingContextPara = BuildTilingContextWithBadInput(6,
        {{{4, 2, 65536}, {4, 2, 65536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_cache_dim0_le_0)
{
    // k_cache Bn = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(6,
        {{{0, 2, 512, 128}, {0, 2, 512, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_cache_dim2_le_0)
{
    // k_cache Bs = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(6,
        {{{4, 2, 0, 128}, {4, 2, 0, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_cache_dim3_le_0)
{
    // k_cache D = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(6,
        {{{4, 2, 512, 0}, {4, 2, 512, 0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckKScaleCacheValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_scale_cache_not_5d)
{
    // k_scale_cache as 4D instead of 5D
    auto tilingContextPara = BuildTilingContextWithBadInput(7,
        {{{4, 2, 512, 4}, {4, 2, 512, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_scale_cache_last_dim_not_2)
{
    // k_scale_cache last dim = 3 instead of 2
    auto tilingContextPara = BuildTilingContextWithBadInput(7,
        {{{4, 2, 512, 2, 3}, {4, 2, 512, 2, 3}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_scale_cache_dim2_le_0)
{
    // k_scale_cache Bs = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(7,
        {{{4, 2, 0, 2, 2}, {4, 2, 0, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_scale_cache_dim3_le_0)
{
    // k_scale_cache D/32/2 = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(7,
        {{{4, 2, 512, 0, 2}, {4, 2, 512, 0, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckVCacheValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_cache_not_4d)
{
    // v_cache as 3D instead of 4D
    auto tilingContextPara = BuildTilingContextWithBadInput(8,
        {{{4, 2, 65536}, {4, 2, 65536}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_cache_dim0_le_0)
{
    // v_cache Bn = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(8,
        {{{0, 2, 512, 128}, {0, 2, 512, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_cache_dim2_le_0)
{
    // v_cache Bs = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(8,
        {{{4, 2, 0, 128}, {4, 2, 0, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_cache_dim3_le_0)
{
    // v_cache D = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(8,
        {{{4, 2, 512, 0}, {4, 2, 512, 0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== CheckVScaleCacheValid error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_scale_cache_not_5d)
{
    // v_scale_cache as 4D instead of 5D
    auto tilingContextPara = BuildTilingContextWithBadInput(9,
        {{{4, 2, 8, 128}, {4, 2, 8, 128}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_scale_cache_last_dim_not_2)
{
    // v_scale_cache last dim = 3 instead of 2
    auto tilingContextPara = BuildTilingContextWithBadInput(9,
        {{{4, 2, 8, 128, 3}, {4, 2, 8, 128, 3}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_scale_cache_dim2_le_0)
{
    // v_scale_cache Bs/32/2 = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(9,
        {{{4, 2, 0, 128, 2}, {4, 2, 0, 128, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_scale_cache_dim3_le_0)
{
    // v_scale_cache D = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(9,
        {{{4, 2, 8, 0, 2}, {4, 2, 8, 0, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== numHeadQ <= 0 error path ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_numHeadQ_le_0)
{
    // numHead = numHeadK + numHeadV => numHeadQ = 0
    // Set qkv N dim = 4 (= numHeadK + numHeadV), so numHeadQ = 4 - 2 - 2 = 0
    auto tilingContextPara = BuildTilingContext(2048, 20, 2, 2, 512);
    // Override qkv shape so total numHead = numHeadK + numHeadV = 4
    tilingContextPara.inputTensorDesc_[0] =
        {{{2048, 4, 128}, {2048, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== FP16 dtype success path ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_success_fp16_20_2_2)
{
    auto tilingContextPara = BuildTilingContextCustom(2048, 20, 2, 2, 512, 128, ge::DT_FLOAT16);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_GT(tilingInfo.tilingDataSize, 0U);
}

// ==================== Additional QKV T/N dimension error paths ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_qkv_t_le_0)
{
    // qkv T dimension = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(0,
        {{{0, 24, 128}, {0, 24, 128}}, ge::DT_BF16, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_qkv_n_le_0)
{
    // qkv N dimension = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(0,
        {{{2048, 0, 128}, {2048, 0, 128}}, ge::DT_BF16, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== Additional k_scale_cache Bn/Nk <= 0 ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_scale_cache_dim0_le_0)
{
    // k_scale_cache Bn = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(7,
        {{{0, 2, 512, 2, 2}, {0, 2, 512, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== Additional v_scale_cache Bn/Nv <= 0 ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_scale_cache_dim0_le_0)
{
    // v_scale_cache Bn = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(9,
        {{{0, 2, 8, 128, 2}, {0, 2, 8, 128, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== Additional k_cache Nk <= 0 ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_k_cache_dim1_le_0)
{
    // k_cache Nk = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(6,
        {{{4, 0, 512, 128}, {4, 0, 512, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== Additional v_cache Nv <= 0 ====================

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantTiling, tiling_fail_when_v_cache_dim1_le_0)
{
    // v_cache Nv = 0
    auto tilingContextPara = BuildTilingContextWithBadInput(8,
        {{{4, 0, 512, 128}, {4, 0, 512, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
