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
 * \file test_causal_conv1d_fn_tiling.cpp
 * \brief Unit tests for CausalConv1dFn tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/causal_conv1d_fn_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class CausalConv1dFnTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dFnTiling SetUp" << std::endl;
        // Force linker to include tiling implementation with registration
        (void)sizeof(optiling::CausalConv1dFnTiling);
    }

    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dFnTiling TearDown" << std::endl;
    }
};

// Test case 1: Basic FP16 test with small batch and sequence
// batch=2, each sequence length=128, dim=512, kernel_width=4
// cu_seq_len = 2 * 128 = 256
// cache_max_size=4 (> batch=2, allows some buffer)
TEST_F(CausalConv1dFnTiling, CausalConv1dFn_950_tiling_basic_fp16)
{
    optiling::CausalConv1dFnCompileInfo compileInfo = {
        64, 32, 196608, 524288, 131072, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dFn",
        {
            // Input 0: x - (cu_seq_len=256, dim=512)
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=4, cache_width=3, dim=512)
            // cache_max_size must be > batch, here 4 > 2
            {{{4, 3, 512}, {4, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: cacheIndices - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: queryStartLoc/seqStartIndex - (batch+1=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (cu_seq_len=256, dim=512)
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=4, cache_width=3, dim=512)
            {{{4, 3, 512}, {4, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo, "Ascend950", 64, 262144, 8192);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "4294967297 30064771079 2199023256064 7 0 6 4294967297 25769803783 2199023256064 17179869248 256 512 2 2 0 256 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 3: Edge case with single batch
// batch=1, sequence length=64, dim=256, kernel_width=4
// cu_seq_len = 64
// cache_max_size=2 (> batch=1)
TEST_F(CausalConv1dFnTiling, CausalConv1dFn_950_tiling_single_batch)
{
    optiling::CausalConv1dFnCompileInfo compileInfo = {
        64, 32, 196608, 524288, 131072, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dFn",
        {
            // Input 0: x - (cu_seq_len=64, dim=256)
            {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=256)
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=2, cache_width=3, dim=256)
            // cache_max_size must be > batch, here 2 > 1
            {{{2, 3, 256}, {2, 3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: cacheIndices - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: queryStartLoc/seqStartIndex - (batch+1=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (cu_seq_len=64, dim=256)
            {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=2, cache_width=3, dim=256)
            {{{2, 3, 256}, {2, 3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo, "Ascend950", 64, 262144, 8192);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "4294967297 17179869188 1099511628032 4 0 4 4294967297 17179869188 1099511628032 17179869245 64 256 1 1 0 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 4: Medium size test with FP16
// batch=8, each sequence length=512, dim=768, kernel_width=4
// cu_seq_len = 8 * 512 = 4096
// cache_max_size=16 (> batch=8)
TEST_F(CausalConv1dFnTiling, CausalConv1dFn_950_tiling_medium_fp16)
{
    optiling::CausalConv1dFnCompileInfo compileInfo = {
        64, 32, 196608, 524288, 131072, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dFn",
        {
            // Input 0: x - (cu_seq_len=4096, dim=768)
            {{{4096, 768}, {4096, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=768)
            {{{4, 768}, {4, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=16, cache_width=3, dim=768)
            // cache_max_size must be > batch, here 16 > 8
            {{{16, 3, 768}, {16, 3, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: cacheIndices - (batch=8)
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: queryStartLoc/seqStartIndex - (batch+1=9)
            {{{9}, {9}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=8)
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (cu_seq_len=4096, dim=768)
            {{{4096, 768}, {4096, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=16, cache_width=3, dim=768)
            {{{16, 3, 768}, {16, 3, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo, "Ascend950", 64, 262144, 8192);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "4294967297 287762808899 3298534884096 67 0 66 4294967297 283467841603 3298534884096 17179869248 4096 768 8 8 0 4096 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 5: Variable sequence lengths test
// batch=3 with variable sequence lengths (128, 256, 64), dim=512, kernel_width=4
// cu_seq_len = 128 + 256 + 64 = 448
// cache_max_size=6 (> batch=3)
TEST_F(CausalConv1dFnTiling, CausalConv1dFn_950_tiling_variable_seqlen)
{
    optiling::CausalConv1dFnCompileInfo compileInfo = {
        64, 32, 196608, 524288, 131072, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dFn",
        {
            // Input 0: x - (cu_seq_len=448, dim=512)
            {{{448, 512}, {448, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=6, cache_width=3, dim=512)
            // cache_max_size must be > batch, here 6 > 3
            {{{6, 3, 512}, {6, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: cacheIndices - (batch=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: queryStartLoc/seqStartIndex - (batch+1=4) = [0, 128, 384, 448]
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (cu_seq_len=448, dim=512)
            {{{448, 512}, {448, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=6, cache_width=3, dim=512)
            {{{6, 3, 512}, {6, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo, "Ascend950", 64, 262144, 8192);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "4294967297 42949672970 2199023256064 10 0 9 4294967297 38654705674 2199023256064 17179869248 448 512 3 3 0 448 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


// Test case: Large size test with BF16
// batch=4, cu_seq_len=32768, dim=512, kernel_width=4
// cache_max_size=256 (> batch=4)
TEST_F(CausalConv1dFnTiling, CausalConv1dFn_950_tiling_large_bf16)
{
    optiling::CausalConv1dFnCompileInfo compileInfo = {
        64, 32, 196608, 524288, 131072, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dFn",
        {
            // Input 0: x - (cu_seq_len=32768, dim=512)
            {{{32768, 512}, {32768, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=256, cache_width=3, dim=512)
            // cache_max_size must be > batch, here 256 > 4
            {{{256, 3, 512}, {256, 3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: cacheIndices - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: queryStartLoc/seqStartIndex - (batch+1=5)
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (cu_seq_len=32768, dim=512)
            {{{32768, 512}, {32768, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=256, cache_width=3, dim=512)
            {{{256, 3, 512}, {256, 3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo, "Ascend950", 64, 262144, 8192);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "17179869186 68719477238 549755814016 515 0 514 17179869186 64424509942 549755814016 17179869248 32768 512 4 4 0 32768 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case: Extra large size test with FP16 (dim split mode)
// batch=256, cu_seq_len=65536, dim=8192, kernel_width=3
// cache_max_size=1024 (> batch=256)
TEST_F(CausalConv1dFnTiling, CausalConv1dFn_950_tiling_xlarge_fp16)
{
    optiling::CausalConv1dFnCompileInfo compileInfo = {
        64, 32, 196608, 524288, 131072, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dFn",
        {
            // Input 0: x - (cu_seq_len=65536, dim=8192)
            {{{65536, 8192}, {65536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=8192)
            {{{3, 8192}, {3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=1024, cache_width=2, dim=8192)
            // cache_max_size must be > batch, here 1024 > 256
            {{{1024, 2, 8192}, {1024, 2, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: cacheIndices - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: queryStartLoc/seqStartIndex - (batch+1=257)
            {{{257}, {257}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (cu_seq_len=65536, dim=8192)
            {{{65536, 8192}, {65536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=1024, cache_width=3, dim=8192)
            {{{1024, 3, 8192}, {1024, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo, "Ascend950", 64, 262144, 8192);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "274877906947 103079215607 549755814016 1026 0 1025 274877906947 98784248311 549755814016 12884901952 65536 8192 256 256 0 65536 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}