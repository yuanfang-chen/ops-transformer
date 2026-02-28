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
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
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
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
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
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
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
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
