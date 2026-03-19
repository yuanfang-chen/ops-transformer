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
 * \file test_causal_conv1d_update_tiling.cpp
 * \brief Unit tests for CausalConv1dUpdate tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/causal_conv1d_update_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class CausalConv1dUpdateTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dUpdateTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dUpdateTiling TearDown" << std::endl;
    }
};


TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b4_s1_d512)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residualConnection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (batch=4, seq_len=1, dim=512)
            {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=4, cache_len=3+1-2=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - optional for 3D x
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=512)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (batch=4, seq_len=1, dim=512)
            {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=4, cache_len=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 20000;
    std::string expectTilingData = "16 4 4 4 0 128 128 4 0 1 1 0 3 1 1 1 1 128 128 1 1 1 1 128 128 4 1 0 512 3 2 0 0 0 1 0 ";

    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b1_s4_d1024)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residualConnection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (batch=1, seq_len=4, dim=1024)
            {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=1024)
            {{{3, 1024}, {3, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=1, cache_len=3+4-2=5, dim=1024)
            // cache_len = kernel_size + seq_len - 2 = 3 + 4 - 2 = 5
            {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=2)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=1024)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (batch=1, seq_len=4, dim=1024)
            {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=1, cache_len=5, dim=1024)
            {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 20000;
    std::string expectTilingData = "8 8 1 8 0 128 128 1 0 1 1 0 0 1 1 1 1 128 128 1 1 1 1 128 128 1 4 0 1024 3 5 0 0 0 1 0 ";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b1_s4_d512_x2d)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residualConnection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (cuSeqLen=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=1, cache_len=3+4-2=5, dim=512)
            // cache_len = kernel_size + seq_len - 2 = 3 + 4 - 2 = 5
            {{{1, 5, 512}, {1, 5, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=1)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=512)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (cuSeqLen=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=1, cache_len=5, dim=512)
            {{{1, 5, 512}, {1, 5, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 20000;
    std::string expectTilingData = "4 4 1 4 0 128 128 1 0 1 1 0 0 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 512 3 5 0 0 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Helper to build a 2D-Input TilingContextPara
static gert::TilingContextPara Make2DTilingPara(int64_t batch, int64_t cuSeqLen, int64_t dim)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"residualConnection", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
    };

    const int64_t k = 3;
    const int64_t state_len = 5; 

    return gert::TilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (cu_seq_len, dim)
            {{{cuSeqLen, dim}, {cuSeqLen, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (k, dim)
            {{{k, dim}, {k, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch, state_len, dim)
            {{{batch, state_len, dim}, {batch, state_len, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1)
            {{{batch + 1}, {batch + 1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch)
            {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch) [empty as existing case]
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - (batch)
            {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len, dim) for 2D path
            {{{cuSeqLen, dim}, {cuSeqLen, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch, state_len, dim)
            {{{batch, state_len, dim}, {batch, state_len, dim}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);
}

// batch=1, cuSeqLen=4
TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b1_s4_d768_x2d)
{
    auto para = Make2DTilingPara(1, 4, 768);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "6 6 1 6 0 128 128 1 0 1 1 0 0 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 768 3 5 0 0 1 1 1 ", {});
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b1_s4_d4096_x2d)
{
    auto para = Make2DTilingPara(1, 4, 4096);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "32 32 1 32 0 128 128 1 0 1 1 0 0 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 4096 3 5 0 0 1 1 1 ", {});
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b1_s4_d8192_x2d)
{
    auto para = Make2DTilingPara(1, 4, 8192);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 64 1 64 0 128 128 1 0 1 1 0 0 1 1 1 1 128 128 1 1 1 1 128 128 1 6 4 8192 3 5 0 0 1 1 1 ", {});
}

// batch=32, cuSeqLen=128
TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b32_s4_d512_x2d)
{
    auto para = Make2DTilingPara(32, 128, 512);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 4 16 4 0 128 128 16 0 2 2 0 31 1 1 2 2 128 128 1 1 2 2 128 128 32 6 128 512 3 5 0 0 1 1 1 ", {});
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b32_s4_d768_x2d)
{
    auto para = Make2DTilingPara(32, 128, 768);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 4 16 2 2 256 128 16 0 2 2 0 31 1 1 2 2 256 256 1 1 2 2 128 128 32 6 128 768 3 5 0 0 1 1 1 ", {});
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b32_s4_d4096_x2d)
{
    auto para = Make2DTilingPara(32, 128, 4096);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 32 2 32 0 128 128 2 0 16 16 0 31 1 1 16 16 128 128 1 1 16 16 128 128 32 6 128 4096 3 5 0 0 1 1 1 ", {});
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf_b32_s4_d8192_x2d)
{
    auto para = Make2DTilingPara(32, 128, 8192);
    int64_t expectTilingKey = 20000;
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, "64 64 1 64 0 128 128 1 0 32 32 0 31 1 1 32 32 128 128 1 1 32 32 128 128 32 6 128 8192 3 5 0 0 1 1 1 ", {});
}
