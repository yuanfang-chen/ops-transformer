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


TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf16)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (batch=2, seq_len=6, dim=16384)
            {{{256, 6, 16384}, {256, 6, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=16384)
            {{{3, 16384}, {3, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=256, cache_len=4+3-2=5, dim=16384)
            // cache_len = kernel_size + seq_len - 2 = 3 + 6 - 2 = 7
            {{{256, 7, 16384}, {256, 7, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=3)
            {{{257}, {257}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (batch=256, seq_len=3, dim=16384)
            {{{256, 6, 16384}, {256, 6, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=256, cache_len=7, dim=16384)
            {{{256, 7, 16384}, {2, 7, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 30000;
    std::string expectTilingData = "64 64 1 256 256 256 256 0 255 166 128 2 2 256 6 0 16384 3 7 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_fp16_batch4_seq1_dim512)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (batch=4, seq_len=1, dim=512)
            {{{4, 1, 512}, {4, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=512)
            {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=4, cache_len=3+1-2=2, dim=512)
            // cache_len = kernel_size + seq_len - 2 = 3 + 1 - 2 = 2
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=5)
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=512)
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (batch=4, seq_len=1, dim=512)
            {{{4, 1, 512}, {4, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=4, cache_len=2, dim=512)
            {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 30000;
    std::string expectTilingData = "1 1 1 512 512 4 4 0 3 4 512 1 1 4 1 0 512 3 2 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_fp16_batch1_seq4_dim1024)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {
        64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (batch=1, seq_len=4, dim=1024)
            {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=3, dim=1024)
            {{{3, 1024}, {3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=1, cache_len=3+4-2=5, dim=1024)
            // cache_len = kernel_size + seq_len - 2 = 3 + 4 - 2 = 5
            {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=1024)
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (batch=1, seq_len=4, dim=1024)
            {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=1, cache_len=5, dim=1024)
            {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 30000;
    std::string expectTilingData = "1 1 1 1024 1024 1 1 0 0 1 1024 1 1 1 4 0 1024 3 5 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


