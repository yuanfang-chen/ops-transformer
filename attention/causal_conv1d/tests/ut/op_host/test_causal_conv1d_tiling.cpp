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
 * \file test_causal_conv1d_tiling.cpp
 * \brief Unit tests for CausalConv1dUpdate tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/causal_conv1d_update_tiling_arch35.h"
#include "../../../op_host/causal_conv1d_fn_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// class CausalConv1dUpdateTiling : public testing::Test {
// protected:
//     static void SetUpTestCase()
//     {
//         std::cout << "CausalConv1dUpdateTiling SetUp" << std::endl;
//     }

//     static void TearDownTestCase()
//     {
//         std::cout << "CausalConv1dUpdateTiling TearDown" << std::endl;
//     }
// };


// TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_bf16)
// {
//     optiling::CausalConv1dUpdateCompileInfo compileInfo = {
//         64, 261888};

//     std::vector<gert::TilingContextPara::OpAttr> attrs = {
//         {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//         {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
//         {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//     };

//     gert::TilingContextPara tilingContextPara(
//         "CausalConv1d",  // Changed from CausalConv1dUpdate to CausalConv1d
//         {
//             // Input 0: x - (batch=2, seq_len=6, dim=16384)
//             {{{256, 6, 16384}, {256, 6, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
//             // Input 1: weight - (kernel_size=3, dim=16384)
//             {{{3, 16384}, {3, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
//             // Input 2: convStates - (batch=256, cache_len=4+3-2=5, dim=16384)
//             // cache_len = kernel_size + seq_len - 2 = 3 + 6 - 2 = 7
//             {{{256, 7, 16384}, {256, 7, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
//             // Input 3: queryStartLoc - (batch+1=3)
//             {{{257}, {257}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 4: cacheIndices - (batch=256)
//             {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 5: hasInitialState - (batch=256)
//             {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
//             },
//         {
//             // Output 0: y - (batch=256, seq_len=3, dim=16384)
//             {{{256, 6, 16384}, {256, 6, 16384}}, ge::DT_BF16, ge::FORMAT_ND},
//             // Output 1: cacheStates - (batch=256, cache_len=7, dim=16384)
//             {{{256, 7, 16384}, {2, 7, 512}}, ge::DT_BF16, ge::FORMAT_ND},
//         },
//         attrs,
//         &compileInfo);

//     int64_t expectTilingKey = 20000;
//     std::string expectTilingData = "64 64 1 256 256 256 256 0 255 167 128 2 2 256 6 0 16384 3 7 0 0 0 0 ";
//     std::vector<size_t> expectWorkspaces = {};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

// TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_fp16_batch4_seq1_dim512)
// {
//     optiling::CausalConv1dUpdateCompileInfo compileInfo = {
//         64, 261888};

//     std::vector<gert::TilingContextPara::OpAttr> attrs = {
//         {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//         {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
//         {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//     };

//     gert::TilingContextPara tilingContextPara(
//         "CausalConv1d",  // Changed from CausalConv1dUpdate to CausalConv1d
//         {
//             // Input 0: x - (batch=4, seq_len=1, dim=512)
//             {{{4, 1, 512}, {4, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 1: weight - (kernel_size=3, dim=512)
//             {{{3, 512}, {3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 2: convStates - (batch=4, cache_len=3+1-2=2, dim=512)
//             // cache_len = kernel_size + seq_len - 2 = 3 + 1 - 2 = 2
//             {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 3: queryStartLoc - (batch+1=5)
//             {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 4: cacheIndices - (batch=4)
//             {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 5: hasInitialState - (batch=4)
//             {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 6: bias - optional, (dim=512)
//             {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 7: numAcceptedTokens - optional, (batch=4)
//             {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
//             },
//         {
//             // Output 0: y - (batch=4, seq_len=1, dim=512)
//             {{{4, 1, 512}, {4, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Output 1: cacheStates - (batch=4, cache_len=2, dim=512)
//             {{{4, 2, 512}, {4, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         attrs,
//         &compileInfo);

//     int64_t expectTilingKey = 20000;
//     std::string expectTilingData = "1 1 1 512 512 4 4 0 3 4 512 1 1 4 1 0 512 3 2 0 0 0 1 ";
//     std::vector<size_t> expectWorkspaces = {};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

// TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_fp16_batch1_seq4_dim1024)
// {
//     optiling::CausalConv1dUpdateCompileInfo compileInfo = {
//         64, 261888};

//     std::vector<gert::TilingContextPara::OpAttr> attrs = {
//         {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
//         {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
//         {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}
//     };

//     gert::TilingContextPara tilingContextPara(
//         "CausalConv1d",  // Changed from CausalConv1dUpdate to CausalConv1d
//         {
//             // Input 0: x - (batch=1, seq_len=4, dim=1024)
//             {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 1: weight - (kernel_size=3, dim=1024)
//             {{{3, 1024}, {3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 2: convStates - (batch=1, cache_len=3+4-2=5, dim=1024)
//             // cache_len = kernel_size + seq_len - 2 = 3 + 4 - 2 = 5
//             {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 3: queryStartLoc - (batch+1=2)
//             {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 4: cacheIndices - (batch=1)
//             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 5: hasInitialState - (batch=1)
//             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
//             // Input 6: bias - optional, (dim=1024)
//             {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Input 7: numAcceptedTokens - optional, (batch=1)
//             {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
//             },
//         {
//             // Output 0: y - (batch=1, seq_len=4, dim=1024)
//             {{{1, 4, 1024}, {1, 4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             // Output 1: cacheStates - (batch=1, cache_len=5, dim=1024)
//             {{{1, 5, 1024}, {1, 5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         attrs,
//         &compileInfo);

//     int64_t expectTilingKey = 20000;
//     std::string expectTilingData = "1 1 1 1024 1024 1 1 0 0 1 1024 1 1 1 4 0 1024 3 5 0 0 0 1 ";
//     std::vector<size_t> expectWorkspaces = {};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

// ============================================================
// CausalConv1d Fn Mode Tests (runMode=0)
// ============================================================

class CausalConv1dFnTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dFnTiling SetUp" << std::endl;
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
TEST_F(CausalConv1dFnTiling, CausalConv1d_950_tiling_basic_fp16)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            // Input 0: x - (cu_seq_len=256, dim=512)
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=4, cache_width=3, dim=512)
            {{{4, 3, 512}, {4, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=256, dim=512)
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=4, cache_width=3, dim=512)
            {{{4, 3, 512}, {4, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;  // FP16 类型
    std::string expectTilingData = "1 2 256 256 128 43 171 1 170 1 2 256 256 128 42 3 4 256 512 2 0 2 0 256 512 512 ";
    std::vector<size_t> expectWorkspaces = {16780288};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 2: Edge case with single batch
// batch=1, sequence length=64, dim=256, kernel_width=4
// cu_seq_len = 64
TEST_F(CausalConv1dFnTiling, CausalConv1d_950_tiling_single_batch)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            // Input 0: x - (cu_seq_len=64, dim=256)
            {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=256)
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=2, cache_width=3, dim=256)
            {{{2, 3, 256}, {2, 3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=1)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=64, dim=256)
            {{{64, 256}, {64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=2, cache_width=3, dim=256)
            {{{2, 3, 256}, {2, 3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 1 64 64 256 256 64 0 64 1 1 64 64 256 256 1 4 64 256 1 0 1 0 64 256 256 ";
    std::vector<size_t> expectWorkspaces = {16777728};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 3: Medium size test with FP16
// batch=8, each sequence length=512, dim=768, kernel_width=4
// cu_seq_len = 8 * 512 = 4096
TEST_F(CausalConv1dFnTiling, CausalConv1d_950_tiling_medium_fp16)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            // Input 0: x - (cu_seq_len=4096, dim=768)
            {{{4096, 768}, {4096, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=768)
            {{{4, 768}, {4, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=16, cache_width=3, dim=768)
            {{{16, 3, 768}, {16, 3, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=9)
            {{{9}, {9}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=8)
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=8)
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=4096, dim=768)
            {{{4096, 768}, {4096, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=16, cache_width=3, dim=768)
            {{{16, 3, 768}, {16, 3, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 2 85 85 640 128 85 0 84 1 2 85 84 640 128 50 4 4096 768 8 0 8 0 4096 768 768 ";
    std::vector<size_t> expectWorkspaces = {16854016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 4: Variable sequence lengths test
// batch=3 with variable sequence lengths (128, 256, 64), dim=512, kernel_width=4
// cu_seq_len = 128 + 256 + 64 = 448
TEST_F(CausalConv1dFnTiling, CausalConv1d_950_tiling_variable_seqlen)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            // Input 0: x - (cu_seq_len=448, dim=512)
            {{{448, 512}, {448, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=6, cache_width=3, dim=512)
            {{{6, 3, 512}, {6, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=448, dim=512)
            {{{448, 512}, {448, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=6, cache_width=3, dim=512)
            {{{6, 3, 512}, {6, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 1 115 115 512 512 115 0 114 1 1 115 114 512 512 4 4 448 512 3 0 3 0 448 512 512 ";
    std::vector<size_t> expectWorkspaces = {16781312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 5: Large size test with BF16
// batch=4, cu_seq_len=32768, dim=512, kernel_width=4
TEST_F(CausalConv1dFnTiling, CausalConv1d_950_tiling_large_bf16)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            // Input 0: x - (cu_seq_len=32768, dim=512)
            {{{32768, 512}, {32768, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=256, cache_width=3, dim=512)
            {{{256, 3, 512}, {256, 3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=5)
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=4)
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=32768, dim=512)
            {{{32768, 512}, {32768, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=256, cache_width=3, dim=512)
            {{{256, 3, 512}, {256, 3, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "2 4 491 27 128 128 515 0 514 2 4 491 26 128 128 64 4 32768 512 4 0 4 0 32768 512 512 ";
    std::vector<size_t> expectWorkspaces = {16842752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 6: Extra large size test with FP16 (dim split mode)
// batch=256, cu_seq_len=65536, dim=8192, kernel_width=3
TEST_F(CausalConv1dFnTiling, CausalConv1d_950_tiling_xlarge_fp16)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            // Input 0: x - (cu_seq_len=65536, dim=8192)
            {{{65536, 8192}, {65536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=3, dim=8192)
            {{{3, 8192}, {3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=1024, cache_width=2, dim=8192)
            {{{1024, 2, 8192}, {1024, 2, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=257)
            {{{257}, {257}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=256)
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=65536, dim=8192)
            {{{65536, 8192}, {65536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=1024, cache_width=3, dim=8192)
            {{{1024, 3, 8192}, {1024, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "3 64 486 58 128 128 1026 0 1025 3 64 486 57 128 128 64 3 65536 8192 256 0 256 0 65536 8192 8192 ";
    std::vector<size_t> expectWorkspaces = {17825792};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 7: Small size test with FP16
TEST_F(CausalConv1dFnTiling, CausalConv1d_950_tiling_small_fp16)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1d",
        {
            // Input 0: x - (cu_seq_len=8, dim=512)
            {{{8, 512}, {8, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_width=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates/cacheStates - (cache_max_size=4, cache_width=3, dim=512)
            {{{4, 3, 512}, {4, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: initialStateMode - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // Output 0: y - (cu_seq_len=8, dim=512)
            {{{8, 512}, {8, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (cache_max_size=4, cache_width=3, dim=512)
            {{{4, 3, 512}, {4, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 10001;
    std::string expectTilingData = "1 1 8 8 512 512 8 0 8 1 1 8 8 512 512 1 4 8 512 2 0 2 0 8 512 512 ";
    std::vector<size_t> expectWorkspaces = {16778240};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
