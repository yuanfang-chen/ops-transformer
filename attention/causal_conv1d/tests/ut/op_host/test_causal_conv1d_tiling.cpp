/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/causal_conv1d_tiling_data.h"

using namespace std;
using namespace ge;

class CausalConv1dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CausalConv1dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dTiling TearDown" << std::endl;
    }
};

std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};

TEST_F(CausalConv1dTiling, causal_conv1d_0) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: (num_cache_lines, state_len, dim)
                                                    {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},          // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},          // cache_indices: (batch)
                                                    {{{2}, {2}}, ge::DT_BOOL, ge::FORMAT_ND},           // has_initial_state: (batch)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 0;
    // dim, cuSeqlen, seqLen, inputMode, width, stateLen, batch, activationMode, padSlotId,
    // hasBias, dimTileSize, blocksPerSeq
    string expectTilingData = "1024 5 0 0 4 3 2 0 -1 1 1024 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_1) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},         // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},         // cache_indices
                                                    {{{2}, {2}}, ge::DT_BOOL, ge::FORMAT_ND},          // has_initial_state
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_BF16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1024 5 0 0 4 3 2 1 -1 1 1024 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_1d_tiling_no_bias) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // bias: optional (absent)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},            // query_start_loc: (batch+1)
                                                    {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},            // cache_indices: (batch)
                                                    {{{1}, {1}}, ge::DT_BOOL, ge::FORMAT_ND},             // has_initial_state: (batch)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1024 5 0 0 4 3 1 0 -1 0 1024 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_3d_batch_mode) {
    // Test 3D batch mode: x.shape = (batch, seqlen, dim)
    // This triggers inputMode=1 with fixed seqLen for all sequences
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{4, 8, 1024}, {4, 8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (batch=4, seqlen=8, dim=1024)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight: (width, dim)
                                                    {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // bias: (dim)
                                                    {{{16, 3, 1024}, {16, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states: (num_cache_lines, state_len, dim)
                                                    {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},               // query_start_loc: (batch+1) - still required but ignored in 3D mode
                                                    {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},               // cache_indices: (batch)
                                                    {{{4}, {4}}, ge::DT_BOOL, ge::FORMAT_ND},                // has_initial_state: (batch)
                                                },
                                                {
                                                    {{{4, 8, 1024}, {4, 8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y: (batch, seqlen, dim)
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    // 3D batch mode: batch=4, dim=1024, seqLen=8, cuSeqlen=32 (4*8), inputMode=1
    uint64_t expectTilingKey = 0;
    // Expected: dim=1024, cuSeqlen=32, seqLen=8, inputMode=1, width=4, stateLen=3, batch=4,
    //           activationMode=1, padSlotId=-1, hasBias=1, dimTileSize=1024, blocksPerSeq=1
    string expectTilingData = "1024 32 8 1 4 3 4 1 -1 1 1024 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_weight_shape_width_dim) {
    // Verify (width, dim) weight shape is accepted (dim contiguous).
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x: (cu_seqlen, dim)
                                                    {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight: (width, dim)
                                                    {{}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // bias: optional (absent)
                                                    {{{8, 3, 1024}, {8, 3, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // conv_states
                                                    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},            // query_start_loc: (batch+1)
                                                    {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},            // cache_indices: (batch)
                                                    {{{1}, {1}}, ge::DT_BOOL, ge::FORMAT_ND},             // has_initial_state: (batch)
                                                },
                                                {
                                                    {{{5, 1024}, {5, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 0;
    // dim, cuSeqlen, seqLen, inputMode, width, stateLen, batch, activationMode, padSlotId,
    // hasBias, dimTileSize, blocksPerSeq
    string expectTilingData = "1024 5 0 0 4 3 1 0 -1 0 1024 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
