/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
                                                    {{{16, 5}, {16, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x: (dim, cu_seqlen)
                                                    {{{16, 4}, {16, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // weight: (dim, width)
                                                    {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 16, 3}, {8, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, // conv_states: (num_cache_lines, dim, state_len)
                                                    {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},          // query_start_loc: (batch+1)
                                                    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},          // cache_indices: (batch)
                                                    {{{2}, {2}}, ge::DT_BOOL, ge::FORMAT_ND},           // has_initial_state: (batch)
                                                },
                                                {
                                                    {{{16, 5}, {16, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "16 5 4 3 2 0 -1 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CausalConv1dTiling, causal_conv1d_1) {
    struct CausalConv1dCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("CausalConv1d",
                                                {
                                                    {{{16, 5}, {16, 5}}, ge::DT_BF16, ge::FORMAT_ND}, // x: (dim, cu_seqlen)
                                                    {{{16, 4}, {16, 4}}, ge::DT_BF16, ge::FORMAT_ND}, // weight: (dim, width)
                                                    {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},        // bias: (dim)
                                                    {{{8, 16, 3}, {8, 16, 3}}, ge::DT_BF16, ge::FORMAT_ND}, // conv_states
                                                    {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},         // query_start_loc
                                                    {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},         // cache_indices
                                                    {{{2}, {2}}, ge::DT_BOOL, ge::FORMAT_ND},          // has_initial_state
                                                },
                                                {
                                                    {{{16, 5}, {16, 5}}, ge::DT_BF16, ge::FORMAT_ND}, // y
                                                },
                                                {
                                                    {"activation_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"pad_slot_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                },
                                                &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "16 5 4 3 2 1 -1 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
