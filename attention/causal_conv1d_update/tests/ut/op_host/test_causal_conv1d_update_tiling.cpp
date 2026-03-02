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
        // Force linker to include tiling implementation with registration
        (void)sizeof(optiling::CausalConv1dUpdateTiling);
    }

    static void TearDownTestCase()
    {
        std::cout << "CausalConv1dUpdateTiling TearDown" << std::endl;
    }
};

// Test case 1: Basic FP16 test with 3D input
// batch=2, seq_len=3, dim=512, kernel_size=4
TEST_F(CausalConv1dUpdateTiling, CausalConv1dUpdate_950_tiling_basic_fp16)
{
    optiling::CausalConv1dUpdateCompileInfo compileInfo = {
        64, 262144};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"activationMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"padSlotId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"residualConnMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"runMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
    };

    gert::TilingContextPara tilingContextPara(
        "CausalConv1dUpdate",
        {
            // Input 0: x - (batch=2, seq_len=3, dim=512)
            {{{2, 3, 512}, {2, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: weight - (kernel_size=4, dim=512)
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 2: convStates - (batch=2, cache_len=4+3-2=5, dim=512)
            // cache_len = kernel_size + seq_len - 2 = 4 + 3 - 2 = 5
            {{{2, 5, 512}, {2, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 3: queryStartLoc - (batch+1=3)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 4: cacheIndices - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 5: hasInitialState - (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // Input 6: bias - optional, (dim=512)
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 7: numAcceptedTokens - optional, (batch=2)
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        {
            // Output 0: y - (batch=2, seq_len=3, dim=512)
            {{{2, 3, 512}, {2, 3, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Output 1: cacheStates - (batch=2, cache_len=5, dim=512)
            {{{2, 5, 512}, {2, 5, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs,
        &compileInfo);

    int64_t expectTilingKey = 30000;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
