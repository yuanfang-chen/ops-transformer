/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_causal_conv1d_tiling.cpp
 * \brief Unit tests for AggregateHiddenGrad tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/aggregate_hidden_grad_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AggregateHiddenGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AggregateHiddenGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AggregateHiddenGradTiling TearDown" << std::endl;
    }
};


TEST_F(AggregateHiddenGradTiling, AggregateHiddenGrad_950_tiling_bf_b4_s1_d512)
{
    optiling::AggregateHiddenGradArch35CompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {

    };

    gert::TilingContextPara tilingContextPara("AggregateHiddenGrad",
                                              {
                                                  // Input 0: x - (batch=4, seq_len=1, dim=512)
                                                  {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  // Input 1: weight - (kernel_size=3, dim=512)
                                                  {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  // Input 2: convStates - (batch=4, cache_len=3+1-2=2, dim=512)
                                                  {{{3, 512}, {3, 512}}, ge::DT_BF16, ge::FORMAT_ND},

                                              },
                                              {
                                                  // Output 0: y - (batch=4, seq_len=1, dim=512)
                                                  {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  // Output 1: cacheStates - (batch=4, cache_len=2, dim=512)
                                                  {{{4, 1, 512}, {4, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 0;
    std::string expectTilingData =
        "16 4 4 4 0 128 128 4 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 4 1 0 512 3 2 512 1024 512 -1 0 1 0 ";

    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
