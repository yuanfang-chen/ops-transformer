/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/expert_dispatch_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "test_cube_util.h"

#include "opdev/op_log.h"
#define private public
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

using namespace std;
using namespace ge;

class ExpertDispatchTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ExpertDispatchTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ExpertDispatchTiling TearDown" << std::endl;
    }
};

void RunNormalCase(int64_t N, int64_t H, int64_t K, ge::DataType xDataType,std::vector<int64_t> expertRange,
    ge::graphStatus result, int64_t expectTilingKey, string expectTilingData, vector<size_t> expectWorkspaces,
    uint64_t coreNum = 64, uint64_t ubSize = 262144, std::string socVersion = "Ascend910B")
{
    optiling::ExpertDispatchCompileInfo compileInfo = {};
    int64_t E = expertRange[1] - expertRange[0];
    gert::TilingContextPara tilingContextPara(
        "ExpertDispatch",
        {
            {{{N, H}, {N, H}}, xDataType, ge::FORMAT_ND},
            {{{N, K}, {N, K}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{N}, {N}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{N * K, H}, {N * K, H}}, xDataType, ge::FORMAT_ND},
            {{{N * K}, {N * K}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{E}, {E}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{N * K}, {N * K}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {
                "expertRangeListPtr_",
                Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(expertRange)
            },
            {
                "expertStart_",
                Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertRange[0])
            },
            {
                "expertEnd_",
                Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertRange[1])
            }
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize
    );

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 单核+直方图全载+GatherOut全载   1000000
TEST_F(ExpertDispatchTiling, expert_dispatch_tiling_01)
{
    int64_t expectTilingKey = 1000000;
    string expectTilingData = "64 4 3 5 1 7 6 1 20 1 20 20 20 1 20 20 8096 0 1024 20 1 1 1 1 1 1 1 1 20 1 1 1 1 1 1 1 1 1 3 3 ";
    std::vector<size_t> expectWorkspaces = {};
    RunNormalCase(
        4, 3, 5, ge::DT_FLOAT, {1, 7}, ge::GRAPH_SUCCESS,
        expectTilingKey, expectTilingData, expectWorkspaces,
        64, 261888
    );
}

// 学员补充：其他Tilingkey模板
TEST_F(ExpertDispatchTiling, expert_dispatch_tiling_02)
{
    int64_t expectTilingKey = 1001000;
    string expectTilingData = "8 8 131072 1 0 1 1 1 8 1 8 8 8 1 8 8 2016 0 1024 8 1 1 1 1 1 1 1 1 8 1 1 1 1 1 1 1 1 32 4096 4096 ";
    std::vector<size_t> expectWorkspaces = {};
    RunNormalCase(
        8, 131072, 1, ge::DT_FLOAT, {0, 1}, ge::GRAPH_SUCCESS,
        expectTilingKey, expectTilingData, expectWorkspaces,
        8, 65536
    );
}

// 1010000 impossible scenario

// 1011000 impossible scenario

TEST_F(ExpertDispatchTiling, expert_dispatch_tiling_05)
{
    int64_t expectTilingKey = 1100000;
    string expectTilingData = "64 4096 7168 8 0 8 8 16 2048 1 2048 2048 2048 1 2048 2048 8096 4 1024 64 512 512 1 512 512 1 512 512 64 512 512 1 512 512 1 512 512 1 7168 7168 ";
    std::vector<size_t> expectWorkspaces = {};
    RunNormalCase(
        4096, 7168, 8, ge::DT_INT8, {0, 8}, ge::GRAPH_SUCCESS,
        expectTilingKey, expectTilingData, expectWorkspaces,
        64, 261888
    );
}

TEST_F(ExpertDispatchTiling, expert_dispatch_tiling_06)
{
    int64_t expectTilingKey = 1101000;
    string expectTilingData = "2 65536 64 1 0 1 1 2 32768 5 8160 128 32768 5 6560 6528 8160 0 1024 2 32768 32768 1 32768 32768 1 32768 32768 2 32768 32768 2 32680 88 2 32680 88 1 64 64 ";
    std::vector<size_t> expectWorkspaces = {};
    RunNormalCase(
        65536, 64, 1, ge::DT_INT8, {0, 1}, ge::GRAPH_SUCCESS,
        expectTilingKey, expectTilingData, expectWorkspaces,
        2, 261888
    );
}

// 1110000 impossible scenario

TEST_F(ExpertDispatchTiling, expert_dispatch_tiling_08)
{
    int64_t expectTilingKey = 1111000;
    string expectTilingData = "16 4096 7168 256 0 8 8 16 65536 33 2016 1024 65536 33 2016 1024 2016 4 1024 16 65536 65536 5 13108 13104 5 13108 13104 16 65536 65536 15 4568 1584 15 4568 1584 1 7168 7168 ";
    std::vector<size_t> expectWorkspaces = {};
    RunNormalCase(
        4096, 7168, 256, ge::DT_BF16, {0, 8}, ge::GRAPH_SUCCESS,
        expectTilingKey, expectTilingData, expectWorkspaces,
        16, 65536
    );
}
