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
 * \file test_moe_distribute_combine_teardown_tiling.cpp
 * \brief host侧tiling ut
 */
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeCombineTeardownUT {

const std::string OP_NAME = "MoeDistributeCombineTeardown";

struct MoeDistributeCombineTeardownTestParam {
    std::string caseName;
    // x
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    // output
    std::initializer_list<int64_t> outputShape;
    ge::DataType outputDtype;
    ge::Format outputFormat;
    // attrs
    std::string groupEpAttr;
    int64_t epWorldSizeAttr;
    int64_t epRankIdAttr;
    int64_t moeExpertNumAttr;
    int64_t sharedExpertNumAttr;
    int64_t sharedExpertRankNumAttr;
    int64_t quantModeAttr;
    // soc version
    std::string socVersion;
    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
    uint64_t rankNum;
};

inline std::ostream& operator<<(std::ostream& os, const MoeDistributeCombineTeardownTestParam& param)
{
    return os << param.caseName;
}

// 构造ut用例：这里可以按照正常用例/异常用例分开声明
static MoeDistributeCombineTeardownTestParam g_testCases[] = {
    {"combine_teardown_1",
    {8, 4096, 6}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {8, 4096, 6}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "hccl_world_group", 2, 0, 32, 0, 0, 1,
    "3510",
    ge::GRAPH_SUCCESS, 0UL, "", {}, MC2_TILING_DATA_RESERVED_LEN, 8},
};

class MoeDistributeCombineTeardownTilingTest : public testing::TestWithParam<MoeDistributeCombineTeardownTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownTilingTest TearDown." << std::endl;
    }
};

static struct MoeDistributeCombineTeardownCompileInfo {} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const MoeDistributeCombineTeardownTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    // 参数封装
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape outputShape = {param.outputShape, param.outputShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_({
        {xShape, param.xDtype, param.xFormat}
    });
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({
        {outputShape, param.outputDtype, param.outputFormat}
    });
    std::vector<gert::TilingContextPara::OpAttr> attrs_({
        {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupEpAttr)},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSizeAttr)},
        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epRankIdAttr)},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.moeExpertNumAttr)},
        {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertNumAttr)},
        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sharedExpertRankNumAttr)},
        {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quantModeAttr)}
    });
    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

static void ThreadFunction(const MoeDistributeCombineTeardownTestParam *testCases, size_t caseNum, size_t threadIdx,
                           size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);
    }
}

static void TestExecMultiThread(const MoeDistributeCombineTeardownTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(MoeDistributeCombineTeardownTilingTest, GeneralCasesTest)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey, param.expectTilingData,
                       param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

TEST_F(MoeDistributeCombineTeardownTilingTest, GeneralCasesMultiThreadTest)
{
    TestExecMultiThread(g_testCases, sizeof(g_testCases) / sizeof(MoeDistributeCombineTeardownTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(MoeDistributeCombineTeardownTilingUT, MoeDistributeCombineTeardownTilingTest, testing::ValuesIn(g_testCases));

} // namespace
