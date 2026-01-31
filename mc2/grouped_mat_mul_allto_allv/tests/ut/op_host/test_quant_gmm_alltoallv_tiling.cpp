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
 * \file test_quant_gmm_alltoallv_tiling.cpp
 * \brief tiling ut
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {

using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "GroupedMatMulAlltoAllv";

struct QuantGmmAlltoAllvTestParam {
    std::string caseName;
    // input
    // gmmX
    std::initializer_list<int64_t> gmmXShape;
    ge::DataType gmmXDtype;
    ge::Format gmmXFormat;

    // gmmWeight
    std::initializer_list<int64_t> gmmWeightShape;
    ge::DataType gmmWeightDtype;
    ge::Format gmmWeightFormat;

    // sendCountsTensor
    std::initializer_list<int64_t> sendCountsTensorShape;
    ge::DataType sendCountsTensorDtype;
    ge::Format sendCountsTensorFormat;

    // recvCountsTensor
    std::initializer_list<int64_t> recvCountsTensorShape;
    ge::DataType recvCountsTensorDtype;
    ge::Format recvCountsTensorFormat;

    // mmX
    std::initializer_list<int64_t> mmXShape;
    ge::DataType mmXDtype;
    ge::Format mmXFormat;

    // mmWeight
    std::initializer_list<int64_t> mmWeightShape;
    ge::DataType mmWeightDtype;
    ge::Format mmWeightFormat;

    // gmmXScale
    std::initializer_list<int64_t> gmmXScaleShape;
    ge::DataType gmmXScaleDtype;
    ge::Format gmmXScaleFormat;

    // gmmWeightScale
    std::initializer_list<int64_t> gmmWeightScaleShape;
    ge::DataType gmmWeightScaleDtype;
    ge::Format gmmWeightScaleFormat;

    // mmXScale
    std::initializer_list<int64_t> mmXScaleShape;
    ge::DataType mmXScaleDtype;
    ge::Format mmXScaleFormat;

    // mmWeightScale
    std::initializer_list<int64_t> mmWeightScaleShape;
    ge::DataType mmWeightScaleDtype;
    ge::Format mmWeightScaleFormat;

    // commScale
    std::initializer_list<int64_t> commScaleShape;
    ge::DataType commScaleDtype;
    ge::Format commScaleFormat;

    // output
    // gmmY
    std::initializer_list<int64_t> gmmYShape;
    ge::DataType gmmYDtype;
    ge::Format gmmYFormat;
    
    // mmY
    std::initializer_list<int64_t> mmYShape;
    ge::DataType mmYDtype;
    ge::Format mmYFormat;

    // attrs
    int64_t gmmXQuantModeAttr;
    int64_t gmmWeightQuantModeAttr;
    int64_t mmXQuantModeAttr;
    int64_t mmWeightQuantModeAttr;
    int64_t commQuantModeAttr;
    int64_t commQuantDtypeAttr;

    std::string groupAttr;
    int64_t epWorldSizeAttr;
    std::vector<int64_t> sendCountsAttr;
    std::vector<int64_t> recvCountsAttr;
    bool transposeGmmWeightAttr;
    bool transposeMmWeightAttr;
    // soc version
    std::string socVersion;
    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

// ut/
// expectWorkspaces = 16 * 1024 * 1024
// tilingDataReservedLen = 43tilingDatamc2InitTilingmc2CcTiling
static QuantGmmAlltoAllvTestParam test_cases[] =
{
    // legal
    {
        "qgmm_alltoallv_case1_gmm_dtype_hifloat8_nobias_nomm",
        // gmm
        {80, 128}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 128, 256}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {}, ge::DT_FLOAT, ge::FORMAT_ND,
        {}, ge::DT_FLOAT, ge::FORMAT_ND,
        {}, ge::DT_FLOAT, ge::FORMAT_ND,
        {}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // scale
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {}, ge::DT_FLOAT, ge::FORMAT_ND,
        {}, ge::DT_FLOAT, ge::FORMAT_ND,
        {}, ge::DT_FLOAT, ge::FORMAT_ND,

        // output
        {40, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
        {}, ge::DT_FLOAT16, ge::FORMAT_ND,
        // attr
        1, 1, 0, 0, 0, 0, // quantMode
        "group",
        2,                // ep_worldsize
        {8, 16, 24, 32},
        {32, 24, 16, 8},
        false, false,     // trans
        "Ascend910_95",
        ge::GRAPH_SUCCESS,
        33UL,             // tilingKey
        "",               // tilingData
        {16822272},       // workspace
        0
    },
};

// setup & teardown
class TestQuantGmmAlltoAllvTiling : public testing::TestWithParam<QuantGmmAlltoAllvTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TestQuantGmmAlltoAllvTiling SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TestQuantGmmAlltoAllvTiling TearDown." << std::endl;
    }
};

struct QuantGmmAlltoAllvCompileInfo {
} compileInfo;

// ut
static void TestOneParamCase(const QuantGmmAlltoAllvTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    //
    //  Shape  tensor  shape  gert::StorageShape
    gert::StorageShape gmmXShape = {param.gmmXShape, param.gmmXShape};
    gert::StorageShape gmmWeightShape = {param.gmmWeightShape, param.gmmWeightShape};
    gert::StorageShape sendCountsTensorShape = {param.sendCountsTensorShape, param.sendCountsTensorShape};
    gert::StorageShape recvCountsTensorShape = {param.recvCountsTensorShape, param.recvCountsTensorShape};
    gert::StorageShape mmXShape = {param.mmXShape, param.mmXShape};
    gert::StorageShape mmWeightShape = {param.mmWeightShape, param.mmWeightShape};
    gert::StorageShape gmmXScaleShape = {param.gmmXScaleShape, param.gmmXScaleShape};
    gert::StorageShape gmmWeightScaleShape = {param.gmmWeightScaleShape, param.gmmWeightScaleShape};
    gert::StorageShape mmXScaleShape = {param.mmXScaleShape, param.mmXScaleShape};
    gert::StorageShape mmWeightScaleShape = {param.mmWeightScaleShape, param.mmWeightScaleShape};
    gert::StorageShape commScaleShape = {param.commScaleShape, param.commScaleShape};
    gert::StorageShape gmmYShape = {param.gmmYShape, param.gmmYShape};
    gert::StorageShape mmYShape = {param.mmYShape, param.mmYShape};

    //  input tensor
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {
            {gmmXShape, param.gmmXDtype, param.gmmXFormat},
            {gmmWeightShape, param.gmmWeightDtype, param.gmmWeightFormat},
            {sendCountsTensorShape, param.sendCountsTensorDtype, param.sendCountsTensorFormat},
            {recvCountsTensorShape, param.recvCountsTensorDtype, param.recvCountsTensorFormat},
            {mmXShape, param.mmXDtype, param.mmXFormat},
            {mmWeightShape, param.mmWeightDtype, param.mmWeightFormat},
            {gmmXScaleShape, param.gmmXScaleDtype, param.gmmXScaleFormat},
            {gmmWeightScaleShape, param.gmmWeightScaleDtype, param.gmmWeightScaleFormat},
            {mmXScaleShape, param.mmXScaleDtype, param.mmXScaleFormat},
            {mmWeightScaleShape, param.mmWeightDtype, param.mmWeightFormat},
            {commScaleShape, param.commScaleDtype, param.commScaleFormat}
        }
    );

    //  output tensor
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {
            {gmmYShape, param.gmmYDtype, param.gmmYFormat},
            {mmYShape, param.mmYDtype, param.mmYFormat},
        }
    );

    //  attributes
    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {
            {"gmmX_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmXQuantModeAttr)},
            {"gmmWeight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmWeightQuantModeAttr)},
            {"mmX_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmXQuantModeAttr)},
            {"mmWeight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmWeightQuantModeAttr)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantModeAttr)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtypeAttr)},
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupAttr)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSizeAttr)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.sendCountsAttr)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.recvCountsAttr)},
            {"transpose_gmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeGmmWeightAttr)},
            {"transpose_mmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeMmWeightAttr)}
        }
    );
    //
    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.socVersion);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);                                          
}

static void ThreadFunction(const QuantGmmAlltoAllvTestParam *testCases, size_t caseNum, size_t threadIdx, size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        TestOneParamCase(testCases[idx]);
    }
}

static void TestExecMultiThread(const QuantGmmAlltoAllvTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(TestQuantGmmAlltoAllvTiling, general_cases)
{
    TestOneParamCase(GetParam());
}

TEST_F(TestQuantGmmAlltoAllvTiling, general_cases_multi_thread)
{
    TestExecMultiThread(test_cases, sizeof(test_cases) / sizeof(QuantGmmAlltoAllvTestParam), 1);
}

INSTANTIATE_TEST_CASE_P(QuantGmmAlltoAllvTilingUT, TestQuantGmmAlltoAllvTiling, testing::ValuesIn(test_cases));

} // namespace
