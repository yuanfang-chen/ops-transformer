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
 * \file test_matmul_allto_all_tiling.cpp
 * \brief tiling ut
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace MatmulAlltoAllUT {

using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "MatmulAlltoAll";

struct MatmulAlltoAllTestParam {
    std::string caseName;
    // input
    // x1
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;

    // x2
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;

    // bias
    std::initializer_list<int64_t> biasShape;
    ge::DataType biasDtype;
    ge::Format biasFormat;

    // x1_scale
    std::initializer_list<int64_t> x1ScaleShape;
    ge::DataType x1ScaleDtype;
    ge::Format x1ScaleFormat;

    // x2_scale
    std::initializer_list<int64_t> x2ScaleShape;
    ge::DataType x2ScaleDtype;
    ge::Format x2ScaleFormat;

    // comm_scale
    std::initializer_list<int64_t> commScaleShape;
    ge::DataType commScaleDtype;
    ge::Format commScaleFormat;

    // x1_offset
    std::initializer_list<int64_t> x1OffsetShape;
    ge::DataType x1OffsetDtype;
    ge::Format x1OffsetFormat;

    // x2_offset
    std::initializer_list<int64_t> x2OffsetShape;
    ge::DataType x2OffsetDtype;
    ge::Format x2OffsetFormat;

    // output
    // y
    std::initializer_list<int64_t> yShape;
    ge::DataType youtputDtype;
    ge::Format youtputFormat;

    // attrs
    std::string groupAttr;
    int64_t worldSizeAttr;
    int64_t alltoAllAxesAttr;
    int64_t yDtypeAttr;
    int64_t x1QuantModeAttr;
    int64_t x2QuantModeAttr;
    int64_t commQuantModeAttr;
    int64_t commQuantDtypeAttr;
    bool transposex1Attr;
    bool transposex2Attr;
    int64_t groupSizeAttr;
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
static MatmulAlltoAllTestParam g_testCases[] = {
    // legal
    {"matmul_alltoall_case_normal_dtype_float16",
    {88, 128}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {128, 256}, ge::DT_FLOAT16,ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    {176, 128},ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", 2, 0, 0, 0, 0, 0, 0, false, false, 0,
    "Ascend910_93",
    ge::GRAPH_SUCCESS,
    0UL, "", {16867328}, 0},
};

// setup & teardown
class MatmulAlltoAllA3TilingTest : public testing::TestWithParam<MatmulAlltoAllTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAlltoAllA3TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAlltoAllA3TilingTest TearDown." << std::endl;
    }
};

struct MatmulAlltoAllCompileInfo {
} compileInfo;

// ut
static void TestOneParamCase(const MatmulAlltoAllTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    //
    //  Shape  tensor  shape  gert::StorageShape
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape biasShape = {param.biasShape, param.biasShape};
    gert::StorageShape x1ScaleShape = {param.x1ScaleShape, param.x1ScaleShape};
    gert::StorageShape x2ScaleShape = {param.x2ScaleShape, param.x2ScaleShape};
    gert::StorageShape commScaleShape = {param.commScaleShape, param.commScaleShape};
    gert::StorageShape x1OffsetShape = {param.x1OffsetShape, param.x1OffsetShape};
    gert::StorageShape x2OffsetShape = {param.x2OffsetShape, param.x2OffsetShape};

    gert::StorageShape yShape = {param.yShape, param.yShape};

    //  input tensor
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1Shape, param.x1Dtype, param.x1Format},
         {x2Shape, param.x2Dtype, param.x2Format},
         {biasShape, param.biasDtype, param.biasFormat},
         {x1ScaleShape, param.x1ScaleDtype, param.x1ScaleFormat},
         {x2ScaleShape, param.x2ScaleDtype, param.x2ScaleFormat},
         {commScaleShape, param.commScaleDtype, param.commScaleFormat},
         {x1OffsetShape, param.x1OffsetDtype, param.x1OffsetFormat},
         {x2OffsetShape, param.x2OffsetDtype, param.x2OffsetFormat}});

    //  output tensor
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.youtputDtype, param.youtputFormat}});

    //  attributes
    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupAttr)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSizeAttr)},
         {"all2all_axes",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.alltoAllAxesAttr))},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.yDtypeAttr))},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantModeAttr)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantModeAttr)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantModeAttr)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtypeAttr)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposex1Attr)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposex2Attr)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSizeAttr)}});
    //
    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.socVersion);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);                                          
}

static void ThreadFunction(const MatmulAlltoAllTestParam *testCases, size_t caseNum, size_t threadIdx, size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        TestOneParamCase(testCases[idx]);
    }
}

static void TestExecMultiThread(const MatmulAlltoAllTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(MatmulAlltoAllA3TilingTest, GeneralCasesTest)
{
    TestOneParamCase(GetParam());
}

TEST_F(MatmulAlltoAllA3TilingTest, GeneralCasesMultiThread)
{
    TestExecMultiThread(g_testCases, sizeof(g_testCases) / sizeof(MatmulAlltoAllTestParam), 1);
}

INSTANTIATE_TEST_CASE_P(MatmulAlltoAllTilingUT, MatmulAlltoAllA3TilingTest, testing::ValuesIn(g_testCases));

} // namespace
