// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /*!
//  * \file test_quant_gmm_alltoallv_tiling.cpp
//  * \brief tiling ut
//  */

// #include <iostream>
// #include <fstream>
// #include <thread>
// #include <vector>
// #include <gtest/gtest.h>
// #include "tiling_context_faker.h"
// #include "tiling_case_executor.h"

// namespace {

// using namespace std;
// using namespace ge;
// using namespace gert;

// static const std::string OP_NAME = "GroupedMatMulAlltoAllv";

// struct QuantGmmAlltoAllvTestParam {
//     std::string caseName;
//     // input
//     // gmmX
//     std::initializer_list<int64_t> gmmXShape;
//     ge::DataType gmmXDtype;
//     ge::Format gmmXFormat;
//     // gmmWeight
//     std::initializer_list<int64_t> gmmWeightShape;
//     ge::DataType gmmWeightDtype;
//     ge::Format gmmWeightFormat;

//     // sendCountsTensor
//     std::initializer_list<int64_t> sendCountsTensorShape;
//     ge::DataType sendCountsTensorDtype;
//     ge::Format sendCountsTensorFormat;
//     // recvCountsTensor
//     std::initializer_list<int64_t> recvCountsTensorShape;
//     ge::DataType recvCountsTensorDtype;
//     ge::Format recvCountsTensorFormat;

//     // mmX
//     std::initializer_list<int64_t> mmXShape;
//     ge::DataType mmXDtype;
//     ge::Format mmXFormat;
//     // mmWeight
//     std::initializer_list<int64_t> mmWeightShape;
//     ge::DataType mmWeightDtype;
//     ge::Format mmWeightFormat;

//     // gmmXScale
//     std::initializer_list<int64_t> gmmXScaleShape;
//     ge::DataType gmmXScaleDtype;
//     ge::Format gmmXScaleFormat;
//     // gmmWeightScale
//     std::initializer_list<int64_t> gmmWeightScaleShape;
//     ge::DataType gmmWeightScaleDtype;
//     ge::Format gmmWeightScaleFormat;

//     // gmmXOffset
//     std::initializer_list<int64_t> gmmXOffsetShape;
//     ge::DataType gmmXOffsetDtype;
//     ge::Format gmmXOffsetFormat;
//     // gmmWeightOffset
//     std::initializer_list<int64_t> gmmWeightOffsetShape;
//     ge::DataType gmmWeightOffsetDtype;
//     ge::Format gmmWeightOffsetFormat;

//     // mmXScale
//     std::initializer_list<int64_t> mmXScaleShape;
//     ge::DataType mmXScaleDtype;
//     ge::Format mmXScaleFormat;
//     // mmWeightScale
//     std::initializer_list<int64_t> mmWeightScaleShape;
//     ge::DataType mmWeightScaleDtype;
//     ge::Format mmWeightScaleFormat;

//     // mmXOffset
//     std::initializer_list<int64_t> mmXOffsetShape;
//     ge::DataType mmXOffsetDtype;
//     ge::Format mmXOffsetFormat;
//     // mmWeightOffset
//     std::initializer_list<int64_t> mmWeightOffsetShape;
//     ge::DataType mmWeightOffsetDtype;
//     ge::Format mmWeightOffsetFormat;

//     // commScale
//     std::initializer_list<int64_t> commScaleShape;
//     ge::DataType commScaleDtype;
//     ge::Format commScaleFormat;

//     // output
//     // gmmY
//     std::initializer_list<int64_t> gmmYShape;
//     ge::DataType gmmYDtype;
//     ge::Format gmmYFormat;
    
//     // mmY
//     std::initializer_list<int64_t> mmYShape;
//     ge::DataType mmYDtype;
//     ge::Format mmYFormat;

//     // attrs
//     std::string groupAttr;
//     int64_t epWorldSizeAttr;
//     std::vector<int64_t> sendCountsAttr;
//     std::vector<int64_t> recvCountsAttr;
//     bool transposeGmmWeightAttr;
//     bool transposeMmWeightAttr;

//     int64_t gmmXQuantModeAttr;
//     int64_t gmmWeightQuantModeAttr;
//     int64_t mmXQuantModeAttr;
//     int64_t mmWeightQuantModeAttr;
//     int64_t commQuantModeAttr;
//     int64_t groupSizeAttr;
//     int64_t gmmYDtypeAttr;
//     int64_t mmYDtypeAttr;
//     int64_t commQuantDtypeAttr;

//     // soc version
//     std::string socVersion;
//     // expert result
//     ge::graphStatus status;
//     uint64_t expectTilingKey;
//     std::string expectTilingData;
//     std::vector<size_t> expectWorkspaces;
//     uint64_t mc2TilingDataReservedLen;
// };

// // ut/
// // expectWorkspaces = 16 * 1024 * 1024
// // tilingDataReservedLen = 43tilingDatamc2InitTilingmc2CcTiling
// static QuantGmmAlltoAllvTestParam test_cases[] =
// {
//     // legal
//     {
//         "qgmm_alltoallv_case1_gmm_dtype_hifloat8_nobias_nomm",
//         // gmm
//         {80, 128}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
//         {4, 128, 256}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT, ge::FORMAT_ND,
//         {1}, ge::DT_FLOAT, ge::FORMAT_ND, // scale
//         {1}, ge::DT_FLOAT, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT, ge::FORMAT_ND,

//         // output
//         {40, 256}, ge::DT_FLOAT16, ge::FORMAT_ND,
//         {}, ge::DT_FLOAT16, ge::FORMAT_ND,
//         // attr
//         1, 1, 0, 0, 0, 0, // quantMode
//         "group",
//         2,                // ep_worldsize
//         {8, 16, 24, 32},
//         {32, 24, 16, 8},
//         false, false,     // trans
//         "Ascend950",
//         ge::GRAPH_SUCCESS,
//         33UL,             // tilingKey
//         "",               // tilingData
//         {16822272},       // workspace
//         0
//     },
// };

// // setup & teardown
// class TestQuantGmmAlltoAllvTiling : public testing::TestWithParam<QuantGmmAlltoAllvTestParam> {
// protected:
//     static void SetUpTestCase()
//     {
//         std::cout << "TestQuantGmmAlltoAllvTiling SetUp." << std::endl;
//     }

//     static void TearDownTestCase()
//     {
//         std::cout << "TestQuantGmmAlltoAllvTiling TearDown." << std::endl;
//     }
// };

// struct QuantGmmAlltoAllvCompileInfo {
// } compileInfo;

// // ut
// static void TestOneParamCase(const QuantGmmAlltoAllvTestParam &param)
// {
//     std::cout << "[TEST_CASE] " << param.caseName << std::endl;
//     //
//     //  Shape  tensor  shape  gert::StorageShape
//     gert::StorageShape gmmXShape = {param.gmmXShape, param.gmmXShape};
//     gert::StorageShape gmmWeightShape = {param.gmmWeightShape, param.gmmWeightShape};
//     gert::StorageShape sendCountsTensorShape = {param.sendCountsTensorShape, param.sendCountsTensorShape};
//     gert::StorageShape recvCountsTensorShape = {param.recvCountsTensorShape, param.recvCountsTensorShape};
//     gert::StorageShape mmXShape = {param.mmXShape, param.mmXShape};
//     gert::StorageShape mmWeightShape = {param.mmWeightShape, param.mmWeightShape};
//     gert::StorageShape gmmXScaleShape = {param.gmmXScaleShape, param.gmmXScaleShape};
//     gert::StorageShape gmmWeightScaleShape = {param.gmmWeightScaleShape, param.gmmWeightScaleShape};
//     gert::StorageShape mmXScaleShape = {param.mmXScaleShape, param.mmXScaleShape};
//     gert::StorageShape mmWeightScaleShape = {param.mmWeightScaleShape, param.mmWeightScaleShape};
//     gert::StorageShape commScaleShape = {param.commScaleShape, param.commScaleShape};
//     gert::StorageShape gmmYShape = {param.gmmYShape, param.gmmYShape};
//     gert::StorageShape mmYShape = {param.mmYShape, param.mmYShape};

//     //  input tensor
//     std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
//         {
//             {gmmXShape, param.gmmXDtype, param.gmmXFormat},
//             {gmmWeightShape, param.gmmWeightDtype, param.gmmWeightFormat},
//             {sendCountsTensorShape, param.sendCountsTensorDtype, param.sendCountsTensorFormat},
//             {recvCountsTensorShape, param.recvCountsTensorDtype, param.recvCountsTensorFormat},
//             {mmXShape, param.mmXDtype, param.mmXFormat},
//             {mmWeightShape, param.mmWeightDtype, param.mmWeightFormat},
//             {gmmXScaleShape, param.gmmXScaleDtype, param.gmmXScaleFormat},
//             {gmmWeightScaleShape, param.gmmWeightScaleDtype, param.gmmWeightScaleFormat},
//             {mmXScaleShape, param.mmXScaleDtype, param.mmXScaleFormat},
//             {mmWeightScaleShape, param.mmWeightDtype, param.mmWeightFormat},
//             {commScaleShape, param.commScaleDtype, param.commScaleFormat}
//         }
//     );

//     //  output tensor
//     std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
//         {
//             {gmmYShape, param.gmmYDtype, param.gmmYFormat},
//             {mmYShape, param.mmYDtype, param.mmYFormat},
//         }
//     );

//     //  attributes
//     std::vector<gert::TilingContextPara::OpAttr> attrs_(
//         {
//             {"gmmX_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmXQuantModeAttr)},
//             {"gmmWeight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmWeightQuantModeAttr)},
//             {"mmX_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmXQuantModeAttr)},
//             {"mmWeight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmWeightQuantModeAttr)},
//             {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantModeAttr)},
//             {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtypeAttr)},
//             {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupAttr)},
//             {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSizeAttr)},
//             {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.sendCountsAttr)},
//             {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.recvCountsAttr)},
//             {"transpose_gmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeGmmWeightAttr)},
//             {"transpose_mmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeMmWeightAttr)}
//         }
//     );
//     //
//     gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
//                                               param.socVersion);
//     ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
//                         param.expectWorkspaces, param.mc2TilingDataReservedLen);                                          
// }

// static void ThreadFunction(const QuantGmmAlltoAllvTestParam *testCases, size_t caseNum, size_t threadIdx, size_t threadNum)
// {
//     for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
//         TestOneParamCase(testCases[idx]);
//     }
// }

// static void TestExecMultiThread(const QuantGmmAlltoAllvTestParam *testCases, size_t testCaseNum, size_t threadNum)
// {
//     std::thread threads[threadNum];
//     for (size_t idx = 0; idx < threadNum; ++idx) {
//         threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
//     }
//     for (size_t idx = 0; idx < threadNum; ++idx) {
//         threads[idx].join();
//     }
// }

// TEST_P(TestQuantGmmAlltoAllvTiling, general_cases)
// {
//     TestOneParamCase(GetParam());
// }

// TEST_F(TestQuantGmmAlltoAllvTiling, general_cases_multi_thread)
// {
//     TestExecMultiThread(test_cases, sizeof(test_cases) / sizeof(QuantGmmAlltoAllvTestParam), 1);
// }

// INSTANTIATE_TEST_CASE_P(QuantGmmAlltoAllvTilingUT, TestQuantGmmAlltoAllvTiling, testing::ValuesIn(test_cases));

// } // namespace

#include <iostream>
#include <gtest/gtest.h>

#include "mc2_tiling_case_executor.h"
#include "../../../op_host/op_tiling/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"

using namespace std;

struct GroupedMatMulAlltoAllvTilingTestParam {
    string caseName;

    // input
    std::vector<int64_t> gmmXShape;
    ge::DataType gmmXDataType;
    ge::Format gmmXFormat;

    std::vector<int64_t> gmmWeightShape;
    ge::DataType gmmWeightDataType;
    ge::Format gmmWeightFormat;

    std::vector<int64_t> gmmXScaleShape;
    ge::DataType gmmXScaleDataType;
    ge::Format gmmXScaleFormat;

    std::vector<int64_t> gmmWeightScaleShape;
    ge::DataType gmmWeightScaleDataType;
    ge::Format gmmWeightScaleFormat;

    std::vector<int64_t> mmXShape;
    ge::DataType mmXDataType;
    ge::Format mmXFormat;

    std::vector<int64_t> mmWeightShape;
    ge::DataType mmWeightDataType;
    ge::Format mmWeightFormat;

    std::vector<int64_t> mmXScaleShape;
    ge::DataType mmXScaleDataType;
    ge::Format mmXScaleFormat;

    std::vector<int64_t> mmWeightScaleShape;
    ge::DataType mmWeightScaleDataType;
    ge::Format mmWeightScaleFormat;

    std::vector<int64_t> sendCounts;
    std::vector<int64_t> recvCounts;

    // output: expected output tensor
    std::vector<int64_t> gmmYShape;
    ge::DataType gmmYDataType;
    ge::Format gmmYFormat;

    std::vector<int64_t> mmYShape;
    ge::DataType mmYDataType;
    ge::Format mmYFormat;

    // Attributes
    bool gmm_x_quant_mode;
    bool gmm_weight_quant_mode;
    bool mm_x_quant_mode;
    bool mm_weight_quant_mode;

    bool trans_gmm_weight_flag;
    bool trans_mm_weight_flag;

    int64_t world_size;
    int64_t ep_world_size;
    int64_t graph_type;

    // Expected result
    ge::graphStatus expectedStatus;

    // Expected tiling key
    uint64_t expectTilingKey;
};

static const vector<GroupedMatMulAlltoAllvTilingTestParam> groupedMatMulAlltoAllvTilingTestParam = {
    // // 非量化用例
    // {
    //     "Test_gmmWeight_size",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,        // gmmX
    //     {64, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,    // gmmWeight (e=64)
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,                    // gmmXScale
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,                    // gmmWeightScale
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,                  // mmX
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,                  // mmWeight
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,                    // mmXScale
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,                    // mmWeightScale
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}, // sendCounts
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}, // recvCounts
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,        // gmmY
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,                  // mmY
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,        // permuteOut
    //     false, false, false, false,                         // quant modes
    //     false, false, true, true,                           // trans flags, permute, mm_out
    //     8, 8, 0,                                            // world_size, ep_world_size, graph_type
    //     ge::GRAPH_FAILED, 0                                 // expected status, tiling key
    // },
    
    // // Test_ep_world_size
    // {
    //     "Test_ep_world_size",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 4, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_e (same as Test_gmmWeight_size)
    // {
    //     "Test_e",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {64, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_e_multi_ep
    // {
    //     "Test_e_multi_ep",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {32, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     vector<int64_t>(512, 128), // sendCounts 512 elements
    //     vector<int64_t>(512, 128), // recvCounts 512 elements
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     16, 16, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_send_counts_size
    // {
    //     "Test_send_counts_size",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {32, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}, // 32 elements, but epWorldSize=16
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     16, 16, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_BSK_1
    // {
    //     "Test_BSK_1",
    //     {52428800, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_BS_1
    // {
    //     "Test_BS_1",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {52428800, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // BS=52428800
    //     {7168, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {52428800, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_H1
    // {
    //     "Test_H1",
    //     {4096, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND, // H1=65536
    //     {4, 65536, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_H2
    // {
    //     "Test_H2",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2048, 12289}, ge::DT_FLOAT16, ge::FORMAT_ND, // H2=12289
    //     {12289, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,   // mmWeightDim0=12289
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2048, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_N1
    // {
    //     "Test_N1",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND, // N1=65536
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_H_1 (H1 and gmmWeightDim1 mismatch)
    // {
    //     "Test_H_1",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // H1=7168
    //     {4, 7169, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmWeightDim1=7169
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_H_3 (H2 and mmWeightDim0 mismatch)
    // {
    //     "Test_H_3",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2048, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // H2=7168
    //     {7169, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,   // mmWeightDim0=7169
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2048, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_H_4 (same as Test_H1)
    // {
    //     "Test_H_4",
    //     {4096, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 65536, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 65536}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_send_counts_0
    // {
    //     "Test_send_counts_0",
    //     {16386, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // BSK=16386
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_recv_counts_0
    // {
    //     "Test_recv_counts_0",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {8193, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // BS=8193
    //     {7168, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {8193, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {16386, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // A=16386
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // Test_recv_counts_1
    // {
    //     "Test_recv_counts_1",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {8193, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // BS=8193
    //     {7168, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {8193, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {16386, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // A=16386
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestH4 (from separate test)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestH4",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096, 7169}, ge::DT_FLOAT16, ge::FORMAT_ND, // permuteOutShape mismatch
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestA1
    // {
    //     "GroupedMatmulAlltoAllvTilingTestA1",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4097, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // A=4097 mismatch
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestBS1
    // {
    //     "GroupedMatmulAlltoAllvTilingTestBS1",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2048, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {7168, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2047, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, // BS mismatch
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestDim1 (3D gmmX)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestDim1",
    //     {2, 4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // 3D shape
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestDim2 (2D gmmWeight)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestDim2",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // 2D shape
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestDim3 (normal case but should fail)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestDim3",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestDim5 (3D mmX)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestDim5",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2, 2048, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND, // 3D shape
    //     {7168, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2047, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, // mismatch
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestDim6 (3D mmWeight)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestDim6",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2048, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2, 7168, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, // 3D shape
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2047, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, // mismatch
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestDim7 (same as Dim6)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestDim7",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2048, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2, 7168, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {2047, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestDim10 (1D permuteOut)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestDim10",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // 1D shape
    //     false, false, false, false,
    //     false, false, true, true,
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    
    // // GroupedMatmulAlltoAllvTilingTestTransMmWeight1 (normal case)
    // {
    //     "GroupedMatmulAlltoAllvTilingTestTransMmWeight1",
    //     {4096, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    //      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
    //     {4096, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     {}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     false, false, false, false,
    //     false, false, true, true, // permute_out_flag=true
    //     8, 8, 0,
    //     ge::GRAPH_FAILED, 0
    // },
    // hif8 全量化
    // 正常测试用例
    {
        "gmmalltoallv_hif8_quant_normal",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmXScale
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmWeightScale
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmX
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmWeight
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // mmXScale
        {1}, ge::DT_FLOAT, ge::FORMAT_ND, // mmWeightScale
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024}, // sendCounts
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024}, // recvCounts
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmYShape
        {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // mmYShape
        1, 1, 1, 1, // gmmXQuantMode gmmWeightQuantMode mmXQuantMode mmWeightQuantMode
        false, false, // gmmTrans mmTrans
        2, 2, 0, // worldSize epWorldSize graphType
        ge::GRAPH_SUCCESS, 137 // expectedStatus expectTilingKey
    },

    // 异常测试用例
    // 数据类型非法异常
    {
        "gmmalltoallv_hif8_quant_exception_gmmx_datatype_invalid",
        {8192, 7168}, ge::DT_INT8, ge::FORMAT_ND, // gmmX数据类型非法
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmweight_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_INT8, ge::FORMAT_ND, // gmmWeight数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmy_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_INT8, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY数据类型非法
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmxscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // gmmXScale数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_gmmweightscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // gmmWeightScale数据类型非法
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmx_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_INT8, ge::FORMAT_ND, // mmX数据类型非法
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmweight_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_INT8, ge::FORMAT_ND, // mmWeight数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmxscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // mmXScale数据类型非法
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },
    {
        "gmmalltoallv_hif8_quant_exception_mmweightscale_datatype_invalid",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_INT8, ge::FORMAT_ND, // mmWeightScale数据类型非法
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        1, 1, 1, 1, false, false, 2, 2, 0, ge::GRAPH_FAILED, 0
    },

    // // 数据格式非法异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmx_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_FRACTAL_NZ, // gmmX数据格式非法
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweight_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_FRACTAL_NZ, // gmmWeight数据格式非法
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmy_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY数据格式非法
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmxscale_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, // gmmXScale数据格式非法
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweightscale_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, // gmmWeightScale数据格式非法
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mmx_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_FRACTAL_NZ, // mmX数据格式非法
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mmweight_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_FRACTAL_NZ, // mmWeight数据格式非法
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mmy_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, // mmY数据格式非法
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mmxscale_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, // mmXScale数据格式非法
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mmweightscale_format_invalid",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, // mmWeightScale数据格式非法
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweight_not_3d",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight维度不为3D
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmy_not_2d",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096,1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY维度不为2D
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmxscale_not_1d",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1,1}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmXScale维度不为1D
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweightscale_not_1d",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1,1}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmWeightScale维度不为1D
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmx_not_2d",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168, 1}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mm不为空时mmX维度不为2D
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmweight_not_2d",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096, 1}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mm不为空时mmWeight维度不为2D
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmy_not_1d",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096,1}, ge::DT_FLOAT16, ge::FORMAT_ND, // mm不为空时mmY维度不为1D
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // 空tensor异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmx_empty_dim0",
    //     {0, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX为空tensor，第一维度为0
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmx_empty_dim1",
    //     {8192, 0}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX为空tensor，第二维度为0
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweight_empty_dim0",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {0, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight为空tensor，第一维度为0
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweight_empty_dim1",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 0, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight为空tensor，第二维度为0
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweight_empty_dim2",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 0}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmWeight为空tensor，第三维度为0
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmy_empty_dim0",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {0,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY为空tensor，第一维为0
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmy_empty_dim1",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,0}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // gmmY为空tensor，第二维为0
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // K轴不匹配异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmx_gmmweight_k_mismatch_no_transpose",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 8192, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX和gmmWeight的k轴不匹配（无转置），gmmX的K=7168，gmmWeight的K=8192
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmx_gmmweight_k_mismatch_transpose",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // gmmX和gmmWeight的k轴不匹配（转置），gmmX的K=7168，gmmWeight转置后K=4096
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // trans_gmm_weight_flag=true
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmx_mmweight_k_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {8192, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mm不为空时mmX和mmWeight的K轴不匹配，mmX的K=7168，mmWeight的K=8192
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // mmX/mmWeight/mmY不同时为空tensor
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_tensors_not_all_null",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmX为空
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmWeight非空
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // mmY非空
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // // mmX/mmWeight/mmY不同时为非空tensor
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_tensors_not_all_nonnull",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmX非空
    //     {}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mmWeight为空
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {}, ge::DT_FLOAT16, ge::FORMAT_ND, // mmY为空
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // mm不为空时mmX的第一维与mmWeight的第二维不匹配
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmx_dim0_mmweight_dim1_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 8192}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // mm不为空时mmX的第一维(4096)与mmWeight的第二维(8192)不匹配
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // // mm不为空时mmY与mmX的第一维不匹配
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmy_mmx_dim0_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, // mmY(8192)与mmX的第一维(4096)不匹配
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // scale的shape不匹配
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmxscale_shape_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {2}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmXScale的shape不匹配，应为[1]
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweightscale_shape_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmWeightScale的shape不匹配，应为[1]
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmxscale_shape_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {2}, ge::DT_FLOAT, ge::FORMAT_ND, // mm不为空时，mmXScale的shape不匹配，应为[1]
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmweightscale_shape_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {2}, ge::DT_FLOAT, ge::FORMAT_ND, // mm不为空时，mmWeightScale的shape不匹配，应为[1]
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // QuantMode异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmx_quantmode_not_1",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     2, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // gmmXQuantMode取值不为1(pertensor)
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweight_quantmode_not_1",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 2, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // gmmWeightQuantMode取值不为1(pertensor)
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmx_quantmode_not_1",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 2, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // mm不为空时，mmXQuantMode取值不为1(pertensor)
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mm_not_null_mmweight_quantmode_not_1",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 2, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // mm不为空时，mmWeightQuantMode取值不为1(pertensor)
    // },

    // // group相关异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_group_length_exceed_128",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024,1024,1024,1024,1024,1024,1024,1024,1024}, // 9个元素，超过8
    //     {1024,1024,1024,1024,1024,1024,1024,1024,1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // group长度超过128（这里sendCounts长度超过8）
    // },

    // // trans flag与shape不匹配
    // {
    //     "gmmalltoallv_hif8_quant_exception_transgmmweight_value_shape_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // shape为[4, 4096, 7168]
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // transGmmWeight=false但shape不匹配（应该是[4,7168,4096]）
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_transmmweight_value_shape_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // shape为[4096,7168]
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0 // transMmWeight=false但shape不匹配（应该是[7168,4096]）
    // },

    // // Scale与QuantMode不匹配
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmxscale_gmmxquantmode_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {8192}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmXScale的shape为[8192]，但QuantMode为1(pertensor，应为[1])
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_gmmweightscale_gmmweightquantmode_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4}, ge::DT_FLOAT, ge::FORMAT_ND, // gmmWeightScale的shape为[4]，但QuantMode为1(pertensor，应为[1])
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mmxscale_mmxquantmode_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4096}, ge::DT_FLOAT, ge::FORMAT_ND, // mmXScale的shape为[4096]，但QuantMode为1(pertensor，应为[1])
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_mmweightscale_mmweightquantmode_mismatch",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {7168}, ge::DT_FLOAT, ge::FORMAT_ND, // mmWeightScale的shape为[7168]，但QuantMode为1(pertensor，应为[1])
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // epWorldSize异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_epworldsize_not_equal_256",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 4, 0, ge::GRAPH_FAILED, 0 // e*epWorldSize不等于256 (2*4=8)
    // },

    // // TopK异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_topk_greater_than_8",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {9, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND, // TopK的值大于8 (第一维=9)
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },

    // // sendCounts和recvCounts总和异常
    // {
    //     "gmmalltoallv_hif8_quant_exception_sendcounts_sum_not_equal_gmmx_dim0",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024,1024,1024,1024,1024,1024,1024,1025}, // 总和=8193，不等于gmmX的第一维大小8192
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // },
    // {
    //     "gmmalltoallv_hif8_quant_exception_recvcounts_sum_not_equal_gmmx_dim0",
    //     {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1}, ge::DT_FLOAT, ge::FORMAT_ND,
    //     {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    //     {1024,1024,1024,1024,1024,1024,1024,1025}, // 总和=8193，不等于gmmX的第一维大小8192
    //     {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
    //     {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    //     1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_FAILED, 0
    // }
};

class GroupedMatMulAlltoAllvTilingTest : public ::testing::TestWithParam<GroupedMatMulAlltoAllvTilingTestParam> {
protected:
    static void SetUpTestCase() {
        cout << "GroupedMatMulAlltoAllvTilingTest SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "GroupedMatMulAlltoAllvTilingTest TearDown" << endl;
    }
    void SetUp() override {
        const auto& param = GetParam();
        cout << "Running test case: " << param.caseName << endl;
    }
};

INSTANTIATE_TEST_SUITE_P(
    GroupedMatMulAlltoAllvTilingTestSuite,
    GroupedMatMulAlltoAllvTilingTest,
    testing::ValuesIn(groupedMatMulAlltoAllvTilingTestParam),
    [](const testing::TestParamInfo<GroupedMatMulAlltoAllvTilingTestParam>& info) {
        return info.param.caseName;
    }
);

TEST_P(GroupedMatMulAlltoAllvTilingTest, test_grouped_quant_mat_mul_allto_allv_tiling) {
    const auto& param = GetParam();

    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    size_t tilingDataSize = sizeof(QuantGmmA2avTilingData);

    gert::StorageShape mmXStorageShape;
    if (param.mmXShape.size() > 0 && param.mmXShape[0] > 0) {
        mmXStorageShape = {{param.mmXShape[0], param.mmXShape[1]}, {param.mmXShape[0], param.mmXShape[1]}};
    } else {
        mmXStorageShape = {};
    }
    
    gert::StorageShape mmWeightStorageShape;
    if (param.mmWeightShape.size() > 0 && param.mmWeightShape[0] > 0) {
        mmWeightStorageShape = {{param.mmWeightShape[0], param.mmWeightShape[1]}, {param.mmWeightShape[0], param.mmWeightShape[1]}};
    } else {
        mmWeightStorageShape = {};
    }
    
    gert::StorageShape mmYStorageShape;
    if (param.mmYShape.size() > 0 && param.mmYShape[0] > 0) {
        mmYStorageShape = {{param.mmYShape[0], param.mmYShape[1]}, {param.mmYShape[0], param.mmYShape[1]}};
    } else {
        mmYStorageShape = {};
    }

    // gmmXScaleStorageShape
    gert::StorageShape gmmXScaleStorageShape;
    if (param.gmmXScaleShape.size() > 0 && param.gmmXScaleShape[0] > 0) {
        gmmXScaleStorageShape = {{param.gmmXScaleShape[0]}, {param.gmmXScaleShape[0]}};
    } else {
        gmmXScaleStorageShape = {};
    }

    // gmmWeightScaleStorageShape
    gert::StorageShape gmmWeightScaleStorageShape;
    if (param.gmmWeightScaleShape.size() > 0 && param.gmmWeightScaleShape[0] > 0) {
        gmmWeightScaleStorageShape = {{param.gmmWeightScaleShape[0]}, {param.gmmWeightScaleShape[0]}};
    } else {
        gmmWeightScaleStorageShape = {};
    }

    // mmXScaleStorageShape
    gert::StorageShape mmXScaleStorageShape;
    if (param.mmXScaleShape.size() > 0 && param.mmXScaleShape[0] > 0) {
        mmXScaleStorageShape = {{param.mmXScaleShape[0]}, {param.mmXScaleShape[0]}};
    } else {
        mmXScaleStorageShape = {};
    }

    // mmWeightScaleStorageShape
    gert::StorageShape mmWeightScaleStorageShape;
    if (param.mmWeightScaleShape.size() > 0 && param.mmWeightScaleShape[0] > 0) {
        mmWeightScaleStorageShape = {{param.mmWeightScaleShape[0]}, {param.mmWeightScaleShape[0]}};
    } else {
        mmWeightScaleStorageShape = {};
    }

    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{param.gmmXShape[0], param.gmmXShape[1]},{param.gmmXShape[0], param.gmmXShape[1]}}, param.gmmXDataType, param.gmmXFormat},
            {{{param.gmmWeightShape[0], param.gmmWeightShape[1],param.gmmWeightShape[2]},{param.gmmWeightShape[0], param.gmmWeightShape[1],param.gmmWeightShape[2]}},
                 param.gmmWeightDataType, param.gmmWeightFormat},
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // send_counts_tensor
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // recv_counts_tensor
            {mmXStorageShape, param.mmXDataType, param.mmXFormat},
            {mmWeightStorageShape,param.mmWeightDataType, param.mmWeightFormat},
            {gmmXScaleStorageShape, param.gmmXScaleDataType, param.gmmXScaleFormat},
            {gmmWeightScaleStorageShape, param.gmmWeightScaleDataType, param.gmmWeightScaleFormat},
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmXOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // gmmWeightOffset
            {mmXScaleStorageShape, param.mmXScaleDataType, param.mmXScaleFormat},
            {mmWeightScaleStorageShape, param.mmWeightScaleDataType, param.mmWeightScaleFormat},
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // mmXOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // mmWeightOffset
            {{}, ge::DT_INT64, ge::FORMAT_ND}  // commQunatScale
        },
        {
            {{{param.gmmYShape[0], param.gmmYShape[1]},{param.gmmYShape[0], param.gmmYShape[1]}}, param.gmmYDataType, param.gmmYFormat},
            {mmYStorageShape, param.mmYDataType, param.mmYFormat}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.sendCounts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.recvCounts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_gmm_weight_flag)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_mm_weight_flag)},
            {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_x_quant_mode)},
            {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_weight_quant_mode)},
            {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_x_quant_mode)},
            {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_weight_quant_mode)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"gmm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"mm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo,
        "Ascend950",
        coreNum,
        ubSize,
        tilingDataSize
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectedStatus, param.expectTilingKey);
}
