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

#include "mc2_tiling_case_executor.h"

using namespace std;

struct AlltoAllvGroupedMatMulTilingTestParam {
    string caseName;

    // gmm: input tensor
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

    // mm: input tensor.
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

    std::vector<int64_t> permuteOutShape;
    ge::DataType permuteOutDataType;
    ge::Format permuteOutFormat;

    // Attributes
    bool gmm_x_quant_mode;
    bool gmm_weight_quant_mode;
    bool mm_x_quant_mode;
    bool mm_weight_quant_mode;

    bool trans_gmm_weight_flag;
    bool trans_mm_weight_flag;
    bool permute_out_flag;
    bool mm_out_flag;
    int64_t world_size;
    int64_t ep_world_size;
    int64_t graph_type;

    // Expected result
    ge::graphStatus expectedStatus;

    // Expected tiling key
    uint64_t expectTilingKey;
};

class AlltoAllvGroupedMatMulTilingTest : public ::testing::TestWithParam<AlltoAllvGroupedMatMulTilingTestParam> {
protected:
    static void SetUpTestCase() {
        cout << "AlltoAllvGroupedMatMulTilingTest SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "AlltoAllvGroupedMatMulTilingTest TearDown" << endl;
    }
    void SetUp() override {
        const auto& param = GetParam();
        cout << "Running test case: " << param.caseName << endl;
    }
};

static const vector<AlltoAllvGroupedMatMulTilingTestParam> alltoAllvGroupedMatMulTilingTestParam = {
    {
        "quant_8192_7168_4_7168_4096_hif8_hif8_ND_deepseekv3_alltoallvgmm_ID0001",
        {8192, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {4, 7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4096, 7168}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {7168, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {128,128,128,128,128,128,128,128},
        {128,128,128,128,128,128,128,128},
        {8192,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, {4096,4096}, ge::DT_FLOAT16, ge::FORMAT_ND, 
        {8192,7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
        1, 1, 1, 1, false, false, true, true, 2, 2, 0, ge::GRAPH_SUCCESS, 0 //tilingkey 不明白
    }

};


INSTANTIATE_TEST_SUITE_P(
    AlltoAllvGroupedMatMulTilingTestSuite,
    AlltoAllvGroupedMatMulTilingTest,
    testing::ValuesIn(alltoAllvGroupedMatMulTilingTestParam),
    [](const testing::TestParamInfo<AlltoAllvGroupedMatMulTilingTestParam>& info) {
        return info.param.caseName;
    }
);

TEST_P(AlltoAllvGroupedMatMulTilingTest, test_allto_allv_grouped_quant_mat_mul_tiling) {
    const auto& param = GetParam();

    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{param.gmmXShape[0], param.gmmXShape[1]},{param.gmmXShape[0], param.gmmXShape[1]}}, param.gmmXDataType, param.gmmXFormat},
            {{{param.gmmWeightShape[0], param.gmmWeightShape[1],param.gmmWeightShape[2]},{param.gmmWeightShape[0], param.gmmWeightShape[1],param.gmmWeightShape[2]}},
                 param.gmmWeightDataType, param.gmmWeightFormat},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}, // bias
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // send_counts_tensor
            {{}, ge::DT_INT64, ge::FORMAT_ND}, // recv_counts_tensor
            {{{param.mmXShape[0], param.mmXShape[1]},{param.mmXShape[0], param.mmXShape[1]}}, param.mmXDataType, param.mmXFormat},
            {{{param.mmWeightShape[0], param.mmWeightShape[1]}, {param.mmWeightShape[0], param.mmWeightShape[1]}},param.mmWeightDataType, param.mmWeightFormat},
            {{{param.gmmXScaleShape[0]},{param.gmmXScaleShape[0]}}, param.gmmXScaleDataType, param.gmmXScaleFormat},
            {{{param.gmmWeightScaleShape[0]}, {param.gmmWeightScaleShape[0]}},param.gmmWeightScaleDataType, param.gmmWeightScaleFormat},
            {{{param.mmXScaleShape[0]},{param.mmXScaleShape[0]}}, param.mmXScaleDataType, param.mmXScaleFormat},
            {{{param.mmWeightScaleShape[0]},{param.mmWeightScaleShape[0]}}, param.mmWeightScaleDataType, param.mmWeightScaleFormat}
        },
        {
            {{{param.gmmYShape[0], param.gmmYShape[1]},{param.gmmYShape[0], param.gmmYShape[1]}}, param.gmmYDataType, param.gmmYFormat},
            {{{param.mmYShape[0], param.mmYShape[1]},{param.mmYShape[0], param.mmYShape[1]}}, param.mmYDataType, param.mmYFormat},
            {{{param.permuteOutShape[0],param.permuteOutShape[1]},{param.permuteOutShape[0],param.permuteOutShape[1]}},param.permuteOutDataType, param.permuteOutFormat}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.sendCounts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.recvCounts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_gmm_weight_flag)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_mm_weight_flag)},
            {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.permute_out_flag)},
        },
        &compileInfo,
        "Ascend910_95",
        coreNum,
        ubSize
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectedStatus, param.expectTilingKey);
}