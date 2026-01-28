/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "matmul_all_reduce_host_ut_param.h"
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace matmul_all_reduce_ut {

// inferShape 用例 =====================================================================================================
class InferShapeUT : public testing::TestWithParam<MatmulAllReduceInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce InferShapeUT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce InferShapeUT TearDown" << std::endl;
    }
};

std::vector<MatmulAllReduceInferShapeUtParam> casesInferShape {
    // 正确用例
    {"2dim", ID({{32, 64}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), ID({{64, 128}, {}}, ge::DT_INT32, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, ID({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, ge::DT_UNDEFINED, 0, 8, ge::GRAPH_SUCCESS, {{32, 128}}},
    {"3dim", ID({{4, 8, 64}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), ID({{64, 128}, {}}, ge::DT_INT32, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, ID({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, ge::DT_UNDEFINED, 0, 8, ge::GRAPH_SUCCESS, {{4, 8, 128}}},
    {"invalid_zero_k", ID({{32, 0}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), ID({{0, 128}, {}}, ge::DT_INT32, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, ID({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, ge::DT_UNDEFINED, 0, 8, ge::GRAPH_SUCCESS, {{32, 128}}},
    {"3dim_quant_v4", ID({{4, 8, 64}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), ID({{64, 128}, {}}, ge::DT_INT32, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, ID({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, ge::DT_UNDEFINED, 0, 8, ge::GRAPH_SUCCESS, {{4, 8, 128}}},
    {"add_rms_norm", ID({{4, 8, 64}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), ID({{64, 128}, {}}, ge::DT_INT32, ge::FORMAT_ND), ID({{128}, {}}, ge::DT_INT32, ge::FORMAT_ND), ID({{4, 8, 128}, {}}, ge::DT_INT32, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, ID({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, ge::DT_UNDEFINED, 0, 8, ge::GRAPH_SUCCESS, {{4, 8, 128}}},
    // 失败用例
    {"invalid_k", ID({{32, 8}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), ID({{64, 128}, {}}, ge::DT_INT32, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, ID({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, ge::DT_UNDEFINED, 0, 8, ge::GRAPH_FAILED}
};

TEST_P(InferShapeUT, paramTest)
{
    auto param = GetParam();
    std::vector<ID> inputTensorDesc;
    if (param.inputInstanceNum[0] == 1) inputTensorDesc.emplace_back(param.x1);
    if (param.inputInstanceNum[1] == 1) inputTensorDesc.emplace_back(param.x2);
    if (param.inputInstanceNum[2] == 1) inputTensorDesc.emplace_back(param.bias);
    if (param.inputInstanceNum[3] == 1) inputTensorDesc.emplace_back(param.x3);
    if (param.inputInstanceNum[4] == 1) inputTensorDesc.emplace_back(param.antiquant_scale);
    if (param.inputInstanceNum[5] == 1) inputTensorDesc.emplace_back(param.antiquant_offset);
    if (param.inputInstanceNum[6] == 1) inputTensorDesc.emplace_back(param.dequant_scale);
    if (param.inputInstanceNum[7] == 1) inputTensorDesc.emplace_back(param.pertoken_scale);
    if (param.inputInstanceNum[8] == 1) inputTensorDesc.emplace_back(param.comm_quant_scale_1);
    if (param.inputInstanceNum[9] == 1) inputTensorDesc.emplace_back(param.comm_quant_scale_2);
    gert::InfershapeContextPara inferShapeContextPara(
        "MatmulAllReduce",
        inputTensorDesc,
        {
            param.y
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_quant_mode)}
        },
        param.inputInstanceNum, param.outputInstanceNum
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", param.ranksize}
    };
    Mc2ExecuteTestCase(inferShapeContextPara, hcomTopologyMockValues, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    InferShapeUT,
    testing::ValuesIn(casesInferShape),
    GetCaseInfoString<MatmulAllReduceInferShapeUtParam>
);

// inferDataType 用例 ==================================================================================================
class InferDataTypeUT : public testing::TestWithParam<MatmulAllReduceInferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce InferDataTypeUT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce InferDataTypeUT TearDown" << std::endl;
    }
};

std::vector<MatmulAllReduceInferDataTypeUtParam> casesInferDataType {
    {"basic", ge::DT_FLOAT16, ge::DT_FLOAT16, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, "group", "sum", false, false, 0, 0, 0, ge::DT_FLOAT16, 0, ge::GRAPH_SUCCESS, ge::DT_FLOAT16}
};

TEST_P(InferDataTypeUT, paramTest)
{
    auto param = GetParam();
    ge::DataType x1 = ge::DT_FLOAT16;
    ge::DataType x2 = ge::DT_FLOAT16;
    std::vector<void*> inputDataTypes;
    if (param.inputInstanceNum[0] == 1) inputDataTypes.emplace_back(&param.x1);
    if (param.inputInstanceNum[1] == 1) inputDataTypes.emplace_back(&param.x2);
    if (param.inputInstanceNum[2] == 1) inputDataTypes.emplace_back(&param.bias);
    if (param.inputInstanceNum[3] == 1) inputDataTypes.emplace_back(&param.x3);
    if (param.inputInstanceNum[4] == 1) inputDataTypes.emplace_back(&param.antiquant_scale);
    if (param.inputInstanceNum[5] == 1) inputDataTypes.emplace_back(&param.antiquant_offset);
    if (param.inputInstanceNum[6] == 1) inputDataTypes.emplace_back(&param.dequant_scale);
    if (param.inputInstanceNum[7] == 1) inputDataTypes.emplace_back(&param.pertoken_scale);
    if (param.inputInstanceNum[8] == 1) inputDataTypes.emplace_back(&param.comm_quant_scale_1);
    if (param.inputInstanceNum[9] == 1) inputDataTypes.emplace_back(&param.comm_quant_scale_2);

    auto contextHolder = gert::InferDataTypeContextFaker()
        .SetOpType("MatmulAllReduce")
        .IrInstanceNum(param.inputInstanceNum, param.outputInstanceNum)
        .InputDataTypes(inputDataTypes)
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_quant_mode)}
        })
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MatmulAllReduce")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.y);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    InferDataTypeUT,
    testing::ValuesIn(casesInferDataType),
    GetCaseInfoString<MatmulAllReduceInferDataTypeUtParam>
);

} // namespace matmul_all_reduce_ut
