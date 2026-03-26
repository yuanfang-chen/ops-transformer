/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "test_allto_all_matmul_host_ut_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AlltoAllMatmulUT {

class InferDataTypeTest : public testing::TestWithParam<AlltoAllMatmulInferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmul InferDataTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmul InferDataTypeTest TearDown" << std::endl;
    }
};

TEST_P(InferDataTypeTest, param)
{
    auto param = GetParam();
    std::vector<void*> inputDataTypes;
    if (param.inputInstance[0] == 1) inputDataTypes.emplace_back(&param.x1);
    if (param.inputInstance[1] == 1) inputDataTypes.emplace_back(&param.x2);

    auto contextHolder = gert::InferDataTypeContextFaker()
        .SetOpType("AlltoAllMatmul")
        .IrInstanceNum(param.inputInstance, param.outputInstance)
        .InputDataTypes(inputDataTypes)
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.world_size)},
            {"allto_all_axes", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.allto_all_axes)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.y_dtype))},
            {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1_quant_mode)},
            {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2_quant_mode)},
            {"trans_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_x1)},
            {"trans_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_x2)},
            {"alltoall_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.alltoall_out_flag)}
        })
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllMatmul")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.y);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), param.alltoallout);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllMatmul,
    InferDataTypeTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulInferDataTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllMatmulInferDataTypeUtParam>
);

} // namespace AlltoAllMatmulUT