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
#include <gtest/gtest.h>

#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "grouped_matmul_finalize_routing_host_ut_param.h"

namespace GroupedMatmulFinalizeRoutingUT {

class GroupedMatmulFinalizeRoutingInferShapeTest
    : public testing::TestWithParam<GroupedMatmulFinalizeRoutingInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRouting Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRouting Proto Test TearDown" << std::endl;
    }
};

TEST_P(GroupedMatmulFinalizeRoutingInferShapeTest, param)
{
    auto param = GetParam();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1) inputTensorDesc.emplace_back(param.x);
    if (param.inputInstance[1] == 1) inputTensorDesc.emplace_back(param.w);
    if (param.inputInstance[2] == 1) inputTensorDesc.emplace_back(param.scale);
    if (param.inputInstance[3] == 1) inputTensorDesc.emplace_back(param.bias);
    if (param.inputInstance[4] == 1) inputTensorDesc.emplace_back(param.pertoken_scale);
    if (param.inputInstance[5] == 1) inputTensorDesc.emplace_back(param.group_list);
    if (param.inputInstance[6] == 1) inputTensorDesc.emplace_back(param.shared_input);
    if (param.inputInstance[7] == 1) inputTensorDesc.emplace_back(param.logit);
    if (param.inputInstance[8] == 1) inputTensorDesc.emplace_back(param.row_index);

    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulFinalizeRouting",
        inputTensorDesc,
        {
            param.y,
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.dtype)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(param.shared_input_weight)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.shared_input_offset)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transpose_x)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transpose_w)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.output_bs)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_list_type)},
        },
        param.inputInstance, param.outputInstance);

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    GroupedMatmulFinalizeRouting,
    GroupedMatmulFinalizeRoutingInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<GroupedMatmulFinalizeRoutingInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GroupedMatmulFinalizeRoutingInferShapeUtParam>);

}  // namespace GroupedMatmulFinalizeRoutingUT
