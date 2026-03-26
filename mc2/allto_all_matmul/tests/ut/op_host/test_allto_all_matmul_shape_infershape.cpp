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
#include "mc2_infer_shape_case_executor.h"

namespace AlltoAllMatmulUT {

class InferShapeTest : public testing::TestWithParam<AlltoAllMatmulInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmul InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmul InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(InferShapeTest, param)
{
    auto param = GetParam();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1) inputTensorDesc.emplace_back(param.x1);
    if (param.inputInstance[1] == 1) inputTensorDesc.emplace_back(param.x2);

    gert::InfershapeContextPara inferShapeContextPara(
        "AlltoAllMatmul",
        inputTensorDesc,
        {
            param.y,
            param.y  // alltoallout output (same as y placeholder)
        },
        {
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.world_size)},
            {"allto_all_axes", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.allto_all_axes)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.y_dtype))},
            {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1_quant_mode)},
            {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2_quant_mode)},
            {"trans_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_x1)},
            {"trans_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_x2)},
            {"alltoall_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.alltoall_out_flag)}
        },
        param.inputInstance, param.outputInstance
    );

    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"world_size", param.world_size}
    };
    Mc2ExecuteTestCase(inferShapeContextPara, hcomTopologyMockValues, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllMatmul,
    InferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllMatmulInferShapeUtParam>
);

} // namespace AlltoAllMatmulUT