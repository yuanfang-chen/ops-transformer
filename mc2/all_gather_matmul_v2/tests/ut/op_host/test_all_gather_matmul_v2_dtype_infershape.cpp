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
#include "all_gather_matmul_v2_host_ut_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AllGatherMatmulV2UT {

class AllGatherMatmulV2InferDataTypeTest : public testing::TestWithParam<AllGatherMatmulV2InferDataTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2 InferDataTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2 InferDataTypeTest TearDown" << std::endl;
    }
};

TEST_P(AllGatherMatmulV2InferDataTypeTest, param)
{
    auto param = GetParam();
    std::vector<void*> inputDataTypes;
    if (param.inputInstance[0] == 1) inputDataTypes.emplace_back(&param.x1);
    if (param.inputInstance[1] == 1) inputDataTypes.emplace_back(&param.x2);
    if (param.inputInstance[2] == 1) inputDataTypes.emplace_back(&param.bias);
    if (param.inputInstance[3] == 1) inputDataTypes.emplace_back(&param.x1Scale);
    if (param.inputInstance[4] == 1) inputDataTypes.emplace_back(&param.x2Scale);
    if (param.inputInstance[5] == 1) inputDataTypes.emplace_back(&param.quantScale);

    auto contextHolder = gert::InferDataTypeContextFaker()
        .SetOpType("AllGatherMatmulV2")
        .IrInstanceNum(param.inputInstance, param.outputInstance)
        .InputDataTypes(inputDataTypes)
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.is_gather_out)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_amax_out)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(param.y_dtype))}
        })
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmulV2")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.y);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), param.gatherOut);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), param.amaxOut);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV2,
    AllGatherMatmulV2InferDataTypeTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV2InferDataTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulV2InferDataTypeUtParam>
);

} // namespace AllGatherMatmulV2UT
