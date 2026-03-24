/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "grouped_matmul_infershape_host_ut_param.h"

#define private public
#include "platform/platform_info.h"

namespace GroupedMatmulUT {

namespace {
void SetSocVersionIfNeeded(const std::string &socVersion)
{
    if (socVersion.empty()) {
        return;
    }
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = socVersion;
    platformInfo.str_info.short_soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}
} // namespace

class GroupedMatmulInferShapeCsvTest : public testing::TestWithParam<GroupedMatmulInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulInferShapeCsvTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulInferShapeCsvTest TearDown" << std::endl;
    }
};

TEST_P(GroupedMatmulInferShapeCsvTest, param)
{
    auto param = GetParam();
    SetSocVersionIfNeeded(param.soc_version);

    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1) inputTensorDesc.emplace_back(param.x);
    if (param.inputInstance[1] == 1) inputTensorDesc.emplace_back(param.weight);
    if (param.inputInstance[2] == 1) inputTensorDesc.emplace_back(param.bias);
    if (param.inputInstance[3] == 1) inputTensorDesc.emplace_back(param.scale);
    if (param.inputInstance[4] == 1) inputTensorDesc.emplace_back(param.offset);
    if (param.inputInstance[5] == 1) inputTensorDesc.emplace_back(param.antiquant_scale);
    if (param.inputInstance[6] == 1) inputTensorDesc.emplace_back(param.antiquant_offset);
    if (param.inputInstance[7] == 1) inputTensorDesc.emplace_back(param.group_list);
    if (param.inputInstance[8] == 1) inputTensorDesc.emplace_back(param.pertoken_scale);

    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmul",
        inputTensorDesc,
        {
            param.y,
        },
        {
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.split_item)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.dtype)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transpose_weight)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transpose_x)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_type)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_list_type)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.act_type)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        param.inputInstance, param.outputInstance);

    ExecuteTestCase(infershapeContextPara, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    GroupedMatmulInferShapeCsv,
    GroupedMatmulInferShapeCsvTest,
    testing::ValuesIn(GetCasesFromCsv<GroupedMatmulInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GroupedMatmulInferShapeUtParam>);

} // namespace GroupedMatmulUT

