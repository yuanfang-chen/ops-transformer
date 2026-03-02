/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "math/apply_adamw_v3/op_host/arch32/apply_adamw_v3_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

constexpr size_t SYS_WORKSPACE = 16777216;

class ApplyAdamwV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyAdamwV3Tiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ApplyAdamwV3Tiling TearDown" << std::endl;
    }
};

TEST_F(ApplyAdamwV3Tiling, ascend910b_test_tiling_fp32_basic)
{
    optiling::ApplyAdamwV3CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ApplyAdamwV3",
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("amsgrad", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("maximize", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, std::vector<size_t>{SYS_WORKSPACE});
}

TEST_F(ApplyAdamwV3Tiling, ascend910b_test_tiling_fp16_basic)
{
    optiling::ApplyAdamwV3CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ApplyAdamwV3",
        {
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("amsgrad", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("maximize", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, std::vector<size_t>{SYS_WORKSPACE});
}

TEST_F(ApplyAdamwV3Tiling, ascend910b_test_tiling_bf16_basic)
{
    optiling::ApplyAdamwV3CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ApplyAdamwV3",
        {
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("amsgrad", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("maximize", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, std::vector<size_t>{SYS_WORKSPACE});
}

TEST_F(ApplyAdamwV3Tiling, ascend910b_test_tiling_fp32_amsgrad)
{
    optiling::ApplyAdamwV3CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ApplyAdamwV3",
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("amsgrad", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("maximize", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 13;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, std::vector<size_t>{SYS_WORKSPACE});
}

TEST_F(ApplyAdamwV3Tiling, ascend910b_test_tiling_fp32_maximize)
{
    optiling::ApplyAdamwV3CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ApplyAdamwV3",
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("amsgrad", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("maximize", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, std::vector<size_t>{SYS_WORKSPACE});
}

TEST_F(ApplyAdamwV3Tiling, ascend910b_test_tiling_fp32_multi_dim)
{
    optiling::ApplyAdamwV3CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "ApplyAdamwV3",
        {
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("amsgrad", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("maximize", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, std::vector<size_t>{SYS_WORKSPACE});
}
