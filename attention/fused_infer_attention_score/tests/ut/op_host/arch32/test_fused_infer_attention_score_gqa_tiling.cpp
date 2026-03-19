/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/fused_infer_attention_score_tiling_compile_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

extern std::string Fia_tiling_A2SocInfo;

class FusedInferAttentionScoreGqaTilingArch32 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedInferAttentionScoreGqaTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedInferAttentionScoreGqaTiling TearDown" << std::endl;
    }
};

TEST_F(FusedInferAttentionScoreGqaTilingArch32, GQA_HighPerf_BNSD_FP16_HeadDim128)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 512, 128}, {4, 16, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 4, 1024, 128}, {4, 4, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 4, 1024, 128}, {4, 4, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 16, 512, 128}, {4, 16, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388347648318f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 103000000000100000;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreGqaTilingArch32, GQA_HighPerf_BSND_BF16_HeadDim64)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 512, 16, 64}, {4, 512, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 4, 64}, {4, 1024, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 4, 64}, {4, 1024, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 512, 16, 64}, {4, 512, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 103000000010122221;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreGqaTilingArch32, GQA_HighPerf_BNSD_FP16_HeadDim192)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 512, 192}, {4, 16, 512, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 4, 1024, 192}, {4, 4, 1024, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 4, 1024, 128}, {4, 4, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 16, 512, 128}, {4, 16, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.072168783648703f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 103000000000100000;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}