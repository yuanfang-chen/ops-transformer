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

class FusedInferAttentionScoreMlaTilingArch32 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedInferAttentionScoreMlaTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedInferAttentionScoreMlaTiling TearDown" << std::endl;
    }
};

TEST_F(FusedInferAttentionScoreMlaTilingArch32, MLA_BNSD_FP16_gSize1_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 1, 512, 512}, {4, 1, 512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1024, 512}, {4, 1, 1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1024, 512}, {4, 1, 1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {{{4, 1, 512, 64}, {4, 1, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1024, 64}, {4, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 1, 512, 512}, {4, 1, 512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824159f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
    int64_t expectTilingKey = 105000000000100000;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreMlaTilingArch32, MLA_BNSD_BF16_gSize2_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 2, 512, 512}, {4, 2, 512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1, 1024, 512}, {4, 1, 1024, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1, 1024, 512}, {4, 1, 1024, 512}}, ge::DT_BF16, ge::FORMAT_ND},
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
            {{{4, 2, 512, 64}, {4, 2, 512, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1, 1024, 64}, {4, 1, 1024, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 2, 512, 512}, {4, 2, 512, 512}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824159f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
    int64_t expectTilingKey = 105000000000022220;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreMlaTilingArch32, MLA_BSND_FP16_gSize4_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 512, 4, 512}, {4, 512, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 1, 512}, {4, 1024, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 1, 512}, {4, 1024, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {{{4, 512, 4, 64}, {4, 512, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 512, 4, 512}, {4, 512, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824159f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
    int64_t expectTilingKey = 105000000010000001;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreMlaTilingArch32, MLA_BSH_FP16_gSize8_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 512, 4096}, {4, 512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 512}, {4, 1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 512}, {4, 1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {{{4, 512, 512}, {4, 512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1024, 64}, {4, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 512, 4096}, {4, 512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824159f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
    int64_t expectTilingKey = 105000000010100001;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreMlaTilingArch32, MLA_BSND_BF16_gSize16_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 512, 16, 512}, {4, 512, 16, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 1, 512}, {4, 1024, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 1, 512}, {4, 1024, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
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
            {{{4, 512, 16, 64}, {4, 512, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024, 1, 64}, {4, 1024, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 512, 16, 512}, {4, 512, 16, 512}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824159f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
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
    int64_t expectTilingKey = 105000000010122221;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}