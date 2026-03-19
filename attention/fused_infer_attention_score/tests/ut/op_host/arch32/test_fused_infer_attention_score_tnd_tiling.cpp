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

class FusedInferAttentionScoreTndTilingArch32 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedInferAttentionScoreTndTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedInferAttentionScoreTndTiling TearDown" << std::endl;
    }
};

TEST_F(FusedInferAttentionScoreTndTilingArch32, TND_MLA_FP16_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417382415922f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
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
    int64_t expectTilingKey = 105000000030000002;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTndTilingArch32, TND_NTD_MLA_FP16_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{64, 12, 512}, {64, 12, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666664f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND_NTD")},
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
    int64_t expectTilingKey = 105000000030000002;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTndTilingArch32, NTD_MHA_FP16_HeadDim128)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{5, 6, 128}, {5, 6, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 768, 128}, {5, 768, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 768, 128}, {5, 768, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2, 4, 6}, {2, 4, 6}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{256, 512, 768}, {256, 512, 768}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
        {{{{5, 6, 128}, {5, 6, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
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
    int64_t expectTilingKey = 103000000050000005;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTndTilingArch32, TND_BF16_HeadDim128_GQA)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{1024, 16, 128}, {1024, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 4, 128}, {2048, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 4, 128}, {2048, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{256, 512, 768, 1024}, {256, 512, 768, 1024}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{512, 1024, 1536, 2048}, {512, 1024, 1536, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
        {{{{1024, 16, 128}, {1024, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388347648318f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
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
    int64_t expectTilingKey = 5000000000000200200;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTndTilingArch32, TND_FP16_HeadDim64_SparseMode3)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{512, 8, 64}, {512, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 8, 64}, {1024, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 8, 64}, {1024, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{128, 256, 384, 512}, {128, 256, 384, 512}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{256, 512, 768, 1024}, {256, 512, 768, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
        {{{{512, 8, 64}, {512, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTndTilingArch32, NTD_TND_FP16_HeadDim128)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{8, 256, 128}, {8, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 512, 128}, {8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 512, 128}, {8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64, 128, 192, 256}, {64, 128, 192, 256}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{128, 256, 384, 512}, {128, 256, 384, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
        {{{{256, 8, 128}, {256, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388347648318f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD_TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
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
    int64_t expectTilingKey = 103000000050000005;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTndTilingArch32, TND_FP16_HeadDim256_SparseMode4)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{256, 4, 256}, {256, 4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 4, 256}, {512, 4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 4, 256}, {512, 4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64, 128, 192, 256}, {64, 128, 192, 256}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{128, 256, 384, 512}, {128, 256, 384, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
        {{{{256, 4, 256}, {256, 4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
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
    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(FusedInferAttentionScoreTndTilingArch32, TND_MLA_BF16_HeadDim512)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{12, 64, 512}, {12, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 1, 512}, {4096, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{3, 6, 9, 12}, {3, 6, 9, 12}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1024, 2048, 3072, 4096}, {1024, 2048, 3072, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{12, 64, 64}, {12, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 1, 64}, {4096, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{12, 64, 512}, {12, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417382415922f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
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
    int64_t expectTilingKey = 105000000030022222;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}