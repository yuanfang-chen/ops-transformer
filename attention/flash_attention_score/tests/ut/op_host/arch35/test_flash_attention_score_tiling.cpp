/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/flash_attention_score_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FlashAttentionScoreTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreTiling TearDown" << std::endl;
    }
};

// BSH
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_0)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // k
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // v
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // real_shift
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // atten_mask
         {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_kvlen
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // q_start_idx
          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // kv_start_idx
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},         
         // dScaleQ
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleK
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleV
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // queryRope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // keyRope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
         // softmaxMax
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // attentionOut
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2400418603268571936;
    std::string expectTilingData = "1 1 1 256 256 0 128 128 0 4446465452318654464 65536 65536 0 0 0 16908546 1099511627776 1 0 0 0 0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


// BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_1)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // k
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // v
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // real_shift
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // drop_mask
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         // atten_mask
         {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_kvlen
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // q_start_idx
          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         // kv_start_idx
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},         
         // dScaleQ
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleK
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleV
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // queryRope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // keyRope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
         // softmaxMax
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2526519393640251952;
    std::string expectTilingData = "1 1 1 128 128 0 256 256 0 4467570831413529805 2147483647 2147483647 128 128 1 281474993618947 549755813888 1 0 0 2147483647 2147483647 0 0 204 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {17170432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_2)
{
    int64_t actual_seq_qlist[] = {13,14,81};
    int64_t actual_seq_kvlist[] = {13,14,81};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND910_95};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
         // q
        {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
         // k
         {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
         // v
         {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
         // real_shift
         {{{1,2,1024,67}, {1,2,1024,67}}, ge::DT_BF16, ge::FORMAT_ND},
         // drop_mask
         {{{81,2,192}, {81,2,192}}, ge::DT_UINT8, ge::FORMAT_ND},
         // padding_mask
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // atten_mask
        {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
         // prefix
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // actual_seq_qlen
          {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_qlist},
         // actual_seq_kvlen
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_kvlist},
         // q_start_idx
          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         // kv_start_idx
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},         
         // dScaleQ
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleK
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // dScaleV
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // queryRope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // keyRope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
         // softmaxMax
        {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum
         {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0,0,0}, {0,0,0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{81, 2, 192}, {81, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}
         },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",64,262144,8192);
    int64_t expectTilingKey = 2522015793744446016;
    std::string expectTilingData = "3 2 1 81 67 0 192 192 0 4437115660700903014 67 0 1024 67 1 74027927481745412 8796093026561 0 0 0 67 0 2 0 229 0 0 0 0 0 0 0 0 0 0 0 8 8 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 1179648 8 8 8 ";
    std::vector<size_t> expectWorkspaces = {17966592};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}