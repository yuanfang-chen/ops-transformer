/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <climits>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
struct QSFAPCompileInfo {
    int64_t coreNum;
};
} // namespace

class QSFAPTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QSFAPTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QSFAPTiling TearDown" << std::endl;
    }
};

// PA_BSND layout with Sink (param_sink feature)
// query(BSND): (1, 1, 8, 576) BF16
// key(PA_BSND): (4, 128, 1, 656) INT8
// value(PA_BSND): (4, 128, 1, 656) INT8
// key_sink/value_sink: (128, 1, 576) BF16
TEST_F(QSFAPTiling, PA_BSND_WithSink_TilingSuccess)
{
    QSFAPCompileInfo compileInfo = {24};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        {
            // 0: query (BSND) BF16
            {{{1, 1, 8, 576}, {1, 1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key (PA_BSND) INT8
            {{{4, 128, 1, 656}, {4, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value (PA_BSND) INT8
            {{{4, 128, 1, 656}, {4, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: sparse_indices (BSND) INT32
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 4: key_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 5: value_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 6: block_table (B=1, maxBlockPerBatch=4) INT32
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 7: actual_seq_lengths_query (B=1) INT32
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // 8: actual_seq_lengths_kv (B=1) INT32
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // 9: key_sink (128, 1, 576) BF16
            {{{128, 1, 576}, {128, 1, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 10: value_sink (128, 1, 576) BF16
            {{{128, 1, 576}, {128, 1, 576}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output 0: attention_out (B=1, S=1, N1=8, D_out=512) BF16
            {{{1, 1, 8, 512}, {1, 1, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        },
        &compileInfo, "Ascend910B", 24, 262144, 8192);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_GT(tilingInfo.tilingKey, static_cast<int64_t>(0));
}

// PA_BSND layout without Sink
TEST_F(QSFAPTiling, PA_BSND_NoSink_TilingSuccess)
{
    QSFAPCompileInfo compileInfo = {24};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        {
            // 0: query (BSND) BF16
            {{{1, 1, 8, 576}, {1, 1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key (PA_BSND) INT8
            {{{4, 128, 1, 656}, {4, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value (PA_BSND) INT8
            {{{4, 128, 1, 656}, {4, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: sparse_indices (BSND) INT32
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 4: key_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 5: value_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 6: block_table (B=1, maxBlockPerBatch=4) INT32
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 7: actual_seq_lengths_query (B=1) INT32
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // 8: actual_seq_lengths_kv (B=1) INT32
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // 9: key_sink (not provided)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // 10: value_sink (not provided)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 8, 512}, {1, 1, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        },
        &compileInfo, "Ascend910B", 24, 262144, 8192);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_GT(tilingInfo.tilingKey, static_cast<int64_t>(0));
}

// BSND Batch Continuous layout (no page attention, no sink)
// query(BSND): (1, 1, 8, 576) BF16
// key(BSND): (1, 512, 1, 656) INT8
// value(BSND): (1, 512, 1, 656) INT8
TEST_F(QSFAPTiling, BSND_BatchContinuous_TilingSuccess)
{
    QSFAPCompileInfo compileInfo = {24};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        {
            // 0: query (BSND) BF16
            {{{1, 1, 8, 576}, {1, 1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key (BSND) INT8
            {{{1, 512, 1, 656}, {1, 512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value (BSND) INT8
            {{{1, 512, 1, 656}, {1, 512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: sparse_indices (BSND) INT32
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 4: key_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 5: value_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 6: block_table (not provided for BSND)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 7: actual_seq_lengths_query (not provided)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 8: actual_seq_lengths_kv (not provided)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 9: key_sink (not provided)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // 10: value_sink (not provided)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 8, 512}, {1, 1, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        },
        &compileInfo, "Ascend910B", 24, 262144, 8192);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_GT(tilingInfo.tilingKey, static_cast<int64_t>(0));
}

// Negative: Sink provided with BSND layout (non-PA) should fail
// CheckKeySinkValueSink requires PA_BSND when sink is provided
TEST_F(QSFAPTiling, Sink_NonPA_Layout_Fail)
{
    QSFAPCompileInfo compileInfo = {24};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        {
            // 0: query (BSND) BF16
            {{{1, 1, 8, 576}, {1, 1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key (BSND) INT8
            {{{1, 512, 1, 656}, {1, 512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value (BSND) INT8
            {{{1, 512, 1, 656}, {1, 512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: sparse_indices (BSND) INT32
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 4: key_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 5: value_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 6: block_table (not provided)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 7: actual_seq_lengths_query (not provided)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 8: actual_seq_lengths_kv (not provided)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 9: key_sink (provided but BSND layout - invalid)
            {{{128, 1, 576}, {128, 1, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 10: value_sink (provided but BSND layout - invalid)
            {{{128, 1, 576}, {128, 1, 576}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 8, 512}, {1, 1, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        },
        &compileInfo, "Ascend910B", 24, 262144, 8192);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: Only key_sink provided without value_sink should fail
TEST_F(QSFAPTiling, SinkKeyOnly_Fail)
{
    QSFAPCompileInfo compileInfo = {24};
    gert::TilingContextPara tilingContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        {
            // 0: query (BSND) BF16
            {{{1, 1, 8, 576}, {1, 1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key (PA_BSND) INT8
            {{{4, 128, 1, 656}, {4, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value (PA_BSND) INT8
            {{{4, 128, 1, 656}, {4, 128, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: sparse_indices (BSND) INT32
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 4: key_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 5: value_dequant_scale FLOAT
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 6: block_table (B=1, maxBlockPerBatch=4) INT32
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 7: actual_seq_lengths_query (B=1) INT32
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // 8: actual_seq_lengths_kv (B=1) INT32
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            // 9: key_sink provided
            {{{128, 1, 576}, {128, 1, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 10: value_sink NOT provided
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 8, 512}, {1, 1, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        },
        &compileInfo, "Ascend910B", 24, 262144, 8192);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
