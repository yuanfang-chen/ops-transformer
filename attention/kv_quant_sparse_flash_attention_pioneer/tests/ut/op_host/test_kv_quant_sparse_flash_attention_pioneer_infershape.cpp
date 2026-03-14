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
#include <climits>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class QSFAPProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QSFAPProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QSFAPProto TearDown" << std::endl;
    }
};

// BSND layout: output shape = (B, S, N, qHeadDim - ropeHeadDim) = (1, 1, 8, 512)
TEST_F(QSFAPProto, BSND_InferShape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        {
            // 0: query (BSND) BF16
            {{{1, 1, 8, 576}, {1, 1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key INT8
            {{{1, 512, 1, 656}, {1, 512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value INT8
            {{{1, 512, 1, 656}, {1, 512, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: sparse_indices INT32
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 4: key_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 5: value_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 6: block_table
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 7: actual_seq_lengths_query
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 8: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 9: key_sink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // 10: value_sink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output: attention_out (to be inferred)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
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
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 8, 512},  // B=1, S=1, N=8, D=576-64=512
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND layout: output shape = (T, N, qHeadDim - ropeHeadDim) = (10, 8, 512)
TEST_F(QSFAPProto, TND_InferShape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvQuantSparseFlashAttentionPioneer",
        {
            // 0: query (TND) BF16
            {{{10, 8, 576}, {10, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key INT8
            {{{10, 1, 656}, {10, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 2: value INT8
            {{{10, 1, 656}, {10, 1, 656}}, ge::DT_INT8, ge::FORMAT_ND},
            // 3: sparse_indices INT32
            {{{10, 1, 4}, {10, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            // 4: key_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 5: value_dequant_scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // 6: block_table
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 7: actual_seq_lengths_query
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 8: actual_seq_lengths_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // 9: key_sink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // 10: value_sink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output: attention_out (to be inferred)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10, 8, 512},  // T=10, N=8, D=576-64=512
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
