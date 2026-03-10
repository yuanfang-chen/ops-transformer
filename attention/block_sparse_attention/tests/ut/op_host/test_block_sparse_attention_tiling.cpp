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
#include <iostream>
#include <vector>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

#define private public
#include "platform/platform_info.h"

#include "../../../op_host/block_sparse_attention_tiling.h"

using namespace optiling;

class BlockSparseAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BlockSparseAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BlockSparseAttentionTiling TearDown" << std::endl;
    }
};

TEST_F(BlockSparseAttentionTiling, block_sparse_attention_tiling_tnd_fp16)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    gert::TilingContextPara tilingContextPara(
    "BlockSparseAttention",
    { // input info
        {{{qSeqlen * batch, numHeads, embeddingSize}, {qSeqlen * batch, numHeads, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query
        {{{kvSeqlen * batch, kvHeads, embeddingSize}, {kvSeqlen * batch, kvHeads, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key
        {{{kvSeqlen * batch, kvHeads, embeddingSize}, {kvSeqlen * batch, kvHeads, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value
        {{{batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, {batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}}, ge::DT_UINT8, ge::FORMAT_ND},  // blockSparseMask
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},  // attenMask
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},  // blockShape
        {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},  // actualSeqLengths
        {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},  // actualSeqLengthsKv
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},  // blockTable
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // attentionOut
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmaxLse
    }, 
    { // attr
        {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<const char*>("TND")},
        {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<const char*>("TND")},
        {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvHeads)},
        {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f / std::sqrt(embeddingSize))},
        {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
        {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    });
    
    ExecuteTilingTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(BlockSparseAttentionTiling, block_sparse_attention_tiling_bnsd_fp16)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    gert::TilingContextPara tilingContextPara(
    "BlockSparseAttention",
    { // input info
        {{{batch, numHeads, qSeqlen, embeddingSize}, {batch, numHeads, qSeqlen, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query
        {{{batch, kvHeads, kvSeqlen, embeddingSize}, {batch, kvHeads, kvSeqlen, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key
        {{{batch, kvHeads, kvSeqlen, embeddingSize}, {batch, kvHeads, kvSeqlen, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value
        {{{batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, {batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}}, ge::DT_UINT8, ge::FORMAT_ND},  // blockSparseMask
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},  // attenMask
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},  // blockShape
        {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},  // actualSeqLengths
        {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},  // actualSeqLengthsKv
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},  // blockTable
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // attentionOut
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmaxLse
    }, 
    { // attr
        {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<const char*>("BNSD")},
        {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<const char*>("BNSD")},
        {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvHeads)},
        {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f / std::sqrt(embeddingSize))},
        {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
        {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    });
    
    ExecuteTilingTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(BlockSparseAttentionTiling, block_sparse_attention_tiling_invalid_layout)
{
    size_t batch = 2;
    size_t numHeads = 32;
    size_t kvHeads = 8;
    size_t embeddingSize = 128;
    size_t qSeqlen = 1024;
    size_t kvSeqlen = 1024;
    size_t blockSize = 128;
    
    gert::TilingContextPara tilingContextPara(
    "BlockSparseAttention",
    { // input info
        {{{qSeqlen * batch, numHeads, embeddingSize}, {qSeqlen * batch, numHeads, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query
        {{{kvSeqlen * batch, kvHeads, embeddingSize}, {kvSeqlen * batch, kvHeads, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key
        {{{kvSeqlen * batch, kvHeads, embeddingSize}, {kvSeqlen * batch, kvHeads, embeddingSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // value
        {{{batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}, {batch, numHeads, qSeqlen / blockSize, kvSeqlen / blockSize}}, ge::DT_UINT8, ge::FORMAT_ND},  // blockSparseMask
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},  // attenMask
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},  // blockShape
        {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},  // actualSeqLengths
        {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},  // actualSeqLengthsKv
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},  // blockTable
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // attentionOut
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmaxLse
    }, 
    { // attr
        {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<const char*>("TND")},
        {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<const char*>("BNSD")},
        {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvHeads)},
        {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f / std::sqrt(embeddingSize))},
        {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
        {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"softmaxLseFlag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    });
    
    ExecuteTilingTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
