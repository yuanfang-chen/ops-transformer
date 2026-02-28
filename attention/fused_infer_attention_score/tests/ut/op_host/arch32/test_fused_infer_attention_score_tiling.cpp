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

// 构造芯片版本
std::string Fia_tiling_A2SocInfo = "{\n"
                                   "  \"hardware_info\": {\n"
                                   "    \"BT_SIZE\": 0,\n"
                                   "    \"load3d_constraints\": \"1\",\n"
                                   "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                   "    \"Intrinsic_data_move_l12ub\": true,\n"
                                   "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                   "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                   "    \"UB_SIZE\": 196608,\n"
                                   "    \"L2_SIZE\": 201326592,\n"
                                   "    \"L1_SIZE\": 524288,\n"
                                   "    \"L0A_SIZE\": 65536,\n"
                                   "    \"L0B_SIZE\": 65536,\n"
                                   "    \"L0C_SIZE\": 131072,\n"
                                   "    \"vector_core_cnt\": 40,\n"
                                   "    \"cube_core_cnt\": 20,\n"
                                   "    \"socVersion\": \"Ascend910_B3\"\n"
                                   "  }\n"
                                   "}";

class FusedInferAttentionScoreTilingArch32 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedInferAttentionScoreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedInferAttentionScoreTiling TearDown" << std::endl;
    }
};

TEST_F(FusedInferAttentionScoreTilingArch32, FusedInferAttentionScoreTiling_tBSH_FP16)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 8192}, {4, 13, 8192}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // query-input0 (BSH: 3 dims, H=8192=16*512)
            {{{4, 10347, 512}, {4, 10347, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // key-input1 (BSH: 3 dims, H=512=1*512)
            {{{4, 10347, 512}, {4, 10347, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                                                // value-input2 (BSH: 3 dims, H=512=1*512)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // pse_shift-input3
            {{{4, 13, 10347}, {4, 13, 10347}}, ge::DT_INT8, ge::FORMAT_ND}, // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                       // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table-input12 (先不使能)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                        // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                        // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                        // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                        // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                        // kv_start_idx-input27
        },
        {                                                                 // 输出Tensor
         {{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                        // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.041666666666666f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5128)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 104000000010500001;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}


// ============================================================================
// 测试用例 1.3: EmptyTensor_FP16_BNSD（触发空tensor）
// 目的：触发 FiaTilingEmptyTensor 模板，提升 fia_tiling_empty_tensor.cpp.cpp 覆盖率
// ============================================================================
TEST_F(FusedInferAttentionScoreTilingArch32, EmptyTensor_FP16_BNSD)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 0, 512}, {4, 16, 0, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // query-input0 (BNSD: 4 dims, S=0. 空tensor)
            {{{4, 4, 0, 512}, {4, 4, 0, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // key-input1 (BNSD: 4 dims, S=0 空tensor)
            {{{4, 4, 0, 512}, {4, 4, 0, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                           // value-input2 (BNSD: 4 dims, S=0 空tensor)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // kv_start_idx-input27
        },
        {                                                                     // 输出Tensor
         {{{4, 16, 0, 512}, {4, 16, 0, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                            // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824162f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}, // GQA
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 100000000000000020;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}


// ============================================================================
// 以下为补充的测试用例 - 用于提升 fused_infer_attention_score 覆盖率
// 添加时间: 2026-02-27
// 目的: 第1轮 - BF16 + BNSD Layout, GQA 非量化
// ============================================================================

TEST_F(FusedInferAttentionScoreTilingArch32, FusedInferAttentionScoreTiling_BF16_BNSD)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 512, 512}, {4, 16, 512, 512}},
             ge::DT_BF16,
             ge::FORMAT_ND}, // query-input0 (BNSD: 4 dims, B=4, N=16, S=512, D=512)
            {{{4, 4, 10347, 512}, {4, 4, 10347, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key-input1 (BNSD: 4 dims)
            {{{4, 4, 10347, 512}, {4, 4, 10347, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value-input2 (BNSD: 4 dims)
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                 // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                               // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                // kv_padding_size-input14
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // key_shared_prefix-input19
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // query_rope-input21
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // key_rope-input22
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // dequant_scale_query-input24
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                // kv_start_idx-input27
        },
        {                                                                      // 输出Tensor
         {{{4, 16, 512, 512}, {4, 16, 512, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                             // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824162f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 104000000000122220;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}
TEST_F(FusedInferAttentionScoreTilingArch32, FusedInferAttentionScoreTiling_GQA_NoQuant)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 512, 512}, {4, 16, 512, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // query-input0 (BNSD: 4 dims, B=4, N=16, S=512, D=512)
            {{{4, 16, 10347, 512}, {4, 16, 10347, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // key-input1 (BNSD: 4 dims, GQA 比例 4:1)
            {{{4, 16, 10347, 512}, {4, 16, 10347, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2 (BNSD: 4 dims)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                      // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                    // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                     // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                     // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                     // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                   // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // kv_start_idx-input27
        },
        {                                                                         // 输出Tensor
         {{{4, 16, 512, 512}, {4, 16, 512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824162f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}, // No GQA
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 104000000000100000;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// ============================================================================
// 测试用例 1.2: GQA_FP16_BNSD_InnerPrecise0（触发 104 系列）
// 目的：触发 FiaTilingNonQuant 模板，提升 fia_tiling_nonquant.cpp 覆盖率
// ============================================================================
TEST_F(FusedInferAttentionScoreTilingArch32, GQA_FP16_BNSD_InnerPrecise0)
{
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "FusedInferAttentionScore",
        {
            {{{4, 16, 512, 512}, {4, 16, 512, 512}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // query-input0 (BNSD: 4 dims, B=4, N=16, S=512, D=512)
            {{{4, 4, 10347, 512}, {4, 4, 10347, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key-input1 (BNSD: 4 dims)
            {{{4, 4, 10347, 512}, {4, 4, 10347, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value-input2 (BNSD: 4 dims)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // pse_shift-input3
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                    // atten_mask-input4
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},                                  // actual_seq_lengths-空
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // actual_seq_lengths_kv-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale1-input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale1-input6
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // dequant_scale2-input7
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_scale2-input8
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // quant_offset2-input9
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_scale-input10
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // antiquant_offset-input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // block_table-input12
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // query_padding_size-input13
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // kv_padding_size-input14
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_scale-input15
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_antiquant_offset-input16
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_scale-input17
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_antiquant_offset-input18
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_shared_prefix-input19
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // value_shared_prefix-input20
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                                   // actual_shared_prefix_len-空
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope-input21
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // key_rope-input22
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key_rope_antiquant_scale-input23
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dequant_scale_query-input24
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // learnable_sink-input25
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // q_start_idx-input26
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // kv_start_idx-input27
        },
        {                                                                         // 输出Tensor
         {{{4, 16, 512, 512}, {4, 16, 512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attentionOut
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},                                // softmax_lse
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173824162f)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}, // GQA
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, // High Precision
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "Ascend910B", Fia_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 104000000000100000;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}
