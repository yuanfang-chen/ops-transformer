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
            {{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 10347, 512}, {4, 10347, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 10347, 512}, {4, 10347, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 13, 10347}, {4, 13, 10347}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{{{4, 13, 8192}, {4, 13, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
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