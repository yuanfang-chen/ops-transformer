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
#include "../../../../op_kernel/attention_to_ffn_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AttentionToFFNUT {

class AttentionToFFNArch32TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AttentionToFFNArch32TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AttentionToFFNArch32TilingTest TearDown" << std::endl;
    }
};

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNNoQuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNNoQuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 2UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNQuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 1UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNQuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 3UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNNoQuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );
    uint64_t expectTilingKey = 4UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNNoQuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 6UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNQuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 5UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNQuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    uint64_t expectTilingKey = 7UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNMicroBatchNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{2, 16, 7168}, {2, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNBsInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 513, 7168}, {1, 513, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNKInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 16}, {1, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNHInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 8196}, {1, 16, 8196}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNLayerNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 9, 4}, {2, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNMoeExpertNumInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1026, 4}, {1, 1026, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNFfnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNFfnTokenDataShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNAttnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNXDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168, 1}, {1, 16, 7168, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNSessionIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNMicroBatchIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNLayerIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNExpertIdsDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNExpertRankTableDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{9, 4}, {9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNScalesDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168, 1}, {1, 9, 7168, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNActiveMaskDimInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNXDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNSessionIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNMicroBatchIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNLayerIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNExpertIdsDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNExpertRankTableDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNScalesDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNActiveMaskDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNXFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNSessionIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNMicroBatchIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNLayerIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNExpertIdsFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNExpertRankTableDormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNScalesFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch32TilingTest, AttentionToFFNActiveMaskFormatInvalid)
{
    struct AttentionToFFNCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AttentionToFFN",
        {
            {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {{}, ge::DT_INT64, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
            {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
            {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910_93"
    );

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // AttentionToFFNUT