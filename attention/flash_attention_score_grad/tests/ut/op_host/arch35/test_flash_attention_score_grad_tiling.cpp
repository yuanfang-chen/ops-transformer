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
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/include/tiling_base/tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

std::string A5SocInfo = "{\n"
                        "  \"hardware_info\": {\n"
                        "    \"BT_SIZE\": 0,\n"
                        "    \"load3d_constraints\": \"1\",\n"
                        "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                        "    \"Intrinsic_data_move_l12ub\": true,\n"
                        "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                        "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                        "    \"UB_SIZE\": 262144,\n"
                        "    \"L2_SIZE\": 134217728,\n"
                        "    \"L1_SIZE\": 524288,\n"
                        "    \"L0A_SIZE\": 65536,\n"
                        "    \"L0B_SIZE\": 65536,\n"
                        "    \"L0C_SIZE\": 262144,\n"
                        "    \"CORE_NUM\": 32,\n"
                        "    \"socVersion\": \"Ascend910_95\"\n"
                        "  }\n"
                        "}";

class FlashAttentionScoreGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_0)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                         // aivNum
        32,                                         // aicNum
        196608,                                     // ubSize
        524288,                                     // l1Size
        65536,                                      // l0aSize
        65536,                                      // l0bSize
        131072,                                     // l0cSize
        33554432,                                   // l2CacheSize
        32,                                         // coreNum
        platform_ascendc::SocVersion::ASCEND910_95  // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1,1,256,8}, {1,1,256,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1,1,256,8}, {1,1,256,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{256,1,128}, {256,1,128}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dv
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   
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
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",A5SocInfo,4096);
    int64_t expectTilingKey = 2410129757049872;
    std::string expectTilingData = "32 1 1 1 256 256 128 128 4575657222443697349 255 65536 65536 0 0 0 2 1 0 0 4294967297 2 1099511627776 0 2 549755813952 549755813952 2 549755814016 4294967300 1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 32768 32768 32768 65536 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 65536 16384 8192 4 35184372097024 4 35184372097024 4 17179877376 131941395394560 131941395447808 70368744177665 70368744177665 1 4295000064 32768 70368744194048 1 32768 70368744194048 1 32768 70368744194048 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND
TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_1)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                         // aivNum
        32,                                         // aicNum
        196608,                                     // ubSize
        524288,                                     // l1Size
        65536,                                      // l0aSize
        65536,                                      // l0bSize
        131072,                                     // l0cSize
        33554432,                                   // l2CacheSize
        32,                                         // coreNum
        platform_ascendc::SocVersion::ASCEND910_95  // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{4, 2, 5, 1}, {4, 2, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{4, 1, 5, 1}, {4, 1, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{4, 1, 5, 1}, {4, 1, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{4, 2, 5, 1}, {4, 2, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk
         {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dv
         {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",A5SocInfo,4096);
    int64_t expectTilingKey = 2691604599542960;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_2)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                         // aivNum
        32,                                         // aicNum
        196608,                                     // ubSize
        524288,                                     // l1Size
        65536,                                      // l0aSize
        65536,                                      // l0bSize
        131072,                                     // l0cSize
        33554432,                                   // l2CacheSize
        32,                                         // coreNum
        platform_ascendc::SocVersion::ASCEND910_95  // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{1, 1}, {1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
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
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         },
        {
         // 输出Tensor
        // dq
         {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk
         {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        // dv
         {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        // dpse
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dq_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        // dk_rope
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},   
         },
        {
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
         },
                &compileInfo,"Ascend910_95",A5SocInfo,4096);
    int64_t expectTilingKey = 2550867111187490;
    std::string expectTilingData = "32 1 1 1 1 1 1 1 4575657222473777152 255 65536 0 2 0 0 2 1 0 0 4294967297 2 4294967296 0 1 4294967360 4294967297 1 4294967424 4294967297 1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 32 1 1 4294967297 1 4294967297 1 4294967297 131941395394560 131941395447808 137438953473 137438953473 1 4295000064 1 4294983680 1 1 4294983680 1 1 4294983680 65536 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}