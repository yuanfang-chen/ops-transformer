/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file test_grouped_weight_quant_batch_matmul_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "test_grouped_matmul_utils.h"

using namespace std;
using namespace ge;

class GroupedWeightQuantBatchMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedWeightQuantBatchMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedWeightQuantBatchMatmulTiling TearDown" << std::endl;
    }
};

// A16W4 BF16 - perchannel antiquant
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_notransw)
{
    size_t M = 512;
    size_t K = 2048;
;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antantiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - perchannel antiquant with offset
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_withoffset_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 FP16 - perchannel antiquant
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4ofp16_perchannel_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},            //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT4, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - perchannel antiquant - transposed weight
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_transw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, {E, N}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        1, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - perchannel antiquant - FP32 bias
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_fp32bias_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - no bias
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_nobias_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - perchannel antiquant - grouplist type count
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_grouplistcount_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - larger N dimension
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_largen_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 4096;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - small N dimension
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_smalln_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 256;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - FP32 weight (INT32)
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_int32_perchannel_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT32, ge::FORMAT_ND},            //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT32, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 FP16 - FP32 weight (INT32)
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4ofp16_int32_perchannel_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},            //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT32, ge::FORMAT_ND},            //weight
                                                    {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //antiquantScale
                                                    {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_FLOAT16, // D_T_A
        DT_INT32, // D_T_B
        DT_FLOAT16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - split_item 2 (s-m-s)
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_splititem2_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}

// A16W4 BF16 - NO_SPLIT group_type
TEST_F(GroupedWeightQuantBatchMatmulTiling, test_tiling_a16w4obf16_perchannel_nosplit_notransw)
{
    size_t M = 512;
    size_t K = 2048;
    size_t N = 1024;
    size_t E = 8;
    optiling::GMMCompileInfo compileInfo = {
        24,//aicNum
        48,//aivNum
        196608,//ubSize
        524288,//l1Size
        196608,//l2Size
        131072,//l0CSize
        65536,//l0ASize
        65536,//l0BSize
        platform_ascendc::SocVersion::ASCEND910B,//ASCEND910B
        NpuArch::DAV_2201,
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", // op_name
                                                { // input info
                                                    {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},               //x
                                                    {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND},             //weight
                                                    {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND},                 //bias
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
                                                    {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                  //antiquantScale
                                                    {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                          //antiquantOffset
                                                    {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         //perTokenScale
                                                }, 
                                                { // output info
                                                    {{{M}, {N}}, ge::DT_BF16, ge::FORMAT_ND}
                                                }, 
                                                { // attr
                                                    {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                    {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                    {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
                                                }, &compileInfo);
    int64_t expectTilingKey = gmmTestUtils::GMMEncodeTilingKey(
        DT_BF16, // D_T_A
        DT_INT4, // D_T_B
        DT_BF16, // D_T_Y
        0, // TRANS_A
        0, // TRANS_B
        GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
        0, // IS_STATIC_TILING_API
        GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
        GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
        GROUPED_MATMUL_AIV_AIC_RATIO_1, // AIV_AIC_RATIO
        0 //IS_ENABLE_FIXED_AXIS
    ); // tilngkey
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,230);
}
