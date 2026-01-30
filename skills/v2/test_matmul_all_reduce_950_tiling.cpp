/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "matmul_all_reduce_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace matmul_all_reduce_ut {

class MatmulAllReduce950TilingUT : public testing::TestWithParam<MatmulAllReduceTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce950TilingUT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce950TilingUT TearDown" << std::endl;
    }
};

std::vector<MatmulAllReduceTilingUtParam> cases950 {
    {"float16_empty_k", TD({{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 8192}, {0, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 0UL, "dd58009770bc8ce6"},
    {"bfloat16", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "2ba71b00fe7fc087"},
    {"float16_support_3_dim", TD({{1, 8192, 1536}, {1, 8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1, 8192, 12288}, {1, 8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "2ba71b00fe7fc087"},
    {"float16_5", TD({{256, 1536}, {256, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "026e1a40228ed0ad"},
    {"float16_4", TD({{1024, 1536}, {1024, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "5ce2a841298167cd"},
    {"float16_3", TD({{128, 1536}, {128, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "5cf5960462215a75"},
    {"float16_2", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", true, true, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "a1ffc4efb4f5111f"},
    {"float16_win2win", TD({{12290, 15360}, {12290, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{15360, 12288}, {15360, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{12290, 12288}, {12290, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "1079c9c5f300ef5d"},
    {"big_K", TD({{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0xFFFFFFF, 12288}, {0xFFFFFFF, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 65UL, "946053bef319c31b"},
    {"big_N", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 0xFFFFFFF}, {1536, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 65UL, "6a3e0f73aee8ec33"},
    {"float16_unaligned", TD({{1, 65536}, {1, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{65536, 128}, {65536, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1, 128}, {1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "a575e667ab29b825"},
    {"float16_1_cube", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 1UL, "eb53c6e14dcdb043"},
    {"float16_1", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 65UL, "dc7c83080ae1e6ed"},
    {"int8_bf16", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_BF16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 2UL, "53a1cd5204d1bd4b"},
    {"int8_1", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 2UL, "df41afbc79bf1aa3"},
    {"int8_2", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND), TD({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND), std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 258UL, "ba0568fb2c7b3714"},
    {"a8w8_mCut_2", TD({{4096, 1024}, {4096, 1024}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1024, 8192}, {1024, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 2050UL, "d1af34e462d99b54"},
    {"a8w8_mCut_1", TD({{4096, 6272}, {4096, 6272}}, ge::DT_INT8, ge::FORMAT_ND), TD({{6272, 8192}, {6272, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 2050UL, "a342fc081229be80"},
    {"a8w8_scaleDimNum2", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1,8192}, {1,8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{1,8192}, {1,8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1,8192}, {1,8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 2050UL, "8aacab28eb60431e"},
    {"a8w8", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 2050UL, "8aacab28eb60431e"},
    // ============================================================================
    // 以下为补充的测试用例 - 用于提升 weight_quant_matmul_all_reduce_tiling_910_95 覆盖率
    // 添加时间: 2026-01-29
    // 目的: 触发 A16W8 权重量化场景，提升覆盖率
    // 注意: 950不支持FRACTAL_NZ格式的A16W8场景，使用ND格式
    // ============================================================================
    {"a16w8_weight_quant_nz_unsupported", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 8, ge::GRAPH_FAILED},
    {"a16w8_weight_quant_nd", TD({{256, 1536}, {256, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 5767187UL, ""},
    {"a16w8_weight_quant_with_bias", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_INT8, ge::FORMAT_ND), TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 5816339UL, ""},
    {"a16w8_weight_quant_transB", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096, 2048}, {4096, 2048}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 5767187UL, ""},
    {"a16w8_weight_quant_n_unaligned", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 4097}, {1024, 4097}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, TD({{4097}, {4097}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4097}, {4097}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 4097}, {512, 4097}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 5816339UL, ""},
    {"invalid_ranksize_128", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 128, ge::GRAPH_FAILED},
    {"weight_quant_missing_scale", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED},
    {"empty_tensor_test", TD({{0, 0}, {0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED}
};

TEST_P(MatmulAllReduce950TilingUT, arc35)
{
    auto param = GetParam();
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    std::string soc = "Ascend910_95";
    uint64_t coreNum = 64;
    uint64_t ubSize = 262144;
    gert::TilingContextPara tilingContextPara(
        "MatmulAllReduce",
        {
            param.x1,
            param.x2,
            param.bias,
            param.x3,
            param.antiquant_scale,
            param.antiquant_offset,
            param.dequant_scale,
            param.pertoken_scale,
            param.comm_quant_scale_1,
            param.comm_quant_scale_2
        },
        {
            param.y
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_quant_mode)}
        },
        param.inputInstanceNum, param.outputInstanceNum,
        &compileInfo,
        soc, coreNum, ubSize
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", param.ranksize}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
        param.expectTilingDataHash, {}, 0, true);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    MatmulAllReduce950TilingUT,
    testing::ValuesIn(cases950),
    GetCaseInfoString<MatmulAllReduceTilingUtParam>
);

} // namespace matmul_all_reduce_ut
