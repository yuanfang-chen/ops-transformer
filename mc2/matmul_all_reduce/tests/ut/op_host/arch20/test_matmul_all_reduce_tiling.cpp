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
#include "../matmul_all_reduce_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace matmul_all_reduce_ut {

class Arch20TilingTest : public testing::TestWithParam<MatmulAllReduceTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce Arch20TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce Arch20TilingTest TearDown" << std::endl;
    }
};

std::vector<MatmulAllReduceTilingUtParam> casesArch20 {
    // // 正确用例
    // {"float16_basic_nd", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 33554497UL, "952926732179b875"},
    // {"float16_weight_nz", TD({{2048, 3072}, {2048, 3072}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{3072, 8192}, {3072, 8192}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{2048, 8192}, {2048, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 134217985UL, "521182932539e487"},
    // {"float16_small_shape", TD({{16, 32}, {16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 33554433UL, "df6fc78cddb7612d"},
    // {"float16_large_shape", TD({{8192, 15360}, {8192, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{15360, 8192}, {15360, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 33554433UL, "934328e85e28c037"},
    // {"float16_k_zero", TD({{512, 0}, {512, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 17UL, "05765b922f012a8b"},
    // {"float16_transB", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096, 2048}, {4096, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 33554497UL, "6f0ab654e34c5333"},
    // {"float16_with_bias", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 33554433UL, "433a83581d371f75"},
    // {"float16_with_x3", TD({{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{256, 1024}, {256, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{256, 1024}, {256, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 33554433UL, "2e81248b4c4f7df0"},
    // {"a8w8_quant_basic", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 9UL, "ad0cdfa00cae6a27"},
    // {"a8w8_with_bias", TD({{128, 768}, {128, 768}}, ge::DT_INT8, ge::FORMAT_ND), TD({{768, 4096}, {768, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 9UL, "854e0e7d73b0d2c5"},
    // {"a16w8_weight_quant_basic", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 139460621UL, "986ada0c8c8a11a1"},
    // {"a16w4_weight_quant_basic", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 139460621UL, "986ada0c8c8a11a1"},
    // {"fp8_as_a16w8_scene", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 33554433UL, "a9ad7bf4d2537f0a"},
    // {"ranksize_1", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 1, ge::GRAPH_SUCCESS, 33554433UL, "9dd10c51322046a7"},
    // {"ranksize_4", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 4, ge::GRAPH_SUCCESS, 33554433UL, "aaaefab5bd549022"},
    // {"float16_split_k", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 1, 2, ge::GRAPH_SUCCESS, 33554497UL, "5cd39af47afb84f7"},
    // {"weight_quant_n_unaligned", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 4097}, {1024, 4097}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, TD({{4097}, {4097}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4097}, {4097}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 4097}, {512, 4097}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 105906189UL, "65ac5c9264b45588"},
    // {"int8_weight_with_scale", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_SUCCESS, 105906189UL, "d046155562d465ea"},
    // // 失败用例
    // {"invalid_ranksize_8", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_FAILED},
    // {"a8w8_invalid_n_aligned", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 4097}, {1536, 4097}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{4097}, {4097}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 4097}, {256, 4097}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED},
    // {"a8w8_invalid_k_aligned", TD({{256, 1535}, {256, 1535}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1535, 4096}, {1535, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 4096}, {256, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED},
    // {"a8w8_invalid_transpose", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 4096}, {1536, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 4096}, {256, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", true, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED},
    // {"a8w8_invalid_weight_format", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 4096}, {1536, 4096}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{4096}, {4096}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 4096}, {256, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED},
    // {"weight_quant_missing_scale", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED},
    // {"empty_tensor_test", TD({{0, 0}, {0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED}
};

TEST_P(Arch20TilingTest, param)
{
    auto param = GetParam();
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    std::string soc = "Ascend310P";
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
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
        param.inputInstance, param.outputInstance,
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
    Arch20TilingTest,
    testing::ValuesIn(casesArch20),
    GetCaseInfoString<MatmulAllReduceTilingUtParam>
);

} // namespace matmul_all_reduce_ut
