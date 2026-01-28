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

class MatmulAllReduce910TilingUT : public testing::TestWithParam<MatmulAllReduceTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce910TilingUT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce910TilingUT TearDown" << std::endl;
    }
};

std::vector<MatmulAllReduceTilingUtParam> cases910 {
    // 正确用例
    {"float16_empty_k", TD({{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 8192}, {0, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 16UL, "e8b9177dbe1151a6"},
    {"bfloat16", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "5cc307fc6bcaa579"},
    {"float16_support_3_dim", TD({{1, 8192, 1536}, {1, 8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1, 8192, 12288}, {1, 8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "5cc307fc6bcaa579"},
    {"float16_5", TD({{256, 1536}, {256, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "a1806934a7dc046f"},
    {"float16_4", TD({{1024, 1536}, {1024, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "fc167ab5e8660a15"},
    {"float16_3", TD({{128, 1536}, {128, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "4ac3e9f1d34a8314"},
    {"float16_2", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", true, true, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "8b5920239ab717c4"},
    {"float16_win2win", TD({{12290, 15360}, {12290, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{15360, 12288}, {15360, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{12290, 12288}, {12290, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "f5ae8863c7710567"},
    {"big_K", TD({{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0xFFFFFFF, 12288}, {0xFFFFFFF, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 256UL, "31a2bcb3f02d4d2e"},
    {"big_N", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 0xFFFFFFF}, {1536, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 256UL, "12502e032a479b8d"},
    {"float16_unaligned", TD({{1, 65536}, {1, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{65536, 128}, {65536, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1, 128}, {1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "e1fe81871a7e7e11"},
    {"float16_1_cube", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 260UL, "3684b2fecdd8cde9"},
    {"float16_1", TD({{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 256UL, "a28c54ae9cdb85ab"},
    {"int8_bf16", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_BF16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 8UL, "05e0806a43ad772e"},
    {"int8_1", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 8UL, "4c373ab8721e536a"},
    {"int8_2", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND), TD({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND), std::nullopt, std::nullopt, TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 16392UL, "fb6a99a6932310d6"},
    {"a8w8_mCut_2", TD({{4096, 1024}, {4096, 1024}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1024, 8192}, {1024, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 40UL, "bffc39307d61606d"},
    {"a8w8_mCut_1", TD({{4096, 6272}, {4096, 6272}}, ge::DT_INT8, ge::FORMAT_ND), TD({{6272, 8192}, {6272, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 40UL, "57b8ddce86bd4ca4"},
    {"a8w8_scaleDimNum2", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1,8192}, {1,8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{1,8192}, {1,8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1,8192}, {1,8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 40UL, "c12740a21997b1af"},
    {"a8w8", TD({{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND), TD({{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND), std::nullopt, TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 8, ge::GRAPH_SUCCESS, 40UL, "c12740a21997b1af"},
    // 失败用例
    {"invalid_ranksize_16", TD({{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 16, ge::GRAPH_FAILED},
    {"weight_quant_missing_scale", TD({{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{2048, 4096}, {2048, 4096}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, true, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED},
    {"empty_tensor_test", TD({{0, 0}, {0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND), TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, TD({{0, 1024}, {0, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND), "group", "sum", false, false, 0, 0, 0, 0, 0, 2, ge::GRAPH_FAILED}
};

TEST_P(MatmulAllReduce910TilingUT, arc32)
{
    auto param = GetParam();
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    std::string soc = "Ascend910B";
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
    MatmulAllReduce910TilingUT,
    testing::ValuesIn(cases910),
    GetCaseInfoString<MatmulAllReduceTilingUtParam>
);

} // namespace matmul_all_reduce_ut
