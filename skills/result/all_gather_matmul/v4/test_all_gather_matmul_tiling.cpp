/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_kernel/all_gather_matmul_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AllGatherMatmulUT {

class AllGatherMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AllGatherMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AllGatherMatmulTiling TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_1) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{512, 12288}, {512, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288, 3904}, {12288, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{512, 3904}, {512, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 12288}, {512, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_2) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2048, 1536}, {2048, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_3) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{327680, 15360}, {327680, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 10240}, {15360, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{327680, 10240}, {327680, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{327680, 15360}, {327680, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_bfloat16) {
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2048, 1536}, {2048, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_l2cache) {
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{8192, 5120}, {8192, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 12288}, {5120, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5120}, {8192, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_n_0) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 256}, {1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{1024, 0}, {1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 256}, {1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ============================================================================
// 以下为补充的测试用例 - 用于提升 all_gather_matmul tiling 覆盖率
// 添加时间: 2026-02-02
// 目的: 覆盖 bias 场景、不同 rankSize、以及边界条件
// ============================================================================

// 补充测试用例1: bias 场景 (tilingKey = 7, FULL_MESH=1, ND2NZ_OPT=1, BIAS_CAST=1)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_with_bias) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536}, {1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // bias input
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2048, 1536}, {2048, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 7UL;  // FULL_MESH=1, ND2NZ_OPT=1, BIAS_CAST=1
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例2: rankSize = 2 (最小多卡场景)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_ranksize_2) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1024}, {2048, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例3: rankSize = 4
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_ranksize_4) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1024}, {2048, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例4: rankSize = 16
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_ranksize_16) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 512}, {1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 16}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例5: rankSize = 32 (最大支持)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_ranksize_32) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 256}, {512, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 32}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ============================================================================
// 以下为补充的测试用例 v4 - 用于提升 all_gather_matmul tiling 覆盖率
// 添加时间: 2026-02-02
// 目的: 覆盖更多场景，包括不同的数据类型、边界条件等
// ============================================================================

// 补充测试用例6: FLOAT32 数据类型 (覆盖 inputDtypeSize == 4 分支)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float32) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 1024}, {2048, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{512, 1024}, {512, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例7: 无 bias 场景验证 (isND2NZ == 0 的情况)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_no_bias) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2048}, {4096, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例8: 小形状场景 (测试 tiny m 场景)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_tiny_shape) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{128, 512}, {128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 256}, {512, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{128, 256}, {128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 512}, {128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例9: 大 K 值场景 (测试 large K 场景)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_large_k) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 32768}, {2048, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32768, 1024}, {32768, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2048, 1024}, {2048, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 32768}, {2048, 32768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例10: 大 M 大 N 场景 (测试 bwGrowthByShape 分支)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_large_m_n) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{16384, 8192}, {16384, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 16384}, {8192, 16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{16384, 16384}, {16384, 16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 8192}, {16384, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例11: 中等 M 值场景 (测试 medianMFlag 相关逻辑)
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_median_m) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 2048}, {4096, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2048, 2048}, {2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例12: bias + 无转置场景
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_bias_no_trans) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1024}, {2048, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例13: 测试最小 rankSize=2 且 bias 场景
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_rank2_bias) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 256}, {512, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 7UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 补充测试用例14: 测试 isStorageGather = false 分支 (通过空 gatherOut 输出)
// 注意: 这是正常测试用例，因为算子可能不需要存储 gather 输出
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_no_gather_out) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{512, 1024}, {512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 512}, {1024, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            // 空 gatherOut 输出 (只设置 matmul 输出)
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 空 gatherOut
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 3UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // AllGatherMatmulUT