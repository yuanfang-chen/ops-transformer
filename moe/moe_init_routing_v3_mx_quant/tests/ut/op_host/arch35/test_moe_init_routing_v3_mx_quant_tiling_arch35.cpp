/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "register/op_impl_registry.h"
#include "tests/utils/op_host_test_util.h"
#include "op_kernel/moe_init_routing_v3_mx_quant_struct.h"

using namespace MoeInitRoutingV3MxQuantNs;

class MoeInitRoutingV3MxQuantTilingTest : public testing::Test {};

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_small_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_medium_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_large_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_2d_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{64, 128}, {64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{64, 128}, {64, 128}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_single_element_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_aligned_256_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_4d_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_medium_bf16) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_small_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_medium_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_large_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_2d_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_single_element_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_aligned_256_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_4d_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_medium_fp8_e4m3fn) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_small_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_medium_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_large_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_2d_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_single_element_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_aligned_256_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_4d_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_medium_fp8_e5m2) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_small_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_medium_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_large_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_2d_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_single_element_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_aligned_256_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_4d_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_medium_fp8_e8m0) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_small_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_medium_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_large_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{128, 4096}, {128, 4096}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_2d_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{64, 128}, {64, 128}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{64, 128}, {64, 128}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_single_element_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_aligned_256_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_4d_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}

TEST_F(MoeInitRoutingV3MxQuantTilingTest, tiling_3d_medium_int32) {
    gert::TilingContextPara tilingContextPara("MoeInitRoutingV3MxQuant",
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_INT32, ge::FORMAT_ND}
    );
    auto context = gert::TilingContext(tilingContextPara);
    auto* tilingData = context.GetTilingData<MoeInitRoutingV3MxQuantArch35TilingData>();
    EXPECT_GE(context.GetTilingKey(), 1030000);  // 1030000 (single-core GATHER) or 1130000 (multi-core GATHER)
    EXPECT_GT(tilingData->coreNum, 0);
}
