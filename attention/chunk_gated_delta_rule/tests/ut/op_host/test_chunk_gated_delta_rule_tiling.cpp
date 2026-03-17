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
 * \file test_chunk_gated_delta_rule_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/chunk_gated_delta_rule_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

class ChunkGatedDeltaRuleTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ChunkGatedDeltaRuleTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ChunkGatedDeltaRuleTilingTest TearDown" << std::endl;
    }
};

TEST_F(ChunkGatedDeltaRuleTilingTest, Test0)
{
    // 编译信息：向量核数(aivNum)与UB大小(ubSize)
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};

    // 形状参数：t = bs * seqLen
    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4;
    uint32_t nv = 8;
    uint32_t dk = 128;
    uint32_t dv = 128;

    // 输入 shape（逻辑形状=存储形状）
    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
        {
            // 输入：query/key/value/beta/initial_state/actual_seq_lengths/g
            {queryShape, ge::DT_BF16, ge::FORMAT_ND},
            {keyShape, ge::DT_BF16, ge::FORMAT_ND},
            {valueShape, ge::DT_BF16, ge::FORMAT_ND},
            {betaShape, ge::DT_BF16, ge::FORMAT_ND},
            {stateShape, ge::DT_BF16, ge::FORMAT_ND},
            {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出：out / final_state
            {{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 常量输入：scale_value
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 0UL;
    TilingInfo tilingInfo;
    // 执行 tiling 并校验 tilingKey
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}
