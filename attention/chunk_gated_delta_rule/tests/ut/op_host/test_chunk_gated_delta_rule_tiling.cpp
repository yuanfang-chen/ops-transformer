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
 * \brief Chunk Gated Delta Rule Tiling 单元测试
 *
 * 测试类别:
 *   - 基础正确性测试 (有/无Gamma)
 *   - 不同Shape组合测试 (batch, heads, dims)
 *   - 边界测试 (最小/最大shape, 维度约束)
 *   - GQA模式测试 (Grouped Query Attention)
 *   - Scale参数变化测试
 *   - 异常输入测试 (错误dtype, 错误shape, 违反约束)
 *
 * 约束条件:
 *   - nv 必须是 nk 的整数倍 (nv % nk == 0)
 *   - nk, nv 最大为 48
 *   - dk, dv 最大为 128
 *   - bs (batch size) 最大为 8
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>

#include "../../../op_host/chunk_gated_delta_rule_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

// ============================================================================
// 约束常量定义
// ============================================================================

constexpr uint32_t MAX_BATCH_SIZE = 8;      // batch size 最大值
constexpr uint32_t MAX_HEAD_NUM = 48;       // nk, nv 最大值
constexpr uint32_t MAX_DIM = 128;           // dk, dv 最大值

// ============================================================================
// 辅助结构和函数
// ============================================================================

/**
 * @brief 测试参数结构体
 */
struct TilingTestParams {
    uint32_t bs;           // batch size
    uint32_t seqLen;       // sequence length per batch
    uint32_t nk;           // key head num
    uint32_t nv;           // value head num
    uint32_t dk;           // key dimension
    uint32_t dv;           // value dimension
    bool hasGamma;         // 是否有gamma输入
    float scale;           // scale因子
    string description;    // 测试描述

    uint32_t GetT() const { return bs * seqLen; }

    string ToString() const {
        stringstream ss;
        ss << "bs=" << bs << ", seqLen=" << seqLen << ", nk=" << nk << ", nv=" << nv
           << ", dk=" << dk << ", dv=" << dv << ", hasGamma=" << hasGamma << ", scale=" << scale;
        return ss.str();
    }

    // 验证参数是否在约束范围内
    bool IsValid() const {
        return (bs <= MAX_BATCH_SIZE) &&
               (nk <= MAX_HEAD_NUM) && (nv <= MAX_HEAD_NUM) &&
               (dk <= MAX_DIM) && (dv <= MAX_DIM) &&
               (nv % nk == 0);  // nv 必须是 nk 的整数倍
    }
};

/**
 * @brief 创建TilingContextPara
 */
gert::TilingContextPara CreateTilingContext(const TilingTestParams& params,
                                            int64_t expectTilingKey = 0) {
    uint32_t t = params.GetT();

    // 输入 shapes
    gert::StorageShape queryShape = {{t, params.nk, params.dk}, {t, params.nk, params.dk}};
    gert::StorageShape keyShape = {{t, params.nk, params.dk}, {t, params.nk, params.dk}};
    gert::StorageShape valueShape = {{t, params.nv, params.dv}, {t, params.nv, params.dv}};
    gert::StorageShape betaShape = {{t, params.nv}, {t, params.nv}};
    gert::StorageShape stateShape = {{params.bs, params.nv, params.dv, params.dk},
                                      {params.bs, params.nv, params.dv, params.dk}};
    gert::StorageShape seqLengthsShape = {{params.bs}, {params.bs}};
    gert::StorageShape gShape = {{t, params.nv}, {t, params.nv}};

    // 编译信息
    static optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};

    // 构建输入列表
    vector<gert::StorageShapeTensorInfo> inputs = {
        {queryShape, ge::DT_BF16, ge::FORMAT_ND},
        {keyShape, ge::DT_BF16, ge::FORMAT_ND},
        {valueShape, ge::DT_BF16, ge::FORMAT_ND},
        {betaShape, ge::DT_BF16, ge::FORMAT_ND},
        {stateShape, ge::DT_BF16, ge::FORMAT_ND},
        {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
    };

    // 如果有gamma，添加gamma输入
    if (params.hasGamma) {
        inputs.push_back({gShape, ge::DT_FLOAT, ge::FORMAT_ND});
    }

    // 输出列表
    vector<gert::StorageShapeTensorInfo> outputs = {
        {{{t, params.nv, params.dv}, {t, params.nv, params.dv}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{params.bs, params.nv, params.dv, params.dk}, {params.bs, params.nv, params.dv, params.dk}}, ge::DT_BF16, ge::FORMAT_ND},
    };

    return gert::TilingContextPara("ChunkGatedDeltaRule", inputs, outputs,
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(params.scale)}},
        &compileinfo);
}

// ============================================================================
// 测试类
// ============================================================================

class ChunkGatedDeltaRuleTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ChunkGatedDeltaRuleTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ChunkGatedDeltaRuleTilingTest TearDown" << std::endl;
    }

    void RunTilingTest(const TilingTestParams& params, int64_t expectTilingKey = 0) {
        std::cout << "Test: " << params.description << std::endl;
        std::cout << "Config: " << params.ToString() << std::endl;

        gert::TilingContextPara tilingContextPara = CreateTilingContext(params, expectTilingKey);
        TilingInfo tilingInfo;

        EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    }

    void RunTilingTestExpectFail(const TilingTestParams& params) {
        std::cout << "Test (expect fail): " << params.description << std::endl;
        std::cout << "Config: " << params.ToString() << std::endl;

        gert::TilingContextPara tilingContextPara = CreateTilingContext(params);
        TilingInfo tilingInfo;

        // 期望 tiling 失败
        EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
    }
};

// ============================================================================
// 1. 基础正确性测试 - 有/无Gamma
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, BasicWithGamma) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 100,
        .nk = 4,
        .nv = 8,
        .dk = 128,
        .dv = 128,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "basic_with_gamma"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BasicWithoutGamma) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 100,
        .nk = 4,
        .nv = 8,
        .dk = 128,
        .dv = 128,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "basic_without_gamma"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 2. 不同Batch Size测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, SingleBatch) {
    TilingTestParams params{
        .bs = 1,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "single_batch"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, MultiBatch2) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "multi_batch_2"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, MultiBatch4) {
    TilingTestParams params{
        .bs = 4,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "multi_batch_4"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, MultiBatch8) {
    // 最大batch size (bs <= 8)
    TilingTestParams params{
        .bs = 8,
        .seqLen = 32,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "max_batch_8"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 3. 不同Head数测试 (包括GQA模式)
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, SingleHead) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 1,
        .nv = 1,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "single_head"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, EqualHeads) {
    // nk = nv (Multi-Query Attention)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 8,
        .nv = 8,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "equal_heads_mqa"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, GQAMode2x) {
    // GQA: nv = 2 * nk
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 8,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "gqa_mode_2x"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, GQAMode4x) {
    // GQA: nv = 4 * nk
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 2,
        .nv = 8,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "gqa_mode_4x"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, GQAMode8x) {
    // GQA: nv = 8 * nk
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 1,
        .nv = 8,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "gqa_mode_8x"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, MaxHeads) {
    // 最大head数 (nk, nv <= 48)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 48,
        .nv = 48,
        .dk = 32,
        .dv = 32,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "max_heads_48"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 4. 不同维度测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, Dim32) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 32,
        .dv = 32,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "dim_32"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, Dim64) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "dim_64"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, Dim96) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 96,
        .dv = 96,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "dim_96"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, Dim128) {
    // 最大维度 (dk, dv <= 128)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 128,
        .dv = 128,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "dim_128_max"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, DifferentDKDV) {
    // dk != dv
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 128,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "different_dk_dv"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, NonAlignedDim48) {
    // 非对齐维度
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 48,
        .dv = 48,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "non_aligned_dim_48"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, NonAlignedDim80) {
    // 非对齐维度
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 80,
        .dv = 80,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "non_aligned_dim_80"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 5. 不同序列长度测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, ShortSeqLen32) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 32,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "short_seqlen_32"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, SeqLen64) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "seqlen_64"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, SeqLen128) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 128,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "seqlen_128"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, SeqLen256) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 256,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "seqlen_256"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, SeqLen512) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 512,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "seqlen_512"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, LongSeqLen1024) {
    TilingTestParams params{
        .bs = 1,
        .seqLen = 1024,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "long_seqlen_1024"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 6. Scale参数变化测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, Scale0_5) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 0.5f,
        .description = "scale_0_5"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, ScaleInvSqrt) {
    // scale = 1/sqrt(dk) = 1/sqrt(64) = 0.125
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 0.125f,
        .description = "scale_inv_sqrt_64"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, ScaleInvSqrt128) {
    // scale = 1/sqrt(128) ≈ 0.0884
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 128,
        .dv = 128,
        .hasGamma = true,
        .scale = 0.0884f,
        .description = "scale_inv_sqrt_128"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, Scale2_0) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 2.0f,
        .description = "scale_2_0"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, ScaleSmall) {
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 0.01f,
        .description = "scale_small_0_01"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 7. 组合边界测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, MinimalShape) {
    // 最小有效shape
    TilingTestParams params{
        .bs = 1,
        .seqLen = 1,
        .nk = 1,
        .nv = 1,
        .dk = 32,
        .dv = 32,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "minimal_shape"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, MaxDimsWithGQA) {
    // 最大维度 + GQA组合 (nk=8, nv=32 <= 48)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 8,
        .nv = 32,  // nv = 4 * nk
        .dk = 128,
        .dv = 128,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "max_dims_with_gqa"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, LargeBatchSmallSeq) {
    // 最大batch小序列 (bs=8, 即最大batch)
    TilingTestParams params{
        .bs = 8,
        .seqLen = 16,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "max_batch_small_seq"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, SmallBatchLongSeq) {
    // 小batch长序列
    TilingTestParams params{
        .bs = 1,
        .seqLen = 2048,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "small_batch_long_seq"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, NoGammaMaxDims) {
    // 无Gamma + 最大维度
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 16,
        .nv = 16,
        .dk = 128,
        .dv = 128,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "no_gamma_max_dims"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, GammaMinDims) {
    // 有Gamma + 最小维度
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 1,
        .nv = 1,
        .dk = 32,
        .dv = 32,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "gamma_min_dims"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 8. 典型模型配置测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, LLMSmall) {
    // 类似小模型配置
    TilingTestParams params{
        .bs = 4,
        .seqLen = 512,
        .nk = 8,
        .nv = 8,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 0.125f,
        .description = "llm_small_config"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, LLMMedium) {
    // 类似中等模型配置
    TilingTestParams params{
        .bs = 2,
        .seqLen = 1024,
        .nk = 16,
        .nv = 16,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 0.125f,
        .description = "llm_medium_config"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, LLMLargeWithGQA) {
    // 类似大模型GQA配置 (如LLaMA)
    TilingTestParams params{
        .bs = 1,
        .seqLen = 2048,
        .nk = 8,
        .nv = 32,  // GQA 4x (nv = 4 * nk)
        .dk = 128,
        .dv = 128,
        .hasGamma = true,
        .scale = 0.0884f,
        .description = "llm_large_gqa_config"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InferenceBatch) {
    // 推理场景batch配置 (最大batch=8)
    TilingTestParams params{
        .bs = 8,
        .seqLen = 16,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 0.125f,
        .description = "inference_batch_config"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 9. 特殊场景测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, SeqLenEqualsChunkSize) {
    // 序列长度等于chunk size (64)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "seqlen_equals_chunk_size"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, SeqLenNonAligned) {
    // 非对齐序列长度
    TilingTestParams params{
        .bs = 2,
        .seqLen = 100,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "seqlen_non_aligned_100"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, TotalTokensLarge) {
    // 大规模总token数
    TilingTestParams params{
        .bs = 8,
        .seqLen = 512,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "large_total_tokens"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// 10. 异常输入测试 - 期望失败
// ============================================================================

// --------------------- nv 不是 nk 的整数倍 ---------------------

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNvNotMultipleOfNk_3and4) {
    // nv=3, nk=4: 3 % 4 != 0, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 3,  // nv 不是 nk 的倍数
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nv_not_multiple_of_nk_3_4"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNvNotMultipleOfNk_5and8) {
    // nv=5, nk=8: 5 % 8 != 0, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 8,
        .nv = 5,  // nv 不是 nk 的倍数
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nv_not_multiple_of_nk_5_8"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNvNotMultipleOfNk_7and4) {
    // nv=7, nk=4: 7 % 4 != 0, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 7,  // nv 不是 nk 的倍数
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nv_not_multiple_of_nk_7_4"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNvNotMultipleOfNk_10and3) {
    // nv=10, nk=3: 10 % 3 != 0, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 3,
        .nv = 10,  // nv 不是 nk 的倍数
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nv_not_multiple_of_nk_10_3"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNvNotMultipleOfNk_Prime) {
    // nv=17, nk=8: 17 % 8 != 0, 期望失败 (质数)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 8,
        .nv = 17,  // nv 不是 nk 的倍数 (质数)
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nv_not_multiple_of_nk_prime"
    };
    RunTilingTestExpectFail(params);
}

// --------------------- nk 超过最大值 48 ---------------------

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNkExceedMax_49) {
    // nk=49 > 48, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 49,  // 超过最大值 48
        .nv = 49,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nk_exceed_max_49"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNkExceedMax_64) {
    // nk=64 > 48, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 64,  // 超过最大值 48
        .nv = 64,
        .dk = 32,
        .dv = 32,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nk_exceed_max_64"
    };
    RunTilingTestExpectFail(params);
}

// --------------------- nv 超过最大值 48 ---------------------

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNvExceedMax_49) {
    // nv=49 > 48, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 1,
        .nv = 49,  // 超过最大值 48
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nv_exceed_max_49"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidNvExceedMax_64) {
    // nv=64 > 48, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 1,
        .nv = 64,  // 超过最大值 48
        .dk = 32,
        .dv = 32,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_nv_exceed_max_64"
    };
    RunTilingTestExpectFail(params);
}

// --------------------- batch size 超过最大值 8 ---------------------

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidBatchExceedMax_9) {
    // bs=9 > 8, 期望失败
    TilingTestParams params{
        .bs = 9,  // 超过最大值 8
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_batch_exceed_max_9"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidBatchExceedMax_16) {
    // bs=16 > 8, 期望失败
    TilingTestParams params{
        .bs = 16,  // 超过最大值 8
        .seqLen = 32,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "invalid_batch_exceed_max_16"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidBatchExceedMax_32) {
    // bs=32 > 8, 期望失败
    TilingTestParams params{
        .bs = 32,  // 超过最大值 8
        .seqLen = 16,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = false,
        .scale = 1.0f,
        .description = "invalid_batch_exceed_max_32"
    };
    RunTilingTestExpectFail(params);
}

// --------------------- dk 超过最大值 128 ---------------------

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidDkExceedMax_129) {
    // dk=129 > 128, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 129,  // 超过最大值 128
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_dk_exceed_max_129"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidDkExceedMax_256) {
    // dk=256 > 128, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 256,  // 超过最大值 128
        .dv = 256,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_dk_exceed_max_256"
    };
    RunTilingTestExpectFail(params);
}

// --------------------- dv 超过最大值 128 ---------------------

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidDvExceedMax_129) {
    // dv=129 > 128, 期望失败
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 129,  // 超过最大值 128
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_dv_exceed_max_129"
    };
    RunTilingTestExpectFail(params);
}

// --------------------- 组合异常 ---------------------

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidCombo_NvNotMultipleAndBatchExceed) {
    // nv 不是 nk 倍数 + batch 超限
    TilingTestParams params{
        .bs = 10,  // 超过 8
        .seqLen = 64,
        .nk = 4,
        .nv = 5,  // 不是 4 的倍数
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_combo_nv_batch"
    };
    RunTilingTestExpectFail(params);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, InvalidCombo_AllExceed) {
    // 多个约束同时违反
    TilingTestParams params{
        .bs = 16,   // 超过 8
        .seqLen = 64,
        .nk = 64,   // 超过 48
        .nv = 100,  // 超过 48 且不是 nk 的倍数
        .dk = 256,  // 超过 128
        .dv = 256,  // 超过 128
        .hasGamma = true,
        .scale = 1.0f,
        .description = "invalid_combo_all_exceed"
    };
    RunTilingTestExpectFail(params);
}

// ============================================================================
// 11. 边界值精确测试
// ============================================================================

TEST_F(ChunkGatedDeltaRuleTilingTest, BoundaryBatchMax8) {
    // 边界: batch = 8 (刚好等于最大值)
    TilingTestParams params{
        .bs = 8,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "boundary_batch_max_8"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BoundaryNkMax48) {
    // 边界: nk = 48 (刚好等于最大值)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 48,
        .nv = 48,
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "boundary_nk_max_48"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BoundaryNvMax48) {
    // 边界: nv = 48 (刚好等于最大值), nk = 1
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 1,
        .nv = 48,  // nv = 48 * nk
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "boundary_nv_max_48"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BoundaryDkMax128) {
    // 边界: dk = 128 (刚好等于最大值)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 128,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "boundary_dk_max_128"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BoundaryDvMax128) {
    // 边界: dv = 128 (刚好等于最大值)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 4,
        .nv = 4,
        .dk = 64,
        .dv = 128,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "boundary_dv_max_128"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BoundaryNvEqualsNk) {
    // 边界: nv = nk (1倍, 最小倍数)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 16,
        .nv = 16,  // nv = 1 * nk
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "boundary_nv_equals_nk"
    };
    RunTilingTest(params, 0);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BoundaryNvMaxMultipleOfNk) {
    // 边界: nv = 48, nk = 1 (最大倍数 48x)
    TilingTestParams params{
        .bs = 2,
        .seqLen = 64,
        .nk = 1,
        .nv = 48,  // nv = 48 * nk (最大倍数)
        .dk = 64,
        .dv = 64,
        .hasGamma = true,
        .scale = 1.0f,
        .description = "boundary_nv_max_multiple_48x"
    };
    RunTilingTest(params, 0);
}

// ============================================================================
// Main函数
// ============================================================================

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
