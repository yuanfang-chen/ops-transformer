/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../op_host/mhc_post_infershape.h"
#include "common/infer_shape_context_faker.h"
#include "common/infer_shape_case_executor.h"

using namespace ge;
using namespace gert;

class MhcPostInferShape : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MhcPostInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MhcPostInferShape TearDown" << std::endl;
    }
};

TEST_F(MhcPostInferShape, test_mhc_post_infer_shape_3d_fp16)
{
    InferShapePara inferShapePara("MhcPost",
        {{{{1024, 4, 5120}, {1024, 4, 5120}}, ge::DT_FLOAT16},
         {{{1024, 4, 4}, {1024, 4, 4}}, ge::DT_FLOAT},
         {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT16},
         {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT});
    TensorDesc outputDesc;

    const int64_t expectDims[] = {1024, 4, 5120};
    ge::graphStatus status = ExecuteTestCase(inferShapePara, outputDesc);

    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(outputDesc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(outputDesc.GetShape().GetDimNum(), 3);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(outputDesc.GetShape().GetDim(i), expectDims[i]);
    }
}

TEST_F(MhcPostInferShape, test_mhc_post_infer_shape_4d_fp16)
{
    InferShapePara inferShapePara("MhcPost",
        {{{{1, 1024, 4, 5120}, {1, 1024, 4, 5120}}, ge::DT_FLOAT16},
         {{{1, 1024, 4, 4}, {1, 1024, 4, 4}}, ge::DT_FLOAT},
         {{{1, 1024, 5120}, {1, 1024, 5120}}, ge::DT_FLOAT16},
         {{{1, 1024, 4}, {1, 1024, 4}}, ge::DT_FLOAT});
    TensorDesc outputDesc;

    const int64_t expectDims[] = {1, 1024, 4, 5120};
    ge::graphStatus status = ExecuteTestCase(inferShapePara, outputDesc);

    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(outputDesc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(outputDesc.GetShape().GetDimNum(), 4);
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(outputDesc.GetShape().GetDim(i), expectDims[i]);
    }
}

TEST_F(MhcPostInferShape, test_mhc_post_infer_shape_3d_bf16)
{
    InferShapePara inferShapePara("MhcPost",
        {{{{256, 8, 2048}, {256, 8, 2048}}, ge::DT_BF16,
         {{{256, 8, 8}, {256, 8, 8}}, ge::DT_FLOAT,
         {{{256, 2048}, {256, 2048}}, ge::DT_BF16,
         {{{256, 8}, {256, 8}}, ge::DT_FLOAT});
    TensorDesc outputDesc;

    const int64_t expectDims[] = {256, 8, 2048};
    ge::graphStatus status = ExecuteTestCase(inferShapePara, outputDesc);

    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), ge::DT_BF16);
    EXPECT_EQ(outputDesc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(outputDesc.GetShape().GetDimNum(), 3);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(outputDesc.GetShape().GetDim(i), expectDims[i]);
    }
}