/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_ffn_worker_batching_infershape.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "common/utils/ut_op_common.h"
#include "experiment_ops.h"

class FfnWorkerBatchingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerBatchingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerBatchingTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerBatchingTest, ffn_worker_batching_infer_shape_001)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FfnWorkerBatching"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FfnWorkerBatching")->infer_shape;

    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape y_shape = {{1152, 4096}, {1152, 4096}};
    gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    gert::StorageShape session_ids_shape = {{1152}, {1152}};
    gert::StorageShape micro_batch_ids_shape = {{1152}, {1152}};
    gert::StorageShape token_ids_shape = {{1152}, {1152}};
    gert::StorageShape expert_offsets_shape = {{1152}, {1152}};
    gert::StorageShape dynamic_scale_shape = {{1152}, {1152}};
    gert::StorageShape actual_token_num_shape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 8)
                      .IrInstanceNum({1})
                      .InputShapes({&schedule_context_shape})
                      .OutputShapes(
                          {&y_shape, &group_list_shape, &session_ids_shape, &micro_batch_ids_shape, &token_ids_shape,
                           &expert_offsets_shape, &dynamic_scale_shape, &actual_token_num_shape})
                      .NodeAttrs(
                          {{"expert_num", ge::AnyValue::CreateFrom<int64_t>(8)},
                           {"max_out_shape", ge::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
                           {"token_dtype", ge::AnyValue::CreateFrom<int64_t>(0)},
                           {"need_schedule", ge::AnyValue::CreateFrom<int64_t>(1)},
                           {"layer_num", ge::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputy = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*outputy), "[1152, 4096]");
}

TEST_F(FfnWorkerBatchingTest, ffn_worker_batching_infer_dtype_001)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FfnWorkerBatching"), nullptr);
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FfnWorkerBatching")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    ge::DataType xD = ge::DT_INT8;
    ge::DataType yD = ge::DT_UNDEFINED;
    ge::DataType groupListD = ge::DT_UNDEFINED;
    ge::DataType sessionIdD = ge::DT_UNDEFINED;
    ge::DataType microBatchIdD = ge::DT_UNDEFINED;
    ge::DataType tokenIdD = ge::DT_UNDEFINED;
    ge::DataType expertOffD = ge::DT_UNDEFINED;
    ge::DataType dynamicD = ge::DT_UNDEFINED;
    ge::DataType actualTokenD = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                                .IrInputNum(1)
                                .NodeIoNum(1, 8)
                                .InputDataTypes({&xD})
                                .OutputDataTypes({&yD, &groupListD, &sessionIdD, &microBatchIdD, &tokenIdD,
                                    &expertOffD, &dynamicD, &actualTokenD})
                                .NodeAttrs(
                                    {{"expert_num", ge::AnyValue::CreateFrom<int64_t>(8)},
                                    {"max_out_shape", ge::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
                                    {"token_dtype", ge::AnyValue::CreateFrom<int64_t>(0)},
                                    {"need_schedule", ge::AnyValue::CreateFrom<int64_t>(1)},
                                    {"layer_num", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT64);
}
