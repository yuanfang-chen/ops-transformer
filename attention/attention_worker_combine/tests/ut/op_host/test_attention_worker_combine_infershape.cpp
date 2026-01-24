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
 * @file test_AttentionWorkerCombine_proto.cpp
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
#include "fusion_ops.h"

class AttentionWorkerCombine : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AttentionWorkerCombine Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AttentionWorkerCombine Proto Test TearDown" << std::endl;
  }
};

TEST_F(AttentionWorkerCombine, AttentionWorkerCombine_infershape) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AttentionWorkerCombine"), nullptr);
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AttentionWorkerCombine")->infer_shape;

  ASSERT_NE(inferShapeFunc, nullptr);

  gert::StorageShape schedule_context_shape = {{1024}, {1024}};
  gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
  gert::StorageShape layer_id_shape = {{1}, {1}};
  gert::StorageShape y_shape_out = {{32, 7168}, {32, 7168}};
  gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

  auto holder = gert::InferShapeContextFaker()
      .NodeIoNum(3, 2)
      .IrInstanceNum({1, 1, 1})
      .InputShapes({&schedule_context_shape, &expert_scales_shape, &layer_id_shape})
      .OutputShapes({&y_shape_out, &next_layer_id_shape_out})
      .NodeAttrs({{"hidden_size", ge::AnyValue::CreateFrom<int64_t>(7168)},
                  {"token_dtype", ge::AnyValue::CreateFrom<int64_t>(0)},
                  {"layer_id", ge::AnyValue::CreateFrom<int64_t>(15)}})
      .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);

  ASSERT_EQ(ge::Shape2String(*output0), "[32, 7168]");
  ASSERT_EQ(ge::Shape2String(*output1), "[1]");
}

TEST_F(AttentionWorkerCombine, AttentionWorkerCombine_inferdtype_fp16) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AttentionWorkerCombine"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AttentionWorkerCombine")->infer_datatype;
  ASSERT_NE(data_type_func, nullptr);
  ge::DataType input_0 = ge::DT_INT8;
  ge::DataType input_1 = ge::DT_FLOAT;
  ge::DataType input_2 = ge::DT_INT32;
  ge::DataType output_0 = ge::DT_FLOAT16;
  ge::DataType output_1 = ge::DT_INT32;
  auto context_holder = gert::InferDataTypeContextFaker()
      .IrInputNum(3)
      .NodeIoNum(3, 4)
      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeAttrs({{"hidden_size", ge::AnyValue::CreateFrom<int64_t>(7168)},
                  {"token_dtype", ge::AnyValue::CreateFrom<int64_t>(0)},
                  {"layer_id", ge::AnyValue::CreateFrom<int64_t>(15)}})
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .InputDataTypes({&input_0, &input_1, &input_2})
      .OutputDataTypes({&output_0, &output_1})
      .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
  EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

TEST_F(AttentionWorkerCombine, AttentionWorkerCombine_inferdtype_bf16) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AttentionWorkerCombine"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AttentionWorkerCombine")->infer_datatype;
  ASSERT_NE(data_type_func, nullptr);
  ge::DataType input_0 = ge::DT_INT8;
  ge::DataType input_1 = ge::DT_FLOAT;
  ge::DataType input_2 = ge::DT_INT32;
  ge::DataType output_0 = ge::DT_FLOAT16;
  ge::DataType output_1 = ge::DT_INT32;
  auto context_holder = gert::InferDataTypeContextFaker()
      .IrInputNum(3)
      .NodeIoNum(3, 4)
      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeAttrs({{"hidden_size", ge::AnyValue::CreateFrom<int64_t>(7168)},
                  {"token_dtype", ge::AnyValue::CreateFrom<int64_t>(1)},
                  {"layer_id", ge::AnyValue::CreateFrom<int64_t>(15)}})
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .InputDataTypes({&input_0, &input_1, &input_2})
      .OutputDataTypes({&output_0, &output_1})
      .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
  EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}
