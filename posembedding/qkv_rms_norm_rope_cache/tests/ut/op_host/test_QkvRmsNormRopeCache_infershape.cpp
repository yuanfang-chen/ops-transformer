/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_QkvRmsNormRopeCache_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
// #include <gtest/gtest.h>
// #include <iostream>
// #include "infershape_test_util.h"
// #include "ut_op_common.h"
// #include "log/log.h"
// #include "kernel_run_context_facker.h"
// #include "../../../op_graph/qkv_rms_norm_rope_cache_proto.h"
// #include "runtime/infer_shape_range_context.h"
// #include "exe_graph/runtime/storage_format.h"
// #include "exe_graph/runtime/storage_shape.h"
// #include "register/op_impl_registry.h"

class QkvRmsNormRopeCache : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "QkvRmsNormRopeCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QkvRmsNormRopeCache Proto Test TearDown" << std::endl;
    }
};

std::vector<int64_t> ToString(const gert::Shape& shape) {
    size_t shape_size = shape.GetDimNum();
    std::vector<int64_t> shape_vec(shape_size, 0);
    for (size_t i = 0; i < shape_size; i++) {
        shape_vec[i] = shape.GetDim(i);
    }
    return shape_vec;
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_infershapeA)
{
    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    gert::InfershapeContextPara infershapeContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });

    std::vector<int64_t> expectedoutput0 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput1 = {block_num, Nk * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput2 = {block_num, Nv * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput3 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput4 = {batch_size * seq_len, Nk * dim};
    std::vector<int64_t> expectedoutput5 = {batch_size * seq_len, Nv * dim};

    std::vector<std::vector<int64_t>> expectOutputShape = {
        expectedoutput0, expectedoutput1, expectedoutput2, expectedoutput3, expectedoutput4, expectedoutput5};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_inferdtype_test1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry->GetOpImpl("QkvRmsNormRopeCache"), nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("QkvRmsNormRopeCache")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType qkv_ref = ge::DT_FLOAT16;
        ge::DataType cos_ref = ge::DT_FLOAT16;
        ge::DataType sin_ref = ge::DT_FLOAT16;
        ge::DataType qscale_ref = ge::DT_FLOAT;
        ge::DataType qoffset_ref = ge::DT_INT32;
        ge::DataType kcache_ref = ge::DT_INT8;
        ge::DataType vcache_ref = ge::DT_INT8;
        ge::DataType indice_ref = ge::DT_INT32;
        ge::DataType output_q_ref = ge::DT_FLOAT16;
        ge::DataType output_k_ref = ge::DT_FLOAT16;
        ge::DataType output_v_ref = ge::DT_FLOAT16;
        ge::DataType output_kcache_ref = ge::DT_INT8;
        ge::DataType output_vcache_ref = ge::DT_INT8;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(8)
                                  .NodeIoNum(8, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(6, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&qkv_ref, &cos_ref, &sin_ref, &qscale_ref, &qoffset_ref, &kcache_ref, &vcache_ref, &indice_ref})
                                  .OutputDataTypes({&output_q_ref, &output_k_ref, &output_v_ref, &output_kcache_ref, &output_vcache_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), qkv_ref);
        EXPECT_EQ(context->GetInputDataType(1), cos_ref);
        EXPECT_EQ(context->GetInputDataType(2), sin_ref);
        EXPECT_EQ(context->GetInputDataType(3), qscale_ref);
        EXPECT_EQ(context->GetInputDataType(4), qoffset_ref);
        EXPECT_EQ(context->GetInputDataType(5), kcache_ref);
        EXPECT_EQ(context->GetInputDataType(6), vcache_ref);
        EXPECT_EQ(context->GetInputDataType(7), indice_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_q_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_k_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_v_ref);
        EXPECT_EQ(context->GetOutputDataType(3), output_kcache_ref);
        EXPECT_EQ(context->GetOutputDataType(4), output_vcache_ref);
    }
}


TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_infershapeA)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QkvRmsNormRopeCache")->infer_shape;

    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 128;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape gammaV_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaK_shape, &gammaV_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .NodeAttrs(
                          {{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);
    auto output4 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(4);
    auto output5 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(5);

    std::vector<int64_t> expectedoutput0 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput1 = {block_num, Nk * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput2 = {block_num, Nv * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput3 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput4 = {batch_size * seq_len, Nk * dim};
    std::vector<int64_t> expectedoutput5 = {batch_size * seq_len, Nv * dim};

    ASSERT_EQ(ToString(*output0), expectedoutput0);
    ASSERT_EQ(ToString(*output1), expectedoutput1);
    ASSERT_EQ(ToString(*output2), expectedoutput2);
    ASSERT_EQ(ToString(*output3), expectedoutput3);
    ASSERT_EQ(ToString(*output4), expectedoutput4);
    ASSERT_EQ(ToString(*output5), expectedoutput5);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_infershapeB)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QkvRmsNormRopeCache")->infer_shape;

    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape gammaV_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaK_shape, &gammaV_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .NodeAttrs(
                          {{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);
    auto output4 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(4);
    auto output5 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(5);

    std::vector<int64_t> expectedoutput0 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput1 = {block_num, Nk * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput2 = {block_num, Nv * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput3 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput4 = {batch_size * seq_len, Nk * dim};
    std::vector<int64_t> expectedoutput5 = {batch_size * seq_len, Nv * dim};

    ASSERT_EQ(ToString(*output0), expectedoutput0);
    ASSERT_EQ(ToString(*output1), expectedoutput1);
    ASSERT_EQ(ToString(*output2), expectedoutput2);
    ASSERT_EQ(ToString(*output3), expectedoutput3);
    ASSERT_EQ(ToString(*output4), expectedoutput4);
    ASSERT_EQ(ToString(*output5), expectedoutput5);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_infershapeC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QkvRmsNormRopeCache")->infer_shape;

    int batch_size = 16;
    int seq_len = 3;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    gert::StorageShape qkv_shape = {{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}};
    gert::StorageShape gammaK_shape = {{dim}, {dim}};
    gert::StorageShape gammaV_shape = {{dim}, {dim}};
    gert::StorageShape cos_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape sin_shape = {{batch_size * seq_len, dim}, {batch_size * seq_len, dim}};
    gert::StorageShape index_shape = {{batch_size * seq_len}, {batch_size * seq_len}};
    gert::StorageShape qOut_shape = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape kScale_shape = {{Nk, dim}, {Nk, dim}};
    gert::StorageShape vScale_shape = {{Nv, dim}, {Nv, dim}};

    gert::StorageShape qOut_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_shape_out = {{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}};
    gert::StorageShape vCache_shape_out = {{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}};
    gert::StorageShape qOut_proto_shape_out = {{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}};
    gert::StorageShape kCache_proto_shape_out = {{batch_size * seq_len, Nk * dim}, {batch_size * seq_len, Nk * dim}};
    gert::StorageShape vCache_proto_shape_out = {{batch_size * seq_len, Nv * dim}, {batch_size * seq_len, Nv * dim}};

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 6)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&qkv_shape, &gammaK_shape, &gammaV_shape, &cos_shape, &sin_shape, &index_shape, &qOut_shape, &kCache_shape,
                           &vCache_shape, &kScale_shape, &vScale_shape, nullptr, nullptr})
                      .OutputShapes({&qOut_shape_out, &kCache_shape_out, &vCache_shape_out, &qOut_proto_shape_out, &kCache_proto_shape_out, &vCache_proto_shape_out})
                      .NodeAttrs(
                          {{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                           {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                           {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                           {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                           {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);
    auto output4 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(4);
    auto output5 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(5);

    std::vector<int64_t> expectedoutput0 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput1 = {block_num, Nk * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput2 = {block_num, Nv * dim / 32, block_size, 32};
    std::vector<int64_t> expectedoutput3 = {batch_size * seq_len, Nq * dim};
    std::vector<int64_t> expectedoutput4 = {batch_size * seq_len, Nk * dim};
    std::vector<int64_t> expectedoutput5 = {batch_size * seq_len, Nv * dim};

    ASSERT_EQ(ToString(*output0), expectedoutput0);
    ASSERT_EQ(ToString(*output1), expectedoutput1);
    ASSERT_EQ(ToString(*output2), expectedoutput2);
    ASSERT_EQ(ToString(*output3), expectedoutput3);
    ASSERT_EQ(ToString(*output4), expectedoutput4);
    ASSERT_EQ(ToString(*output5), expectedoutput5);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_inferdtype_test1)
{
    
    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 128;

    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QkvRmsNormRopeCache")->infer_datatype;
    
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType input_3 = ge::DT_INT8;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_INT8;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(13)
                              .NodeIoNum(13, 6)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                                   {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                                   {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                   {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                                   {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes(
                                  {&input_0, &input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_3, &input_3,
                                   &input_2, &input_2, &input_2, &input_2})
                              .OutputDataTypes({&output_0, &output_1, &output_1, &output_0, &output_0, &output_0})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(5), ge::DT_FLOAT16);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_inferdtype_test2)
{
    
    int batch_size = 72;
    int seq_len = 2;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QkvRmsNormRopeCache")->infer_datatype;
    
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType input_3 = ge::DT_INT8;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_INT8;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(13)
                              .NodeIoNum(13, 6)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                                   {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                                   {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                   {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                                   {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes(
                                  {&input_0, &input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_3, &input_3,
                                   &input_2, &input_2, &input_2, &input_2})
                              .OutputDataTypes({&output_0, &output_1, &output_1, &output_0, &output_0, &output_0})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(5), ge::DT_FLOAT16);
}

TEST_F(QkvRmsNormRopeCache, QkvRmsNormRopeCache_inferdtype_test3)
{
    
    int batch_size = 16;
    int seq_len = 3;
    int Nqkv = 18;
    int Nq = 16;
    int Nk = 1;
    int Nv = 1;
    int dim = 128;
    int block_num = 72;
    int block_size = 11898;

    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QkvRmsNormRopeCache")->infer_datatype;
    
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT64;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType input_3 = ge::DT_INT8;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_INT8;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(13)
                              .NodeIoNum(13, 6)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(
                                  {{"qkv_size", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(qkv_size)},
                                   {"head_nums", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(head_nums)},
                                   {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
                                   {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
                                   {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes(
                                  {&input_0, &input_0, &input_0, &input_0, &input_0, &input_1, &input_0, &input_3, &input_3,
                                   &input_2, &input_2, &input_2, &input_2})
                              .OutputDataTypes({&output_0, &output_1, &output_1, &output_0, &output_0, &output_0})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(4), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(5), ge::DT_FLOAT16);
}