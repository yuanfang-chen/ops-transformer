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
 * \file test_allto_all_matmul_infershape.cpp
 * \brief infershape ut
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "mc2_infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "mc2_hcom_topology_mocker.h"

class AlltoAllMatmulInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulInfershape TearDown" << std::endl;
    }
};

TEST_F(AlltoAllMatmulInfershape, infer_shape_for_2p) {
    gert::StorageShape x1_shape = {{228344, 1152}, {}};
    gert::StorageShape x2_shape = {{3072, 2304}, {}};
    gert::StorageShape bias_shape = {{}, {}};
    gert::StorageShape all2all_out_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("AlltoAllMatmul",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {bias_shape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {all2all_out_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"alltoall_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{114172, 3072}, {114172, 2304}};
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(AlltoAllMatmulInfershape, infer_dtype) {
    ge::DataType x1 = ge::DT_FLOAT16;
    ge::DataType x2 = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(2, 1)
        .InputDataTypes({&x1, &x2})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
            {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
            {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"alltoall_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        })
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllMatmul")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}