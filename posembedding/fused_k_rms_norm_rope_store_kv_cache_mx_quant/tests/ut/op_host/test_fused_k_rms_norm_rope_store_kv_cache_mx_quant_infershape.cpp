/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t QKV_DTYPE_SCALE_GRANULARITY = 32;
constexpr int64_t MX_SCALE_PACK = 2;
constexpr float EPSILON = 1e-5f;
constexpr size_t OUTPUT_IDX_Q = 0;
constexpr size_t OUTPUT_IDX_Q_SCALE = 1;
constexpr size_t OUTPUT_IDX_K_CACHE = 2;
constexpr size_t OUTPUT_IDX_K_SCALE_CACHE = 3;
constexpr size_t OUTPUT_IDX_V_CACHE = 4;
constexpr size_t OUTPUT_IDX_V_SCALE_CACHE = 5;

class FusedKRmsNormRopeStoreKvCacheMxQuantInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedKRmsNormRopeStoreKvCacheMxQuantInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedKRmsNormRopeStoreKvCacheMxQuantInferShape TearDown" << std::endl;
    }
};

gert::InfershapeContextPara BuildInferShapeContext(int64_t seqLengthSum,
                                                   int64_t numHeadQ,
                                                   int64_t numHeadK,
                                                   int64_t numHeadV,
                                                   int64_t blockSize)
{
    const int64_t totalNumHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (seqLengthSum + blockSize - 1) / blockSize;
    const int64_t qScaleDim = HEAD_DIM / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK;
    const int64_t vScaleSeq = std::max<int64_t>(1, blockSize / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK);
    const int64_t vScaleSlotNum = std::max<int64_t>(1, seqLengthSum / QKV_DTYPE_SCALE_GRANULARITY / MX_SCALE_PACK);

    return gert::InfershapeContextPara(
        "FusedKRmsNormRopeStoreKvCacheMxQuant",
        {
            {{{seqLengthSum, totalNumHead, HEAD_DIM}, {seqLengthSum, totalNumHead, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{seqLengthSum, 1, HEAD_DIM}, {seqLengthSum, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{seqLengthSum, 1, HEAD_DIM}, {seqLengthSum, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{HEAD_DIM}, {HEAD_DIM}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{seqLengthSum}, {seqLengthSum}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{vScaleSlotNum}, {vScaleSlotNum}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, HEAD_DIM}, {blockNum, numHeadK, blockSize, HEAD_DIM}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}, {blockNum, numHeadK, blockSize, qScaleDim, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{blockNum, numHeadV, blockSize, HEAD_DIM}, {blockNum, numHeadV, blockSize, HEAD_DIM}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{blockNum, numHeadV, vScaleSeq, HEAD_DIM, MX_SCALE_PACK}, {blockNum, numHeadV, vScaleSeq, HEAD_DIM, MX_SCALE_PACK}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(EPSILON)},
        });
}

void CheckInferDataType()
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("FusedKRmsNormRopeStoreKvCacheMxQuant")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType qkvDtype = ge::DT_BF16;
    ge::DataType gammaDtype = ge::DT_FLOAT;
    ge::DataType mappingDtype = ge::DT_INT64;
    ge::DataType fp8Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType fp8ScaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType qOutDtype = ge::DT_FLOAT;
    ge::DataType qScaleOutDtype = ge::DT_FLOAT;
    ge::DataType kCacheOutDtype = ge::DT_FLOAT;
    ge::DataType kScaleCacheOutDtype = ge::DT_FLOAT;
    ge::DataType vCacheOutDtype = ge::DT_FLOAT;
    ge::DataType vScaleCacheOutDtype = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(10)
                             .NodeIoNum(10, 6)
                             .NodeInputTd(0, qkvDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, qkvDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, qkvDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, gammaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, mappingDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, mappingDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, fp8Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, fp8ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(8, fp8Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(9, fp8ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&qkvDtype,
                                              &qkvDtype,
                                              &qkvDtype,
                                              &gammaDtype,
                                              &mappingDtype,
                                              &mappingDtype,
                                              &fp8Dtype,
                                              &fp8ScaleDtype,
                                              &fp8Dtype,
                                              &fp8ScaleDtype})
                             .OutputDataTypes({&qOutDtype,
                                               &qScaleOutDtype,
                                               &kCacheOutDtype,
                                               &kScaleCacheOutDtype,
                                               &vCacheOutDtype,
                                               &vScaleCacheOutDtype})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_Q), fp8Dtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_Q_SCALE), fp8ScaleDtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_K_CACHE), fp8Dtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_K_SCALE_CACHE), fp8ScaleDtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_V_CACHE), fp8Dtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_V_SCALE_CACHE), fp8ScaleDtype);
}
} // namespace

TEST_F(FusedKRmsNormRopeStoreKvCacheMxQuantInferShape, infershape_success_bf16_20_2_2)
{
    auto infershapeContextPara = BuildInferShapeContext(2048, 20, 2, 2, 512);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2048, 20, 128},
        {2048, 20, 2, 2},
        {4, 2, 512, 128},
        {4, 2, 512, 2, 2},
        {4, 2, 512, 128},
        {4, 2, 8, 128, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    CheckInferDataType();
}
