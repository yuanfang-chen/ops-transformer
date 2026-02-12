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
 * \file ts_aclnn_mhc_post.cpp
 * \brief MhcPost ACLNN 测试用例.
 */

#include "ts_aclnn_mhc_post.h"

namespace {

// 辅助函数：构建AclnnMhcPostCase
inline AclnnMhcPostParam BuildMhcPostParam(const std::vector<ops::adv::tests::utils::Tensor> &tensors)
{
    AclnnMhcPostParam param;
    std::map<std::string, ops::adv::tests::utils::Tensor> tensorMap;
    for (const auto &tensor : tensors) {
        tensorMap[tensor.Name()] = tensor;
    }
    param.mParamData["tensors"] = tensorMap;
    return param;
}

TEST_P(Ts_Aclnn_Mhc_Post_WithParam_Ascend950, Tc_Aclnn_Mhc_Post_FP16_3D)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

TEST_P(Ts_Aclnn_Mhc_Post_WithParam_Ascend950, Tc_Aclnn_Mhc_Post_BF16_3D)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

// 测试用例定义
const auto Tc_Mhc_Post_3D_FP16_Case = ::testing::Values(

    AclnnMhcPostCase("Test_3D_FP16_001", true,                                       /* CaseName,Enable */
                     "",                                                              /* DebugInfo */
                     ops::adv::tests::utils::OpInfo(
                         ops::adv::tests::utils::ControlInfo(true, true),            /* RunTiling,RunKernel */
                         ops::adv::tests::utils::ExpectInfo(true,                    /* ExpectSuccess */
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingKey,
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingBlockDim)),
                     BuildMhcPostParam({
                         GenTensor("x", {1024, 4, 5120}, ge::DataType::DT_FLOAT16),   // [T, n, D]
                         GenTensor("h_res", {1024, 4, 4}, ge::DataType::DT_FLOAT),     // [T, n, n]
                         GenTensor("h_out", {1024, 5120}, ge::DataType::DT_FLOAT16),   // [T, D]
                         GenTensor("h_post", {1024, 4}, ge::DataType::DT_FLOAT),       // [T, n]
                         GenTensor("y", {1024, 4, 5120}, ge::DataType::DT_FLOAT16)})), // [T, n, D]

    AclnnMhcPostCase("Test_3D_FP16_002", true,
                     "",
                     ops::adv::tests::utils::OpInfo(
                         ops::adv::tests::utils::ControlInfo(true, true),
                         ops::adv::tests::utils::ExpectInfo(true,
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingKey,
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingBlockDim)),
                     BuildMhcPostParam({
                         GenTensor("x", {512, 6, 4096}, ge::DataType::DT_FLOAT16),
                         GenTensor("h_res", {512, 6, 6}, ge::DataType::DT_FLOAT),
                         GenTensor("h_out", {512, 4096}, ge::DataType::DT_FLOAT16),
                         GenTensor("h_post", {512, 6}, ge::DataType::DT_FLOAT),
                         GenTensor("y", {512, 6, 4096}, ge::DataType::DT_FLOAT16)})));

const auto Tc_Mhc_Post_3D_BF16_Case = ::testing::Values(

    AclnnMhcPostCase("Test_3D_BF16_001", true,                                       /* CaseName,Enable */
                     "",                                                              /* DebugInfo */
                     ops::adv::tests::utils::OpInfo(
                         ops::adv::tests::utils::ControlInfo(true, true),            /* RunTiling,RunKernel */
                         ops::adv::tests::utils::ExpectInfo(true,                    /* ExpectSuccess */
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingKey,
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingBlockDim)),
                     BuildMhcPostParam({
                         GenTensor("x", {128, 8, 2048}, ge::DataType::DT_BF16),      // [T, n, D]
                         GenTensor("h_res", {128, 8, 8}, ge::DataType::DT_FLOAT),     // [T, n, n]
                         GenTensor("h_out", {128, 2048}, ge::DataType::DT_BF16),      // [T, D]
                         GenTensor("h_post", {128, 8}, ge::DataType::DT_FLOAT),       // [T, n]
                         GenTensor("y", {128, 8, 2048}, ge::DataType::DT_BF16)})));    // [T, n, D]

INSTANTIATE_TEST_SUITE_P(MhcPost_3D_FP16, Ts_Aclnn_Mhc_Post_WithParam_Ascend950, Tc_Mhc_Post_3D_FP16_Case);
INSTANTIATE_TEST_SUITE_P(MhcPost_3D_BF16, Ts_Aclnn_Mhc_Post_WithParam_Ascend950, Tc_Mhc_Post_3D_BF16_Case);

// 4D BSND格式测试用例
const auto Tc_Mhc_Post_4D_FP16_Case = ::testing::Values(

    AclnnMhcPostCase("Test_4D_FP16_001", true,                                       /* CaseName,Enable */
                     "",                                                              /* DebugInfo */
                     ops::adv::tests::utils::OpInfo(
                         ops::adv::tests::utils::ControlInfo(true, true),            /* RunTiling,RunKernel */
                         ops::adv::tests::utils::ExpectInfo(true,                    /* ExpectSuccess */
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingKey,
                                                            ops::adv::tests::utils::ExpectInfo::kInvalidTilingBlockDim)),
                     BuildMhcPostParam({
                         GenTensor("x", {1, 1024, 4, 5120}, ge::DataType::DT_FLOAT16),   // [B, S, n, D]
                         GenTensor("h_res", {1, 1024, 4, 4}, ge::DataType::DT_FLOAT),     // [B, S, n, n]
                         GenTensor("h_out", {1, 1024, 5120}, ge::DataType::DT_FLOAT16),   // [B, S, D]
                         GenTensor("h_post", {1, 1024, 4}, ge::DataType::DT_FLOAT),       // [B, S, n]
                         GenTensor("y", {1, 1024, 4, 5120}, ge::DataType::DT_FLOAT16)}))  // [B, S, n, D]
);

INSTANTIATE_TEST_SUITE_P(MhcPost_4D_FP16, Ts_Aclnn_Mhc_Post_WithParam_Ascend950, Tc_Mhc_Post_4D_FP16_Case);

} // namespace