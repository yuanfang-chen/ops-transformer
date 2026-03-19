/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array> 
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

class sparse_flash_attention_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "sparse_flash_attention_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "sparse_flash_attention_opapi_ut TearDown" << endl;
    }
};

TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_aclnn_0) {

/*
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,

    double           scaleValue,
    int64_t          sparseBlockSizeOptional,
    char             *layoutQueryOptional,
    char             *layoutKvOptional,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    int64_t          attentionMode,
    bool             returnSoftmaxLse,
*/

    // auto tensorQ = TensorDesc({1, 128, 128, 512}, ACL_BF16, ACL_FORMAT_ND)
    //     .ValueRange(-10, 10);  
    // auto tensorK = TensorDesc({1, 128, 1, 512}, ACL_BF16, ACL_FORMAT_ND)
    //     .ValueRange(-1000, 1000);
    // auto tensorV = TensorDesc({1, 128, 1, 512}, ACL_BF16, ACL_FORMAT_ND)
    //     .ValueRange(-1000, 1000);
    // auto tensorSparseIndices = TensorDesc({1, 128, 1, 2048}, ACL_INT32, ACL_FORMAT_ND)
    //     .ValueRange(-1, 0);
    // auto tensorBlockTable = TensorDesc({1, 1}, ACL_INT32, ACL_FORMAT_ND)
    //     .ValueRange(0, 1);
    // auto tensorasl_Q = TensorDesc({{113}}, ACL_INT32, ACL_FORMAT_ND)
    //     .Value(vector<uint8_t>{1});
    // auto tensorasl_Kv = TensorDesc({{113}}, ACL_INT32, ACL_FORMAT_ND)
    //     .Value(vector<uint8_t>{1});
    // auto tensorQueryRope = TensorDesc({1, 128, 128, 64}, ACL_BF16, ACL_FORMAT_ND)
    //     .ValueRange(-500, 100);
    // auto tensorKeyRope = TensorDesc({1, 128, 1, 64}, ACL_BF16, ACL_FORMAT_ND)
    //     .ValueRange(-500, 100);
    // 输出
    // auto tensorAttentionOut = TensorDesc({1, 128, 128, 512}, ACL_BF16, ACL_FORMAT_ND);
    // auto tensorSoftmaxMax = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    // auto tensorSoftmaxSum = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    const double scaleValue = 0.0416666666666667;
    const int64_t sparseBlockSize = 64;
    char layoutQ[] = "BSND";
    char layoutKv[] = "BSND";
    const int64_t sparseMode = 3;
    const int64_t preTokens = INT64_MAX;
    const int64_t nextTokens = INT64_MAX;
    const int64_t attentionMode = 0;
    const bool returnSoftmaxLse = false;

    auto ut = OP_API_UT(
        aclnnSparseFlashAttention,
        INPUT(
            TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),
            TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),
            TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),
            TensorDesc({24, 1, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1),
            nullptr,
            nullptr,                  // asl_Q
            nullptr,                  // asl_Kv
            nullptr,
            nullptr,
            scaleValue,
            sparseBlockSize,
            layoutQ,
            layoutKv,
            sparseMode,
            preTokens,
            nextTokens,
            attentionMode,
            returnSoftmaxLse
        ),
        OUTPUT(
            TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND),
            TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND),
            TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND)
        )
    );

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}