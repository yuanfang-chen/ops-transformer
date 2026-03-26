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
 * \file test_aclnn_moe_token_unpermute.cpp
 * \brief
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_token_unpermute.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_unpermute_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_unpermute_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_unpermute_test TearDown" << endl;
    }
};

// ==================== 正常场景测试 ====================

// dtype fp32
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_fp32)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_unpermute_test, Ascend910_9589_moe_token_unpermute_fp32)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype fp16
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_fp16)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype bfloat16
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_bf16)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_BF16, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ================= probsOptional 为 nullptr 的场景 ====================

TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_null_probs)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, (aclTensor*)nullptr, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 不同 Shape 测试 ====================

// 大 batch size
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_large_batch)
{
    auto permuteTokens = TensorDesc({32, 128}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({32}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({32, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({32, 128}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 大 topK
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_large_topK)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 边界情况：最小 shape
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_min_shape)
{
    auto permuteTokens = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 锭误场景测试 ====================

// 空指针测试：permuteTokens 为 nullptr
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_null_permuteTokens)
{
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT((aclTensor*)nullptr, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：SortedIndices 为 nullptr
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_null_SortedIndices)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, (aclTensor*)nullptr, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：out 为 nullptr
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_null_out)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT((aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：workspaceSize 为 nullptr
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_null_workspaceSize)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t* workspaceSize = nullptr;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：executor 为 nullptr
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_null_executor)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor** executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 数据类型不匹配：permuteTokens 和 out 类型不一致
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_dtype_mismatch)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 非连续 Tensor 测试 ====================

// 非连续 permuteTokens
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_noncontiguous_permuteTokens)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10).Noncontiguous();
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 非连续 SortedIndices
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_noncontiguous_SortedIndices)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0).Noncontiguous();
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 执行阶段测试 ====================

// 测试完整的两段式调用
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_full_execute)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    // 测试第二段接口
    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}

// 测试 BFLOAT16 类型的完整执行
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_bf16_full_execute)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_BF16, ACL_FORMAT).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, probsOptional, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}

// 测试 probsOptional 为 nullptr 的完整执行
TEST_F(l2_moe_token_unpermute_test, Ascend910B2_moe_token_unpermute_null_probs_full_execute)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT);
    auto ut = OP_API_UT(aclnnMoeTokenUnpermute,
                        INPUT(permuteTokens, SortedIndices, (aclTensor*)nullptr, false, (aclIntArray*)nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}
