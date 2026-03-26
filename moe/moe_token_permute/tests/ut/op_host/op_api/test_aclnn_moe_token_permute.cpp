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
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_token_permute.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_permute_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_permute_test TearDown" << endl;
    }
};

// ==================== 正常场景测试 ====================

// dtype fp32
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_fp32)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_test, Ascend910_9589_moe_token_permute_fp32)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype fp16
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_fp16)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype bfloat16
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_bf16)
{
    auto tokens = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype int8
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_int8)
{
    auto tokens = TensorDesc({1, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// indices dtype int64
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_indices_int64)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 不同参数组合测试 ====================

// numOutTokens = 0 (不删除任何token)
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_numOutTokens_zero)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// numOutTokens = 正数 (删除部分token)
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_numOutTokens_positive)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 5, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// numOutTokens = 负数 (负数切片)
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_numOutTokens_negative)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, -1, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 不同 Shape 测试 ====================

// 大 batch size
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_large_batch)
{
    auto tokens = TensorDesc({32, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({32, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto permuteTokensOut = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({128}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 大 topK
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_large_topK)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 32}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 32);
    auto permuteTokensOut = TensorDesc({32, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 边界情况：最小 shape
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_min_shape)
{
    auto tokens = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto permuteTokensOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 错误场景测试 ====================

// 空指针测试：tokens 为 nullptr
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_null_tokens)
{
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT((aclTensor*)nullptr, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：indices 为 nullptr
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_null_indices)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, (aclTensor*)nullptr, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：permuteTokensOut 为 nullptr
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_null_permuteTokensOut)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT((aclTensor*)nullptr, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：sortedIndicesOut 为 nullptr
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_null_sortedIndicesOut)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：workspaceSize 为 nullptr
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_null_workspaceSize)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t* workspaceSize = nullptr;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：executor 为 nullptr
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_null_executor)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor** executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 数据类型不匹配：tokens 和 permuteTokensOut 类型不一致
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_dtype_mismatch)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 非连续 Tensor 测试 ====================

// 非连续 tokens
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_noncontiguous_tokens)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Noncontiguous();
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 非连续 indices
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_noncontiguous_indices)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8).Noncontiguous();
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 不同数据格式测试 ====================

// ACL_FORMAT_NCHW 格式
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_format_nchw)
{
    auto tokens = TensorDesc({1, 64, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8, 1, 1}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto sortedIndicesOut = TensorDesc({8, 1, 1, 1}, ACL_INT32, ACL_FORMAT_NCHW);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 执行阶段测试 ====================

// 测试完整的两段式调用
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_full_execute)
{
    auto tokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    // 测试第二段接口
    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}

// 测试 BFLOAT16 类型的完整执行
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_bf16_full_execute)
{
    auto tokens = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}

// 测试 INT8 类型的完整执行
TEST_F(l2_moe_token_permute_test, Ascend910B2_moe_token_permute_int8_full_execute)
{
    auto tokens = TensorDesc({1, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto indices = TensorDesc({1, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto permuteTokensOut = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto sortedIndicesOut = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeTokenPermute, INPUT(tokens, indices, 0, false), OUTPUT(permuteTokensOut, sortedIndicesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}
