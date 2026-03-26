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
#include "../../../../op_host/op_api/aclnn_moe_token_permute_grad.h"

#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_permute_grad_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_permute_grad_test TearDown" << endl;
    }
};

// ==================== 正常场景测试 ====================

// dtype fp32
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_fp32)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_token_permute_grad_test, Ascend910_9589_moe_token_permute_grad_fp32)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype fp16
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_fp16)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype bfloat16
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_bf16)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 不同参数组合测试 ====================

// numTopk = 1
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_numTopk_1)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// numTopk = 8
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_numTopk_8)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 8, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// numTopk = 32
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_numTopk_32)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 32, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 不同 Shape 测试 ====================

// 大 batch size
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_large_batch)
{
    auto permutedOutputGrad = TensorDesc({32, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({32, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 4, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 大 hidden size
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_large_hidden)
{
    auto permutedOutputGrad = TensorDesc({1, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 边界情况：最小 shape
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_min_shape)
{
    auto permutedOutputGrad = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 错误场景测试 ====================

// 空指针测试：permutedOutputGrad 为 nullptr
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_null_permutedOutputGrad)
{
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT((aclTensor*)nullptr, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：SortedIndices 为 nullptr
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_null_SortedIndices)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, (aclTensor*)nullptr, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：out 为 nullptr
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_null_out)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT((aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：workspaceSize 为 nullptr
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_null_workspaceSize)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t* workspaceSize = nullptr;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针测试：executor 为 nullptr
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_null_executor)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor** executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 数据类型不匹配：permutedOutputGrad 和 out 类型不一致
TEST_F(l2_moe_token_permute_grad_test, (Ascend910B2_moe_token_permute_grad_dtype_mismatch)
{
    auto permuted = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permuted, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 非连续 Tensor 测试 ====================

// 非连续 permutedOutputGrad
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_noncontiguous_permutedOutputGrad)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Noncontiguous();
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 非连续 SortedIndices
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_noncontiguous_SortedIndices)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0).Noncontiguous();
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ==================== 执行阶段测试 ====================

// 测试完整的两段式调用
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_full_execute)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    // 测试第二段接口
    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}

// 测试 BFLOAT16 类型的完整执行
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_bf16_full_execute)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}

// 测试 FLOAT16 类型的完整执行
TEST_F(l2_moe_token_permute_grad_test, Ascend910B2_moe_token_permute_grad_fp16_full_execute)
{
    auto permutedOutputGrad = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeTokenPermuteGrad, INPUT(permutedOutputGrad, SortedIndices, 1, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    aclnnStatus executeResult = ut.TestExecuteWithNNopbaseInner();
    EXPECT_EQ(executeResult, ACLNN_SUCCESS);
}
