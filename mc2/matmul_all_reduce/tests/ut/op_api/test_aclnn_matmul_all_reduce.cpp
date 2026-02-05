/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include "opdev/platform.h"
#include "matmul_all_reduce_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_matmul_all_reduce.h"

namespace MatmulAllReduceUT {

class AclnnMatmulAllReduceTest : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "MatmulAllReduce AclnnMatmulAllReduceTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "MatmulAllReduce AclnnMatmulAllReduceTest TearDown" << std::endl;
    }
};

// 测试 aclTensor 入参为 nullptr 的场景 =================================================================================
TEST_F(AclnnMatmulAllReduceTest, aclTensorNull)
{
    TensorDesc x1 = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 8;
    int64_t streamMode = 1;
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut_null_x1 = OP_API_UT(
        aclnnMatmulAllReduce,
        INPUT(nullptr, x2, bias, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(
        aclnnMatmulAllReduce,
        INPUT(x1, nullptr, bias, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_bias = OP_API_UT(
        aclnnMatmulAllReduce,
        INPUT(x1, x2, nullptr, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnMatmulAllReduce,
        INPUT(x1, x2, bias, group, reduceOp, commTurn, streamMode),
        OUTPUT(nullptr)
    );
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// 测试 aclTensor 入参不为 nullptr 的场景 ================================================================================
TEST_P(AclnnMatmulAllReduceTest, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(
        aclnnMatmulAllReduce,
        INPUT(param.x1, param.x2, param.bias, param.group, param.reduceOp, param.commTurn, param.streamMode),
        OUTPUT(param.output)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
    } else {
        EXPECT_EQ(param.expectAclnnStatus, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    AclnnMatmulAllReduceTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceApiUtParam>
);

} // namespace matmul_all_reduce_ut
