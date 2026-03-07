/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <gtest/gtest.h>
#include "../../../../op_host/op_api/aclnn_mhc_post.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class MhcPostOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcPostOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcPostOpapiUt TearDown" << endl;
    }
};

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_basic)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(
        aclnnMhcPost,
        INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost),
        OUTPUT(tensorOut)
    );
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}