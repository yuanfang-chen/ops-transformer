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
 * \file test_aclnn_moe_distribute_combine_teardown.cpp
 * \brief aclnn ut
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_combine_teardown.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class TestAclnnMoeDistributeCombineTeardown : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnMoeDistributeCombineTeardown SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnMoeDistributeCombineTeardown TearDown" << endl;
    }
};

struct MoeDistributeCombineTeardownAclnnTestParam {
    string case_name;
    vector<int64_t> expandXShape;
    vector<int64_t> quantExpandXShape;
    vector<int64_t> expertIdsShape;
    vector<int64_t> expandIdxShape;
    vector<int64_t> expertScalesShape;
    vector<int64_t> commCmdInfoShape;
    vector<int64_t> xActiveMaskOptionalShape;
    vector<int64_t> sharedExpertXOptionalShape;
    vector<int64_t> xOutShape;
    char* groupEp;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t globalBs;
    int64_t commQuantMode;
    int64_t commType;
    char* commAlg;
    aclDataType expandXDtype;
    aclDataType quantExpandXDtype;
    aclDataType expertIdsDtype;
    aclDataType expandIdxDtype;
    aclDataType expertScalesDtype;
    aclDataType commCmdInfoDtype;
    aclDataType xActiveMaskOptionalDtype;
    aclDataType sharedExpertXOptionalDtype;
    aclDataType xOutDtype;
    aclnnStatus aclnnStatusUt;
};

static MoeDistributeCombineTeardownAclnnTestParam g_casesParams[] = {
    {"combine_teardown_1", {48, 4096}, {48, 4096}, {48, 1}, {48, 1}, {48, 1}, {1024}, {48}, {48, 4096}, {8, 4096},
        "moe_distribute_combine_teardown_test_group", 2, 0, 6, 0, 0, 0, 0, 0, 0, nullptr,
        ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_BOOL, ACL_FLOAT16, ACL_FLOAT16, ACLNN_SUCCESS},
};

static void TestOneParamCase(const MoeDistributeCombineTeardownAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    if (param.groupEp == nullptr) {
        std::cerr << "[ERROR]: groupEp is null" << std::endl;
        return;
    }
    vector<int64_t> expandXShape = param.expandXShape;
    vector<int64_t> quantExpandXShape = param.quantExpandXShape;
    vector<int64_t> expertIdsShape = param.expertIdsShape;
    vector<int64_t> expandIdxShape = param.expandIdxShape;
    vector<int64_t> expertScalesShape = param.expertScalesShape;
    vector<int64_t> commCmdInfoShape = param.commCmdInfoShape;
    vector<int64_t> xActiveMaskOptionalShape = param.xActiveMaskOptionalShape;
    vector<int64_t> sharedExpertXOptionalShape = param.sharedExpertXOptionalShape;
    vector<int64_t> xOutShape = param.xOutShape;
    char* groupEp = param.groupEp;
    int64_t epWorldSize = param.epWorldSize;
    int64_t epRankId = param.epRankId;
    int64_t moeExpertNum = param.moeExpertNum;
    int64_t expertShardType = param.expertShardType;
    int64_t sharedExpertNum = param.sharedExpertNum;
    int64_t sharedExpertRankNum = param.sharedExpertRankNum;
    int64_t globalBs = param.globalBs;
    int64_t commQuantMode = param.commQuantMode;
    int64_t commType = param.commType;
    char* commAlg = param.commAlg;
    aclDataType expandXDtype = param.expandXDtype;
    aclDataType quantExpandXDtype = param.quantExpandXDtype;
    aclDataType expertIdsDtype = param.expertIdsDtype;
    aclDataType expandIdxDtype = param.expandIdxDtype;
    aclDataType expertScalesDtype = param.expertScalesDtype;
    aclDataType commCmdInfoDtype = param.commCmdInfoDtype;
    aclDataType xActiveMaskOptionalDtype = param.xActiveMaskOptionalDtype;
    aclDataType sharedExpertXOptionalDtype = param.sharedExpertXOptionalDtype;
    aclDataType xOutDtype = param.xOutDtype;
    aclnnStatus retStatus = param.aclnnStatusUt;
    TensorDesc expandX = TensorDesc(expandXShape, expandXDtype, ACL_FORMAT_ND);
    TensorDesc quantExpandX = TensorDesc(quantExpandXShape, quantExpandXDtype, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc(expertIdsShape, expertIdsDtype, ACL_FORMAT_ND);
    TensorDesc expandIdx = TensorDesc(expandIdxShape, expandIdxDtype, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc(expertScalesShape, expertScalesDtype, ACL_FORMAT_ND);
    TensorDesc commCmdInfo = TensorDesc(commCmdInfoShape, commCmdInfoDtype, ACL_FORMAT_ND);
    TensorDesc xActiveMaskOptional = TensorDesc(xActiveMaskOptionalShape, xActiveMaskOptionalDtype, ACL_FORMAT_ND);
    TensorDesc sharedExpertXOptional = TensorDesc(sharedExpertXOptionalShape, sharedExpertXOptionalDtype, ACL_FORMAT_ND);
    TensorDesc xOut = TensorDesc(xOutShape, xOutDtype, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeDistributeCombineTeardown,
                        INPUT(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo,
                              &xActiveMaskOptional, &sharedExpertXOptional,
                              groupEp, epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertNum,
                              sharedExpertRankNum, globalBs, commQuantMode, commType, commAlg),
                        OUTPUT(xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
}

TEST_F(TestAclnnMoeDistributeCombineTeardown, CasesParamsTest)
{
    if (std::size(g_casesParams) != 0) {
        uint64_t numCases = sizeof(g_casesParams) / sizeof(g_casesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_casesParams[idx]);
        }
    }
}