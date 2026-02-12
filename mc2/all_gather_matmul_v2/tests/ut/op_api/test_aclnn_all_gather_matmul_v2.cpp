/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "all_gather_matmul_v2_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_all_gather_matmul_v2.h"

namespace AllGatherMatmulV2UT {

class AclnnAllGatherMatmulV2Test : public testing::TestWithParam<AllGatherMatmulV2ApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        std::cout << "AllGatherMatmulV2 AclnnAllGatherMatmulV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "AllGatherMatmulV2 AclnnAllGatherMatmulV2Test TearDown" << std::endl;
    }
};

TEST_P(AclnnAllGatherMatmulV2Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale, param.quantScale,
              param.blockSize, param.group.c_str(), param.gatherIndex, param.commTurn,
              param.streamMode, param.groupSize, param.commMode.c_str()),
        OUTPUT(param.output, param.gatherOut, param.amaxOut)
    );
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV2,
    AclnnAllGatherMatmulV2Test,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV2ApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulV2ApiUtParam>
);

} // namespace AllGatherMatmulV2UT
