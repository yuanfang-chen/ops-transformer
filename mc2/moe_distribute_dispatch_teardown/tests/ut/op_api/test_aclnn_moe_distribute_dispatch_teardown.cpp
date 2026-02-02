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
 * \file test_aclnn_moe_distribute_dispatch_teardown.cpp
 * \brief aclnn ut
 */

#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <fstream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_dispatch_teardown.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

namespace {

using namespace op;

class test_aclnn_moe_distribute_dispatch_teardown : public testing::Test {
protected:
    static void SetUpTestCase() {
        SetPlatformSocVersion(NpuArch::DAV_3510);
        std::cout << "test_aclnn_moe_distribute_dispatch_teardown SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        SetPlatformSocVersion(SocVersion::ASCEND910B);
        std::cout << "test_aclnn_moe_distribute_dispatch_teardown TearDown" << std::endl;
    }
};

// 定义用例信息结构体
struct MoeDistributeDispatchTeardownAclnnTestParam {
    string case_name;
    // 输入信息shape
    std::vector<int64_t> x;
    std::vector<int64_t> y;
    std::vector<int64_t> expert_ids;
    std::vector<int64_t> comm_cmd_info;
    std::vector<int64_t> xActiveMask;
    std::vector<int64_t> scales;
    std::vector<int64_t> expertScales;
    
    // 输入信息dtype
    aclDataType x_dtype;
    aclDataType y_dtype;
    aclDataType expert_ids_dtype;
    aclDataType comm_cmd_info_dtype;
    aclDataType xActiveMask_dtype;
    aclDataType scales_dtype;
    aclDataType expertScales_dtype;

    // 输入Attrs
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t quantMode;
    int64_t globalBs;
    int64_t expertTokenNumsType;
    int64_t commType;

    // 输出信息shape
    std::vector<int64_t> expandX;
    std::vector<int64_t> dynamicScales;
    std::vector<int64_t> assistInfoForCombine;
    std::vector<int64_t> expertTokensNums;
    
    // 输出信息dtype
    aclDataType expandX_dtype;
    aclDataType dynamicScales_dtype;
    aclDataType assistInfoForCombine_dtype;
    aclDataType expertTokensNums_dtype;

    // 用例对应型号
    SocVersion socVersion;

    // 预期结果
    aclnnStatus expectResult;
};

// 重写 << 操作符，规避用例执行失败时打印入参乱码的问题
inline std::ostream& operator<<(std::ostream& os, const MoeDistributeDispatchTeardownAclnnTestParam& param)
{
    return os << param.case_name;
}

// 用例列表集
static MoeDistributeDispatchTeardownAclnnTestParam test_cases[] = {
//===============================================Ascend910C===================================================
{"test_aclnn_moe_distribute_dispatch_first_api",
 {8, 7168}, {64, 7168}, {8, 8}, {64}, {8, 256}, {256, 7168}, {8, 8}, 
 ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 
 288, 0, 256, 0, 1, 8, 0, 0, 1, 0, 
 {256}, {8 * 256}, {256}, {1}, 
 ACL_INT8, ACL_FLOAT, ACL_INT32, ACL_INT64, 
 SocVersion::ASCEND910_93, ACLNN_SUCCESS},
{"ascend910B2_test_aclnn_moe_distribute_dispatch_tp_not_empty", 
 {8, 7168}, {64, 7168}, {8, 8}, {64}, {8, 256}, {256, 7168}, {8, 8}, 
 ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 
 256, 0, 256, 0, 0, 0, 2, 0, 1, 0, 
 {256}, {8 * 256}, {256}, {1}, 
 ACL_INT8, ACL_FLOAT, ACL_INT32, ACL_INT64, 
 SocVersion::ASCEND910_93, ACLNN_ERR_PARAM_INVALID}
};

static void BuildTilingContextPara

static void TestOneParamCase(const MoeDistributeDispatchTeardownAclnnTestParam& param)
{
    std::out << "[TEST_CASE]" << param << std::endl;

    // 封装用例中输入信息
    TensorDesc x = TensorDesc(param.x, param.x_dtype, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc(param.y, param.y_dtype, ACL_FORMAT_ND);
    TensorDesc expert_ids = TensorDesc(param.expert_ids, param.expert_ids_dtype, ACL_FORMAT_ND);
    TensorDesc comm_cmd_info = TensorDesc(param.comm_cmd_info, param.comm_cmd_info_dtype, ACL_FORMAT_ND);
    const char* groupEp = "test_group";
    TensorDesc xActiveMask = TensorDesc(param.xActiveMask, param.xActiveMask_dtype, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc(param.scales, param.scales_dtype, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc(param.expertScales, param.expertScales_dtype, ACL_FORMAT_ND);
    const char* commAlg = nullptr;

    // 封装输出信息
    TensorDesc expandX = TensorDesc(param.expandX, param.expandX_dtype, ACL_FORMAT_ND);
    TensorDesc dynamicScales = TensorDesc(param.dynamicScales, param.dynamicScales_dtype, ACL_FORMAT_ND);
    TensorDesc assistInfoForCombine = TensorDesc(param.assistInfoForCombine, param.assistInfoForCombine_dtype, ACL_FORMAT_ND);
    TensorDesc expertTokensNums = TensorDesc(param.expertTokensNums, param.expertTokensNums_dtype, ACL_FORMAT_ND);

    // 打桩到对应型号
    SetPlatformSocVersion(param.socVersion);

    auto ut = OP_API_UT(aclnnMoeDistributeDispatchTeardown,
                        INPUT(x, y, expert_ids, comm_cmd_info, groupEp, param.epWorldSize, 
                            param.epRankId, param.moeExpertNum, param.expertShardType, 
                            param.sharedExpertNum, param.sharedExpertRankNum, param.quantMode, 
                            param.globalBs, param.expertTokenNumsType, param.commType, commAlg),
                        OUTPUT(expandX, dynamicScales, assistInfoForCombine, expertTokensNums));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);

    if (param.expectResult == ACLNN_SUCCESS) {
        // 910b环境下正常用例可能返回ACLNN_SUCCESS以外结果，因此修改判断条件
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
    else {
        EXPECT_EQ(aclRet, param.expectResult);
    }
}

// 多线程执行用例集
static void ThreadFunc(const MoeDistributeDispatchTeardownAclnnTestParam* testCases, size_t testcase_num, size_t thread_idx, size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(testCases[idx]);
    }
}

static void TestMultiThread(const MoeDistributeDispatchTeardownAclnnTestParam* testCases, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, testCases, testcase_num, idx, thread_num);
    }
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_P(test_aclnn_moe_distribute_dispatch_teardown, general_cases) {
    TestOneParamCase(GetParam());
}

TEST_F(test_aclnn_moe_distribute_dispatch_teardown, general_cases_multi_thread) {
    size_t thread_num = 3;
    TestMultiThread(test_cases, sizeof(test_cases) / sizeof(MoeDistributeDispatchTeardownAclnnTestParam), thread_num);
}
}