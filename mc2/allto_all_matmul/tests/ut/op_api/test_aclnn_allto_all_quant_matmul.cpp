/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_allto_all_quant_matmul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class TestAclnnAlltoAllQuantMatmul : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_95);
        cout << "TestAclnnAlltoAllQuantMatmul SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "TestAclnnAlltoAllQuantMatmul TearDown" << endl;
    }
};

// ut用例结构体
struct AlltoAllQuantMatmulAclnnTestParam {
    // 用例名
    string caseName;
    // 通信域卡数，ut测试默认为2
    int worldSize;
    // 量化模式
    int64_t x1Quantmode; // x1量化模式
    int64_t x2Quantmode; // x2量化模式
    // 数据形状
    vector<int64_t> x1Shape; // x1数据shape，正常为（BS，H）
    vector<int64_t> x2Shape; // x2数据shape，正常为（H * world_size，N）
    vector<int64_t> biasShape; // bias数据shape，正常为（N）
    vector<int64_t> x1ScaleOptionalShape; // x1ScaleOptional数据shape，正常为（BS）
    vector<int64_t> x2ScaleShape; // x2scales数据shape，正常为（N）
    vector<int64_t> outputShape; // output数据shape，正常为（BS / world_size，N）
    vector<int64_t> alltoalloutputShape; // alltoalloutput数据shape，正常为（BS / ranksize，H * ranksize）
    // 数据类型
    aclDataType x1Dtype; // x1数据dtype，仅支持bfloat16和float16
    aclDataType x2Dtype; // x2数据dtype，仅支持float8_e5m2和float8_e4m3fn
    aclDataType biasDtype; // bias数据dtype，仅支持float32
    aclDataType x1ScaleOptionalDtype; // x1ScaleOptional数据dtype，仅支持bfloat16和float16，要求和x1Dtype一致
    aclDataType x2ScaleDtype; // x2scales数据dtype，仅支持float32
    aclDataType outputDtype; // output数据dtype，支持bfloat16、float16和float32
    aclDataType alltoalloutputDtype; // alltoalloutput数据dtype，仅支持bfloat16和float16，要求和x1Dtype一致
    // 数据格式
    aclFormat x1Format; // x1数据format，仅支持ND
    aclFormat x2Format; // x2数据format，仅支持ND
    aclFormat biasFormat; // bias数据format，仅支持ND
    aclFormat x1ScaleOptionalFormat; // x1ScaleOptional数据format，仅支持ND
    aclFormat x2ScaleFormat; // x2Scale数据format，仅支持ND
    aclFormat outputFormat; // output数据format，仅支持ND
    aclFormat alltoalloutputFormat; // alltoalloutputoutput数据format，仅支持ND
    // 其它属性
    int64_t x1Quantdtype; // x1量化数据类型，仅支持配置35（表示ACL_FLOAT8_E5M2）或36（表示ACL_FLOAT8_E4M3FN）
    vector<int64_t> alltoAllAxesOptional; // alltoall数据交换的方向，只能为空或者[-2,-1]
    char* group; // 通信域标识，字符串，长度要求（0，128）
    bool transposeX1; // x1是否转置，现不支持为true
    bool transposeX2; // x2是否转置，为true时x2shape为（H2，H1）
    // ut用例期望返回状态
    aclnnStatus aclnnStatusUT; //期望状态
};

static AlltoAllQuantMatmulAclnnTestParam g_casesParams[] = {
    // 正常用例 192条，caseid按照[算子名-x1-x2-bias-x1scale-x2scale-output-alltoallout-format-transpose-x1quantdtype-id]构成
    // 等待补充
    {"AclnnAlltoAllQuantMatmul-bf16-e4m3-f32-bf16-f32-bf16-bf16-nd-notrans-35-01",
        2, 7, 2, {256, 64}, {128, 256}, {256}, {256}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS},
    {"AclnnAlltoAllQuantMatmul-bf16-e4m3-f32-bf16-f32-bf16-bf16-nd-notrans-35-01",
        2, 7, 2, {256, 64}, {256, 128}, {256}, {256}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},

    // 异常用例 很多很多条，caseid按照[error-算子名-异常原因-id]构成
    // 等待补充
    {"error-AclnnAlltoAllQuantMatmul-x1dtype_invalid-01",
        2, 7, 2, {256, 64}, {128, 256}, {256}, {256}, {256}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnAlltoAllQuantMatmul-x2dtype_invalid-02",
        2, 7, 2, {256, 64}, {128, 256}, {256}, {256}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID},
};

static void TestOneParamCase(const AlltoAllQuantMatmulAclnnTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    // 从结构体list中获取实际用例属性
    int64_t x1quantmode = param.x1Quantmode;
    int64_t x2quantmode = param.x2Quantmode;
    vector<int64_t> x1Shape = param.x1Shape;
    vector<int64_t> x2Shape = param.x2Shape;
    vector<int64_t> biasShape = param.biasShape;
    vector<int64_t> x1scalesShape = param.x1ScaleOptionalShape;
    vector<int64_t> x2scalesShape = param.x2ScaleShape;
    vector<int64_t> outputShape = param.outputShape;
    vector<int64_t> alltoalloutShape = param.alltoalloutputShape;
    aclDataType x1Dtype = param.x1Dtype;
    aclDataType x2Dtype = param.x2Dtype;
    aclDataType biasDtype = param.biasDtype;
    aclDataType x1scalesDtype = param.x1ScaleOptionalDtype;
    aclDataType x2scalesDtype = param.x2ScaleDtype;
    aclDataType outputDtype = param.outputDtype;
    aclDataType alltoalloutDtype = param.alltoalloutputDtype;
    aclFormat x1Format = param.x1Format;
    aclFormat x2Format = param.x2Format;
    aclFormat biasFormat = param.biasFormat;
    aclFormat x1ScaleFormat = param.x1ScaleOptionalFormat;
    aclFormat x2ScaleFormat = param.x2ScaleFormat;
    aclFormat outputFormat = param.outputFormat;
    aclFormat alltoalloutFormat = param.alltoalloutputFormat;
    int64_t x1quantdtype = param.x1Quantdtype;
    vector<int64_t> axesAcl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axesAcl.data(), axesAcl.size());
    const char* group = param.group;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnnStatusUT;
    TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1Format);
    TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2Format);
    TensorDesc x2scales = TensorDesc(x2scalesShape, x2scalesDtype, x2ScaleFormat);
    TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
    // 三个可能为空指针的，需要特殊处理
    TensorDesc bias = TensorDesc(biasShape, biasDtype, biasFormat);
    TensorDesc alltoallout = TensorDesc(alltoalloutShape, alltoalloutDtype, alltoalloutFormat);
    TensorDesc x1scales = TensorDesc(x1scalesShape, x1scalesDtype, x1ScaleFormat);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ut = OP_API_UT(aclnnAlltoAllQuantMatmul,
                INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, nullptr, group, alltoAllAxesOptional,
                    x1quantmode, x2quantmode, 0, -1, x1quantdtype, 0, transposeX1, transposeX2),
                OUTPUT(output, alltoallout));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, retStatus);
    std::cout << "end case " <<  param.caseName << std::endl;
}

TEST_F(TestAclnnAlltoAllQuantMatmul, g_casesParams)
{
    if (std::size(g_casesParams) != 0) {
        uint64_t numCases = sizeof(g_casesParams) / sizeof(g_casesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_casesParams[idx]);
        }
    }
}