/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_matmul_allto_all.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class test_aclnn_matmul_allto_all : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_95);
        cout << "test_aclnn_matmul_allto_all SetUp" << endl;
    }

    static void TearDownTestCase() { cout << "test_aclnn_matmul_allto_all TearDown" << endl; }
};

// ut用例结构体
struct MatmulAlltoAllAclnnTestParam {
    // 用例名
    string case_name;
    // 数据形状
    vector<int64_t> x1_shape; // x1数据shape，正常为（BS，H1）
    vector<int64_t> x2_shape; // x2数据shape，正常为（H1，H2）
    vector<int64_t> bias_shape; // bias数据shape，正常为（H2）
    vector<int64_t> output_shape; // output数据shape，正常为（BS，H2）
    // 数据类型
    aclDataType x1_dtype; // x1数据dtype，仅支持bfloat16和float16
    aclDataType x2_dtype; // x2数据dtype，仅支持bfloat16和float16
    aclDataType bias_dtype; // bias数据dtype，仅支持float32
    aclDataType output_dtype; // 输出数据dtype，支持bfloat16、float16和float32
    // 数据格式
    aclFormat x1_format; // x1数据format，仅支持ND
    aclFormat x2_format; // x2数据format，仅支持ND
    aclFormat bias_format; // bias数据format，仅支持ND
    aclFormat output_format; // output数据format，仅支持ND
    // 其它属性
    vector<int64_t> alltoAllAxesOptional; // alltoall数据交换的方向，只能为空或者[-2,-1]
    char* group; // 通信域标识，字符串，长度要求（0，128）
    bool transposeX1; // x1是否转置，现不支持为true
    bool transposeX2; // x2是否转置，为true时x2shape为（H2，H1）
    // ut用例期望返回状态
    aclnnStatus aclnn_status; //期望状态
};

static MatmulAlltoAllAclnnTestParam cases_params[] = {
    // 正常用例 12条
    // ========================bfloat16 系列（6条）========================
    // x和output为bfloat16，按bias分组
    // 1. Bias=BF16 (2)
    {"test_aclnn_matmul_allto_all_bf16_biasbf16_nd_notrans_01", {256, 128}, {128, 256}, {256}, {256, 256}, 
     ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"test_aclnn_matmul_allto_all_bf16_biasbf16_nd_trans_03", {256, 128}, {256, 128}, {256}, {256, 256}, 
     ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},

    // 2. Bias=FLOAT32 (2)
    {"test_aclnn_matmul_allto_all_bf16_biasf32_nd_notrans_09", {256, 128}, {128, 256}, {256}, {256, 256}, 
     ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"test_aclnn_matmul_allto_all_bf16_biasf32_nd_trans_11", {256, 128}, {256, 128}, {256}, {256, 256}, 
     ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},

    // 3. Bias=Null (2) 
    {"test_aclnn_matmul_allto_all_bf16_biasnull_nd_notrans_17", {256, 128}, {128, 256}, {}, {256, 256},
     ACL_BF16, ACL_BF16, ACL_DT_UNDEFINED, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"test_aclnn_matmul_allto_all_bf16_biasnull_nd_trans_19", {256, 128}, {256, 128}, {}, {256, 256},
     ACL_BF16, ACL_BF16, ACL_DT_UNDEFINED, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},

    // ========================float16 系列（6条）========================
    // x和output为float16，按bias分组
    // 1. Bias=FP16 (2)
    {"test_aclnn_matmul_allto_all_fp16_biasfp16_nd_notrans_01", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"test_aclnn_matmul_allto_all_fp16_biasfp16_nd_trans_03", {256, 128}, {256, 128}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},
    
    // 2. Bias=FLOAT32 (2)
    {"test_aclnn_matmul_allto_all_fp16_biasf32_nd_notrans_09", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"test_aclnn_matmul_allto_all_fp16_biasf32_nd_trans_11", {256, 128}, {256, 128}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},

    // 3. Bias=Null (2) 
    {"test_aclnn_matmul_allto_all_fp16_biasnull_nd_notrans_17", {256, 128}, {128, 256}, {}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_DT_UNDEFINED, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"test_aclnn_matmul_allto_all_fp16_biasnull_nd_trans_19", {256, 128}, {256, 128}, {}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_DT_UNDEFINED, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},

    // 异常用例 13条
    // 1. x1 dtype不合法(ACL_INT8)
    {"test_aclnn_matmul_allto_all_x1dtype_invalid_00", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_INT8, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},

    // 2. x2 dtype不合法 (ACL_UINT8)
    {"test_aclnn_matmul_allto_all_x2dtype_invalid_01", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_UINT8, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},

    // 3. bias dtype不合法 不等于xdtype或float32(ACL_BF16)
    {"test_aclnn_matmul_allto_all_biasdtype_invalid_02", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_BF16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},

    // 4. output dtype不合法 (2)
    // 4.1 不等于x dtype (ACL_FLOAT)
    {"test_aclnn_matmul_allto_all_outdtype_mismatch_03", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 4.2 等于不支持类型 (ACL_FLOAT8_E5M2)
    {"test_aclnn_matmul_allto_all_outdtype_unsupport_04", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT8_E5M2,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},

    // 7. 空tensor (1)
    // 7.1 x2有维度为0
    {"test_aclnn_matmul_allto_all_x2empty_10", {256, 128}, {128, 0}, {0}, {256, 0},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},

    // 8. format为私有格式(4)
    {"test_aclnn_matmul_allto_all_private_fmt1_12", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_matmul_allto_all_private_fmt2_13", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_NCL, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_matmul_allto_all_private_fmt3_14", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_matmul_allto_all_private_fmt4_15", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_NCL,
     {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},

    // 9. AlltoAllAxes不合法
    {"test_aclnn_matmul_allto_all_invalid_axes_16", {256, 128}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {3, 2, 1}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},

    // 10. group不合法 (待补充)

    // 11. transposeX1=true
    {"test_aclnn_matmul_allto_all_transx1_19", {128, 256}, {128, 256}, {256}, {256, 256},
     ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_matmul_allto_all", true, false, ACLNN_ERR_PARAM_INVALID}

};

static void TestOneParamCase(const MatmulAlltoAllAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    // 从结构体list中获取实际用例属性
    vector<int64_t> x1Shape = param.x1_shape;
    vector<int64_t> x2Shape = param.x2_shape;
    vector<int64_t> biasShape = param.bias_shape;
    vector<int64_t> outputShape = param.output_shape;
    aclDataType x1Dtype = param.x1_dtype;
    aclDataType x2Dtype = param.x2_dtype;
    aclDataType biasDtype = param.bias_dtype;
    aclDataType outputDtype = param.output_dtype;
    aclFormat x1Format = param.x1_format;
    aclFormat x2Format = param.x2_format;
    aclFormat biasFormat = param.bias_format;
    aclFormat outputFormat = param.output_format;
    vector<int64_t> axes_acl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axes_acl.data(), axes_acl.size());
    const char* group = param.group;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnn_status;
    TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1Format);
    TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2Format);
    TensorDesc bias = TensorDesc(biasShape, biasDtype, biasFormat);
    TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    bool emptyBias = biasShape.empty();
    if (emptyBias) {
        auto ut = OP_API_UT(aclnnMatmulAlltoAll,
                        INPUT(x1, x2, nullptr, alltoAllAxesOptional, group, transposeX1, transposeX2),
                        OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        EXPECT_EQ(aclRet, retStatus);
    } else {
        TensorDesc bias = TensorDesc(biasShape, biasDtype, biasFormat);
        auto ut = OP_API_UT(aclnnMatmulAlltoAll,
                        INPUT(x1, x2, bias, alltoAllAxesOptional, group, transposeX1, transposeX2),
                        OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.case_name << std::endl;
}

TEST_F(test_aclnn_matmul_allto_all, cases_params)
{
    if (std::size(cases_params) != 0) {
        uint64_t numCases = sizeof(cases_params) / sizeof(cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(cases_params[idx]);
        }
    }
}
