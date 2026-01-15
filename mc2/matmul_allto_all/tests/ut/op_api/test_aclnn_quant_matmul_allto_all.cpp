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
#include "../../../op_api/aclnn_quant_matmul_allto_all.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class test_aclnn_quant_matmul_allto_all : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_95);
        cout << "test_aclnn_quant_matmul_allto_all SetUp" << endl;
    }

    static void TearDownTestCase() { cout << "test_aclnn_matmul_allto_all TearDown" << endl; }
};

struct QuantMatmulAlltoAllAclnnTestParam {
    // 用例名
    string case_name;
    // 数据形状
    int64_t x1_quantmode; // x1量化模式
    int64_t x2_quantmode; // x2量化模式
    vector<int64_t> x1_shape; // x1数据shape
    vector<int64_t> x2_shape; // x2数据shape
    vector<int64_t> bias_shape; // bias数据shape
    vector<int64_t> x1_scale_shape; // x1scales数据shape
    vector<int64_t> x2_scale_shape; // x2scales数据shape
    vector<int64_t> comm_scale_optional_shape; // 低比特通信的量化系数，暂不支持。
    vector<int64_t> x1_offset_optional_shape; // 左矩阵的量化偏置，暂不支持。
    vector<int64_t> x2_offset_optional_shape; // 右矩阵的量化偏置，暂不支持。
    vector<int64_t> output_shape; // output数据shape
    // 数据类型
    aclDataType x1_dtype; // x1数据dtype
    aclDataType x2_dtype; // x2数据dtype
    aclDataType bias_dtype; // bias数据dtype
    aclDataType x1_scale_dtype; // x1scales数据dtype
    aclDataType x2_scale_dtype; // x2scales数据dtype
    aclDataType output_dtype; // 输出数据dtype
    // 数据格式
    aclFormat x1_format; // x1数据format，仅支持ND
    aclFormat x2_format; // x2数据format，仅支持ND
    aclFormat bias_format; // bias数据format，仅支持ND
    aclFormat x1_scale_format; // x1Scale数据format，仅支持ND
    aclFormat x2_scale_format; // x2Scale数据format，仅支持ND
    aclFormat output_format; // output数据format，仅支持ND
    // 其它属性
    vector<int64_t> alltoAllAxesOptional; // alltoall数据交换的方向，只能为空或者[-2,-1]
    char* group; // 通信域标识，字符串，长度要求（0，128）
    int64_t commQuantMode; // 低比特通信的量化模式，预留参数，当前仅支持配置为0，表示不量化。
    int64_t commQuantDtype; // 低比特通信的量化类型，预留参数，当前仅支持配置为-1，表示ACL_DT_UNDEFINED。
    int64_t groupSize; // 用于Matmul计算三个方向上的量化分组大小，预留参数，T-C量化模式下仅支持配置为0，取值不生效。
    bool transposeX1; // x1是否转置，现不支持为true
    bool transposeX2; // x2是否转置，为true时x2shape为（H2，H1）
    aclnnStatus aclnn_status; //期望状态
};

static QuantMatmulAlltoAllAclnnTestParam quant_cases_params[] = {
    // 正常用例 24条
    // ========================float8_e4m3fn 系列（12条）========================
    // ------------------------输出float16（4条）------------------------
    // 基础组合：bias有/无 + trans=false/true
    {"test_aclnn_quant_matmul_allto_all_e4m3_float16_02", 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_float16_04", 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_float16_10", 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_float16_12", 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    // ------------------------输出bfloat16（4条）------------------------
    {"test_aclnn_quant_matmul_allto_all_e4m3_bfloat16_18", 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_bfloat16_20", 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_bfloat16_26", 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_bfloat16_28", 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    // ------------------------输出float32（4条）------------------------
    {"test_aclnn_quant_matmul_allto_all_e4m3_float32_34", 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_float32_36", 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_float32_42", 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e4m3_float32_44", 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    // ========================float8_e5m2 系列（12条）========================
    // ------------------------输出float16（4条）------------------------
    // 基础组合：bias有/无 + trans=false/true
    {"test_aclnn_quant_matmul_allto_all_e5m2_float16_02", 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_float16_04", 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_float16_10", 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_float16_12", 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    // ------------------------输出bfloat16（4条）------------------------
    {"test_aclnn_quant_matmul_allto_all_e5m2_bfloat16_18", 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_bfloat16_20", 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_bfloat16_26", 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_bfloat16_28", 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_BF16,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    // ------------------------输出float32（4条）------------------------
    {"test_aclnn_quant_matmul_allto_all_e5m2_float32_34", 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_float32_36", 3, 2, {256, 128}, {128, 256}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_float32_42", 3, 2, {256, 128}, {256, 128}, {256}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    {"test_aclnn_quant_matmul_allto_all_e5m2_float32_44", 3, 2, {256, 128}, {256, 128}, {}, {256}, {256}, {}, {}, {}, {256, 256},
     ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_DT_UNDEFINED, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, true, ACLNN_SUCCESS},

    // 异常用例
    // 示例1，x1的dtype不符合要求，其它如正常示例1
    {"test_aclnn_quant_matmul_allto_all_false_01", 3, 2, {256, 128}, {128, 256}, {256}, {256}, {256}, {}, {}, {}, {256, 256}, 
     ACL_FLOAT16, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 
     ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, 
     {-1, -2}, "ut_test_quant_matmul_allto_all", 0, 28, 0, false, false, ACLNN_ERR_PARAM_INVALID}
};

static void TestQuantOneParamCase(const QuantMatmulAlltoAllAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    // 从结构体list中获取实际用例属性
    int64_t x1quantmode = param.x1_quantmode;
    int64_t x2quantmode = param.x2_quantmode;
    vector<int64_t> x1Shape = param.x1_shape;
    vector<int64_t> x2Shape = param.x2_shape;
    vector<int64_t> biasShape = param.bias_shape;
    vector<int64_t> x1scalesShape = param.x1_scale_shape;
    vector<int64_t> x2scalesShape = param.x2_scale_shape;
    vector<int64_t> comm_scale_optional_shape = param.comm_scale_optional_shape;
    vector<int64_t> x1_offset_optional_shape = param.x1_offset_optional_shape;
    vector<int64_t> x2_offset_optional_shape = param.x2_offset_optional_shape;
    vector<int64_t> outputShape = param.output_shape;
    aclDataType x1Dtype = param.x1_dtype;
    aclDataType x2Dtype = param.x2_dtype;
    aclDataType biasDtype = param.bias_dtype;
    aclDataType x1scalesDtype = param.x1_scale_dtype;
    aclDataType x2scalesDtype = param.x2_scale_dtype;
    aclDataType outputDtype = param.output_dtype;
    aclFormat x1_format = param.x1_format;
    aclFormat x2_format = param.x2_format;
    aclFormat bias_format = param.bias_format;
    aclFormat x1_scale_format = param.x1_scale_format;
    aclFormat x2_scale_format = param.x2_scale_format;
    aclFormat output_format = param.output_format;
    vector<int64_t> axes_acl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axes_acl.data(), axes_acl.size());
    const char* group = param.group;
    int64_t commQuantMode = param.commQuantMode;
    int64_t commQuantDtype = param.commQuantDtype;
    int64_t groupSize = param.groupSize;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnn_status;
    TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1_format);
    TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2_format);
    TensorDesc x1scales = TensorDesc(x1scalesShape, x1scalesDtype, x1_scale_format);
    TensorDesc x2scales = TensorDesc(x2scalesShape, x2scalesDtype, x2_scale_format);
    TensorDesc output = TensorDesc(outputShape, outputDtype, output_format);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    bool emptyBias = biasShape.empty();
    if (emptyBias) {
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
                            INPUT(x1, x2, nullptr, x1scales, x2scales, nullptr, nullptr, nullptr, alltoAllAxesOptional, group,
                                  x1quantmode, x2quantmode, commQuantMode, commQuantDtype, groupSize, transposeX1, transposeX2),
                            OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        EXPECT_EQ(aclRet, retStatus);
    } else {
        TensorDesc bias = TensorDesc(biasShape, biasDtype, bias_format);
        auto ut = OP_API_UT(aclnnQuantMatmulAlltoAll,
                            INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, nullptr, alltoAllAxesOptional, group,
                                  x1quantmode, x2quantmode, commQuantMode, commQuantDtype, groupSize, transposeX1, transposeX2),
                            OUTPUT(output));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.case_name << std::endl;
}

TEST_F(test_aclnn_quant_matmul_allto_all, quant_cases_params)
{
    if (std::size(quant_cases_params) != 0) {
        uint64_t numCases = sizeof(quant_cases_params) / sizeof(quant_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestQuantOneParamCase(quant_cases_params[idx]);
        }
    }
}