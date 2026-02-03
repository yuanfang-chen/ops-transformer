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
#include "../../../op_api/aclnn_quant_allto_allv_grouped_mat_mul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_quant_allto_allv_grouped_mat_mul_test : public testing::Test {
 protected:
  static void SetUpTestCase()
  {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
	cout << "l2_quant_allto_allv_grouped_mat_mul_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "l2_quant_allto_allv_grouped_mat_mul_test TearDown" << endl; }
};

struct QuantAlltoAllvGroupedMatmulAclnnTestParam {
    // 用例名
    string case_name;
    // gmm
    vector<int64_t> gmmX;
    vector<int64_t> gmmWeight;
	aclDataType gmmX_dtype;
    aclDataType gmmWeight_dtype;
	aclFormat gmmX_format;
    aclFormat gmmWeight_format;

	// mm
	vector<int64_t> mmX;
    vector<int64_t> mmWeight;
	aclDataType mmX_dtype;
    aclDataType mmWeight_dtype;
	aclFormat mmX_format;
    aclFormat mmWeight_format;

	// gmmY
	vector<int64_t> gmmY;
	aclDataType gmmY_dtype;
	aclFormat gmmY_format;

	// mmY
    vector<int64_t> mmYOptional;
	aclDataType mmYOptional_dtype;
	aclFormat mmYOptional_format;

    char* group;
	bool send;
	bool recv;
    bool transGmmWeight;
    bool transMmWeight;
	bool permuteOutFlag;
    aclnnStatus aclnn_status;
};

static QuantAlltoAllvGroupedMatmulAclnnTestParam quant_cases_params[] = {
	// float16 正常用例
    {"test_quant_allto_allv_grouped_mat_mul_test_float16_00",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_SUCCESS},

	// 异常 sendCounts null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_01",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", true, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// recvCounts null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_02",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, true, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// gmmx null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_03",
		{}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// gmmWeight null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_04",
		{4096, 7168}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// gmmY null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_05",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// group ep null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_06",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"", false, false, false, false, false, ACLNN_ERR_PARAM_NULLPTR},

	// group ep invalid
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_07",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group_"
		"test_allto_allv_grouped_mat_mul_ep_group_"
		"test_allto_allv_grouped_mat_mul_ep_group_"
		"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// mmx not_null mmweight null mmy null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_08",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1024, 7168}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// mmx null mmweight not_null mmy null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_09",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {7168,1024}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// mmx null mmweight null mmy not_null
	{"test_quant_allto_allv_grouped_mat_mul_test_float16_10",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{1024}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"test_quant_allto_allv_grouped_mat_mul_test_float16_11",
		{4096, 7168}, {4, 7168, 4096}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {}, ACL_FLOAT16, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096,4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, true, ACLNN_ERR_PARAM_NULLPTR}

};

static void TestQuantParamCase(const QuantAlltoAllvGroupedMatmulAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    // 从结构体list中获取实际用例属性
    TensorDesc gmmX_ = TensorDesc(param.gmmX, param.gmmX_dtype, param.gmmX_format);
    TensorDesc gmmWeight_ = TensorDesc(param.gmmWeight, param.gmmWeight_dtype, param.gmmWeight_format);
	TensorDesc mmX_ = TensorDesc(param.mmX, param.mmX_dtype, param.mmX_format);
    TensorDesc mmWeight_ = TensorDesc(param.mmWeight, param.mmWeight_dtype, param.mmWeight_format);
    TensorDesc gmmY_ = TensorDesc(param.gmmY, param.gmmY_dtype, param.gmmY_format);
	TensorDesc mmY_ = TensorDesc(param.mmYOptional, param.mmYOptional_dtype, param.mmYOptional_format);

    const char* group = param.group;
	bool send_ = param.send;
	bool recv_ = param.recv;
    bool transGmmWeight = param.transGmmWeight;
    bool transMmWeight = param.transMmWeight;
	bool permuteOutFlag = param.permuteOutFlag;

    aclnnStatus retStatus = param.aclnn_status;

	constexpr int64_t epWorldSize = 8;
	constexpr int64_t BS = 4096;
	constexpr int64_t K = 2;
	constexpr int64_t H = 7168;
	constexpr int64_t e = 4;
	std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
	std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
	aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
	aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

	if(send_) sendCounts = nullptr;
	if(recv_) recvCounts = nullptr;

    auto ut = OP_API_UT(aclnnQuantAlltoAllvGroupedMatMul,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, mmX_, mmWeight_, nullptr,
                                 nullptr, nullptr, nullptr, 1, 1, 1, 1, 1, 1, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
								 transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_, mmY_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, retStatus);
    std::cout << "end case " <<  param.case_name << std::endl;
}

TEST_F(l2_quant_allto_allv_grouped_mat_mul_test, quant_cases_params)
{
    if (std::size(quant_cases_params) != 0) {
        uint64_t numCases = sizeof(quant_cases_params) / sizeof(quant_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestQuantParamCase(quant_cases_params[idx]);
        }
    }
}
