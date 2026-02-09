/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_allto_allv_quant_grouped_mat_mul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_quant_allto_allv_grouped_mat_mul_test : public testing::Test {
 protected:
  static void SetUpTestCase()
  {
    op::(NpuArch::DAV_3510);
	cout << "l2_quant_allto_allv_grouped_mat_mul_test SetUp" << endl;
  }

  static void TearDownTestCase()
  {
	op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
	cout << "l2_quant_allto_allv_grouped_mat_mul_test TearDown" << endl;
  }
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

	// gmm scale
	vector<int64_t> gmmXScale;
    vector<int64_t> gmmWeightScale;
	aclDataType gmmXScale_dtype;
    aclDataType gmmWeightScale_dtype;
	aclFormat gmmXScale_format;
    aclFormat gmmWeightScale_format;

	// mm
	vector<int64_t> mmX;
    vector<int64_t> mmWeight;
	aclDataType mmX_dtype;
    aclDataType mmWeight_dtype;
	aclFormat mmX_format;
    aclFormat mmWeight_format;

	// mm scale
	vector<int64_t> mmXScale;
    vector<int64_t> mmWeightScale;
	aclDataType mmXScale_dtype;
    aclDataType mmWeightScale_dtype;
	aclFormat mmXScale_format;
    aclFormat mmWeightScale_format;

	// gmmY
	vector<int64_t> gmmY;
	aclDataType gmmY_dtype;
	aclFormat gmmY_format;

	// mmY
    vector<int64_t> mmYOptional;
	aclDataType mmYOptional_dtype;
	aclFormat mmYOptional_format;

	// permuteOut
	// vector<int64_t> permuteOutOptional;
	// aclDataType permuteOutOptional_dtype;
	// aclFormat permuteOutOptional_format;

	int64_t gmmX_quant_mode;
	int64_t gmmWeight_quant_mode;
	int64_t mmX_quant_mode;
	int64_t mmWeight_quant_mode;
	int64_t epWorldSize;
	int64_t BS;
	int64_t K;
	int64_t H;
	int64_t e;
    char* group;
	bool send;
	bool recv;
    bool transGmmWeight;
    bool transMmWeight;
	bool permuteOutFlag;
    aclnnStatus aclnn_status;
};

static QuantAlltoAllvGroupedMatmulAclnnTestParam quant_cases_params[] = {
	// hifloat8 正常用例
    {"AclnnAlltoAllvQuantGMM_hifloat8_normal_00",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_SUCCESS},
	
	// 数据类型异常 10条
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_01",
		{8192, 7168}, {4, 7168, 4096}, ACL_FLOAT, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_02",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_03",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT16, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_04",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_05",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_FLOAT, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_06",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_07",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT16, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_08",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_09",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_dtype_10",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	// 数据格式异常 10条
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_01",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_02",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_03",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_04",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_05",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_06",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_07",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_08",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_09",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_Z,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_nd_10",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_Z,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	// Quantmode异常:4条
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_quant_mode_01",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		0, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_quant_mode_02",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 0, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_quant_mode_03",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 0, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_quant_mode_04",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 0, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// scale异常：4条
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_scale_01",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_scale_02",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_scale_03",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_scale_04",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	// shape 异常
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_01",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_02",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{1023}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_03",
		{}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_04",
		{8192, 7168}, {}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_05",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_06",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_07",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_08",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_09",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_10",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{256, 7168}, {7168, 256}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{256}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_11",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{8192, 7168}, {7168, 8192}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{8192}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_12",
		{8192, 7168}, {33, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_13",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{2}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_14",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {2}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_15",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{2}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_16",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {2}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_17",
		{52428801, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {2}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{52428801, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_18",
		{8192, 65537}, {4, 65537, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_19",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 12289}, {12289, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_20",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 65537}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 65537}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},
	
	// epworldsize error
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_shape_21",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 3, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// 异常 sendCounts null
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_sendcount",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", true, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// recvCounts null
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_recvcount",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, true, false, false, false, ACLNN_ERR_PARAM_INVALID},

	
	// group ep null
	{"AclnnAlltoAllvQuantGMM_hifloat8_no_ep_null",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},

	// group ep invalid
	{"AclnnAlltoAllvQuantGMM_hifloat8_ep_invaild",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group_"
		"test_allto_allv_grouped_mat_mul_ep_group_"
		"test_allto_allv_grouped_mat_mul_ep_group_"
		"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, false, ACLNN_ERR_PARAM_INVALID},


	{"AclnnAlltoAllvQuantGMM_hifloat8_no_permute",
		{8192, 7168}, {4, 7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
	 	{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{4096, 7168}, {7168, 4096}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FORMAT_ND, ACL_FORMAT_ND,
		{1}, {1}, ACL_FLOAT, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND,
     	{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
	 	{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND,
		1, 1, 1, 1, 8, 4096, 2, 7168, 4,
     	"test_allto_allv_grouped_mat_mul_ep_group", false, false, false, false, true, ACLNN_ERR_PARAM_INVALID}

};

static void TestQuantParamCase(const QuantAlltoAllvGroupedMatmulAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    // 从结构体list中获取实际用例属性
    TensorDesc gmmX_ = TensorDesc(param.gmmX, param.gmmX_dtype, param.gmmX_format);
    TensorDesc gmmWeight_ = TensorDesc(param.gmmWeight, param.gmmWeight_dtype, param.gmmWeight_format);
	TensorDesc gmmXScale_ = TensorDesc(param.gmmXScale, param.gmmXScale_dtype, param.gmmXScale_format);
	TensorDesc gmmWeightScale_ = TensorDesc(param.gmmWeightScale, param.gmmWeightScale_dtype, param.gmmWeightScale_format);
	TensorDesc mmX_ = TensorDesc(param.mmX, param.mmX_dtype, param.mmX_format);
    TensorDesc mmWeight_ = TensorDesc(param.mmWeight, param.mmWeight_dtype, param.mmWeight_format);
	TensorDesc mmXScale_ = TensorDesc(param.mmXScale, param.mmXScale_dtype, param.mmXScale_format);
	TensorDesc mmWeightScale_ = TensorDesc(param.mmWeightScale, param.mmWeightScale_dtype, param.mmWeightScale_format);
    TensorDesc gmmY_ = TensorDesc(param.gmmY, param.gmmY_dtype, param.gmmY_format);
	TensorDesc mmY_ = TensorDesc(param.mmYOptional, param.mmYOptional_dtype, param.mmYOptional_format);
	// TensorDesc permuteOut_ = TensorDesc(param.permuteOutOptional, param.permuteOutOptional_dtype, param.permuteOutOptional_format);

	int64_t gmmX_quant_mode_ = param.gmmX_quant_mode;
	int64_t gmmWeight_quant_mode_ = param.gmmWeight_quant_mode;
	int64_t mmX_quant_mode_ = param.mmX_quant_mode;
	int64_t mmWeight_quant_mode_ = param.mmWeight_quant_mode;

    const char* group = param.group;
	bool send_ = param.send;
	bool recv_ = param.recv;
    bool transGmmWeight = param.transGmmWeight;
    bool transMmWeight = param.transMmWeight;
	bool permuteOutFlag = param.permuteOutFlag;

    aclnnStatus retStatus = param.aclnn_status;

	int64_t epWorldSize_ = param.epWorldSize;
	int64_t BS_ = param.BS;
	int64_t K_ = param.K;
	int64_t H_ = param.H;
	int64_t e_ = param.e;
	std::vector<int64_t> sendCountsList(epWorldSize_ * e_, BS_ * K_ / (epWorldSize_ * e_));
	std::vector<int64_t> recvCountsList(epWorldSize_ * e_, BS_ * K_ / (epWorldSize_ * e_));
	aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
	aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
	uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

	if(send_) sendCounts = nullptr;
	if(recv_) recvCounts = nullptr;

    auto ut = OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, mmX_,
                              mmWeight_, mmXScale_, mmWeightScale_, nullptr, nullptr, gmmX_quant_mode_,
                              gmmWeight_quant_mode_, mmX_quant_mode_, mmWeight_quant_mode_, group, epWorldSize_,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, 0, permuteOutFlag),
                        OUTPUT(gmmY_, mmY_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
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
