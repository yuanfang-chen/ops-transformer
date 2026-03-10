/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

#define private public
#include "platform/platform_info.h"

class GroupedMatmulInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulProto TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_0)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},              //x
        {{{E, K, N}, {E, K, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //weight
        {{{M, N}, {M, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                //bias
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //scale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //offset
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantScale
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                      //groupList
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        //perTokenScale
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    }, 
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_group_size)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                //x
        {{{E, N, K}, {E, N, K}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},    //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                         //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //offset
        {{{E, N, K/16}, {E, N, K/16}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},      //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                              //groupList
        {{{M, K/32}, {M, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},            //perTokenScale
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_pertokenscale_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                //x
        {{{E, N, K}, {E, N, K}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},    //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                         //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //offset
        {{{E, N, K/32}, {E, N, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},      //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                              //groupList
        {{{M, 1, K/32}, {M, 1, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},      //perTokenScale
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_group_type)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},      //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_ND},   //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},      //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},              //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},      //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},            //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},              //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},              //perTokenScale
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    }, 
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_antiquant_scale_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, 1, N}, {E, 1, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    }, 
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_mx_antiquant_scale_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //x
        {{{E, K, N/8}, {E, K, N/8}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},      //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},                  //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                            //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                              //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //perTokenScale
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    }, 
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_a16w4_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},  //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_x_weight_shape)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K + 1}, {E, N, K + 1}}, ge::DT_INT8, ge::FORMAT_ND}, //weight, K mismatch
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_bias_dtype)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_INT32, ge::FORMAT_ND},            //bias, invalid dtype
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_grouplist_shape)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},            //groupList, invalid dim
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_a16w8_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_bf16_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},             //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},             //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},             //antiquantScale
        {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND},                   //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_antiquant_offset_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{E, 1}, {E, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantOffset, invalid dim
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_with_per_token_scale_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ},           //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                         //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //offset
        {{{E, N, K/32}, {E, N, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},      //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                              //groupList
        {{{M, K/32}, {M, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},            //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_x_dim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K, 1, 1, 1, 1, 1}, {M, K, 1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, //x, 7 dims invalid
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_transpose_x_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{K, M}, {K, M}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x, transposed
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Multi/Multi/Multi Scenario Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - multi scenario: multiple tensors in lists
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info - multi output
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},  // X_Y_SEPARATED for multi scenario
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, // GMM_NO_SPLIT
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_invalid_x_size)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - x and weight size mismatch
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0] only one x
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1] two weights
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_invalid_bias_size)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - bias size != weight size
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0] only one bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_invalid_x_dim_num)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - x has 7 dims (invalid)
        {{{M, K, 1, 1, 1, 1, 1}, {M, K, 1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, //x[0] 7 dims
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_with_antiquant_offset)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - with antiquantOffset
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset[1]
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info - multi output
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== S8S4 (INT8-INT4) Scenario Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_s8s4_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 512;
    size_t N = 512;
    size_t E = 2;
    size_t groupSize = 128;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - S8S4 with required params
        {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                     //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},          //weight (4bit packed in int32)
        {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                    //bias
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT, ge::FORMAT_ND},  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT16, ge::FORMAT_ND}, //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset (empty for S8S4)
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale (1 dim for S8S4)
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_s8s4_invalid_scale_dtype)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 512;
    size_t N = 512;
    size_t E = 2;
    size_t groupSize = 128;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - invalid scale dtype (should be float)
        {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                     //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},          //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                    //bias
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT16, ge::FORMAT_ND}, //scale (wrong dtype)
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT16, ge::FORMAT_ND}, //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_s8s4_missing_scale)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 512;
    size_t N = 512;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - missing scale (required for S8S4)
        {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                     //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},          //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                    //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale (empty)
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N, K/128}, {E, N, K/128}}, ge::DT_FLOAT16, ge::FORMAT_ND},    //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== A16FP8 (FP16/BF16-FP8) Scenario Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_a16fp8_e4m3_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - FP16-FP8_E4M3
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //x
        {{{E, N, K}, {E, N, K}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},      //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //antiquantScale
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_bf16_fp8_e5m2_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - BF16-FP8_E5M2
        {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},                     //x
        {{{E, N, K}, {E, N, K}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},        //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                     //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                     //antiquantScale
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                     //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},      // BF16 output
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Activation Function Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_activation)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr - with activation (not supported)
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  // ReLU (not supported)
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Split Item Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_split_item_for_single)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr - invalid split_item for single scenario
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},  // X_Y_SEPARATED not valid for single
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_split_item_for_multi)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr - invalid split_item for multi scenario
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},  // NO_SEPARATED not valid for multi
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== FP4_E2M1 with FP8_E8M0 antiquantScale Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_fp16_fp4_e2m1_mx_success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 256;
    size_t N = 256;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - FP16-FP4 with MX quantization
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //x
        {{{E, K, N/8}, {E, K, N/8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ}, //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                     //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, K/32, N}, {E, K/32, N}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},  //antiquantScale (per-group)
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset (empty for FP4)
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M, K/32}, {M, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},        //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_fp16_fp4_with_antiquant_offset_fail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 256;
    size_t N = 256;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - FP4 with antiquantOffset (not allowed)
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //x
        {{{E, K, N/8}, {E, K, N/8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ}, //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                     //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, K/32, N}, {E, K/32, N}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},  //antiquantScale
        {{{E, K/32, N}, {E, K/32, N}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},  //antiquantOffset (not allowed for FP4)
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M, K/32}, {M, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},        //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== 4-bit alignment Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_n_align_4bit)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 256;
    size_t N = 65;  // Not aligned to 64
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - N not aligned to 64 for 4bit weight
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //x
        {{{E, K, N/8}, {E, K, N/8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ}, //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                     //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, K/32, N}, {E, K/32, N}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},  //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M, K/32}, {M, K/32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},        //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== GroupList Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_grouplist_size_zero)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList - size 0
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_grouplist_too_large)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{1025}, {1025}}, ge::DT_INT64, ge::FORMAT_ND},            //groupList - exceeds max 1024
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Unsupported dtype combination Tests ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_invalid_dtype_combination)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - unsupported dtype combination
        {{{M, K}, {M, K}}, ge::DT_INT32, ge::FORMAT_ND},            //x - int32 not supported
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Multi scenario with Y separated ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_y_separated)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - multi scenario with Y separated (split_item=1)
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info - multi output
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  // Y_SEPARATED
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Multi scenario K mismatch ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_k_mismatch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - K mismatch between x and weight
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K + 64}, {N, K + 64}}, ge::DT_INT8, ge::FORMAT_ND},   //weight[1] - K mismatch
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Multi scenario N mismatch ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_n_mismatch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - N mismatch between weight and antiquantScale
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N + 64}, {N + 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},      //antiquantScale[1] - N mismatch
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Multi scenario with negative M dim ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_scenario_negative_m)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - negative M dim
        {{{-1, K}, {-1, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //x[0] with unknown M
        {{{-1, K}, {-1, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},        //x[1] with unknown M
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, N}, {-1, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== S8S4 with groupsize 192 ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_s8s4_groupsize_192)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 576;  // 192 * 3
    size_t N = 512;
    size_t E = 2;
    size_t groupSize = 192;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - S8S4 with groupsize 192
        {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                     //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},          //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                    //bias
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT, ge::FORMAT_ND},  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT16, ge::FORMAT_ND}, //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== S8S4 with invalid groupsize ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_s8s4_invalid_groupsize)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 640;  // Not divisible by 128/192/256/512
    size_t N = 512;
    size_t E = 2;
    size_t groupSize = 160;  // Invalid group size
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - S8S4 with invalid groupsize
        {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                     //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},          //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT, ge::FORMAT_ND},                    //bias
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT, ge::FORMAT_ND},  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N, K/groupSize}, {E, N, K/groupSize}}, ge::DT_FLOAT16, ge::FORMAT_ND}, //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== FP8_E4M3-FP4 with invalid pertoken scale dtype ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_mx_a8w4_invalid_pertoken_dtype)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 256;
    size_t N = 256;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - FP8_E4M3-FP4 with wrong pertoken scale dtype
        {{{M, K}, {M, K}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},            //x
        {{{E, K, N/8}, {E, K, N/8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ}, //weight
        {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},                     //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, K/32, N}, {E, K/32, N}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},  //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{M, K/32}, {M, K/32}}, ge::DT_FLOAT, ge::FORMAT_ND},              //perTokenScale - wrong dtype
    },
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== A16W4 INT32 weight format ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_a16w4_int32_weight)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 512;
    size_t N = 512;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - A16W4 with INT32 packed weight
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //x
        {{{E, N, K/8}, {E, N, K/8}}, ge::DT_INT32, ge::FORMAT_ND},          //weight (4bit in int32)
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //antiquantScale
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                  //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Multi scenario weight dim != 2 ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_weight_dim_invalid)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - weight has 3 dims (invalid for multi scenario)
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K, 1}, {N, K, 1}}, ge::DT_INT8, ge::FORMAT_ND},       //weight[0] - 3 dims
        {{{N, K, 1}, {N, K, 1}}, ge::DT_INT8, ge::FORMAT_ND},       //weight[1] - 3 dims
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Multi scenario with antiquantOffset size mismatch ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_antiquant_offset_size_mismatch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - antiquantOffset size != weight size
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset[0] - only one
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Multi scenario with antiquantScale size mismatch ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_multi_antiquant_scale_size_mismatch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 128;
    size_t K = 1280;
    size_t N = 1280;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - antiquantScale size != weight size
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[0]
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x[1]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[0]
        {{{N, K}, {N, K}}, ge::DT_INT8, ge::FORMAT_ND},             //weight[1]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[0]
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //bias[1]
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{N}, {N}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantScale[0] - only one
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}, {M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// ==================== Single scenario with X separated ====================
TEST_F(GroupedMatmulInfershape, grouped_matmul_infershape_weight_quant_single_x_separated)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo opti_compilation_info;
    opti_compilation_info.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    size_t M = 3;
    size_t K = 1280;
    size_t N = 1280;
    size_t E = 2;
    gert::InfershapeContextPara infershapeContextPara(
    "GroupedMatmul",
    { // input info - single scenario with X separated (split_item=2)
        {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //x
        {{{E, N, K}, {E, N, K}}, ge::DT_INT8, ge::FORMAT_ND},       //weight
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //bias
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //scale
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //offset
        {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},          //antiquantScale
        {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},                //antiquantOffset
        {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  //groupList
        {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                  //perTokenScale
    },
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    },
    { // attr
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},  // X_SEPARATED
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{M, N}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}