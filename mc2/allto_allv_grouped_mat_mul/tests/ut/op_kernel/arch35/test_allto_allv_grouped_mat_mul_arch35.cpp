/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "../allto_allv_grouped_mat_mul_tiling_def.h"
#include "../../../../op_kernel/allto_allv_gmm_utils.h"
#include "../../../../op_kernel/allto_allv_gmm.h"

// 引用arch35 op_kernel中的常量并打印验证
constexpr uint32_t TEST_UB_BLOCK_UNIT_SIZE = ALLTO_ALLV_GMM::UB_BLOCK_UNIT_SIZE;
constexpr uint32_t TEST_THRESHOLD_BLOCK_NUM = ALLTO_ALLV_GMM::thresholdBlockNum;

class allto_allv_grouped_mat_mul_arch35_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "[arch35] allto_allv_grouped_mat_mul_arch35_test SetUp" << std::endl;
        // 打印从op_kernel引用的常量值
        std::cout << "[arch35] UB_BLOCK_UNIT_SIZE from op_kernel: " << TEST_UB_BLOCK_UNIT_SIZE << std::endl;
        std::cout << "[arch35] thresholdBlockNum from op_kernel: " << TEST_THRESHOLD_BLOCK_NUM << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "[arch35] allto_allv_grouped_mat_mul_arch35_test TearDown" << std::endl;
    }
};

// 简单测试：验证常量引用正确
TEST_F(allto_allv_grouped_mat_mul_arch35_test, arch35_constant_test)
{
    std::cout << "[arch35] Running arch35_constant_test" << std::endl;

    // 验证从op_kernel引用的常量值
    EXPECT_EQ(TEST_UB_BLOCK_UNIT_SIZE, 32u);
    EXPECT_EQ(TEST_THRESHOLD_BLOCK_NUM, 8u);

    std::cout << "[arch35] TEST_UB_BLOCK_UNIT_SIZE = " << TEST_UB_BLOCK_UNIT_SIZE << std::endl;
    std::cout << "[arch35] TEST_THRESHOLD_BLOCK_NUM = " << TEST_THRESHOLD_BLOCK_NUM << std::endl;
    std::cout << "[arch35] arch35_constant_test PASSED" << std::endl;
}

// 简单测试：验证tiling数据结构
TEST_F(allto_allv_grouped_mat_mul_arch35_test, arch35_tiling_struct_test)
{
    std::cout << "[arch35] Running arch35_tiling_struct_test" << std::endl;

    AlltoAllvGmmTilingData tilingData;
    tilingData.commonTilingInfo.BSK = 4096;
    tilingData.commonTilingInfo.H1 = 2048;

    EXPECT_EQ(tilingData.commonTilingInfo.BSK, 4096u);
    EXPECT_EQ(tilingData.commonTilingInfo.H1, 2048u);

    std::cout << "[arch35] BSK = " << tilingData.commonTilingInfo.BSK << std::endl;
    std::cout << "[arch35] H1 = " << tilingData.commonTilingInfo.H1 << std::endl;
    std::cout << "[arch35] arch35_tiling_struct_test PASSED" << std::endl;
}
