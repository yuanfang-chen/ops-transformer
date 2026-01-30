/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gmm_expert_op.cpp
 * \brief GmmExpertOp 单元测试
 */

#include <gtest/gtest.h>
#include "op_kernel/mc2_templates/compute/gmm_expert_op.h"

namespace MC2KernelTemplate {
namespace test {

/**
 * 测试 add 函数的基本功能
 */
TEST(AddFunctionTest, BasicAddition)
{
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(0, 0), 0);
    EXPECT_EQ(add(-1, 1), 0);
    EXPECT_EQ(add(100, 200), 300);
}

/**
 * 测试 add 函数的边界值
 */
TEST(AddFunctionTest, BoundaryValues)
{
    EXPECT_EQ(add(INT_MAX, 0), INT_MAX);
    EXPECT_EQ(add(INT_MIN, 0), INT_MIN);
    EXPECT_EQ(add(1, INT_MAX - 1), INT_MAX);
}

/**
 * 测试 add 函数处理负数
 */
TEST(AddFunctionTest, NegativeNumbers)
{
    EXPECT_EQ(add(-1, -2), -3);
    EXPECT_EQ(add(-100, 50), -50);
    EXPECT_EQ(add(-5, -5), -10);
}

/**
 * 测试 add 函数处理大数
 */
TEST(AddFunctionTest, LargeNumbers)
{
    EXPECT_EQ(add(1000000, 2000000), 3000000);
    EXPECT_EQ(add(-500000, 500000), 0);
}

} // namespace test
} // namespace MC2KernelTemplate
