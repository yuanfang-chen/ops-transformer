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
#include "grouped_mat_mul_allto_allv_tiling_def.h"
#include "../../../op_kernel/grouped_mat_mul_allto_allv.cpp"

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class grouped_mat_mul_allto_allv_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "grouped_mat_mul_allto_allv_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "grouped_mat_mul_allto_allv_test TearDown\n" << std::endl;
    }
};

// shard = 1
TEST_F(grouped_mat_mul_allto_allv_test, grouped_mat_mul_allto_allv_test_0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 10 * 1024 * 1024;
    size_t usrWorkspaceSize = 10 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(GroupedMatMulAlltoAllvTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

}