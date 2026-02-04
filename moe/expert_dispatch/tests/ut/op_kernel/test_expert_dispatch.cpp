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

#include "gtest/gtest.h"
#include "expert_dispatch_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>
using namespace std;

extern "C" __global__ __aicore__ void expert_dispatch(uint8_t* x, uint8_t* expertId, uint8_t* scale,
                                                      uint8_t* dispatchedX, uint8_t* dispatchedRowIdx,
                                                      uint8_t* expertTokensCount, uint8_t* expertTotalCount,
                                                      uint8_t* dispatchedScale, uint8_t* workspace, uint8_t* tiling);
class expert_dispatch_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "expert_dispatch_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "expert_dispatch_test TearDown\n" << endl;
    }
};

// 单核排序、直方图非全载、GatherOut非全载
TEST_F(expert_dispatch_test, test_case_0)
{
    size_t n = 4;
    size_t cols = 3;
    size_t k = 5;
    size_t expert_num = 6;
    uint64_t tilingKey = 1000000;
    uint32_t blockDim = 24;

    size_t x_Size = n * cols * sizeof(float);
    size_t expertId_Size = n * k * sizeof(int32_t);
    size_t scale_Size = n * sizeof(float);
    size_t dispatchedX_Size = n * k * cols * sizeof(float);
    size_t dispatchedRowIdx_Size = n * k * sizeof(int32_t);
    size_t expertTokensCount_Size = expert_num * sizeof(int32_t);
    size_t expertTotalCount_Size = 1 * sizeof(int32_t);
    size_t dispatchedScale_Size = n * k * sizeof(float);
    size_t workspace_Size = (n * k + expert_num + 1) * sizeof(float) * 7 + blockDim * 32 * 2 + 16781184;
    size_t tiling_Size = sizeof(ExpertDispatchTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_Size);
    uint8_t* expertId = (uint8_t*)AscendC::GmAlloc(expertId_Size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_Size);
    uint8_t* dispatchedX = (uint8_t*)AscendC::GmAlloc(dispatchedX_Size);
    uint8_t* dispatchedRowIdx = (uint8_t*)AscendC::GmAlloc(dispatchedRowIdx_Size);
    uint8_t* expertTokensCount = (uint8_t*)AscendC::GmAlloc(expertTokensCount_Size);
    uint8_t* expertTotalCount = (uint8_t*)AscendC::GmAlloc(expertTotalCount_Size);
    uint8_t* dispatchedScale = (uint8_t*)AscendC::GmAlloc(dispatchedScale_Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_Size);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_Size);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(expert_dispatch, blockDim, x, expertId, scale, dispatchedX, dispatchedRowIdx, expertTokensCount,
                expertTotalCount, dispatchedScale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)expertId);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)dispatchedX);
    AscendC::GmFree((void*)dispatchedRowIdx);
    AscendC::GmFree((void*)expertTokensCount);
    AscendC::GmFree((void*)expertTotalCount);
    AscendC::GmFree((void*)dispatchedScale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// 学员补充：其他数据类型（全载）和非全载用例