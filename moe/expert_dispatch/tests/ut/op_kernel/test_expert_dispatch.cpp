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

    size_t x_FileSize = n * cols * sizeof(float);
    size_t expertId_FileSize = n * k * sizeof(int32_t);
    size_t scale_FileSize = n * sizeof(float);
    size_t dispatchedX_FileSize = n * k * cols * sizeof(float);
    size_t dispatchedRowIdx_FileSize = n * k * sizeof(int32_t);
    size_t expertTokensCount_FileSize = expert_num * sizeof(int32_t);
    size_t expertTotalCount_FileSize = 1 * sizeof(int32_t);
    size_t dispatchedScale_FileSize = n * k * sizeof(float);
    size_t workspace_FileSize = (n * k + expert_num + 1) * sizeof(float) * 7 + blockDim * 32 * 2 + 16781184;
    size_t tiling_FileSize = sizeof(ExpertDispatchTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_FileSize);
    uint8_t* expertId = (uint8_t*)AscendC::GmAlloc(expertId_FileSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_FileSize);
    uint8_t* dispatchedX = (uint8_t*)AscendC::GmAlloc(dispatchedX_FileSize);
    uint8_t* dispatchedRowIdx = (uint8_t*)AscendC::GmAlloc(dispatchedRowIdx_FileSize);
    uint8_t* expertTokensCount = (uint8_t*)AscendC::GmAlloc(expertTokensCount_FileSize);
    uint8_t* expertTotalCount = (uint8_t*)AscendC::GmAlloc(expertTotalCount_FileSize);
    uint8_t* dispatchedScale = (uint8_t*)AscendC::GmAlloc(dispatchedScale_FileSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspace_FileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_FileSize);

    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/expert_dispatch/expert_dispatch_data ./");
    // system("chmod -R 755 ./expert_dispatch_data/");
    // system("cd ./expert_dispatch_data/ && rm -rf ./*bin");
    // system("cd ./expert_dispatch_data/ && python3 gen_data.py 4 3 5 float32");
    // system("cd ./expert_dispatch_data/ && python3 gen_tiling.py case0");

    // char* path_ = get_current_dir_name();
    // string path(path_);
    // ReadFile(path + "/expert_dispatch_data/input_x.bin", x_FileSize, x, x_FileSize);
    // ReadFile(path + "/expert_dispatch_data/input_expertId.bin", expertId_FileSize, expertId, expertId_FileSize);
    // ReadFile(path + "/expert_dispatch_data/scale.bin", scale_FileSize, scale, scale_FileSize);
    // ReadFile(path + "/expert_dispatch_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

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
    // free(path_);
}

// 学员补充，其它tilingKey模板