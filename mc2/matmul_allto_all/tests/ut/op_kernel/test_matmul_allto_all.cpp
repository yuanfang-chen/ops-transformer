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
 * \file test_matmul_allto_all.cpp
 * \brief kernel ut
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "matmul_allto_all_tiling_def.h"
#include "../../../op_kernel/matmul_allto_all.cpp"

extern uint8_t* g_hcclContextReserved[2];

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class matmul_allto_all_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        size_t ctxSize = sizeof(HcclCombinOpParam);
        g_hcclContextReserved[0] = (uint8_t*)AscendC::GmAlloc(ctxSize);
        std::cout << "matmul_allto_all_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        AscendC::GmFree((void*)g_hcclContextReserved[0]);
        std::cout << "matmul_allto_all_test TearDown\n" << std::endl;
    }
};

TEST_F(matmul_allto_all_test, matmul_allto_all_test_0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint32_t blockDim = 20;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 38191616;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulAlltoAllTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    MatmulAlltoAllInfo mmAlltoAllInfo{4096, 7168, 7168, 2, false, false};
    MatmulAlltoAllTilingData* tiling_data = reinterpret_cast<MatmulAlltoAllTilingData*>(tiling);
    tiling_data->matmulAlltoAllInfo = mmAlltoAllInfo;

    uint8_t* x1GM = (uint8_t*)AscendC::GmAlloc(mmAlltoAllInfo.M * mmAlltoAllInfo.K * sizeof(uint16_t));
    uint8_t* x2GM = (uint8_t*)AscendC::GmAlloc(mmAlltoAllInfo.K * mmAlltoAllInfo.N * sizeof(uint16_t));
    uint8_t* biasGM = nullptr;
    uint8_t* x1scaleGM = nullptr;
    uint8_t* x2scaleGM = nullptr;
    uint8_t* commscaleGM = nullptr;
    uint8_t* x1offsetGM = nullptr;
    uint8_t* x2offsetGM = nullptr;
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(mmAlltoAllInfo.M * mmAlltoAllInfo.N * sizeof(uint16_t));

    ICPU_SET_TILING_KEY(1000000);
    ICPU_RUN_KF(matmul_allto_all 20, x1GM, x2GM, biasGM, x1scaleGM, x2scaleGM, commscaleGM,
                x1offsetGM, x2offsetGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x1GM);
    AscendC::GmFree((void*)x2GM);
    AscendC::GmFree((void*)yGM);
}