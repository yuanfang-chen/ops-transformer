/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

// Tiling struct used by update kernel
#include "../../../op_kernel/arch35/causal_conv1d_update_struct.h"

using namespace std;


extern "C" __global__ __aicore__ void causal_conv1d(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convStates,
    GM_ADDR queryStartLoc,
    GM_ADDR cacheIndices,
    GM_ADDR initialStateMode,
    GM_ADDR bias,
    GM_ADDR numAcceptedToken,
    GM_ADDR y,
    GM_ADDR outputConvStates,
    GM_ADDR workspace,
    GM_ADDR tiling);

class causal_conv1d_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "causal_conv1d_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "causal_conv1d_test TearDown\n" << endl; }
};

// Shape aligned with host UT case: CausalConv1dUpdate_950_bf_b4_s1_d512
TEST_F(causal_conv1d_test, CausalConv1dUpdate_950_bf_b4_s1_d512)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // Shapes
    const int64_t batch = 4;
    const int64_t seqLen = 1;
    const int64_t dim = 512;
    const int64_t kernelSize = 3; // K
    const int64_t stateLen = 2;   // K - 1 + seqLen = 2

    // Allocate GM buffers (bf16/half both 2 bytes; we don't access contents in this UT)
    size_t bytesElem2 = sizeof(uint16_t);
    size_t xBytes = static_cast<size_t>(batch * seqLen * dim) * bytesElem2;
    size_t wBytes = static_cast<size_t>(kernelSize * dim) * bytesElem2;
    size_t stateBytes = static_cast<size_t>(batch * stateLen * dim) * bytesElem2;
    size_t yBytes = static_cast<size_t>(batch * seqLen * dim) * bytesElem2;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yBytes);
    uint8_t* outStates = (uint8_t*)AscendC::GmAlloc(stateBytes);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

    // Allocate tiling buffer
    size_t cacheIdxBytes = static_cast<size_t>(batch) * sizeof(int32_t);
    size_t numAcceptedBytes = static_cast<size_t>(batch) * sizeof(int32_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xBytes);
    uint8_t* w = (uint8_t*)AscendC::GmAlloc(wBytes);
    uint8_t* convStates = (uint8_t*)AscendC::GmAlloc(stateBytes);

    uint8_t* cacheIndices = (uint8_t*)AscendC::GmAlloc(cacheIdxBytes);
    uint8_t* numAcceptedToken = (uint8_t*)AscendC::GmAlloc(numAcceptedBytes);
    
    size_t tilingSize = sizeof(CausalConv1dUpdateTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    auto* td = reinterpret_cast<CausalConv1dUpdateTilingData*>(tiling);

    // Fill tiling data to mirror the host-UT expected tiling string
    // "16 4 4 4 0 128 128 4 0 1 1 1 1 1 1 128 128 1 1 1 1 128 128 4 1 0 512 3 2 512 1024 512 -1 0 1 0 "
    td->usedCoreNum = 16;
    td->dimCoreCnt = 4;
    td->batchCoreCnt = 4;

    td->dimMainCoreCnt = 4;
    td->dimTailCoreCnt = 0;
    td->mainCoredimLen = 128;
    td->tailCoredimLen = 128;

    td->batchMainCoreCnt = 4;
    td->batchTailCoreCnt = 0;
    td->mainCoreBatchNum = 1;
    td->tailCoreBatchNum = 1;

    td->loopNumBS = 1;
    td->loopNumDim = 1;
    td->ubMainFactorBS = 1;
    td->ubTailFactorBS = 1;
    td->ubMainFactorDim = 128;
    td->ubTailFactorDim = 128;
    td->tailBlockloopNumBS = 1;
    td->tailBlockloopNumDim = 1;
    td->tailBlockubFactorBS = 1;
    td->tailBlockubTailFactorBS = 1;
    td->tailBlockubFactorDim = 128;
    td->tailBlockubTailFactorDim = 128;

    td->batchSize = 4;
    td->seqLen = 1;
    td->cuSeqLen = 0;
    td->dim = 512;
    td->kernelSize = 3;
    td->stateLen = 2;

    td->xStride = 512;     // seqLen stride in x
    td->cacheStride0 = 1024; // stride on batch for convStates
    td->cacheStride1 = 512;  // stride on stateLen for convStates
    td->padSlotId = -1;
    td->xInputMode = 0;         // 3D input
    td->hasAcceptTokenNum = 1;  // input 7 provided
    td->residualConnection = 0; // disabled

    uint32_t blockDim = 16; // match usedCoreNum

    ICPU_SET_KEY(20000); // TILING_KEY_UPDATE_BF16
    ICPU_RUN_KF(causal_conv1d, blockDim,
                x, w, convStates, queryStartLoc, cacheIndices, initialStateMode, bias, numAcceptedToken,
                y, outStates, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(w);
    AscendC::GmFree(convStates);
    AscendC::GmFree(queryStartLoc);
    AscendC::GmFree(cacheIndices);
    AscendC::GmFree(initialStateMode);
    AscendC::GmFree(bias);
    AscendC::GmFree(numAcceptedToken);
    AscendC::GmFree(y);
    AscendC::GmFree(outStates);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
