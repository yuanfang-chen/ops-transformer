/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_kernel/arch35/causal_conv1d_cut_bh_struct.h"
#include "../../../op_kernel/arch35/causal_conv1d_cut_bsh_struct.h"

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

// Shape aligned with host UT case: CausalConv1dCutBH_950_bf_b4_s1_d512
TEST_F(causal_conv1d_test, CausalConv1dCutBH_950_bf_b4_s1_d512)
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

    size_t tilingSize = sizeof(CausalConv1dCutBHTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    auto* td = reinterpret_cast<CausalConv1dCutBHTilingData*>(tiling);

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
static void FillSingleCoreTiling(CausalConv1dCutBSHTilingDataTest *td,
    uint32_t batch, uint32_t dim, uint32_t K, uint32_t cuSeqLen, uint32_t residualConnection)
{
    td->loopNumBS = 1;
    td->loopNumDim = 1;
    td->ubFactorBS = cuSeqLen;
    td->ubTailFactorBS = cuSeqLen;
    td->ubFactorDim = dim;
    td->ubTailFactorDim = dim;
    td->tailBlockloopNumBS = 1;
    td->tailBlockloopNumDim = 1;
    td->tailBlockubFactorBS = cuSeqLen;
    td->tailBlockubTailFactorBS = cuSeqLen;
    td->tailBlockubFactorDim = dim;
    td->tailBlockubTailFactorDim = dim;
    td->dimCoreNum = 1;
    td->dimRemainderCores = 1;
    td->dimBlockFactor = dim;
    td->dimBlockTailFactor = dim;
    td->bsCoreNum = 1;
    td->bsRemainderCores = 1;
    td->bsBlockFactor = cuSeqLen;
    td->bsBlockTailFactor = cuSeqLen;
    td->realCoreNum = 1;
    td->kernelWidth = K;
    td->cuSeqLen = cuSeqLen;
    td->dim = dim;
    td->batch = batch;
    td->padSlotId = -1;
    td->xStride = dim;
    td->cacheStride0 = (K - 1) * dim;
    td->cacheStride1 = dim;
    td->residualConnection = residualConnection;
}

TEST_F(causal_conv1d_test, test_bsh_fp16_single_batch)
{
    uint32_t batch = 1;
    uint32_t dim = 128;
    uint32_t K = 3;
    uint32_t cuSeqLen = 16;
    uint32_t blockDim = 1;

    size_t x_size = cuSeqLen * dim * sizeof(half);
    size_t weight_size = K * dim * sizeof(half);
    size_t cache_size = batch * (K - 1) * dim * sizeof(half);
    size_t y_size = cuSeqLen * dim * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_size);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weight_size);
    uint8_t *convStates = (uint8_t *)AscendC::GmAlloc(cache_size);
    uint8_t *queryStartLoc = (uint8_t *)AscendC::GmAlloc((batch + 1) * sizeof(int32_t));
    uint8_t *cacheIndices = (uint8_t *)AscendC::GmAlloc(batch * sizeof(int32_t));
    uint8_t *initialStateMode = (uint8_t *)AscendC::GmAlloc(batch * sizeof(int32_t));
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(16);
    uint8_t *numAcceptedToken = (uint8_t *)AscendC::GmAlloc(16);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(y_size);
    uint8_t *outputConvStates = (uint8_t *)AscendC::GmAlloc(16);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(CausalConv1dCutBSHTilingDataTest));

    memset(x, 0, x_size);
    memset(weight, 0, weight_size);
    memset(convStates, 0, cache_size);

    reinterpret_cast<int32_t *>(queryStartLoc)[0] = 0;
    reinterpret_cast<int32_t *>(queryStartLoc)[1] = 16;
    reinterpret_cast<int32_t *>(cacheIndices)[0] = 0;
    reinterpret_cast<int32_t *>(initialStateMode)[0] = 1;

    CausalConv1dCutBSHTilingDataTest *td =
        reinterpret_cast<CausalConv1dCutBSHTilingDataTest *>(tiling);
    FillSingleCoreTiling(td, batch, dim, K, cuSeqLen, 0);

    ICPU_SET_TILING_KEY(10001);
    ICPU_RUN_KF(causal_conv1d, blockDim, x, weight, convStates, queryStartLoc,
                cacheIndices, initialStateMode, bias, numAcceptedToken, y,
                outputConvStates, workspace, (uint8_t *)(td));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(convStates);
    AscendC::GmFree(queryStartLoc);
    AscendC::GmFree(cacheIndices);
    AscendC::GmFree(initialStateMode);
    AscendC::GmFree(bias);
    AscendC::GmFree(numAcceptedToken);
    AscendC::GmFree(y);
    AscendC::GmFree(outputConvStates);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}