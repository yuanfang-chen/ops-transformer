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
 * \file test_aclnn_moe_init_routing_v3_mx_quant.cpp
 * \brief Example for aclnn MoeInitRoutingV3MxQuant operator end-to-end test.
 *
 * This example:
 *   - Creates BF16 input tensors x [num_tokens, hidden_size] and expert_idx [num_tokens, topk]
 *   - Calls aclnnMoeInitRoutingV3MxQuant to produce y (FP8), mxscale (FP8_E8M0),
 *     expanded_row_idx (INT32)
 *   - Verifies outputs are non-zero and prints basic statistics
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <fstream>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_moe_init_routing_v3_mx_quant.h"

#define CHECK_ACL(expr)                                                               \
    do {                                                                              \
        aclError _ret = (expr);                                                       \
        if (_ret != ACL_SUCCESS) {                                                    \
            std::cerr << "[ERROR] " << #expr << " failed with code " << _ret         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

static void FillBF16Random(uint16_t* buf, size_t n, float lo = -1.f, float hi = 1.f)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) {
        float v = dist(rng);
        // Convert float to BF16 (truncate mantissa)
        uint32_t u;
        std::memcpy(&u, &v, 4);
        buf[i] = static_cast<uint16_t>(u >> 16);
    }
}

static void FillInt32Sequential(int32_t* buf, size_t n, int32_t mod)
{
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<int32_t>(i % mod);
    }
}

int main(int argc, char* argv[])
{
    // ===== Configuration =====
    const int64_t numTokens = 4;
    const int64_t hiddenSize = 64;
    const int64_t topk = 2;
    const int64_t expertNum = 4;
    const int64_t blocksize = 32;
    const int64_t dstType = 0;  // 0 = fp8_e4m3fn

    // Derived
    const int64_t numExpandedTokens = numTokens * topk;
    const int64_t scaleColsPerRow = (hiddenSize + blocksize - 1) / blocksize;

    // ===== ACL initialization =====
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ===== Allocate host buffers =====
    size_t xBytes = numTokens * hiddenSize * sizeof(uint16_t);       // BF16
    size_t expertIdxBytes = numTokens * topk * sizeof(int32_t);      // INT32
    size_t yBytes = numExpandedTokens * hiddenSize * sizeof(uint8_t); // FP8 (1 byte/elem)
    size_t mxscaleBytes = numExpandedTokens * scaleColsPerRow * sizeof(uint8_t); // FP8_E8M0
    size_t expandedRowIdxBytes = numExpandedTokens * sizeof(int32_t);

    std::vector<uint16_t> hostX(numTokens * hiddenSize);
    std::vector<int32_t> hostExpertIdx(numTokens * topk);
    std::vector<uint8_t> hostY(numExpandedTokens * hiddenSize, 0);
    std::vector<uint8_t> hostMxscale(numExpandedTokens * scaleColsPerRow, 0);
    std::vector<int32_t> hostExpandedRowIdx(numExpandedTokens, 0);

    FillBF16Random(hostX.data(), hostX.size());
    FillInt32Sequential(hostExpertIdx.data(), hostExpertIdx.size(), expertNum);

    // ===== Allocate device buffers =====
    void *devX, *devExpertIdx, *devY, *devMxscale, *devExpandedRowIdx;
    CHECK_ACL(aclrtMalloc(&devX, xBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&devExpertIdx, expertIdxBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&devY, yBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&devMxscale, mxscaleBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&devExpandedRowIdx, expandedRowIdxBytes, ACL_MEM_MALLOC_NORMAL_ONLY));

    CHECK_ACL(aclrtMemcpy(devX, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devExpertIdx, expertIdxBytes, hostExpertIdx.data(),
                           expertIdxBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ===== Build aclTensor =====
    std::vector<int64_t> xShape = {numTokens, hiddenSize};
    std::vector<int64_t> expertIdxShape = {numTokens, topk};
    std::vector<int64_t> yShape = {numExpandedTokens, hiddenSize};
    std::vector<int64_t> mxscaleShape = {numExpandedTokens, scaleColsPerRow};
    std::vector<int64_t> expandedRowIdxShape = {numExpandedTokens};

    auto xTensor = aclCreateTensor(xShape.data(), 2, ACL_BF16, nullptr, 0, ACL_FORMAT_ND,
                                    xShape.data(), 2, devX);
    auto expertIdxTensor = aclCreateTensor(expertIdxShape.data(), 2, ACL_INT32, nullptr, 0,
                                            ACL_FORMAT_ND, expertIdxShape.data(), 2, devExpertIdx);
    auto yTensor = aclCreateTensor(yShape.data(), 2, ACL_FLOAT8_E4M3FN, nullptr, 0,
                                    ACL_FORMAT_ND, yShape.data(), 2, devY);
    auto mxscaleTensor = aclCreateTensor(mxscaleShape.data(), 2, ACL_FLOAT8_E8M0, nullptr, 0,
                                          ACL_FORMAT_ND, mxscaleShape.data(), 2, devMxscale);
    auto expandedRowIdxTensor = aclCreateTensor(expandedRowIdxShape.data(), 1, ACL_INT32,
                                                  nullptr, 0, ACL_FORMAT_ND,
                                                  expandedRowIdxShape.data(), 1, devExpandedRowIdx);

    // ===== Call operator =====
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int64_t activeNum = -1;
    int64_t expertCapacity = -1;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t axis = -1;
    char roundMode[] = "rint";
    int64_t scaleAlg = 0;

    CHECK_ACL(aclnnMoeInitRoutingV3MxQuantGetWorkspaceSize(
        xTensor, expertIdxTensor,
        nullptr,  // scale (optional)
        nullptr,  // offset (optional)
        activeNum, expertCapacity, expertNum, dropPadMode,
        expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
        axis, roundMode, dstType, blocksize, scaleAlg,
        yTensor, mxscaleTensor, expandedRowIdxTensor,
        nullptr,  // expert_tokens_count_or_cumsum (optional)
        nullptr,  // expanded_scale (optional)
        &workspaceSize, &executor));

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }

    CHECK_ACL(aclnnMoeInitRoutingV3MxQuant(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // ===== Copy back and verify =====
    CHECK_ACL(aclrtMemcpy(hostY.data(), yBytes, devY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(hostExpandedRowIdx.data(), expandedRowIdxBytes,
                           devExpandedRowIdx, expandedRowIdxBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    CHECK_ACL(aclrtMemcpy(hostMxscale.data(), mxscaleBytes, devMxscale, mxscaleBytes,
                           ACL_MEMCPY_DEVICE_TO_HOST));

    // Save inputs and outputs as .bin files for precision verification
    auto saveBin = [](const char* fname, const void* data, size_t bytes) {
        std::ofstream f(fname, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data), bytes);
        std::cout << "[INFO] Saved " << fname << " (" << bytes << " bytes)" << std::endl;
    };
    saveBin("input_x.bin", hostX.data(), xBytes);
    saveBin("input_expert_idx.bin", hostExpertIdx.data(), expertIdxBytes);
    saveBin("output_y.bin", hostY.data(), yBytes);
    saveBin("output_mxscale.bin", hostMxscale.data(), mxscaleBytes);
    saveBin("output_expanded_row_idx.bin", hostExpandedRowIdx.data(), expandedRowIdxBytes);

    // Basic validation: expanded_row_idx should be in [0, numTokens)
    bool idxOk = true;
    for (auto idx : hostExpandedRowIdx) {
        if (idx < 0 || idx >= static_cast<int32_t>(numTokens)) {
            idxOk = false;
            break;
        }
    }

    // Count non-zero y elements
    int nonZeroY = 0;
    for (auto v : hostY) { if (v != 0) nonZeroY++; }

    std::cout << "[INFO] numTokens=" << numTokens << " hiddenSize=" << hiddenSize
              << " topk=" << topk << " expertNum=" << expertNum << std::endl;
    std::cout << "[INFO] numExpandedTokens=" << numExpandedTokens << std::endl;
    std::cout << "[INFO] expanded_row_idx range check: " << (idxOk ? "PASS" : "FAIL") << std::endl;
    std::cout << "[INFO] y non-zero elements: " << nonZeroY << "/" << hostY.size() << std::endl;

    // ===== Cleanup =====
    aclDestroyTensor(xTensor);
    aclDestroyTensor(expertIdxTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(mxscaleTensor);
    aclDestroyTensor(expandedRowIdxTensor);

    if (workspace) { CHECK_ACL(aclrtFree(workspace)); }
    CHECK_ACL(aclrtFree(devX));
    CHECK_ACL(aclrtFree(devExpertIdx));
    CHECK_ACL(aclrtFree(devY));
    CHECK_ACL(aclrtFree(devMxscale));
    CHECK_ACL(aclrtFree(devExpandedRowIdx));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());

    std::cout << "[INFO] Example test completed." << std::endl;
    return idxOk ? 0 : 1;
}
