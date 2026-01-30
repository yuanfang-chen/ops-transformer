/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_mc2_comm_server.cpp
 * \brief
 */

// 定义宏以启用C11边界检查接口

#include <cstdio>
#include <string>
#include "acl/acl.h"
#include "runtime/runtime/kernel.h"
#include "graph/types.h"
#include "securec.h"
#include "aclnn_mc2_comm_server.h"

uint32_t AscFormatKfcArgs(const uintptr_t* resCtxList, uint64_t resCtxNum, rtAicpuArgsEx_t &aicpuArgs) {
    aicpuArgs.args = g_hostArgs;

    CommKfcParamDesc desc{};
    desc.version = 2;
    desc.itemNum = resCtxNum;
    uint64_t cpyLen = sizeof(desc);
    (void)memcpy_s(g_hostArgs, sizeof(g_hostArgs), &desc, cpyLen);
    aicpuArgs.argsSize = cpyLen;

    cpyLen = resCtxNum * sizeof(uintptr_t);
    (void)memcpy_s(g_hostArgs + aicpuArgs.argsSize, sizeof(g_hostArgs) - aicpuArgs.argsSize, resCtxList, cpyLen);
    aicpuArgs.argsSize += cpyLen;
    CHECK_RET(aicpuArgs.argsSize <= sizeof(g_hostArgs),
        printf("Args size %u exceeded the max value %u.", aicpuArgs.argsSize, sizeof(g_hostArgs));
        return 1);

    cpyLen = sizeof(KFC_SERVER_SO_NAME);
    (void)strcpy_s(g_hostArgs + aicpuArgs.argsSize, sizeof(g_hostArgs) - aicpuArgs.argsSize, KFC_SERVER_SO_NAME);
    aicpuArgs.soNameAddrOffset = aicpuArgs.argsSize;
    aicpuArgs.argsSize += cpyLen;
    CHECK_RET(aicpuArgs.argsSize <= sizeof(g_hostArgs),
        printf("Args size %u exceeded the max value %u.", aicpuArgs.argsSize, sizeof(g_hostArgs));
        return 1);

    cpyLen = sizeof(KFC_KERNEL_NAME);
    (void)strcpy_s(g_hostArgs + aicpuArgs.argsSize, sizeof(g_hostArgs) - aicpuArgs.argsSize, KFC_KERNEL_NAME);
    aicpuArgs.kernelNameAddrOffset = aicpuArgs.argsSize;
    aicpuArgs.argsSize += cpyLen;
    CHECK_RET(aicpuArgs.argsSize <= sizeof(g_hostArgs),
        printf("Args size %u exceeded the max value %u.", aicpuArgs.argsSize, sizeof(g_hostArgs));
        return 1);

    return 0;
}

extern "C" aclnnStatus aclnnInnerMc2CommServerGetWorkspaceSize(
    const aclTensor* ctx,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);


extern "C" void NnopbaseGetInputTensorAddr(void *executor, size_t index, void **addr);

extern "C" aclnnStatus aclnnMc2CommServerGetWorkspaceSize(
    const aclTensor* ctx,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
    // OP_CHECK_NULL(ctx, return false);
    if (ctx == nullptr) {
        return -1;
    }
    return aclnnInnerMc2CommServerGetWorkspaceSize(ctx, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnMc2CommServer(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream) {

    uintptr_t ctxAddr[1] = {0};
    NnopbaseGetInputTensorAddr(executor, 0, (void**)&ctxAddr[0]);
    uint32_t ret = 0;
    rtAicpuArgsEx_t aicpuArgs{};
    CHECK_RET(AscFormatKfcArgs(ctxAddr, 1, aicpuArgs) == 0, printf("Failed to format KFC args."); return 1);
    ret = rtAicpuKernelLaunchExWithArgs(5U, "AscComm", 1U, &aicpuArgs, nullptr, stream, 0x100U);
    CHECK_RET(ret == 0, printf("Failed to AscRtAicpuKernelLaunchExWithArgs."); return 1);

    return 0;
}