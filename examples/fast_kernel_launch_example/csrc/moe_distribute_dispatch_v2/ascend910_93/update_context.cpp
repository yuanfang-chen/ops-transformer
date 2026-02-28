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
 * \file abs.cpp
 * \brief
 */

#include <torch/extension.h>
#include "hccl_common.h"
#include "update_context.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;

constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t OP_TYPE_ALL_TO_ALLV = 8;
constexpr uint32_t EP_OP_TYPE = OP_TYPE_ALL_TO_ALLV;
const std::string EP_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";

static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

static int32_t CreatMc2Context(HcclComm &comm, int64_t worldSize, Mc2ContextStru *mc2Context)
{
    printf("dlopen success3");
    uint32_t ctxIndex =  0;

    auto getFuncHcclGetRankId = GetHcclFuncAddr<_HcclGetRankId>("HcclGetRankId");
    if (getFuncHcclGetRankId == nullptr) {
        printf("getFuncHcclGetRankId failed.");
        return -1;
    }
    printf("getFuncHcclGetRankId success.");
    auto getFuncHcclGetHcclBuffer = GetHcclFuncAddr<_HcclGetHcclBuffer>("HcclGetHcclBuffer");
    if (getFuncHcclGetHcclBuffer == nullptr) {
        printf("getFuncHcclGetHcclBuffer failed.");
        return -1;
    }
    printf("getFuncHcclGetHcclBuffer success.");
    auto getFuncHcclGetRemoteIpcHcclBuf = GetHcclFwkFuncAddr<_HcclGetRemoteIpcHcclBuf>("HcclGetRemoteIpcHcclBuf");
    if (getFuncHcclGetRemoteIpcHcclBuf == nullptr) {
        printf("getFuncHcclGetRemoteIpcHcclBuf failed.");
        return -1;
    }
    printf("getFuncHcclGetRemoteIpcHcclBuf success.");
    uint32_t rankId;
    (void)getFuncHcclGetRankId(comm, &rankId);
    mc2Context[ctxIndex].epRankId = rankId;
    printf("HcclGetRankId success3, rankId=%d", rankId);

    for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
        void *remoteAddr = nullptr;
        uint64_t commSize = 0; // 出参, HCCL_BUFFER_SIZE=, 校验
        HcclResult ret;
        if (rankId == remoteRankId) {
            ret = static_cast<HcclResult>(getFuncHcclGetHcclBuffer(comm, &remoteAddr, &commSize));
        } else {
            ret = static_cast<HcclResult>(getFuncHcclGetRemoteIpcHcclBuf(comm, remoteRankId, &remoteAddr, &commSize));
        }
        if (commSize == 0 || ret != HCCL_SUCCESS) {
            printf("getFuncHcclGetRemoteIpcHcclBuf failed, commSize=%lu, ret=%d.");
            return -1;
        }
        printf("HcclGetRemoteIpcHcclBuf success3");
        mc2Context[ctxIndex].epHcclBuffer_[remoteRankId] = (uint64_t)remoteAddr;
        printf("HcclGetRemoteIpcHcclBuf success4, rankId=%d, remoteId=%d, remoteAddr = %p comm:%p",
            rankId, remoteRankId, mc2Context[ctxIndex].epHcclBuffer_[remoteRankId], comm);
    }

    printf("HcclGetBuff success, rankId=%d", rankId);
    return 0;
}

static int32_t CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize,
                                     const char* groupName, std::string algConfig, uint32_t opType)
{
    auto getFuncHcclKfcOpArgsSetAlgConfig = GetHcclFuncAddr<_HcclKfcOpArgsSetAlgConfig>("HcclKfcOpArgsSetAlgConfig");
    if (getFuncHcclKfcOpArgsSetAlgConfig == nullptr) {
        printf("getFuncHcclKfcOpArgsSetAlgConfig failed.");
        return -1;
    }
    printf("getFuncHcclKfcOpArgsSetAlgConfig success.");
    auto getFuncHcclCommGetHandleWithName= GetHcclFwkFuncAddr<_HcclCommGetHandleWithName>("HcclCommGetHandleWithName");
    if (getFuncHcclCommGetHandleWithName == nullptr) {
        printf("getFuncHcclCommGetHandleWithName failed.");
        return -2;
    }
    printf("getFuncHcclCommGetHandleWithName success.");

    auto getFuncHcclCreateOpResCtx = GetHcclFuncAddr<_HcclCreateOpResCtx>("HcclCreateOpResCtx");
    if (getFuncHcclCreateOpResCtx == nullptr) {
        printf("getFuncHcclCreateOpResCtx failed.");
        return -3;
    }
    printf("getFuncHcclCreateOpResCtx success.");

    auto getFuncHcclGetRankId = GetHcclFuncAddr<_HcclGetRankId>("HcclGetRankId");
    if (getFuncHcclGetRankId == nullptr) {
        printf("getFuncHcclGetRankId failed.");
        return -4;
    }
    printf("getFuncHcclGetRankId success.");

    auto getFuncHcclGetRankSize = GetHcclFuncAddr<_HcclGetRankSize>("HcclGetRankSize");
    if (getFuncHcclGetRankSize == nullptr) {
        printf("getFuncHcclGetRankSize failed.");
        return -5;
    }
    printf("getFuncHcclGetRankSize success.");

    HcclResult ret = static_cast<HcclResult>(getFuncHcclKfcOpArgsSetAlgConfig(opArgs, const_cast<char*>(algConfig.c_str())));
    if(ret != 0) {
        printf("HcclKfcOpArgsSetAlgConfig failed.");
        return -6;
    }
    printf("try HcclCommGetHandleWithName groupName %s.", groupName);
    ret = static_cast<HcclResult>(getFuncHcclCommGetHandleWithName(groupName, &commHandle));
    if(ret != 0) {
        printf("HcclGetCommHandle failed, groupName %s, ret %d", groupName, ret);
        return -7;
    }
    printf("HcclCommGetHandleWithName success2, groupName %s, ret %d.", groupName, ret);

    void *opsResCtx;
    ret = static_cast<HcclResult>(getFuncHcclCreateOpResCtx(commHandle, opType, opArgs, &opsResCtx)); // 构造Context
    if(ret != HCCL_SUCCESS) {
        printf("HcclCreateOpResCtx failed.");
        return -8;
    }
    printf("HcclCreateOpResCtx success2");

    // Get Comm world Size and rank
    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = static_cast<HcclResult>(getFuncHcclGetRankSize(commHandle, &worldSizeHccl));
        if(ret != HCCL_SUCCESS) {
        printf("HcclGetRankSize failed, ret %d", ret);
        return -9;
    }
    printf("HcclGetRankSize success2");
    ret = static_cast<HcclResult>(getFuncHcclGetRankId(commHandle, &rankId));
    if(ret != HCCL_SUCCESS) {
        printf("HcclGetRankSize failed, ret %d", ret);
        return -10;
    }
    printf("HcclGetRankId success2");
    if (rankId >= worldSizeHccl || worldSize != worldSizeHccl) {
        TORCH_CHECK(rankId < worldSizeHccl, "rankId ", ret, "worldSizeHccl ", worldSizeHccl, "worldSize ", worldSize);
        return -11;
    }
    return 0;
}


static int32_t GetMc2Context(Mc2ContextStru* mc2ContextHost, int64_t epWorldSize, const char* groupEpStr)
{

    auto getHcclKfcAllocOpArgs = GetHcclFuncAddr<_HcclKfcAllocOpArgs>("HcclKfcAllocOpArgs");
    if (getHcclKfcAllocOpArgs == nullptr) {
        printf("getHcclKfcAllocOpArgs failed.");
        return -20;
    }

    auto getHcclKfcFreeOpArgs = GetHcclFuncAddr<_HcclKfcFreeOpArgs>("HcclKfcFreeOpArgs");
    if (getHcclKfcFreeOpArgs == nullptr) {
        printf("getHcclKfcFreeOpArgs failed.");
        return -21;
    }
    auto getHcclKfcOpArgsSetCommEngine = GetHcclFuncAddr<_HcclKfcOpArgsSetCommEngine>("HcclKfcOpArgsSetCommEngine");
    if (getHcclKfcOpArgsSetCommEngine == nullptr) {
        printf("getHcclKfcOpArgsSetCommEngine failed.");
        return -22;
    }
    void* opArgs = nullptr;
    HcclResult ret = static_cast<HcclResult>(getHcclKfcAllocOpArgs(&opArgs));
    if(ret != 0) {
        printf("HcclKfcAllocOpArgs failed, ret=%d", ret);
        return -23;
    }

    printf("HcclKfcAllocOpArgs success");
    uint8_t commEngine = COMM_ENGINE_AIV; //默认AIV引擎，应该是4
    ret = static_cast<HcclResult>(getHcclKfcOpArgsSetCommEngine(opArgs, (uint8_t)commEngine));
    if(ret != 0) {
        printf("HcclKfcOpArgsSetCommEngine failed, ret=%d", ret);
        return -24;
    }
    printf("getHcclKfcOpArgsSetCommEngine success");

    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE);
    if (contextRet != 0) {
        printf("CreateHcclContext failed.");
        return contextRet;
    }
    printf("CreateHcclContext success");
    contextRet = CreatMc2Context(epCommHandle, epWorldSize, mc2ContextHost);
    if (contextRet != 0) {
        printf("CreatMc2Context failed.");
        return contextRet;
    }
    printf("CreatMc2Context success");

    printf("HcclKfcFreeOpArgs begin");
    // opArgs is no longer needed after creating hcclCommHandle
    ret = static_cast<HcclResult>(getHcclKfcFreeOpArgs(opArgs));
    if(ret != 0) {
        printf("HcclKfcFreeOpArgs failed.");
        return -27;
    }

    return 0;
}


/**
 * @brief ACLNN Warpper for aclnnAbs
 * @param x Input Tensor (on NPU)
 * @return Result Tensor
 */
at::Tensor update_context(std::string group_ep, int64_t ep_world_size)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclLibName());
    TORCH_CHECK(opApiHandler != nullptr, "dlopen failed");
    auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetHcclLibName(), "HcclKfcFreeOpArgs");
    TORCH_CHECK(funcAddr != nullptr, "dlsym failed");
    Mc2ContextStru mc2ContextHost[2];
    int32_t ret = GetMc2Context(mc2ContextHost, ep_world_size, group_ep.c_str());
    TORCH_CHECK(ret == 0, "GetMc2Context failed, ret", ret);
    at::Tensor hostContext = at::from_blob(&mc2ContextHost, {sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)}, at::kInt);

    at::Tensor output = at::empty({sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)}, at::TensorOptions().dtype(at::kInt)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    output.copy_(hostContext);
    return output;
}


// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("update_context", &update_context, "update_context");
}
} // op_api