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

#include "update_context_torch.h"

namespace ascend_ops {
namespace Update_Context {

// constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t OP_TYPE_ALL_TO_ALLV = 8;
constexpr uint32_t EP_OP_TYPE = OP_TYPE_ALL_TO_ALLV;
const std::string EP_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";
static bool isInit = false;
static std::string last_group_ep_str = nullptr;
static at::Tensor output;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("updateContext(Tensor x, str group_ep, int ep_world_size) -> Tensor");
}

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
    // printf("dlopen success3");
    uint32_t ctxIndex =  0;


    uint32_t rankId;
    (void)HcclGetRankId(comm, &rankId);
    mc2Context[ctxIndex].epRankId = rankId;
    // printf("HcclGetRankId success3, rankId=%d", rankId);


    for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
        void *remoteAddr = nullptr;
        uint64_t commSize = 0; // 出参, HCCL_BUFFER_SIZE=, 校验
        HcclResult ret;
        if (rankId == remoteRankId) {
            ret = static_cast<HcclResult>(HcclGetHcclBuffer(comm, &remoteAddr, &commSize));
        } else {
            ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBuf(comm, remoteRankId, &remoteAddr, &commSize));
        }
        if (commSize == 0 || ret != HCCL_SUCCESS) {
            // printf("HcclGetRemoteIpcHcclBuf failed, commSize=%lu, ret=%d.");
            return -1;
        }
        // printf("HcclGetRemoteIpcHcclBuf success3");
        mc2Context[ctxIndex].epHcclBuffer_[remoteRankId] = (uint64_t)remoteAddr;
        // printf("HcclGetRemoteIpcHcclBuf success4, rankId=%d, remoteId=%d, remoteAddr = %p comm:%p",
            // rankId, remoteRankId, mc2Context[ctxIndex].epHcclBuffer_[remoteRankId], comm);
    }


    // printf("HcclGetBuff success, rankId=%d", rankId);
    return 0;
}


static int32_t CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize,
                                    const char* groupName, std::string algConfig, uint32_t opType)
{
    HcclResult ret = static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfig(opArgs, const_cast<char*>(algConfig.c_str())));
    if(ret != 0) {
        // printf("HcclKfcOpArgsSetAlgConfig failed.");
        return -6;
    }
    // printf("try HcclCommGetHandleWithName groupName %s.", groupName);
    ret = static_cast<HcclResult>(HcclCommGetHandleWithName(groupName, &commHandle));
    if(ret != 0) {
        // printf("HcclGetCommHandle failed, groupName %s, ret %d", groupName, ret);
        return -7;
    }
    // printf("HcclCommGetHandleWithName success2, groupName %s, ret %d.", groupName, ret);


    void *opsResCtx;
    ret = static_cast<HcclResult>(HcclCreateOpResCtx(commHandle, opType, opArgs, &opsResCtx)); // 构造Context
    if(ret != HCCL_SUCCESS) {
        // printf("HcclCreateOpResCtx failed.");
        return -8;
    }
    // printf("HcclCreateOpResCtx success2");


    // Get Comm world Size and rank
    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = static_cast<HcclResult>(HcclGetRankSize(commHandle, &worldSizeHccl));
        if(ret != HCCL_SUCCESS) {

        // printf("HcclGetRankSize failed, ret %d", ret);
        return -9;
    }
    // printf("HcclGetRankSize success2");
    ret = static_cast<HcclResult>(HcclGetRankId(commHandle, &rankId));
    if(ret != HCCL_SUCCESS) {
        // printf("HcclGetRankSize failed, ret %d", ret);
        return -10;
    }
    // printf("HcclGetRankId success2");
    if (rankId >= worldSizeHccl || worldSize != worldSizeHccl) {
        TORCH_CHECK(rankId < worldSizeHccl, "rankId ", ret, "worldSizeHccl ", worldSizeHccl, "worldSize ", worldSize);
        return -11;
    }
    return 0;
}




static int32_t GetMc2Context(Mc2ContextStru* mc2ContextHost, int64_t epWorldSize, const char* groupEpStr)
{
    void* opArgs = nullptr;
    HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgs(&opArgs));
    if(ret != 0) {
        // printf("HcclKfcAllocOpArgs failed, ret=%d", ret);
        return -23;
    }


    // printf("HcclKfcAllocOpArgs success");
    uint8_t commEngine = COMM_ENGINE_AIV; //默认AIV引擎，应该是4
    ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngine(opArgs, (uint8_t)commEngine));
    if(ret != 0) {
        // printf("HcclKfcOpArgsSetCommEngine failed, ret=%d", ret);
        return -24;
    }
    // printf("HcclKfcOpArgsSetCommEngine success");


    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE);
    if (contextRet != 0) {
        // printf("CreateHcclContext failed.");
        return contextRet;
    }
    // printf("CreateHcclContext success");
    contextRet = CreatMc2Context(epCommHandle, epWorldSize, mc2ContextHost);
    if (contextRet != 0) {
        // printf("CreatMc2Context failed.");
        return contextRet;
    }
    // printf("CreatMc2Context success");


    // printf("HcclKfcFreeOpArgs begin");
    // opArgs is no longer needed after creating hcclCommHandle
    ret = static_cast<HcclResult>(HcclKfcFreeOpArgs(opArgs));
    if(ret != 0) {
        // printf("HcclKfcFreeOpArgs failed.");
        return -27;
    }


    return 0;
}




/**
* @brief ACLNN Warpper for aclnnAbs
* @param x Input Tensor (on NPU)
* @return Result Tensor
*/
at::Tensor update_context(const at::Tensor &x, c10::string_view group_ep, int64_t ep_world_size)
{
    bool isSameGroup = false;
    std::string new_group_ep_str = std::string(group_ep);
    isSameGroup = (new_group_ep_str == last_group_ep_str);
    if (isInit && isSameGroup) {
        return output;
    }

    Mc2ContextStru mc2ContextHost[2];
    int32_t ret = GetMc2Context(mc2ContextHost, ep_world_size, new_group_ep_str.c_str());
    TORCH_CHECK(ret == 0, "GetMc2Context failed, ret", ret);
    at::Tensor hostContext = at::from_blob(&mc2ContextHost, {sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)}, at::kInt);

    output = at::empty({sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)}, at::TensorOptions().dtype(at::kInt)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    output.copy_(hostContext);
    last_group_ep_str = new_group_ep_str;
    isInit = true;
    return output;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("updateContext", update_context);
}
}
}