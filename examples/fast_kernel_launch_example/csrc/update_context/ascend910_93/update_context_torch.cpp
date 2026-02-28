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
static std::string last_group_ep_str = "";
static at::Tensor output;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("updateContext(Tensor x, str group_ep, int ep_world_size) -> Tensor");
}

static int32_t CreatMc2Context(HcclComm &comm, int64_t worldSize, int64_t cclBufferSize, Mc2ContextStru &mc2Context)
{
    uint32_t ctxIndex = 0;
    uint32_t rankId;
    (void)HcclGetRankIdFunc(comm, &rankId);
    mc2Context.epRankId = rankId;
    for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
        void *remoteAddr = nullptr;
        uint64_t commSize = 0;
        HcclResult ret;
        if (rankId == remoteRankId) {
            ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));  // 获取本卡地址
        } else {
            ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBufFunc(comm, remoteRankId, &remoteAddr, &commSize)); // 获取远端地址
        }
        TORCH_CHECK(((commSize >= cclBufferSize) && (ret == HCCL_SUCCESS)), "HcclGetRemoteIpcHcclBuf failed, commSize=", commSize, " ret=", ret);
        mc2Context.epHcclBuffer_[remoteRankId] = (uint64_t)remoteAddr;
    }

    return 0;
}

static int32_t CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize, const char* groupName,
    std::string algConfig, uint32_t opType)
{
    HcclResult ret = static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfigFunc(opArgs, const_cast<char *>(algConfig.c_str())));  // 设置通信类型
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);
    ret = static_cast<HcclResult>(HcclCommGetHandleWithNameFunc(groupName, &commHandle));  // 通过groupName获取groupHandle
    TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);
    void *opsResCtx;
    ret = static_cast<HcclResult>(HcclCreateOpResCtxFunc(commHandle, opType, opArgs, &opsResCtx)); // 创建HcclContext
    TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);

    // Get Comm world Size and rank
    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = static_cast<HcclResult>(HcclGetRankSizeFunc(commHandle, &worldSizeHccl)); // 获取通信域大小
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);
    ret = static_cast<HcclResult>(HcclGetRankIdFunc(commHandle, &rankId)); // 获取卡号
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);
    if ((rankId >= worldSizeHccl) || (worldSize != worldSizeHccl)) {
        TORCH_CHECK(rankId < worldSizeHccl, "rankId:", rankId, " worldSizeHccl:", worldSizeHccl, " worldSize:", worldSize);
        return -1;
    }
    return 0;
}

static int32_t GetMc2Context(Mc2ContextStru &mc2ContextHost, int64_t epWorldSize, int64_t cclBufferSize, const char* groupEpStr)
{
    InitHcclFunctions();
    void* opArgs = nullptr;
    HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgsFunc(&opArgs));  // 通信配置对象创建
    TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);
    uint8_t commEngine = COMM_ENGINE_AIV;
    ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngineFunc(opArgs, (uint8_t)commEngine)); // 设置通信方式
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);
    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE);
    TORCH_CHECK(contextRet == 0, "CreateHcclContext failed, ret:", contextRet);
    ret = static_cast<HcclResult>(HcclKfcFreeOpArgsFunc(opArgs)); // 释放通信配置对象
    TORCH_CHECK(ret == 0, "getHcclKfcFreeOpArgs failed, ret:", ret);
    contextRet = CreatMc2Context(epCommHandle, epWorldSize, cclBufferSize, mc2ContextHost);
    TORCH_CHECK(contextRet == 0, "CreatMc2Context failed, ret:", contextRet);

    return 0;
}

/**
* @param x Input Tensor (on NPU)
* @return Result Tensor
*/
at::Tensor update_context(const at::Tensor &x, c10::string_view group_ep, int64_t cclBufferSize, int64_t ep_world_size)
{
    bool isSameGroup = false;
    std::string new_group_ep_str = std::string(group_ep);
    isSameGroup = (new_group_ep_str == last_group_ep_str);
    if (isInit && isSameGroup) {
        return output;
    }

    Mc2ContextStru mc2ContextHost[2];
    int32_t ret = GetMc2Context(mc2ContextHost, ep_world_size, cclBufferSize, new_group_ep_str.c_str());
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
