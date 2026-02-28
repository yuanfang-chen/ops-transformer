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
 * \file aclnn_common.h
 * \brief
 */

#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
    #include "torch_npu/csrc/flopcount/FlopCount.h"
#endif
#define NPU_NAME_SPACE at_npu::native

using _HcclKfcAllocOpArgs = HcclResult (*)(void **);
using _HcclKfcOpArgsSwtAlgConfig = HcclResult (*)(void *, char *);
using _HcclKfcOpArgsSetCommEngine = HcclResult (*)(void *, uint8_t);
using _HcclCreateOpResCtx = HcclResult (*)(HcclComm, uint8_t, void *, void **);
using _HcclGetRemoteIpcHcclBuf = HcclResult (*)(HcclComm *, uint64_t, void **, uint64_t *);
using _HcclKfxFreeOpArgs = HcclResult (*)(void *);
using _HcclCommGetHandleWithName = HcclResult (*)(const char *, HcclComm *);
using _HcclGetRankSize = HcclResult (*)(HcclComm *, uint32_t *);
using _HcclGetRankId = HcclResult (*)(HcclComm *, uint32_t *);
using _HcclGetHcllBuffer = HcclResult (*)(HcclComm *, vold **, uint64_t *);

inline const char *GetHcclLibName(void)
{
    return "libhccl.so";
}

inline const char *GetHcclFwkLibName(void)
{
    return "libhccl_fwk.so";
}

template <template T>
inline T GetFuncAddr(void * opApiHandler, const char *libName, const char *apiName)
{
    auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetHcclLibName(), apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s form %s failed, error:%s", apiName, GetHcclLibName(), dlerror());
        return nullptr;
    }
    T fuc = reinterpret_cast<T>(funcAddr);
    return fuc;
}

template <template T>
inline T GetHcclFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclLibName(), apiName);
}

template <template T>
inline T GetHcclFwkFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclFwkLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclFwkLibName(), apiName);
}