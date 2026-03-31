# MC2 通信结构体实现

对于需要 HCCL 通信的算子（如 MoeDistribute），需要在 `update_context` 目录创建通信结构体获取逻辑。

## 目录结构

```
update_context/ascend910_93/
├── CMakeLists.txt
├── update_context_torch.h    # MC2通信结构体定义
└── update_context_torch.cpp  # HCCL通信配置实现
```

## update_context_torch.h

```cpp
#ifndef UPDATE_CONTEXT_TORCH_H
#define UPDATE_CONTEXT_TORCH_H
#include <torch/extension.h>
#include <hccl/hccl.h>

namespace ascend_ops {
namespace Update_Context {

constexpr uint32_t COMM_ENGINE_AIV = 4;
constexpr uint32_t OP_TYPE_ALL_TO_ALLV = 8;

struct Mc2ContextStru {
    uint64_t epRankId;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer[1024];  // 各 rank 的 buffer 地址
};

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("updateContext(Tensor x, str group_ep, int cclBufferSize, int ep_world_size) -> Tensor");
}

}}
#endif
```

## update_context_torch.cpp

```cpp
#include "update_context_torch.h"
#include <acl/acl.h>

namespace ascend_ops {
namespace Update_Context {

constexpr uint32_t EP_OP_TYPE = OP_TYPE_ALL_TO_ALLV;
const std::string EP_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";

static bool isInit = false;
static std::string last_group_ep_str = "";
static at::Tensor output;

// ========== 创建 MC2 Context ==========
static int32_t CreateMc2Context(
    HcclComm &comm, 
    int64_t worldSize, 
    int64_t cclBufferSize, 
    Mc2ContextStru *mc2Context)
{
    uint32_t ctxIndex = 0;
    uint32_t rankId;
    (void)HcclGetRankId(comm, &rankId);
    mc2Context[ctxIndex].epRankId = rankId;

    // 获取所有 rank 的 buffer 地址
    for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
        void *remoteAddr = nullptr;
        uint64_t commSize = 0;
        HcclResult ret;
        
        if (rankId == remoteRankId) {
            ret = static_cast<HcclResult>(HcclGetHcclBuffer(comm, &remoteAddr, &commSize));
        } else {
            ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBuf(comm, remoteRankId, &remoteAddr, &commSize));
        }
        
        TORCH_CHECK((commSize >= cclBufferSize) && (ret == HCCL_SUCCESS),
            "HcclGetRemoteIpcHcclBuf failed, commSize=", commSize, " ret=", ret);
        
        mc2Context[ctxIndex].epHcclBuffer[remoteRankId] = (uint64_t)remoteAddr;
    }
    return 0;
}

// ========== 创建 HCCL Context ==========
static int32_t CreateHcclContext(
    HcclComm &commHandle,
    void *opArgs,
    int64_t worldSize,
    const char* groupName,
    std::string algConfig,
    uint32_t opType)
{
    // 设置算法配置
    HcclResult ret = static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfig(opArgs, const_cast<char*>(algConfig.c_str())));
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);

    // 获取通信组句柄
    ret = static_cast<HcclResult>(HcclCommGetHandleWithName(groupName, &commHandle));
    TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);

    // 创建通信资源上下文
    void *opsResCtx;
    ret = static_cast<HcclResult>(HcclCreateOpResCtx(commHandle, opType, opArgs, &opsResCtx));
    TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);

    // 获取 world size 和 rank
    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = static_cast<HcclResult>(HcclGetRankSize(commHandle, &worldSizeHccl));
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);
    ret = static_cast<HcclResult>(HcclGetRankId(commHandle, &rankId));
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);

    TORCH_CHECK(rankId < worldSizeHccl && worldSize == worldSizeHccl,
        "rankId/worldSize mismatch");
    
    return 0;
}

// ========== 获取完整 MC2 Context ==========
static int32_t GetMc2Context(
    Mc2ContextStru* mc2ContextHost, 
    int64_t epWorldSize, 
    int64_t cclBufferSize, 
    const char* groupEpStr)
{
    // 分配操作参数
    void* opArgs = nullptr;
    HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgs(&opArgs));
    TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);

    // 设置通信引擎（AIV）
    ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngine(opArgs, COMM_ENGINE_AIV));
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);

    // 创建 HCCL Context
    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE);
    TORCH_CHECK(contextRet == 0, "CreateHcclContext failed, ret:", contextRet);

    // 创建 MC2 Context
    contextRet = CreateMc2context(epCommHandle, epWorldSize, cclBufferSize, mc2ContextHost);
    TORCH_CHECK(contextRet == 0, "CreateMc2Context failed, ret:", contextRet);

    // 释放操作参数
    ret = static_cast<HcclResult>(HcclKfcFreeOpArgs(opArgs));
    TORCH_CHECK(ret == 0, "HcclKfcFreeOpArgs failed, ret:", ret);

    return 0;
}

// ========== PTA 接口实现 ==========
at::Tensor update_context(
    const at::Tensor &x, 
    c10::string_view group_ep, 
    int64_t cclBufferSize, 
    int64_t ep_world_size)
{
    std::string new_group_ep_str = std::string(group_ep);
    
    // 缓存：相同 group 直接返回
    if (isInit && (new_group_ep_str == last_group_ep_str)) {
        return output;
    }

    // 获取 MC2 Context
    Mc2ContextStru mc2ContextHost[2];
    int32_t ret = GetMc2Context(mc2ContextHost, ep_world_size, cclBufferSize, new_group_ep_str.c_str());
    TORCH_CHECK(ret == 0, "GetMc2Context failed, ret", ret);

    // 转为 Tensor
    at::Tensor hostContext = at::from_blob(&mc2ContextHost, 
        {sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)}, at::kInt);
    
    output = at::empty({sizeof(Mc2ContextStru) * 2 / sizeof(int32_t)}, 
        at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1)
        .memory_format(c10::MemoryFormat::Contiguous));
    output.copy_(hostContext);
    
    last_group_ep_str = new_group_ep_str;
    isInit = true;
    return output;
}

// ========== 注册实现 ==========
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("updateContext", update_context);
}

}}
```

## Python 调用

```python
import torch
import torch_npu
import torch.distributed as dist

class MoeDistributeBuffer:
    def __init__(self, group, ccl_buffer_size: int = 0):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False)
        self.ccl_buffer_size = ccl_buffer_size * 1024 * 1024  # MB to bytes
        
        # 获取通信上下文
        x_context = torch.zeros(4, dtype=torch.int32).npu()
        self.context = torch.ops.ascend_ops.updateContext(
            x_context, self.group_name, self.ccl_buffer_size, self.world_size)
    
    def update_ctx(self, new_group):
        self.group = new_group
        self.rank_id = torch.distributed.get_rank(new_group)
        self.group_name = new_group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False)
        x_context = torch.zeros(4, dtype=torch.int32).npu()
        self.context = torch.ops.ascend_ops.updateContext(
            x_context, self.group_name, self.ccl_buffer_size, self.world_size)
```

## 关键 HCCL API

| API | 说明 |
|-----|------|
| `HcclKfcAllocOpArgs` | 分配操作参数 |
| `HcclKfcOpArgsSetCommEngine` | 设置通信引擎（AIV=4） |
| `HcclKfcOpArgsSetAlgConfig` | 设置算法配置 |
| `HcclCommGetHandleWithName` | 获取通信组句柄 |
| `HcclCreateOpResCtx` | 创建通信资源上下文 |
| `HcclGetRankId` | 获取当前 rank |
| `HcclGetRankSize` | 获取 world size |
| `HcclGetHcclBuffer` | 获取本端 buffer 地址 |
| `HcclGetRemoteIpcHcclBuf` | 获取远端 buffer 地址 |
| `HcclKfcFreeOpArgs` | 释放操作参数 |