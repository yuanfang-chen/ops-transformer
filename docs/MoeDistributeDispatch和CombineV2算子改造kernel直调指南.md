## 1. 概述

本文档介绍将 A3 版本的`MoeDistributeDispatchV2` 和 `MoeDistributeCombineV2` 算子从传统的 **Host/Kernel 传递模式** 改造为 **Kernel 直调模式** 的方法，实现从 aclnn 到 `<<<>>>` 直调的转换，使算子能够通过 torch.library 被 PyTorch 直接调用。

### 1.1 改造背景

原始算子位于 `mc2/moe_distribute_dispatch_v2` 和 `mc2/moe_distribute_combine_v2` 目录下，采用传统的 Host/Kernel 传递模式：

- **op_host**: 包含 Tiling 校验、Shape 推导等 Host 侧代码
- **op_kernel**: 包含 分支选择、通信计算等 Kernel 侧代码
- **op_api**: 包含 aclnn 二段式接口封装

改造后的算子位于 `examples/fast_kernel_launch_example/csrc/` 目录下，采用 Kernel 直调模式，通过 `<<<>>>` 语法直接启动核函数。

### 1.2 改造目标

参考链接：[PR #2141](https://gitcode.com/cann/ops-transformer/pull/2141)

- 在 example 目录中新增 dispatchV2/combineV2 算子 `<<<>>>` 直调方法
- 提供从 PTA 接口到 Kernel 的一整套流程
- 提供 demo 调用以供参考

### 1.3 改造前后对比

| 维度 | 传递模式（原始） | 直调模式（改造） |
|------|-----------------|-------------------|
| 代码组织 | `op_host` + `op_kernel` 分离 | 单一目录统一编译 |
| Host 入口 | `gert::TilingContext*` GE 上下文 | 自定义 struct 直接传参 |
| Kernel 启动 | 二段式调度 | `<<<blockDim, nullptr, stream>>>` |
| Python 绑定 | aclnn API | pybind11 直接绑定 |
| TilingKey | 全量支持 | 仅保留必要分支 |

---

## 2. 目录结构对比

### 2.1 原始分离模式目录（DispatchV2为例）

```
mc2/moe_distribute_dispatch_v2/
├── CMakeLists.txt
├── README.md
├── op_api/                      # aclnn 接口
│   ├── aclnn_moe_distribute_dispatch_v2.h
│   └── v2_proto.haclnn_moe_distribute_dispatch_v2.cpp
├── op_host/                     # Host 侧代码
│   ├── CMakeLists.txt
│   ├── moe_distribute_dispatch_v2_def.cpp # 算子原型
│   ├── moe_distribute_dispatch_v2_infershape.cpp
│   └── op_tiling/               # Tiling 实现
│       ├── moe_distribute_dispatch_v2_tiling.cpp
│       ├── moe_distribute_dispatch_tiling_v2.h
├── op_kernel/                   # Kernel 侧代码
│   ├── CMakeLists.txt
│   ├── moe_distribute_dispatch_v2.cpp
│   ├── moe_distribute_dispatch_v2.h
│   ├── moe_distribute_dispatch_v2_tiling.h
├── op_graph/                     # GE入图
│   ├── CMakeLists.txt
│   └── moe_distribute_dispatch_v2.cpp
└── examples/                     # 调用示例
    └── test_aclnn_moe_distribute_dispatch_v2.cpp
```

### 2.2 改造直调模式目录（DispatchV2为例）

```
examples/fast_kernel_launch_example/csrc/
├── moe_distribute_dispatch_v2/
│   └── ascend910_93/
│       ├── CMakeLists.txt                          # 编译配置（仅指定架构）
│       ├── moe_distribute_dispatch_v2_entry.h      # <<<>>>调用函数声明
│       ├── moe_distribute_dispatch_v2_entry.cpp    # Kernel入口 + <<<>>>调用
│       ├── moe_distribute_dispatch_v2_torch.h      # PyTorch工具函数
│       ├── moe_distribute_dispatch_v2_torch.cpp    # Host侧实现 + PyTorch绑定
│       └── op_kernel/                              # 精简的Kernel头文件
│           ├── moe_distribute_dispatch_v2.h
│           ├── moe_distribute_dispatch_v2_tiling.h
│           └── ...
├── update_context/
│   └── ascend910_93/
│       ├── CMakeLists.txt                          # 编译配置（仅指定架构）
│       ├── update_context_torch.h                  # 定义MC2通信结构体
│       ├── update_context_torch.cpp                # MC2通信结构体实现文件
```

---

## 3. Tiling 侧改造详解

### 3.1 去掉 GE Context 依赖

**传递模式**：使用 `gert::TilingContext*` 创建`context`获取输入信息

```cpp
// 原始代码 (mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp)
ge::graphStatus MoeDistributeDispatchA3TilingFuncImplPublic(gert::TilingContext *context, DispatchV2Config &config)
{
	const char *nodeName = context->GetNodeName();
	MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
	tilingData.moeDistributeDispatchV2Info.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
   	tilingData.moeDistributeDispatchV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
	...
	// 校验UB大小
    CheckUBSize(context, *tilingData, nodeName)
    // 校验win区大小
 	CheckAndCalWinSize(context, *tilingData, nodeName, isSetFullMeshV2, localMoeExpertNum, isLayered, config)
    // 设置workspace
    SetWorkSpace(context, nodeName）
    // 配置通信域信息
    SetHCommCfg(context, tilingData, groupEp, groupTp, tpWorldSize, isLayered)
    // 计算tilingkey
    CalTilingKey(context, isScales, quantMode, tpWorldSize, isSetFullMeshV2, isLayered);
}
```

**直调模式**：直接从 `at::Tensor` 获取参数传递`Tilingdata`数据

```cpp
// 改造后代码 (moe_distribute_dispatch_v2_torch.cpp)
void CalculateTilingData(MoeDistributeDispatchV2Info &tilingData, int64_t ep_world_size, int64_t ep_rank_id, ...）
{
	auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);
    // 初始化TilingData结构体的成员变量
    tilingData.epWorldSize = ep_world_size;                
    tilingData.epRankId = ep_rank_id;                  
	...
}
```

### 3.2 关键 API 改造对照表

| 原始 API (传递模式) | 改造后 API (直调模式) | 说明 |
|-------------------|---------------------|------|
| `gert::TilingContext*` | 直接传参 | 无需 GE 上下文 |
| `context->GetInputShape()` | `tensor.sizes()` | PyTorch API |
| `context->GetOutputShape()` | 根据输入推导 | 预留创建空tensor |
| `OP_LOGE()` / `OP_LOGI()` | `std::cout` | 标准输出 |

### 3.3 TilingData 结构改造

**传递模式**：包含业务参数与通信参数

```cpp
// mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h
struct MoeDistributeDispatchV2Info {
    uint32_t epWorldSize;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    ...
};

struct MoeDistributeDispatchV2TilingData {
    Mc2InitTiling mc2InitTiling;           // 直调模式移除
    Mc2CcTiling mc2CcTiling1;              // 直调模式移除
    Mc2CcTiling mc2CcTiling2;              // 直调模式移除
    MoeDistributeDispatchV2Info moeDistributeDispatchV2Info;
};
```

**直调模式**：`Mc2InitTiling`与`Mc2CcTiling`通信配置转移到update_context处理

```cpp
// examples/fast_kernel_launch_example/csrc/.../op_kernel/moe_distribute_dispatch_v2_tiling.h
struct MoeDistributeDispatchV2Info {
    uint32_t epWorldSize;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    ...
};
```

---

## 4. Kernel 侧改造详解

### 4.1 基本原则

Kernel 侧代码基本保持不变，主要变化：

1. 移除 `GET_TILING_DATA_WITH_STRUCT` 宏，改为直接传递 `TilingData`
2. 移除 `moe_distribute_dispatch_v2_tiling_key.h` 文件，自定义裁剪tilingkey分支

### 4.2 Kernel 入口改造

**传递模式**：使用宏注册，再获取 Tiling 数据

```cpp
// mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp
template<bool HasTp, uint8_t QuantMode, ...>
__global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, ..., GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
    op.Init(mc2Context, x, expertIds, ...,  &pipe, &tilingData); 
	op.Process();
}
```

**直调模式**：直接传递 TilingData 参数
ps：workspace由PTA接口内构造，构造一个Tensor作为输入传进kernel

```cpp
// moe_distribute_dispatch_v2_entry.cpp
// 模板函数
template<typename XType, typename ExpandxType, int32_t QuantMode, bool IsSmoothScaleExist>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, ..., MoeDistributeDispatchV2Info tilingData)
{
    MoeDistributeDispatchV2<XType, ExpandxType, QuantMode, IsSmoothScaleExist> op;
    op.Init(mc2Context, x, expertIds, ..., &pipe, tilingData);
    op.Process();
}

// 全局核函数入口
extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2_generic(
    int32_t tilingKey,
    GM_ADDR x, GM_ADDR expertIds, ..., 
    MoeDistributeDispatchV2Info tilingData)  // 直接传参
{
    switch (tilingKey) {
        case 10000:
            moe_distribute_dispatch_v2<float16_t, float16_t, UNQUANT, false>(
                x, expertIds, ..., tilingData);
            break;
        // ... 更多 case
    }
}
```

### 4.3 TilingKey 设计

**传递模式**：参照[Tiling模板编程-CANN社区版8.3.RC1.alpha001-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)
Host侧调用`GET_TPL_TILING_KEY`宏计算tilingkey，通过`context`传递至kernel入口拆分为模板函数参数

```cpp
// mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp
static uint64_t CalTilingKey(const gert::TilingContext *context, ..., bool isLayered)
{
    uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
    uint32_t tilingKeyQuantMode = quantMode;
    ...
    tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode, 
                        templateDispatch, commMode, TILINGKEY_TPL_A3);
    }
    return tilingKey;
}
context->SetTilingKey(tilingKey);
// mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp
   template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
   __global__ __aicore__ void moe_distribute_dispatch_v2(
       GM_ADDR x, GM_ADDR expertIds, ..., GM_ADDR tilingGM)
   {
       ...
   }

```

​**直调模式**​：使用简化的 TilingKey 编码，自定义裁剪tilingkey

```cpp
/*
 * A3 tilingkey说明
 * 5位的十进制数
 * 第1位（个位）：quantMode:
 *     0: 不量化, 1: 静态量化, 2: 动态量化
 * 第2位（十位）：x输入类型:
 *     0: float16, 1: bfloat16
 * 第3位（百位）：是否有smoothScale:
 *     0: 无, 1: 有
 * 第4位（千位）：是否走fullmesh_v2模板:
 *     0: 不做, 1: 做
 * 第5位（万位）：无实际含义
 */

// 示例 TilingKey:
case 10000:  // float16, 不量化, 无smoothScale, 非fullmesh
case 10002:  // float16, 动态量化, 无smoothScale, 非fullmesh
case 11000:  // float16, 不量化, 无smoothScale, fullmesh
case 11012:  // bfloat16, 动态量化, 无smoothScale, fullmesh
case 10110:  // bfloat16, 不量化, 有smoothScale, 非fullmesh
```

---

## 5. 通信结构体转换

### 5.1 通信域信息配置

**传递模式**：在Host侧调用内置函数，完成通信域信息预配置
参考文档：[TilingData结构体-CANN商用版8.5.0-昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_10048.html)

```cpp
// 原始代码 (mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp)
static ge::graphStatus SetHCommCfg(const gert::TilingContext *context, MoeDistributeCombineV2TilingData *tiling,
    const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize, bool isLayered)
{
    const char* nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombineV2 groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    std::string algConfigAllToAllStr = isLayered ? "AlltoAll=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigReduceScatterStr = "ReduceScatter=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);   // 通过不拉起AICPU，提高算子退出性能

    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling1 GetTiling mc2CcTiling1 failed"), return ge::GRAPH_FAILED);

    if (tpWorldSize > 1) {
        OP_LOGD(nodeName, "MoeDistributeCombineV2 groupTp = %s", groupTp.c_str());
        mc2CcTilingConfig.SetGroupName(groupTp);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfigReduceScatterStr);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling2 GetTiling  mc2CcTiling2 failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
```

**直调模式**：抽离出`hcclcontext（mc2context）`，单独设置`update_context`目录获取通信信息作为kernel入参
把 HCCL 通信句柄、本端 / 对端 buffer 地址、rank 信息 → 打包成一个 Tensor → 传给 MOE 算子用。

```cpp
// examples/fast_kernel_launch_example/csrc/update_context/.../update_context_torch.h
struct Mc2ContextStru {
       uint64_t epRankId;
       uint64_t kfcContextAddr;
       uint64_t epHcclBuffer[1024];
   };
// examples/fast_kernel_launch_example/csrc/update_context/.../update_context_torch.cpp
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
    m.def("updateContext(Tensor x, str group_ep, int cclBufferSize, int ep_world_size) -> Tensor");
}

static int32_t CreateMc2Context(HcclComm &comm, int64_t worldSize, int64_t cclBufferSize, Mc2ContextStru *mc2Context)
{
    uint32_t ctxIndex =  0;

    uint32_t rankId;
    (void)HcclGetRankId(comm, &rankId);
    mc2Context[ctxIndex].epRankId = rankId;

    for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
        void *remoteAddr = nullptr;
        uint64_t commSize = 0; // 出参, HCCL_BUFFER_SIZE=, 校验
        HcclResult ret;
        if (rankId == remoteRankId) {
            ret = static_cast<HcclResult>(HcclGetHcclBuffer(comm, &remoteAddr, &commSize));
        } else {
            ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBuf(comm, remoteRankId, &remoteAddr, &commSize));
        }
        TORCH_CHECK(((commSize >= cclBufferSize) && (ret == HCCL_SUCCESS)), "HcclGetRemoteIpcHcclBuf failed, commSize=",
            commSize, " ret=", ret);
        mc2Context[ctxIndex].epHcclBuffer[remoteRankId] = (uint64_t)remoteAddr;
    }

    return 0;
}


static int32_t CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize,
                                    const char* groupName, std::string algConfig, uint32_t opType)
{
    HcclResult ret = static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfig(opArgs, const_cast<char*>(algConfig.c_str())));
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);

    ret = static_cast<HcclResult>(HcclCommGetHandleWithName(groupName, &commHandle));
    TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);

    void *opsResCtx;
    ret = static_cast<HcclResult>(HcclCreateOpResCtx(commHandle, opType, opArgs, &opsResCtx)); // 构造Context
    TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);

    // Get Comm world Size and rank
    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = static_cast<HcclResult>(HcclGetRankSize(commHandle, &worldSizeHccl));
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);

    ret = static_cast<HcclResult>(HcclGetRankId(commHandle, &rankId));
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);

    if (rankId >= worldSizeHccl || worldSize != worldSizeHccl) {
        TORCH_CHECK(rankId < worldSizeHccl, "rankId ", ret, "worldSizeHccl ", worldSizeHccl, "worldSize ", worldSize);
        return -11;
    }
    return 0;
}

static int32_t GetMc2Context(Mc2ContextStru* mc2ContextHost, int64_t epWorldSize, int64_t cclBufferSize, const char* groupEpStr)
{
    void* opArgs = nullptr;
    HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgs(&opArgs));
    TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);

    uint8_t commEngine = COMM_ENGINE_AIV; //默认AIV引擎，应该是4
    ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngine(opArgs, (uint8_t)commEngine));
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);

    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE);
    TORCH_CHECK(contextRet == 0, "CreateHcclContext failed, ret:", contextRet);\

    contextRet = CreateMc2Context(epCommHandle, epWorldSize, cclBufferSize, mc2ContextHost);
    TORCH_CHECK(contextRet == 0, "CreateMc2Context failed, ret:", contextRet);

    // 释放通信配置对象
    ret = static_cast<HcclResult>(HcclKfcFreeOpArgs(opArgs));
    TORCH_CHECK(ret == 0, "HcclKfcFreeOpArgs failed, ret:", ret);

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

```

## 6. 直调模式实现

### 6.1 <<<>>> 调用

直调模式的核心 ：使用 CUDA 风格的 `<<<>>>` 语法直接启动核函数：

**头文件声明** (`moe_distribute_dispatch_v2_entry.h`):

```cpp
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_ENTRY_H
#define MOE_DISTRIBUTE_DISPATCH_V2_ENTRY_H
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"

// <<<>>>调用函数声明
void moe_distribute_dispatch_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, 
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context,
    MoeDistributeDispatchV2Info tilingData);

#endif // MOE_DISTRIBUTE_DISPATCH_V2_ENTRY_H
```

**实现文件** (`moe_distribute_dispatch_v2_entry.cpp`):

```cpp
// <<<>>>调用函数实现
void moe_distribute_dispatch_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, 
    GM_ADDR x, GM_ADDR expertIds, ..., MoeDistributeDispatchV2Info tilingData)
{
    // 核心：使用 <<<>>> 语法直接启动核函数
    moe_distribute_dispatch_v2_generic<<<blockDim, nullptr, stream>>>(
        tilingKey, x, expertIds, scales, ...,
        mc2Context, tilingData
    );
}
```

**参数说明**：

- `blockDim`: 核函数启动的 block 数量（对应 AIV 核数，从 `tilingData.aivNum` 获取）
- `nullptr`: 保留参数
- `stream`: ACL 流，用于异步执行

### 6.2 PyTorch 绑定（`moe_distribute_dispatch_v2_torch.cpp`）

**Schema 注册**:

```cpp
// 1. 注册算子 Schema
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("MoeDistributeDispatchV2(Tensor x, Tensor expert_ids, Tensor mc2_context, "
          "str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, "
          "int total_winsize_ep, *, Tensor? scales=None, Tensor? x_active_mask=None, "
          "Tensor? expert_scales=None, Tensor? performance_info=None, "
          "int expert_shard_type=0, int shared_expert_num=0, "
          "int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, "
          "int expert_token_nums_type=0, str comm_alg=\"\", "
          "int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) "
          "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
```

**NPU 调用函数**：

```cpp
// 输入采用at::tensor，输出预设值空at::tensor
// 3. 实现 NPU 调用函数
void moe_distribute_dispatch_v2_api(
    aclrtStream stream, bool is_fullmesh_v2, const at::Tensor &x, ...)
{
    // 获取 Tensor 数据指针
    auto x_ptr = get_first_tensor_address<at::Tensor>(x.scalar_type(), x, false);
    ...
    // 计算 TilingKey
    int32_t tilingKey = ...;
    // 调用 <<<>>> 入口函数
    moe_distribute_dispatch_v2_entry(tilingKey, tilingData.aivNum, (void*)stream, 
        (GM_ADDR)x_ptr, (GM_ADDR)expertIds_ptr, ..., tilingData);
}

static auto npu_moe_distribute_dispatch_v2(
    const at::Tensor &x, const at::Tensor &expert_ids, ...)
{
    // 获取当前 NPU 流
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
	// 获取tilingdata数据，等效完成Host侧工具
	calculate_tilingdata(tilingData, ep_world_size, ...）
    // 调用 API 函数
    moe_distribute_dispatch_v2_api(stream, is_fullmesh_v2, x, expert_ids, ...);
    return std::tie(expand_x, dynamic_scales, assist_info, 
                    expert_token_nums, ep_recv_counts, expand_scales);
}
```

**Meta 函数实现**（图模式：InferShape + InferDtype）：

```cpp
// 2. 实现 Meta 函数
static void npu_moe_distribute_dispatch_v2_meta(
    const at::Tensor &x, const at::Tensor &expert_ids, ...)
{
    // 推导输出形状和数据类型
    auto output_sizes = std::vector<int64_t>{...};
    ...
}
```

**实现注册**：

```cpp
// 4. 注册实现
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("MoeDistributeDispatchV2", TORCH_FN(npu_moe_distribute_dispatch_v2));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("MoeDistributeDispatchV2", &npu_moe_distribute_dispatch_v2_meta);
}

```

**整体调用**：（`examples/fast_kernel_launch_example/ascend_ops/moe_distribute.py`）

```python
Python 侧：MoeDistributeBuffer 初始化，获取通信组、rank、group_name
Python 调用：torch.ops.ascend_ops.updateContext(...)
进入 C++：执行 update_context(...) 函数
获取卡号：调用 HcclGetRankId(comm, &rankId) 获取当前 NPU 卡号
构造上下文：填充 mc2ContextHost[2]（包含 rankId、所有 NPU buffer 地址）
返回 Python：将结构体转为 Tensor 并传回
保存上下文：存入 self.context
送入算子：将 mc2_context=self.context 传给 MoeDistributeDispatchV2

import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error

X_CONTEXT_SIZE=4

class MoeDistributeBuffer:
    def __init__(self, group, ccl_buffer_size: int = 0, comm_alg: int = 0):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        mb_buffer_size = 200 if ccl_buffer_size == 0 else ccl_buffer_size
        self.ccl_buffer_size = mb_buffer_size * 1024 * 1024 # convert from mb
        x_context = torch.zeros(X_CONTEXT_SIZE, dtype=torch.int32).npu()
        self.context = torch.ops.ascend_ops.updateContext(x_context, self.group_name, self.ccl_buffer_size, self.world_size)

    @staticmethod
    def get_ccl_buffer_size(world_size: int, num_max_dispatch_tokens_per_rank: int, hidden: int,
                                        num_moe_expert: int, topk: int, num_shared_expert: int = 0,
                                        num_shared_expert_ranks: int = 0, comm_alg: str = "",
                                        ) -> int:
        def inline_align(value, base):
            return (value + base - 1) // base * base

        max_out_dtype_size = 2 # sizeof(int32)
        mb_conversion = 1024 * 1024
        ub_align = 32 # 32B
        scale_expand_index_buffer = 44 # scale 32B + 3 * 4 expand_idx
        full_mesh_data_align = 480
        win_addr_align = 512

        comm_alg_support_list = ["fullmesh_v1", "fullmesh_v2", ""]
        torch._check(comm_alg in comm_alg_support_list,
                     lambda: (f"comm_alg only support {comm_alg_support_list=} "
                              f"but got {comm_alg=}."),)

        token_actual_len = inline_align(hidden * max_out_dtype_size, ub_align) + scale_expand_index_buffer
        if (comm_alg == "fullmesh_v2"):
            token_need_size_dispatch = inline_align(token_actual_len, full_mesh_data_align) // full_mesh_data_align * \
                win_addr_align
        else: 
            token_need_size_dispatch = inline_align(token_actual_len, win_addr_align)
        token_need_size_combine = inline_align(hidden * max_out_dtype_size, win_addr_align)

        local_moe_expert_num = num_moe_expert // (world_size - num_shared_expert_ranks)
        minimum_buffer_size = 2 * (
            (num_max_dispatch_tokens_per_rank * token_need_size_dispatch * world_size * local_moe_expert_num) + \
            (num_max_dispatch_tokens_per_rank * token_need_size_combine * (topk + num_shared_expert))) + mb_conversion
        return inline_align(minimum_buffer_size, mb_conversion) // mb_conversion

    def update_ctx(self, new_group):
        self.group = new_group
        self.rank_id = torch.distributed.get_rank(new_group)
        self.group_name = new_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        x_context = torch.zeros(X_CONTEXT_SIZE, dtype=torch.int32).npu()
        self.context = torch.ops.ascend_ops.updateContext(x_context, self.group_name, self.ccl_buffer_size, self.world_size)
        return

    def npu_moe_distribute_dispatch_v2(self, x, expert_ids,
            moe_expert_num, *, scales=None, x_active_mask=None,
            expert_scales=None, performance_info=None, expert_shard_type=0, shared_expert_num=0,
            shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=0,
            comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
        (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, expand_scales) \
            = torch.ops.ascend_ops.MoeDistributeDispatchV2(
                                             mc2_context=self.context,
                                             x=x,
                                             expert_ids=expert_ids,
                                             group_ep=self.group_name,
                                             ep_world_size=self.world_size,
                                             ep_rank_id=self.rank_id,
                                             moe_expert_num=moe_expert_num,
                                             total_winsize_ep=self.ccl_buffer_size,
                                             scales=scales,
                                             x_active_mask=x_active_mask,
                                             expert_scales=expert_scales,
                                             performance_info=performance_info,
                                             expert_shard_type=expert_shard_type,
                                             shared_expert_num=shared_expert_num,
                                             shared_expert_rank_num=shared_expert_rank_num,
                                             quant_mode=quant_mode,
                                             expert_token_nums_type=expert_token_nums_type,
                                             global_bs=global_bs,
                                             comm_alg=comm_alg,
                                             zero_expert_num=zero_expert_num,
                                             copy_expert_num=copy_expert_num,
                                             const_expert_num=const_expert_num)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, expand_scales


    def npu_moe_distribute_combine_v2(self, expand_x, expert_ids, assist_info_for_combine,
            ep_send_counts, expert_scales,
            moe_expert_num, *,
            x_active_mask=None, shared_expert_x=None, ori_x=None,
            const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None,
            performance_info=None, expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=0,
            global_bs=0, comm_quant_mode=0,
            comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
        return torch.ops.ascend_ops.MoeDistributeCombineV2(
                                             mc2_context=self.context,
                                             expand_x=expand_x,
                                             expert_ids=expert_ids,
                                             assist_info_for_combine=assist_info_for_combine,
                                             ep_send_counts=ep_send_counts,
                                             expert_scales=expert_scales,
                                             group_ep=self.group_name,
                                             ep_world_size=self.world_size,
                                             ep_rank_id=self.rank_id,
                                             moe_expert_num=moe_expert_num,
                                             total_winsize_ep=self.ccl_buffer_size,
                                             x_active_mask=x_active_mask,
                                             shared_expert_x=shared_expert_x,
                                             ori_x=ori_x,
                                             const_expert_alpha_1=const_expert_alpha_1,
                                             const_expert_alpha_2=const_expert_alpha_2,
                                             const_expert_v=const_expert_v,
                                             performance_info=performance_info,
                                             expert_shard_type=expert_shard_type,
                                             shared_expert_num=shared_expert_num,
                                             shared_expert_rank_num=shared_expert_rank_num,
                                             global_bs=global_bs,
                                             comm_quant_mode=comm_quant_mode,
                                             comm_alg=comm_alg,
                                             zero_expert_num=zero_expert_num,
                                             copy_expert_num=copy_expert_num,
                                             const_expert_num=const_expert_num)



```

## 7. 构建与安装

### 7.1 CMakeLists.txt

直调模式的 CMakeLists.txt 非常简洁：

```cmake
# moe_distribute_dispatch_v2/ascend910_93/CMakeLists.txt
add_sources("--npu-arch=dav-2201")  # 仅指定架构参数
```

### 7.2 编译命令

```bash
# 进入 fast_kernel_launch_example 目录
cd examples/fast_kernel_launch_example
# 安装依赖
python3 -m pip install -r requirements.txt
# 指定SOCversion：
export NPU_ARCH=ascend910_93
# 构建 Wheel 包
python3 -m build --wheel -n
# 安装
python3 -m pip install dist/*.whl --force-reinstall --no-deps
```

---

## 8. Python 调用示例

```python
# 核心依赖
import os
from typing import List, Optional, Dict, Any, Tuple
from torch.multiprocessing import Process, Manager, set_start_method
import torch
import torch_npu
import torch.distributed as dist
import torchair
from torchair.configs.compiler_config import CompilerConfig
import ascend_ops
from ascend_ops import MoeDistributeBuffer

# 配置（只保留核心）
class Config:
    world_size = 16
    ep_world_size = 16
    moe_expert_num = 128
    batch_size = 8
    hidden_size = 7168
    topk = 8
    dispatch_comm_alg = "fullmesh_v1"
cfg = Config()

# 模型：仅保留 MOE 分发 + 聚合核心
class MOECascadeModel(torch.nn.Module):
    def __init__(self, ep_group):
        super().__init__()
        self.distribute_buffer = ascend_ops.MoeDistributeBuffer(ep_group)

    def forward(self, x, expert_ids, x_active_mask, expert_scales):
        # ============= 核心算子 1：专家分发 =============
        out = self.distribute_buffer.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            x_active_mask=x_active_mask,
            expert_scales=expert_scales,
            moe_expert_num=cfg.moe_expert_num,
            comm_alg=cfg.dispatch_comm_alg
        )

        # ============= 核心算子 2：专家结果聚合 =============
        combine_out = self.distribute_buffer.npu_moe_distribute_combine_v2(
            expand_x=out[0],
            expert_ids=expert_ids,
            assist_info_for_combine=out[2],
            expert_scales=expert_scales
        )
        return combine_out

# 分布式进程入口（单卡逻辑）
def run_rank(rank, x, expert_ids, mask, scales):
    # 1. 初始化 NPU + 分布式通信
    dist.init_process_group(backend="hccl", rank=rank, world_size=cfg.world_size)
    ep_group = dist.new_group()

    # 2. 编译模型
    model = torch.compile(MOECascadeModel(ep_group), backend=torchair.get_npu_backend())

    # 3. 运行 MOE 核心流程
    model(x.npu(), expert_ids.npu(), mask.npu(), scales.npu())

# 主函数：启动多进程 + 构造数据
if __name__ == "__main__":
    # 构造输入数据
    x = torch.randn(cfg.batch_size * cfg.world_size, cfg.hidden_size)
    expert_ids = torch.randint(0, cfg.moe_expert_num, (cfg.batch_size * cfg.world_size, cfg.topk))
    mask = torch.ones(cfg.batch_size * cfg.world_size, cfg.topk).bool()
    scales = torch.rand(cfg.batch_size * cfg.world_size, cfg.topk)

    # 多进程启动分布式任务
    for rank in range(cfg.world_size):
        proc = torch.multiprocessing.Process(
            target=run_rank,
            args=(rank, x.chunk(cfg.world_size)[rank], ...)
        )
        proc.start()
```

---

## 9. 改造清单

### 9.1 必须改造项

| 序号 | 改造项 | 说明 |
|-----|--------|------|
| 1 | 创建 `moe_distribute_dispatch_v2_entry.h` | 声明 <<<>>> 调用函数 |
| 2 | 创建 `moe_distribute_dispatch_v2_entry.cpp` | 实现 Kernel 入口 + <<<>>> 调用 |
| 3 | 创建 `moe_distribute_dispatch_v2_torch.h` | PyTorch 工具函数 |
| 4 | 创建 `moe_distribute_dispatch_v2_torch.cpp` | Host 侧实现 + PyTorch 绑定 |
| 5 | 精简 TilingData 结构 | 移除 通信结构体 相关字段 |
| 6 | 实现 Tiling 计算函数 | 替换 GE Context 依赖 |
| 7 | 精简 TilingKey | 仅保留必要分支 |
| 8 | 调取mc2context | pta侧完成通信配置 |

### 9.2 可选改造项

| 序号 | 改造项 | 说明 |
|-----|--------|------|
| 1 | 移除 TP 相关逻辑 | 如不需要张量并行 |
| 2 | 移除 Elastic Info | 如不需要弹性信息 |
| 3 | 精简 Kernel 模板 | 仅保留必要数据类型 |

---

## 10. 新增算子<<<>>>实现步骤

- 在/examples/fast_kernel_launch_example/csrc下新增文件夹
  - 算子名/SocVersion
- 在文件夹下新增CmakeList
  - add_sources("--npu-arch=dav-2201")
- 新增算子.cpp
  - 例 moe_distribute_dispatch_v2_torch.cpp

## 11. 注意事项

### 11.1 TilingKey 选择

模式的 TilingKey 与原始模式不同，需要根据实际场景重新设计：

- 传递模式：使用模板化的 TilingKey 编码（包含 ArchTag、CommMode 等）
- 直调模式：使用简化的 5 位十进制编码

### 11.2 数据类型映射

| PyTorch 类型 | Ascend C 类型 |
|-------------|--------------|
| `torch.float16` | `float16_t` |
| `torch.bfloat16` | `bfloat16_t` |
| `torch.int32` | `int32_t` |
| `torch.int8` | `int8_t` |

### 11.3 NPU 架构配置

不同 NPU 架构需要设置不同的编译参数：

- `A2`: `--npu-arch=dav-2201`
- `A3`: `--npu-arch=dav-2201`
- `A5`: `--npu-arch=dav-3510`

### 11.4 与原始 aclnn 接口的对比

| 特性 | aclnn 接口 | 直调模式 |
|-----|-----------|---------|
| 调用方式 | `aclnnMoeDistributeDispatchV2(...)` | `torch.ops.ascend_ops.MoeDistributeDispatchV2(...)` |
| 同步方式 | 异步（需调用 `aclrtSynchronizeStream`） | 异步（通过 PyTorch 流管理） |
| 输入参数 | 需构造 `aclTensor` | 直接使用 `torch.Tensor` |
| 输出获取 | 通过 `aclrtMemcpy` | 直接返回 `torch.Tensor` |

---

## 12. 总结

通过直调<<<>>>改造，`MoeDistributeDispatchV2` 和 `MoeDistributeCombineV2` 算子实现了以下目标：

### 12.1 PTA接口侧修改：

- PTA对外接口

```python
import torch
import ascend_ops
torch.ops.ascend_ops.MoeDistributeDispatchV2
torch.ops.ascend_ops.MoeDistributeCombineV2
```

- PTA侧

```c++
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_distribute_dispatch_v2(
    const at::Tensor &x, const at::Tensor &expert_ids, const at::Tensor &mc2_context, c10::string_view group_ep,
    int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, int64_t total_winsize_ep, 
    const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &expert_scales, const c10::optional<at::Tensor> &performance_info,
    int64_t expert_shard_type, int64_t shared_expert_num,
    int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs,
    int64_t expert_token_nums_type, c10::string_view comm_alg,
    int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num)
{
```

- PTA调用kernel

```c++
moe_distribute_dispatch_v2_generic<<<blockDim, nullptr, stream>>>(
        tilingKey, x, expertIds, scales, xActiveMask, expertScales, performanceInfo, expandXOut, dynamicScalesOut,
        assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM,
        mc2Context, tilingData
    );
```

- kernel侧

```c++
template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR performanceInfo, 
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, 
    GM_ADDR workspaceGM, MoeDistributeDispatchV2Info tilingData, TPipe *pipe)

```

### 12.2 PTA侧详细流程：

- PTA对外接口定义

```c++
 std::tuple npu_moe_distribute_dispatch_v2(
  const at::Tensor &x, const at::Tensor &expert_ids, const at::Tensor &mc2_context, ...)
```

- PTA对外接口注册，PTA对外接口schema注册

```c++
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("MoeDistributeDispatchV2(Tensor x, Tensor expert_ids, xxxx, *, xxxx int const_expert_num=0) " \
          "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m) {m.impl("MoeDistributeDispatchV2", TORCH_FN(npu_moe_distribute_dispatch_v2)); }
TORCH_LIBRARY_IMPL(ascend_ops, Meta, m) {m.impl("MoeDistributeDispatchV2", &npu_moe_distribute_dispatch_v2_meta); }
```

- PTA接口内检测输入是否合规（同opplugin）
- PTA接口内构造输出

```c++
at::Tensor ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
```

- PTA接口构造tilingdata

```
利用PTA的输入，获取输入shape、输入datatype、输入是否为空等属性构造
```

- PTA结构构造kernel输入

```
将输入转换为GMADDR: auto x_ptr = get_first_tensor_address<at::Tensor>(x.scalar_type(), x, false);
获取blockDim（也就是aivNum），构造stream
```

- <<<>>>直调kernel

```c++
void moe_distribute_dispatch_v2_demo(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR x, GM_ADDR expertIds, xxxx, GM_ADDR workspaceGM, GM_ADDR mc2Context, MoeDistributeDispatchV2Info tilingData) {
    moe_distribute_dispatch_v2_Generic<<<blockDim, nullptr, stream>>>(tilingKey, x, expertIds, xxxx, workspaceGM, mc2Context, tilingData);}
```

- 根据tilingkey选择调用模板及模板参数

```c++
switch (tilingKey) { case 10000:
    moe_distribute_dispatch_v2 <float16_t, float16_t, MoeDistributeDispatchV2Impl::UNQUANT, false> (
        x, xxxx, workspaceGM, mc2Context, tilingData); break;
case xxxxx}
```

---

## 参考链接

- [原始算子代码 - moe_distribute_dispatch_v2](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_v2)
- [原始算子代码 - moe_distribute_combine_v2](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_combine_v2)
- [改造后算子代码 - dispatch_v2](https://gitcode.com/cann/ops-transformer/tree/master/examples/fast_kernel_launch_example/csrc/moe_distribute_dispatch_v2)
- [改造后算子代码 - combine_v2](https://gitcode.com/cann/ops-transformer/tree/master/examples/fast_kernel_launch_example/csrc/moe_distribute_combine_v2)
- [PR #2141 - dispatchV2 combineV2算子新增<<<>>>直调](https://gitcode.com/cann/ops-transformer/pull/2141)
- [Fast Kernel Launch 示例](https://gitcode.com/cann/ops-nn/tree/master/examples/fast_kernel_launch_example)
