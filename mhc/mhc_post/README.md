# MhcPost

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

### 算子功能

MhcPost是Manifold-Constraint Hyper-Connection (mHC)架构中的核心算子之一，用于实现Post Mapping和残差连接。该算子基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_l$进行Res Mapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。

### 计算公式

$$x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}$$

其中：
- $(H_{l}^{res})^{T} \times x_l$：表示对输入$x_l$使用转置后的$h_{res}$矩阵（双随机矩阵）进行矩阵乘法变换，实现Res Mapping
- $h_{l}^{out} \otimes H_{t}^{post}$：表示将Atten/MLP层的输出$h_{out}$与$h_{post}$变换矩阵进行逐元素相乘，实现Post Mapping
- 最后将两个结果相加，得到下一层的输入$x_{l+1}$

### 应用场景

MhcPost算子主要应用于以下场景：
- **mHC网络架构**：在Manifold-Constraint Hyper-Connection架构中连接不同层
- **残差连接优化**：通过可学习的双随机矩阵$h_{res}$和变换矩阵$h_{post}$优化传统的残差连接
- **多分支网络**：支持将单一流分派到多个流进行并行处理

## 参数说明

### 输入参数

| 参数名 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:-------|:-----|:---------|:---------|:------------|:------------:|
| x | 待计算的张量，表示网络中mHC层的输入数据 | FLOAT16、BFLOAT16 | ND | [B,S,N,D]、[T,N,D] | √ |
| h_res | mHC的h_res变换矩阵，是做完sinkhorn变换后的双随机矩阵 | FLOAT32 | ND | [B,S,N,N]、[T,N,N] | √ |
| h_out | Atten/MLP层的输出 | FLOAT16、BFLOAT16 | ND | [B,S,D]、[T,D] | √ |
| h_post | mHC的h_post变换矩阵 | FLOAT32 | ND | [B,S,N]、[T,N] | √ |

### 输出参数

| 参数名 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:-------|:-----|:---------|:---------|:------------|:------------:|
| out | 网络中mHC层的输出数据，作为下一层的输入。数据类型与x相同 | FLOAT16、BFLOAT16 | ND | [B,S,N,D]、[T,N,D] | √ |

### 参数说明

- **B**: Batch size，批处理大小
- **S**: Sequence length，序列长度
- **N**: Number of streams/heads，流数量或头数
- **D**: Hidden dimension，隐藏层维度
- **T**: Total tokens (B × S)，总token数

## 约束说明

### 维度约束

1. 输入x的维度必须是3维[T,N,D]或4维[B,S,N,D]
2. 输入h_res的维度必须与x相同，为3维[T,N,N]或4维[B,S,N,N]
3. 输入h_out的维度必须是2维[T,D]或3维[B,S,D]
4. 输入h_post的维度必须与h_out相同，为2维[T,N]或3维[B,S,N]
5. 输出out的维度与输入x相同

### 数据类型约束

1. x和h_out的数据类型必须一致，支持FLOAT16或BFLOAT16
2. h_res和h_post的数据类型必须是FLOAT32
3. out的数据类型与x相同

### Shape匹配约束

1. x和h_res的维度数必须相等，且除最后两维外的所有维度必须相同
2. h_out和h_post的维度数必须相等，且所有维度必须相同
3. h_res的最后两维必须相等（N × N的方阵）
4. x的最后一维(D)必须与h_out的最后一维(D)相同
5. h_post的最后一维(N)必须与h_res的最后一维(N)相同
6. h_res和h_out除最后两维外的维度必须匹配

### 其他约束

- 仅支持Ascend 950系列芯片
- 支持非连续Tensor输入

## 接口说明

### ACLNN接口（推荐）

算子执行接口为两段式接口，必须先调用`aclnnMhcPostGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMhcPost`接口执行计算。

```c++
aclnnStatus aclnnMhcPostGetWorkspaceSize(
    const aclTensor  *x,
    const aclTensor  *h_res,
    const aclTensor  *h_out,
    const aclTensor  *h_post,
    aclTensor        *out,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnMhcPost(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

详细接口说明请参考：[aclnnMhcPost接口文档](./docs/aclnnMhcPost.md)

### L0 Kernel接口

```c++
const aclTensor *MhcPost(
    const aclTensor *x,
    const aclTensor *h_res,
    const aclTensor *h_out,
    const aclTensor *h_post,
    aclOpExecutor *executor);
```

## 算子实现架构

### 文件结构

```
mhc_post/
├── op_host/                          # Host侧实现
│   ├── op_api/
│   │   ├── aclnn_mhc_post.h         # ACLNN接口头文件
│   │   ├── aclnn_mhc_post.cpp       # ACLNN接口实现
│   │   ├── mhc_post.h               # L0接口头文件
│   │   └── mhc_post.cpp             # L0接口实现
│   ├── config/ascend950/            # 算子配置
│   │   ├── mhc_post_binary.json     # 算子二进制配置
│   │   └── mhc_post_simplified_key.ini  # 简化key配置
│   ├── mhc_post_tiling.h            # Tiling头文件
│   ├── mhc_post_tiling.cpp          # Tiling实现
│   ├── mhc_post_tiling_base.cpp     # 基础Tiling实现
│   ├── mhc_post_infershape.cpp      # Shape推导实现
│   ├── mhc_post_def.cpp             # 算子定义
│   └── CMakeLists.txt               # Host侧编译配置
├── op_kernel/                        # Kernel侧实现
│   ├── arch35/                       # Ascend 950架构
│   │   ├── mhc_post.h               # Kernel实现头文件
│   │   ├── mhc_post_tiling_data.h   # Tiling数据结构
│   │   └── mhc_post_tiling_key.h    # Tiling Key定义
│   ├── mhc_post_apt.cpp             # Kernel入口
│   └── CMakeLists.txt               # Kernel侧编译配置
├── op_graph/                         # Graph构图
│   ├── mhc_post_proto.h             # 算子原型定义
│   └── CMakeLists.txt               # Graph侧编译配置
├── tests/                            # 测试代码
│   └── ut/                          # 单元测试
│       └── op_host/                 # Host侧测试
│           ├── test_mhc_post_tiling.cpp
│           ├── test_mhc_post_infershape.cpp
│           └── CMakeLists.txt
├── docs/                             # 文档
│   └── aclnnMhcPost.md              # ACLNN接口详细文档
├── CMakeLists.txt                    # 主编译配置
└── README.md                         # 本文件
```

### 实现细节

#### Double Buffer优化

Kernel实现采用Double Buffer技术提升Memory Bound算子性能：
- 数据队列（hOutTileQueue_, xTileQueue_）使用Double Buffer，depth=2
- 实现数据加载与计算的重叠执行，隐藏内存访问延迟

#### 自适应分块策略

根据输入shape自动选择并行化策略：
- **Strategy 0 (usePermanentX=0)**: 逐列处理x，适用于一般场景
- **Strategy 1 (usePermanentX=1)**: 一次性加载所有x数据，适用于大batch场景

#### Tiling参数

| 参数名 | 说明 |
|:-------|:-----|
| n | 流数量/头数 |
| D | 隐藏层维度 |
| usedCoreNum | 使用的核数 |
| normalCoreProcessNum | 普通核处理的数据量 |
| tailCoreProcessNum | 尾核处理的数据量 |
| bsInner/bsOuter/bsTail | Batch维度分块参数 |
| dInner/dOuter/dTail/dTailAlign | 维度D分块参数 |

## 调用示例

### C++ ACLNN调用示例

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/mhc_post.h"
#include "securec.h"

using namespace std;

// 初始化资源
int Init(int32_t deviceId, aclrtStream *stream) {
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        printf("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

// 创建AclTensor
template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, 
                    void **deviceAddr, aclDataType dataType, aclTensor **tensor) {
    int64_t size = 1;
    for (auto i : shape) size *= i;
    auto dataSize = size * sizeof(T);
    
    auto ret = aclrtMalloc(deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;
    
    ret = aclrtMemcpy(*deviceAddr, dataSize, hostData.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return ret;

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, 
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != 0) return ret;

    // 2. 构造输入和输出 (BSND格式)
    std::vector<int64_t> xShape = {1, 1024, 4, 5120};     // BSND
    std::vector<int64_t> hResShape = {1, 1024, 4, 4};     // BSNN
    std::vector<int64_t> hOutShape = {1, 1024, 5120};     // BSD
    std::vector<int64_t> hPostShape = {1, 1024, 4};       // BSn
    std::vector<int64_t> outShape = {1, 1024, 4, 5120};   // BSND

    // 分配设备内存并创建Tensor
    void *xDeviceAddr = nullptr, *hResDeviceAddr = nullptr;
    void *hOutDeviceAddr = nullptr, *hPostDeviceAddr = nullptr, *outDeviceAddr = nullptr;
    aclTensor *xTensor = nullptr, *hResTensor = nullptr;
    aclTensor *hOutTensor = nullptr, *hPostTensor = nullptr, *outTensor = nullptr;

    // 准备Host数据并创建Tensor (示例使用全1数据)
    int64_t xSize = 1 * 1024 * 4 * 5120;
    int64_t hResSize = 1 * 1024 * 4 * 4;
    int64_t hOutSize = 1 * 1024 * 5120;
    int64_t hPostSize = 1 * 1024 * 4;
    int64_t outSize = 1 * 1024 * 4 * 5120;

    std::vector<half> xHostData(xSize, static_cast<half>(1.0f));
    std::vector<float> hResHostData(hResSize, 1.0f);
    std::vector<half> hOutHostData(hOutSize, static_cast<half>(1.0f));
    std::vector<float> hPostHostData(hPostSize, 1.0f);
    std::vector<half> outHostData(outSize, static_cast<half>(0.0f));

    CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    CreateAclTensor(hResHostData, hResShape, &hResDeviceAddr, aclDataType::ACL_FLOAT, &hResTensor);
    CreateAclTensor(hOutHostData, hOutShape, &hOutDeviceAddr, aclDataType::ACL_FLOAT16, &hOutTensor);
    CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPostTensor);
    CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    
    // 第一段接口：获取workspace大小
    ret = aclnnMhcPostGetWorkspaceSize(xTensor, hResTensor, hOutTensor, hPostTensor, 
                                       outTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("aclnnMhcPostGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    // 申请workspace内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段接口：执行计算
    ret = aclnnMhcPost(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("aclnnMhcPost failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. 同步等待
    aclrtSynchronizeStream(stream);

    // 5. 获取结果
    std::vector<half> resultData(outSize);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(half), outDeviceAddr,
                outSize * sizeof(half), ACL_MEMCPY_DEVICE_TO_HOST);

    // 6. 释放资源
    aclDestroyTensor(xTensor);
    aclDestroyTensor(hResTensor);
    aclDestroyTensor(hOutTensor);
    aclDestroyTensor(hPostTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(xDeviceAddr);
    aclrtFree(hResDeviceAddr);
    aclrtFree(hOutDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    return 0;
}
```

完整的示例代码请参考：[aclnnMhcPost接口文档](./docs/aclnnMhcPost.md)

## 性能优化建议

1. **数据类型选择**：对于推理场景，建议使用BFLOAT16以获得更好的性能和精度平衡
2. **内存布局**：虽然支持非连续Tensor，但使用连续Tensor可以获得更好的性能
3. **Batch大小**：较大的Batch size可以更好地利用并行计算能力
4. **维度对齐**：建议将D维度对齐到32或64的倍数，以获得更好的内存访问效率

## 相关算子

- **MhcPre**: mHC架构中的Pre Mapping算子
- **MhcRes**: mHC架构中的残差处理算子

## 参考资源

- [aclnnMhcPost接口文档](./docs/aclnnMhcPost.md) - ACLNN接口详细说明
- [mHC架构论文](https://arxiv.org/) - Manifold-Constraint Hyper-Connection论文

## 版权信息

Copyright (c) 2026 Huawei Technologies Co., Ltd.

This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
